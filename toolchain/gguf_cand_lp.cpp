// Teacher-forced logprob of candidate strings given each prompt, for GGUF models.
// Input: prompts file (one escaped-per-line prompt), candidates file (one escaped string per line).
// Output TSV: one row per prompt, one column per candidate = summed next-token logprob.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static std::string unescape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            if (c == 'n') out += '\n';
            else if (c == 't') out += '\t';
            else if (c == 'r') out += '\r';
            else if (c == '\\') out += '\\';
            else out += c;
        } else {
            out += s[i];
        }
    }
    return out;
}

static std::vector<std::string> read_lines(const std::string &path) {
    std::ifstream f(path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) out.push_back(unescape(line));
    }
    return out;
}

int main(int argc, char **argv) {
    std::string model, prompts_path, cands_path, out_path;
    int n_ctx = 1024, n_threads = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = val();
        else if (a == "--prompts") prompts_path = val();
        else if (a == "--cands") cands_path = val();
        else if (a == "--out") out_path = val();
        else if (a == "--n-ctx") n_ctx = std::stoi(val());
        else if (a == "--threads") n_threads = std::stoi(val());
    }
    if (model.empty() || prompts_path.empty() || cands_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: gguf_cand_lp --model M --prompts P --cands C --out O [--n-ctx N] [--threads T]\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model *m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) {
        std::fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }
    const llama_vocab *vocab = llama_model_get_vocab(m);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_free_model(m);
        return 1;
    }
    llama_set_n_threads(ctx, n_threads, n_threads);

    std::vector<std::vector<llama_token>> cand_tok;
    for (const auto &c : read_lines(cands_path)) {
        std::vector<llama_token> t(64);
        int32_t n = llama_tokenize(vocab, c.c_str(), (int32_t)c.size(), t.data(), (int32_t)t.size(),
                                   /*add_bos=*/false, /*special=*/true);
        if (n < 0) {
            t.resize((size_t)-n + 4);
            n = llama_tokenize(vocab, c.c_str(), (int32_t)c.size(), t.data(), (int32_t)t.size(), false, true);
        }
        t.resize(n);
        cand_tok.push_back(t);
    }

    std::ofstream out(out_path);
    std::vector<std::string> prompts;
    {
        std::ifstream f(prompts_path);
        std::string line;
        while (std::getline(f, line)) prompts.push_back(line);
    }

    std::vector<llama_token> buf;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string text = unescape(prompts[pi]);
        std::vector<llama_token> pt(text.size() + 8);
        int32_t np = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), pt.data(),
                                    (int32_t)pt.size(), /*add_bos=*/false, /*special=*/true);
        if (np <= 0) {
            std::fprintf(stderr, "tokenize failed at prompt %zu\n", pi);
            out << "\n";
            continue;
        }
        pt.resize(np);

        // Fast path: all candidates are single tokens, so one prompt forward pass yields the
        // next-token distribution for every candidate at once. Multi-token candidates fall back
        // to decoding prompt + candidate and summing per-token logprobs.
        std::vector<double> lp(cand_tok.size(), 0.0);
        bool multi = false;
        for (const auto & t : cand_tok) if (t.size() != 1) multi = true;

        if (!multi) {
            if (np > n_ctx) { std::fprintf(stderr, "prompt %zu too long\n", pi); for (size_t ci = 0; ci < lp.size(); ++ci) lp[ci] = -INFINITY; }
            else {
                llama_batch batch = llama_batch_init(np, 0, 1);
                for (int32_t j = 0; j < np; ++j) {
                    batch.token[j]    = pt[j];
                    batch.pos[j]      = j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j]   = (j == np - 1) ? 1 : 0;
                }
                batch.n_tokens = np;
                if (llama_decode(ctx, batch) != 0) {
                    std::fprintf(stderr, "decode failed at prompt %zu\n", pi);
                    for (auto & v : lp) v = -std::numeric_limits<double>::infinity();
                } else {
                    const float *lg = llama_get_logits_ith(ctx, np - 1);
                    if (!lg) {
                        std::fprintf(stderr, "no logits at prompt %zu\n", pi);
                        for (auto & v : lp) v = -std::numeric_limits<double>::infinity();
                    } else {
                        float mx = lg[0];
                        for (int32_t v = 1; v < n_vocab; ++v) mx = std::max(mx, lg[v]);
                        double sum = 0.0;
                        for (int32_t v = 0; v < n_vocab; ++v) sum += std::exp((double)lg[v] - mx);
                        const double lse = mx + std::log(sum);
                        for (size_t ci = 0; ci < cand_tok.size(); ++ci) {
                            lp[ci] = (double)lg[cand_tok[ci][0]] - lse;
                        }
                    }
                }
                llama_batch_free(batch);
            }
            llama_memory_clear(llama_get_memory(ctx), /*clear_cache=*/false);
        } else {
            // Each candidate is scored from the same prompt prefix: decode prompt + candidate
            // separately, reading logits at positions np-1 .. np+len(cand)-2.
            for (size_t ci = 0; ci < cand_tok.size(); ++ci) {
                buf = pt;
                for (auto id : cand_tok[ci]) buf.push_back(id);
                const int32_t n_all = (int32_t)buf.size();

                // llama_batch_get_one() leaves batch.logits == nullptr in current llama.cpp,
                // so the output flags must be set on a batch allocated with llama_batch_init.
                llama_batch batch = llama_batch_init(n_all, 0, 1);
                for (int32_t j = 0; j < n_all; ++j) {
                    batch.token[j]    = buf[j];
                    batch.pos[j]      = j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j]   = (j >= np - 1 && j < n_all - 1) ? 1 : 0;
                }
                batch.n_tokens = n_all;

                if (llama_decode(ctx, batch) != 0) {
                    std::fprintf(stderr, "decode failed at prompt %zu cand %zu\n", pi, ci);
                    lp[ci] = -std::numeric_limits<double>::infinity();
                } else {
                    int cursor = np - 1;
                    for (size_t k = 0; k < cand_tok[ci].size(); ++k) {
                        const float *lg = llama_get_logits_ith(ctx, cursor);
                        if (!lg) {
                            std::fprintf(stderr, "no logits at %d\n", cursor);
                            lp[ci] = -std::numeric_limits<double>::infinity();
                            break;
                        }
                        float mx = lg[0];
                        for (int32_t v = 1; v < n_vocab; ++v) mx = std::max(mx, lg[v]);
                        double sum = 0.0;
                        for (int32_t v = 0; v < n_vocab; ++v) sum += std::exp((double)lg[v] - mx);
                        lp[ci] += (double)lg[cand_tok[ci][k]] - (mx + std::log(sum));
                        ++cursor;
                    }
                }
                llama_batch_free(batch);
                llama_memory_clear(llama_get_memory(ctx), /*clear_cache=*/false);
            }
        }
        for (size_t ci = 0; ci < lp.size(); ++ci) {
            if (ci) out << '\t';
            out << lp[ci];
        }
        out << '\n';
        if (pi % 100 == 0) std::fprintf(stderr, "[%zu/%zu]\n", pi, prompts.size());
    }
    out.close();
    llama_free(ctx);
    llama_free_model(m);
    llama_backend_free();
    return 0;
}

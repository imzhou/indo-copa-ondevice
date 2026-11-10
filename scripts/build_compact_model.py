#!/usr/bin/env python3
"""Build a task-adapted vocab-subset compact model with identical A/B margins."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_D = "/path/to/data"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def copa_prompt(r):
    q = "penyebab" if r.get("question") == "cause" else "akibat"
    return (
        "Premis: " + r["premise"] + "\n"
        "Apa " + q + " yang paling mungkin?\n"
        "A. " + r["choice1"] + "\n"
        "B. " + r["choice2"] + "\n"
        "Jawaban:"
    )


def ecare_prompt(r):
    q = "penyebab" if r.get("ask-for") == "cause" else "akibat"
    return (
        "Premis: " + r["premise"] + "\n"
        "Apa " + q + " yang paling mungkin?\n"
        "A. " + r["hypothesis1"] + "\n"
        "B. " + r["hypothesis2"] + "\n"
        "Jawaban:"
    )


def collect_prompts(data_root):
    d = Path(data_root) / "data"
    groups = {}
    p = d / "copa_id_clean.jsonl"
    if p.exists():
        groups["copa_train"] = [copa_prompt(r) for r in read_jsonl(p)]
    p = d / "ecare_dev_id.jsonl"
    if p.exists():
        groups["ecare_dev"] = [ecare_prompt(r) for r in read_jsonl(p) if not r.get("drop")]
    p = d / "ecare_pilot1500" / "ecare_pilot_id.jsonl"
    if p.exists():
        groups["ecare_pilot"] = [ecare_prompt(r) for r in read_jsonl(p)]
    p = d / "ecare_kd_train.jsonl"
    if p.exists():
        groups["ecare_kd"] = [
            copa_prompt(r) if "choice1" in r else ecare_prompt(r) for r in read_jsonl(p)
        ]
    from datasets import load_dataset

    xv = load_dataset("cambridgeltl/xcopa", "id", split="validation")
    groups["xcopa_val"] = [copa_prompt(r) for r in xv]
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", default=DEFAULT_D)
    ap.add_argument("--base-vocab", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_root = Path(args.data_root)
    base_vocab = (
        Path(args.base_vocab)
        if args.base_vocab
        else data_root / "ckpts/run_id/latest/id_vocab.json"
    )
    meta = json.loads(base_vocab.read_text())
    kept = list(meta["kept_ids"])
    kept_set = set(kept)
    tok = AutoTokenizer.from_pretrained(args.model)
    groups = collect_prompts(args.data_root)
    oov = Counter()
    stats = {}
    for name, prompts in groups.items():
        n_tok = n_oov = n_prompt_oov = 0
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=False)
            bad = [i for i in ids if i not in kept_set]
            n_tok += len(ids)
            n_oov += len(bad)
            n_prompt_oov += int(bool(bad))
            oov.update(bad)
        stats[name] = {
            "prompts": len(prompts),
            "tokens": n_tok,
            "oov_tokens": n_oov,
            "oov_rate": n_oov / max(1, n_tok),
            "prompts_with_oov": n_prompt_oov,
        }
    added = [i for i in oov if i not in kept_set]
    kept.extend(added)
    for sp in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id):
        if sp is not None and sp not in kept_set:
            kept.append(sp)
            kept_set.add(sp)
    v = len(kept)
    pos = {old: i for i, old in enumerate(kept)}
    print("base_vocab", len(meta["kept_ids"]), "oov_unique", len(added), "final_vocab", v)
    for name, s in stats.items():
        print(name, s)
    print("top_oov", [(i, tok.decode([i]), oov[i]) for i in [x for x, _ in oov.most_common(12)]])

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    old_emb = model.get_input_embeddings().weight.data
    hidden = old_emb.shape[1]
    idx = torch.tensor(kept, dtype=torch.long)
    new_emb = torch.nn.Embedding(v, hidden, dtype=old_emb.dtype)
    new_emb.weight.data.copy_(old_emb[idx])
    model.set_input_embeddings(new_emb)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    new_head = torch.nn.Linear(hidden, v, bias=False, dtype=old_emb.dtype)
    if tied:
        new_head.weight = new_emb.weight
    else:
        new_head.weight.data.copy_(model.lm_head.weight.data[idx])
    model.lm_head = new_head
    model.config.vocab_size = v
    model.config.tie_word_embeddings = tied
    if tok.eos_token_id in pos:
        model.config.eos_token_id = pos[tok.eos_token_id]
        model.generation_config.eos_token_id = pos[tok.eos_token_id]
    assert model.get_input_embeddings().weight.shape[0] == v
    assert model.lm_head.weight.shape[0] == v
    params = sum(p.numel() for p in model.parameters()) / 1e6
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    src_ct = Path(args.model) / "chat_template.jinja"
    if src_ct.exists():
        shutil.copy2(src_ct, out / "chat_template.jinja")
    (out / "id_vocab.json").write_text(
        json.dumps({"kept_ids": kept, "unk_old": kept[0], "vocab_size": v})
    )
    manifest = {
        "source_model": args.model,
        "base_vocab": str(base_vocab),
        "final_vocab": v,
        "oov_added": len(added),
        "params_m": params,
        "tied": tied,
        "group_stats": stats,
        "top_oov": [[i, tok.decode([i]), oov[i]] for i in [x for x, _ in oov.most_common(20)]],
    }
    (out / "compact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print("saved", out, "params_m", round(params, 3), "tied", tied)


if __name__ == "__main__":
    main()

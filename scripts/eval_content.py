#!/usr/bin/env python3
"""Content-based (option-text likelihood) vs letter-logprob scoring, order-swapped."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rag import load_items  # noqa: E402
from id_vocab import load_any_tokenizer  # noqa: E402
from probe_teacher_format import (  # noqa: E402
    bootstrap_ci,
    question_word,
    score_condition,
    score_pairs,
)


def prompt_pair(ex, q):
    a = (f"Premis: {ex['premise']}\nApa {q} yang paling mungkin?\n"
         f"A. {ex['choice1']}\nB. {ex['choice2']}\nJawaban:")
    b = (f"Premis: {ex['premise']}\nApa {q} yang paling mungkin?\n"
         f"A. {ex['choice2']}\nB. {ex['choice1']}\nJawaban:")
    return a, b


def load_model(path: str, device: str, load4bit: bool):
    from transformers import AutoModelForCausalLM

    if load4bit:
        from transformers import BitsAndBytesConfig

        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=qc, dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True
            ).eval()
        except ValueError as e:
            if "dispatched on the CPU" not in str(e):
                raise
            print("[load4bit] GPU tight; pin layers to GPU, embed/lm_head to CPU", flush=True)
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
            n = int(getattr(cfg, "num_hidden_layers", 0))
            dm = {"model.embed_tokens": "cpu", "model.norm": "cuda:0"}
            if not bool(getattr(cfg, "tie_word_embeddings", False)):
                dm["lm_head"] = "cpu"
            for i in range(n):
                dm[f"model.layers.{i}"] = "cuda:0"
            qc2 = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=qc2, dtype=torch.bfloat16,
                device_map=dm, trust_remote_code=True
            ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
    return model


def metrics(pf, mf, ps, ms, y):
    ps_choice = [1 - p for p in ps]
    ms_choice = [-m for m in ms]
    avg = [(a + b) / 2 for a, b in zip(mf, ms_choice)]
    p_avg = [int(v > 0) for v in avg]

    def acc(p):
        return sum(1 for a, t in zip(p, y) if a == t) / len(y)

    def auc(scores):
        pos = [s for s, t in zip(scores, y) if t == 1]
        neg = [s for s, t in zip(scores, y) if t == 0]
        return sum(1 for p in pos for q in neg if p > q) / (len(pos) * len(neg))

    return {
        "n": len(y),
        "acc_as_written": acc(pf),
        "acc_swapped": acc(ps_choice),
        "acc_order_averaged": acc(p_avg),
        "auc_as_written": auc(mf),
        "auc_order_averaged": auc(avg),
        "order_consistency": sum(1 for a, b in zip(pf, ps_choice) if a == b) / len(y),
        "predB_as_written": sum(pf) / len(y),
        "predB_swapped": sum(ps_choice) / len(y),
        "ci_order_averaged": bootstrap_ci([int(a == t) for a, t in zip(p_avg, y)]),
    }, {"fwd_margin": mf, "swp_margin": ms_choice, "avg_margin": avg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="xcopa_val,copal_test,copal_test_colloquial")
    ap.add_argument("--scoring", default="both", choices=["both", "content", "letter"])
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load4bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok, _mapped = load_any_tokenizer(args.model)
    model = load_model(args.model, args.device, args.load4bit)

    out = {"model": args.model, "scoring": args.scoring, "conditions": {}, "items": {}, "labels": {}}

    for setname in args.sets.split(","):
        items = load_items(setname)
        if args.limit:
            items = items[: args.limit]
        y = [int(ex["label"]) for ex in items]
        fwd = [prompt_pair(ex, question_word(ex))[0] for ex in items]
        swp = [prompt_pair(ex, question_word(ex))[1] for ex in items]
        out["labels"][setname] = y

        if args.scoring in ("both", "letter"):
            pf, mf = score_condition(model, tok, fwd, args.device, bs=args.bs)
            ps, ms = score_condition(model, tok, swp, args.device, bs=args.bs)
            rec, it = metrics(pf, mf, ps, ms, y)
            out["conditions"][f"{setname}/letter"] = rec
            out["items"][f"{setname}/letter"] = it
            print(f"{setname:22s} letter  avg={rec['acc_order_averaged']:.3f} "
                  f"auc={rec['auc_order_averaged']:.3f} cons={rec['order_consistency']:.3f} "
                  f"ci={rec['ci_order_averaged']}", flush=True)

        if args.scoring in ("both", "content"):
            pf, mf = content_margins(model, tok, items, fwd, args.device, args.bs)
            ps, ms = content_margins(model, tok, items, swp, args.device, args.bs)
            rec, it = metrics(pf, mf, ps, ms, y)
            out["conditions"][f"{setname}/content"] = rec
            out["items"][f"{setname}/content"] = it
            print(f"{setname:22s} content avg={rec['acc_order_averaged']:.3f} "
                  f"auc={rec['auc_order_averaged']:.3f} cons={rec['order_consistency']:.3f} "
                  f"ci={rec['ci_order_averaged']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print("wrote", args.out, flush=True)


@torch.no_grad()
def content_margins(model, tok, items, prompts, device, bs):
    pairs = []
    for i, ex in enumerate(items):
        pairs.append((2 * i, prompts[i], " " + ex["choice1"]))
        pairs.append((2 * i + 1, prompts[i], " " + ex["choice2"]))
    scores = score_pairs(model, tok, pairs, device, bs=bs)
    preds, margins = [], []
    for i in range(len(items)):
        s0, s1 = scores[2 * i], scores[2 * i + 1]
        preds.append(int(s1 > s0))
        margins.append(s1 - s0)
    return preds, margins


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""7B-Instruct nf4 letter agreement on translated pilot items (0/3-shot).
Gate for a 7B-teacher KD arm: 3-shot >=70% and clearly above 0.6B's 64.2%."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ecare_pilot import block, letter_margins, load_ds, load_jsonl, wilson

D = "/path/to/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=f"{D}/data/ecare_pilot1500/ecare_pilot_id.jsonl")
    ap.add_argument("--model", default=f"{D}/ckpts/teacher_7b")
    ap.add_argument("--out", default=f"{D}/data/ecare_pilot1500/teacher7b_agree.json")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.pilot) if not r.get("drop")]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    xv = load_ds("cambridgeltl/xcopa", "id", split="validation")
    demos = []
    for i in range(3):
        ex = xv[i]
        letter = "A" if ex["label"] == 0 else "B"
        demos.append(block(ex["premise"], ex["question"], ex["choice1"], ex["choice2"], letter))
    demo_str = "\n\n".join(demos) + "\n\n"

    p0 = [block(r["premise"], r["ask-for"], r["hypothesis1"], r["hypothesis2"]) for r in rows]
    p3 = [demo_str + p for p in p0]
    m0 = letter_margins(model, tok, p0, chunk=2)
    print("0-shot scoring done", flush=True)
    m3 = letter_margins(model, tok, p3, chunk=2)
    print("3-shot scoring done", flush=True)
    for r, a, b in zip(rows, m0, m3):
        r["pred_7b_0shot"] = 0 if a > 0 else 1
        r["pred_7b_3shot"] = 0 if b > 0 else 1

    report = {"model": args.model, "gate": "3shot >= 0.70 and > 0.6B 0.642"}
    for name, sub in (
        ("ecare", [r for r in rows if r["src"] == "ecare"]),
        ("copa_control", [r for r in rows if r["src"] == "copa_control"]),
    ):
        rep = {}
        for shot, key in (("0shot", "pred_7b_0shot"), ("3shot", "pred_7b_3shot")):
            k = sum(1 for r in sub if r[key] == int(r["label"]))
            lo, hi = wilson(k, len(sub))
            rep[shot] = {"agree": k, "n": len(sub), "acc": k / len(sub), "ci95": [lo, hi]}
            print(f"{name:13s} {shot}: {k}/{len(sub)} = {k / len(sub):.3f}  CI95 [{lo:.3f}, {hi:.3f}]", flush=True)
        report[name] = rep
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Score a student ckpt on translated e-CARE dev: 0/3-shot letter agreement vs gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ecare_pilot import block, letter_margins, load_ds, load_jsonl, wilson
from id_vocab import load_any_tokenizer

D = "/path/to/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--dev", default=f"{D}/data/ecare_dev_id.jsonl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.dev) if not r.get("drop")]
    tok, _mapped = load_any_tokenizer(args.student)
    model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16).cuda().eval()

    xv = load_ds("cambridgeltl/xcopa", "id", split="validation")
    demos = []
    for i in range(3):
        ex = xv[i]
        letter = "A" if ex["label"] == 0 else "B"
        demos.append(block(ex["premise"], ex["question"], ex["choice1"], ex["choice2"], letter))
    demo_str = "\n\n".join(demos) + "\n\n"

    p0 = [block(r["premise"], r["ask-for"], r["hypothesis1"], r["hypothesis2"]) for r in rows]
    p3 = [demo_str + p for p in p0]
    m0 = letter_margins(model, tok, p0)
    m3 = letter_margins(model, tok, p3)

    report = {"student": args.student, "n": len(rows)}
    for shot, margins in (("0shot", m0), ("3shot", m3)):
        k = sum(1 for r, m in zip(rows, margins) if (0 if m > 0 else 1) == int(r["label"]))
        lo, hi = wilson(k, len(rows))
        report[shot] = {"agree": k, "n": len(rows), "acc": k / len(rows), "ci95": [lo, hi]}
        print(f"{shot}: {k}/{len(rows)} = {k / len(rows):.3f}  CI95 [{lo:.3f}, {hi:.3f}]", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
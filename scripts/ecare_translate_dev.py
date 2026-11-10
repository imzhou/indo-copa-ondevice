#!/usr/bin/env python3
"""Translate full e-CARE dev to Indonesian for KD model selection (no scoring)."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ecare_pilot import load_jsonl, translate_batch, write_jsonl

D = "/path/to/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=f"{D}/data/ecare_dev.jsonl")
    ap.add_argument("--out", default=f"{D}/data/ecare_dev_id.jsonl")
    ap.add_argument("--translator", default=f"{D}/ckpts/teacher_7b")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    items = [
        {
            "premise": r["premise"],
            "ask-for": r["ask-for"],
            "hypothesis1": r["hypothesis1"],
            "hypothesis2": r["hypothesis2"],
            "label": int(r["label"]),
            "src": "ecare_dev",
            "premise_en": r["premise"],
            "h1_en": r["hypothesis1"],
            "h2_en": r["hypothesis2"],
        }
        for r in rows
        if r.get("ask-for") in ("cause", "effect")
    ]
    out = Path(args.out)
    ckpt = out.with_name(out.stem + ".ckpt.jsonl")
    cache = {}
    if ckpt.exists():
        for it in load_jsonl(ckpt):
            if not it.get("drop"):
                cache[(it["premise_en"], it["h1_en"], it["h2_en"])] = (
                    it["premise"], it["hypothesis1"], it["hypothesis2"]
                )
    pending = []
    for it in items:
        tr = cache.get((it["premise_en"], it["h1_en"], it["h2_en"]))
        if tr:
            it["premise"], it["hypothesis1"], it["hypothesis2"] = tr
        else:
            pending.append(it)
    print(f"dev items {len(items)}, cached {len(items) - len(pending)}", flush=True)

    failed = 0
    if pending:
        tok = AutoTokenizer.from_pretrained(args.translator)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.translator, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        for s in range(0, len(pending), args.batch):
            batch = pending[s : s + args.batch]
            res = translate_batch(tok, model, batch)
            retry = [it for it, tr in zip(batch, res) if tr is None]
            if retry:
                res2 = translate_batch(tok, model, retry, strict=True)
                rmap = {id(it): tr for it, tr in zip(retry, res2)}
                res = [rmap.get(id(it), tr) if tr is None else tr for it, tr in zip(batch, res)]
            for it, tr in zip(batch, res):
                if tr is None:
                    it["drop"] = True
                    failed += 1
                else:
                    it["premise"], it["hypothesis1"], it["hypothesis2"] = tr
            if (s // args.batch) % 10 == 0:
                print(f"  translated {min(s + args.batch, len(pending))}/{len(pending)} failed={failed}", flush=True)
                write_jsonl(ckpt, items)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_jsonl(ckpt, items)
    write_jsonl(out, items)
    print(f"dev translation done, failed={failed}, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
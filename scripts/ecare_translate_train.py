#!/usr/bin/env python3
"""Translate the full e-CARE train split into Indonesian (resumable, stop-anytime).

Unlike ecare_translate_dev.py, the checkpoint file only contains *completed* items, so
interrupting and resuming never mistakes an untranslated row for a translated one.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ecare_pilot import load_jsonl, translate_batch, write_jsonl

D = "/path/to/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=f"{D}/data/ecare_train.jsonl")
    ap.add_argument("--out", default=f"{D}/data/ecare_train_id.jsonl")
    ap.add_argument("--translator", default=f"{D}/ckpts/teacher_7b")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="translate at most N unique items")
    args = ap.parse_args()

    out = Path(args.out)
    ckpt = out.with_name(out.stem + ".ckpt.jsonl")

    rows = load_jsonl(args.input)
    uniq = {}
    for r in rows:
        if r.get("ask-for") not in ("cause", "effect"):
            continue
        key = (r["premise"], r["hypothesis1"], r["hypothesis2"])
        uniq.setdefault(key, {"premise_en": key[0], "h1_en": key[1], "h2_en": key[2]})
    items = list(uniq.values())
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]

    done = {}
    if ckpt.exists():
        for it in load_jsonl(ckpt):
            done[(it["premise_en"], it["h1_en"], it["h2_en"])] = it
    pending = [it for it in items if (it["premise_en"], it["h1_en"], it["h2_en"]) not in done]
    print(f"train rows {len(rows)}  unique triples {len(uniq)}  to translate {len(items)}  "
          f"already done {len(done)}  pending {len(pending)}", flush=True)

    if not pending:
        write_jsonl(out, items)
        print(f"nothing to do, wrote {out}", flush=True)
        return

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

    t0 = time.time()
    failed = 0
    n_ok = 0
    for s in range(0, len(pending), args.batch):
        batch = pending[s : s + args.batch]
        res = translate_batch(tok, model, batch)
        retry = [it for it, tr in zip(batch, res) if tr is None]
        if retry:
            rmap = dict(zip(map(id, retry), translate_batch(tok, model, retry, strict=True)))
            res = [rmap.get(id(it), tr) if tr is None else tr for it, tr in zip(batch, res)]
        for it, tr in zip(batch, res):
            if tr is None:
                it = {**it, "drop": True, "translated": False}
                failed += 1
            else:
                it = {**it, "premise": tr[0], "hypothesis1": tr[1], "hypothesis2": tr[2],
                      "translated": True}
                n_ok += 1
            done[(it["premise_en"], it["h1_en"], it["h2_en"])] = it
        if (s // args.batch) % 5 == 0:
            rate = (s + args.batch) / max(1e-9, time.time() - t0)
            eta = (len(pending) - s - args.batch) / max(1e-9, rate) / 3600
            print(f"  {s + args.batch}/{len(pending)} ok={n_ok} drop={failed} "
                  f"{rate:.2f} item/s ETA {eta:.2f} h", flush=True)
            write_jsonl(ckpt, list(done.values()))
    write_jsonl(ckpt, list(done.values()))
    write_jsonl(out, [done[(it["premise_en"], it["h1_en"], it["h2_en"])] for it in items
                      if (it["premise_en"], it["h1_en"], it["h2_en"]) in done])
    print(f"train translation done ok={n_ok} drop={failed} took {(time.time() - t0) / 3600:.2f} h",
          flush=True)


if __name__ == "__main__":
    main()

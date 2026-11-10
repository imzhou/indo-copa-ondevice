#!/usr/bin/env python3
"""Stream Indonesian Wikipedia to jsonl. Skip if train.jsonl already exists."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def dump_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            text = (r.get("text") or "").strip()
            if len(text) < 80:
                continue
            f.write(json.dumps({"text": text, "lang": r.get("lang", "id")}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} lines -> {path}")


def stream_wiki(config: str, lang: str, max_docs: int | None) -> list[dict]:
    print(f"streaming wikipedia {config} ...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        rows.append({"text": ex["text"], "lang": lang})
        if (i + 1) % 5000 == 0:
            print(f"  {lang} {i+1} docs", flush=True)
        if max_docs and i + 1 >= max_docs:
            break
    print(f"  {lang}: {len(rows)} docs", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    p.add_argument("--id-max", type=int, default=0, help="0 = all ID wiki docs")
    p.add_argument("--valid-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    train_path = out / "train.jsonl"
    valid_path = out / "valid.jsonl"
    if train_path.exists() and valid_path.exists() and train_path.stat().st_size > 1000:
        print(f"data already exists: {train_path}")
        return

    id_max = args.id_max or None
    rows = stream_wiki("20231101.id", "id", id_max)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_valid = min(args.valid_n, max(1, len(rows) // 50))
    dump_rows(rows[:n_valid], valid_path)
    dump_rows(rows[n_valid:], train_path)


if __name__ == "__main__":
    main()

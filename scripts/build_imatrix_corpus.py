#!/usr/bin/env python3
"""Render training-split MC prompts as plain text for llama-imatrix calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_content import prompt_pair  # noqa: E402
from probe_teacher_format import question_word  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    src = []
    for line in open(args.jsonl, encoding="utf-8"):
        ex = json.loads(line)
        if ex.get("source", "").endswith("_swap"):
            continue
        src.append(ex)
    picked = src[:: args.stride][: args.rows]

    texts = []
    for ex in picked:
        q = question_word(ex)
        fwd, swp = prompt_pair(ex, q)
        texts.append(fwd)
        texts.append(swp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(texts) + "\n", encoding="utf-8")
    chars = sum(len(t) for t in texts)
    print(f"rows {len(picked)} of {len(src)} -> prompts {len(texts)}, chars {chars}")


if __name__ == "__main__":
    main()

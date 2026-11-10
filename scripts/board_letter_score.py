#!/usr/bin/env python3
"""Score on-device letter TSVs against the host TSV / HF probe on the same slice."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_content import metrics  # noqa: E402
from eval_gguf import letter_scores  # noqa: E402
from probe_teacher_format import mcnemar_exact  # noqa: E402


def read_tsv(path: str, n: int):
    rows = [[float(x) for x in l.split("\t")] for l in
            Path(path).read_text().strip().split("\n") if l.strip()]
    if len(rows) != 2 * n:
        raise SystemExit(f"{path}: expected {2 * n} rows, got {len(rows)}")
    return rows


def flags_of(scores, labels):
    return [int(m > 0) == int(t == 1) for m, t in zip(scores, labels)]


def report(name: str, rec: dict) -> None:
    print(f"  {name:14s} acc={rec['acc_order_averaged']:.3f} "
          f"auc={rec.get('auc_order_averaged', float('nan')):.3f} "
          f"cons={rec.get('order_consistency', float('nan')):.3f} "
          f"ci={rec.get('ci_order_averaged')}")


def paired(tag: str, a: str, b: str, fa, fb, ma, mb, y) -> None:
    agree = sum(1 for x, z in zip(fa, fb) if x == z) / len(fa)
    pear = float(np.corrcoef(ma, mb)[0, 1])
    mc = mcnemar_exact(fa, fb, [1] * len(y))
    gap = sum(fa) - sum(fb)
    print(f"  [{tag}] {a} vs {b}: agree={agree:.3f} pearson={pear:.4f} "
          f"mcnemar b={mc['b']}(only {a} right) c={mc['c']}(only {b} right) p={mc['p']:.3f} "
          f"self-check b-c={mc['b'] - mc['c']} vs acc_gap*n={gap:+d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--tsv", action="append", default=[], help="name=path; every unordered pair is tested")
    ap.add_argument("--hf", default="", help="HF bf16 probe JSON to slice by the same indices")
    ap.add_argument("--hf-cond", default="ecare_dev/letter")
    args = ap.parse_args()

    rec = json.load(open(args.subset))
    y, n, idx = rec["labels"], rec["n"], rec["indices"]
    print(f"== slice {rec['set']} n={n} seed={rec['seed']} tok_max={rec['prompt_tokens_max']} "
          f"n_ctx={rec['n_ctx_pow2']}")

    stats, margins, flags = {}, {}, {}
    for spec in args.tsv:
        name, path = spec.split("=", 1)
        rows = read_tsv(path, n)
        pf, mf = letter_scores(rows[:n])
        ps, ms = letter_scores(rows[n:])
        m, it = metrics(pf, mf, ps, ms, y)
        stats[name] = m
        margins[name] = it["avg_margin"]
        flags[name] = flags_of(it["avg_margin"], y)
        print(f"[{name}]")
        report(name, m)
        json.dump({"rec": m, "items": it, "labels": y},
                  open(Path(path).with_suffix(".metrics.json"), "w"), ensure_ascii=False)

    if args.hf:
        hf = json.load(open(args.hf))
        hy = hf["labels"]
        hy = hy[rec["set"]] if isinstance(hy, dict) else hy
        hfull = hf["items"][args.hf_cond]["avg_margin"]
        if [hy[i] for i in idx] != y:
            print(f"[hf] {Path(args.hf).name}: label mismatch on slice - skipped")
        else:
            name = "hf_bf16"
            hm = [hfull[i] for i in idx]
            margins[name] = hm
            flags[name] = flags_of(hm, y)
            print(f"[{name}] ({Path(args.hf).name})")
            report(name, {"acc_order_averaged": sum(flags[name]) / n})

    names = list(margins)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            paired("pair", a, b, flags[a], flags[b], margins[a], margins[b], y)
    print("BOARD_LETTER_DONE", flush=True)


if __name__ == "__main__":
    main()

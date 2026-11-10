#!/usr/bin/env python3
"""Stage a fixed eval slice as gguf_cand_lp input files, then score those exact
bytes on the host, so the same file can be pushed to the A133 board and compared
item-for-item with the on-device TSV.

Nothing here reads XCOPA test or e-CARE test: the default slice is ecare_dev.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_content import metrics, prompt_pair  # noqa: E402
from eval_gguf import CANDS, esc, letter_scores  # noqa: E402
from eval_rag import load_items  # noqa: E402
from probe_teacher_format import mcnemar_exact, question_word  # noqa: E402

TOOL_X86 = "/path/to/data/toolchain/gguf_cand_lp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="ecare_dev")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tokenizer", default="/path/to/data/ckpts/teacher")
    ap.add_argument("--gguf", action="append", default=[], help="host GGUF to score on the same bytes")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    items = load_items(args.set)
    idx = sorted(random.Random(args.seed).sample(range(len(items)), args.n))
    chosen = [items[i] for i in idx]
    y = [int(ex["label"]) for ex in chosen]

    pairs = [prompt_pair(ex, question_word(ex)) for ex in chosen]
    lines = [p for p, _ in pairs] + [s for _, s in pairs]
    (outdir / "p_dev.txt").write_text("\n".join(esc(l) for l in lines) + "\n", encoding="utf-8")
    (outdir / "c.txt").write_text("\n".join(esc(c) for c in CANDS) + "\n", encoding="utf-8")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lens = [len(tok.encode(l, add_special_tokens=False)) for l in lines]
    n_ctx = 1
    while n_ctx < max(lens) + 2:
        n_ctx *= 2

    rec = {"set": args.set, "n": args.n, "seed": args.seed, "indices": idx, "labels": y,
           "prompt_tokens_max": max(lens), "prompt_tokens_mean": sum(lens) / len(lens),
           "n_ctx_pow2": n_ctx, "lines": len(lines), "cands": CANDS}
    (outdir / "subset.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[stage] {args.set} n={args.n} seed={args.seed} -> {outdir}/p_dev.txt "
          f"({len(lines)} lines) tok max={max(lens)} n_ctx_pow2={n_ctx}", flush=True)

    for gguf in args.gguf:
        tsv = outdir / f"letter_host_{Path(gguf).stem}.tsv"
        print(f"[host] {gguf} {time.strftime('%FT%T%z')}", flush=True)
        subprocess.run([TOOL_X86, "--model", gguf, "--prompts", str(outdir / "p_dev.txt"),
                        "--cands", str(outdir / "c.txt"), "--out", str(tsv),
                        "--threads", str(args.threads), "--n-ctx", str(n_ctx)], check=True)
        rows = [[float(x) for x in l.split("\t")] for l in tsv.read_text().strip().split("\n")]
        pf, mf = letter_scores(rows[: args.n])
        ps, ms = letter_scores(rows[args.n:])
        m, it = metrics(pf, mf, ps, ms, y)
        print(f"   {Path(gguf).stem:22s} acc={m['acc_order_averaged']:.3f} auc={m['auc_order_averaged']:.3f} "
              f"cons={m['order_consistency']:.3f} ci={m['ci_order_averaged']}", flush=True)
        (outdir / f"host_metrics_{Path(gguf).stem}.json").write_text(
            json.dumps({"rec": m, "items": it, "labels": y}, ensure_ascii=False), encoding="utf-8")
    print("STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()

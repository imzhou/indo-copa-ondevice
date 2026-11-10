#!/usr/bin/env python3
"""Score a GGUF with the same letter + order-swap protocol as the HF probes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_content import metrics, prompt_pair  # noqa: E402
from eval_rag import load_items  # noqa: E402
from probe_teacher_format import mcnemar_exact, question_word  # noqa: E402

TOOL = "/path/to/data/toolchain/gguf_cand_lp"
CANDS = [" A", "A", " B", "B"]  # mirrors score_condition's max(' X','X') per letter


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def tool_rows(gguf: str, prompts: list[str], threads: int, n_ctx: int) -> list[list[float]]:
    with tempfile.TemporaryDirectory() as td:
        pf, cf = Path(td, "p.txt"), Path(td, "c.txt")
        of, ef = Path(td, "o.tsv"), Path(td, "err.log")
        pf.write_text("\n".join(esc(p) for p in prompts) + "\n")
        cf.write_text("\n".join(esc(c) for c in CANDS) + "\n")
        with open(ef, "w") as err:
            proc = subprocess.run(
                [TOOL, "--model", gguf, "--prompts", str(pf), "--cands", str(cf),
                 "--out", str(of), "--threads", str(threads), "--n-ctx", str(n_ctx)],
                check=False, stdout=subprocess.DEVNULL, stderr=err)
        if proc.returncode != 0:
            sys.stderr.write(ef.read_text()[-4000:])
            raise SystemExit(f"tool failed rc={proc.returncode} on {gguf}")
        rows = []
        for line in of.read_text().strip().split("\n"):
            rows.append([float(x) for x in line.split("\t")])
    return rows


def letter_scores(rows: list[list[float]]):
    preds, margins = [], []
    for r in rows:
        a, b = max(r[0], r[1]), max(r[2], r[3])
        preds.append(int(b > a))
        margins.append(b - a)
    return preds, margins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="xcopa_val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--compare", default="", help="HF probe JSON with a letter condition")
    ap.add_argument("--compare-cond", default="xcopa_val/letter")
    args = ap.parse_args()

    out = {"model": args.gguf, "conditions": {}, "items": {}, "labels": {}}
    for setname in args.sets.split(","):
        items = load_items(setname)
        if args.limit:
            items = items[: args.limit]
        y = [int(ex["label"]) for ex in items]
        out["labels"][setname] = y
        fwd = [prompt_pair(ex, question_word(ex))[0] for ex in items]
        swp = [prompt_pair(ex, question_word(ex))[1] for ex in items]
        rows = tool_rows(args.gguf, fwd + swp, args.threads, args.n_ctx)
        pf, mf = letter_scores(rows[: len(items)])
        ps, ms = letter_scores(rows[len(items):])
        rec, it = metrics(pf, mf, ps, ms, y)
        out["conditions"][f"{setname}/letter"] = rec
        out["items"][f"{setname}/letter"] = it
        print(f"{setname:22s} letter  avg={rec['acc_order_averaged']:.3f} "
              f"auc={rec['auc_order_averaged']:.3f} cons={rec['order_consistency']:.3f} "
              f"ci={rec['ci_order_averaged']}", flush=True)

    if args.compare:
        ref = json.load(open(args.compare))
        out["comparison"] = {}
        for setname in args.sets.split(","):
            cond = f"{setname}/letter"
            if cond not in ref.get("items", {}):
                continue
            r = ref["items"][cond]
            ref_labels = ref["labels"]
            ry = ref_labels[setname] if isinstance(ref_labels, dict) else ref_labels
            gg = out["items"][cond]
            gg_labels = out["labels"][setname]
            n = min(len(ry), len(gg_labels))
            if n != len(ry) or n != len(gg_labels):
                print(f"[compare] {setname}: label length mismatch ref={len(ry)} gguf={len(gg_labels)}", flush=True)
                continue
            if ry != gg_labels:
                nd = sum(1 for a, b in zip(ry, gg_labels) if a != b)
                print(f"[compare] {setname}: label mismatch on {nd}/{n} items - skipping", flush=True)
                continue
            ref_correct = [int((m > 0) == (t == 1)) for m, t in zip(r["avg_margin"], ry)]
            gg_correct = [int((m > 0) == (t == 1)) for m, t in zip(gg["avg_margin"], gg_labels)]
            agree = sum(1 for a, b in zip(ref_correct, gg_correct) if a == b) / n
            # mcnemar_exact's third arg is gold; both inputs here are already
            # correct-flags, so all-ones reduces it to the discordant-pair count.
            mc = mcnemar_exact(ref_correct, gg_correct, [1] * n)
            corr = None
            try:
                import numpy as np
                corr = float(np.corrcoef(r["avg_margin"], gg["avg_margin"])[0, 1])
            except Exception:
                pass
            print(f"vs HF bf16 ({Path(args.compare).name}) [{setname}] "
                  f"decision-agreement={agree:.3f} margin-pearson={'' if corr is None else f'{corr:.4f}'} "
                  f"mcnemar b={mc['b']} c={mc['c']} p={mc['p']:.3f}", flush=True)
            out["comparison"][setname] = {
                "ref": args.compare, "ref_model": ref.get("model"), "agreement": agree,
                "margin_pearson": corr, "mcnemar": mc}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

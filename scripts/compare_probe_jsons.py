#!/usr/bin/env python3
"""Cross-compare saved probe JSONs (same items, same protocol) without re-running inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_teacher_format import mcnemar_exact  # noqa: E402


def labels_of(d, setname):
    lab = d.get("labels")
    return lab if isinstance(lab, list) else lab.get(setname)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--files", required=True, help="comma-separated JSONs")
    ap.add_argument("--cond", default="/letter")
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    print(f"ref = {Path(args.ref).name}  ({ref.get('model')})")
    for path in args.files.split(","):
        d = json.load(open(path))
        print(f"\n== {Path(path).name}  ({d.get('model')})")
        for ck, rec in d["conditions"].items():
            if not ck.endswith(args.cond) or ck not in ref["items"]:
                continue
            setname = ck.split("/")[0]
            r, c = ref["items"][ck], d["items"][ck]
            yr, yc = labels_of(ref, setname), labels_of(d, setname)
            if yr != yc:
                print(f"  {setname:22s} label mismatch - skipped")
                continue
            rc = [int((m > 0) == (t == 1)) for m, t in zip(r["avg_margin"], yr)]
            cc = [int((m > 0) == (t == 1)) for m, t in zip(c["avg_margin"], yr)]
            agree = sum(1 for a, b in zip(rc, cc) if a == b) / len(yr)
            # mcnemar_exact's third arg is gold; both inputs here are already
            # correct-flags, so all-ones reduces it to the discordant-pair count.
            mc = mcnemar_exact(rc, cc, [1] * len(yr))
            try:
                import numpy as np
                pear = float(np.corrcoef(r["avg_margin"], c["avg_margin"])[0, 1])
            except Exception:
                pear = float("nan")
            print(f"  {setname:22s} acc {rec['acc_order_averaged']:.3f} "
                  f"(ref {ref['conditions'][ck]['acc_order_averaged']:.3f})  "
                  f"auc {rec['auc_order_averaged']:.3f} (ref {ref['conditions'][ck]['auc_order_averaged']:.3f})  "
                  f"agree {agree:.3f}  pearson {pear:.4f}  mcnemar b={mc['b']} c={mc['c']} p={mc['p']:.3f}")


if __name__ == "__main__":
    main()

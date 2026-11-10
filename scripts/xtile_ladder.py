#!/usr/bin/env python3
"""Cross-lingual quantisation ladder: does the Q4_K_M -> Q4_0 cliff move when the
item language changes but the prompt frame and scoring protocol do not?

The protocol is imported, never re-implemented, so it cannot drift from the ladder
numbers already in the paper: CANDS / esc / tool_rows / letter_scores come from
eval_gguf, the metric bundle from eval_content. The `id` frame condition on Indonesian
items is asserted byte-for-byte against eval_content.prompt_pair before scoring.

Sealed-data discipline: `--split test` additionally requires --authorise-test-read and
refuses if a test result JSON already exists in --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_content import metrics, prompt_pair  # noqa: E402
from eval_gguf import CANDS, esc, letter_scores, tool_rows  # noqa: E402
from probe_teacher_format import (  # noqa: E402
    bootstrap_diff_ci,
    mcnemar_exact,
    question_word,
)
ROOT = "/path/to/data"
ARMS = f"{ROOT}/gguf_arms"
GGUF_A = f"{ROOT}/gguf_a"

# ---- frozen protocol constants (editing any of these = a new protocol version) ---
PROTOCOL = "xtile-1"
QW = {"cause": {"id": "penyebab", "en": "cause"},
      "effect": {"id": "akibat", "en": "effect"}}
FRAME_SHAPE = "{head}{premise}\n{question}\nA. {c_left}\nB. {c_right}\n{tail}"
FRAMES = {
    "id": {"head": "Premis: ", "tail": "Jawaban:", "question": "Apa {w} yang paling mungkin?"},
    "en": {"head": "Premise: ", "tail": "Answer:", "question": "What is the most likely {w}?"},
}
LANGS_PRIMARY = ["id", "vi", "th"]
RUNGS = {
    "n06ctl": ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_XS",
               "Q2_K", "Q2_K_imx", "Q3_K_M", "Q4_0", "Q4_K_M", "Q8_0", "f16"],
    "n06kl": ["IQ2_XXS", "IQ2_S", "IQ3_M", "IQ4_XS", "Q4_0", "Q4_K_M", "f16"],
    "n06mid": ["IQ2_XXS", "IQ2_S", "IQ3_M", "IQ4_XS", "Q4_0", "Q4_K_M", "f16"],
    "base": ["Q4_0", "Q4_K_M", "f16"],
}
FAMILY_DIR = {"n06ctl": ARMS, "n06kl": ARMS, "n06mid": ARMS, "base": GGUF_A}
FAMILY_STEM = {"n06ctl": "n06ctl", "n06kl": "n06kl", "n06mid": "n06mid", "base": "qwen3_0.6b"}
CLIFF = ("Q4_K_M", "Q4_0")  # the rung pair whose existence is being tested
TEST_READ_PHRASE = "XTILE-TEST-READ-ONCE"
ANCHOR_ACC, ANCHOR_TOL = 0.711, 0.005  # qwen3_0.6b_f16.gguf on xcopa_val, eval_gguf, 2026-11-03


def xtile_pair(ex, frame: str) -> tuple[str, str]:
    f = FRAMES[frame]
    q = f["question"].format(w=QW[ex["question"]][frame])
    a = FRAME_SHAPE.format(head=f["head"], premise=ex["premise"], question=q,
                           c_left=ex["choice1"], c_right=ex["choice2"], tail=f["tail"])
    b = FRAME_SHAPE.format(head=f["head"], premise=ex["premise"], question=q,
                           c_left=ex["choice2"], c_right=ex["choice1"], tail=f["tail"])
    return a, b


def load_xcopa(lang: str, split: str) -> list[dict]:
    from datasets import load_dataset

    rows = list(load_dataset("cambridgeltl/xcopa", lang)[split])
    return [{"premise": r["premise"], "question": r["question"], "choice1": r["choice1"],
             "choice2": r["choice2"], "label": int(r["label"]), "idx": int(r["idx"])}
            for r in rows]


def model_list(fam: str) -> list[dict]:
    out = []
    for rung in RUNGS[fam]:
        p = f"{FAMILY_DIR[fam]}/{FAMILY_STEM[fam]}_{rung}.gguf"
        if Path(p).exists():
            out.append({"fam": fam, "rung": rung, "name": f"{fam}:{rung}", "path": p,
                        "bytes": Path(p).stat().st_size})
    return out


def prereg_blob() -> dict:
    return {"protocol": PROTOCOL, "cands": CANDS, "frames": FRAMES, "frame_shape": FRAME_SHAPE,
            "question_words": QW, "langs": LANGS_PRIMARY,
            "rungs": RUNGS, "cliff_pair": list(CLIFF), "metric": "acc_order_averaged",
            "stats": ["exact two-sided McNemar on order-averaged preds",
                      "bootstrap 2000 seed 0 for acc, seed 1 for the diff"]}


def prereg_hash() -> str:
    return hashlib.sha256(json.dumps(prereg_blob(), sort_keys=True).encode()).hexdigest()


def anchor_check(threads: int) -> float:
    """End-to-end regression on the prompt/row layout against one published number.

    0.711 = qwen3_0.6b_f16.gguf on xcopa_val (97 items) via scripts/eval_gguf.py, measured
    2026-11-03 on 175. This is the check that catches blocked-vs-interleaved row ordering,
    which no per-item assertion can see.
    """
    from eval_rag import load_items

    gguf = f"{GGUF_A}/qwen3_0.6b_f16.gguf"
    items = load_items("xcopa_val")
    y = [int(e["label"]) for e in items]
    pairs = [prompt_pair(e, question_word(e)) for e in items]
    lines = [p for p, _ in pairs] + [s for _, s in pairs]
    rows = tool_rows(gguf, lines, threads, 128)
    pf, mf = letter_scores(rows[:len(items)])
    ps, ms = letter_scores(rows[len(items):])
    rec, _ = metrics(pf, mf, ps, ms, y)
    acc = rec["acc_order_averaged"]
    ok = abs(acc - ANCHOR_ACC) <= ANCHOR_TOL
    print(f"[anchor] {Path(gguf).name} xcopa_val acc={acc:.3f} want {ANCHOR_ACC:.3f}"
          f"±{ANCHOR_TOL} auc={rec['auc_order_averaged']:.3f} "
          f"cons={rec['order_consistency']:.3f} -> {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("[anchor FAIL] row layout, scorer or prompts changed; scoring stopped")
    return acc


def paired_diff_ci(g: list[int], ref: list[int], n_boot: int = 2000, seed: int = 2) -> list[float]:
    """Bootstrap CI of mean(g - ref); position i is the same underlying COPA item in both lists."""
    rng = random.Random(seed)
    n = len(g)
    d = sorted(sum((g[i] - ref[i] for i in (rng.randrange(n) for _ in range(n))), start=0) / n
               for _ in range(n_boot))
    return [d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]]


def ladder_analysis(cells: dict, models: dict) -> dict:
    """cells: (model_name, lang, frame) -> {rec, items, labels}. Ladder ordered by bytes on disk."""
    ana = {}
    gvec = {}  # (fam, lang, frame) -> per-item correct(Q4_0) - correct(Q4_K_M)
    for fam, ms in models.items():
        sized = sorted(ms, key=lambda m: -m["bytes"])
        for lang in LANGS_PRIMARY:
            for frame in ("id", "en"):
                rows = []
                for m in sized:
                    key = (m["name"], lang, frame)
                    if key not in cells:
                        continue
                    rec = cells[key]["rec"]
                    rows.append({"rung": m["rung"], "bytes": m["bytes"],
                                 "acc": rec["acc_order_averaged"], "auc": rec["auc_order_averaged"],
                                 "cons": rec["order_consistency"], "ci": rec["ci_order_averaged"]})
                adj = [{"from": a["rung"], "to": b["rung"], "d_acc": round(b["acc"] - a["acc"], 4)}
                       for a, b in zip(rows, rows[1:])]
                cliff = None
                try:
                    hi = next(r for r in rows if r["rung"] == CLIFF[0])
                    lo = next(r for r in rows if r["rung"] == CLIFF[1])
                except StopIteration:
                    hi = lo = None
                if hi and lo:
                    kh, kl = (f"{fam}:{CLIFF[0]}", lang, frame), (f"{fam}:{CLIFF[1]}", lang, frame)
                    y = cells[kh]["labels"]
                    pa = [int(m > 0) for m in cells[kh]["items"]["avg_margin"]]
                    pb = [int(m > 0) for m in cells[kl]["items"]["avg_margin"]]
                    ca = [int(p == t) for p, t in zip(pa, y)]
                    cb = [int(p == t) for p, t in zip(pb, y)]
                    cliff = {"acc_hi": hi["acc"], "acc_lo": lo["acc"], "d_acc": lo["acc"] - hi["acc"],
                             "d_acc_ci95": bootstrap_diff_ci(cb, ca),
                             "mcnemar_exact": mcnemar_exact(pa, pb, y),
                             "order_avg_agree": sum(1 for a, b in zip(pa, pb) if a == b) / len(pa)}
                    gvec[(fam, lang, frame)] = [b - a for a, b in zip(ca, cb)]
                worst = min(adj, key=lambda a: a["d_acc"]) if adj else None
                ana[f"{fam}/{lang}/{frame}"] = {
                    "ladder": rows, "adjacent_drops": adj, "worst_adjacent": worst,
                    "cliff_contrast": cliff,
                    "rungs_ci_upper_at_or_below_chance":
                        [r["rung"] for r in rows if r["ci"][1] <= 0.5],
                }
    # cross-language: same items, so the cliff DEPTH can be differenced item-for-item
    for (fam, lang, frame), g in gvec.items():
        ref = gvec.get((fam, "id", frame))
        if ref is None or lang == "id" or len(ref) != len(g):
            continue
        d = sum(a - b for a, b in zip(g, ref)) / len(g)
        ana[f"{fam}/{lang}/{frame}"]["cliff_depth_vs_id"] = {
            "d_acc_lang": sum(g) / len(g), "d_acc_id": sum(ref) / len(ref),
            "depth_minus_id": d, "depth_minus_id_ci95": paired_diff_ci(g, ref)}
    return ana


def score_cell(model: dict, lang: str, frame: str, lines: list[str], n_items: int,
               labels: list[int], threads: int, n_ctx: int):
    """One (model x language x frame) cell: 2*n_items prompt lines -> metric bundle.

    Each cell is an independent gguf_cand_lp subprocess with --threads unchanged, so
    running cells concurrently is a scheduling choice, not a protocol change.
    """
    t0 = time.time()
    rows = tool_rows(model["path"], lines, threads, n_ctx)
    if len(rows) != 2 * n_items:
        raise SystemExit(f"[fatal] {model['name']} {lang}/{frame}: {len(rows)} rows, want {2*n_items}")
    pf, mf = letter_scores(rows[:n_items])
    ps, ms = letter_scores(rows[n_items:])
    rec, it = metrics(pf, mf, ps, ms, labels)
    key = (model["name"], lang, frame)
    return key, rec, it, labels, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--langs", default=",".join(LANGS_PRIMARY))
    ap.add_argument("--frames", default="id,en")
    ap.add_argument("--families", default="n06ctl,base", help="from " + ",".join(RUNGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1, help="concurrent cells (subprocesses)")
    ap.add_argument("--tokenizer", default=f"{ROOT}/ckpts/teacher")
    ap.add_argument("--outdir", default=f"{ROOT}/xtile")
    ap.add_argument("--authorise-test-read", default="")
    ap.add_argument("--tokenize-only", action="store_true")
    ap.add_argument("--anchor-check", action="store_true", help="only run the layout regression")
    ap.add_argument("--skip-anchor", action="store_true")
    ap.add_argument("--print-prereg-hash", action="store_true")
    args = ap.parse_args()

    if args.print_prereg_hash:
        print(prereg_hash())
        return
    if args.anchor_check:
        anchor_check(args.threads)
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    frames = [x.strip() for x in args.frames.split(",") if x.strip()]
    fams = [x.strip() for x in args.families.split(",") if x.strip()]

    if args.split == "test":
        if args.authorise_test_read != TEST_READ_PHRASE:
            raise SystemExit(f"[refused] test is sealed; pass --authorise-test-read {TEST_READ_PHRASE}")
        prior = sorted(outdir.glob("xtile_test*.json"))
        if prior:
            raise SystemExit(f"[refused] the single test read is already recorded: {prior[0]}")

    items = {l: load_xcopa(l, args.split) for l in langs}
    for l in langs:
        if args.limit:
            items[l] = items[l][: args.limit]
    idx_sig = {l: [e["idx"] for e in items[l]] for l in langs}
    aligned = all(idx_sig[l] == idx_sig[langs[0]] for l in langs)

    prompts: dict[tuple[str, str], list[str]] = {}
    for l in langs:
        for fr in frames:
            pairs = []
            for ex in items[l]:
                pair = xtile_pair(ex, fr)
                if (l, fr) == ("id", "id"):
                    assert pair == prompt_pair(ex, question_word(ex)), \
                        "the id-frame template drifted from the published protocol"
                pairs.append(pair)
            # eval_gguf / board_letter_stage layout: every as-written prompt, then every
            # swapped one. letter_scores splits the rows at n_items, so interleaving
            # (a,b,a,b) silently pairs rows with the wrong labels and flattens acc to chance.
            prompts[(l, fr)] = [p for p, _ in pairs] + [s for _, s in pairs]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lens = {}
    for (l, fr), lines in prompts.items():
        v = [len(tok.encode(s, add_special_tokens=False)) for s in lines]
        lens[f"{l}/{fr}"] = {"mean": sum(v) / len(v), "max": max(v)}
    mx = max(v["max"] for v in lens.values())
    n_ctx = 128
    while n_ctx < mx + 2:  # frozen rule: smallest power of two that fits every prompt
        n_ctx *= 2

    meta = {"protocol": PROTOCOL, "prereg_sha256": prereg_hash(), "split": args.split,
            "langs": langs, "frames": frames, "families": fams, "limit": args.limit,
            "n_items": {l: len(items[l]) for l in langs},
            "labels": {l: [e["label"] for e in items[l]] for l in langs},
            "idx": idx_sig, "idx_aligned_across_langs": aligned,
            "cands": CANDS, "n_ctx": n_ctx, "threads": args.threads,
            "prompt_tokens": lens,
            "prompt_lines": {f"{l}/{fr}": len(v) for (l, fr), v in prompts.items()},
            "prompt_file_sha256": {f"{l}/{fr}": hashlib.sha256(
                "\n".join(esc(x) for x in v).encode()).hexdigest()
                for (l, fr), v in prompts.items()},
            "started": datetime.now(timezone.utc).isoformat()}
    if args.tokenize_only:
        print(json.dumps(meta, ensure_ascii=False, indent=1))
        return

    # the layout regression gates every scoring run, not only the explicit --anchor-check
    meta["anchor_acc"] = None if args.skip_anchor else anchor_check(args.threads)

    models = {f: model_list(f) for f in fams}
    todo = [(m, l, fr) for fam in fams for m in models[fam] for l in langs for fr in frames]
    cells: dict = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(score_cell, m, l, fr, prompts[(l, fr)], len(items[l]),
                          [e["label"] for e in items[l]], args.threads, n_ctx) for (m, l, fr) in todo]
        for fu in as_completed(futs):
            key, rec, it, y, dt = fu.result()
            cells[key] = {"rec": rec, "items": it, "labels": y}
            print(f"[{time.strftime('%F %T')}] {key[0]:16s} {key[1]}/{key[2]} "
                  f"acc={rec['acc_order_averaged']:.3f} auc={rec['auc_order_averaged']:.3f} "
                  f"cons={rec['order_consistency']:.3f} ci={rec['ci_order_averaged']} "
                  f"{dt:.0f}s  ({len(cells)}/{len(todo)})", flush=True)

    out = {"meta": meta, "models": models,
           "conditions": {f"{k[0]}/{k[1]}/{k[2]}": v["rec"] for k, v in cells.items()},
           "items": {f"{k[0]}/{k[1]}/{k[2]}": v["items"] for k, v in cells.items()},
           "analysis": ladder_analysis(cells, models),
           "finished": datetime.now(timezone.utc).isoformat()}
    stem = f"xtile_{args.split}" + (f"_limit{args.limit}" if args.limit else "")
    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {path}", flush=True)
    print("XTILE_DONE", prereg_hash()[:12], flush=True)


if __name__ == "__main__":
    main()

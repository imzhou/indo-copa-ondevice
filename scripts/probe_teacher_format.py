#!/usr/bin/env python3
"""Teacher-side protocol probes for XCOPA-id under letter log-prob scoring.

Three pre-registered questions:
  1. How much of the teacher's 3-shot gain is prompt-format anchoring rather than
     label-conditioned in-context learning?  (directive-only, shuffled-label and
     placeholder-content arms)
  2. How stable is the 3-shot number under demo resampling?
  3. Does the raw-completion protocol differ from the chat-template protocol?

Only the teacher is scored. No student weights are touched and no model selection
is performed on the test split.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = list("ABCD")


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def copa_prompt(ex, q: str) -> str:
    return (
        f"Premis: {ex['premise']}\n"
        f"Apa {q} yang paling mungkin?\n"
        f"A. {ex['choice1']}\n"
        f"B. {ex['choice2']}\n"
        f"Jawaban:"
    )


def question_word(ex) -> str:
    return "penyebab" if ex["question"] == "cause" else "akibat"


def demo_block(ex, letter: str) -> str:
    return copa_prompt(ex, question_word(ex)) + f" {letter}"


def placeholder_block(q: str, letter: str) -> str:
    return (
        f"Premis: ...\n"
        f"Apa {q} yang paling mungkin?\n"
        f"A. ...\n"
        f"B. ...\n"
        f"Jawaban: {letter}"
    )


def shots_prefix(demos: list[str]) -> str:
    return "\n\n".join(demos) + "\n\n" if demos else ""


@torch.no_grad()
def score_pairs(model, tok, pairs, device, bs: int = 8) -> dict:
    """pairs: list of (idx, prompt, choice). Returns idx -> mean logprob."""
    out: dict[int, float] = {}
    groups: dict[tuple, list] = defaultdict(list)
    for idx, prompt, choice in pairs:
        cids = tuple(encode(tok, choice))
        if not cids:
            out[idx] = -1e9
            continue
        groups[cids].append((idx, prompt))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for cids, items in groups.items():
        k = len(cids)
        for j in range(0, len(items), bs):
            chunk = items[j : j + bs]
            seqs = [encode(tok, p) + list(cids) for _, p in chunk]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, L - len(s) :] = torch.tensor(s)
                attn[r, L - len(s) :] = 1
            ids = ids.to(device)
            attn = attn.to(device)
            try:
                logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=k + 1).logits
            except TypeError:
                logits = model(input_ids=ids, attention_mask=attn).logits[:, -(k + 1) :]
            lp = F.log_softmax(logits.float(), dim=-1)
            for r, (idx, _) in enumerate(chunk):
                s = 0.0
                for t in range(k):
                    s += float(lp[r, t, cids[t]])
                out[idx] = s / k
    return out


@torch.no_grad()
def score_condition(model, tok, prompts, device, bs: int = 8):
    """Letter scoring identical to scripts/eval_nlu.py: max over ' X' and 'X'."""
    pairs = []
    for i, prompt in enumerate(prompts):
        for li in range(2):
            pairs.append((4 * i + 2 * li, prompt, f" {LETTERS[li]}"))
            pairs.append((4 * i + 2 * li + 1, prompt, LETTERS[li]))
    lps = score_pairs(model, tok, pairs, device, bs=bs)
    preds, margins = [], []
    for i in range(len(prompts)):
        scores = []
        for li in range(2):
            a = max(lps[4 * i + 2 * li], lps[4 * i + 2 * li + 1])
            scores.append(a)
        preds.append(int(scores[1] > scores[0]))
        margins.append(scores[1] - scores[0])
    return preds, margins


def mcnemar_exact(c1: list[int], c2: list[int], y: list[int]):
    b = sum(1 for a, c, t in zip(c1, c2, y) if a == t and c != t)
    c = sum(1 for a, cc, t in zip(c1, c2, y) if a != t and cc == t)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(b, c)
    tail = sum(_comb(n, i) for i in range(0, k + 1)) / (2**n)
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def _comb(n: int, k: int) -> int:
    import math

    return math.comb(n, k)


def bootstrap_ci(correct: list[int], n_boot: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    n = len(correct)
    accs = []
    for _ in range(n_boot):
        accs.append(sum(correct[rng.randrange(n)] for _ in range(n)) / n)
    accs.sort()
    return [accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)]]


def bootstrap_diff_ci(c1: list[int], c2: list[int], n_boot: int = 2000, seed: int = 1):
    rng = random.Random(seed)
    n = len(c1)
    d = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(c1[i] - c2[i] for i in idx) / n)
    d.sort()
    return [d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]]


def summarize(name, preds, margins, y):
    correct = [int(p == t) for p, t in zip(preds, y)]
    acc = sum(correct) / len(correct)
    lo, hi = bootstrap_ci(correct)
    absm = sorted(abs(m) for m in margins)
    near = sum(1 for m in absm if m < 0.5) / len(absm)
    return {
        "name": name,
        "n": len(correct),
        "acc": acc,
        "acc_ci95": [lo, hi],
        "margin_median_abs": statistics.median(absm),
        "frac_abs_margin_lt_0.5": near,
        "correct": correct,
        "preds": preds,
        "margins": margins,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/path/to/data/ckpts/teacher_instruct")
    ap.add_argument("--out", default="/path/to/data/data/probe_format/probe.json")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("cambridgeltl/xcopa", "id")
    val, test = list(ds["validation"]), list(ds["test"])
    y = [int(ex["label"]) for ex in test]
    print(f"val={len(val)} test={len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(args.device)
        .eval()
    )

    def base_prompts(prefix: str = "", directive: str | None = None, where: str = "q") -> list[str]:
        out = []
        for ex in test:
            q = question_word(ex)
            if directive and where == "q":
                body = (
                    f"Premis: {ex['premise']}\n"
                    f"Apa {q} yang paling mungkin?\n"
                    f"{directive}\n"
                    f"A. {ex['choice1']}\n"
                    f"B. {ex['choice2']}\n"
                    f"Jawaban:"
                )
            elif directive and where == "top":
                body = directive + "\n\n" + copa_prompt(ex, q)
            else:
                body = copa_prompt(ex, q)
            out.append(prefix + body)
        return out

    def chat_prompts(prefix: str = "") -> list[str]:
        out = []
        for ex in test:
            content = prefix + copa_prompt(ex, question_word(ex))
            msgs = [{"role": "user", "content": content}]
            try:
                out.append(
                    tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        return out

    def val_prefix(k: int, rng=None, flip: bool = False) -> str:
        idxs = list(range(len(val))) if rng is None else [rng.randrange(len(val)) for _ in range(k)]
        if rng is None:
            idxs = idxs[:k]
        demos = []
        for i in idxs:
            ex = val[i]
            lab = int(ex["label"])
            if flip:
                lab = 1 - lab
            demos.append(demo_block(ex, LETTERS[lab]))
        return shots_prefix(demos)

    conditions = {
        "0shot": base_prompts(),
        "dir_q": base_prompts(directive="Jawab hanya dengan satu huruf: A atau B."),
        "dir_strict": base_prompts(
            directive="Jawab hanya dengan satu huruf, A atau B, tanpa penjelasan."
        ),
        "dir_top": base_prompts(
            directive="Jawab hanya dengan satu huruf: A atau B.", where="top"
        ),
        "1shot": base_prompts(val_prefix(1)),
        "2shot": base_prompts(val_prefix(2)),
        "3shot": base_prompts(val_prefix(3)),
        "5shot": base_prompts(val_prefix(5)),
        "8shot": base_prompts(val_prefix(8)),
        "3shot_shuf": base_prompts(val_prefix(3, flip=True)),
        "3shot_fmt": base_prompts(
            shots_prefix(
                [
                    placeholder_block(question_word(val[0]), "A"),
                    placeholder_block(question_word(val[1]), "B"),
                    placeholder_block(question_word(val[2]), "B"),
                ]
            )
        ),
        "chat0": chat_prompts(),
        "chat3": chat_prompts(val_prefix(3)),
    }

    results = {}
    for name, prompts in conditions.items():
        preds, margins = score_condition(model, tok, prompts, args.device, bs=args.bs)
        results[name] = summarize(name, preds, margins, y)
        print(f"{name:12s} acc={results[name]['acc']:.3f} "
              f"ci={results[name]['acc_ci95'][0]:.3f}-{results[name]['acc_ci95'][1]:.3f}", flush=True)

    draws = []
    rng = random.Random(1234)
    for d in range(args.draws):
        prefix = val_prefix(3, rng=rng)
        preds, margins = score_condition(model, tok, base_prompts(prefix), args.device, bs=args.bs)
        acc = sum(int(p == t) for p, t in zip(preds, y)) / len(y)
        draws.append(acc)
        print(f"draw {d:02d} acc={acc:.3f}", flush=True)

    names = list(results)
    mcn = {}
    for a, b in [
        ("0shot", "dir_q"),
        ("0shot", "dir_strict"),
        ("0shot", "3shot"),
        ("3shot", "3shot_shuf"),
        ("3shot", "3shot_fmt"),
        ("0shot", "chat0"),
        ("3shot", "chat3"),
        ("3shot", "5shot"),
    ]:
        if a in results and b in results:
            mcn[f"{a}_vs_{b}"] = mcnemar_exact(results[a]["preds"], results[b]["preds"], y)
    diffs = {}
    for a, b in [("3shot", "dir_q"), ("3shot", "0shot"), ("3shot", "3shot_fmt")]:
        if a in results and b in results:
            lo, hi = bootstrap_diff_ci(results[a]["correct"], results[b]["correct"])
            diffs[f"{a}_minus_{b}"] = [lo, hi]

    payload = {
        "model": args.model,
        "split": "xcopa-id test n=500, letter logprob, max(' X','X')",
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk not in ("correct", "preds", "margins")}
                       for k, v in results.items()},
        "items": {k: {"preds": v["preds"], "margins": v["margins"]} for k, v in results.items()},
        "labels": y,
        "demo_draws": {
            "n_draws": args.draws,
            "accs": draws,
            "mean": statistics.mean(draws),
            "sd": statistics.pstdev(draws),
            "min": min(draws),
            "max": max(draws),
        },
        "mcnemar": mcn,
        "bootstrap_diff_ci95": diffs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("\n==== SUMMARY ====")
    for k in names:
        print(f"{k:12s} {results[k]['acc']*100:6.2f}  "
              f"[{results[k]['acc_ci95'][0]*100:5.1f},{results[k]['acc_ci95'][1]*100:5.1f}]")
    print("demo draws: mean=%.3f sd=%.3f min=%.3f max=%.3f" % (
        statistics.mean(draws), statistics.pstdev(draws), min(draws), max(draws)))
    for k, v in mcn.items():
        print(f"mcnemar {k}: b={v['b']} c={v['c']} p={v['p']:.4g}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

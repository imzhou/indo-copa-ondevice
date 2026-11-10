#!/usr/bin/env python3
"""Offline teacher letter-agree: 0-shot vs 3-shot. Shots from gold COPA-train only, never XCOPA val/test."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def mc_prompt(row: dict) -> str:
    q = "penyebab" if row.get("question") == "cause" else "akibat"
    return (
        "Premis: " + row["premise"] + "\n"
        "Apa " + q + " yang paling mungkin?\n"
        "A. " + row["choice1"] + "\n"
        "B. " + row["choice2"] + "\n"
        "Jawaban:"
    )


def filled(row: dict) -> str:
    letter = " A" if int(row["label"]) == 0 else " B"
    return mc_prompt(row) + letter


def pack_prompt(row: dict, shots: list) -> str:
    if not shots:
        return mc_prompt(row)
    return "\n\n".join(filled(s) for s in shots) + "\n\n" + mc_prompt(row)


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("premise") and r.get("choice1") and r.get("choice2") and r.get("label") in (0, 1, "0", "1"):
                r["label"] = int(r["label"])
                rows.append(r)
    return rows


def letter_ids(tok):
    a = tok.encode(" A", add_special_tokens=False) or tok.encode("A", add_special_tokens=False)
    b = tok.encode(" B", add_special_tokens=False) or tok.encode("B", add_special_tokens=False)
    return a[0], b[0]


@torch.no_grad()
def pred_letter(model, tok, prompt: str, a0: int, b0: int, device: str) -> int:
    ids = tok.encode(prompt, add_special_tokens=False)[-2048:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    lp = F.log_softmax(model(input_ids=x).logits[0, -1].float(), dim=-1)
    return 0 if float(lp[a0]) >= float(lp[b0]) else 1


def score_split(model, tok, rows, shots_pool, a0, b0, n_shot: int, device: str, tag: str) -> dict:
    rng = random.Random(0)
    ok0 = okk = 0
    n = len(rows)
    for i, row in enumerate(rows):
        gold = int(row["label"])
        pred0 = pred_letter(model, tok, pack_prompt(row, []), a0, b0, device)
        ok0 += int(pred0 == gold)
        shots = []
        if n_shot and shots_pool:
            guard = 0
            while len(shots) < n_shot and guard < 40:
                guard += 1
                s = shots_pool[rng.randrange(len(shots_pool))]
                if s is row:
                    continue
                if s.get("premise") == row.get("premise") and s.get("choice1") == row.get("choice1"):
                    continue
                shots.append(s)
        predk = pred_letter(model, tok, pack_prompt(row, shots), a0, b0, device)
        okk += int(predk == gold)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(
                "%s %d/%d 0shot=%.3f %dshot=%.3f"
                % (tag, i + 1, n, ok0 / (i + 1), n_shot, okk / (i + 1)),
                flush=True,
            )
    rep = {
        "tag": tag,
        "n": n,
        "acc_0shot": ok0 / max(1, n),
        "acc_%dshot" % n_shot: okk / max(1, n),
    }
    print("AGREE", json.dumps(rep), flush=True)
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--gold", required=True, help="copa_id_clean.jsonl")
    p.add_argument("--synth", required=True, help="copa_id_real.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n-shot", type=int, default=3)
    args = p.parse_args()

    gold_all = load_jsonl(args.gold)
    gold_uniq = [r for r in gold_all if r.get("source") == "copa_en_train"]
    synth_uniq = [r for r in load_jsonl(args.synth) if r.get("source") == "copa_real_gen"]
    print("gold_uniq", len(gold_uniq), "gold_all", len(gold_all), "synth_uniq", len(synth_uniq), flush=True)

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    a0, b0 = letter_ids(tok)
    print("letter tokens", a0, b0, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    report = {
        "teacher": args.teacher,
        "n_shot": args.n_shot,
        "shots": "copa_en_train unique, never xcopa val/test",
        "gold": score_split(model, tok, gold_uniq, gold_uniq, a0, b0, args.n_shot, "cuda", "gold"),
        "synth": score_split(model, tok, synth_uniq, gold_uniq, a0, b0, args.n_shot, "cuda", "synth"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("WROTE", args.out, flush=True)
    g3 = report["gold"]["acc_%dshot" % args.n_shot]
    s3 = report["synth"]["acc_%dshot" % args.n_shot]
    print("VERDICT gold_3shot=%.3f synth_3shot=%.3f" % (g3, s3), flush=True)
    if s3 >= 0.65:
        print("ROUTE_ALIVE synth 3-shot agree >= 0.65, worth 0-shot student KD", flush=True)
    else:
        print("ROUTE_DEAD synth 3-shot agree < 0.65, do not KD this set", flush=True)


if __name__ == "__main__":
    main()

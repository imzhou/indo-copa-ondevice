#!/usr/bin/env python3
"""Indo NLU eval. MC uses letter scoring (A/B/C/D), not full-choice logprob."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from id_vocab import load_any_tokenizer

LETTERS = list("ABCD")


def load_model(path: str, device: str):
    tok, mapped = load_any_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    return tok, mapped, model


def encode_ids(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


@torch.no_grad()
def continuation_logprob(model, tok, prompt: str, choice: str, device: str) -> float:
    p_ids = encode_ids(tok, prompt)
    c_ids = encode_ids(tok, choice)
    if not c_ids:
        return -1e9
    ids = p_ids + c_ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    lp = F.log_softmax(model(input_ids=x).logits[0].float(), dim=-1)
    s = 0.0
    off = len(p_ids)
    for i, tid in enumerate(c_ids):
        s += float(lp[off + i - 1, tid])
    return s / max(1, len(c_ids))


def score_letters(model, tok, prompt: str, n_choices: int, device: str) -> int:
    """Score  A /  B / ... as the continuation (standard LM MC)."""
    variants = [f" {LETTERS[i]}" for i in range(n_choices)]
    # also try unprefixed "A" if space-letter is OOV-ish
    lps = []
    for v in variants:
        lp = continuation_logprob(model, tok, prompt, v, device)
        lp2 = continuation_logprob(model, tok, prompt, v.strip(), device)
        lps.append(max(lp, lp2))
    return int(max(range(len(lps)), key=lambda i: lps[i]))


def xcopa_prompt(ex, shots: list[str]) -> str:
    q = "penyebab" if ex["question"] == "cause" else "akibat"
    body = (
        f"Premis: {ex["premise"]}\n"
        f"Apa {q} yang paling mungkin?\n"
        f"A. {ex["choice1"]}\n"
        f"B. {ex["choice2"]}\n"
        f"Jawaban:"
    )
    if shots:
        return "\n\n".join(shots) + "\n\n" + body
    return body


def eval_xcopa(model, tok, device, n_shot: int = 3) -> dict:
    from datasets import load_dataset

    ds = load_dataset("cambridgeltl/xcopa", "id")
    val, test = ds["validation"], ds["test"]
    shots = []
    for ex in val:
        if len(shots) >= n_shot:
            break
        letter = LETTERS[int(ex["label"])]
        shots.append(xcopa_prompt(ex, []) + f" {letter}")
    n, ok = 0, 0
    for ex in test:
        prompt = xcopa_prompt(ex, shots)
        pred = score_letters(model, tok, prompt, 2, device)
        ok += int(pred == int(ex["label"]))
        n += 1
        if n % 100 == 0:
            print(f"  xcopa {n}/{len(test)} acc={ok/n:.3f}", flush=True)
    return {"n": n, "acc": ok / max(1, n)}


def eval_belebele(model, tok, device, limit: int = 0) -> dict:
    from datasets import load_dataset

    ds = load_dataset("facebook/belebele", "ind_Latn", split="test")
    n, ok = 0, 0
    for ex in ds:
        ch = [ex[f"mc_answer{i}"] for i in range(1, 5)]
        gold = int(ex["correct_answer_num"]) - 1
        lines = "\n".join(f"{LETTERS[i]}. {ch[i]}" for i in range(4))
        prompt = (
            f"Teks: {ex["flores_passage"]}\n"
            f"Pertanyaan: {ex["question"]}\n"
            f"{lines}\n"
            f"Jawaban:"
        )
        pred = score_letters(model, tok, prompt, 4, device)
        ok += int(pred == gold)
        n += 1
        if limit and n >= limit:
            break
        if n % 100 == 0:
            print(f"  belebele {n} acc={ok/n:.3f}", flush=True)
    return {"n": n, "acc": ok / max(1, n)}


def gold_hit(gold: str, gen: str) -> bool:
    g = re.sub(r"[*_`]+", "", gen).lower()
    a = gold.lower()
    if a in g or g[:50] in a:
        return True
    keys = {
        "jakarta", "jawa", "borobudur", "magelang", "1945", "soekarno",
        "hatta", "pancasila", "nusantara", "garuda", "rendang", "batik",
        "yogyakarta", "surabaya", "bandung", "medan", "bali",
    }
    gw = set(re.findall(r"[a-z0-9]+", a))
    need = [k for k in keys if k in gw] or [w for w in gw if len(w) > 4]
    if not need:
        return False
    return sum(k in g for k in need) >= max(1, (len(need) + 1) // 2)


def eval_gold(model, tok, mapped, device, gold_path: str, max_new: int = 80) -> dict:
    rows = [json.loads(l) for l in Path(gold_path).read_text().splitlines() if l.strip()]
    n, ok = 0, 0
    samples = []
    for r in rows:
        q, a = r["instruction"], r["output"]
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = q + "\nJawaban: "
        ids = encode_ids(tok, text)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][len(ids) :].tolist()
        gen = tok.decode(gen_ids, skip_special_tokens=True)
        gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.S).strip()
        hit = gold_hit(a, gen)
        ok += int(hit)
        n += 1
        samples.append({"q": q, "gold": a, "gen": gen.replace("\n", " / ")[:240], "hit": bool(hit)})
    return {"n": n, "acc": ok / max(1, n), "samples": samples}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gold", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", default="xcopa,gold", help="comma: xcopa,belebele,gold")
    p.add_argument("--belebele-limit", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--n-shot", type=int, default=3)
    args = p.parse_args()
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    print(f"eval {args.model} tasks={sorted(tasks)}", flush=True)
    tok, mapped, model = load_model(args.model, args.device)
    print(f"mapped_vocab={mapped}", flush=True)
    report = {"model": args.model, "mapped": mapped}
    if "xcopa" in tasks:
        report["xcopa_id"] = eval_xcopa(model, tok, args.device, n_shot=args.n_shot)
        report["n_shot"] = args.n_shot
        print("XCOPA-id", report["xcopa_id"], flush=True)
    if "belebele" in tasks:
        report["belebele_ind"] = eval_belebele(model, tok, args.device, args.belebele_limit)
        print("Belebele ind_Latn", report["belebele_ind"], flush=True)
    if "gold" in tasks and args.gold:
        report["gold_qa"] = eval_gold(model, tok, mapped, args.device, args.gold)
        print("gold_qa", {k: report["gold_qa"][k] for k in ("n", "acc")}, flush=True)
        for s in report["gold_qa"].get("samples", []):
            print("  Q {q}\n  G {gold}\n  A {gen}\n  hit={hit}".format(**s), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("wrote", args.out, flush=True)
    slim = {k: v for k, v in report.items() if k != "gold_qa"}
    if "gold_qa" in report:
        slim["gold_qa"] = {k: report["gold_qa"][k] for k in ("n", "acc")}
    print("EVAL_NLU_DONE", json.dumps(slim, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

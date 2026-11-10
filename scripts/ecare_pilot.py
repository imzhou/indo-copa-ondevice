#!/usr/bin/env python3
"""e-CARE pilot: translate a stratified sample to Indonesian, probe Qwen3-0.6B-Instruct
0-shot/3-shot letter agreement vs gold labels. Gate >=65% (3-shot) before committing
to full translation + KD. Includes a COPA-train positive control (expect ~67% 3-shot)
to validate that this translation+scoring pipeline reproduces the known gold result.

Output: {out_dir}/ecare_pilot.json (report) and ecare_pilot_id.jsonl (translated items
with per-item margins; reusable as training data if the gate passes).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

D = "/path/to/data"
QID = {"cause": "penyebab", "effect": "akibat"}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_ds(*a, **k):
    from datasets import load_dataset

    try:
        return load_dataset(*a, trust_remote_code=True, **k)
    except TypeError:
        return load_dataset(*a, **k)


def block(premise, askfor, c1, c2, answer=None):
    q = QID.get(askfor, askfor)
    s = f"Premis: {premise}\nApa {q} yang paling mungkin?\nA. {c1}\nB. {c2}\nJawaban:"
    if answer is not None:
        s += f" {answer}"
    return s


def translate_prompt(premise, h1, h2, strict=False):
    extra = " Remember: output exactly three lines (Premis/A/B), Indonesian only." if strict else ""
    return (
        "Translate the following three English sentences into natural, accurate Indonesian. "
        "Preserve the causal meaning exactly. Output ONLY three lines in this exact format, "
        "nothing else:\n"
        "Premis: <Indonesian translation of the premise>\n"
        "A: <Indonesian translation of choice 1>\n"
        "B: <Indonesian translation of choice 2>\n\n"
        f"Premis: {premise}\nA: {h1}\nB: {h2}" + extra
    )


def parse_translation(text):
    got = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(Premis|A|B)\s*:\s*(.+?)\s*$", line)
        if m:
            got[m.group(1)] = m.group(2)
    if all(k in got and len(got[k]) > 2 for k in ("Premis", "A", "B")):
        return got["Premis"], got["A"], got["B"]
    return None


def apply_template(tok, user):
    messages = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(
            messages, enable_thinking=False, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def translate_batch(tok, model, items, max_new=220, strict=False):
    texts = [
        apply_template(tok, translate_prompt(it["premise_en"], it["h1_en"], it["h2_en"], strict))
        for it in items
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.cuda() for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id
    )
    res = []
    for i, seq in enumerate(out):
        plen = int(enc["attention_mask"][i].sum().item())
        gen = tok.decode(seq[plen:], skip_special_tokens=True)
        res.append(parse_translation(gen))
    return res


@torch.no_grad()
def letter_margins(model, tok, prompts, chunk=4):
    """scoreA - scoreB per prompt; each letter scored as max mean-token-logprob over
    ' X' / 'X' continuations (MEMO eval protocol)."""
    seqs, meta = [], []
    for pi, p in enumerate(prompts):
        pid = tok(p, add_special_tokens=False)["input_ids"]
        for L in ("A", "B"):
            for cand in (f" {L}", L):
                cid = tok(cand, add_special_tokens=False)["input_ids"]
                seqs.append(pid + cid)
                meta.append((pi, L, len(pid), cid))
    best = {}
    for s in range(0, len(seqs), chunk):
        batch, m = seqs[s : s + chunk], meta[s : s + chunk]
        maxlen = max(len(x) for x in batch)
        ids = torch.full((len(batch), maxlen), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, x in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            att[i, : len(x)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
        lsm = torch.log_softmax(logits, dim=-1)
        for i, (pi, L, lp, cid) in enumerate(m):
            pos = torch.arange(lp - 1, lp - 1 + len(cid))
            lpv = lsm[i, pos, torch.tensor(cid)].mean().item()
            key = (pi, L)
            if key not in best or lpv > best[key]:
                best[key] = lpv
    return [best[(pi, "A")] - best[(pi, "B")] for pi in range(len(prompts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecare-train", default=f"{D}/data/ecare_train.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--control-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--translator", default=f"{D}/ckpts/teacher_7b")
    ap.add_argument("--scorer", default=f"{D}/ckpts/teacher_instruct")
    ap.add_argument("--out-dir", default=f"{D}/data/ecare_pilot")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="reuse saved translations, skip 7B")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.ecare_train)
    by = {"cause": [], "effect": []}
    for r in rows:
        if r.get("ask-for") in by:
            by[r["ask-for"]].append(r)
    sample = []
    for k in ("cause", "effect"):
        pool = by[k][:]
        rng.shuffle(pool)
        sample.extend(pool[: args.n // 2])
    items = [
        {
            "premise": r["premise"],
            "ask-for": r["ask-for"],
            "hypothesis1": r["hypothesis1"],
            "hypothesis2": r["hypothesis2"],
            "label": int(r["label"]),
            "src": "ecare",
            "index": r.get("index", ""),
        }
        for r in sample
    ]
    control = []
    if args.control_n > 0:
        copa = load_ds("aps/super_glue", "copa", split="train")
        idx = list(range(len(copa)))
        rng.shuffle(idx)
        for i in idx[: args.control_n]:
            ex = copa[i]
            control.append(
                {
                    "premise": ex["premise"],
                    "ask-for": ex["question"],
                    "hypothesis1": ex["choice1"],
                    "hypothesis2": ex["choice2"],
                    "label": int(ex["label"]),
                    "src": "copa_control",
                    "index": f"copa-{i}",
                }
            )
    items.extend(control)
    for it in items:
        it["premise_en"], it["h1_en"], it["h2_en"] = (
            it["premise"],
            it["hypothesis1"],
            it["hypothesis2"],
        )
    print(f"pilot items: {len(items)} (ecare {len(sample)}, copa control {len(control)})", flush=True)

    ckpt = out / "ecare_pilot_translated.jsonl"
    if args.resume and ckpt.exists():
        items = load_jsonl(ckpt)
        failed = sum(1 for it in items if it.get("drop"))
        print(f"resumed {len(items)} translated items, failed={failed}", flush=True)
    else:
        cache = {}
        if ckpt.exists():
            for it in load_jsonl(ckpt):
                if not it.get("drop"):
                    cache[(it["premise_en"], it["h1_en"], it["h2_en"])] = (
                        it["premise"], it["hypothesis1"], it["hypothesis2"]
                    )
        pending = []
        for it in items:
            tr = cache.get((it["premise_en"], it["h1_en"], it["h2_en"]))
            if tr:
                it["premise"], it["hypothesis1"], it["hypothesis2"] = tr
            else:
                pending.append(it)
        failed = 0
        if pending:
            tok = AutoTokenizer.from_pretrained(args.translator)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.translator, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda"
            ).eval()
            for s in range(0, len(pending), args.batch):
                batch = pending[s : s + args.batch]
                res = translate_batch(tok, model, batch)
                retry = [it for it, tr in zip(batch, res) if tr is None]
                if retry:
                    res2 = translate_batch(tok, model, retry, strict=True)
                    rmap = {id(it): tr for it, tr in zip(retry, res2)}
                    res = [rmap.get(id(it), tr) if tr is None else tr for it, tr in zip(batch, res)]
                for it, tr in zip(batch, res):
                    if tr is None:
                        it["drop"] = True
                        failed += 1
                    else:
                        it["premise"], it["hypothesis1"], it["hypothesis2"] = tr
                if (s // args.batch) % 10 == 0:
                    print(f"  translated {min(s + args.batch, len(pending))}/{len(pending)} failed={failed}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        print(f"translation done, failed={failed}, cached={len(items) - len(pending)}", flush=True)
        write_jsonl(ckpt, items)

    stok = AutoTokenizer.from_pretrained(args.scorer)
    smodel = AutoModelForCausalLM.from_pretrained(args.scorer, torch_dtype=torch.bfloat16).cuda().eval()
    xv = load_ds("cambridgeltl/xcopa", "id", split="validation")
    demos = []
    for i in range(3):
        ex = xv[i]
        letter = "A" if ex["label"] == 0 else "B"
        demos.append(block(ex["premise"], ex["question"], ex["choice1"], ex["choice2"], letter))
    demo_str = "\n\n".join(demos) + "\n\n"

    kept = [it for it in items if not it.get("drop")]
    p0 = [block(it["premise"], it["ask-for"], it["hypothesis1"], it["hypothesis2"]) for it in kept]
    p3 = [demo_str + p for p in p0]
    m0 = letter_margins(smodel, stok, p0)
    m3 = letter_margins(smodel, stok, p3)
    for it, a, b in zip(kept, m0, m3):
        it["margin_0shot"], it["margin_3shot"] = a, b
        it["pred_0shot"] = 0 if a > 0 else 1
        it["pred_3shot"] = 0 if b > 0 else 1
    print("scoring done", flush=True)

    def agree(sub, key):
        k = sum(1 for it in sub if it[key] == it["label"])
        n = len(sub)
        return {"n": n, "agree": k, "acc": k / n if n else 0.0, "ci95": wilson(k, n)}

    report = {"gate_threshold": 0.65, "seed": args.seed, "translation_failed": failed}
    for name, sub in (
        ("ecare", [it for it in kept if it["src"] == "ecare"]),
        ("copa_control", [it for it in kept if it["src"] == "copa_control"]),
    ):
        rep = {}
        for shot, key in (("0shot", "pred_0shot"), ("3shot", "pred_3shot")):
            rep[shot] = agree(sub, key)
        for q in ("cause", "effect"):
            rep[f"3shot_{q}"] = agree([it for it in sub if it["ask-for"] == q], "pred_3shot")
        report[name] = rep

    e3 = report["ecare"]["3shot"]["acc"]
    c3 = report["copa_control"]["3shot"]["acc"]
    control_ok = c3 >= 0.60
    if e3 >= 0.65 and control_ok:
        gate, note = "GO", "proceed to full train translation + dev val translation, then KD"
    elif e3 >= 0.62 and control_ok:
        gate, note = "GREY", "extend pilot to n=1500 before deciding (CI still wide)"
    else:
        gate, note = "STOP", "do not KD e-CARE; control_ok=%s" % control_ok
    report["gate"] = gate
    report["gate_note"] = note

    with (out / "ecare_pilot.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_jsonl(out / "ecare_pilot_id.jsonl", kept)

    print("\n=== e-CARE pilot report ===", flush=True)
    for name in ("ecare", "copa_control"):
        for shot in ("0shot", "3shot"):
            r = report[name][shot]
            lo, hi = r["ci95"]
            print(f"{name:13s} {shot}: {r['agree']}/{r['n']} = {r['acc']:.3f}  CI95 [{lo:.3f}, {hi:.3f}]", flush=True)
    print(f"GATE: {gate} — {note}", flush=True)
    print(f"wrote {out / 'ecare_pilot.json'} and ecare_pilot_id.jsonl ({len(kept)} items)", flush=True)


if __name__ == "__main__":
    main()
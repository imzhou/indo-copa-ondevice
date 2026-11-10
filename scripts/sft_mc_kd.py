#!/usr/bin/env python3
"""Letter-level KD from Qwen3-0.6B-Instruct onto student, same prompt as eval_nlu."""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from id_vocab import load_any_tokenizer


def mc_prompt(row: dict) -> str:
    q = "penyebab" if row.get("question") == "cause" else "akibat"
    return (
        "Premis: " + row["premise"] + "\n"
        "Apa " + q + " yang paling mungkin?\n"
        "A. " + row["choice1"] + "\n"
        "B. " + row["choice2"] + "\n"
        "Jawaban:"
    )


def letter_ids(tok) -> tuple[list[int], list[int]]:
    a = tok.encode(" A", add_special_tokens=False) or tok.encode("A", add_special_tokens=False)
    b = tok.encode(" B", add_special_tokens=False) or tok.encode("B", add_special_tokens=False)
    return a, b


@torch.no_grad()
def teacher_letter_logprob(model, tok, prompt: str, device: str) -> tuple[float, float]:
    p_ids = tok.encode(prompt, add_special_tokens=False)

    def cont(c_ids: list[int]) -> float:
        ids = p_ids + c_ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        lp = F.log_softmax(model(input_ids=x).logits[0].float(), dim=-1)
        s = 0.0
        off = len(p_ids)
        for i, tid in enumerate(c_ids):
            s += float(lp[off + i - 1, tid])
        return s / max(1, len(c_ids))

    scores = []
    for letter in ("A", "B"):
        best = None
        for cand in (f" {letter}", letter):
            c_ids = tok.encode(cand, add_special_tokens=False)
            if not c_ids:
                continue
            v = cont(c_ids)
            best = v if best is None else max(best, v)
        scores.append(best)
    return scores[0], scores[1]


def save_ckpt(model, tok, step, out: Path, keep: int) -> None:
    ckpt = out / ("step_" + str(step))
    tmp = out / (".tmp_step_" + str(step))
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.rename(ckpt)
    latest = out / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    os.symlink(ckpt.name, latest)
    steps = sorted([p for p in out.glob("step_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
    for p in steps[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
    print("saved", ckpt, flush=True)


class LetterDS(Dataset):
    def __init__(self, rows, tok, max_len: int, n_shot: int, rng: random.Random):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len
        self.n_shot = n_shot
        self.rng = rng
        self.gold_pool = [r for r in rows if r.get("source") in ("copa_en_train", "copa_en_train_swap")]
        if not self.gold_pool:
            self.gold_pool = rows

    def __len__(self):
        return len(self.rows)

    def _shots(self, i: int) -> str:
        if self.n_shot <= 0:
            return ""
        picks = []
        guard = 0
        while len(picks) < self.n_shot and guard < 20:
            guard += 1
            j = self.rng.randrange(len(self.gold_pool))
            if self.gold_pool[j] is self.rows[i]:
                continue
            r = self.gold_pool[j]
            letter = " A" if int(r["label"]) == 0 else " B"
            picks.append(mc_prompt(r) + letter)
        return "\n\n".join(picks)

    def __getitem__(self, i):
        row = self.rows[i]
        prefix = self._shots(i)
        prompt = (prefix + "\n\n" + mc_prompt(row)) if prefix else mc_prompt(row)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        letter = " A" if int(row["label"]) == 0 else " B"
        a_ids = self.tok(letter, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            a_ids = a_ids + [self.tok.eos_token_id]
        ids = (p_ids + a_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        tlp = row.get("teacher_lp") or [0.0, 0.0]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "prompt_len": torch.tensor(min(len(p_ids), len(ids))),
            "teacher_lp": torch.tensor(tlp, dtype=torch.float32),
            "hard": torch.tensor(int(row["label"])),
        }


def collate(batch, pad_id: int):
    m = max(len(x["input_ids"]) for x in batch)
    ids, lab, attn, plen, tlp, hard = [], [], [], [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = m - n
        ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id)]))
        lab.append(torch.cat([x["labels"], torch.full((pad,), -100)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        plen.append(x["prompt_len"])
        tlp.append(x["teacher_lp"])
        hard.append(x["hard"])
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(lab),
        "attention_mask": torch.stack(attn),
        "prompt_len": torch.stack(plen),
        "teacher_lp": torch.stack(tlp),
        "hard": torch.stack(hard),
    }


def load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("premise") and r.get("choice1") and r.get("choice2") and r.get("label") in (0, 1, "0", "1"):
                    r["label"] = int(r["label"])
                    rows.append(r)
    return rows


@torch.no_grad()
def annotate_teacher(rows, teacher, tok, device, cache_path: Path, demo_str: str = "") -> list[dict]:
    if cache_path.exists():
        cached = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
        if len(cached) == len(rows):
            print("teacher cache", cache_path, flush=True)
            return cached
    out = []
    agree = 0
    for i, r in enumerate(rows):
        prompt = demo_str + mc_prompt(r)
        lp_a, lp_b = teacher_letter_logprob(teacher, tok, prompt, device)
        pred = 0 if lp_a >= lp_b else 1
        r = dict(r)
        r["teacher_lp"] = [lp_a, lp_b]
        r["teacher_pred"] = pred
        r["teacher_agree"] = int(pred == int(r["label"]))
        agree += r["teacher_agree"]
        out.append(r)
        if (i + 1) % 50 == 0:
            print("teacher score", i + 1, "/", len(rows), "agree", agree / (i + 1), flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("teacher agree", agree / max(1, len(out)), "n", len(out), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher-cache", default="")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--alpha", type=float, default=0.7, help="weight on teacher KL")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--n-shot-train", type=int, default=0)
    p.add_argument("--teacher-shot", type=int, default=0, help="shots in teacher labeling prompt (XCOPA val demos)")
    p.add_argument("--teacher-4bit", action="store_true", help="load teacher in nf4 (required for 7B)")
    p.add_argument("--filter-disagree", action="store_true")
    p.add_argument("--keep-gold-always", action="store_true")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-last", type=int, default=6)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_jsonl(args.data)
    print("raw rows", len(rows), flush=True)

    ttok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if args.teacher_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, quantization_config=bnb, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="cuda",
        ).eval()
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
    cache = Path(args.teacher_cache or (args.out + "/teacher_scored.jsonl"))
    demo_str = ""
    if args.teacher_shot > 0:
        from datasets import load_dataset

        try:
            xv = load_dataset("cambridgeltl/xcopa", "id", split="validation", trust_remote_code=True)
        except TypeError:
            xv = load_dataset("cambridgeltl/xcopa", "id", split="validation")
        demos = []
        for i in range(args.teacher_shot):
            ex = xv[i]
            letter = " A" if ex["label"] == 0 else " B"
            demos.append(mc_prompt(ex) + letter)
        demo_str = "\n\n".join(demos) + "\n\n"
        print("teacher labeling shots", args.teacher_shot, flush=True)
    rows = annotate_teacher(rows, teacher, ttok, "cuda", cache, demo_str)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    kept = []
    for r in rows:
        is_gold = str(r.get("source", "")).startswith("copa_en_train")
        if args.filter_disagree:
            if is_gold and args.keep_gold_always:
                kept.append(r)
            elif r.get("teacher_agree"):
                kept.append(r)
        else:
            kept.append(r)
    print("kept", len(kept), "filter", args.filter_disagree, flush=True)
    rows = kept
    if len(rows) < 50:
        raise SystemExit("too few rows")

    tok, _mapped = load_any_tokenizer(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model.gradient_checkpointing_enable()
    model.train()
    model.config.use_cache = False
    a_ids, b_ids = letter_ids(tok)
    a0, b0 = a_ids[0], b_ids[0]
    print("letter tokens", a_ids, b_ids, flush=True)

    rng = random.Random(0)
    ds = LetterDS(rows, tok, args.seq_len, args.n_shot_train, rng)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    opt_per_epoch = max(1, (len(loader) + args.grad_accum - 1) // args.grad_accum)
    max_opt = opt_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, min(50, max_opt // 10), max_opt)
    opt_step = 0
    micro = 0
    running = 0.0
    running_ce = 0.0
    running_kd = 0.0
    opt.zero_grad(set_to_none=True)
    print("start letter kd opt_steps", max_opt, "n", len(ds), "alpha", args.alpha, flush=True)
    stop = False
    T = args.temp
    for epoch in range(args.epochs):
        if stop:
            break
        for batch in loader:
            ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attn = batch["attention_mask"].cuda()
            plen = batch["prompt_len"].cuda()
            tlp = batch["teacher_lp"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                ce = out.loss
                logits = out.logits.float()
                bsz = ids.size(0)
                s_lp = []
                for i in range(bsz):
                    pos = int(plen[i].item()) - 1
                    pos = max(0, min(pos, logits.size(1) - 1))
                    lp = F.log_softmax(logits[i, pos], dim=-1)
                    s_lp.append(torch.stack([lp[a0], lp[b0]]))
                s_lp = torch.stack(s_lp)
                t_lp = F.log_softmax(tlp / T, dim=-1)
                s_lp_t = F.log_softmax(s_lp / T, dim=-1)
                kd = F.kl_div(s_lp_t, t_lp.exp(), reduction="batchmean") * (T * T)
                loss = (1.0 - args.alpha) * ce + args.alpha * kd
            (loss / args.grad_accum).backward()
            running += float(loss.detach())
            running_ce += float(ce.detach())
            running_kd += float(kd.detach())
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.log_every == 0:
                    den = args.log_every * args.grad_accum
                    print(
                        "kd opt %d/%d ep %d loss %.3f ce %.3f kd %.3f"
                        % (opt_step, max_opt, epoch, running / den, running_ce / den, running_kd / den),
                        flush=True,
                    )
                    running = running_ce = running_kd = 0.0
                if opt_step % args.save_every == 0:
                    save_ckpt(model, tok, opt_step, Path(args.out), args.keep_last)
                if opt_step >= max_opt:
                    stop = True
                    break
    save_ckpt(model, tok, opt_step, Path(args.out), args.keep_last)
    print("letter kd done", flush=True)


if __name__ == "__main__":
    main()

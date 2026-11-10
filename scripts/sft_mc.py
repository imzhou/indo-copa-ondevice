#!/usr/bin/env python3
"""Completion SFT in the exact XCOPA letter format used at eval."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from id_vocab import load_any_tokenizer


def mc_text(row: dict) -> tuple[str, str]:
    q = "penyebab" if row.get("question") == "cause" else "akibat"
    qtext = row.get("question_text") or ("Apa " + q + " yang paling mungkin?")
    prompt = (
        "Premis: " + row["premise"] + "\n"
        + qtext + "\n"
        + "A. " + row["choice1"] + "\n"
        + "B. " + row["choice2"] + "\n"
        + "Jawaban:"
    )
    letter = " A" if int(row["label"]) == 0 else " B"
    return prompt, letter


class MCJsonl(Dataset):
    def __init__(self, path: str, tok, max_len: int, n_shot: int = 0):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("premise") and r.get("choice1") and r.get("choice2") and r.get("label") in (0, 1, "0", "1"):
                    self.rows.append(r)
        self.tok = tok
        self.max_len = max_len
        self.n_shot = n_shot
        print("mc examples", len(self.rows), "n_shot", n_shot, flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt, letter = mc_text(self.rows[i])
        if self.n_shot > 0 and len(self.rows) > 1:
            shots = []
            guard = 0
            while len(shots) < self.n_shot and guard < 24:
                guard += 1
                j = torch.randint(0, len(self.rows), (1,)).item()
                if j == i:
                    continue
                sp, sl = mc_text(self.rows[j])
                shots.append(sp + sl)
            if shots:
                prompt = chr(10).join(['']*0) or (chr(10)+chr(10)).join(shots) + (chr(10)+chr(10)) + prompt
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(letter, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            a_ids = a_ids + [self.tok.eos_token_id]
        ids = (p_ids + a_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        if len(ids) < 8:
            ids = p_ids[:8]
            labels = [-100] * len(ids)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id: int):
    m = max(len(x["input_ids"]) for x in batch)
    ids, lab, attn = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = m - n
        ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id)]))
        lab.append(torch.cat([x["labels"], torch.full((pad,), -100)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(lab),
        "attention_mask": torch.stack(attn),
    }


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq-len", type=int, default=384)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--n-shot-train", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
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
    ds = MCJsonl(args.data, tok, args.seq_len, n_shot=args.n_shot_train)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    opt_per_epoch = max(1, (len(loader) + args.grad_accum - 1) // args.grad_accum)
    max_opt = min(args.max_steps, opt_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, min(50, max_opt // 10), max_opt)
    opt_step = 0
    micro = 0
    t0 = time.time()
    running = 0.0
    opt.zero_grad(set_to_none=True)
    print("start mc sft opt_steps", max_opt, "n", len(ds), flush=True)
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for batch in loader:
            ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attn = batch["attention_mask"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            (loss / args.grad_accum).backward()
            running += float(loss.detach())
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.log_every == 0:
                    print(
                        "mc opt %d/%d ep %d loss %.3f"
                        % (opt_step, max_opt, epoch, running / (args.log_every * args.grad_accum)),
                        flush=True,
                    )
                    running = 0.0
                    t0 = time.time()
                if opt_step % args.save_every == 0:
                    save_ckpt(model, tok, opt_step, Path(args.out), args.keep_last)
                if opt_step >= max_opt:
                    stop = True
                    break
    save_ckpt(model, tok, opt_step, Path(args.out), args.keep_last)
    print("mc sft done", flush=True)


if __name__ == "__main__":
    main()

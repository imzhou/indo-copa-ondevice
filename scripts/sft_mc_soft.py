#!/usr/bin/env python3
"""Letter-level SFT with an optional order-symmetric soft target (row field `p_b`).

Loss = alpha * CE(gold letter + EOS) + (1 - alpha) * KL(two-way letter distribution || target).
Rows without `p_b` contribute CE only. Keeps the CLI/ckpt layout of scripts/sft_mc.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from id_vocab import load_any_tokenizer
from sft_mc import mc_text


class SoftMC(Dataset):
    def __init__(self, path: str, tok, max_len: int):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("premise") and r.get("choice1") and r.get("choice2") and r.get("label") in (0, 1, "0", "1"):
                    self.rows.append(r)
        self.tok = tok
        self.max_len = max_len
        self.n_soft = sum(1 for r in self.rows if r.get("p_b") is not None)
        print("mc examples", len(self.rows), "with p_b", self.n_soft, flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt, letter = mc_text(self.rows[i])
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(letter, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            a_ids = a_ids + [self.tok.eos_token_id]
        ids = (p_ids + a_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        pb = self.rows[i].get("p_b")
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "plen": torch.tensor(min(len(p_ids), len(ids))),
            "p_b": torch.tensor(-1.0 if pb is None else float(pb)),
        }


def collate(batch, pad_id: int):
    m = max(len(x["input_ids"]) for x in batch)
    ids, lab, attn, plen, pbs = [], [], [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = m - n
        ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=x["input_ids"].dtype)]))
        lab.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=x["labels"].dtype)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        plen.append(x["plen"])
        pbs.append(x["p_b"])
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(lab),
        "attention_mask": torch.stack(attn),
        "plen": torch.stack(plen),
        "p_b": torch.stack(pbs),
    }


def save_ckpt(model, tok, step, out: Path, keep: int) -> None:
    ckpt = out / ("step_" + str(step))
    tmp = out / (".tmp_step_" + str(step))
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    shutil.rmtree(ckpt, ignore_errors=True)
    tmp.rename(ckpt)
    latest = out / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt.name)
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
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--alpha", type=float, default=0.5, help="CE weight; 1-alpha goes to soft KL")
    p.add_argument("--kl-scale", type=float, default=1.0, help="divide soft KL by this factor")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--seed", type=int, default=-1, help=">=0 reseeds shuffling/dropout for seed replicates")
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.seed >= 0:
        import random as _random

        _random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tok, _ = load_any_tokenizer(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    id_a = tok(" A", add_special_tokens=False)["input_ids"][0]
    id_b = tok(" B", add_special_tokens=False)["input_ids"][0]
    print("letter token ids", id_a, id_b, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model.gradient_checkpointing_enable()
    model.train()
    model.config.use_cache = False

    ds = SoftMC(args.data, tok, args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt_per_epoch = max(1, (len(loader) + args.grad_accum - 1) // args.grad_accum)
    max_opt = opt_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, min(20, max_opt // 10), max_opt)
    print(f"start soft sft opt_steps={max_opt} n={len(ds)} lr={args.lr} alpha={args.alpha}", flush=True)

    opt_step = micro = 0
    run_ce = run_kl = 0.0
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in loader:
            ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attn = batch["attention_mask"].cuda()
            plen = batch["plen"].cuda()
            pbs = batch["p_b"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=attn).logits
                ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                     labels[:, 1:].reshape(-1), ignore_index=-100)
                rows = (pbs >= 0) & (plen > 0) & (plen < ids.size(1))
                if rows.any():
                    idx = rows.nonzero(as_tuple=True)[0]
                    pos = plen[idx] - 1
                    sel = logits[idx, pos, :].float()
                    two = torch.stack([sel[:, id_a], sel[:, id_b]], dim=1).log_softmax(dim=-1)
                    tgt = torch.stack([1.0 - pbs[idx], pbs[idx]], dim=1).clamp(1e-6, 1 - 1e-6)
                    kl = (tgt * (tgt.log() - two)).sum(dim=1).mean()
                else:
                    kl = ce.new_zeros(())
                loss = args.alpha * ce + (1.0 - args.alpha) * kl / args.kl_scale
            (loss / args.grad_accum).backward()
            run_ce += float(ce.detach())
            run_kl += float(kl.detach())
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.log_every == 0:
                    k = args.log_every * args.grad_accum
                    print(f"soft opt {opt_step}/{max_opt} ep {epoch} ce {run_ce / k:.3f} "
                          f"kl {run_kl / k:.4f} pb_mean {pbs[pbs >= 0].mean().item() if (pbs >= 0).any() else float('nan'):.3f}",
                          flush=True)
                    run_ce = run_kl = 0.0
                if opt_step % args.save_every == 0:
                    save_ckpt(model, tok, opt_step, Path(args.out), args.keep_last)
    save_ckpt(model, tok, opt_step, Path(args.out), args.keep_last)
    print("soft sft done", flush=True)


if __name__ == "__main__":
    main()

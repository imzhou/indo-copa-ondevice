#!/usr/bin/env python3
"""Indonesian 32k vocab as a subset of Qwen tokens (aligned logit KD)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_map(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "id_vocab.json"
    return json.loads(p.read_text())


def save_map(meta: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(meta))


class IndoTokenizer:
    """Encode with Qwen tokenizer, remap to kept 32k ids; decode via old ids."""

    def __init__(self, base, kept_ids: list[int], unk_old: int):
        self.base = base
        self.kept_ids = list(kept_ids)
        self.old_to_new = {old: i for i, old in enumerate(self.kept_ids)}
        self.unk_id = int(self.old_to_new.get(unk_old, 0))
        self.vocab_size = len(self.kept_ids)
        self.eos_token_id = self.old_to_new.get(getattr(base, "eos_token_id", None), self.unk_id)
        self.bos_token_id = self.old_to_new.get(getattr(base, "bos_token_id", None), self.eos_token_id)
        self.pad_token_id = self.old_to_new.get(getattr(base, "pad_token_id", None), self.eos_token_id)
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        self.eos_token = getattr(base, "eos_token", None)
        self.pad_token = getattr(base, "pad_token", None) or self.eos_token
        self.chat_template = getattr(base, "chat_template", None)
        self.padding_side = "right"

    def remap(self, old_ids: list[int]) -> list[int]:
        m = self.old_to_new
        u = self.unk_id
        return [m.get(int(i), u) for i in old_ids]

    def in_keep_mask(self, old_ids: list[int]) -> list[bool]:
        m = self.old_to_new
        return [int(i) in m for i in old_ids]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        old = self.base.encode(text, add_special_tokens=add_special_tokens)
        return self.remap(old)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        old = [self.kept_ids[int(i)] if 0 <= int(i) < len(self.kept_ids) else self.kept_ids[self.unk_id] for i in ids]
        return self.base.decode(old, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, *args, **kwargs):
        return self.base.apply_chat_template(*args, **kwargs)

    def save_pretrained(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(out)
        save_map(
            {
                "kept_ids": self.kept_ids,
                "unk_old": self.kept_ids[self.unk_id],
                "vocab_size": self.vocab_size,
            },
            out / "id_vocab.json",
        )

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


def load_indo_tokenizer(model_dir: str | Path) -> IndoTokenizer:
    d = Path(model_dir)
    meta = load_map(d)
    base = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    if base.pad_token_id is None:
        base.pad_token = base.eos_token
    return IndoTokenizer(base, meta["kept_ids"], meta["unk_old"])


def load_any_tokenizer(model_dir: str | Path):
    d = Path(model_dir)
    if (d / "id_vocab.json").exists():
        return load_indo_tokenizer(d), True
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, False

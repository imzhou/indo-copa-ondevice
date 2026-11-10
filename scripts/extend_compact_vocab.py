#!/usr/bin/env python3
"""Extend a vocab-subset model with merge-closure tokens (for exact BPE)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_closure(kept: list[int], tokenizer_json: dict) -> list[int]:
    id2tok = {i: s for s, i in tokenizer_json["model"]["vocab"].items()}
    special = {a["id"]: a["content"] for a in tokenizer_json.get("added_tokens", [])}
    tok2id = tokenizer_json["model"]["vocab"]
    pair = {}
    for a, b in tokenizer_json["model"]["merges"]:
        pair[a + b] = (a, b)

    need_str = {special.get(o, id2tok.get(o)) for o in kept}
    stack = list(need_str)
    while stack:
        t = stack.pop()
        p = pair.get(t)
        if p is None:
            continue
        for x in p:
            if x not in need_str:
                need_str.add(x)
                stack.append(x)

    kept_str = {special.get(o, id2tok.get(o)) for o in kept}
    extra_tokens = need_str - kept_str
    return [tok2id[s] for s in extra_tokens]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--orig-model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    meta = json.loads((src / "id_vocab.json").read_text())
    kept = [int(x) for x in meta["kept_ids"]]
    tokenizer_json = json.loads((src / "tokenizer.json").read_text())
    extra = merge_closure(kept, tokenizer_json)
    kept2 = kept + [e for e in extra if e not in set(kept)]
    v = len(kept2)
    print(f"kept={len(kept)} closure_extra={len(kept2) - len(kept)} final={v}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.orig_model, torch_dtype=torch.bfloat16)
    old_emb = model.get_input_embeddings().weight.data
    hidden = old_emb.shape[1]
    idx = torch.tensor(kept2, dtype=torch.long)
    new_emb = torch.nn.Embedding(v, hidden, dtype=old_emb.dtype)
    new_emb.weight.data.copy_(old_emb[idx])
    model.set_input_embeddings(new_emb)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    new_head = torch.nn.Linear(hidden, v, bias=False, dtype=old_emb.dtype)
    if tied:
        new_head.weight = new_emb.weight
    else:
        new_head.weight.data.copy_(model.lm_head.weight.data[idx])
    model.lm_head = new_head
    model.config.vocab_size = v
    model.config.tie_word_embeddings = tied
    model.config.bos_token_id = None
    model.config.eos_token_id = 0
    model.config.pad_token_id = 0
    model.generation_config.eos_token_id = 0
    model.generation_config.pad_token_id = 0

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tok.save_pretrained(out)
    for name in ["chat_template.jinja", "tokenizer.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    (out / "id_vocab.json").write_text(
        json.dumps({"kept_ids": kept2, "unk_old": kept2[0], "vocab_size": v})
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    (out / "compact_manifest.json").write_text(
        json.dumps(
            {
                "source_model": args.orig_model,
                "base_compact": str(src),
                "final_vocab": v,
                "closure_extra": len(kept2) - len(kept),
                "params_m": params,
                "tied": tied,
            },
            indent=2,
        )
    )
    print("saved", out, "params_m", round(params, 3), "tied", tied, flush=True)


if __name__ == "__main__":
    main()

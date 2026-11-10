#!/usr/bin/env python3
"""Create ~0.2B Qwen3 student, copy sliced embeddings from the 0.6B teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def copy_sliced(dst: torch.nn.Parameter, src: torch.Tensor) -> None:
    slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    with torch.no_grad():
        dst[slices].copy_(src[slices].to(dtype=dst.dtype, device=dst.device))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    with open(args.config) as f:
        raw = json.load(f)
    skip = {"architectures", "transformers_version", "torch_dtype"}
    cfg_kwargs = {k: v for k, v in raw.items() if k not in skip}
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    cfg_kwargs["vocab_size"] = int(teacher.config.vocab_size)
    cfg_kwargs["bos_token_id"] = teacher.config.bos_token_id
    cfg_kwargs["eos_token_id"] = teacher.config.eos_token_id
    student_cfg = Qwen3Config(**cfg_kwargs)
    student = Qwen3ForCausalLM(student_cfg)

    copy_sliced(student.model.embed_tokens.weight, teacher.model.embed_tokens.weight.data)
    copy_sliced(student.model.norm.weight, teacher.model.norm.weight.data)
    student.tie_weights()

    total = count_params(student)
    print(f"student params: {total / 1e6:.2f}M")
    student.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "init_meta.json").write_text(json.dumps({"teacher": args.teacher, "params": total}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

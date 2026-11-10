#!/usr/bin/env python3
"""Build a GGUF-ready dir whose tokenizer matches the sliced 32,870 vocab.

The compact model keeps a subset of Qwen token ids, so the stock 151k
tokenizer is inconsistent with its embedding. Here we emit a tokenizer.json
containing only kept tokens (in kept order) and merges whose parts and result
are all kept. Also audits tokenization against the training-time remap path.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_subset_tokenizer(tok_json: dict, kept: list[int]) -> tuple[dict, dict]:
    old2new = {int(o): i for i, o in enumerate(kept)}
    vocab_old = tok_json["model"]["vocab"]
    id2tok = {i: s for s, i in vocab_old.items()}
    specials = {a["id"]: a for a in tok_json.get("added_tokens", [])}

    new_vocab: dict[str, int] = {}
    new_specials: list[dict] = []
    for new_id, old_id in enumerate(kept):
        old_id = int(old_id)
        if old_id in specials:
            ent = dict(specials[old_id])
            ent["id"] = new_id
            new_specials.append(ent)
        else:
            tok = id2tok[old_id]
            new_vocab[tok] = new_id

    new_merges = []
    for m in tok_json["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ")
        if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
            new_merges.append([a, b])

    tj = json.loads(json.dumps(tok_json))
    tj["model"]["vocab"] = new_vocab
    tj["model"]["merges"] = new_merges
    tj["added_tokens"] = new_specials
    return tj, {"old2new": old2new, "merges_kept": len(new_merges)}


def audit(full_tok, sub_tok, old2new: dict, texts: list[str], label: str) -> None:
    same = 0
    oov_tokens = 0
    total_tokens = 0
    bad = 0
    for t in texts:
        old = full_tok.encode(t, add_special_tokens=False)
        total_tokens += len(old)
        remapped = [old2new.get(int(i), 0) for i in old]
        oov_tokens += sum(1 for i in old if int(i) not in old2new)
        try:
            new = sub_tok.encode(t, add_special_tokens=False)
        except Exception:
            bad += 1
            continue
        if new == remapped:
            same += 1
    n = max(len(texts), 1)
    print(
        f"[{label}] prompts={len(texts)} exact_same={same}/{n}={same / n:.4f} "
        f"encode_err={bad} oov_tokens={oov_tokens}/{total_tokens}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="compact model dir (has id_vocab.json)")
    p.add_argument("--out", required=True, help="GGUF-ready dir")
    p.add_argument("--base-tokenizer", default="", help="full Qwen tokenizer.json dir")
    p.add_argument("--audit-data", nargs="*", default=[])
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = [int(x) for x in json.loads((src / "id_vocab.json").read_text())["kept_ids"]]

    tok_dir = Path(args.base_tokenizer) if args.base_tokenizer else src
    tok_json = json.loads((tok_dir / "tokenizer.json").read_text())
    tj, info = build_subset_tokenizer(tok_json, kept)
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    print(f"vocab={len(tj['model']['vocab'])} specials={len(tj['added_tokens'])} merges={info['merges_kept']}", flush=True)

    tc = json.loads((tok_dir / "tokenizer_config.json").read_text())
    tc.pop("added_tokens_decoder", None)
    jinja = src / "chat_template.jinja"
    if jinja.exists():
        tc["chat_template"] = jinja.read_text()
    (out / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False))

    for name in ["config.json", "generation_config.json", "chat_template.jinja", "id_vocab.json"]:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    for f in src.glob("*.safetensors"):
        if not (out / f.name).exists():
            (out / f.name).symlink_to(f.resolve())

    if args.audit_data:
        from transformers import AutoTokenizer

        full = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        sub = AutoTokenizer.from_pretrained(out, trust_remote_code=True, local_files_only=True)
        print(f"full_vocab={len(full)} sub_vocab={len(sub)}", flush=True)
        skip = {"ask-for", "question", "label", "src", "source"}
        for path in args.audit_data:
            texts = []
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                for k, v in r.items():
                    if isinstance(v, str) and k not in skip and not k.endswith("_en"):
                        texts.append(v)
            audit(full, sub, info["old2new"], texts, Path(path).name)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()

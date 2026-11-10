# indo-copa-ondevice

Code and measurement artifacts for **on-device letter-score evaluation** of quantised Indonesian causal-reasoning language models on an Allwinner A133 tablet (4× Cortex-A53, Android 10, 32-bit userspace, ~992 MB RAM).

Weights are published separately on Hugging Face:

https://huggingface.co/jifengzhou/indo-copa-n06ctl

This repository contains **source code**, **checksums**, **per-item margin vectors**, and **board/host logs**. It does **not** contain manuscript drafts or lab notebooks.

## Layout

```
scripts/          host + board letter scoring, quant ladder, SFT helpers
configs/          student configs + pre-registered ladder protocol
deploy/           A133 push / bench helpers (32-bit armeabi-v7a)
toolchain/        gguf_cand_lp.cpp letter scorer source
artifacts/
  checksums/      SHA256 of HF weights and of files in this repo
  board/          p_dev.txt, per-item TSV margins, metrics JSON
  probe/          host-side letter-score JSON dumps (n06ctl ladder family)
  logs/           board letter run logs and small bench/gen traces
```

## Key artifacts

| Path | Description |
|---|---|
| `artifacts/board/p_dev.txt` | Byte-identical 400-line prompt stream used on board and host (MD5 `33e7bf81c84052f951d1ef83601b49a7`) |
| `artifacts/board/letter_*.tsv` | Per-item margin vectors (board) |
| `artifacts/board/letter_host_*.tsv` | Per-item margin vectors (host twin) |
| `artifacts/board/*.metrics.json` | Aggregated letter metrics |
| `artifacts/checksums/hf_indo-copa-n06ctl.sha256` | SHA256 of the published HF model + GGUF files |
| `artifacts/checksums/artifacts.sha256` | SHA256 of files under `artifacts/` |

## Reproduce (host)

```bash
# build letter scorer (link against your llama.cpp as documented in deploy/)
# then:
python scripts/eval_gguf.py \
  --gguf path/to/n06ctl_Q4_K_M.gguf \
  --sets ecare_dev --threads 8 --n-ctx 128
```

## Board

See `deploy/`. The device is **32-bit** `armeabi-v7a`; do not push arm64 binaries.

## License

- Code in this repository: Apache-2.0 (see `LICENSE`)
- Model weights: Qwen license via base `Qwen/Qwen3-0.6B` (see the HF model card)

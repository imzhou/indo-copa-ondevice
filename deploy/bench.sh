#!/bin/sh
# On-board benchmark for copa3 GGUF. Usage: sh bench.sh [model.gguf] [threads]
set -e
MODEL=${1:-copa3_Q4_K_M.gguf}
THREADS=${2:-4}
BIN=${BIN:-./llama.cpp/build/bin/llama-bench}
"$BIN" -m "$MODEL" -p 64 -n 32 -t "$THREADS" -r 3

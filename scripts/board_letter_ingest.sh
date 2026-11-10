#!/usr/bin/env bash
# Take the board's letter TSVs out of deploy/logs/<TS>/letter_<boardTS>/ (this host = the
# Windows Z: tree), copy them to 175, and run the paired board-vs-host-vs-HF scoring there.
# Usage: scripts/board_letter_ingest.sh [ctl_base] [base_base]
set -uo pipefail
H=USER@TRAIN_HOST
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=/path/to/code
B=/path/to/data/board/letter
M=/path/to/data
CTL=${1:-n06ctl_Q4_K_M}
BASE=${2:-qwen3_0.6b_Q4_K_M}
N=400

SRC=""
for d in $(ls -1dt "$ROOT"/deploy/logs/*/letter_* 2>/dev/null); do
  c=$(grep -c '' "$d/letter_$CTL.tsv" 2>/dev/null || echo 0)
  b=$(grep -c '' "$d/letter_$BASE.tsv" 2>/dev/null || echo 0)
  if [[ $c -eq $N && $b -eq $N ]]; then SRC=$d; break; fi
done
[[ -n $SRC ]] || { echo "[wait] no letter_* dir with $N rows for $CTL + $BASE under $ROOT/deploy/logs"; exit 3; }
echo "[src] $SRC"
for f in "letter_$CTL.tsv" "letter_$BASE.tsv" "letter_$CTL.txt" "letter_$BASE.txt" \
         "letter_$CTL.log" "letter_$BASE.log" sysinfo.txt cpufreq.txt progress.txt; do
  [[ -e "$SRC/$f" ]] && printf '[file] %-32s lines=%s\n' "$f" "$(grep -c '' "$SRC/$f")"
done || true

ssh "$H" "mkdir -p $B"
scp -q "$SRC/letter_$CTL.tsv" "$SRC/letter_$BASE.tsv" "$H:$B/"
for f in "letter_$CTL.txt" "letter_$BASE.txt" "letter_$CTL.log" "letter_$BASE.log" \
         sysinfo.txt cpufreq.txt progress.txt; do
  [[ -e "$SRC/$f" ]] && scp -q "$SRC/$f" "$H:$B/"
done
echo "[push] -> $H:$B"

OUT=$ROOT/deploy/logs/board_letter_score_${CTL}_${BASE}.log
ssh "$H" "cd $REMOTE && .venv/bin/python -u scripts/board_letter_score.py \
  --subset $B/subset.json \
  --tsv board_ctl=$B/letter_$CTL.tsv \
  --tsv host_ctl=$B/letter_host_$CTL.tsv \
  --tsv board_base=$B/letter_$BASE.tsv \
  --tsv host_base=$B/letter_host_$BASE.tsv \
  --hf $M/data/probe_format/dev_ecare_n06_ctl_step_1884.json --hf-cond ecare_dev/letter" \
  2>&1 | tee "$OUT"
echo "INGEST_DONE $OUT"

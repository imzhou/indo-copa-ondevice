#!/bin/bash
# Task #20: 0.6B 主线候选的 2/3-bit 量化阶梯（imatrix 标定 -> IQ2/IQ3 档位 -> letter 评测）
# 标定语料只取训练分布（mix_native），dev/COPAL 一律不进 imatrix。
# Usage: bash scripts/quant_ladder.sh [n06ctl|n06kl|n06mid]
#   env: ROWS=2000 QUANTS=IQ2_XXS,IQ2_XS,IQ2_S,IQ2_M,IQ3_XXS,IQ3_S
set -uo pipefail
D=/path/to/data
cd /path/to/code || exit 1
export HF_HOME=$D/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
P=.venv/bin/python
T=$D/toolchain/llama.cpp

MODEL=${1:-n06ctl}
case "$MODEL" in
  n06ctl) REF=n06_ctl;  REFSTEP=1884 ;;
  n06kl)  REF=n06_kl;   REFSTEP=1906 ;;
  n06mid) REF=n06_mid;  REFSTEP=1906 ;;
  *) REF=$MODEL; REFSTEP=1906 ;;
esac
ROWS=${ROWS:-2000}
QUANTS=${QUANTS:-IQ2_XXS,IQ2_XS,IQ2_S,IQ2_M,IQ3_XXS,IQ3_S}
F16=$D/gguf_arms/${MODEL}_f16.gguf
REF_DEV=$D/data/probe_format/dev_ecare_${REF}_step_${REFSTEP}.json
REF_COP=$D/data/probe_format/copal_final_${REF}_step_${REFSTEP}.json
echo "[cfg] MODEL=$MODEL REF=$REF ROWS=$ROWS QUANTS=$QUANTS"
[[ -f $F16 ]] || { echo "[FATAL] missing $F16"; exit 1; }
for f in "$REF_DEV" "$REF_COP"; do [[ -s $f ]] || { echo "[FATAL] missing ref $f"; exit 1; }; done

IMX=$D/gguf_arms/${MODEL}_imatrix.gguf
CORPUS=$D/data/imatrix_${MODEL}.txt

echo "[1] corpus $(date -Is)"
$P -u scripts/build_imatrix_corpus.py --jsonl $D/data/mix_native.jsonl \
  --out $CORPUS --rows $ROWS || { echo "[FATAL] corpus failed"; exit 1; }
head -c 260 $CORPUS; echo

if [[ ! -s $IMX ]]; then
  echo "[2] imatrix $(date -Is)"
  $T/build/bin/llama-imatrix -m $F16 -f $CORPUS -o $IMX \
    -c 512 -t 8 --output-format gguf --no-ppl > $D/logs/imatrix_${MODEL}.log 2>&1 \
    || { echo "[FATAL] imatrix failed"; tail -30 $D/logs/imatrix_${MODEL}.log; exit 1; }
else
  echo "[2] imatrix present, skip"
fi
ls -l $IMX

IFS=',' read -ra QS <<< "$QUANTS"
for q in "${QS[@]}"; do
  o=$D/gguf_arms/${MODEL}_${q}.gguf
  if [[ -s $o ]]; then echo "[3] $q exists, skip quantize"; continue; fi
  echo "[3] quantize $q $(date -Is)"
  $T/build/bin/llama-quantize --imatrix $IMX $F16 $o $q > $D/logs/quant_${MODEL}_${q}.log 2>&1 \
    || { echo "[skip] quantize $q failed"; tail -8 $D/logs/quant_${MODEL}_${q}.log; continue; }
  ls -l $o
done

for q in "${QS[@]}"; do
  o=$D/gguf_arms/${MODEL}_${q}.gguf
  [[ -s $o ]] || continue
  for s in ecare_dev copal_test; do
    j=$D/data/probe_format/gguf_${MODEL}_${q}_${s}.json
    if [[ -s $j ]]; then echo "[4] $q/$s exists, skip eval"; continue; fi
    if [[ $s == ecare_dev ]]; then REFJ=$REF_DEV; else REFJ=$REF_COP; fi
    echo "[4] eval $q on $s $(date -Is)"
    $P -u scripts/eval_gguf.py --gguf $o --sets $s --threads 16 --out $j \
      --compare $REFJ --compare-cond $s/letter > $D/logs/eval_${MODEL}_${q}_${s}.log 2>&1 \
      || { echo "[skip] eval $q/$s failed"; tail -20 $D/logs/eval_${MODEL}_${q}_${s}.log; continue; }
    grep -E "letter|vs HF" $D/logs/eval_${MODEL}_${q}_${s}.log
  done
done
echo "QUANT_LADDER_DONE $(date -Is)"

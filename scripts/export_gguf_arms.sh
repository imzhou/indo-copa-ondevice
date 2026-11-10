#!/usr/bin/env bash
# Export the 0.6B mainline candidates to GGUF (f16 -> Q4_K_M/Q4_0), stage them in deploy/ for
# the adb push, and check engine parity (GGUF vs HF bf16, same letter+order-swap protocol)
# before spending board time.
set -uo pipefail
D=/path/to/data
T=$D/toolchain/llama.cpp
cd /path/to/code
export HF_HOME=$D/hf_home
P=.venv/bin/python
LOG=$D/logs/gguf_export.log
exec >> "$LOG" 2>&1
echo "===== START $(date -Is) ====="
mkdir -p $D/gguf_arms $D/deploy

pick () { ls -d $D/ckpts/$1/step_* 2>/dev/null | sort -V | tail -1; }

export_arm () {  # ckpt tag
  local ck=$1 tag=$2
  local f16=$D/gguf_arms/${tag}_f16.gguf
  if [[ ! -s $f16 ]]; then
    echo "[convert] $tag <- $ck $(date -Is)"
    timeout 1800 $P -u $T/convert_hf_to_gguf.py "$ck" --outtype f16 --outfile "$f16" \
      > "$D/logs/convert_$tag.log" 2>&1 || { echo "EXPORT $tag convert FAILED (logs/convert_$tag.log)"; return; }
  fi
  for q in Q4_K_M Q4_0; do
    local o=$D/gguf_arms/${tag}_$q.gguf
    [[ -s $o ]] || timeout 1800 $T/build/bin/llama-quantize "$f16" "$o" $q > /dev/null 2>&1 \
      || echo "EXPORT $tag quant $q FAILED"
    [[ -s $o ]] && cp -f "$o" $D/deploy/
  done
  echo "EXPORT $tag <- $(basename "$ck")  $(ls -l $D/deploy/${tag}_Q4*.gguf 2>/dev/null | awk '{print $9" "int($5/1048576)"MB"}' | tr '\n' ' ')"
}

for spec in "native_n06_ctl n06ctl" "native_n06_kl n06kl" "native_n06_mid n06mid"; do
  set -- $spec
  ck=$(pick "$1")
  [[ -n "$ck" ]] && export_arm "$ck" "$2"
done

echo "[parity] GGUF Q4_K_M vs HF bf16, same protocol (dev n=2125 selection set + COPAL n=559) $(date -Is)"
parity () {  # tag devref copalref
  local tag=$1 devref=$2 copref=$3
  local g=$D/gguf_arms/${tag}_Q4_K_M.gguf
  [[ -s $g ]] || { echo "PARITY $tag missing gguf"; return; }
  for spec in "ecare_dev $devref" "copal_test $copref"; do
    set -- $spec
    local j=$D/data/probe_format/gguf_${tag}_Q4_K_M_$1.json
    timeout 7200 $P -u scripts/eval_gguf.py --gguf "$g" --sets "$1" --threads 16 \
      --out "$j" --compare "$D/data/probe_format/$2" --compare-cond "$1/letter" \
      2>&1 | grep -Ev "^llama_|^sched_|^print_info|^main|Warning:" | sed "s/^/PARITY $tag /"
  done
}
parity n06ctl dev_ecare_n06_ctl_step_1884.json copal_final_n06_ctl_step_1884.json
parity n06kl  dev_ecare_n06_kl_step_1906.json  copal_final_n06_kl_step_1906.json
parity n06mid dev_ecare_n06_mid_step_1906.json copal_final_n06_mid_step_1906.json

echo "[staged] $(ls -l $D/deploy/*.gguf | awk '{print $9" "int($5/1048576)"MB"}')"
df -h /path/to/data | tail -1
echo GGUF_EXPORT_DONE

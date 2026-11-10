#!/system/bin/sh
# On-device pipeline: sysinfo, llama-bench, smoke generation, peak RSS.
# Usage: sh a133_run_all.sh <threads> <llama_dir> <model.gguf> [more models...]
set -u

THREADS=${1:-4}
LLAMA_PASS=${2:-}
shift 2 2>/dev/null || true
MODELS="$*"

DEV_DIR=$(cd "$(dirname "$0")" && pwd)
LOG="$DEV_DIR/logs"
rm -rf "$LOG"
mkdir -p "$LOG"

find_llama() {
    for d in "$LLAMA_PASS" /data/local/tmp/copa3/llama /data/local/tmp/llama.cpp \
             "$HOME/llama.cpp" /data/data/com.termux/files/home/llama.cpp "$DEV_DIR/llama"; do
        if [ -n "$d" ] && [ -x "$d/llama-completion" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if ! LLAMA_DIR=$(find_llama); then
    echo "llama-completion not found; pass its directory as argument 2" > "$LOG/fatal.txt"
    cat "$LOG/fatal.txt"
    exit 1
fi
export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"
CLI="$LLAMA_DIR/llama-completion"
BENCH=""
[ -x "$LLAMA_DIR/llama-bench" ] && BENCH="$LLAMA_DIR/llama-bench"

{
    echo "== date"
    date
    echo "== uname"
    uname -a
    echo "== model"
    getprop ro.product.model 2>/dev/null
    echo "== android"
    getprop ro.build.version.release 2>/dev/null
    echo "== cpu"
    grep -m1 -E "model name|Processor|Hardware" /proc/cpuinfo
    echo "== cores"
    grep -c ^processor /proc/cpuinfo
    echo "== mem"
    grep MemTotal /proc/meminfo
    echo "== governor"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
    echo "== llama dir"
    echo "$LLAMA_DIR"
} > "$LOG/sysinfo.txt" 2>&1

for m in $MODELS; do
    base=$(basename "$m" .gguf)
    if [ -n "$BENCH" ]; then
        "$BENCH" -m "$DEV_DIR/$m" -p 128,256 -n 32 -t "$THREADS" -r 3 > "$LOG/bench_$base.txt" 2>&1
    else
        echo "llama-bench not found in $LLAMA_DIR" > "$LOG/bench_$base.txt"
    fi
done

PROMPT='Premis: Badan saya menimbulkan bayang-bayang di atas rumput.
Apa penyebab yang paling mungkin?
A. Matahari terbit.
B. Rumput telah dipotong.
Jawaban:'

for m in $MODELS; do
    base=$(basename "$m" .gguf)
    out="$LOG/gen_$base.txt"
    "$CLI" -m "$DEV_DIR/$m" -c 256 -n 32 --temp 0 -t "$THREADS" -st --no-warmup \
        -p "$PROMPT" > "$out" 2>&1 &
    pid=$!
    max=0
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null)
        if [ -n "${rss:-}" ] && [ "$rss" -gt "$max" ]; then
            max=$rss
        fi
        sleep 0.2 2>/dev/null || sleep 1
    done
    wait "$pid"
    rc=$?
    echo "exit=$rc peak_rss_kb=$max" >> "$out"
done

echo "==== SUMMARY ===="
cat "$LOG/sysinfo.txt"
grep -h -E "pp128|pp256|tg32|t/s" "$LOG"/bench_*.txt 2>/dev/null
grep -h "peak_rss_kb" "$LOG"/gen_*.txt 2>/dev/null

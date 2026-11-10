# A133 deploy helpers

Scripts for pushing quantised GGUF models to an Allwinner A133 tablet
(4× Cortex-A53, Android 10, **32-bit** `armeabi-v7a` userspace, ~992 MB RAM)
and running bench / smoke / letter-score jobs.

Published weights live on Hugging Face:

https://huggingface.co/jifengzhou/indo-copa-n06ctl

Typical board filenames used by the helpers below are `n06ctl_Q4_K_M.gguf`
and `n06ctl_Q8_0.gguf` (place them next to these scripts, or edit the bat
variables). Older local aliases such as `copa3_*.gguf` may still appear in
script defaults — rename or adjust paths before running.

> The board is 32-bit Android. Official arm64 llama.cpp builds will not run.

## 0. One-shot pipeline (recommended)

`run_a133_all.bat` (Windows host with `adb` on PATH):

```bat
rem Q4_K_M, 4 threads
run_a133_all.bat

rem Q8_0 / both quants
run_a133_all.bat Q8
run_a133_all.bat both

rem custom thread count; optionally scp logs back to a training host
run_a133_all.bat --threads 4 --upload
```

What it does: check the adb device → push GGUF (size + SHA256 verify) →
push the bundled armeabi-v7a llama.cpp package on first use → run
`llama-bench`, a short generation smoke test, and peak RSS → pull logs
into `deploy/logs/<timestamp>/` → print a short summary.

Log files: `sysinfo.txt`, `bench_*.txt`, `gen_*.txt`
(last line includes `exit=… peak_rss_kb=…`).

Linux equivalent: `a133_run_all.sh`.

## 1. Manual push

Android (adb):

```bash
adb push n06ctl_Q4_K_M.gguf /data/local/tmp/
adb push bench.sh /data/local/tmp/
```

Linux (scp):

```bash
scp n06ctl_Q4_K_M.gguf bench.sh user@<board-ip>:~/
```

### Windows helper

`deploy_a133.bat`:

```bat
rem optional: fetch GGUFs from a remote host first (edit SSH_HOST / SRC_DIR)
deploy_a133.bat --fetch

rem push Q4_K_M + bench.sh (size + SHA256 check)
deploy_a133.bat

rem push then run llama-bench on device
deploy_a133.bat --bench

deploy_a133.bat Q8
deploy_a133.bat both
```

Default on-device llama.cpp path is `/data/local/tmp/copa3/llama`
(populated by `run_a133_all.bat`). Override via `DEV_LLAMA` at the top of the bat.

## 2. On-device llama.cpp (armeabi-v7a)

`run_a133_all.bat` pushes `llama-android-armv7/` (`llama-bench` +
`llama-completion`) to `/data/local/tmp/copa3/llama`. Cross-build on a
Linux host with NDK (example r27d):

```bash
bash deploy/build_android_armv7.sh
```

Build notes: `armeabi-v7a` + NEON + `-march=armv8-a` (equivalent to
`-mcpu=cortex-a53 -mfpu=neon-fp-armv8`), statically linked C++, and
`unicode-data.cpp` prebuilt at `-O0`. Resulting binaries are ELF 32-bit ARM
with Tag_CPU_arch v7 / NEONv1.

Source tweak: in `ggml-cpu-impl.h`, narrow the `vcvtnq_s32_f32` 32-bit
compat shim to `__ARM_ARCH < 8`, otherwise it conflicts with clang 18
`arm_neon.h` on armv8 32-bit (intrinsic already present).

Optional Termux rebuild:

```bash
pkg install -y cmake clang git
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-completion llama-bench
```

## 3. Throughput / memory

```bash
# 4 threads = all A133 cores
./llama/llama-bench -m n06ctl_Q4_K_M.gguf -p 128,256 -n 32 -t 4 -r 3
```

Peak RSS smoke (keep context small on ~992 MB RAM):

```bash
./llama/llama-completion -m n06ctl_Q4_K_M.gguf -c 256 -n 8 --temp 0 -t 4 \
  -p "Apa ibu kota Indonesia?" &
ps -A -o RSS,NAME | grep llama
```

## 4. Functional smoke

```bash
./llama/llama-completion -m n06ctl_Q4_K_M.gguf -c 256 -n 16 --temp 0 -t 4 -st --no-warmup \
  -p "Premis: Badan saya menimbulkan bayang-bayang di atas rumput.
Apa penyebab yang paling mungkin?
A. Matahari terbit.
B. Rumput telah dipotong.
Jawaban:"
```

Expect a continuation that restates an option (e.g. `A. Matahari terbit`).

## 5. Host reference numbers (x86 only — not comparable to A133)

| Quant | Size | pp128 t/s | tg32 t/s |
|---|---|---|---|
| Q4_K_M | ~90 MB | 1717 | 170 |
| Q8_0 | ~119 MB | 2231 | 151 |
| F16 | ~222 MB | 2749 | 107 |

(16 threads on a 24-core x86 host.) On A133, decode is typically on the
order of ~8–15 tok/s with llama.cpp NEON kernels.

## 6. Letter-score artifacts

Per-item margin vectors, metrics JSON, checksums, and board logs shipped
with this repository live under `artifacts/`. See the top-level README.

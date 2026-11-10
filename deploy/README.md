# copa3 114M 端侧部署包（A133，4×Cortex-A53）

模型：`copa3_Q4_K_M.gguf`（90MB）/ `copa3_Q8_0.gguf`（119MB，精度参照）。
词表 34,775（32k 印尼任务子集 + 870 任务 OOV + 1,905 merge 闭包 token），与训练分词逐条一致。

## 0. 一键全流程（推荐）

`Z:\SDK\paper\indo02b\deploy\run_a133_all.bat`

```bat
cd /d Z:\SDK\paper\indo02b\deploy

rem Q4_K_M，4 线程
run_a133_all.bat

rem Q8_0 / 两个都跑
run_a133_all.bat Q8
run_a133_all.bat both

rem 指定线程；跑完把日志也传回 175
run_a133_all.bat --threads 4 --upload
```

自动完成：检查 adb 设备 → 推模型（大小 + SHA256 校验）→ 首次推送内置的 llama.cpp
armv7 包（`llama-android-armv7/`，32 位 A53 交叉编译，见 §2）→ 板上跑 llama-bench、
冒烟生成、峰值 RSS → 回拉日志到 `deploy\logs\<时间戳>\` → 打印摘要。

注意：这块 A133 是 **32 位 Android**（armeabi-v7a userspace），官方 arm64 包不能用。

日志：`sysinfo.txt`、`bench_*.txt`、`gen_*.txt`（末行 `exit=… peak_rss_kb=…`）。

## 1. 推送到板子

Android（adb）：
```bash
adb push copa3_Q4_K_M.gguf /data/local/tmp/
adb push bench.sh /data/local/tmp/
```

Linux（scp）：
```bash
scp copa3_Q4_K_M.gguf bench.sh user@<board-ip>:~/
```

## 1b. Windows（Z: 盘）一键部署

脚本：`Z:\SDK\paper\indo02b\deploy\deploy_a133.bat`（需 adb 在 PATH）。

```bat
cd /d Z:\SDK\paper\indo02b\deploy

rem 首次：从 175 拉取 GGUF 到本目录
deploy_a133.bat --fetch

rem 推送 Q4_K_M + bench.sh，校验大小与 SHA256
deploy_a133.bat

rem 推送后直接在板上跑 llama-bench
deploy_a133.bat --bench

rem 只推 Q8，或 Q4+Q8 都推
deploy_a133.bat Q8
deploy_a133.bat both
```

板上 llama.cpp 路径默认 `/data/local/tmp/copa3/llama`（由 `run_a133_all.bat` 推送的 armv7 包），
不同则改 bat 顶部 `DEV_LLAMA`。

## 2. 板上 llama.cpp（armv7，已内置）

`run_a133_all.bat` 会把 `llama-android-armv7/`（`llama-bench` + `llama-completion`）推到
`/data/local/tmp/copa3/llama`，无需在板上编译。32 位包在 175 上交叉编译（NDK r27d）：

```bash
bash deploy/build_android_armv7.sh
```

要点（与 `paper/qwen` 板上验证过的配方一致）：armeabi-v7a + NEON + `-march=armv8-a`
（等价 `-mcpu=cortex-a53 -mfpu=neon-fp-armv8`），静态链接 c++，`unicode-data.cpp` 用 -O0 预编译。
产物属性：ELF 32-bit ARM, Tag_CPU_arch v7 / NEONv1，与 qwen 的 a53 包完全一致。

源码补丁：`ggml-cpu-impl.h` 里 `vcvtnq_s32_f32` 的 32 位兼容 shim 需按 `__ARM_ARCH < 8`
收窄，否则与 clang 18 的 arm_neon.h 冲突（armv8 32-bit 已自带该 intrinsic）。

Termux 自编译（可选）：
```bash
pkg install -y cmake clang git
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-completion llama-bench
```

Debian/Ubuntu：
```bash
sudo apt install -y cmake g++ git
# 同上 cmake/build
```

## 3. 测速 / 测内存

```bash
# 吞吐：4 线程 = A133 全部核心（与 qwen 板上协议一致）
./llama/llama-bench -m copa3_Q4_K_M.gguf -p 128,256 -n 32 -t 4 -r 3
```

内存（任一方式）：
```bash
# Android：c=256 是为 992MB 内存留余量
./llama/llama-completion -m copa3_Q4_K_M.gguf -c 256 -n 8 --temp 0 -t 4 \
  -p "Apa ibu kota Indonesia?" &
ps -A -o RSS,NAME | grep llama
```

## 4. 功能自检

```bash
./llama/llama-completion -m copa3_Q4_K_M.gguf -c 256 -n 16 --temp 0 -t 4 -st --no-warmup \
  -p "Premis: Badan saya menimbulkan bayang-bayang di atas rumput.
Apa penyebab yang paling mungkin?
A. Matahari terbit.
B. Rumput telah dipotong.
Jawaban:"
```
期望续写复述选项（如 "A. Matahari terbit"）。

## 5. 参考值（x86 24 核，仅作对照，不可与 A133 比）

| 量化 | 文件 | pp128 t/s | tg32 t/s |
|---|---|---|---|
| Q4_K_M | 90MB | 1717 | 170 |
| Q8_0 | 119MB | 2231 | 151 |
| F16 | 222MB | 2749 | 107 |

（16 线程；Q4 的 tg 高于 Q8 是因为带宽受限，精度低反而更快。）

A133 预期：decode 8–15 tok/s（4×A53 无 NEON 加速库依赖，llama.cpp 自带 NEON）。

## 6. 精度记录（e-CARE dev 2130，0-shot 字母协议）

| 后端 | 一致数 | 准确率 |
|---|---|---|
| HF bf16（34,775 词表） | 1067 | 50.09% |
| GGUF F16 | 1075 | 50.47% |
| GGUF Q8_0 | 1077 | 50.56% |
| GGUF Q4_K_M | 1053 | 49.44% |

HF↔GGUF 符号一致率 ≥92%，所有翻标签均发生在 |margin|≤0.15 的近平局。

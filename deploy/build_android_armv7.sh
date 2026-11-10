#!/usr/bin/env bash
# Build 32-bit armeabi-v7a llama.cpp for A133 (Cortex-A53, Android 10).
# Recipe mirrors paper/qwen/scripts/build_android_cpu.sh (a53 variant):
# armeabi-v7a + NEON + -mcpu=cortex-a53 -mfpu=neon-fp-armv8 via GGML_CPU_ARM_ARCH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${NDK:-/home//user/code/ndk/android-ndk-r27d}"
LLAMA="${LLAMA:-$ROOT/llama.cpp-src}"
BUILD="${BUILD:-$ROOT/build-android-armv7}"
OUT="${OUT:-$ROOT/deploy/llama-android-armv7}"
API="${API:-24}"

TC="$NDK/build/cmake/android.toolchain.cmake"
[ -f "$TC" ] || { echo "missing NDK toolchain: $TC" >&2; exit 1; }
STRIP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
CLANGXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
SYSROOT="$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot"

cmake -S "$LLAMA" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$TC" \
  -DANDROID_ABI=armeabi-v7a \
  -DANDROID_PLATFORM="android-$API" \
  -DANDROID_ARM_NEON=ON \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_OPENCL=OFF \
  -DCMAKE_UNITY_BUILD=OFF \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_CXX_FLAGS="-Wno-c++11-narrowing" \
  -DGGML_CPU_ARM_ARCH=armv8-a

# unicode-data.cpp at -O3 with the A53 schedule takes very long; precompile -O0.
OBJ="$BUILD/src/CMakeFiles/llama.dir/unicode-data.cpp.o"
mkdir -p "$(dirname "$OBJ")"
"$CLANGXX" --target="armv7-none-linux-androideabi$API" --sysroot="$SYSROOT" \
  -DGGML_USE_CPU -I"$LLAMA/src/." -I"$BUILD/src" -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -DANDROID -fdata-sections -ffunction-sections -funwind-tables -no-canonical-prefixes \
  -march=armv7-a -mthumb -O0 -DNDEBUG -fPIC -o "$OBJ" -c "$LLAMA/src/unicode-data.cpp"

cmake --build "$BUILD" -j"$(nproc)" --target llama-bench llama-completion

mkdir -p "$OUT"
for b in llama-bench llama-completion; do
  cp -f "$BUILD/bin/$b" "$OUT/$b"
  [ -x "$STRIP" ] && "$STRIP" "$OUT/$b" || true
  echo "== $b"
  file "$OUT/$b"
  readelf -A "$OUT/$b" | grep -E "Tag_CPU_arch|Tag_FP_arch|Tag_Advanced_SIMD_arch" || true
done
echo "OK: $OUT"

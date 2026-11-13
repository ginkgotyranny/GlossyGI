#!/usr/bin/env bash
# Compiles the probe shaders with Falcor's slangc, builds gpu_probe.cpp with the OpenHarmony NDK,
# pushes both to the phone and runs the tests. See gpu_probe.cpp for what each test answers.
#
# Usage (Git Bash, phone connected over hdc):
#   ohos/probes/gpu/run.sh [info|encode|scatter|adam|rays|features|features64|readback|load16|pipelines ...]     default: all
# Results: ohos/_build/gpu_probe/results.txt (RESULT lines) and full.log (everything).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/../.."
REPO="$ROOT/.."
SDK="${OHOS_NATIVE_SDK:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native}"
HDC="${HDC:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe}"
SLANGC="${SLANGC:-$REPO/external/packman/slang/bin/slangc.exe}"
OUT="$ROOT/_build/gpu_probe"
DEVICE_DIR=/data/local/tmp/ncr_gpu_probe
SHADERS="PrimEncode ClsEncode PrimScatter GridAdam RayThroughput Features ReadAfterWrite Load16"

mkdir -p "$OUT"
for s in $SHADERS; do
  "$SLANGC" "$HERE/$s.cs.slang" -I "$HERE" -target spirv -entry main -stage compute -o "$OUT/$s.spv"
done
"$SLANGC" "$HERE/Features.cs.slang" -DUSE_INT64=1 -target spirv -entry main -stage compute -o "$OUT/Features64.spv"
"$SDK/llvm/bin/clang++.exe" --target=aarch64-linux-ohos --sysroot="$SDK/sysroot" -std=c++17 -O2 \
  "$HERE/gpu_probe.cpp" -o "$OUT/gpu_probe" -lvulkan -static-libstdc++

# Layout files for the test "pipelines" (before MSYS_NO_PATHCONV: python is a Windows program).
if [ -n "${NCR_BAKE_SPIRV:-}" ]; then
  python "$ROOT/tools/spirv_info.py" --layout --summary "$NCR_BAKE_SPIRV"/*.spv > "$OUT/baked_summary.txt"
fi

export MSYS_NO_PATHCONV=1
"$HDC" shell "mkdir -p $DEVICE_DIR" > /dev/null
for f in gpu_probe Features64.spv $(for s in $SHADERS; do echo "$s.spv"; done); do
  (cd "$OUT" && "$HDC" file send "$f" "$DEVICE_DIR/$f" > /dev/null)
done
# Test "pipelines": the SPIR-V of a Falcor bake (NCR_BAKE_DIR), given as NCR_BAKE_SPIRV=<bake dir>/spirv.
if [ -n "${NCR_BAKE_SPIRV:-}" ]; then
  "$HDC" shell "rm -rf $DEVICE_DIR/baked && mkdir -p $DEVICE_DIR/baked" > /dev/null
  (cd "$NCR_BAKE_SPIRV" && ls *.spv | sed 's/\.spv$//' > list.txt &&
    for f in list.txt *.spv *.layout; do "$HDC" file send "$f" "$DEVICE_DIR/baked/$f" > /dev/null; done)
fi
"$HDC" shell "cd $DEVICE_DIR && chmod 755 gpu_probe && ./gpu_probe $* 2>&1" | tr -d '\r' | tee "$OUT/full.log"
grep -E "^(RESULT|FAIL)" "$OUT/full.log" > "$OUT/results.txt" || true
echo "results: $OUT/results.txt"

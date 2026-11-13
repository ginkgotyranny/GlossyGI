#!/usr/bin/env bash
# Builds npu_probe.cpp with the OpenHarmony NDK, pushes it with the models exported by
# ohos/npu/export_models.py to the phone, runs every model and collects the RESULT lines.
#
# Usage (Git Bash, phone connected over hdc):
#   ohos/probes/npu_mlp/run.sh [--runs N] [--kind om|omc|both] [name filter ...]
# Results: ohos/_build/npu_probe/results.txt (RESULT lines) and full.log (everything).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/../.."
SDK="${OHOS_NATIVE_SDK:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native}"
HDC="${HDC:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe}"
MODELS="${NCR_NPU_MODELS:-$ROOT/_build/npu_models}"
OUT="$ROOT/_build/npu_probe"
DEVICE_DIR=/data/local/tmp/ncr_npu_probe

RUNS=30
KIND=both
FILTERS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    *) FILTERS+=("$1"); shift ;;
  esac
done

mkdir -p "$OUT"
"$SDK/llvm/bin/clang++.exe" --target=aarch64-linux-ohos --sysroot="$SDK/sysroot" -std=c++17 -O2 \
  "$HERE/npu_probe.cpp" -o "$OUT/npu_probe" -lneural_network_core -lqos -static-libstdc++

export MSYS_NO_PATHCONV=1
"$HDC" shell "mkdir -p $DEVICE_DIR" > /dev/null
(cd "$OUT" && "$HDC" file send npu_probe "$DEVICE_DIR/npu_probe" > /dev/null)
"$HDC" shell "chmod 755 $DEVICE_DIR/npu_probe"

: > "$OUT/full.log"
for onnx in "$MODELS"/*.onnx; do
  name="$(basename "$onnx" .onnx)"
  if [ ${#FILTERS[@]} -gt 0 ]; then
    keep=0
    for f in "${FILTERS[@]}"; do case "$name" in *"$f"*) keep=1 ;; esac; done
    [ $keep -eq 1 ] || continue
  fi
  files=()
  [ "$KIND" != omc ] && [ -f "$MODELS/${name}_fp16.om" ] && files+=("${name}_fp16.om")
  if [ "$KIND" != om ]; then
    for omc in "$MODELS/${name}"_fp16_*.omc; do [ -f "$omc" ] && files+=("$(basename "$omc")"); done
  fi
  [ ${#files[@]} -gt 0 ] || { echo "skip $name: no converted model" | tee -a "$OUT/full.log"; continue; }
  data_arg=""
  if [ -f "$MODELS/$name.in0.bin" ]; then
    for v in "$MODELS/$name".in*.bin "$MODELS/$name".out*.bin; do
      (cd "$MODELS" && "$HDC" file send "$(basename "$v")" "$DEVICE_DIR/$(basename "$v")" > /dev/null)
    done
    data_arg="--data $DEVICE_DIR/$name"
  fi
  for file in "${files[@]}"; do
    (cd "$MODELS" && "$HDC" file send "$file" "$DEVICE_DIR/$file" > /dev/null)
    echo "=== $file" >> "$OUT/full.log"
    "$HDC" shell "cd $DEVICE_DIR && ./npu_probe $DEVICE_DIR/$file --runs $RUNS $data_arg 2>&1" | tr -d '\r' | tee -a "$OUT/full.log" | grep -E "^(RESULT|FAIL)" || echo "FAIL $file: no result line (see full.log)"
  done
  # Test vectors are large; do not leave them on the phone.
  "$HDC" shell "rm -f $DEVICE_DIR/$name.in*.bin $DEVICE_DIR/$name.out*.bin" > /dev/null
done
grep -E "^(RESULT|FAIL)" "$OUT/full.log" > "$OUT/results.txt" || true
echo "results: $OUT/results.txt"

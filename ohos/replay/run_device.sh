#!/usr/bin/env bash
# Builds the replayer for OpenHarmony, pushes it with a packed bake, a checkpoint, its own shaders
# and the NPU models to the phone, replays the baked frame there and brings the results back for
# the comparison with Falcor.
#
# Usage (Git Bash, phone connected over hdc):
#   ohos/replay/run_device.sh <bake dir> <checkpoint.ckpt> [replayer options, e.g. --networks npu --stages]
# The bake must have been packed (python ohos/tools/bake_pack.py <bake dir>). For --networks npu the
# models must exist: python ohos/npu/export_models.py --only inf_256x256 --only _train_ --om
# Results: ohos/_build/replay/device_out (the same files the PC replayer writes) and device.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
REPO="$ROOT/.."
SDK="${OHOS_NATIVE_SDK:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native}"
HDC="${HDC:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe}"
SLANGC="${SLANGC:-$REPO/external/packman/slang/bin/slangc.exe}"
MODELS="${NCR_NPU_MODELS:-$ROOT/_build/npu_models}"
BAKE="$(cd "$1" && pwd)"
CHECKPOINT="$2"
shift 2
OUT="$ROOT/_build/replay"
DEVICE_DIR=/data/local/tmp/ncr_replay

mkdir -p "$OUT/device/shaders" "$OUT/device/models"
SLANG_ARGS=(-I "$HERE/shaders" -target spirv -entry main -stage compute)
"$SLANGC" "$HERE/shaders/EncodeInputs.cs.slang" -DSPECULAR=0 "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/EncodeDiff.spv"
"$SLANGC" "$HERE/shaders/EncodeInputs.cs.slang" -DSPECULAR=1 "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/EncodeSpec.spv"
"$SLANGC" "$HERE/shaders/DecodeOutputs.cs.slang" "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/DecodeOutputs.spv"
"$SLANGC" "$HERE/shaders/EncodeTargets.cs.slang" "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/EncodeTargets.spv"
"$SLANGC" "$HERE/shaders/ScatterGradients.cs.slang" -DSPECULAR=0 "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/ScatterDiff.spv"
"$SLANGC" "$HERE/shaders/ScatterGradients.cs.slang" -DSPECULAR=1 "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/ScatterSpec.spv"
"$SLANGC" "$HERE/shaders/GridOptimizer.cs.slang" "${SLANG_ARGS[@]}" -o "$OUT/device/shaders/GridOptimizer.spv"
"$SDK/llvm/bin/clang++.exe" --target=aarch64-linux-ohos --sysroot="$SDK/sysroot" -std=c++17 -O2 \
  "$HERE/ncr_replay.cpp" -o "$OUT/device/ncr_replay" -lvulkan -lneural_network_core -lqos -static-libstdc++
for net in diff spec; do
  for graph in inf_256x256 train_4096 train_8192 train_16384; do
    if [ -f "$MODELS/ncr_${net}_${graph}_fp16.om" ]; then cp "$MODELS/ncr_${net}_${graph}_fp16.om" "$OUT/device/models/"; fi
  done
done
cp "$CHECKPOINT" "$OUT/device/checkpoint.ckpt"

# What the phone needs from the bake: the index, the SPIR-V, and the data the passes read. The
# reference results (the ray-batch buffers and the images the passes write) stay on the PC.
python - "$BAKE" "$OUT/device" "$OUT/bundle.tar" <<'EOF'
import os, sys, tarfile
bake, device, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = ["replay.txt"]
for line in open(os.path.join(bake, "replay.txt")):
    p = line.split()
    if p[0] == "pass":
        files.append(f"spirv/{p[1]}.spv")
    elif p[0] == "buffer" and p[4] == "1":
        files.append(f"data/buffer{p[1]}.bin")
    elif p[0] == "texture" and p[10] == "0":
        files.append(f"data/texture{p[1]}.bin")
    elif p[0] == "element" and p[5] == "uniform":
        files.append(p[7])
files = sorted(set(files))
with tarfile.open(out, "w") as tar:
    for f in files:
        tar.add(os.path.join(bake, f), arcname="bake/" + f)
    for folder, _, names in os.walk(device):
        for name in names:
            path = os.path.join(folder, name)
            tar.add(path, arcname=os.path.relpath(path, device).replace(os.sep, "/"))
print(f"bundle: {len(files)} bake files plus the replayer, {os.path.getsize(out) / 1e6:.1f} MB")
EOF

export MSYS_NO_PATHCONV=1
"$HDC" shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR/out" > /dev/null
(cd "$OUT" && "$HDC" file send bundle.tar "$DEVICE_DIR/bundle.tar" > /dev/null)
"$HDC" shell "cd $DEVICE_DIR && tar -xf bundle.tar && rm bundle.tar && chmod 755 ncr_replay && ./ncr_replay bake checkpoint.ckpt out --models models $* 2>&1" | tr -d '\r' | tee "$OUT/device.log"

# The results, as one file.
"$HDC" shell "cd $DEVICE_DIR && tar -cf out.tar out" > /dev/null
rm -rf "$OUT/device_out" "$OUT/out.tar"
(cd "$OUT" && "$HDC" file recv "$DEVICE_DIR/out.tar" out.tar > /dev/null && tar -xf out.tar && mv out device_out && rm out.tar)
"$HDC" shell "rm -rf $DEVICE_DIR/out.tar $DEVICE_DIR/out" > /dev/null
echo "results: $OUT/device_out"

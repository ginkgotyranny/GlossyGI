#!/usr/bin/env bash
# Builds the replayer for OpenHarmony, pushes it with a packed bake, the model, the runtime's shaders
# and the NPU models to the phone, runs it there from the shell and brings the results back for the
# comparison with Falcor.
#
# Usage (Git Bash, phone connected over hdc):
#   ohos/replay/run_device.sh <bake dir> <bake|checkpoint.ckpt> [replayer options, e.g. --networks npu --frames 60]
# "bake" replays from the model at the start of the baked frame (the bake's before.ckpt and
# before_working.bin); a checkpoint path is loaded the way Falcor loads one.
# The bake must have been packed (python ohos/tools/bake_pack.py <bake dir>). For --networks npu the
# models must exist: python ohos/npu/export_models.py --only inf_256x256 --only _train_ --om
# Results: ohos/_build/replay/device_out (the same files the PC replayer writes) and device.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
SDK="${OHOS_NATIVE_SDK:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native}"
HDC="${HDC:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe}"
MODELS="${NCR_NPU_MODELS:-$ROOT/_build/npu_models}"
BAKE="$(cd "$1" && pwd)"
MODEL="$2"
shift 2
OUT="$ROOT/_build/replay"
DEVICE_DIR=/data/local/tmp/ncr_replay

rm -rf "$OUT/device"
mkdir -p "$OUT/device/shaders" "$OUT/device/models"
python "$HERE/compile_shaders.py" "$OUT/device/shaders"
"$SDK/llvm/bin/clang++.exe" --target=aarch64-linux-ohos --sysroot="$SDK/sysroot" -std=c++17 -O2 \
  "$HERE/ncr_replay.cpp" "$HERE/ncr_runtime.cpp" -o "$OUT/device/ncr_replay" \
  -lvulkan -lneural_network_core -lnative_buffer -lnative_window -lqos -static-libstdc++
for net in diff spec; do
  for graph in inf_256x256 train_2048 train_4096 train_8192 train_16384; do
    if [ -f "$MODELS/ncr_${net}_${graph}_fp16.om" ]; then cp "$MODELS/ncr_${net}_${graph}_fp16.om" "$OUT/device/models/"; fi
  done
done
DEVICE_MODEL=bake
if [ "$MODEL" != "bake" ]; then
  cp "$MODEL" "$OUT/device/checkpoint.ckpt"
  DEVICE_MODEL=checkpoint.ckpt
fi

# What the phone needs from the bake (ohos/tools/bake_files.py); the reference results stay on the PC.
python - "$BAKE" "$OUT/device" "$OUT/bundle.tar" "$ROOT/tools" <<'EOF'
import os, sys, tarfile
bake, device, out, tools = sys.argv[1:5]
sys.path.insert(0, tools)
from bake_files import runtime_files
files = runtime_files(bake)
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
"$HDC" shell "cd $DEVICE_DIR && tar -xf bundle.tar && rm bundle.tar && chmod 755 ncr_replay && ./ncr_replay bake $DEVICE_MODEL out --models models $* 2>&1" | tr -d '\r' | tee "$OUT/device.log"

# The results, as one file.
"$HDC" shell "cd $DEVICE_DIR && tar -cf out.tar out" > /dev/null
rm -rf "$OUT/device_out" "$OUT/out.tar"
(cd "$OUT" && "$HDC" file recv "$DEVICE_DIR/out.tar" out.tar > /dev/null && tar -xf out.tar && mv out device_out && rm out.tar)
"$HDC" shell "rm -rf $DEVICE_DIR/out.tar $DEVICE_DIR/out" > /dev/null
echo "results: $OUT/device_out"

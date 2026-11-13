#!/usr/bin/env bash
# Builds the replayer for OpenHarmony, pushes it with a packed bake and a checkpoint to the phone,
# replays the baked frame there and brings the results back for the comparison with Falcor.
#
# Usage (Git Bash, phone connected over hdc):
#   ohos/replay/run_device.sh <bake dir> <checkpoint.ckpt> [--stages]
# The bake must have been packed (python ohos/tools/bake_pack.py <bake dir>).
# Results: ohos/_build/replay/device_out (the same files the PC replayer writes) and device.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
SDK="${OHOS_NATIVE_SDK:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native}"
HDC="${HDC:-/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe}"
BAKE="$(cd "$1" && pwd)"
CHECKPOINT="$2"
OUT="$ROOT/_build/replay"
DEVICE_DIR=/data/local/tmp/ncr_replay

mkdir -p "$OUT"
"$SDK/llvm/bin/clang++.exe" --target=aarch64-linux-ohos --sysroot="$SDK/sysroot" -std=c++17 -O2 \
  "$HERE/ncr_replay.cpp" -o "$OUT/ncr_replay" -lvulkan -static-libstdc++

# What the phone needs from the bake: the index, the SPIR-V, and the data the passes read. The
# reference results (the ray-batch buffers and the images the passes write) stay on the PC.
python - "$BAKE" "$OUT/bundle.tar" <<'EOF'
import os, sys, tarfile
bake, out = sys.argv[1], sys.argv[2]
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
print(f"bundle: {len(files)} files, {os.path.getsize(out) / 1e6:.1f} MB")
EOF
cp "$CHECKPOINT" "$OUT/checkpoint.ckpt"

export MSYS_NO_PATHCONV=1
"$HDC" shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR/out" > /dev/null
for f in ncr_replay bundle.tar checkpoint.ckpt; do
  (cd "$OUT" && "$HDC" file send "$f" "$DEVICE_DIR/$f" > /dev/null)
done
"$HDC" shell "cd $DEVICE_DIR && tar -xf bundle.tar && rm bundle.tar && chmod 755 ncr_replay && ./ncr_replay bake checkpoint.ckpt out ${3:-} 2>&1" | tr -d '\r' | tee "$OUT/device.log"

# The results, as one file.
"$HDC" shell "cd $DEVICE_DIR && tar -cf out.tar out" > /dev/null
rm -rf "$OUT/device_out" "$OUT/out.tar"
(cd "$OUT" && "$HDC" file recv "$DEVICE_DIR/out.tar" out.tar > /dev/null && tar -xf out.tar && mv out device_out && rm out.tar)
"$HDC" shell "rm -rf $DEVICE_DIR/out.tar $DEVICE_DIR/out" > /dev/null
echo "results: $OUT/device_out"

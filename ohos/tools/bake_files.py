#!/usr/bin/env python
"""The files of a packed bake that the runtime reads.

    python ohos/tools/bake_files.py <bake dir>

Prints them relative to the bake, one per line: the index (replay.txt), the SPIR-V of every pass, the
buffers and textures the passes read, the uniform blocks, and the model at the start of the baked
frame (before.ckpt, before_working.bin) when the bake has them. The reference results Falcor left at
the end of the frame (the buffers and images the passes write) are not among them: they stay on the
PC. Used by ohos/replay/run_device.sh and the app's asset sync. Standard library only.
"""
import os
import sys


def runtime_files(bake):
    files = ["replay.txt"]
    for line in open(os.path.join(bake, "replay.txt")):
        p = line.split()
        if not p:
            continue
        if p[0] == "pass":
            files.append(f"spirv/{p[1]}.spv")
        elif p[0] == "buffer" and p[4] == "1":
            files.append(f"data/buffer{p[1]}.bin")
        elif p[0] == "texture" and p[10] == "0":
            files.append(f"data/texture{p[1]}.bin")
        elif p[0] == "element" and p[5] == "uniform":
            files.append(p[7])
    for model in ("before.ckpt", "before_working.bin"):
        if os.path.exists(os.path.join(bake, model)):
            files.append(model)
    return sorted(set(files))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for f in runtime_files(sys.argv[1]):
        print(f)


if __name__ == "__main__":
    main()

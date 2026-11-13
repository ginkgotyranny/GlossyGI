#!/usr/bin/env python
"""Compares what the replayer produced (ohos/replay) with what Falcor produced in the baked frame.

    python ohos/tools/replay_compare.py <bake dir> <replay output dir> [--all]

The bake holds, as reference, every image and ray-batch buffer as it was at the end of the frame
Falcor baked; the replayer writes the same resources after replaying that frame. For each one this
prints how many bytes differ and, read as the type the bake records for it, how large the difference
is. Resources are named by the shader variables they are bound to.

Without --all only the resources the render path writes are listed (those the replay left untouched
and Falcor left zero are skipped). Exit code: 0 if the G-buffer agrees (mean difference below 0.1% of the
mean value; for the hit records, the same triangle in 99.9% of the pixels) and the final image is within
50 dB PSNR of Falcor's. The final image is judged by PSNR
because FXAA turns differences of 1e-4 in its input into visible ones at the few pixels that sit on
its edge threshold.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import bake_check  # noqa: E402

TEXTURE_TYPES = {"RGBA32Float": np.float32, "R32Float": np.float32, "RGBA32Uint": np.uint32, "R32Uint": np.uint32,
                 "RGBA16Float": np.float16, "RG32Float": np.float32}
# The images that decide: the final color and the G-buffer of the primary hits.
DECISIVE = {"gColor", "gPos", "gNormal", "gAlbedo", "gRoughness", "gVBuffer", "gDir"}


def names_of(bake, kind, id_):
    """Shader variables a resource is bound to, shortest first: 'Pass.path'."""
    found = set()
    for call in bake["calls"].values():
        for (path, index), record in call["binds"].items():
            if record[0] == kind and int(record[1]) == id_:
                found.add(path)
    roles = [role for role, (k, i) in bake["named"].items() if kind == "buffer" and i == id_]
    return sorted(found, key=len), roles


def compare(reference, ours, dtype):
    n = min(reference.size, ours.size) // np.dtype(dtype).itemsize * np.dtype(dtype).itemsize
    different_bytes = int((reference[:n] != ours[:n]).sum())
    a = reference[:n].view(dtype)
    b = ours[:n].view(dtype)
    if np.issubdtype(dtype, np.floating):
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        finite = np.isfinite(a64) & np.isfinite(b64)
        difference = np.abs(np.where(finite, a64 - b64, 0.0))
        scale = np.abs(np.where(finite, a64, 0.0)).mean()
        return different_bytes, n, f"mean |difference| {difference.mean():.3e} of mean |value| {scale:.3e}, largest {difference.max():.3e}", \
            difference.mean() / max(scale, 1e-30)
    different = int((a != b).sum())
    return different_bytes, n, f"{different} of {a.size} values differ", different / max(a.size, 1)


def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    if len(arguments) != 2:
        sys.exit(__doc__)
    bake = bake_check.load(arguments[0])
    failures = 0
    rows = []
    for file in sorted(os.listdir(arguments[1])):
        if not file.endswith(".bin"):
            continue
        kind = "texture" if file.startswith("texture") else "buffer"
        id_ = int(file[len(kind):-4])
        reference = np.fromfile(os.path.join(arguments[0], "data", file), dtype=np.uint8)
        ours = np.fromfile(os.path.join(arguments[1], file), dtype=np.uint8)
        paths, roles = names_of(bake, kind, id_)
        label = (roles[0] if roles else "") + (" = " if roles and paths else "") + ", ".join(paths[:3])
        if kind == "texture":
            dtype = TEXTURE_TYPES.get(bake["textures"][id_]["format"], np.uint8)
        else:
            # Ray-batch buffers: fp16 network inputs, otherwise 32-bit floats or integers.
            dtype = np.float16 if any(p.endswith(("diffInput", "specInput")) for p in paths) else \
                np.uint32 if any(p.endswith(("Active", "Index", "Src", "Counts", "counts", "VBuffer")) for p in paths) else np.float32
        if not show_all and not reference.any() and not ours.any():
            continue
        different_bytes, n, text, relative = compare(reference, ours, dtype)
        decisive = kind == "texture" and bool(DECISIVE & set(paths))
        ok = relative < 1e-3
        if kind == "texture" and "gVBuffer" in paths:
            # Hit records: the first two words say which triangle was hit, the other two are the
            # barycentrics as float bits, which differ in their last places from one GPU to another.
            a = reference.view(np.uint32).reshape(-1, 4)
            b = ours.view(np.uint32).reshape(-1, 4)
            same = float(((a[:, 0] == b[:, 0]) & (a[:, 1] == b[:, 1])).mean())
            text += f"; same triangle in {100.0 * same:.3f}% of the pixels"
            ok = same >= 0.999
        if kind == "texture" and "gColor" in paths:
            a = np.clip(reference.view(np.float32).reshape(-1, 4)[:, :3], 0.0, 1.0).astype(np.float64)
            b = np.clip(ours.view(np.float32).reshape(-1, 4)[:, :3], 0.0, 1.0).astype(np.float64)
            psnr = 10.0 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-30))
            text += f"; PSNR {psnr:.1f} dB"
            ok = psnr >= 50.0
        if decisive and not ok:
            failures += 1
        rows.append((not decisive, kind, id_, f"{'*' if decisive else ' '} {kind} {id_:3d} {label[:58]:58s} "
                     f"{100.0 * different_bytes / max(n, 1):6.2f}% bytes differ; {text}{'' if ok else '   <-- DIFFERENT'}"))
    for _, _, _, line in sorted(rows):
        print(line)
    print("* decides the result: " + ("the final image and the G-buffer agree with Falcor" if not failures else f"{failures} DIFFERENT"))
    sys.exit(failures)


if __name__ == "__main__":
    main()

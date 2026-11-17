#!/usr/bin/env python
"""Checks the track of an animated bake against a second bake of the same scene at a later frame.

    python ohos/tools/track_check.py <bake dir> <reference bake dir>

The track (track.txt, track.bin: BakeExport.cpp) says what the scene's data is at every sample of
the animation, as the bytes that differ from the baked frame. Sample k is the scene k frames after
the baked frame, so a bake of that later frame, made with the clock at the track's rate
(NCR_SMOKE_FPS), holds the same scene. This rebuilds the scene of that sample from the bake and its
track and compares it, byte for byte, with what the reference bake holds:

  - the scene's uniform block (camera, light counts, environment)
  - every scene buffer the passes have bound, except vertices and indices (transforms, lights, ...)
  - the transform of every geometry instance (what the top-level acceleration structure is built from)
  - the scene bounds, which the passes get as bboxMin / bboxMax

Padding is left out: Falcor does not initialise it, so it differs from run to run. That is the
members named _padding in the uniform block, and in a buffer every byte outside the track's ranges:
the track is made by comparing every byte of every frame with the baked frame, so a byte it does
not list did not change in that run, and a difference there against another run is not the scene's.

Exit code: the number of differences. Standard library and numpy.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import bake_check  # noqa: E402

HEADER = 28   # of a record: events (uint32), bounds (6 floats)


def scene_resources(bake):
    """Shader path -> buffer id, and the scene block's file, from the first call that binds the scene."""
    for step in sorted(bake["calls"]):
        binds = bake["calls"][step]["binds"]
        if ("gScene", 0) not in binds:
            continue
        buffers = {}
        for (path, index), record in binds.items():
            if record[0] == "buffer" and path.startswith("gScene.") and path not in ("gScene.vertices.data", "gScene.indexData.data"):
                buffers[f"{path}[{index}]"] = int(record[1])
        return buffers, binds[("gScene", 0)][2]
    sys.exit("no call binds the scene")


def bounds_of(bake, folder):
    """bboxMin and bboxMax as some pass got them."""
    for step in sorted(bake["calls"]):
        call = bake["calls"][step]
        for block, members in bake["uniforms"].get(call["pass"], {}).items():
            if "bboxMin" in members and (block, 0) in call["binds"] and call["binds"][(block, 0)][0] == "uniform":
                data = open(os.path.join(folder, call["binds"][(block, 0)][2]), "rb").read()
                return struct.unpack_from("3f", data, members["bboxMin"][0]) + struct.unpack_from("3f", data, members["bboxMax"][0])
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    folder, reference_folder = sys.argv[1], sys.argv[2]
    bake, reference = bake_check.load(folder), bake_check.load(reference_folder)
    lines = [line.split() for line in open(os.path.join(folder, "track.txt")) if line.strip()]
    samples, rate, record_bytes = int(lines[0][1]), float(lines[0][2]), int(lines[0][3])
    sample = reference["frame"]["frameCount"] - bake["frame"]["frameCount"]
    if not 0 <= sample < samples:
        sys.exit(f"the reference is {sample} frames after the bake; the track has {samples} samples")
    with open(os.path.join(folder, "track.bin"), "rb") as f:
        f.seek(20 + sample * record_bytes)
        record = f.read(record_bytes)
    events = struct.unpack_from("I", record, 0)[0]
    names = [name for bit, name in ((1, "camera"), (2, "edit"), (4, "light"), (8, "object")) if events & bit]
    print(f"sample {sample} of {samples} ({sample / rate:.3f} s after the baked frame); events: {'+'.join(names) or 'none'}")

    # The scene of the baked frame, then the sample's bytes over it.
    buffers, block_file = scene_resources(bake)
    reference_buffers, reference_block_file = scene_resources(reference)
    state = {("uniform", 0): bytearray(open(os.path.join(folder, block_file), "rb").read())}
    for path, id_ in buffers.items():
        state[("buffer", id_)] = bytearray(open(os.path.join(folder, "data", f"buffer{id_}.bin"), "rb").read())
    count = max(g[0] for g in bake["geometry"]) + 1
    transforms = np.zeros((count, 12), dtype=np.float32)
    for g in bake["geometry"]:
        transforms[g[0]] = g[12:24]
    state[("geometry", 0)] = bytearray(transforms.tobytes())
    at, changed = HEADER, {}
    for kind, id_, offset, size in ((p[1], int(p[2]), int(p[3]), int(p[4])) for p in lines[1:]):
        target = state[(kind, id_)]
        target[offset:offset + size] = record[at:at + size]
        changed[(kind, id_)] = changed.get((kind, id_), 0) + size
        at += size

    failures = 0
    tracked_bytes = {}   # (kind, id) -> mask of the bytes the track lists
    for kind, id_, offset, size in ((p[1], int(p[2]), int(p[3]), int(p[4])) for p in lines[1:]):
        mask = tracked_bytes.setdefault((kind, id_), np.zeros(len(state[(kind, id_)]), dtype=bool))
        mask[offset:offset + size] = True

    def report(label, key, theirs, decisive=None):
        nonlocal failures
        ours = state[key]
        a, b = np.frombuffer(bytes(ours), dtype=np.uint8), np.frombuffer(theirs, dtype=np.uint8)
        tracked = tracked_bytes.get(key, np.zeros(a.size, dtype=bool))
        if a.size != b.size:
            failures += 1
            print(f"  {label:58s} {a.size} bytes against {b.size} DIFFER")
            return
        decisive = tracked if decisive is None else decisive
        different, elsewhere = int(((a != b) & decisive).sum()), int(((a != b) & ~decisive).sum())
        failures += 1 if different else 0
        print(f"  {label:58s} {a.size:9d} bytes, {int(tracked.sum()):6d} in the track: " + ("equal" if not different else f"{different} bytes DIFFER")
              + (f" ({elsewhere} bytes of padding differ between the runs)" if elsewhere else ""))

    # In the uniform block everything counts but the padding members.
    block = np.ones(len(state[("uniform", 0)]), dtype=bool)
    for members in bake["uniforms"].values():
        for name, (offset, size) in members.get("gScene", {}).items():
            if "_padding" in name:
                block[offset:offset + size] = False
    report("scene uniform block (gScene)", ("uniform", 0), open(os.path.join(reference_folder, reference_block_file), "rb").read(), block)
    for path, id_ in sorted(buffers.items()):
        if path not in reference_buffers:
            print(f"  {path}: not bound in the reference")
            failures += 1
            continue
        theirs = open(os.path.join(reference_folder, "data", f"buffer{reference_buffers[path]}.bin"), "rb").read()
        report(path, ("buffer", id_), theirs)
    theirs = np.zeros((count, 12), dtype=np.float32)
    for g in reference["geometry"]:
        theirs[g[0]] = g[12:24]
    ours = np.frombuffer(bytes(state[("geometry", 0)]), dtype=np.float32).reshape(-1, 12)
    # The geometry records are text: compare as numbers.
    worst = float(np.abs(ours - theirs).max())
    moved = int((np.abs(ours - transforms).max(axis=1) > 0).sum())
    failures += 1 if worst > 1e-6 else 0
    print(f"  {'geometry transforms':58s} {count:9d} instances, {moved} moved: " + ("equal" if worst <= 1e-6 else f"largest difference {worst:.3e} DIFFERS"))
    ours_bounds, theirs_bounds = struct.unpack_from("6f", record, 4), bounds_of(reference, reference_folder)
    if theirs_bounds is not None:
        same = ours_bounds == theirs_bounds
        failures += 0 if same else 1
        print(f"  {'scene bounds':58s} " + ("equal" if same else f"{ours_bounds} against {theirs_bounds} DIFFER"))
    print("the track's sample is the reference's scene" if not failures else f"{failures} DIFFERENT")
    sys.exit(failures)


if __name__ == "__main__":
    main()

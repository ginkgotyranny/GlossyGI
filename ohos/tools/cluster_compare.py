#!/usr/bin/env python
"""The cluster-reuse state of a replay against Falcor's, per pixel.

    python ohos/tools/cluster_compare.py <bake> <replay out> [--reference <bake of the last frame>] [--again <second replay out>]

Since the desktop's stage 4 snapshots, ConeTraceScreen hands its rays out from a per-frame budget with
atomic counters, so which pixel gets a ray when the budget runs short depends on the order the GPU's
threads arrive in: two runs of the same frame on the same GPU no longer give the same history, in
Falcor or here. The buffers are therefore not expected to be equal. What is compared:

  - the share of pixels whose entry differs, for the history and lock-state pairs, the visibility
    check's recovery countdown, the frequency check's counters and the previous frame's specular mask;
  - with --again (the same replay run a second time), the same between the two replays: the noise
    floor. A replay that differs from Falcor about as much as from itself is as good as it gets;
  - the ray budget's counters (farValidationStats) side by side.

<bake> is the bake the replay ran (its replay.txt names the buffers); --reference is the bake whose
contents are Falcor's state after the replay's last frame (default: <bake> itself, for one frame).
The pairs swap roles every frame, so with an odd number of frames between the two the buffer written
last is the other one: both pairings are tried and the closer one reported.
"""
import argparse
import os

import numpy as np

ROLES = ["cluster.history", "cluster.lockState", "cluster.vcdRecovery"]
SINGLE = ["cluster.frequencyCounts", "cluster.prevSpecActive"]


def named(folder):
    ids, pixels = {}, 0
    for line in open(os.path.join(folder, "replay.txt")):
        p = line.split()
        if p and p[0] == "named":
            ids[p[1]] = int(p[2])
        elif p and p[0] == "frame":
            pixels = int(p[1]) * int(p[2])
    return ids, pixels


def load(folder, ids, role, data):
    path = os.path.join(folder, "data" if data else "", f"buffer{ids[role]}.bin")
    return np.fromfile(path, dtype=np.uint8) if os.path.exists(path) else None


def pixels_differing(a, b, pixels):
    n = min(a.size, b.size) // pixels * pixels
    if n == 0:
        return float("nan")
    return float(np.mean(np.any((a[:n] != b[:n]).reshape(pixels, -1), axis=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bake")
    parser.add_argument("out")
    parser.add_argument("--reference")
    parser.add_argument("--again")
    args = parser.parse_args()
    ids, pixels = named(args.bake)
    ref_dir = args.reference or args.bake
    ref_ids, _ = named(ref_dir)

    def report(label, ours, theirs, again):
        text = f"  {label:28s} pixels differing from Falcor {100 * pixels_differing(ours, theirs, pixels):7.3f}%"
        if again is not None:
            text += f"   between two replays {100 * pixels_differing(ours, again, pixels):7.3f}%"
        print(text)

    for role in ROLES:
        if role + "0" not in ids or role + "0" not in ref_ids:
            continue
        ours = [load(args.out, ids, role + str(i), False) for i in (0, 1)]
        theirs = [load(ref_dir, ref_ids, role + str(i), True) for i in (0, 1)]
        straight = sum(pixels_differing(ours[i], theirs[i], pixels) for i in (0, 1))
        crossed = sum(pixels_differing(ours[i], theirs[1 - i], pixels) for i in (0, 1))
        swap = crossed < straight
        for i in (0, 1):
            again = load(args.again, ids, role + str(i), False) if args.again else None
            report(f"{role}{i}" + (" (swapped)" if swap else ""), ours[i], theirs[1 - i if swap else i], again)
    for role in SINGLE:
        if role not in ids or role not in ref_ids:
            continue
        again = load(args.again, ids, role, False) if args.again else None
        report(role, load(args.out, ids, role, False), load(ref_dir, ref_ids, role, True), again)
    role = "cluster.farValidationStats"
    if role in ids and role in ref_ids:
        print("  ray budget counters, Falcor:", load(ref_dir, ref_ids, role, True).view(np.uint32).tolist())
        print("  ray budget counters, replay:", load(args.out, ids, role, False).view(np.uint32)[:16].tolist())
        if args.again:
            print("  ray budget counters, again: ", load(args.again, ids, role, False).view(np.uint32)[:16].tolist())


if __name__ == "__main__":
    main()

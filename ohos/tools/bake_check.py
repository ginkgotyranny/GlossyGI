#!/usr/bin/env python
"""Checks a Falcor bake (NCR_BAKE_DIR, see BakeExport.cpp) for completeness.

    python ohos/tools/bake_check.py <bake dir> [--verbose]

The bake records the frame as it executed: every call of a compute pass with what was bound at each
shader path, and the host steps in between. For every call this resolves each binding of the pass's
SPIR-V module, by its shader path, to what the bake recorded as bound there: a buffer, a texture, a
sampler, a uniform block or the acceleration structure. The replayer does the same join, so a
binding this tool cannot resolve is one the phone could not bind. Also prints what the bake
contains. Exit code: number of unresolved bindings.

`load(bake_dir)` returns the parsed index for other tools.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import spirv_info  # noqa: E402


def load(folder):
    bake = {"folder": folder, "passes": {}, "buffers": {}, "textures": {}, "samplers": {}, "uniforms": {},
            "calls": {}, "program": [], "geometry": [], "frame": None, "named": {}, "values": {}}
    with open(os.path.join(folder, "bake.txt")) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            kind = p[0]
            if kind == "frame":
                bake["frame"] = {"width": int(p[1]), "height": int(p[2]), "frameCount": int(p[3])}
            elif kind == "pass":
                bake["passes"][p[1]] = {"group": [int(x) for x in p[2:5]]}
            elif kind == "call":
                step = int(p[1])
                bake["calls"][step] = {"pass": p[2], "dims": [int(x) for x in p[3:6]], "binds": {}}
                bake["program"].append(("call", step))
            elif kind == "cbind":
                # cbind <step> <path> <index> <kind> [...]
                bake["calls"][int(p[1])]["binds"][(p[2], int(p[3]))] = p[4:]
            elif kind == "host":
                bake["program"].append(("host", int(p[1]), p[2:]))
            elif kind == "buffer":
                bake["buffers"][int(p[1])] = {"bytes": int(p[2]), "structSize": int(p[3]), "elements": int(p[4])}
            elif kind == "texture":
                bake["textures"][int(p[1])] = {"type": p[2], "format": p[3], "width": int(p[4]), "height": int(p[5]),
                                               "depth": int(p[6]), "mips": int(p[7]), "layers": int(p[8])}
            elif kind == "sampler":
                bake["samplers"][int(p[1])] = p[2:]
            elif kind == "uniform":
                bake["uniforms"].setdefault(p[1], {}).setdefault(p[2], {})[p[3]] = (int(p[4]), int(p[5]))
            elif kind == "named":
                bake["named"][p[1]] = (p[2], int(p[3]))
            elif kind == "value":
                bake["values"][p[1]] = float(p[2])
            elif kind == "geometry":
                bake["geometry"].append([int(x) for x in p[1:13]] + [float(x) for x in p[13:25]])
    return bake


def call_batch(bake, call):
    """The ray batch (render, lhs, rhs) whose buffers a call has bound, or '-'."""
    roles = {}
    for role, (kind, id_) in bake["named"].items():
        roles.setdefault(id_, set()).add(role.split(".")[0])
    votes = {}
    for record in call["binds"].values():
        if record[0] == "buffer":
            for batch in roles.get(int(record[1]), ()):
                if batch != "dummy":
                    votes[batch] = votes.get(batch, 0) + 1
    if not votes:
        return "-"
    # A pass over two batches (SampleRHS, ResolveRHS) is filed under the one it reads the samples from.
    return max(sorted(votes), key=lambda b: votes[b])


EXPECTED = {
    "storage_buffer": {"buffer"},
    "uniform_buffer": {"uniform"},
    "storage_image": {"texture"},
    "sampled_image": {"texture"},
    "sampler": {"sampler"},
    "acceleration_structure": {"accel"},
}


def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    verbose = "--verbose" in sys.argv
    if len(arguments) != 1:
        sys.exit(__doc__)
    bake = load(arguments[0])
    frame = bake["frame"]
    print(f"frame {frame['width']}x{frame['height']}, baked at frame {frame['frameCount']}; {len(bake['passes'])} passes in "
          f"{len(bake['calls'])} calls, {len(bake['buffers'])} buffers ({sum(b['bytes'] for b in bake['buffers'].values()) / 1e6:.1f} MB), "
          f"{len(bake['textures'])} textures, {len(bake['samplers'])} samplers, {len(bake['geometry'])} geometry instances in "
          f"{len({g[1] for g in bake['geometry']})} TLAS instances")
    formats = {}
    for t in bake["textures"].values():
        formats[t["format"]] = formats.get(t["format"], 0) + 1
    print("texture formats: " + ", ".join(f"{name} x{count}" for name, count in sorted(formats.items())))

    modules = {name: spirv_info.reflect(os.path.join(arguments[0], "spirv", name + ".spv")) for name in bake["passes"]}
    unresolved = 0
    for entry in bake["program"]:
        if entry[0] == "host":
            print(f"  {entry[1]:3d} host  {' '.join(entry[2])}")
            continue
        step = entry[1]
        call = bake["calls"][step]
        missing, null, total = [], [], 0
        for b in modules[call["pass"]]["bindings"]:
            for index in range(max(b["count"], 1)):
                total += 1
                record = call["binds"].get((b["name"], index))
                if record is None:
                    missing.append(f"{b['name']}[{index}] ({b['descriptor']})")
                elif record[0] == "null":
                    null.append(f"{b['name']}[{index}]")
                elif record[0] not in EXPECTED.get(b["descriptor"], set()):
                    missing.append(f"{b['name']}[{index}]: SPIR-V wants {b['descriptor']}, the bake has {record[0]}")
        unresolved += len(missing)
        print(f"  {step:3d} call  {call['pass']:20s} on {call_batch(bake, call):6s} dims {call['dims']}  {total:4d} binding elements, "
              f"{len(null):3d} bound to nothing, {len(missing)} unresolved")
        for m in missing[:12]:
            print(f"          unresolved: {m}")
        if verbose:
            for n in null[:12]:
                print(f"          null: {n}")
    print(f"{unresolved} unresolved binding(s)" if unresolved else "every binding resolves")
    sys.exit(min(unresolved, 255))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Repairs what the Slang version Falcor ships (2024.1.34) leaves out of its SPIR-V and a strict
Vulkan driver needs.

    python ohos/tools/spirv_fix.py <file.spv> [...]        patches the files in place

Non-uniform resource indices. Falcor's material system picks a texture and a sampler out of
descriptor arrays by an index that differs from pixel to pixel (the material of whatever was hit).
Vulkan requires such an access to be decorated NonUniform on the pointer into the array; without it
a driver may use one index for a whole group of invocations. Desktop drivers index per invocation
regardless. The Maleoon 955 follows the specification: undecorated, the pixels of a thread group
take the texture of a neighbour's material.

Slang decorates only the loaded object, and for a texture handed to a generic sampling function
not even that, whether or not the source says NonUniformResourceIndex. So this tool does not rely
on what is there: every access into an array of images or samplers with an index that is not a
constant gets the decoration, on the pointer, on what is loaded through it and on the sampled images
built from that. It also declares the capabilities this needs.

`fix(words)` returns (new words, number of decorations added); used by bake_pack.py.
"""
import struct
import sys

OP_CAPABILITY, OP_DECORATE, OP_MEMBER_DECORATE = 17, 71, 72
OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE, OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_POINTER = 25, 26, 27, 28, 29, 32
OP_CONSTANT, OP_VARIABLE, OP_LOAD, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN = 43, 59, 61, 65, 66
OP_SAMPLED_IMAGE, OP_IMAGE = 86, 100
NON_UNIFORM = 5300
CAP_SHADER_NON_UNIFORM, CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM = 5301, 5307


def fix(words):
    words = list(words)
    instructions = []          # (offset, opcode, args)
    i = 5
    while i < len(words):
        opcode, count = words[i] & 0xffff, words[i] >> 16
        instructions.append((i, opcode, words[i + 1:i + count]))
        i += count

    opaque, arrays, pointers, constants, variables = set(), {}, {}, set(), {}
    decorated, capabilities = set(), set()
    for _, op, args in instructions:
        if op in (OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE):
            opaque.add(args[0])
        elif op in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY):
            arrays[args[0]] = args[1]
        elif op == OP_TYPE_POINTER:
            pointers[args[0]] = args[2]
        elif op == OP_CONSTANT:
            constants.add(args[1])
        elif op == OP_VARIABLE:
            variables[args[1]] = args[0]
        elif op == OP_DECORATE and len(args) >= 2 and args[1] == NON_UNIFORM:
            decorated.add(args[0])
        elif op == OP_CAPABILITY:
            capabilities.add(args[0])

    # Variables that are arrays of images or samplers.
    resource_arrays = {var for var, pointer in variables.items() if arrays.get(pointers.get(pointer)) in opaque}

    marked = set(decorated)
    # Pointers into those arrays through an index that is not a constant.
    for _, op, args in instructions:
        if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and args[2] in resource_arrays and len(args) > 3 and args[3] not in constants:
            marked.add(args[1])
    # The pointer of a load Slang marked, what is loaded through a marked pointer, and what is built from it.
    changed = True
    while changed:
        changed = False
        for _, op, args in instructions:
            new = None
            if op == OP_LOAD:
                if args[2] in marked and args[1] not in marked:
                    new = args[1]
                elif args[1] in marked and args[2] not in marked and args[2] not in variables:
                    new = args[2]
            elif op == OP_SAMPLED_IMAGE and (args[2] in marked or args[3] in marked) and args[1] not in marked:
                new = args[1]
            elif op == OP_IMAGE and args[2] in marked and args[1] not in marked:
                new = args[1]
            if new is not None:
                marked.add(new)
                changed = True
    wanted = marked - decorated
    needed = [c for c in (CAP_SHADER_NON_UNIFORM, CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM) if c not in capabilities] if marked else []
    if not wanted and not needed:
        return words, 0

    last_capability = max(offset for offset, op, _ in instructions if op == OP_CAPABILITY)
    last_decorate = max(offset for offset, op, _ in instructions if op in (OP_DECORATE, OP_MEMBER_DECORATE))
    out = []
    for offset, opcode, args in instructions:
        out.append((len(args) + 1) << 16 | opcode)
        out.extend(args)
        if offset == last_capability:
            for capability in needed:
                out.extend([2 << 16 | OP_CAPABILITY, capability])
        if offset == last_decorate:
            for target in sorted(wanted):
                out.extend([3 << 16 | OP_DECORATE, target, NON_UNIFORM])
    return words[:5] + out, len(wanted)


def fix_file(path):
    with open(path, "rb") as f:
        data = f.read()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError(f"{path}: not SPIR-V")
    fixed, added = fix(words)
    if len(fixed) != len(words):
        with open(path, "wb") as f:
            f.write(struct.pack(f"<{len(fixed)}I", *fixed))
    return added


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        print(f"{path}: {fix_file(path)} NonUniform decorations added")


if __name__ == "__main__":
    main()

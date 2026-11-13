#!/usr/bin/env python
"""Lists what a SPIR-V module needs from a Vulkan device and which resources it binds.

    python ohos/tools/spirv_info.py <file.spv> [...]          capabilities, extensions, bindings
    python ohos/tools/spirv_info.py --summary <file.spv> ...  one line per file
    python ohos/tools/spirv_info.py --layout <file.spv> ...   also write <file>.layout: "set binding type count"
                                                              lines, read by ohos/probes/gpu (test "pipelines")

Used on the SPIR-V Falcor bakes for the phone (NCR_BAKE_DIR): the Maleoon driver rejects a module
that declares a capability it lacks (for example Int64), and the replayer needs every binding.
`reflect(path)` returns the same information as a dictionary.
"""
import struct
import sys

CAPABILITIES = {
    0: "Matrix", 1: "Shader", 9: "Float16", 10: "Float64", 11: "Int64", 12: "Int64Atomics", 22: "Int16", 39: "Int8",
    49: "StorageImageExtendedFormats", 50: "ImageQuery", 55: "StorageImageReadWithoutFormat",
    56: "StorageImageWriteWithoutFormat", 61: "GroupNonUniform", 62: "GroupNonUniformVote",
    63: "GroupNonUniformArithmetic", 64: "GroupNonUniformBallot", 65: "GroupNonUniformShuffle",
    4433: "StorageBuffer16BitAccess", 4434: "UniformAndStorageBuffer16BitAccess", 4435: "StoragePushConstant16",
    4472: "RayQueryKHR", 4478: "RayTraversalPrimitiveCullingKHR", 4479: "RayTracingKHR",
    5301: "ShaderNonUniform", 5302: "RuntimeDescriptorArray", 5306: "UniformBufferArrayNonUniformIndexing",
    5307: "SampledImageArrayNonUniformIndexing", 5308: "StorageBufferArrayNonUniformIndexing",
    5347: "PhysicalStorageBufferAddresses", 6016: "DemoteToHelperInvocation",
}
# What the Maleoon 955 driver does not have (ohos/README.md): a module declaring any of these fails.
UNSUPPORTED_ON_PHONE = {"Int64", "Int64Atomics", "Float64"}
STORAGE_CLASSES = {0: "UniformConstant", 2: "Uniform", 9: "PushConstant", 12: "StorageBuffer"}


def reflect(path):
    with open(path, "rb") as f:
        data = f.read()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError(f"{path}: not SPIR-V")
    version = f"{(words[1] >> 16) & 0xff}.{(words[1] >> 8) & 0xff}"

    def string(args):
        raw = b"".join(struct.pack("<I", w) for w in args)
        return raw.split(b"\0", 1)[0].decode("utf-8", "replace")

    capabilities, extensions, names, decorations, types, constants, variables = [], [], {}, {}, {}, {}, []
    i = 5
    while i < len(words):
        opcode, count = words[i] & 0xffff, words[i] >> 16
        args = words[i + 1:i + count]
        if opcode == 17:
            capabilities.append(CAPABILITIES.get(args[0], str(args[0])))
        elif opcode == 10:
            extensions.append(string(args))
        elif opcode == 5:
            names[args[0]] = string(args[1:])
        elif opcode == 71:
            decorations.setdefault(args[0], {})[args[1]] = args[2] if len(args) > 2 else True
        elif opcode == 43:      # OpConstant
            constants[args[1]] = args[2]
        elif opcode == 32:      # OpTypePointer
            types[args[0]] = ("pointer", args[2])
        elif opcode == 30:
            types[args[0]] = ("struct",)
        elif opcode == 28:
            types[args[0]] = ("array", args[1], args[2])
        elif opcode == 29:
            types[args[0]] = ("runtime_array", args[1])
        elif opcode == 25:      # OpTypeImage: sampled type, dim, depth, arrayed, ms, sampled, format
            types[args[0]] = ("image", args[2], args[6])
        elif opcode == 26:
            types[args[0]] = ("sampler",)
        elif opcode == 27:
            types[args[0]] = ("sampled_image",)
        elif opcode == 5341:
            types[args[0]] = ("acceleration_structure",)
        elif opcode == 59:      # OpVariable: type, id, storage class
            if args[2] in STORAGE_CLASSES:
                variables.append((args[1], args[0], args[2]))
        i += count

    bindings = []
    for var, pointer_type, storage in variables:
        pointee = types.get(pointer_type, ("?", 0))[1]
        count = 1
        kind = types.get(pointee, ("?",))
        if kind[0] == "array":
            count, pointee = constants.get(kind[2], 0), kind[1]
        elif kind[0] == "runtime_array":
            count, pointee = 0, kind[1]          # 0: unsized
        kind = types.get(pointee, ("?",))
        type_decorations = decorations.get(pointee, {})
        if STORAGE_CLASSES[storage] == "PushConstant":
            descriptor = "push_constant"
        elif kind[0] == "struct":
            # Block in Uniform storage is a uniform buffer; BufferBlock, or Block in StorageBuffer, a storage buffer.
            descriptor = "storage_buffer" if (3 in type_decorations or storage == 12) else "uniform_buffer"
        elif kind[0] == "image":
            descriptor = "storage_image" if kind[2] == 2 else "sampled_image"
        elif kind[0] == "sampled_image":
            descriptor = "combined_image_sampler"
        else:
            descriptor = kind[0]
        d = decorations.get(var, {})
        bindings.append({"name": names.get(var, f"%{var}"), "set": d.get(34), "binding": d.get(33),
                         "descriptor": descriptor, "count": count})
    bindings.sort(key=lambda b: (b["set"] if b["set"] is not None else -1, b["binding"] if b["binding"] is not None else -1))
    return {"version": version, "capabilities": capabilities, "extensions": extensions, "bindings": bindings}


def main():
    arguments = sys.argv[1:]
    summary = "--summary" in arguments
    layout = "--layout" in arguments
    paths = [a for a in arguments if not a.startswith("--")]
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        info = reflect(path)
        blocked = sorted(set(info["capabilities"]) & UNSUPPORTED_ON_PHONE)
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if layout:
            with open(path[:-4] + ".layout", "w", newline="\n") as f:
                for b in info["bindings"]:
                    if b["descriptor"] != "push_constant":
                        f.write(f"{b['set']} {b['binding']} {b['descriptor']} {b['count']}\n")
        if summary:
            sets = sorted({b["set"] for b in info["bindings"] if b["set"] is not None})
            print(f"{name:26s} SPIR-V {info['version']}  {len(info['bindings']):3d} bindings in sets {sets}  "
                  f"caps: {' '.join(c for c in info['capabilities'] if c not in ('Shader', 'Matrix'))}"
                  + (f"  ** not on the phone: {', '.join(blocked)}" if blocked else ""))
            continue
        print(f"{path}: SPIR-V {info['version']}")
        print("  capabilities: " + ", ".join(info["capabilities"]))
        print("  extensions:   " + ", ".join(info["extensions"]))
        if blocked:
            print("  NOT AVAILABLE ON THE PHONE: " + ", ".join(blocked))
        for b in info["bindings"]:
            where = "push constant" if b["descriptor"] == "push_constant" else f"set {b['set']} binding {b['binding']}"
            array = "" if b["count"] == 1 else ("[]" if b["count"] == 0 else f"[{b['count']}]")
            print(f"  {where:22s} {b['descriptor'] + array:28s} {b['name']}")


if __name__ == "__main__":
    main()

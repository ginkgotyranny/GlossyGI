#!/usr/bin/env python
"""Turns a Falcor bake (NCR_BAKE_DIR) into the index the replayer reads: <bake dir>/replay.txt.

    python ohos/tools/bake_pack.py <bake dir>

The bake names what is bound by shader path; the SPIR-V modules name their bindings the same way and
carry the descriptor set and binding numbers. This tool joins the two once, on the PC, so that the
replayer (ohos/replay), which also runs on the phone, needs neither a SPIR-V parser nor Falcor's
names. It also repairs the SPIR-V in place where Falcor's Slang leaves out what a strict driver needs
(spirv_fix.py). One record per line:

    frame <width> <height> <frameCount>
    buffer <id> <bytes> <structSize> <load 0|1>      0: a ray-batch buffer, written by the passes; its baked
                                                     contents are reference results, not input
    texture <id> <2D|Cube|3D> <VkFormat> <bytesPerPixel> <width> <height> <depth> <mips> <layers> <storage 0|1>
    sampler <id> <min> <mag> <mip> <addressU> <addressV> <addressW> <maxAnisotropy> <lodBias> <minLod> <maxLod>
    scene vertices <bufferId> <stride> | scene indices <bufferId>
    geometry <globalInstanceID> <tlasInstance> <geometryIndex> <vbOffset> <ibOffset> <vertexCount> <indexCount>
             <index16> <frontFaceCW> <opaque> <12 floats: 3x4 transform>
    pass <Pass> <groupX> <groupY> <groupZ>
    binding <Pass> <set> <binding> <descriptor type> <count> <shader path>
    uniform <Pass> <block> <member> <byteOffset> <byteSize>
  the frame as it executed, <step> counting calls and host steps together:
    call <step> <Pass> <dimX> <dimY> <dimZ> <batch>     <batch>: render, lhs, rhs or - (whose buffers it has bound)
    element <step> <set> <binding> <index> buffer|texture|sampler <id>
    element <step> <set> <binding> <index> uniform <bytes> <file>
    element <step> <set> <binding> <index> accel|null
    host <step> clearcounts <batch> | prefixsum <batch> diff|spec <count> | infer <batch> | train <batch>
    named <role> <bufferId>
    value <name> <number>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import bake_check  # noqa: E402
import spirv_fix  # noqa: E402
import spirv_info  # noqa: E402

# Falcor ResourceFormat name -> (VkFormat, bytes per pixel)
FORMATS = {
    "R8Unorm": (9, 1), "RG8Unorm": (16, 2), "RGBA8Unorm": (37, 4), "RGBA8UnormSrgb": (43, 4),
    "BGRA8Unorm": (44, 4), "BGRA8UnormSrgb": (50, 4), "BGRX8Unorm": (44, 4), "BGRX8UnormSrgb": (50, 4),
    "R16Float": (76, 2), "RG16Float": (83, 4), "RGBA16Float": (97, 8),
    "R32Uint": (98, 4), "R32Float": (100, 4), "RG32Uint": (101, 8), "RG32Float": (103, 8),
    "RGB32Float": (106, 12), "RGBA32Uint": (107, 16), "RGBA32Float": (109, 16),
    "R16Unorm": (70, 2), "RGBA16Unorm": (91, 8),
}
DESCRIPTORS = ["sampler", "combined_image_sampler", "sampled_image", "storage_image", "uniform_texel_buffer",
               "storage_texel_buffer", "uniform_buffer", "storage_buffer"]   # index = VkDescriptorType
ACCELERATION_STRUCTURE = 1000150000


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = sys.argv[1]
    bake = bake_check.load(folder)
    out = []
    frame = bake["frame"]
    out.append(f"frame {frame['width']} {frame['height']} {frame['frameCount']}")

    problems, repaired, modules, pass_lines = [], 0, {}, []
    for name, info in bake["passes"].items():
        repaired += spirv_fix.fix_file(os.path.join(folder, "spirv", name + ".spv"))
        modules[name] = spirv_info.reflect(os.path.join(folder, "spirv", name + ".spv"))
        pass_lines.append(f"pass {name} {info['group'][0]} {info['group'][1]} {info['group'][2]}")
        for b in modules[name]["bindings"]:
            if b["descriptor"] == "push_constant":
                problems.append(f"{name}: push constants are not supported ({b['name']})")
                continue
            descriptor = ACCELERATION_STRUCTURE if b["descriptor"] == "acceleration_structure" else DESCRIPTORS.index(b["descriptor"])
            pass_lines.append(f"binding {name} {b['set']} {b['binding']} {descriptor} {max(b['count'], 1)} {b['name']}")
        for block, members in bake["uniforms"].get(name, {}).items():
            for member, (offset, size) in members.items():
                pass_lines.append(f"uniform {name} {block} {member} {offset} {size}")

    # The frame: every call joined with its module, and which textures a shader writes (storage usage).
    program_lines, storage_textures, scene = [], set(), {}
    for entry in bake["program"]:
        if entry[0] == "host":
            program_lines.append(f"host {entry[1]} " + " ".join(entry[2]))
            continue
        step = entry[1]
        call = bake["calls"][step]
        program_lines.append(f"call {step} {call['pass']} {call['dims'][0]} {call['dims'][1]} {call['dims'][2]} {bake_check.call_batch(bake, call)}")
        for b in modules[call["pass"]]["bindings"]:
            if b["descriptor"] == "push_constant":
                continue
            for index in range(max(b["count"], 1)):
                record = call["binds"].get((b["name"], index))
                if record is None:
                    problems.append(f"step {step} ({call['pass']}): nothing recorded for {b['name']}[{index}]")
                    continue
                if record[0] == "texture" and b["descriptor"] == "storage_image":
                    storage_textures.add(int(record[1]))
                if record[0] == "buffer" and b["name"] in ("gScene.vertices.data", "gScene.indexData.data"):
                    scene[b["name"]] = int(record[1])
                program_lines.append(f"element {step} {b['set']} {b['binding']} {index} " + " ".join(record))

    scratch = {id_ for kind, id_ in bake["named"].values()}
    for id_, b in sorted(bake["buffers"].items()):
        out.append(f"buffer {id_} {b['bytes']} {b['structSize']} {0 if id_ in scratch else 1}")
    for id_, t in sorted(bake["textures"].items()):
        if t["format"] not in FORMATS:
            problems.append(f"texture {id_}: format {t['format']} is not in the table of bake_pack.py")
            continue
        vk_format, bytes_per_pixel = FORMATS[t["format"]]
        out.append(f"texture {id_} {t['type']} {vk_format} {bytes_per_pixel} {t['width']} {t['height']} {t['depth']} "
                   f"{t['mips']} {t['layers']} {1 if id_ in storage_textures else 0}")
    for id_, s in sorted(bake["samplers"].items()):
        out.append(f"sampler {id_} " + " ".join(s[:10]))
    if "gScene.vertices.data" in scene:
        vertices = scene["gScene.vertices.data"]
        out.append(f"scene vertices {vertices} {bake['buffers'][vertices]['structSize']}")
    if "gScene.indexData.data" in scene:
        out.append(f"scene indices {scene['gScene.indexData.data']}")
    for g in bake["geometry"]:
        # bake: id tlas geometryIndex meshID materialID vbOffset ibOffset vertexCount indexCount index16 frontFaceCW opaque + 12 floats
        fields = [g[0], g[1], g[2], g[5], g[6], g[7], g[8], g[9], g[10], g[11]]
        out.append("geometry " + " ".join(str(x) for x in fields) + " " + " ".join(repr(x) for x in g[12:24]))
    out += pass_lines + program_lines
    for role, (kind, id_) in bake["named"].items():
        out.append(f"named {role} {id_}")
    for name, value in bake["values"].items():
        out.append(f"value {name} {value:g}")

    if problems:
        for p in problems:
            print("PROBLEM: " + p)
        sys.exit(1)
    with open(os.path.join(folder, "replay.txt"), "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"{folder}/replay.txt: {len(out)} records, {len(bake['passes'])} passes in {len(bake['calls'])} calls, "
          f"{len(storage_textures)} of {len(bake['textures'])} textures are shader outputs; "
          f"{repaired} NonUniform decorations added to the SPIR-V")


if __name__ == "__main__":
    main()

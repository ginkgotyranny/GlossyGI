#!/usr/bin/env python
"""Puts what the app reads at run time into the HAP: entry/src/main/resources/rawfile/ncr.

    python ohos/app/tools/sync_assets.py [--scenes <dir>] [--scene <name> ...] [--models <dir>] [--train-tile N]

  scenes/<name>/   the runtime's files of each bake (ohos/tools/bake_files.py) in <dir>
                   (default ohos/_build/scenes, filled by ohos/tools/bake_scene.py); --scene picks some
  shaders/         the runtime's own SPIR-V (ohos/replay/compile_shaders.py)
  models/          the NPU graphs: inference (256x256) and training (--train-tile samples, default
                   4096) of both networks (ohos/npu/export_models.py --om)
  manifest.txt     "<sha1>\\t<path>" per file; the app extracts the assets again when it changes

Files are only copied when their content changed, and files that are no longer wanted are removed, so
that hvigor's incremental build stays quick. Standard library only.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
APP = os.path.dirname(HERE)
OHOS = os.path.dirname(APP)
RAWFILE = os.path.join(APP, "entry", "src", "main", "resources", "rawfile")
TARGET = os.path.join(RAWFILE, "ncr")
sys.path.insert(0, os.path.join(OHOS, "tools"))
from bake_files import runtime_files  # noqa: E402


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=os.path.join(OHOS, "_build", "scenes"))
    parser.add_argument("--scene", action="append", help="only these scenes (repeatable)")
    parser.add_argument("--models", default=os.path.join(OHOS, "_build", "npu_models"))
    parser.add_argument("--train-tile", type=int, default=4096)
    args = parser.parse_args()

    # Source path for every asset path.
    wanted = {}
    if not os.path.isdir(args.scenes):
        sys.exit(f"no scenes in {args.scenes}: bake one first (python ohos/tools/bake_scene.py bathroom)")
    names = args.scene or sorted(n for n in os.listdir(args.scenes) if os.path.exists(os.path.join(args.scenes, n, "replay.txt")))
    if not names:
        sys.exit(f"no packed bake in {args.scenes}")
    for name in names:
        bake = os.path.join(args.scenes, name)
        if not os.path.exists(os.path.join(bake, "before.ckpt")):
            sys.exit(f"{bake} has no before.ckpt: bake it again with the current Falcor build")
        for f in runtime_files(bake):
            wanted[f"scenes/{name}/{f}"] = os.path.join(bake, f)
    shader_dir = os.path.join(OHOS, "_build", "app_shaders")
    subprocess.run([sys.executable, os.path.join(OHOS, "replay", "compile_shaders.py"), shader_dir], check=True)
    for f in sorted(os.listdir(shader_dir)):
        wanted[f"shaders/{f}"] = os.path.join(shader_dir, f)
    for net in ("diff", "spec"):
        for graph in ("inf_256x256", f"train_{args.train_tile}"):
            model = os.path.join(args.models, f"ncr_{net}_{graph}_fp16.om")
            if not os.path.exists(model):
                sys.exit(f"missing {model}: python ohos/npu/export_models.py --only inf_256x256 --only _train_ --om")
            wanted[f"models/{os.path.basename(model)}"] = model

    # Copy what changed, remove what is no longer wanted.
    copied, lines, total = 0, [], 0
    for path in sorted(wanted):
        source, target = wanted[path], os.path.join(TARGET, path)
        digest = sha1(source)
        if not os.path.exists(target) or os.path.getsize(target) != os.path.getsize(source) or sha1(target) != digest:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
        lines.append(f"{digest}\t{path}")
        total += os.path.getsize(source)
    removed = 0
    for folder, _, files in os.walk(TARGET):
        for f in files:
            rel = os.path.relpath(os.path.join(folder, f), TARGET).replace(os.sep, "/")
            if rel != "manifest.txt" and rel not in wanted:
                os.remove(os.path.join(folder, f))
                removed += 1
    with open(os.path.join(TARGET, "manifest.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"assets: {len(wanted)} files, {total / 1e6:.1f} MB (scenes: {', '.join(names)}); {copied} copied, {removed} removed")


if __name__ == "__main__":
    main()

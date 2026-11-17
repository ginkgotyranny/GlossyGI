#!/usr/bin/env python
"""Bakes a scene for the phone: Falcor records frame 0 of the NeuralRadiosity pass in Online Train mode,
and the bake is packed for the runtime.

    python ohos/tools/bake_scene.py <scene> [static|anim] [--frame N] [--res WxH] [--props "{...}"] [--rate HZ] [--out <dir>]
    python ohos/tools/bake_scene.py --all          every scene and variant under media/scenes
    python ohos/tools/bake_scene.py --list         what --all would bake

<scene> is a folder under media/scenes; the variant is the scene file, as for run.bat: static is
scene-v4.pyscene, anim is scene-animation.pyscene (the default is static when the scene has it).
The bake goes to ohos/_build/scenes/<scene>/<variant> unless --out says otherwise;
ohos/app/tools/push_scenes.sh takes the scenes from there. Falcor runs headless on the Vulkan backend
(the bake needs SPIR-V) with the pass properties of the project's defaults plus --props, and also
captures the frame (<out>/frames) for ohos/tools/display_check.py. Mogwai is
build/windows-vs2022/bin/Release/Mogwai.exe unless MOGWAI is set.

Frame 0 is what the runtime should start from: Falcor's model is fresh then, its optimizer moments
empty, as the runtime's are.

An animated scene also gets its track: Falcor plays one loop of the scene's animation at --rate
samples per second (default 30) and the bake records what the scene's data is at every sample
(BakeExport.cpp), which the runtime plays back.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


SCENE_FILES = {"static": "scene-v4.pyscene", "anim": "scene-animation.pyscene"}


def variants():
    """Every (scene, variant) under media/scenes that has its scene file."""
    root = os.path.join(REPO, "media", "scenes")
    found = []
    for scene in sorted(os.listdir(root)):
        for variant, file in SCENE_FILES.items():
            if os.path.exists(os.path.join(root, scene, file)):
                found.append((scene, variant))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?")
    parser.add_argument("variant", nargs="?", choices=sorted(SCENE_FILES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--res", default="640x360")
    parser.add_argument("--props", default="", help="extra pass properties, a Python dict literal")
    parser.add_argument("--rate", type=int, default=30, help="track samples per second of an animated scene")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.list:
        for scene, variant in variants():
            print(scene, variant)
        return
    if args.all:
        failed = []
        for scene, variant in variants():
            if not bake(args, scene, variant, None):
                failed.append(f"{scene} {variant}")
        if failed:
            sys.exit("failed: " + ", ".join(failed))
        return
    if not args.scene:
        parser.error("a scene, --all or --list")
    available = [v for s, v in variants() if s == args.scene]
    if not available:
        sys.exit(f"no scene file for {args.scene} under media/scenes")
    variant = args.variant or ("static" if "static" in available else available[0])
    if variant not in available:
        sys.exit(f"{args.scene} has no {SCENE_FILES[variant]}")
    if not bake(args, args.scene, variant, args.out):
        sys.exit(1)


def bake(args, scene, variant, out):
    scene_file = os.path.join(REPO, "media", "scenes", scene, SCENE_FILES[variant])
    out = os.path.abspath(out or os.path.join(REPO, "ohos", "_build", "scenes", scene, variant))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    mogwai = os.environ.get("MOGWAI", os.path.join(REPO, "build", "windows-vs2022", "bin", "Release", "Mogwai.exe"))
    props = args.props.strip()
    env = dict(os.environ)
    env.update({
        "NCR_BAKE_DIR": out,
        "NCR_BAKE_FRAME": str(args.frame),
        "NCR_SMOKE_SCENE": scene_file,
        "NCR_SMOKE_FRAMES": str(args.frame + 1),
        "NCR_SMOKE_RES": args.res,
        "NCR_SMOKE_MODE": "onlineTrain",
        "NCR_SMOKE_OUT": os.path.join(out, "frames"),
        "NCR_SMOKE_PROPS": props if props else "{}",
    })
    if variant == "anim":
        env["NCR_BAKE_TRACK_RATE"] = repr(args.rate)
    log = out + ".log"
    with open(log, "w") as f:
        result = subprocess.run([mogwai, "--headless", "--device-type", "vulkan", "--script", os.path.join(REPO, "scripts", "ncr_smoke.py")],
                                cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not os.path.exists(os.path.join(out, "bake.txt")):
        print(f"{scene} {variant}: Falcor failed (exit {result.returncode}); see {log}")
        return False
    if subprocess.run([sys.executable, os.path.join(HERE, "bake_pack.py"), out]).returncode != 0:
        print(f"{scene} {variant}: packing failed")
        return False
    print(f"{scene} {variant}: frame {args.frame} baked to {out} (Falcor's log: {log})")
    return True


if __name__ == "__main__":
    main()

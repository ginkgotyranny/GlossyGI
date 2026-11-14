#!/usr/bin/env python
"""Bakes a scene for the phone: Falcor records frame 0 of the NeuralRadiosity pass in Online Train mode,
and the bake is packed for the runtime.

    python ohos/tools/bake_scene.py <scene> [--frame N] [--res WxH] [--props "{...}"] [--out <dir>]

<scene> is a folder under media/scenes with a scene-v4.pyscene. The bake goes to
ohos/_build/scenes/<scene> unless --out says otherwise; ohos/app/tools/sync_assets.py takes the scenes
from there. Falcor runs headless on the Vulkan backend (the bake needs SPIR-V) with the pass properties
of the project's defaults plus --props, and also captures the frame (<out>/frames) for
ohos/tools/display_check.py. Mogwai is build/windows-vs2022/bin/Release/Mogwai.exe unless MOGWAI is set.

Frame 0 is what the runtime should start from: Falcor's model is fresh then, its optimizer moments
empty, as the runtime's are.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--res", default="640x360")
    parser.add_argument("--props", default="", help="extra pass properties, a Python dict literal")
    parser.add_argument("--out")
    args = parser.parse_args()

    scene_file = os.path.join(REPO, "media", "scenes", args.scene, "scene-v4.pyscene")
    if not os.path.exists(scene_file):
        sys.exit(f"no scene {scene_file}")
    out = os.path.abspath(args.out or os.path.join(REPO, "ohos", "_build", "scenes", args.scene))
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
    log = out + ".log"
    with open(log, "w") as f:
        result = subprocess.run([mogwai, "--headless", "--device-type", "vulkan", "--script", os.path.join(REPO, "scripts", "ncr_smoke.py")],
                                cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not os.path.exists(os.path.join(out, "bake.txt")):
        sys.exit(f"Falcor failed (exit {result.returncode}); see {log}")
    subprocess.run([sys.executable, os.path.join(HERE, "bake_pack.py"), out], check=True)
    print(f"{args.scene}: frame {args.frame} baked to {out} (Falcor's log: {log})")


if __name__ == "__main__":
    main()

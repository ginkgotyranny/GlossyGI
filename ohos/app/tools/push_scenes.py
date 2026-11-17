#!/usr/bin/env python
"""Puts baked scenes on the phone, where the app finds them: <files>/scenes/<scene>/<variant>.

    python ohos/app/tools/push_scenes.py [<scene>[/<variant>] ...] [--scenes <dir>] [--force] [--list]

Without names: every packed bake under ohos/_build/scenes (ohos/tools/bake_scene.py --all). A scene
whose files on the phone are those of the bake is skipped (--force sends it anyway). --list shows
what is on the phone. The app must have been installed once, so that its files folder exists; a
running app sees new scenes after "Rescan Scenes" in its panel.

The scenes are not in the HAP: a bake is 60 to 400 MB and there are fourteen of them. Only the files
the runtime reads are sent (ohos/tools/bake_files.py), as one tar per scene over hdc.
HDC overrides the path of hdc. Standard library only.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.realpath(__file__))
APP = os.path.dirname(HERE)
OHOS = os.path.dirname(APP)
sys.path.insert(0, os.path.join(OHOS, "tools"))
from bake_files import runtime_files  # noqa: E402

HDC = os.environ.get("HDC", r"C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe")


def hdc(*args, check=True):
    result = subprocess.run([HDC, *args], capture_output=True, text=True)
    text = (result.stdout + result.stderr).replace("\r", "")
    if check and (result.returncode != 0 or "[Fail]" in text):
        sys.exit(f"hdc {' '.join(args)} failed:\n{text}")
    return text


def signature(bake, files):
    """Changes when any file the runtime reads changes."""
    h = hashlib.sha1()
    for f in files:
        info = os.stat(os.path.join(bake, f))
        h.update(f"{f}\t{info.st_size}\t{info.st_mtime_ns}\n".encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="<scene> or <scene>/<variant>")
    parser.add_argument("--scenes", default=os.path.join(OHOS, "_build", "scenes"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    bundle = re.search(r'"bundleName"\s*:\s*"([^"]+)"', open(os.path.join(APP, "AppScope", "app.json5")).read()).group(1)
    files_dir = f"/data/app/el2/100/base/{bundle}/haps/entry/files"
    target = files_dir + "/scenes"
    if "No such file" in hdc("shell", f"ls -d {files_dir}", check=False):
        sys.exit(f"the app {bundle} has no files folder on the phone: install it first (bash ohos/app/tools/deploy.sh)")
    if args.list:
        print(hdc("shell", f"cd {target} 2>/dev/null && du -sh */* 2>/dev/null || echo 'no scenes on the phone'").strip())
        return

    # The packed bakes: <scenes>/<scene>/<variant>/replay.txt.
    bakes = []
    for scene in sorted(os.listdir(args.scenes)) if os.path.isdir(args.scenes) else []:
        for variant in ("static", "anim"):
            if os.path.exists(os.path.join(args.scenes, scene, variant, "replay.txt")):
                bakes.append((scene, variant))
    if args.names:
        wanted = [(n.split("/")[0], n.split("/")[1] if "/" in n else None) for n in args.names]
        missing = [n for n, (s, v) in zip(args.names, wanted) if not any(b[0] == s and v in (None, b[1]) for b in bakes)]
        if missing:
            sys.exit(f"no packed bake for {', '.join(missing)} in {args.scenes} (python ohos/tools/bake_scene.py <scene> [static|anim])")
        bakes = [b for b in bakes if any(b[0] == s and v in (None, b[1]) for s, v in wanted)]
    if not bakes:
        sys.exit(f"no packed bake in {args.scenes}: python ohos/tools/bake_scene.py --all")

    owner = hdc("shell", f"stat -c %u:%g {files_dir}").strip()
    work = os.path.join(OHOS, "_build", "push")
    os.makedirs(work, exist_ok=True)
    sent, total = 0, 0.0
    for scene, variant in bakes:
        bake = os.path.join(args.scenes, scene, variant)
        files = runtime_files(bake)
        mark = signature(bake, files)
        remote = f"{target}/{scene}/{variant}"
        if not args.force and hdc("shell", f"cat {remote}/.pushed 2>/dev/null", check=False).strip() == mark:
            print(f"{scene}/{variant}: up to date")
            continue
        tar_path = os.path.join(work, f"{scene}-{variant}.tar")
        size = 0
        with tarfile.open(tar_path, "w") as tar:
            for f in files:
                tar.add(os.path.join(bake, f), arcname=f"{scene}/{variant}/{f}")
                size += os.path.getsize(os.path.join(bake, f))
        hdc("file", "send", tar_path, "/data/local/tmp/ncr_scene.tar")
        os.remove(tar_path)
        # The old files go first: a bake that has fewer files must not leave any behind.
        hdc("shell", f"mkdir -p {target} && rm -rf {remote} && cd {target} && tar -xf /data/local/tmp/ncr_scene.tar && rm /data/local/tmp/ncr_scene.tar"
                     f" && echo {mark} > {remote}/.pushed && chown -R {owner} {target} && chmod -R u+rwX,g+rwX {target}")
        if hdc("shell", f"cat {remote}/.pushed", check=False).strip() != mark:
            sys.exit(f"{scene}/{variant}: the files did not arrive on the phone")
        print(f"{scene}/{variant}: {len(files)} files, {size / 1e6:.0f} MB sent")
        sent += 1
        total += size
    print(f"{sent} of {len(bakes)} scenes sent ({total / 1e6:.0f} MB); on the phone: {target}")


if __name__ == "__main__":
    main()

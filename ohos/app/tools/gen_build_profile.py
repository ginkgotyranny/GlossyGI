#!/usr/bin/env python
"""Writes the two files of the app project that depend on the developer's debug signing:
build-profile.json5 (from build-profile.template.json5 and the signing configuration) and
AppScope/app.json5 (from app.template.json5 and the bundle name the signing profile is made for).

    python ohos/app/tools/gen_build_profile.py [--from <build-profile.json5 of another DevEco project>]

The signing configuration lives in signing.local.json (not in git): {"signingConfigs": [...]}, the
array DevEco Studio writes into a project's build-profile.json5. It gets there in one of two ways:
  - DevEco Studio: open ohos/app, File > Project Structure > Signing Configs, "Automatically generate
    signature". DevEco writes the configuration into build-profile.json5; the next run of this script
    keeps it in signing.local.json.
  - --from: borrow another project's configuration. The bundle name then becomes that project's, and
    the two apps replace each other on the phone.
Without a signing configuration the HAP is built unsigned (it does not install) and the bundle name is
com.example.ncrgi. Standard library only.
"""
import argparse
import json
import os
import re
import sys

APP = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_BUNDLE = "com.example.ncrgi"


def load_json5(path):
    """DevEco's JSON5 as it writes it: JSON with // comments and trailing commas."""
    text = open(path, encoding="utf-8").read()
    text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*', lambda m: m.group(1) or "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def profile_bundle_name(configs):
    """The bundle name the first signing profile (.p7b) is made for."""
    for config in configs:
        profile = config.get("material", {}).get("profile")
        if profile and os.path.exists(profile):
            match = re.search(rb'"bundle-name"\s*:\s*"([^"]+)"', open(profile, "rb").read())
            if match:
                return match.group(1).decode()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", help="build-profile.json5 of a project whose signing to borrow")
    args = parser.parse_args()

    signing_path = os.path.join(APP, "signing.local.json")
    profile_path = os.path.join(APP, "build-profile.json5")
    if args.source:
        configs = load_json5(args.source)["app"].get("signingConfigs", [])
        if not configs:
            sys.exit(f"{args.source} has no signingConfigs")
        json.dump({"signingConfigs": configs}, open(signing_path, "w", encoding="utf-8"), indent=2)
        print(f"signing configuration borrowed from {args.source}")
    elif not os.path.exists(signing_path) and os.path.exists(profile_path):
        # DevEco Studio has generated a signature into the project: keep it.
        configs = load_json5(profile_path)["app"].get("signingConfigs", [])
        if configs:
            json.dump({"signingConfigs": configs}, open(signing_path, "w", encoding="utf-8"), indent=2)
            print("signing configuration taken from build-profile.json5 (written by DevEco Studio) into signing.local.json")

    configs = json.load(open(signing_path, encoding="utf-8"))["signingConfigs"] if os.path.exists(signing_path) else []
    profile = load_json5(os.path.join(APP, "build-profile.template.json5"))
    profile["app"]["signingConfigs"] = configs
    with open(profile_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")

    bundle = profile_bundle_name(configs) or DEFAULT_BUNDLE
    template = open(os.path.join(APP, "AppScope", "app.template.json5"), encoding="utf-8").read()
    with open(os.path.join(APP, "AppScope", "app.json5"), "w", encoding="utf-8", newline="\n") as f:
        f.write(template.replace("@BUNDLE_NAME@", bundle))
    if configs:
        print(f"build-profile.json5 with signing; bundle name {bundle}")
    else:
        print(f"WARNING: no signing configuration (see this script's help); the HAP will be unsigned, bundle name {bundle}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Drives the app's screen on the phone, for checks without a hand on it: what the panel shows, taps,
swipes, screenshots. Uses the phone's uitest (dumpLayout, uiInput) over hdc.

    python ohos/app/tools/ui.py texts                    every text on the screen, with its bounds
    python ohos/app/tools/ui.py tap <text> [n]           tap the n-th element (default 0) whose text is <text>
    python ohos/app/tools/ui.py toggle <label> [on|off]  the switch in the row of <label>: print its state,
                                                         and tap it unless it already is on / off
    python ohos/app/tools/ui.py tapxy <x> <y>
    python ohos/app/tools/ui.py swipe <x1> <y1> <x2> <y2> [velocity]
    python ohos/app/tools/ui.py shot [name]              screenshot to ohos/_build/<name>.jpeg (default ui)

Coordinates are screen pixels (2848x1320 on the Mate 90 Pro Max in landscape). The panel scrolls:
swipe inside it to reach a control that is not on the screen (tap fails with the texts it sees).
HDC overrides the path of hdc. Standard library only.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "_build")
HDC = os.environ.get("HDC", r"C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe")
BUNDLE = "com.example.ncrgi"


def hdc(*args):
    result = subprocess.run([HDC, *args], capture_output=True, text=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    return (result.stdout + result.stderr).replace("\r", "")


def nodes():
    """Every element on the screen: (type, text, (x1, y1, x2, y2), attributes)."""
    hdc("shell", f"uitest dumpLayout -p /data/local/tmp/ncr_layout.json -b {BUNDLE}")
    os.makedirs(OUT, exist_ok=True)
    local = os.path.join(OUT, "ui_layout.json")
    hdc("file", "recv", "/data/local/tmp/ncr_layout.json", local)
    found = []

    def walk(node):
        a = node.get("attributes", {})
        m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", a.get("bounds", ""))
        if m:
            found.append((a.get("type", ""), a.get("text", ""), tuple(int(v) for v in m.groups()), a))
        for child in node.get("children", []):
            walk(child)

    walk(json.load(open(local, encoding="utf-8")))
    return found


def center(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def click(x, y):
    hdc("shell", f"uitest uiInput click {x} {y}")


def visible_texts(all_nodes):
    return [f"{t!r} {box}" for kind, t, box, a in all_nodes if t]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command, args = sys.argv[1], sys.argv[2:]
    if command == "texts":
        for kind, text, box, a in nodes():
            if text:
                print(f"{kind:10s} {box}  {text}")
    elif command == "tap":
        index = int(args[1]) if len(args) > 1 else 0
        matches = [n for n in nodes() if n[1] == args[0]]
        if len(matches) <= index:
            sys.exit(f"no element with the text {args[0]!r} (#{index}) on the screen")
        x, y = center(matches[index][2])
        click(x, y)
        print(f"tapped {args[0]!r} at {x},{y}")
    elif command == "toggle":
        all_nodes = nodes()
        labels = [n for n in all_nodes if n[1] == args[0]]
        if not labels:
            sys.exit(f"no label {args[0]!r} on the screen")
        label = labels[0][2]
        row = (label[1] + label[3]) // 2
        switches = [n for n in all_nodes if n[0] == "Toggle" and n[2][1] <= row <= n[2][3] and n[2][0] > label[0]]
        if not switches:
            sys.exit(f"no switch in the row of {args[0]!r}")
        box, attributes = switches[0][2], switches[0][3]
        on = attributes.get("checked") == "true"
        print(f"{args[0]}: {'on' if on else 'off'} ({'enabled' if attributes.get('enabled') == 'true' else 'disabled'})")
        want = args[1] if len(args) > 1 else None
        if want is None or (want == "on") != on:
            x, y = center(box)
            click(x, y)
            print(f"tapped the switch at {x},{y}")
    elif command == "tapxy":
        click(int(args[0]), int(args[1]))
    elif command == "swipe":
        velocity = args[4] if len(args) > 4 else "600"
        hdc("shell", f"uitest uiInput swipe {args[0]} {args[1]} {args[2]} {args[3]} {velocity}")
    elif command == "shot":
        name = (args[0] if args else "ui") + ".jpeg"
        hdc("shell", "snapshot_display -f /data/local/tmp/ncr_ui.jpeg")
        hdc("file", "recv", "/data/local/tmp/ncr_ui.jpeg", os.path.join(OUT, name))
        print(os.path.join(OUT, name))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

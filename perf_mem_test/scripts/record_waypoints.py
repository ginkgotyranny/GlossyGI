"""
Camera waypoint recorder for Falcor benchmark — key-triggered version.

Usage in Mogwai Python console (loaded via --script):
    m.script('perf_mem_test/scripts/record_waypoints.py')

Behavior:
    - After loading, move camera to a viewpoint and press R to record it.
    - Each press appends one waypoint (position + target) to
      BENCH_WAYPOINTS_FILE (default <repo>/perf_mem_test/scene_waypoints.txt)
      and prints it to the console.
    - Press Q to stop and restore the previous key callback.

No stillness detection, no auto-sampling — fully manual, one key per point.
"""

import os

_REPO_ROOT = os.environ.get("BENCH_REPO_ROOT", os.getcwd())
OUT_FILE = os.environ.get("BENCH_WAYPOINTS_FILE",
                          os.path.join(_REPO_ROOT, "perf_mem_test", "scene_waypoints.txt"))

# 'R' and 'Q' are ASCII codes in Falcor's Input::Key enum.
KEY_R = ord('R')
KEY_Q = ord('Q')

_state = {
    "running": False,
    "recorded_count": 0,
    "orig_callback": None,
}

def _waypoint_key_callback(pressed, key):
    """Key callback: R = record, Q = stop."""
    if not pressed:
        return False

    if key == KEY_R:
        cam = m.scene.camera
        pos = cam.position
        tgt = cam.target
        line = f"({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f},  {tgt.x:.4f}, {tgt.y:.4f}, {tgt.z:.4f},  0.5),"
        with open(OUT_FILE, "a") as f:
            f.write(line + "\n")
        _state["recorded_count"] += 1
        print(f"[waypoint #{_state['recorded_count']}] {line}")
        return True  # consumed

    if key == KEY_Q:
        stop_waypoint_recorder()
        return True

    return False

def start_waypoint_recorder():
    """Begin recording. Truncates the output file and installs the key callback."""
    if _state["running"]:
        print("Recorder already running. Press R to record, Q to stop.")
        return
    _state["recorded_count"] = 0
    with open(OUT_FILE, "w") as f:
        f.write("# Camera waypoints recorded by record_waypoints.py\n")
    _state["orig_callback"] = m.keyCallback
    m.keyCallback = _waypoint_key_callback
    _state["running"] = True
    print(f"Waypoint recorder started. Output: {os.path.abspath(OUT_FILE)}")
    print("  R = record current camera pose")
    print("  Q = stop recording")

def stop_waypoint_recorder():
    """Stop recording and restore the previous key callback."""
    if not _state["running"]:
        print("Recorder not running.")
        return
    _state["running"] = False
    m.keyCallback = _state["orig_callback"]
    n = _state["recorded_count"]
    print(f"Waypoint recorder stopped. Captured {n} waypoints to {os.path.abspath(OUT_FILE)}")

# Auto-start on load
start_waypoint_recorder()

"""
Benchmark runner for Falcor — measures FPS and memory along a fixed camera trajectory.

Usage (inside Mogwai Python console, after the render graph is loaded):
    m.script('scripts/benchmark.py')               # default 30s measurement
    m.script('scripts/benchmark.py', '60')          # 60s measurement

Behavior:
    1. Loads the camera trajectory (Catmull-Rom through recorded waypoints).
    2. Warms up for N frames (not measured).
    3. Installs a per-frame callback that drives the camera along the trajectory
       and records frame time + RSS. The callback runs inside Mogwai's normal
       render loop — it does NOT block the Python console, so you can still
       type commands while it runs.
    4. After MEASURE_SECONDS seconds, the callback automatically writes CSV,
       prints summary, and restores the original scene callback.

No key presses needed. The measurement auto-stops after the configured duration.
"""

import os
import time
import datetime

# ---- configuration ---------------------------------------------------------

WARMUP_FRAMES = 30          # frames to render before measurement
MEASURE_SECONDS = 30        # default measurement duration
TRAJECTORY_LOOPS = 1000     # large = effectively infinite looping
SEGMENT_TIME = 4.0          # seconds per trajectory segment (higher = slower camera)
OUTPUT_DIR = os.environ.get("BENCH_RESULTS_DIR", os.path.join(os.environ.get("BENCH_REPO_ROOT", os.getcwd()), "perf_mem_test", "results"))

# ---- helpers ---------------------------------------------------------------

def _read_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0

# ---- benchmark state (module-level so callback can reach it) ---------------

_state = {
    "phase": "idle",          # idle -> warming -> measuring -> done
    "frame_times": [],
    "rss_values": [],
    "traj_times": [],
    "t_start": 0.0,
    "measure_seconds": MEASURE_SECONDS,
    "out_path": "",
    "traj_callback": None,
    "orig_callback": None,
    "orig_time": 0.0,
    "orig_cam_animated": True,
    "clock": None,
    "cam": None,
    "_prev_t": 0.0,
}

def _benchmark_callback(scene, current_time):
    """Per-frame callback: drive camera + record data + auto-stop."""
    s = _state

    if s["phase"] == "measuring":
        now = time.time()
        elapsed = now - s["t_start"]

        # record previous frame's wall time (callback-to-callback interval)
        wall_ms = (now - s["_prev_t"]) * 1000.0
        s["_prev_t"] = now

        s["frame_times"].append(wall_ms)
        s["rss_values"].append(_read_rss_mb())
        s["traj_times"].append(current_time)

        # check if time is up
        if elapsed >= s["measure_seconds"]:
            _finish_benchmark()
            return

        # periodic progress
        n = len(s["frame_times"])
        if n % 300 == 0:
            recent = s["frame_times"][-300:]
            avg_ft = sum(recent) / len(recent)
            print(f"  [benchmark] frame {n}  avg_fps={1000/avg_ft:.1f}  "
                  f"elapsed={elapsed:.1f}/{s['measure_seconds']}s")

    # drive camera along trajectory
    if s["traj_callback"]:
        s["traj_callback"](scene, current_time)

def _finish_benchmark():
    """Write CSV, print summary, restore original state."""
    s = _state
    s["phase"] = "done"

    ft = s["frame_times"]
    rss = s["rss_values"]
    total_wall = time.time() - s["t_start"]

    # write CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(s["out_path"], "w") as f:
        f.write("frame,wall_time_ms,fps,rss_mb,traj_time_s\n")
        for i in range(len(ft)):
            fps = 1000.0 / ft[i] if ft[i] > 0 else 0
            f.write(f"{i},{ft[i]:.3f},{fps:.2f},{rss[i]:.2f},{s['traj_times'][i]:.3f}\n")

    # summary
    valid_ft = [x for x in ft if x > 0]
    avg_ft = sum(valid_ft) / len(valid_ft) if valid_ft else 0
    avg_fps = 1000.0 / avg_ft if avg_ft > 0 else 0
    sorted_ft = sorted(valid_ft)
    n = len(sorted_ft)
    p50 = sorted_ft[n // 2] if n else 0
    p95 = sorted_ft[int(n * 0.95)] if n else 0
    p99 = sorted_ft[int(n * 0.99)] if n else 0

    valid_rss = [x for x in rss if x > 0]
    base_rss = valid_rss[0] if valid_rss else 0
    avg_rss = sum(valid_rss) / len(valid_rss) if valid_rss else 0
    peak_rss = max(valid_rss) if valid_rss else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"CSV            : {os.path.abspath(s['out_path'])}")
    print(f"Frames measured: {len(ft)}")
    print(f"Total wall time: {total_wall:.2f}s")
    print()
    print("--- Frame Rate ---")
    print(f"  Avg FPS       : {avg_fps:.2f}")
    print(f"  Avg frame time: {avg_ft:.2f} ms")
    print(f"  p50           : {p50:.2f} ms  ({1000/p50:.1f} FPS)" if p50 > 0 else "")
    print(f"  p95           : {p95:.2f} ms  ({1000/p95:.1f} FPS)" if p95 > 0 else "")
    print(f"  p99           : {p99:.2f} ms  ({1000/p99:.1f} FPS)" if p99 > 0 else "")
    print()
    print("--- Memory (process RSS) ---")
    print(f"  Baseline RSS  : {base_rss:.1f} MB")
    print(f"  Average RSS   : {avg_rss:.1f} MB")
    print(f"  Peak RSS      : {peak_rss:.1f} MB")
    print(f"  Delta         : {peak_rss - base_rss:.1f} MB")
    print("=" * 60)

    # restore original state
    clock = s["clock"]
    cam = s["cam"]
    clock.time = s["orig_time"]
    cam.animated = s["orig_cam_animated"]
    m.sceneUpdateCallback = s["orig_callback"]
    clock.play()
    print("[benchmark] done, scene restored.")

# ---- main entry ------------------------------------------------------------

def run_benchmark(measure_seconds=None):
    import sys
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.path.join(os.getcwd(), "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)
    from camera_trajectory import build_callback

    if measure_seconds is not None:
        _state["measure_seconds"] = float(measure_seconds)
    else:
        _state["measure_seconds"] = MEASURE_SECONDS

    # build trajectory
    callback, traj_duration = build_callback(loops=TRAJECTORY_LOOPS, segment_time=SEGMENT_TIME)

    cam = m.scene.camera
    clock = m.clock

    # save original state
    _state["orig_callback"] = m.sceneUpdateCallback
    _state["orig_time"] = clock.time
    _state["orig_cam_animated"] = cam.animated
    _state["traj_callback"] = callback
    _state["clock"] = clock
    _state["cam"] = cam
    _state["frame_times"] = []
    _state["rss_values"] = []
    _state["traj_times"] = []

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _state["out_path"] = os.path.join(OUTPUT_DIR, f"benchmark_{ts}.csv")

    # disable camera animation + pause clock (prevents scene->update from
    # overriding our camera positions)
    clock.pause()
    cam.animated = False

    # warmup: render a few frames at trajectory start to stabilize
    _state["phase"] = "warming"
    one_loop = traj_duration / TRAJECTORY_LOOPS
    print(f"[benchmark] warming up {WARMUP_FRAMES} frames...")
    for i in range(WARMUP_FRAMES):
        clock.time = (i / WARMUP_FRAMES) * one_loop
        m.renderFrame()
    print(f"[benchmark] measuring for {_state['measure_seconds']:.0f}s "
          f"(segment_time={SEGMENT_TIME}s, {len([w for w in []])} waypoints)...")

    # start measurement: install callback that drives camera + records data
    _state["phase"] = "measuring"
    _state["t_start"] = time.time()
    _state["_prev_t"] = _state["t_start"]
    m.sceneUpdateCallback = _benchmark_callback
    clock.play()  # let clock advance so trajectory time progresses naturally

    print("[benchmark] running... (auto-stops, console is still usable)")

# auto-run on load — accept optional duration argument
# m.script('scripts/benchmark.py')        -> 30s
# m.script('scripts/benchmark.py', '60')  -> 60s
import sys as _sys
_dur = _sys.argv[1] if len(_sys.argv) > 1 else None
run_benchmark(_dur)

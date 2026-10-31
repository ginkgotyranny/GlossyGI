"""
Auto-benchmark script for Falcor performance & memory testing.

Loaded via --script. Automatically:
  1. Loads the render graph (ptdn / ptnrd / ncr selected via BENCH_ALGO env var)
  2. Drives camera along a fixed trajectory (Catmull-Rom through waypoints)
  3. Records FPS at fixed time intervals + memory via nvitop, for BENCH_DURATION s
  4. Writes two CSVs (FPS + memory) to BENCH_RESULTS_DIR (default perf_mem_test/results)
  5. Calls os._exit(0) to terminate Mogwai

Environment variables:
  BENCH_ALGO          : ptdn | ptnrd | ncr  (default: ptdn)
  BENCH_SCENE         : scene name           (default: bathroom)
  BENCH_DURATION      : seconds to measure   (default: 60)
  BENCH_WARMUP_S      : static-camera warmup seconds before measuring (default: 30)
  BENCH_FPS_INTERVAL  : FPS sample interval in seconds (default: 0.030 = 30ms)
  BENCH_MEM_INTERVAL_S: memory sample interval in seconds (default: 2.0)
  BENCH_RTXDI         : 0 = rtxdi_off tag, 1 = rtxdi_on tag (filename only)
  BENCH_STATIC        : 1 = don't move camera (control experiment)
  BENCH_WAYPOINTS_FILE: path to waypoints file (default: <repo>/perf_mem_test/scene_waypoints.txt)
  BENCH_RESULTS_DIR   : output directory for CSV/meta (default: <repo>/perf_mem_test/results)
  BENCH_GRAPH_DIR     : directory containing render-graph scripts (default: <repo>/scripts)
  BENCH_DEBUG_LOG     : debug log path (default: <BENCH_RESULTS_DIR>/debug.log)
"""

import os
import time
import datetime

# Repo root = Mogwai's cwd when launched via --script (set by the shell runners).
_REPO_ROOT = os.environ.get("BENCH_REPO_ROOT", os.getcwd())

# Debug log to file (Mogwai may swallow stdout)
_RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR",
                              os.path.join(_REPO_ROOT, "perf_mem_test", "results"))
_DEBUG_LOG = os.environ.get("BENCH_DEBUG_LOG",
                            os.path.join(_RESULTS_DIR, "debug.log"))
def _dlog(msg):
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG), exist_ok=True)
        with open(_DEBUG_LOG, "a") as f:
            f.write(f"[{time.time():.3f}] {msg}\n")
    except:
        pass

# ---- config from env -------------------------------------------------------

ALGO = os.environ.get("BENCH_ALGO", "ptdn")
DURATION = float(os.environ.get("BENCH_DURATION", "60"))
OUTPUT_DIR = _RESULTS_DIR

_GRAPH_DIR = os.environ.get("BENCH_GRAPH_DIR",
                            os.path.join(_REPO_ROOT, "scripts"))
GRAPH_SCRIPTS = {
    "ptdn":  os.path.join(_GRAPH_DIR, "OptixDenoiser.py"),
    "ptnrd": os.path.join(_GRAPH_DIR, "PathTracerNRD_VK.py"),
    "ncr":   os.path.join(_GRAPH_DIR, "NeuralRadiosity.py"),
}

# ---- waypoints (parsed from scene_waypoints.txt by BENCH_SCENE) ------------

SCENE_NAME = os.environ.get("BENCH_SCENE", "bathroom")
WAYPOINTS_FILE = os.environ.get("BENCH_WAYPOINTS_FILE",
                                os.path.join(_REPO_ROOT, "perf_mem_test", "scene_waypoints.txt"))

def _parse_waypoints(path, scene):
    """Parse scene_waypoints.txt, return list of waypoints for `scene`."""
    waypoints = []
    current = None
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("#"):
                    low = s.lower()
                    if "recorded in" in low and "scene" in low:
                        after = low.split("recorded in", 1)[1]
                        name = after.split("scene", 1)[0].strip().strip(",，")
                        current = name
                    continue
                if not s:
                    current = None
                    continue
                if current != scene:
                    continue
                body = s.strip().rstrip(",").strip()
                if body.startswith("(") and body.endswith(")"):
                    body = body[1:-1]
                nums = [float(p.strip()) for p in body.split(",") if p.strip() != ""]
                if len(nums) >= 7:
                    waypoints.append(tuple(nums[:7]))
    except Exception as e:
        _dlog(f"waypoint parse error: {e}")
    return waypoints

WAYPOINTS = _parse_waypoints(WAYPOINTS_FILE, SCENE_NAME)
if not WAYPOINTS:
    _dlog(f"WARNING: no waypoints found for scene='{SCENE_NAME}' in {WAYPOINTS_FILE}")
    print(f"[bench] WARNING: no waypoints for scene '{SCENE_NAME}', using single static point")
    WAYPOINTS = [(0, 0, 0, 0, 0, -1, 0.3)]

_dlog(f"scene={SCENE_NAME}  waypoints={len(WAYPOINTS)} from {WAYPOINTS_FILE}")
print(f"[bench] scene={SCENE_NAME}  waypoints={len(WAYPOINTS)}")

SEGMENT_TIME = 2.0  # seconds per segment of travel between waypoints

# ---- Catmull-Rom interpolation (inlined, no imports needed) -----------------

def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3)

def _interp(points, i, t):
    n = len(points)
    p0 = points[(i-1) % n]
    p1 = points[i]
    p2 = points[(i+1) % n]
    p3 = points[(i+2) % n]
    return (_catmull_rom(p0[0],p1[0],p2[0],p3[0],t),
            _catmull_rom(p0[1],p1[1],p2[1],p3[1],t),
            _catmull_rom(p0[2],p1[2],p2[2],p3[2],t))

def _trajectory_at(t, seg_durations, one_loop):
    if t >= one_loop:
        return (WAYPOINTS[0][0:3], WAYPOINTS[0][3:6])
    positions = [(w[0],w[1],w[2]) for w in WAYPOINTS]
    targets   = [(w[3],w[4],w[5]) for w in WAYPOINTS]
    elapsed = 0.0
    n = len(WAYPOINTS)
    for i in range(n):
        seg_dur = seg_durations[i]
        if t < elapsed + seg_dur:
            local_t = t - elapsed
            travel = seg_dur - WAYPOINTS[i][6]
            if local_t < travel:
                u = local_t / travel if travel > 0 else 0
                return (_interp(positions, i, u), _interp(targets, i, u))
            return (positions[(i+1)%n], targets[(i+1)%n])
        elapsed += seg_dur
    return (positions[0], targets[0])

# ---- build trajectory params -----------------------------------------------

_seg_durations = [SEGMENT_TIME + w[6] for w in WAYPOINTS]
_one_loop = sum(_seg_durations)

# ---- benchmark state -------------------------------------------------------
#
# Sampling design (revised):
#   FPS  — time-based, every FPS_SAMPLE_INTERVAL_S (30ms) wall-clock. The
#          callback fires once per rendered frame (needed to drive the camera),
#          but a sample is RECORDED only when 30ms have elapsed since the last
#          sample. FPS = frames_in_window / window_dt. This decouples sample
#          density from the algorithm's frame rate (so ptdn @50fps and ncr
#          @120fps produce the same ~33 samples/sec) and minimizes per-frame
#          recording overhead inside the render callback.
#   MEM  — background thread, every MEM_SAMPLE_INTERVAL_S (2s) wall-clock.
#          Both CPU-MEM and GPU-MEM are read from a single `nvitop -1 --ascii
#          --pid <PID>` call: GPU-MEM from the process-row MiB column, CPU-MEM
#          from %MEM × MemTotal (equals process RSS). Same frequency for both.
#          Background thread avoids any subprocess fork inside the render loop.

FPS_SAMPLE_INTERVAL_S = float(os.environ.get("BENCH_FPS_INTERVAL", "0.030"))  # 30ms
MEM_SAMPLE_INTERVAL_S = float(os.environ.get("BENCH_MEM_INTERVAL_S", "2.0"))  # 2s

# FPS samples: list of (elapsed_s, fps)
_fps_samples = []
# Memory samples: list of (elapsed_s, cpu_mem_mb, gpu_mem_mb)
_mem_samples = []
# total frame counter (for harmonic-mean FPS over the whole run)
_total_frames = 0

_t_start = 0.0
_warmup_count = 0
WARMUP_FRAMES = 30  # legacy, kept for meta compatibility
WARMUP_SECONDS = float(os.environ.get("BENCH_WARMUP_S", "30"))
_warmup_t0 = 0.0
_clock = None
_cam = None

# FPS window state
_last_fps_sample_t = 0.0
_frames_in_window = 0

# Memory background thread
_mem_thread = None
_mem_thread_stop = False
_last_cpu_mem = -1.0
_last_gpu_mem = -1.0
_mem_t0 = 0.0  # wall-clock time when measurement started (for elapsed in mem samples)

# Round tag: "rtxdi_on" or "rtxdi_off"
RTXDI_TAG = "rtxdi_on" if os.environ.get("BENCH_RTXDI", "0") == "1" else "rtxdi_off"

# Scene/trajectory metadata written to a sidecar JSON for the report.
BENCH_META = {
    "algo": os.environ.get("BENCH_ALGO", "ptdn"),
    "scene": SCENE_NAME,
    "rtxdi": RTXDI_TAG,
    "duration_s": float(os.environ.get("BENCH_DURATION", "60")),
    "warmup_frames": WARMUP_FRAMES,
    "warmup_seconds": WARMUP_SECONDS,
    "segment_time_s": 2.0,
    "resolution": "1280x720",
    "fps_sample_interval_s": FPS_SAMPLE_INTERVAL_S,
    "mem_sample_interval_s": MEM_SAMPLE_INTERVAL_S,
    "waypoints": [],
    "waypoint_count": len(WAYPOINTS),
}

# ---- MemTotal (for %MEM → absolute CPU-MEM conversion) ----------------------
_MEM_TOTAL_MB = 0.0
try:
    with open("/proc/meminfo") as _f:
        for _line in _f:
            if _line.startswith("MemTotal:"):
                _MEM_TOTAL_MB = int(_line.split()[1]) / 1024.0
                break
except:
    pass
_dlog(f"MemTotal = {_MEM_TOTAL_MB:.0f} MB")

def _mem_sampler_thread():
    """Background thread: poll nvitop every MEM_SAMPLE_INTERVAL_S for both
    CPU-MEM (%MEM × MemTotal = RSS) and GPU-MEM (process-row MiB).

    Uses `nvitop -1 --ascii --pid <PID>`. The process row format (ascii):
      | 0  <PID>  X  <user>  <GPU-MEM>MiB  %SM  %CPU  %MEM  <time>  <cmd> |
    GPU-MEM = token containing 'MiB'; %MEM = token at index 7 after splitting.
    Falls back to nvidia-smi pmon for GPU-MEM if nvitop can't find the process,
    and to /proc/self/status VmRSS for CPU-MEM if nvitop %MEM can't be parsed.
    """
    global _last_cpu_mem, _last_gpu_mem, _mem_thread_stop
    import subprocess, re
    my_pid = os.getpid()
    mib_re = re.compile(r"(\d+(?:\.\d+)?)\s*MiB", re.IGNORECASE)
    while not _mem_thread_stop:
        cpu_mem = -1.0
        gpu_mem = -1.0
        # --- primary: nvitop (both CPU-MEM and GPU-MEM) ---
        try:
            out = subprocess.check_output(
                ["nvitop", "-1", "--ascii", "--pid", str(my_pid)],
                timeout=8, stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            for line in out.splitlines():
                if str(my_pid) not in line:
                    continue
                # GPU-MEM: the MiB token
                m = mib_re.search(line)
                if m:
                    gpu_mem = float(m.group(1))
                # %MEM: strip '|', split. Layout: GPU PID type USER GPU-MEM %SM %CPU %MEM TIME CMD
                body = line.strip().strip("|").strip()
                toks = body.split()
                if len(toks) >= 8:
                    try:
                        pct = float(toks[7])
                        cpu_mem = pct * _MEM_TOTAL_MB / 100.0
                    except (ValueError, IndexError):
                        pass
                break
        except Exception as e:
            _dlog(f"nvitop mem parse failed: {e}")
        # --- fallback GPU-MEM: nvidia-smi pmon (compute + graphics) ---
        if gpu_mem < 0:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "pmon", "-c", "1", "-s", "m"],
                    timeout=8, stderr=subprocess.DEVNULL
                ).decode(errors="ignore")
                for line in out.splitlines():
                    if line.lstrip().startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == str(my_pid):
                        try:
                            gpu_mem = float(parts[3])
                        except ValueError:
                            pass
                        break
            except:
                pass
        # --- fallback CPU-MEM: /proc/self/status RSS ---
        if cpu_mem < 0:
            try:
                with open("/proc/self/status") as f:
                    for ln in f:
                        if ln.startswith("VmRSS:"):
                            cpu_mem = int(ln.split()[1]) / 1024.0
                            break
            except:
                pass
        if cpu_mem >= 0:
            _last_cpu_mem = cpu_mem
        if gpu_mem >= 0:
            _last_gpu_mem = gpu_mem
        # record a memory sample with elapsed time (only during measurement)
        now = time.time()
        if _mem_t0 > 0 and cpu_mem >= 0 and gpu_mem >= 0:
            _mem_samples.append((now - _mem_t0, cpu_mem, gpu_mem))
        time.sleep(MEM_SAMPLE_INTERVAL_S)

def _start_mem_thread():
    global _mem_thread
    import threading
    _mem_thread = threading.Thread(target=_mem_sampler_thread, daemon=True)
    _mem_thread.start()
    _dlog("mem sampler thread started")

def _bench_callback(scene, current_time):
    global _warmup_count, _t_start, _warmup_t0, _mem_t0
    global _last_fps_sample_t, _frames_in_window, _total_frames

    now = time.time()
    if _warmup_count == 0:
        _warmup_t0 = now
        _dlog(f"first callback: current_time={current_time:.3f}, warmup_t0={_warmup_t0:.3f}")
        print(f"[bench] holding camera still for {WARMUP_SECONDS}s warmup (shader compile / first-frame)...")
    _warmup_count += 1

    # warmup: hold camera still for WARMUP_SECONDS, no recording, no trajectory.
    if (now - _warmup_t0) < WARMUP_SECONDS:
        return

    # first measuring frame: reset timers
    if _t_start == 0.0:
        _t_start = now
        _mem_t0 = now  # global — set measurement start for mem thread elapsed
        _last_fps_sample_t = now
        _frames_in_window = 0
        print(f"[bench] warmup done ({WARMUP_SECONDS:.0f}s, {_warmup_count} frames). "
              f"Measuring for {DURATION}s with camera trajectory...")
        _dlog(f"warmup done, t_start={_t_start:.3f}")

    elapsed = now - _t_start
    _total_frames += 1
    _frames_in_window += 1

    # --- time-based FPS sampling: record only when FPS_SAMPLE_INTERVAL_S elapsed ---
    dt = now - _last_fps_sample_t
    if dt >= FPS_SAMPLE_INTERVAL_S:
        fps = _frames_in_window / dt if dt > 0 else 0
        _fps_samples.append((elapsed, fps))
        _frames_in_window = 0
        _last_fps_sample_t = now

    # drive camera (every frame — needed for smooth trajectory)
    _drive_camera(current_time)

    # check if time is up
    if elapsed >= DURATION:
        _finish()
        return

    # periodic progress
    n = len(_fps_samples)
    if n > 0 and n % 500 == 0:
        recent = _fps_samples[-500:]
        avg_fps = sum(f for _, f in recent) / len(recent)
        print(f"  [bench] sample {n}  fps={avg_fps:.1f}  "
              f"elapsed={elapsed:.1f}/{DURATION}s  cpu={_last_cpu_mem:.0f}MB  gpu={_last_gpu_mem:.0f}MB")

_STATIC_CAM = os.environ.get("BENCH_STATIC", "0") == "1"

def _drive_camera(current_time):
    if _STATIC_CAM:
        return
    t = current_time % _one_loop
    pos, tgt = _trajectory_at(t, _seg_durations, _one_loop)
    _cam.position = float3(pos[0], pos[1], pos[2])
    _cam.target = float3(tgt[0], tgt[1], tgt[2])

def _finish():
    """Write CSVs, print summary, exit."""
    global _mem_thread_stop
    _mem_thread_stop = True
    _dlog(f"finish called, fps_samples={len(_fps_samples)}, total_frames={_total_frames}, mem_samples={len(_mem_samples)}")
    total_wall = time.time() - _t_start

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- FPS CSV: elapsed_s, fps ----
    fps_path = os.path.join(OUTPUT_DIR, f"bench_{SCENE_NAME}_{ALGO}_{RTXDI_TAG}.csv")
    with open(fps_path, "w") as f:
        f.write("elapsed_s,fps\n")
        for elapsed, fps in _fps_samples:
            f.write(f"{elapsed:.4f},{fps:.2f}\n")

    # ---- Memory CSV: elapsed_s, cpu_mem_mb, gpu_mem_mb ----
    mem_path = os.path.join(OUTPUT_DIR, f"bench_{SCENE_NAME}_{ALGO}_{RTXDI_TAG}_mem.csv")
    with open(mem_path, "w") as f:
        f.write("elapsed_s,cpu_mem_mb,gpu_mem_mb\n")
        for elapsed, cpu, gpu in _mem_samples:
            f.write(f"{elapsed:.4f},{cpu:.2f},{gpu:.2f}\n")

    # write metadata sidecar JSON (for report)
    import json
    meta_path = os.path.join(OUTPUT_DIR, f"bench_{SCENE_NAME}_{ALGO}_{RTXDI_TAG}.meta.json")
    BENCH_META["waypoints"] = WAYPOINTS
    BENCH_META["one_loop_s"] = _one_loop
    BENCH_META["total_frames"] = _total_frames
    BENCH_META["fps_samples"] = len(_fps_samples)
    BENCH_META["mem_samples"] = len(_mem_samples)
    BENCH_META["wall_time_s"] = total_wall

    with open(meta_path, "w") as f:
        json.dump(BENCH_META, f, indent=2)

    # ---- summary ----
    # Harmonic-mean FPS over the whole run = total_frames / total_wall
    avg_fps = _total_frames / total_wall if total_wall > 0 else 0
    fps_vals = [f for _, f in _fps_samples if f > 0]

    cpu_vals = [c for _, c, _ in _mem_samples if c > 0]
    gpu_vals = [g for _, _, g in _mem_samples if g > 0]
    avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0
    peak_cpu = max(cpu_vals) if cpu_vals else 0
    avg_gpu = sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0
    peak_gpu = max(gpu_vals) if gpu_vals else 0

    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS ({ALGO})  scene={SCENE_NAME}  rtxdi={RTXDI_TAG}")
    print("=" * 60)
    print(f"FPS CSV        : {os.path.abspath(fps_path)}")
    print(f"MEM CSV        : {os.path.abspath(mem_path)}")
    print(f"Total frames   : {_total_frames}")
    print(f"FPS samples    : {len(_fps_samples)}  (interval {FPS_SAMPLE_INTERVAL_S*1000:.0f}ms)")
    print(f"MEM samples    : {len(_mem_samples)}  (interval {MEM_SAMPLE_INTERVAL_S:.0f}s)")
    print(f"Wall time      : {total_wall:.2f}s")
    print()
    print("--- Frame Rate ---")
    print(f"  Avg FPS (total_frames/wall) : {avg_fps:.2f}")
    if fps_vals:
        print(f"  Avg FPS (sample mean)       : {sum(fps_vals)/len(fps_vals):.2f}")
        print(f"  Min / Max sample FPS        : {min(fps_vals):.1f} / {max(fps_vals):.1f}")
    print()
    print("--- Memory (nvitop) ---")
    print(f"  CPU-MEM (RSS): avg={avg_cpu:.1f}  peak={peak_cpu:.1f} MB")
    print(f"  GPU-MEM      : avg={avg_gpu:.1f}  peak={peak_gpu:.1f} MB")
    print(f"  Total avg    : {avg_cpu+avg_gpu:.1f} MB")
    print("=" * 60)
    print(f"\nFPS CSV: {fps_path}")
    print(f"MEM CSV: {mem_path}")

    # terminate Mogwai
    os._exit(0)

# ---- main: load graph + install callback -----------------------------------

# 1. Load the render graph
_dlog(f"loading graph: {GRAPH_SCRIPTS[ALGO]}")
print(f"[bench] loading graph: {GRAPH_SCRIPTS[ALGO]}")
m.script(GRAPH_SCRIPTS[ALGO])
_dlog("graph loaded")

# 2. Get camera + clock
_cam = m.scene.camera
_clock = m.clock
_dlog(f"camera={_cam}, clock={_clock}")

# 3. Disable camera animation (prevents scene->update from overriding our positions)
_cam.animated = False
_dlog("camera animation disabled")

# 4. Let clock play so time advances naturally (callback uses current_time)
if os.environ.get("BENCH_PAUSE_CLOCK", "0") != "1":
    _clock.play()
    _dlog("clock playing")
else:
    _dlog("clock PAUSED (control experiment)")

# 5. Start memory sampler thread (background, non-blocking)
_start_mem_thread()

# 6. Install benchmark callback
m.sceneUpdateCallback = _bench_callback
_dlog("callback installed")

print(f"[bench] algo={ALGO}  scene={SCENE_NAME}  duration={DURATION}s  warmup={WARMUP_SECONDS:.0f}s (static)  "
      f"fps_interval={FPS_SAMPLE_INTERVAL_S*1000:.0f}ms  mem_interval={MEM_SAMPLE_INTERVAL_S:.0f}s  "
      f"segment_time={SEGMENT_TIME}s  one_loop={_one_loop:.1f}s")
print("[bench] running...")
_dlog("benchmark started")

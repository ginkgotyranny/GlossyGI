"""
Staged memory measurement — loads graph incrementally and measures PSS
at each stage to isolate where CPU memory is allocated.

Stages:
  0: Mogwai bare (no graph)
  1: + scene loaded
  2: + graph compiled (before first frame)
  3: + first frame rendered (shader compile / PSO creation)
  4: + 100 frames (steady state)
"""
import os, time

ALGO = os.environ.get("BENCH_ALGO", "ptdn")
OUTPUT_DIR = os.environ.get("BENCH_RESULTS_DIR", os.path.join(os.environ.get("BENCH_REPO_ROOT", os.getcwd()), "perf_mem_test", "results"))

GRAPHS = {
    "ptdn":  "scripts/OptixDenoiser.py",
    "ptnrd": "scripts/PathTracerNRD_VK.py",
    "ncr":   "scripts/NeuralRadiosity.py",
}

def _read_pss():
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024.0
    except:
        pass
    return -1.0

def _read_rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except:
        pass
    return -1.0

_samples = []
_stage = [0]
_frame_count = [0]

def _record_stage(label):
    pss = _read_pss()
    rss = _read_rss()
    _samples.append((label, pss, rss))
    print(f"[mem] stage={label}  PSS={pss:.0f}MB  RSS={rss:.0f}MB")

def _cb(scene, current_time):
    fc = _frame_count[0]
    if fc == 0:
        _record_stage("3_after_first_frame")
    if fc == 100:
        _record_stage("4_steady_state_100frames")
        # write results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"mem_breakdown_{ALGO}.txt"), "w") as f:
            for label, pss, rss in _samples:
                f.write(f"{label}\t{pss:.1f}\t{rss:.1f}\n")
        print(f"[mem] breakdown written for {ALGO}")
        os._exit(0)
    _frame_count[0] += 1

# Stage 0: bare Mogwai (before graph load)
_record_stage("0_bare_mogwai")

# Load graph
m.script(GRAPHS[ALGO])

# Stage 1: graph loaded (scene already loaded by --deferred)
_record_stage("1_graph_loaded")

# Install callback to measure after first frame and steady state
m.sceneUpdateCallback = _cb
print(f"[mem] {ALGO}: staged measurement started")

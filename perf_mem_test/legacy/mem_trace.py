"""
Fine-grained memory trace — samples PSS every 10 frames for 200 frames
to see how memory grows over time.
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

_samples = []
_fc = [0]

def _cb(scene, current_time):
    fc = _fc[0]
    if fc % 10 == 0:
        pss = _read_pss()
        _samples.append((fc, pss))
        print(f"[trace] frame={fc:4d}  PSS={pss:.0f}MB")
    if fc >= 200:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"mem_trace_{ALGO}.csv"), "w") as f:
            f.write("frame,pss_mb\n")
            for frame, pss in _samples:
                f.write(f"{frame},{pss:.1f}\n")
        print(f"[trace] done for {ALGO}")
        os._exit(0)
    _fc[0] += 1

m.script(GRAPHS[ALGO])
m.sceneUpdateCallback = _cb
print(f"[trace] {ALGO}: tracing PSS every 10 frames for 200 frames")

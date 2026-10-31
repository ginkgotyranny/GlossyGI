"""
Ultra-fine trace: sample PSS every frame for first 30 frames to find
the exact frame where the 2.6GB jump happens.
"""
import os

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
    pss = _read_pss()
    _samples.append((fc, pss))
    if fc <= 30 or fc % 50 == 0:
        print(f"[fine] frame={fc:4d}  PSS={pss:.0f}MB")
    if fc >= 200:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"mem_fine_{ALGO}.csv"), "w") as f:
            f.write("frame,pss_mb\n")
            for frame, pss in _samples:
                f.write(f"{frame},{pss:.1f}\n")
        print(f"[fine] done for {ALGO}")
        os._exit(0)
    _fc[0] += 1

m.script(GRAPHS[ALGO])
m.sceneUpdateCallback = _cb
print(f"[fine] {ALGO}: per-frame PSS for first 200 frames")

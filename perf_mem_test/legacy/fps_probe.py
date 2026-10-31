"""
FPS probe with warmup — measures true interactive FPS after shader compile.
"""
import os, time

ALGO = os.environ.get("BENCH_ALGO", "ptdn")
WARMUP = float(os.environ.get("BENCH_WARMUP_S", "15"))
DURATION = float(os.environ.get("BENCH_DURATION", "10"))
OUTPUT_DIR = os.environ.get("BENCH_RESULTS_DIR", os.path.join(os.environ.get("BENCH_REPO_ROOT", os.getcwd()), "perf_mem_test", "results"))

GRAPHS = {
    "ptdn":  "scripts/OptixDenoiser.py",
    "ptnrd": "scripts/PathTracerNRD_VK.py",
    "ncr":   "scripts/NeuralRadiosity.py",
}

m.script(GRAPHS[ALGO])

_phase = [0]  # 0=warmup, 1=measure, 2=done
_t0 = [0.0]
_warmup_t0 = [0.0]
_count = [0]
_prev = [0.0]
_walls = []

def _cb(scene, current_time):
    now = time.time()
    if _phase[0] == 0:
        _warmup_t0[0] = now
        _phase[0] = 1
        print(f"[probe] {ALGO}: warmup {WARMUP}s...")
        return
    if _phase[0] == 1:
        if now - _warmup_t0[0] < WARMUP:
            return
        _phase[0] = 2
        _t0[0] = now
        _prev[0] = now
        _count[0] = 0
        print(f"[probe] {ALGO}: measuring {DURATION}s...")
        return
    if _phase[0] == 2:
        if _move_cam:
            _move()
        _count[0] += 1
        # Test: add /proc reads like perf_mem_test.py does
        if os.environ.get("BENCH_READ_PROC", "0") == "1":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"): _ = line
            with open("/proc/self/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"): _ = line
        _walls.append((now - _prev[0]) * 1000.0)
        _prev[0] = now
        if now - _t0[0] >= DURATION:
            valid = [w for w in _walls[1:] if w > 0]
            avg = sum(valid) / len(valid) if valid else 0
            fps = 1000.0 / avg if avg > 0 else 0
            print(f"[probe] {ALGO}: frames={len(valid)} avg_wall={avg:.1f}ms fps={fps:.1f}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f"probe_{ALGO}.txt"), "w") as f:
                f.write(f"algo={ALGO} frames={len(valid)} avg_wall_ms={avg:.2f} fps={fps:.1f}\n")
            os._exit(0)

m.sceneUpdateCallback = _cb
_clock = m.clock
_clock.play()
_cam = m.scene.camera
_cam.animated = False

# Test: move camera every frame (like perf_mem_test.py does)
_move_cam = os.environ.get("BENCH_MOVE", "0") == "1"
if _move_cam:
    from falcor import float3
    _wp = [(-0.0384, 0.7121, -0.6126, -1.0226, 0.7627, -0.7817)]
    _mi = [0]
    def _move():
        _mi[0] += 1
        # tiny jitter to trigger CameraPropertiesChanged
        i = _mi[0] % 100
        _cam.position = float3(_wp[0][0]+i*0.0001, _wp[0][1], _wp[0][2])
        _cam.target = float3(_wp[0][3], _wp[0][4], _wp[0][5])

print(f"[probe] {ALGO}: callback installed, warmup={WARMUP}s measure={DURATION}s, move_cam={_move_cam}")

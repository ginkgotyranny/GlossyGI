#!/bin/bash
# Compare ptdn / ptnrd / ncr on a scene: FPS + RSS + GPU power. (legacy)
#
# Usage:
#   ./compare_algos.sh                 # default: bathroom, 25s each
#   ./compare_algos.sh kitchen 30      # kitchen scene, 30s each
#
# Output (under BENCH_RESULTS_DIR, default <repo>/perf_mem_test/results):
#   <results>/benchmark_<algo>_<ts>.csv   — per-frame FPS/RSS (one per algo)
#   <results>/power_<algo>_<ts>.csv       — GPU power samples (one per algo)
#   <results>/compare_report.html         — side-by-side comparison report

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PERF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PERF_DIR/.." && pwd)"
cd "$REPO_ROOT"

export BENCH_REPO_ROOT="$REPO_ROOT"
RESULTS_DIR="${BENCH_RESULTS_DIR:-$PERF_DIR/results}"
mkdir -p "$RESULTS_DIR"
export BENCH_RESULTS_DIR="$RESULTS_DIR"
export BENCH_WAYPOINTS_FILE="${BENCH_WAYPOINTS_FILE:-$PERF_DIR/scene_waypoints.txt}"
export BENCH_GRAPH_DIR="${BENCH_GRAPH_DIR:-$REPO_ROOT/scripts}"

SCENE=${1:-bathroom}
DURATION=${2:-25}
SCENE_FILE="scene-animation.pyscene"
if [ "$SCENE" = "teapot" ]; then
    SCENE_FILE="scene-v4.pbrt"
fi
SCENE_PATH="media/scenes/${SCENE}/${SCENE_FILE}"

if [ ! -f "$SCENE_PATH" ]; then
    echo "ERROR: scene file not found: $SCENE_PATH"
    exit 1
fi

MOGWAI="$REPO_ROOT/build/linux-gcc/bin/Release/Mogwai"
if [ ! -f "$MOGWAI" ]; then
    echo "ERROR: Mogwai binary not found: $MOGWAI"
    exit 1
fi

ALGOS=("ptdn" "ptnrd" "ncr")
export DISPLAY=${DISPLAY:-:10}

echo "============================================="
echo "Algorithm Comparison Benchmark (legacy)"
echo "  Repo root : $REPO_ROOT"
echo "  Results   : $RESULTS_DIR"
echo "  Scene     : $SCENE ($SCENE_FILE)"
echo "  Duration  : ${DURATION}s per algorithm"
echo "  Algorithms: ${ALGOS[*]}"
echo "============================================="
echo ""

# ---- power sampler ----------------------------------------------------------
start_power_sampler() {
    local out="$1"
    local pid_file="$2"
    echo "timestamp_s,power_w,temp_c,util_pct" > "$out"
    nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu \
               --format=csv,noheader,nounits -lms 500 >> "$out" &
    echo $! > "$pid_file"
}

stop_power_sampler() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        kill "$(cat $pid_file)" 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

# ---- run each algorithm -----------------------------------------------------
for algo in "${ALGOS[@]}"; do
    echo ""
    echo "============================================="
    echo "Running $algo ..."
    echo "============================================="

    ts=$(date +%Y%m%d_%H%M%S)
    power_csv="$RESULTS_DIR/power_${algo}_${ts}.csv"
    pid_file="/tmp/bench_power_pid_$$"

    start_power_sampler "$power_csv" "$pid_file"
    echo "  Power sampling -> $power_csv"

    BENCH_ALGO=$algo BENCH_DURATION=$DURATION BENCH_SCENE="$SCENE" \
    LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16 \
        "$MOGWAI" \
        --width 1280 --height 720 \
        --deferred \
        --script "$PERF_DIR/scripts/perf_mem_test.py" \
        --scene "$SCENE_PATH" \
        || true

    stop_power_sampler "$pid_file"
    echo "  $algo done."
done

# ---- generate comparison report ---------------------------------------------
echo ""
echo "All algorithms done. Generating comparison report..."
python3 "$SCRIPT_DIR/plot_compare.py"

echo ""
echo "Done. Open $RESULTS_DIR/compare_report.html in a browser."

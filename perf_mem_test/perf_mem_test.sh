#!/bin/bash
# Single-shot performance & memory benchmark for Falcor (one algo × one scene).
#
# Usage:
#   ./perf_mem_test.sh                       # default: ptdn + bathroom + 25s
#   ./perf_mem_test.sh ptnrd                 # test ptnrd algorithm
#   ./perf_mem_test.sh ncr 30                # test ncr for 30 seconds
#   ./perf_mem_test.sh ptdn 25 kitchen       # test ptdn on kitchen scene
#
# Output (under BENCH_RESULTS_DIR, default <repo>/perf_mem_test/results):
#   <results>/bench_<scene>_<algo>_rtxdi_off.csv      — FPS samples (perf_mem_test.py)
#   <results>/bench_<scene>_<algo>_rtxdi_off_mem.csv  — memory samples
#   (plus a benchmark_report.html via plot_benchmark.py if legacy CSVs exist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export BENCH_REPO_ROOT="$REPO_ROOT"

RESULTS_DIR="${BENCH_RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
export BENCH_RESULTS_DIR="$RESULTS_DIR"

export BENCH_WAYPOINTS_FILE="${BENCH_WAYPOINTS_FILE:-$SCRIPT_DIR/scene_waypoints.txt}"
export BENCH_GRAPH_DIR="${BENCH_GRAPH_DIR:-$REPO_ROOT/scripts}"

ALGO=${1:-ptdn}
DURATION=${2:-25}
SCENE=${3:-bathroom}
SCENE_FILE="scene-animation.pyscene"

# teapot has no pyscene, use pbrt
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

echo "============================================="
echo "Performance & Memory Benchmark"
echo "  Repo root : $REPO_ROOT"
echo "  Results   : $RESULTS_DIR"
echo "  Algorithm : $ALGO"
echo "  Scene     : $SCENE ($SCENE_FILE)"
echo "  Duration  : ${DURATION}s"
echo "============================================="
echo ""

# Run Mogwai with auto-benchmark script.
# Note: perf_mem_test.py calls os._exit(0) to terminate Mogwai; that is an
# abnormal exit that may bypass C++ destructors, so we capture the exit code
# and do NOT let `set -e` abort the script before plot generation.
export DISPLAY=${DISPLAY:-:10}
export BENCH_ALGO="$ALGO"
export BENCH_SCENE="$SCENE"
export BENCH_DURATION="$DURATION"
LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16 \
    "$MOGWAI" \
    --width 1280 --height 720 \
    --deferred \
    --script "$SCRIPT_DIR/scripts/perf_mem_test.py" \
    --scene "$SCENE_PATH" \
    || true

echo ""
echo "Benchmark complete. Generating plots..."

# Generate HTML report (legacy single-shot curves; uses plot_benchmark.py
# which reads benchmark_*.csv — only produced by the older benchmark.py).
# For 7-scene reports use plot_7scene_report.py instead.
python3 "$SCRIPT_DIR/scripts/plot_benchmark.py" || \
    echo "  (plot_benchmark.py skipped — no legacy benchmark_*.csv found)"

echo ""
echo "Done. Results in: $RESULTS_DIR"

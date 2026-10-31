#!/bin/bash
# 7-scene × 3-algo benchmark runner (single round).
#
# Usage:
#   ./run_benchmark_round.sh                  # round 1: BENCH_RTXDI=0 (useRTXDI=false)
#   ./run_benchmark_round.sh 1                # round 2: BENCH_RTXDI=1 (useRTXDI=true)
#
# Each run: 30s warmup (static camera) + 60s measure (moving camera) = ~90s.
# Total per round: 7 scenes × 3 algos × ~90s ≈ 32 min + scene load time.
#
# Output (under BENCH_RESULTS_DIR, default <repo>/perf_mem_test/results):
#   <results>/bench_<scene>_<algo>_<rtxdi_on|off>.csv
#   <results>/bench_<scene>_<algo>_<rtxdi_on|off>.meta.json
#
# Prerequisites:
#   - Mogwai built at <repo>/build/linux-gcc/bin/Release/Mogwai
#   - <repo>/scripts/OptixDenoiser.py & <repo>/scripts/PathTracerNRD_VK.py
#     useRTXDI matches the round (the orchestrator toggles these for you)

set -uo pipefail

# Resolve repo root = parent of this script's directory (perf_mem_test/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Export repo root for the Python script (which runs inside Mogwai's cwd)
export BENCH_REPO_ROOT="$REPO_ROOT"

# Default results dir (orchestrator overrides this with a date-stamped path)
RESULTS_DIR="${BENCH_RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
export BENCH_RESULTS_DIR="$RESULTS_DIR"

# Default waypoints + graph scripts (kept under perf_mem_test/ and <repo>/scripts/)
export BENCH_WAYPOINTS_FILE="${BENCH_WAYPOINTS_FILE:-$SCRIPT_DIR/scene_waypoints.txt}"
export BENCH_GRAPH_DIR="${BENCH_GRAPH_DIR:-$REPO_ROOT/scripts}"

RTXDI_VAL=${1:-0}                      # 0 = useRTXDI=false (round 1), 1 = useRTXDI=true (round 2)
export BENCH_RTXDI="$RTXDI_VAL"
RTXDI_TAG="rtxdi_on"
[ "$RTXDI_VAL" = "0" ] && RTXDI_TAG="rtxdi_off"

DURATION=${BENCH_DURATION:-60}
WARMUP=${BENCH_WARMUP_S:-30}
export BENCH_DURATION="$DURATION"
export BENCH_WARMUP_S="$WARMUP"
export DISPLAY=${DISPLAY:-:10}

MOGWAI="$REPO_ROOT/build/linux-gcc/bin/Release/Mogwai"
LD_PRELOAD="/lib/x86_64-linux-gnu/libpng16.so.16"
BENCH_SCRIPT="$SCRIPT_DIR/scripts/perf_mem_test.py"

# Pre-flight GPU/process contention check
source "$SCRIPT_DIR/scripts/wait_gpu_free.sh"

SCENES=("bathroom" "bathroom2" "cornell-box" "kitchen" "living-room" "teapot" "veach-ajar")
ALGOS=("ptdn" "ptnrd" "ncr")

# Scene file: all use scene-animation.pyscene except teapot (only scene-v4.pbrt)
scene_file() {
    case "$1" in
        teapot) echo "scene-v4.pbrt" ;;
        *)      echo "scene-animation.pyscene" ;;
    esac
}

if [ ! -f "$MOGWAI" ]; then
    echo "ERROR: Mogwai binary not found at $MOGWAI"
    exit 1
fi
if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: bench script not found: $BENCH_SCRIPT"
    exit 1
fi

echo "============================================="
echo "7-Scene Benchmark — Round: $RTXDI_TAG"
echo "  Repo root  : $REPO_ROOT"
echo "  Results dir: $RESULTS_DIR"
echo "  RTXDI      : $RTXDI_VAL"
echo "  Warmup     : ${WARMUP}s (static camera)"
echo "  Measure    : ${DURATION}s (moving camera)"
echo "  Scenes     : ${SCENES[*]}"
echo "  Algorithms : ${ALGOS[*]}"
echo "============================================="

ROUND_START=$(date +%s)

for scene in "${SCENES[@]}"; do
    sfile=$(scene_file "$scene")
    spath="media/scenes/${scene}/${sfile}"
    if [ ! -f "$spath" ]; then
        echo "ERROR: scene file not found: $spath — skipping"
        continue
    fi
    for algo in "${ALGOS[@]}"; do
        echo ""
        echo "---------------------------------------------"
        echo "[$(date +%H:%M:%S)] scene=$scene  algo=$algo  rtxdi=$RTXDI_TAG"
        echo "---------------------------------------------"

        export BENCH_ALGO="$algo"
        export BENCH_SCENE="$scene"

        # Stale CSV path (used to detect success after the run)
        csv_out="$RESULTS_DIR/bench_${scene}_${algo}_${RTXDI_TAG}.csv"

        # Wait for GPU to be free before launching (avoids contention skewing data)
        wait_gpu_free

        LD_PRELOAD="$LD_PRELOAD" "$MOGWAI" \
            --width 1280 --height 720 \
            --deferred \
            --script "$BENCH_SCRIPT" \
            --scene "$spath" \
            || true

        if [ -f "$csv_out" ]; then
            lines=$(wc -l < "$csv_out")
            echo "  -> $csv_out (${lines} lines)"
        else
            echo "  -> WARNING: no CSV produced for $scene/$algo"
        fi
    done
done

ROUND_END=$(date +%s)
echo ""
echo "============================================="
echo "Round $RTXDI_TAG complete. Elapsed: $((ROUND_END - ROUND_START))s"
echo "Results: $RESULTS_DIR"
echo "============================================="

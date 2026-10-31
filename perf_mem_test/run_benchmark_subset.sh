#!/bin/bash
# Subset runner: only run specified scenes (space-separated env BENCH_SCENES).
# Reuses the same logic as run_benchmark_round.sh.
#
# Usage:
#   BENCH_SCENES="cornell-box veach-ajar" ./run_benchmark_subset.sh 0   # rtxdi_off
#   BENCH_SCENES="cornell-box veach-ajar" ./run_benchmark_subset.sh 1   # rtxdi_on

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export BENCH_REPO_ROOT="$REPO_ROOT"

RESULTS_DIR="${BENCH_RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
export BENCH_RESULTS_DIR="$RESULTS_DIR"

export BENCH_WAYPOINTS_FILE="${BENCH_WAYPOINTS_FILE:-$SCRIPT_DIR/scene_waypoints.txt}"
export BENCH_GRAPH_DIR="${BENCH_GRAPH_DIR:-$REPO_ROOT/scripts}"

RTXDI_VAL=${1:-0}
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

source "$SCRIPT_DIR/scripts/wait_gpu_free.sh"

SCENES=(${BENCH_SCENES:-"cornell-box veach-ajar"})
ALGOS=("ptdn" "ptnrd" "ncr")

scene_file() {
    case "$1" in
        teapot) echo "scene-v4.pbrt" ;;
        *)      echo "scene-animation.pyscene" ;;
    esac
}

echo "============================================="
echo "Subset benchmark round: $RTXDI_TAG"
echo "  Repo root  : $REPO_ROOT"
echo "  Results dir: $RESULTS_DIR"
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

        csv_out="$RESULTS_DIR/bench_${scene}_${algo}_${RTXDI_TAG}.csv"

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

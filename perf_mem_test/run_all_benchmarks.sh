#!/bin/bash
# =============================================================================
# run_all_benchmarks.sh — End-to-end 7-scene × 3-algo × 2-round benchmark.
#
# One command runs the WHOLE pipeline:
#   1. Round 1 (useRTXDI = false): 7 scenes × 3 algos × (30s warmup + 60s measure)
#   2. Auto-toggle useRTXDI=true in OptixDenoiser.py & PathTracerNRD_VK.py
#      (runtime shader define — no C++ recompile needed)
#   3. Round 2 (useRTXDI = true):  same 7 × 3 matrix
#   4. Restore the original useRTXDI values (always, even on error / Ctrl-C)
#   5. Generate benchmark_7scene_report.html from both rounds' CSVs
#
# All artifacts are archived under a date-stamped folder — never overwritten:
#   <repo>/perf_mem_test/results/<YYYY-MM-DD_HHMMSS>/
#       bench_<scene>_<algo>_<rtxdi_off|on>.csv
#       bench_<scene>_<algo>_<rtxdi_off|on>_mem.csv
#       bench_<scene>_<algo>_<rtxdi_off|on>.meta.json
#       debug.log
#       round1_rtxdi_off.log
#       round2_rtxdi_on.log
#       benchmark_7scene_report.html
#
# Usage:
#   ./run_all_benchmarks.sh                # full run, ~70 min
#   BENCH_DURATION=30 ./run_all_benchmarks.sh   # shorter measure window
#   SKIP_ROUND2=1 ./run_all_benchmarks.sh       # only round 1 (RTXDI off)
#   SKIP_REPORT=1 ./run_all_benchmarks.sh       # skip HTML generation
#   DRY_RUN=1   ./run_all_benchmarks.sh         # print plan, run nothing
#
# Prerequisites:
#   - Mogwai built at <repo>/build/linux-gcc/bin/Release/Mogwai
#   - 7 scenes under <repo>/media/scenes/{bathroom,bathroom2,cornell-box,
#     kitchen,living-room,teapot,veach-ajar}/
#   - GPU free (RTX 3090 shared — wait_gpu_free blocks until idle)
#   - DISPLAY set (defaults to :10)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- date-stamped results dir (never overwrite) -----------------------------
STAMP=$(date +%Y-%m-%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/$STAMP"
mkdir -p "$RESULTS_DIR"

# Export env for child scripts (run_benchmark_round.sh, perf_mem_test.py)
export BENCH_REPO_ROOT="$REPO_ROOT"
export BENCH_RESULTS_DIR="$RESULTS_DIR"
export BENCH_WAYPOINTS_FILE="${BENCH_WAYPOINTS_FILE:-$SCRIPT_DIR/scene_waypoints.txt}"
export BENCH_GRAPH_DIR="${BENCH_GRAPH_DIR:-$REPO_ROOT/scripts}"
export BENCH_DURATION="${BENCH_DURATION:-60}"
export BENCH_WARMUP_S="${BENCH_WARMUP_S:-30}"
export DISPLAY="${DISPLAY:-:10}"

MOGWAI="$REPO_ROOT/build/linux-gcc/bin/Release/Mogwai"
GRAPH_FILES=(
    "$REPO_ROOT/scripts/OptixDenoiser.py"
    "$REPO_ROOT/scripts/PathTracerNRD_VK.py"
)

# ---- preflight --------------------------------------------------------------
echo "============================================="
echo "Falcor 7-Scene Full Benchmark"
echo "  Repo root  : $REPO_ROOT"
echo "  Results dir: $RESULTS_DIR"
echo "  Warmup     : ${BENCH_WARMUP_S}s (static camera)"
echo "  Measure    : ${BENCH_DURATION}s (moving camera)"
echo "  RTXDI      : round1=off -> toggle -> round2=on"
echo "  Display    : $DISPLAY"
[ "${DRY_RUN:-0}" = "1" ] && echo "  DRY_RUN    : 1 (no Mogwai launches)"
[ "${SKIP_ROUND2:-0}" = "1" ] && echo "  SKIP_ROUND2: 1"
[ "${SKIP_REPORT:-0}" = "1" ] && echo "  SKIP_REPORT: 1"
echo "============================================="

if [ ! -f "$MOGWAI" ]; then
    echo "ERROR: Mogwai binary not found: $MOGWAI"
    exit 1
fi
for gf in "${GRAPH_FILES[@]}"; do
    if [ ! -f "$gf" ]; then
        echo "ERROR: render-graph script not found: $gf"
        exit 1
    fi
done

# Snapshot current useRTXDI values so we can restore them on exit (any exit).
declare -A ORIG_RTXDI
for gf in "${GRAPH_FILES[@]}"; do
    # match 'useRTXDI': <val> or "useRTXDI": <val> (True/False)
    val=$(grep -oE "['\"]useRTXDI['\"][[:space:]]*:[[:space:]]*(True|False)" "$gf" \
          | head -1 | grep -oE "(True|False)")
    ORIG_RTXDI["$gf"]="${val:-unknown}"
    echo "  [preflight] $gf -> useRTXDI=${val:-unknown}"
done

restore_rtxdi() {
    local rc=$?
    echo ""
    echo "[restore] restoring original useRTXDI values..."
    for gf in "${GRAPH_FILES[@]}"; do
        local want="${ORIG_RTXDI[$gf]}"
        if [ "$want" = "unknown" ]; then
            echo "  [restore] $gf: original value unknown, leaving as-is"
            continue
        fi
        # set 'useRTXDI': <want>  AND  "useRTXDI": <want>  (covers both quote styles)
        sed -i -E "s/(['\"]useRTXDI['\"][[:space:]]*:[[:space:]]*)(True|False)/\1$want/g" "$gf"
        local now
        now=$(grep -oE "['\"]useRTXDI['\"][[:space:]]*:[[:space:]]*(True|False)" "$gf" | head -1 | grep -oE "(True|False)")
        echo "  [restore] $gf: useRTXDI=$now (target=$want)"
    done
    echo "[restore] done. Exiting with rc=$rc."
    exit $rc
}
trap restore_rtxdi EXIT INT TERM

# ---- helper: force a specific useRTXDI value in both graph files -----------
set_rtxdi() {
    local want="$1"  # True | False
    for gf in "${GRAPH_FILES[@]}"; do
        sed -i -E "s/(['\"]useRTXDI['\"][[:space:]]*:[[:space:]]*)(True|False)/\1$want/g" "$gf"
    done
    echo "  [set_rtxdi] useRTXDI -> $want in: $(basename "${GRAPH_FILES[0]}"), $(basename "${GRAPH_FILES[1]}")"
}

# ---- round 1: useRTXDI = false ----------------------------------------------
echo ""
echo "#############################################"
echo "# ROUND 1: useRTXDI = false (rtxdi_off)"
echo "#############################################"
set_rtxdi "False"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] would run: $SCRIPT_DIR/run_benchmark_round.sh 0"
else
    "$SCRIPT_DIR/run_benchmark_round.sh" 0 \
        > "$RESULTS_DIR/round1_rtxdi_off.log" 2>&1 \
        || echo "WARNING: round 1 returned non-zero (see $RESULTS_DIR/round1_rtxdi_off.log)"
fi

# ---- round 2: useRTXDI = true -----------------------------------------------
if [ "${SKIP_ROUND2:-0}" != "1" ]; then
    echo ""
    echo "#############################################"
    echo "# ROUND 2: useRTXDI = true (rtxdi_on)"
    echo "#############################################"
    set_rtxdi "True"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] would run: $SCRIPT_DIR/run_benchmark_round.sh 1"
    else
        "$SCRIPT_DIR/run_benchmark_round.sh" 1 \
            > "$RESULTS_DIR/round2_rtxdi_on.log" 2>&1 \
            || echo "WARNING: round 2 returned non-zero (see $RESULTS_DIR/round2_rtxdi_on.log)"
    fi
else
    echo "SKIP_ROUND2=1 — skipping round 2."
fi

# ---- report -----------------------------------------------------------------
if [ "${SKIP_REPORT:-0}" != "1" ]; then
    echo ""
    echo "#############################################"
    echo "# Generating 7-scene report"
    echo "#############################################"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] would run: python3 $SCRIPT_DIR/scripts/plot_7scene_report.py $RESULTS_DIR"
    else
        python3 "$SCRIPT_DIR/scripts/plot_7scene_report.py" "$RESULTS_DIR" \
            || echo "WARNING: report generation failed"
    fi
else
    echo "SKIP_REPORT=1 — skipping report generation."
fi

# ---- summary ----------------------------------------------------------------
echo ""
echo "============================================="
echo "Full benchmark complete."
echo "  Results dir: $RESULTS_DIR"
echo "  CSVs        : $(ls "$RESULTS_DIR"/bench_*.csv 2>/dev/null | wc -l)"
echo "  Meta JSON   : $(ls "$RESULTS_DIR"/bench_*.meta.json 2>/dev/null | wc -l)"
if [ -f "$RESULTS_DIR/benchmark_7scene_report.html" ]; then
    echo "  Report      : $RESULTS_DIR/benchmark_7scene_report.html"
fi
echo "Original useRTXDI values will be restored on exit."
echo "============================================="
# trap will fire restore_rtxdi on normal exit too
exit 0

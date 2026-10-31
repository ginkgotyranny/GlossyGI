#!/bin/bash
# wait_gpu_free.sh — pre-flight check before launching a benchmark run.
#
# Waits until no other GPU-consuming processes or same-name Mogwai processes
# are running. Prevents GPU contention from skewing benchmark data.
#
# Usage: source this file, then call `wait_gpu_free` before each Mogwai launch.
# Env vars:
#   WAIT_GPU_TIMEOUT_S  (default 1800 = 30min) — max seconds to wait before giving up
#   WAIT_GPU_POLL_S     (default 15)           — poll interval
#   GPU_FREE_MIN_MB     (default 1000)         — treat GPU as busy if any process
#                                                 uses more than this many MB of FB

wait_gpu_free() {
    local timeout=${WAIT_GPU_TIMEOUT_S:-1800}
    local poll=${WAIT_GPU_POLL_S:-15}
    local min_mb=${GPU_FREE_MIN_MB:-1000}
    local waited=0
    local my_pid=$$

    while [ "$waited" -lt "$timeout" ]; do
        # 1) Check for other Mogwai processes not owned by this script tree.
        #    `--deferred` runs Mogwai headless; we match any Mogwai binary path.
        local mogwai_pids
        mogwai_pids=$(pgrep -f "build/linux-gcc/bin/Release/Mogwai" 2>/dev/null | grep -v "^${my_pid}$" || true)
        # Also filter out child PIDs of this shell (the runner forks Mogwai directly,
        # but those are the runs we *want* — we only block on *pre-existing* ones).
        # A pre-existing Mogwai will have a parent PID that is NOT in our process tree.
        local foreign=""
        if [ -n "$mogwai_pids" ]; then
            for pid in $mogwai_pids; do
                # Skip if this pid is a direct child of the current runner
                local ppid
                ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
                if [ "$ppid" != "$my_pid" ] && [ "$pid" != "$my_pid" ]; then
                    foreign="$foreign $pid"
                fi
            done
        fi

        # 2) Check nvidia-smi pmon for any process using GPU framebuffer memory.
        #    `pmon -c 1 -s m` lists compute+graphics processes with their fb (MB).
        #    We ignore the header lines (starting with '#') and our own future run.
        local gpu_procs=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            # pmon output: gpu pid type fb ccpm command
            gpu_procs=$(nvidia-smi pmon -c 1 -s m 2>/dev/null \
                | grep -v '^#' \
                | awk -v mb="$min_mb" '$4+0 > mb {print $2":"$4":"$6}' || true)
        fi

        if [ -z "$foreign" ] && [ -z "$gpu_procs" ]; then
            # GPU is free
            if [ "$waited" -gt 0 ]; then
                echo "[wait_gpu_free] GPU free after ${waited}s of waiting."
            fi
            return 0
        fi

        # Report what's blocking
        local msg=""
        if [ -n "$foreign" ]; then
            msg="Mogwai PIDs:$foreign"
        fi
        if [ -n "$gpu_procs" ]; then
            msg="${msg:+$msg }GPU procs: $(echo "$gpu_procs" | tr '\n' ', ')"
        fi
        echo "[wait_gpu_free] $(date +%H:%M:%S) waiting (${waited}s/${timeout}s) — blocked by: $msg"

        sleep "$poll"
        waited=$((waited + poll))
    done

    echo "[wait_gpu_free] TIMEOUT after ${timeout}s — GPU still busy. Proceeding anyway." >&2
    return 1
}

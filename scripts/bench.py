#!/usr/bin/env python3
"""Measure frame time and process VRAM for the ptnrd.sh and ncr.sh launches.

The scene, resolution, and render-graph script are read from those shell files.
The animation clock keeps running, so the measurement includes the scene animation.

    ./bench.sh --seconds 30
    ./bench.sh ptnrd --seconds 60 --warmup 15
    ./bench.sh ncr --scene media/scenes/sponza/scene-animation.pyscene

Per-frame rows go to bench_results/*.csv. The summary is printed after Mogwai exits.
"""

import argparse
import csv
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCH = {
    "ptnrd": ROOT / "ptnrd.sh",
    "ncr": ROOT / "ncr.sh",
}

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1


class _NvmlProcess(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class ProcessVram:
    """Mogwai's GPU memory from NVML, in bytes."""

    def __init__(self):
        self.lib = None
        self.devices = []
        self._graphics = None
        self._compute = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            lib.nvmlInit_v2.restype = ctypes.c_int
            if lib.nvmlInit_v2() != NVML_SUCCESS:
                return
            lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
            count = ctypes.c_uint()
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != NVML_SUCCESS:
                return
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            devices = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == NVML_SUCCESS:
                    devices.append(handle)
            self._graphics = lib.nvmlDeviceGetGraphicsRunningProcesses_v3
            self._compute = lib.nvmlDeviceGetComputeRunningProcesses_v3
            for fn in (self._graphics, self._compute):
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(_NvmlProcess)]
                fn.restype = ctypes.c_int
            self.lib = lib
            self.devices = devices
        except Exception:
            self.lib = None
            self.devices = []

    def bytes_for_pid(self, pid):
        if self.lib is None:
            return None
        total = 0
        found = False
        for device in self.devices:
            samples = []
            for fn in (self._graphics, self._compute):
                for info in self._processes(device, fn):
                    if info.pid == pid and info.usedGpuMemory != NVML_VALUE_NOT_AVAILABLE:
                        samples.append(int(info.usedGpuMemory))
            if samples:
                found = True
                total += max(samples)
        return total if found else None

    def _processes(self, device, fn):
        for _ in range(3):
            count = ctypes.c_uint(0)
            rc = fn(device, ctypes.byref(count), None)
            if count.value == 0:
                return []
            if rc not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
                return []
            array = (_NvmlProcess * count.value)()
            filled = ctypes.c_uint(count.value)
            rc = fn(device, ctypes.byref(filled), array)
            if rc == NVML_SUCCESS:
                return [array[i] for i in range(filled.value)]
            if rc != NVML_ERROR_INSUFFICIENT_SIZE:
                return []
        return []


def _in_mogwai():
    return "m" in globals()


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize(rows):
    times = [row["frame_time_s"] for row in rows]
    vram = [row["vram_mib"] for row in rows if row["vram_mib"] is not None]
    ordered = sorted(times)
    mean_time = statistics.fmean(times)
    summary = {
        "frames": len(rows),
        "duration_s": sum(times),
        "frame_time_s": {
            "mean": mean_time,
            "median": _percentile(ordered, 0.5),
            "p95": _percentile(ordered, 0.95),
            "min": ordered[0],
            "max": ordered[-1],
        },
        "fps_mean": (1.0 / mean_time) if mean_time > 0.0 else None,
    }
    if vram:
        summary["vram_mib"] = {
            "mean": statistics.fmean(vram),
            "min": min(vram),
            "max": max(vram),
        }
    else:
        summary["vram_mib"] = None
    return summary


def _run_benchmark():
    graph = os.environ["BENCH_GRAPH"]
    scene = os.environ["BENCH_SCENE"]
    seconds = float(os.environ["BENCH_SECONDS"])
    warmup = float(os.environ["BENCH_WARMUP"])
    csv_path = Path(os.environ["BENCH_CSV"])
    summary_path = Path(os.environ["BENCH_SUMMARY"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    m.ui = False
    graph_code = Path(graph).read_text()
    exec(compile(graph_code, graph, "exec"), globals())
    m.loadScene(scene)

    vram = ProcessVram()
    pid = os.getpid()
    rows = []
    handle = csv_path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=["frame", "phase", "frame_time_s", "vram_mib"])
    writer.writeheader()
    handle.flush()

    def sample(phase):
        start = time.perf_counter()
        m.renderFrame()
        frame_time = time.perf_counter() - start
        used = vram.bytes_for_pid(pid)
        row = {
            "frame": len(rows) + 1,
            "phase": phase,
            "frame_time_s": frame_time,
            "vram_mib": None if used is None else used / (1024.0 * 1024.0),
        }
        rows.append(row)
        writer.writerow({
            "frame": row["frame"],
            "phase": row["phase"],
            "frame_time_s": "%.6f" % row["frame_time_s"],
            "vram_mib": "" if row["vram_mib"] is None else "%.3f" % row["vram_mib"],
        })
        handle.flush()
        return frame_time

    elapsed = 0.0
    while elapsed < warmup:
        elapsed += sample("warmup")
    elapsed = 0.0
    while elapsed < seconds:
        elapsed += sample("measure")
    handle.close()

    measured = [row for row in rows if row["phase"] == "measure"]
    if not measured:
        raise RuntimeError("no measured frames")
    summary = _summarize(measured)
    summary.update({
        "method": os.environ["BENCH_METHOD"],
        "scene": scene,
        "width": int(os.environ["BENCH_WIDTH"]),
        "height": int(os.environ["BENCH_HEIGHT"]),
        "warmup_s": warmup,
        "seconds": seconds,
        "warmup_frames": sum(1 for row in rows if row["phase"] == "warmup"),
        "csv": str(csv_path),
    })
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    exit()


def _format_summary(summary):
    frame = summary["frame_time_s"]
    lines = [
        "方法: %s" % summary["method"],
        "场景: %s" % summary["scene"],
        "分辨率: %dx%d" % (summary["width"], summary["height"]),
        "预热: %.1f s，%d 帧（不计入统计）" % (summary["warmup_s"], summary["warmup_frames"]),
        "测量帧数: %d" % summary["frames"],
        "测量时长: %.3f s" % summary["duration_s"],
        "帧时间: 平均 %.2f ms，中位 %.2f ms，p95 %.2f ms，最小 %.2f ms，最大 %.2f ms" % (
            frame["mean"] * 1e3,
            frame["median"] * 1e3,
            frame["p95"] * 1e3,
            frame["min"] * 1e3,
            frame["max"] * 1e3,
        ),
        "平均 FPS: %.2f" % summary["fps_mean"],
    ]
    if frame["median"] > 0.0 and frame["max"] > 3.0 * frame["median"]:
        lines.append("提示: 最慢帧远高于中位数，预热可能不够，着色器或网络还在第一次编译。")
    vram = summary.get("vram_mib")
    if vram:
        lines.append("显存: 平均 %.1f MiB，最小 %.1f MiB，最大 %.1f MiB" % (vram["mean"], vram["min"], vram["max"]))
    else:
        lines.append("显存: 未能读到本进程的 GPU 占用")
    lines.append("明细: %s" % summary["csv"])
    return "\n".join(lines)


def _parse_launch(path):
    text = path.read_text()
    def value(pattern):
        match = re.search(pattern, text)
        if not match:
            raise SystemExit("无法从 %s 解析 %s" % (path, pattern))
        return match.group(1)
    return {
        "preload": value(r"LD_PRELOAD=(\S+)"),
        "mogwai": value(r"(\./\S*Mogwai)"),
        "width": int(value(r"--width\s+(\d+)")),
        "height": int(value(r"--height\s+(\d+)")),
        "graph": value(r"--script\s+(\S+)"),
        "scene": value(r"--scene\s+(\S+)"),
    }


def _host_main():
    parser = argparse.ArgumentParser(description="测量 ptnrd.sh 和 ncr.sh 的平均 FPS 与进程显存。")
    parser.add_argument("method", nargs="*", choices=sorted(LAUNCH), help="要测的方法，默认两个都测")
    parser.add_argument("--seconds", type=float, default=30.0, help="计入统计的渲染时长（秒），默认 30")
    parser.add_argument("--warmup", type=float, default=10.0, help="开始统计前先渲染的时长（秒），默认 10")
    parser.add_argument("--scene", help="覆盖 sh 文件里的场景")
    parser.add_argument("--width", type=int, help="覆盖 sh 文件里的宽度")
    parser.add_argument("--height", type=int, help="覆盖 sh 文件里的高度")
    parser.add_argument("--out", default="bench_results", help="CSV 输出目录，默认 bench_results")
    parser.add_argument("--headless", action="store_true", help="不打开窗口")
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.warmup < 0.0:
        raise SystemExit("--seconds 必须大于 0，--warmup 不能为负")

    methods = args.method or ["ptnrd", "ncr"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for method in methods:
        launch = _parse_launch(LAUNCH[method])
        scene = args.scene or launch["scene"]
        width = args.width or launch["width"]
        height = args.height or launch["height"]
        stem = "%s_%s_%s" % (method, Path(scene).stem, stamp)
        csv_path = out_dir / (stem + ".csv")
        summary_path = out_dir / (stem + ".json")
        env = os.environ.copy()
        env["LD_PRELOAD"] = launch["preload"]
        env["PYTHONUNBUFFERED"] = "1"
        env.update({
            "BENCH_METHOD": method,
            "BENCH_GRAPH": launch["graph"],
            "BENCH_SCENE": scene,
            "BENCH_SECONDS": str(args.seconds),
            "BENCH_WARMUP": str(args.warmup),
            "BENCH_WIDTH": str(width),
            "BENCH_HEIGHT": str(height),
            "BENCH_CSV": str(csv_path),
            "BENCH_SUMMARY": str(summary_path),
        })
        command = [launch["mogwai"], "--width", str(width), "--height", str(height), "--script", "scripts/bench.py"]
        if args.headless:
            command.append("--headless")
        print("启动 %s  %s  %dx%d  测量 %.1f s" % (method, scene, width, height, args.seconds), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env)
        if not summary_path.is_file():
            print("%s 失败，退出码 %s，没有写出统计文件" % (method, completed.returncode), file=sys.stderr)
            return completed.returncode or 1
        summary = json.loads(summary_path.read_text())
        if summary.get("error"):
            print(summary["error"], file=sys.stderr)
            return completed.returncode or 1
        summaries.append(summary)
        print()
        print(_format_summary(summary))
        print()

    if len(summaries) > 1:
        print("对比")
        print("%-8s %10s %12s %14s %14s" % ("方法", "平均FPS", "帧时间ms", "显存平均MiB", "显存峰值MiB"))
        for summary in summaries:
            vram = summary["vram_mib"] or {}
            print("%-8s %10.2f %12.2f %14s %14s" % (
                summary["method"],
                summary["fps_mean"],
                summary["frame_time_s"]["mean"] * 1e3,
                "%.1f" % vram["mean"] if vram else "-",
                "%.1f" % vram["max"] if vram else "-",
            ))
    return 0


if _in_mogwai():
    try:
        _run_benchmark()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        summary_path = os.environ.get("BENCH_SUMMARY")
        if summary_path:
            Path(summary_path).write_text(json.dumps({"error": traceback.format_exc()}) + "\n")
        exit(1)
elif __name__ == "__main__":
    sys.exit(_host_main())

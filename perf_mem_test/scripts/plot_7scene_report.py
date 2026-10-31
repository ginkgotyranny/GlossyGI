"""
7-scene benchmark report generator (revised).

Reads <results_dir>/bench_<scene>_<algo>_<rtxdi_tag>.csv (FPS, time-based)
and bench_<scene>_<algo>_<rtxdi_tag>_mem.csv (memory, nvitop) and emits a
self-contained HTML report with:
  - FPS line charts (per round, 7 scenes × 3 algos, time-based x-axis)
  - FPS summary table (cols=7 scenes + diff, rows=RTXDI on/off × ptdn/ptnrd/ncr)
  - Memory summary tables: avg + peak (cols=7 scenes × CPU/GPU/Total + diff,
    rows=RTXDI on/off × ptdn/ptnrd/ncr)
  - Methodology section

Requirements (per user spec):
  - NCR must be ≥50% faster than ptdn/ptnrd (perf target). Cells where NCR
    does NOT beat ptdn/ptnrd by 50% are highlighted red.
  - NCR memory must be ≤ ptdn/ptnrd + 200MB (memory target). Cells where NCR
    exceeds ptdn/ptnrd by more than 200MB are highlighted red.

Usage:
  python3 perf_mem_test/scripts/plot_7scene_report.py [results_dir]

If results_dir is omitted, falls back to $BENCH_RESULTS_DIR, then
<repo>/perf_mem_test/results/<latest dated subdir>, then <repo>/perf_mem_test/results.
"""

import os
import sys
import csv
import datetime
import html

_REPO_ROOT = os.environ.get("BENCH_REPO_ROOT", os.getcwd())
_DEFAULT_RESULTS_ROOT = os.path.join(_REPO_ROOT, "perf_mem_test", "results")

def _resolve_results_dir():
    # 1) CLI arg
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    # 2) env var
    env = os.environ.get("BENCH_RESULTS_DIR")
    if env:
        return os.path.abspath(env)
    # 3) latest dated subdir under results/
    if os.path.isdir(_DEFAULT_RESULTS_ROOT):
        dated = sorted(d for d in os.listdir(_DEFAULT_RESULTS_ROOT)
                       if os.path.isdir(os.path.join(_DEFAULT_RESULTS_ROOT, d)))
        if dated:
            return os.path.join(_DEFAULT_RESULTS_ROOT, dated[-1])
    # 4) results root itself
    return _DEFAULT_RESULTS_ROOT

RESULTS_DIR = _resolve_results_dir()
OUT_HTML = os.path.join(RESULTS_DIR, "benchmark_7scene_report.html")

SCENES = ["bathroom", "bathroom2", "cornell-box", "kitchen",
          "living-room", "teapot", "veach-ajar"]
ALGOS = ["ptdn", "ptnrd", "ncr"]
ROUNDS = ["rtxdi_off", "rtxdi_on"]
ROUND_LABEL = {"rtxdi_off": "useRTXDI = false", "rtxdi_on": "useRTXDI = true"}
ALGO_COLORS = {"ptdn": "#2563eb", "ptnrd": "#dc2626", "ncr": "#16a34a"}

# Targets
PERF_TARGET_PCT = 50.0   # NCR must be >= (ptdn or ptnrd) * 1.5
MEM_TARGET_MB = 200.0    # NCR total mem must be <= (ptdn or ptnrd) + 200


def _load_fps(path):
    if not os.path.isfile(path):
        return None
    ts, fps = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["elapsed_s"]))
                fps.append(float(row["fps"]))
            except (ValueError, KeyError):
                continue
    return {"t": ts, "fps": fps, "n": len(fps)} if ts else None


def _load_mem(path):
    if not os.path.isfile(path):
        return None
    cpu, gpu = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                cpu.append(float(row["cpu_mem_mb"]))
                gpu.append(float(row["gpu_mem_mb"]))
            except (ValueError, KeyError):
                continue
    if not cpu:
        return None
    tot = [c + g for c, g in zip(cpu, gpu)]
    return {"cpu": cpu, "gpu": gpu, "tot": tot}


def _harmonic_fps_from_total_frames(meta_path):
    """Avg FPS = total_frames / wall_time (most accurate, from meta JSON)."""
    import json
    if not os.path.isfile(meta_path):
        return None
    try:
        d = json.load(open(meta_path))
        tf = d.get("total_frames", 0)
        wt = d.get("wall_time_s", 0)
        if wt > 0:
            return tf / wt
    except:
        pass
    return None


def _avg(vals):
    v = [x for x in vals if x > 0]
    return sum(v) / len(v) if v else 0.0


def _peak(vals):
    v = [x for x in vals if x > 0]
    return max(v) if v else 0.0


def _line_svg(series_dict, w=760, h=260, ylabel="", title=""):
    """SVG line chart. series_dict: {label: (x_list, y_list, color)}."""
    margin_l, margin_r, margin_t, margin_b = 56, 16, 28, 36
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    all_x, all_y = [], []
    for _, (xs, ys, _) in series_dict.items():
        all_x.extend(xs)
        all_y.extend([y for y in ys if y is not None and y > 0])
    if not all_x or not all_y:
        return f'<div class="empty">无数据: {html.escape(title)}</div>'
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if ymax <= ymin:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.08
    ymin = max(0, ymin - pad)
    ymax += pad

    def sx(x):
        return margin_l + (x - xmin) / (xmax - xmin + 1e-9) * plot_w

    def sy(y):
        return margin_t + (1 - (y - ymin) / (ymax - ymin + 1e-9)) * plot_h

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w/2}" y="16" text-anchor="middle" class="chart-title">{html.escape(title)}</text>')
    for i in range(5):
        gy = ymin + (ymax - ymin) * i / 4
        yy = sy(gy)
        parts.append(f'<line x1="{margin_l}" y1="{yy:.1f}" x2="{w-margin_r}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_l-6}" y="{yy+3:.1f}" text-anchor="end" class="axis">{gy:.0f}</text>')
    parts.append(f'<text x="{margin_l}" y="{h-8}" class="axis">t=0s</text>')
    parts.append(f'<text x="{w-margin_r}" y="{h-8}" text-anchor="end" class="axis">t={xmax:.0f}s</text>')
    parts.append(f'<text x="{12}" y="{h/2}" transform="rotate(-90 12 {h/2})" text-anchor="middle" class="axis">{html.escape(ylabel)}</text>')
    for label, (xs, ys, color) in series_dict.items():
        pts = []
        for x, y in zip(xs, ys):
            if y is None or y <= 0:
                continue
            pts.append(f"{sx(x):.1f},{sy(y):.1f}")
        if pts:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.4" opacity="0.9"/>')
    lx = margin_l + 6
    for label, (_, _, color) in series_dict.items():
        parts.append(f'<rect x="{lx}" y="{h-12}" width="10" height="3" fill="{color}"/>')
        parts.append(f'<text x="{lx+13}" y="{h-8}" class="legend">{html.escape(label)}</text>')
        lx += 70
    parts.append('</svg>')
    return "".join(parts)


def _perf_ok(ncr_val, other_val):
    """NCR must be >= other * 1.5 (50% faster)."""
    if ncr_val is None or other_val is None or other_val <= 0:
        return True  # can't judge missing data
    return ncr_val >= other_val * (1 + PERF_TARGET_PCT / 100.0)


def _mem_ok(ncr_val, other_val):
    """NCR total mem must be <= other + 200MB."""
    if ncr_val is None or other_val is None:
        return True
    return ncr_val <= other_val + MEM_TARGET_MB


def build_report():
    # ---- load all data ----
    fps_data = {}   # (scene, algo, round) -> {t, fps, n}
    mem_data = {}   # (scene, algo, round) -> {cpu, gpu, tot}
    for scene in SCENES:
        for algo in ALGOS:
            for rnd in ROUNDS:
                fps_data[(scene, algo, rnd)] = _load_fps(
                    os.path.join(RESULTS_DIR, f"bench_{scene}_{algo}_{rnd}.csv"))
                mem_data[(scene, algo, rnd)] = _load_mem(
                    os.path.join(RESULTS_DIR, f"bench_{scene}_{algo}_{rnd}_mem.csv"))

    # ---- FPS stats (total_frames/wall from meta = harmonic mean) ----
    fps_stats = {}  # (algo, round)[scene] = avg_fps
    for algo in ALGOS:
        for rnd in ROUNDS:
            fps_stats[(algo, rnd)] = {}
            for scene in SCENES:
                mp = os.path.join(RESULTS_DIR, f"bench_{scene}_{algo}_{rnd}.meta.json")
                v = _harmonic_fps_from_total_frames(mp)
                fps_stats[(algo, rnd)][scene] = v

    # ---- memory stats ----
    # mem_stats[(algo, round)][scene] = (avg_cpu, peak_cpu, avg_gpu, peak_gpu, avg_tot, peak_tot)
    mem_stats = {}
    for algo in ALGOS:
        for rnd in ROUNDS:
            mem_stats[(algo, rnd)] = {}
            for scene in SCENES:
                d = mem_data.get((scene, algo, rnd))
                if not d:
                    mem_stats[(algo, rnd)][scene] = None
                    continue
                mem_stats[(algo, rnd)][scene] = (
                    _avg(d["cpu"]), _peak(d["cpu"]),
                    _avg(d["gpu"]), _peak(d["gpu"]),
                    _avg(d["tot"]), _peak(d["tot"]),
                )

    # ---- build HTML ----
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_runs = sum(1 for v in fps_data.values() if v)

    H = []
    H.append(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>7-Scene Benchmark Report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #1f2937; max-width: 1400px; }}
h1 {{ font-size: 22px; border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
h2 {{ font-size: 18px; margin-top: 36px; color: #1e40af; }}
h3 {{ font-size: 15px; margin-top: 22px; color: #374151; }}
h4 {{ font-size: 13px; margin-top: 16px; color: #4b5563; margin-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 12px 0; font-size: 12px; }}
th, td {{ border: 1px solid #d1d5db; padding: 4px 7px; text-align: right; }}
th {{ background: #eff6ff; font-weight: 600; }}
td.scene, th.scene {{ text-align: left; font-weight: 600; background: #f9fafb; }}
td.diff {{ color: #6b7280; font-style: italic; }}
tr.round-sep td {{ border-top: 2px solid #9ca3af; }}
.bad {{ background: #fecaca !important; }}  /* red — target NOT met */
.missing {{ color: #9ca3af; }}
.chart {{ width: 100%; max-width: 760px; height: auto; margin: 8px 0; }}
.chart-title {{ font-size: 13px; font-weight: 600; fill: #374151; }}
.axis {{ font-size: 10px; fill: #6b7280; }}
.legend {{ font-size: 10px; fill: #374151; }}
.grid {{ stroke: #e5e7eb; stroke-width: 0.5; }}
.empty {{ color: #9ca3af; font-style: italic; padding: 12px; }}
.method {{ background: #f9fafb; border-left: 3px solid #2563eb; padding: 12px 16px; margin: 12px 0; font-size: 13px; line-height: 1.7; }}
.method code {{ background: #e5e7eb; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
.note {{ font-size: 12px; color: #6b7280; }}
.legend-box {{ display: inline-block; width: 14px; height: 14px; border: 1px solid #991b1b; background: #fecaca; vertical-align: middle; margin: 0 4px; }}
</style></head><body>
<h1>Falcor 7-Scene Performance &amp; Memory Benchmark</h1>
<p class="note">Generated: {now} &middot; Runs loaded: {n_runs} / {len(SCENES)*len(ALGOS)*len(ROUNDS)} &middot;
Resolution: 1280x720 &middot; Warmup: 30s static &middot; Measure: 60s moving camera</p>
<p class="note"><span class="legend-box"></span> 红色背景 = 不达标：
帧率要求 NCR 比 ptdn/ptnrd 快 ≥{PERF_TARGET_PCT:.0f}%；
内存要求 NCR 总量 ≤ ptdn/ptnrd + {MEM_TARGET_MB:.0f}MB</p>
""")

    # ---- Methodology ----
    H.append(f"""<h2>一、测试方法</h2>
<div class="method">
<p><b>测试流程</b>：每个场景对每个算法（ptdn / ptnrd / ncr）独立运行一次 Mogwai。
应用启动后先静置相机 <b>30 秒</b>（warmup，等首帧 shader 编译 / BLAS 构建 / CUDA 初始化完成），
随后启动相机沿航点轨迹运动，采集 <b>60 秒</b>数据后退出。两轮测试分别对应
<code>useRTXDI=false</code>（round 1）与 <code>useRTXDI=true</code>（round 2）。</p>

<p><b>RTXDI 开关</b>：<code>useRTXDI</code> 是 PathTracer pass 的运行时选项，通过 Slang shader
define <code>USE_RTXDI</code> 在 shader 运行时编译期注入，<b>无需 C++ 重新编译</b>——切换后下次
启动即生效。NCR 不使用 PathTracer，无此开关，两轮结果作对照。</p>

<p><b>场景与航点</b>：7 个场景各自的相机航点记录在 <code>perf_mem_test/scene_waypoints.txt</code>，通过
<code>BENCH_SCENE</code> 环境变量读取，用 <b>Catmull-Rom 样条</b>插值生成平滑轨迹并循环。
teapot 场景无 pyscene，加载 <code>scene-v4.pbrt</code>。</p>

<p><b>FPS 采集（定时采样）</b>：渲染回调 <code>sceneUpdateCallback</code> 每帧触发（用于驱动相机），
但<b>仅每隔 30ms 墙钟时间记录一次</b>帧率样本：FPS = 窗口内帧数 / 窗口时长。
这样无论算法实际帧率高低（ptdn ~50fps 或 ncr ~120fps），采样密度都一致（~33 样本/秒），
且最小化回调内的记录开销，避免影响实际渲染帧率。汇总采用<b>调和平均</b> FPS = 总帧数 / 总时长
（从 meta JSON 的 <code>total_frames / wall_time_s</code> 取得，最精确）。</p>

<p><b>内存采集（nvitop，CPU-MEM + GPU-MEM 同频）</b>：后台线程每 <b>2 秒</b>调用一次
<code>nvitop -1 --ascii --pid &lt;PID&gt;</code>，从进程行同时解析：
<br>· <b>GPU-MEM</b>：进程行 <code>GPU-MEM</code> 列（MiB）；
<br>· <b>CPU-MEM</b>：进程行 <code>%MEM</code> 列 × <code>MemTotal</code>（= 进程 RSS，MB）。
<br>nvitop 对计算（CUDA）与图形（Vulkan）进程均可显示。若 nvitop 未列出进程，GPU-MEM 回退到
<code>nvidia-smi pmon -c 1 -s m</code>（fb 列），CPU-MEM 回退到 <code>/proc/self/status</code> VmRSS。
后台线程方式避免在渲染回调中 fork 子进程。CPU-MEM 与 GPU-MEM 采样频率<b>完全相同</b>（2s）。</p>

<p><b>内存汇总指标</b>：avg = 测量期内有效采样的算术平均；peak = 测量期内最大值。
Total = CPU-MEM + GPU-MEM。内存波动小，仅以表格汇总，不绘制折线图。</p>

<p><b>达标判定</b>：
<br>· 帧率：NCR 须比 ptdn/ptnrd 快 ≥{PERF_TARGET_PCT:.0f}%（即 NCR_fps ≥ other_fps × 1.5），否则该格标红；
<br>· 内存：NCR 总内存须 ≤ ptdn/ptnrd + {MEM_TARGET_MB:.0f}MB，否则该格标红。
<br>差值列 = NCR − ptdn（或 NCR − ptnrd），正值=超出的内存 / 提升的帧率。</p>
</div>
""")

    # ---- FPS section ----
    H.append("<h2>二、帧率（FPS）</h2>")

    # FPS line charts: per round, per scene (3 algo lines), x=elapsed_s
    for rnd in ROUNDS:
        H.append(f'<h3>{ROUND_LABEL[rnd]}</h3>')
        for scene in SCENES:
            series = {}
            for algo in ALGOS:
                d = fps_data.get((scene, algo, rnd))
                if d:
                    series[algo] = (d["t"], d["fps"], ALGO_COLORS[algo])
            if series:
                H.append(_line_svg(series, ylabel="FPS",
                                   title=f"{scene} — {ROUND_LABEL[rnd]}"))

    # FPS summary table: one table per round, scenes as rows, per-scene diff cols
    H.append("<h3>FPS 汇总（调和平均 = 总帧数/总时长，单位 fps）</h3>")
    for rnd in ROUNDS:
        H.append(f'<h4>{ROUND_LABEL[rnd]}</h4>')
        H.append('<table><thead><tr>'
                 '<th class="scene">场景</th>'
                 '<th>ptdn</th><th>ptnrd</th><th>ncr</th>'
                 '<th>Δ(ptdn−ncr)</th><th>Δ(ptnrd−ncr)</th>'
                 '<th>达标?</th></tr></thead><tbody>')
        n_bad = 0
        n_total = 0
        for scene in SCENES:
            pv = fps_stats.get(("ptdn", rnd), {}).get(scene)
            tv = fps_stats.get(("ptnrd", rnd), {}).get(scene)
            nv = fps_stats.get(("ncr", rnd), {}).get(scene)
            H.append(f'<tr><td class="scene">{scene}</td>')
            H.append(f'<td>{pv:.1f}</td>' if pv else '<td class="missing">—</td>')
            H.append(f'<td>{tv:.1f}</td>' if tv else '<td class="missing">—</td>')
            # NCR cell: red if NCR exists and fails to beat EITHER ptdn or ptnrd by 50%
            ncr_bad = False
            if nv:
                if pv and not _perf_ok(nv, pv):
                    ncr_bad = True
                if tv and not _perf_ok(nv, tv):
                    ncr_bad = True
            n_total += 1
            if ncr_bad:
                n_bad += 1
            H.append(f'<td{" class=\"bad\"" if ncr_bad else ""}>{nv:.1f}</td>' if nv else '<td class="missing">—</td>')
            # diff cols: Δ = other - ncr (negative = NCR faster, which is good for perf)
            for ov in (pv, tv):
                if nv is not None and ov is not None and ov > 0:
                    diff = ov - nv  # negative means NCR faster
                    ok = _perf_ok(nv, ov)
                    H.append(f'<td class="diff{" bad" if not ok else ""}">{diff:+.1f}</td>')
                else:
                    H.append('<td class="diff">—</td>')
            # pass/fail summary
            if nv is None:
                H.append('<td class="missing">—</td>')
            elif ncr_bad:
                H.append('<td class="bad">FAIL</td>')
            else:
                H.append('<td>OK</td>')
            H.append('</tr>')
        H.append('</tbody></table>')
        H.append(f'<p class="note">{ROUND_LABEL[rnd]}：{n_total-n_bad}/{n_total} 场景达标。'
                 'Δ = other − ncr（负值=NCR 更快，符合优化目标）。'
                 f'红色 = NCR 未比 ptdn/ptnrd 快 ≥{PERF_TARGET_PCT:.0f}%。</p>')

    # ---- Memory section (tables only, no line charts) ----
    H.append("<h2>三、内存（CPU-MEM / GPU-MEM / CPU+GPU）</h2>")
    H.append('<p class="note">内存波动小，仅以表格汇总（avg + peak），不绘制折线图。'
             'CPU-MEM = 进程 RSS（nvitop %MEM × MemTotal）；GPU-MEM = nvitop 进程行 MiB。</p>')

    # Per-scene memory table: one table per (round, stat), scenes as rows.
    # stat_idx = (cpu_idx, gpu_idx, tot_idx) into the 6-tuple.
    for stat_label, stat_idx in [("平均值 (avg, MB)", (0, 2, 4)), ("峰值 (peak, MB)", (1, 3, 5))]:
        H.append(f"<h3>内存汇总 — {stat_label}</h3>")
        ci, gi, ti = stat_idx
        for rnd in ROUNDS:
            H.append(f'<h4>{ROUND_LABEL[rnd]} — Total (CPU+GPU)</h4>')
            H.append('<table><thead><tr>'
                     '<th class="scene">场景</th>'
                     '<th>ptdn</th><th>ptnrd</th><th>ncr</th>'
                     '<th>Δ(ptdn−ncr)</th><th>Δ(ptnrd−ncr)</th>'
                     '<th>达标?</th></tr></thead><tbody>')
            n_bad = 0
            n_total = 0
            for scene in SCENES:
                ps = mem_stats.get(("ptdn", rnd), {}).get(scene)
                ts = mem_stats.get(("ptnrd", rnd), {}).get(scene)
                ns = mem_stats.get(("ncr", rnd), {}).get(scene)
                pv = ps[ti] if ps else None
                tv = ts[ti] if ts else None
                nv = ns[ti] if ns else None
                H.append(f'<tr><td class="scene">{scene}</td>')
                H.append(f'<td>{pv:.0f}</td>' if pv else '<td class="missing">—</td>')
                H.append(f'<td>{tv:.0f}</td>' if tv else '<td class="missing">—</td>')
                ncr_bad = False
                if nv:
                    if pv and not _mem_ok(nv, pv):
                        ncr_bad = True
                    if tv and not _mem_ok(nv, tv):
                        ncr_bad = True
                n_total += 1
                if ncr_bad:
                    n_bad += 1
                H.append(f'<td{" class=\"bad\"" if ncr_bad else ""}>{nv:.0f}</td>' if nv else '<td class="missing">—</td>')
                # Δ = ncr - other (positive = NCR uses more, bad for mem)
                for ov in (pv, tv):
                    if nv is not None and ov is not None:
                        diff = nv - ov
                        ok = _mem_ok(nv, ov)
                        H.append(f'<td class="diff{" bad" if not ok else ""}">{diff:+.0f}</td>')
                    else:
                        H.append('<td class="diff">—</td>')
                if nv is None:
                    H.append('<td class="missing">—</td>')
                elif ncr_bad:
                    H.append('<td class="bad">FAIL</td>')
                else:
                    H.append('<td>OK</td>')
                H.append('</tr>')
            H.append('</tbody></table>')
            H.append(f'<p class="note">{ROUND_LABEL[rnd]}：{n_total-n_bad}/{n_total} 场景达标。'
                     'Δ = ncr − other（正值=NCR 用内存更多，超出预算=不达标）。'
                     f'红色 = NCR 总内存超过 ptdn/ptnrd +{MEM_TARGET_MB:.0f}MB。</p>')

    # ---- Per-component detail (CPU/GPU breakdown) ----
    H.append("<h3>内存分项明细（CPU-MEM / GPU-MEM，avg）</h3>")
    H.append('<table><thead><tr><th class="scene">RTXDI</th><th class="scene">算法</th>')
    for s in SCENES:
        H.append(f'<th colspan="2">{s}</th>')
    H.append('</tr><tr><th></th><th></th>')
    for _ in SCENES:
        H.append('<th>CPU</th><th>GPU</th>')
    H.append('</tr></thead><tbody>')
    for ri, rnd in enumerate(ROUNDS):
        for ai, algo in enumerate(ALGOS):
            cls = ' class="round-sep"' if (ai == 0 and ri > 0) else ''
            H.append(f'<tr{cls}><td class="scene">{ROUND_LABEL[rnd]}</td><td class="scene">{algo}</td>')
            for scene in SCENES:
                s = mem_stats.get((algo, rnd), {}).get(scene)
                if s:
                    H.append(f'<td>{s[0]:.0f}</td><td>{s[2]:.0f}</td>')
                else:
                    H.append('<td class="missing">—</td><td class="missing">—</td>')
            H.append('</tr>')
    H.append('</tbody></table>')

    H.append("</body></html>")

    with open(OUT_HTML, "w") as f:
        f.write("".join(H))
    print(f"Report written to {OUT_HTML}  ({n_runs} runs loaded)")


if __name__ == "__main__":
    build_report()

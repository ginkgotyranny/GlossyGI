#!/usr/bin/env python3
"""
生成 ptdn / ptnrd / ncr 三算法对比报告（中文版）。

读取 $BENCH_RESULTS_DIR (默认 <repo>/perf_mem_test/results) 下最新的
benchmark_<algo>_*.csv、power_<algo>_*.csv 和 benchmark_<algo>_*.meta.json，
生成 HTML 对比报告。

输出: <results_dir>/compare_report.html
包含:
  - 测试方法说明（场景、时长、相机轨迹、采样方式）
  - 性能对比（FPS 柱状图 + 表格）
  - 内存对比（RSS/PSS/VRAM/Total 柱状图 + 表格）
  - 实测 vs 代码预估内存对比表
  - Falcor SceneStats 精确内存分解表
"""

import os
import sys
import glob
import json
import datetime

ALGOS = ["ptdn", "ptnrd", "ncr"]
ALGO_COLORS = {"ptdn": "#4ecdc4", "ptnrd": "#f7b731", "ncr": "#ff6b6b"}
RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR", os.path.join(os.environ.get("BENCH_REPO_ROOT", os.getcwd()), "perf_mem_test", "results"))

# ---- 代码预估内存 (MB) -------------------------------------------------------
# 基于 1280x720 分辨率，从源码分析各 render graph 的 pass 配置、缓冲分配、纹理格式。
# 分辨率: 1280x720 = 921,600 px; RGBA32F = 16 B/px = 14.7 MB/纹理。
#
# 重要：实测发现 Falcor 第一帧渲染时 VRAM 暴涨 ~1GB，原因是 Slang shader 编译 →
# SPIR-V → Vulkan PSO 创建。这部分开销不在 SceneStats 追踪范围内，但被
# nvidia-smi 计入进程 VRAM。预估表已加入 shader 编译开销。

MEMORY_ESTIMATES = {
    "ptdn": {
        "common": {
            "场景数据 (SceneStats: 纹理89 + BLAS 21 + 顶点12 + 索引3 + 其他)": 126,
            "Vulkan 框架 + 插件 + 交换链": 52,
            "Shader 编译产物 (Slang→SPIR-V→PSO, 第一帧分配)": 670,
            "CUDA Context + OptiX Context (OptiX denoiser 需要)": 100,
        },
        "algo_specific": {
            "GBufferRT (vbuffer, viewW, mvec, normW, posW, linearZ)": 70,
            "PathTracer 输出 (color, albedo, emission, reflectance)": 73,
            "AccumulatePass (累积颜色 + 权重)": 18,
            "OptixDenoiser (内部状态 + 输出 + staging buffers)": 75,
            "ToneMapper 输出": 15,
        },
        "estimated_total": 1199,
        "notes": "1spp + OptiX HDR 降噪器。移动相机下 AccumulatePass 每帧重置。"
                 "Shader 编译开销为 Falcor 框架层面，三个算法共享但各自独立编译。",
    },
    "ptnrd": {
        "common": {
            "场景数据 (SceneStats: 纹理89 + BLAS 21 + 顶点12 + 索引3 + 其他)": 126,
            "Vulkan 框架 + 插件 + 交换链": 52,
            "Shader 编译产物 (Slang→SPIR-V→PSO, 第一帧分配)": 670,
            "CUDA Context + OptiX Context": 100,
        },
        "algo_specific": {
            "GBufferRT (vbuffer, viewW, mvec, mvecW, normW, posW, linearZ)": 85,
            "PathTracer NRD 输出 (diffuse/specular radiance+hitdist, delta refl/trans, residual, reflectance, emission)": 220,
            "NRD ReblurDiffuseSpecular (历史缓冲 + 内部纹理)": 140,
            "NRD RelaxDiffuse x2 (delta reflection + transmission)": 220,
            "NRD SpecularReflectionMv + SpecularDeltaMv": 70,
            "AccumulatePass (参考路径)": 18,
            "ModulateIllumination (合成器)": 15,
            "2x ToneMapper": 30,
        },
        "estimated_total": 1746,
        "notes": "1spp + 5 个 NRD 降噪器实例 (Reblur + 2x Relax + 2x 运动向量)。"
                 "NRD 的历史缓冲和内部纹理比 OptiX denoiser 多用 ~300MB。",
    },
    "ncr": {
        "common": {
            "场景数据 (SceneStats: 纹理89 + BLAS 21 + 顶点12 + 索引3 + 其他)": 126,
            "Vulkan 框架 + 插件 + 交换链": 52,
            "Shader 编译产物 (Slang→SPIR-V→PSO, 第一帧分配)": 670,
            "CUDA Context (NeuralRadiosity 需要, 含 tiny-cuda-nn JIT)": 150,
        },
        "algo_specific": {
            "NCR 哈希网格 (主网格 6层x2特征 + 聚类网格 4簇x6层x4特征, log2HashMapSize=18)": 124,
            "NCR 优化器状态 (EMA+Adam, 3x 参数量)": 125,
            "NCR MLP 权重 (diffuse 11K + specular 18K 参数)": 1,
            "NCR 渲染批次缓冲 (逐像素结构化缓冲, 1280x720)": 333,
            "NCR 训练批次缓冲 (LHS 32K + RHS 262K 元素)": 152,
            "NCR 帧临时纹理 (2x RGBA32F)": 29,
            "NCR 内部路径追踪输出": 30,
            "ToneMapper 输出": 15,
        },
        "estimated_total": 1807,
        "notes": "自包含 pass (自带 GBuffer + 路径追踪 + 神经推理)。"
                 "哈希网格: 6 层, 2 特征/层, log2HashMapSize=18。MLP: 64 隐藏单元, 3-4 层。"
                 "在线训练增加优化器状态 + 训练批次缓冲。"
                 "CUDA Context 比 ptdn/ptnrd 大 ~50MB (tiny-cuda-nn JIT 编译缓存)。",
    },
}

# ---- CSV 解析 ----------------------------------------------------------------

def find_latest_csv(prefix, algo):
    pattern = os.path.join(RESULTS_DIR, f"{prefix}_{algo}_*.csv")
    csvs = sorted(glob.glob(pattern))
    return csvs[-1] if csvs else None

def find_latest_meta(algo):
    pattern = os.path.join(RESULTS_DIR, f"benchmark_{algo}_*.meta.json")
    metas = sorted(glob.glob(pattern))
    return metas[-1] if metas else None

def parse_benchmark_csv(path):
    d = {"frames": [], "wall_ms": [], "fps": [], "rss": [], "pss": [], "vram": [], "traj_t": []}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            d["frames"].append(int(parts[0]))
            d["wall_ms"].append(float(parts[1]))
            d["fps"].append(float(parts[2]))
            d["rss"].append(float(parts[3]))
            d["pss"].append(float(parts[4]) if len(parts) > 5 and parts[4] else -1.0)
            d["vram"].append(float(parts[5]) if len(parts) > 6 and parts[5] else -1.0)
            d["traj_t"].append(float(parts[6]) if len(parts) > 7 and parts[6] else float(parts[4]) if len(parts) > 4 else 0.0)
    return d

def parse_power_csv(path):
    d = {"power_w": [], "temp_c": [], "util_pct": []}
    if not path:
        return d
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            try:
                if len(parts) >= 2:
                    d["power_w"].append(float(parts[1].strip()))
                if len(parts) >= 3:
                    d["temp_c"].append(float(parts[2].strip()))
                if len(parts) >= 4:
                    d["util_pct"].append(float(parts[3].strip()))
            except ValueError:
                continue
    return d

# ---- 统计 --------------------------------------------------------------------

def avg(lst):
    valid = [x for x in lst if x > 0]
    return sum(valid) / len(valid) if valid else 0

def percentile(sorted_lst, p):
    if not sorted_lst:
        return 0
    idx = int(len(sorted_lst) * p)
    if idx >= len(sorted_lst):
        idx = len(sorted_lst) - 1
    return sorted_lst[idx]

def summarize(bench, power):
    wall = [x for x in bench["wall_ms"] if x > 0]
    rss = [x for x in bench["rss"] if x > 0]
    pss = [x for x in bench["pss"] if x > 0]
    vram = [x for x in bench["vram"] if x > 0]
    sorted_wall = sorted(wall)
    n = len(sorted_wall)
    s = {
        "frames": len(bench["wall_ms"]),
        "avg_fps": 1000.0 / avg(wall) if avg(wall) > 0 else 0,
        "avg_frame_ms": avg(wall),
        "p50_ms": percentile(sorted_wall, 0.5) if n else 0,
        "p95_ms": percentile(sorted_wall, 0.95) if n else 0,
        "p99_ms": percentile(sorted_wall, 0.99) if n else 0,
        "base_rss": rss[0] if rss else 0,
        "avg_rss": avg(rss),
        "peak_rss": max(rss) if rss else 0,
        "base_pss": pss[0] if pss else 0,
        "avg_pss": avg(pss),
        "peak_pss": max(pss) if pss else 0,
        "base_vram": vram[0] if vram else 0,
        "avg_vram": avg(vram),
        "peak_vram": max(vram) if vram else 0,
        "avg_power": avg(power["power_w"]),
        "peak_power": max(power["power_w"]) if power["power_w"] else 0,
        "avg_temp": avg(power["temp_c"]),
        "avg_util": avg(power["util_pct"]),
    }
    s["p50_fps"] = 1000.0 / s["p50_ms"] if s["p50_ms"] > 0 else 0
    s["p95_fps"] = 1000.0 / s["p95_ms"] if s["p95_ms"] > 0 else 0
    s["avg_total"] = s["avg_pss"] + s["avg_vram"]
    s["peak_total"] = s["peak_pss"] + s["peak_vram"]
    return s

# ---- SVG 柱状图 --------------------------------------------------------------

def bar_chart(title, labels, values, unit, color_map, max_val=None):
    n = len(labels)
    if n == 0 or max(values) == 0:
        return f"<p>{title}: 无数据</p>"

    W = 480
    H = 260
    margin_l = 50
    margin_r = 20
    margin_t = 40
    margin_b = 50
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    if max_val is None:
        max_val = max(values) * 1.15
    if max_val <= 0:
        max_val = 1

    bar_w = plot_w / (n * 1.5)
    gap = bar_w * 0.5

    bars = ""
    for i in range(n):
        x = margin_l + i * (bar_w + gap) + gap / 2
        h = (values[i] / max_val) * plot_h
        y = margin_t + plot_h - h
        color = color_map.get(labels[i], "#888")
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="3"/>'
        bars += f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" fill="#e0e0e0" font-size="12" text-anchor="middle">{values[i]:.1f}</text>'
        bars += f'<text x="{x+bar_w/2:.1f}" y="{margin_t+plot_h+18:.1f}" fill="#888" font-size="11" text-anchor="middle">{labels[i]}</text>'

    ticks = ""
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        v = max_val * frac
        ty = margin_t + plot_h - frac * plot_h
        ticks += f'<line x1="{margin_l}" y1="{ty:.1f}" x2="{margin_l+plot_w}" y2="{ty:.1f}" stroke="#333" stroke-width="0.5"/>'
        ticks += f'<text x="{margin_l-5}" y="{ty+4:.1f}" fill="#666" font-size="10" text-anchor="end">{v:.0f}</text>'

    svg = f'''<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;font-family:monospace;display:inline-block;margin:10px">
  <text x="{margin_l}" y="25" fill="#e0e0e0" font-size="13" font-weight="bold">{title}</text>
  <rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#0f0f23" stroke="#333"/>
  {ticks}
  {bars}
</svg>'''
    return svg

# ---- 测试方法说明 ------------------------------------------------------------

def build_methodology_html(metas):
    meta = None
    for a in ALGOS:
        if a in metas and metas[a]:
            meta = metas[a]
            break
    if not meta:
        return "<p>无元数据</p>"

    waypoints = meta.get("waypoints", [])
    wp_rows = ""
    for i, wp in enumerate(waypoints):
        wp_rows += (
            f"<tr>"
            f'<td style="padding:3px 10px;color:#888">{i}</td>'
            f'<td style="padding:3px 10px;color:#ccc">({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})</td>'
            f'<td style="padding:3px 10px;color:#ccc">({wp[3]:.2f}, {wp[4]:.2f}, {wp[5]:.2f})</td>'
            f'<td style="padding:3px 10px;color:#ccc">{wp[6]:.1f}s</td>'
            f"</tr>"
        )

    html = f'''
<h3 style="color:#e0e0e0">测试方法</h3>
<table style="border-collapse:collapse;margin:10px 0">
<tr><td style="padding:4px 20px;color:#888">场景</td><td style="color:#e0e0e0">bathroom (scene-animation.pyscene, 851 个网格, 591K 三角形)</td></tr>
<tr><td style="padding:4px 20px;color:#888">分辨率</td><td style="color:#e0e0e0">{meta.get("resolution", "1280x720")}</td></tr>
<tr><td style="padding:4px 20px;color:#888">每个算法测试时长</td><td style="color:#e0e0e0">{meta.get("duration_s", "?")} 秒</td></tr>
<tr><td style="padding:4px 20px;color:#888">预热帧数</td><td style="color:#e0e0e0">{meta.get("warmup_frames", 30)} 帧（相机移动中，不记录数据；用于 shader 编译和 GPU 预热）</td></tr>
<tr><td style="padding:4px 20px;color:#888">相机轨迹</td><td style="color:#e0e0e0">Catmull-Rom 样条插值，{meta.get("waypoint_count", 10)} 个航点，一圈 ≈ {meta.get("one_loop_s", 0):.1f}s，段间时间={meta.get("segment_time_s", 2.0)}s</td></tr>
<tr><td style="padding:4px 20px;color:#888">相机动画</td><td style="color:#e0e0e0">已禁用 (cam.animated=False)，由 sceneUpdateCallback 逐帧驱动</td></tr>
<tr><td style="padding:4px 20px;color:#888">帧率测量</td><td style="color:#e0e0e0">连续两次 sceneUpdateCallback 调用之间的挂钟时间间隔（= 逐帧间隔）</td></tr>
<tr><td style="padding:4px 20px;color:#888">RSS (CPU)</td><td style="color:#e0e0e0">/proc/self/status VmRSS — 进程驻留内存（含共享库），逐帧采样</td></tr>
<tr><td style="padding:4px 20px;color:#888">PSS (CPU)</td><td style="color:#e0e0e0">/proc/self/smaps_rollup Pss — 比例分摊内存（私有 + 共享库按比例分摊），逐帧采样</td></tr>
<tr><td style="padding:4px 20px;color:#888">VRAM (GPU)</td><td style="color:#e0e0e0">nvidia-smi --query-compute-apps — 进程级 GPU 显存，后台线程每 2 秒采样</td></tr>
<tr><td style="padding:4px 20px;color:#888">GPU 功耗</td><td style="color:#e0e0e0">nvidia-smi --query-gpu power.draw — 后台进程每 500ms 采样</td></tr>
</table>

<h4 style="color:#ccc;margin-top:15px">相机航点（位置, 朝向, 停留时间）</h4>
<table style="border-collapse:collapse;margin:5px 0;font-size:12px">
<tr><th style="padding:3px 10px;color:#888;border-bottom:1px solid #444;text-align:left">#</th>
<th style="padding:3px 10px;color:#888;border-bottom:1px solid #444;text-align:left">位置 (x, y, z)</th>
<th style="padding:3px 10px;color:#888;border-bottom:1px solid #444;text-align:left">朝向 (x, y, z)</th>
<th style="padding:3px 10px;color:#888;border-bottom:1px solid #444;text-align:left">停留</th></tr>
{wp_rows}
</table>
<p style="color:#888;font-size:11px;margin-top:8px">注：相机移动导致 AccumulatePass 每帧重置（CameraPropertiesChanged 标志）。
这测的是相机移动下的稳态渲染性能，是对比 ptdn/ptnrd（逐帧降噪）与 ncr（无时序累积的神经推理）的正确条件。</p>
'''
    return html

# ---- SceneStats 精确内存分解 -------------------------------------------------

def build_scene_stats_html(metas):
    """从 meta.json 读取 SceneStats，生成精确内存分解表。"""
    html = '''
<h3 style="color:#e0e0e0">Falcor SceneStats 精确内存分解</h3>
<p style="color:#888;font-size:12px">SceneStats 是 Falcor 框架在运行时统计的场景数据内存分配，由 m.scene.stats 返回。
注意：SceneStats 不包含 shader 编译产物、Vulkan PSO、CUDA context、render graph 纹理等非场景资源。</p>
'''
    has_stats = False
    for algo in ALGOS:
        meta = metas.get(algo)
        if not meta or not meta.get("scene_stats"):
            continue
        ss = meta["scene_stats"]
        if not ss.get("totalMemoryMB"):
            continue
        has_stats = True
        color = ALGO_COLORS[algo]
        html += f'''
<h4 style="color:{color}">{algo}</h4>
<table style="border-collapse:collapse;margin:5px 0;font-size:12px">
<tr><td style="padding:3px 15px;color:#888">场景总内存</td><td style="padding:3px 15px;color:{color};text-align:right">{ss["totalMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">网格数</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss.get("meshCount", 0)}</td></tr>
<tr><td style="padding:3px 15px;color:#888">三角形数</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss.get("uniqueTriangleCount", 0):,}</td></tr>
<tr><td style="padding:3px 15px;color:#888">顶点数</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss.get("uniqueVertexCount", 0):,}</td></tr>
<tr><td style="padding:3px 15px;color:#888">材质纹理 ({ss.get("textureCount", 0)} 张, {ss.get("textureCompressedCount", 0)} 张压缩)</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["textureMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">BLAS 加速结构</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["blasMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">BLAS scratch</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["blasScratchMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">顶点缓冲</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["vertexMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">索引缓冲</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["indexMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">几何数据</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["geometryMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">动画数据</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["animationMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">TLAS</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["tlasMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">材质数据</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["materialMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">发光光源</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["emissiveMemoryMB"]:.1f} MB</td></tr>
<tr><td style="padding:3px 15px;color:#888">环境贴图</td><td style="padding:3px 15px;color:#ccc;text-align:right">{ss["envMapMemoryMB"]:.1f} MB</td></tr>
</table>
'''
    if not has_stats:
        html += "<p style='color:#888'>无 SceneStats 数据（需要重新运行基准测试）</p>"

    html += '''
<p style="color:#888;font-size:11px;margin-top:10px">
VRAM 差异分析：实测 VRAM (nvidia-smi) = SceneStats (126 MB) + Render Graph 纹理 (~250 MB) +
Shader 编译产物 + Vulkan PSO (~670 MB) + CUDA/OptiX Context (~100 MB)。
Shader 编译产物是最大单项，在第一帧渲染时一次性分配，后续保持不变。</p>
'''
    return html

# ---- 实测 vs 预估对比表 ------------------------------------------------------

def build_memory_estimate_html(results):
    rows = ""
    for algo in ALGOS:
        if algo not in results:
            continue
        est = MEMORY_ESTIMATES.get(algo, {})
        actual = results[algo]["stats"]
        color = ALGO_COLORS[algo]

        common_items = est.get("common", {})
        spec_items = est.get("algo_specific", {})
        est_total = est.get("estimated_total", 0)
        notes = est.get("notes", "")

        actual_vram = actual["avg_vram"]
        actual_total = actual["avg_pss"] + actual["avg_vram"] if actual["avg_vram"] > 0 else actual["avg_pss"]
        diff = actual_vram - est_total
        diff_pct = (diff / est_total * 100) if est_total > 0 else 0

        breakdown = ""
        for name, mb in common_items.items():
            breakdown += f'<tr><td style="padding:2px 10px;color:#666;font-size:11px">&nbsp;&nbsp;{name}</td><td style="padding:2px 10px;color:#666;font-size:11px;text-align:right">{mb}</td></tr>'
        breakdown += f'<tr><td style="padding:2px 10px;color:#888;font-size:11px;font-weight:bold">算法特定:</td><td></td></tr>'
        for name, mb in spec_items.items():
            breakdown += f'<tr><td style="padding:2px 10px;color:#aaa;font-size:11px">&nbsp;&nbsp;{name}</td><td style="padding:2px 10px;color:#aaa;font-size:11px;text-align:right">{mb}</td></tr>'

        diff_color = "#4ecdc4" if abs(diff_pct) < 15 else ("#f7b731" if abs(diff_pct) < 30 else "#ff6b6b")

        rows += f'''
<tr style="border-bottom:2px solid #444">
  <td style="padding:8px 15px;color:{color};font-weight:bold;font-size:14px" rowspan="{len(common_items)+len(spec_items)+3}">{algo}</td>
  <td style="padding:2px 10px;color:#888;font-size:11px;font-weight:bold">公共部分:</td>
  <td></td>
  <td style="padding:8px 15px;color:{color};text-align:right" rowspan="{len(common_items)+len(spec_items)+3}">{actual_vram:.0f} MB</td>
  <td style="padding:8px 15px;color:{color};text-align:right" rowspan="{len(common_items)+len(spec_items)+3}">{est_total} MB</td>
  <td style="padding:8px 15px;color:{diff_color};text-align:right" rowspan="{len(common_items)+len(spec_items)+3}">{diff:+.0f} MB ({diff_pct:+.1f}%)</td>
</tr>
{breakdown}
<tr><td style="padding:4px 10px;color:#888;font-size:11px;font-weight:bold">预估总计</td><td style="padding:4px 10px;color:#888;text-align:right">{est_total}</td></tr>
<tr><td colspan="2" style="padding:4px 10px;color:#666;font-size:11px;font-style:italic">{notes}</td></tr>
'''

    html = f'''
<h3 style="color:#e0e0e0">实测 vs 代码预估内存对比</h3>
<p style="color:#888;font-size:12px">预估基于 1280x720 分辨率，从 render graph 源码分析各 pass 的缓冲分配、纹理格式、CUDA 分配。
实测 = nvidia-smi 报告的进程 VRAM 均值。
差异主要来自 shader 编译产物（第一帧一次性分配 ~670MB，已纳入预估）和驱动内部开销。</p>
<table style="border-collapse:collapse;margin:10px 0;font-size:12px">
<tr>
  <th style="padding:6px 15px;color:#888;border-bottom:2px solid #555;text-align:left">算法</th>
  <th style="padding:6px 15px;color:#888;border-bottom:2px solid #555;text-align:left">组件</th>
  <th style="padding:6px 15px;color:#888;border-bottom:2px solid #555;text-align:right">预估 MB</th>
  <th style="padding:6px 15px;color:#888;border-bottom:2px solid #555;text-align:right">实测 VRAM</th>
  <th style="padding:6px 15px;color:#888;border-bottom:2px solid #555;text-align:right">预估总计</th>
  <th style="padding:6px 15px;color:#888;border-bottom:2px solid #555;text-align:right">差异</th>
</tr>
{rows}
</table>
<p style="color:#888;font-size:11px">差异 = 实测 - 预估。绿色 &lt;15%, 黄色 &lt;30%, 红色 &gt;30%。</p>
'''
    return html

# ---- 主流程 ------------------------------------------------------------------

def main():
    results = {}
    metas = {}
    for algo in ALGOS:
        bench_csv = find_latest_csv("benchmark", algo)
        power_csv = find_latest_csv("power", algo)
        meta_path = find_latest_meta(algo)
        if not bench_csv:
            print(f"警告: 未找到 {algo} 的 benchmark CSV，跳过")
            continue
        bench = parse_benchmark_csv(bench_csv)
        power = parse_power_csv(power_csv)
        meta = None
        if meta_path:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except:
                pass
        results[algo] = {
            "bench_csv": os.path.basename(bench_csv),
            "power_csv": os.path.basename(power_csv) if power_csv else None,
            "stats": summarize(bench, power),
            "bench": bench,
            "power": power,
        }
        metas[algo] = meta
        s = results[algo]["stats"]
        print(f"{algo}: {s['avg_fps']:.1f} FPS, PSS={s['avg_pss']:.0f} MB, VRAM={s['avg_vram']:.0f} MB, 总计={s['avg_total']:.0f} MB, 功耗={s['avg_power']:.1f} W")

    if not results:
        print("错误: 未找到任何基准数据")
        sys.exit(1)

    algos_present = [a for a in ALGOS if a in results]
    labels = algos_present

    fps_chart = bar_chart("平均 FPS", labels,
                          [results[a]["stats"]["avg_fps"] for a in labels], "FPS", ALGO_COLORS)
    p95_chart = bar_chart("p95 FPS", labels,
                          [results[a]["stats"]["p95_fps"] for a in labels], "FPS", ALGO_COLORS)
    pss_chart = bar_chart("平均 PSS — CPU 内存 (MB)", labels,
                          [results[a]["stats"]["avg_pss"] for a in labels], "MB", ALGO_COLORS)
    vram_chart = bar_chart("平均 VRAM — GPU 显存 (MB)", labels,
                           [results[a]["stats"]["avg_vram"] for a in labels], "MB", ALGO_COLORS)
    total_chart = bar_chart("平均总内存 PSS+VRAM (MB)", labels,
                            [results[a]["stats"]["avg_total"] for a in labels], "MB", ALGO_COLORS)
    power_chart = bar_chart("平均 GPU 功耗 (W)", labels,
                            [results[a]["stats"]["avg_power"] for a in labels], "W", ALGO_COLORS)

    # 性能 + 内存汇总表
    table_rows = ""
    metrics = [
        ("测量帧数", "frames", "", ".0f"),
        ("平均 FPS", "avg_fps", "", ".2f"),
        ("p50 FPS", "p50_fps", "", ".2f"),
        ("p95 FPS", "p95_fps", "", ".2f"),
        ("平均帧时间", "avg_frame_ms", "ms", ".2f"),
        ("p95 帧时间", "p95_ms", "ms", ".2f"),
        ("平均 RSS (CPU)", "avg_rss", "MB", ".1f"),
        ("平均 PSS (CPU)", "avg_pss", "MB", ".1f"),
        ("平均 VRAM (GPU)", "avg_vram", "MB", ".1f"),
        ("峰值 VRAM", "peak_vram", "MB", ".1f"),
        ("平均总内存 (PSS+VRAM)", "avg_total", "MB", ".1f"),
        ("峰值总内存", "peak_total", "MB", ".1f"),
        ("平均 GPU 功耗", "avg_power", "W", ".1f"),
        ("峰值 GPU 功耗", "peak_power", "W", ".1f"),
        ("平均 GPU 温度", "avg_temp", "°C", ".1f"),
    ]
    for label, key, unit, fmt in metrics:
        row = f'<tr><td style="padding:6px 20px;color:#888;border-bottom:1px solid #333">{label}</td>'
        for a in algos_present:
            val = results[a]["stats"].get(key, 0)
            color = ALGO_COLORS.get(a, "#e0e0e0")
            row += f'<td style="padding:6px 20px;color:{color};border-bottom:1px solid #333;text-align:right">{format(val, fmt)} {unit}</td>'
        row += "</tr>"
        table_rows += row

    # NCR 提升分析
    improvement_html = ""
    if "ncr" in results and ("ptdn" in results or "ptnrd" in results):
        ncr_fps = results["ncr"]["stats"]["avg_fps"]
        improvements = []
        for base in ["ptdn", "ptnrd"]:
            if base in results:
                base_fps = results[base]["stats"]["avg_fps"]
                if base_fps > 0:
                    pct = (ncr_fps - base_fps) / base_fps * 100
                    improvements.append(f"vs {base}: <span style='color:{'#4ecdc4' if pct>=0 else '#ff6b6b'}'>{pct:+.1f}%</span>")
        ncr_total = results["ncr"]["stats"]["avg_total"]
        mem_comparisons = []
        for base in ["ptdn", "ptnrd"]:
            if base in results:
                base_total = results[base]["stats"]["avg_total"]
                delta = ncr_total - base_total
                mem_comparisons.append(f"vs {base}: <span style='color:{'#4ecdc4' if delta<200 else '#ff6b6b'}'>{delta:+.0f} MB</span>")

        if improvements:
            improvement_html = f'''
<h3 style="color:#e0e0e0">NCR 与基线对比</h3>
<table style="border-collapse:collapse;margin:10px 0">
<tr><td style="padding:6px 20px;color:#888">FPS 提升</td><td style="font-family:monospace;color:#e0e0e0">{" &nbsp;|&nbsp; ".join(improvements)}</td></tr>
<tr><td style="padding:6px 20px;color:#888">内存变化 (总内存)</td><td style="font-family:monospace;color:#e0e0e0">{" &nbsp;|&nbsp; ".join(mem_comparisons)}</td></tr>
<tr><td style="padding:6px 20px;color:#888">目标</td><td style="font-family:monospace;color:#888;font-size:12px">FPS: ≥50% 提升 &nbsp;|&nbsp; 内存: 新增 &lt;200MB</td></tr>
</table>'''

    header_cells = "".join(
        f'<th style="padding:6px 20px;color:{ALGO_COLORS.get(a, "#e0e0e0")};border-bottom:2px solid #555">{a}</th>'
        for a in algos_present
    )

    methodology_html = build_methodology_html(metas)
    estimate_html = build_memory_estimate_html(results)
    scene_stats_html = build_scene_stats_html(metas)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f'''<!DOCTYPE html>
<html><head><title>算法对比报告</title>
<meta charset="utf-8">
</head>
<body style="background:#1a1a2e;color:#e0e0e0;font-family:monospace;margin:20px">
<h2>算法对比报告 — {", ".join(algos_present)} ({ts})</h2>

{methodology_html}

<h3 style="color:#e0e0e0">性能与内存汇总</h3>
<table style="border-collapse:collapse;margin:10px 0">
<tr><th style="padding:6px 20px;color:#888;border-bottom:2px solid #555;text-align:left">指标</th>{header_cells}</tr>
{table_rows}
</table>

{improvement_html}

<h3 style="color:#e0e0e0">图表</h3>
<div>
{fps_chart}
{p95_chart}
</div>
<div>
{pss_chart}
{vram_chart}
</div>
<div>
{total_chart}
{power_chart}
</div>

{estimate_html}

{scene_stats_html}

<h3 style="color:#e0e0e0">源文件</h3>
<table style="border-collapse:collapse;margin:10px 0;font-size:12px">
<tr><th style="padding:4px 12px;color:#888;text-align:left">算法</th><th style="padding:4px 12px;color:#888;text-align:left">Benchmark CSV</th><th style="padding:4px 12px;color:#888;text-align:left">功耗 CSV</th></tr>
'''
    for a in algos_present:
        r = results[a]
        html += f'<tr><td style="padding:4px 12px;color:{ALGO_COLORS[a]}">{a}</td><td style="padding:4px 12px;color:#888">{r["bench_csv"]}</td><td style="padding:4px 12px;color:#888">{r["power_csv"] or "(无)"}</td></tr>'
    html += '''</table>

</body></html>'''

    out_html = os.path.join(RESULTS_DIR, "compare_report.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"\n报告: {os.path.abspath(out_html)}")

if __name__ == "__main__":
    main()

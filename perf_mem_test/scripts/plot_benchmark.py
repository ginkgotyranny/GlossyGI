#!/usr/bin/env python3
"""
Parse benchmark CSV and generate an HTML report with FPS + RSS curves.

Usage:
    python3 perf_mem_test/scripts/plot_benchmark.py [csv_path]

If no path given, uses the most recent CSV in $BENCH_RESULTS_DIR
(default <repo>/perf_mem_test/results).
Output: <results_dir>/benchmark_report.html (open in browser)
"""

import os
import sys
import glob
import datetime

_REPO_ROOT = os.environ.get("BENCH_REPO_ROOT", os.getcwd())
_RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR",
                              os.path.join(_REPO_ROOT, "perf_mem_test", "results"))

def find_latest_csv():
    csvs = sorted(glob.glob(os.path.join(_RESULTS_DIR, "benchmark_*.csv")))
    if not csvs:
        print(f"No benchmark CSV found in {_RESULTS_DIR}")
        sys.exit(1)
    return csvs[-1]

def parse_csv(path):
    frames, wall_ms, fps, rss, traj_t = [], [], [], [], []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            frames.append(int(parts[0]))
            wall_ms.append(float(parts[1]))
            fps.append(float(parts[2]))
            rss.append(float(parts[3]))
            traj_t.append(float(parts[4]))
    return frames, wall_ms, fps, rss, traj_t

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

def generate_svg(frames, fps, rss, avg_fps, avg_rss, peak_rss, p50_fps, p95_fps, title):
    """Generate inline SVG with two subplots: FPS and RSS over time."""
    n = len(frames)
    if n == 0:
        return "<p>No data</p>"

    # SVG dimensions
    W = 1000
    H = 450
    margin_l = 60
    margin_r = 30
    margin_t = 40
    margin_b = 50
    plot_w = (W - margin_l - margin_r - 20) // 2
    plot_h = H - margin_t - margin_b

    # FPS plot range
    fps_max = max(fps) * 1.1 if max(fps) > 0 else 100
    fps_min = 0

    # RSS plot range
    rss_min = min(rss) * 0.95 if min(rss) > 0 else 0
    rss_max = max(rss) * 1.05 if max(rss) > 0 else 100

    def x_scale(i):
        return margin_l + (i / max(n - 1, 1)) * plot_w

    def y_fps(v):
        return margin_t + plot_h - ((v - fps_min) / (fps_max - fps_min + 0.001)) * plot_h

    def y_rss(v):
        return margin_t + plot_h - ((v - rss_min) / (rss_max - rss_min + 0.001)) * plot_h

    # FPS polyline
    fps_points = " ".join(f"{x_scale(i):.1f},{y_fps(fps[i]):.1f}" for i in range(n))
    # RSS polyline
    rss_x0 = margin_l + plot_w + 20
    rss_points = " ".join(f"{rss_x0 + (i / max(n-1,1)) * plot_w:.1f},{y_rss(rss[i]):.1f}" for i in range(n))

    # avg lines
    avg_fps_y = y_fps(avg_fps)
    avg_rss_y = y_rss(avg_rss)

    svg = f'''<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;font-family:monospace">
  <!-- FPS subplot -->
  <text x="{margin_l}" y="25" fill="#e0e0e0" font-size="14" font-weight="bold">FPS over time</text>
  <rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#0f0f23" stroke="#333"/>
  <!-- grid lines -->
  <line x1="{margin_l}" y1="{avg_fps_y:.1f}" x2="{margin_l+plot_w}" y2="{avg_fps_y:.1f}" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>
  <text x="{margin_l+plot_w-5}" y="{avg_fps_y-4:.1f}" fill="#ff6b6b" font-size="11" text-anchor="end">avg={avg_fps:.1f}</text>
  <!-- FPS curve -->
  <polyline points="{fps_points}" fill="none" stroke="#4ecdc4" stroke-width="1.5"/>
  <!-- axes labels -->
  <text x="{margin_l+plot_w//2}" y="{H-15}" fill="#888" font-size="11" text-anchor="middle">Frame</text>
  <text x="{margin_l-40}" y="{margin_t+plot_h//2}" fill="#888" font-size="11" text-anchor="middle" transform="rotate(-90 {margin_l-40} {margin_t+plot_h//2})">FPS</text>
  <!-- y-axis ticks -->
  <text x="{margin_l-5}" y="{margin_t+8}" fill="#666" font-size="10" text-anchor="end">{fps_max:.0f}</text>
  <text x="{margin_l-5}" y="{margin_t+plot_h}" fill="#666" font-size="10" text-anchor="end">{fps_min:.0f}</text>

  <!-- RSS subplot -->
  <text x="{rss_x0}" y="25" fill="#e0e0e0" font-size="14" font-weight="bold">RSS (MB) over time</text>
  <rect x="{rss_x0}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#0f0f23" stroke="#333"/>
  <!-- avg line -->
  <line x1="{rss_x0}" y1="{avg_rss_y:.1f}" x2="{rss_x0+plot_w}" y2="{avg_rss_y:.1f}" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>
  <text x="{rss_x0+plot_w-5}" y="{avg_rss_y-4:.1f}" fill="#ff6b6b" font-size="11" text-anchor="end">avg={avg_rss:.0f}MB</text>
  <!-- RSS curve -->
  <polyline points="{rss_points}" fill="none" stroke="#f7b731" stroke-width="1.5"/>
  <!-- axes labels -->
  <text x="{rss_x0+plot_w//2}" y="{H-15}" fill="#888" font-size="11" text-anchor="middle">Frame</text>
  <text x="{rss_x0-40}" y="{margin_t+plot_h//2}" fill="#888" font-size="11" text-anchor="middle" transform="rotate(-90 {rss_x0-40} {margin_t+plot_h//2})">MB</text>
  <text x="{rss_x0-5}" y="{margin_t+8}" fill="#666" font-size="10" text-anchor="end">{rss_max:.0f}</text>
  <text x="{rss_x0-5}" y="{margin_t+plot_h}" fill="#666" font-size="10" text-anchor="end">{rss_min:.0f}</text>
</svg>'''
    return svg

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_csv()
    print(f"Parsing: {csv_path}")

    frames, wall_ms, fps, rss, traj_t = parse_csv(csv_path)
    if not frames:
        print("CSV is empty")
        sys.exit(1)

    avg_fps = avg(fps)
    avg_rss = avg(rss)
    peak_rss = max(rss) if rss else 0
    base_rss = rss[0] if rss else 0
    sorted_fps = sorted([f for f in fps if f > 0])
    p50_fps = 1000.0 / percentile(sorted(sorted([w for w in wall_ms if w > 0])), 0.5) if sorted_fps else 0
    p95_fps = 1000.0 / percentile(sorted(sorted([w for w in wall_ms if w > 0])), 0.95) if sorted_fps else 0

    # extract algo name from filename
    basename = os.path.basename(csv_path)
    algo = "unknown"
    if "ptdn" in basename: algo = "ptdn"
    elif "ptnrd" in basename: algo = "ptnrd"
    elif "ncr" in basename: algo = "ncr"

    title = f"Benchmark: {algo} ({len(frames)} frames, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"

    svg = generate_svg(frames, fps, rss, avg_fps, avg_rss, peak_rss, p50_fps, p95_fps, title)

    html = f'''<!DOCTYPE html>
<html><head><title>Benchmark Report - {algo}</title></head>
<body style="background:#1a1a2e;color:#e0e0e0;font-family:monospace;margin:20px">
<h2>{title}</h2>
<table style="border-collapse:collapse;margin:10px 0">
<tr><td style="padding:4px 20px;color:#888">Avg FPS</td><td style="color:#4ecdc4;font-size:18px;font-weight:bold">{avg_fps:.2f}</td></tr>
<tr><td style="padding:4px 20px;color:#888">p50 FPS</td><td>{p50_fps:.2f}</td></tr>
<tr><td style="padding:4px 20px;color:#888">p95 FPS</td><td>{p95_fps:.2f}</td></tr>
<tr><td style="padding:4px 20px;color:#888">Avg RSS</td><td style="color:#f7b731;font-size:18px;font-weight:bold">{avg_rss:.1f} MB</td></tr>
<tr><td style="padding:4px 20px;color:#888">Peak RSS</td><td>{peak_rss:.1f} MB</td></tr>
<tr><td style="padding:4px 20px;color:#888">Base RSS</td><td>{base_rss:.1f} MB</td></tr>
<tr><td style="padding:4px 20px;color:#888">Delta</td><td>{peak_rss - base_rss:.1f} MB</td></tr>
</table>
{svg}
</body></html>'''

    out_html = os.path.join(os.path.dirname(csv_path), "benchmark_report.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"Report: {os.path.abspath(out_html)}")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print(f"  Avg RSS: {avg_rss:.1f} MB")
    print(f"  Peak RSS: {peak_rss:.1f} MB")

if __name__ == "__main__":
    main()

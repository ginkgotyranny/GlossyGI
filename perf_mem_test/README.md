# perf_mem_test/ — Falcor 性能 & 内存基准测试套件

7 场景 × 3 算法 × 2 轮 (RTXDI on/off) 端到端基准测试，一键执行。

## 快速开始

```bash
cd /data/wy/Falcor_part3_ver3/Falcor

# 一键全量测试（约 70 分钟）
./perf_mem_test/run_all_benchmarks.sh
```

执行完后查看报告：

```bash
# 找最新的结果目录
ls -dt perf_mem_test/results/*/
# 打开 HTML（用 Read 工具或浏览器；不要用 xdg-open，moz-proxy 会拦截 file://）
# 报告路径: perf_mem_test/results/<timestamp>/benchmark_7scene_report.html
```

## 目录结构

```
perf_mem_test/
├── run_all_benchmarks.sh      # ★ 总编排脚本（一键全量测试）
├── run_benchmark_round.sh     # 单轮 runner（7 场景 × 3 算法 × 1 轮 RTXDI 状态）
├── run_benchmark_subset.sh    # 子集 runner（只跑 BENCH_SCENES 指定的场景）
├── perf_mem_test.sh           # 单次测试（1 算法 × 1 场景，调试用）
├── scene_waypoints.txt        # 7 场景相机航点（Catmull-Rom 插值用）
├── README.md                  # 本文件
├── scripts/                   # 内部脚本（被 runner 调用，不直接执行）
│   ├── perf_mem_test.py       # Mogwai 注入脚本：warmup + 采集 FPS/内存 + 写 CSV + 退出
│   ├── plot_7scene_report.py  # 7 场景报告生成器（HTML+SVG，无网络依赖）
│   ├── plot_benchmark.py      # 单次测试报告生成器（旧式 benchmark_*.csv）
│   ├── record_waypoints.py    # 航点录制器（Mogwai 内 R 录制 / Q 停止）
│   ├── camera_trajectory.py   # Catmull-Rom 轨迹库（被 perf_mem_test.py 内联使用）
│   └── wait_gpu_free.sh       # GPU 空闲检测（避免共享卡 contention）
├── legacy/                    # 早期/调试脚本，不参与主流程
│   ├── compare_algos.sh       # 三算法对比 + GPU 功耗采样
│   ├── plot_compare.py        # 对比报告生成器
│   ├── benchmark.py           # 早期单次测试脚本（已废弃，用 perf_mem_test.py）
│   ├── fps_probe.py           # FPS 探针（调试 /proc stall 用）
│   ├── mem_trace.py           # 内存追踪（粗粒度）
│   ├── mem_trace_fine.py      # 内存追踪（细粒度）
│   └── mem_breakdown.py       # 内存分解报告
└── results/                   # 测试结果归档（按日期戳，不覆盖）
    └── <YYYY-MM-DD_HHMMSS>/   # 每次执行一个独立文件夹
        ├── bench_<scene>_<algo>_<rtxdi_off|on>.csv       # FPS 采样
        ├── bench_<scene>_<algo>_<rtxdi_off|on>_mem.csv   # 内存采样
        ├── bench_<scene>_<algo>_<rtxdi_off|on>.meta.json # 元数据
        ├── round1_rtxdi_off.log                          # round 1 完整日志
        ├── round2_rtxdi_on.log                           # round 2 完整日志
        ├── debug.log                                     # perf_mem_test.py 调试日志
        └── benchmark_7scene_report.html                  # 最终报告
```

## run_all_benchmarks.sh（推荐入口）

端到端完成：round 1 (RTXDI off) → 自动切换 useRTXDI → round 2 (RTXDI on) → 恢复
原 useRTXDI 值 → 生成 HTML 报告。**useRTXDI 是 runtime shader define，无需重编译 C++**。

```bash
./perf_mem_test/run_all_benchmarks.sh                # 全量，~70 min
```

环境变量（可选）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `BENCH_DURATION` | 60 | 测量时长（秒） |
| `BENCH_WARMUP_S` | 30 | 静置 warmup 时长（秒） |
| `DISPLAY` | `:10` | X display |
| `SKIP_ROUND2` | 0 | 1=只跑 round 1 |
| `SKIP_REPORT` | 0 | 1=跳过 HTML 生成 |
| `DRY_RUN` | 0 | 1=只打印计划，不启动 Mogwai（验证用） |

**RTXDI 恢复保证**：脚本用 `trap ... EXIT INT TERM` 在任何退出路径（正常完成、
错误、Ctrl-C）都恢复 `OptixDenoiser.py` 和 `PathTracerNRD_VK.py` 的原始
`useRTXDI` 值。

## 单轮 / 子集 / 单次测试（手动分步）

```bash
# 单轮（手动控制 RTXDI 切换）
./perf_mem_test/run_benchmark_round.sh 0   # RTXDI off
./perf_mem_test/run_benchmark_round.sh 1   # RTXDI on

# 子集（只跑指定场景）
BENCH_SCENES="cornell-box veach-ajar" ./perf_mem_test/run_benchmark_subset.sh 0

# 单次（1 算法 × 1 场景，调试用）
./perf_mem_test/perf_mem_test.sh ptdn 25 bathroom
./perf_mem_test/perf_mem_test.sh ncr 30 kitchen

# 手动生成报告（从指定结果目录）
python3 perf_mem_test/scripts/plot_7scene_report.py perf_mem_test/results/<timestamp>
# 或不指定路径，自动用最新的 results 子目录
python3 perf_mem_test/scripts/plot_7scene_report.py
```

## 测试方法（写入报告）

- **流程**：每个场景对每个算法独立运行一次 Mogwai，启动后静置 30s warmup，再沿
  航点轨迹运动 60s 采集数据后退出。两轮分别对应 useRTXDI=false / true。
- **相机轨迹**：`scene_waypoints.txt` 中各场景航点，Catmull-Rom 样条插值循环。
- **FPS 采集**：渲染回调每帧触发，但仅每 30ms 墙钟时间记录一次（FPS = 窗口内帧数
  / 窗口时长），采样密度与算法帧率解耦。汇总用调和平均 FPS = 总帧数 / 总时长。
- **内存采集**：后台线程每 2s 调用 `nvitop -1 --ascii --pid <PID>` 取 GPU-MEM
  (MiB) 和 %MEM（× MemTotal = RSS）；若 nvitop 找不到进程，回退到
  `nvidia-smi pmon -c 1 -s m`（compute + graphics 进程）和 `/proc/self/status`
  VmRSS。后台线程避免渲染循环内 fork 子进程导致 GPU stall。

## 前置条件

- Mogwai 已编译：`build/linux-gcc/bin/Release/Mogwai`
- 7 个场景齐备：`media/scenes/{bathroom,bathroom2,cornell-box,kitchen,
  living-room,teapot,veach-ajar}/`（teapot 用 `scene-v4.pbrt`，其余用
  `scene-animation.pyscene`）
- GPU 空闲（RTX 3090 共享环境，`wait_gpu_free` 会等待他人退出）
- `DISPLAY` 可用（默认 `:10`，可用 `DISPLAY=:0` 覆盖）
- `nvitop` 已安装（`/usr/bin/nvitop`，CLI 模式即可，不需要 Python 模块）

## 注意事项

- **不覆盖**：每次执行 `run_all_benchmarks.sh` 都新建 `results/<timestamp>/`
  目录，旧数据保留。如需清理用 `rm -rf perf_mem_test/results/<old_timestamp>/`。
- **共享 GPU**：`/data/qz/Falcor-stage3/` 的 Mogwai 可能占用 98% GPU 导致 FPS
  下降——`wait_gpu_free` 会阻塞等待，但不会无限等（默认 30min 超时）。
- **报告查看**：moz-proxy 会拦截 `file://` URL，不要用 `xdg-open` 打开 HTML。
  用 Read 工具读取，或复制到本地浏览器。
- **matplotlib 不可用**（无网络）：报告是纯 HTML+SVG，不依赖 matplotlib。

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of NVIDIA Falcor 8.0 (real-time rendering framework; C++/Slang/CMake; DX12 + Vulkan) carrying a research project on neural glossy global illumination. Everything up to commit `9dc819c1` ("Falcor 8.0") is untouched upstream; all commits after are project work. Active development is on `stage4dev` (`ncr` holds an earlier stage, `master` is upstream).

The research module is the `NeuralRadiosity` render pass: a two-lobe (diffuse + specular) neural radiosity cache whose networks are trained and evaluated through tiny-cuda-nn (hash-grid encodings + MLPs, half precision), plus a customized NRD denoiser pipeline. Two platforms are in use: the Linux box (Ubuntu 24.04, Ampere GPU, **Vulkan** backend, `linux-gcc` preset) and the local Windows laptop (RTX 5070 Ti Blackwell, CUDA 12.9, VS 2022, **D3D12** backend, `windows-vs2022` preset). `FALCOR_HAS_VULKAN` is `ON` on both platforms (upstream default, restored Nov 2026); on Windows pick the backend at launch with `--device-type=d3d12|vulkan` (both verified on this laptop, GPU index 0 is the RTX under either API, index 1 is the AMD iGPU). The working tree carries the Linux "stage 4" NeuralRadiosity pipeline imported verbatim from the non-git snapshot `../Falcor-stage3-new/` (Nov 2026) plus a small set of Windows adaptations found by building and running on D3D12; one tree now serves both platforms.

## Build (Linux)

```sh
git submodule update --init --recursive        # external/tiny-cuda-nn etc.
cmake --preset linux-gcc -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux-gcc
```

- Resulting app: `build/linux-gcc/bin/Release/Mogwai`.
- Requires the CUDA toolkit: `NeuralRadiosity` and tiny-cuda-nn compile `.cu` sources. The plugin's CUDA architecture is the cache variable `NCR_CUDA_ARCH` (default `86`, Ampere Linux box); configure this laptop with `-DNCR_CUDA_ARCH=120` (Blackwell), since sm_86 SASS cannot load on it.
- `-DFALCOR_SKIP_PACKMAN_PULL=ON` skips dependency download at configure time when `external/packman` is already populated.
- Root CMakeLists Linux workarounds that must not be undone: `-D_GLIBCXX_USE_CXX11_ABI=0` applied to C++ **and** CUDA, and `-U_FORTIFY_SOURCE` for nvcc (nvcc 12.0 bundled GCC vs glibc 2.34 fortified headers).
- NRD is linked on Linux against `libNRD.so.3.1.0` (see `external/CMakeLists.txt`).
## Build (Windows)

```bat
setup_vs2022.bat                                     :: submodules + packman + configure (VS 2022, -T host=x86,cuda=<CUDA dir>)
cmake --build build\windows-vs2022 --config Release
```

- Resulting app: `build\windows-vs2022\bin\Release\Mogwai.exe`. Shaders compile at runtime, so Slang-only edits need no rebuild.
- Two build trees exist on this laptop: `build\windows-vs2022` and `build\windows-vs2022-vk`, both Vulkan + D3D12 since the 2026-11-03 reconfigure (the -vk tree: full build takes ~5 min on 32 cores). `run.bat` takes `-d vulkan|d3d12`, `--gpu N`, and honours `NCR_BUILD_DIR=build\windows-vs2022-vk`. NRDPass calls `vk*` functions directly, so `find_package(Vulkan REQUIRED)` needs an import library. The Vulkan SDK that `VULKAN_SDK` pointed at (`E:\UE\ue5.5.4\vulkansdk_dl\sdk`) is gone since Nov 2026; instead `build\vulkan-implib\vulkan-1.lib` is generated from the driver's `C:\Windows\System32\vulkan-1.dll` (`dumpbin -exports` -> `.def` -> `lib -def:... -machine:x64`) and passed with `-DVulkan_LIBRARY=<that lib> -DVulkan_INCLUDE_DIR=<repo>/external/vulkan-headers/include`. Installing a Vulkan SDK and reconfiguring without those two flags also works.
- `setup_vs2022.bat` prefers `external/packman/cuda` if present; otherwise the system CUDA (12.9 here) is used via the toolset string.
- **Charset vs the precompiled header** (resolved Nov 2026): a PCH built under code page 936 fails with C2855 once the system ANSI code page changes (now 1252). Both trees have since rebuilt their PCH under 1252. If it recurs, delete `Source/Falcor/Falcor.dir/Release/cmake_pch.pch` in the build tree, or build with `MSYS_NO_PATHCONV=1 CL="-source-charset:.936 -execution-charset:.936"`.

## Run

Windows (`run.bat` mirrors `run.sh`'s script registry and scene resolution; `NCR_RUN_DRYRUN=1` prints the command instead of launching):

```bat
run.bat                            :: interactive
run.bat ncr bathroom anim          :: NeuralRadiosity, animated scene
run.bat ptdn cornell-box --width 1920 --height 1080
```

Linux:

```sh
./run.sh                       # interactive: pick script + scene
./run.sh ncr living-room       # NeuralRadiosity, static scene
./run.sh ncr living-room anim  # animated variant
./run.sh ptnrd kitchen         # PathTracer + NRD (Vulkan)
./run.sh ptdn cornell-box      # PathTracer + OptiX denoiser
```

`run.sh` is the unified launcher (script registry + scene resolution live there). It execs Mogwai with `LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16`, which is required (system libpng conflict) — keep that in any manual launch.

- Scenes: `media/scenes/<name>/scene-v4.pyscene` (static) or `scene-animation.pyscene` (animated). `scripts/python/pbrt2pyscene.py` converts pbrt-v4 scenes to pyscene.
- NeuralRadiosity render modes (pass UI or `renderMode` property: `train|onlineTrain|render|idle`): `Train` (offline; saves `<steps>.ckpt` into the working directory every `saveCkptInterval` steps, stops at `totalTrainSteps`), `Online Train` (**default**; per-frame train + render), `Render`/`Idle`. `loadModelPath` loads a checkpoint right after the model is created.
- Headless smoke test: `scripts/ncr_smoke.py` loads the scene itself (Mogwai runs `--script` before `--scene`), steps the paused clock, captures frames, exits. Driven by env vars: `NCR_SMOKE_SCENE`, `NCR_SMOKE_OUT`, `NCR_SMOKE_FRAMES=30,90`, `NCR_SMOKE_MODE`, `NCR_SMOKE_PROPS="{'enableTemporalSpecReuse': True}"`, `NCR_SMOKE_RES=1280x720`, `NCR_SMOKE_OUTPUTS=albedo,roughness`. Run with `Mogwai.exe --headless --script scripts/ncr_smoke.py`. A healthy frame has no pixels exactly equal to 1.0 (raw throughput); the `active` channel is R32Uint and exports as zeros, so don't use it as a probe.
- Benchmarks: `bench.sh` → `scripts/bench.py` (frame time + NVML VRAM around the `ncr`/`ptnrd` launchers) and `perf_mem_test/` (7 scenes × 3 algorithms with Catmull-Rom camera paths, HTML report; README is in Chinese). Both are Linux shell pipelines.

## Tests

From the repo root; requires packman python (`setup.sh` bootstraps `tools/.packman/python` if missing):

```sh
tests/run_unit_tests.sh      # C++ unit tests (FalcorTest); forwards args to testing/run_unit_tests.py
tests/run_unit_tests.sh -f "Sqrt" -c gpu -d vulkan   # filter regex / category all|cpu|gpu / device d3d12|vulkan
tests/run_image_tests.sh     # image comparison tests
tests/run_python_tests.sh    # python binding tests
```

Windows equivalents are `tests\run_unit_tests.bat`, `run_image_tests.bat`, `run_python_tests.bat` (same args; use `-d d3d12`). Unit tests are implemented in `Source/Tools/FalcorTest` (`CPU_TEST`/`GPU_TEST` macros); exit code = number of failed tests.

## Architecture

### Falcor layer (upstream)

- `Source/Falcor` — core framework (API/device abstraction, scene, materials, render graph).
- `Source/Mogwai` — the interactive renderer app that loads `--script` / `--scene`.
- `Source/RenderPasses/*` — render passes built as plugins (`add_plugin()`), registered via `FALCOR_PLUGIN_CLASS`, instantiated from Python as `createPass("Name")`.
- `scripts/*.py` — Python render-graph definitions: passes are created and wired with `g.addEdge("Pass.output", "Other.input")`, outputs marked with `g.markOutput(...)`. The trailing `try: m.addGraph(...) except NameError: None` makes scripts work inside Mogwai and headless.
- Shaders are Slang. Compute shaders in this project use the `.cs.slang` suffix; `target_copy_shaders` copies them into the output tree at build time.
- `tools/make_new_render_pass.bat <Name>` scaffolds a new pass plugin.
- `scripts/inv-rendering/` + `environment.yml` (conda env `falcor-pytorch`) belong to the upstream differentiable/inverse-rendering demos, not the neural GI research.

### Research module: `Source/RenderPasses/NeuralRadiosity`

One render pass orchestrating a GPU pipeline split between Slang compute shaders and CUDA/tiny-cuda-nn:

- `NeuralRadiosity.cpp/.h` — the pass. `render()`: FirstSmooth (primary rays through delta surfaces, writes G-buffer + `gPrimaryPos` + diff/spec masks) → PrefixSum + CompactPass → cone trace → CUDA inference → Resolve → PostFilter + FXAA. `train()`: RandomSmooth (LHS) → SampleRHS → RHS batch (compact, diffuse-only inference, resolveBatch) → LHS batch (compact, coneTraceBatch, resolveRHS, `modelTrainCUDA`). Batches live in `RayBatchBuffer`: render, train-LHS, train-RHS (RHS has no spec group, `needSpec=false`; RHS always queries the diffuse net), plus a 1-element `mpBindDummyBatch` so shaders declaring the full layout can still be bound. Only the four CUDA I/O buffers carry `ResourceBindFlags::Shared`.
- Hyperparameters: `NCRConfig.h` is the single source for C++, CUDA and Slang (CMake copies it next to the shaders). Checkpoints embed the element counts and refuse to load when they differ. `NCRConfig-stage4.h` is a kept alternative preset.
- Specular lobe, default render path (`coneTraceBatch`): `ConeTrace.cs.slang` runs one thread group per spec sample (`NUM_SPEC_RAYS` threads, T=32 render / 128 train from NCRConfig), each thread GGX-samples a ray; hit distances are clustered with integer-domain KMeans into 4 clusters written to `SpecNetInput` via `writeClusterOutput`. `SpecPrimary.cs.slang` (one thread per sample) then writes the position and env-color slots; with `enableMirrorAlbedo` (default on) a mirror ray replaces the env color by the hit surface's albedo (`MirrorAlbedo.slang`). The spec G-buffer is gathered on demand through `specSrc` (screen pixel index on the render path, pre-compact batch index on the training path).
- Optional render path, `enableTemporalSpecReuse=True` (default from `NCR_ENABLE_CLUSTER_REUSE`: **off** in `NCRConfig.h`, on in `NCRConfig-stage4.h`): `clusterReuse()` = `ClusterReproject` → `ConeTraceScreen` → `ClusterSpatialReuse` over a per-pixel packed `SpecClusterHistory` (ping-pong buffers, packed `ClusterLockState` change detector, previous-frame G-buffer copies saved by `savePrevGBuffer`). `enableClusterSpatialReuse` gates the neighbourhood gather. Training batches always use ConeTrace. Tunables are the `cluster*` properties; the pass also exposes `clearHistory()` and `trainHeat` (online LR/EMA schedule, `trainLr*`/`trainEma*` properties) to Python. Local schedule changes, not in the Linux snapshots (keep them when merging a snapshot): (a) while the camera is fixed, `updateTrainSchedule` grades scene changes instead of sending all of them to `trainSceneHeat`: one-off edits (materials, light parameters, environment, added/removed geometry) still use `trainSceneHeat`; moving lights (`LightsMoved`, or `LightCollectionChanged`, which Falcor raises only when an emissive instance's transform changes) use `trainLightMotionHeat` (0.7); moving non-emissive geometry alone uses `trainObjectMotionHeat` (0.2). While the camera moves, ver4's rule applies unchanged. (b) Events that keep the heat at its level no longer freeze the warmup countdown, so the warmup ends N frames after load even while a camera path plays. (c) `trainHeatLogInterval` (default 30, 0 = off) logs a `[TrainSchedule]` line every N training frames and whenever an event raises the heat: heat, hold/cooling, LR, EMA decay, and the events seen since the previous line (`camera`/`edit`/`light`/`object`).
- Render-batch network I/O is shared on Vulkan (`RayBatchBuffer::sharedIO`: specular rows first, diffuse rows after the padded specular rows, offsets from `counts` on the GPU and `diffInputOffset()`/`diffOutputOffset()` on the CPU).
- Env light: `preparePrefiltEnvCubemap` is still commented out in `execute()`, so the env-color input falls back to `gScene.envMap.eval`.
- Checkpoints (`Model.cu`, format "NRCK" v2): header with element counts, then 4 f32 master blocks and 4 f16 EMA inference blocks; load rebuilds the f16 working copies from the masters. Adam moments are not stored.
- **Verified D3D12 findings (Nov 2026)**: (1) an `out PathState` parameter is undefined on entry, so `generatePath` must start with `path = {}`; relying on the caller's `= {}` made DXIL fold the flag bits and `isActive()` read false, which zeroed every diff/spec mask (image = raw throughput, 56% of pixels exactly 1.0). Vulkan happened to keep the caller's zeros. (2) DXC cannot unroll a loop whose start depends on an outer induction variable; `ClusterHistory.slang`'s pair loop iterates the full range with `if (j <= i) continue`. (3) Linux loads plugins with `RTLD_LAZY`, so a declared-but-unimplemented method only fails on Windows (LNK2019); keep every `NRModel` method implemented. (4) `f32tof16` does not round to nearest even on D3D12. The cluster history re-packs t/sigma every frame, so the reuse path drifted ~2% darker than Vulkan (bathroom, fixed weights) once ver4 normalized `clsScale`; `ClusterHistory.slang` packs with an integer RNE conversion (`f32tof16Rne`), which reproduces Vulkan's images exactly. (5) Shared render-batch I/O binds one buffer as `RWStructuredBuffer<DiffNetInput>` and `<SpecNetInput>`. D3D12 views take the buffer's struct size, so `prepareResources` enables `sharedIO` on Vulkan only (on this RTX it happened to work on D3D12 too, which D3D12 does not guarantee). (6) MSVC builds with `/WX`: a local that hides a member (C4458) fails the Windows build while GCC accepts it.
- Training-sample generation: `randomSmooth`, `sampleRHS` (NEE + analytic lights + env light prefiltered cubemap), `resolveRHS`, `compactBatch`, `resolveBatch`.
- `Model/` — CUDA side. `NRModel` (Model.h/.cu) owns two hash grids (`HashGrid` primary encoding, `HashGridInterp` cluster encoding that interpolates between two grid levels chosen by cluster scale), `diffNet`/`specNet` FullyFusedMLPs (SquarePlus output), four EMA-wrapped Adam optimizers, and dual parameter sets: f32 masters for training vs f16 EMA copies (`custom_weights`) for inference (`precision_t = __half`, ModelTypes.h). Separate train/inference CUDA streams. `NeuralModel.cu` / `NeuralConeModel.cu` are compiled but unused.
- Interop: buffers are created with `ResourceBindFlags::Shared` and mapped through `Utils/CudaUtils.h` (`cuda_utils::CudaDevice`); `NetworkParams.slang` / `PathState.slang` define the shared GPU-side layouts — keep these in sync with the C++ struct layout when changing either side.
- Its CMakeLists globs sources (`GLOB_RECURSE`), so newly added files are only picked up after CMake re-runs (touch the CMakeLists or reconfigure).

### NRDPass changes (fork vs upstream)

Upstream NRDPass was D3D12-only. This fork adds a Vulkan backend (`createVulkanPipelines`/`dispatchVulkan`, 16 buffered frames of descriptor pools) and delta-lobe support, all guarded by `FALCOR_HAS_D3D12` / `FALCOR_HAS_VULKAN`. Consumed by `scripts/PathTracerNRD_VK.py` (path tracer + REBLUR/RELAX with separate delta reflection/transmission paths) as the denoised baseline for comparisons.

`OptixDenoiser` pass + `scripts/OptixDenoiser.py` provide the other comparison baseline (needs the OptiX SDK under `external/packman/optix`).

## Conventions

- Formatting: `.clang-format` at root (`tools/format_code.sh`); VS Code uses the clang-format from `tools/.packman/clang-format`.
- Upstream dev docs live in `docs/development/` (cmake.md, coding-conventions.md, error-handling.md, unit-testing.md); usage docs and tutorials under `docs/`. `AGENTS.md` (repo root) has the fuller naming/CMake convention summary.
- Project docs (Chinese): `docs/GLOSSY_GI_RENDER_PIPELINES.md` maps every `scripts/*.py` render graph to its pass chain with Mermaid diagrams; `docs/ncr/NCR管线说明文档.md` is a line-by-line walkthrough of the NeuralRadiosity Online-Train pipeline, with an interactive flowchart HTML and an offline hash-collision analysis script alongside.
- Do not start VS Code from Git Bash (breaks CMake's PATH) — see README.md.

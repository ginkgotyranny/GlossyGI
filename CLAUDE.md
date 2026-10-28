# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of NVIDIA Falcor 8.0 (real-time rendering framework; C++/Slang/CMake; DX12 + Vulkan) carrying a research project on neural glossy global illumination. Everything up to commit `9dc819c1` ("Falcor 8.0") is untouched upstream; all commits after are project work. Active development is on `stage4dev` (`ncr` holds an earlier stage, `master` is upstream).

The research module is the `NeuralRadiosity` render pass: a two-lobe (diffuse + specular) neural radiosity cache whose networks are trained and evaluated through tiny-cuda-nn (hash-grid encodings + MLPs, half precision), plus a customized NRD denoiser pipeline. The working platform is Ubuntu 24.04 with the **Vulkan** backend and the `linux-gcc` preset; Windows/D3D12 is secondary.

## Build (Linux)

```sh
git submodule update --init --recursive        # external/tiny-cuda-nn etc.
cmake --preset linux-gcc -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux-gcc
```

- Resulting app: `build/linux-gcc/bin/Release/Mogwai`.
- Requires the CUDA toolkit: `NeuralRadiosity` and tiny-cuda-nn compile `.cu` sources (NeuralRadiosity hardcodes `CUDA_ARCHITECTURES 86` in its CMakeLists.txt).
- `-DFALCOR_SKIP_PACKMAN_PULL=ON` skips dependency download at configure time when `external/packman` is already populated.
- Root CMakeLists Linux workarounds that must not be undone: `-D_GLIBCXX_USE_CXX11_ABI=0` applied to C++ **and** CUDA, and `-U_FORTIFY_SOURCE` for nvcc (nvcc 12.0 bundled GCC vs glibc 2.34 fortified headers).
- NRD is linked on Linux against `libNRD.so.3.1.0` (see `external/CMakeLists.txt`).
- Windows alternative: `setup_vs2022.bat` → `build/windows-vs2022`.

## Run

```sh
./run.sh                       # interactive: pick script + scene
./run.sh ncr living-room       # NeuralRadiosity, static scene
./run.sh ncr living-room anim  # animated variant
./run.sh ptnrd kitchen         # PathTracer + NRD (Vulkan)
./run.sh ptdn cornell-box      # PathTracer + OptiX denoiser
```

`run.sh` is the unified launcher (script registry + scene resolution live there). It execs Mogwai with `LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16`, which is required (system libpng conflict) — keep that in any manual launch.

- Scenes: `media/scenes/<name>/scene-v4.pyscene` (static) or `scene-animation.pyscene` (animated). `scripts/python/pbrt2pyscene.py` converts pbrt-v4 scenes to pyscene.
- NeuralRadiosity render modes (pass UI): `Train` (offline training; checkpoints saved to the working directory), `Online train` (per-frame updates, for animated scenes), `Render`/idle inference. Checkpoints are loaded through the pass UI — see NCR-README.md.

## Tests

From the repo root; requires packman python (`setup.sh` bootstraps `tools/.packman/python` if missing):

```sh
tests/run_unit_tests.sh      # C++ unit tests (FalcorTest); forwards args to testing/run_unit_tests.py
tests/run_image_tests.sh     # image comparison tests
tests/run_python_tests.sh    # python binding tests
```

Unit tests are implemented in `Source/Tools/FalcorTest`; exit code = number of failed tests.

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

- `NeuralRadiosity.cpp/.h` — the pass. Per frame it runs screen-space passes (FirstSmooth primary-hit smoothing, Compact stream compaction via `PrefixSum`, Resolve, PostFilter incl. FXAA) and ray-batch passes over `RayBatchBuffer` (shared interop structured buffers holding pos/dir/normal/albedo/roughness/vbuffer plus per-lobe compaction in/out). Three batches exist: render, train-LHS, train-RHS.
- Specular lobe handling: fires `NUM_SPEC_RAYS` rays per pixel (32 render / 128 train), clusters them with KMeans (`numClusters = 4`; 3 iters render / 10 train), feeds cluster features to the spec network. Temporal reuse strategies are switched by `ConeTraceReuseMode`.
- Training-sample generation: `randomSmooth`, `sampleRHS` (NEE + analytic lights + env light prefiltered cubemap), `resolveRHS`, `compactBatch`, `resolveBatch`.
- `Model/` — CUDA side. `NRModel` (Model.h/.cu) owns two hash grids (`HashGrid` primary encoding, `HashGridInterp` cluster encoding), `diffNet`/`specNet` MLPs, per-component optimizers, and dual parameter sets: full-precision training weights vs half-precision (`precision_t = __half`, ModelTypes.h) inference copies. Separate train/inference CUDA streams; online mode swaps EMA weights in. Custom tcnn models live in `NeuralModel.cu` / `NeuralConeModel.cu`.
- Interop: buffers are created with `ResourceBindFlags::Shared` and mapped through `Utils/CudaUtils.h` (`cuda_utils::CudaDevice`); `NetworkParams.slang` / `PathState.slang` define the shared GPU-side layouts — keep these in sync with the C++ struct layout when changing either side.
- Its CMakeLists globs sources (`GLOB_RECURSE`), so newly added files are only picked up after CMake re-runs (touch the CMakeLists or reconfigure).

### NRDPass changes (fork vs upstream)

Upstream NRDPass was D3D12-only. This fork adds a Vulkan backend (`createVulkanPipelines`/`dispatchVulkan`, 16 buffered frames of descriptor pools) and delta-lobe support, all guarded by `FALCOR_HAS_D3D12` / `FALCOR_HAS_VULKAN`. Consumed by `scripts/PathTracerNRD_VK.py` (path tracer + REBLUR/RELAX with separate delta reflection/transmission paths) as the denoised baseline for comparisons.

`OptixDenoiser` pass + `scripts/OptixDenoiser.py` provide the other comparison baseline (needs the OptiX SDK under `external/packman/optix`).

## Conventions

- Formatting: `.clang-format` at root (`tools/format_code.sh`); VS Code uses the clang-format from `tools/.packman/clang-format`.
- Upstream dev docs live in `docs/development/` (cmake.md, coding-conventions.md, error-handling.md, unit-testing.md); usage docs and tutorials under `docs/`.
- Do not start VS Code from Git Bash (breaks CMake's PATH) — see README.md.

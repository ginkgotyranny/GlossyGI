# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

Falcor is a real-time rendering framework supporting DirectX 12 and Vulkan, designed for rendering research and prototyping. This fork (branch `stage4dev`) extends Falcor with neural radiosity (`NeuralRadiosity`), NRD denoising integration, and CUDA/tiny-cuda-nn-based neural rendering passes.

## Build Commands

### Prerequisites
- Linux: Ubuntu 22.04+ with `sudo apt install xorg-dev libgtk-3-dev`
- Windows: Visual Studio 2022, Windows 10 SDK 10.0.19041.0
- CUDA (optional, required for NeuralRadiosity): CUDA 11.6.2+
- GPU with raytracing support (NVIDIA RTX)

### First-time setup
```bash
# Linux
./setup.sh                    # fetches git submodules + packman deps, patches NVTT

# Windows (VS2022 solution)
setup_vs2022.bat

# Windows (VS Code / Ninja)
setup.bat
```

### Configuring and building (Linux GCC, the primary dev workflow)
```bash
cmake --preset linux-gcc -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux-gcc --config Release
```

Other presets: `linux-clang`, `windows-vs2022`, `windows-ninja-msvc`. Use `cmake --list-presets` to see all. Each preset has a `-ci` variant (e.g. `linux-gcc-ci`) that enables `FALCOR_ENABLE_NRC`, header validation (`FALCOR_VALIDATE_HEADERS=ON`), and disables PCH.

CUDA is auto-detected by `check_language(CUDA)` in the root `CMakeLists.txt`; if `nvcc` is on `PATH`, `FALCOR_HAS_CUDA` is set `ON` and `find_package(CUDAToolkit REQUIRED)` runs. There is no `FALCOR_ENABLE_CUDA` toggle — to disable CUDA, remove it from `PATH`. Default `CMAKE_CUDA_ARCHITECTURES` is `75-virtual` (Turing); targets override via the `CUDA_ARCHITECTURES` property (NeuralRadiosity uses `86` for Ampere).

### Binary output
- Binaries: `build/<preset>/bin/<Config>/`
- Shaders: `build/<preset>/bin/<Config>/shaders/`
- Plugins: `build/<preset>/bin/<Config>/plugins/`
- The `Mogwai` executable is the main renderer application.

### Running the renderer
```bash
# NeuralRadiosity demo
./ncr.sh

# PathTracer + NRD (Vulkan)
./ptnrd.sh

# OptiX denoiser
./ptdn.sh
```
These scripts run `build/linux-gcc/bin/Release/Mogwai` with `LD_PRELOAD` for libpng16 and point at `media/scenes/living-room/scene-animation.pyscene`. Edit scene paths in the scripts as needed.

## Testing

### Unit tests (custom framework in `Source/Tools/FalcorTest`)
```bash
# Linux
./tests/run_unit_tests.sh                           # all tests
./tests/run_unit_tests.sh -f "Sqrt"                 # filter by regex
./tests/run_unit_tests.sh -c gpu                    # category: all|cpu|gpu
./tests/run_unit_tests.sh -d vulkan                 # device: d3d12|vulkan
```
Direct: `build/<preset>/bin/<Config>/FalcorTest [options]`. Return code = number of failed tests.

Tests live in `Source/Tools/FalcorTest/Tests/`. Use `CPU_TEST(Name)` / `GPU_TEST(Name)` macros from `UnitTest.h`. GPU tests get a `GPUUnitTestContext` named `ctx`. Skip with `CPU_TEST(Name, "reason")`.

### Image tests
```bash
./tests/run_image_tests.sh
```

### Python tests (require conda environment with `environment.yml`)
```bash
./tests/run_python_tests.sh
```

## Code Architecture

### Core library (`Source/Falcor/`)
- **Core/API/**: Low-level GPU abstraction over D3D12/Vulkan — `Device`, `RenderContext`, `Buffer`, `Texture`, `RenderContext`, raytracing acceleration structures. This is the hardware abstraction layer.
- **Core/Program/**: Shader compilation and reflection via Slang. `ProgramManager` handles program loading/caching; `ProgramReflection` provides parameter binding metadata.
- **Core/Pass/**: Base pass types — `ComputePass`, `FullScreenPass`, `RasterPass`, `BaseGraphicsPass`. Render passes build on these.
- **RenderGraph/**: The render graph system — `RenderGraph` (graph of passes), `RenderPass` (base class for all passes), `RenderGraphCompiler`, `ResourceCache`. Passes declare inputs/outputs via `RenderPassReflection`; the graph resolves resource dependencies and execution order.
- **Scene/**: Scene representation — `Scene`/`SceneBuilder` (the central scene object, ~200k lines), `Material/MaterialSystem`, `Lights`, `Camera`, `Animation`, `SDFs`, `Volume`. Host-side scene data is mirrored to GPU via Slang modules (`Scene.slang`, `MaterialSystem.slang`, etc.) that are imported by render pass shaders.
- **Rendering/**: Rendering subsystems shared across passes — `Lights/` (emissive samplers, env map sampler, light BVH), `Materials/` (BSDF implementations in Slang), `RTXDI/` (RTX Direct Illumination integration).
- **Utils/**: Math, sampling, scripting (Python bindings via pybind11), settings, logging, CUDA interop (`CudaUtils`).

### Render passes (`Source/RenderPasses/`)
Each render pass is a shared library plugin loaded at runtime. Passes inherit from `RenderPass` and register via the plugin system (`FALCOR_PLUGIN_CLASS` macro). Key passes:
- **GBuffer/VBuffer**: G-buffer generation (raytraced `VBufferRT`, rasterized `GBufferRT`).
- **PathTracer**: Reference path tracer with NRD output support. Produces diffuse/specular radiance-hitdist, reflectance, emission, and delta reflection/transmission paths.
- **NRDPass**: NVIDIA RTX Denoiser wrapper — wraps NRD SDK methods (ReblurDiffuseSpecular, RelaxDiffuse, motion vector passes).
- **NeuralRadiosity**: Neural radiosity estimator using CUDA + tiny-cuda-nn. Online/offline training, inference via hash-grid-encoded MLPs. Has its own `Model/` subdirectory with CUDA implementations (`Model.cu`, `HashGrid.cu`, `HashGridInterp.cu`). Built with `CUDA_ARCHITECTURES 86` (Ampere).
- **AccumulatePass, ToneMapper, ModulateIllumination, TAA, SVGF**: Standard post/denoising/compositing passes.
- **DLSSPass, RTXDIPass, OptixDenoiser**: SDK integration passes.

### Render graphs (`scripts/*.py`)
Render graphs are Python scripts that wire passes together. They use `from falcor import *`, create a `RenderGraph`, add passes via `createPass("Name", {options})`, connect them with `g.addEdge("Pass.output", "Pass2.input")`, and mark outputs. At the bottom the graph is attached to the live renderer via `m.addGraph(...)` where `m` is the Mogwai instance injected into the script namespace. Run with `Mogwai --script <path>.py --scene <scene>`. Subdirectories `scripts/inv-rendering/` and `scripts/sdf-editor/` hold specialized graph setups.

### Applications
- **Mogwai** (`Source/Mogwai/`): Main interactive renderer application with ImGui-based UI, scripting, and render graph loading.
- **FalcorTest** (`Source/Tools/FalcorTest/`): Unit test runner.
- **RenderGraphEditor**: Standalone graph editor.
- **Samples** (`Source/Samples/`): Minimal example apps (HelloDXR, CudaInterop, ShaderToy, etc.).

### Plugin system
- Plugins are shared libraries (`add_plugin()` in CMake) placed in the `plugins/` output directory.
- A `plugins.json` manifest is auto-generated listing all plugin libraries.
- `PluginBase` classes (e.g. `RenderPass`) use `FALCOR_PLUGIN_BASE_CLASS`; concrete plugins use `FALCOR_PLUGIN_CLASS`.
- Scene importers (`Source/plugins/importers/`) are also plugins: Assimp, Mitsuba, PBRT, USD, Python.

### Shaders (Slang)
- Shaders use the [Slang](https://github.com/shader-slang/slang) language with extensions `.slang` (importable modules) and `.slangh` (included headers).
- Entry point suffixes: `.rt.slang` (ray tracing), `.cs.slang` (compute), `.3d.slang` (graphics), `.ps.slang`/`.vs.slang`/`.gs.slang` (single-stage).
- `target_copy_shaders(<target> <output_dir>)` in CMake copies shader sources to the build output. Shaders are compiled at runtime by the `ProgramManager` using the Slang compiler bundled via packman.
- Slang files are co-located with their C++ host code in the directory tree.

### Dependencies
- **Git submodules** (`.gitmodules`): pybind11, glfw, args, fmt, imgui, vulkan-headers, tiny-cuda-nn.
- **Packman** (`dependencies.xml`): Binary deps pulled to `external/packman/` — Slang compiler, CUDA toolkit, DLSS, NRD, RTXDI, NVTT, python, cmake, ninja, nanovdb, USD.
- **tiny-cuda-nn** (`external/tiny-cuda-nn/`): Neural network framework used by NeuralRadiosity. Linked as `tiny-cuda-nn` + `tiny-cuda-nn-resources`.

## Coding Conventions

- **C++17**, 4-space indentation, column limit 140. See `.clang-format` (Chromium-based) and `.editorconfig`.
- **Naming**: `PascalCase` for types/enums, `camelCase` for functions/variables, `mCamelCase` for private members, `sCamelCase` for static members, `kCamelCase` for constants, `UPPER_SNAKE_CASE` for macros.
- **Smart pointers**: Use `ref<T>` (Falcor's `std::shared_ptr` alias). Prefer `make_ref<T>()`.
- **Namespace**: All code under `Falcor` namespace (PascalCase). `using namespace` only in `.cpp` files, never in headers. No `using namespace std`.
- **Slang formatting**: Same conventions as C++. Structs hold both data and code. Use `1.f` for float literals.
- **Headers**: `#pragma once`. Headers must be self-sufficient. Source files include matching header first, then local includes, then third-party, then STL.
- **Error handling**: See `docs/development/error-handling.md`. Avoid C-style error codes; use exceptions/asserts per the policy.
- **Comments**: Doxygen-style `/** ... */` for functions (new style with `@param`/`@return`), `///<` for struct fields, `//` for inline.

## CMake Conventions
- Add new source/shader files to `target_sources()` in the relevant `CMakeLists.txt`, sorted alphabetically.
- New render pass: `add_plugin(Name)` + `target_sources` + `target_copy_shaders` + `target_source_group`.
- New executable: `add_falcor_executable(Name)`.
- CUDA targets set `CUDA_ARCHITECTURES` property (default 75-Turing; NeuralRadiosity uses 86-Ampere).
- On Linux, code is compiled with `_GLIBCCXX_USE_CXX11_ABI=0` for compatibility with prebuilt dependencies.

## NeuralRadiosity-specific notes
- Requires CUDA + tiny-cuda-nn. The pass does online training and inference of hash-grid-encoded MLPs for radiance prediction.
- Training modes: "Train" (offline), "Online train" (dynamic scenes), "Idle" (inference only with loaded checkpoint).
- Checkpoints saved to/loaded from `media/ckpt/` (the pass UI exposes a "Load models" action). Pre-trained checkpoints exist for `living-room` and `kitchen`.
- The `Model/` directory contains CUDA kernels (`Model.cu`, `HashGrid.cu`, `HashGridInterp.cu`, `NeuralModel.cu`, `NeuralConeModel.cu`) that run on separate CUDA streams from rendering. The pass CMake uses `file(GLOB_RECURSE)` over `*.cpp/*.h/*.cu/*.slang`, so new files in the pass directory are picked up automatically without editing `target_sources`.
- Falcor-CUDA interop uses Vulkan external memory / D3D12 shared handles via `CudaUtils`.
- See `NCR-README.md` at the repo root for the end-user workflow (running, online training, offline training). The demo scripts `ncr.sh` / `ptnrd.sh` / `ptdn.sh` at the repo root launch `Mogwai` against `media/scenes/living-room/scene-animation.pyscene` with `LD_PRELOAD=/lib/x86_64-linux-gnu/libpng16.so.16`.

## Further reading
- `docs/development/error-handling.md` — full error-handling policy (asserts vs. `FALCOR_THROW`/`FALCOR_CHECK` exceptions vs. return values; `logWarning`/`reportError`/`reportFatalError`).
- `docs/development/coding-conventions.md` — detailed naming, formatting, and Slang rules (this file is the authoritative source; the summary above is a quick reference).
- `docs/development/cmake.md` — CMake target helpers (`add_plugin`, `add_falcor_executable`, `target_copy_shaders`, `target_source_group`).
- `docs/development/unit-testing.md` — full unit-test API (`CPU_TEST`/`GPU_TEST`, `EXPECT_*` macros, `GPUUnitTestContext`).
- `README.md` — upstream Falcor overview, NVAPI/NSight Aftermath/OptiX setup, SDK licenses.

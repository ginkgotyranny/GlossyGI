# NCR on OpenHarmony / HarmonyOS

Work in progress on the branch `ohos-port`: running the NeuralRadiosity (NCR) pipeline on a HarmonyOS
phone, with the two MLPs inferred and trained on the NPU in real time. The reference for the platform
side is the finished OpenHarmony port of Device MRRC (a separate repository, not part of this one).

Falcor and CUDA / tiny-cuda-nn do not exist on the phone, so the port is a new native runtime that
reuses NCR's algorithm, its shader logic and its checkpoints.

## Where each part of the network runs

| Part | Parameters | Runs on | Why |
|---|---|---|---|
| DiffNet, SpecNet (inference; forward + backward for training) | 11,264 + 17,408 | NPU, through NNRt on the HiAI Foundation device | bias-free MLPs of 1x1 convolutions, all channel counts multiples of 16 |
| Primary and cluster hash grids (lookup, gradient scatter, Adam + EMA) | 7,051,018 | GPU compute | gather / scatter by hashed index, not a matrix operation |
| Adam + EMA of the MLP weights | | CPU, float32 | tiny, and keeps tiny-cuda-nn's float32 master weights |

The grid lookup writes the encoding channels of the network input, as the CUDA kernels in
`Source/RenderPasses/NeuralRadiosity/Model/HashGrid*.cu` do, and the training graph returns the gradient
of those channels for the scatter. Network tensors are channel-planar (NCHW): the NPU accepts no other
layout.

## Layout

```
ohos/npu/ncr_npu_graphs.py     the two MLPs as NPU graphs (PyTorch): inference, and training with hand-written gradients
ohos/npu/ncr_reference.py      the rest of the model in numpy: checkpoint format, input layout, grid lookup and scatter, Adam + EMA
ohos/npu/verify_tcnn.py        the reference model against data dumped from the CUDA implementation, piece by piece
ohos/npu/train_loop_check.py   convergence of the NPU-style training loop against exact training
ohos/npu/export_models.py      gradient check vs autograd, ONNX, test vectors, .om / .omc conversion (WSL)
ohos/probes/npu_mlp/           on-device probe: load a model through NNRt, check the outputs, time RunSync
ohos/probes/gpu/               on-device probe: hash grids in compute, ray query throughput, shader features, baked pipelines
ohos/replay/                   the replayer: runs baked frames on any Vulkan device (PC and phone), rendering and training,
                               with the MLPs on the NPU (phone) or on the CPU
ohos/tools/spirv_info.py       capabilities and bindings of a SPIR-V module (and .layout files for the probe)
ohos/tools/bake_check.py       resolves every SPIR-V binding of a bake to the resource the bake recorded for it
ohos/tools/bake_pack.py        joins bake and SPIR-V into the index the replayer reads; repairs the SPIR-V (spirv_fix.py)
ohos/tools/replay_compare.py   a replay's images and buffers against Falcor's from the baked frame
ohos/tools/train_check.py      one training step of the replayer against the reference model, stage by stage, and against Falcor
ohos/tools/train_compare.py    N frames of training in the replayer against N frames of training in Falcor
ohos/tools/raw_to_png.py       a dumped RGBA32F image as PNG
ohos/tools/png_stats.py        mean / saturation / difference of PNG frames, standard library only
ohos/_build/                   generated files (git-ignored)
```

## Running the probes

Needs DevEco Studio (OpenHarmony NDK and `hdc`), a phone connected over `hdc`, Git Bash, and for the NPU
models a Python with torch and onnx plus the two converters in WSL (`omg` of the HiAI DDK tools, `atc` of
the CANN toolkit; `HIAI_OMG` overrides the `omg` path).

```bash
python ohos/npu/export_models.py --om --omc     # ONNX + test vectors + .om + Kirin9050 .omc into ohos/_build/npu_models
bash ohos/probes/npu_mlp/run.sh                 # every model on the phone: load, numerical check, timing
bash ohos/probes/gpu/run.sh                     # hash grids, ray query, shader features
```

Both probes are plain executables run from a shell on the phone (`/data/local/tmp`), so neither needs a
HAP. Results land in `ohos/_build/{npu,gpu}_probe/results.txt`.

## Probe results

HUAWEI Mate 90 Pro Max (Kirin 9050 Pro, Maleoon 955, HarmonyOS 7.1 / API 26, HiAI Foundation
111.636.120.010), 2026-11-13. Times are medians of 30 runs (NPU) or the best of three submissions (GPU).

### NPU (portable `.om`, compiled on the phone)

| Graph | Samples | Time | Notes |
|---|---|---|---|
| DiffNet inference, 256x256 | 65,536 | 1.6 ms | whole graph on the NPU; output L2 error 6e-4 vs float32 |
| SpecNet inference, 256x256 | 65,536 | 1.9 ms | same |
| DiffNet inference, 512x512 | 262,144 | 7.7 ms | |
| SpecNet inference, 512x512 | 262,144 | 9.4 ms | about 480 G multiply-adds per second |
| SpecNet inference, 65536x1 | 65,536 | 33 ms | the NPU refuses these convolutions: batches must be 2-D tiles |
| DiffNet training graph | 4,096 | 6.1 ms | whole graph on the NPU; gradient L2 error 1e-2 vs float32 |
| SpecNet training graph | 2,048 / 4,096 / 8,192 / 16,384 | 5.1 / 7.7 / 11.6 / 20.6 ms | same |

- Every inference and training graph is one NPU partition, except where noted. The training graph
  returns the weight gradients and the gradient of the network input.
- ReLU derivative: `a / (a + 1e-4)` and the clip form stay on the NPU (same time); `Sign` is refused
  and falls back to the CPU (20 ms). Neither NPU form is an exact step: activations close to zero get
  a partial mask, about 2% L2 error in the gradients against autograd.
- 16 output channels cost the same as 4.
- The precompiled `.omc` loads in 25-85 ms instead of 300-650 ms but runs up to 40% slower than the `.om`
  here.

### GPU (Slang compute shaders compiled to SPIR-V by Falcor's `slangc`)

| Test | Size | Time | Check |
|---|---|---|---|
| Primary grid lookup | 65,536 / 262,144 samples | 4.3 / 16.7 ms | equals the CPU reference (fp16 rounding) |
| Cluster grid lookup | 65,536 / 131,072 samples x 4 clusters | 6.5 / 12.6 ms | same |
| Primary grid gradient scatter (integer atomics) | 8,192 / 32,768 / 65,536 samples | 3.1 / 12.0 / 24.1 ms | exact, no lost update |
| Adam + EMA pass, primary / cluster grid | 2,693,314 / 4,357,704 parameters | 2.6-4.5 / 3.9-4.5 ms | equals the CPU formula |
| Ray query, incoherent rays | 640x360 x 1 / 8 / 32 rays | 5.4 / 42 / 166 ms (44 M rays/s) | every ray hits the closed box |
| Ray query, rays in a cone | 640x360 x 1 / 8 / 32 rays | 3.1 / 22 / 87 ms (85 M rays/s) | same |
| Subgroup sum, groupshared memory | 32 threads per group | | correct; one subgroup holds the whole group |

Driver facts: Vulkan 1.4, `shaderInt64 = 0` (a shader with any 64-bit integer cannot be compiled:
"Unsupported capability (11)"), `shaderInt16`, `shaderFloat16` and 16-bit storage present, subgroup size
32 with arithmetic in compute, ray query and acceleration structures present, `VK_OHOS_external_memory`
present.

### What follows from them

- The NPU is not the limit. At 640x360 with a bathroom-like split (93k diffuse, 130k specular samples) the
  render inference is about 7.5 ms, the 65,536 diffuse queries of a training step 1.7 ms, and the two
  training graphs at 4,096 samples each 14 ms.
- Variable-length batches are run as fixed 2-D tiles (for example 256x256), one `RunSync` per tile.
- The GPU is the limit. Grid lookups for the same frame cost about 27 ms, and 32 cone rays per specular
  sample about 50 ms; the per-frame K-Means cone trace is out of reach, the cluster-reuse path
  (`ConeTraceScreen`) is the one to port.
- Falcor shader modules that use 64-bit integers have to be replaced.

## The reference model and its parity with the CUDA implementation

`ncr_reference.py` and `ncr_npu_graphs.py` together are NRModel without tiny-cuda-nn; the phone runtime
is written against them. They are checked against the CUDA implementation itself:

```bat
set NCR_DUMP_DIR=<dir> & set NCR_DUMP_STEP=300
set NCR_SMOKE_SCENE=media/scenes/bathroom/scene-v4.pyscene & set NCR_SMOKE_FRAMES=305
build\windows-vs2022\bin\Release\Mogwai.exe --headless --script scripts/ncr_smoke.py
python ohos/npu/verify_tcnn.py <dir>
python ohos/npu/train_loop_check.py <dir>
python ohos/npu/export_models.py --real <dir> --only ncr_real --om && bash ohos/probes/npu_mlp/run.sh --kind om ncr_real
```

With `NCR_DUMP_DIR` set, `NRModel` writes what one training step read and produced (inputs, targets,
outputs, every gradient, the optimizers' moments, checkpoints before and after) and the next render
inference. Without the variable the code path is untouched.

Result on bathroom, training step 300 (18,356 diffuse and 14,411 specular samples), 2026-11-13: all 27
checks pass.

| Quantity recomputed from the inputs of its stage | L2 error against CUDA |
|---|---|
| Primary and cluster encodings (EMA and training grids) | 2e-4 (fp16 rounding) |
| DiffNet / SpecNet outputs, inference and training | 5e-4 |
| Loss gradient | 2e-4 |
| MLP weight gradients, gradient of the encodings | 4e-3 to 2e-2 (tiny-cuda-nn computes them in fp16) |
| Grid gradients after the scatter | 7e-4 to 1e-3 |
| Change of the float32 master weights, all four optimizers | 2e-5 |
| EMA weights, all four optimizers | 5e-5 to 7e-5 |

The same inputs and weights on the phone's NPU: inference L2 error 6e-4 against the float32 graph (outputs
up to 20), training gradients 3e-3 to 2e-2. Training SpecNet's MLP from scratch on that batch, the loss
falls alike with exact gradients, with the NPU graph's fp16 inputs and outputs, and with 2% noise on top
(0.089 / 0.087 / 0.087 after 400 steps, from 12.6).

What the comparison established about the model, beyond what its source suggests:

- **Grid indexing is dense modulo the level size on every level.** The prime hash in `HashGrid*.cu` is
  never selected: its test compares the level size, already clamped to the table size, with the table
  size. Checkpoints are trained with the modulo indexing, so the port reproduces it.
- **Positions leave the unit cube** (cluster positions range from about -2 to 3). CUDA's float-to-unsigned
  conversion saturates at 0 and the trilinear weights then extrapolate; the shaders clamp explicitly,
  because SPIR-V leaves that conversion undefined.
- **Adam treats grids and MLPs differently.** A grid entry with an exactly zero gradient is skipped and has
  its own step count for the bias correction; every MLP weight is updated each step, with an L2 term of
  1e-8. The EMA runs over all entries, on the fp16 training weights, with tiny-cuda-nn's debiasing.
- **Small grid gradients matter**, since Adam moves an entry by about the learning rate whatever the
  magnitude of its gradient. The GPU sums gradients as integers in units of 2^-24 (the smallest fp16
  value); a coarser unit would drop 7-15% of the entries a step touches.
- **A batch is padded to a multiple of 256 rows and the padding takes part**: the loss is normalised by
  the padded count.
- **SquarePlus needs care in fp16**: `(10 z)^2` overflows from z = 25.6 on. The graphs take the root as
  `64 * sqrt((q / 64)^2 + 4 / 64^2)`, good up to z = 1638.

## Rendering on the phone: Falcor bakes, the phone replays

NCR's shaders lean on Falcor's shader library (materials, emissive light sampling, inline ray tracing,
the scene's GPU data). Instead of rewriting that for the phone, Falcor is used as the offline compiler:

```bat
set NCR_BAKE_DIR=<dir> & set NCR_BAKE_FRAME=5
set NCR_SMOKE_SCENE=media/scenes/bathroom/scene-v4.pyscene & set NCR_SMOKE_FRAMES=8
build\windows-vs2022\bin\Release\Mogwai.exe --headless --device-type vulkan --script scripts/ncr_smoke.py
python ohos/tools/spirv_info.py --summary <dir>/spirv/*.spv
NCR_BAKE_SPIRV=<dir>/spirv bash ohos/probes/gpu/run.sh pipelines
```

With `NCR_BAKE_DIR` set, the NeuralRadiosity pass (`BakeExport.cpp`) writes, at frame `NCR_BAKE_FRAME`:

- `spirv/<Pass>.spv`: each of its compute passes exactly as Falcor compiled it for the loaded scene and
  the current settings. The modules carry their bindings by shader path (`gScene.geometryInstances`,
  `gScene.materials.materialTextures`, ...).
- `data/`: the contents of every buffer and texture those passes have bound (geometry, materials, lights,
  textures; and the pass's own buffers and images as they are at the end of the frame, which are the
  reference results), every sampler, and the uniform block of every parameter block.
- `bake.txt`: the index. The frame is recorded as it executes: every call of a compute pass in order,
  with its dispatch size, what was bound at each shader path and its uniform block as it was then (a
  pass such as ConeTrace or CompactBatch runs several times per frame, on different ray batches), and
  between the calls the steps Falcor does on the host: clearing the counts, the prefix sums, network
  inference, the training step. Besides: the byte offset of every uniform member, so the replayer can
  set per-frame values by name; the ray batches by role; and per geometry instance what is needed to
  rebuild the acceleration structures the way Falcor lays them out (`InstanceID() + GeometryIndex()` is
  the global instance index).

`python ohos/tools/bake_check.py <dir>` joins the two: every binding of every module must resolve to a
baked resource. It does for cornell-box and for bathroom (852 geometry instances, 9 material textures,
70 buffers; 232 MB with the reference data), 2026-11-13.

Result on the phone, bathroom scene, 2026-11-13: **all 15 passes create compute pipelines on the
Maleoon 955** (FirstSmoothPass, CompactPass, ConeTrace, SpecPrimary, ClusterReproject, ConeTraceScreen,
ClusterSpatialReuse, ResolvePass, PostFilterPass, PostFXAAPass, RandomSmooth, SampleRHS, CompactBatch,
ResolveBatch, ResolveRHS). Creation takes 6 ms to 3.0 s per pass (ConeTraceScreen 3.0 s, SampleRHS 2.3 s),
7.2 s in total: a pipeline cache is needed for start-up.

One change to Falcor's shader library made that possible. The modules need Int16, Float16, 16-bit storage,
ray query, the ray tracing capability, storage images without format and, in ConeTrace, subgroup
arithmetic: all present on the phone. They also needed Int64, which is not: Falcor's default sample
generator seeds itself with SplitMix64. `Utils/Sampling/Pseudorandom/SplitMix64.slang` now has a variant
that computes the same sequence with 32-bit arithmetic (`FALCOR_SPLITMIX64_32BIT=1`, checked against
64-bit arithmetic on 400,000 draws); the NeuralRadiosity pass defines it, so the desktop and the phone
run the same code. The default is unchanged for every other pass.

### The replayer

`ohos/replay` loads a packed bake and executes the recorded frame, without Falcor: it creates the
buffers, textures and samplers of the bake, rebuilds the acceleration structures, creates a compute
pipeline per baked module and descriptor sets per recorded call, then runs the calls in order and does
the host steps between them itself. For a frame in render mode that is FirstSmoothPass, the prefix sums,
CompactPass, ConeTrace, SpecPrimary, the networks, ResolvePass, PostFilterPass, PostFXAAPass. The same
source builds for Windows and for OpenHarmony. `--per-pass` submits every call on its own and prints
its time.

The networks run the way they will in the app (`--networks`):

- `tiles` (default): a GPU pass of the replayer looks up the hash grids and lays the samples out as the
  NPU reads them, channel-planar tiles of 65,536 samples (`shaders/EncodeInputs`); the MLPs run on the
  tiles; a second pass writes the outputs back as the rows ResolvePass reads (`shaders/DecodeOutputs`).
  Here the MLPs run on the CPU, standing in for the NPU.
- `npu` (OpenHarmony): the same with the MLPs on the NPU through NNRt (`replay_nnrt.h`). The tiles are
  memory the GPU and the NPU share: an `OH_NativeBuffer` imported into Vulkan as a storage buffer
  (`VK_OHOS_external_memory`, `Device::makeSharedBuffer`) and given to NNRt by its file descriptor, one
  tensor per tile at its offset in the buffer (`OH_NNTensor_CreateWithFd`). `--copy-tiles` keeps the
  tiles in ordinary memory and copies them to and from tensors NNRt owns: the results are the same to
  the bit, the frame is 216 ms longer.
- `rows`: the float32 reference on the CPU, lookups included (`replay_nn.h`).

The prefix sums between the passes (mask to compacted index, and the count) are a compute pass of the
replayer as well (`shaders/PrefixSum`), so a frame only waits for the host where the networks run.
`--cpu-prefix-sum` does them on the CPU instead. Training is described below.

```bat
python ohos/tools/bake_pack.py <bake dir>
ohos\replay\build_pc.bat
ohos\_build\replay\ncr_replay.exe <bake dir> <checkpoint.ckpt> <out dir>
python ohos/tools/replay_compare.py <bake dir> <out dir>
python ohos/npu/export_models.py --only inf_256x256 --om            # the NPU models, for --networks npu
bash ohos/replay/run_device.sh <bake dir> <checkpoint.ckpt> --networks npu     # the same on the phone
python ohos/tools/replay_compare.py <bake dir> ohos/_build/replay/device_out
```

The bake keeps, as reference, every image and ray-batch buffer as Falcor left it at the end of the baked
frame; the replayer writes the same resources after replaying that frame.

Results, bathroom at 640x360 in render mode with a 300-step checkpoint (92,764 diffuse and 130,423
specular samples), 2026-11-13:

| | PC (the GPU Falcor ran on) | Phone (Maleoon 955) |
|---|---|---|
| Diffuse / specular masks, compacted indices, counts | identical | identical |
| Hit records (same triangle) | 100% of the pixels | 99.998% |
| Position, direction, normal, albedo, roughness (mean difference / mean value) | identical | 1e-6 to 1e-4 |
| Network inputs after cone trace and encodings (same measure) | 1e-4 | 2e-4 |
| Network outputs, CPU float32 against tiny-cuda-nn fp16 (same measure) | 5e-4 | 5e-4 |
| Final image against Falcor, networks on the CPU (rows / tiles) | 74.7 / 74.6 dB PSNR | 61.6 / 61.6 dB PSNR |
| Final image against Falcor, MLPs on the NPU | | 61.3 dB PSNR |

GPU time of that frame on the phone: FirstSmoothPass 2.2-2.9 ms; CompactPass + ConeTrace (32 rays for each
of 130,423 samples, K-Means) + SpecPrimary 68-78 ms; ResolvePass + PostFilterPass + PostFXAAPass 6.5-6.9 ms.
Acceleration structures (591,288 triangles) build in 22-31 ms, the 15 pipelines in 3.5-3.9 s.

The networks of that frame on the phone with `--networks npu`: grid lookups into tiles on the GPU 15-21 ms;
the MLPs 17-19 ms inside `RunSync` for two SpecNet and two DiffNet tiles (the first runs after loading;
the probe measured 1.6-1.9 ms per tile in steady state) plus 10-28 ms copying tiles in and out, which
zero-copy buffers remove; outputs back to rows about 3 ms. The whole frame, first run, about 120 ms.
The prefix sums ran on the CPU in that measurement (1.5 ms on a local copy of the buffer; 82 ms when walking the shared
buffer directly).

### Training

A frame baked in a training mode contains the training step (`host <k> train lhs`), and the bake holds
the model before and after the frame (`before.ckpt`, `after.ckpt`) and the learning rate and EMA decay
the step ran with. With `--networks tiles` or `npu` the replayer executes it:

| Step | Where | What |
|---|---|---|
| Inputs and targets | GPU: `EncodeInputs` with the training grids, `EncodeTargets` | tiles of the training graph's size (4,096 samples by default); the target tile carries a weight per slot, 0 beyond the batch |
| Gradients of the MLP | NPU: `ncr_<net>_train_<N>_fp16.om`, one run per tile | dL/dw, summed over the tiles on the CPU, and dL/d(input) per sample |
| Gradients of the grids | GPU: `ScatterGradients` | the encoding channels of dL/d(input) added to the grid entries the lookup read, in 2^-24 fixed point (integer atomics) |
| Grid optimizers | GPU: `GridOptimizer`, once per grid | Adam on the float32 masters where the gradient is not zero, per-entry step counts; EMA over all entries; fp16 copies rounded to nearest even |
| MLP optimizers | CPU | Adam + EMA on 28,672 weights, then the new weights go into the graphs' weight tensors |

`--frames N` replays the frame N times as consecutive frames: the frame counter that seeds the samplers
advances and the uniforms that depend on the sample counts are set as `NeuralRadiosity.cpp` sets them;
camera and scene stay as baked. The weights after the last frame are written as `trained.ckpt`, a
checkpoint Falcor loads.

```bat
set NCR_BAKE_DIR=<bake0> & set NCR_BAKE_FRAME=0 & set NCR_SMOKE_FRAMES=1 & set NCR_SMOKE_MODE=onlineTrain
set NCR_SMOKE_PROPS={'loadModelPath': '<start.ckpt>'}
build\windows-vs2022\bin\Release\Mogwai.exe --headless --device-type vulkan --script scripts/ncr_smoke.py
python ohos/tools/bake_pack.py <bake0>
python ohos/npu/export_models.py --only inf_256x256 --only _train_ --om
bash ohos/replay/run_device.sh <bake0> <bake0>/before.ckpt --networks npu --dump-train
python ohos/tools/train_check.py <bake0> ohos/_build/replay/device_out --npu      # one step
bash ohos/replay/run_device.sh <bake0> <bake0>/before.ckpt --networks npu --frames 60 --loss 10
python ohos/tools/train_compare.py <bake0> ohos/_build/replay/device_out <bake59>  # <bake59>: the same run baked at frame 59
```

Falcor has to start from a loaded checkpoint and the bake has to be of frame 0: then Falcor and the
replayer both begin with the fp16 training copies rebuilt from the masters and with empty Adam moments.

Results, bathroom at 640x360, online training from a 300-step checkpoint, 32,768 training samples per
frame (18,429 diffuse and 14,339 specular rows, 229,738 RHS queries), learning rate 5e-3, EMA 0.95,
`randomTrainDir` off (see below), 2026-11-13:

| | PC, MLPs on the CPU | Phone, MLPs on the NPU |
|---|---|---|
| One step, each stage against the reference model from its own inputs: encodings | 2e-4 | 2e-4 |
| MLP weight gradients / input gradients | 1e-4 / 2e-4 | 8e-3 to 2e-2 / 9e-3 to 1e-2 |
| Grid gradients after the scatter | 1e-5 | 1e-5 |
| Masters after the optimizers | within 2e-7 | within 2e-7 |
| One step against Falcor: cosine between the two updates, primary / cluster grid | 0.9996 / 0.9996 | 0.995 / 0.993 |
| the same, DiffNet / SpecNet | 0.993 / 0.980 | 0.983 / 0.969 |
| Image rendered after that step, against Falcor's | 53.5 dB | 54.4 dB |
| 60 frames against Falcor: cosine between the two updates, grids | 0.991 / 0.992 | 0.985 / 0.987 |
| the same, DiffNet / SpecNet | 0.992 / 0.984 | 0.958 / 0.940 |
| Image after frame 60, against Falcor's after its 60 frames | 51.3 dB | 50.9 dB |

For scale: Falcor's own image moves by 17-19 dB between its first and its 60th frame of that run. The
checkpoint the phone wrote after its 60 frames, loaded into Falcor (D3D12, render mode), gives the image
Falcor's own 60-frame checkpoint gives (mean difference 0.0012 of the 8-bit range). The PC column was
measured with the default `randomTrainDir`, the phone column needs it off, for the reason given below.

A frame of online training on the phone at these sizes, steady state over 60 frames: 316 ms
(364 ms with `--cpu-prefix-sum`, 580 ms with `--copy-tiles` as well).

| Part of the frame | Time |
|---|---|
| RandomSmooth + SampleRHS (32,768 samples, 8 RHS paths each), RHS prefix sum and compaction | 62 ms |
| RHS: grid lookups for 229,738 queries | 18 ms |
| RHS: DiffNet on the NPU, 4 tiles | 13 ms |
| RHS outputs; LHS prefix sums, compaction, cone trace (32 rays per specular sample), targets | 10 ms |
| Training: inputs and targets into tiles | 5 ms |
| Training: gradients on the NPU, 4 + 5 tiles of 4,096 | 80 ms (7.3-8.3 ms per tile in `RunSync`) |
| Training: grid gradient scatter and the two grid optimizers | 15 ms |
| Training: MLP optimizers on the CPU | 1-10 ms |
| Render: G-buffer, prefix sums, compaction, cone trace | 69 ms |
| Render: grid lookups, outputs, resolve, filters | 23 ms |
| Render: 2 + 2 tiles on the NPU | 14 ms |

That is about 200 ms on the GPU, 107 ms on the NPU and up to 10 ms on the CPU, with the sample counts
of the desktop defaults: the configuration for the phone is still to be chosen.

#### What the comparison showed about the desktop pipeline

Two properties of the pipeline as it runs on the desktop, found because the replay did not agree at first:

1. **The first training step after a random initialisation changes nothing.** The fp16 copies of the
   weights that training reads are first written by the optimizers; before that they are zero, the
   network output is constant and every gradient is zero. From a loaded checkpoint this does not
   happen: loading rebuilds the copies. (Baked frame 0 of a fresh run: Falcor's grids unchanged, MLP
   weights changed by rounding and weight decay only.)
2. **29% of the training samples are shaded from behind their surface**, and their direct light is
   decided by how a GPU treats an intersection 1e-4 from a ray's origin. RandomSmooth gives each
   sample a random direction and flips it to the side the path came from, but the test multiplies
   components of a local-space vector with the world-space face normal (`dot(wiLocal, sd.faceN)`), so
   the outcome depends on the orientation of the surface in the world (9,364 of 32,768 samples here).
   SampleRHS then shades such a sample from the back, while its stored position is offset to the
   front: the shadow ray toward a light behind the surface starts 1.2e-4 in front of the triangle it
   has to pass. On thick walls the far side stops that ray anyway. On the thin panel beside the
   bathroom's window, with the window light 0.4 behind it, the RTX lets most of those rays through and
   the Maleoon stops most: 17,387 against 7,425 in summed direct light on 2,228 samples of one frame,
   the whole difference between the two GPUs (elsewhere 19,261 against 19,168). With the default
   setting the phone therefore drifts from Falcor (35.6 dB after 60 frames) although every stage agrees
   with the reference model.

The pass property `randomTrainDir` (default on, the behaviour as it was) turns the random direction
off: a sample then keeps the direction its path arrived with. The comparisons above use that to have a
frame both GPUs must agree on. Whether the test in RandomSmooth should be corrected is a question for
the training algorithm, not for the port; it changes what the desktop learns from.

### What the phone's driver does differently

Each of these made the first replays on the phone wrong while the PC was right. The first two are
reproduced by `ohos/probes/gpu`.

1. **A load from a storage image after a store to the same texel, in the same invocation, returns the
   value from before the store** (test `readback`: 0 of 4,096; the store itself lands, and storage
   buffers behave). ResolvePass read the throughput from `gColor`, stored the product and read `gColor`
   back for the next pass, which therefore got the bare throughput: a white image. It now keeps the
   result in a local. No other NCR shader reads an image back after writing it.
2. **At most 65,535 thread groups per dimension, and a larger dispatch runs only that many without an
   error.** ConeTrace is dispatched with one group per specular sample (146,725 here), so half of the
   samples were never traced. Its shader already strides a group over further samples when there are
   fewer groups than samples; the replayer caps the group count at the device limit and sets the stride.
3. **Descriptor arrays are indexed per group unless the access is decorated NonUniform**, as Vulkan
   specifies. Falcor's material system indexes its texture and sampler arrays by material without the
   decoration, and the Slang version it ships does not emit it on the pointer even when asked, so pixels
   took the texture of a neighbour's material (part of a picture showed its wooden frame). `spirv_fix.py`,
   run by `bake_pack.py`, adds the decoration to every access into an array of images or samplers with a
   non-constant index, and declares the capabilities.

A 16-bit index load at a byte offset that is not a multiple of four, as Falcor does for every odd
triangle, was suspected first and is fine (test `load16`).

## Stages

1. Device probes (done).
2. The network split on the PC against the CUDA implementation (done, above).
3. Bake and replay. Done: the bake; the replayer rendering the baked frame on the PC and on the phone
   in agreement with Falcor, with the grid lookups on the GPU and the MLPs on the NPU; online training
   in the replayer, frame after frame, with the MLP gradients on the NPU, the grids trained on the GPU
   and the result in agreement with Falcor's training (above); the tiles shared by Vulkan and NNRt.
   The prefix sums run on the GPU.
   Next: sample counts for the phone; the cluster-reuse cone trace; a moving camera; the training
   schedule.
4. The HarmonyOS app: platform layer, NNRt backend, zero-copy buffers, frame synchronisation, cross-frame
   training; then performance.

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
ohos/probes/gpu/               on-device probe: hash grids in compute, ray query throughput, shader features
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

## Stages

1. Device probes (done).
2. The network split on the PC against the CUDA implementation (done, above).
3. A stand-alone Vulkan renderer on the PC with ONNX Runtime in place of the NPU, compared with Falcor.
4. The HarmonyOS app: platform layer, NNRt backend, zero-copy buffers, frame synchronisation, cross-frame
   training; then performance.

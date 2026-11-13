#!/usr/bin/env python
"""Does the training loop of the port converge like exact training?

Trains SpecNet's MLP from a fresh initialisation on the specular batch of a Falcor dump (the
encodings in the rows stay fixed), 4,096 samples per step, with tiny-cuda-nn's Adam, three ways:

  exact   autograd gradients in float64
  npu     the NPU training graph as it runs on the phone: fp16 weights and samples in, the
          a / (a + eps) ReLU derivative, fp16 gradients out
  noisy   the same with 2% relative noise on every gradient, twice the error the phone's fp16
          arithmetic showed against float32 in the probe

and prints the relative-L2-luminance loss over the whole batch. The three should fall together.
This checks the gradient approximations and the optimizer loop, not the renderer.

    python ohos/npu/train_loop_check.py <dump dir> [steps]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import ncr_npu_graphs as graphs  # noqa: E402
import ncr_reference as ref  # noqa: E402

BATCH = 4096
LEARNING_RATE, EMA_DECAY = 5e-3, 0.95


def loss(kernels, x, target):
    """RelativeL2Luminance over all rows, float64, without the loss scale."""
    with torch.no_grad():
        h = x
        for w in kernels[:-1]:
            h = F.relu(F.conv2d(h, w))
        y, _ = graphs.squareplus(F.conv2d(h, kernels[-1]))
        luminance = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
        difference = y[:, :ref.OUTPUT_DIM] - target[:, :ref.OUTPUT_DIM]
        return float((difference * difference / (luminance * luminance + graphs.LOSS_EPS)).mean())


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dump = ref.Dump(sys.argv[1])
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rows = dump["train_spec_input"][:int(dump.scalars["spec_size"])].astype(np.float64)
    target = np.zeros((rows.shape[0], ref.OUTPUT_PAD))
    target[:, :ref.OUTPUT_DIM] = dump["train_spec_target"][:rows.shape[0]]
    target[:, graphs.MASK_CHANNEL] = 1.0
    x_all = torch.from_numpy(ref.rows_to_planar(rows))
    t_all = torch.from_numpy(ref.rows_to_planar(target))
    print(f"{rows.shape[0]} specular samples, target radiance up to {target.max():.1f}, {steps} steps of {BATCH}")

    rng = np.random.default_rng(7)
    initial = np.concatenate([rng.uniform(-1, 1, o * i) * np.sqrt(6.0 / (o + i)) for o, i in ref.SPEC_LAYERS]).astype(np.float32)
    npu_graph = graphs.Training(graphs.SPEC, BATCH, mask="div")

    def gradients(mode, half_weights, batch_rows, batch_target):
        kernels = ref.split_layers(half_weights, ref.SPEC_LAYERS)
        if mode == "exact":
            g = graphs.reference_gradients(graphs.SPEC, [torch.from_numpy(k.astype(np.float64)) for k in kernels],
                                           torch.from_numpy(ref.rows_to_planar(batch_rows)),
                                           torch.from_numpy(ref.rows_to_planar(batch_target)))
            return np.concatenate([w.numpy().reshape(-1) for w in g[:-1]])
        inputs = [torch.from_numpy(k.astype(np.float32)) for k in kernels]
        inputs += [torch.from_numpy(ref.rows_to_planar(batch_rows.astype(np.float16).astype(np.float32))),
                   torch.from_numpy(ref.rows_to_planar(batch_target.astype(np.float16).astype(np.float32)))]
        with torch.no_grad():
            g = np.concatenate([w.numpy().reshape(-1) for w in npu_graph(inputs)[:-1]])
        if mode == "noisy":
            g = g * (1.0 + 0.02 * noise.standard_normal(g.shape))
        return g.astype(np.float16)

    results = {}
    for mode in ["exact", "npu", "noisy"]:
        noise = np.random.default_rng(11)
        order = np.random.default_rng(3)
        master, half = initial.copy(), initial.astype(np.float16)
        m, v = np.zeros_like(master), np.zeros_like(master)
        param_steps, ema = np.zeros(master.shape, dtype=np.uint32), np.zeros(master.shape, dtype=np.float16)
        curve = []
        for step in range(1, steps + 1):
            pick = order.choice(rows.shape[0], BATCH, replace=False)
            g = gradients(mode, half, rows[pick], target[pick])
            master, half, m, v, param_steps, ema = ref.adam_ema_step(
                master, half, g, m, v, param_steps, ema, step=step, learning_rate=LEARNING_RATE,
                ema_decay=EMA_DECAY, loss_scale=graphs.LOSS_SCALE, matrix_params=True)
            if step % 50 == 0 or step == 1:
                kernels = [torch.from_numpy(k.astype(np.float64)) for k in ref.split_layers(ema, ref.SPEC_LAYERS)]
                curve.append((step, loss(kernels, x_all, t_all)))
        results[mode] = curve

    print("loss of the EMA weights over the whole batch")
    print("   step      exact        npu      noisy")
    for i, (step, value) in enumerate(results["exact"]):
        print(f"  {step:5d}  {value:9.4f}  {results['npu'][i][1]:9.4f}  {results['noisy'][i][1]:9.4f}")
    final = {mode: curve[-1][1] for mode, curve in results.items()}
    first = results["exact"][0][1]
    ok = all(value < 0.5 * first for value in final.values()) and max(final.values()) < 1.25 * min(final.values())
    print("converges alike" if ok else "DIFFERENT CONVERGENCE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

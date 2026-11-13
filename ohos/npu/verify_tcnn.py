#!/usr/bin/env python
"""Checks the reference model of the port (ncr_reference.py, ncr_npu_graphs.py) against the CUDA /
tiny-cuda-nn implementation, piece by piece.

The data comes from a Falcor run with NCR_DUMP_DIR set (see DebugDump in Model/Model.cu):

    set NCR_DUMP_DIR=<dir> & set NCR_DUMP_STEP=300
    set NCR_SMOKE_SCENE=media/scenes/bathroom/scene-v4.pyscene & set NCR_SMOKE_FRAMES=305
    build\\windows-vs2022\\bin\\Release\\Mogwai.exe --headless --script scripts/ncr_smoke.py

    python ohos/npu/verify_tcnn.py <dir>

Each line compares one quantity the CUDA code produced with the same quantity recomputed here from
the inputs of that stage, so an error cannot hide behind an earlier one. tiny-cuda-nn computes the
MLPs and accumulates the grid gradients in fp16; the reference uses float32 / float64, so agreement
is to fp16 accuracy, not to the bit. Exit code: number of failed checks.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import ncr_npu_graphs as graphs  # noqa: E402
import ncr_reference as ref  # noqa: E402

failures = 0


def check(name, ours, theirs, tolerance, absolute=None):
    """L2 error relative to the norm of the CUDA result (or, with `absolute`, the largest element error)."""
    global failures
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    finite = np.isfinite(theirs)
    difference = np.where(finite, ours - theirs, 0.0)
    norm = np.linalg.norm(np.where(finite, theirs, 0.0))
    l2 = np.linalg.norm(difference) / max(norm, 1e-30)
    worst = np.abs(difference).max()
    value = worst if absolute else l2
    ok = value <= tolerance
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} L2 error {l2:.2e}  largest element error {worst:.2e}  "
          f"(largest |value| {np.abs(np.where(finite, theirs, 0.0)).max():.3g})")


def mlp_forward(rows, kernels):
    """Float32 forward of the NPU graph on sample rows -> (N, 16)."""
    model = graphs.Inference.__new__(graphs.Inference)
    torch.nn.Module.__init__(model)
    with torch.no_grad():
        inputs = [torch.from_numpy(ref.rows_to_planar(rows.astype(np.float32)))]
        inputs += [torch.from_numpy(k.astype(np.float32)) for k in kernels]
        return ref.planar_to_rows(graphs.Inference.forward(model, inputs).numpy())


def training_gradients(spec, rows, target, kernels):
    """Float64 gradients of the training graph (exact ReLU derivative) for a whole batch."""
    model = graphs.Training(spec, rows.shape[0], mask="sign").double()
    padded_target = np.zeros((rows.shape[0], ref.OUTPUT_PAD), dtype=np.float64)
    padded_target[:, :ref.OUTPUT_DIM] = target
    padded_target[:, graphs.MASK_CHANNEL] = 1.0   # every row of the batch counts, as in tiny-cuda-nn
    inputs = [torch.from_numpy(k.astype(np.float64)) for k in kernels]
    inputs += [torch.from_numpy(ref.rows_to_planar(rows.astype(np.float64))), torch.from_numpy(ref.rows_to_planar(padded_target))]
    with torch.no_grad():
        outputs = model(inputs)
    weight_gradients = np.concatenate([g.numpy().reshape(-1) for g in outputs[:-1]])
    return weight_gradients, ref.planar_to_rows(outputs[-1].numpy())


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    dump = ref.Dump(sys.argv[1])
    before = ref.load_checkpoint(os.path.join(dump.folder, "train_before.ckpt"))
    after = ref.load_checkpoint(os.path.join(dump.folder, "train_after.ckpt"))
    loss_scale = dump.scalars["loss_scale"]
    print(f"training step {int(dump.scalars['diff_opt_step']) + 1}: {int(dump.scalars['diff_size'])} diffuse and "
          f"{int(dump.scalars['spec_size'])} specular samples, learning rate {dump.scalars['learning_rate']}, "
          f"EMA decay {dump.scalars['ema_decay']}")

    # ---- Render inference: EMA weights, as stored in the checkpoint written after the step.
    print("render inference (EMA weights)")
    rows = dump["infer_diff_input"].astype(np.float32)
    check("DiffNet: primary encoding", ref.prim_encode(after["prim_grid_ema"], rows[:, ref.DIFF_POS:ref.DIFF_POS + 3]),
          rows[:, :ref.PRIM_ENC_DIM], 2e-3, absolute=True)
    check("DiffNet: output", mlp_forward(rows, ref.split_layers(after["diff_ema"], ref.DIFF_LAYERS))[:, :ref.OUTPUT_DIM],
          dump["infer_diff_output"], 2e-2)
    rows = dump["infer_spec_input"].astype(np.float32)
    check("SpecNet: primary encoding", ref.prim_encode(after["prim_grid_ema"], rows[:, ref.SPEC_POS:ref.SPEC_POS + 3]),
          rows[:, :ref.PRIM_ENC_DIM], 2e-3, absolute=True)
    check("SpecNet: cluster encoding", ref.cls_encode(after["cls_grid_ema"], *ref.spec_clusters(rows)),
          rows[:, ref.SPEC_CLS_ENC:ref.SPEC_CLS_ENC + ref.CLS_ENC_DIM], 2e-3, absolute=True)
    check("SpecNet: output", mlp_forward(rows, ref.split_layers(after["spec_ema"], ref.SPEC_LAYERS))[:, :ref.OUTPUT_DIM],
          dump["infer_spec_output"], 2e-2)

    # ---- Training step: fp16 training copies of the weights.
    print("training step: forward and gradients (fp16 training weights)")
    prim_half, cls_half = dump["before_prim_grid_half"], dump["before_cls_grid_half"]
    grid_gradients = {}
    nets = [("diff", graphs.DIFF, ref.DIFF_LAYERS, ref.DIFF_POS), ("spec", graphs.SPEC, ref.SPEC_LAYERS, ref.SPEC_POS)]
    for name, spec, layers, pos_offset in nets:
        label = "DiffNet" if name == "diff" else "SpecNet"
        rows = dump[f"train_{name}_input"].astype(np.float32)
        target = dump[f"train_{name}_target"]
        kernels = ref.split_layers(dump[f"before_{name}_params_half"], layers)
        pos = rows[:, pos_offset:pos_offset + 3]
        check(f"{label}: primary encoding", ref.prim_encode(prim_half, pos), rows[:, :ref.PRIM_ENC_DIM], 2e-3, absolute=True)
        if name == "spec":
            check(f"{label}: cluster encoding", ref.cls_encode(cls_half, *ref.spec_clusters(rows)),
                  rows[:, ref.SPEC_CLS_ENC:ref.SPEC_CLS_ENC + ref.CLS_ENC_DIM], 2e-3, absolute=True)
        output = dump[f"train_{name}_output"].astype(np.float64)
        check(f"{label}: output (16 channels)", mlp_forward(rows, kernels), output, 2e-2)

        # Loss gradient from the CUDA output itself.
        luminance = 0.299 * output[:, 0] + 0.587 * output[:, 1] + 0.114 * output[:, 2]
        g = np.zeros_like(output)
        g[:, :ref.OUTPUT_DIM] = (2.0 * loss_scale / (rows.shape[0] * ref.OUTPUT_DIM)) * (output[:, :ref.OUTPUT_DIM] - target) \
            / (luminance * luminance + graphs.LOSS_EPS)[:, None]
        check(f"{label}: loss gradient dL/dy", g, dump[f"train_{name}_dL_doutput"], 2e-2)

        weight_gradients, input_gradient = training_gradients(spec, rows, target, kernels)
        check(f"{label}: weight gradients", weight_gradients, dump[f"train_{name}_params_grad"], 5e-2)
        cuda_input_gradient = dump[f"train_{name}_dL_dinput"]
        enc_dim = spec.encoding_dim
        check(f"{label}: dL/d(encoding)", input_gradient[:, :enc_dim], cuda_input_gradient[:, :enc_dim], 5e-2)

        # Scatter of the CUDA input gradient itself.
        grid_gradients[name] = ref.prim_scatter(cuda_input_gradient[:, :ref.PRIM_ENC_DIM], pos)
        if name == "diff":
            check("primary grid gradient after the diffuse batch", grid_gradients["diff"], dump["train_prim_grid_grad_diff"], 5e-2)
        else:
            check("primary grid gradient, both batches", grid_gradients["diff"] + grid_gradients["spec"], dump["train_prim_grid_grad"], 5e-2)
            cls_gradient = ref.cls_scatter(cuda_input_gradient[:, ref.SPEC_CLS_ENC:ref.SPEC_CLS_ENC + ref.CLS_ENC_DIM], *ref.spec_clusters(rows))
            check("cluster grid gradient", cls_gradient, dump["train_cls_grid_grad"], 5e-2)

    # ---- Optimizers, from the CUDA gradients.
    print("optimizer step (Adam on the float32 masters, EMA of the fp16 weights)")
    blocks = [("prim_grid", "before_prim_grid_half", "train_prim_grid_grad", False),
              ("cls_grid", "before_cls_grid_half", "train_cls_grid_grad", False),
              ("diff", "before_diff_params_half", "train_diff_params_grad", True),
              ("spec", "before_spec_params_half", "train_spec_params_grad", True)]
    for block, half_name, gradient_name, matrix in blocks:
        optimizer = block + "_opt"
        gradient = dump[gradient_name]
        new_master, _, _, _, steps, new_ema = ref.adam_ema_step(
            before[block + "_master"], dump[half_name], gradient, dump[optimizer + "_first_moments"],
            dump[optimizer + "_second_moments"], dump[optimizer + "_param_steps"], before[block + "_ema"],
            step=int(dump.scalars[optimizer + "_step"]) + 1, learning_rate=dump.scalars["learning_rate"],
            ema_decay=dump.scalars["ema_decay"], loss_scale=loss_scale, matrix_params=matrix)
        updated = int((after[block + "_master"] != before[block + "_master"]).sum())
        print(f"  {block}: {updated} of {gradient.size} parameters changed in the CUDA step, "
              f"{int((gradient != 0).sum())} have a non-zero gradient")
        check(f"{block}: change of the master weights", new_master - before[block + "_master"],
              after[block + "_master"] - before[block + "_master"], 1e-3)
        check(f"{block}: EMA weights", new_ema, after[block + "_ema"], 1e-3)

    print(f"{failures} check(s) failed" if failures else "all checks passed")
    sys.exit(failures)


if __name__ == "__main__":
    main()

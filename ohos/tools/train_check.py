#!/usr/bin/env python
"""Checks a training step of the replayer against the reference model and against Falcor.

    python ohos/tools/train_check.py <bake dir> <replay output dir> [--npu]

The bake must be of a frame that trains, baked at frame 0 (NCR_BAKE_FRAME=0): Falcor then starts,
like the replayer, from <bake>/before.ckpt with fresh optimizer moments. The replay must have run one
frame with --dump-train, from the bake's model (`ncr_replay <bake> bake <out> --dump-train`).

Step by step, each from the replayer's own inputs to that step, against ohos/npu (the model that was
checked against the CUDA implementation by verify_tcnn.py):
  1. the grid encodings in the input tiles      (GPU: EncodeInputs with the training grids)
  2. the gradients of the MLP weights and inputs (NPU or its CPU stand-in: the training graph)
  3. the gradients scattered into the grids      (GPU: ScatterGradients)
  4. the weights after the optimizers            (GPU: GridOptimizer; CPU: the MLPs)
and the result against Falcor:
  5. the weights in trained.ckpt against <bake>/after.ckpt. After one Adam step from fresh moments
     every updated weight has moved by the learning rate in the direction opposite to its gradient,
     so the comparison is: do the same weights move, and in the same direction.

--npu widens the tolerance of step 2 to what the NPU's training graph is accepted with (fp16
arithmetic and the smooth ReLU derivative; see ohos/README.md).
Run with a Python that has numpy and torch, e.g. E:/MyProjects/MRRCGI/.venv/Scripts/python.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "npu"))
import ncr_npu_graphs as graphs  # noqa: E402
import ncr_reference as ref  # noqa: E402

failures = 0


def check(name, value, limit, detail=""):
    global failures
    ok = value <= limit
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {value:.2e} (limit {limit:.0e}){' ' + detail if detail else ''}")


def relative_l2(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def load_working(path):
    """The four fp16 blocks of an NRWK file (NRModel::saveWorkingCopies), in checkpoint order."""
    data = open(path, "rb").read()
    magic, version, *counts = __import__("struct").unpack_from("<4sI4I", data, 0)
    assert magic == b"NRWK" and version == 1, path
    blocks, offset = [], 24
    for count in counts:
        blocks.append(np.frombuffer(data, dtype=np.float16, count=count, offset=offset).copy())
        offset += count * 2
    return blocks


def tiles_to_rows(path, channels, tile, count):
    """Tiles as the replayer lays them out, (tiles, C, tile) fp16 -> rows (count, C) float32."""
    data = np.fromfile(path, dtype=np.float16).reshape(-1, channels, tile)
    return np.ascontiguousarray(data.transpose(0, 2, 1)).reshape(-1, channels)[:count].astype(np.float32), data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bake")
    parser.add_argument("out")
    parser.add_argument("--npu", action="store_true")
    args = parser.parse_args()

    info = {}
    for line in open(os.path.join(args.out, "train.txt")):
        p = line.split()
        info[p[0]] = [float(v) for v in p[1:]]
    tile = int(info["tile"][0])
    learning_rate, ema_decay = info["learning_rate"][0], info["ema_decay"][0]
    before = ref.load_checkpoint(os.path.join(args.bake, "before.ckpt"))
    falcor = ref.load_checkpoint(os.path.join(args.bake, "after.ckpt"))
    trained = ref.load_checkpoint(os.path.join(args.out, "trained.ckpt"))
    # The fp16 copies the step computed with, as the replay dumped them (train_working.bin). With the
    # bake's model they are Falcor's (before_working.bin): the masters in fp16 after loading a
    # checkpoint, zero in a fresh model, whose first step therefore changes nothing but the weight decay.
    working = load_working(os.path.join(args.out, "train_working.bin"))
    half = dict(zip(ref.BLOCKS, working))
    baked = os.path.join(args.bake, "before_working.bin")
    if os.path.exists(baked):
        same = all(np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in zip(working, load_working(baked)))
        check("training copies equal the bake's before_working.bin (mismatching blocks)", 0 if same else 1, 0)
    print(f"training step: tiles of {tile}, learning rate {learning_rate:g}, EMA decay {ema_decay:g}")

    nets = {"diff": (graphs.DIFF, ref.DIFF_LAYERS, ref.DIFF_POS), "spec": (graphs.SPEC, ref.SPEC_LAYERS, ref.SPEC_POS)}
    prim_gradient = np.zeros(ref.PRIM_GRID_ELEMS, dtype=np.float64)
    cls_gradient = np.zeros(ref.CLS_GRID_ELEMS, dtype=np.float64)
    weight_gradients = {}
    for name, (spec, layers, pos_offset) in nets.items():
        if name not in info:
            continue
        size, tiles, rows_total = (int(v) for v in info[name])
        prefix = os.path.join(args.out, f"train_{name}")
        rows, x_tiles = tiles_to_rows(prefix + "_x.bin", spec.input_dim, tile, size)
        _, target_tiles = tiles_to_rows(prefix + "_target.bin", graphs.OUTPUT_PAD, tile, size)
        gx_rows, _ = tiles_to_rows(prefix + "_gx.bin", spec.input_dim, tile, size)
        gw = np.fromfile(prefix + "_gw.bin", dtype=np.float32)
        to_batch = tile / rows_total
        print(f"{name}: {size} samples in {tiles} tiles, loss normalised by {rows_total} rows")

        # 1. Encodings.
        pos = rows[:, pos_offset:pos_offset + 3]
        check("primary encoding in the input tiles", relative_l2(rows[:, :ref.PRIM_ENC_DIM], ref.prim_encode(half["prim_grid"], pos)), 2e-3)
        if name == "spec":
            cls_pos, cls_scale, cls_weight = ref.spec_clusters(rows)
            expected = ref.cls_encode(half["cls_grid"], cls_pos, cls_scale, cls_weight)
            check("cluster encoding in the input tiles",
                  relative_l2(rows[:, ref.SPEC_CLS_ENC:ref.SPEC_CLS_ENC + ref.CLS_ENC_DIM], expected), 2e-3)
        mask = target_tiles[:, graphs.MASK_CHANNEL, :].reshape(-1)
        check("sample weights in the target tiles", float(np.abs(mask - (np.arange(mask.size) < size)).max()), 0)

        # 2. The training graph, tile by tile, in float64 with autograd.
        kernels = [torch.from_numpy(k.astype(np.float64)) for k in ref.split_layers(half[name], layers)]
        expected_gw = np.zeros(gw.shape, dtype=np.float64)
        expected_gx = np.zeros((tiles, spec.input_dim, tile), dtype=np.float64)
        for t in range(tiles):
            x = torch.from_numpy(x_tiles[t].astype(np.float64)).reshape(1, spec.input_dim, tile, 1)
            target = torch.from_numpy(target_tiles[t].astype(np.float64)).reshape(1, graphs.OUTPUT_PAD, tile, 1)
            gradients = graphs.reference_gradients(spec, kernels, x, target)
            expected_gw += np.concatenate([g.numpy().reshape(-1) for g in gradients[:-1]]) * to_batch
            expected_gx[t] = gradients[-1].numpy().reshape(spec.input_dim, tile)
        expected_gx_rows = np.ascontiguousarray(expected_gx.transpose(0, 2, 1)).reshape(-1, spec.input_dim)[:size]
        limit = 5e-2 if args.npu else 2e-3
        check("gradient of the weights", relative_l2(gw, expected_gw), limit)
        check("gradient of the inputs", relative_l2(gx_rows, expected_gx_rows), limit)
        weight_gradients[name] = gw

        # 3. What the grids should receive from this network: its input gradients, as dumped.
        scale = to_batch * ref.GRID_GRADIENT_SCALE
        prim_gradient += ref.prim_scatter(gx_rows[:, :ref.PRIM_ENC_DIM], pos) * scale
        if name == "spec":
            cls_gradient += ref.cls_scatter(gx_rows[:, ref.SPEC_CLS_ENC:ref.SPEC_CLS_ENC + ref.CLS_ENC_DIM], cls_pos, cls_scale, cls_weight) * scale

    print("grids:")
    grid_gradients = {}
    for name, expected in (("prim", prim_gradient), ("cls", cls_gradient)):
        got = np.fromfile(os.path.join(args.out, f"train_{name}_grad.bin"), dtype=np.int32)
        grid_gradients[name + "_grid"] = got
        # Every sample's contribution is rounded to a fixed-point unit before it is added.
        check(f"{name} grid gradient", relative_l2(got, expected), 2e-3,
              f"{np.count_nonzero(got)} of {got.size} elements touched, largest {np.abs(got).max() / ref.GRID_GRADIENT_SCALE:.3g}")

    # 4. The optimizers, from the replayer's own gradients.
    print("optimizers (reference step from the replayer's gradients):")
    for name in ref.BLOCKS:
        is_grid = name.endswith("_grid")
        if is_grid:
            gradient = grid_gradients[name].astype(np.float32) / np.float32(ref.GRID_GRADIENT_SCALE)
            if name == "cls_grid" and "spec" not in info:
                continue
        else:
            if name not in weight_gradients:
                continue
            gradient = weight_gradients[name]
        master = before[name + "_master"]
        zeros = np.zeros(master.shape, dtype=np.float32)
        new_master, _, _, _, _, new_ema = ref.adam_ema_step(
            master, half[name], gradient, zeros, zeros, np.zeros(master.shape, dtype=np.uint32), before[name + "_ema"],
            step=1, learning_rate=learning_rate, ema_decay=ema_decay, loss_scale=graphs.LOSS_SCALE, matrix_params=not is_grid)
        # The GPU's pow and sqrt are a few float32 steps off, which moves a master by up to 2e-7.
        check(f"{name} masters", float(np.abs(trained[name + "_master"] - new_master).max()), 1e-6)
        # The fp16 copies: within one fp16 step of the reference, or within those 2e-7 for the
        # smallest weights, where fp16 steps are finer than that.
        expected = new_ema.astype(np.float64)
        tolerance = np.maximum(np.abs(expected) * 2.0 ** -10, 5e-7)
        check(f"{name} EMA copies (error / tolerance)", float((np.abs(trained[name + "_ema"].astype(np.float64) - expected) / tolerance).max()), 1)

    # 5. Against Falcor.
    print("against Falcor (after.ckpt):")
    for name in ref.BLOCKS:
        ours = trained[name + "_master"].astype(np.float64) - before[name + "_master"]
        theirs = falcor[name + "_master"].astype(np.float64) - before[name + "_master"]
        moved_ours, moved_theirs = ours != 0, theirs != 0
        if not moved_ours.any() and not moved_theirs.any():
            print(f"  {name}: no weight moved, here or in Falcor (a fresh model's first step)")
            continue
        both = moved_ours & moved_theirs
        same_set = both.sum() / max((moved_ours | moved_theirs).sum(), 1)
        same_direction = (np.sign(ours[both]) == np.sign(theirs[both])).mean() if both.any() else 1.0
        cosine = float(ours @ theirs / max(np.linalg.norm(ours) * np.linalg.norm(theirs), 1e-30))
        print(f"  {name}: {moved_theirs.sum()} weights moved in Falcor, {moved_ours.sum()} here; the same weights {same_set:.4f}, "
              f"of those in the same direction {same_direction:.4f}; cosine of the two updates {cosine:.4f}")
        check(f"{name} update differs from Falcor's (1 - cosine)", 1.0 - cosine, 0.1)

    print("PASS" if failures == 0 else f"FAIL: {failures} checks")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

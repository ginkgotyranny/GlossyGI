"""NCR's model without tiny-cuda-nn: the reference the phone runtime is written against.

Everything NRModel (Source/RenderPasses/NeuralRadiosity/Model) does, in numpy:
  - the NRCK checkpoint format,
  - the layout of the two network inputs (NetworkParams.slang),
  - the primary and cluster hash-grid lookups and their gradient scatter (GPU compute on the phone),
  - the MLPs (NPU graphs on the phone, see ncr_npu_graphs.py),
  - tiny-cuda-nn's Adam and EMA steps, with the differences between MLP and grid parameters.
verify_tcnn.py checks each piece against data dumped from the CUDA implementation.
"""
import os
import struct

import numpy as np

# ---------------------------------------------------------------------------------------------
# NCRConfig.h
LOG2_HASHMAP_SIZE = 18
HASHMAP_SIZE = 1 << LOG2_HASHMAP_SIZE
PRIM_LEVELS, PRIM_FEATURES, PRIM_BASE_RES = 6, 2, 32
CLS_LEVELS, CLS_FEATURES, CLS_BASE_RES = 6, 4, 16
NUM_CLUSTERS = 4
INTERP_RATIO = 1.0
OUTPUT_DIM, OUTPUT_PAD = 4, 16
BATCH_ALIGN = 256                               # kModelBatchAlign

# Grid gradients are summed on the GPU as fixed-point integers (Vulkan has no floating-point atomic
# add): units of 2^-24, the smallest non-zero fp16 value, which is what tiny-cuda-nn's fp16 sums
# resolve. It matters because Adam skips a grid entry only when its gradient is exactly zero and
# otherwise moves it by about the learning rate whatever the magnitude: in a real step 7% (primary)
# and 15% (cluster) of the touched entries have gradients below 2^-20. int32 then holds +-128, against
# sums of at most 0.5 in that step.
GRID_GRADIENT_SCALE = float(1 << 24)

# NetworkParams.slang: offsets into one input row.
PRIM_ENC_DIM = PRIM_LEVELS * PRIM_FEATURES      # 12
CLS_ENC_DIM = CLS_FEATURES * NUM_CLUSTERS       # 16
DIFF_INPUT_DIM, SPEC_INPUT_DIM = 32, 64
DIFF_POS = PRIM_ENC_DIM                         # 12: pos, dir, normal, albedo, roughness follow
SPEC_CLS_ENC = PRIM_ENC_DIM                     # 12
SPEC_POS = PRIM_ENC_DIM + CLS_ENC_DIM           # 28
SPEC_CLS_POS = SPEC_POS + 3 * 5 + 1             # 44: after pos, dir, normal, albedo, roughness, env color
SPEC_CLS_SCALE = SPEC_CLS_POS + NUM_CLUSTERS * 3  # 56
SPEC_CLS_WEIGHT = SPEC_CLS_SCALE + NUM_CLUSTERS   # 60

DIFF_LAYERS = [(64, 32), (64, 64), (64, 64), (16, 64)]             # (Cout, Cin), tiny-cuda-nn order
SPEC_LAYERS = [(64, 64), (64, 64), (64, 64), (64, 64), (16, 64)]


# ---------------------------------------------------------------------------------------------
# Grids
def level_entries(base_res, level):
    res1 = (base_res << level) + 1
    return min(res1 ** 3, HASHMAP_SIZE)


def grid_entries(base_res, levels):
    return sum(level_entries(base_res, i) for i in range(levels))


PRIM_GRID_ELEMS = grid_entries(PRIM_BASE_RES, PRIM_LEVELS) * PRIM_FEATURES   # 2,693,314
CLS_GRID_ELEMS = grid_entries(CLS_BASE_RES, CLS_LEVELS) * CLS_FEATURES       # 4,357,704


def level_corners(pos, base_res, level):
    """pos (N, 3) float32 -> entries (N, 8) int64 into the whole grid, weights (N, 8) float32.
    `level` is an int, or an int array (N,) to pick a level per sample (cluster grid)."""
    level = np.broadcast_to(np.asarray(level, dtype=np.int64), pos.shape[:1])
    res = (base_res << level).astype(np.int64)
    res1 = res + 1
    size = np.minimum(res1 ** 3, HASHMAP_SIZE)
    sizes = np.array([level_entries(base_res, i) for i in range(8)], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(sizes)])[level]

    grid = pos.astype(np.float32) * res[:, None].astype(np.float32)
    # float -> unsigned as CUDA converts it: truncation, saturating at 0 for negative coordinates.
    # Cluster positions do leave the unit cube; the weights then extrapolate from the border cell.
    base = np.clip(np.trunc(grid), 0, 4294967295).astype(np.int64)
    frac = grid - base.astype(np.float32)
    entries = np.empty((pos.shape[0], 8), dtype=np.int64)
    weights = np.empty((pos.shape[0], 8), dtype=np.float32)
    for c in range(8):
        o = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1], dtype=np.int64)
        p = base + o
        # The CUDA code tests "level size > table size" to choose its prime hash, but the level size
        # is already clamped to the table size, so every level takes the dense index modulo its
        # size (in 32-bit arithmetic). Checkpoints are trained with exactly this indexing.
        dense = ((res1 * p[:, 0] + p[:, 1]) * res1 + p[:, 2]) & 0xffffffff
        entries[:, c] = offset + dense % size
        w = np.ones(pos.shape[0], dtype=np.float32)
        for a in range(3):
            w = w * (frac[:, a] if o[a] else (np.float32(1) - frac[:, a]))
        weights[:, c] = w
    return entries, weights


def prim_encode(grid, pos):
    """HashGrid::forwardKernel. grid (entries, 2), pos (N, 3) -> (N, 12) float32."""
    grid = grid.reshape(-1, PRIM_FEATURES).astype(np.float32)
    out = np.empty((pos.shape[0], PRIM_ENC_DIM), dtype=np.float32)
    for level in range(PRIM_LEVELS):
        entries, weights = level_corners(pos, PRIM_BASE_RES, level)
        out[:, level * PRIM_FEATURES:(level + 1) * PRIM_FEATURES] = (grid[entries] * weights[:, :, None]).sum(axis=1)
    return out


def prim_scatter(grad_enc, pos, loss_weight=None):
    """HashGrid::backwardKernel. grad_enc (N, 12): dL/d(encoding) -> gradient (entries * 2,) float64."""
    grad = np.zeros((PRIM_GRID_ELEMS // PRIM_FEATURES, PRIM_FEATURES), dtype=np.float64)
    for level in range(PRIM_LEVELS):
        entries, weights = level_corners(pos, PRIM_BASE_RES, level)
        g = grad_enc[:, level * PRIM_FEATURES:(level + 1) * PRIM_FEATURES].astype(np.float64)
        for c in range(8):
            np.add.at(grad, entries[:, c], g * weights[:, c:c + 1])
    return grad.reshape(-1)


def cluster_levels(scale):
    """HashGridInterp: the two levels bracketing the cluster radius, and their weights.
    scale (N,) float32 -> upper, lower (int), upper_weight, lower_weight (float32)."""
    voxel = np.array([INTERP_RATIO / (CLS_BASE_RES << i) for i in range(CLS_LEVELS)], dtype=np.float32)
    scale = scale.astype(np.float32)
    n = scale.shape[0]
    # Finer level: the first level whose voxel is smaller than the radius.
    greater = scale[:, None] > voxel[None, :]
    level = np.where(greater.any(axis=1), greater.argmax(axis=1), CLS_LEVELS)
    upper = np.clip(level, 0, CLS_LEVELS - 1)
    inner = (level > 0) & (level < CLS_LEVELS)
    coarser, finer = voxel[np.clip(level - 1, 0, CLS_LEVELS - 1)], voxel[upper]
    upper_weight = np.where(level == 0, 0.0, np.where(level == CLS_LEVELS, 1.0,
                            (coarser - scale) / np.where(inner, coarser - finer, 1.0))).astype(np.float32)
    # Coarser level: the last level whose voxel is larger than the radius.
    less = scale[:, None] < voxel[None, :]
    level = np.where(less.any(axis=1), CLS_LEVELS - 1 - less[:, ::-1].argmax(axis=1), -1)
    lower = np.clip(level, 0, CLS_LEVELS - 1)
    inner = (level >= 0) & (level < CLS_LEVELS - 1)
    coarser, finer = voxel[lower], voxel[np.clip(level + 1, 0, CLS_LEVELS - 1)]
    lower_weight = np.where(level == -1, 1.0, np.where(level == CLS_LEVELS - 1, 0.0,
                            (scale - finer) / np.where(inner, coarser - finer, 1.0))).astype(np.float32)
    assert upper.shape == (n,)
    return upper, lower, upper_weight, lower_weight


def cls_encode(grid, cls_pos, cls_scale, cls_weight):
    """HashGridInterp::forwardKernel. grid (entries, 4); cls_pos (N, 4, 3); cls_scale, cls_weight (N, 4)
    -> (N, 16) float32."""
    grid = grid.reshape(-1, CLS_FEATURES).astype(np.float32)
    n = cls_pos.shape[0]
    out = np.empty((n, CLS_ENC_DIM), dtype=np.float32)
    for k in range(NUM_CLUSTERS):
        upper, lower, upper_weight, lower_weight = cluster_levels(cls_scale[:, k])
        pos = cls_pos[:, k].astype(np.float32)
        eu, wu = level_corners(pos, CLS_BASE_RES, upper)
        el, wl = level_corners(pos, CLS_BASE_RES, lower)
        vu = (grid[eu] * wu[:, :, None]).sum(axis=1)
        vl = (grid[el] * wl[:, :, None]).sum(axis=1)
        value = (vl * lower_weight[:, None] + vu * upper_weight[:, None]) * cls_weight[:, k:k + 1].astype(np.float32)
        out[:, k * CLS_FEATURES:(k + 1) * CLS_FEATURES] = value
    return out


def cls_scatter(grad_enc, cls_pos, cls_scale, cls_weight):
    """HashGridInterp::backwardKernel -> gradient (entries * 4,) float64."""
    grad = np.zeros((CLS_GRID_ELEMS // CLS_FEATURES, CLS_FEATURES), dtype=np.float64)
    for k in range(NUM_CLUSTERS):
        upper, lower, upper_weight, lower_weight = cluster_levels(cls_scale[:, k])
        pos = cls_pos[:, k].astype(np.float32)
        g = grad_enc[:, k * CLS_FEATURES:(k + 1) * CLS_FEATURES].astype(np.float64) * cls_weight[:, k:k + 1].astype(np.float64)
        eu, wu = level_corners(pos, CLS_BASE_RES, upper)
        el, wl = level_corners(pos, CLS_BASE_RES, lower)
        for c in range(8):
            np.add.at(grad, eu[:, c], g * upper_weight[:, None] * wu[:, c:c + 1])
            np.add.at(grad, el[:, c], g * lower_weight[:, None] * wl[:, c:c + 1])
    return grad.reshape(-1)


def spec_clusters(rows):
    """Cluster position, radius and weight columns of SpecNet input rows (N, 64)."""
    cls_pos = rows[:, SPEC_CLS_POS:SPEC_CLS_POS + NUM_CLUSTERS * 3].reshape(-1, NUM_CLUSTERS, 3)
    cls_scale = rows[:, SPEC_CLS_SCALE:SPEC_CLS_SCALE + NUM_CLUSTERS]
    cls_weight = rows[:, SPEC_CLS_WEIGHT:SPEC_CLS_WEIGHT + NUM_CLUSTERS]
    return cls_pos, cls_scale, cls_weight


# ---------------------------------------------------------------------------------------------
# Checkpoints ("NRCK" v2, Model.cu): header, 4 float32 master blocks, 4 fp16 EMA blocks, each in the
# order primary grid, cluster grid, DiffNet, SpecNet.
BLOCKS = ["prim_grid", "cls_grid", "diff", "spec"]


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, *counts = struct.unpack_from("<4sI4I", data, 0)
    if magic != b"NRCK" or version != 2:
        raise ValueError(f"{path}: not an NRCK v2 checkpoint")
    expected = [PRIM_GRID_ELEMS, CLS_GRID_ELEMS, sum(o * i for o, i in DIFF_LAYERS), sum(o * i for o, i in SPEC_LAYERS)]
    if counts != expected:
        raise ValueError(f"{path}: element counts {counts} differ from this configuration {expected}")
    offset = struct.calcsize("<4sI4I")
    checkpoint = {}
    for name, count in zip(BLOCKS, counts):
        checkpoint[name + "_master"] = np.frombuffer(data, dtype=np.float32, count=count, offset=offset).copy()
        offset += count * 4
    for name, count in zip(BLOCKS, counts):
        checkpoint[name + "_ema"] = np.frombuffer(data, dtype=np.float16, count=count, offset=offset).copy()
        offset += count * 2
    assert offset == len(data), "trailing bytes in the checkpoint"
    return checkpoint


class Dump:
    """Arrays and scalars written by a Falcor run with NCR_DUMP_DIR set (DebugDump in Model/Model.cu)."""
    DTYPES = {"f16": np.float16, "f32": np.float32, "u32": np.uint32}

    def __init__(self, folder):
        self.folder = folder
        self.arrays, self.scalars = {}, {}
        with open(os.path.join(folder, "manifest.txt")) as f:
            for line in f:
                parts = line.split()
                if parts and parts[0] == "array":
                    self.arrays[parts[1]] = (parts[2], int(parts[3]), int(parts[4]))
                elif parts and parts[0] == "scalar":
                    self.scalars[parts[1]] = float(parts[2])

    def __getitem__(self, name):
        dtype, rows, cols = self.arrays[name]
        data = np.fromfile(os.path.join(self.folder, name + ".bin"), dtype=self.DTYPES[dtype])
        assert data.size == rows * cols, name
        return data.reshape(rows, cols) if cols > 1 else data


def split_layers(params, layers):
    """Flat MLP parameters -> conv kernels (Cout, Cin, 1, 1), the NPU weight layout. tiny-cuda-nn stores
    each matrix row-major with rows = outputs, which is the same memory order."""
    kernels, offset = [], 0
    for cout, cin in layers:
        kernels.append(np.asarray(params[offset:offset + cout * cin]).reshape(cout, cin, 1, 1))
        offset += cout * cin
    assert offset == len(params)
    return kernels


def rows_to_planar(rows):
    """(N, C) sample rows -> (1, C, N, 1), the channel-planar tensor of the NPU graphs."""
    return np.ascontiguousarray(rows.T).reshape(1, rows.shape[1], rows.shape[0], 1)


def planar_to_rows(tensor):
    return np.ascontiguousarray(tensor.reshape(tensor.shape[1], -1).T)


# ---------------------------------------------------------------------------------------------
# Optimizers (tiny-cuda-nn EMA(Adam), as configured in Model.cu)
ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON = 0.9, 0.999, 1e-8
ADAM_L2_REG = 1e-8   # matrix (MLP) parameters only


def adam_ema_step(master, weights_half, gradient, first_moment, second_moment, param_steps, ema, *,
                  step, learning_rate, ema_decay, loss_scale, matrix_params):
    """One optimizer step. `step` is the optimizer's step count after this step (1 for the first).
    matrix_params: True for the MLPs, whose every weight is updated with an L2 term. False for the
    grids, where an entry with a zero gradient is skipped altogether and has its own step count.
    Returns the new (master, weights_half, first_moment, second_moment, param_steps, ema)."""
    f = np.float32
    master, first_moment, second_moment = master.astype(f), first_moment.astype(f), second_moment.astype(f)
    g = gradient.astype(f) / f(loss_scale)
    g = np.where(np.isfinite(g), g, f(0))
    update = np.ones(master.shape, dtype=bool) if matrix_params else g != 0
    if matrix_params:
        g = g + f(ADAM_L2_REG) * master
    m = np.where(update, f(ADAM_BETA1) * first_moment + f(1 - ADAM_BETA1) * g, first_moment).astype(f)
    v = np.where(update, f(ADAM_BETA2) * second_moment + f(1 - ADAM_BETA2) * g * g, second_moment).astype(f)
    steps = param_steps + update.astype(param_steps.dtype)
    t = np.maximum(steps, 1).astype(f)
    lr = f(learning_rate) * np.sqrt(f(1) - np.power(f(ADAM_BETA2), t)) / (f(1) - np.power(f(ADAM_BETA1), t))
    new_master = np.where(update, master - lr / (np.sqrt(v) + f(ADAM_EPSILON)) * m, master).astype(f)
    new_half = np.where(update, new_master.astype(np.float16), weights_half)
    debias_old = f(1 - ema_decay ** (step - 1))
    debias_new = f(1 / (1 - ema_decay ** step))
    new_ema = ((ema.astype(f) * f(ema_decay) * debias_old + new_half.astype(f) * f(1 - ema_decay)) * debias_new).astype(np.float16)
    return new_master, new_half, m, v, steps, new_ema

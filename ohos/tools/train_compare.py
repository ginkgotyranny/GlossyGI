#!/usr/bin/env python
"""Compares N frames of training in the replayer with N frames of training in Falcor.

    python ohos/tools/train_compare.py <bake of frame 0> <replay output dir> <bake of frame N-1>

Falcor is run twice from the same checkpoint (loadModelPath), once baking frame 0 and once baking
frame N-1. The replayer starts from the first bake's before.ckpt and replays its frame N times
(--frames N): the same scene and camera, the same sample sequence (the frame counter seeds it), its
own networks and optimizers. What it arrives at is compared with what Falcor arrived at:

  - the weights: trained.ckpt against the second bake's after.ckpt, as the update each made to the
    checkpoint both started from;
  - the image: the final color of the last frame against Falcor's of frame N-1.

The two runs are not expected to be equal: every step differs a little (fp16 rounding, the order
of the sums), and training amplifies that. They should have learned the same thing.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "npu"))
import bake_check  # noqa: E402
import ncr_reference as ref  # noqa: E402

# Accepted: the cosine between the two updates of a weight block, and the PSNR of the final image.
MIN_COSINE = 0.9
MIN_PSNR = 35.0


def color_texture(bake):
    """Id of the texture bound as gColor (the pass's output)."""
    for call in bake["calls"].values():
        for (path, index), record in call["binds"].items():
            if path == "gColor" and record[0] == "texture":
                return int(record[1])
    raise SystemExit("the bake has no gColor texture")


def image(path):
    return np.clip(np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3], 0.0, 1.0).astype(np.float64)


def psnr(a, b):
    return 10.0 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-30))


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    first_dir, out_dir, last_dir = sys.argv[1:]
    first, last = bake_check.load(first_dir), bake_check.load(last_dir)
    first_frame, last_frame = first["frame"]["frameCount"], last["frame"]["frameCount"]
    print(f"{last_frame - first_frame + 1} frames: Falcor frames {first_frame} to {last_frame}")
    failures = 0

    before = ref.load_checkpoint(os.path.join(first_dir, "before.ckpt"))
    falcor = ref.load_checkpoint(os.path.join(last_dir, "after.ckpt"))
    ours = ref.load_checkpoint(os.path.join(out_dir, "trained.ckpt"))
    print("weights (what each run changed in the checkpoint both started from):")
    for name in ref.BLOCKS:
        for kind in ("master", "ema"):
            key = f"{name}_{kind}"
            start = before[key].astype(np.float64)
            a, b = ours[key].astype(np.float64) - start, falcor[key].astype(np.float64) - start
            cosine = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
            ok = cosine >= MIN_COSINE
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'} {key:16s} update norm {np.linalg.norm(a):9.4f} here, {np.linalg.norm(b):9.4f} in Falcor; "
                  f"cosine {cosine:.4f}; difference / update {np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30):.3f}")

    reference = image(os.path.join(last_dir, "data", f"texture{color_texture(last)}.bin"))
    replayed = image(os.path.join(out_dir, f"texture{color_texture(first)}.bin"))
    untrained = image(os.path.join(first_dir, "data", f"texture{color_texture(first)}.bin"))
    value = psnr(replayed, reference)
    ok = value >= MIN_PSNR
    failures += 0 if ok else 1
    print(f"image after the last frame: PSNR {value:.1f} dB against Falcor's ({'ok' if ok else 'FAIL'}, limit {MIN_PSNR:.0f} dB); "
          f"for scale, Falcor's first frame against its last: {psnr(untrained, reference):.1f} dB")
    print("PASS" if failures == 0 else f"FAIL: {failures} checks")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

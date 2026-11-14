#!/usr/bin/env python
"""The image the runtime shows against the one Falcor's graph outputs.

    python ohos/tools/display_check.py <display.bin> <falcor.png> [<out.png>]

<display.bin> is what the runtime's Display pass wrote (8-bit RGBA rows, the replayer's output
directory or the app's), <falcor.png> the ToneMapper output Falcor captured of the same frame
(scripts/ncr_smoke.py, smoke.ToneMapper.dst.<n>.png). The size is taken from the PNG. Prints the
largest difference per channel, how many pixels differ, and the PSNR; with <out.png> the runtime's
image is also written as a PNG.

The comparison decides on the PSNR (>= 40 dB): the two images come from different GPUs, or from the
same GPU with the networks on different processors, and the tone curve amplifies small HDR differences
in dark regions; a wrong tone mapping, colour order or gamma costs far more than that.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from png_stats import read_png  # noqa: E402
from raw_to_png import write_png  # noqa: E402

MIN_PSNR = 40.0


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    width, height, channels, rows = read_png(sys.argv[2])
    falcor = np.frombuffer(b"".join(bytes(r) for r in rows), dtype=np.uint8).reshape(height, width, channels)[:, :, :3]
    ours = np.fromfile(sys.argv[1], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    if len(sys.argv) > 3:
        write_png(sys.argv[3], np.ascontiguousarray(ours))
    difference = np.abs(ours.astype(np.int32) - falcor.astype(np.int32))
    error = np.mean((difference / 255.0) ** 2)
    psnr = 10.0 * np.log10(1.0 / max(error, 1e-30))
    per_pixel = difference.max(axis=2)
    print(f"{width}x{height}: largest difference per channel {difference.reshape(-1, 3).max(axis=0).tolist()} (of 255); "
          f"pixels differing by more than 1: {100.0 * np.mean(per_pixel > 1):.3f}%, by more than 8: {100.0 * np.mean(per_pixel > 8):.3f}%")
    print(f"mean {ours.mean():.2f} here, {falcor.mean():.2f} in Falcor; PSNR {psnr:.1f} dB")
    ok = psnr >= MIN_PSNR
    print("PASS" if ok else f"FAIL: PSNR below {MIN_PSNR:.0f} dB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

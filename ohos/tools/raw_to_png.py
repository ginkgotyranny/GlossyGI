#!/usr/bin/env python
"""Writes a raw RGBA32F image (as the bake and the replayer dump them) as an 8-bit sRGB PNG.

    python ohos/tools/raw_to_png.py <in.bin> <width> <height> <out.png> [exposure]

Standard library only. Values are multiplied by `exposure` (default 1), clipped to [0, 1] and
sRGB-encoded with a 2.2 gamma; not a number shows as magenta.
"""
import struct
import sys
import zlib

import numpy as np


def write_png(path, rgb):
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    width, height = int(sys.argv[2]), int(sys.argv[3])
    exposure = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    image = np.fromfile(sys.argv[1], dtype=np.float32).reshape(height, width, 4)[:, :, :3] * exposure
    bad = ~np.isfinite(image).all(axis=2)
    out = (np.clip(np.nan_to_num(image), 0.0, 1.0) ** (1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)
    out[bad] = (255, 0, 255)
    write_png(sys.argv[4], out)
    print(f"{sys.argv[4]}: {width}x{height}, mean {np.nan_to_num(image).mean():.4f}, {int(bad.sum())} non-finite pixels")


if __name__ == "__main__":
    main()

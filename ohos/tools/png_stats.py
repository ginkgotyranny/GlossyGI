#!/usr/bin/env python
"""Mean and saturation statistics of 8-bit PNG files, with no dependency beyond the standard library.

    python ohos/tools/png_stats.py <a.png> [<b.png> ...]

With two or more files, also prints the mean absolute difference of each to the first. Used to compare
frames of the same scene from different backends or runtimes (Falcor D3D12 / Vulkan, the phone).
`read_png(path)` returns (width, height, channels, rows of bytes).
"""
import struct
import sys
import zlib


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG"
    pos, idat, header = 8, b"", None
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    width, height, depth, color, _, _, interlace = header
    assert depth == 8 and interlace == 0 and color in (2, 6), f"{path}: only 8-bit RGB / RGBA, not interlaced"
    channels = 3 if color == 2 else 4
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, previous = [], bytearray(stride)
    for y in range(height):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        previous = line
    return width, height, channels, rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    first = None
    for path in sys.argv[1:]:
        width, height, channels, rows = read_png(path)
        rgb = [bytes(row[i] for i in range(len(row)) if i % channels < 3) for row in rows]
        total = sum(sum(row) for row in rgb)
        white = sum(1 for row in rgb for x in range(width) if row[3 * x:3 * x + 3] == b"\xff\xff\xff")
        line = f"{path}: {width}x{height}, mean {total / (width * height * 3) / 255:.4f}, white pixels {100.0 * white / (width * height):.3f}%"
        if first is None:
            first = rgb
        elif len(first) == len(rgb) and len(first[0]) == len(rgb[0]):
            difference = sum(abs(a - b) for r0, r1 in zip(first, rgb) for a, b in zip(r0, r1))
            line += f", mean |difference| to the first {difference / (width * height * 3) / 255:.4f}"
        print(line)


if __name__ == "__main__":
    main()

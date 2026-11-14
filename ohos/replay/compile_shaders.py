#!/usr/bin/env python
"""Compiles the runtime's own shaders (ohos/replay/shaders) to SPIR-V.

    python ohos/replay/compile_shaders.py <output dir>

The one list of the runtime's shaders: build_pc.bat, run_device.sh and the app's asset sync use it.
The compiler is Falcor's slangc (external/packman/slang); SLANGC overrides the path.
Standard library only.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SLANGC = os.environ.get("SLANGC", os.path.join(REPO, "external", "packman", "slang", "bin", "slangc.exe"))

# (output name, source, extra arguments)
SHADERS = [
    ("EncodeDiff", "EncodeInputs.cs.slang", ["-DSPECULAR=0"]),
    ("EncodeSpec", "EncodeInputs.cs.slang", ["-DSPECULAR=1"]),
    ("DecodeOutputs", "DecodeOutputs.cs.slang", []),
    ("EncodeTargets", "EncodeTargets.cs.slang", []),
    ("ScatterDiff", "ScatterGradients.cs.slang", ["-DSPECULAR=0"]),
    ("ScatterSpec", "ScatterGradients.cs.slang", ["-DSPECULAR=1"]),
    ("GridOptimizer", "GridOptimizer.cs.slang", []),
    ("PrefixSum", "PrefixSum.cs.slang", []),
    ("Display", "Display.cs.slang", []),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    shaders = os.path.join(HERE, "shaders")
    for name, source, extra in SHADERS:
        command = [SLANGC, os.path.join(shaders, source), *extra, "-I", shaders, "-target", "spirv", "-entry", "main", "-stage", "compute",
                   "-o", os.path.join(out, name + ".spv")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout + result.stderr)
            sys.exit(f"slangc failed on {source} ({name})")
    print(f"{len(SHADERS)} shaders compiled to {out}")


if __name__ == "__main__":
    main()

"""The numbers of Source/RenderPasses/NeuralRadiosity/NCRConfig.h the port's Python side depends on.

NCRConfig.h is the single source for the desktop (C++, CUDA, Slang). The NPU graphs and the reference
model read the MLP sizes from it too, so a change of preset (e.g. SpecNet with three hidden layers
instead of four) needs no edit here: export the models again and bake again. The phone runtime takes
the same sizes from the checkpoint a bake carries (replay_nn.h).
"""
import os
import re

PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "Source", "RenderPasses",
                                     "NeuralRadiosity", "NCRConfig.h"))


def load(path=PATH):
    """Every `#define NCR_<NAME> <integer>` of the header, as a dict of ints."""
    values = {}
    with open(path) as f:
        for line in f:
            match = re.match(r"\s*#define\s+(NCR_\w+)\s+(\d+)\s*(//.*)?$", line)
            if match:
                values[match.group(1)] = int(match.group(2))
    return values


_values = load()
DIFF_NEURONS = _values["NCR_DIFF_N_NEURONS"]
DIFF_HIDDEN_LAYERS = _values["NCR_DIFF_N_HIDDEN_LAYERS"]
SPEC_NEURONS = _values["NCR_SPEC_N_NEURONS"]
SPEC_HIDDEN_LAYERS = _values["NCR_SPEC_N_HIDDEN_LAYERS"]

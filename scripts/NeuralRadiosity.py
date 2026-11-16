import os
from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("NeuralRadiosity")

    # Launcher option (run.bat / run.sh): NCR_MIRROR_ALBEDO=1 or 0 sets the pass's "Mirror Albedo" box.
    # Unset: the pass's default, which is on.
    props = {}
    if "NCR_MIRROR_ALBEDO" in os.environ:
        props["enableMirrorAlbedo"] = os.environ["NCR_MIRROR_ALBEDO"] == "1"

    NeuralRadiosity = createPass("NeuralRadiosity", props)
    g.addPass(NeuralRadiosity, "NeuralRadiosity")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    # OptixDenoiser = createPass("OptixDenoiser", {'enabled': False})
    # g.addPass(OptixDenoiser, "Denoiser")

    g.addEdge("NeuralRadiosity.color", "ToneMapper.src")
    # g.addEdge("NeuralRadiosity.color", "Denoiser.color")
    # g.addEdge("Denoiser.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

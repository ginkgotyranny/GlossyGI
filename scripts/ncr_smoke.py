"""Headless smoke test for the NeuralRadiosity pass.

Run with:
    Mogwai.exe --headless --script scripts/ncr_smoke.py --scene media/scenes/<name>/scene-v4.pyscene

Environment variables (all optional):
    NCR_SMOKE_OUT      output directory for captured frames (default: ncr_smoke_out)
    NCR_SMOKE_FRAMES   comma-separated frame indices to capture (default: 30,90)
    NCR_SMOKE_MODE     render mode property passed to the pass, if supported
    NCR_SMOKE_PROPS    extra pass properties as Python dict literal, e.g. "{'enableTemporalSpecReuse': True}"
    NCR_SMOKE_RES      WxH (default: 640x360)

The clock is paused and stepped manually so results are deterministic per frame index.
"""
import os
from falcor import *

# Absolute: Mogwai resolves a relative capture directory against the executable's directory, not
# the working directory, and aborts when the image writer cannot create its file there.
out_dir = os.path.abspath(os.environ.get("NCR_SMOKE_OUT", "ncr_smoke_out"))
frames = [int(f) for f in os.environ.get("NCR_SMOKE_FRAMES", "30,90").split(",")]
res = os.environ.get("NCR_SMOKE_RES", "640x360").lower().split("x")
props = eval(os.environ.get("NCR_SMOKE_PROPS", "{}"))
if "NCR_SMOKE_MODE" in os.environ:
    props["renderMode"] = os.environ["NCR_SMOKE_MODE"]

def render_graph_NeuralRadiosity():
    g = RenderGraph("NeuralRadiosity")
    nr = createPass("NeuralRadiosity", props)
    g.addPass(nr, "NeuralRadiosity")
    tm = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(tm, "ToneMapper")
    g.addEdge("NeuralRadiosity.color", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    g.markOutput("NeuralRadiosity.color")
    for extra in [o for o in os.environ.get("NCR_SMOKE_OUTPUTS", "").split(",") if o]:
        g.markOutput("NeuralRadiosity." + extra)
    return g

try:
    # Mogwai runs --script before --scene, so a synchronous render loop must load
    # the scene itself. Pass the scene via NCR_SMOKE_SCENE instead of --scene.
    scene = os.environ.get("NCR_SMOKE_SCENE")
    if scene:
        m.loadScene(scene)
    # NCR_SMOKE_GRAPH=<scripts/X.py>: run that render-graph script instead of the
    # NeuralRadiosity graph (the script adds its own graph via m.addGraph).
    graph_file = os.environ.get("NCR_SMOKE_GRAPH")
    if graph_file:
        exec(compile(open(graph_file).read(), graph_file, "exec"), globals())
    else:
        m.addGraph(render_graph_NeuralRadiosity())
    m.resizeFrameBuffer(int(res[0]), int(res[1]))
    m.ui = False
    m.clock.framerate = 60
    m.clock.time = 0
    m.clock.pause()
    os.makedirs(out_dir, exist_ok=True)
    m.frameCapture.outputDir = out_dir
    m.frameCapture.baseFilename = "smoke"

    frame = 0
    for capture_frame in sorted(frames):
        while frame < capture_frame:
            frame += 1
            m.clock.frame = frame
            m.renderFrame()
        m.frameCapture.capture()
        print(f"[ncr_smoke] captured frame {frame}")
    print("[ncr_smoke] done")
    exit()
except NameError:
    pass

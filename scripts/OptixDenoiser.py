from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("OptixDenoiser")
    PathTracer = createPass("PathTracer", {
        'samplesPerPixel': 1,
        'maxDiffuseBounces': 16,
        'maxSpecularBounces': 16,
        'useRTXDI': True,
    })
    g.addPass(PathTracer, "PathTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")
    OptixDenoiser = createPass("OptixDenoiser", {'model': 'HDR'})
    g.addPass(OptixDenoiser, "Denoiser")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("GBufferRT.mvec", "PathTracer.mvec")

    g.addEdge("PathTracer.albedo",         "Denoiser.albedo")
    g.addEdge("GBufferRT.normW",          "Denoiser.normal")
    g.addEdge("GBufferRT.mvec",           "Denoiser.mvec")
    g.addEdge("PathTracer.color",          "AccumulatePass.input")
    g.addEdge("AccumulatePass.output",         "Denoiser.color")

    g.addEdge("Denoiser.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

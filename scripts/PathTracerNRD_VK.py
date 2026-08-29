from falcor import *


def render_graph_PathTracerNRD():
    g = RenderGraph("PathTracerNRD_VK")

    GBufferRT = createPass("GBufferRT", {"samplePattern": "Halton", "sampleCount": 32, "useAlphaTest": True})
    g.addPass(GBufferRT, "GBufferRT")

    PathTracer = createPass("PathTracer", {"samplesPerPixel": 1, "maxSurfaceBounces": 16, "useRTXDI": True})
    g.addPass(PathTracer, "PathTracer")

    # Reference path.
    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapperReference = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapperReference, "ToneMapperReference")

    # NRD path.
    NRDDenoiser = createPass("NRD", {"method": "ReblurDiffuseSpecular", "maxIntensity": 250.0, "worldSpaceMotion": True})
    g.addPass(NRDDenoiser, "NRDDenoiser")
    Compositor = createPass("ModulateIllumination", {"useResidualRadiance": False})
    g.addPass(Compositor, "Compositor")
    ToneMapperNRD = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapperNRD, "ToneMapperNRD")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")

    # Reference path graph.
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapperReference.src")

    # NRD path inputs.
    g.addEdge("PathTracer.nrdDiffuseRadianceHitDist", "NRDDenoiser.diffuseRadianceHitDist")
    g.addEdge("PathTracer.nrdSpecularRadianceHitDist", "NRDDenoiser.specularRadianceHitDist")
    g.addEdge("GBufferRT.mvecW", "NRDDenoiser.mvec")
    g.addEdge("GBufferRT.normWRoughnessMaterialID", "NRDDenoiser.normWRoughnessMaterialID")
    g.addEdge("GBufferRT.linearZ", "NRDDenoiser.viewZ")

    # NRD outputs -> compositor.
    g.addEdge("NRDDenoiser.filteredDiffuseRadianceHitDist", "Compositor.diffuseRadiance")
    g.addEdge("NRDDenoiser.filteredSpecularRadianceHitDist", "Compositor.specularRadiance")
    g.addEdge("PathTracer.nrdDiffuseReflectance", "Compositor.diffuseReflectance")
    g.addEdge("PathTracer.nrdSpecularReflectance", "Compositor.specularReflectance")
    g.addEdge("PathTracer.nrdEmission", "Compositor.emission")
    g.addEdge("PathTracer.nrdResidualRadianceHitDist", "Compositor.residualRadiance")
    g.addEdge("Compositor.output", "ToneMapperNRD.src")

    g.markOutput("ToneMapperNRD.dst")
    g.markOutput("ToneMapperReference.dst")

    return g


PathTracerNRD = render_graph_PathTracerNRD()
try:
    m.addGraph(PathTracerNRD)
except NameError:
    None

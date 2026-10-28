from falcor import *


def render_graph_PathTracerNRD():
    g = RenderGraph("PathTracerNRD_VK")

    GBufferRT = createPass("GBufferRT", {"samplePattern": "Center", "sampleCount": 32, "useAlphaTest": True})
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
    NRDDeltaReflection = createPass(
        "NRD",
        {
            "method": "RelaxDiffuse",
            "maxIntensity": 250.0,
            "worldSpaceMotion": False,
            "enableReprojectionTestSkippingWithoutMotion": True,
            "spatialVarianceEstimationHistoryThreshold": 1,
        },
    )
    g.addPass(NRDDeltaReflection, "NRDDeltaReflection")
    NRDDeltaTransmission = createPass(
        "NRD",
        {
            "method": "RelaxDiffuse",
            "maxIntensity": 250.0,
            "worldSpaceMotion": False,
            "enableReprojectionTestSkippingWithoutMotion": True,
        },
    )
    g.addPass(NRDDeltaTransmission, "NRDDeltaTransmission")
    NRDReflectionMotionVectors = createPass("NRD", {"method": "SpecularReflectionMv", "worldSpaceMotion": False})
    g.addPass(NRDReflectionMotionVectors, "NRDReflectionMotionVectors")
    NRDTransmissionMotionVectors = createPass("NRD", {"method": "SpecularDeltaMv", "worldSpaceMotion": False})
    g.addPass(NRDTransmissionMotionVectors, "NRDTransmissionMotionVectors")
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

    # Delta reflection path.
    g.addEdge("PathTracer.nrdDeltaReflectionHitDist", "NRDReflectionMotionVectors.specularHitDist")
    g.addEdge("GBufferRT.linearZ", "NRDReflectionMotionVectors.viewZ")
    g.addEdge("GBufferRT.normWRoughnessMaterialID", "NRDReflectionMotionVectors.normWRoughnessMaterialID")
    g.addEdge("GBufferRT.mvec", "NRDReflectionMotionVectors.mvec")

    g.addEdge("PathTracer.nrdDeltaReflectionRadianceHitDist", "NRDDeltaReflection.diffuseRadianceHitDist")
    g.addEdge("NRDReflectionMotionVectors.reflectionMvec", "NRDDeltaReflection.mvec")
    g.addEdge("PathTracer.nrdDeltaReflectionNormWRoughMaterialID", "NRDDeltaReflection.normWRoughnessMaterialID")
    g.addEdge("PathTracer.nrdDeltaReflectionPathLength", "NRDDeltaReflection.viewZ")

    # Delta transmission path.
    g.addEdge("GBufferRT.posW", "NRDTransmissionMotionVectors.deltaPrimaryPosW")
    g.addEdge("PathTracer.nrdDeltaTransmissionPosW", "NRDTransmissionMotionVectors.deltaSecondaryPosW")
    g.addEdge("GBufferRT.mvec", "NRDTransmissionMotionVectors.mvec")

    g.addEdge("PathTracer.nrdDeltaTransmissionRadianceHitDist", "NRDDeltaTransmission.diffuseRadianceHitDist")
    g.addEdge("NRDTransmissionMotionVectors.deltaMvec", "NRDDeltaTransmission.mvec")
    g.addEdge("PathTracer.nrdDeltaTransmissionNormWRoughMaterialID", "NRDDeltaTransmission.normWRoughnessMaterialID")
    g.addEdge("PathTracer.nrdDeltaTransmissionPathLength", "NRDDeltaTransmission.viewZ")

    # NRD outputs -> compositor.
    g.addEdge("NRDDenoiser.filteredDiffuseRadianceHitDist", "Compositor.diffuseRadiance")
    g.addEdge("NRDDenoiser.filteredSpecularRadianceHitDist", "Compositor.specularRadiance")
    g.addEdge("PathTracer.nrdDiffuseReflectance", "Compositor.diffuseReflectance")
    g.addEdge("PathTracer.nrdSpecularReflectance", "Compositor.specularReflectance")
    g.addEdge("PathTracer.nrdEmission", "Compositor.emission")
    g.addEdge("NRDDeltaReflection.filteredDiffuseRadianceHitDist", "Compositor.deltaReflectionRadiance")
    g.addEdge("PathTracer.nrdDeltaReflectionReflectance", "Compositor.deltaReflectionReflectance")
    g.addEdge("PathTracer.nrdDeltaReflectionEmission", "Compositor.deltaReflectionEmission")
    g.addEdge("NRDDeltaTransmission.filteredDiffuseRadianceHitDist", "Compositor.deltaTransmissionRadiance")
    g.addEdge("PathTracer.nrdDeltaTransmissionReflectance", "Compositor.deltaTransmissionReflectance")
    g.addEdge("PathTracer.nrdDeltaTransmissionEmission", "Compositor.deltaTransmissionEmission")
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

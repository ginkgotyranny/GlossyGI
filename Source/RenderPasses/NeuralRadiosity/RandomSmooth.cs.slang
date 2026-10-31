import Scene.Shading;
import Scene.RaytracingInline;
import PathState;
import RayTraceHelper;
import NetworkParams;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;

// Ray batch buffers
RWStructuredBuffer<float4> allPos;
RWStructuredBuffer<float4> allDir;
RWStructuredBuffer<float4> allNormal;
RWStructuredBuffer<float4> allAlbedo;
RWStructuredBuffer<float> allRoughness;
RWStructuredBuffer<PackedHitInfo> allVBuffer;
RWStructuredBuffer<float4> allColor;
RWStructuredBuffer<float4> allEmission;

RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<DiffNetInput> diffInput;
RWStructuredBuffer<NetworkOutput> diffColor;

RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<SpecNetInput> specInput;
RWStructuredBuffer<NetworkOutput> specColor;
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<uint2> specPixel;
RWStructuredBuffer<float4> specPos;
RWStructuredBuffer<float4> specDir;
RWStructuredBuffer<float> specRoughness;


#define is_valid(name) (is_valid_##name != 0)

#ifndef USE_ALPHA_TEST
#define USE_ALPHA_TEST true
#endif

#ifndef ADJUST_SHADING_NORMALS
#define ADJUST_SHADING_NORMALS true
#endif

#ifndef USE_ENV_LIGHT
#define USE_ENV_LIGHT true
#endif

#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#ifndef SPECULAR_ROUGHNESS_THRESHOLD
#define SPECULAR_ROUGHNESS_THRESHOLD 0.5f
#endif

#ifndef RS_RR_PROB
#define RS_RR_PROB 0.3f
#endif


struct RandomSmooth
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kEnvMapDepth = 100000000.0f; // Arbitrary big number
    static const int kMaxRecursionDepth = 16;
    static const float kSpecularRoughnessThreshold = SPECULAR_ROUGHNESS_THRESHOLD;
    static const uint kRandomBase = 2654435761u;
    static const float kRSRRProb = RS_RR_PROB;
    static const bool kRandomWi = true;

    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void generatePath(const uint idx, const uint loopIdx, out PathState path)
    {
        path.setActive();
        path.id = idx;
        path.thp = float3(1.f);

        // Create sample generator.
        path.sg = SampleGenerator(uint2(idx & 0xFFFu, idx >> 12), batchCount * kMaxRecursionDepth + loopIdx);

        // Unspecified:
        // - path.origin
        // - path.dir
        // - path.setHit(hit);
    }

    /** Computes a random camera ray for a random point assuming a pinhole camera model.
        \param[in] uv UV coordinate in [0,1] with origin in top-left.
        \return Returns the camera ray.
    */
    Ray computeRandomRayPinhole(float2 uv)
    {
        Ray ray;

        // Compute the normalized ray direction assuming a pinhole camera.
        ray.origin = gScene.camera.data.posW;

        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 ndc = float2(2, -2) * uv + float2(-1, 1);

        // Compute the non-normalized ray direction assuming a pinhole camera.
        ray.dir = normalize(ndc.x * gScene.camera.data.cameraU + ndc.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW);

        float invCos = 1.f / dot(normalize(gScene.camera.data.cameraW), ray.dir);
        ray.tMin = gScene.camera.data.nearZ * invCos;
        ray.tMax = gScene.camera.data.farZ * invCos;

        return ray;
    }

    void execute(const uint idx)
    {
        if (any(idx >= batchSize))
            return;
        
        // Trace rays.
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        HitInfo hit;
        float3 color = float3(0.f);

        // G-Buffer values.
        float3 pos = float3(0.f);
        float3 dir = float3(0.f);
        float3 normal = float3(0.f);
        float3 albedo = float3(0.f);
        float roughness = 1.f;

        // Initialize path state.
        bool resample = false;
        ShadingData sd;
        IMaterialInstance mi;
        PathState path = {};
        generatePath(idx, 0, path);
        bool pathActive = true;  // Local tracking to work around Slang bitfield bug

        // Compute primary ray.
        Ray ray = computeRandomRayPinhole(sampleNext2D(path.sg));

        for (int i = 0; i < kMaxRecursionDepth; i++)
        {
            float rayT = 0;
            hit = sceneRayQuery.traceRay(ray, rayT, kRayFlags);
            resample = !hit.isValid();

            // Miss: Write background color.
            if (!resample)
            {
                // Load shading data. This is a long latency operation.
                sd = loadShadingData(hit, ray.origin, ray.dir);

                // Create material instance.
                let lod = ExplicitLodTextureSampler(0.f);
                let hints = getMaterialInstanceHints(hit, i == 0, kAdjustShadingNormals);
                mi = gScene.materials.getMaterialInstance(sd, lod, hints);
                const BSDFProperties bsdfProperties = mi.getProperties(sd);

                // Update path state.
                path.setHit(hit);
                path.incrementVertexIndex();
                path.sceneLength += float16_t(rayT);
                path.origin = sd.computeRayOrigin();
                path.dir = ray.dir;

                // Emissive surface or backside surface: resample.
                if (any(bsdfProperties.emission > 0.f) || (!sd.mtl.isDoubleSided() && !sd.frontFacing))
                {
                    resample = true;
                }
                // Sample next ray segment for delta specular or RR is active.
                else if (sd.mtl.isDeltaSpecular() || sampleNext1D(path.sg) >= kRSRRProb)
                {
                    // Sample next ray segment.
                    BSDFSample result = {};
                    bool valid = mi.sample(sd, path.sg, result);
                    if (valid) valid = generateScatterRay(result, sd, mi, path, kSpecularRoughnessThreshold);

                    // Resample ray if the BSDF sample is invalid.
                    resample = resample || !valid;

                    // Update ray.
                    ray = path.getScatterRay();
                }
                // Select this point as a training sample.
                else
                {
                    pos = path.origin;
                    if (kRandomWi)
                    {
                        float wiPdf = 0.f;
                        float3 wiLocal = sample_cosine_hemisphere_concentric(sampleNext2D(path.sg), wiPdf);
                        if ((dot(wiLocal, sd.faceN) * dot(sd.frame.toLocal(path.dir), sd.faceN)) < 0)
                        {
                            wiLocal = -wiLocal;
                        }
                        dir = sd.frame.fromLocal(wiLocal);
                    }
                    else
                    {
                        dir = path.dir;
                    }
                    normal = sd.frame.N;
                    roughness = bsdfProperties.roughness;
                    albedo = all(bsdfProperties.diffuseReflectionAlbedo < 0.01f) ?
                        bsdfProperties.specularReflectionAlbedo : bsdfProperties.diffuseReflectionAlbedo;
                    color = path.thp;
                    break;
                }
            }

            if (resample)
            {
                pos = float3(0.f);
                dir = float3(0.f);
                normal = float3(0.f);
                albedo = float3(0.f);
                roughness = 1.f;
                path = {};
                generatePath(idx, i + 1, path);
                pathActive = true;  // reset for new path
                ray = computeRandomRayPinhole(sampleNext2D(path.sg));
            }
        }
        if (resample)
        {
            path.terminate();
            pathActive = false;
        }

        // Active mask
        uint diffAct = (pathActive && (roughness > kSpecularRoughnessThreshold)) ? 1 : 0;
        uint specAct = (pathActive && (roughness <= kSpecularRoughnessThreshold)) ? 1 : 0;

        // Write outputs
        allColor[idx] = float4(color, 0.f);
        allEmission[idx] = float4(0.f);
        allVBuffer[idx] = hit.pack();

        // Write G-Buffer
        allPos[idx] = float4(pos, 1.f);
        allDir[idx] = float4(dir, 0.f);
        allNormal[idx] = float4(normal, 0.f);
        allAlbedo[idx] = float4(albedo, 0.f);
        allRoughness[idx] = roughness;

        // Write active mask
        diffActive[idx] = diffAct;
        diffIndex[idx] = diffAct;
        specActive[idx] = specAct;
        specIndex[idx] = specAct;
    }
};

ParameterBlock<RandomSmooth> gRandomSmooth;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gRandomSmooth.execute(dispatchThreadId.x);
}

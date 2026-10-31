import Scene.Shading;
import Scene.RaytracingInline;
import Rendering.Lights.EmissiveLightSampler;
import Utils.Sampling.SampleGenerator;
import PathState;
import RayTraceHelper;
import NetworkParams;

// LHS batch buffers
RWStructuredBuffer<float4> lhsPos;
RWStructuredBuffer<float4> lhsDir;
RWStructuredBuffer<PackedHitInfo> lhsVBuffer;
RWStructuredBuffer<uint> lhsDiffActive;
RWStructuredBuffer<uint> lhsSpecActive;

// RHS batch buffers
RWStructuredBuffer<float4> rhsPos;
RWStructuredBuffer<float4> rhsDir;
RWStructuredBuffer<float4> rhsNormal;
RWStructuredBuffer<float4> rhsAlbedo;
RWStructuredBuffer<float> rhsRoughness;
RWStructuredBuffer<PackedHitInfo> rhsVBuffer;
RWStructuredBuffer<float4> rhsColor;
RWStructuredBuffer<float4> rhsEmission;

RWStructuredBuffer<uint> rhsDiffActive;
RWStructuredBuffer<uint> rhsDiffIndex;
RWStructuredBuffer<uint> rhsSpecActive;
RWStructuredBuffer<uint> rhsSpecIndex;


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

#ifndef NUM_RHS
#define NUM_RHS 32
#endif


struct SampleRHS
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kEnvMapDepth = 100000000.0f; // Arbitrary big number
    static const int kMaxRecursionDepth = 16;
    static const float kSpecularRoughnessThreshold = SPECULAR_ROUGHNESS_THRESHOLD;
    static const uint kRandomBase = 2654435761u;
    static const uint numRHS = NUM_RHS;

    uint batchSize = 0;
    uint batchCount = 0;

    bool useNEE = false;
    bool useEmissiveLights = false;
    bool useAnalyticLights = false;
    bool useEnvLights = false;
    EmissiveLightSampler emissiveSampler;

    void generatePath(const uint idx, out PathState path)
    {
        path.setActive();
        path.id = idx;
        path.thp = float3(1.f);

        // Create sample generator.
        path.sg = SampleGenerator(uint2(idx & 0xFFFu, idx >> 12), batchCount);

        // Unspecified:
        // - path.origin
        // - path.dir
        // - path.setHit(hit);
    }

    void execute(const uint idx, const uint rhsIdx)
    {
        if (any(idx >= batchSize))
            return;

        const uint rayIdx = idx * numRHS + rhsIdx;

        // Load LHS data.
        // loadLHSData(idx, rhsIdx);
        float3 lPos = lhsPos[idx].xyz;
        float3 lDir = lhsDir[idx].xyz;
        HitInfo lHit = unpackHitInfo(lhsVBuffer[idx]);
        bool valid = lhsDiffActive[idx] != 0 || lhsSpecActive[idx] != 0;

        // Load shading data.
        ShadingData lSd = loadShadingData(lHit, lPos, lDir);
        lSd.posW = lPos;

        // Create material instance.
        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(lHit, true, kAdjustShadingNormals);
        IMaterialInstance lMi = gScene.materials.getMaterialInstance(lSd, lod, hints);
        BSDFProperties lBsdfProperties = lMi.getProperties(lSd);

        // Initialize path state.
        PathState path = {};
        generatePath(rayIdx, path);
        path.origin = lPos;
        path.dir = lDir;
        path.setHit(lHit);
        bool pathActive = true;  // Local tracking to work around Slang bitfield bug

        // Prepare ray query structures.
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        HitInfo hit;
        float3 color = float3(0.f);

        // G-Buffer values.
        float3 pos = float3(0.f);
        float3 dir = float3(0.f);
        float3 normal = float3(0.f);
        float3 albedo = float3(0.f);
        float roughness = 1.f;

        // Sample NEE.
        LightSample ls = {};
        bool neeValid = false;
        if (valid && useNEE)
        {
            uint lightType;
            float selectionPdf;
            if (!selectLightType(lightType, selectionPdf, path.sg, useEmissiveLights, useAnalyticLights, useEnvLights))
            {
                neeValid = false;
            }
            else if (lightType == (uint)LightType::Analytic)
            {
                // Sample analytic light.
                neeValid = generateAnalyticLightSample(lSd, path.sg, ls);
                ls.lightType = (uint)LightType::Analytic;
            }
            else if (lightType == (uint)LightType::Emissive)
            {
                // Sample emissive light.
                neeValid = generateEmissiveSample(emissiveSampler, lSd, true, path.sg, ls);
                ls.lightType = (uint)LightType::Emissive;
            }
            
            if (neeValid)
            {
                ls.pdf *= selectionPdf;
                ls.Li /= selectionPdf;
                neeValid &= sceneRayQuery.traceVisibilityRay(ls.getVisibilityRay(), kRayFlags);
            }
        }

        // Sample BSDF.
        BSDFSample bs = {};
        bool bsdfValid = false;
        if (valid) bsdfValid = lMi.sample(lSd, path.sg, bs);
        if (valid) bsdfValid = generateScatterRay(bs, lSd, lMi, path, kSpecularRoughnessThreshold);

        // Cross-term pdfs for MIS
        float pEdB = 0;  // wait until first hit
        float pBdE = neeValid ? lMi.evalPdf(lSd, ls.dir, true) : 0.f;
        float3 neeWeight = neeValid ? lMi.eval(lSd, ls.dir, path.sg) : 0.f;

        // Update ray.
        Ray ray = path.getScatterRay();

        if (bsdfValid)
        {
            // Trace the "first smooth" ray
            for (int i = 0; i < kMaxRecursionDepth; i++)
            {
                float rayT = 0;
                hit = sceneRayQuery.traceRay(ray, rayT, kRayFlags);

                // Miss: Write background color.
                if (!hit.isValid())
                {
                    if (kUseEnvLight)
                    {
                        color = gScene.envMap.eval(ray.dir) * path.thp;
                    }
                    path.terminate();
                    pathActive = false;
                    break;
                }

                // Load shading data. This is a long latency operation.
                ShadingData sd = loadShadingData(hit, ray.origin, ray.dir);

                // Create material instance.
                let lod = ExplicitLodTextureSampler(0.f);
                let hints = getMaterialInstanceHints(hit, i == 0, kAdjustShadingNormals);
                const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
                const BSDFProperties bsdfProperties = mi.getProperties(sd);

                // Update path state.
                path.setHit(hit);
                path.incrementVertexIndex();
                path.sceneLength += float16_t(rayT);
                path.origin = sd.computeRayOrigin();
                path.dir = ray.dir;

                // NEE pdf for BSDF sample.
                if ((i == 0) && (ls.lightType == (uint)LightType::Emissive))
                {
                    pEdB = emissiveSampler.evalPdf(lSd.posW, lSd.getOrientedFaceNormal(), true, getTriangleLightHit(hit, sd));
                }

                // Emissive surface: terminate path and write emitted radiance.
                if (any(bsdfProperties.emission > 0.f))
                {
                    color = bsdfProperties.emission * path.thp;
                    path.terminate();
                    pathActive = false;
                    break;
                }

                // Backside surface: terminate path.
                if (!sd.mtl.isDoubleSided() && !sd.frontFacing)
                {
                    path.terminate();
                    pathActive = false;
                    break;
                }

                // Break the loop if the hit is not on delta only material
                if (!sd.mtl.isDeltaSpecular())
                {
                    pos = path.origin;
                    dir = path.dir;
                    normal = sd.frame.N;
                    roughness = bsdfProperties.roughness;
                    albedo = all(bsdfProperties.diffuseReflectionAlbedo < 0.01f) ? 
                        bsdfProperties.specularReflectionAlbedo : bsdfProperties.diffuseReflectionAlbedo;

                    color = path.thp;
                    break;
                }

                // Sample next ray segment.
                BSDFSample result = {};
                bool valid = mi.sample(sd, path.sg, result);
                if (valid) valid = generateScatterRay(result, sd, mi, path, kSpecularRoughnessThreshold);

                // Break the loop if the BSDF sample is invalid.
                if (!valid)
                {
                    path.terminate();
                    pathActive = false;
                    break;
                }

                // Update ray.
                ray = path.getScatterRay();
            }
        }
        else
        {
            path.terminate();
            pathActive = false;
        }

        // MIS evaluation
        float misNEE = neeValid ? ((ls.lightType == (uint)LightType::Analytic) ? 1.0f : (ls.pdf / (ls.pdf + pBdE))) : 0.f;
        float misBSDF = bsdfValid ? (bs.pdf / (bs.pdf + pEdB)) : 0.f;

        // Active mask
        uint diffAct = (pathActive && (roughness > kSpecularRoughnessThreshold)) ? 1 : 0;
        uint specAct = (pathActive && (roughness <= kSpecularRoughnessThreshold)) ? 1 : 0;

        // Write outputs
        rhsColor[rayIdx] = float4(color * misBSDF, 0.f);
        rhsEmission[rayIdx] = float4(ls.Li * neeWeight * misNEE, 0.f);
        rhsVBuffer[rayIdx] = hit.pack();

        // Write G-Buffer
        rhsPos[rayIdx] = float4(pos, 1.f);
        rhsDir[rayIdx] = float4(dir, 0.f);
        rhsNormal[rayIdx] = float4(normal, 0.f);
        rhsAlbedo[rayIdx] = float4(albedo, 0.f);
        rhsRoughness[rayIdx] = roughness;

        // Write active mask
        rhsDiffActive[rayIdx] = diffAct;
        rhsDiffIndex[rayIdx] = diffAct;
        rhsSpecActive[rayIdx] = specAct;
        rhsSpecIndex[rayIdx] = specAct;
    }
};


ParameterBlock<SampleRHS> gSampleRHS;


[numthreads(NUM_RHS, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    gSampleRHS.execute(groupID.x, groupThreadID.x);
}

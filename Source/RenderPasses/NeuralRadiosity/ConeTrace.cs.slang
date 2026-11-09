import Scene.Shading;
import Scene.RaytracingInline;
import Rendering.Materials.IsotropicGGX;
import PathState;
import RayTraceHelper;
import NetworkParams;
import ClusterHistory;

#include "Utils/Math/MathConstants.slangh"

#ifndef USE_ALPHA_TEST
#define USE_ALPHA_TEST true
#endif

#ifndef ADJUST_SHADING_NORMALS
#define ADJUST_SHADING_NORMALS true
#endif

#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#include "NCRConfig.h"

#ifndef SPECULAR_ROUGHNESS_THRESHOLD
#define SPECULAR_ROUGHNESS_THRESHOLD NCR_SPECULAR_ROUGHNESS_THRESHOLD
#endif

#ifndef NUM_KMEANS_ITERS
#define NUM_KMEANS_ITERS NCR_NUM_KMEANS_ITERS_RENDER
#endif

#ifndef NUM_SPEC_RAYS
#define NUM_SPEC_RAYS NCR_NUM_SPEC_RAYS_RENDER
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS NCR_NUM_CLUSTERS
#endif

#ifndef WAVE_SIZE
#define WAVE_SIZE 32
#endif

/// Ceil(NUM_SPEC_RAYS / WAVE_SIZE). A group of <=32 rays is one wave, so this stays 1.
#define CONE_WAVE_COUNT ((NUM_SPEC_RAYS + WAVE_SIZE - 1) / WAVE_SIZE)


RWStructuredBuffer<uint> specSrc;
RWStructuredBuffer<SpecNetInput> specInput;
/// Compacted counts from PrefixSum: [0] diffuse, [1] specular. Read on the GPU so
/// the dispatch can be sized without a CPU readback.
ByteAddressBuffer gBatchCounts;

// Train path: gather G-Buffer from the pre-compact batch.
RWStructuredBuffer<float4> allPos;
RWStructuredBuffer<float4> allDir;
RWStructuredBuffer<float> allRoughness;
RWStructuredBuffer<PackedHitInfo> allVBuffer;

// Render fallback: gather G-Buffer from the screen textures.
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float> gRoughness;


groupshared ShadingData sd;
groupshared IMaterialInstance mi;
groupshared BSDFProperties bsdfProperties;

groupshared float3 pos;
groupshared float3 dir;
groupshared float roughness;
groupshared float tanLobe;
groupshared float cosLobe;

// Wave reduction buffers. Unused when the group fits in one wave.
groupshared float gsWaveMin[CONE_WAVE_COUNT];
groupshared float gsWaveMax[CONE_WAVE_COUNT];
groupshared float gsWaveSum[CONE_WAVE_COUNT];
// K-Means
groupshared uint gsClusterIdx[NUM_SPEC_RAYS];
groupshared float gsCenters[NUM_CLUSTERS];
groupshared float gsVars[NUM_CLUSTERS];
groupshared uint gsCounts[NUM_CLUSTERS];
groupshared float gsNewCenters[NUM_CLUSTERS];

// Wave reduction buffers (int). Unused when the group fits in one wave.
groupshared int gsWaveMinInt[CONE_WAVE_COUNT];
groupshared int gsWaveMaxInt[CONE_WAVE_COUNT];
groupshared int gsWaveSumInt[CONE_WAVE_COUNT];
// K-Means (int)
groupshared int gsCentersInt[NUM_CLUSTERS];
groupshared int gsVarsInt[NUM_CLUSTERS];
groupshared int gsNewCentersInt[NUM_CLUSTERS];


float GroupActiveSum(float value, uint threadID, uint groupSize)
{
    uint laneIdx  = WaveGetLaneIndex();
    uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();
    uint waveIdx  = threadID / waveSize;
    uint numWaves = (groupSize + waveSize - 1) / waveSize;

    float waveSum = WaveActiveSum(value);

    // One wave already holds every lane, so its WaveActiveSum is the group sum.
    if (numWaves <= 1)
        return waveSum;

    if (laneIdx == 0)
        gsWaveSum[waveIdx] = waveSum;

    GroupMemoryBarrierWithGroupSync();

    float groupSum = 0.0f;

    if (waveIdx == 0)
    {
        float val = (laneIdx < numWaves) ? gsWaveSum[laneIdx] : 0.0f;
        groupSum = WaveActiveSum(val);
    }

    GroupMemoryBarrierWithGroupSync();

    return groupSum;
}


struct ConeTrace
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kSpecularRoughnessThreshold = SPECULAR_ROUGHNESS_THRESHOLD;
    static const uint kRandomBase = 805459861u;
    static const uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();

    static const uint numKMeansIters = NUM_KMEANS_ITERS;
    static const uint numSpecRays = NUM_SPEC_RAYS;
    static const uint numClusters = NUM_CLUSTERS;

    uint batchSize = 0;
    uint batchCount = 0;
    /// Number of groups actually dispatched. Each group loops over samples with this stride.
    uint dispatchGroupCount = 1;
    uint2 frameDim = {};
    uint useScreenGBuffer = 0;

    float3 bboxMin;
    float3 bboxMax;

    void loadSpecGBuffer(const uint compactIdx, out float3 outPos, out float3 outDir, out float outRoughness, out HitInfo hit)
    {
        const uint src = specSrc[compactIdx];
        if (useScreenGBuffer != 0)
        {
            const uint2 pixel = uint2(src % frameDim.x, src / frameDim.x);
            outPos = gPos[pixel].xyz;
            outDir = gDir[pixel].xyz;
            outRoughness = gRoughness[pixel];
            hit = unpackHitInfo(gVBuffer[pixel]);
        }
        else
        {
            outPos = allPos[src].xyz;
            outDir = allDir[src].xyz;
            outRoughness = allRoughness[src];
            hit = unpackHitInfo(allVBuffer[src]);
        }
    }

    uint2 specRngPixel(const uint compactIdx)
    {
        const uint src = specSrc[compactIdx];
        // Render: real screen pixel. Train: packed raw idx, matching the old specPixel encoding.
        if (useScreenGBuffer != 0)
            return uint2(src % frameDim.x, src / frameDim.x);
        return uint2(src & 0xFFFu, src >> 12);
    }

    float2 hammersleySample(in uint2 index, in float2 randomOffset)
    {
        // index = (i, N)
        float2 u = {};
        u.y = (float(index.x) + 0.5f) / float(index.y);

        uint bits = index.x;
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        u.x = float(bits) * 2.3283064365386963e-10; // / 0x100000000

        // u += clamp(0.01 * (randomOffset - 0.5f), float2(1e-3f, 1e-3f), float2(0.999f, 0.999f));

        return u;
    }

    bool stratifiedSample(in ShadingData sd, in float alpha, in uint2 index, in SampleGenerator sg, out BSDFSample result)
    {
        float3 wi = sd.frame.toLocal(sd.V);
        float3 wo = {};

        {
            // Default initialization to avoid divergence at returns.
            result.weight = {};
            result.pdf = 0.f;
            result.lobeType = (uint)LobeType::SpecularReflection;

            if (wi.z < kMinCosTheta)
                return false;

            // Sample the GGX distribution to find a microfacet normal (half vector).
            float3 h = sampleGGX_NDF(alpha, hammersleySample(index, sampleNext2D(sg)), result.pdf); // pdf = D(h) * h.z
            // float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), result.pdf); // pdf = D(h) * h.z

            // Reflect the incident direction to find the outgoing direction.
            float wiDotH = dot(wi, h);
            wo = 2.f * wiDotH * h - wi;
            if (wo.z < kMinCosTheta)
                return false;

            result.pdf /= (4.f * wiDotH); // Jacobian of the reflection operator.
        }

        result.wo = sd.frame.fromLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sd.frame, wi, wo, result.wo)) return false;

        return true;
    }

    void loadGroupData(const uint pixIdx, const uint threadIdx)
    {
        if (threadIdx == 0)
        {
            float3 localPos;
            float3 localDir;
            float localRoughness;
            HitInfo hit;
            loadSpecGBuffer(pixIdx, localPos, localDir, localRoughness, hit);
            pos = localPos;
            dir = localDir;
            roughness = localRoughness;

            // Load shading data.
            sd = loadShadingData(hit, pos, dir);
            sd.posW = pos;

            // Disable diffuse lobes
            uint activeLobes = sd.mtl.getActiveLobes();
            activeLobes &= ~(uint)LobeType::Diffuse;
            sd.mtl.setActiveLobes(activeLobes);

            // Create material instance.
            let lod = ExplicitLodTextureSampler(0.f);
            let hints = getMaterialInstanceHints(hit, true, kAdjustShadingNormals);
            mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            bsdfProperties = mi.getProperties(sd);

            // Calculate lobe info. Shared with the render path so the cone
            // geometry behind every cluster scale stays identical.
            dir = reflect(dir, sd.frame.N);
            float tanLobeLocal;
            float cosLobeLocal;
            evalConeLobe(roughness, tanLobeLocal, cosLobeLocal);
            tanLobe = tanLobeLocal;
            cosLobe = cosLobeLocal;
        }
        
        GroupMemoryBarrierWithGroupSync();
    }

    bool traceRandomDepth(const uint pixIdx, const uint threadIdx, out float rayT)
    {
        uint2 pixel = specRngPixel(pixIdx);
        pixel.y = pixel.y * numSpecRays + threadIdx;

        SampleGenerator sg = SampleGenerator(pixel, 0);
        // SampleGenerator sg = SampleGenerator(uint2(pixIdx & 0xFFFu, pixIdx >> 12), batchCount);

        BSDFSample result = {};
        bool valid = false;
        for (int i = 0; i < 10 && !valid; ++i)
        {
            if (false)
            {
                float alpha = bsdfProperties.roughness * bsdfProperties.roughness;
                valid = stratifiedSample(sd, alpha, uint2(threadIdx, numSpecRays), sg, result);
            }
            else
            {
                valid = mi.sample(sd, sg, result);
            }
            // valid = valid && (dot(result.wo, reflect(-sd.V, sd.frame.N)) > cosLobe);
        }
        if (!valid) return false;

        SceneRayQuery<ConeTrace::kUseAlphaTest> sceneRayQuery;
        Ray ray;
        ray.origin = sd.posW;
        ray.dir = result.wo;
        ray.tMin = 0.f;
        ray.tMax = 1e30f;

        HitInfo nextHit = sceneRayQuery.traceRay(ray, rayT, ConeTrace::kRayFlags);

        GroupMemoryBarrierWithGroupSync();

        return nextHit.isValid();
    }

    void kMeans(const uint threadIdx, float rayT, bool valid)
    {
        static_assert(numSpecRays <= 256, "Thread group size exceeds the square of warp size: 16^2");

        uint laneIdx = WaveGetLaneIndex();
        uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();
        uint waveIdx = threadIdx / waveSize;
        uint numWaves = (numSpecRays + waveSize - 1) / waveSize;

        const float laneMin = valid ? rayT : 0.0f;
        const float laneMax = valid ? rayT : -FLT_MAX;

        if (numSpecRays <= waveSize)
        {
            const float bMin = WaveActiveMin(laneMin);
            const float bMax = WaveActiveMax(laneMax);
            if (laneIdx < numClusters)
                gsCenters[laneIdx] = bMin + (bMax - bMin) * laneIdx / (numClusters - 1);
        }
        else
        {
            // 1.1 Min-max reduction inside a wave.
            float waveMin = WaveActiveMin(laneMin);
            float waveMax = WaveActiveMax(laneMax);

            if (WaveIsFirstLane())
            {
                gsWaveMin[waveIdx] = waveMin;
                gsWaveMax[waveIdx] = waveMax;
            }
            GroupMemoryBarrierWithGroupSync();

            // 1.2 Min-max reduction over waves, then write to cluster centers.
            if (waveIdx == 0)
            {
                float bMin = 0.0f;
                float bMax = -FLT_MAX;

                if (laneIdx < numWaves)
                {
                    bMin = gsWaveMin[laneIdx];
                    bMax = gsWaveMax[laneIdx];
                }

                bMin = WaveActiveMin(bMin);
                bMax = WaveActiveMax(bMax);

                if (laneIdx < numClusters)
                    gsCenters[laneIdx] = bMin + (bMax - bMin) * laneIdx / (numClusters - 1);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. K-Means clustering.
        for (int iter = 0; iter < numKMeansIters; ++iter)
        {
            if (threadIdx < numClusters)
            {
                gsNewCenters[threadIdx] = 0.0f;
                gsCounts[threadIdx] = 0;
            }
            GroupMemoryBarrierWithGroupSync();

            int bestK = 0;
            float bestDist = fabs(rayT - gsCenters[0]);

            for (int k = 1; k < numClusters; ++k)
            {
                float dist = fabs(rayT - gsCenters[k]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestK = k;
                }
            }
            gsClusterIdx[threadIdx] = bestK;

            InterlockedAdd(gsCounts[bestK], valid ? 1 : 0);
            // InterlockedAdd(gsNewCenters[bestK], valid ? rayT : 0.0f);
            for (int k = 0; k < numClusters; ++k)
            {
                bool inCluster = valid && (bestK == k);
                float sum = GroupActiveSum(inCluster ? rayT : 0.0f, threadIdx, numSpecRays);
            
                if (threadIdx == 0)
                    gsNewCenters[k] = sum;
            }
            GroupMemoryBarrierWithGroupSync();

            if (threadIdx < numClusters)
            {
                float count = max(gsCounts[threadIdx], 1.0f);
                gsCenters[threadIdx] = gsNewCenters[threadIdx] / count;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Compute variance.
        int bestK = gsClusterIdx[threadIdx];
        float diff = rayT - gsCenters[bestK];
        // InterlockedAdd(gsVars[bestK], diff * diff);
        for (int k = 0; k < numClusters; ++k)
        {
            bool inCluster = valid && (bestK == k);
            float sum = GroupActiveSum(inCluster ? diff * diff : 0.0f, threadIdx, numSpecRays);
        
            if (threadIdx == 0)
                gsVars[k] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    void kMeansInt(const uint threadIdx, float rayT, bool valid)
    {
        static_assert(numSpecRays <= 1024, "Thread group size exceeds the square of warp size: 32^2");

        uint laneIdx = WaveGetLaneIndex();
        uint waveIdx = threadIdx / waveSize;
        uint numWaves = (numSpecRays + waveSize - 1) / waveSize;

        // float to int
        static const float scale = 1e5f;
        int rayTInt = (int) (rayT * scale);

        const int laneMin = valid ? rayTInt : INT32_MAX;
        const int laneMax = valid ? rayTInt : 0;

        if (numSpecRays <= waveSize)
        {
            const int bMin = WaveActiveMin(laneMin);
            const int bMax = WaveActiveMax(laneMax);
            if (laneIdx < numClusters)
                gsCentersInt[laneIdx] = bMin + (bMax - bMin) * laneIdx / (numClusters - 1);
        }
        else
        {
            // 1.1 Min-max reduction inside a wave.
            int waveMin = WaveActiveMin(laneMin);
            int waveMax = WaveActiveMax(laneMax);

            if (WaveIsFirstLane())
            {
                gsWaveMinInt[waveIdx] = waveMin;
                gsWaveMaxInt[waveIdx] = waveMax;
            }
            GroupMemoryBarrierWithGroupSync();

            // 1.2 Min-max reduction over waves, then write to cluster centers.
            if (waveIdx == 0)
            {
                int bMin = INT32_MAX;
                int bMax = 0;

                if (laneIdx < numWaves)
                {
                    bMin = gsWaveMinInt[laneIdx];
                    bMax = gsWaveMaxInt[laneIdx];
                }

                bMin = WaveActiveMin(bMin);
                bMax = WaveActiveMax(bMax);

                if (laneIdx < numClusters)
                    gsCentersInt[laneIdx] = bMin + (bMax - bMin) * laneIdx / (numClusters - 1);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. K-Means clustering.
        for (int iter = 0; iter < numKMeansIters; ++iter)
        {
            if (threadIdx < numClusters)
            {
                gsNewCentersInt[threadIdx] = 0;
                gsCounts[threadIdx] = 0;
            }
            GroupMemoryBarrierWithGroupSync();

            int bestK = 0;
            int bestDist = abs(rayTInt - gsCentersInt[0]);

            for (int k = 1; k < numClusters; ++k)
            {
                int dist = abs(rayTInt - gsCentersInt[k]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestK = k;
                }
            }
            gsClusterIdx[threadIdx] = bestK;

            InterlockedAdd(gsCounts[bestK], valid ? 1 : 0);
            InterlockedAdd(gsNewCentersInt[bestK], valid ? rayTInt : 0);
            GroupMemoryBarrierWithGroupSync();

            if (threadIdx < numClusters)
            {
                int count = max(gsCounts[threadIdx], 1);
                gsCentersInt[threadIdx] = gsNewCentersInt[threadIdx] / count;
            }
        }
        if (threadIdx < numClusters)
        {
            gsVarsInt[threadIdx] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Compute variance.
        int bestK = gsClusterIdx[threadIdx];
        int diff = rayTInt - gsCentersInt[bestK];
        int varsInt = (int) ((float) diff / scale * diff);
        InterlockedAdd(gsVarsInt[bestK], valid ? varsInt : 0);

        GroupMemoryBarrierWithGroupSync();

        if (threadIdx < numClusters)
        {
            gsVars[threadIdx] = gsVarsInt[threadIdx] / scale;
            gsCenters[threadIdx] = gsCentersInt[threadIdx] / scale;
        }
    }

    void writeOutput(const uint pixIdx, const uint threadIdx)
    {
        // envColor is written by SpecPrimary. This group only writes cluster slots.
        if (threadIdx < numClusters)
        {
            float mean = gsCenters[threadIdx];
            float size = gsCounts[threadIdx];
            float std = sqrt(gsVars[threadIdx] / max(size, 1.0f));
            // Denominator counts misses, so sum(clsWeight) < 1 when the cone
            // sends part of its energy to the environment.
            float clsWeight = size / numSpecRays;
            writeClusterOutput(
                specInput, pixIdx, threadIdx,
                pos, dir, tanLobe, mean, std, clsWeight,
                bboxMin, bboxMax
            );
        }
    }

    void execute(const uint pixIdx, const uint threadIdx)
    {
        // Load interaction.
        loadGroupData(pixIdx, threadIdx);

        float rayT = 0.0f;
        bool valid = traceRandomDepth(pixIdx, threadIdx, rayT);

        kMeansInt(threadIdx, rayT, valid);

        // Write output
        writeOutput(pixIdx, threadIdx);
    }
}


ParameterBlock<ConeTrace> gConeTrace;


[numthreads(NUM_SPEC_RAYS, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    // The CPU does not know this frame's specular count when it records this
    // dispatch. It sizes the grid from the previous frame's count, and each group
    // strides over any samples beyond it. The stride is uniform across the group,
    // so the barriers inside execute() stay convergent.
    const uint specCount = gBatchCounts.Load(4);
    for (uint pixIdx = groupID.x; pixIdx < specCount; pixIdx += gConeTrace.dispatchGroupCount)
    {
        gConeTrace.execute(pixIdx, groupThreadID.x);
        // The next sample overwrites the groupshared interaction and cluster state.
        GroupMemoryBarrierWithGroupSync();
    }
}

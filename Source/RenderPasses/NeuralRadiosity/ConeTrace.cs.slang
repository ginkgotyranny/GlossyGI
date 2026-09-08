import Scene.Shading;
import Scene.RaytracingInline;
import Rendering.Materials.IsotropicGGX;
import PathState;
import RayTraceHelper;
import NetworkParams;
import PrefiltEnvCubemap;

#include "Utils/Math/MathConstants.slangh"

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

#ifndef NUM_KMEANS_ITERS
#define NUM_KMEANS_ITERS 10
#endif

#ifndef NUM_SPEC_RAYS
#define NUM_SPEC_RAYS 128
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

#ifndef WAVE_SIZE
#define WAVE_SIZE 32
#endif


struct TemporalSpecHistory
{
    float mean[NUM_CLUSTERS];
    float std[NUM_CLUSTERS];
    float count[NUM_CLUSTERS];
};


RWStructuredBuffer<uint2> specPixel;
RWStructuredBuffer<float4> specPos;
RWStructuredBuffer<float4> specDir;
RWStructuredBuffer<float> specRoughness;
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<SpecNetInput> specInput;
// RWStructuredBuffer<TemporalSpecHistory> temporalHistory;


groupshared ShadingData sd;
groupshared IMaterialInstance mi;
groupshared BSDFProperties bsdfProperties;

groupshared float3 pos;
groupshared float3 dir;
groupshared float roughness;
groupshared float tanLobe;
groupshared float cosLobe;

// Wave reduction buffers
groupshared float gsWaveMin[NUM_SPEC_RAYS / 32];
groupshared float gsWaveMax[NUM_SPEC_RAYS / 32];
groupshared float gsWaveSum[NUM_SPEC_RAYS / 32];
// K-Means
groupshared uint gsClusterIdx[NUM_SPEC_RAYS];
groupshared float gsCenters[NUM_CLUSTERS];
groupshared float gsVars[NUM_CLUSTERS];
groupshared uint gsCounts[NUM_CLUSTERS];
groupshared float gsNewCenters[NUM_CLUSTERS];
groupshared uint gsHistoryActive[NUM_CLUSTERS];

// Wave reduction buffers (int)
groupshared int gsWaveMinInt[NUM_SPEC_RAYS / 32];
groupshared int gsWaveMaxInt[NUM_SPEC_RAYS / 32];
groupshared int gsWaveSumInt[NUM_SPEC_RAYS / 32];
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
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kSpecularRoughnessThreshold = SPECULAR_ROUGHNESS_THRESHOLD;
    static const uint kRandomBase = 805459861u;
    static const uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();

    static const uint numKMeansIters = NUM_KMEANS_ITERS;
    static const uint numSpecRays = NUM_SPEC_RAYS;
    static const uint numClusters = NUM_CLUSTERS;

    static const uint kLutSize = 32;
    static const float gTanLobeLUT[kLutSize] = {
        0.0,                    0.009716673754155636,   0.03886669501662254,    0.0874500647187233,
        0.15546678006649017,    0.24291685223579407,    0.3498002588748932,     0.4761170446872711,
        0.6218671202659607,     0.7870506048202515,     0.9716674089431763,     1.1757175922393799,
        1.3992010354995728,     1.642117977142334,      1.9044681787490845,     2.186251640319824,
        2.4874684810638428,     2.8081188201904297,     3.148202419281006,      3.5077192783355713,
        3.886669635772705,      4.285053253173828,      4.7028703689575195,     5.140120506286621,
        5.596804141998291,      6.072921276092529,      6.568471908569336,      7.083455562591553,
        7.617872714996338,      8.171723365783691,      8.745006561279297,      9.337723731994629
    };
    static const float gCosLobeLUT[kLutSize] = {
        1.0,                    0.9999527931213379,     0.9992455244064331,     0.9961980581283569,
        0.9881297945976257,     0.9717404246330261,     0.9439170956611633,     0.9028862714767456,
        0.8491918444633484,     0.7858079075813293,     0.7171937823295593,     0.6478891372680664,
        0.5814579129219055,     0.5201177000999451,     0.46489009261131287,    0.4159560799598694,
        0.37300193309783936,    0.33547356724739075,    0.3027360737323761,     0.27416205406188965,
        0.24917441606521606,    0.22726285457611084,    0.20798613131046295,    0.19096757471561432,
        0.17588791251182556,    0.16247735917568207,    0.15050818026065826,    0.13978791236877441,
        0.13015364110469818,    0.12146709859371185,    0.11361058801412582,    0.10648360103368759
    };

    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;
    uint enableTemporalHistory = 0;
    uint useTemporalReuse = 0;
    uint coneTraceReuseMode = 1;

    float3 bboxMin;
    float3 bboxMax;

    static const uint kConeTraceReuseModeTemporalMerge = 0;
    static const uint kConeTraceReuseModeTemporalToHistory = 1;
    static const uint kConeTraceReuseModeScreenSpaceDenoise = 2;

    // void sortClusters(const uint threadIdx)
    // {
    //     if (threadIdx == 0)
    //     {
    //         for (uint i = 0; i + 1 < numClusters; ++i)
    //         {
    //             for (uint j = 0; j + i + 1 < numClusters; ++j)
    //             {
    //                 bool swapClusters = false;
    //                 const bool leftActive = gsCounts[j] > 0;
    //                 const bool rightActive = gsCounts[j + 1] > 0;

    //                 if (!leftActive && rightActive)
    //                 {
    //                     swapClusters = true;
    //                 }
    //                 else if (leftActive == rightActive && gsCenters[j] > gsCenters[j + 1])
    //                 {
    //                     swapClusters = true;
    //                 }

    //                 if (swapClusters)
    //                 {
    //                     const float center = gsCenters[j];
    //                     gsCenters[j] = gsCenters[j + 1];
    //                     gsCenters[j + 1] = center;

    //                     const float var = gsVars[j];
    //                     gsVars[j] = gsVars[j + 1];
    //                     gsVars[j + 1] = var;

    //                     const uint count = gsCounts[j];
    //                     gsCounts[j] = gsCounts[j + 1];
    //                     gsCounts[j + 1] = count;
    //                 }
    //             }
    //         }
    //     }

    //     GroupMemoryBarrierWithGroupSync();
    // }

    // bool hasActiveHistoryClusters()
    // {
    //     [unroll]
    //     for (uint k = 0; k < numClusters; ++k)
    //     {
    //         if (gsHistoryActive[k] != 0)
    //         {
    //             return true;
    //         }
    //     }

    //     return false;
    // }

    // bool hasEmptyHistoryClusters()
    // {
    //     [unroll]
    //     for (uint k = 0; k < numClusters; ++k)
    //     {
    //         if (gsHistoryActive[k] == 0)
    //         {
    //             return true;
    //         }
    //     }

    //     return false;
    // }

    // int findNearestCluster(float rayT)
    // {
    //     int bestK = 0;
    //     float bestDist = fabs(rayT - gsCenters[0]);

    //     [unroll]
    //     for (uint k = 1; k < numClusters; ++k)
    //     {
    //         const float dist = fabs(rayT - gsCenters[k]);
    //         if (dist < bestDist)
    //         {
    //             bestDist = dist;
    //             bestK = (int) k;
    //         }
    //     }

    //     return bestK;
    // }

    // void seedEmptyHistoryClusters(const uint threadIdx, float rayT, bool valid)
    // {
    //     uint laneIdx = WaveGetLaneIndex();
    //     uint waveIdx = threadIdx / waveSize;
    //     uint numWaves = (numSpecRays + waveSize - 1) / waveSize;

    //     float waveMin = WaveActiveMin(valid ? rayT : FLT_MAX);
    //     float waveMax = WaveActiveMax(valid ? rayT : -FLT_MAX);

    //     if (WaveIsFirstLane())
    //     {
    //         gsWaveMin[waveIdx] = waveMin;
    //         gsWaveMax[waveIdx] = waveMax;
    //     }
    //     GroupMemoryBarrierWithGroupSync();

    //     if (waveIdx == 0)
    //     {
    //         float bMin = FLT_MAX;
    //         float bMax = -FLT_MAX;

    //         if (laneIdx < numWaves)
    //         {
    //             bMin = gsWaveMin[laneIdx];
    //             bMax = gsWaveMax[laneIdx];
    //         }

    //         bMin = WaveActiveMin(bMin);
    //         bMax = WaveActiveMax(bMax);

    //         if (laneIdx == 0 && bMin <= bMax)
    //         {
    //             uint emptyCount = 0;
    //             [unroll]
    //             for (uint k = 0; k < numClusters; ++k)
    //             {
    //                 if (gsHistoryActive[k] == 0) emptyCount++;
    //             }

    //             uint emptyRank = 0;
    //             [unroll]
    //             for (uint k = 0; k < numClusters; ++k)
    //             {
    //                 if (gsHistoryActive[k] == 0)
    //                 {
    //                     const float t = ((float) emptyRank + 0.5f) / max((float) emptyCount, 1.0f);
    //                     gsCenters[k] = lerp(bMin, bMax, t);
    //                     emptyRank++;
    //                 }
    //             }
    //         }
    //     }
    //     GroupMemoryBarrierWithGroupSync();
    // }

    // void refineEmptyHistoryClusters(const uint threadIdx, float rayT, bool valid)
    // {
    //     for (uint iter = 0; iter < numKMeansIters; ++iter)
    //     {
    //         if (threadIdx < numClusters)
    //         {
    //             gsNewCenters[threadIdx] = 0.0f;
    //             gsCounts[threadIdx] = 0;
    //         }
    //         GroupMemoryBarrierWithGroupSync();

    //         const int bestK = findNearestCluster(rayT);
    //         gsClusterIdx[threadIdx] = bestK;
    //         InterlockedAdd(gsCounts[bestK], valid ? 1 : 0);

    //         [unroll]
    //         for (uint k = 0; k < numClusters; ++k)
    //         {
    //             const bool inCluster = valid && (bestK == (int) k);
    //             const float sum = GroupActiveSum(inCluster ? rayT : 0.0f, threadIdx, numSpecRays);

    //             if (threadIdx == 0)
    //             {
    //                 gsNewCenters[k] = sum;
    //             }
    //         }
    //         GroupMemoryBarrierWithGroupSync();

    //         if (threadIdx < numClusters && gsHistoryActive[threadIdx] == 0 && gsCounts[threadIdx] > 0)
    //         {
    //             gsCenters[threadIdx] = gsNewCenters[threadIdx] / (float) gsCounts[threadIdx];
    //         }
    //         GroupMemoryBarrierWithGroupSync();
    //     }
    // }

    // void computeCurrentClusterStats(const uint threadIdx, float rayT, bool valid)
    // {
    //     if (threadIdx < numClusters)
    //     {
    //         gsNewCenters[threadIdx] = 0.0f;
    //         gsCounts[threadIdx] = 0;
    //     }
    //     GroupMemoryBarrierWithGroupSync();

    //     const int bestK = findNearestCluster(rayT);
    //     gsClusterIdx[threadIdx] = bestK;
    //     InterlockedAdd(gsCounts[bestK], valid ? 1 : 0);

    //     [unroll]
    //     for (uint k = 0; k < numClusters; ++k)
    //     {
    //         const bool inCluster = valid && (bestK == (int) k);
    //         const float sum = GroupActiveSum(inCluster ? rayT : 0.0f, threadIdx, numSpecRays);

    //         if (threadIdx == 0)
    //         {
    //             gsNewCenters[k] = sum;
    //         }
    //     }
    //     GroupMemoryBarrierWithGroupSync();

    //     if (threadIdx < numClusters)
    //     {
    //         const float count = (float) gsCounts[threadIdx];
    //         if (count > 0.0f)
    //         {
    //             gsNewCenters[threadIdx] /= count;
    //         }
    //         gsVars[threadIdx] = 0.0f;
    //     }
    //     GroupMemoryBarrierWithGroupSync();

    //     const int assignedK = gsClusterIdx[threadIdx];
    //     const float diff = valid ? (rayT - gsNewCenters[assignedK]) : 0.0f;

    //     [unroll]
    //     for (uint k = 0; k < numClusters; ++k)
    //     {
    //         const bool inCluster = valid && (assignedK == (int) k);
    //         const float sum = GroupActiveSum(inCluster ? diff * diff : 0.0f, threadIdx, numSpecRays);

    //         if (threadIdx == 0)
    //         {
    //             gsVars[k] = sum;
    //         }
    //     }
    //     GroupMemoryBarrierWithGroupSync();
    // }

    // void mergeCurrentClustersWithHistory(const uint pixIdx, const uint threadIdx)
    // {
    //     if (threadIdx == 0)
    //     {
    //         const float kMaxTemporalCount = 102400.0f;
    //         TemporalSpecHistory history = temporalHistory[pixIdx];

    //         [unroll]
    //         for (uint k = 0; k < numClusters; ++k)
    //         {
    //             const float curCount = (float) gsCounts[k];
    //             const float curMean = gsCenters[k];
    //             const float curM2 = gsVars[k];
    //             const float prevCount = min(history.count[k], kMaxTemporalCount);
    //             const float prevMean = history.mean[k];
    //             const float prevStd = max(history.std[k], 0.0f);
    //             const float prevM2 = prevStd * prevStd * prevCount;

    //             float outCount = 0.0f;
    //             float outMean = 0.0f;
    //             float outM2 = 0.0f;

    //             if (prevCount > 0.0f)
    //             {
    //                 if (curCount > 0.0f)
    //                 {
    //                     const float mergedCount = prevCount + curCount;
    //                     const float delta = curMean - prevMean;
    //                     const float mergedMean = (prevMean * prevCount + curMean * curCount) / mergedCount;
    //                     const float mergedM2 = prevM2 + curM2; // + delta * delta * prevCount * curCount / mergedCount;
    //                     const float countScale = min(kMaxTemporalCount / mergedCount, 1.0f);

    //                     outCount = mergedCount * countScale;
    //                     outMean = mergedMean;
    //                     outM2 = mergedM2 * countScale;
    //                 }
    //                 else
    //                 {
    //                     outCount = prevCount;
    //                     outMean = prevMean;
    //                     outM2 = prevM2;
    //                 }
    //             }
    //             else if (curCount > 0.0f)
    //             {
    //                 outCount = curCount;
    //                 outMean = curMean;
    //                 outM2 = curM2;
    //             }

    //             gsCounts[k] = (uint) (outCount + 0.5f);
    //             gsCenters[k] = outMean;
    //             gsVars[k] = outM2;
    //         }
    //     }

    //     GroupMemoryBarrierWithGroupSync();
    // }

    // void clusterToHistory(const uint pixIdx, const uint threadIdx, float rayT, bool valid)
    // {
    //     if (threadIdx < numClusters)
    //     {
    //         TemporalSpecHistory history = temporalHistory[pixIdx];
    //         gsCenters[threadIdx] = history.mean[threadIdx];
    //         gsHistoryActive[threadIdx] = history.count[threadIdx] > 0.0f ? 1u : 0u;
    //     }
    //     GroupMemoryBarrierWithGroupSync();

    //     if (!hasActiveHistoryClusters())
    //     {
    //         kMeansInt(threadIdx, rayT, valid);
    //         return;
    //     }

    //     if (hasEmptyHistoryClusters())
    //     {
    //         seedEmptyHistoryClusters(threadIdx, rayT, valid);
    //         refineEmptyHistoryClusters(threadIdx, rayT, valid);
    //     }

    //     computeCurrentClusterStats(threadIdx, rayT, valid);

    //     if (threadIdx < numClusters)
    //     {
    //         gsCenters[threadIdx] = gsNewCenters[threadIdx];
    //     }

    //     GroupMemoryBarrierWithGroupSync();
    //     mergeCurrentClustersWithHistory(pixIdx, threadIdx);
    // }

    // void clusterCurrentThenMergeHistory(const uint pixIdx, const uint threadIdx, float rayT, bool valid)
    // {
    //     kMeansInt(threadIdx, rayT, valid);
    //     sortClusters(threadIdx);
    //     mergeCurrentClustersWithHistory(pixIdx, threadIdx);
    // }

    // void accumulateTemporal(const uint pixIdx, const uint threadIdx)
    // {
    //     sortClusters(threadIdx);

    //     if (enableTemporalHistory == 0)
    //     {
    //         return;
    //     }

    //     if (threadIdx == 0)
    //     {
    //         TemporalSpecHistory history = temporalHistory[pixIdx];

    //         [unroll]
    //         for (uint k = 0; k < numClusters; ++k)
    //         {
    //             const float count = (float) gsCounts[k];
    //             history.mean[k] = gsCenters[k];
    //             history.std[k] = sqrt(gsVars[k] / max(count, 1.0f));
    //             history.count[k] = count;
    //         }

    //         temporalHistory[pixIdx] = history;
    //     }

    //     GroupMemoryBarrierWithGroupSync();
    // }

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
            pos = specPos[pixIdx].xyz;
            dir = specDir[pixIdx].xyz;
            roughness = specRoughness[pixIdx];
            HitInfo hit = unpackHitInfo(specVBuffer[pixIdx]);

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

            // Calculate lobe info.
            dir = reflect(dir, sd.frame.N);
            float lutPos = roughness * kLutSize;
            uint lutIdx = min((uint) lutPos, kLutSize - 2);
            float frac = saturate(lutPos - lutIdx);
            tanLobe = lerp(gTanLobeLUT[lutIdx], gTanLobeLUT[lutIdx + 1], frac);
            cosLobe = lerp(gCosLobeLUT[lutIdx], gCosLobeLUT[lutIdx + 1], frac);
        }
        
        GroupMemoryBarrierWithGroupSync();
    }

    bool traceRandomDepth(const uint pixIdx, const uint threadIdx, out float rayT)
    {
        uint2 pixel = specPixel[pixIdx];
        pixel.y = pixel.y * numSpecRays + threadIdx;

        SampleGenerator sg = SampleGenerator(pixel, 0);

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

        // 1.1 Min-max reduction inside a wave.
        float waveMin = WaveActiveMin(valid ? rayT : 0.0f);
        float waveMax = WaveActiveMax(valid ? rayT : -FLT_MAX);

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
            {
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

        // 1.1 Min-max reduction inside a wave.
        int waveMin = WaveActiveMin(valid ? rayTInt : INT32_MAX);
        int waveMax = WaveActiveMax(valid ? rayTInt : 0);

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
            {
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
        if (threadIdx < numClusters)
        {
            float mean = gsCenters[threadIdx];
            float size = gsCounts[threadIdx];
            float std = sqrt(gsVars[threadIdx] / max(size, 1.0f));
            float totalCount = 0.0f;
            [unroll]
            for (uint k = 0; k < numClusters; ++k) totalCount += gsCounts[k];
            // Calculate cluster info
            float3 clsPos = pos + mean * dir;
            float clsScale = (std + tanLobe * mean) / 2;
            // float clsWeight = size / max(totalCount, 1.0f);
            float clsWeight = size / numSpecRays;
            writeSpecInputClsBuffer(
                specInput, pixIdx, threadIdx,
                (Real3) ((clsPos - bboxMin) / (bboxMax - bboxMin)),
                (Real) clsScale, (Real) clsWeight
            );
        }
        if (threadIdx == 0)
        {
            // Normalize to [0, 1]
            pos = (pos - bboxMin) / (bboxMax - bboxMin);
            specInput[pixIdx].data[SpecNetInput::posOffset + 0] = (Real) pos.x;
            specInput[pixIdx].data[SpecNetInput::posOffset + 1] = (Real) pos.y;
            specInput[pixIdx].data[SpecNetInput::posOffset + 2] = (Real) pos.z;

            // Write prefiltered env map color
            float3 envColor = gPrefiltEnv.valid != 0 ? gPrefiltEnv.eval(dir, roughness) :
                (gPrefiltEnv.fallbackValid != 0 ? gScene.envMap.eval(dir) : float3(0.f));
            // float3 envColor = float3(0.f);
            specInput[pixIdx].data[SpecNetInput::envColorOffset + 0] = (Real) envColor.x;
            specInput[pixIdx].data[SpecNetInput::envColorOffset + 1] = (Real) envColor.y;
            specInput[pixIdx].data[SpecNetInput::envColorOffset + 2] = (Real) envColor.z;
        }
    }

    void execute(const uint pixIdx, const uint threadIdx)
    {
        if (pixIdx >= specSize)
            return;

        // Load interaction.
        loadGroupData(pixIdx, threadIdx);

        float rayT = 0.0f;
        bool valid = traceRandomDepth(pixIdx, threadIdx, rayT);

        // if (enableTemporalHistory != 0 && useTemporalReuse != 0)
        // {
        //     if (coneTraceReuseMode == kConeTraceReuseModeTemporalMerge)
        //         clusterCurrentThenMergeHistory(pixIdx, threadIdx, rayT, valid);
        //     else if (coneTraceReuseMode == kConeTraceReuseModeTemporalToHistory)
        //         clusterToHistory(pixIdx, threadIdx, rayT, valid);
        //     else
        //         // Placeholder until a dedicated screen-space denoise path is implemented.
        //         clusterToHistory(pixIdx, threadIdx, rayT, valid);
        // }
        // else
        //     kMeansInt(threadIdx, rayT, valid);
        // accumulateTemporal(pixIdx, threadIdx);
        kMeansInt(threadIdx, rayT, valid);

        // Write output
        writeOutput(pixIdx, threadIdx);
    }
}


ParameterBlock<ConeTrace> gConeTrace;


[numthreads(NUM_SPEC_RAYS, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    gConeTrace.execute(groupID.x, groupThreadID.x);
}

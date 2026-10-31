/** Screen-space cone trace for the render path.

    One thread per glossy pixel (not one group per pixel as on the training
    path): there is no group-wide K-Means, no wave reduction and no groupshared
    state here. The pixel already carries a multi-modal history, so the single
    baseline ray only has to answer "does this depth fall inside a known mode?".

    Rays stay GGX importance sampled through mi.sample() with a frame-varying
    seed — not seed=0, not Hammersley. Variance drops by accumulating ~N_lock
    random samples with α = 1/(n+1), then hard-locking the mixture. A locked
    pixel traces one ray only to measure residual; the blend is frozen unless
    the cone test already emptied the slot or the residual exceeds λ for M
    consecutive frames.

    Adaptive extra rays run only while unlocked. Locked silhouettes therefore
    stop being tugged by the near/far coin-flip of the current frame.
*/
import Scene.Shading;
import Scene.RaytracingInline;
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

#ifndef CONE_TRACE_SCREEN_GROUP_SIZE
#define CONE_TRACE_SCREEN_GROUP_SIZE 64
#endif

// Compacted specular source index (screen pixIdx). G-Buffer is gathered below.
RWStructuredBuffer<uint> specSrc;

// Screen G-Buffer channels.
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float> gRoughness;

// Cluster history (reprojected in place by ClusterReproject, updated here).
RWStructuredBuffer<SpecClusterHistory> clusterHistory;
// Consecutive residual-outlier count, ping-ponged with the history.
RWStructuredBuffer<uint> residualStreak;


struct ConeTraceScreen
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const uint kRayFlags = RAY_FLAGS;
    static const uint kMaxRayBudget = 4;

    uint2 frameDim = {};
    uint frameCount = 0;
    uint specSize = 0;

    /// Scene-scale epsilon, guards the residual denominators.
    float epsScene = 1e-4f;
    /// lambda: normalized residual below which a sample joins an existing mode.
    float assignThreshold = 2.0f;
    /// N_max: caps the per-mode (mu, sigma) count during accumulation.
    float maxCount = 64.0f;
    /// Modes below this weight release their slot.
    float weakWeight = 0.02f;
    /// Newborn sigma = sigmaPriorScale * tanLobe * t.
    float sigmaPriorScale = 0.5f;
    /// Observation count at which the mixture hard-locks (α = 0).
    float lockCount = 32.0f;
    /// Consecutive residual outliers required to unlock a frozen pixel.
    uint unlockStreak = 2;
    /// Rays traced per pixel per frame, 1..kMaxRayBudget. Locked pixels use 1.
    uint maxRays = 4;

    /** Trace one GGX importance sampled reflection ray, return its hit distance. */
    bool traceDepthSample(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, out float rayT)
    {
        rayT = 0.f;

        BSDFSample result = {};
        bool valid = false;
        for (int i = 0; i < 10 && !valid; ++i)
        {
            valid = mi.sample(sd, sg, result);
        }
        if (!valid) return false;

        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        Ray ray;
        ray.origin = sd.posW;
        ray.dir = result.wo;
        ray.tMin = 0.f;
        ray.tMax = 1e30f;

        HitInfo nextHit = sceneRayQuery.traceRay(ray, rayT, kRayFlags);
        return nextHit.isValid();
    }

    bool accumulateRay(
        inout ClusterSet cs,
        const ShadingData sd,
        const IMaterialInstance mi,
        inout SampleGenerator sg,
        float tanLobe,
        inout uint numRays,
        inout uint numOutliers)
    {
        float rayT = 0.f;
        const bool hitValid = traceDepthSample(sd, mi, sg, rayT);
        ++numRays;

        // t is the raw hit distance along the sampled direction while the
        // anchor sits on the cone axis. That is the convention the training
        // path uses (K-Means over rayT, clsPos = pos + mean * reflectDir),
        // so both paths must agree on it.
        const bool outlier = integrateClusterSample(
            cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene,
            assignThreshold, maxCount, weakWeight, sigmaPriorScale
        );
        if (outlier) ++numOutliers;
        return outlier;
    }

    void execute(const uint idx)
    {
        if (idx >= specSize) return;

        const uint2 pixel = decodeSpecSrcPixel(specSrc[idx], frameDim);
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        const float3 pos = gPos[pixel].xyz;
        const float3 rayDir = gDir[pixel].xyz;
        const float roughness = gRoughness[pixel];
        const HitInfo hit = unpackHitInfo(gVBuffer[pixel]);

        // Load shading data, mirroring ConeTrace::loadGroupData().
        ShadingData sd = loadShadingData(hit, pos, rayDir);
        sd.posW = pos;

        // Disable diffuse lobes.
        uint activeLobes = sd.mtl.getActiveLobes();
        activeLobes &= ~(uint)LobeType::Diffuse;
        sd.mtl.setActiveLobes(activeLobes);

        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(hit, true, kAdjustShadingNormals);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);

        // Cone axis. gNormal was written from sd.frame.N by FirstSmoothPass; using
        // the texture keeps all three passes of this pipeline on the same axis.
        const float3 reflectDir = normalize(reflect(rayDir, normalize(gNormal[pixel].xyz)));

        float tanLobe = 0.f;
        float cosLobe = 1.f;
        evalConeLobe(roughness, tanLobe, cosLobe);

        ClusterSet cs = unpackClusterSet(clusterHistory[pixIdx]);

        // Frame-varying seed so successive locked-window samples are independent
        // GGX draws. Freezing this to 0 would lock a single direction, which is
        // not the same as the old T=32 one-shot estimator.
        SampleGenerator sg = SampleGenerator(pixel, frameCount);

        const uint rayBudget = clamp(maxRays, 1u, kMaxRayBudget);
        const uint streakLimit = max(unlockStreak, 1u);
        uint streak = residualStreak[pixIdx];
        uint numRays = 0;
        uint numOutliers = 0;

        if (isClusterLocked(cs, lockCount))
        {
            // One ray, residual only. Do not fold it into (t, σ, w).
            float rayT = 0.f;
            const bool hitValid = traceDepthSample(sd, mi, sg, rayT);
            ++numRays;

            const float residual = clusterSampleResidual(
                cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene
            );
            const bool outlier = residual >= assignThreshold;
            if (outlier)
            {
                ++numOutliers;
                streak += 1u;
            }
            else
            {
                streak = 0;
            }

            if (streak >= streakLimit)
            {
                unlockClusterSet(cs);
                streak = 0;

                bool keepGoing = integrateClusterSample(
                    cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene,
                    assignThreshold, maxCount, weakWeight, sigmaPriorScale
                );

                [loop]
                for (uint r = 1; r < rayBudget && keepGoing; ++r)
                {
                    keepGoing = accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers);
                }
            }
        }
        else
        {
            streak = 0;

            [loop]
            for (uint r = 0; r < rayBudget; ++r)
            {
                if (!accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers)) break;
            }
        }

        releaseWeakClusters(cs, weakWeight);
        clusterHistory[pixIdx] = packClusterSet(cs);
        residualStreak[pixIdx] = streak;
    }
}


ParameterBlock<ConeTraceScreen> gConeTraceScreen;


[numthreads(CONE_TRACE_SCREEN_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gConeTraceScreen.execute(dispatchThreadId.x);
}

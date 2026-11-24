/** Screen-space cluster tracing: one baseline GGX ray per glossy pixel.
    Unlocked pixels accumulate independent samples. Stationary locked pixels
    reuse the baseline as a repeated-depth detector; repeated draws never enter
    the estimator. Pixel-staggered eight-frame windows cover the GGX lobe.

    Two global extra-ray pools bound each dispatch to N + floor(N/2) samples:
    floor(N/8) for adaptive accumulation, the remainder for far-mode validation.
    Diagnostic legacy mode retains the previous independent probe for A/B tests.
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
// Packed ClusterLockState (change-detection EMAs), ping-ponged with the history.
RWStructuredBuffer<uint> clusterLockState;
RWStructuredBuffer<uint> farValidationStats;


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
    float assignThreshold = 0.1f;
    /// N_max: caps the per-mode (mu, sigma) count during accumulation.
    float maxCount = 64.0f;
    /// Modes below this weight release their slot.
    float weakWeight = 0.02f;
    /// Newborn sigma = sigmaPriorScale * tanLobe * t.
    float sigmaPriorScale = 0.0f;
    /// Observation count at which the mixture hard-locks (α = 0).
    float lockCount = 32.0f;
    /// lambda_d: residual gate of the change detector (wider than assignThreshold).
    float detectThreshold = 1.0f;
    /// beta: EMA rate of the change detector, i.e. ~2/beta frames of memory.
    float detectRate = 0.125f;
    /// Outlier share at which a locked pixel unlocks (new geometry in the cone).
    float unlockOutlierRate = 0.2f;
    /// Anchor drift since lock, in cone radii, at which a locked pixel unlocks.
    float unlockDrift = 0.5f;
    /// Per-pixel adaptive loop limit; the global extra-ray cap also applies.
    uint maxRays = 4;
    /// Share the locked baseline with repeated-depth detection.
    uint enableVisibilityCheck = 1;
    uint enableFarModeValidation = 1;
    uint enableRayBudget = 1;
    uint visibilityFrame = 0;
    uint visibilityFrameAdvanced = 1;

    /** Trace one GGX importance sampled reflection ray, return its hit distance. */
    bool traceDepthSample(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, out float rayT, inout uint tracedRays)
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

        ++tracedRays;
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
        inout uint numOutliers,
        inout uint tracedRays)
    {
        float rayT = 0.f;
        const bool hitValid = traceDepthSample(sd, mi, sg, rayT, tracedRays);
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

    bool cameraMoved()
    {
        [unroll]
        for (uint r = 0; r < 4; ++r)
            if (any(gScene.camera.data.viewProjMatNoJitter[r] != gScene.camera.data.prevViewProjMatNoJitter[r])) return true;
        return false;
    }

    /** A fixed, independent batch estimates probabilities without retaining the
        sample that triggered validation. Misses remain in the denominator.
        Rebuild mode counts from the batch weights, since streaming slot merges
        otherwise lose the sample count used by the historical estimator.
    */
    ClusterSet validateFarModes(const ShadingData sd, const IMaterialInstance mi, uint2 pixel, float tanLobe, inout uint tracedRays)
    {
        ClusterSet fresh = emptyClusterSet();
        SampleGenerator validationSg = SampleGenerator(pixel, frameCount ^ 0xa511e9b3u);
        [loop]
        for (uint r = 0; r < 32u; ++r)
        {
            float t = 0.f;
            const bool hitValid = traceDepthSample(sd, mi, validationSg, t, tracedRays);
            int slot = -1;
            if (hitValid)
            {
                float residual = FLT_MAX;
                slot = findNearestCluster(fresh, t, tanLobe, epsScene, residual);
                if (slot < 0 || residual >= assignThreshold)
                {
                    slot = allocateClusterSlot(fresh, weakWeight, tanLobe, epsScene);
                    birthCluster(fresh, slot, t, max(sigmaPriorScale * tanLobe * t, epsScene));
                }
                else updateCluster(fresh, slot, t, maxCount);
            }
            updateClusterWeights(fresh, 1.f / float(r + 1u), slot);
            [unroll]
            for (uint k = 0; k < kClusterCount; ++k)
                if (isClusterActive(fresh, k)) fresh.n[k] = max(1.f, fresh.w[k] * float(r + 1u));
        }
        return fresh;
    }

    /** Counter layout: validated pixels, regular/validation reservations,
        actual traceRay calls, denied regular/validation requests, visibility
        resets, deferred resets, regular/validation ticket cursors.
        A wave reserves consecutive tickets with one atomic operation. Only
        whole requests inside the pool are accepted; no contended spin loop.
    */
    bool reserveExtraRays(uint count, bool validation)
    {
        if (enableRayBudget == 0) return true;
        const uint slot = validation ? 2u : 1u;
        const uint limit = validation ? specSize / 2u - specSize / 8u : specSize / 8u;
        const uint waveCount = WaveActiveSum(1u);
        const uint offset = WavePrefixSum(1u);
        uint start = 0u;
        if (WaveIsFirstLane()) InterlockedAdd(farValidationStats[validation ? 9u : 8u], waveCount, start);
        start = WaveReadLaneFirst(start);
        const bool accepted = start + offset < limit / count;
        const uint approved = WaveActiveSum(accepted ? count : 0u);
        const uint denied = WaveActiveSum(accepted ? 0u : 1u);
        if (WaveIsFirstLane())
        {
            InterlockedAdd(farValidationStats[slot], approved);
            InterlockedAdd(farValidationStats[validation ? 5u : 4u], denied);
        }
        return accepted;
    }

    void updateBudgetedHistory(
        inout ClusterSet cs, inout ClusterLockState state,
        const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg,
        uint2 pixel, float tanLobe, uint rayBudget, inout uint tracedRays)
    {
        if (state.visibilityDepth == kVisibilityResetPending)
        {
            cs = emptyClusterSet();
            state = emptyClusterLockState();
        }
        uint numRays = 0;
        uint numOutliers = 0;
        bool needsFreshSample = true;
        if (isClusterLocked(cs, lockCount))
        {
            float t = 0.f;
            bool hitValid;
            bool changed = false;
            bool newDirection = true;
            const bool sharedDetector = enableVisibilityCheck != 0 && !cameraMoved();
            if (sharedDetector)
            {
                // Phase 0 checks the OLD direction with the baseline ray, then
                // retains its reference. Phase 1 starts the new window. On the
                // first geometry-motion frame the CPU holds this clock, so an
                // abrupt edit is checked before any reference is replaced.
                uint h = (pixel.y * frameDim.x + pixel.x) * 0x9e3779b9u;
                h ^= h >> 16;
                const uint phaseFrame = visibilityFrame + (h & 7u);
                const uint phase = phaseFrame & 7u;
                const uint epoch = phaseFrame / 8u;
                const uint seedEpoch = phase == 0u && epoch > 0u ? epoch - 1u : epoch;
                SampleGenerator detectorSg = SampleGenerator(pixel, seedEpoch ^ 0x68bc21ebu);
                hitValid = traceDepthSample(sd, mi, detectorSg, t, tracedRays);
                newDirection = state.visibilityDepth == 0u || (visibilityFrameAdvanced != 0u && phase == 1u);
                changed = !newDirection && visibilityDepthChanged(state.visibilityDepth, hitValid, t, epsScene);
                if (newDirection) state.visibilityDepth = packVisibilityDepth(hitValid, t);
            }
            else
            {
                state.visibilityDepth = 0u;
                hitValid = traceDepthSample(sd, mi, sg, t, tracedRays);
            }
            ++numRays;
            // Do not count the same repeated outlier as eight independent votes.
            if (newDirection)
                updateClusterLockOutlier(state, cs, hitValid, t, tanLobe, epsScene, detectThreshold, detectRate);
            const bool unlock = state.outlier >= unlockOutlierRate || state.drift >= unlockDrift;
            if (changed || unlock)
            {
                if (changed) InterlockedAdd(farValidationStats[6], 1u);
                if (sharedDetector)
                {
                    // Repeated detector samples are never accumulation samples.
                    // Reserve a fresh draw before clearing, to avoid a black frame.
                    if (!reserveExtraRays(1u, false))
                    {
                        if (changed) state.visibilityDepth = kVisibilityResetPending;
                        InterlockedAdd(farValidationStats[7], 1u);
                        return;
                    }
                    if (changed) cs = emptyClusterSet();
                    else unlockClusterSet(cs);
                    state = emptyClusterLockState();
                    needsFreshSample = accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers, tracedRays);
                }
                else
                {
                    unlockClusterSet(cs);
                    needsFreshSample = integrateClusterSample(cs, hitValid, t, tanLobe, epsScene,
                        assignThreshold, maxCount, weakWeight, sigmaPriorScale);
                }
            }
            else return;
        }
        else state.visibilityDepth = 0u;

        [loop]
        while (numRays < rayBudget && needsFreshSample)
        {
            // One unconditional baseline; every later draw must reserve a token.
            if (numRays > 0u)
            {
                if (!reserveExtraRays(1u, false)) break;
            }
            needsFreshSample = accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers, tracedRays);
        }
    }

    void execute(const uint idx)
    {
        if (idx >= specSize) return;

        // Rotate the compacted work order so budget exhaustion does not always
        // disadvantage the same screen region. Pixel RNG remains unchanged.
        const uint sourceIdx = enableRayBudget != 0 ? (idx + (frameCount * 2654435761u) % specSize) % specSize : idx;
        const uint2 pixel = decodeSpecSrcPixel(specSrc[sourceIdx], frameDim);
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

        // Independent accumulation RNG. The shared locked detector uses its
        // own window seed and never contributes repeated draws to the mixture.
        SampleGenerator sg = SampleGenerator(pixel, frameCount);

        uint tracedRays = 0;
        const uint rayBudget = clamp(maxRays, 1u, kMaxRayBudget);
        ClusterLockState lockState = unpackClusterLockState(clusterLockState[pixIdx]);
        if (enableRayBudget != 0)
            updateBudgetedHistory(cs, lockState, sd, mi, sg, pixel, tanLobe, rayBudget, tracedRays);
        else
        {
            if (enableVisibilityCheck != 0)
            {
                // Repeat a direction for eight frames and compare against the first
                // depth, not the preceding frame: slow motion must accumulate too.
                // Rotate directions between windows to cover the full GGX lobe.
                // This ray is detection-only; integrating it would bias the mixture.
                const uint probePeriod = 8u;
                const bool rotateProbe = (frameCount % probePeriod) == 0u;
                bool visibilityChanged = false;
                if (rotateProbe && frameCount >= probePeriod && lockState.visibilityDepth != 0u)
                {
                    // Validate the old direction before replacing its reference.
                    // Otherwise a geometry edit on this exact frame is invisible
                    // to every probe, and stale mixtures can remain locked forever.
                    SampleGenerator oldProbeSg = SampleGenerator(pixel, (frameCount / probePeriod - 1u) ^ 0x68bc21ebu);
                    float oldProbeT = 0.f;
                    const bool oldProbeHit = traceDepthSample(sd, mi, oldProbeSg, oldProbeT, tracedRays);
                    visibilityChanged = visibilityDepthChanged(lockState.visibilityDepth, oldProbeHit, oldProbeT, epsScene);
                }
                SampleGenerator probeSg = SampleGenerator(pixel, (frameCount / probePeriod) ^ 0x68bc21ebu);
                float probeT = 0.f;
                const bool probeHit = traceDepthSample(sd, mi, probeSg, probeT, tracedRays);
                const bool newProbe = rotateProbe || lockState.visibilityDepth == 0u;
                visibilityChanged = visibilityChanged ||
                    (!newProbe && visibilityDepthChanged(lockState.visibilityDepth, probeHit, probeT, epsScene));
                if (visibilityChanged)
                {
                    // The old cone distribution is no longer trustworthy, even if
                    // the new hit happens to match another old mode (or is a miss).
                    // Reset only this pixel; unaffected static pixels remain locked.
                    cs = emptyClusterSet();
                    lockState = emptyClusterLockState();
                }
                if (newProbe || lockState.visibilityDepth == 0u)
                    lockState.visibilityDepth = packVisibilityDepth(probeHit, probeT);
            }
            else lockState.visibilityDepth = 0;
            uint numRays = 0;
            uint numOutliers = 0;

            if (isClusterLocked(cs, lockCount))
            {
                // One ray, detection only. Do not fold it into (t, σ, w).
                float rayT = 0.f;
                const bool hitValid = traceDepthSample(sd, mi, sg, rayT, tracedRays);
                ++numRays;

                updateClusterLockOutlier(
                    lockState, cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene, detectThreshold, detectRate
                );

                if (lockState.outlier >= unlockOutlierRate || lockState.drift >= unlockDrift)
                {
                    unlockClusterSet(cs);

                    bool keepGoing = integrateClusterSample(
                        cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene,
                        assignThreshold, maxCount, weakWeight, sigmaPriorScale
                    );

                    [loop]
                    for (uint r = 1; r < rayBudget && keepGoing; ++r)
                    {
                        keepGoing = accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers, tracedRays);
                    }
                }
            }
            else
            {
                [loop]
                for (uint r = 0; r < rayBudget; ++r)
                {
                    if (!accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers, tracedRays)) break;
                }
            }
        }

        // Camera motion keeps the ordinary estimator near a four-sample window.
        // Narrow lobes with rare separated depths are especially sensitive to a
        // single draw. Verify only those pixels, leaving broad/static lobes alone.
        if (enableFarModeValidation != 0 && roughness <= 0.15f && cameraMoved() &&
            needsFarModeValidation(cs, tanLobe, epsScene))
        {
            // Limit the extra estimator to highly reflective, non-transmitting
            // mirror-like surfaces. Low-roughness diffuse ceramics/plastics keep
            // their existing temporal path instead of trading noise elsewhere.
            const BSDFProperties properties = mi.getProperties(sd);
            if (all(properties.diffuseReflectionAlbedo < 0.01f) &&
                all(properties.specularTransmissionAlbedo < 0.01f) &&
                any(properties.specularReflectionAlbedo > 0.5f))
            {
                if (reserveExtraRays(32u, true))
                {
                    cs = validateFarModes(sd, mi, pixel, tanLobe, tracedRays);
                    InterlockedAdd(farValidationStats[0], 1u);
                }
            }
        }

        releaseWeakClusters(cs, weakWeight);

        // The detector only measures change since the mixture froze.
        if (!isClusterLocked(cs, lockCount))
        {
            lockState.outlier = 0.f;
            lockState.drift = 0.f;
        }
        const float u0 = sampleNext1D(sg);
        const float u1 = sampleNext1D(sg);

        const uint waveRays = WaveActiveSum(tracedRays);
        if (WaveIsFirstLane()) InterlockedAdd(farValidationStats[3], waveRays);
        clusterHistory[pixIdx] = packClusterSet(cs);
        clusterLockState[pixIdx] = packClusterLockState(lockState, u0, u1);
    }
}


ParameterBlock<ConeTraceScreen> gConeTraceScreen;


[numthreads(CONE_TRACE_SCREEN_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gConeTraceScreen.execute(dispatchThreadId.x);
}

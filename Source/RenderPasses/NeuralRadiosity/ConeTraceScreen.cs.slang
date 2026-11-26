/** Screen-space cluster tracing: one baseline GGX ray per glossy pixel.
    Unlocked pixels accumulate independent samples. Stationary locked pixels
    reuse the baseline as a repeated-depth detector; repeated draws never enter
    the estimator. Pixel-staggered eight-frame windows cover the GGX lobe.

    Two global extra-ray pools bound each dispatch to N + floor(N/2) samples:
    While moving: floor(N/8) for accumulation, the remainder for validation.
    With a fixed camera, validation is idle and its pool supports short rebuilds.
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
RWStructuredBuffer<uint> frequencyCounts;
RWStructuredBuffer<uint> vcdRecovery;


struct ConeTraceScreen
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const uint kRayFlags = RAY_FLAGS;
    static const uint kMaxRayBudget = 4;

    uint2 frameDim = {};
    uint frameCount = 0;
    uint diagnosticSamplingPeriod = 0; // 0: production; 32: frozen-model cyclic RNG test.
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
    float unlockOutlierRate = 0.05f;
    /// Anchor drift since lock, in cone radii, at which a locked pixel unlocks.
    float unlockDrift = 0.5f;
    /// Per-pixel adaptive loop limit; the global extra-ray cap also applies.
    uint maxRays = 4;
    /// Share the locked baseline with repeated-depth detection.
    uint enableVisibilityCheck = 1;
    uint enableFarModeValidation = 1;
    uint enableRayBudget = 1;
    // Preserve counts during fixed-camera accumulation. Motion uses truncated
    // pseudo-counts; retain that policy until separately validated.
    uint conserveMergeCounts = 1;
    uint enableFastRebuild = 1;
    uint enableVCDRecovery = 1;
    uint diagnosticVCDImmediateUnlock = 0; // Frozen-model comparison with the previous implementation.
    float recoveryThreshold = 0.1f;
    float recoveryUnlockRate = 0.1f;
    uint recoveryFrames = 64;
    uint visibilityResetSamples = 0; // Independent fresh batch after a confirmed VCD reset.
    uint checkFrequency = 0;
    uint gradualFrequencyUpdate = 1;
    uint diagnosticEagerFrequencySurvey = 0; // Reproduce the old biased survey only in frozen-model comparisons.
    static const uint kSurveyUpdating = 0xfffbu;
    static const uint kAdapting = 0xfffau;
    uint frequencySurveyFrame = 0;
    // Reserved fp16 NaN, distinct from the legacy pending reset and miss.
    static const uint kFrequencyResetPending = 0xfffcu;
    static const uint kFrequencyRebuild = 0xfffdu;
    uint visibilityFrame = 0;
    uint visibilityFrameAdvanced = 1;

    float recoveryAmount(uint pixelIndex)
    {
        return saturate(float(vcdRecovery[pixelIndex]) / float(max(recoveryFrames, 1u)));
    }

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
            assignThreshold, maxCount, weakWeight, sigmaPriorScale, conserveMergeCounts != 0 && !cameraMoved()
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
    ClusterSet sampleFreshModes(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator batchSg,
        float tanLobe, uint sampleCount, inout uint tracedRays)
    {
        ClusterSet fresh = emptyClusterSet();
        [loop]
        for (uint r = 0; r < sampleCount; ++r)
        {
            float t = 0.f;
            const bool hitValid = traceDepthSample(sd, mi, batchSg, t, tracedRays);
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

    ClusterSet validateFarModes(const ShadingData sd, const IMaterialInstance mi, uint2 pixel, float tanLobe, inout uint tracedRays)
    {
        const uint validationSeed = diagnosticSamplingPeriod != 0u
            ? frameCount % diagnosticSamplingPeriod + diagnosticSamplingPeriod
            : frameCount ^ 0xa511e9b3u;
        SampleGenerator validationSg = SampleGenerator(pixel, validationSeed);
        return sampleFreshModes(sd, mi, validationSg, tanLobe, 32u, tracedRays);
    }

    // A separate pool prevents reset batches from taking regular rebuild or
    // mirror-validation reservations. At most floor(specSize/2) extra attempts.
    // Counters 10..14: batch pixels, reserved rays, denied batches, ticket cursor,
    // actual batch traceRay calls; counter 15 counts successful deferred batches.
    // Every request in this pool has the same size.
    bool rebuildVisibilityBatch(inout ClusterSet cs, const ShadingData sd, const IMaterialInstance mi,
        inout SampleGenerator sg, float tanLobe, inout uint tracedRays)
    {
        if (visibilityResetSamples == 0u || cameraMoved()) return false;
        const uint waveCount = WaveActiveSum(1u);
        const uint offset = WavePrefixSum(1u);
        uint start = 0u;
        if (WaveIsFirstLane()) InterlockedAdd(farValidationStats[13], waveCount, start);
        start = WaveReadLaneFirst(start);
        const bool accepted = start + offset < (specSize / 2u) / visibilityResetSamples;
        const uint approved = WaveActiveSum(accepted ? 1u : 0u);
        if (WaveIsFirstLane())
        {
            InterlockedAdd(farValidationStats[10], approved);
            InterlockedAdd(farValidationStats[11], approved * visibilityResetSamples);
            InterlockedAdd(farValidationStats[12], waveCount - approved);
        }
        if (!accepted) return false;
        const uint before = tracedRays;
        // The detector draw selected this event: never count it in the estimate.
        // A fixed batch counts leading misses too, and does not stop at an inlier.
        cs = sampleFreshModes(sd, mi, sg, tanLobe, visibilityResetSamples, tracedRays);
        const uint waveTraces = WaveActiveSum(tracedRays - before);
        if (WaveIsFirstLane()) InterlockedAdd(farValidationStats[14], waveTraces);
        return true;
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
        // Far-mode validation runs only with a moving camera. Reclaim its
        // otherwise idle pool for newly rebuilt fixed-camera pixels.
        const uint regularLimit = enableFastRebuild != 0 && !cameraMoved() ? specSize / 2u : specSize / 8u;
        const uint limit = validation ? specSize / 2u - regularLimit : regularLimit;
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

    // Four four-bit hit counts followed by a draw count. Misses are implicit.
    // Only a newly selected direction votes, once per visibility window.
    bool frequencyChanged(uint pixelIndex, ClusterSet cs, bool hit, float t, float tanLobe)
    {
        uint counts = frequencyCounts[pixelIndex] + (1u << 16);
        if (hit)
        {
            float residual;
            int slot = findNearestCluster(cs, t, tanLobe, epsScene, residual);
            if (slot >= 0) counts += 1u << (4u * uint(slot));
        }
        const uint draws = counts >> 16;
        if (draws < 12u)
        {
            frequencyCounts[pixelIndex] = counts;
            return false;
        }
        frequencyCounts[pixelIndex] = 0u;
        uint hits = 0u;
        bool changed = false;
        [unroll]
        for (uint k = 0; k < kClusterCount; ++k)
        {
            uint observed = (counts >> (4u * k)) & 15u;
            hits += observed;
            changed = changed || abs(float(observed) / draws - cs.w[k]) > 0.5f;
        }
        return changed || abs(float(hits) / draws - saturate(clusterWeightSum(cs))) > 0.35f;
    }

    // Signed residuals of the miss probability and the sample's depth rank.
    // A well-calibrated mixture has expected midpoint rank 1/2 (miss is last).
    // Comparing against the pre-update mixture also works when slots are merged.
    float2 frequencyResidual(ClusterSet cs, bool hit, float t, float tanLobe)
    {
        float hitMass = saturate(clusterWeightSum(cs));
        float rank = 0.5f * (1.f + hitMass);
        if (hit)
        {
            float residual;
            int slot = findNearestCluster(cs, t, tanLobe, epsScene, residual);
            if (slot >= 0)
            {
                rank = 0.5f * cs.w[slot];
                [unroll]
                for (uint k = 0; k < kClusterCount; ++k)
                    if (cs.t[k] < cs.t[slot]) rank += cs.w[k];
            }
        }
        return float2(hit ? hitMass - 1.f : hitMass, rank - 0.5f);
    }

    void adaptFrequencySample(inout ClusterSet cs, bool hit, float t, float tanLobe)
    {
        integrateClusterSample(cs, hit, t, tanLobe, epsScene, assignThreshold,
            maxCount, weakWeight, sigmaPriorScale, conserveMergeCounts != 0);
    }

    void updateBudgetedHistory(
        inout ClusterSet cs, inout ClusterLockState state,
        const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg,
        uint2 pixel, float tanLobe, uint rayBudget, inout uint tracedRays)
    {
        const uint pixelIndex = pixel.y * frameDim.x + pixel.x;
        const float recovery = recoveryAmount(pixelIndex);
        float activeGate = lerp(detectThreshold, recoveryThreshold, recovery);
        float activeUnlockRate = lerp(unlockOutlierRate, recoveryUnlockRate, recovery);
        if (state.visibilityDepth == kAdapting)
        {
            // Keep the old probabilities and anchors. Shorten their effective
            // sample history for the next independent draws, then let it grow.
            uint remaining = frequencyCounts[pixelIndex];
            // The gather also reads these counts as confidence. Ease toward
            // the short history so its neighbour weight changes progressively.
            float targetCount = max(8.f, 0.84f * clusterObservationCount(cs));
            capClusterObservations(cs, targetCount);
            uint attempts = 0, outliers = 0;
            accumulateRay(cs, sd, mi, sg, tanLobe, attempts, outliers, tracedRays);
            frequencyCounts[pixelIndex] = remaining > 0u ? remaining - 1u : 0u;
            if (remaining <= 1u) state.visibilityDepth = kFrequencyRebuild;
            return;
        }
        if (gradualFrequencyUpdate != 0u && frequencySurveyFrame != 0u && enableVisibilityCheck != 0 &&
            !cameraMoved() && (isClusterLocked(cs, lockCount) || state.visibilityDepth == kSurveyUpdating) &&
            state.visibilityDepth != kVisibilityResetPending && state.visibilityDepth != kFrequencyRebuild)
        {
            bool continuing = state.visibilityDepth == kSurveyUpdating && frequencySurveyFrame > 1u;
            float2 evidence = continuing ? unpackHalf2(frequencyCounts[pixelIndex]) : float2(0.f);
            float t;
            bool hit = traceDepthSample(sd, mi, sg, t, tracedRays);
            evidence = 0.8f * evidence + frequencyResidual(cs, hit, t, tanLobe);
            // A survey measures evidence against the frozen mixture. Updating
            // here silently exceeds N_lock even in unchanged pixels. Eventually
            // a newborn mode receives 1/(N_obs+1) < weakWeight, is discarded,
            // and real hit probability leaks into the environment component.
            // Only the confirmed kAdapting path may shorten counts and update.
            if (diagnosticEagerFrequencySurvey != 0) adaptFrequencySample(cs, hit, t, tanLobe);
            state = emptyClusterLockState();
            if (any(abs(evidence) > float2(1.8f, 1.2f)))
            {
                state.visibilityDepth = kAdapting;
                frequencyCounts[pixelIndex] = 20u;
            }
            else
            {
                state.visibilityDepth = kSurveyUpdating;
                frequencyCounts[pixelIndex] = packHalf2(evidence.x, evidence.y);
            }
            return;
        }
        if (state.visibilityDepth == kSurveyUpdating)
        {
            state = emptyClusterLockState();
            frequencyCounts[pixelIndex] = 0u;
            if (!isClusterLocked(cs, lockCount)) state.visibilityDepth = kFrequencyRebuild;
        }
        if (state.visibilityDepth == kFrequencyResetPending)
        {
            cs = emptyClusterSet();
            state = emptyClusterLockState();
            state.visibilityDepth = kFrequencyRebuild;
        }
        if (state.visibilityDepth == kFrequencyRebuild)
        {
            // Rebuild using only the ordinary baseline, one fresh draw per frame.
            uint attempts = 0, outliers = 0;
            accumulateRay(cs, sd, mi, sg, tanLobe, attempts, outliers, tracedRays);
            if (isClusterLocked(cs, lockCount)) state = emptyClusterLockState();
            return;
        }
        if (checkFrequency != 0 && (frequencySurveyFrame == 1u || !isClusterLocked(cs, lockCount) ||
            (frequencySurveyFrame == 0u && (state.visibilityDepth == 0u || visibilityFrameAdvanced == 0u))))
            frequencyCounts[pixelIndex] = 0u;
        if (state.visibilityDepth == kVisibilityResetPending)
        {
            if (rebuildVisibilityBatch(cs, sd, mi, sg, tanLobe, tracedRays))
            {
                InterlockedAdd(farValidationStats[15], 1u);
                state = emptyClusterLockState();
                return;
            }
            cs = emptyClusterSet();
            state = emptyClusterLockState();
        }
        uint numRays = 0;
        uint numOutliers = 0;
        bool needsFreshSample = true;
        const bool allowFastRebuild = enableFastRebuild != 0 && !cameraMoved();
        // Spend a short batch on new mixtures instead of stopping at their
        // first inlier. This adds fresh evidence, not extra historical memory.
        // While measuring the post-edit distribution, all accumulating pixels
        // also use one baseline per frame. Resume the usual budget afterwards.
        if (frequencySurveyFrame != 0u && !cameraMoved()) rayBudget = 1u;
        bool rebuilding = allowFastRebuild && clusterObservationCount(cs) < 8.f;
        if (isClusterLocked(cs, lockCount))
        {
            float t = 0.f;
            bool hitValid;
            bool changed = false;
            bool newDirection = true;
            const bool sharedDetector = enableVisibilityCheck != 0 && !cameraMoved();
            const bool survey = sharedDetector && frequencySurveyFrame != 0u;
            if (survey)
            {
                // After geometry settles, spend twelve baseline rays on distinct
                // directions. The preceding motion frame retained its old-depth check.
                hitValid = traceDepthSample(sd, mi, sg, t, tracedRays);
                // Invalidate the repeated-direction reference when this survey ends.
                state.visibilityDepth = 0u;
            }
            else if (sharedDetector)
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
            if (changed)
            {
                InterlockedAdd(farValidationStats[6], 1u);
                if (enableVCDRecovery != 0)
                {
                    // VCD only changes detector sensitivity. Preserve counts, the
                    // mixture, detector EMA and depth reference until a real unlock.
                    vcdRecovery[pixelIndex] = recoveryFrames + 1u;
                    if (diagnosticVCDImmediateUnlock == 0)
                    {
                        activeGate = recoveryThreshold;
                        activeUnlockRate = recoveryUnlockRate;
                    }
                }
            }
            // Do not count the same repeated outlier as eight independent votes.
            if (newDirection && !survey)
                updateClusterLockOutlier(state, cs, hitValid, t, tanLobe, epsScene, activeGate, detectRate);
            const bool unlock = state.outlier >= activeUnlockRate || state.drift >= unlockDrift;
            bool frequencyMismatch = false;
            if (!changed && !unlock && sharedDetector && newDirection && checkFrequency != 0)
                frequencyMismatch = frequencyChanged(pixelIndex, cs, hitValid, t, tanLobe);
            if (frequencyMismatch)
            {
                // Keep the current frame intact. The next baseline starts a new
                // estimator, independently of the direction that triggered it.
                state.visibilityDepth = gradualFrequencyUpdate != 0u ? kAdapting : kFrequencyResetPending;
                if (gradualFrequencyUpdate != 0u) frequencyCounts[pixelIndex] = 20u;
                return;
            }
            const bool forcedVCD = changed && (enableVCDRecovery == 0 || diagnosticVCDImmediateUnlock != 0);
            if (forcedVCD || unlock)
            {
                rebuilding = allowFastRebuild;
                if (sharedDetector)
                {
                    if (changed && enableVCDRecovery != 0)
                    {
                        // Reached only after detector/drift unlock, unless explicitly
                        // reproducing the previous immediate-unlock experiment.
                        vcdRecovery[pixelIndex] = recoveryFrames + 1u;
                        unlockClusterSet(cs);
                        state = emptyClusterLockState();
                        if (!reserveExtraRays(1u, false)) return;
                        needsFreshSample = accumulateRay(cs, sd, mi, sg, tanLobe, numRays, numOutliers, tracedRays);
                    }
                    else
                    {
                        // Keep the side-effecting batch call inside an explicit
                        // branch: shader logical operators need not short-circuit.
                        if (changed)
                        {
                            if (rebuildVisibilityBatch(cs, sd, mi, sg, tanLobe, tracedRays))
                            {
                                state = emptyClusterLockState();
                                return;
                            }
                        }
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
                }
                else
                {
                    unlockClusterSet(cs);
                    needsFreshSample = integrateClusterSample(cs, hitValid, t, tanLobe, epsScene,
                        assignThreshold, maxCount, weakWeight, sigmaPriorScale, conserveMergeCounts != 0 && !cameraMoved());
                }
            }
            else return;
        }
        else state.visibilityDepth = 0u;

        [loop]
        while (numRays < rayBudget &&
            (needsFreshSample || (rebuilding && clusterCountSum(cs) > 0.f)))
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
        uint remaining = vcdRecovery[pixIdx];
        if (enableVCDRecovery == 0 || enableVisibilityCheck == 0 || cameraMoved()) remaining = 0u;
        vcdRecovery[pixIdx] = remaining > 0u ? remaining - 1u : 0u;
        const float recovery = recoveryAmount(pixIdx);
        float activeGate = lerp(detectThreshold, recoveryThreshold, recovery);
        float activeUnlockRate = lerp(unlockOutlierRate, recoveryUnlockRate, recovery);

        // Independent accumulation RNG. The shared locked detector uses its
        // own window seed and never contributes repeated draws to the mixture.
        const uint sampleSeed = diagnosticSamplingPeriod != 0u ? frameCount % diagnosticSamplingPeriod : frameCount;
        SampleGenerator sg = SampleGenerator(pixel, sampleSeed);

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
                    if (enableVCDRecovery != 0)
                    {
                        vcdRecovery[pixIdx] = recoveryFrames + 1u;
                        if (diagnosticVCDImmediateUnlock != 0)
                        {
                            unlockClusterSet(cs);
                            lockState = emptyClusterLockState();
                        }
                        else
                        {
                            activeGate = recoveryThreshold;
                            activeUnlockRate = recoveryUnlockRate;
                        }
                    }
                    else
                    {
                        cs = emptyClusterSet();
                        lockState = emptyClusterLockState();
                    }
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
                    lockState, cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene, activeGate, detectRate
                );

                if (lockState.outlier >= activeUnlockRate || lockState.drift >= unlockDrift)
                {
                    unlockClusterSet(cs);

                    bool keepGoing = integrateClusterSample(
                        cs, hitValid, hitValid ? rayT : 0.f, tanLobe, epsScene,
                        assignThreshold, maxCount, weakWeight, sigmaPriorScale, conserveMergeCounts != 0 && !cameraMoved()
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

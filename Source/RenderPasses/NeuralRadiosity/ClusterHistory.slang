/** Shared data structures and helpers for the spatiotemporal cluster reuse
    pipeline: ClusterReproject -> ConeTraceScreen -> ClusterSpatialReuse.

    The render path keeps a persistent K-mode mixture of reflection depths per
    pixel. The slot index *is* the mode identity and is never reordered, so a
    pixel can hold a near and a far mode simultaneously and the network sees a
    genuine mixture instead of a lerp between the two.

    `t_k` is stored in the reference frame of the pixel that wrote it, i.e.
    relative to (pos, reflectDir). The world anchor is recovered as
    `pos + t_k * reflectDir` and re-projected into the reader's frame, which is
    what makes the history meaningful across camera motion.
*/
import NetworkParams;

#include "Utils/Math/MathConstants.slangh"

#include "NCRConfig.h"

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS NCR_NUM_CLUSTERS
#endif

#ifndef NCR_NORMALIZE_CLS_SCALE
#error "NCRConfig.h must define NCR_NORMALIZE_CLS_SCALE (0 or 1)."
#endif

/// Slots addressable by the 24 B packed layout below.
#define CLUSTER_MAX_SLOTS 4

#if NUM_CLUSTERS > CLUSTER_MAX_SLOTS
#error "SpecClusterHistory packs at most 4 clusters per pixel."
#endif

static const uint kMaxClusterSlots = CLUSTER_MAX_SLOTS;
/// Slots actually used by the network input layout.
static const uint kClusterCount = NUM_CLUSTERS;

/// Decode CompactPass's specSrc (pixIdx = y * width + x) back to a screen pixel.
uint2 decodeSpecSrcPixel(uint src, uint2 frameDim)
{
    return uint2(src % frameDim.x, src / frameDim.x);
}

/// Packed per-pixel reflection-depth mixture. 24 B.
struct SpecClusterHistory
{
    uint2 packedT;      ///< 4 x fp16: t_k in the writing frame's (pos, reflectDir) frame.
    uint2 packedSigma;  ///< 4 x fp16: axial standard deviation sigma_k.
    uint2 packedWN;     ///< 4 x (u8 weight w_k, u8 observation count n_k).
};

/// Unpacked working set. Lives in registers inside a single thread.
struct ClusterSet
{
    float t[CLUSTER_MAX_SLOTS];
    float sigma[CLUSTER_MAX_SLOTS];
    float w[CLUSTER_MAX_SLOTS];
    float n[CLUSTER_MAX_SLOTS];
};

/// Weighted accumulator used by the spatial gather. Slot-aligned with ClusterSet.
struct ClusterAccum
{
    float sumWT[CLUSTER_MAX_SLOTS];
    float sumWSigma[CLUSTER_MAX_SLOTS];
    float sumW[CLUSTER_MAX_SLOTS];
};

ClusterAccum emptyClusterAccum()
{
    ClusterAccum acc;
    [unroll]
    for (uint k = 0; k < CLUSTER_MAX_SLOTS; ++k)
    {
        acc.sumWT[k] = 0.f;
        acc.sumWSigma[k] = 0.f;
        acc.sumW[k] = 0.f;
    }
    return acc;
}


/******************************************************************************
    Cone lobe lookup

    Shared with the training path (ConeTrace.cs.slang) so the cone geometry used
    to build cluster scales can never drift between the two.
******************************************************************************/

static const uint kConeLutSize = 32;

static const float kConeTanLobeLUT[kConeLutSize] = {
    0.0,                    0.009716673754155636,   0.03886669501662254,    0.0874500647187233,
    0.15546678006649017,    0.24291685223579407,    0.3498002588748932,     0.4761170446872711,
    0.6218671202659607,     0.7870506048202515,     0.9716674089431763,     1.1757175922393799,
    1.3992010354995728,     1.642117977142334,      1.9044681787490845,     2.186251640319824,
    2.4874684810638428,     2.8081188201904297,     3.148202419281006,      3.5077192783355713,
    3.886669635772705,      4.285053253173828,      4.7028703689575195,     5.140120506286621,
    5.596804141998291,      6.072921276092529,      6.568471908569336,      7.083455562591553,
    7.617872714996338,      8.171723365783691,      8.745006561279297,      9.337723731994629
};

static const float kConeCosLobeLUT[kConeLutSize] = {
    1.0,                    0.9999527931213379,     0.9992455244064331,     0.9961980581283569,
    0.9881297945976257,     0.9717404246330261,     0.9439170956611633,     0.9028862714767456,
    0.8491918444633484,     0.7858079075813293,     0.7171937823295593,     0.6478891372680664,
    0.5814579129219055,     0.5201177000999451,     0.46489009261131287,    0.4159560799598694,
    0.37300193309783936,    0.33547356724739075,    0.3027360737323761,     0.27416205406188965,
    0.24917441606521606,    0.22726285457611084,    0.20798613131046295,    0.19096757471561432,
    0.17588791251182556,    0.16247735917568207,    0.15050818026065826,    0.13978791236877441,
    0.13015364110469818,    0.12146709859371185,    0.11361058801412582,    0.10648360103368759
};

/** Look up the cone half-angle tangent and cosine for a roughness value.
    Identical interpolation to the original ConeTrace implementation.
*/
void evalConeLobe(float roughness, out float tanLobe, out float cosLobe)
{
    float lutPos = roughness * kConeLutSize;
    uint lutIdx = min((uint)lutPos, kConeLutSize - 2);
    float frac = saturate(lutPos - lutIdx);
    tanLobe = lerp(kConeTanLobeLUT[lutIdx], kConeTanLobeLUT[lutIdx + 1], frac);
    cosLobe = lerp(kConeCosLobeLUT[lutIdx], kConeCosLobeLUT[lutIdx + 1], frac);
}


/******************************************************************************
    Packing
******************************************************************************/

/** fp32 -> fp16 bits, round to nearest even, in integer math.

    The history is unpacked, updated and re-packed every frame, so the rounding of
    the pack accumulates. The f32tof16 intrinsic does not round the same way on
    every backend: on D3D12 (DXIL) the reuse path drifted ~2% darker than Vulkan
    in bathroom once clsScale was bbox-normalized, and this function removed the
    gap. On Vulkan it reproduces f32tof16 exactly (identical images).
*/
uint f32tof16Rne(float f)
{
    const uint x = asuint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint e8 = (x >> 23) & 0xffu;
    uint mant = x & 0x7fffffu;
    if (e8 == 0xffu) return sign | 0x7c00u | (mant != 0u ? 0x200u : 0u); // Inf / NaN
    const int e = int(e8) - 127 + 15;
    if (e >= 31) return sign | 0x7c00u; // Overflow to Inf.
    if (e <= 0)
    {
        // fp16 denormal or zero.
        if (e < -10) return sign;
        mant |= 0x800000u;
        const uint shift = uint(14 - e);
        uint h = mant >> shift;
        const uint rem = mant & ((1u << shift) - 1u);
        const uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) h++;
        return sign | h;
    }
    // A mantissa carry propagates into the exponent, which is the correct rounding.
    uint h = sign | (uint(e) << 10) | (mant >> 13);
    const uint rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h++;
    return h;
}

uint packHalf2(float a, float b)
{
    return (f32tof16Rne(a) & 0xffffu) | ((f32tof16Rne(b) & 0xffffu) << 16);
}

float2 unpackHalf2(uint v)
{
    return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16));
}

ClusterSet emptyClusterSet()
{
    ClusterSet cs;
    [unroll]
    for (uint k = 0; k < kMaxClusterSlots; ++k)
    {
        cs.t[k] = 0.f;
        cs.sigma[k] = 0.f;
        cs.w[k] = 0.f;
        cs.n[k] = 0.f;
    }
    return cs;
}

ClusterSet unpackClusterSet(SpecClusterHistory h)
{
    ClusterSet cs;

    float2 t01 = unpackHalf2(h.packedT.x);
    float2 t23 = unpackHalf2(h.packedT.y);
    cs.t[0] = t01.x; cs.t[1] = t01.y; cs.t[2] = t23.x; cs.t[3] = t23.y;

    float2 s01 = unpackHalf2(h.packedSigma.x);
    float2 s23 = unpackHalf2(h.packedSigma.y);
    cs.sigma[0] = s01.x; cs.sigma[1] = s01.y; cs.sigma[2] = s23.x; cs.sigma[3] = s23.y;

    [unroll]
    for (uint k = 0; k < kMaxClusterSlots; ++k)
    {
        uint packed = (k < 2) ? h.packedWN.x : h.packedWN.y;
        uint shift = (k & 1) * 16;
        cs.w[k] = float((packed >> shift) & 0xffu) * (1.f / 255.f);
        cs.n[k] = float((packed >> (shift + 8)) & 0xffu);
    }

    return cs;
}

SpecClusterHistory packClusterSet(ClusterSet cs)
{
    SpecClusterHistory h;

    h.packedT = uint2(packHalf2(cs.t[0], cs.t[1]), packHalf2(cs.t[2], cs.t[3]));
    h.packedSigma = uint2(packHalf2(cs.sigma[0], cs.sigma[1]), packHalf2(cs.sigma[2], cs.sigma[3]));

    uint lo = 0;
    uint hi = 0;
    [unroll]
    for (uint k = 0; k < kMaxClusterSlots; ++k)
    {
        uint qw = (uint)clamp(round(cs.w[k] * 255.f), 0.f, 255.f);
        uint qn = (uint)clamp(round(cs.n[k]), 0.f, 255.f);
        uint field = (qw | (qn << 8)) << ((k & 1) * 16);
        if (k < 2) lo |= field;
        else hi |= field;
    }
    h.packedWN = uint2(lo, hi);

    return h;
}


/******************************************************************************
    Mixture statistics
******************************************************************************/

bool isClusterActive(ClusterSet cs, uint k)
{
    return cs.n[k] > 0.5f;
}

float clusterWeightSum(ClusterSet cs)
{
    float sum = 0.f;
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k) sum += cs.w[k];
    return sum;
}

float clusterCountSum(ClusterSet cs)
{
    float sum = 0.f;
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k) sum += cs.n[k];
    return sum;
}

/** Weight-averaged depth of the mixture, used to size the spatial gather. */
float clusterMeanDepth(ClusterSet cs)
{
    float sumW = 0.f;
    float sumT = 0.f;
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k)) continue;
        sumW += cs.w[k];
        sumT += cs.w[k] * cs.t[k];
    }
    return sumW > 1e-5f ? sumT / sumW : 0.f;
}


/******************************************************************************
    Cone geometry
******************************************************************************/

/// World-space anchor of a cluster owned by the frame (pos, reflectDir).
float3 clusterAnchor(float3 pos, float3 reflectDir, float t)
{
    return pos + t * reflectDir;
}

/** Re-express a world anchor in the (pos, reflectDir) frame and test whether it
    still lies inside the reflection cone.

    This single geometric test replaces the per-pixel global invalidation of the
    previous attempt: as the cone sweeps off an object the corresponding mode
    falls out of the cone and dies, instead of lingering as a ghost.
*/
bool clusterInCone(float3 anchor, float3 pos, float3 reflectDir, float cosLobe, out float t)
{
    float3 d = anchor - pos;
    t = dot(d, reflectDir);
    float len = length(d);
    if (t <= 0.f || len < 1e-8f) return false;
    return (t / len) > cosLobe;
}

/** Normalized residual of a depth sample against mode k. The denominator is the
    mode's own spread widened by the cone footprint at that depth, so the gate
    automatically loosens for rough materials and far modes.
*/
float clusterResidual(float t, float mu, float sigma, float tanLobe, float epsScene)
{
    return abs(t - mu) / max(sigma + 0.5f * tanLobe * mu, epsScene);
}


/******************************************************************************
    Gated assignment, mode birth / death
******************************************************************************/

/** Nearest active mode to a depth sample, or -1 if the mixture is empty. */
int findNearestCluster(ClusterSet cs, float t, float tanLobe, float epsScene, out float bestDist)
{
    int best = -1;
    bestDist = FLT_MAX;

    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k)) continue;
        float d = clusterResidual(t, cs.t[k], cs.sigma[k], tanLobe, epsScene);
        if (d < bestDist)
        {
            bestDist = d;
            best = (int)k;
        }
    }

    return best;
}

/** Fold mode j into mode i (weighted mean plus law of total variance) and free j. */
void mergeClusterPair(inout ClusterSet cs, int i, int j)
{
    float wi = max(cs.w[i], 1e-6f);
    float wj = max(cs.w[j], 1e-6f);
    float wsum = wi + wj;

    float mu = (wi * cs.t[i] + wj * cs.t[j]) / wsum;
    float di = cs.t[i] - mu;
    float dj = cs.t[j] - mu;
    float var = (wi * (cs.sigma[i] * cs.sigma[i] + di * di) + wj * (cs.sigma[j] * cs.sigma[j] + dj * dj)) / wsum;

    cs.t[i] = mu;
    cs.sigma[i] = sqrt(max(var, 0.f));
    cs.w[i] = cs.w[i] + cs.w[j];
    cs.n[i] = max(cs.n[i], cs.n[j]);

    cs.t[j] = 0.f;
    cs.sigma[j] = 0.f;
    cs.w[j] = 0.f;
    cs.n[j] = 0.f;
}

/** Free a slot for a newly observed mode: prefer an unused slot, then the
    weakest mode below the release threshold, and only if every slot carries
    real weight merge the closest pair (K=4 means just 6 candidate pairs).
*/
int allocateClusterSlot(inout ClusterSet cs, float weakWeight, float tanLobe, float epsScene)
{
    int weakest = 0;
    float weakestW = FLT_MAX;

    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k)) return (int)k;
        if (cs.w[k] < weakestW)
        {
            weakestW = cs.w[k];
            weakest = (int)k;
        }
    }

    if (weakestW < weakWeight) return weakest;

    int mi = 0;
    int mj = (kClusterCount > 1) ? 1 : 0;
    float bestD = FLT_MAX;

    // Constant trip counts on both loops: DXC's unroller cannot deduce the bound of
    // an inner loop whose start depends on the outer induction variable (j = i + 1),
    // so iterate the full range and skip j <= i. The same pairs are visited.
    [unroll]
    for (uint i = 0; i < kClusterCount; ++i)
    {
        [unroll]
        for (uint j = 0; j < kClusterCount; ++j)
        {
            if (j <= i) continue;
            float denom = cs.sigma[i] + cs.sigma[j] + 0.5f * tanLobe * (cs.t[i] + cs.t[j]);
            float d = abs(cs.t[i] - cs.t[j]) / max(denom, epsScene);
            if (d < bestD)
            {
                bestD = d;
                mi = (int)i;
                mj = (int)j;
            }
        }
    }

    if (mi != mj) mergeClusterPair(cs, mi, mj);
    return mj;
}

void birthCluster(inout ClusterSet cs, int k, float t, float sigmaPrior)
{
    cs.t[k] = t;
    // Seed with the cone prior rather than 0 so that a single-sample mode still
    // queries the hash grid at cone scale instead of degenerating to a point query.
    cs.sigma[k] = sigmaPrior;
    cs.n[k] = 1.f;
    cs.w[k] = 0.f; // Set by updateClusterWeights() below.
}

/** Welford-style EMA update of (mu, sigma) for mode k. */
void updateCluster(inout ClusterSet cs, int k, float t, float maxCount)
{
    cs.n[k] = min(cs.n[k] + 1.f, maxCount);
    float alpha = 1.f / max(cs.n[k], 1.f);

    float delta = t - cs.t[k];
    cs.t[k] += alpha * delta;

    float var = cs.sigma[k] * cs.sigma[k];
    var = (1.f - alpha) * (var + alpha * delta * delta);
    cs.sigma[k] = sqrt(max(var, 0.f));
}

/** Number of rays, hits *and* misses, behind the current weights.

    sum(n_k) counts hits and sum(w_k) estimates the hit ratio, so their quotient
    recovers the sample count without spending any bits on it. The weight update
    needs this to stay unbiased: a plain "first hit takes full weight" rule pins
    sum(w_k) at 1 for as long as the pixel keeps being re-seeded, which is a large
    systematic over-estimate of the cone's hit ratio.
*/
float clusterObservationCount(ClusterSet cs)
{
    float hits = clusterCountSum(cs);
    if (hits <= 0.f) return 0.f;
    return hits / max(clusterWeightSum(cs), 1e-3f);
}

/** True once enough random GGX samples have been folded in that the mixture
    should freeze. Matches the old T=32 one-shot estimator: accumulate, then lock.
*/
bool isClusterLocked(ClusterSet cs, float lockCount)
{
    return clusterObservationCount(cs) >= lockCount;
}

/** Re-open accumulation without emptying the slots.

    `isClusterActive` is `n_k > 0.5`, so unlocking cannot zero the counts. Setting
    each live `n_k` to 1 drops `n_obs` well below `N_lock` while keeping (t, σ, w).
*/
void unlockClusterSet(inout ClusterSet cs)
{
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k)) continue;
        cs.n[k] = 1.f;
    }
}

/** EMA estimate of p_k = T_k / T. `hitSlot < 0` means the ray escaped to the
    environment, in which case every weight decays and sum(w_k) drops below 1 -
    the same semantics as `clsWeight = size / numSpecRays` on the training path,
    whose denominator also counts misses.
*/
void updateClusterWeights(inout ClusterSet cs, float alphaWeight, int hitSlot)
{
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        float target = ((int)k == hitSlot) ? 1.f : 0.f;
        cs.w[k] = (1.f - alphaWeight) * cs.w[k] + alphaWeight * target;
    }
}

/** Release modes whose weight has decayed to noise so their slot can be reused. */
void releaseWeakClusters(inout ClusterSet cs, float weakWeight)
{
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k)) continue;
        if (cs.w[k] >= weakWeight) continue;
        cs.t[k] = 0.f;
        cs.sigma[k] = 0.f;
        cs.w[k] = 0.f;
        cs.n[k] = 0.f;
    }
}

/******************************************************************************
    Change detection for locked pixels

    A locked mixture is frozen, so a locked pixel needs its own evidence that the
    cone now sees something else. Two signals, both kept in the u32 that travels
    with the mixture (ClusterLockState), so detection costs no memory:

    outlier  EMA (rate beta) of "this frame's ray hit outside every mode". The gate
             here is lambda_d, deliberately separate from the assignment gate
             lambda: lambda is tuned small (0.1) to partition the cone as finely as
             the training K-Means, so a converged static pixel already puts ~30% of
             its rays outside every lambda gate, while lambda_d = 1 leaves ~1%.
             A "M consecutive outliers" rule on lambda either unlocks constantly
             (M=1, static flicker) or almost never (M=4, smearing in motion).

    drift    Lateral offset of the mode anchors from the current cone axis, in
             units of the cone radius tanLobe * t, summed since the pixel locked.
             ClusterReproject re-expresses every anchor as a depth on the *current*
             axis, so the anchor slides along with the cone and the per-cluster
             cone test only fires on a large single-frame jump. Smooth camera
             motion would otherwise leave a stale mixture locked indefinitely.

    outlier and drift are reset to 0 whenever the pixel is not locked, so they only
    measure change since the mixture froze.

    While the camera moves relative to a pixel, ClusterReproject also caps its
    observation count (capClusterObservations), so the pixel cannot lock on a
    mixture averaged along the motion path. Without the cap such a mixture locks,
    the camera stops, every smeared mode still sits inside a detection gate, and
    the reflection's motion trail stays frozen on screen.
******************************************************************************/

struct ClusterLockState
{
    float outlier;
    float drift;
};

/// Drift quantization: 8 bits in units of 1/64 cone radius (saturates at ~4).
static const float kLockDriftScale = 64.f;

ClusterLockState emptyClusterLockState()
{
    ClusterLockState s;
    s.outlier = 0.f;
    s.drift = 0.f;
    return s;
}

ClusterLockState unpackClusterLockState(uint packed)
{
    ClusterLockState s;
    s.outlier = float(packed & 0xffu) * (1.f / 255.f);
    s.drift = float((packed >> 8) & 0xffu) * (1.f / kLockDriftScale);
    return s;
}

/** `u0`, `u1` are uniforms in [0,1) for stochastic rounding, which keeps the 8-bit
    EMA and the 8-bit running sum unbiased under small per-frame increments.
*/
uint packClusterLockState(ClusterLockState s, float u0, float u1)
{
    const uint o = (uint)clamp(floor(saturate(s.outlier) * 255.f + u0), 0.f, 255.f);
    const uint d = (uint)clamp(floor(max(s.drift, 0.f) * kLockDriftScale + u1), 0.f, 255.f);
    return o | (d << 8);
}

/** Fold one locked-pixel ray into the outlier EMA. A miss is not an outlier: an
    escaping ray has no depth to compare, and the escape share is part of the
    mixture (1 - sum(w)).
*/
void updateClusterLockOutlier(
    inout ClusterLockState s,
    ClusterSet cs,
    bool hitValid,
    float t,
    float tanLobe,
    float epsScene,
    float detectThreshold,
    float rate)
{
    bool outlier = false;
    if (hitValid)
    {
        float bestDist = FLT_MAX;
        const int best = findNearestCluster(cs, t, tanLobe, epsScene, bestDist);
        outlier = best < 0 || bestDist >= detectThreshold;
    }
    s.outlier += rate * ((outlier ? 1.f : 0.f) - s.outlier);
}

/** Rescale the counts so the mixture holds at most `maxObs` observations' worth of
    evidence (n_obs = sum(n)/sum(w)), keeping (t, sigma, w). Each live mode keeps
    n >= 1. Further samples then move (t, sigma, w) at roughly 1/(maxObs+1), i.e.
    the mixture tracks a short recent window, and it cannot reach lockCount.
*/
void capClusterObservations(inout ClusterSet cs, float maxObs)
{
    const float nObs = clusterObservationCount(cs);
    if (nObs <= maxObs) return;
    const float scale = maxObs / nObs;
    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k)) continue;
        cs.n[k] = max(1.f, floor(cs.n[k] * scale));
    }
}

/** Lateral distance of `anchor` from the cone axis, in units of the cone radius at
    its depth `t` along that axis.
*/
float clusterLateralDrift(float3 anchor, float3 pos, float3 reflectDir, float t, float tanLobe, float epsScene)
{
    const float3 d = anchor - pos;
    const float lateral = length(d - t * reflectDir);
    return lateral / max(tanLobe * t, epsScene);
}


/** Integrate a single depth sample into the mixture.

    History already encodes the multi-modal structure, so one ray per frame only
    has to answer "does this sample fall inside a known mode?". Failing that test
    is the only genuinely new information, and is what drives extra rays.

    Weight update is the unbiased running average `1/(n+1)`. The caller hard-locks
    at `N_lock`, so this must not saturate into a residual EMA — that leftover
    `1/64` tug is what kept silhouettes flickering after they should have frozen.

    \return true if the sample is an outlier w.r.t. the current mixture.
*/
bool integrateClusterSample(
    inout ClusterSet cs,
    bool hitValid,
    float t,
    float tanLobe,
    float epsScene,
    float assignThreshold,
    float maxCount,
    float weakWeight,
    float sigmaPriorScale)
{
    const float alphaW = 1.f / (clusterObservationCount(cs) + 1.f);

    if (!hitValid)
    {
        updateClusterWeights(cs, alphaW, -1);
        return false;
    }

    float bestDist = FLT_MAX;
    int best = findNearestCluster(cs, t, tanLobe, epsScene, bestDist);
    bool outlier = (best < 0) || (bestDist >= assignThreshold);

    int slot;
    if (outlier)
    {
        slot = allocateClusterSlot(cs, weakWeight, tanLobe, epsScene);
        birthCluster(cs, slot, t, max(sigmaPriorScale * tanLobe * t, epsScene));
    }
    else
    {
        slot = best;
        updateCluster(cs, slot, t, maxCount);
    }

    updateClusterWeights(cs, alphaW, slot);
    return outlier;
}


/** Pick the slot a neighbouring depth belongs to during the spatial gather.

    Prefers one of the pixel's own modes, then a slot already seeded by another
    neighbour at the same depth, then any completely unused slot. This keeps slot
    identity stable across the gather instead of re-sorting by depth.
*/
int routeNeighborCluster(
    ClusterSet cs,
    ClusterAccum acc,
    float t,
    float tanLobe,
    float epsScene,
    float assignThreshold)
{
    int best = -1;
    float bestDist = FLT_MAX;

    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        float mu;
        float sigma;
        if (isClusterActive(cs, k))
        {
            mu = cs.t[k];
            sigma = cs.sigma[k];
        }
        else if (acc.sumW[k] > 0.f)
        {
            mu = acc.sumWT[k] / acc.sumW[k];
            sigma = 0.f;
        }
        else
        {
            continue;
        }

        float d = clusterResidual(t, mu, sigma, tanLobe, epsScene);
        if (d < bestDist)
        {
            bestDist = d;
            best = (int)k;
        }
    }

    if (best >= 0 && bestDist < assignThreshold) return best;

    [unroll]
    for (uint k = 0; k < kClusterCount; ++k)
    {
        if (!isClusterActive(cs, k) && acc.sumW[k] <= 0.f) return (int)k;
    }

    return best >= 0 ? best : 0;
}

void accumulateNeighborCluster(inout ClusterAccum acc, int slot, float weight, float t, float sigma)
{
    acc.sumWT[slot] += weight * t;
    acc.sumWSigma[slot] += weight * sigma;
    acc.sumW[slot] += weight;
}


/******************************************************************************
    Network input

    Both the training path (group-wide K-Means over T=128 rays) and the render
    path funnel through these two helpers, so the specInput layout and the
    cluster formulas cannot drift apart and existing checkpoints stay valid.
******************************************************************************/

/** The three per-cluster quantities the network consumes. */
struct ClusterOutput
{
    float3 clsPos;    ///< bbox-normalized world position of the cluster center.
    float clsScale;   ///< Hash grid query radius, bbox-normalized like clsPos.
    float clsWeight;  ///< Mixture weight.
};

/** Single source of truth for the cluster formulas:
      clsPos    = (pos + t * reflectDir - bboxMin) / (bboxMax - bboxMin)
      clsScale  = (sigma + tanLobe * t) / 2 / max(bboxMax - bboxMin)
      clsWeight = w

    HashGridInterp picks levels by comparing clsScale with 1/resolution, a
    bbox-normalized length, so clsScale is normalized too. One scalar (the
    longest bbox axis) keeps the footprint isotropic; it is the voxel's longest
    side that matches the footprint at the chosen level.
*/
ClusterOutput evalClusterOutput(
    float3 pos,
    float3 reflectDir,
    float tanLobe,
    float t,
    float sigma,
    float weight,
    float3 bboxMin,
    float3 bboxMax)
{
    ClusterOutput o;
    o.clsPos = (pos + t * reflectDir - bboxMin) / (bboxMax - bboxMin);
    o.clsScale = (sigma + tanLobe * t) / 2;
#if NCR_NORMALIZE_CLS_SCALE
    const float3 extent = bboxMax - bboxMin;
    o.clsScale /= max(max(extent.x, extent.y), max(extent.z, 1e-6f));
#endif
    o.clsWeight = weight;
    return o;
}

/** Write one cluster slot into the network input buffer. */
void writeClusterOutput(
    RWStructuredBuffer<SpecNetInput> buffer,
    uint idx,
    uint clusterIdx,
    float3 pos,
    float3 reflectDir,
    float tanLobe,
    float t,
    float sigma,
    float weight,
    float3 bboxMin,
    float3 bboxMax)
{
    ClusterOutput o = evalClusterOutput(pos, reflectDir, tanLobe, t, sigma, weight, bboxMin, bboxMax);

    writeSpecInputClsBuffer(
        buffer, idx, clusterIdx,
        (Real3)o.clsPos, (Real)o.clsScale, (Real)o.clsWeight
    );
}

/** Order the mixture by ascending depth, packing live modes into the low slots.

    The training path seeds its 1-D Lloyd iteration with centers spread
    monotonically over [t_min, t_max], and 1-D k-means preserves that ordering,
    so the network was trained with slot 0 nearest and slot 3 farthest. Slot
    identity in *history* is deliberately stable (that is what stops the two
    depth modes from being lerped together), so the training convention is
    restored only at the point where specInput is written.
*/
ClusterSet sortClusterSetByDepth(ClusterSet cs)
{
    ClusterSet sorted = emptyClusterSet();

    bool taken[CLUSTER_MAX_SLOTS];
    [unroll]
    for (uint i = 0; i < CLUSTER_MAX_SLOTS; ++i) taken[i] = false;

    uint dst = 0;
    [unroll]
    for (uint iter = 0; iter < kClusterCount; ++iter)
    {
        int best = -1;
        float bestT = FLT_MAX;

        [unroll]
        for (uint k = 0; k < kClusterCount; ++k)
        {
            if (taken[k] || cs.w[k] <= 0.f || cs.t[k] <= 0.f) continue;
            if (cs.t[k] < bestT)
            {
                bestT = cs.t[k];
                best = (int)k;
            }
        }

        if (best < 0) break;

        taken[best] = true;
        sorted.t[dst] = cs.t[best];
        sorted.sigma[dst] = cs.sigma[best];
        sorted.w[dst] = cs.w[best];
        sorted.n[dst] = cs.n[best];
        ++dst;
    }

    return sorted;
}

/** Write the per-pixel (non-cluster) part of the specular network input:
    the normalized shading position and the encoded reflected color
    (sampleMirrorAlbedo: hit albedo, or the encoded environment color on a miss).
*/
void writeSpecPrimaryOutput(
    RWStructuredBuffer<SpecNetInput> buffer,
    uint idx,
    float3 pos,
    float3 envColor,
    float3 bboxMin,
    float3 bboxMax)
{
    float3 normPos = (pos - bboxMin) / (bboxMax - bboxMin);

    buffer[idx].data[SpecNetInput::posOffset + 0] = (Real)normPos.x;
    buffer[idx].data[SpecNetInput::posOffset + 1] = (Real)normPos.y;
    buffer[idx].data[SpecNetInput::posOffset + 2] = (Real)normPos.z;

    buffer[idx].data[SpecNetInput::envColorOffset + 0] = (Real)envColor.x;
    buffer[idx].data[SpecNetInput::envColorOffset + 1] = (Real)envColor.y;
    buffer[idx].data[SpecNetInput::envColorOffset + 2] = (Real)envColor.z;
}

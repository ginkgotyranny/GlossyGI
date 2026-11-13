/** Reproject the previous frame's per-pixel cluster mixture into the current frame.

    Addressing uses the primary hit position (gPrimaryPos) rather than gPos:
    FirstSmoothPass recurses through delta specular surfaces, so gPos can be
    behind a mirror and would reproject to the wrong pixel. The camera already
    carries prevViewProjMatNoJitter, so no external motion vector is needed.

    Validation happens at two levels:
      - per pixel: off-screen, previously non-glossy, normal / roughness / parallax
        mismatch (the last one catches broken specular chains);
      - per cluster: the world anchor must still lie inside the current cone.

    The second test is what replaces the previous attempt's global "camera moved,
    clear the whole buffer" invalidation, which effectively disabled temporal
    reuse whenever the user touched the mouse.

    Emptying the pixel (every mode fails the cone, or the pixel is rejected) is
    also the immediate unlock: n_obs drops to 0, so ConeTraceScreen starts a new
    1/(n+1) accumulation window. Losing only some modes unlocks too once the lost
    weight is real (kConeDropUnlockWeight): n_obs = sum(n)/sum(w) barely moves when
    a mode is dropped, so without this the pixel would stay locked with weights
    that no longer sum to its hit ratio, which the network reads as "escapes to
    the environment".

    Each kept anchor is re-expressed as a depth on the *current* axis, i.e. it is
    slid back onto the axis every frame. That keeps t well defined, but it also
    means a smoothly moving cone never pushes an anchor out of the cone test. The
    per-frame lateral offset is therefore added to the pixel's ClusterLockState
    drift, which ConeTraceScreen compares against unlockDrift. While the camera
    moves (its view-projection matrix changed this frame), or this frame's offset
    alone exceeds motionDrift, the pixel's observation count is capped at
    motionMemory: it keeps re-accumulating
    from a short recent window and cannot lock on samples smeared along the
    motion path (see capClusterObservations). The detector state is copied only
    when at least one mode survives, so a new occupant cannot inherit the
    previous pixel's evidence.
*/
import Scene.Shading;
import Utils.Math.MathHelpers;
import NetworkParams;
import ClusterHistory;

// Screen G-Buffer channels (current frame).
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float> gRoughness;
RWTexture2D<float4> gPrimaryPos;

// Previous frame G-Buffer copies.
Texture2D<float4> gPrevPos;
Texture2D<float4> gPrevDir;
Texture2D<float4> gPrevNormal;
Texture2D<float> gPrevRoughness;
StructuredBuffer<uint> prevSpecActive;

// Specular activity mask (current frame).
RWStructuredBuffer<uint> specActive;

// Cluster history ping-pong.
StructuredBuffer<SpecClusterHistory> clusterHistoryIn;
RWStructuredBuffer<SpecClusterHistory> clusterHistoryOut;
// Packed ClusterLockState travels with the mixture so a new occupant of this
// screen pixel cannot inherit the previous occupant's detector evidence.
StructuredBuffer<uint> clusterLockStateIn;
RWStructuredBuffer<uint> clusterLockStateOut;

/// Weight that must leave through the cone test before a locked pixel unlocks.
static const float kConeDropUnlockWeight = 0.1f;

/// Uniform in [0,1) from a 32-bit hash (PCG output permutation), for stochastic
/// rounding of the drift sum. Quality is irrelevant, only independence per pixel.
float hashUniform(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * (1.f / 4294967296.f);
}


struct ClusterReproject
{
    uint2 frameDim = {};
    uint frameCount = 0;

    /// 0 on the first frame after a clear: nothing to reproject from.
    uint historyValid = 0;

    /// Scene-scale epsilon, guards the residual denominators.
    float epsScene = 1e-4f;
    /// Minimum dot(N_prev, N_cur) to accept a reprojected pixel.
    float normalThreshold = 0.9f;
    /// Maximum |roughness_prev - roughness_cur| to accept a reprojected pixel.
    float roughnessThreshold = 0.05f;
    /// Maximum |p_prev - p_cur|, relative to the distance to the camera.
    float posThresholdRel = 0.02f;
    /// Per-frame anchor drift (cone radii) above which the pixel counts as moving.
    float motionDrift = 0.002f;
    /// Observation cap while moving: the mixture only remembers ~this many samples.
    float motionMemory = 4.f;

    /** True when the camera moved since the previous frame. A still camera gives
        bit-identical matrices. Per-pixel drift alone is not enough: on a rough
        surface the cone is so wide that a moving camera barely shifts the anchors
        relative to it, yet the reflection still smears along the motion path.
    */
    bool cameraMoved()
    {
        const float4x4 a = gScene.camera.data.viewProjMatNoJitter;
        const float4x4 b = gScene.camera.data.prevViewProjMatNoJitter;
        [unroll]
        for (uint r = 0; r < 4; ++r)
        {
            if (any(a[r] != b[r])) return true;
        }
        return false;
    }

    /** Screen position of a world point in the previous frame, in pixels. */
    bool reprojectPixel(const uint2 pixel, const float3 worldPos, out int2 prevPixel)
    {
        prevPixel = int2(0, 0);

        float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(worldPos, 1.f));
        if (prevPosH.w <= 1e-6f) return false;

        const float2 pixelCrd = float2(pixel) + float2(0.5f, 0.5f);
        const float2 frameDimF = float2(frameDim);
        // calcMotionVector() owns the clip-to-screen convention (incl. FALCOR_FLIP_Y).
        const float2 prevCrd = calcMotionVector(pixelCrd, prevPosH, frameDimF) + pixelCrd / frameDimF;

        const float2 prevPixelF = prevCrd * frameDimF;
        prevPixel = int2(floor(prevPixelF));

        return all(prevPixel >= int2(0, 0)) && all(prevPixel < int2(frameDim));
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        ClusterSet cs = emptyClusterSet();
        uint lockStateOut = 0;
        bool anyKept = false;
        float droppedWeight = 0.f;
        float driftSum = 0.f;
        float keptWeight = 0.f;

        if (specActive[pixIdx] != 0 && historyValid != 0)
        {
            const float3 posCur = gPos[pixel].xyz;
            const float3 normalCur = normalize(gNormal[pixel].xyz);
            const float3 reflectCur = normalize(reflect(gDir[pixel].xyz, normalCur));
            const float roughnessCur = gRoughness[pixel];

            float tanLobe = 0.f;
            float cosLobe = 1.f;
            evalConeLobe(roughnessCur, tanLobe, cosLobe);

            const float4 primary = gPrimaryPos[pixel];

            int2 prevPixelI;
            if (primary.w > 0.f && reprojectPixel(pixel, primary.xyz, prevPixelI))
            {
                const uint2 prevPixel = uint2(prevPixelI);
                const uint prevIdx = prevPixel.y * frameDim.x + prevPixel.x;

                if (prevSpecActive[prevIdx] != 0)
                {
                    const float3 posPrev = gPrevPos[prevPixel].xyz;
                    const float3 normalPrev = normalize(gPrevNormal[prevPixel].xyz);
                    const float3 reflectPrev = normalize(reflect(gPrevDir[prevPixel].xyz, normalPrev));
                    const float roughnessPrev = gPrevRoughness[prevPixel];

                    const float camDist = length(posCur - gScene.camera.getPosition());
                    const float posTolerance = posThresholdRel * camDist + epsScene;

                    const bool accept =
                        dot(normalPrev, normalCur) > normalThreshold &&
                        abs(roughnessPrev - roughnessCur) < roughnessThreshold &&
                        length(posPrev - posCur) < posTolerance;

                    if (accept)
                    {
                        const ClusterSet prev = unpackClusterSet(clusterHistoryIn[prevIdx]);

                        [unroll]
                        for (uint k = 0; k < kClusterCount; ++k)
                        {
                            if (!isClusterActive(prev, k)) continue;

                            const float3 anchor = clusterAnchor(posPrev, reflectPrev, prev.t[k]);

                            float tCur = 0.f;
                            if (!clusterInCone(anchor, posCur, reflectCur, cosLobe, tCur))
                            {
                                droppedWeight += prev.w[k];
                                continue;
                            }

                            driftSum += prev.w[k] * clusterLateralDrift(anchor, posCur, reflectCur, tCur, tanLobe, epsScene);
                            keptWeight += prev.w[k];

                            // Slot index is the mode identity: never reorder.
                            cs.t[k] = tCur;
                            cs.sigma[k] = prev.sigma[k];
                            cs.w[k] = prev.w[k];
                            cs.n[k] = prev.n[k];
                            anyKept = true;
                        }

                        // Cone-fail of every mode empties the pixel (n_obs = 0),
                        // which is the immediate unlock. Only carry the detector
                        // state when the mixture itself survived.
                        if (anyKept)
                        {
                            ClusterLockState lockState = unpackClusterLockState(clusterLockStateIn[prevIdx]);
                            const float frameDrift = keptWeight > 0.f ? driftSum / keptWeight : 0.f;
                            lockState.drift += frameDrift;
                            if (cameraMoved() || frameDrift > motionDrift) capClusterObservations(cs, motionMemory);
                            const uint seed = pixIdx * 2u + frameCount * 0x9E3779B9u;
                            lockStateOut = packClusterLockState(lockState, hashUniform(seed), hashUniform(seed + 1u));
                            if (droppedWeight >= kConeDropUnlockWeight) unlockClusterSet(cs);
                        }
                    }
                }
            }
        }

        clusterHistoryOut[pixIdx] = packClusterSet(cs);
        clusterLockStateOut[pixIdx] = lockStateOut;
    }
}


ParameterBlock<ClusterReproject> gClusterReproject;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gClusterReproject.execute(dispatchThreadId.xy);
}

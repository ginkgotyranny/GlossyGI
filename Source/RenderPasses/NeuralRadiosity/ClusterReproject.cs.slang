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
    1/(n+1) accumulation window. The residual-outlier streak is copied only when
    at least one mode survives, so a new occupant cannot inherit the previous
    pixel's unlock countdown.
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
// Residual-outlier streak travels with the mixture so a new occupant of this
// screen pixel cannot inherit the previous occupant's unlock countdown.
StructuredBuffer<uint> residualStreakIn;
RWStructuredBuffer<uint> residualStreakOut;


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
        uint streakOut = 0;
        bool anyKept = false;

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
                            if (!clusterInCone(anchor, posCur, reflectCur, cosLobe, tCur)) continue;

                            // Slot index is the mode identity: never reorder.
                            cs.t[k] = tCur;
                            cs.sigma[k] = prev.sigma[k];
                            cs.w[k] = prev.w[k];
                            cs.n[k] = prev.n[k];
                            anyKept = true;
                        }

                        // Cone-fail of every mode empties the pixel (n_obs = 0),
                        // which is the immediate unlock. Only copy the streak
                        // when the mixture itself survived.
                        if (anyKept) streakOut = residualStreakIn[prevIdx];
                    }
                }
            }
        }

        clusterHistoryOut[pixIdx] = packClusterSet(cs);
        residualStreakOut[pixIdx] = streakOut;
    }
}


ParameterBlock<ClusterReproject> gClusterReproject;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gClusterReproject.execute(dispatchThreadId.xy);
}

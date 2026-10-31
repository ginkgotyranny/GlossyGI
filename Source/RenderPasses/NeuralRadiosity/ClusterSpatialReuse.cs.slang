/** Spatial cluster reuse and specInput assembly for the render path.

    Temporal accumulation converges too slowly on freshly disoccluded pixels, so
    the neighbourhood fills in the missing modes for the current frame. Taps are
    gathered inside the cone's screen-space projection and accepted by exactly
    the same geometric test as the temporal path: the neighbour's world anchor
    must lie inside *our* cone. Being purely geometric, that test rejects curved
    surfaces, roughness jumps and occlusion differences on its own, which fits
    the problem better than normal/depth bilateral weights.

    The merged mixture is a convex combination, `M = (1-g) M_self + g M_neigh`,
    with g rising as the pixel's own evidence (sum n_k) falls. It is written to
    specInput only and deliberately *not* stored back into history: otherwise the
    blend would diffuse across reflection silhouettes frame after frame and decay
    into an average depth.

    Locked pixels skip the gather entirely. A neighbour still accumulating would
    otherwise leak jitter into an already-frozen mixture. Tap rotation is a
    per-pixel hash, not `frameCount`, so an unlocked pixel's ring does not
    crawl across the screen every frame.
*/
import Scene.Shading;
import NetworkParams;
import PrefiltEnvCubemap;
import ClusterHistory;
import MirrorAlbedo;

#include "Utils/Math/MathConstants.slangh"

#ifndef CLUSTER_SPATIAL_REUSE_GROUP_SIZE
#define CLUSTER_SPATIAL_REUSE_GROUP_SIZE 64
#endif

/// Golden angle in radians; used with a per-pixel hash, not the frame index.
static const float kGoldenAngle = 2.39996323f;

// Compacted specular source index (screen pixIdx). G-Buffer is gathered below.
RWStructuredBuffer<uint> specSrc;
RWStructuredBuffer<SpecNetInput> specInput;
RWStructuredBuffer<uint> specActive;

// Screen G-Buffer channels, used for self and neighbour cone frames.
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float> gRoughness;

// Cluster history (read only here).
StructuredBuffer<SpecClusterHistory> clusterHistory;


struct ClusterSpatialReuse
{
    uint2 frameDim = {};
    uint specSize = 0;

    float3 bboxMin;
    float3 bboxMax;

    /// Scene-scale epsilon, guards the residual denominators.
    float epsScene = 1e-4f;
    /// lambda: residual below which a neighbour depth joins one of our modes.
    float assignThreshold = 2.0f;
    /// Observation count at which the pixel's own mixture is frozen.
    float lockCount = 32.0f;

    /// 0 disables the gather and writes the temporal result unchanged.
    uint enableSpatialReuse = 1;
    /// 0 keeps the environment color and skips the mirror ray.
    uint enableMirrorAlbedo = 1;
    /// Number of taps on the gather ring.
    uint numTaps = 5;
    /// Gather radius scale applied to the cone's screen-space projection.
    float gatherRadiusScale = 1.0f;
    /// Clamp on the gather radius, in pixels.
    float maxGatherRadius = 8.0f;
    /// g when the pixel has no evidence of its own.
    float gammaMax = 0.6f;
    /// g when the pixel is fully converged.
    float gammaMin = 0.05f;
    /// sum n_k at which the pixel is considered fully converged.
    float confidenceCount = 32.0f;

    /** World size of one pixel at `worldPos`, used to convert the cone's world
        footprint into a screen-space gather radius.
    */
    float pixelFootprint(const float3 worldPos)
    {
        const float camDist = length(worldPos - gScene.camera.getPosition());
        const float tanHalfFovY = 0.5f * gScene.camera.data.frameHeight / max(gScene.camera.data.focalLength, 1e-6f);
        return max(2.f * camDist * tanHalfFovY / max(float(frameDim.y), 1.f), 1e-8f);
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
        const float3 reflectDir = normalize(reflect(rayDir, normalize(gNormal[pixel].xyz)));

        float tanLobe = 0.f;
        float cosLobe = 1.f;
        evalConeLobe(roughness, tanLobe, cosLobe);

        const ClusterSet self = unpackClusterSet(clusterHistory[pixIdx]);
        ClusterSet merged = self;

        // Locked pixels keep the frozen mixture. Gathering from neighbours that
        // are still accumulating would write their jitter into specInput.
        if (enableSpatialReuse != 0 && numTaps > 0 && !isClusterLocked(self, lockCount))
        {
            ClusterAccum acc = emptyClusterAccum();

            const float meanT = clusterMeanDepth(self);
            const float radius = clamp(
                gatherRadiusScale * tanLobe * meanT / pixelFootprint(pos), 1.f, maxGatherRadius
            );
            const uint hash = pixel.x * 1664525u + pixel.y * 1013904223u;
            const float baseAngle = float(hash % 64u) * kGoldenAngle;

            uint validTaps = 0;

            [loop]
            for (uint i = 0; i < numTaps; ++i)
            {
                const float theta = baseAngle + M_2PI * float(i) / float(numTaps);
                const int2 offset = int2(round(radius * float2(cos(theta), sin(theta))));
                if (all(offset == int2(0, 0))) continue;

                const int2 tapI = int2(pixel) + offset;
                if (any(tapI < int2(0, 0)) || any(tapI >= int2(frameDim))) continue;

                const uint2 tap = uint2(tapI);
                const uint tapIdx = tap.y * frameDim.x + tap.x;
                if (specActive[tapIdx] == 0) continue;

                const float3 tapPos = gPos[tap].xyz;
                const float3 tapNormal = normalize(gNormal[tap].xyz);
                const float3 tapReflect = normalize(reflect(gDir[tap].xyz, tapNormal));
                const ClusterSet neighbor = unpackClusterSet(clusterHistory[tapIdx]);

                uint neighborModes = 0;
                uint accepted = 0;

                [unroll]
                for (uint k = 0; k < kClusterCount; ++k)
                {
                    if (!isClusterActive(neighbor, k) || neighbor.w[k] <= 0.f) continue;
                    ++neighborModes;

                    const float3 anchor = clusterAnchor(tapPos, tapReflect, neighbor.t[k]);

                    float tCur = 0.f;
                    if (!clusterInCone(anchor, pos, reflectDir, cosLobe, tCur)) continue;
                    ++accepted;

                    const int slot = routeNeighborCluster(self, acc, tCur, tanLobe, epsScene, assignThreshold);
                    accumulateNeighborCluster(acc, slot, neighbor.w[k], tCur, neighbor.sigma[k]);
                }

                // A tap only votes when its evidence is actually about our cone.
                // A neighbour whose modes all sit outside the cone says nothing
                // about us, so counting it would shrink our weights for free; a
                // neighbour with no modes at all does carry information (its cone
                // escapes to the environment) and is counted.
                if (accepted > 0 || neighborModes == 0) ++validTaps;
            }

            if (validTaps > 0)
            {
                // A pixel that has barely observed anything leans on its neighbours.
                const float confidence = saturate(clusterCountSum(self) / max(confidenceCount, 1e-3f));
                const float gamma = lerp(gammaMax, gammaMin, confidence);
                const float invTaps = 1.f / float(validTaps);

                [unroll]
                for (uint k = 0; k < kClusterCount; ++k)
                {
                    if (acc.sumW[k] > 1e-6f)
                    {
                        const float neighborT = acc.sumWT[k] / acc.sumW[k];
                        const float neighborSigma = acc.sumWSigma[k] / acc.sumW[k];

                        if (isClusterActive(self, k))
                        {
                            merged.t[k] = lerp(merged.t[k], neighborT, gamma);
                            merged.sigma[k] = lerp(merged.sigma[k], neighborSigma, gamma);
                        }
                        else
                        {
                            merged.t[k] = neighborT;
                            merged.sigma[k] = neighborSigma;
                        }
                    }

                    // Convex combination keeps sum(w_k) <= 1 automatically.
                    merged.w[k] = (1.f - gamma) * merged.w[k] + gamma * (acc.sumW[k] * invTaps);
                }
            }
        }

        // Restore the training path's slot convention (slot 0 nearest) before the
        // mixture reaches the network. History keeps its own stable slot identity.
        const ClusterSet outCs = sortClusterSetByDepth(merged);

        [unroll]
        for (uint k = 0; k < kClusterCount; ++k)
        {
            const ClusterOutput o = evalClusterOutput(
                pos, reflectDir, tanLobe, outCs.t[k], outCs.sigma[k], outCs.w[k], bboxMin, bboxMax
            );

            writeSpecInputClsBuffer(
                specInput, idx, k, (Real3)o.clsPos, (Real)o.clsScale, (Real)o.clsWeight
            );
        }

        const float3 envColor = gPrefiltEnv.valid != 0 ? gPrefiltEnv.eval(reflectDir, roughness) :
            (gPrefiltEnv.fallbackValid != 0 ? gScene.envMap.eval(reflectDir) : float3(0.f));

        writeSpecPrimaryOutput(specInput, idx, pos, sampleMirrorAlbedo(pos, reflectDir, envColor, enableMirrorAlbedo), bboxMin, bboxMax);
    }
}


ParameterBlock<ClusterSpatialReuse> gClusterSpatialReuse;


[numthreads(CLUSTER_SPATIAL_REUSE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gClusterSpatialReuse.execute(dispatchThreadId.x);
}

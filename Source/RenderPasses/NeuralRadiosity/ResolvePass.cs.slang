import Scene.Shading;
import NetworkParams;

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;

RWTexture2D<float4> tempColor;

// Output G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;

// First smooth buffers
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<NetworkOutput> diffColor;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<NetworkOutput> specColor;
RWStructuredBuffer<PackedHitInfo> specVBuffer;

// debug
RWStructuredBuffer<float4> debug;
struct TemporalSpecHistory
{
    float mean[4];
    float std[4];
    float count[4];
};
// RWStructuredBuffer<TemporalSpecHistory> temporalHistory;

// Debug: per-spec-pixel dominant grid level (written by ConeTrace).
RWStructuredBuffer<float> gSpecLevel;


#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

#ifndef DEBUG_RENDER_DIFFUSE_ONLY
#define DEBUG_RENDER_DIFFUSE_ONLY 0
#endif

#ifndef DEBUG_RENDER_SPECULAR_ONLY
#define DEBUG_RENDER_SPECULAR_ONLY 0
#endif

#ifndef DEBUG_SHOW_SPEC_LEVEL
#define DEBUG_SHOW_SPEC_LEVEL 0
#endif

#ifndef N_GRID_LEVELS
#define N_GRID_LEVELS 6
#endif

struct ResolvePass
{
    static const uint kRayFlags = RAY_FLAGS;
    static const uint numClusters = NUM_CLUSTERS;
    static const bool kDebugDiffuseOnly = (DEBUG_RENDER_DIFFUSE_ONLY != 0);
    static const bool kDebugSpecularOnly = (DEBUG_RENDER_SPECULAR_ONLY != 0);

    uint2 frameDim = {};
    uint frameCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        uint pixIdx = pixel.y * frameDim.x + pixel.x;
        float4 thp = gColor[pixel];

        // Debug: visualize per-spec-pixel grid level as color.
        // Level 0 (coarsest) -> red, level N-1 (finest) -> blue, mid -> green.
        // Diffuse / inactive pixels -> black so only spec regions are shown.
        if (DEBUG_SHOW_SPEC_LEVEL != 0)
        {
            if (specActive[pixIdx] != 0)
            {
                const uint specIdx = specIndex[pixIdx];
                float lvl = gSpecLevel[specIdx];
                if (lvl < 0.0f)
                {
                    gColor[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
                }
                else
                {
                    float t = saturate(lvl / float(N_GRID_LEVELS - 1));
                    // Red (0) -> Green (0.5) -> Blue (1)
                    float3 col;
                    if (t < 0.5f)
                    {
                        col = lerp(float3(1, 0, 0), float3(0, 1, 0), t * 2.0f);
                    }
                    else
                    {
                        col = lerp(float3(0, 1, 0), float3(0, 0, 1), (t - 0.5f) * 2.0f);
                    }
                    gColor[pixel] = float4(col, 1.0f);
                }
            }
            else
            {
                gColor[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
            }
            tempColor[pixel] = gColor[pixel];
            return;
        }

        // When isolating a component, the other component's active mask was
        // zeroed in FirstSmoothPass, so the inactive branch naturally falls
        // through to the kept branch. No extra branching needed here.
        if (specActive[pixIdx] != 0)
        {
            const uint specIdx = specIndex[pixIdx];
            gColor[pixel] = thp * specColor[specIdx].toFloat4();
        }
        else if (diffActive[pixIdx] != 0)
        {
            const uint diffIdx = diffIndex[pixIdx];
            gColor[pixel] = thp * diffColor[diffIdx].toFloat4();
        }
        else if (kDebugDiffuseOnly || kDebugSpecularOnly)
        {
            // Isolated component had nothing to contribute for this pixel:
            // write pure black so the screen clearly shows only the kept lobe.
            gColor[pixel] = float4(0.f, 0.f, 0.f, 0.f);
        }

        tempColor[pixel] = gColor[pixel];
    }
}


ParameterBlock<ResolvePass> gResolvePass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gResolvePass.execute(dispatchThreadId.xy);
}

import Scene.Shading;

// Debug dump pass: copies per-spec-pixel debug buffers into screen-sized
// textures for CPU readback and disk save. Driven by DEBUG_DUMP define on
// the host side. Non-spec pixels receive a sentinel value (NaN for floats,
// 0xFFFFFFFF for uint) so they are easy to identify in the saved images.

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

// Per-spec-pixel debug buffers (written by ConeTrace).
RWStructuredBuffer<uint>   gSpecActive;        // active mask (1 = spec pixel)
RWStructuredBuffer<uint>   gSpecIndex;         // spec pixel index per screen pixel
RWStructuredBuffer<float>  gSpecLevel;         // weighted-avg grid level
RWStructuredBuffer<float>  gSpecTanLobe;       // tanLobe value
RWStructuredBuffer<float4> gSpecClusterMean;   // 4 cluster means (xyzw)
RWStructuredBuffer<float4> gSpecClusterScale;  // 4 cluster scales (xyzw)
RWStructuredBuffer<float4> gSpecClusterWeight; // 4 cluster weights (xyzw)
RWStructuredBuffer<float4> gSpecClusterPosX;   // 4 cluster pos.x (xyzw)
RWStructuredBuffer<float4> gSpecClusterPosY;   // 4 cluster pos.y (xyzw)
RWStructuredBuffer<float4> gSpecClusterPosZ;   // 4 cluster pos.z (xyzw)
RWStructuredBuffer<float2> gSpecMaterialData;  // (materialID, roughness)

// Output textures (all RGBA32Float).
RWTexture2D<float4> gDumpSpecLevel;      // r=level, gba=0
RWTexture2D<float4> gDumpTanLobe;        // r=tanLobe, gba=0
RWTexture2D<float4> gDumpClusterMean;    // xyzw = 4 cluster means
RWTexture2D<float4> gDumpClusterScale;   // xyzw = 4 cluster scales
RWTexture2D<float4> gDumpClusterWeight;  // xyzw = 4 cluster weights
RWTexture2D<float4> gDumpClusterPosX;    // xyzw = 4 cluster pos.x
RWTexture2D<float4> gDumpClusterPosY;    // xyzw = 4 cluster pos.y
RWTexture2D<float4> gDumpClusterPosZ;    // xyzw = 4 cluster pos.z
RWTexture2D<float4> gDumpMaterialData;   // r=materialID, g=roughness, ba=0

struct DebugDumpPass
{
    static const uint numClusters = NUM_CLUSTERS;

    uint2 frameDim = {};

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        if (gSpecActive[pixIdx] != 0)
        {
            const uint specIdx = gSpecIndex[pixIdx];
            gDumpSpecLevel[pixel]      = float4(gSpecLevel[specIdx], 0.f, 0.f, 1.f);
            gDumpTanLobe[pixel]        = float4(gSpecTanLobe[specIdx], 0.f, 0.f, 1.f);
            gDumpClusterMean[pixel]    = gSpecClusterMean[specIdx];
            gDumpClusterScale[pixel]   = gSpecClusterScale[specIdx];
            gDumpClusterWeight[pixel]  = gSpecClusterWeight[specIdx];
            gDumpClusterPosX[pixel]    = gSpecClusterPosX[specIdx];
            gDumpClusterPosY[pixel]    = gSpecClusterPosY[specIdx];
            gDumpClusterPosZ[pixel]    = gSpecClusterPosZ[specIdx];
            const float2 matData       = gSpecMaterialData[specIdx];
            gDumpMaterialData[pixel]   = float4(matData.x, matData.y, 0.f, 1.f);
        }
        else
        {
            // Non-spec pixel: sentinel values.
            const float nan = asfloat(0x7FC00000u);
            gDumpSpecLevel[pixel]      = float4(nan, 0.f, 0.f, 0.f);
            gDumpTanLobe[pixel]        = float4(nan, 0.f, 0.f, 0.f);
            gDumpClusterMean[pixel]    = float4(nan, nan, nan, nan);
            gDumpClusterScale[pixel]   = float4(nan, nan, nan, nan);
            gDumpClusterWeight[pixel]  = float4(nan, nan, nan, nan);
            gDumpClusterPosX[pixel]    = float4(nan, nan, nan, nan);
            gDumpClusterPosY[pixel]    = float4(nan, nan, nan, nan);
            gDumpClusterPosZ[pixel]    = float4(nan, nan, nan, nan);
            gDumpMaterialData[pixel]   = float4(nan, nan, 0.f, 0.f);
        }
    }
}

ParameterBlock<DebugDumpPass> gDebugDumpPass;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gDebugDumpPass.execute(dispatchThreadId.xy);
}

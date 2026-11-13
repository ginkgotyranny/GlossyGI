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
RWTexture2D<float4> gPrimaryPos;

// First smooth buffers
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<NetworkOutput> diffColor;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<NetworkOutput> specColor;

// debug
RWStructuredBuffer<float4> debug;


#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#include "NCRConfig.h"

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS NCR_NUM_CLUSTERS
#endif

struct ResolvePass
{
    static const uint kRayFlags = RAY_FLAGS;
    static const uint numClusters = NUM_CLUSTERS;

    uint2 frameDim = {};
    uint frameCount = 0;
    uint diffSize = 0;
    uint specSize = 0;
    /// First diffuse row in diffColor; nonzero when it shares a buffer with specColor.
    uint diffColorOffset = 0;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        uint pixIdx = pixel.y * frameDim.x + pixel.x;
        float4 thp = gColor[pixel];

        if (specActive[pixIdx] != 0)
        {
            const uint specIdx = specIndex[pixIdx];
            gColor[pixel] = thp * specColor[specIdx].toFloat4();
        }
        else if (diffActive[pixIdx] != 0)
        {
            const uint diffIdx = diffColorOffset + diffIndex[pixIdx];
            gColor[pixel] = thp * diffColor[diffIdx].toFloat4();
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

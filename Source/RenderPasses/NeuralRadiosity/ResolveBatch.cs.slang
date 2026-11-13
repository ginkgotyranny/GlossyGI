import Scene.Shading;
import NetworkParams;

// Ray batch buffers
RWStructuredBuffer<float4> allPos;
RWStructuredBuffer<float4> allDir;
RWStructuredBuffer<float4> allNormal;
RWStructuredBuffer<float4> allAlbedo;
RWStructuredBuffer<float> allRoughness;
RWStructuredBuffer<PackedHitInfo> allVBuffer;
RWStructuredBuffer<float4> allColor;
RWStructuredBuffer<float4> allEmission;

RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<DiffNetInput> diffInput;
RWStructuredBuffer<NetworkOutput> diffColor;

RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<SpecNetInput> specInput;
RWStructuredBuffer<NetworkOutput> specColor;
RWStructuredBuffer<uint> specSrc;


#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#include "NCRConfig.h"

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS NCR_NUM_CLUSTERS
#endif

struct ResolveBatch
{
    static const uint kRayFlags = RAY_FLAGS;
    static const uint numClusters = NUM_CLUSTERS;

    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;
    uint enableDiff = 1;
    uint enableSpec = 1;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        // allColor is the BSDF-path throughput when the path ended on a surface the
        // network evaluates, and its finished radiance otherwise (emitter hit, miss,
        // backface, invalid sample). allEmission is the NEE term of the same RHS
        // sample. It is independent of where the BSDF ray went, so it must be added
        // in every case: dropping it for terminated paths loses direct light in
        // proportion to how often BSDF rays hit emitters (living-room-2 windows).
        float4 color = allColor[idx];

        if (enableSpec != 0 && specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            color *= specColor[specIdx].toFloat4();
        }
        else if (enableDiff != 0 && diffActive[idx] != 0)
        {
            const uint diffIdx = diffIndex[idx];
            color *= diffColor[diffIdx].toFloat4();
        }
        allColor[idx] = color + allEmission[idx];
    }
}


ParameterBlock<ResolveBatch> gResolveBatch;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gResolveBatch.execute(dispatchThreadId.x);
}

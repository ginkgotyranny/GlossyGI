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
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<uint2> specPixel;
RWStructuredBuffer<float4> specPos;
RWStructuredBuffer<float4> specDir;
RWStructuredBuffer<float> specRoughness;


#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

struct ResolveBatch
{
    static const uint kRayFlags = RAY_FLAGS;
    static const uint numClusters = NUM_CLUSTERS;

    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        float4 thp = allColor[idx];

        if (specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            allColor[idx] = thp * specColor[specIdx].toFloat4() + allEmission[idx];
        }
        else if (diffActive[idx] != 0)
        {
            const uint diffIdx = diffIndex[idx];
            allColor[idx] = thp * diffColor[diffIdx].toFloat4() + allEmission[idx];
        }
    }
}


ParameterBlock<ResolveBatch> gResolveBatch;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gResolveBatch.execute(dispatchThreadId.x);
}

import Scene.Shading;
import NetworkParams;


// RHS color buffer
RWStructuredBuffer<float4> rhsColor;
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

#ifndef NUM_RHS
#define NUM_RHS 32
#endif

struct ResolveRHS
{
    static const uint kRayFlags = RAY_FLAGS;
    static const uint numClusters = NUM_CLUSTERS;
    static const uint numRHS = NUM_RHS;

    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        float4 lhsColor = float4(0.f);

        for (uint i = 0; i < numRHS; i++)
        {
            const uint rhsIdx = idx * numRHS + i;
            lhsColor += rhsColor[rhsIdx];
        }
        lhsColor /= numRHS;
        lhsColor = select(isnan(lhsColor) | !isfinite(lhsColor), float4(0.f), lhsColor);

        if (specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            specColor[specIdx].clear();
            specColor[specIdx].fromFloat4(lhsColor);
        }
        else if (diffActive[idx] != 0)
        {
            const uint diffIdx = diffIndex[idx];
            diffColor[diffIdx].clear();
            diffColor[diffIdx].fromFloat4(lhsColor);
        }
    }
}


ParameterBlock<ResolveRHS> gResolveRHS;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gResolveRHS.execute(dispatchThreadId.x);
}

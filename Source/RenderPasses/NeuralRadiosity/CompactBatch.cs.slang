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


struct CompactBatch
{
    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;
    uint enableDiff = 1;
    uint enableSpec = 1;

    float3 bboxMin;
    float3 bboxMax;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        if (enableDiff != 0 && diffActive[idx] != 0)
        {
            const uint diffIdx = diffIndex[idx];
            writeDiffInputGBuffer(
                diffInput, diffIdx,
                (Real3) ((allPos[idx].xyz - bboxMin) / (bboxMax - bboxMin)),
                (Real3) allDir[idx].xyz,
                (Real3) allNormal[idx].xyz,
                (Real3) allAlbedo[idx].xyz,
                (Real) allRoughness[idx]
            );
        }

        if (enableSpec != 0 && specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            specSrc[specIdx] = idx;
            writeSpecInputGBuffer(
                specInput, specIdx,
                (Real3) allPos[idx].xyz,
                (Real3) allDir[idx].xyz,
                (Real3) allNormal[idx].xyz,
                (Real3) allAlbedo[idx].xyz,
                (Real) allRoughness[idx]
            );
        }
    }
}


ParameterBlock<CompactBatch> gCompactBatch;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCompactBatch.execute(dispatchThreadId.x);
}

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


struct CompactBatch
{
    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    float3 bboxMin;
    float3 bboxMax;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        if (diffActive[idx] != 0)
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

        if (specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            specVBuffer[specIdx] = allVBuffer[idx];
            specPixel[specIdx] = uint2(idx & 0xFFFu, idx >> 12);
            specPos[specIdx] = allPos[idx];
            specDir[specIdx] = allDir[idx];
            specRoughness[specIdx] = allRoughness[idx];
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

import Scene.Shading;
import NetworkParams;

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;

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
RWStructuredBuffer<DiffNetInput> diffInput;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<SpecNetInput> specInput;
RWStructuredBuffer<uint> specSrc;


struct CompactPass
{
    uint2 frameDim = {};
    uint frameCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    float3 bboxMin;
    float3 bboxMax;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        if (diffActive[pixIdx] != 0)
        {
            const uint diffIdx = diffIndex[pixIdx];
            writeDiffInputGBuffer(
                diffInput, diffIdx,
                (Real3) ((gPos[pixel].xyz - bboxMin) / (bboxMax - bboxMin)),
                (Real3) (gDir[pixel].xyz),
                (Real3) (gNormal[pixel].xyz),
                (Real3) (gAlbedo[pixel].xyz),
                (Real) (gRoughness[pixel])
            );
        }

        if (specActive[pixIdx] != 0)
        {
            const uint specIdx = specIndex[pixIdx];
            specSrc[specIdx] = pixIdx;
            writeSpecInputGBuffer(
                specInput, specIdx,
                (Real3) (gPos[pixel].xyz),
                (Real3) (gDir[pixel].xyz),
                (Real3) (gNormal[pixel].xyz),
                (Real3) (gAlbedo[pixel].xyz),
                (Real) (gRoughness[pixel])
            );
        }
    }
}


ParameterBlock<CompactPass> gCompactPass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCompactPass.execute(dispatchThreadId.xy);
}

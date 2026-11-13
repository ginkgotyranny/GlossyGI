// Probe: primary hash-grid lookup (HashGrid::forwardKernel), one thread per sample.
// Output is channel-planar fp16, the layout the NPU reads: feature c of sample n at c * capacity + n.
import HashGrid;

struct Params
{
    uint count;      // samples
    uint capacity;   // plane length of gEnc
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float4> gSamples;       // xyz: position in [0, 1]^3
[[vk::binding(1, 0)]] StructuredBuffer<float16_t2> gPrimGrid;
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> gEnc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint n = tid.x;
    if (n >= gParams.count) return;
    const float3 pos = gSamples[n].xyz;

    [ForceUnroll]
    for (uint level = 0; level < kPrimLevels; level++)
    {
        const GridLevel l = gridLevel(kPrimBaseRes, level);
        const float3 posGrid = pos * float(l.res);
        const uint3 base = gridBase(posGrid);
        const float3 frac = posGrid - float3(base);

        float2 entry = 0.0f;
        [ForceUnroll]
        for (uint c = 0; c < 8; c++)
        {
            const uint3 corner = cornerOffset(c);
            entry += float2(gPrimGrid[l.offset + gridIndex(base + corner, l)]) * cornerWeight(corner, frac);
        }
        gEnc[(level * kPrimFeatures + 0) * gParams.capacity + n] = float16_t(entry.x);
        gEnc[(level * kPrimFeatures + 1) * gParams.capacity + n] = float16_t(entry.y);
    }
}

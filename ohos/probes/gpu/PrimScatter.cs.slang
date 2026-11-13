// Probe: primary hash-grid backward (HashGrid::backwardKernel). Each sample adds its encoding
// gradient, weighted by the trilinear weights, to the eight entries it read. CUDA does this with a
// floating-point atomicAdd; here the sum is a fixed-point integer, the atomic add Vulkan has.
import HashGrid;

struct Params
{
    uint count;
    uint capacity;
    float gradScale;   // fixed-point units per 1.0
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float4> gSamples;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> gGradIn;     // planar dL/d(encoding): c * capacity + n
[[vk::binding(2, 0)]] RWStructuredBuffer<int> gGrad;           // [entry * 2 + f]

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
        const uint3 base = uint3(posGrid);
        const float3 frac = posGrid - float3(base);
        const float2 g = float2(float(gGradIn[(level * kPrimFeatures + 0) * gParams.capacity + n]),
                                float(gGradIn[(level * kPrimFeatures + 1) * gParams.capacity + n])) * gParams.gradScale;
        [ForceUnroll]
        for (uint c = 0; c < 8; c++)
        {
            const uint3 corner = cornerOffset(c);
            const uint entry = l.offset + gridIndex(base + corner, l);
            const float2 v = g * cornerWeight(corner, frac);
            InterlockedAdd(gGrad[entry * kPrimFeatures + 0], int(round(v.x)));
            InterlockedAdd(gGrad[entry * kPrimFeatures + 1], int(round(v.y)));
        }
    }
}

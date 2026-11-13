// Backward pass of the hash grids: the training graph on the NPU returns the loss gradient with
// respect to every network input; the channels that hold the grid encodings go back into the grids
// here (HashGrid::backwardKernel and HashGridInterp::backwardKernel in CUDA on the desktop).
//
// Each sample adds its gradient, weighted like the lookup was, to the entries the lookup read.
// CUDA sums with a floating-point atomicAdd; Vulkan has integer atomics only, so the sum is kept in
// fixed point, in units the caller sets with gradScale (2^24 per 1.0 keeps the smallest fp16
// gradient and leaves room for the largest sums; see ohos/README.md).
//
// The positions are read from the input tiles, where EncodeInputs copied them.
// Compiled twice: SPECULAR=0 for DiffNet (primary grid), SPECULAR=1 for SpecNet (both grids).
import HashGrid;

#ifndef SPECULAR
#define SPECULAR 0
#endif

#if SPECULAR
static const uint kChannels = 64;
static const uint kPosOffset = 28;
static const uint kClsEncOffset = 12;
static const uint kClsPosOffset = 44;
static const uint kClsScaleOffset = 56;
static const uint kClsWeightOffset = 60;
#else
static const uint kChannels = 32;
static const uint kPosOffset = 12;
#endif

struct Params
{
    uint count;       // samples
    uint tileShift;   // log2 of the samples per tile
    float gradScale;  // fixed-point units per 1.0 of the graph's gradient
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t> gInputs;      // network input tiles
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> gGradients;   // dL/d(input) tiles, same layout
[[vk::binding(2, 0)]] RWStructuredBuffer<int> gPrimGrad;        // [entry * 2 + feature]
#if SPECULAR
[[vk::binding(3, 0)]] RWStructuredBuffer<int> gClsGrad;         // [entry * 4 + feature]
#endif

uint tileIndex(uint n, uint channel)
{
    const uint tile = 1u << gParams.tileShift;
    return (((n >> gParams.tileShift) * kChannels + channel) << gParams.tileShift) + (n & (tile - 1));
}

float input(uint n, uint channel) { return float(gInputs[tileIndex(n, channel)]); }
float gradient(uint n, uint channel) { return float(gGradients[tileIndex(n, channel)]); }

/// To fixed point. A gradient that is not finite (fp16 overflow on the NPU) counts as zero, as the
/// optimizer on the desktop treats it.
int fixedPoint(float v)
{
    const float limit = 1073741824.0f;   // 2^30
    return (v > -limit && v < limit) ? int(round(v)) : 0;
}

#if SPECULAR
void scatterCluster(float3 pos, uint level, float4 g)
{
    const GridLevel l = gridLevel(kClsBaseRes, level);
    const float3 posGrid = pos * float(l.res);
    const uint3 base = gridBase(posGrid);
    const float3 frac = posGrid - float3(base);
    [ForceUnroll]
    for (uint c = 0; c < 8; c++)
    {
        const uint3 corner = cornerOffset(c);
        const uint entry = (l.offset + gridIndex(base + corner, l)) * kClsFeatures;
        const float4 v = g * cornerWeight(corner, frac);
        [ForceUnroll]
        for (uint f = 0; f < kClsFeatures; f++)
        {
            const int add = fixedPoint(v[f]);
            if (add != 0) InterlockedAdd(gClsGrad[entry + f], add);
        }
    }
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint n = tid.x;
    if (n >= gParams.count) return;

    const float3 pos = float3(input(n, kPosOffset), input(n, kPosOffset + 1), input(n, kPosOffset + 2));
    [ForceUnroll]
    for (uint level = 0; level < kPrimLevels; level++)
    {
        const GridLevel l = gridLevel(kPrimBaseRes, level);
        const float3 posGrid = pos * float(l.res);
        const uint3 base = gridBase(posGrid);
        const float3 frac = posGrid - float3(base);
        const float2 g = float2(gradient(n, level * kPrimFeatures), gradient(n, level * kPrimFeatures + 1)) * gParams.gradScale;
        [ForceUnroll]
        for (uint c = 0; c < 8; c++)
        {
            const uint3 corner = cornerOffset(c);
            const uint entry = (l.offset + gridIndex(base + corner, l)) * kPrimFeatures;
            const float2 v = g * cornerWeight(corner, frac);
            const int addX = fixedPoint(v.x), addY = fixedPoint(v.y);
            if (addX != 0) InterlockedAdd(gPrimGrad[entry], addX);
            if (addY != 0) InterlockedAdd(gPrimGrad[entry + 1], addY);
        }
    }

#if SPECULAR
    [ForceUnroll]
    for (uint k = 0; k < kNumClusters; k++)
    {
        const float3 clsPos = float3(input(n, kClsPosOffset + k * 3), input(n, kClsPosOffset + k * 3 + 1), input(n, kClsPosOffset + k * 3 + 2));
        const ClusterLevels levels = clusterLevels(input(n, kClsScaleOffset + k));
        float4 g;
        [ForceUnroll]
        for (uint f = 0; f < kClsFeatures; f++) g[f] = gradient(n, kClsEncOffset + k * kClsFeatures + f);
        // The lookup multiplied the blended entry by the cluster weight; so does its gradient.
        g *= input(n, kClsWeightOffset + k) * gParams.gradScale;
        scatterCluster(clsPos, levels.upper, g * levels.upperWeight);
        scatterCluster(clsPos, levels.lower, g * levels.lowerWeight);
    }
#endif
}

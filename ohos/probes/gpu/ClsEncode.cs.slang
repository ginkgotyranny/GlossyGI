// Probe: cluster hash-grid lookup (HashGridInterp::forwardKernel), one thread per sample, four
// clusters each: two levels chosen by the cluster radius, blended, times the cluster weight.
import HashGrid;

struct Params
{
    uint count;
    uint capacity;
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float4> gClusters;      // [n * 4 + k]: xyz position, w radius
[[vk::binding(1, 0)]] StructuredBuffer<float> gClusterWeights; // [n * 4 + k]
[[vk::binding(2, 0)]] StructuredBuffer<float16_t4> gClsGrid;
[[vk::binding(3, 0)]] RWStructuredBuffer<float16_t> gEnc;      // planar: (k * 4 + f) * capacity + n

float4 lookup(float3 pos, uint level)
{
    const GridLevel l = gridLevel(kClsBaseRes, level);
    const float3 posGrid = pos * float(l.res);
    const uint3 base = gridBase(posGrid);
    const float3 frac = posGrid - float3(base);
    float4 entry = 0.0f;
    [ForceUnroll]
    for (uint c = 0; c < 8; c++)
    {
        const uint3 corner = cornerOffset(c);
        entry += float4(gClsGrid[l.offset + gridIndex(base + corner, l)]) * cornerWeight(corner, frac);
    }
    return entry;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint n = tid.x;
    if (n >= gParams.count) return;

    [ForceUnroll]
    for (uint k = 0; k < kNumClusters; k++)
    {
        const float4 cluster = gClusters[n * kNumClusters + k];
        const ClusterLevels levels = clusterLevels(cluster.w);
        const float4 upper = lookup(cluster.xyz, levels.upper);
        const float4 lower = lookup(cluster.xyz, levels.lower);
        const float4 value = (lower * levels.lowerWeight + upper * levels.upperWeight) * gClusterWeights[n * kNumClusters + k];
        [ForceUnroll]
        for (uint f = 0; f < kClsFeatures; f++)
        {
            gEnc[(k * kClsFeatures + f) * gParams.capacity + n] = float16_t(value[f]);
        }
    }
}

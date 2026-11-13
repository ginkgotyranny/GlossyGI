// Network inputs for the NPU: looks up the hash grids and lays the samples out as the NPU reads them.
//
// The NCR passes write one row of fp16 values per sample (NetworkParams.slang: DiffNetInput 32 values,
// SpecNetInput 64), with the encoding channels still empty; on the desktop a CUDA kernel fills those
// in place. The NPU takes channel-planar tensors of a fixed shape instead: tiles of TILE samples,
// channel c of sample n at ((n / TILE) * C + c) * TILE + n % TILE. The inference graphs take tiles
// of 65,536 samples, (1, C, 256, 256); the training graphs smaller ones, (1, C, TILE, 1).
// This pass does both: it copies a row into its tile and writes the encodings there. Which grids
// it reads is the caller's choice: the EMA copies for inference, the training copies for training.
//
// Compiled twice: SPECULAR=0 for DiffNet (primary grid), SPECULAR=1 for SpecNet (primary grid and
// the four cluster lookups).
import HashGrid;

#ifndef SPECULAR
#define SPECULAR 0
#endif

#if SPECULAR
static const uint kChannels = 64;                // SpecNetInput::paddedDim
static const uint kPosOffset = 28;               // after the primary (12) and cluster (16) encodings
static const uint kClsEncOffset = 12;
static const uint kClsPosOffset = 44;
static const uint kClsScaleOffset = 56;
static const uint kClsWeightOffset = 60;
#else
static const uint kChannels = 32;                // DiffNetInput::paddedDim
static const uint kPosOffset = 12;
#endif

struct Params
{
    uint count;      // samples
    uint rowBase;    // index of the first value of the first row in gRows (the diffuse rows of a shared buffer start after the specular ones)
    uint tileShift;  // log2 of the samples per tile
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t> gRows;        // sample rows, kChannels values each
[[vk::binding(1, 0)]] StructuredBuffer<float16_t2> gPrimGrid;
[[vk::binding(2, 0)]] StructuredBuffer<float16_t4> gClsGrid;
[[vk::binding(3, 0)]] RWStructuredBuffer<float16_t> gTiles;

float16_t row(uint n, uint channel)
{
    return gRows[gParams.rowBase + n * kChannels + channel];
}

uint tileIndex(uint n, uint channel)
{
    const uint tile = 1u << gParams.tileShift;
    return (((n >> gParams.tileShift) * kChannels + channel) << gParams.tileShift) + (n & (tile - 1));
}

#if SPECULAR
float4 clusterLookup(float3 pos, uint level)
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
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint n = tid.x;
    if (n >= gParams.count) return;

    // Everything after the encodings is copied as it is.
    for (uint c = kPosOffset; c < kChannels; c++)
    {
        gTiles[tileIndex(n, c)] = row(n, c);
    }

    const float3 pos = float3(row(n, kPosOffset), row(n, kPosOffset + 1), row(n, kPosOffset + 2));
    [ForceUnroll]
    for (uint level = 0; level < kPrimLevels; level++)
    {
        const GridLevel l = gridLevel(kPrimBaseRes, level);
        const float3 posGrid = pos * float(l.res);
        const uint3 base = gridBase(posGrid);
        const float3 frac = posGrid - float3(base);
        float2 entry = 0.0f;
        [ForceUnroll]
        for (uint corner = 0; corner < 8; corner++)
        {
            const uint3 o = cornerOffset(corner);
            entry += float2(gPrimGrid[l.offset + gridIndex(base + o, l)]) * cornerWeight(o, frac);
        }
        gTiles[tileIndex(n, level * kPrimFeatures + 0)] = float16_t(entry.x);
        gTiles[tileIndex(n, level * kPrimFeatures + 1)] = float16_t(entry.y);
    }

#if SPECULAR
    [ForceUnroll]
    for (uint k = 0; k < kNumClusters; k++)
    {
        const float3 clsPos = float3(row(n, kClsPosOffset + k * 3), row(n, kClsPosOffset + k * 3 + 1), row(n, kClsPosOffset + k * 3 + 2));
        const ClusterLevels levels = clusterLevels(float(row(n, kClsScaleOffset + k)));
        const float4 upper = clusterLookup(clsPos, levels.upper);
        const float4 lower = clusterLookup(clsPos, levels.lower);
        const float4 value = (lower * levels.lowerWeight + upper * levels.upperWeight) * float(row(n, kClsWeightOffset + k));
        [ForceUnroll]
        for (uint f = 0; f < kClsFeatures; f++)
        {
            gTiles[tileIndex(n, kClsEncOffset + k * kClsFeatures + f)] = float16_t(value[f]);
        }
    }
#endif
}

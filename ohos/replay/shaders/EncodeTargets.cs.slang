// Training targets for the NPU: the radiance each training sample should have, laid out as the
// training graph reads it.
//
// ResolveRHS leaves one row of four floats per sample (NetworkOutput) in the batch's color buffer.
// The training graph takes (1, 16, TILE, 1) fp16, channel-planar like its input: the radiance in
// channels 0-3 and in channel 15 a weight, 1 for a sample and 0 for the slots of the last tile that
// lie beyond the batch (ohos/npu/ncr_npu_graphs.py, MASK_CHANNEL). Channels 4-14 are never written
// and stay zero.
static const uint kChannels = 16;
static const uint kMaskChannel = 15;
static const float kLargestTarget = 60000.0f;   // fp16 ends at 65504

struct Params
{
    uint count;      // samples
    uint rowBase;    // first row in gRows
    uint tileShift;  // log2 of the samples per tile
    uint slots;      // tiles * samples per tile: every slot gets its weight
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float4> gRows;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> gTiles;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint n = tid.x;
    if (n >= gParams.slots) return;
    const uint tile = 1u << gParams.tileShift;
    const uint base = (((n >> gParams.tileShift) * kChannels) << gParams.tileShift) + (n & (tile - 1));
    const bool sample = n < gParams.count;
    float4 target = 0.0f;
    if (sample)
    {
        target = gRows[gParams.rowBase + n];
        // A NaN target would poison every weight; min/max drop it (the second operand wins).
        target = min(max(target, 0.0f), kLargestTarget);
    }
    gTiles[base] = float16_t(target.x);
    gTiles[base + tile] = float16_t(target.y);
    gTiles[base + 2 * tile] = float16_t(target.z);
    gTiles[base + 3 * tile] = float16_t(target.w);
    gTiles[base + kMaskChannel * tile] = float16_t(sample ? 1.0f : 0.0f);
}

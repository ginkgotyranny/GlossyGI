// Network outputs from the NPU back to the layout the NCR passes read.
//
// The NPU writes channel-planar fp16 tiles of TILE samples with 16 channels (the MLPs' padded output
// width), of which the first four are the radiance the passes use. ResolvePass reads one row of four
// floats per sample (NetworkOutput).
static const uint kTile = 65536;
static const uint kChannels = 16;

struct Params
{
    uint count;     // samples
    uint rowBase;   // first row in gRows (the diffuse rows of a shared buffer start after the padded specular ones)
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t> gTiles;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gRows;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint n = tid.x;
    if (n >= gParams.count) return;
    const uint base = (n / kTile) * kChannels * kTile + n % kTile;
    gRows[gParams.rowBase + n] = float4(float(gTiles[base]), float(gTiles[base + kTile]), float(gTiles[base + 2 * kTile]),
                                         float(gTiles[base + 3 * kTile]));
}

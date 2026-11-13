// Probe: shader features NCR's ConeTrace relies on and Device MRRC's GLSL never used on Maleoon:
// subgroup arithmetic (WaveActiveSum) and groupshared memory with a group barrier, in thread groups
// of 32 threads as ConeTrace uses per specular sample. Compiled a second time with USE_INT64=1 to
// test 64-bit integers, which Falcor's shader library uses in places: a device without shaderInt64
// rejects that whole module, so the two are separate pipelines.
#ifndef USE_INT64
#define USE_INT64 0
#endif
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> gOut;   // [thread * 8]

groupshared uint gsValues[32];

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint value = gtid.x + 1;
    gsValues[gtid.x] = value;
    GroupMemoryBarrierWithGroupSync();
    uint groupSum = 0;
    for (uint i = 0; i < 32; i++) groupSum += gsValues[i];

    const uint waveSum = WaveActiveSum(value);
#if USE_INT64
    const uint64_t wide = uint64_t(tid.x + 1) * 0x100000003ull;
    const uint wideLow = uint(wide & 0xffffffffull);
    const uint wideHigh = uint(wide >> 32);
#else
    const uint wideLow = 0;
    const uint wideHigh = 0;
#endif

    const uint o = tid.x * 8;
    gOut[o + 0] = WaveGetLaneIndex();
    gOut[o + 1] = WaveGetLaneCount();
    gOut[o + 2] = waveSum;
    gOut[o + 3] = groupSum;
    gOut[o + 4] = wideLow;
    gOut[o + 5] = wideHigh;
    gOut[o + 6] = value;
    gOut[o + 7] = 0xabcd1234u;
}

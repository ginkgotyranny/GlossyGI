// Probe: 16-bit loads from a byte-address buffer at offsets that are not multiples of four.
// Falcor reads the 16-bit indices of triangle t with Load<uint16_t3>(base + t * 6): every odd
// triangle starts two bytes into a 32-bit word. The buffer holds the 16-bit values 0, 1, 2, ...
// The second variant reads the same bytes through aligned 32-bit words.
[[vk::binding(0, 0)]] ByteAddressBuffer gData;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gOut;   // [triangle * 8]: direct xyz, pad, via words xyz, pad

uint3 loadViaWords(uint byteOffset)
{
    const uint2 words = gData.Load2(byteOffset & ~3u);
    return (byteOffset & 2u) != 0 ? uint3(words.x >> 16, words.y & 0xffffu, words.y >> 16)
                                  : uint3(words.x & 0xffffu, words.x >> 16, words.y & 0xffffu);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint triangle = tid.x;
    const uint3 direct = uint3(gData.Load<uint16_t3>(triangle * 6));
    const uint3 words = loadViaWords(triangle * 6);
    gOut[triangle * 8 + 0] = direct.x;
    gOut[triangle * 8 + 1] = direct.y;
    gOut[triangle * 8 + 2] = direct.z;
    gOut[triangle * 8 + 3] = 0;
    gOut[triangle * 8 + 4] = words.x;
    gOut[triangle * 8 + 5] = words.y;
    gOut[triangle * 8 + 6] = words.z;
    gOut[triangle * 8 + 7] = 0;
}

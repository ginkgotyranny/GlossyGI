// Probe: does a load see a store made earlier in the same invocation?
// NCR's ResolvePass reads a texel of a storage image, conditionally stores to it and reads it again.
// Both the image and the buffer below start at 0, 1, 2, ...; each thread adds 100 to its element when
// gParams.enable is set (a run-time condition, as in ResolvePass) and reads it back.
struct Params
{
    uint width;
    uint enable;
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] RWTexture2D<float4> gImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gBuffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gOut;   // image: first load, second load; buffer: first load, second load

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    const uint2 pixel = uint2(i % gParams.width, i / gParams.width);

    const float4 imageBefore = gImage[pixel];
    if (gParams.enable != 0)
    {
        gImage[pixel] = imageBefore + 100.0f;
    }
    const float4 imageAfter = gImage[pixel];

    const float4 bufferBefore = gBuffer[i];
    if (gParams.enable != 0)
    {
        gBuffer[i] = bufferBefore + 100.0f;
    }
    const float4 bufferAfter = gBuffer[i];

    gOut[i] = float4(imageBefore.x, imageAfter.x, bufferBefore.x, bufferAfter.x);
}

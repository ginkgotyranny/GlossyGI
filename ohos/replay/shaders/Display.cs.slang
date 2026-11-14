// The pass's HDR output as it is shown. The project's render graphs end in Falcor's ToneMapper with
// autoExposure off and exposureCompensation 0; with its other defaults (film speed 100, f-number 1,
// shutter 1, no white balance) the colour transform is the identity, the operator is ACES and the
// result is clamped. The graph's output is RGBA8UnormSrgb, so the writes are sRGB-encoded. This pass
// does the same and packs one RGBA8 value per pixel, rows top to bottom, for a copy into a swapchain
// image or a file (RuntimeDisplay in ncr_runtime.cpp).
[[vk::binding(0, 0)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> gColor;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gPixels;

struct Params
{
    uint width;
    uint height;
    uint bgra;   // 1: the swapchain stores blue first
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

// ToneMapping.ps.slang, toneMapAces.
float3 toneMapAces(float3 color)
{
    color *= 0.6;
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return saturate((color * (A * color + B)) / (color * (C * color + D) + E));
}

float linearToSrgb(float c)
{
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.width * gParams.height) return;
    const uint2 xy = uint2(i % gParams.width, i / gParams.width);
    const float3 c = saturate(toneMapAces(gColor[xy].rgb));
    uint3 q = uint3(float3(linearToSrgb(c.r), linearToSrgb(c.g), linearToSrgb(c.b)) * 255.0 + 0.5);
    if (gParams.bgra != 0) q = q.zyx;
    gPixels[i] = q.x | (q.y << 8) | (q.z << 16) | (255u << 24);
}

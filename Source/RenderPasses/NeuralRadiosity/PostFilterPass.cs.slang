import Scene.Shading;
import NetworkParams;

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;
RWTexture2D<float4> gFilteredColor;

Texture2D<float4> tempColor;
Texture2D<float4> gFxaaInput;

static const float kFXAAReduceMin = 1.f / 128.f;
static const float kFXAAReduceMul = 1.f / 8.f;
static const float kFXAASpanMax = 8.f;

// Output G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;

// First smooth buffers
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;


struct PostFilterPass
{
    uint2 frameDim = {};
    uint frameCount = 0;

    // Parameters
    int kernelRadius;
    float sigmaSpatial;
    float sigmaNormal;
    float sigmaAlbedo;
    float sigmaColor;
    uint enableFXAA = 1;

    float gaussianWeight(float distance2, float sigma)
    {
        sigma = max(sigma, 1e-4f);
        return exp(-distance2 / (2.0f * sigma * sigma));
    }

    int2 frameDimI()
    {
        return int2(int(frameDim.x), int(frameDim.y));
    }

    float2 frameDimF()
    {
        return float2(float(frameDim.x), float(frameDim.y));
    }

    uint2 clampPixel(const int2 pixel)
    {
        return uint2(clamp(pixel, int2(0, 0), frameDimI() - int2(1, 1)));
    }

    float4 loadTempColor(const int2 pixel)
    {
        return tempColor[clampPixel(pixel)];
    }

    float4 loadFxaaColor(const int2 pixel)
    {
        return gFxaaInput[clampPixel(pixel)];
    }

    float luma(const float3 color)
    {
        return dot(color, float3(0.299f, 0.587f, 0.114f));
    }

    float3 sampleFxaaBilinear(const float2 samplePos)
    {
        const float2 clampedPos = clamp(samplePos, float2(0.5f, 0.5f), frameDimF() - float2(0.5f, 0.5f));
        const float2 pixelPos = clampedPos - float2(0.5f, 0.5f);
        const int2 p0 = int2(floor(pixelPos));
        const int2 p1 = min(p0 + int2(1, 1), frameDimI() - int2(1, 1));
        const float2 t = pixelPos - float2(float(p0.x), float(p0.y));

        const float3 c00 = loadFxaaColor(p0).xyz;
        const float3 c10 = loadFxaaColor(int2(p1.x, p0.y)).xyz;
        const float3 c01 = loadFxaaColor(int2(p0.x, p1.y)).xyz;
        const float3 c11 = loadFxaaColor(p1).xyz;

        const float3 top = lerp(c00, c10, t.x);
        const float3 bottom = lerp(c01, c11, t.x);
        return lerp(top, bottom, t.y);
    }

    float3 fxaaPixel(const uint2 pixel)
    {
        const int2 p = int2(pixel);
        const float3 rgbNW = loadFxaaColor(p + int2(-1, -1)).xyz;
        const float3 rgbNE = loadFxaaColor(p + int2( 1, -1)).xyz;
        const float3 rgbSW = loadFxaaColor(p + int2(-1,  1)).xyz;
        const float3 rgbSE = loadFxaaColor(p + int2( 1,  1)).xyz;
        const float3 rgbM = loadFxaaColor(p).xyz;

        const float lumaNW = luma(rgbNW);
        const float lumaNE = luma(rgbNE);
        const float lumaSW = luma(rgbSW);
        const float lumaSE = luma(rgbSE);
        const float lumaM = luma(rgbM);
        const float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
        const float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

        float2 dir;
        dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
        dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

        const float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * kFXAAReduceMul), kFXAAReduceMin);
        const float rcpDirMin = 1.f / (min(abs(dir.x), abs(dir.y)) + dirReduce);
        dir = clamp(dir * rcpDirMin, float2(-kFXAASpanMax, -kFXAASpanMax), float2(kFXAASpanMax, kFXAASpanMax));

        const float2 fragCoord = float2(float(pixel.x), float(pixel.y)) + float2(0.5f, 0.5f);
        const float3 sampleA0 = sampleFxaaBilinear(fragCoord + dir * (1.f / 3.f - 0.5f));
        const float3 sampleA1 = sampleFxaaBilinear(fragCoord + dir * (2.f / 3.f - 0.5f));
        const float3 rgbA = 0.5f * (sampleA0 + sampleA1);

        const float3 sampleB0 = sampleFxaaBilinear(fragCoord + dir * -0.5f);
        const float3 sampleB1 = sampleFxaaBilinear(fragCoord + dir *  0.5f);
        const float3 rgbB = 0.5f * rgbA + 0.25f * (sampleB0 + sampleB1);

        const float lumaB = luma(rgbB);
        return (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
    }

    void executeBilateral(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        const float4 centerSample = loadTempColor(int2(pixel));
        const float3 centerColor = centerSample.xyz;

        if (specActive[pixIdx] != 0)
        {
            const float3 centerNormal = normalize(gNormal[pixel].xyz);
            const float3 centerAlbedo = gAlbedo[pixel].xyz;

            float3 filteredColor = float3(0.f);
            float weightSum = 0.f;

            const int radius = kernelRadius;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int2 samplePixelI = int2(pixel) + int2(dx, dy);
                    const uint2 samplePixel = clampPixel(samplePixelI);

                    // if (specActive[samplePixIdx] == 0 || gActive[samplePixel] == 0) continue;

                    const float3 sampleColor = tempColor[samplePixel].xyz;
                    const float3 sampleNormal = normalize(gNormal[samplePixel].xyz);
                    const float3 sampleAlbedo = gAlbedo[samplePixel].xyz;

                    const float spatialDist2 = float(dx * dx + dy * dy);
                    const float normalDist2 = 1.f - saturate(dot(centerNormal, sampleNormal));
                    const float albedoDist2 = dot(centerAlbedo - sampleAlbedo, centerAlbedo - sampleAlbedo);
                    const float colorDist2 = dot(centerColor - sampleColor, centerColor - sampleColor);

                    const float weight =
                        gaussianWeight(spatialDist2, sigmaSpatial) *
                        gaussianWeight(normalDist2, sigmaNormal) *
                        gaussianWeight(albedoDist2, sigmaAlbedo) *
                        gaussianWeight(colorDist2, sigmaColor);

                    filteredColor += weight * sampleColor;
                    weightSum += weight;
                }
            }

            gFilteredColor[pixel] = float4(weightSum > 0.f ? filteredColor / weightSum : centerColor, centerSample.w);
            return;
        }

        gFilteredColor[pixel] = centerSample;
    }

    void executeFXAA(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        const float4 centerSample = loadFxaaColor(int2(pixel));
        gColor[pixel] = enableFXAA != 0 ? float4(fxaaPixel(pixel), centerSample.w) : centerSample;
    }
}


ParameterBlock<PostFilterPass> gPostFilterPass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPostFilterPass.executeBilateral(dispatchThreadId.xy);
}


[numthreads(16, 16, 1)]
void mainFXAA(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPostFilterPass.executeFXAA(dispatchThreadId.xy);
}

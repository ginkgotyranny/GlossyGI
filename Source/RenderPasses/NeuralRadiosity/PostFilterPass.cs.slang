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
RWTexture2D<float4> gPrimaryPos;

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
    uint enableColorWeight = 1;
    uint enableFXAA = 1;
    float highlightMinExcess = 0.1f; // Linear HDR difference after bilateral/FXAA.
    uint highlightRequireQuietRing = 0; // Legacy diagnostic; FXAA can spread peaks into the ring.

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

    /// FXAA runs on HDR radiance, where one bilinear tap of an emitter or a window
    /// (~100) turns a neighbouring wall pixel white. Filter in a reversible
    /// tone-mapped space instead (c / (1 + luma(c)), Karis 2014), so a bright
    /// neighbour weighs at most ~1 and the result maps back to HDR afterwards.
    float3 fxaaCompress(const float3 c)
    {
        return c / (1.f + luma(max(c, 0.f)));
    }

    float3 fxaaExpand(const float3 c)
    {
        return c / max(1.f - luma(max(c, 0.f)), 1e-4f);
    }

    float4 loadFxaaColor(const int2 pixel)
    {
        const float4 c = gFxaaInput[clampPixel(pixel)];
        return float4(fxaaCompress(c.xyz), c.w);
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
        return fxaaExpand((lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB);
    }

    /** Conservative single-pixel HDR outlier repair, after bilateral and FXAA. All eight neighbours must belong to a locally continuous
        glossy surface and be strictly dimmer than the center. This cannot distinguish
        a genuine isolated subpixel reflection from noise; use a high threshold.
        Read only this stage's immutable input, never this dispatch's output.
    */
    float3 suppressIsolatedHighlight(uint2 pixel, float3 center)
    {
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;
        if (specActive[pixIdx] == 0 || gActive[pixel] == 0) return center;
        // Require eight distinct neighbours; never duplicate edge pixels.
        if (any(pixel == 0u) || any(pixel + 1u >= frameDim)) return center;
        if (!all(isfinite(center))) return center;
        const float centerLuma = luma(max(center, 0.f));
        if (centerLuma <= highlightMinExcess) return center;
        const float3 normal = normalize(gNormal[pixel].xyz);
        const float3 albedo = gAlbedo[pixel].xyz;
        const float roughness = gRoughness[pixel];
        const float3 pos = gPos[pixel].xyz;
        const float tanHalfFovY = 0.5f * gScene.camera.data.frameHeight / max(gScene.camera.data.focalLength, 1e-6f);
        const float footprint = max(2.f * length(pos - gScene.camera.getPosition()) * tanHalfFovY / float(frameDim.y), 1e-6f);
        float3 sumColor = 0.f;
        float minLuma = 3.402823466e+38f;
        float maxLuma = 0.f;
        float sumLuma = 0.f;
        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0) continue;
                const uint2 tap = uint2(int2(pixel) + int2(dx, dy));
                const float3 c = tempColor[tap].xyz;
                if (!all(isfinite(c))) return center;
                const float y = luma(max(c, 0.f));
                // Most pixels cannot exceed even the first neighbour. Reject
                // before geometry loads; the final blend would be exactly zero.
                if (centerLuma - y <= max(highlightMinExcess, y)) return center;
                if (specActive[tap.y * frameDim.x + tap.x] == 0 || gActive[tap] == 0) return center;
                if (dot(normal, normalize(gNormal[tap].xyz)) < 0.995f ||
                    any(abs(albedo - gAlbedo[tap].xyz) > 0.03f) ||
                    abs(roughness - gRoughness[tap]) > 0.02f) return center;
                const float3 delta = gPos[tap].xyz - pos;
                if (length(delta) > 4.f * footprint || abs(dot(normal, delta)) > 0.1f * footprint) return center;

                minLuma = min(minLuma, y);
                maxLuma = max(maxLuma, y);
                sumLuma += y;
                sumColor += c;
            }
        }
        // The old raw-image ring-flatness gate rejects FXAA halos. In the final
        // image, keep the strict all-eight-neighbours contrast test instead.
        // Connected bright lines still fail that test; retain the old gate for A/B.
        if (highlightRequireQuietRing != 0 && maxLuma - minLuma > max(0.1f, 0.5f * sumLuma / 8.f)) return center;
        // At least twice the brightest neighbour AND highlightMinExcess above it.
        // The post-filtered peak is weaker than raw resolve, so its default floor
        // is 0.1 HDR. Full replacement at 3x (or twice that floor), continuously.
        const float threshold = max(highlightMinExcess, maxLuma);
        const float strength = smoothstep(threshold, 2.f * threshold, centerLuma - maxLuma);
        return lerp(center, sumColor / 8.f, strength);
    }

    void executeIsolatedHighlight(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const float4 center = tempColor[pixel];
        gColor[pixel] = float4(suppressIsolatedHighlight(pixel, center.xyz), center.w);
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
                        (enableColorWeight != 0 ? gaussianWeight(colorDist2, sigmaColor) : 1.f);

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

        const float4 centerSample = gFxaaInput[pixel];
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

[numthreads(16, 16, 1)]
void mainIsolatedHighlight(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPostFilterPass.executeIsolatedHighlight(dispatchThreadId.xy);
}

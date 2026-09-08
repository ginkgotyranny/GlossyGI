import Scene.Shading;

#ifndef PREFILT_ENV_CUBEMAP_BUILD_PASS
#define PREFILT_ENV_CUBEMAP_BUILD_PASS 0
#endif

struct PrefiltEnvCubemap
{
    Texture2DArray<float4> cubemap;
    SamplerState sampler;
    uint faceSize = 0;
    uint mipCount = 0;
    uint valid = 0;
    uint fallbackValid = 0;

    void dirToFaceUv(float3 dir, out uint face, out float2 uv)
    {
        float3 a = abs(dir);
        float2 p = float2(0.f);

        if (a.x >= a.y && a.x >= a.z)
        {
            if (dir.x >= 0.f)
            {
                face = 0;
                p = float2(-dir.z, dir.y) / a.x;
            }
            else
            {
                face = 1;
                p = float2(dir.z, dir.y) / a.x;
            }
        }
        else if (a.y >= a.x && a.y >= a.z)
        {
            if (dir.y >= 0.f)
            {
                face = 2;
                p = float2(dir.x, -dir.z) / a.y;
            }
            else
            {
                face = 3;
                p = float2(dir.x, dir.z) / a.y;
            }
        }
        else
        {
            if (dir.z >= 0.f)
            {
                face = 4;
                p = float2(dir.x, dir.y) / a.z;
            }
            else
            {
                face = 5;
                p = float2(-dir.x, dir.y) / a.z;
            }
        }

        uv = float2(p.x, -p.y) * 0.5f + 0.5f;
    }

    float3 eval(float3 dir, float roughness)
    {
        if (valid == 0 || mipCount == 0)
            return float3(0.f);

        uint face = 0;
        float2 uv = float2(0.f);
        dirToFaceUv(normalize(dir), face, uv);

        static const float kNdfCdf = 0.9f;
        float r = saturate(roughness) / 2.0f;
        float alpha = r * r;
        float tanTheta = alpha * sqrt(kNdfCdf / (1.f - kNdfCdf));
        float lod = clamp(log2(max(tanTheta * float(faceSize), 1.f)), 0.f, float(mipCount - 1));
        return cubemap.SampleLevel(sampler, float3(uv, face), lod).rgb;
    }
};

#if !PREFILT_ENV_CUBEMAP_BUILD_PASS
ParameterBlock<PrefiltEnvCubemap> gPrefiltEnv;
#endif

#if PREFILT_ENV_CUBEMAP_BUILD_PASS
struct PrefiltEnvCubemapPass
{
    uint2 outputDim = {};

    RWTexture2DArray<float4> output;
    Texture2DArray<float4> source;
    SamplerState sampler;

    float3 texelToCubeDir(uint2 pixel, uint face, uint2 dim)
    {
        float2 uv = (float2(pixel) + 0.5f) / float2(dim);
        float2 p = uv * 2.f - 1.f;
        p.y = -p.y;

        float3 dir = float3(0.f);
        switch (face)
        {
        case 0: dir = float3(1.f, p.y, -p.x); break;
        case 1: dir = float3(-1.f, p.y, p.x); break;
        case 2: dir = float3(p.x, 1.f, -p.y); break;
        case 3: dir = float3(p.x, -1.f, p.y); break;
        case 4: dir = float3(p.x, p.y, 1.f); break;
        default: dir = float3(-p.x, p.y, -1.f); break;
        }

        return normalize(dir);
    }

    void init(uint3 pixel)
    {
        if (any(pixel.xy >= outputDim) || pixel.z >= 6)
            return;

        float3 dir = texelToCubeDir(pixel.xy, pixel.z, outputDim);
        output[pixel] = float4(gScene.envMap.eval(dir), 1.f);
    }

    void downsample(uint3 pixel)
    {
        if (any(pixel.xy >= outputDim) || pixel.z >= 6)
            return;

        uint2 parentDim = outputDim * 2;
        uint2 basePixel = pixel.xy * 2;
        float3 color = float3(0.f);

        [unroll]
        for (uint y = 0; y < 2; ++y)
        {
            [unroll]
            for (uint x = 0; x < 2; ++x)
            {
                float2 parentUv = (float2(basePixel + uint2(x, y)) + 0.5f) / float2(parentDim);
                color += source.SampleLevel(sampler, float3(parentUv, pixel.z), 0.f).rgb;
            }
        }

        output[pixel] = float4(color * 0.25f, 1.f);
    }
};

ParameterBlock<PrefiltEnvCubemapPass> gPrefiltEnvCubemapPass;

[numthreads(16, 16, 1)]
void init(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPrefiltEnvCubemapPass.init(dispatchThreadId);
}

[numthreads(16, 16, 1)]
void downsample(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPrefiltEnvCubemapPass.downsample(dispatchThreadId);
}
#endif

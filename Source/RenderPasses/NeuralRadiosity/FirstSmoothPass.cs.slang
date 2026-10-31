import Scene.Shading;
import Scene.RaytracingInline;
import PathState;
import RayTraceHelper;

// Input channels

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;

// Output G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;

// Active buffers
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;


#define is_valid(name) (is_valid_##name != 0)

#ifndef USE_ALPHA_TEST
#define USE_ALPHA_TEST true
#endif

#ifndef ADJUST_SHADING_NORMALS
#define ADJUST_SHADING_NORMALS true
#endif

#ifndef USE_ENV_LIGHT
#define USE_ENV_LIGHT true
#endif

#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#ifndef SPECULAR_ROUGHNESS_THRESHOLD
#define SPECULAR_ROUGHNESS_THRESHOLD 0.5f
#endif

#ifndef DEBUG_RENDER_DIFFUSE_ONLY
#define DEBUG_RENDER_DIFFUSE_ONLY 0
#endif

#ifndef DEBUG_RENDER_SPECULAR_ONLY
#define DEBUG_RENDER_SPECULAR_ONLY 0
#endif

struct FirstSmoothPass
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kEnvMapDepth = 100000000.0f; // Arbitrary big number
    static const int kMaxRecursionDepth = 16;
    static const float kSpecularRoughnessThreshold = SPECULAR_ROUGHNESS_THRESHOLD;
    static const uint kRandomBase = 2654435761u;
    static const bool kDebugDiffuseOnly = (DEBUG_RENDER_DIFFUSE_ONLY != 0);
    static const bool kDebugSpecularOnly = (DEBUG_RENDER_SPECULAR_ONLY != 0);

    uint2 frameDim = {};
    uint frameCount = 0;

    void generatePath(const uint2 pixel, out PathState path)
    {
        path.setActive();
        path.id = pixel.x | (pixel.y << 12);  // | (sampleIdx << 24);
        path.thp = float3(1.f);

        // Create sample generator.
        path.sg = SampleGenerator(pixel, frameCount);

        // Unspecified:
        // - path.origin
        // - path.dir
        // - path.setHit(hit);
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim))
            return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        // Compute primary ray.
        Ray ray = gScene.camera.computeRayPinhole(pixel, frameDim);
        
        // Trace rays.
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        HitInfo hit;
        float3 color = float3(0.f);

        // G-Buffer values.
        float3 pos = float3(0.f);
        float3 dir = float3(0.f);
        float3 normal = float3(0.f);
        float3 albedo = float3(0.f);
        float roughness = 1.f;

        // Initialize path state.
        PathState path = {};
        generatePath(pixel, path);
        bool pathActive = true;  // Local tracking to work around Slang bitfield bug

        for (int i = 0; i < kMaxRecursionDepth; i++)
        {
            float rayT = 0;
            hit = sceneRayQuery.traceRay(ray, rayT, kRayFlags);

            // Miss: Write background color.
            if (!hit.isValid())
            {
                if (kUseEnvLight)
                {
                    color = gScene.envMap.eval(ray.dir) * path.thp;
                }
                path.terminate();
                pathActive = false;
                break;
            }

            // Load shading data. This is a long latency operation.
            ShadingData sd = loadShadingData(hit, ray.origin, ray.dir);

            // Create material instance.
            let lod = ExplicitLodTextureSampler(0.f);
            let hints = getMaterialInstanceHints(hit, i == 0, kAdjustShadingNormals);
            const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            const BSDFProperties bsdfProperties = mi.getProperties(sd);

            // Update path state.
            path.setHit(hit);
            path.incrementVertexIndex();
            path.sceneLength += float16_t(rayT);
            path.origin = sd.computeRayOrigin();
            path.dir = ray.dir;

            // Emissive surface: terminate path and write emitted radiance.
            if (any(bsdfProperties.emission > 0.f))
            {
                color = bsdfProperties.emission * path.thp;
                path.terminate();
                pathActive = false;
                break;
            }

            // Backside surface: terminate path.
            if (!sd.mtl.isDoubleSided() && !sd.frontFacing)
            {
                path.terminate();
                pathActive = false;
                break;
            }

            // Break the loop if the hit is not on delta only material
            if (!sd.mtl.isDeltaSpecular())
            {
                pos = path.origin;
                dir = path.dir;
                normal = sd.frame.N;
                roughness = bsdfProperties.roughness;
                albedo = all(bsdfProperties.diffuseReflectionAlbedo < 0.01f) ?
                    bsdfProperties.specularReflectionAlbedo : bsdfProperties.diffuseReflectionAlbedo;

                color = path.thp;
                break;
            }

            // Sample next ray segment.
            BSDFSample result = {};
            bool valid = mi.sample(sd, path.sg, result);
            if (valid) valid = generateScatterRay(result, sd, mi, path, kSpecularRoughnessThreshold);

            // Break the loop if the BSDF sample is invalid.
            if (!valid)
            {
                path.terminate();
                pathActive = false;
                break;
            }

            // Update ray.
            ray = path.getScatterRay();
        }

        // Active mask
        uint diffAct = (pathActive && (roughness > kSpecularRoughnessThreshold)) ? 1 : 0;
        uint specAct = (pathActive && (roughness <= kSpecularRoughnessThreshold)) ? 1 : 0;
        // Debug isolation: zero out the component that should not be rendered.
        if (kDebugDiffuseOnly) specAct = 0;
        if (kDebugSpecularOnly) diffAct = 0;

        // Write outputs
        gColor[pixel] = float4(color, 0.f);
        gVBuffer[pixel] = hit.pack();
        gActive[pixel] = pathActive ? 1u : 0u;

        // Write G-Buffer
        gPos[pixel] = float4(pos, 1.f);
        gDir[pixel] = float4(dir, 0.f);
        gNormal[pixel] = float4(normal, 0.f);
        gAlbedo[pixel] = float4(albedo, 0.f);
        gRoughness[pixel] = roughness;

        // Write active mask
        diffActive[pixIdx] = diffAct;
        diffIndex[pixIdx] = diffAct;
        specActive[pixIdx] = specAct;
        specIndex[pixIdx] = specAct;
    }
};

ParameterBlock<FirstSmoothPass> gFirstSmoothPass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gFirstSmoothPass.execute(dispatchThreadId.xy);
}
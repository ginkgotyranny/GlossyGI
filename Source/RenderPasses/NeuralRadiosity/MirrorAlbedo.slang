/** One mirror ray for the specular primary color.

    Both writers of the envColor slot (SpecPrimary and ClusterSpatialReuse) call
    this so the feature cannot drift. A hit replaces the blurry environment
    color with that surface's albedo. A miss keeps the environment color.

    Hits and misses share the same 3 input channels, but albedo is in [0, 1] while
    the prefiltered environment radiance can be 10-1000. encodeReflectedColor maps
    both into one bounded range before they reach the network (see
    kEnvColorEncoding*).
*/
import Scene.RaytracingInline;
import RayTraceHelper;

#ifndef USE_ALPHA_TEST
#define USE_ALPHA_TEST true
#endif

#ifndef ADJUST_SHADING_NORMALS
#define ADJUST_SHADING_NORMALS true
#endif

#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

static const bool kMirrorAlbedoUseAlphaTest = USE_ALPHA_TEST;
static const bool kMirrorAlbedoAdjustShadingNormals = ADJUST_SHADING_NORMALS;
static const uint kMirrorAlbedoRayFlags = RAY_FLAGS;

/// Raw values: hit albedo or environment radiance (legacy).
static const uint kEnvColorEncodingLinear = 0;
/// Hit albedo as is; miss -log(1 + env). The sign tells the first layer whether the
/// mirror ray hit, and the log keeps a 1000x brighter sky within ~7 units.
static const uint kEnvColorEncodingSignedLog = 1;
/// log(1 + env) on a miss, without the sign (ablation of the hit/miss flag).
static const uint kEnvColorEncodingLog = 2;

float3 encodeReflectedColor(float3 color, bool hit, uint encoding)
{
    if (hit || encoding == kEnvColorEncodingLinear) return color;
    const float3 l = log(1.f + max(color, 0.f));
    return encoding == kEnvColorEncodingSignedLog ? -l : l;
}

/// Encoded network input for the envColor slot: the mirror-hit albedo, or the
/// environment color on a miss (or when the mirror ray is disabled).
float3 sampleMirrorAlbedo(float3 origin, float3 reflectDir, float3 envColor, uint enable, uint encoding)
{
    if (enable == 0) return encodeReflectedColor(envColor, false, encoding);

    SceneRayQuery<kMirrorAlbedoUseAlphaTest> sceneRayQuery;
    Ray ray;
    ray.origin = origin;
    ray.dir = reflectDir;
    ray.tMin = 0.f;
    ray.tMax = 1e30f;

    float rayT = 0.f;
    HitInfo hit = sceneRayQuery.traceRay(ray, rayT, kMirrorAlbedoRayFlags);
    if (!hit.isValid()) return encodeReflectedColor(envColor, false, encoding);

    ShadingData sd = loadShadingData(hit, ray.origin, ray.dir);
    let lod = ExplicitLodTextureSampler(0.f);
    let hints = getMaterialInstanceHints(hit, false, kMirrorAlbedoAdjustShadingNormals);
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    const BSDFProperties bsdfProperties = mi.getProperties(sd);

    const float3 albedo = all(bsdfProperties.diffuseReflectionAlbedo < 0.01f) ?
        bsdfProperties.specularReflectionAlbedo : bsdfProperties.diffuseReflectionAlbedo;
    return encodeReflectedColor(albedo, true, encoding);
}

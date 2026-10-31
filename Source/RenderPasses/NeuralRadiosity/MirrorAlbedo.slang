/** One mirror ray for the specular primary color.

    Both writers of the envColor slot (SpecPrimary and ClusterSpatialReuse) call
    this so the feature cannot drift. A hit replaces the blurry environment
    color with that surface's albedo. A miss keeps the environment color.
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

float3 sampleMirrorAlbedo(float3 origin, float3 reflectDir, float3 envColor, uint enable)
{
    if (enable == 0) return envColor;

    SceneRayQuery<kMirrorAlbedoUseAlphaTest> sceneRayQuery;
    Ray ray;
    ray.origin = origin;
    ray.dir = reflectDir;
    ray.tMin = 0.f;
    ray.tMax = 1e30f;

    float rayT = 0.f;
    HitInfo hit = sceneRayQuery.traceRay(ray, rayT, kMirrorAlbedoRayFlags);
    if (!hit.isValid()) return envColor;

    ShadingData sd = loadShadingData(hit, ray.origin, ray.dir);
    let lod = ExplicitLodTextureSampler(0.f);
    let hints = getMaterialInstanceHints(hit, false, kMirrorAlbedoAdjustShadingNormals);
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    const BSDFProperties bsdfProperties = mi.getProperties(sd);

    return all(bsdfProperties.diffuseReflectionAlbedo < 0.01f) ?
        bsdfProperties.specularReflectionAlbedo : bsdfProperties.diffuseReflectionAlbedo;
}

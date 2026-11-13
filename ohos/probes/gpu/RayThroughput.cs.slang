// Probe: inline ray query throughput. Every thread traces raysPerPixel rays inside a closed,
// tessellated box, so every ray must hit: the hit count checks correctness, the time gives rays/s.
//   mode 0: incoherent, random origin and direction per ray
//   mode 1: cone, one origin for all pixels and rays within a narrow cone around the direction of
//           the pixel, as ConeTrace does for one specular sample
struct Params
{
    uint width;
    uint height;
    uint raysPerPixel;
    uint mode;
    uint seed;
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure gScene;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gOut;   // [pixel * 2]: hits, sum of hit distances in 1/1024

uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint state)
{
    return float(pcg(state) >> 8) * (1.0f / 16777216.0f);
}

float3 sphereDirection(float u, float v)
{
    const float z = 1.0f - 2.0f * u;
    const float r = sqrt(max(0.0f, 1.0f - z * z));
    const float phi = 6.2831853f * v;
    return float3(r * cos(phi), r * sin(phi), z);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gParams.width || tid.y >= gParams.height) return;
    const uint pixel = tid.y * gParams.width + tid.x;
    uint state = pixel * 9781u + gParams.seed * 6271u + 1u;

    const float2 uv = (float2(tid.xy) + 0.5f) / float2(gParams.width, gParams.height);
    const float3 cameraDir = normalize(float3(uv * 2.0f - 1.0f, 1.0f));

    uint hits = 0;
    float distance = 0.0f;
    for (uint r = 0; r < gParams.raysPerPixel; r++)
    {
        RayDesc ray;
        if (gParams.mode == 0)
        {
            ray.Origin = float3(rnd(state), rnd(state), rnd(state)) - 0.5f;
            ray.Direction = sphereDirection(rnd(state), rnd(state));
        }
        else
        {
            ray.Origin = float3(0.0f, 0.0f, -0.5f);
            ray.Direction = normalize(cameraDir + 0.1f * sphereDirection(rnd(state), rnd(state)));
        }
        ray.TMin = 0.0f;
        ray.TMax = 100.0f;

        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(gScene, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
        query.Proceed();
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            hits++;
            distance += query.CommittedRayT();
        }
    }
    gOut[pixel * 2 + 0] = hits;
    gOut[pixel * 2 + 1] = uint(distance * 1024.0f);
}

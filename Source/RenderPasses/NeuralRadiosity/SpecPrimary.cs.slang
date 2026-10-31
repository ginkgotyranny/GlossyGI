/** Environment color and mirror albedo for the coneTraceBatch path.

    ConeTrace runs one group of NUM_SPEC_RAYS threads per glossy sample, so a
    mirror ray on thread 0 stalls the other lanes. This pass is one thread per
    sample and writes only the pos / envColor slots. Cluster slots stay in
    ConeTrace. ClusterSpatialReuse already has this occupancy and keeps its own
    copy of the same write.
*/
import Scene.Shading;
import RayTraceHelper;
import NetworkParams;
import PrefiltEnvCubemap;
import ClusterHistory;
import MirrorAlbedo;

#ifndef SPEC_PRIMARY_GROUP_SIZE
#define SPEC_PRIMARY_GROUP_SIZE 64
#endif

RWStructuredBuffer<uint> specSrc;
RWStructuredBuffer<SpecNetInput> specInput;

// Train path: gather G-Buffer from the pre-compact batch.
RWStructuredBuffer<float4> allPos;
RWStructuredBuffer<float4> allDir;
RWStructuredBuffer<float> allRoughness;
RWStructuredBuffer<PackedHitInfo> allVBuffer;

// Render path: gather G-Buffer from the screen textures.
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float> gRoughness;


struct SpecPrimary
{
    uint specSize = 0;
    uint2 frameDim = {};
    /// Nonzero: specSrc is a screen pixIdx. Zero: specSrc is a pre-compact batch index.
    uint useScreenGBuffer = 0;
    /// 0 keeps the environment color and skips the mirror ray.
    uint enableMirrorAlbedo = 1;

    float3 bboxMin;
    float3 bboxMax;

    void loadSpecGBuffer(const uint compactIdx, out float3 outPos, out float3 outDir, out float outRoughness, out HitInfo hit)
    {
        const uint src = specSrc[compactIdx];
        if (useScreenGBuffer != 0)
        {
            const uint2 pixel = decodeSpecSrcPixel(src, frameDim);
            outPos = gPos[pixel].xyz;
            outDir = gDir[pixel].xyz;
            outRoughness = gRoughness[pixel];
            hit = unpackHitInfo(gVBuffer[pixel]);
        }
        else
        {
            outPos = allPos[src].xyz;
            outDir = allDir[src].xyz;
            outRoughness = allRoughness[src];
            hit = unpackHitInfo(allVBuffer[src]);
        }
    }

    void execute(const uint idx)
    {
        if (idx >= specSize) return;

        float3 pos;
        float3 incoming;
        float roughness;
        HitInfo hit;
        loadSpecGBuffer(idx, pos, incoming, roughness, hit);

        // Same shading frame ConeTrace uses for its cone axis.
        ShadingData sd = loadShadingData(hit, pos, incoming);
        sd.posW = pos;
        const float3 reflectDir = reflect(incoming, sd.frame.N);

        float3 envColor = gPrefiltEnv.valid != 0 ? gPrefiltEnv.eval(reflectDir, roughness) :
            (gPrefiltEnv.fallbackValid != 0 ? gScene.envMap.eval(reflectDir) : float3(0.f));
        envColor = sampleMirrorAlbedo(sd.posW, reflectDir, envColor, enableMirrorAlbedo);
        writeSpecPrimaryOutput(specInput, idx, pos, envColor, bboxMin, bboxMax);
    }
}


ParameterBlock<SpecPrimary> gSpecPrimary;


[numthreads(SPEC_PRIMARY_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gSpecPrimary.execute(dispatchThreadId.x);
}

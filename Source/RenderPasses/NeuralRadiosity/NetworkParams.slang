

#ifndef N_LEVELS
#define N_LEVELS 6
#endif

#ifndef N_FEATURES_PER_LEVEL
#define N_FEATURES_PER_LEVEL 2
#endif

#ifndef N_LEVELS_SPEC
#define N_LEVELS_SPEC 6
#endif

#ifndef N_FEATURES_PER_LEVEL_SPEC
#define N_FEATURES_PER_LEVEL_SPEC 4
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

#ifndef OUTPUT_DIM
#define OUTPUT_DIM 16
#endif


struct DiffNetInput
{
    static const uint padDim = 16;
    static const uint numClusters = NUM_CLUSTERS;
    static const uint encDim = N_FEATURES_PER_LEVEL * N_LEVELS;

    // static const uint posOffset = encDim;
    static const uint posOffset = 0;
    static const uint dirOffset = posOffset + 3;
    static const uint normalOffset = dirOffset + 3;
    static const uint albedoOffset = normalOffset + 3;
    static const uint roughnessOffset = albedoOffset + 3;
    static const uint totalDim = roughnessOffset + 1;
    // static const uint paddedDim = (totalDim + padDim - 1) / padDim * padDim;
    static const uint paddedDim = totalDim;

    float data[paddedDim];

    float3 getPos() { return float3(data[posOffset], data[posOffset + 1], data[posOffset + 2]); }
    float3 getDir() { return float3(data[dirOffset], data[dirOffset + 1], data[dirOffset + 2]); }
    float3 getNormal() { return float3(data[normalOffset], data[normalOffset + 1], data[normalOffset + 2]); }
    float3 getAlbedo() { return float3(data[albedoOffset], data[albedoOffset + 1], data[albedoOffset + 2]); }
    float getRoughness() { return data[roughnessOffset]; }

    [mutating]
    void clear()
    {
        [unroll]
        for (uint i = 0; i < paddedDim; i++) data[i] = 0.f;
    }

    [mutating]
    void setPos(float3 v)
    {
        data[posOffset] = v.x;
        data[posOffset + 1] = v.y;
        data[posOffset + 2] = v.z;
    }

    [mutating]
    void setDir(float3 v)
    {
        data[dirOffset] = v.x;
        data[dirOffset + 1] = v.y;
        data[dirOffset + 2] = v.z;
    }

    [mutating]
    void setNormal(float3 v)
    {
        data[normalOffset] = v.x;
        data[normalOffset + 1] = v.y;
        data[normalOffset + 2] = v.z;
    }

    [mutating]
    void setAlbedo(float3 v)
    {
        data[albedoOffset] = v.x;
        data[albedoOffset + 1] = v.y;
        data[albedoOffset + 2] = v.z;
    }

    [mutating]
    void setRoughness(float v) {
        data[roughnessOffset] = v;
    }
};


struct SpecNetInput
{
    static const uint padDim = 16;
    static const uint numClusters = NUM_CLUSTERS;
    static const uint primEncDim = N_FEATURES_PER_LEVEL * N_LEVELS;
    static const uint clsEncDim = N_FEATURES_PER_LEVEL_SPEC * NUM_CLUSTERS;

    // static const uint posOffset = primEncDim + clsEncDim;
    static const uint posOffset = 0;
    static const uint dirOffset = posOffset + 3;
    static const uint normalOffset = dirOffset + 3;
    static const uint albedoOffset = normalOffset + 3;
    static const uint roughnessOffset = albedoOffset + 3;
    static const uint clsPosOffset = roughnessOffset + 1;
    static const uint clsScaleOffset = clsPosOffset + numClusters * 3;
    static const uint clsWeightOffset = clsScaleOffset + numClusters;
    static const uint totalDim = clsWeightOffset + numClusters;
    // static const uint paddedDim = (totalDim + padDim - 1) / padDim * padDim;
    static const uint paddedDim = totalDim;

    float data[paddedDim];

    float3 getPos() { return float3(data[posOffset], data[posOffset + 1], data[posOffset + 2]); }
    float3 getDir() { return float3(data[dirOffset], data[dirOffset + 1], data[dirOffset + 2]); }
    float3 getNormal() { return float3(data[normalOffset], data[normalOffset + 1], data[normalOffset + 2]); }
    float3 getAlbedo() { return float3(data[albedoOffset], data[albedoOffset + 1], data[albedoOffset + 2]); }
    float getRoughness() { return data[roughnessOffset]; }

    float3 getClsPos(uint clusterIdx)
    {
        return float3(data[clsPosOffset + clusterIdx * 3], data[clsPosOffset + clusterIdx * 3 + 1], data[clsPosOffset + clusterIdx * 3 + 2]);
    }

    float getClsScale(uint clusterIdx)
    {
        return data[clsScaleOffset + clusterIdx];
    }

    float getClsWeight(uint clusterIdx)
    {
        return data[clsWeightOffset + clusterIdx];
    }

    [mutating]
    void clear()
    {
        [unroll]
        for (uint i = 0; i < paddedDim; i++) data[i] = 0.f;
    }

    [mutating]
    void setPos(float3 v)
    {
        data[posOffset] = v.x;
        data[posOffset + 1] = v.y;
        data[posOffset + 2] = v.z;
    }

    [mutating]
    void setDir(float3 v)
    {
        data[dirOffset] = v.x;
        data[dirOffset + 1] = v.y;
        data[dirOffset + 2] = v.z;
    }

    [mutating]
    void setNormal(float3 v)
    {
        data[normalOffset] = v.x;
        data[normalOffset + 1] = v.y;
        data[normalOffset + 2] = v.z;
    }

    [mutating]
    void setAlbedo(float3 v)
    {
        data[albedoOffset] = v.x;
        data[albedoOffset + 1] = v.y;
        data[albedoOffset + 2] = v.z;
    }

    [mutating]
    void setRoughness(float v) {
        data[roughnessOffset] = v;
    }

    [mutating]
    void setClsPos(uint clusterIdx, float3 v)
    {
        data[clsPosOffset + clusterIdx * 3] = v.x;
        data[clsPosOffset + clusterIdx * 3 + 1] = v.y;
        data[clsPosOffset + clusterIdx * 3 + 2] = v.z;
    }

    [mutating]
    void setClsScale(uint clusterIdx, float v)
    {
        data[clsScaleOffset + clusterIdx] = v;
    }

    [mutating]
    void setClsWeight(uint clusterIdx, float v)
    {
        data[clsWeightOffset + clusterIdx] = v;
    }
};


struct NetworkOutput
{
    static const uint outputDim = OUTPUT_DIM;

    float data[outputDim];

    float3 toFloat3()
    {
        return float3(data[0], data[1], data[2]);
    }

    float4 toFloat4()
    {
        return float4(data[0], data[1], data[2], data[3]);
    }

    [mutating]
    void clear()
    {
        [unroll]
        for (uint i = 0; i < outputDim; i++) data[i] = 0.f;
    }

    [mutating]
    void fromFloat3(float3 v)
    {
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
    }

    [mutating]
    void fromFloat4(float4 v)
    {
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
        data[3] = v.w;
    }
};


void writeDiffInputGBuffer(
    RWStructuredBuffer<DiffNetInput> buffer,
    uint idx,
    float3 pos,
    float3 dir,
    float3 normal,
    float3 albedo,
    float roughness)
{
    buffer[idx].data[DiffNetInput::posOffset + 0] = pos.x;
    buffer[idx].data[DiffNetInput::posOffset + 1] = pos.y;
    buffer[idx].data[DiffNetInput::posOffset + 2] = pos.z;

    buffer[idx].data[DiffNetInput::dirOffset + 0] = dir.x;
    buffer[idx].data[DiffNetInput::dirOffset + 1] = dir.y;
    buffer[idx].data[DiffNetInput::dirOffset + 2] = dir.z;

    buffer[idx].data[DiffNetInput::normalOffset + 0] = normal.x;
    buffer[idx].data[DiffNetInput::normalOffset + 1] = normal.y;
    buffer[idx].data[DiffNetInput::normalOffset + 2] = normal.z;

    buffer[idx].data[DiffNetInput::albedoOffset + 0] = albedo.x;
    buffer[idx].data[DiffNetInput::albedoOffset + 1] = albedo.y;
    buffer[idx].data[DiffNetInput::albedoOffset + 2] = albedo.z;

    buffer[idx].data[DiffNetInput::roughnessOffset] = roughness;

    [unroll]
    for (uint i = DiffNetInput::totalDim; i < DiffNetInput::paddedDim; i++) buffer[idx].data[i] = 0.f;
}


void writeSpecInputGBuffer(
    RWStructuredBuffer<SpecNetInput> buffer,
    uint idx,
    float3 pos,
    float3 dir,
    float3 normal,
    float3 albedo,
    float roughness)
{
    buffer[idx].data[SpecNetInput::posOffset + 0] = pos.x;
    buffer[idx].data[SpecNetInput::posOffset + 1] = pos.y;
    buffer[idx].data[SpecNetInput::posOffset + 2] = pos.z;

    buffer[idx].data[SpecNetInput::dirOffset + 0] = dir.x;
    buffer[idx].data[SpecNetInput::dirOffset + 1] = dir.y;
    buffer[idx].data[SpecNetInput::dirOffset + 2] = dir.z;

    buffer[idx].data[SpecNetInput::normalOffset + 0] = normal.x;
    buffer[idx].data[SpecNetInput::normalOffset + 1] = normal.y;
    buffer[idx].data[SpecNetInput::normalOffset + 2] = normal.z;

    buffer[idx].data[SpecNetInput::albedoOffset + 0] = albedo.x;
    buffer[idx].data[SpecNetInput::albedoOffset + 1] = albedo.y;
    buffer[idx].data[SpecNetInput::albedoOffset + 2] = albedo.z;

    buffer[idx].data[SpecNetInput::roughnessOffset] = roughness;

    [unroll]
    for (uint i = SpecNetInput::totalDim; i < SpecNetInput::paddedDim; i++) buffer[idx].data[i] = 0.f;
}

void writeSpecInputClsBuffer(
    RWStructuredBuffer<SpecNetInput> buffer,
    uint idx,
    uint clusterIdx,
    float3 clsPos,
    float clsScale,
    float clsWeight)
{
    buffer[idx].data[SpecNetInput::clsPosOffset + clusterIdx * 3 + 0] = clsPos.x;
    buffer[idx].data[SpecNetInput::clsPosOffset + clusterIdx * 3 + 1] = clsPos.y;
    buffer[idx].data[SpecNetInput::clsPosOffset + clusterIdx * 3 + 2] = clsPos.z;

    buffer[idx].data[SpecNetInput::clsScaleOffset + clusterIdx] = clsScale;
    buffer[idx].data[SpecNetInput::clsWeightOffset + clusterIdx] = clsWeight;
}

float nan_to_num(float x, float fallback = 0.f)
{
    return (isnan(x) || !isfinite(x)) ? fallback : x;
}

float2 nan_to_num(float2 v, float fallback = 0.f)
{
    return select(isnan(v) | !isfinite(v), float2(fallback), v);
}

float3 nan_to_num(float3 v, float fallback = 0.f)
{
    return select(isnan(v) | !isfinite(v), float3(fallback), v);
}

float4 nan_to_num(float4 v, float fallback = 0.f)
{
    return select(isnan(v) | !isfinite(v), float4(fallback), v);
}
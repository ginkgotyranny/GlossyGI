#ifndef NR_USE_FP16
#define NR_USE_FP16 1
#endif

#if NR_USE_FP16
typealias Real  = float16_t;
typealias Real2 = float16_t2;
typealias Real3 = float16_t3;
typealias Real4 = float16_t4;
#else
typealias Real  = float;
typealias Real2 = float2;
typealias Real3 = float3;
typealias Real4 = float4;
#endif

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

    static const uint posOffset = encDim;
    // static const uint posOffset = 0;
    static const uint dirOffset = posOffset + 3;
    static const uint normalOffset = dirOffset + 3;
    static const uint albedoOffset = normalOffset + 3;
    static const uint roughnessOffset = albedoOffset + 3;
    static const uint totalDim = roughnessOffset + 1;
    static const uint paddedDim = (totalDim + padDim - 1) / padDim * padDim;
    // static const uint paddedDim = totalDim;

    Real data[paddedDim];

    Real3 getPos() { return Real3(data[posOffset], data[posOffset + 1], data[posOffset + 2]); }
    Real3 getDir() { return Real3(data[dirOffset], data[dirOffset + 1], data[dirOffset + 2]); }
    Real3 getNormal() { return Real3(data[normalOffset], data[normalOffset + 1], data[normalOffset + 2]); }
    Real3 getAlbedo() { return Real3(data[albedoOffset], data[albedoOffset + 1], data[albedoOffset + 2]); }
    Real getRoughness() { return data[roughnessOffset]; }

    [mutating]
    void clear()
    {
        [unroll]
        for (uint i = 0; i < paddedDim; i++) data[i] = (Real) 0.f;
    }

    [mutating]
    void setPos(Real3 v)
    {
        data[posOffset] = v.x;
        data[posOffset + 1] = v.y;
        data[posOffset + 2] = v.z;
    }

    [mutating]
    void setDir(Real3 v)
    {
        data[dirOffset] = v.x;
        data[dirOffset + 1] = v.y;
        data[dirOffset + 2] = v.z;
    }

    [mutating]
    void setNormal(Real3 v)
    {
        data[normalOffset] = v.x;
        data[normalOffset + 1] = v.y;
        data[normalOffset + 2] = v.z;
    }

    [mutating]
    void setAlbedo(Real3 v)
    {
        data[albedoOffset] = v.x;
        data[albedoOffset + 1] = v.y;
        data[albedoOffset + 2] = v.z;
    }

    [mutating]
    void setRoughness(Real v) {
        data[roughnessOffset] = v;
    }
};


struct SpecNetInput
{
    static const uint padDim = 16;
    static const uint numClusters = NUM_CLUSTERS;
    static const uint primEncDim = N_FEATURES_PER_LEVEL * N_LEVELS;
    static const uint clsEncDim = N_FEATURES_PER_LEVEL_SPEC * NUM_CLUSTERS;

    static const uint posOffset = primEncDim + clsEncDim;
    // static const uint posOffset = 0;
    static const uint dirOffset = posOffset + 3;
    static const uint normalOffset = dirOffset + 3;
    static const uint albedoOffset = normalOffset + 3;
    static const uint roughnessOffset = albedoOffset + 3;
    static const uint envColorOffset = roughnessOffset + 1;
    // static const uint clsPosOffset = roughnessOffset + 1;
    static const uint clsPosOffset = envColorOffset + 3;
    static const uint clsScaleOffset = clsPosOffset + numClusters * 3;
    static const uint clsWeightOffset = clsScaleOffset + numClusters;
    static const uint totalDim = clsWeightOffset + numClusters;
    static const uint paddedDim = (totalDim + padDim - 1) / padDim * padDim;
    // static const uint paddedDim = totalDim;

    Real data[paddedDim];

    Real3 getPos() { return Real3(data[posOffset], data[posOffset + 1], data[posOffset + 2]); }
    Real3 getDir() { return Real3(data[dirOffset], data[dirOffset + 1], data[dirOffset + 2]); }
    Real3 getNormal() { return Real3(data[normalOffset], data[normalOffset + 1], data[normalOffset + 2]); }
    Real3 getAlbedo() { return Real3(data[albedoOffset], data[albedoOffset + 1], data[albedoOffset + 2]); }
    Real getRoughness() { return data[roughnessOffset]; }

    Real3 getClsPos(uint clusterIdx)
    {
        return Real3(data[clsPosOffset + clusterIdx * 3], data[clsPosOffset + clusterIdx * 3 + 1], data[clsPosOffset + clusterIdx * 3 + 2]);
    }

    Real getClsScale(uint clusterIdx)
    {
        return data[clsScaleOffset + clusterIdx];
    }

    Real getClsWeight(uint clusterIdx)
    {
        return data[clsWeightOffset + clusterIdx];
    }

    [mutating]
    void clear()
    {
        [unroll]
        for (uint i = 0; i < paddedDim; i++) data[i] = (Real) 0.f;
    }

    [mutating]
    void setPos(Real3 v)
    {
        data[posOffset] = v.x;
        data[posOffset + 1] = v.y;
        data[posOffset + 2] = v.z;
    }

    [mutating]
    void setDir(Real3 v)
    {
        data[dirOffset] = v.x;
        data[dirOffset + 1] = v.y;
        data[dirOffset + 2] = v.z;
    }

    [mutating]
    void setNormal(Real3 v)
    {
        data[normalOffset] = v.x;
        data[normalOffset + 1] = v.y;
        data[normalOffset + 2] = v.z;
    }

    [mutating]
    void setAlbedo(Real3 v)
    {
        data[albedoOffset] = v.x;
        data[albedoOffset + 1] = v.y;
        data[albedoOffset + 2] = v.z;
    }

    [mutating]
    void setRoughness(Real v) {
        data[roughnessOffset] = v;
    }

    [mutating]
    void setClsPos(uint clusterIdx, Real3 v)
    {
        data[clsPosOffset + clusterIdx * 3] = v.x;
        data[clsPosOffset + clusterIdx * 3 + 1] = v.y;
        data[clsPosOffset + clusterIdx * 3 + 2] = v.z;
    }

    [mutating]
    void setClsScale(uint clusterIdx, Real v)
    {
        data[clsScaleOffset + clusterIdx] = v;
    }

    [mutating]
    void setClsWeight(uint clusterIdx, Real v)
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
    Real3 pos,
    Real3 dir,
    Real3 normal,
    Real3 albedo,
    Real roughness)
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
    for (uint i = DiffNetInput::totalDim; i < DiffNetInput::paddedDim; i++) buffer[idx].data[i] = (Real) 0.f;
}


void writeSpecInputGBuffer(
    RWStructuredBuffer<SpecNetInput> buffer,
    uint idx,
    Real3 pos,
    Real3 dir,
    Real3 normal,
    Real3 albedo,
    Real roughness)
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
    for (uint i = SpecNetInput::totalDim; i < SpecNetInput::paddedDim; i++) buffer[idx].data[i] = (Real) 0.f;
}

void writeSpecInputClsBuffer(
    RWStructuredBuffer<SpecNetInput> buffer,
    uint idx,
    uint clusterIdx,
    Real3 clsPos,
    Real clsScale,
    Real clsWeight)
{
    buffer[idx].data[SpecNetInput::clsPosOffset + clusterIdx * 3 + 0] = clsPos.x;
    buffer[idx].data[SpecNetInput::clsPosOffset + clusterIdx * 3 + 1] = clsPos.y;
    buffer[idx].data[SpecNetInput::clsPosOffset + clusterIdx * 3 + 2] = clsPos.z;

    buffer[idx].data[SpecNetInput::clsScaleOffset + clusterIdx] = clsScale;
    buffer[idx].data[SpecNetInput::clsWeightOffset + clusterIdx] = clsWeight;
}

Real nan_to_num(Real x, Real fallback = (Real) 0.f)
{
    return (isnan(x) || !isfinite(x)) ? fallback : x;
}

Real2 nan_to_num(Real2 v, Real fallback = (Real) 0.f)
{
    return select(isnan(v) | !isfinite(v), Real2(fallback), v);
}

Real3 nan_to_num(Real3 v, Real fallback = (Real) 0.f)
{
    return select(isnan(v) | !isfinite(v), Real3(fallback), v);
}

Real4 nan_to_num(Real4 v, Real fallback = (Real) 0.f)
{
    return select(isnan(v) | !isfinite(v), Real4(fallback), v);
}
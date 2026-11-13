// One optimizer step over a hash grid: tiny-cuda-nn's EMA(Adam) as Model.cu configures it, which
// runs in CUDA on the desktop. Follows adam_ema_step() in ohos/npu/ncr_reference.py, the form that
// was checked against the CUDA implementation.
//
// Per entry: an entry whose gradient is exactly zero is skipped by Adam altogether (most of a grid
// in any one step) and has its own step count for the bias correction. The EMA runs over every
// entry, on the fp16 training copy, and its result is the fp16 copy inference reads.
//
// The fp16 copies are written through an explicit round-to-nearest-even conversion. The EMA
// re-rounds its own result every step, and a driver that truncates instead would pull every
// weight toward zero by up to 1 / (1 - decay) fp16 steps.
static const float kBeta1 = 0.9f;
static const float kBeta2 = 0.999f;
static const float kEpsilon = 1e-8f;

struct Params
{
    uint count;          // grid elements (entries * features)
    float learningRate;
    float gradToFloat;   // 1 / (fixed-point units per 1.0 * loss scale)
    float emaDecay;
    float emaDebiasOld;  // 1 - decay^(step - 1)
    float emaDebiasNew;  // 1 / (1 - decay^step)
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] RWStructuredBuffer<int> gGradient;         // fixed point; cleared here
[[vk::binding(1, 0)]] RWStructuredBuffer<float> gMaster;         // float32 weights
[[vk::binding(2, 0)]] RWStructuredBuffer<float> gFirstMoment;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> gSecondMoment;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> gSteps;           // Adam steps taken per element
[[vk::binding(5, 0)]] RWStructuredBuffer<uint16_t> gTraining;    // fp16 bits: what the training lookups read
[[vk::binding(6, 0)]] RWStructuredBuffer<uint16_t> gInference;   // fp16 bits: the EMA, what inference reads

/// float -> fp16 bits, round to nearest even (what a float -> __half cast does in CUDA).
uint halfBits(float f)
{
    const uint x = asuint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint e8 = (x >> 23) & 0xffu;
    uint mant = x & 0x7fffffu;
    if (e8 == 0xffu) return sign | 0x7c00u | (mant != 0u ? 0x200u : 0u);
    const int e = int(e8) - 127 + 15;
    if (e >= 31) return sign | 0x7c00u;
    if (e <= 0)
    {
        if (e < -10) return sign;
        mant |= 0x800000u;
        const uint shift = uint(14 - e);
        uint h = mant >> shift;
        const uint rem = mant & ((1u << shift) - 1u);
        const uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) h++;
        return sign | h;
    }
    uint h = sign | (uint(e) << 10) | (mant >> 13);   // a mantissa carry runs into the exponent: still correct
    const uint rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h++;
    return h;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.count) return;

    uint training = uint(gTraining[i]);
    const int fixedGradient = gGradient[i];
    if (fixedGradient != 0)
    {
        gGradient[i] = 0;
        const float g = float(fixedGradient) * gParams.gradToFloat;
        const float m = kBeta1 * gFirstMoment[i] + (1.0f - kBeta1) * g;
        const float v = kBeta2 * gSecondMoment[i] + (1.0f - kBeta2) * g * g;
        const uint steps = gSteps[i] + 1;
        gFirstMoment[i] = m;
        gSecondMoment[i] = v;
        gSteps[i] = steps;
        const float t = float(steps);
        const float rate = gParams.learningRate * sqrt(1.0f - pow(kBeta2, t)) / (1.0f - pow(kBeta1, t));
        const float weight = gMaster[i] - rate / (sqrt(v) + kEpsilon) * m;
        gMaster[i] = weight;
        training = halfBits(weight);
        gTraining[i] = uint16_t(training);
    }
    const float ema = (f16tof32(uint(gInference[i])) * gParams.emaDecay * gParams.emaDebiasOld + f16tof32(training) * (1.0f - gParams.emaDecay)) *
                      gParams.emaDebiasNew;
    gInference[i] = uint16_t(halfBits(ema));
}

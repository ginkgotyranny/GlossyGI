// Probe: one dense optimizer pass over a grid, as tiny-cuda-nn's Adam + EMA do on the desktop:
// fixed-point gradient -> Adam on the f32 master -> fp16 training copy and fp16 EMA copy; the
// gradient is cleared for the next step. A timing probe; the exact update rule is settled later.
struct Params
{
    uint count;
    float learningRate;     // already bias-corrected: lr * sqrt(1 - b2^t) / (1 - b1^t)
    float gradToFloat;      // 1 / (fixed-point scale * loss scale)
    float emaDecay;
    float emaDebiasOld;
    float emaDebiasNew;
};
[[vk::push_constant]] ConstantBuffer<Params> gParams;

[[vk::binding(0, 0)]] RWStructuredBuffer<int> gGrad;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> gMaster;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> gFirstMoment;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> gSecondMoment;
[[vk::binding(4, 0)]] RWStructuredBuffer<float16_t> gWeights;
[[vk::binding(5, 0)]] RWStructuredBuffer<float16_t> gEma;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.count) return;

    const float weight = gMaster[i];
    const float gradient = float(gGrad[i]) * gParams.gradToFloat + 1e-8f * weight;   // l2_reg
    gGrad[i] = 0;
    const float m = 0.9f * gFirstMoment[i] + 0.1f * gradient;
    const float v = 0.999f * gSecondMoment[i] + 0.001f * gradient * gradient;
    gFirstMoment[i] = m;
    gSecondMoment[i] = v;
    const float updated = weight - gParams.learningRate * m / (sqrt(v) + 1e-8f);
    gMaster[i] = updated;
    gWeights[i] = float16_t(updated);
    const float ema = (float(gEma[i]) * gParams.emaDecay * gParams.emaDebiasOld + updated * (1.0f - gParams.emaDecay)) * gParams.emaDebiasNew;
    gEma[i] = float16_t(ema);
}

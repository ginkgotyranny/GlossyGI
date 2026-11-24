#pragma once

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/loss.h>

namespace ncr {

// Samples are contiguous columns in TCNN's column-major matrices. The MLP
// requires aligned batches, but padding is not training data. Give its forward
// pass finite inputs so even zero-gradient padded lanes cannot produce 0 * NaN.
template<typename T>
void clearTrainingInputPadding(cudaStream_t stream, tcnn::GPUMatrix<T>& input, uint32_t validSamples)
{
    CHECK_THROW(validSamples <= input.n());
    if (validSamples == input.n()) return;
    const size_t offset = size_t(validSamples) * input.m();
    const size_t count = size_t(input.n() - validSamples) * input.m();
    CUDA_CHECK_THROW(cudaMemsetAsync(input.data() + offset, 0, count * sizeof(T), stream));
}

// Reuse the original loss on views of only the valid samples. This both avoids
// reading stale targets and preserves the loss's normalization by real samples.
// Backward still sees an aligned matrix, with exactly zero gradient in its tail.
template<typename T>
void evaluateValidSamples(
    cudaStream_t stream, float lossScale, tcnn::Loss<T>& loss,
    tcnn::GPUMatrix<T>& prediction, const tcnn::GPUMatrix<float>& target,
    tcnn::GPUMatrix<float>& values, tcnn::GPUMatrix<T>& gradients)
{
    const uint32_t validSamples = target.n();
    CHECK_THROW(validSamples <= prediction.n());
    CHECK_THROW(values.m() == prediction.m() && values.n() == prediction.n());
    CHECK_THROW(gradients.m() == prediction.m() && gradients.n() == prediction.n());
    const size_t tailOffset = size_t(validSamples) * prediction.m();
    const size_t tailCount = size_t(prediction.n() - validSamples) * prediction.m();
    if (tailCount)
    {
        CUDA_CHECK_THROW(cudaMemsetAsync(values.data() + tailOffset, 0, tailCount * sizeof(float), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(gradients.data() + tailOffset, 0, tailCount * sizeof(T), stream));
    }
    if (!validSamples) return;
    tcnn::GPUMatrix<T> validPrediction(prediction.data(), prediction.m(), validSamples);
    tcnn::GPUMatrix<float> validValues(values.data(), values.m(), validSamples);
    tcnn::GPUMatrix<T> validGradients(gradients.data(), gradients.m(), validSamples);
    loss.evaluate(stream, lossScale, validPrediction, target, validValues, validGradients);
}

} // namespace ncr

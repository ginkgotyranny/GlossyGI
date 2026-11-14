#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "ModelTypes.h"
#include "../NCRConfig.h"
#include "HashGrid.h"
#include "HashGridInterp.h"
#include "tiny-cuda-nn/object.h"


namespace tcnn
{
    template <typename T>
    class GPUMemory;

    template <typename T>
    class Encoding;

    template <typename T, typename PARAMS_T>
    class Network;

    template <typename T>
    class NetworkWithInputEncoding;

    template <typename T, typename PARAMS_T, typename COMPUTE_T>
    class Trainer;

    template <typename T>
    class Optimizer;

    template <typename T>
    class Loss;

    // Custom models
    template<typename T>
    class NeuralModel;

    template<typename T>
    class NeuralConeModel;

    struct Context;

    struct pcg32;
}

struct ModelIOPtrs
{
    precision_t* inputPtr;
    float* outputPtr;

    uint32_t size;
};

inline constexpr uint32_t padUp(uint32_t x, uint32_t align) { return (x + align - 1) & ~(align - 1); }
/// NRModel pads every batch to this many rows. The hash-grid encoders and the
/// MLP read and write the padded rows, so I/O buffers must have room for them.
inline constexpr uint32_t kModelBatchAlign = 256;


class NRModel
{
public:
    NRModel();
    ~NRModel();

    void loadState(const std::string& path);
    void saveState(const std::string& path);
    /// The fp16 copies the training step computes with ("NRWK" v1: header with the four element
    /// counts, then the primary grid, cluster grid, DiffNet and SpecNet blocks). Not part of a
    /// checkpoint: loading one rebuilds them from the masters, while a fresh model starts with
    /// whatever the optimizers have not yet written. The OpenHarmony bake records them.
    void saveWorkingCopies(const std::string& path);

    void setOnline(bool online);
    /// Learning rate and EMA decay for all four online optimizers (grids and MLPs).
    /// Takes effect on the next train() step. Adam keeps its moments and the EMA
    /// keeps its weights, so changing them mid-run is continuous.
    void setOptimizerHyperparams(float learningRate, float emaDecay);
    float learningRate() const { return mLearningRate; }
    float emaDecay() const { return mEmaDecay; }

    void inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);
    void train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);

    cudaStream_t stream(bool isTraining) const { return isTraining ? mTrainStream : mInferenceStream; }

private:
    void createOnlineModels();
    void createOfflineModelsIfNeeded();
    
    cudaStream_t mInferenceStream;
    cudaStream_t mTrainStream;
    bool mOnline = false;
    float mLearningRate = NCR_ADAM_LR;
    float mEmaDecay = NCR_EMA_DECAY;

    std::shared_ptr<tcnn::pcg32> mpRng;

    uint32_t pixelCount = 1u << 20;

    HashGrid::Config mPrimGridConfig;
    HashGridInterp::Config mClsGridConfig;

    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpPrimGrids;
    std::shared_ptr<tcnn::GPUMemory<float>>       mpPrimGridsFull;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpPrimGridsGradient;
    precision_t*                                  mpPrimGridsInference;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpClsGrids;
    std::shared_ptr<tcnn::GPUMemory<float>>       mpClsGridsFull;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpClsGridsGradient;
    precision_t*                                  mpClsGridsInference;
    
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> mpDiffNet;
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> mpSpecNet;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpDiffParams;
    std::shared_ptr<tcnn::GPUMemory<float>>       mpDiffParamsFull;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpDiffParamsGradient;
    precision_t*                                  mpDiffParamsInference;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpSpecParams;
    std::shared_ptr<tcnn::GPUMemory<float>>       mpSpecParamsFull;
    std::shared_ptr<tcnn::GPUMemory<precision_t>> mpSpecParamsGradient;
    precision_t*                                  mpSpecParamsInference;
    
    std::shared_ptr<tcnn::Optimizer<precision_t>> mpPrimGridOpt;
    std::shared_ptr<tcnn::Optimizer<precision_t>> mpClsGridOpt;
    std::shared_ptr<tcnn::Optimizer<precision_t>> mpDiffNetOpt;
    std::shared_ptr<tcnn::Optimizer<precision_t>> mpSpecNetOpt;
    std::shared_ptr<tcnn::Loss<precision_t>> mpLoss;

    uint32_t mDiffGBufferDim;
    uint32_t mSpecGBufferDim;
    uint32_t mDiffInputDim;
    uint32_t mSpecInputDim;
    /// Channels stored in diffColor / specColor and supervised by the loss.
    uint32_t mOutputDim;
    /// FullyFusedMLP forward/backward width. 16 when mOutputDim <= 16.
    uint32_t mPaddedOutputDim;

    uint32_t mDiffEncOffset;
    uint32_t mDiffPosOffset;
    uint32_t mSpecPrimEncOffset;
    uint32_t mSpecClsEncOffset;
    uint32_t mSpecPosOffset;
    uint32_t mSpecClsPosOffset;
    uint32_t mSpecClsScaleOffset;
    uint32_t mSpecClsWeightOffset;
};

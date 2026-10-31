#include "Model.h"
#include "HashGrid.h"
#include "HashGridInterp.h"
// #include "NeuralModel.h"
// #include "NeuralConeModel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "tiny-cuda-nn/config.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/trainer.h"
#include <nlohmann/json.hpp>


// Model code

namespace {

template<typename SrcT, typename DstT>
void castMatrix(cudaStream_t stream, const tcnn::GPUMatrixDynamic<SrcT>& src, tcnn::GPUMatrixDynamic<DstT>& dst) {
    CHECK_THROW(src.m() == dst.m());
    CHECK_THROW(src.n() == dst.n());

    const uint32_t count = src.n_elements();
    tcnn::parallel_for_gpu(stream, count, [srcData = src.data(), dstData = dst.data()] __device__ (size_t i) {
        dstData[i] = (DstT)srcData[i];
    });
}

template<typename SrcT, typename DstT>
void moveGBuffer(cudaStream_t stream, const tcnn::GPUMatrixDynamic<SrcT>& gbuf, tcnn::GPUMatrixDynamic<DstT>& netInput, uint32_t gBufOffset) {
    CHECK_THROW(gbuf.n() == netInput.n());
    CHECK_THROW(gBufOffset + gbuf.m() <= netInput.m());

    const uint32_t gbufRows = gbuf.m();
    const uint32_t gbufStride = gbuf.stride();
    const tcnn::MatrixLayout gbufLayout = gbuf.layout();
    const uint32_t netInputStride = netInput.stride();
    const tcnn::MatrixLayout netInputLayout = netInput.layout();

    const uint32_t count = gbuf.n_elements();
    tcnn::parallel_for_gpu(stream, count, [gbufData = gbuf.data(), netInputData = netInput.data(), gBufOffset, gbufRows, gbufStride, gbufLayout, netInputStride, netInputLayout] __device__ (size_t i) {
        const uint32_t row = i % gbufRows;
        const uint32_t col = i / gbufRows;

        const uint32_t gbufIndex = gbufLayout == tcnn::CM ? row + col * gbufStride : col + row * gbufStride;
        const uint32_t netInputRow = gBufOffset + row;
        const uint32_t netInputIndex = netInputLayout == tcnn::CM ? netInputRow + col * netInputStride : col + netInputRow * netInputStride;

        netInputData[netInputIndex] = (DstT)gbufData[gbufIndex];
    });
}

template<typename T>
__global__ void clampGradients(
    T* gradients,
    uint32_t count,
    float threshold
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float g = (float)gradients[idx];
    if (!isfinite(g)) {
        gradients[idx] = (T)0.0f;
        return;
    }

    gradients[idx] = (T)fminf(fmaxf(g, -threshold), threshold);
}

static uint64_t gpuUsedMBs() {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK_THROW(cudaMemGetInfo(&freeBytes, &totalBytes));
    return (totalBytes - freeBytes) / (1024 * 1024);
}

} // namespace

NRModel::NRModel() {
    CUDA_CHECK_THROW(cudaStreamCreate(&mTrainStream));
    CUDA_CHECK_THROW(cudaStreamCreate(&mInferenceStream));

    mDiffGBufferDim = 3 * 4 + 1;
    mSpecGBufferDim = 3 * 5 + 1 + numClusters * (3 + 1 + 1);
    mDiffInputDim = padUp(nFeaturesPerLevel * nLevels + mDiffGBufferDim, 16);
    mSpecInputDim = padUp(nFeaturesPerLevel * nLevels + nInterpFeaturesPerLevel * numClusters + mSpecGBufferDim, 16);
    mOutputDim = 16;

    int seed = 1337;
    std::seed_seq seq{seed};
    std::vector<uint32_t> seeds(2);
    seq.generate(std::begin(seeds), std::end(seeds));
    mpRng = std::make_shared<tcnn::pcg32>(seeds.front());

    mPrimGridConfig = {
        nLevels,
        nFeaturesPerLevel,
        log2HashMapSize,
        baseResolution,
        perLevelScale
    };
    mClsGridConfig = {
        numClusters,
        nInterpLevels,
        nInterpFeaturesPerLevel,
        log2InterpHashMapSize,
        baseInterpResolution,
        perLevelInterpScale,
        interpRatio
    };

    mDiffEncOffset = 0;
    mDiffPosOffset = mDiffEncOffset + mPrimGridConfig.nLevels * mPrimGridConfig.nFeaturesPerLevel;
    mSpecPrimEncOffset = 0;
    mSpecClsEncOffset = mSpecPrimEncOffset + mPrimGridConfig.nLevels * mPrimGridConfig.nFeaturesPerLevel;
    mSpecPosOffset = mSpecClsEncOffset + mClsGridConfig.nClusters * mClsGridConfig.nFeaturesPerLevel;
    mSpecClsPosOffset = mSpecPosOffset + 3 * 5 + 1;
    mSpecClsScaleOffset = mSpecClsPosOffset + mClsGridConfig.nClusters * 3;
    mSpecClsWeightOffset = mSpecClsScaleOffset + mClsGridConfig.nClusters * 1;

    createOnlineModels();

    mOnline = true;
}

void NRModel::createOnlineModels()
{
    {
        HashGrid::initializeConstants(mPrimGridConfig);

        float resolution = mPrimGridConfig.baseResolution;
        uint32_t gridSize = 0;
        for (int i = 0; i < mPrimGridConfig.nLevels; i++)
        {
            uint64_t elems64 = (uint64_t) resolution + 1;
            elems64 = elems64 * elems64 * elems64;
            uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << mPrimGridConfig.log2HashMapSize));

            resolution *= mPrimGridConfig.perLevelScale;
            gridSize += elems;
        }
        gridSize = gridSize * mPrimGridConfig.nFeaturesPerLevel;

        mpPrimGrids = std::make_shared<tcnn::GPUMemory<precision_t>>(gridSize);
        mpPrimGridsGradient = std::make_shared<tcnn::GPUMemory<precision_t>>(gridSize);
        mpPrimGridsFull = std::make_shared<tcnn::GPUMemory<float>>(gridSize);
    }

    {
        HashGridInterp::initializeConstants(mClsGridConfig);

        float resolution = mClsGridConfig.baseResolution;
        uint32_t gridSize = 0;
        for (int i = 0; i < mClsGridConfig.nLevels; i++)
        {
            uint64_t elems64 = (uint64_t) resolution + 1;
            elems64 = elems64 * elems64 * elems64;
            uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << mClsGridConfig.log2HashMapSize));

            resolution *= mClsGridConfig.perLevelScale;
            gridSize += elems;
        }
        gridSize = gridSize * mClsGridConfig.nFeaturesPerLevel;

        mpClsGrids = std::make_shared<tcnn::GPUMemory<precision_t>>(gridSize);
        mpClsGridsGradient = std::make_shared<tcnn::GPUMemory<precision_t>>(gridSize);
        mpClsGridsFull = std::make_shared<tcnn::GPUMemory<float>>(gridSize);
    }

    const tcnn::json diffNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mDiffInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", 64},
        {"n_hidden_layers", 3}
    };

    const tcnn::json specNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mSpecInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", 64},
        {"n_hidden_layers", 4}
    };

    mpDiffNet = std::shared_ptr<tcnn::Network<precision_t, precision_t>>(tcnn::create_network<precision_t>(diffNetConfig));
    mpSpecNet = std::shared_ptr<tcnn::Network<precision_t, precision_t>>(tcnn::create_network<precision_t>(specNetConfig));
    mpDiffParams         = std::make_shared<tcnn::GPUMemory<precision_t>>(mpDiffNet->n_params());
    mpDiffParamsGradient = std::make_shared<tcnn::GPUMemory<precision_t>>(mpDiffNet->n_params());
    mpDiffParamsFull     = std::make_shared<tcnn::GPUMemory<float>>(mpDiffNet->n_params());
    mpSpecParams         = std::make_shared<tcnn::GPUMemory<precision_t>>(mpSpecNet->n_params());
    mpSpecParamsGradient = std::make_shared<tcnn::GPUMemory<precision_t>>(mpSpecNet->n_params());
    mpSpecParamsFull     = std::make_shared<tcnn::GPUMemory<float>>(mpSpecNet->n_params());
    mpDiffNet->initialize_params(*mpRng, mpDiffParamsFull->data());
    mpSpecNet->initialize_params(*mpRng, mpSpecParamsFull->data());

    const tcnn::json offlineOptConfig = {
        {"otype", "Adam"},
        {"learning_rate", 5e-3f}
    };
    const tcnn::json onlineOptConfig = {
        {"otype", "EMA"},
        {"decay", 0.95f},
        {"nested", offlineOptConfig}
    };
    const tcnn::json lossConfig = {
        {"otype", "RelativeL2Luminance"}
    };

    mpPrimGridOpt = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(onlineOptConfig));
    mpClsGridOpt = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(onlineOptConfig));
    mpDiffNetOpt = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(onlineOptConfig));
    mpSpecNetOpt = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(onlineOptConfig));
    mpLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(lossConfig));

    mpPrimGridOpt->allocate(mpPrimGrids->size());
    mpClsGridOpt->allocate(mpClsGrids->size());
    mpDiffNetOpt->allocate(mpDiffNet);
    mpSpecNetOpt->allocate(mpSpecNet);

    mpPrimGridsInference = mpPrimGridOpt->custom_weights() ? mpPrimGridOpt->custom_weights() : mpPrimGrids->data();
    mpClsGridsInference = mpClsGridOpt->custom_weights() ? mpClsGridOpt->custom_weights() : mpClsGrids->data();
    mpDiffParamsInference = mpDiffNetOpt->custom_weights() ? mpDiffNetOpt->custom_weights() : mpDiffParams->data();
    mpSpecParamsInference = mpSpecNetOpt->custom_weights() ? mpSpecNetOpt->custom_weights() : mpSpecParams->data();

    mpDiffNet->set_params(mpDiffParams->data(), mpDiffParamsInference, mpDiffParamsGradient->data());
    mpSpecNet->set_params(mpSpecParams->data(), mpSpecParamsInference, mpSpecParamsGradient->data());
}

void NRModel::createOfflineModelsIfNeeded()
{
//     if (mpDiffOfflineNet && mpSpecOfflineNet && mpDiffOfflineTrainer && mpSpecOfflineTrainer) {
//         return;
//     }

//     HashGrid::Config gridConfig{nLevels, nFeaturesPerLevel, log2HashMapSize, baseResolution, perLevelScale};
//     HashGrid::Config primGridConfig{nLevels, nFeaturesPerLevel, log2HashMapSize, baseResolution, perLevelScale};
//     HashGridInterp::Config clsGridConfig{
//         numClusters,
//         nInterpLevels,
//         nInterpFeaturesPerLevel,
//         log2InterpHashMapSize,
//         baseInterpResolution,
//         perLevelInterpScale,
//         interpRatio
//     };

//     mpDiffOfflineNet = std::make_shared<tcnn::NeuralModel<precision_t>>(gridConfig);
//     {
//         auto offlineOptimizer = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(mOfflineOptConfig));
//         auto offlineLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(mLossConfig));
//         mpDiffOfflineTrainer = std::make_unique<Trainer>(mpDiffOfflineNet, offlineOptimizer, offlineLoss);
//     }

//     mpSpecOfflineNet = std::make_shared<tcnn::NeuralConeModel<precision_t>>(primGridConfig, clsGridConfig);
//     {
//         auto offlineOptimizer = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(mOfflineOptConfig));
//         auto offlineLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(mLossConfig));
//         mpSpecOfflineTrainer = std::make_unique<Trainer>(mpSpecOfflineNet, offlineOptimizer, offlineLoss);
//     }
}

NRModel::~NRModel() {
    mpPrimGrids.reset();
    mpPrimGridsGradient.reset();
    mpClsGrids.reset();
    mpClsGridsGradient.reset();

    mpDiffNet.reset();
    mpSpecNet.reset();
    mpDiffParams.reset();
    mpDiffParamsGradient.reset();
    mpSpecParams.reset();
    mpSpecParamsGradient.reset();

    mpPrimGridOpt.reset();
    mpClsGridOpt.reset();
    mpDiffNetOpt.reset();
    mpSpecNetOpt.reset();
    mpLoss.reset();

    if (mpPrimCollisionCounters) { cudaFree(mpPrimCollisionCounters); mpPrimCollisionCounters = nullptr; }
    if (mpPrimCollisionMarker)   { cudaFree(mpPrimCollisionMarker);   mpPrimCollisionMarker   = nullptr; }
    if (mpClsCollisionCounters)  { cudaFree(mpClsCollisionCounters);  mpClsCollisionCounters  = nullptr; }
    if (mpClsCollisionMarker)    { cudaFree(mpClsCollisionMarker);    mpClsCollisionMarker    = nullptr; }
    if (mpClsLevelHistogram)     { cudaFree(mpClsLevelHistogram);     mpClsLevelHistogram     = nullptr; }

    if (mTrainStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mTrainStream));
    }
    if (mInferenceStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mInferenceStream));
    }
}

void NRModel::resetCollisionStats() {
    if (!mCollisionStatsEnabled) return;
    HashGrid::resetCollisionStats(mInferenceStream, mPrimGridConfig);
    HashGridInterp::resetCollisionStats(mInferenceStream, mClsGridConfig);
    HashGridInterp::resetLevelHistogram(mInferenceStream, mClsGridConfig);
    HashGrid::resetCollisionStats(mTrainStream, mPrimGridConfig);
    HashGridInterp::resetCollisionStats(mTrainStream, mClsGridConfig);
    HashGridInterp::resetLevelHistogram(mTrainStream, mClsGridConfig);
}

void NRModel::setCollisionStatsEnabled(bool enabled) {
    if (enabled == mCollisionStatsEnabled) return;
    mCollisionStatsEnabled = enabled;

    if (enabled) {
        // Allocate counters (2 uint32) + marker buffer (per level).
        const uint32_t primMaxHash = 1u << mPrimGridConfig.log2HashMapSize;
        mPrimMarkerSize = primMaxHash * mPrimGridConfig.nLevels;
        cudaMalloc(&mpPrimCollisionCounters, sizeof(uint32_t) * 2);
        cudaMalloc(&mpPrimCollisionMarker,   sizeof(uint32_t) * mPrimMarkerSize);
        cudaMemset(mpPrimCollisionCounters, 0, sizeof(uint32_t) * 2);
        cudaMemset(mpPrimCollisionMarker,   0, sizeof(uint32_t) * mPrimMarkerSize);

        const uint32_t clsMaxHash = 1u << mClsGridConfig.log2HashMapSize;
        mClsMarkerSize = clsMaxHash * mClsGridConfig.nLevels;
        cudaMalloc(&mpClsCollisionCounters, sizeof(uint32_t) * 2);
        cudaMalloc(&mpClsCollisionMarker,   sizeof(uint32_t) * mClsMarkerSize);
        cudaMemset(mpClsCollisionCounters, 0, sizeof(uint32_t) * 2);
        cudaMemset(mpClsCollisionMarker,   0, sizeof(uint32_t) * mClsMarkerSize);

        // Specular level-usage histogram: nLevels * nLevels uint32_t.
        const uint32_t histSize = mClsGridConfig.nLevels * mClsGridConfig.nLevels;
        cudaMalloc(&mpClsLevelHistogram, sizeof(uint32_t) * histSize);
        cudaMemset(mpClsLevelHistogram, 0, sizeof(uint32_t) * histSize);
        mClsGridConfig.levelHistogram = mpClsLevelHistogram;

        mPrimGridConfig.counters = mpPrimCollisionCounters;
        mPrimGridConfig.slotMarker = mpPrimCollisionMarker;
        mClsGridConfig.counters = mpClsCollisionCounters;
        mClsGridConfig.slotMarker = mpClsCollisionMarker;
    } else {
        mPrimGridConfig.counters = nullptr;
        mPrimGridConfig.slotMarker = nullptr;
        mClsGridConfig.counters = nullptr;
        mClsGridConfig.slotMarker = nullptr;
        mClsGridConfig.levelHistogram = nullptr;

        if (mpPrimCollisionCounters) { cudaFree(mpPrimCollisionCounters); mpPrimCollisionCounters = nullptr; }
        if (mpPrimCollisionMarker)   { cudaFree(mpPrimCollisionMarker);   mpPrimCollisionMarker   = nullptr; }
        if (mpClsCollisionCounters)  { cudaFree(mpClsCollisionCounters);  mpClsCollisionCounters  = nullptr; }
        if (mpClsCollisionMarker)    { cudaFree(mpClsCollisionMarker);    mpClsCollisionMarker    = nullptr; }
        if (mpClsLevelHistogram)     { cudaFree(mpClsLevelHistogram);     mpClsLevelHistogram     = nullptr; }
        mPrimMarkerSize = 0;
        mClsMarkerSize = 0;
    }
}

NRModel::CollisionStats NRModel::getCollisionStats() const {
    if (!mCollisionStatsEnabled) return {0, 0, 0, 0};

    // Make sure all counter writes on both streams are visible to host.
    if (mInferenceStream) cudaStreamSynchronize(mInferenceStream);
    if (mTrainStream)      cudaStreamSynchronize(mTrainStream);

    uint32_t primHost[2] = {0, 0};
    uint32_t clsHost[2] = {0, 0};
    if (mpPrimCollisionCounters) {
        cudaMemcpy(primHost, mpPrimCollisionCounters, sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost);
    }
    if (mpClsCollisionCounters) {
        cudaMemcpy(clsHost, mpClsCollisionCounters, sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost);
    }
    mLastCollisionStats.primUniqueSlots   = primHost[0];
    mLastCollisionStats.primTotalQueries  = primHost[1];
    mLastCollisionStats.clsUniqueSlots    = clsHost[0];
    mLastCollisionStats.clsTotalQueries   = clsHost[1];
    return mLastCollisionStats;
}

void NRModel::getLevelHistogram(std::vector<uint32_t>& outHostHistogram) const {
    const uint32_t nLevels = mClsGridConfig.nLevels;
    outHostHistogram.assign((size_t)nLevels * nLevels, 0u);
    if (!mCollisionStatsEnabled || mpClsLevelHistogram == nullptr) return;

    if (mInferenceStream) cudaStreamSynchronize(mInferenceStream);
    if (mTrainStream)      cudaStreamSynchronize(mTrainStream);

    cudaMemcpy(outHostHistogram.data(), mpClsLevelHistogram,
               sizeof(uint32_t) * (size_t)nLevels * nLevels,
               cudaMemcpyDeviceToHost);
}

std::vector<float> NRModel::getClusterGridScales() const {
    const uint32_t nLevels = mClsGridConfig.nLevels;
    std::vector<float> scales(nLevels);
    float resolution = (float)mClsGridConfig.baseResolution;
    for (uint32_t i = 0; i < nLevels; ++i) {
        scales[i] = mClsGridConfig.interpRatio / resolution;
        resolution *= mClsGridConfig.perLevelScale;
    }
    return scales;
}

// --- Runtime hyperparameter tuning ---

// Helper: cast optimizer to EmaOptimizer to access decay
static float getEmaDecay(const std::shared_ptr<tcnn::Optimizer<precision_t>>& opt) {
    return opt->hyperparams()["decay"];
}

static void setEmaDecay(std::shared_ptr<tcnn::Optimizer<precision_t>>& opt, float v) {
    tcnn::json hp = opt->hyperparams();
    hp["decay"] = v;
    opt->update_hyperparams(hp);
}

float NRModel::getPrimGridEmaDecay() const { return getEmaDecay(mpPrimGridOpt); }
float NRModel::getClsGridEmaDecay()  const { return getEmaDecay(mpClsGridOpt); }
float NRModel::getDiffNetEmaDecay()  const { return getEmaDecay(mpDiffNetOpt); }
float NRModel::getSpecNetEmaDecay()  const { return getEmaDecay(mpSpecNetOpt); }

void NRModel::setPrimGridEmaDecay(float v) { setEmaDecay(mpPrimGridOpt, v); }
void NRModel::setClsGridEmaDecay(float v)  { setEmaDecay(mpClsGridOpt, v); }
void NRModel::setDiffNetEmaDecay(float v)  { setEmaDecay(mpDiffNetOpt, v); }
void NRModel::setSpecNetEmaDecay(float v)  { setEmaDecay(mpSpecNetOpt, v); }

void NRModel::setInterpRatio(float v) {
    if (v <= 0.f) return;
    mClsGridConfig.interpRatio = v;
    HashGridInterp::initializeConstants(mClsGridConfig);
}

// Learning rate accessors
float NRModel::getPrimGridLr() const { return mpPrimGridOpt->learning_rate(); }
float NRModel::getClsGridLr()  const { return mpClsGridOpt->learning_rate(); }
float NRModel::getDiffNetLr()  const { return mpDiffNetOpt->learning_rate(); }
float NRModel::getSpecNetLr()  const { return mpSpecNetOpt->learning_rate(); }

void NRModel::setPrimGridLr(float v) { mpPrimGridOpt->set_learning_rate(v); }
void NRModel::setClsGridLr(float v)  { mpClsGridOpt->set_learning_rate(v); }
void NRModel::setDiffNetLr(float v)  { mpDiffNetOpt->set_learning_rate(v); }
void NRModel::setSpecNetLr(float v)  { mpSpecNetOpt->set_learning_rate(v); }

uint32_t NRModel::getOptimizerStep() const { return mpDiffNetOpt->step(); }

void NRModel::setOnline(bool online)
{
    if (online == mOnline) {
        return;
    }

    // if (online) {
    //     syncTrainerParams(*mpDiffTrainer, *mpDiffOnlineTrainer);
    //     syncTrainerParams(*mpSpecTrainer, *mpSpecOnlineTrainer);
    //     mpDiffNet = mpDiffOnlineNet.get();
    //     mpSpecNet = mpSpecOnlineNet.get();
    //     mpDiffTrainer = mpDiffOnlineTrainer.get();
    //     mpSpecTrainer = mpSpecOnlineTrainer.get();
    // } else {
    //     createOfflineModelsIfNeeded();
    //     syncTrainerParams(*mpDiffTrainer, *mpDiffOfflineTrainer);
    //     syncTrainerParams(*mpSpecTrainer, *mpSpecOfflineTrainer);
    //     mpDiffNet = mpDiffOfflineNet.get();
    //     mpSpecNet = mpSpecOfflineNet.get();
    //     mpDiffTrainer = mpDiffOfflineTrainer.get();
    //     mpSpecTrainer = mpSpecOfflineTrainer.get();
    // }

    mOnline = online;
}

void NRModel::inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs) {
    const uint32_t diffBatchSize = padUp(diffPtrs.size, 256);
    const uint32_t specBatchSize = padUp(specPtrs.size, 256);

    tcnn::GPUMatrix<precision_t> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
    tcnn::GPUMatrix<precision_t> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
    tcnn::GPUMatrix<float> diffOutput(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specOutput(specPtrs.outputPtr, mOutputDim, specBatchSize);

    if (diffBatchSize > 0) {
        // tcnn::GPUMatrix<precision_t> networkInput{mDiffInputDim, diffBatchSize, mInferenceStream};
        // castMatrix(mInferenceStream, diffInput, networkInput);
        // moveGBuffer(mInferenceStream, diffGBuffer, networkInput, mDiffPosOffset);
        HashGrid::launchForward(
            mInferenceStream,
            mpPrimGridsInference,
            diffInput.data(),
            diffBatchSize,
            mDiffInputDim,
            mDiffEncOffset,
            mDiffPosOffset,
            mPrimGridConfig
        );
        mpDiffNet->inference(mInferenceStream, diffInput, diffOutput);
    }
    if (specBatchSize > 0) {
        // tcnn::GPUMatrix<precision_t> networkInput{mSpecInputDim, specBatchSize, mInferenceStream};
        // castMatrix(mInferenceStream, specInput, networkInput);
        // moveGBuffer(mInferenceStream, specGBuffer, networkInput, mSpecPosOffset);
        HashGrid::launchForward(
            mInferenceStream,
            mpPrimGridsInference,
            specInput.data(),
            specBatchSize,
            mSpecInputDim,
            mSpecPrimEncOffset,
            mSpecPosOffset,
            mPrimGridConfig
        );
        HashGridInterp::launchForward(
            mInferenceStream,
            mpClsGridsInference,
            specInput.data(),
            specBatchSize,
            mSpecInputDim,
            mSpecClsEncOffset,
            mSpecClsPosOffset,
            mSpecClsScaleOffset,
            mSpecClsWeightOffset,
            mClsGridConfig
        );
        mpSpecNet->inference(mInferenceStream, specInput, specOutput);
    }
}

void NRModel::train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs) {
    const uint32_t diffBatchSize = padUp(diffPtrs.size, 256);
    const uint32_t specBatchSize = padUp(specPtrs.size, 256);

    const float lossScale = tcnn::default_loss_scale<precision_t>();
    CUDA_CHECK_THROW(cudaMemsetAsync(mpPrimGridsGradient->data(), 0, mpPrimGridsGradient->bytes(), mTrainStream));
    CUDA_CHECK_THROW(cudaMemsetAsync(mpClsGridsGradient->data(), 0, mpClsGridsGradient->bytes(), mTrainStream));

    tcnn::GPUMatrix<precision_t> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
    tcnn::GPUMatrix<precision_t> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
    tcnn::GPUMatrix<float> diffTarget(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specTarget(specPtrs.outputPtr, mOutputDim, specBatchSize);

    if (diffBatchSize > 0) {
        // tcnn::GPUMatrix<precision_t> networkInput{mDiffInputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> networkOutput{mOutputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkOutput{mOutputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkInput{mDiffInputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<float> loss{mOutputDim, diffBatchSize, mTrainStream};
        // castMatrix(mTrainStream, diffInput, networkInput);
        // moveGBuffer(mTrainStream, diffGBuffer, networkInput, mDiffPosOffset);
        // Forward
        HashGrid::launchForward(
            mTrainStream,
            mpPrimGrids->data(),
            diffInput.data(),
            diffBatchSize,
            mDiffInputDim,
            mDiffEncOffset,
            mDiffPosOffset,
            mPrimGridConfig
        );
        auto ctx = mpDiffNet->forward(mTrainStream, diffInput, &networkOutput);
        // Loss
        mpLoss->evaluate(mTrainStream, lossScale, networkOutput, diffTarget, loss, dL_dnetworkOutput);
        // Backward
        mpDiffNet->backward(mTrainStream, *ctx, diffInput, networkOutput, dL_dnetworkOutput, &dL_dnetworkInput);
        HashGrid::launchBackward(
            mTrainStream,
            mpPrimGrids->data(),
            diffInput.data(),
            dL_dnetworkInput.data(),
            mpPrimGridsGradient->data(),
            diffBatchSize,
            mDiffInputDim,
            mDiffEncOffset,
            mDiffPosOffset,
            mPrimGridConfig
        );
    }
    if (specBatchSize > 0) {
        // tcnn::GPUMatrix<precision_t> networkInput{mSpecInputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> networkOutput{mOutputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkOutput{mOutputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkInput{mSpecInputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<float> loss{mOutputDim, specBatchSize, mTrainStream};
        // castMatrix(mTrainStream, specInput, networkInput);
        // moveGBuffer(mTrainStream, specGBuffer, networkInput, mSpecPosOffset);
        // Forward
        HashGrid::launchForward(
            mTrainStream,
            mpPrimGrids->data(),
            specInput.data(),
            specBatchSize,
            mSpecInputDim,
            mSpecPrimEncOffset,
            mSpecPosOffset,
            mPrimGridConfig
        );
        HashGridInterp::launchForward(
            mTrainStream,
            mpClsGrids->data(),
            specInput.data(),
            specBatchSize,
            mSpecInputDim,
            mSpecClsEncOffset,
            mSpecClsPosOffset,
            mSpecClsScaleOffset,
            mSpecClsWeightOffset,
            mClsGridConfig
        );
        auto ctx = mpSpecNet->forward(mTrainStream, specInput, &networkOutput);
        // Loss
        mpLoss->evaluate(mTrainStream, lossScale, networkOutput, specTarget, loss, dL_dnetworkOutput);
        // Backward
        mpSpecNet->backward(mTrainStream, *ctx, specInput, networkOutput, dL_dnetworkOutput, &dL_dnetworkInput);
        HashGrid::launchBackward(
            mTrainStream,
            mpPrimGrids->data(),
            specInput.data(),
            dL_dnetworkInput.data(),
            mpPrimGridsGradient->data(),
            specBatchSize,
            mSpecInputDim,
            mSpecPrimEncOffset,
            mSpecPosOffset,
            mPrimGridConfig
        );
        HashGridInterp::launchBackward(
            mTrainStream,
            mpClsGrids->data(),
            specInput.data(),
            dL_dnetworkInput.data(),
            mpClsGridsGradient->data(),
            specBatchSize,
            mSpecInputDim,
            mSpecClsEncOffset,
            mSpecClsPosOffset,
            mSpecClsScaleOffset,
            mSpecClsWeightOffset,
            mClsGridConfig
        );
    }
    if (diffBatchSize > 0 && !mFreezeDiffNetOpt) {
        mpDiffNetOpt->step(mTrainStream, lossScale, mpDiffParamsFull->data(), mpDiffParams->data(), mpDiffParamsGradient->data());
    }
    if (specBatchSize > 0 && !mFreezeSpecNetOpt) {
        mpSpecNetOpt->step(mTrainStream, lossScale, mpSpecParamsFull->data(), mpSpecParams->data(), mpSpecParamsGradient->data());
    }
    if (specBatchSize > 0 && !mFreezeClsGridOpt) {
        mpClsGridOpt->step(mTrainStream, lossScale, mpClsGridsFull->data(), mpClsGrids->data(), mpClsGridsGradient->data());
    }
    if ((diffBatchSize > 0 || specBatchSize > 0) && !mFreezePrimGridOpt) {
        mpPrimGridOpt->step(mTrainStream, lossScale, mpPrimGridsFull->data(), mpPrimGrids->data(), mpPrimGridsGradient->data());
    }
}

// ============================================================================
// Checkpoint save / load (from the Windows working branch).
// ============================================================================

namespace {
void writeGPUBuffer(std::ofstream& ofs, const tcnn::GPUMemory<precision_t>& halfBuf, const tcnn::GPUMemory<float>& fullBuf) {
    const uint32_t count = (uint32_t)halfBuf.size();
    std::vector<float> host(count);
    CUDA_CHECK_THROW(cudaMemcpy(host.data(), fullBuf.data(), count * sizeof(float), cudaMemcpyDeviceToHost));
    ofs.write(reinterpret_cast<const char*>(host.data()), count * sizeof(float));
}

void readGPUBuffer(std::ifstream& ifs, tcnn::GPUMemory<precision_t>& halfBuf, tcnn::GPUMemory<float>& fullBuf) {
    const uint32_t count = (uint32_t)halfBuf.size();
    std::vector<float> host(count);
    ifs.read(reinterpret_cast<char*>(host.data()), count * sizeof(float));
    CUDA_CHECK_THROW(cudaMemcpy(fullBuf.data(), host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(halfBuf.data(), host.data(), count * sizeof(precision_t), cudaMemcpyHostToDevice));
}
} // namespace

void NRModel::saveState(const std::string& path) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open file for writing: " + path);

    const uint32_t version = 1;
    const uint32_t nPrim = (uint32_t)mpPrimGrids->size();
    const uint32_t nCls = (uint32_t)mpClsGrids->size();
    const uint32_t nDiff = (uint32_t)mpDiffParams->size();
    const uint32_t nSpec = (uint32_t)mpSpecParams->size();
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&nPrim), sizeof(nPrim));
    ofs.write(reinterpret_cast<const char*>(&nCls), sizeof(nCls));
    ofs.write(reinterpret_cast<const char*>(&nDiff), sizeof(nDiff));
    ofs.write(reinterpret_cast<const char*>(&nSpec), sizeof(nSpec));

    writeGPUBuffer(ofs, *mpPrimGrids, *mpPrimGridsFull);
    writeGPUBuffer(ofs, *mpClsGrids, *mpClsGridsFull);
    writeGPUBuffer(ofs, *mpDiffParams, *mpDiffParamsFull);
    writeGPUBuffer(ofs, *mpSpecParams, *mpSpecParamsFull);

    ofs.close();
    fmt::print("NRModel::saveState: saved to {}\n", path);
}

// Reset the EMA optimizer's nested Adam step counter to a large value so that
// the EMA debias factor is ~1.0 from the first step.
static void resetOptimizerStepCounter(
    const std::shared_ptr<tcnn::Optimizer<precision_t>>& opt,
    uint32_t nWeights)
{
    if (!opt) return;
    tcnn::json state = opt->serialize();
    if (state.contains("nested")) {
        auto& nested = state["nested"];
        if (nested.contains("current_step")) {
            nested["current_step"] = 10000u;
        }
        if (nested.contains("first_moments_binary")) {
            auto& bin = nested["first_moments_binary"].get_binary();
            std::fill(bin.begin(), bin.end(), 0);
        }
        if (nested.contains("second_moments_binary")) {
            auto& bin = nested["second_moments_binary"].get_binary();
            std::fill(bin.begin(), bin.end(), 0);
        }
        if (nested.contains("param_steps_binary")) {
            auto& bin = nested["param_steps_binary"].get_binary();
            std::fill(bin.begin(), bin.end(), 0);
        }
    }
    opt->deserialize(state);
}

void NRModel::loadState(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    uint32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error(fmt::format("Unsupported checkpoint version: {}", version));
    }

    fmt::print("Loading checkpoint: primGrids={}, clsGrids={}, diffParams={}, specParams={}\n",
        mpPrimGrids->size(), mpClsGrids->size(), mpDiffParams->size(), mpSpecParams->size());

    readGPUBuffer(ifs, *mpPrimGrids, *mpPrimGridsFull);
    fmt::print("  primGrids loaded\n");
    readGPUBuffer(ifs, *mpClsGrids, *mpClsGridsFull);
    fmt::print("  clsGrids loaded\n");
    readGPUBuffer(ifs, *mpDiffParams, *mpDiffParamsFull);
    fmt::print("  diffParams loaded\n");
    readGPUBuffer(ifs, *mpSpecParams, *mpSpecParamsFull);
    fmt::print("  specParams loaded\n");

    // Copy loaded weights to EMA optimizer's custom weights so inference uses them
    const uint32_t primGridSize = mpPrimGrids->size();
    const uint32_t clsGridSize = mpClsGrids->size();
    const uint32_t diffNetSize = mpDiffParams->size();
    const uint32_t specNetSize = mpSpecParams->size();

    if (mpPrimGridOpt->custom_weights())
        cudaMemcpy(mpPrimGridOpt->custom_weights(), mpPrimGrids->data(), primGridSize * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    if (mpClsGridOpt->custom_weights())
        cudaMemcpy(mpClsGridOpt->custom_weights(), mpClsGrids->data(), clsGridSize * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    if (mpDiffNetOpt->custom_weights())
        cudaMemcpy(mpDiffNetOpt->custom_weights(), mpDiffParams->data(), diffNetSize * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    if (mpSpecNetOpt->custom_weights())
        cudaMemcpy(mpSpecNetOpt->custom_weights(), mpSpecParams->data(), specNetSize * sizeof(precision_t), cudaMemcpyDeviceToDevice);

    // Reset optimizer step counters so EMA debias doesn't overwrite loaded weights
    resetOptimizerStepCounter(mpPrimGridOpt, primGridSize);
    resetOptimizerStepCounter(mpClsGridOpt, clsGridSize);
    resetOptimizerStepCounter(mpDiffNetOpt, diffNetSize);
    resetOptimizerStepCounter(mpSpecNetOpt, specNetSize);

    mpPrimGridsInference = mpPrimGridOpt->custom_weights() ? mpPrimGridOpt->custom_weights() : mpPrimGrids->data();
    mpClsGridsInference = mpClsGridOpt->custom_weights() ? mpClsGridOpt->custom_weights() : mpClsGrids->data();
    mpDiffParamsInference = mpDiffNetOpt->custom_weights() ? mpDiffNetOpt->custom_weights() : mpDiffParams->data();
    mpSpecParamsInference = mpSpecNetOpt->custom_weights() ? mpSpecNetOpt->custom_weights() : mpSpecParams->data();

    ifs.close();
    fmt::print("NRModel::loadState: loaded from {}\n", path);
}

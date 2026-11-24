#include "Model.h"
#include "HashGrid.h"
#include "HashGridInterp.h"
#include "TrainingBatch.h"
#include "Diagnostics.h"
// #include "NeuralModel.h"
// #include "NeuralConeModel.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <cstring>
#include <limits>
#include <fmt/format.h>
#include "tiny-cuda-nn/random.h"
#include "tiny-cuda-nn/config.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/trainer.h"


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

// Reference data for the OpenHarmony port (ohos/npu/verify_tcnn.py), which re-implements this model
// without tiny-cuda-nn. With the environment variable NCR_DUMP_DIR set, the train() call number
// NCR_DUMP_STEP (default 50) writes everything that step read and produced, and the next render
// inference writes its inputs and outputs. Each array is a raw <name>.bin; manifest.txt lists
// "array name dtype rows cols" and "scalar name value" lines.
struct DebugDump {
    bool enabled = false;
    std::string dir;
    uint32_t step = 50;
    uint32_t trainCalls = 0;
    bool trainDone = false;
    bool inferenceDone = false;

    DebugDump() {
        const char* d = std::getenv("NCR_DUMP_DIR");
        enabled = d && *d;
        if (!enabled) return;
        dir = d;
        if (const char* s = std::getenv("NCR_DUMP_STEP")) step = (uint32_t)std::atoi(s);
        std::ofstream(dir + "/manifest.txt", std::ios::trunc);
    }

    void note(const std::string& line) {
        std::ofstream f(dir + "/manifest.txt", std::ios::app);
        f << line << "\n";
    }

    void writeHost(const std::string& name, const char* dtype, size_t rows, size_t cols, const void* data, size_t bytes) {
        std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
        f.write(static_cast<const char*>(data), (std::streamsize)bytes);
        if (!f.good()) throw std::runtime_error("NCR_DUMP_DIR: cannot write " + name);
        note(fmt::format("array {} {} {} {}", name, dtype, rows, cols));
    }

    void writeDevice(const std::string& name, const char* dtype, size_t rows, size_t cols, const void* devPtr, size_t elemBytes) {
        std::vector<char> host(rows * cols * elemBytes);
        if (!host.empty()) CUDA_CHECK_THROW(cudaMemcpy(host.data(), devPtr, host.size(), cudaMemcpyDeviceToHost));
        writeHost(name, dtype, rows, cols, host.data(), host.size());
    }

    // Adam moments, per-parameter step counts and the step counter of an EMA(Adam) optimizer.
    void writeOptimizer(const std::string& name, const tcnn::Optimizer<precision_t>& optimizer) {
        const tcnn::json data = optimizer.serialize();
        const tcnn::json& adam = data["nested"];
        for (const char* key : {"first_moments", "second_moments", "param_steps"}) {
            const auto& binary = adam[std::string(key) + "_binary"].get_binary();
            writeHost(name + "_" + key, std::string(key) == "param_steps" ? "u32" : "f32", binary.size() / 4, 1, binary.data(), binary.size());
        }
        note(fmt::format("scalar {}_step {}", name, adam["current_step"].get<uint32_t>()));
    }
};
DebugDump gDump;

void syncInitialParameters(
    cudaStream_t stream,
    const tcnn::GPUMemory<float>& master,
    tcnn::GPUMemory<precision_t>& working,
    tcnn::GPUMemory<precision_t>& gradients,
    precision_t* inference)
{
    const float* src = master.data();
    precision_t* dst = working.data();
    tcnn::parallel_for_gpu(stream, master.size(), [src, dst, inference] __device__ (size_t i) {
        const precision_t value = precision_t(src[i]);
        dst[i] = value;
        inference[i] = value;
    });
    CUDA_CHECK_THROW(cudaMemsetAsync(gradients.data(), 0, gradients.bytes(), stream));
}

// Read back once at startup. Working and EMA values must match the rounded
// master parameters exactly; FP32 -> FP16 quantization error itself is expected.
void logInitializationStats(
    const char* name,
    const tcnn::GPUMemory<float>& master,
    const precision_t* working,
    const precision_t* inference,
    const tcnn::Optimizer<precision_t>& optimizer)
{
    const size_t count = master.size();
    CHECK_THROW(count > 0);
    std::vector<float> fp32(count);
    std::vector<precision_t> fp16(count), ema(count);
    CUDA_CHECK_THROW(cudaMemcpy(fp32.data(), master.data(), master.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(fp16.data(), working, count * sizeof(precision_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(ema.data(), inference, count * sizeof(precision_t), cudaMemcpyDeviceToHost));

    double sum = 0., sumSquared = 0.;
    float minValue = std::numeric_limits<float>::infinity();
    float maxValue = -std::numeric_limits<float>::infinity();
    float maxQuantizationError = 0.f, maxEmaError = 0.f;
    size_t nonFiniteMaster = 0, nonFiniteWorking = 0, nonFiniteEma = 0;
    size_t castMismatch = 0, emaMismatch = 0, zeroWorking = 0;
    uint64_t fingerprint = 14695981039346656037ull;
    for (size_t i = 0; i < count; ++i)
    {
        const float x = fp32[i], w = float(fp16[i]), e = float(ema[i]);
        nonFiniteMaster += !std::isfinite(x);
        nonFiniteWorking += !std::isfinite(w);
        nonFiniteEma += !std::isfinite(e);
        castMismatch += w != float(precision_t(x));
        emaMismatch += e != w;
        zeroWorking += w == 0.f;
        if (std::isfinite(x))
        {
            minValue = std::min(minValue, x);
            maxValue = std::max(maxValue, x);
            sum += x;
            sumSquared += double(x) * x;
        }
        if (std::isfinite(x) && std::isfinite(w)) maxQuantizationError = std::max(maxQuantizationError, std::abs(x - w));
        if (std::isfinite(w) && std::isfinite(e)) maxEmaError = std::max(maxEmaError, std::abs(w - e));
        uint32_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        for (uint32_t shift = 0; shift < 32; shift += 8)
        {
            fingerprint ^= (bits >> shift) & 0xffu;
            fingerprint *= 1099511628211ull;
        }
    }
    const double mean = sum / count;
    const double stddev = std::sqrt(std::max(0., sumSquared / count - mean * mean));
    fmt::print(
        "[NCR Init] {} count={} fp32[min={:.7g} max={:.7g} mean={:.7g} std={:.7g}] "
        "nonfinite[fp32={} working={} ema={}] working_zero={} cast_mismatch={} ema_mismatch={} "
        "max_quant_error={:.7g} max_ema_error={:.7g} optimizer_step={} fingerprint={:016x}\n",
        name, count, minValue, maxValue, mean, stddev, nonFiniteMaster, nonFiniteWorking, nonFiniteEma,
        zeroWorking, castMismatch, emaMismatch, maxQuantizationError, maxEmaError, optimizer.step(), fingerprint
    );
    CHECK_THROW(nonFiniteMaster == 0 && nonFiniteWorking == 0 && nonFiniteEma == 0);
    CHECK_THROW(castMismatch == 0 && emaMismatch == 0 && optimizer.step() == 0);
}

} // namespace

NRModel::NRModel(const NRModelConfig& config) : mConfig(config) {
    CHECK_THROW(std::isfinite(mConfig.initGridScale) && mConfig.initGridScale > 0.f && mConfig.initGridScale <= 1.f);
    CUDA_CHECK_THROW(cudaStreamCreate(&mTrainStream));
    CUDA_CHECK_THROW(cudaStreamCreate(&mInferenceStream));

    mDiffGBufferDim = 3 * 4 + 1;
    mSpecGBufferDim = 3 * 5 + 1 + NCR_NUM_CLUSTERS * (3 + 1 + 1);
    mDiffInputDim = padUp(mConfig.nFeaturesPerLevel * mConfig.nLevels + mDiffGBufferDim, 16);
    mSpecInputDim = padUp(mConfig.nFeaturesPerLevel * mConfig.nLevels + mConfig.nFeaturesPerLevelSpec * NCR_NUM_CLUSTERS + mSpecGBufferDim, 16);
    mOutputDim = NCR_OUTPUT_DIM;

    std::seed_seq seq{mConfig.initSeed};
    std::vector<uint32_t> seeds(2);
    seq.generate(std::begin(seeds), std::end(seeds));
    mpRng = std::make_shared<tcnn::pcg32>(seeds.front());

    mPrimGridConfig = {
        mConfig.nLevels,
        mConfig.nFeaturesPerLevel,
        mConfig.log2HashmapSize,
        mConfig.baseResolution,
        mConfig.perLevelScale
    };
    mClsGridConfig = {
        NCR_NUM_CLUSTERS,
        mConfig.nLevelsSpec,
        mConfig.nFeaturesPerLevelSpec,
        mConfig.log2HashmapSizeSpec,
        mConfig.baseResolutionSpec,
        mConfig.perLevelScaleSpec,
        mConfig.interpRatio
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
        {"n_neurons", mConfig.diffNNeurons},
        {"n_hidden_layers", mConfig.diffNHiddenLayers}
    };

    const tcnn::json specNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mSpecInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", mConfig.specNNeurons},
        {"n_hidden_layers", mConfig.specNHiddenLayers}
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

    std::seed_seq gridSeedSeq{mConfig.initSeed, 0x47524944u};
    uint32_t gridSeed;
    gridSeedSeq.generate(&gridSeed, &gridSeed + 1);
    tcnn::pcg32 gridRng(gridSeed);
    tcnn::generate_random_uniform<float>(
        mTrainStream, gridRng, mpPrimGridsFull->size(), mpPrimGridsFull->data(), -mConfig.initGridScale, mConfig.initGridScale
    );
    tcnn::generate_random_uniform<float>(
        mTrainStream, gridRng, mpClsGridsFull->size(), mpClsGridsFull->data(), -mConfig.initGridScale, mConfig.initGridScale
    );
    mPaddedOutputDim = mpDiffNet->padded_output_width();
    CHECK_THROW(mpSpecNet->padded_output_width() == mPaddedOutputDim);
    CHECK_THROW(mpDiffNet->output_width() == mOutputDim);
    CHECK_THROW(mpSpecNet->output_width() == mOutputDim);

    const tcnn::json offlineOptConfig = {
        {"otype", "Adam"},
        {"learning_rate", NCR_ADAM_LR}
    };
    const tcnn::json onlineOptConfig = {
        {"otype", "EMA"},
        {"decay", NCR_EMA_DECAY},
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

    // Like tcnn::Trainer::set_params_full_precision(): initialize all three
    // representations before the first RHS inference/forward. Do NOT take a dummy
    // optimizer step: Adam moments and counters must remain at their initial state.
    // EMA stays in its existing debiased mode. At step 0 this is the inference
    // snapshot; the first real step has ema_debias_old == 0 and uses updated weights.
    syncInitialParameters(mTrainStream, *mpPrimGridsFull, *mpPrimGrids, *mpPrimGridsGradient, mpPrimGridsInference);
    syncInitialParameters(mTrainStream, *mpClsGridsFull, *mpClsGrids, *mpClsGridsGradient, mpClsGridsInference);
    syncInitialParameters(mTrainStream, *mpDiffParamsFull, *mpDiffParams, *mpDiffParamsGradient, mpDiffParamsInference);
    syncInitialParameters(mTrainStream, *mpSpecParamsFull, *mpSpecParams, *mpSpecParamsGradient, mpSpecParamsInference);
    // Required even when diagnostics are off: inference uses a different stream.
    CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));

    if (mConfig.initDiagnostics)
    {
        fmt::print("[NCR Init] seed={} grid_uniform=[{}, {}] mlp=Xavier working_bits={} ema=debiased-step0-snapshot\n",
            mConfig.initSeed, -mConfig.initGridScale, mConfig.initGridScale, 8 * sizeof(precision_t));
        logInitializationStats("primary_grid", *mpPrimGridsFull, mpPrimGrids->data(), mpPrimGridsInference, *mpPrimGridOpt);
        logInitializationStats("cluster_grid", *mpClsGridsFull, mpClsGrids->data(), mpClsGridsInference, *mpClsGridOpt);
        logInitializationStats("diffuse_mlp", *mpDiffParamsFull, mpDiffParams->data(), mpDiffParamsInference, *mpDiffNetOpt);
        logInitializationStats("specular_mlp", *mpSpecParamsFull, mpSpecParams->data(), mpSpecParamsInference, *mpSpecNetOpt);
        fmt::print("[NCR Init] PASS: finite parameters, rounded master == working == EMA, optimizer steps == 0.\n");
        std::fflush(stdout);
    }

    mpDiffNet->set_params(mpDiffParams->data(), mpDiffParamsInference, mpDiffParamsGradient->data());
    mpSpecNet->set_params(mpSpecParams->data(), mpSpecParamsInference, mpSpecParamsGradient->data());
}

void NRModel::createOfflineModelsIfNeeded()
{
//     if (mpDiffOfflineNet && mpSpecOfflineNet && mpDiffOfflineTrainer && mpSpecOfflineTrainer) {
//         return;
//     }

//     HashGrid::Config gridConfig{NCR_N_LEVELS, NCR_N_FEATURES_PER_LEVEL, NCR_LOG2_HASHMAP_SIZE, NCR_BASE_RESOLUTION, NCR_PER_LEVEL_SCALE};
//     HashGrid::Config primGridConfig{NCR_N_LEVELS, NCR_N_FEATURES_PER_LEVEL, NCR_LOG2_HASHMAP_SIZE, NCR_BASE_RESOLUTION, NCR_PER_LEVEL_SCALE};
//     HashGridInterp::Config clsGridConfig{
//         NCR_NUM_CLUSTERS,
//         NCR_N_LEVELS_SPEC,
//         NCR_N_FEATURES_PER_LEVEL_SPEC,
//         NCR_LOG2_HASHMAP_SIZE_SPEC,
//         NCR_BASE_RESOLUTION_SPEC,
//         NCR_PER_LEVEL_SCALE_SPEC,
//         NCR_INTERP_RATIO
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

    if (mTrainStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mTrainStream));
    }
    if (mInferenceStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mInferenceStream));
    }
}

void NRModel::setOptimizerHyperparams(float learningRate, float emaDecay)
{
    if (learningRate == mLearningRate && emaDecay == mEmaDecay) return;
    mLearningRate = learningRate;
    mEmaDecay = emaDecay;
    const tcnn::json emaParams = {{"decay", emaDecay}};
    for (auto* pOpt : {&mpPrimGridOpt, &mpClsGridOpt, &mpDiffNetOpt, &mpSpecNetOpt})
    {
        if (!*pOpt) continue;
        (*pOpt)->set_learning_rate(learningRate);
        (*pOpt)->update_hyperparams(emaParams);
    }
}

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
    const uint32_t diffBatchSize = (diffPtrs.inputPtr && diffPtrs.outputPtr) ? padUp(diffPtrs.size, kModelBatchAlign) : 0;
    const uint32_t specBatchSize = (specPtrs.inputPtr && specPtrs.outputPtr) ? padUp(specPtrs.size, kModelBatchAlign) : 0;

    if (diffBatchSize > 0) {
        tcnn::GPUMatrix<precision_t> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
        tcnn::GPUMatrix<float> diffOutput(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
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
        tcnn::GPUMatrix<precision_t> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
        tcnn::GPUMatrix<float> specOutput(specPtrs.outputPtr, mOutputDim, specBatchSize);
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
    // The first render inference (both networks) after the dumped training step. It reads the EMA
    // weights stored in train_after.ckpt.
    if (gDump.enabled && gDump.trainDone && !gDump.inferenceDone && diffBatchSize > 0 && specBatchSize > 0) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));
        const uint32_t diffRows = std::min(diffPtrs.size, 65536u);
        const uint32_t specRows = std::min(specPtrs.size, 65536u);
        gDump.writeDevice("infer_diff_input", "f16", diffRows, mDiffInputDim, diffPtrs.inputPtr, sizeof(precision_t));
        gDump.writeDevice("infer_diff_output", "f32", diffRows, mOutputDim, diffPtrs.outputPtr, sizeof(float));
        gDump.writeDevice("infer_spec_input", "f16", specRows, mSpecInputDim, specPtrs.inputPtr, sizeof(precision_t));
        gDump.writeDevice("infer_spec_output", "f32", specRows, mOutputDim, specPtrs.outputPtr, sizeof(float));
        gDump.inferenceDone = true;
        fmt::print("NRModel: wrote a render inference to {}\n", gDump.dir);
    }
}

void NRModel::train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs) {
    const uint32_t diffBatchSize = (diffPtrs.inputPtr && diffPtrs.outputPtr) ? padUp(diffPtrs.size, kModelBatchAlign) : 0;
    const uint32_t specBatchSize = (specPtrs.inputPtr && specPtrs.outputPtr) ? padUp(specPtrs.size, kModelBatchAlign) : 0;

    tcnn::json diagnostic;
    const float lossScale = tcnn::default_loss_scale<precision_t>();
    const bool dump = gDump.enabled && !gDump.trainDone && ++gDump.trainCalls == gDump.step;
    if (dump) {
        // What the step starts from: f32 masters and EMA copies (a checkpoint), the fp16 training
        // copies the forward pass reads, and the moments of the optimizers.
        saveState(gDump.dir + "/train_before.ckpt");
        gDump.note(fmt::format("scalar learning_rate {}", mLearningRate));
        gDump.note(fmt::format("scalar ema_decay {}", mEmaDecay));
        gDump.note(fmt::format("scalar loss_scale {}", lossScale));
        gDump.note(fmt::format("scalar diff_size {}", diffBatchSize > 0 ? diffPtrs.size : 0));
        gDump.note(fmt::format("scalar spec_size {}", specBatchSize > 0 ? specPtrs.size : 0));
        gDump.writeDevice("before_prim_grid_half", "f16", mpPrimGrids->size(), 1, mpPrimGrids->data(), sizeof(precision_t));
        gDump.writeDevice("before_cls_grid_half", "f16", mpClsGrids->size(), 1, mpClsGrids->data(), sizeof(precision_t));
        gDump.writeDevice("before_diff_params_half", "f16", mpDiffParams->size(), 1, mpDiffParams->data(), sizeof(precision_t));
        gDump.writeDevice("before_spec_params_half", "f16", mpSpecParams->size(), 1, mpSpecParams->data(), sizeof(precision_t));
        gDump.writeOptimizer("prim_grid_opt", *mpPrimGridOpt);
        gDump.writeOptimizer("cls_grid_opt", *mpClsGridOpt);
        gDump.writeOptimizer("diff_opt", *mpDiffNetOpt);
        gDump.writeOptimizer("spec_opt", *mpSpecNetOpt);
    }
    CUDA_CHECK_THROW(cudaMemsetAsync(mpPrimGridsGradient->data(), 0, mpPrimGridsGradient->bytes(), mTrainStream));
    CUDA_CHECK_THROW(cudaMemsetAsync(mpClsGridsGradient->data(), 0, mpClsGridsGradient->bytes(), mTrainStream));

    if (diffBatchSize > 0) {
        tcnn::GPUMatrix<precision_t> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
        tcnn::GPUMatrix<float> diffTarget(diffPtrs.outputPtr, mOutputDim, diffPtrs.size);
        ncr::clearTrainingInputPadding(mTrainStream, diffInput, diffPtrs.size);
        // Forward writes the padded 16-wide tile. The loss supervises mOutputDim
        // and zeros the gradient on padded channels and padded samples.
        tcnn::GPUMatrix<precision_t> networkOutput{mPaddedOutputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkOutput{mPaddedOutputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkInput{mDiffInputDim, diffBatchSize, mTrainStream};
        tcnn::GPUMatrix<float> loss{mPaddedOutputDim, diffBatchSize, mTrainStream};
        // castMatrix(mTrainStream, diffInput, networkInput);
        // moveGBuffer(mTrainStream, diffGBuffer, networkInput, mDiffPosOffset);
        // Forward
        HashGrid::launchForward(
            mTrainStream,
            mpPrimGrids->data(),
            diffInput.data(),
            diffPtrs.size,
            mDiffInputDim,
            mDiffEncOffset,
            mDiffPosOffset,
            mPrimGridConfig
        );
        auto ctx = mpDiffNet->forward(mTrainStream, diffInput, &networkOutput);
        // Loss
        ncr::evaluateValidSamples(mTrainStream, lossScale, *mpLoss, networkOutput, diffTarget, loss, dL_dnetworkOutput);
        // Backward
        mpDiffNet->backward(mTrainStream, *ctx, diffInput, networkOutput, dL_dnetworkOutput, &dL_dnetworkInput);
        if (mTrainingDiagnostics) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
            diagnostic["diff"]["target"] = ncr::outputStatistics(diffTarget.data(), diffPtrs.size, mOutputDim, mOutputDim);
            diagnostic["diff"]["working_prediction_pre_update"] = ncr::outputStatistics(networkOutput.data(), diffPtrs.size, mPaddedOutputDim, mOutputDim);
            diagnostic["diff"]["input_values"] = ncr::gradientStatistics(diffInput.data(), size_t(diffPtrs.size)*mDiffInputDim, true, 1.f);
            diagnostic["diff"]["full_output_values"] = ncr::gradientStatistics(networkOutput.data(), size_t(diffPtrs.size)*mPaddedOutputDim, true, 1.f);
            diagnostic["diff"]["loss_output_gradient"] = ncr::gradientStatistics(dL_dnetworkOutput.data(), size_t(diffPtrs.size)*mPaddedOutputDim, true, lossScale);
            diagnostic["diff"]["network_input_gradient"] = ncr::gradientStatistics(dL_dnetworkInput.data(), size_t(diffPtrs.size)*mDiffInputDim, true, lossScale);
        }
        HashGrid::launchBackward(
            mTrainStream,
            mpPrimGrids->data(),
            diffInput.data(),
            dL_dnetworkInput.data(),
            mpPrimGridsGradient->data(),
            diffPtrs.size,
            mDiffInputDim,
            mDiffEncOffset,
            mDiffPosOffset,
            mPrimGridConfig
        );
        if (dump) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
            gDump.writeDevice("train_diff_input", "f16", diffBatchSize, mDiffInputDim, diffInput.data(), sizeof(precision_t));
            gDump.writeDevice("train_diff_target", "f32", diffBatchSize, mOutputDim, diffTarget.data(), sizeof(float));
            gDump.writeDevice("train_diff_output", "f16", diffBatchSize, mPaddedOutputDim, networkOutput.data(), sizeof(precision_t));
            gDump.writeDevice("train_diff_dL_doutput", "f16", diffBatchSize, mPaddedOutputDim, dL_dnetworkOutput.data(), sizeof(precision_t));
            gDump.writeDevice("train_diff_dL_dinput", "f16", diffBatchSize, mDiffInputDim, dL_dnetworkInput.data(), sizeof(precision_t));
            // The primary grid gradient so far: the diffuse batch alone.
            gDump.writeDevice("train_prim_grid_grad_diff", "f16", mpPrimGridsGradient->size(), 1, mpPrimGridsGradient->data(), sizeof(precision_t));
        }
    }
    if (specBatchSize > 0) {
        tcnn::GPUMatrix<precision_t> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
        tcnn::GPUMatrix<float> specTarget(specPtrs.outputPtr, mOutputDim, specPtrs.size);
        ncr::clearTrainingInputPadding(mTrainStream, specInput, specPtrs.size);
        tcnn::GPUMatrix<precision_t> networkOutput{mPaddedOutputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkOutput{mPaddedOutputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<precision_t> dL_dnetworkInput{mSpecInputDim, specBatchSize, mTrainStream};
        tcnn::GPUMatrix<float> loss{mPaddedOutputDim, specBatchSize, mTrainStream};
        // castMatrix(mTrainStream, specInput, networkInput);
        // moveGBuffer(mTrainStream, specGBuffer, networkInput, mSpecPosOffset);
        // Forward
        HashGrid::launchForward(
            mTrainStream,
            mpPrimGrids->data(),
            specInput.data(),
            specPtrs.size,
            mSpecInputDim,
            mSpecPrimEncOffset,
            mSpecPosOffset,
            mPrimGridConfig
        );
        HashGridInterp::launchForward(
            mTrainStream,
            mpClsGrids->data(),
            specInput.data(),
            specPtrs.size,
            mSpecInputDim,
            mSpecClsEncOffset,
            mSpecClsPosOffset,
            mSpecClsScaleOffset,
            mSpecClsWeightOffset,
            mClsGridConfig
        );
        auto ctx = mpSpecNet->forward(mTrainStream, specInput, &networkOutput);
        // Loss
        ncr::evaluateValidSamples(mTrainStream, lossScale, *mpLoss, networkOutput, specTarget, loss, dL_dnetworkOutput);
        // Backward
        mpSpecNet->backward(mTrainStream, *ctx, specInput, networkOutput, dL_dnetworkOutput, &dL_dnetworkInput);
        if (mTrainingDiagnostics) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
            diagnostic["spec"]["target"] = ncr::outputStatistics(specTarget.data(), specPtrs.size, mOutputDim, mOutputDim);
            diagnostic["spec"]["working_prediction_pre_update"] = ncr::outputStatistics(networkOutput.data(), specPtrs.size, mPaddedOutputDim, mOutputDim);
            diagnostic["spec"]["input_columns"] = ncr::inputColumnStatistics(specInput.data(), specPtrs.size, mSpecInputDim);
            diagnostic["spec"]["input_layout"] = {{"width",mSpecInputDim},{"primary_encoding",mSpecPrimEncOffset},
                {"cluster_encoding",mSpecClsEncOffset},{"position",mSpecPosOffset},{"cluster_position",mSpecClsPosOffset},
                {"cluster_scale",mSpecClsScaleOffset},{"cluster_weight",mSpecClsWeightOffset},{"clusters",NCR_NUM_CLUSTERS}};
            diagnostic["spec"]["input_values"] = ncr::gradientStatistics(specInput.data(), size_t(specPtrs.size)*mSpecInputDim, true, 1.f);
            diagnostic["spec"]["full_output_values"] = ncr::gradientStatistics(networkOutput.data(), size_t(specPtrs.size)*mPaddedOutputDim, true, 1.f);
            diagnostic["spec"]["loss_output_gradient"] = ncr::gradientStatistics(dL_dnetworkOutput.data(), size_t(specPtrs.size)*mPaddedOutputDim, true, lossScale);
            diagnostic["spec"]["network_input_gradient"] = ncr::gradientStatistics(dL_dnetworkInput.data(), size_t(specPtrs.size)*mSpecInputDim, true, lossScale);
        }
        HashGrid::launchBackward(
            mTrainStream,
            mpPrimGrids->data(),
            specInput.data(),
            dL_dnetworkInput.data(),
            mpPrimGridsGradient->data(),
            specPtrs.size,
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
            specPtrs.size,
            mSpecInputDim,
            mSpecClsEncOffset,
            mSpecClsPosOffset,
            mSpecClsScaleOffset,
            mSpecClsWeightOffset,
            mClsGridConfig
        );
        if (dump) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
            gDump.writeDevice("train_spec_input", "f16", specBatchSize, mSpecInputDim, specInput.data(), sizeof(precision_t));
            gDump.writeDevice("train_spec_target", "f32", specBatchSize, mOutputDim, specTarget.data(), sizeof(float));
            gDump.writeDevice("train_spec_output", "f16", specBatchSize, mPaddedOutputDim, networkOutput.data(), sizeof(precision_t));
            gDump.writeDevice("train_spec_dL_doutput", "f16", specBatchSize, mPaddedOutputDim, dL_dnetworkOutput.data(), sizeof(precision_t));
            gDump.writeDevice("train_spec_dL_dinput", "f16", specBatchSize, mSpecInputDim, dL_dnetworkInput.data(), sizeof(precision_t));
        }
    }
    if (dump) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
        gDump.writeDevice("train_diff_params_grad", "f16", mpDiffParamsGradient->size(), 1, mpDiffParamsGradient->data(), sizeof(precision_t));
        gDump.writeDevice("train_spec_params_grad", "f16", mpSpecParamsGradient->size(), 1, mpSpecParamsGradient->data(), sizeof(precision_t));
        gDump.writeDevice("train_prim_grid_grad", "f16", mpPrimGridsGradient->size(), 1, mpPrimGridsGradient->data(), sizeof(precision_t));
        gDump.writeDevice("train_cls_grid_grad", "f16", mpClsGridsGradient->size(), 1, mpClsGridsGradient->data(), sizeof(precision_t));
    }
    if (mTrainingDiagnostics) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
        diagnostic["diff_count"] = diffPtrs.size;
        diagnostic["spec_count"] = specPtrs.size;
        diagnostic["learning_rate"] = mLearningRate;
        diagnostic["ema_decay"] = mEmaDecay;
        diagnostic["gradient"]["diff_mlp"] = ncr::gradientStatistics(mpDiffParamsGradient->data(), mpDiffParamsGradient->size(), diffBatchSize>0, lossScale);
        diagnostic["gradient"]["spec_mlp"] = ncr::gradientStatistics(mpSpecParamsGradient->data(), mpSpecParamsGradient->size(), specBatchSize>0, lossScale);
        diagnostic["gradient"]["primary_grid"] = ncr::gradientStatistics(mpPrimGridsGradient->data(), mpPrimGridsGradient->size(), diffBatchSize>0 || specBatchSize>0, lossScale);
        diagnostic["gradient"]["cluster_grid"] = ncr::gradientStatistics(mpClsGridsGradient->data(), mpClsGridsGradient->size(), specBatchSize>0, lossScale);
        diagnostic["optimizer_steps_before"] = {mpDiffNetOpt->step(), mpSpecNetOpt->step(), mpPrimGridOpt->step(), mpClsGridOpt->step()};
        mLastTrainingDiagnostics = diagnostic.dump();
    }
    if (diffBatchSize > 0) {
        mpDiffNetOpt->step(mTrainStream, lossScale, mpDiffParamsFull->data(), mpDiffParams->data(), mpDiffParamsGradient->data());
    }
    if (specBatchSize > 0) {
        mpSpecNetOpt->step(mTrainStream, lossScale, mpSpecParamsFull->data(), mpSpecParams->data(), mpSpecParamsGradient->data());
        mpClsGridOpt->step(mTrainStream, lossScale, mpClsGridsFull->data(), mpClsGrids->data(), mpClsGridsGradient->data());
    }
    if (diffBatchSize > 0 || specBatchSize > 0) {
        mpPrimGridOpt->step(mTrainStream, lossScale, mpPrimGridsFull->data(), mpPrimGrids->data(), mpPrimGridsGradient->data());
    }
    if (dump) {
        saveState(gDump.dir + "/train_after.ckpt");
        gDump.trainDone = true;
        fmt::print("NRModel: wrote training step {} to {}\n", gDump.step, gDump.dir);
    }
}

// ============================================================================
// Checkpoint save / load.
//
// Layout ("NRCK" v2): header, then 8 raw blocks in fixed order:
//   4 x f32 master weights  (primary grid, cluster grid, DiffNet, SpecNet)
//   4 x f16 EMA inference copies, same order.
// The f16 working copies are rebuilt from the masters on load. Adam moments are
// not stored; resuming training restarts the moment estimates. The element
// counts in the header are validated on load so a checkpoint produced with a
// different NCRConfig.h fails loudly instead of loading garbage.
// ============================================================================

namespace {

struct CheckpointHeader {
    char magic[4];           // 'N','R','C','K'
    uint32_t version;        // 2
    uint32_t primGridElems;
    uint32_t clsGridElems;
    uint32_t diffParamElems;
    uint32_t specParamElems;
};

void writeDeviceBlock(std::ofstream& f, const void* devPtr, size_t bytes) {
    std::vector<char> host(bytes);
    if (bytes > 0) CUDA_CHECK_THROW(cudaMemcpy(host.data(), devPtr, bytes, cudaMemcpyDeviceToHost));
    f.write(host.data(), (std::streamsize)bytes);
}

void readDeviceBlock(std::ifstream& f, void* devPtr, size_t bytes) {
    std::vector<char> host(bytes);
    f.read(host.data(), (std::streamsize)bytes);
    CHECK_THROW(f.good());
    if (bytes > 0) CUDA_CHECK_THROW(cudaMemcpy(devPtr, host.data(), bytes, cudaMemcpyHostToDevice));
}

void castFullToHalf(cudaStream_t stream, const float* src, precision_t* dst, uint32_t count) {
    if (count == 0) return;
    tcnn::parallel_for_gpu(stream, count, [src, dst] __device__ (size_t i) {
        dst[i] = (precision_t)src[i];
    });
}

} // namespace

void NRModel::saveState(const std::string& path) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));

    CheckpointHeader h = {};
    h.magic[0] = 'N'; h.magic[1] = 'R'; h.magic[2] = 'C'; h.magic[3] = 'K';
    h.version = 2;
    h.primGridElems = (uint32_t)mpPrimGrids->size();
    h.clsGridElems = (uint32_t)mpClsGrids->size();
    h.diffParamElems = (uint32_t)mpDiffParams->size();
    h.specParamElems = (uint32_t)mpSpecParams->size();

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("NRModel::saveState: cannot open '" + path + "' for writing");
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));

    writeDeviceBlock(f, mpPrimGridsFull->data(), mpPrimGridsFull->bytes());
    writeDeviceBlock(f, mpClsGridsFull->data(), mpClsGridsFull->bytes());
    writeDeviceBlock(f, mpDiffParamsFull->data(), mpDiffParamsFull->bytes());
    writeDeviceBlock(f, mpSpecParamsFull->data(), mpSpecParamsFull->bytes());

    writeDeviceBlock(f, mpPrimGridsInference, mpPrimGrids->bytes());
    writeDeviceBlock(f, mpClsGridsInference, mpClsGrids->bytes());
    writeDeviceBlock(f, mpDiffParamsInference, mpDiffParams->bytes());
    writeDeviceBlock(f, mpSpecParamsInference, mpSpecParams->bytes());

    CHECK_THROW(f.good());
    fmt::print("NRModel::saveState: wrote {} bytes to {}\n", (size_t)f.tellp(), path);
}

void NRModel::saveWorkingCopies(const std::string& path) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));

    CheckpointHeader h = {};
    h.magic[0] = 'N'; h.magic[1] = 'R'; h.magic[2] = 'W'; h.magic[3] = 'K';
    h.version = 1;
    h.primGridElems = (uint32_t)mpPrimGrids->size();
    h.clsGridElems = (uint32_t)mpClsGrids->size();
    h.diffParamElems = (uint32_t)mpDiffParams->size();
    h.specParamElems = (uint32_t)mpSpecParams->size();

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("NRModel::saveWorkingCopies: cannot open '" + path + "' for writing");
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    writeDeviceBlock(f, mpPrimGrids->data(), mpPrimGrids->bytes());
    writeDeviceBlock(f, mpClsGrids->data(), mpClsGrids->bytes());
    writeDeviceBlock(f, mpDiffParams->data(), mpDiffParams->bytes());
    writeDeviceBlock(f, mpSpecParams->data(), mpSpecParams->bytes());
    CHECK_THROW(f.good());
}

void NRModel::loadState(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("NRModel::loadState: cannot open '" + path + "' for reading");

    CheckpointHeader h = {};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    CHECK_THROW(f.good());
    if (!(h.magic[0] == 'N' && h.magic[1] == 'R' && h.magic[2] == 'C' && h.magic[3] == 'K')) {
        throw std::runtime_error("NRModel::loadState: '" + path + "' is not an NRCK checkpoint");
    }
    if (h.version != 2) {
        throw std::runtime_error(fmt::format("NRModel::loadState: unsupported checkpoint version {}", h.version));
    }
    auto checkCount = [&](const char* name, uint32_t fileCount, size_t modelCount) {
        if (fileCount != modelCount) {
            throw std::runtime_error(fmt::format(
                "NRModel::loadState: {} element count mismatch (file {}, model {}); NCRConfig.h differs from the one the checkpoint was trained with",
                name, fileCount, modelCount));
        }
    };
    checkCount("primary grid", h.primGridElems, mpPrimGrids->size());
    checkCount("cluster grid", h.clsGridElems, mpClsGrids->size());
    checkCount("DiffNet", h.diffParamElems, mpDiffParams->size());
    checkCount("SpecNet", h.specParamElems, mpSpecParams->size());

    CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));

    // f32 masters, then rebuild the f16 working copies the optimizers operate on.
    readDeviceBlock(f, mpPrimGridsFull->data(), mpPrimGridsFull->bytes());
    readDeviceBlock(f, mpClsGridsFull->data(), mpClsGridsFull->bytes());
    readDeviceBlock(f, mpDiffParamsFull->data(), mpDiffParamsFull->bytes());
    readDeviceBlock(f, mpSpecParamsFull->data(), mpSpecParamsFull->bytes());
    castFullToHalf(mInferenceStream, mpPrimGridsFull->data(), mpPrimGrids->data(), (uint32_t)mpPrimGrids->size());
    castFullToHalf(mInferenceStream, mpClsGridsFull->data(), mpClsGrids->data(), (uint32_t)mpClsGrids->size());
    castFullToHalf(mInferenceStream, mpDiffParamsFull->data(), mpDiffParams->data(), (uint32_t)mpDiffParams->size());
    castFullToHalf(mInferenceStream, mpSpecParamsFull->data(), mpSpecParams->data(), (uint32_t)mpSpecParams->size());

    // EMA inference copies (what inference() actually reads).
    readDeviceBlock(f, mpPrimGridsInference, mpPrimGrids->bytes());
    readDeviceBlock(f, mpClsGridsInference, mpClsGrids->bytes());
    readDeviceBlock(f, mpDiffParamsInference, mpDiffParams->bytes());
    readDeviceBlock(f, mpSpecParamsInference, mpSpecParams->bytes());

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    fmt::print("NRModel::loadState: loaded {}\n", path);
}

// Probe on scratch I/O. inference(false) selects working MLP weights; grids must
// select the corresponding snapshot too. Neither path performs backward/step.
std::string NRModel::probeDiagnostics(ModelIOPtrs specPtrs)
{
    if (!specPtrs.size) return "{}";
    const uint32_t count = padUp(specPtrs.size, kModelBatchAlign);
    tcnn::GPUMatrix<precision_t> input(mSpecInputDim, count, mInferenceStream);
    tcnn::GPUMatrix<float> output(mOutputDim, count, mInferenceStream);
    tcnn::json result;
    for (bool ema : {false, true}) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(input.data(), specPtrs.inputPtr, size_t(specPtrs.size)*mSpecInputDim*sizeof(precision_t), cudaMemcpyDeviceToDevice, mInferenceStream));
        ncr::clearTrainingInputPadding(mInferenceStream, input, specPtrs.size);
        HashGrid::launchForward(mInferenceStream, ema?mpPrimGridsInference:mpPrimGrids->data(), input.data(), specPtrs.size, mSpecInputDim, mSpecPrimEncOffset, mSpecPosOffset, mPrimGridConfig);
        HashGridInterp::launchForward(mInferenceStream, ema?mpClsGridsInference:mpClsGrids->data(), input.data(), specPtrs.size, mSpecInputDim, mSpecClsEncOffset, mSpecClsPosOffset, mSpecClsScaleOffset, mSpecClsWeightOffset, mClsGridConfig);
        mpSpecNet->inference(mInferenceStream, input, output, ema);
        CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));
        result[ema?"ema":"working"] = ncr::outputStatistics(output.data(), specPtrs.size, mOutputDim, mOutputDim);
    }
    return result.dump();
}

#include "Model.h"
#include "HashGrid.h"
#include "HashGridInterp.h"
// #include "NeuralModel.h"
// #include "NeuralConeModel.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
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

} // namespace

NRModel::NRModel() {
    CUDA_CHECK_THROW(cudaStreamCreate(&mTrainStream));
    CUDA_CHECK_THROW(cudaStreamCreate(&mInferenceStream));

    mDiffGBufferDim = 3 * 4 + 1;
    mSpecGBufferDim = 3 * 5 + 1 + NCR_NUM_CLUSTERS * (3 + 1 + 1);
    mDiffInputDim = padUp(NCR_N_FEATURES_PER_LEVEL * NCR_N_LEVELS + mDiffGBufferDim, 16);
    mSpecInputDim = padUp(NCR_N_FEATURES_PER_LEVEL * NCR_N_LEVELS + NCR_N_FEATURES_PER_LEVEL_SPEC * NCR_NUM_CLUSTERS + mSpecGBufferDim, 16);
    mOutputDim = NCR_OUTPUT_DIM;

    int seed = 1337;
    std::seed_seq seq{seed};
    std::vector<uint32_t> seeds(2);
    seq.generate(std::begin(seeds), std::end(seeds));
    mpRng = std::make_shared<tcnn::pcg32>(seeds.front());

    mPrimGridConfig = {
        NCR_N_LEVELS,
        NCR_N_FEATURES_PER_LEVEL,
        NCR_LOG2_HASHMAP_SIZE,
        NCR_BASE_RESOLUTION,
        NCR_PER_LEVEL_SCALE
    };
    mClsGridConfig = {
        NCR_NUM_CLUSTERS,
        NCR_N_LEVELS_SPEC,
        NCR_N_FEATURES_PER_LEVEL_SPEC,
        NCR_LOG2_HASHMAP_SIZE_SPEC,
        NCR_BASE_RESOLUTION_SPEC,
        NCR_PER_LEVEL_SCALE_SPEC,
        NCR_INTERP_RATIO
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
        {"n_neurons", NCR_DIFF_N_NEURONS},
        {"n_hidden_layers", NCR_DIFF_N_HIDDEN_LAYERS}
    };

    const tcnn::json specNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mSpecInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", NCR_SPEC_N_NEURONS},
        {"n_hidden_layers", NCR_SPEC_N_HIDDEN_LAYERS}
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
        tcnn::GPUMatrix<float> diffTarget(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
        // Forward writes the padded 16-wide tile. The loss supervises mOutputDim
        // and zeros the gradient on the padded channels.
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
        tcnn::GPUMatrix<float> specTarget(specPtrs.outputPtr, mOutputDim, specBatchSize);
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

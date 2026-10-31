#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "ModelTypes.h"
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


class NRModel
{
public:
    NRModel();
    ~NRModel();

    void loadState(const std::string& path);
    void saveState(const std::string& path);

    void setOnline(bool online);

    void inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);
    void train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);

    cudaStream_t stream(bool isTraining) const { return isTraining ? mTrainStream : mInferenceStream; }

    // --- Debug: hash collision stats ---
    // Enable/disable per-frame collision counting. When enabled, the model
    // allocates marker buffers and resets them before each forward pass.
    void setCollisionStatsEnabled(bool enabled);
    bool isCollisionStatsEnabled() const { return mCollisionStatsEnabled; }

    // Zero counters and marker buffers on both streams. Call at frame start.
    void resetCollisionStats();

    // Read back counters from the last forward pass. Returns {uniqueSlots, totalQueries}
    // for the primary (diffuse) and cluster (specular) grids respectively.
    // Must be called after inference/train and a stream sync.
    struct CollisionStats {
        uint32_t primUniqueSlots;
        uint32_t primTotalQueries;
        uint32_t clsUniqueSlots;
        uint32_t clsTotalQueries;
    };
    CollisionStats getCollisionStats() const;

    // --- Debug: specular level usage histogram ---
    // Returns pointer to nLevels*nLevels uint32_t histogram (upperLevel * nLevels + lowerLevel).
    // Buffer owned by NRModel; valid until next resetCollisionStats / setCollisionStatsEnabled.
    // Returns nullptr if collision stats are disabled (histogram is allocated alongside).
    void getLevelHistogram(std::vector<uint32_t>& outHostHistogram) const;
    uint32_t getNumLevels() const { return mClsGridConfig.nLevels; }

    // Returns the per-level voxel scales for the cluster (specular) grid.
    // gridScales[i] = interpRatio / (baseResolution * perLevelScale^i).
    // Used by the ConeTrace shader to select the grid level per cluster.
    std::vector<float> getClusterGridScales() const;

    // --- Debug: per-optimizer freeze switches ---
    // When true, the corresponding optimizer's step() is skipped in train(),
    // but forward/backward still run and gradients are still computed (then
    // discarded). Useful for isolating which optimizer's updates contribute
    // to instability. Does NOT affect inference.
    void setFreezePrimGridOpt(bool v) { mFreezePrimGridOpt = v; }
    void setFreezeClsGridOpt(bool v)  { mFreezeClsGridOpt  = v; }
    void setFreezeDiffNetOpt(bool v)  { mFreezeDiffNetOpt  = v; }
    void setFreezeSpecNetOpt(bool v)  { mFreezeSpecNetOpt  = v; }
    bool getFreezePrimGridOpt() const { return mFreezePrimGridOpt; }
    bool getFreezeClsGridOpt() const  { return mFreezeClsGridOpt; }
    bool getFreezeDiffNetOpt() const  { return mFreezeDiffNetOpt; }
    bool getFreezeSpecNetOpt() const  { return mFreezeSpecNetOpt; }

    // --- Runtime hyperparameter tuning ---
    // Get/set Adam learning rate for each optimizer (delegates to nested Adam
    // inside the EMA wrapper). Takes effect immediately on the next step().
    float getPrimGridLr() const;
    float getClsGridLr()  const;
    float getDiffNetLr()  const;
    float getSpecNetLr()  const;
    void setPrimGridLr(float v);
    void setClsGridLr(float v);
    void setDiffNetLr(float v);
    void setSpecNetLr(float v);

    // Get/set EMA decay for each optimizer. Takes effect on the next step().
    float getPrimGridEmaDecay() const;
    float getClsGridEmaDecay()  const;
    float getDiffNetEmaDecay()  const;
    float getSpecNetEmaDecay()  const;
    void setPrimGridEmaDecay(float v);
    void setClsGridEmaDecay(float v);
    void setDiffNetEmaDecay(float v);
    void setSpecNetEmaDecay(float v);

    // Get/set interpRatio for the cluster grid. Requires rebuilding grid
    // scales and re-binding to shaders.
    float getInterpRatio() const { return mClsGridConfig.interpRatio; }
    void setInterpRatio(float v);

    // Current optimizer step count (from nested Adam).
    uint32_t getOptimizerStep() const;

private:
    void createOnlineModels();
    void createOfflineModelsIfNeeded();
    
    cudaStream_t mInferenceStream;
    cudaStream_t mTrainStream;
    bool mOnline = false;

    std::shared_ptr<tcnn::pcg32> mpRng;

    uint32_t pixelCount = 1u << 20;
    uint32_t numClusters = 4;

    // Hyper parameters
    uint32_t nLevels = 6;
    uint32_t nFeaturesPerLevel = 2;
    uint32_t log2HashMapSize = 19;
    uint32_t baseResolution = 32;
    float perLevelScale = 2.0f;

    uint32_t nInterpLevels = 6;
    uint32_t nInterpFeaturesPerLevel = 4;
    uint32_t log2InterpHashMapSize = 19;
    uint32_t baseInterpResolution = 16;
    float perLevelInterpScale = 2.0f;
    float interpRatio = 1.0f;

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
    uint32_t mOutputDim;

    uint32_t mDiffEncOffset;
    uint32_t mDiffPosOffset;
    uint32_t mSpecPrimEncOffset;
    uint32_t mSpecClsEncOffset;
    uint32_t mSpecPosOffset;
    uint32_t mSpecClsPosOffset;
    uint32_t mSpecClsScaleOffset;
    uint32_t mSpecClsWeightOffset;

    // --- Debug: hash collision stats ---
    bool mCollisionStatsEnabled = false;
    // Device buffers for collision counting (host-owned, raw cudaMalloc).
    uint32_t* mpPrimCollisionCounters = nullptr;   // 2 uint32_t: [uniqueSlots, totalQueries]
    uint32_t* mpPrimCollisionMarker   = nullptr;   // (1<<log2HashMapSize) * nLevels uint32_t
    uint32_t* mpClsCollisionCounters  = nullptr;
    uint32_t* mpClsCollisionMarker    = nullptr;
    uint32_t  mPrimMarkerSize = 0;
    uint32_t  mClsMarkerSize  = 0;
    // Device buffer for specular level-usage histogram (nLevels*nLevels uint32_t).
    uint32_t* mpClsLevelHistogram = nullptr;
    // Host mirrors for getCollisionStats()
    mutable CollisionStats mLastCollisionStats{};
    mutable std::vector<uint32_t> mLastLevelHistogram{};

    // --- Debug: per-optimizer freeze switches ---
    bool mFreezePrimGridOpt = false;
    bool mFreezeClsGridOpt  = false;
    bool mFreezeDiffNetOpt  = false;
    bool mFreezeSpecNetOpt  = false;
};

/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Utils/CudaUtils.h"
#include "Utils/Algorithm/PrefixSum.h"
#include "Model/Model.h"

#include <filesystem>
#include <memory>
#include <cstdlib>

using namespace Falcor;

struct RayBatchBuffer
{
    ref<Device> pDevice;
    ShaderVar var;
    uint32_t size = 0;
    uint32_t diffSize = 0;
    uint32_t specSize = 0;
    uint32_t numClusters = 0;
    std::string name;
    bool needAll = true;

    /// G-buffer
    ref<Buffer> pos;
    ref<Buffer> dir;
    ref<Buffer> normal;
    ref<Buffer> albedo;
    ref<Buffer> roughness;
    ref<Buffer> vbuffer;
    ref<Buffer> color;
    ref<Buffer> emission;           // NEE color
    /// Diffuse buffer
    ref<Buffer> diffActive;         // Compaction Input
    ref<Buffer> diffIndex;          // Compaction Output
    ref<Buffer> diffInput;          // Diffuse input for neural network
    ref<Buffer> diffColor;
    /// Specular buffer
    ref<Buffer> specVBuffer;
    ref<Buffer> specPixel;
    ref<Buffer> specPos;
    ref<Buffer> specDir;
    ref<Buffer> specRoughness;
    ref<Buffer> specActive;       // Compaction Input
    ref<Buffer> specIndex;        // Compaction Output
    ref<Buffer> specInput;        // Specular input for neural network
    ref<Buffer> specColor;

    RayBatchBuffer(ref<Device> pDevice, ShaderVar& var, uint32_t size, uint32_t numClusters, const std::string& name, bool needAll = true)
        : pDevice(pDevice), var(var), size(size), numClusters(numClusters), name(name), needAll(needAll)
    {
        initBuffers();
    }

    void resize(uint32_t newSize, uint32_t newNumClusters)
    {
        uint32_t oldSize = size;
        uint32_t oldNumClusters = numClusters;

        size = newSize;
        numClusters = newNumClusters;

        if (size > oldSize || (numClusters * size) > (oldNumClusters * oldSize))
        {
            // Recreate buffers if the new size exceeds the old size
            initBuffers();
        }
    }

    void initBuffers()
    {
        // Create buffers
        if (needAll)
        {
            pos = pDevice->createStructuredBuffer(
                var["allPos"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            dir = pDevice->createStructuredBuffer(
                var["allDir"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            normal = pDevice->createStructuredBuffer(
                var["allNormal"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            albedo = pDevice->createStructuredBuffer(
                var["allAlbedo"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            roughness = pDevice->createStructuredBuffer(
                var["allRoughness"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            vbuffer = pDevice->createStructuredBuffer(
                var["allVBuffer"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            color = pDevice->createStructuredBuffer(
                var["allColor"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            emission = pDevice->createStructuredBuffer(
                var["allEmission"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }

        // Create diffuse pixel buffers
        {
            diffActive = pDevice->createStructuredBuffer(
                var["diffActive"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffIndex = pDevice->createStructuredBuffer(
                var["diffIndex"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffInput = pDevice->createStructuredBuffer(
                var["diffInput"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffColor = pDevice->createStructuredBuffer(
                var["diffColor"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }

        // Create specular pixel buffers
        {
            specVBuffer = pDevice->createStructuredBuffer(
                var["specVBuffer"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specPixel = pDevice->createStructuredBuffer(
                var["specPixel"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specPos = pDevice->createStructuredBuffer(
                var["specPos"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specDir = pDevice->createStructuredBuffer(
                var["specDir"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specRoughness = pDevice->createStructuredBuffer(
                var["specRoughness"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specActive = pDevice->createStructuredBuffer(
                var["specActive"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specIndex = pDevice->createStructuredBuffer(
                var["specIndex"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specInput = pDevice->createStructuredBuffer(
                var["specInput"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specColor = pDevice->createStructuredBuffer(
                var["specColor"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }
    }

    ModelIOPtrs getDiffPtrs()
    {
        return {
            (precision_t*) diffInput->getCudaMemory()->getMappedData(),
            (float*) diffColor->getCudaMemory()->getMappedData(),
            diffSize
        };
    }

    ModelIOPtrs getSpecPtrs()
    {
        return {
            (precision_t*) specInput->getCudaMemory()->getMappedData(),
            (float*) specColor->getCudaMemory()->getMappedData(),
            specSize
        };
    }
};

enum class RenderMode
{
    Idle,
    Render,
    Train,
    OnlineTrain
};

FALCOR_ENUM_INFO(
    RenderMode,
    {
        {RenderMode::Idle, "Idle"},
        {RenderMode::Render, "Render"},
        {RenderMode::Train, "Train"},
        {RenderMode::OnlineTrain, "Online Train"}
    }
);
FALCOR_ENUM_REGISTER(RenderMode);

enum class ConeTraceReuseMode
{
    TemporalMerge,
    TemporalToHistory,
    ScreenSpaceDenoise,
};

FALCOR_ENUM_INFO(
    ConeTraceReuseMode,
    {
        {ConeTraceReuseMode::TemporalMerge, "Temporal Merge"},
        {ConeTraceReuseMode::TemporalToHistory, "Temporal To History"},
        {ConeTraceReuseMode::ScreenSpaceDenoise, "Screen Space Denoise"},
    }
);
FALCOR_ENUM_REGISTER(ConeTraceReuseMode);


class NeuralRadiosity : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralRadiosity, "NeuralRadiosity", "Neural Radiosity Render Pass. (S. Hadadan, ToG 2021)");

    static ref<NeuralRadiosity> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<NeuralRadiosity>(pDevice, props);
    }

    // Returns true if env var NCR_DEBUG_LOG=1 is set (used by NCR_LOG macro).
    static bool debugLogEnvEnabled()
    {
        const char* v = std::getenv("NCR_DEBUG_LOG");
        return v != nullptr && v[0] == '1';
    }

    NeuralRadiosity(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual void onSceneUpdates(RenderContext* pRenderContext, IScene::UpdateFlags sceneUpdates) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void render(RenderContext* pRenderContext, const RenderData& renderData);
    void train(RenderContext* pRenderContext);
    // Screen rendering passes
    void firstSmoothPass(RenderContext* pRenderContext, const RenderData& renderData);
    void compactPass(RenderContext* pRenderContext, const RenderData& renderData);
    void resolvePass(RenderContext* pRenderContext, const RenderData& renderData);
    void postFilterPass(RenderContext* pRenderContext, const RenderData& renderData);
    // Ray-batch rendering passes
    void randomSmooth(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void sampleRHS(RenderContext* pRenderContext);
    void resolveRHS(RenderContext* pRenderContext);
    void compactBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void resolveBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    // Common rendering passes
    void coneTrace(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void modelInferenceCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void modelTrainCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);

    void updateFrameDim(const uint2 frameDim);
    void updatePrograms(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    bool prepareLighting(RenderContext* pRenderContext);
    void preparePrefiltEnvCubemap(RenderContext* pRenderContext);
    void updateRenderTemporalHistory(RenderContext* pRenderContext);
    void clearRenderTemporalHistory(RenderContext* pRenderContext);
    void bindScreenData(ShaderVar& var, const RenderData& renderData, const std::string &name);
    void bindRayBatchData(ShaderVar& var, std::shared_ptr<RayBatchBuffer> pRayBatch, const std::string &name);

    /// Debug image dump: runs DebugDumpPass to copy per-spec-pixel debug
    /// buffers into textures, then saves all debug images (spec level,
    /// material ID, normal, albedo, tanLobe, cluster mean) to disk under
    /// the given directory. Returns the directory path used.
    void dumpDebugImages(RenderContext* pRenderContext, const RenderData& renderData);
    /// Create / resize the debug dump textures and buffers to match frameDim.
    void prepareDebugDumpResources();
    DefineList getShaderDefines(const RenderData& renderData) const;
    bool loadTrainCamerasFromFile(const std::filesystem::path& path);
    void setCamera(uint32_t cameraIdx);
    void setConeParameters(bool train);
    uint32_t getAdaptiveRHSStage(uint32_t completedSteps) const;
    void updateAdaptiveRHSState(uint32_t completedSteps, bool force = false);

    // Internal state & parameters
    uint32_t mFrameCount = 0;
    uint2 mFrameDim = {};
    /// Shader variables changed flag
    bool mVarsChanged = false;
    /// Algorithm parameters
    static const uint32_t NUM_SPEC_RAYS_TRAIN = 128;
    static const uint32_t NUM_SPEC_RAYS_RENDER = 32;
    static const uint32_t NUM_KMEANS_ITERS_TRAIN = 10;
    static const uint32_t NUM_KMEANS_ITERS_RENDER = 3;
    uint32_t mNumSpecRays = NUM_SPEC_RAYS_RENDER;
    uint32_t mNumKMeansIters = NUM_KMEANS_ITERS_RENDER;
    uint32_t mNumClusters = 4;
    /// Russian Roulette probability for random smooth.
    float mRSRRProb = 0.4f;
    /// Enable alpha test.
    bool mUseAlphaTest = true;
    /// Use environment lighting.
    bool mUseEnvLight = true;
    /// User next event estimation.
    bool mUseNEE = true;
    /// Adjust shading normals.
    bool mAdjustShadingNormals = true;
    /// Specular roughness threshold
    float mSpecularRoughnessThreshold = 0.5f;
    /// Enable temporal reuse for render-batch specular cone tracing.
    bool mEnableRenderSpecTemporalReuse = true;
    /// Strategy for integrating the current frame's depth samples with temporal specular clusters.
    ConeTraceReuseMode mConeTraceReuseMode = ConeTraceReuseMode::TemporalToHistory;
    /// Post filter switch
    bool mEnablePostFilter = true;
    /// FXAA switch for post filter.
    bool mEnableFXAA = true;
    /// Debug switches: render only diffuse or specular component (isolate flickering source).
    bool mDebugRenderDiffuseOnly = false;
    bool mDebugRenderSpecularOnly = false;
    /// Debug switches: freeze training updates for one network (still inferred, params not updated).
    bool mDisableDiffUpdate = false;
    bool mDisableSpecUpdate = false;
    /// Debug switches: freeze individual optimizer step() (forward/backward still run).
    bool mFreezePrimGridOpt = false;
    bool mFreezeClsGridOpt  = false;
    bool mFreezeDiffNetOpt  = false;
    bool mFreezeSpecNetOpt  = false;
    /// Training parameter UI mirrors (synced to NRModel on change).
    float mParamPrimGridLr = 5e-3f;
    float mParamClsGridLr  = 5e-3f;
    float mParamDiffNetLr  = 5e-3f;
    float mParamSpecNetLr  = 5e-3f;
    float mParamPrimGridEmaDecay = 0.95f;
    float mParamClsGridEmaDecay  = 0.95f;
    float mParamDiffNetEmaDecay  = 0.95f;
    float mParamSpecNetEmaDecay  = 0.95f;
    float mParamInterpRatio = 1.0f;
    /// Verbose scene-related parameter logging (AABB / envMap / emissive / camera / batches).
    bool mEnableDebugLog = false;
    /// Debug: count hash grid collisions per frame (unique slots vs total queries).
    bool mEnableHashCollisionStats = false;
    /// Debug: visualize per-spec-pixel grid level as color (red=coarse, blue=fine).
    bool mDebugShowSpecLevel = false;
    /// Debug: one-shot flag to dump all debug images to disk on next frame.
    bool mDebugDumpRequested = false;
    /// Force cull mode for all geometry, otherwise set it based on the scene.
    bool mForceCullMode = false;
    /// Cull mode to use for when mForceCullMode is true.
    RasterizerState::CullMode mCullMode = RasterizerState::CullMode::Back;
    /// UI variables
    RenderPassHelpers::IOSize mOutputSizeSelection = RenderPassHelpers::IOSize::Default;
    uint2 mFixedOutputSize = {512, 512};

    // Resources
    std::shared_ptr<RayBatchBuffer> mpRenderBatch;
    std::shared_ptr<RayBatchBuffer> mpTrainLHSBatch;
    std::shared_ptr<RayBatchBuffer> mpTrainRHSBatch;
    ref<Buffer> mpRenderSpecTemporalHistory;
    ref<Texture> mpRenderTempColor;
    ref<Texture> mpPostFilterTempColor;
    ref<Texture> mpPrefiltEnvCubemap;
    /// Debug: per-spec-pixel dominant grid level (one float per screen pixel).
    ref<Buffer> mpSpecLevelBuffer;
    /// Debug dump: per-spec-pixel buffers written by ConeTrace when DEBUG_DUMP is on.
    ref<Buffer> mpSpecTanLobeBuffer;        // float per spec pixel
    ref<Buffer> mpSpecClusterMeanBuffer;    // float4 per spec pixel (4 cluster means)
    ref<Buffer> mpSpecClusterScaleBuffer;   // float4 per spec pixel (4 cluster scales)
    ref<Buffer> mpSpecClusterWeightBuffer;  // float4 per spec pixel (4 cluster weights)
    ref<Buffer> mpSpecClusterPosXBuffer;    // float4 per spec pixel (4 cluster pos.x)
    ref<Buffer> mpSpecClusterPosYBuffer;    // float4 per spec pixel (4 cluster pos.y)
    ref<Buffer> mpSpecClusterPosZBuffer;    // float4 per spec pixel (4 cluster pos.z)
    ref<Buffer> mpSpecMaterialDataBuffer;   // float2 per spec pixel (materialID, roughness)
    /// Debug dump: output textures for CPU readback (DebugDumpPass writes these).
    ref<Texture> mpDumpSpecLevelTex;        // RGBA32Float
    ref<Texture> mpDumpTanLobeTex;          // RGBA32Float
    ref<Texture> mpDumpClusterMeanTex;      // RGBA32Float
    ref<Texture> mpDumpClusterScaleTex;     // RGBA32Float
    ref<Texture> mpDumpClusterWeightTex;    // RGBA32Float
    ref<Texture> mpDumpClusterPosXTex;      // RGBA32Float
    ref<Texture> mpDumpClusterPosYTex;      // RGBA32Float
    ref<Texture> mpDumpClusterPosZTex;      // RGBA32Float
    ref<Texture> mpDumpMaterialDataTex;     // RGBA32Float (r=materialID, g=roughness)
    /// Debug dump compute pass.
    ref<ComputePass> mpDebugDumpPass;
    ref<Sampler> mpPrefiltEnvSampler;
    bool mRenderSpecTemporalHistoryValid = false;
    bool mHasRenderCameraViewProj = false;
    float4x4 mRenderCameraViewProj = float4x4();
    bool mPrefiltEnvDirty = true;

    // Scene & Compute Passes
    ref<Scene> mpScene;
    ref<ComputePass> mpFirstSmoothPass;
    ref<ComputePass> mpCompactPass;
    ref<ComputePass> mpResolvePass;
    ref<ComputePass> mpPostFilterPass;
    ref<ComputePass> mpPostFXAAPass;
    ref<ComputePass> mpRandomSmooth;
    ref<ComputePass> mpSampleRHS;
    ref<ComputePass> mpResolveRHS;
    ref<ComputePass> mpCompactBatch;
    ref<ComputePass> mpResolveBatch;
    ref<ComputePass> mpConeTrace;
    ref<ComputePass> mpPrefiltEnvInitPass;
    ref<ComputePass> mpPrefiltEnvDownsamplePass;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<PrefixSum> mpPrefixSum;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;    ///< Emissive light sampler or nullptr if not used.

    // CUDA
    ref<cuda_utils::CudaDevice> mpCudaDevice;
    std::unique_ptr<NRModel> mpNRModel;

    // Train mode parameters
    RenderMode mRenderMode = RenderMode::Render;
    /// LHS and RHS
    uint32_t mBatchSizeInit = 1u << 15;
    uint32_t mNumRHSInit = 8u;
    uint32_t mBatchSize = mBatchSizeInit;
    uint32_t mNumRHS = mNumRHSInit;
    /// Cameras used for training.
    std::vector<ref<Camera>> mTrainCameras;
    std::filesystem::path mTrainCameraPath;
    /// Total training steps.
    uint32_t mTotalTrainSteps = 20000;
    uint32_t mSaveCKPTInterval = 1000;
    bool mAdaptiveRHS = true;
    uint32_t mAdaptiveRHSStage = 0;
    std::string mLoadModelPath;
};

// Debug-log macro: emits only when NeuralRadiosity::mEnableDebugLog is true,
// OR the NCR_DEBUG_LOG environment variable is set to "1" (allows scripted
// tests to dump scene parameters without GUI interaction).
// Usage: NCR_LOG(pSelf, "fmt {}", arg);  pSelf must point to a NeuralRadiosity.
#define NCR_LOG(pSelf, ...) \
    do { \
        if ((pSelf) && ((pSelf)->mEnableDebugLog || ::NeuralRadiosity::debugLogEnvEnabled())) \
            ::Falcor::logInfo(__VA_ARGS__); \
    } while (0)

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
#include "NCRConfig.h"

#include <filesystem>
#include <memory>

using namespace Falcor;

struct RayBatchBuffer
{
    ref<Device> pDevice;
    ShaderVar var;
    uint32_t size = 0;
    /// Compacted counts on the CPU. Only valid after readCounts(), which needs the
    /// GPU to have finished the compaction (the CUDA interop sync guarantees that).
    uint32_t diffSize = 0;
    uint32_t specSize = 0;
    uint32_t numClusters = 0;
    std::string name;
    /// G-buffer (pos/dir/normal/albedo/roughness/vbuffer/color/emission).
    bool needAll = true;
    /// Compacted diffuse network I/O.
    bool needDiff = true;
    /// Compacted specular network I/O and specSrc (G-Buffer is gathered on demand).
    bool needSpec = true;
    /// Diffuse and specular share one network-input buffer and one output buffer.
    /// Specular rows start at 0 and diffuse rows follow the padded specular rows,
    /// so the diffuse offset changes every frame (see diffRowOffset()). Only the
    /// render batch uses this: diff + spec <= pixel count, so the pool needs
    /// 128 B per pixel instead of 64 + 128 B. The same buffer is bound as both
    /// RWStructuredBuffer<DiffNetInput> and <SpecNetInput>; Vulkan takes the
    /// stride from the shader, D3D12 would need a ByteAddressBuffer instead.
    bool sharedIO = false;
    uint32_t diffStride = 0;

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
    /// Render: y * width + x. Train: pre-compact raw sample index.
    ref<Buffer> specSrc;
    ref<Buffer> specActive;       // Compaction Input
    ref<Buffer> specIndex;        // Compaction Output
    ref<Buffer> specInput;        // Specular input for neural network
    ref<Buffer> specColor;
    /// Compacted counts: [0] diffuse, [1] specular. Written by PrefixSum on the GPU.
    ref<Buffer> counts;
    /// Host-visible copy of `counts`, filled by a copy recorded after compaction.
    ref<Buffer> countsReadback;

    RayBatchBuffer(
        ref<Device> pDevice,
        ShaderVar& var,
        uint32_t size,
        uint32_t numClusters,
        const std::string& name,
        bool needAll = true,
        bool needDiff = true,
        bool needSpec = true,
        bool sharedIO = false
    )
        : pDevice(pDevice)
        , var(var)
        , size(size)
        , numClusters(numClusters)
        , name(name)
        , needAll(needAll)
        , needDiff(needDiff)
        , needSpec(needSpec)
        , sharedIO(sharedIO && needDiff && needSpec)
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
        // Only the four CUDA I/O buffers need exportable memory. Everything else
        // stays on the Vulkan heap so it can be suballocated.
        const auto kBind = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        const auto kBindCuda = kBind | ResourceBindFlags::Shared;

        if (needAll)
        {
            pos = pDevice->createStructuredBuffer(var["allPos"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            dir = pDevice->createStructuredBuffer(var["allDir"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            normal = pDevice->createStructuredBuffer(var["allNormal"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            albedo = pDevice->createStructuredBuffer(var["allAlbedo"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            roughness = pDevice->createStructuredBuffer(var["allRoughness"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            vbuffer = pDevice->createStructuredBuffer(var["allVBuffer"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            color = pDevice->createStructuredBuffer(var["allColor"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            emission = pDevice->createStructuredBuffer(var["allEmission"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
        }

        if (needDiff)
        {
            diffActive = pDevice->createStructuredBuffer(var["diffActive"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            diffIndex = pDevice->createStructuredBuffer(var["diffIndex"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            if (!sharedIO)
            {
                diffInput = pDevice->createStructuredBuffer(var["diffInput"], size, kBindCuda, MemoryType::DeviceLocal, nullptr, false);
                diffColor = pDevice->createStructuredBuffer(var["diffColor"], size, kBindCuda, MemoryType::DeviceLocal, nullptr, false);
            }
        }

        if (needSpec)
        {
            specSrc = pDevice->createStructuredBuffer(var["specSrc"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            specActive = pDevice->createStructuredBuffer(var["specActive"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            specIndex = pDevice->createStructuredBuffer(var["specIndex"], size, kBind, MemoryType::DeviceLocal, nullptr, false);
            if (sharedIO)
            {
                // Worst case with diff + spec <= size and a diffuse row no wider than a
                // specular row: size SpecNetInput rows, plus kModelBatchAlign rows for
                // each of the two paddings and one more for rounding the diffuse start.
                // The outputs are 16 B for both, so size + 2 * kModelBatchAlign rows.
                const uint32_t inputRows = size + 2 * kModelBatchAlign + 1;
                const uint32_t outputRows = size + 2 * kModelBatchAlign;
                diffStride = (uint32_t)var["diffInput"].getType()->unwrapArray()->asResourceType()->getStructType()->getSlangTypeLayout()->getStride();
                specInput = pDevice->createStructuredBuffer(var["specInput"], inputRows, kBindCuda, MemoryType::DeviceLocal, nullptr, false);
                FALCOR_CHECK(diffStride <= specInput->getStructSize(), "Shared network I/O assumes DiffNetInput is not wider than SpecNetInput.");
                specColor = pDevice->createStructuredBuffer(var["specColor"], outputRows, kBindCuda, MemoryType::DeviceLocal, nullptr, false);
                diffInput = specInput;
                diffColor = specColor;
            }
            else
            {
                specInput = pDevice->createStructuredBuffer(var["specInput"], size, kBindCuda, MemoryType::DeviceLocal, nullptr, false);
                specColor = pDevice->createStructuredBuffer(var["specColor"], size, kBindCuda, MemoryType::DeviceLocal, nullptr, false);
            }
        }

        counts = pDevice->createBuffer(2 * sizeof(uint32_t), kBind, MemoryType::DeviceLocal, nullptr);
        countsReadback = pDevice->createBuffer(2 * sizeof(uint32_t), ResourceBindFlags::None, MemoryType::ReadBack, nullptr);
    }

    /// Load diffSize/specSize from countsReadback. The caller must make sure the GPU
    /// has executed the copy into countsReadback, otherwise the values are stale.
    void readCounts()
    {
        const uint32_t* pCounts = static_cast<const uint32_t*>(countsReadback->map());
        diffSize = needDiff ? pCounts[0] : 0;
        specSize = needSpec ? pCounts[1] : 0;
        countsReadback->unmap();
    }

    /// First diffuse row in the shared output buffer (NetworkOutput rows). Needs
    /// specSize, i.e. readCounts(); shaders that run before that use `counts`.
    uint32_t diffOutputOffset() const { return sharedIO ? padUp(specSize, kModelBatchAlign) : 0; }

    /// First diffuse row in the shared input buffer (DiffNetInput rows): the end of
    /// the padded specular rows, rounded up to a whole diffuse row. CompactPass
    /// computes the same value on the GPU.
    uint32_t diffInputOffset() const
    {
        if (!sharedIO)
            return 0;
        const uint64_t specBytes = uint64_t(diffOutputOffset()) * specInput->getStructSize();
        // Not named diffStride: MSVC (C4458, /WX) rejects a local hiding the member.
        const uint32_t rowStride = diffInputStride();
        return uint32_t((specBytes + rowStride - 1) / rowStride);
    }

    /// Bytes per DiffNetInput row. With sharedIO the buffer was created for
    /// SpecNetInput, so this stride is taken from the shader reflection.
    uint32_t diffInputStride() const { return sharedIO ? diffStride : diffInput->getStructSize(); }

    ModelIOPtrs getDiffPtrs()
    {
        if (!needDiff || !diffInput || !diffColor)
            return { nullptr, nullptr, 0 };
        uint8_t* input = (uint8_t*) diffInput->getCudaMemory()->getMappedData();
        float* output = (float*) diffColor->getCudaMemory()->getMappedData();
        if (sharedIO)
        {
            input += uint64_t(diffInputOffset()) * diffInputStride();
            output += uint64_t(diffOutputOffset()) * (diffColor->getStructSize() / sizeof(float));
        }
        return { (precision_t*) input, output, diffSize };
    }

    ModelIOPtrs getSpecPtrs()
    {
        if (!needSpec || !specInput || !specColor)
            return { nullptr, nullptr, 0 };
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

class NeuralRadiosity : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralRadiosity, "NeuralRadiosity", "Neural Radiosity Render Pass. (S. Hadadan, ToG 2021)");

    static ref<NeuralRadiosity> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<NeuralRadiosity>(pDevice, props);
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

    /// Drop the render-path cluster history (same as the "Clear History" button). Scriptable.
    void clearHistory() { clearRenderTemporalHistory(mpDevice->getRenderContext()); }
    /// Current training-schedule heat (0 = cold/settled, 1 = hot/adapting).
    float getTrainHeat() const { return mTrainHeat; }

private:
    void render(RenderContext* pRenderContext, const RenderData& renderData);
    void train(RenderContext* pRenderContext, const RenderData& renderData);
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
    // Batch counts stay on the GPU until the next full sync.
    void computeBatchCounts(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, uint32_t elementCount);
    void stageBatchCounts(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void readBatchCountsAfterSync(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void resolveBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    // Common rendering passes
    void coneTrace(RenderContext* pRenderContext, const RenderData& renderData, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void coneTraceBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, const RenderData& renderData);
    void specPrimaryPass(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, const RenderData& renderData);
    void modelInferenceCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void modelTrainCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    // Spatiotemporal cluster reuse (render path only)
    void clusterReuse(RenderContext* pRenderContext, const RenderData& renderData);
    void clusterReprojectPass(RenderContext* pRenderContext, const RenderData& renderData);
    void coneTraceScreenPass(RenderContext* pRenderContext, const RenderData& renderData);
    void clusterSpatialReusePass(RenderContext* pRenderContext, const RenderData& renderData);
    void savePrevGBuffer(RenderContext* pRenderContext, const RenderData& renderData);
    bool useClusterReuse() const;

    void updateFrameDim(const uint2 frameDim);
    void updatePrograms(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    bool prepareLighting(RenderContext* pRenderContext);
    void preparePrefiltEnvCubemap(RenderContext* pRenderContext);
    void clearRenderTemporalHistory(RenderContext* pRenderContext);
    void resetTrainSchedule();
    void updateTrainSchedule();
    void bindScreenData(ShaderVar& var, const RenderData& renderData, const std::string &name);
    void bindPrefiltEnv(ShaderVar& var);
    void bindRayBatchData(ShaderVar& var, std::shared_ptr<RayBatchBuffer> pRayBatch, const std::string &name);
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
    /// Algorithm parameters. Counts come from NCRConfig.h; train vs render is chosen in setConeParameters.
    uint32_t mNumSpecRays = NCR_NUM_SPEC_RAYS_RENDER;
    uint32_t mNumKMeansIters = NCR_NUM_KMEANS_ITERS_RENDER;
    uint32_t mNumClusters = NCR_NUM_CLUSTERS;
    /// Russian Roulette probability for random smooth.
    float mRSRRProb = NCR_RS_RR_PROB;
    /// Enable alpha test.
    bool mUseAlphaTest = true;
    /// Use environment lighting.
    bool mUseEnvLight = true;
    /// User next event estimation.
    bool mUseNEE = true;
    /// Adjust shading normals.
    bool mAdjustShadingNormals = true;
    /// Specular roughness threshold
    float mSpecularRoughnessThreshold = NCR_SPECULAR_ROUGHNESS_THRESHOLD;
    /// Replace the specular environment color with the mirror-hit albedo.
    bool mEnableMirrorAlbedo = true;
    /// kEnvColorEncoding* in MirrorAlbedo.slang: how a mirror miss enters the shared
    /// env/albedo channels. 1 = signed log, which keeps 10-1000 sky radiance and [0, 1]
    /// albedo in one bounded range and marks misses by their sign.
    uint32_t mEnvColorEncoding = 1;

    /// Online training schedule. "Heat" h in [0, 1] interpolates log(lr) and
    /// log(1 - emaDecay) between the cold (settled) and hot (adapting) settings.
    /// h starts at 1, holds for mTrainWarmupFrames, then relaxes toward 0 with time
    /// constant mTrainCoolFrames. Camera motion pulls it up to mTrainCameraHeat and
    /// scene changes to mTrainSceneHeat, held mTrainHoldFrames after the last event.
    ///
    /// Tuned on the 10 eval scenes (static 2400 frames, animation, stop 600 frames,
    /// vs ptdn 4096 spp): flicker between frames halves (static and after stopping);
    /// mean PSNR equal to the fixed hot setting. A cold LR of 1e-3 or a 300-frame
    /// warmup stops the network while it is still improving (-0.3 to -0.5 dB static).
    /// Camera heat 0 loses ~1 dB on camera-only paths (bathroom2).
    bool mEnableTrainSchedule = true;
    float mTrainLrHot = NCR_ADAM_LR;
    float mTrainLrCold = NCR_ADAM_LR_COLD;
    float mTrainEmaHot = NCR_EMA_DECAY;
    float mTrainEmaCold = NCR_EMA_DECAY_COLD;
    uint32_t mTrainWarmupFrames = 600;
    uint32_t mTrainHoldFrames = 30;
    float mTrainCoolFrames = 300.f;
    float mTrainCameraHeat = 0.5f;
    float mTrainSceneHeat = 1.f;
    float mTrainHeat = 1.f;
    uint32_t mTrainHoldLeft = 600;
    IScene::UpdateFlags mPendingSceneUpdates = IScene::UpdateFlags::None;
    /// SampleRHS::maxGlossyBounces: glossy RHS hits continue along a BSDF sample (with
    /// NEE at each) up to this many times before the diffuse network is queried.
    /// With 0 the diffuse network must also represent glossy, view-dependent radiance;
    /// in living-room-2 it then collapses after ~1000 frames (LR 5e-3) to a state ~40%
    /// too dark that it never leaves. Cost over 0: +0.2-0.4 ms/frame with 2, +0.4-0.8
    /// with 4; 4 gains ~0.1 dB mean, +1.1 dB on living-room-2.
    uint32_t mRhsGlossyBounces = 2;
    /// Train the diffuse network on glossy LHS samples as well. Only needed when glossy
    /// RHS hits query it (mRhsGlossyBounces = 0); training it on glossy targets is
    /// itself what makes it collapse. When off, an RHS path still on a glossy surface
    /// after mRhsGlossyBounces ends there instead of querying the untrained network.
    bool mDiffTrainOnGlossy = false;
    /// Enable spatiotemporal cluster reuse on the render path. When off, the render
    /// path falls back to the original per-frame group-wide K-Means cone trace.
    bool mEnableRenderSpecTemporalReuse = NCR_ENABLE_CLUSTER_REUSE != 0;
    /// Enable the neighbourhood gather stage of the cluster reuse pipeline.
    bool mEnableClusterSpatialReuse = true;
    /// Wipe all history whenever the scene reports moved geometry. Off by default:
    /// a playing animation raises that flag every frame, which would leave the
    /// render path permanently un-accumulated.
    bool mClearHistoryOnGeometryMoved = false;
    /// With cluster reuse off, run the render path at the training ray count so its
    /// clusters can be compared against the T=128 reference. Very slow.
    bool mUseReferenceConeTrace = false;
    /// lambda: normalized residual below which a depth sample joins an existing mode.
    /// Also sets how finely the cone is partitioned, because a sample that fails the
    /// gate starts a new mode. Tuned against the T=32/T=128 K-Means reference on the
    /// living-room scene: lambda=0.4 gives 3.64 live modes vs the reference's 3.70 and
    /// matches its clsScale/(tanLobe*t) to within 2%, at 1.06 rays/pixel. Larger values
    /// collapse the mixture toward a single wide mode and visibly coarsen the hash grid
    /// query, which shifts the tone of every glossy pixel.
    float mClusterAssignThreshold = 0.1f;
    /// alpha_w kept for property compatibility; accumulation is 1/(n+1) until lock.
    float mClusterAlphaWeight = 0.1f;
    /// N_max: caps the per-mode (mu, sigma) count during accumulation.
    float mClusterMaxCount = 64.0f;
    /// Observation count at which the mixture hard-locks (α = 0). Aligns with old T=32.
    float mClusterLockCount = 32.0f;
    /// Change detector for locked pixels (see ClusterLockState in ClusterHistory.slang).
    /// lambda_d: residual gate of the detector. Must be well above the assignment gate
    /// lambda: with lambda = 0.1 a converged static pixel already puts ~30% of its rays
    /// outside every gate, while lambda_d = 1 leaves ~1%.
    float mClusterDetectThreshold = 1.0f;
    /// beta: EMA rate of the detector shares (~2/beta frames of memory).
    float mClusterDetectRate = 0.125f;
    /// Outlier share at which a locked pixel unlocks to grow new modes.
    float mClusterUnlockOutlierRate = 0.2f;
    /// Lateral drift of the frozen anchors since lock, in cone radii, at which a locked
    /// pixel unlocks. Catches smooth camera motion, which the per-frame cone test misses.
    float mClusterUnlockDrift = 0.5f;
    /// Per-frame drift (cone radii) above which a pixel counts as moving with the camera.
    /// A still camera gives ~1e-6 (float round-off); on the bathroom2 camera path 0.02
    /// missed many moving pixels.
    float mClusterMotionDrift = 0.002f;
    /// Observation cap for moving pixels: they re-accumulate from ~this many recent samples
    /// and cannot lock until the camera holds still, so no motion trail gets frozen.
    float mClusterMotionMemory = 4.0f;
    /// Modes below this weight release their slot.
    float mClusterWeakWeight = 0.02f;
    /// Newborn sigma = mClusterSigmaPriorScale * tanLobe * t.
    /// 0 makes a single-sample mode produce exactly the training path's singleton
    /// clsScale of tanLobe*t/2; Welford then grows sigma from later hits. Measured
    /// converged sigma/(tanLobe*t) has a median of 0.21, so any prior much above
    /// that biases every fresh cluster toward a coarser hash grid level.
    float mClusterSigmaPriorScale = 0.0f;
    /// Ray budget per glossy pixel per frame. 1 baseline, more only on outliers.
    uint32_t mClusterMaxRays = 4;
    /// Reprojection rejection thresholds.
    float mClusterNormalThreshold = 0.9f;
    float mClusterRoughnessThreshold = 0.05f;
    float mClusterPosThresholdRel = 0.02f;
    /// Spatial gather configuration.
    uint32_t mClusterNumTaps = 5;
    float mClusterGatherRadiusScale = 1.0f;
    float mClusterMaxGatherRadius = 8.0f;
    float mClusterGammaMax = 0.6f;
    float mClusterGammaMin = 0.05f;
    float mClusterConfidenceCount = 32.0f;
    /// Post filter switch
    bool mEnablePostFilter = true;
    /// FXAA switch for post filter.
    bool mEnableFXAA = true;
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
    /// 1-element stand-in so shaders that declare the full RayBatch layout can
    /// still be bound when a group (usually RHS spec) is not allocated.
    std::shared_ptr<RayBatchBuffer> mpBindDummyBatch;
    ref<Texture> mpRenderTempColor;
    ref<Texture> mpPostFilterTempColor;
    ref<Texture> mpPrefiltEnvCubemap;
    ref<Sampler> mpPrefiltEnvSampler;
    bool mPrefiltEnvDirty = true;

    /// Per-pixel cluster mixture, ping-ponged: [1 - write] holds the previous frame.
    ref<Buffer> mpClusterHistory[2];
    /// Packed change-detector state (one u32 per pixel), ping-ponged with the mixture.
    ref<Buffer> mpClusterLockState[2];
    uint32_t mClusterHistoryWriteIdx = 0;
    /// False until one full frame of history exists (after a clear or a resize).
    bool mClusterHistoryValid = false;
    /// Previous frame G-Buffer copies, needed to rebuild the writer's cone frame.
    ref<Texture> mpPrevPos;
    ref<Texture> mpPrevDir;
    ref<Texture> mpPrevNormal;
    ref<Texture> mpPrevRoughness;
    ref<Buffer> mpPrevSpecActive;

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
    ref<ComputePass> mpSpecPrimary;
    ref<ComputePass> mpClusterReproject;
    ref<ComputePass> mpConeTraceScreen;
    ref<ComputePass> mpClusterSpatialReuse;
    ref<ComputePass> mpPrefiltEnvInitPass;
    ref<ComputePass> mpPrefiltEnvDownsamplePass;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<PrefixSum> mpPrefixSum;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;    ///< Emissive light sampler or nullptr if not used.

    // CUDA
    ref<cuda_utils::CudaDevice> mpCudaDevice;
    std::unique_ptr<NRModel> mpNRModel;

    // Train mode parameters
    RenderMode mRenderMode = RenderMode::OnlineTrain;
    /// LHS and RHS
    uint32_t mBatchSizeInit = NCR_BATCH_SIZE_INIT;
    uint32_t mNumRHSInit = NCR_NUM_RHS_INIT;
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
    /// Checkpoint to load right after the model is created (property `loadModelPath`).
    std::string mLoadModelPath;
};

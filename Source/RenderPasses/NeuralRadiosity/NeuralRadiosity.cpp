#include "NeuralRadiosity.h"

#include "Utils/Math/VectorJson.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>

namespace
{
    // Render pass inputs and outputs.

    const Falcor::ChannelList kInputChannels =
    {
        // { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
        // { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputVBuffer = "vbuffer";
    const std::string kOutputActive = "active";
    const std::string kOutputPos = "pos";
    const std::string kOutputDir = "dir";
    const std::string kOutputNormal = "normal";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputRoughness = "roughness";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,         "gColor",       "Output color",                       false,  ResourceFormat::RGBA32Float },
        { kOutputVBuffer,       "gVBuffer",     "Visibility buffer in packed format", false,  ResourceFormat::RGBA32Uint },
        { kOutputActive,        "gActive",      "If the pixel is active",             false,  ResourceFormat::R32Uint },
        { kOutputPos,           "gPos",         "Output position",                    false,  ResourceFormat::RGBA32Float },
        { kOutputDir,           "gDir",         "Output direction",                   false,  ResourceFormat::RGBA32Float },
        { kOutputNormal,        "gNormal",      "Output normal",                      false,  ResourceFormat::RGBA32Float },
        { kOutputAlbedo,        "gAlbedo",      "Output albedo",                      false,  ResourceFormat::RGBA32Float },
        { kOutputRoughness,     "gRoughness",   "Output roughness",                   false,  ResourceFormat::R32Float },
    };

    // Program files.
    const std::string kFirstSmoothPassFile = "RenderPasses/NeuralRadiosity/FirstSmoothPass.cs.slang";
    const std::string kCompactPassFile = "RenderPasses/NeuralRadiosity/CompactPass.cs.slang";
    const std::string kResolvePassFile = "RenderPasses/NeuralRadiosity/ResolvePass.cs.slang";
    const std::string kPostFilterPassFile = "RenderPasses/NeuralRadiosity/PostFilterPass.cs.slang";

    const std::string kRandomSmoothFile = "RenderPasses/NeuralRadiosity/RandomSmooth.cs.slang";
    const std::string kSampleRHSFile = "RenderPasses/NeuralRadiosity/SampleRHS.cs.slang";
    const std::string kResolveRHSFile = "RenderPasses/NeuralRadiosity/ResolveRHS.cs.slang";
    const std::string kCompactBatchFile = "RenderPasses/NeuralRadiosity/CompactBatch.cs.slang";
    const std::string kResolveBatchFile = "RenderPasses/NeuralRadiosity/ResolveBatch.cs.slang";
    
    const std::string kConeTraceFile = "RenderPasses/NeuralRadiosity/ConeTrace.cs.slang";
    const std::string kPrefiltEnvCubemapFile = "RenderPasses/NeuralRadiosity/PrefiltEnvCubemap.slang";
    const std::string kDebugDumpPassFile = "RenderPasses/NeuralRadiosity/DebugDumpPass.cs.slang";

    const FileDialogFilterVec kCameraJsonFilters = {
        {"json", "JSON Files"},
    };

    const FileDialogFilterVec kModelStateFilters = {
        {"ckpt", "Binary Files"},
    };

    const char kEnableTemporalSpecReuse[] = "enableTemporalSpecReuse";
    const char kConeTraceReuseMode[] = "coneTraceReuseMode";
    const char kTemporalSpecClusterMode[] = "temporalSpecClusterMode";
    const char kEnableFXAA[] = "enableFXAA";
    const char kLoadModelPath[] = "loadModelPath";
    const char kRenderMode[] = "renderMode";

    float3 parseFloat3(const nlohmann::json& value, std::string_view key)
    {
        if (!value.is_array() || value.size() != 3)
        {
            FALCOR_THROW("Camera field '{}' must be an array of 3 numbers.", key);
        }

        return value.get<float3>();
    }

    uint32_t floorPowerOfTwo(uint32_t value)
    {
        if (value == 0) return 1;

        uint32_t result = 1;
        while ((result << 1) <= value)
            result <<= 1;
        return result;
    }

    uint32_t computePrefiltEnvFaceSize(const ref<Texture>& pEnvMap)
    {
        FALCOR_ASSERT(pEnvMap);
        const uint32_t angularSize = std::max(1u, std::min(pEnvMap->getWidth() / 4u, pEnvMap->getHeight() / 2u));
        return floorPowerOfTwo(angularSize);
    }

    uint32_t computeMipCount(uint32_t faceSize)
    {
        uint32_t mipCount = 1;
        while (faceSize > 1)
        {
            faceSize >>= 1;
            mipCount++;
        }
        return mipCount;
    }
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralRadiosity>();
}

static uint64_t gpuUsedMBs() {
    cudaDeviceSynchronize();
    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    return (totalBytes - freeBytes) / (1024 * 1024);
}

NeuralRadiosity::NeuralRadiosity(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kEnableTemporalSpecReuse)
            mEnableRenderSpecTemporalReuse = value;
        else if (key == kEnableFXAA)
            mEnableFXAA = value;
        else if (key == kConeTraceReuseMode || key == kTemporalSpecClusterMode)
            mConeTraceReuseMode = value;
        else if (key == kLoadModelPath)
        {
            std::string path = value;
            mLoadModelPath = path;
        }
        else if (key == kRenderMode)
        {
            std::string mode = value;
            if (mode == "train")
                mRenderMode = RenderMode::Train;
            else if (mode == "onlineTrain")
                mRenderMode = RenderMode::OnlineTrain;
            else if (mode == "render")
                mRenderMode = RenderMode::Render;
            else if (mode == "idle")
                mRenderMode = RenderMode::Idle;
        }
        else
            logWarning("Unknown property '{}' in NeuralRadiosity properties.", key);
    }

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);

    // Environment-variable-driven debug switches (for headless/CI runs).
    if (const char* v = std::getenv("NCR_HASH_STATS"))
    {
        mEnableHashCollisionStats = (v[0] == '1');
    }
    if (const char* v = std::getenv("NCR_DUMP_DEBUG"))
    {
        mDebugDumpRequested = (v[0] == '1');
    }
}

Properties NeuralRadiosity::getProperties() const
{
    Properties props;
    props[kEnableTemporalSpecReuse] = mEnableRenderSpecTemporalReuse;
    props[kConeTraceReuseMode] = mConeTraceReuseMode;
    props[kEnableFXAA] = mEnableFXAA;
    return props;
}

RenderPassReflection NeuralRadiosity::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);

    return reflector;
}

void NeuralRadiosity::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    const auto& pOutput = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutput);

    // Set output frame dimension.
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    // If there is no scene, clear the output and return.
    if (mpScene == nullptr)
    {
        clearRenderPassChannels(pRenderContext, kOutputChannels, renderData);
        return;
    }

    // Prepare lighting data.
    prepareLighting(pRenderContext);

    // Update shader program specialization.
    updatePrograms(pRenderContext, renderData);

    // Prepare the roughness-addressable environment cubemap.
    // preparePrefiltEnvCubemap(pRenderContext);

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    if (mEnableRenderSpecTemporalReuse && (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain))
    {
        updateRenderTemporalHistory(pRenderContext);
    }

    mVarsChanged = false;

    // Debug: reset hash collision counters at frame start.
    if (mEnableHashCollisionStats && mpNRModel)
    {
        mpNRModel->resetCollisionStats();
    }

    // Train the model.
    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
    {
        train(pRenderContext);
    }

    // Render the scene.
    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        render(pRenderContext, renderData);
    }

    // Debug: read back hash collision counters and log.
    if (mEnableHashCollisionStats && mpNRModel)
    {
        // Sync both CUDA streams so counter writes are visible to host.
        pRenderContext->waitForCuda(mpNRModel->stream(false));
        pRenderContext->waitForCuda(mpNRModel->stream(true));
        const auto stats = mpNRModel->getCollisionStats();
        const uint32_t primCollisions = (stats.primTotalQueries > stats.primUniqueSlots)
            ? (stats.primTotalQueries - stats.primUniqueSlots) : 0u;
        const uint32_t clsCollisions = (stats.clsTotalQueries > stats.clsUniqueSlots)
            ? (stats.clsTotalQueries - stats.clsUniqueSlots) : 0u;
        NCR_LOG(this, "[NCR hash collisions] frame={} prim: hashQueries={} uniqueSlots={} collisions={} | "
                "cls: hashQueries={} uniqueSlots={} collisions={}",
                mFrameCount,
                stats.primTotalQueries, stats.primUniqueSlots, primCollisions,
                stats.clsTotalQueries, stats.clsUniqueSlots, clsCollisions);

        // Specular level-usage histogram: (upperLevel, lowerLevel) pair counts.
        // Diffuse (HashGrid) always concatenates all levels, so only spec is logged.
        std::vector<uint32_t> hist;
        mpNRModel->getLevelHistogram(hist);
        const uint32_t nLevels = mpNRModel->getNumLevels();
        uint64_t totalQueries = 0;
        for (auto c : hist) totalQueries += c;
        std::string histStr;
        for (uint32_t u = 0; u < nLevels; ++u) {
            for (uint32_t l = 0; l < nLevels; ++l) {
                const uint32_t cnt = hist[u * nLevels + l];
                if (cnt > 0) {
                    if (!histStr.empty()) histStr += ", ";
                    histStr += fmt::format("U{}L{}={}", u, l, cnt);
                }
            }
        }
        NCR_LOG(this, "[NCR spec level usage] frame={} totalQueries={} pairs=[{}]",
                mFrameCount, totalQueries, histStr.empty() ? "none" : histStr);
    }

    mFrameCount++;
}

void NeuralRadiosity::renderUI(Gui::Widgets& widget)
{
    widget.text(fmt::format("Frame count: {}", mFrameCount));
    if (widget.dropdown("Render Mode", mRenderMode))
    {
        mVarsChanged = true;
        if (mRenderMode == RenderMode::Train)
        {
            mFrameCount = 0;
            setConeParameters(true);
        }
        else if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
        {
            setConeParameters(false);
        }
        if (mpNRModel) mpNRModel->setOnline(mRenderMode != RenderMode::Train);
    }

    if (widget.button("Load model state"))
    {
        std::filesystem::path path;
        if (mpNRModel && openFileDialog(kModelStateFilters, path))
        {
            mpNRModel->loadState(path.string());
        }
    }

    if (widget.button("Save model state"))
    {
        std::filesystem::path path;
        if (mpNRModel && saveFileDialog(kModelStateFilters, path))
        {
            mpNRModel->saveState(path.string());
        }
    }

    if (widget.button("Resume training from ckpt"))
    {
        mVarsChanged = true;
        std::filesystem::path path;
        if (mpNRModel && openFileDialog(kModelStateFilters, path))
        {
            uint32_t frameCountFromFile = 0;
            try
            {
                frameCountFromFile = std::stoi(path.stem().string());
                mFrameCount = frameCountFromFile;
                mRenderMode = RenderMode::Train;
                setConeParameters(true);
            }
            catch(const std::exception& e)
            {
                logError(e.what());
            }

            mpNRModel->loadState(path.string());
        }
    }

    if (widget.button("Load train cameras"))
    {
        std::filesystem::path path;
        if (openFileDialog(kCameraJsonFilters, path))
        {
            loadTrainCamerasFromFile(path);
            mVarsChanged = true;
        }
    }

    widget.text(fmt::format("Train batchsize: {}, RHS samples: {}", mBatchSize, mNumRHS));
    widget.text(fmt::format("Num spec rays: {}, clusters: {}, KMeans iters: {}", mNumSpecRays, mNumClusters, mNumKMeansIters));
    widget.text(fmt::format("Train cameras: {}", mTrainCameras.size()));
    widget.text(mTrainCameraPath.empty() ? "Camera file: N/A" : fmt::format("Camera file: {}", mTrainCameraPath.filename().string()));

    if (!mTrainCameraPath.empty())
    {
        widget.tooltip(mTrainCameraPath.string());
    }

    if (widget.checkbox("Use NEE", mUseNEE))
    {
        mVarsChanged = true;
    }

    if (widget.checkbox("Temporal Spec Reuse", mEnableRenderSpecTemporalReuse))
    {
        mVarsChanged = true;
        if (mpRenderSpecTemporalHistory)
        {
            clearRenderTemporalHistory(mpDevice->getRenderContext());
        }
    }

    if (widget.dropdown("Cone Trace Reuse Mode", mConeTraceReuseMode))
    {
        mVarsChanged = true;
        if (mpRenderSpecTemporalHistory)
        {
            clearRenderTemporalHistory(mpDevice->getRenderContext());
        }
    }

    if (widget.checkbox("Post Filter", mEnablePostFilter))
    {
        mVarsChanged = true;
    }
    if (mEnablePostFilter && widget.checkbox("FXAA", mEnableFXAA))
    {
        mVarsChanged = true;
    }

    if (auto debugGroup = widget.group("Debug: isolate / freeze", false))
    {
        debugGroup.text("Render only one lobe (rebuilds shaders on toggle)");
        if (debugGroup.checkbox("Render Diffuse Only", mDebugRenderDiffuseOnly))
        {
            if (mDebugRenderDiffuseOnly) mDebugRenderSpecularOnly = false;
            mVarsChanged = true;
        }
        if (debugGroup.checkbox("Render Specular Only", mDebugRenderSpecularOnly))
        {
            if (mDebugRenderSpecularOnly) mDebugRenderDiffuseOnly = false;
            mVarsChanged = true;
        }

        debugGroup.separator();

        debugGroup.text("Freeze training updates (network still inferred)");
        if (debugGroup.checkbox("Disable Diffuse Update", mDisableDiffUpdate))
        {
            mVarsChanged = true;
        }
        if (debugGroup.checkbox("Disable Specular Update", mDisableSpecUpdate))
        {
            mVarsChanged = true;
        }

        debugGroup.separator();

        debugGroup.text("Freeze individual optimizer step (fwd/bwd still run)");
        if (debugGroup.checkbox("Freeze PrimGrid Opt", mFreezePrimGridOpt))
        {
            if (mpNRModel) mpNRModel->setFreezePrimGridOpt(mFreezePrimGridOpt);
        }
        if (debugGroup.checkbox("Freeze ClsGrid Opt", mFreezeClsGridOpt))
        {
            if (mpNRModel) mpNRModel->setFreezeClsGridOpt(mFreezeClsGridOpt);
        }
        if (debugGroup.checkbox("Freeze DiffNet Opt", mFreezeDiffNetOpt))
        {
            if (mpNRModel) mpNRModel->setFreezeDiffNetOpt(mFreezeDiffNetOpt);
        }
        if (debugGroup.checkbox("Freeze SpecNet Opt", mFreezeSpecNetOpt))
        {
            if (mpNRModel) mpNRModel->setFreezeSpecNetOpt(mFreezeSpecNetOpt);
        }

        debugGroup.separator();

        if (debugGroup.checkbox("Verbose Scene Log", mEnableDebugLog))
        {
            // No shader rebuild needed; next frame will re-emit logs.
        }
        if (debugGroup.checkbox("Hash Collision Stats", mEnableHashCollisionStats))
        {
            if (mpNRModel) mpNRModel->setCollisionStatsEnabled(mEnableHashCollisionStats);
        }
        if (debugGroup.checkbox("Show Spec Level", mDebugShowSpecLevel))
        {
            mVarsChanged = true;
        }

        debugGroup.separator();
        debugGroup.text("One-shot: dump debug images (spec level, material, normal, tanLobe, cluster mean) to debug_dump_<frame>/");
        if (debugGroup.button("Dump Debug Images"))
        {
            mDebugDumpRequested = true;
            mVarsChanged = true; // rebuild shaders with DEBUG_DUMP=1 next frame
        }
    }

    // --- Training Parameters tab (raw ImGui for DragFloat support) ---
    if (ImGui::CollapsingHeader("Training Parameters", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (!mpNRModel)
        {
            ImGui::TextDisabled("No model loaded");
            return;
        }

        // Optimizer step count
        ImGui::Text("Optimizer step: %u", mpNRModel->getOptimizerStep());
        ImGui::Separator();

        // --- Learning rates ---
        ImGui::Text("Learning Rate (Adam)");
        ImGui::SameLine();
        if (ImGui::SmallButton("Sync All"))
        {
            float lr = mpNRModel->getDiffNetLr();
            mpNRModel->setPrimGridLr(lr);
            mpNRModel->setClsGridLr(lr);
            mpNRModel->setDiffNetLr(lr);
            mpNRModel->setSpecNetLr(lr);
        }

        ImGui::PushItemWidth(80);
        if (ImGui::DragFloat("##PrimGridLr", &mParamPrimGridLr, 1e-4f, 1e-6f, 1.f, "%.6f"))
        {
            mpNRModel->setPrimGridLr(mParamPrimGridLr);
        }
        ImGui::SameLine();
        ImGui::Text("PrimGrid");

        if (ImGui::DragFloat("##ClsGridLr", &mParamClsGridLr, 1e-4f, 1e-6f, 1.f, "%.6f"))
        {
            mpNRModel->setClsGridLr(mParamClsGridLr);
        }
        ImGui::SameLine();
        ImGui::Text("ClsGrid");

        if (ImGui::DragFloat("##DiffNetLr", &mParamDiffNetLr, 1e-4f, 1e-6f, 1.f, "%.6f"))
        {
            mpNRModel->setDiffNetLr(mParamDiffNetLr);
        }
        ImGui::SameLine();
        ImGui::Text("DiffNet");

        if (ImGui::DragFloat("##SpecNetLr", &mParamSpecNetLr, 1e-4f, 1e-6f, 1.f, "%.6f"))
        {
            mpNRModel->setSpecNetLr(mParamSpecNetLr);
        }
        ImGui::SameLine();
        ImGui::Text("SpecNet");
        ImGui::PopItemWidth();

        ImGui::Separator();

        // --- EMA decay ---
        ImGui::Text("EMA Decay");
        ImGui::SameLine();
        if (ImGui::SmallButton("Sync All EMA"))
        {
            float d = mpNRModel->getDiffNetEmaDecay();
            mpNRModel->setPrimGridEmaDecay(d);
            mpNRModel->setClsGridEmaDecay(d);
            mpNRModel->setDiffNetEmaDecay(d);
            mpNRModel->setSpecNetEmaDecay(d);
        }

        ImGui::PushItemWidth(80);
        if (ImGui::DragFloat("##PrimGridEma", &mParamPrimGridEmaDecay, 0.001f, 0.f, 1.f, "%.4f"))
        {
            mpNRModel->setPrimGridEmaDecay(mParamPrimGridEmaDecay);
        }
        ImGui::SameLine();
        ImGui::Text("PrimGrid EMA");

        if (ImGui::DragFloat("##ClsGridEma", &mParamClsGridEmaDecay, 0.001f, 0.f, 1.f, "%.4f"))
        {
            mpNRModel->setClsGridEmaDecay(mParamClsGridEmaDecay);
        }
        ImGui::SameLine();
        ImGui::Text("ClsGrid EMA");

        if (ImGui::DragFloat("##DiffNetEma", &mParamDiffNetEmaDecay, 0.001f, 0.f, 1.f, "%.4f"))
        {
            mpNRModel->setDiffNetEmaDecay(mParamDiffNetEmaDecay);
        }
        ImGui::SameLine();
        ImGui::Text("DiffNet EMA");

        if (ImGui::DragFloat("##SpecNetEma", &mParamSpecNetEmaDecay, 0.001f, 0.f, 1.f, "%.4f"))
        {
            mpNRModel->setSpecNetEmaDecay(mParamSpecNetEmaDecay);
        }
        ImGui::SameLine();
        ImGui::Text("SpecNet EMA");
        ImGui::PopItemWidth();

        ImGui::Separator();

        // --- Algorithm parameters ---
        ImGui::Text("Algorithm Parameters");

        ImGui::PushItemWidth(80);
        if (ImGui::DragInt("##NumSpecRays", (int*)&mNumSpecRays, 1, 1, 512))
        {
            mVarsChanged = true;
        }
        ImGui::SameLine();
        ImGui::Text("Num Spec Rays (%d)", mNumSpecRays);

        if (ImGui::DragInt("##NumKMeansIters", (int*)&mNumKMeansIters, 1, 1, 50))
        {
            mVarsChanged = true;
        }
        ImGui::SameLine();
        ImGui::Text("KMeans Iters (%d)", mNumKMeansIters);

        if (ImGui::DragFloat("##InterpRatio", &mParamInterpRatio, 0.01f, 0.01f, 10.f, "%.4f"))
        {
            mpNRModel->setInterpRatio(mParamInterpRatio);
            mVarsChanged = true; // grid scales changed
        }
        ImGui::SameLine();
        ImGui::Text("InterpRatio (%.4f)", mParamInterpRatio);

        if (ImGui::DragFloat("##RoughnessThreshold", &mSpecularRoughnessThreshold, 0.01f, 0.f, 1.f, "%.3f"))
        {
            mVarsChanged = true;
        }
        ImGui::SameLine();
        ImGui::Text("Roughness Threshold (%.3f)", mSpecularRoughnessThreshold);

        if (ImGui::DragFloat("##RSRRProb", &mRSRRProb, 0.01f, 0.f, 1.f, "%.3f"))
        {
            mVarsChanged = true;
        }
        ImGui::SameLine();
        ImGui::Text("Russian Roulette Prob (%.3f)", mRSRRProb);
        ImGui::PopItemWidth();
    }
}

void NeuralRadiosity::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpFirstSmoothPass = nullptr;
    mpConeTrace = nullptr;
    mpResolvePass = nullptr;
    mpPostFilterPass = nullptr;
    mpPostFXAAPass = nullptr;
    mpPrefiltEnvInitPass = nullptr;
    mpPrefiltEnvDownsamplePass = nullptr;
    mpPrefiltEnvCubemap = nullptr;
    mpRenderSpecTemporalHistory = nullptr;
    mRenderSpecTemporalHistoryValid = false;
    mHasRenderCameraViewProj = false;
    mPrefiltEnvDirty = true;
    mVarsChanged = true;

    // Verbose scene dump for cross-scene parameter comparison.
    NCR_LOG(this, "================ NCR setScene ================");
    if (mpScene)
    {
        const AABB& bb = mpScene->getSceneBounds();
        NCR_LOG(this, "  scene AABB min = ({}, {}, {})", bb.minPoint.x, bb.minPoint.y, bb.minPoint.z);
        NCR_LOG(this, "  scene AABB max = ({}, {}, {})", bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z);
        const float3 diag = bb.maxPoint - bb.minPoint;
        NCR_LOG(this, "  scene AABB diag = ({}, {}, {})", diag.x, diag.y, diag.z);
        NCR_LOG(this, "  useEmissiveLights = {}", mpScene->useEmissiveLights());
        NCR_LOG(this, "  useEnvLight       = {}", mpScene->useEnvLight());
        NCR_LOG(this, "  scene light count = {}", mpScene->getLightCount());
        const auto& pEnvMap = mpScene->getEnvMap();
        if (pEnvMap && pEnvMap->getEnvMap())
        {
            NCR_LOG(this, "  envMap size       = {}x{}", pEnvMap->getEnvMap()->getWidth(), pEnvMap->getEnvMap()->getHeight());
            NCR_LOG(this, "  envMap intensity  = {}", pEnvMap->getIntensity());
            NCR_LOG(this, "  envMap rotation   = ({}, {}, {})", pEnvMap->getRotation().x, pEnvMap->getRotation().y, pEnvMap->getRotation().z);
        }
        else
        {
            NCR_LOG(this, "  envMap            = <none>");
        }
        const auto& pCam = mpScene->getCamera();
        if (pCam)
        {
            const float3& pos = pCam->getPosition();
            const float3& tgt = pCam->getTarget();
            NCR_LOG(this, "  camera pos        = ({}, {}, {})", pos.x, pos.y, pos.z);
            NCR_LOG(this, "  camera target     = ({}, {}, {})", tgt.x, tgt.y, tgt.z);
            NCR_LOG(this, "  camera focalLen   = {}", pCam->getFocalLength());
            NCR_LOG(this, "  camera aspect     = {}", pCam->getAspectRatio());
        }
    }
    else
    {
        NCR_LOG(this, "  scene = <null>");
    }
    NCR_LOG(this, "  mNumClusters      = {}", mNumClusters);
    NCR_LOG(this, "  mNumSpecRays      = {}", mNumSpecRays);
    NCR_LOG(this, "  mNumKMeansIters   = {}", mNumKMeansIters);
    NCR_LOG(this, "  mSpecularRoughnessThreshold = {}", mSpecularRoughnessThreshold);
    NCR_LOG(this, "  mBatchSize        = {}", mBatchSize);
    NCR_LOG(this, "  mNumRHS           = {}", mNumRHS);
    NCR_LOG(this, "=============================================");
}

void NeuralRadiosity::onSceneUpdates(RenderContext* pRenderContext, IScene::UpdateFlags sceneUpdates)
{
    if (is_set(sceneUpdates, IScene::UpdateFlags::GeometryMoved))
    {
        clearRenderTemporalHistory(pRenderContext);
    }

    if (is_set(sceneUpdates, IScene::UpdateFlags::LightIntensityChanged) ||
        is_set(sceneUpdates, IScene::UpdateFlags::EnvMapChanged) ||
        is_set(sceneUpdates, IScene::UpdateFlags::EnvMapPropertiesChanged) ||
        is_set(sceneUpdates, IScene::UpdateFlags::RenderSettingsChanged))
    {
        // Light changed.
        mPrefiltEnvDirty = true;
    }
}

// Private methods

void NeuralRadiosity::render(RenderContext* pRenderContext, const RenderData& renderData)
{
    // First smooth pass.
    firstSmoothPass(pRenderContext, renderData);
    // Compact pass.
    compactPass(pRenderContext, renderData);

    // Cone tracing pass.
    coneTrace(pRenderContext, mpRenderBatch);
    // Model querying: CUDA kernel.
    modelInferenceCUDA(pRenderContext, mpRenderBatch);
    // Resolve pass.
    resolvePass(pRenderContext, renderData);
    // Post-process filtering pass.
    postFilterPass(pRenderContext, renderData);

    // Debug image dump: one-shot. After ConeTrace wrote debug buffers (with
    // DEBUG_DUMP=1), run DebugDumpPass to copy buffers into textures, then
    // save all images to disk. Then request shader rebuild to restore
    // DEBUG_DUMP=0 for normal rendering.
    if (mDebugDumpRequested)
    {
        dumpDebugImages(pRenderContext, renderData);
        mDebugDumpRequested = false;
        mVarsChanged = true; // rebuild shaders without DEBUG_DUMP next frame
    }
}

void NeuralRadiosity::train(RenderContext* pRenderContext)
{
    // Sample random smooth points (LHS)
    randomSmooth(pRenderContext, mpTrainLHSBatch);
    // Sample RHS
    sampleRHS(pRenderContext);

    // Render RHS
    {
        // 1. Compaction
        compactBatch(pRenderContext, mpTrainRHSBatch);
        // 2. Cone tracing
        coneTrace(pRenderContext, mpTrainRHSBatch);
        // 3. Model inference
        modelInferenceCUDA(pRenderContext, mpTrainRHSBatch);
        // 4. Resolve
        resolveBatch(pRenderContext, mpTrainRHSBatch);
    }

    // Train LHS with RHS results
    {
        // 1. Compaction
        compactBatch(pRenderContext, mpTrainLHSBatch);
        // 2. Cone tracing
        coneTrace(pRenderContext, mpTrainLHSBatch);
        // 3. Resolve RHS color (write to lhs->diffColor/specColor)
        resolveRHS(pRenderContext);
        // 4. Model forward (target should be written to diffColor/specColor)
        modelTrainCUDA(pRenderContext, mpTrainLHSBatch);
    }

    if (mRenderMode == RenderMode::Train)
    {
        const uint32_t completedSteps = mFrameCount + 1;

        if (completedSteps % mSaveCKPTInterval == 0)
        {
            std::filesystem::path ckptPath = fmt::format("{:05d}.ckpt", completedSteps);
            if (mpNRModel)
            {
                mpNRModel->saveState(ckptPath.string());
                fmt::print("Saved checkpoint to '{}'\n", ckptPath.string());
            }
        }
        updateAdaptiveRHSState(completedSteps);

        if (completedSteps >= mTotalTrainSteps)
        {
            fmt::print("Finished training after {} steps.\n", completedSteps);
            mRenderMode = RenderMode::Idle;
            mVarsChanged = true;
        }
    }
}

void NeuralRadiosity::firstSmoothPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "FirstSmoothPass");

    ShaderVar var = mpFirstSmoothPass->getRootVar();
    bindScreenData(var, renderData, "gFirstSmoothPass");

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    mpFirstSmoothPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void NeuralRadiosity::compactPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "CompactPass");

    // Prefix sum from active mask to index
    mpPrefixSum->execute(
        pRenderContext,
        mpRenderBatch->diffIndex,
        mFrameDim.x * mFrameDim.y,
        &mpRenderBatch->diffSize
    );

    mpPrefixSum->execute(
        pRenderContext,
        mpRenderBatch->specIndex,
        mFrameDim.x * mFrameDim.y,
        &mpRenderBatch->specSize
    );

    ShaderVar var = mpCompactPass->getRootVar();
    const auto name = "gCompactPass";
    bindScreenData(var, renderData, name);

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    var[name]["diffSize"] = mpRenderBatch->diffSize;
    var[name]["specSize"] = mpRenderBatch->specSize;

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["diffInput"] = mpRenderBatch->diffInput;

    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;
    var["specInput"] = mpRenderBatch->specInput;
    var["specVBuffer"] = mpRenderBatch->specVBuffer;
    var["specPixel"] = mpRenderBatch->specPixel;
    var["specPos"] = mpRenderBatch->specPos;
    var["specDir"] = mpRenderBatch->specDir;
    var["specRoughness"] = mpRenderBatch->specRoughness;

    mpCompactPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void NeuralRadiosity::resolvePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ResolvePass");

    ShaderVar var = mpResolvePass->getRootVar();
    const auto name = "gResolvePass";
    bindScreenData(var, renderData, name);

    var[name]["diffSize"] = mpRenderBatch->diffSize;
    var[name]["specSize"] = mpRenderBatch->specSize;

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["diffColor"] = mpRenderBatch->diffColor;

    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;
    var["specColor"] = mpRenderBatch->specColor;
    var["specVBuffer"] = mpRenderBatch->specVBuffer;

    // var["temporalHistory"] = mpRenderSpecTemporalHistory;
    var["tempColor"] = mpRenderTempColor;

    // Debug: per-spec-pixel grid level visualization buffer (read by ResolvePass
    // when DEBUG_SHOW_SPEC_LEVEL is on to color-code spec regions by grid level).
    if (mDebugShowSpecLevel && mpSpecLevelBuffer)
    {
        var["gSpecLevel"] = mpSpecLevelBuffer;
    }

    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
        var["debug"] = mpTrainRHSBatch->emission;

    mpResolvePass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void NeuralRadiosity::postFilterPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PostFilterPass");

    if (!mEnablePostFilter)
        return;

    {
        ShaderVar var = mpPostFilterPass->getRootVar();
        const auto name = "gPostFilterPass";
        bindScreenData(var, renderData, name);

        var[name]["kernelRadius"] = 3;
        var[name]["sigmaSpatial"] = 4.0f;
        var[name]["sigmaNormal"] = 0.05f;
        var[name]["sigmaAlbedo"] = 0.02f;
        var[name]["sigmaColor"] = 2.0f;

        var["diffActive"] = mpRenderBatch->diffActive;
        var["diffIndex"] = mpRenderBatch->diffIndex;
        var["specActive"] = mpRenderBatch->specActive;
        var["specIndex"] = mpRenderBatch->specIndex;
        var["tempColor"] = mpRenderTempColor;
        var["gFilteredColor"] = mpPostFilterTempColor;

        mpPostFilterPass->execute(pRenderContext, uint3(mFrameDim, 1));
    }

    pRenderContext->uavBarrier(mpPostFilterTempColor.get());

    {
        ShaderVar var = mpPostFXAAPass->getRootVar();
        const auto name = "gPostFilterPass";
        bindScreenData(var, renderData, name);
        var[name]["enableFXAA"] = mEnableFXAA ? 1u : 0u;
        var["gFxaaInput"] = mpPostFilterTempColor;

        mpPostFXAAPass->execute(pRenderContext, uint3(mFrameDim, 1));
    }
}

void NeuralRadiosity::randomSmooth(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "RandomSmooth");

    // Sample a random camera (offline training)
    if (mRenderMode == RenderMode::Train && (mTrainCameras.size() > 0))
    {
        std::mt19937 rng(mFrameCount);
        std::uniform_int_distribution<uint32_t> dist(0, mTrainCameras.size() - 1);
        setCamera(dist(rng));
    }

    ShaderVar var = mpRandomSmooth->getRootVar();
    bindRayBatchData(var, pRayBatch, "gRandomSmooth");

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    mpRandomSmooth->execute(pRenderContext, uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::sampleRHS(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SampleRHS");

    ShaderVar var = mpSampleRHS->getRootVar();
    const auto name = "gSampleRHS";

    // Bind parameters
    var[name]["batchSize"] = mBatchSize;
    var[name]["batchCount"] = mFrameCount;

    // LHS buffers
    {
        var["lhsPos"] = mpTrainLHSBatch->pos;
        var["lhsDir"] = mpTrainLHSBatch->dir;
        var["lhsVBuffer"] = mpTrainLHSBatch->vbuffer;
        var["lhsDiffActive"] = mpTrainLHSBatch->diffActive;
        var["lhsSpecActive"] = mpTrainLHSBatch->specActive;
    }

    // RHS buffers
    {
        var["rhsPos"] = mpTrainRHSBatch->pos;
        var["rhsDir"] = mpTrainRHSBatch->dir;
        var["rhsNormal"] = mpTrainRHSBatch->normal;
        var["rhsAlbedo"] = mpTrainRHSBatch->albedo;
        var["rhsRoughness"] = mpTrainRHSBatch->roughness;
        var["rhsVBuffer"] = mpTrainRHSBatch->vbuffer;
        var["rhsColor"] = mpTrainRHSBatch->color;
        var["rhsEmission"] = mpTrainRHSBatch->emission;

        var["rhsDiffActive"] = mpTrainRHSBatch->diffActive;
        var["rhsDiffIndex"] = mpTrainRHSBatch->diffIndex;
        var["rhsSpecActive"] = mpTrainRHSBatch->specActive;
        var["rhsSpecIndex"] = mpTrainRHSBatch->specIndex;
    }

    var[name]["useNEE"] = false;
    if (mpEmissiveSampler)
    {
        var[name]["useNEE"] = mUseNEE;
        mpEmissiveSampler->bindShaderData(var[name]["emissiveSampler"]);
        var[name]["useEmissiveLights"] = true;
    }
    if (mpScene->getLightCount() > 0)
    {
        var[name]["useNEE"] = mUseNEE;
        var[name]["useAnalyticLights"] = true;
    }

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    mpSampleRHS->execute(pRenderContext, uint3(mBatchSize * mNumRHS, 1, 1));
}

void NeuralRadiosity::resolveRHS(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "ResolveRHS");

    ShaderVar var = mpResolveRHS->getRootVar();
    const auto name = "gResolveRHS";
    
    bindRayBatchData(var, mpTrainLHSBatch, name);
    var["rhsColor"] = mpTrainRHSBatch->color;

    mpResolveRHS->execute(pRenderContext, uint3(mBatchSize, 1, 1));
}

void NeuralRadiosity::compactBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "CompactBatch");

    // Prefix sum from active mask to index
    mpPrefixSum->execute(
        pRenderContext,
        pRayBatch->diffIndex,
        pRayBatch->size,
        &pRayBatch->diffSize
    );

    mpPrefixSum->execute(
        pRenderContext,
        pRayBatch->specIndex,
        pRayBatch->size,
        &pRayBatch->specSize
    );

    ShaderVar var = mpCompactBatch->getRootVar();
    const auto name = "gCompactBatch";
    bindRayBatchData(var, pRayBatch, name);

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    mpCompactBatch->execute(pRenderContext, uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::resolveBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ResolveBatch");

    ShaderVar var = mpResolveBatch->getRootVar();
    bindRayBatchData(var, pRayBatch, "gResolveBatch");

    mpResolveBatch->execute(pRenderContext, uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::coneTrace(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ConeTrace");

    ShaderVar var = mpConeTrace->getRootVar();
    const auto name = "gConeTrace";
    // Bind parameters
    var[name]["batchSize"] = pRayBatch->size;
    var[name]["batchCount"] = mFrameCount;
    var[name]["diffSize"] = pRayBatch->diffSize;
    var[name]["specSize"] = pRayBatch->specSize;

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    var["specPixel"] = pRayBatch->specPixel;
    var["specPos"] = pRayBatch->specPos;
    var["specDir"] = pRayBatch->specDir;
    var["specRoughness"] = pRayBatch->specRoughness;
    var["specVBuffer"] = pRayBatch->specVBuffer;
    var["specInput"] = pRayBatch->specInput;
    // var["temporalHistory"] = mpRenderSpecTemporalHistory;
    var[name]["enableTemporalHistory"] = (pRayBatch == mpRenderBatch && mEnableRenderSpecTemporalReuse) ? 1u : 0u;
    var[name]["useTemporalReuse"] = (pRayBatch == mpRenderBatch && mEnableRenderSpecTemporalReuse && mRenderSpecTemporalHistoryValid) ? 1u : 0u;
    var[name]["coneTraceReuseMode"] = (uint32_t)mConeTraceReuseMode;

    // Debug: per-spec-pixel grid level visualization and dump. Bind gridScales
    // array and output buffer when either mode is active (both need the buffer
    // written by ConeTrace; shader defines guard the actual writes).
    if ((mDebugShowSpecLevel || mDebugDumpRequested) && mpNRModel && mpSpecLevelBuffer)
    {
        const std::vector<float> scales = mpNRModel->getClusterGridScales();
        const uint32_t n = std::min<uint32_t>((uint32_t)scales.size(), 64u);
        float scalesArr[64];
        for (uint32_t i = 0; i < n; ++i) scalesArr[i] = scales[i];
        var[name]["gridScales"].setBlob(scalesArr, n * sizeof(float));
        var["gSpecLevel"] = mpSpecLevelBuffer;
    }

    // Debug dump: bind per-spec-pixel debug buffers. The shader only writes
    // to these when DEBUG_DUMP is defined (one-shot, see dumpDebugImages).
    if (mDebugDumpRequested)
    {
        if (mpSpecLevelBuffer)          var["gSpecLevel"]          = mpSpecLevelBuffer;
        if (mpSpecTanLobeBuffer)        var["gSpecTanLobe"]        = mpSpecTanLobeBuffer;
        if (mpSpecClusterMeanBuffer)    var["gSpecClusterMean"]    = mpSpecClusterMeanBuffer;
        if (mpSpecClusterScaleBuffer)   var["gSpecClusterScale"]   = mpSpecClusterScaleBuffer;
        if (mpSpecClusterWeightBuffer)  var["gSpecClusterWeight"]  = mpSpecClusterWeightBuffer;
        if (mpSpecClusterPosXBuffer)    var["gSpecClusterPosX"]    = mpSpecClusterPosXBuffer;
        if (mpSpecClusterPosYBuffer)    var["gSpecClusterPosY"]    = mpSpecClusterPosYBuffer;
        if (mpSpecClusterPosZBuffer)    var["gSpecClusterPosZ"]    = mpSpecClusterPosZBuffer;
        if (mpSpecMaterialDataBuffer)   var["gSpecMaterialData"]   = mpSpecMaterialDataBuffer;
    }

    var["gPrefiltEnv"]["mipCount"] = mpPrefiltEnvCubemap ? mpPrefiltEnvCubemap->getMipCount() : 0u;
    var["gPrefiltEnv"]["faceSize"] = mpPrefiltEnvCubemap ? mpPrefiltEnvCubemap->getWidth() : 0u;
    var["gPrefiltEnv"]["valid"] = mpPrefiltEnvCubemap ? 1u : 0u;
    var["gPrefiltEnv"]["fallbackValid"] = (mpScene && mpScene->useEnvLight()) ? 1u : 0u;
    var["gPrefiltEnv"]["cubemap"].setTexture(mpPrefiltEnvCubemap);
    var["gPrefiltEnv"]["sampler"].setSampler(mpPrefiltEnvSampler);

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    mpConeTrace->execute(pRenderContext, uint3(pRayBatch->specSize * mNumSpecRays, 1, 1));

    if (pRayBatch == mpRenderBatch && mEnableRenderSpecTemporalReuse)
    {
        mRenderSpecTemporalHistoryValid = true;
    }
}

void NeuralRadiosity::prepareDebugDumpResources()
{
    const uint32_t pixelCount = mFrameDim.x * mFrameDim.y;

    // Per-spec-pixel debug buffers (sized to screen pixel count, indexed by pixIdx).
    auto ensureBuffer = [&](ref<Buffer>& buf, size_t bytes)
    {
        if (!buf || buf->getSize() != bytes)
        {
            buf = mpDevice->createBuffer(
                bytes,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr);
        }
    };
    ensureBuffer(mpSpecLevelBuffer,         pixelCount * sizeof(float));
    ensureBuffer(mpSpecTanLobeBuffer,       pixelCount * sizeof(float));
    ensureBuffer(mpSpecClusterMeanBuffer,   pixelCount * sizeof(float4));
    ensureBuffer(mpSpecClusterScaleBuffer,  pixelCount * sizeof(float4));
    ensureBuffer(mpSpecClusterWeightBuffer, pixelCount * sizeof(float4));
    ensureBuffer(mpSpecClusterPosXBuffer,   pixelCount * sizeof(float4));
    ensureBuffer(mpSpecClusterPosYBuffer,   pixelCount * sizeof(float4));
    ensureBuffer(mpSpecClusterPosZBuffer,   pixelCount * sizeof(float4));
    ensureBuffer(mpSpecMaterialDataBuffer,  pixelCount * sizeof(float2));

    // Output textures for CPU readback.
    auto ensureTex = [&](ref<Texture>& tex, ResourceFormat fmt)
    {
        if (!tex || tex->getWidth() != mFrameDim.x || tex->getHeight() != mFrameDim.y || tex->getFormat() != fmt)
        {
            tex = mpDevice->createTexture2D(
                mFrameDim.x, mFrameDim.y, fmt, 1, 1, nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        }
    };
    ensureTex(mpDumpSpecLevelTex,      ResourceFormat::RGBA32Float);
    ensureTex(mpDumpTanLobeTex,        ResourceFormat::RGBA32Float);
    ensureTex(mpDumpClusterMeanTex,    ResourceFormat::RGBA32Float);
    ensureTex(mpDumpClusterScaleTex,   ResourceFormat::RGBA32Float);
    ensureTex(mpDumpClusterWeightTex,  ResourceFormat::RGBA32Float);
    ensureTex(mpDumpClusterPosXTex,    ResourceFormat::RGBA32Float);
    ensureTex(mpDumpClusterPosYTex,    ResourceFormat::RGBA32Float);
    ensureTex(mpDumpClusterPosZTex,    ResourceFormat::RGBA32Float);
    ensureTex(mpDumpMaterialDataTex,   ResourceFormat::RGBA32Float);
}

void NeuralRadiosity::dumpDebugImages(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpDebugDumpPass || !mpRenderBatch)
    {
        logWarning("NeuralRadiosity::dumpDebugImages: debug dump pass or render batch not ready.");
        return;
    }

    prepareDebugDumpResources();

    // Run DebugDumpPass: copy per-spec-pixel debug buffers into screen-sized textures.
    {
        ShaderVar var = mpDebugDumpPass->getRootVar();
        const auto name = "gDebugDumpPass";
        var[name]["frameDim"] = mFrameDim;

        var["gSpecActive"]        = mpRenderBatch->specActive;
        var["gSpecIndex"]         = mpRenderBatch->specIndex;
        var["gSpecLevel"]         = mpSpecLevelBuffer;
        var["gSpecTanLobe"]       = mpSpecTanLobeBuffer;
        var["gSpecClusterMean"]   = mpSpecClusterMeanBuffer;
        var["gSpecClusterScale"]  = mpSpecClusterScaleBuffer;
        var["gSpecClusterWeight"] = mpSpecClusterWeightBuffer;
        var["gSpecClusterPosX"]   = mpSpecClusterPosXBuffer;
        var["gSpecClusterPosY"]   = mpSpecClusterPosYBuffer;
        var["gSpecClusterPosZ"]   = mpSpecClusterPosZBuffer;
        var["gSpecMaterialData"]  = mpSpecMaterialDataBuffer;

        var["gDumpSpecLevel"]      = mpDumpSpecLevelTex;
        var["gDumpTanLobe"]        = mpDumpTanLobeTex;
        var["gDumpClusterMean"]    = mpDumpClusterMeanTex;
        var["gDumpClusterScale"]   = mpDumpClusterScaleTex;
        var["gDumpClusterWeight"]  = mpDumpClusterWeightTex;
        var["gDumpClusterPosX"]    = mpDumpClusterPosXTex;
        var["gDumpClusterPosY"]    = mpDumpClusterPosYTex;
        var["gDumpClusterPosZ"]    = mpDumpClusterPosZTex;
        var["gDumpMaterialData"]   = mpDumpMaterialDataTex;

        mpDebugDumpPass->execute(pRenderContext, uint3(mFrameDim, 1));
    }

    // Save all debug images to disk. Output dir: debug_dump_<frame>/.
    const std::filesystem::path outDir = fmt::format("debug_dump_{:05d}", mFrameCount);
    std::filesystem::create_directories(outDir);

    auto saveTex = [&](const ref<Texture>& tex, const std::string& filename, Bitmap::FileFormat fmt)
    {
        if (!tex) return;
        const auto path = outDir / filename;
        tex->captureToFile(0, 0, path, fmt, Bitmap::ExportFlags::None, false);
        logInfo("[NCR debug dump] saved {}", path.string());
    };

    // HDR float images as EXR.
    saveTex(mpDumpSpecLevelTex,      "spec_level.exr",       Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpTanLobeTex,        "tan_lobe.exr",         Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpClusterMeanTex,    "cluster_mean.exr",     Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpClusterScaleTex,   "cluster_scale.exr",    Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpClusterWeightTex,  "cluster_weight.exr",   Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpClusterPosXTex,    "cluster_pos_x.exr",    Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpClusterPosYTex,    "cluster_pos_y.exr",    Bitmap::FileFormat::ExrFile);
    saveTex(mpDumpClusterPosZTex,    "cluster_pos_z.exr",    Bitmap::FileFormat::ExrFile);
    // Material data: r = material ID (as float), g = roughness.
    saveTex(mpDumpMaterialDataTex,   "material_data.exr",    Bitmap::FileFormat::ExrFile);

    // Also save the existing GBuffer channels (pos, normal, albedo) for context.
    const auto& pPos    = renderData.getTexture(kOutputPos);
    const auto& pNormal = renderData.getTexture(kOutputNormal);
    const auto& pAlbedo = renderData.getTexture(kOutputAlbedo);
    if (pPos)    pPos->captureToFile(0, 0, outDir / "gbuffer_pos.exr",    Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    if (pNormal) pNormal->captureToFile(0, 0, outDir / "normal.exr",      Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    if (pAlbedo) pAlbedo->captureToFile(0, 0, outDir / "albedo.exr",      Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    logInfo("[NCR debug dump] saved gbuffer_pos.exr, normal.exr, albedo.exr");

    logInfo("[NCR debug dump] all images saved to '{}/'", outDir.string());
}

void NeuralRadiosity::modelInferenceCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ModelInferenceCUDA");

    pRenderContext->waitForFalcor(mpNRModel->stream(false));
    mpNRModel->inference(pRayBatch->getDiffPtrs(), pRayBatch->getSpecPtrs());
    pRenderContext->waitForCuda(mpNRModel->stream(false));
}

void NeuralRadiosity::modelTrainCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ModelTrainCUDA");

    ModelIOPtrs diffPtrs = pRayBatch->getDiffPtrs();
    ModelIOPtrs specPtrs = pRayBatch->getSpecPtrs();

    // Debug freeze: force size to 0 so NRModel::train skips forward/backward/optimizer
    // step for the frozen network. Inference is unaffected (uses the other stream).
    if (mDisableDiffUpdate) diffPtrs.size = 0;
    if (mDisableSpecUpdate) specPtrs.size = 0;

    pRenderContext->waitForFalcor(mpNRModel->stream(true));
    mpNRModel->train(diffPtrs, specPtrs);
    pRenderContext->waitForCuda(mpNRModel->stream(true));
}

void NeuralRadiosity::updateFrameDim(const uint2 frameDim)
{
    FALCOR_ASSERT(frameDim.x > 0 && frameDim.y > 0);

    if (any(mFrameDim != frameDim))
    {
        mVarsChanged = true;
    }
    mFrameDim = frameDim;
}

void NeuralRadiosity::updatePrograms(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mVarsChanged) return;
    const auto startTime = std::chrono::steady_clock::now();

    // On any var change, release existing compute passes so they get rebuilt
    // with the updated DefineList (e.g. DEBUG_RENDER_DIFFUSE_ONLY). Without this,
    // passes are created once and never pick up new shader defines.
    mpFirstSmoothPass = nullptr;
    mpCompactPass = nullptr;
    mpResolvePass = nullptr;
    mpPostFilterPass = nullptr;
    mpPostFXAAPass = nullptr;
    mpRandomSmooth = nullptr;
    mpSampleRHS = nullptr;
    mpResolveRHS = nullptr;
    mpCompactBatch = nullptr;
    mpResolveBatch = nullptr;
    mpConeTrace = nullptr;
    mpPrefiltEnvInitPass = nullptr;
    mpPrefiltEnvDownsamplePass = nullptr;
    mpDebugDumpPass = nullptr;

    if (!mpPrefixSum)
    {
        mpPrefixSum = std::make_unique<PrefixSum>(mpDevice);
    }

    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(mpScene->getTypeConformances());

    DefineList defines;
    defines.add(mpScene->getSceneDefines());
    defines.add(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler)
    {
        defines.add(mpEmissiveSampler->getDefines());
    }
    defines.add(getShaderDefines(renderData));

    if (!mpPrefiltEnvInitPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrefiltEnvCubemapFile).csEntry("init");

        DefineList buildDefines = defines;
        buildDefines.add("PREFILT_ENV_CUBEMAP_BUILD_PASS", "1");
        mpPrefiltEnvInitPass = ComputePass::create(mpDevice, desc, buildDefines, true);

        ShaderVar var = mpPrefiltEnvInitPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
    }

    if (!mpPrefiltEnvDownsamplePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPrefiltEnvCubemapFile).csEntry("downsample");

        DefineList buildDefines = defines;
        buildDefines.add("PREFILT_ENV_CUBEMAP_BUILD_PASS", "1");
        mpPrefiltEnvDownsamplePass = ComputePass::create(mpDevice, desc, buildDefines, true);
    }

    if (!mpFirstSmoothPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kFirstSmoothPassFile).csEntry("main");

        mpFirstSmoothPass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpFirstSmoothPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpCompactPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCompactPassFile).csEntry("main");

        mpCompactPass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpCompactPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpResolvePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kResolvePassFile).csEntry("main");

        mpResolvePass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpResolvePass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpPostFilterPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPostFilterPassFile).csEntry("main");

        mpPostFilterPass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpPostFilterPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpPostFXAAPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPostFilterPassFile).csEntry("mainFXAA");

        mpPostFXAAPass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpPostFXAAPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    const auto passEndTime = std::chrono::steady_clock::now();
    const auto passElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(passEndTime - startTime).count();
    logInfo("Screen passes created: \t{} ms", passElapsedMs);

    if (!mpRandomSmooth)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kRandomSmoothFile).csEntry("main");

        mpRandomSmooth = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpRandomSmooth->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpSampleRHS)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSampleRHSFile).csEntry("main");

        mpSampleRHS = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpSampleRHS->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpResolveRHS)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kResolveRHSFile).csEntry("main");

        mpResolveRHS = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpResolveRHS->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpCompactBatch)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCompactBatchFile).csEntry("main");

        mpCompactBatch = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpCompactBatch->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpResolveBatch)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kResolveBatchFile).csEntry("main");

        mpResolveBatch = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpResolveBatch->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpConeTrace)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kConeTraceFile).csEntry("main");

        mpConeTrace = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpConeTrace->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpDebugDumpPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kDebugDumpPassFile).csEntry("main");

        mpDebugDumpPass = ComputePass::create(mpDevice, desc, defines, true);

        ShaderVar var = mpDebugDumpPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
    }

    const auto batchEndTime = std::chrono::steady_clock::now();
    const auto batchElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(batchEndTime - passEndTime).count();
    logInfo("Batch passes created: \t{} ms", batchElapsedMs);

    if (!mpNRModel)
    {
        mpNRModel = std::make_unique<NRModel>();
        if (mEnableHashCollisionStats)
        {
            mpNRModel->setCollisionStatsEnabled(true);
        }
        // Propagate per-optimizer freeze switches to the new model.
        mpNRModel->setFreezePrimGridOpt(mFreezePrimGridOpt);
        mpNRModel->setFreezeClsGridOpt(mFreezeClsGridOpt);
        mpNRModel->setFreezeDiffNetOpt(mFreezeDiffNetOpt);
        mpNRModel->setFreezeSpecNetOpt(mFreezeSpecNetOpt);

        // Sync training parameter UI mirrors to the new model.
        mpNRModel->setPrimGridLr(mParamPrimGridLr);
        mpNRModel->setClsGridLr(mParamClsGridLr);
        mpNRModel->setDiffNetLr(mParamDiffNetLr);
        mpNRModel->setSpecNetLr(mParamSpecNetLr);
        mpNRModel->setPrimGridEmaDecay(mParamPrimGridEmaDecay);
        mpNRModel->setClsGridEmaDecay(mParamClsGridEmaDecay);
        mpNRModel->setDiffNetEmaDecay(mParamDiffNetEmaDecay);
        mpNRModel->setSpecNetEmaDecay(mParamSpecNetEmaDecay);
        mpNRModel->setInterpRatio(mParamInterpRatio);

        if (!mLoadModelPath.empty())
        {
            logInfo("Auto-loading model state from '{}'", mLoadModelPath);
            try {
                mpNRModel->loadState(mLoadModelPath);
                mpNRModel->setOnline(true);
                logInfo("Model state loaded successfully.");
            } catch (const std::exception& e) {
                logError("Failed to load model state: {}", e.what());
            }
        }
    }

    const auto modelEndTime = std::chrono::steady_clock::now();
    const auto modelElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(modelEndTime - batchEndTime).count();
    logInfo("NRModel created: \t{} ms", modelElapsedMs);
    logInfo("NeuralRadiosity::updatePrograms() took {} ms", passElapsedMs + batchElapsedMs + modelElapsedMs);
}

void NeuralRadiosity::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mVarsChanged) return;
    const auto startTime = std::chrono::steady_clock::now();

    ShaderVar var = mpRandomSmooth->getRootVar();

    if (!mpRenderBatch)
        mpRenderBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mFrameDim.x * mFrameDim.y, mNumClusters, "RenderBatch", false);
    if (!mpTrainLHSBatch)
        mpTrainLHSBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mBatchSize, mNumClusters, "LHSBatch", true);
    if (!mpTrainRHSBatch)
        mpTrainRHSBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mBatchSize * mNumRHS, mNumClusters, "RHSBatch", true);

    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        mpRenderBatch->resize(mFrameDim.x * mFrameDim.y, mNumClusters);
    }
    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
    {
        mpTrainLHSBatch->resize(mBatchSize, mNumClusters);
        mpTrainRHSBatch->resize(mBatchSize * mNumRHS, mNumClusters);
    }

    // const uint32_t historySize = mpRenderBatch ? mpRenderBatch->size : 0u;
    // if (historySize > 0 && (!mpRenderSpecTemporalHistory || mpRenderSpecTemporalHistory->getElementCount() != historySize))
    // {
    //     ShaderVar coneTraceVar = mpConeTrace->getRootVar();
    //     mpRenderSpecTemporalHistory = mpDevice->createStructuredBuffer(
    //         coneTraceVar["temporalHistory"],
    //         historySize,
    //         ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
    //         MemoryType::DeviceLocal,
    //         nullptr,
    //         false
    //     );
    //     clearRenderTemporalHistory(pRenderContext);
    //     mHasRenderCameraViewProj = false;
    // }

    if (!mpRenderTempColor ||
        mpRenderTempColor->getWidth() != mFrameDim.x ||
        mpRenderTempColor->getHeight() != mFrameDim.y)
    {
        mpRenderTempColor = mpDevice->createTexture2D(
            mFrameDim.x, mFrameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }

    if (!mpPostFilterTempColor ||
        mpPostFilterTempColor->getWidth() != mFrameDim.x ||
        mpPostFilterTempColor->getHeight() != mFrameDim.y)
    {
        mpPostFilterTempColor = mpDevice->createTexture2D(
            mFrameDim.x, mFrameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }

    // Debug: per-spec-pixel dominant grid level (one float per pixel).
    const uint32_t specLevelCount = mFrameDim.x * mFrameDim.y;
    if (mDebugShowSpecLevel &&
        (!mpSpecLevelBuffer || mpSpecLevelBuffer->getSize() != specLevelCount * sizeof(float)))
    {
        mpSpecLevelBuffer = mpDevice->createBuffer(
            specLevelCount * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr);
    }

    // Debug dump: allocate per-spec-pixel debug buffers when dump is requested.
    if (mDebugDumpRequested)
    {
        prepareDebugDumpResources();
    }

    const auto endTime = std::chrono::steady_clock::now();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    logInfo("NeuralRadiosity::prepareResources() took {} ms", elapsedMs);

    NCR_LOG(this, "[NCR prepareResources] frameDim={}x={} renderBatchSize={} trainLHS={} trainRHS={} numClusters={}",
            mFrameDim.x, mFrameDim.y,
            mpRenderBatch ? mpRenderBatch->size : 0u,
            mpTrainLHSBatch ? mpTrainLHSBatch->size : 0u,
            mpTrainRHSBatch ? mpTrainRHSBatch->size : 0u,
            mNumClusters);
}

void NeuralRadiosity::updateRenderTemporalHistory(RenderContext* pRenderContext)
{
    if (!mpScene || !mpRenderSpecTemporalHistory) return;

    const auto& pCamera = mpScene->getCamera();
    if (!pCamera) return;

    const float4x4 viewProj = pCamera->getViewProjMatrix();
    
    if (!mHasRenderCameraViewProj || std::memcmp(&viewProj, &mRenderCameraViewProj, sizeof(float4x4)) != 0)
    {
        clearRenderTemporalHistory(pRenderContext);
        mRenderCameraViewProj = viewProj;
        mHasRenderCameraViewProj = true;
    }
}

void NeuralRadiosity::clearRenderTemporalHistory(RenderContext* pRenderContext)
{
    if (!mpRenderSpecTemporalHistory) return;

    pRenderContext->clearUAV(mpRenderSpecTemporalHistory->getUAV().get(), float4(0.f));
    mRenderSpecTemporalHistoryValid = false;
}

bool NeuralRadiosity::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getILightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getILightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);

            mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));

            lightingChanged = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
    }

    // Verbose: emissive triangle count + sampler state.
    if (mpScene && mpScene->useEmissiveLights())
    {
        const auto& pLights = mpScene->getILightCollection(pRenderContext);
        const uint32_t activeTris = pLights ? pLights->getActiveLightCount(pRenderContext) : 0u;
        NCR_LOG(this, "[NCR prepareLighting] useEmissiveLights=true, activeEmissiveTris={}, samplerCreated={}",
                activeTris, mpEmissiveSampler ? 1 : 0);
    }
    else
    {
        NCR_LOG(this, "[NCR prepareLighting] useEmissiveLights=false (no emissive sampler)");
    }

    return lightingChanged;
}

void NeuralRadiosity::preparePrefiltEnvCubemap(RenderContext* pRenderContext)
{
    if (!mPrefiltEnvDirty)
        return;

    if (!mpScene || !mpScene->useEnvLight() || !mpScene->getEnvMap() || !mpScene->getEnvMap()->getEnvMap())
    {
        mpPrefiltEnvCubemap = nullptr;
        mPrefiltEnvDirty = false;
        NCR_LOG(this, "[NCR preparePrefiltEnvCubemap] no envMap, cubemap=nullptr");
        return;
    }

    const auto& pEnvMapTexture = mpScene->getEnvMap()->getEnvMap();
    const uint32_t faceSize = computePrefiltEnvFaceSize(pEnvMapTexture);
    const uint32_t mipCount = computeMipCount(faceSize);

    if (!mpPrefiltEnvCubemap ||
        mpPrefiltEnvCubemap->getWidth() != faceSize ||
        mpPrefiltEnvCubemap->getHeight() != faceSize ||
        mpPrefiltEnvCubemap->getMipCount() != mipCount)
    {
        mpPrefiltEnvCubemap = mpDevice->createTexture2D(
            faceSize,
            faceSize,
            ResourceFormat::RGBA32Float,
            6,
            mipCount,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        mpPrefiltEnvCubemap->setName("NeuralRadiosity.PrefiltEnvCubemap");
    }

    if (!mpPrefiltEnvSampler)
    {
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpPrefiltEnvSampler = mpDevice->createSampler(desc);
    }

    {
        ShaderVar var = mpPrefiltEnvInitPass->getRootVar();
        ShaderVar passVar = var["gPrefiltEnvCubemapPass"];

        passVar["outputDim"] = uint2(faceSize, faceSize);
        passVar["output"].setUav(mpPrefiltEnvCubemap->getUAV(0, 0, 6));
        mpScene->bindShaderData(var["gScene"]);

        mpPrefiltEnvInitPass->execute(pRenderContext, uint3(faceSize, faceSize, 6));
        pRenderContext->uavBarrier(mpPrefiltEnvCubemap.get());
    }

    for (uint32_t mip = 1; mip < mipCount; ++mip)
    {
        const uint32_t mipWidth = mpPrefiltEnvCubemap->getWidth(mip);
        const uint32_t mipHeight = mpPrefiltEnvCubemap->getHeight(mip);

        ShaderVar var = mpPrefiltEnvDownsamplePass->getRootVar();
        ShaderVar passVar = var["gPrefiltEnvCubemapPass"];

        passVar["outputDim"] = uint2(mipWidth, mipHeight);
        passVar["source"].setSrv(mpPrefiltEnvCubemap->getSRV(mip - 1, 1, 0, 6));
        passVar["sampler"].setSampler(mpPrefiltEnvSampler);
        passVar["output"].setUav(mpPrefiltEnvCubemap->getUAV(mip, 0, 6));

        mpPrefiltEnvDownsamplePass->execute(pRenderContext, uint3(mipWidth, mipHeight, 6));
        pRenderContext->uavBarrier(mpPrefiltEnvCubemap.get());
    }

    logInfo("NeuralRadiosity: prefiltered environment cubemap is {}x{} with {} mip levels.", faceSize, faceSize, mipCount);
    NCR_LOG(this, "[NCR preparePrefiltEnvCubemap] cubemap={}x{} mipCount={} dirtyResolved",
            faceSize, faceSize, mipCount);
    mPrefiltEnvDirty = false;
}

void NeuralRadiosity::bindScreenData(ShaderVar& var, const RenderData& renderData, const std::string& name)
{
    // Bind parameters
    var[name]["frameDim"] = mFrameDim;
    var[name]["frameCount"] = mFrameCount;
    // Bind input & output textures
    for (const auto& channel : kInputChannels)
        var[channel.texname] = renderData.getTexture(channel.name);
    for (const auto& channel : kOutputChannels)
        var[channel.texname] = renderData.getTexture(channel.name);
}

void NeuralRadiosity::bindRayBatchData(ShaderVar& var, std::shared_ptr<RayBatchBuffer> pRayBatch, const std::string &name)
{
    // Bind parameters
    var[name]["batchSize"] = pRayBatch->size;
    var[name]["batchCount"] = mFrameCount;
    var[name]["diffSize"] = pRayBatch->diffSize;
    var[name]["specSize"] = pRayBatch->specSize;
    
    // Bind buffers
    var["allPos"] = pRayBatch->pos;
    var["allDir"] = pRayBatch->dir;
    var["allNormal"] = pRayBatch->normal;
    var["allAlbedo"] = pRayBatch->albedo;
    var["allRoughness"] = pRayBatch->roughness;
    var["allVBuffer"] = pRayBatch->vbuffer;
    var["allColor"] = pRayBatch->color;
    var["allEmission"] = pRayBatch->emission;

    // Bind diffuse buffers
    var["diffActive"] = pRayBatch->diffActive;
    var["diffIndex"] = pRayBatch->diffIndex;
    var["diffInput"] = pRayBatch->diffInput;
    var["diffColor"] = pRayBatch->diffColor;

    // Bind specular buffers
    var["specActive"] = pRayBatch->specActive;
    var["specIndex"] = pRayBatch->specIndex;
    var["specInput"] = pRayBatch->specInput;
    var["specColor"] = pRayBatch->specColor;
    var["specVBuffer"] = pRayBatch->specVBuffer;
    var["specPixel"] = pRayBatch->specPixel;
    var["specPos"] = pRayBatch->specPos;
    var["specDir"] = pRayBatch->specDir;
    var["specRoughness"] = pRayBatch->specRoughness;
}

DefineList NeuralRadiosity::getShaderDefines(const RenderData& renderData) const
{
    DefineList defines;

    defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
    defines.add("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mUseEnvLight ? "1" : "0");
    defines.add("SPECULAR_ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    // Add cone trace parameters
    defines.add("NUM_KMEANS_ITERS", std::to_string(mNumKMeansIters));
    defines.add("NUM_SPEC_RAYS", std::to_string(mNumSpecRays));
    defines.add("NUM_CLUSTERS", std::to_string(mNumClusters));
    // Add sample RHS parameters
    defines.add("NUM_RHS", std::to_string(mNumRHS));
    // Add random smooth parameters
    defines.add("RS_RR_PROB", std::to_string(mRSRRProb));
    // Debug render-isolation switches
    defines.add("DEBUG_RENDER_DIFFUSE_ONLY", mDebugRenderDiffuseOnly ? "1" : "0");
    defines.add("DEBUG_RENDER_SPECULAR_ONLY", mDebugRenderSpecularOnly ? "1" : "0");
    // Debug: per-spec-pixel grid level visualization (red=coarse, blue=fine).
    defines.add("DEBUG_SHOW_SPEC_LEVEL", mDebugShowSpecLevel ? "1" : "0");
    defines.add("N_GRID_LEVELS", mpNRModel ? std::to_string(mpNRModel->getNumLevels()) : "6");
    // Debug dump: enables per-spec-pixel debug buffer writes in ConeTrace.
    // This is a one-shot flag — we rebuild shaders with DEBUG_DUMP=1 for one
    // frame, then rebuild with DEBUG_DUMP=0 to restore normal behavior.
    defines.add("DEBUG_DUMP", mDebugDumpRequested ? "1" : "0");
    // Setup ray flags.
    RayFlags rayFlags = RayFlags::None;
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front)
        rayFlags = RayFlags::CullFrontFacingTriangles;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back)
        rayFlags = RayFlags::CullBackFacingTriangles;
    defines.add("RAY_FLAGS", std::to_string((uint32_t)rayFlags));

    // Set 'is_valid_<name>' defines to inform the program of which ones it can access.
    defines.add(getValidResourceDefines(kInputChannels, renderData));
    defines.add(getValidResourceDefines(kOutputChannels, renderData));
    return defines;
}

bool NeuralRadiosity::loadTrainCamerasFromFile(const std::filesystem::path& path)
{
    try
    {
        std::ifstream ifs(path);
        if (!ifs)
        {
            FALCOR_THROW("Failed to open file '{}'.", path.string());
        }

        const nlohmann::json json = nlohmann::json::parse(ifs);
        if (!json.is_array())
        {
            FALCOR_THROW("Camera file '{}' must contain a JSON array.", path.string());
        }

        std::vector<ref<Camera>> cameras;
        cameras.reserve(json.size());

        for (size_t i = 0; i < json.size(); ++i)
        {
            const auto& entry = json.at(i);
            if (!entry.is_object())
            {
                FALCOR_THROW("Camera entry {} must be a JSON object.", i);
            }

            auto camera = Camera::create(fmt::format("train_camera_{}", i));
            camera->setPosition(parseFloat3(entry.at("position"), "position"));
            camera->setTarget(parseFloat3(entry.at("target"), "target"));
            camera->setUpVector(parseFloat3(entry.at("up"), "up"));
            camera->setFocalLength(entry.at("focalLength").get<float>());
            camera->setFocalDistance(entry.at("focalDistance").get<float>());
            camera->setApertureRadius(entry.at("apertureRadius").get<float>());

            if (entry.contains("aspectRatio")) camera->setAspectRatio(entry.at("aspectRatio").get<float>());
            if (entry.contains("nearPlane")) camera->setNearPlane(entry.at("nearPlane").get<float>());
            if (entry.contains("farPlane")) camera->setFarPlane(entry.at("farPlane").get<float>());

            cameras.push_back(camera);
        }

        mTrainCameras = std::move(cameras);
        mTrainCameraPath = path;
        logInfo("Loaded {} training cameras from '{}'.", mTrainCameras.size(), path.string());
        return true;
    }
    catch (const std::exception& e)
    {
        logError("Failed to load training cameras from '{}': {}", path.string(), e.what());
        return false;
    }
}

void NeuralRadiosity::setCamera(uint32_t cameraIdx)
{
    if (!mpScene)
    {
        logWarning("NeuralRadiosity::setCamera() called without an active scene.");
        return;
    }

    if (cameraIdx >= mTrainCameras.size())
    {
        logWarning("NeuralRadiosity::setCamera() camera index {} is out of range ({} cameras loaded).", cameraIdx, mTrainCameras.size());
        return;
    }

    const auto& pSrcCamera = mTrainCameras[cameraIdx];
    const auto& pDstCamera = mpScene->getCamera();
    if (!pSrcCamera || !pDstCamera)
    {
        logWarning("NeuralRadiosity::setCamera() source or destination camera is null.");
        return;
    }

    pDstCamera->setPosition(pSrcCamera->getPosition());
    pDstCamera->setTarget(pSrcCamera->getTarget());
    pDstCamera->setUpVector(pSrcCamera->getUpVector());
    pDstCamera->setAspectRatio(pSrcCamera->getAspectRatio());
    pDstCamera->setFocalLength(pSrcCamera->getFocalLength());
    pDstCamera->setFrameHeight(pSrcCamera->getFrameHeight());
    pDstCamera->setFocalDistance(pSrcCamera->getFocalDistance());
    pDstCamera->setApertureRadius(pSrcCamera->getApertureRadius());
    pDstCamera->setShutterSpeed(pSrcCamera->getShutterSpeed());
    pDstCamera->setISOSpeed(pSrcCamera->getISOSpeed());
    pDstCamera->setNearPlane(pSrcCamera->getNearPlane());
    pDstCamera->setFarPlane(pSrcCamera->getFarPlane());
}

void NeuralRadiosity::setConeParameters(bool train)
{
    if (train)
    {
        if (mNumSpecRays != NUM_SPEC_RAYS_TRAIN || mNumKMeansIters != NUM_KMEANS_ITERS_TRAIN)
        {
            mNumSpecRays = NUM_SPEC_RAYS_TRAIN;
            mNumKMeansIters = NUM_KMEANS_ITERS_TRAIN;
            mpConeTrace = nullptr;
        }
        updateAdaptiveRHSState(0, true);
    }
    else
    {
        if (mNumSpecRays != NUM_SPEC_RAYS_RENDER || mNumKMeansIters != NUM_KMEANS_ITERS_RENDER)
        {
            mNumSpecRays = NUM_SPEC_RAYS_RENDER;
            mNumKMeansIters = NUM_KMEANS_ITERS_RENDER;
            mpConeTrace = nullptr;
        }
    }
}

uint32_t NeuralRadiosity::getAdaptiveRHSStage(uint32_t completedSteps) const
{
    if (!mAdaptiveRHS || mTotalTrainSteps == 0) return 0;

    uint32_t stage = 0;
    for (uint32_t quarter = 1; quarter <= 3; ++quarter)
    {
        const uint32_t threshold = static_cast<uint32_t>(((uint64_t)mTotalTrainSteps * quarter + 3ull) / 4ull);
        if (completedSteps >= threshold) ++stage;
    }

    return stage;
}

void NeuralRadiosity::updateAdaptiveRHSState(uint32_t completedSteps, bool force)
{
    const uint32_t stage = getAdaptiveRHSStage(completedSteps);
    if (!force && stage == mAdaptiveRHSStage) return;

    mAdaptiveRHSStage = stage;

    const uint32_t targetNumRHS = mNumRHSInit << stage;
    const uint32_t targetBatchSize = std::max(1u, mBatchSizeInit >> stage);
    const bool rhsChanged = targetNumRHS != mNumRHS;
    const bool batchChanged = targetBatchSize != mBatchSize;

    mNumRHS = targetNumRHS;
    mBatchSize = targetBatchSize;

    if (mpTrainLHSBatch) mpTrainLHSBatch->size = mBatchSize;

    if (rhsChanged)
    {
        mpSampleRHS = nullptr;
        mpResolveRHS = nullptr;
    }

    if (rhsChanged || batchChanged)
    {
        mVarsChanged = true;
    }
}

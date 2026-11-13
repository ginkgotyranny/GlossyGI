#include "NeuralRadiosity.h"

#include "Utils/Math/VectorJson.h"

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
    const std::string kOutputPrimaryPos = "primaryPos";
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
        { kOutputPrimaryPos,    "gPrimaryPos",  "Primary hit position (xyz), valid flag (w)", false, ResourceFormat::RGBA32Float },
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
    const std::string kSpecPrimaryFile = "RenderPasses/NeuralRadiosity/SpecPrimary.cs.slang";
    const std::string kClusterReprojectFile = "RenderPasses/NeuralRadiosity/ClusterReproject.cs.slang";
    const std::string kConeTraceScreenFile = "RenderPasses/NeuralRadiosity/ConeTraceScreen.cs.slang";
    const std::string kClusterSpatialReuseFile = "RenderPasses/NeuralRadiosity/ClusterSpatialReuse.cs.slang";
    const std::string kPrefiltEnvCubemapFile = "RenderPasses/NeuralRadiosity/PrefiltEnvCubemap.slang";

    /// Thread group size of the two compacted-index cluster passes.
    constexpr uint32_t kClusterGroupSize = 64;
    /// Thread group size of SpecPrimary.
    constexpr uint32_t kSpecPrimaryGroupSize = 64;
    /// Byte offsets into RayBatchBuffer::counts.
    constexpr uint64_t kDiffCountOffset = 0;
    constexpr uint64_t kSpecCountOffset = sizeof(uint32_t);
    /// ConeTrace and SpecPrimary run before the CPU knows this frame's specular
    /// count, and this gfx build has no Vulkan indirect dispatch. The grid is sized
    /// from the previous frame's count plus headroom, and the shaders stride over
    /// any samples beyond it, so a count that grows is still fully covered. A grid
    /// much smaller than the count costs ConeTrace ~1 ms on living-room-2, because
    /// each group then runs many samples back to back.
    constexpr uint32_t kSpecDispatchMinItems = 8192;

    uint32_t specDispatchItems(uint32_t prevSpecCount, uint32_t batchSize)
    {
        const uint32_t withHeadroom = prevSpecCount + prevSpecCount / 8;
        return std::min(batchSize, std::max(kSpecDispatchMinItems, withHeadroom));
    }

    const FileDialogFilterVec kCameraJsonFilters = {
        {"json", "JSON Files"},
    };

    const FileDialogFilterVec kModelStateFilters = {
        {"ckpt", "Binary Files"},
    };

    const char kEnableTemporalSpecReuse[] = "enableTemporalSpecReuse";
    const char kEnableClusterSpatialReuse[] = "enableClusterSpatialReuse";
    const char kUseReferenceConeTrace[] = "useReferenceConeTrace";
    const char kClusterMaxRays[] = "clusterMaxRays";
    const char kClusterSigmaPriorScale[] = "clusterSigmaPriorScale";
    const char kClusterAssignThreshold[] = "clusterAssignThreshold";
    const char kClusterAlphaWeight[] = "clusterAlphaWeight";
    const char kClusterLockCount[] = "clusterLockCount";
    const char kClusterDetectThreshold[] = "clusterDetectThreshold";
    const char kClusterDetectRate[] = "clusterDetectRate";
    const char kClusterUnlockDrift[] = "clusterUnlockDrift";
    const char kClusterMotionDrift[] = "clusterMotionDrift";
    const char kClusterMotionMemory[] = "clusterMotionMemory";
    const char kClusterUnlockOutlierRate[] = "clusterUnlockOutlierRate";
    const char kEnablePostFilter[] = "enablePostFilter";
    const char kEnableFXAA[] = "enableFXAA";
    const char kEnableMirrorAlbedo[] = "enableMirrorAlbedo";
    const char kEnvColorEncoding[] = "envColorEncoding";
    const char kUseNEE[] = "useNEE";
    const char kDiffTrainOnGlossy[] = "diffTrainOnGlossy";
    const char kRandomTrainDir[] = "randomTrainDir";
    const char kTrainBatchSize[] = "trainBatchSize";
    const char kNumRHS[] = "numRHS";
    const char kRhsGlossyBounces[] = "rhsGlossyBounces";
    const char kEnableTrainSchedule[] = "enableTrainSchedule";
    const char kTrainLrHot[] = "trainLrHot";
    const char kTrainLrCold[] = "trainLrCold";
    const char kTrainEmaHot[] = "trainEmaHot";
    const char kTrainEmaCold[] = "trainEmaCold";
    const char kTrainWarmupFrames[] = "trainWarmupFrames";
    const char kTrainHoldFrames[] = "trainHoldFrames";
    const char kTrainCoolFrames[] = "trainCoolFrames";
    const char kTrainCameraHeat[] = "trainCameraHeat";
    const char kTrainSceneHeat[] = "trainSceneHeat";
    const char kTrainLightMotionHeat[] = "trainLightMotionHeat";
    const char kTrainObjectMotionHeat[] = "trainObjectMotionHeat";
    const char kTrainHeatLogInterval[] = "trainHeatLogInterval";
    // Headless / scripted control: initial render mode, checkpoint auto-load, training schedule.
    const char kRenderMode[] = "renderMode";
    const char kLoadModelPath[] = "loadModelPath";
    const char kTotalTrainSteps[] = "totalTrainSteps";
    const char kSaveCkptInterval[] = "saveCkptInterval";

    // Training-schedule event classes (bit mask), as graded and logged by updateTrainSchedule().
    constexpr uint32_t kTrainEventCamera = 1u << 0;
    constexpr uint32_t kTrainEventEdit = 1u << 1;
    constexpr uint32_t kTrainEventLight = 1u << 2;
    constexpr uint32_t kTrainEventObject = 1u << 3;

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

    /// Scene-scale epsilon guarding the cluster residual denominators. All three
    /// cluster passes must agree on this value or their gates disagree.
    float clusterSceneEpsilon(const AABB& aabb)
    {
        return 1e-4f * std::max(length(aabb.extent()), 1e-3f);
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

static void regNeuralRadiosity(pybind11::module& m)
{
    pybind11::class_<NeuralRadiosity, RenderPass, ref<NeuralRadiosity>> pass(m, "NeuralRadiosity");
    pass.def("clearHistory", &NeuralRadiosity::clearHistory);
    pass.def_property_readonly("trainHeat", &NeuralRadiosity::getTrainHeat);
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralRadiosity>();
    ScriptBindings::registerBinding(regNeuralRadiosity);
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
        else if (key == kEnableMirrorAlbedo)
            mEnableMirrorAlbedo = value;
        else if (key == kEnvColorEncoding)
            mEnvColorEncoding = value;
        else if (key == kUseNEE)
            mUseNEE = value;
        else if (key == kDiffTrainOnGlossy)
            mDiffTrainOnGlossy = value;
        else if (key == kRandomTrainDir)
            mRandomTrainDir = value;
        else if (key == kTrainBatchSize)
            mBatchSize = mBatchSizeInit = std::max(256u, (uint32_t)value);
        else if (key == kNumRHS)
            mNumRHS = mNumRHSInit = std::max(1u, (uint32_t)value);
        else if (key == kRhsGlossyBounces)
            mRhsGlossyBounces = value;
        else if (key == kEnableTrainSchedule)
            mEnableTrainSchedule = value;
        else if (key == kTrainLrHot)
            mTrainLrHot = value;
        else if (key == kTrainLrCold)
            mTrainLrCold = value;
        else if (key == kTrainEmaHot)
            mTrainEmaHot = value;
        else if (key == kTrainEmaCold)
            mTrainEmaCold = value;
        else if (key == kTrainWarmupFrames)
            mTrainWarmupFrames = value;
        else if (key == kTrainHoldFrames)
            mTrainHoldFrames = value;
        else if (key == kTrainCoolFrames)
            mTrainCoolFrames = value;
        else if (key == kTrainCameraHeat)
            mTrainCameraHeat = value;
        else if (key == kTrainSceneHeat)
            mTrainSceneHeat = value;
        else if (key == kTrainLightMotionHeat)
            mTrainLightMotionHeat = value;
        else if (key == kTrainObjectMotionHeat)
            mTrainObjectMotionHeat = value;
        else if (key == kTrainHeatLogInterval)
            mTrainHeatLogInterval = value;
        else if (key == kEnableClusterSpatialReuse)
            mEnableClusterSpatialReuse = value;
        else if (key == kUseReferenceConeTrace)
            mUseReferenceConeTrace = value;
        else if (key == kClusterMaxRays)
            mClusterMaxRays = value;
        else if (key == kClusterSigmaPriorScale)
            mClusterSigmaPriorScale = value;
        else if (key == kClusterAssignThreshold)
            mClusterAssignThreshold = value;
        else if (key == kClusterAlphaWeight)
            mClusterAlphaWeight = value;
        else if (key == kClusterLockCount)
            mClusterLockCount = value;
        else if (key == kClusterDetectThreshold)
            mClusterDetectThreshold = value;
        else if (key == kClusterDetectRate)
            mClusterDetectRate = value;
        else if (key == kClusterUnlockDrift)
            mClusterUnlockDrift = value;
        else if (key == kClusterMotionDrift)
            mClusterMotionDrift = value;
        else if (key == kClusterMotionMemory)
            mClusterMotionMemory = value;
        else if (key == kClusterUnlockOutlierRate)
            mClusterUnlockOutlierRate = value;
        else if (key == kEnablePostFilter)
            mEnablePostFilter = value;
        else if (key == kRenderMode)
        {
            const std::string mode = value;
            if (mode == "train") mRenderMode = RenderMode::Train;
            else if (mode == "onlineTrain") mRenderMode = RenderMode::OnlineTrain;
            else if (mode == "render") mRenderMode = RenderMode::Render;
            else if (mode == "idle") mRenderMode = RenderMode::Idle;
            else logWarning("Unknown renderMode '{}' (expected train|onlineTrain|render|idle).", mode);
        }
        else if (key == kLoadModelPath)
        {
            const std::string path = value;
            mLoadModelPath = path;
        }
        else if (key == kTotalTrainSteps)
            mTotalTrainSteps = value;
        else if (key == kSaveCkptInterval)
            mSaveCKPTInterval = value;
        else
            logWarning("Unknown property '{}' in NeuralRadiosity properties.", key);
    }

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);

    // Pick the ConeTrace specialization that matches who still uses it. Must run
    // after the properties above, which can change who that is.
    setConeParameters(mRenderMode == RenderMode::Train);
}

Properties NeuralRadiosity::getProperties() const
{
    Properties props;
    props[kEnableTemporalSpecReuse] = mEnableRenderSpecTemporalReuse;
    props[kEnableClusterSpatialReuse] = mEnableClusterSpatialReuse;
    props[kUseReferenceConeTrace] = mUseReferenceConeTrace;
    props[kClusterMaxRays] = mClusterMaxRays;
    props[kClusterSigmaPriorScale] = mClusterSigmaPriorScale;
    props[kClusterAssignThreshold] = mClusterAssignThreshold;
    props[kClusterAlphaWeight] = mClusterAlphaWeight;
    props[kClusterLockCount] = mClusterLockCount;
    props[kClusterDetectThreshold] = mClusterDetectThreshold;
    props[kClusterDetectRate] = mClusterDetectRate;
    props[kClusterUnlockDrift] = mClusterUnlockDrift;
    props[kClusterMotionDrift] = mClusterMotionDrift;
    props[kClusterMotionMemory] = mClusterMotionMemory;
    props[kClusterUnlockOutlierRate] = mClusterUnlockOutlierRate;
    props[kEnablePostFilter] = mEnablePostFilter;
    props[kEnableFXAA] = mEnableFXAA;
    props[kEnableMirrorAlbedo] = mEnableMirrorAlbedo;
    props[kEnvColorEncoding] = mEnvColorEncoding;
    props[kUseNEE] = mUseNEE;
    props[kDiffTrainOnGlossy] = mDiffTrainOnGlossy;
    props[kRandomTrainDir] = mRandomTrainDir;
    props[kTrainBatchSize] = mBatchSizeInit;
    props[kNumRHS] = mNumRHSInit;
    props[kRhsGlossyBounces] = mRhsGlossyBounces;
    props[kEnableTrainSchedule] = mEnableTrainSchedule;
    props[kTrainLrHot] = mTrainLrHot;
    props[kTrainLrCold] = mTrainLrCold;
    props[kTrainEmaHot] = mTrainEmaHot;
    props[kTrainEmaCold] = mTrainEmaCold;
    props[kTrainWarmupFrames] = mTrainWarmupFrames;
    props[kTrainHoldFrames] = mTrainHoldFrames;
    props[kTrainCoolFrames] = mTrainCoolFrames;
    props[kTrainCameraHeat] = mTrainCameraHeat;
    props[kTrainSceneHeat] = mTrainSceneHeat;
    props[kTrainLightMotionHeat] = mTrainLightMotionHeat;
    props[kTrainObjectMotionHeat] = mTrainObjectMotionHeat;
    props[kTrainHeatLogInterval] = mTrainHeatLogInterval;
    props[kTotalTrainSteps] = mTotalTrainSteps;
    props[kSaveCkptInterval] = mSaveCKPTInterval;
    if (!mLoadModelPath.empty()) props[kLoadModelPath] = mLoadModelPath;
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

    mVarsChanged = false;

    bakeBegin();

    // Train the model.
    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
    {
        updateTrainSchedule();
        train(pRenderContext, renderData);
    }

    // Render the scene.
    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        render(pRenderContext, renderData);
    }

    bakeExport(pRenderContext);
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
    widget.var("RHS Glossy Bounces", mRhsGlossyBounces, 0u, 8u);
    widget.tooltip(
        "Glossy RHS hits continue along a BSDF sample (with NEE) this many times\n"
        "before the diffuse network is queried. 0 queries it at glossy hits, which\n"
        "lets the diffuse network drift dark in glossy, high-albedo rooms."
    );
    widget.checkbox("Diffuse Trains On Glossy", mDiffTrainOnGlossy);
    widget.tooltip("Also train the diffuse network on glossy samples. Only useful with 0 glossy bounces.");
    widget.checkbox("Mirror Albedo", mEnableMirrorAlbedo);
    widget.tooltip("Trace one mirror ray and replace the environment color with the hit albedo.");
    {
        static const Gui::DropdownList kEnvEncodings = {{0, "Linear"}, {1, "Signed log"}, {2, "Log"}};
        widget.dropdown("Env Color Encoding", kEnvEncodings, mEnvColorEncoding);
        widget.tooltip(
            "How a mirror-ray miss enters the shared env/albedo input channels.\n"
            "Signed log: -log(1 + env), so misses are negative and bounded; hits keep the albedo."
        );
    }

    if (auto group = widget.group("Training Schedule", false))
    {
        group.checkbox("Enable", mEnableTrainSchedule);
        group.tooltip("Off: always train with the hot learning rate and EMA decay.");
        group.text(fmt::format("Heat {:.2f}  hold {}", mTrainHeat, mTrainHoldLeft));
        group.var("Log every N frames", mTrainHeatLogInterval, 0u, 10000u);
        group.tooltip("Print heat, hold/cooling, LR, EMA decay and the events seen to the log every N training frames,\n"
                      "and at once when a camera or scene event raises the heat. 0 = off.");
        group.var("LR hot", mTrainLrHot, 1e-5f, 1e-1f, 1e-4f, false, "%.5f");
        group.var("LR cold", mTrainLrCold, 1e-5f, 1e-1f, 1e-4f, false, "%.5f");
        group.var("EMA hot", mTrainEmaHot, 0.f, 0.9999f, 0.001f, false, "%.4f");
        group.var("EMA cold", mTrainEmaCold, 0.f, 0.9999f, 0.001f, false, "%.4f");
        group.var("Warmup frames", mTrainWarmupFrames, 0u, 10000u);
        group.tooltip("Frames held fully hot after the scene loads or the schedule is reset.");
        group.var("Hold frames", mTrainHoldFrames, 0u, 10000u);
        group.tooltip("Frames held at the event level after the last camera or scene change.");
        group.var("Cool frames", mTrainCoolFrames, 1.f, 10000.f, 1.f);
        group.tooltip("Time constant of the decay toward cold once nothing changes.");
        group.var("Camera heat", mTrainCameraHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Heat while the camera moves. The network is world-space, so this only speeds up newly seen regions.");
        group.var("Scene heat", mTrainSceneHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Heat on discrete scene edits (materials, light parameters, environment, added or removed geometry),\n"
                      "and on any scene change while the camera moves.");
        group.var("Light motion heat", mTrainLightMotionHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Camera fixed: heat while lights move (analytic lights, or emissive geometry).");
        group.var("Object motion heat", mTrainObjectMotionHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Camera fixed: heat while only non-emissive geometry moves.\n"
                      "Lower settles the rest of the image; the moving objects' own lighting then lags more.");
        if (group.button("Reset schedule")) resetTrainSchedule();
    }

    if (auto group = widget.group("Spatiotemporal Cluster Reuse", true))
    {
        if (group.checkbox("Enable", mEnableRenderSpecTemporalReuse))
        {
            // Toggling changes whether ConeTrace is still on the render path, which
            // in turn changes its NUM_SPEC_RAYS specialization.
            setConeParameters(mRenderMode == RenderMode::Train);
            clearRenderTemporalHistory(mpDevice->getRenderContext());
            mVarsChanged = true;
        }
        group.tooltip(
            "On: 1 ray/pixel baseline with world-anchored cluster history.\n"
            "Off: original per-frame group-wide K-Means cone trace."
        );

        if (group.button("Clear History")) clearHistory();

        if (!mEnableRenderSpecTemporalReuse)
        {
            if (group.checkbox("Reference T=128", mUseReferenceConeTrace))
            {
                setConeParameters(mRenderMode == RenderMode::Train);
            }
            group.tooltip(
                "Run the render path at the training ray count so its (t_k, sigma_k, w_k)\n"
                "can be compared per pixel against the T=128 reference. Very slow."
            );
        }

        group.checkbox("Spatial Reuse", mEnableClusterSpatialReuse);

        group.checkbox("Clear On Geometry Moved", mClearHistoryOnGeometryMoved);
        group.tooltip(
            "Off by default. A scene with a playing animation raises GeometryMoved\n"
            "every frame, and clearing on it would keep the history permanently empty\n"
            "(1 unfiltered ray per pixel). Moved geometry is handled per cluster by\n"
            "the cone test and the outlier path instead."
        );

        group.var("Max Rays / Pixel", mClusterMaxRays, 1u, 4u);
        group.tooltip("Hard cap. Extra rays are only traced when a sample is an outlier.");

        group.var("Assign Threshold (lambda)", mClusterAssignThreshold, 0.05f, 8.f, 0.01f);
        group.tooltip(
            "How finely the cone is partitioned into depth modes.\n"
            "Lower = more, narrower modes = closer to the K-Means reference."
        );
        group.var("Lock Count (N_lock)", mClusterLockCount, 1.f, 255.f, 1.f);
        group.tooltip(
            "Accumulate random GGX samples with alpha = 1/(n+1) until this many\n"
            "observations, then freeze the mixture. Matches the old T=32 one-shot."
        );
        group.var("Detect Gate (lambda_d)", mClusterDetectThreshold, 0.1f, 8.f, 0.05f);
        group.tooltip(
            "Residual gate of the locked-pixel change detector. Separate from lambda,\n"
            "which is too tight to tell a static pixel from a changed one."
        );
        group.var("Detect Rate (beta)", mClusterDetectRate, 0.02f, 0.5f, 0.005f);
        group.tooltip("EMA rate of the observed ray shares. Lower = steadier, slower to react.");
        group.var("Unlock Outlier Rate", mClusterUnlockOutlierRate, 0.05f, 1.f, 0.01f);
        group.tooltip(
            "When this share of hits lands outside every mode, the pixel unlocks\n"
            "to grow new modes. Losing real weight through the cone test unlocks directly."
        );
        group.var("Motion Drift", mClusterMotionDrift, 0.f, 1.f, 0.001f);
        group.tooltip("Per-frame drift (cone radii) above which the pixel counts as moving with the camera.");
        group.var("Motion Memory", mClusterMotionMemory, 1.f, 32.f, 1.f);
        group.tooltip(
            "Observation cap while moving: the mixture tracks ~this many recent samples\n"
            "and only locks once the camera holds still, so no motion trail is frozen."
        );
        group.var("Unlock Drift", mClusterUnlockDrift, 0.05f, 4.f, 0.05f);
        group.tooltip(
            "Unlock once the cone has slid this many cone radii across the frozen\n"
            "anchors since the pixel locked (smooth camera motion)."
        );
        group.var("Weight EMA (alpha_w)", mClusterAlphaWeight, 0.01f, 1.f, 0.005f);
        group.tooltip("Unused: weights use 1/(n+1) until lock. Kept for property compatibility.");
        group.var("Max Count (N_max)", mClusterMaxCount, 1.f, 255.f, 1.f);
        group.tooltip("Per-mode (mu, sigma) count cap during the accumulation window.");
        group.var("Release Weight", mClusterWeakWeight, 0.f, 0.5f, 0.005f);
        group.var("Sigma Prior Scale", mClusterSigmaPriorScale, 0.f, 2.f, 0.01f);
        group.tooltip(
            "Newborn sigma = scale * tanLobe * t. 0 reproduces the training path's\n"
            "singleton clsScale exactly; Welford grows sigma from later hits."
        );

        if (auto reproj = group.group("Reprojection"))
        {
            reproj.var("Normal Threshold", mClusterNormalThreshold, 0.f, 1.f, 0.005f);
            reproj.var("Roughness Threshold", mClusterRoughnessThreshold, 0.f, 1.f, 0.005f);
            reproj.var("Position Tolerance", mClusterPosThresholdRel, 0.f, 0.5f, 0.001f);
        }

        if (auto spatial = group.group("Spatial Gather"))
        {
            spatial.var("Taps", mClusterNumTaps, 0u, 16u);
            spatial.var("Radius Scale", mClusterGatherRadiusScale, 0.f, 8.f, 0.05f);
            spatial.var("Max Radius (px)", mClusterMaxGatherRadius, 1.f, 32.f, 0.5f);
            spatial.var("Gamma Max", mClusterGammaMax, 0.f, 1.f, 0.01f);
            spatial.var("Gamma Min", mClusterGammaMin, 0.f, 1.f, 0.01f);
            spatial.var("Confidence Count", mClusterConfidenceCount, 1.f, 255.f, 1.f);
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
}

void NeuralRadiosity::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpFirstSmoothPass = nullptr;
    mpConeTrace = nullptr;
    mpSpecPrimary = nullptr;
    mpClusterReproject = nullptr;
    mpConeTraceScreen = nullptr;
    mpClusterSpatialReuse = nullptr;
    mpResolvePass = nullptr;
    mpPostFilterPass = nullptr;
    mpPostFXAAPass = nullptr;
    mpPrefiltEnvInitPass = nullptr;
    mpPrefiltEnvDownsamplePass = nullptr;
    mpPrefiltEnvCubemap = nullptr;
    mpClusterHistory[0] = nullptr;
    mpClusterHistory[1] = nullptr;
    mClusterHistoryValid = false;
    mPrefiltEnvDirty = true;
    mVarsChanged = true;
    resetTrainSchedule();
}

void NeuralRadiosity::onSceneUpdates(RenderContext* pRenderContext, IScene::UpdateFlags sceneUpdates)
{
    // Consumed by the next updateTrainSchedule().
    mPendingSceneUpdates |= sceneUpdates;

    // Neither camera nor geometry motion invalidates history wholesale. Clusters
    // carry world anchors that are validated one at a time against the current
    // cone, and a mode whose object moved but stayed in the cone is replaced by
    // the outlier path within a frame or two.
    //
    // This deliberately does *not* clear on GeometryMoved: any scene with a
    // playing animation raises that flag every single frame, which would wipe the
    // history continuously and silently reduce the render path to one unfiltered
    // ray per pixel. That is exactly the failure mode this design replaces.
    if (mClearHistoryOnGeometryMoved && is_set(sceneUpdates, IScene::UpdateFlags::GeometryMoved))
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

void NeuralRadiosity::resetTrainSchedule()
{
    mTrainHeat = 1.f;
    mTrainHoldLeft = mTrainWarmupFrames;
    mPendingSceneUpdates = IScene::UpdateFlags::None;
    // Print on the first training frame after a reset.
    mTrainHeatLogFramesLeft = 0;
    mTrainHeatLogEvents = 0;
}

void NeuralRadiosity::updateTrainSchedule()
{
    using F = IScene::UpdateFlags;
    const F updates = mPendingSceneUpdates;
    mPendingSceneUpdates = F::None;
    if (!mpNRModel) return;
    if (!mEnableTrainSchedule)
    {
        mpNRModel->setOptimizerHyperparams(mTrainLrHot, mTrainEmaHot);
        logTrainSchedule(0, false, mTrainLrHot, mTrainEmaHot);
        return;
    }

    // Changes to what the network must represent, split by what changed (local
    // addition; ver4 sent all of them to mTrainSceneHeat). Together they are the
    // same flag set as ver4's.
    // One-off edits.
    const F editChange = F::LightIntensityChanged | F::LightPropertiesChanged | F::LightCountChanged | F::MaterialsChanged |
                         F::EmissiveMaterialsChanged | F::EnvMapChanged | F::EnvMapPropertiesChanged | F::GeometryChanged |
                         F::DisplacementChanged | F::SDFGeometryChanged;
    // Moving lights. Analytic lights raise LightsMoved. For emissive geometry the scene
    // raises LightCollectionChanged (only when a mesh light's transform changed) next
    // to GeometryMoved.
    const F lightMotion = F::LightsMoved | F::LightCollectionChanged;
    // Moving geometry. GeometryMoved is raised for every animated instance, emissive or not.
    const F objectMotion = F::GeometryMoved | F::MeshesChanged | F::CurvesMoved | F::CustomPrimitivesMoved;
    const F sceneChange = editChange | lightMotion | objectMotion;
    // The radiance field is in world space, so camera motion changes nothing it
    // has learned. It only exposes regions that have had little training.
    const F cameraChange = F::CameraMoved | F::CameraSwitched | F::CameraPropertiesChanged;

    uint32_t events = 0;
    if (is_set(updates, cameraChange)) events |= kTrainEventCamera;
    if (is_set(updates, editChange)) events |= kTrainEventEdit;
    if (is_set(updates, lightMotion)) events |= kTrainEventLight;
    if (is_set(updates, objectMotion)) events |= kTrainEventObject;

    float target = 0.f;
    if ((events & kTrainEventCamera) != 0)
    {
        // Camera moving: ver4 behavior, any scene change goes to mTrainSceneHeat.
        target = mTrainCameraHeat;
        if (is_set(updates, sceneChange)) target = std::max(target, mTrainSceneHeat);
    }
    else
    {
        // Camera fixed: grade scene changes so the image can settle while only
        // non-emissive geometry moves, and stays warm enough to follow moving lights.
        if ((events & kTrainEventEdit) != 0) target = std::max(target, mTrainSceneHeat);
        if ((events & kTrainEventLight) != 0) target = std::max(target, mTrainLightMotionHeat);
        if ((events & kTrainEventObject) != 0) target = std::max(target, mTrainObjectMotionHeat);
    }

    const float prevHeat = mTrainHeat;
    if (target >= mTrainHeat && target > 0.f)
    {
        mTrainHeat = target;
        // Keep a longer pending hold (the warmup) counting down while events repeat
        // (local fix). ver4 froze it here, so a camera path over an animated scene kept
        // the whole warmup pending until the camera stopped. Same as ver4 for holds up
        // to mTrainHoldFrames.
        mTrainHoldLeft = std::max(mTrainHoldLeft > 0 ? mTrainHoldLeft - 1 : 0u, mTrainHoldFrames);
    }
    else if (mTrainHoldLeft > 0)
    {
        mTrainHoldLeft--;
    }
    else
    {
        // Relax toward the current event level (0 when nothing changes).
        const float k = std::exp(-1.f / std::max(mTrainCoolFrames, 1.f));
        mTrainHeat = target + (mTrainHeat - target) * k;
    }

    // Interpolate log(lr) and log(1 - decay): each heat step changes the step size
    // and the EMA half-life by the same factor.
    const float h = std::clamp(mTrainHeat, 0.f, 1.f);
    const float lr = mTrainLrCold * std::pow(mTrainLrHot / mTrainLrCold, h);
    const float oneMinusDecay = (1.f - mTrainEmaCold) * std::pow((1.f - mTrainEmaHot) / (1.f - mTrainEmaCold), h);
    mpNRModel->setOptimizerHyperparams(lr, 1.f - oneMinusDecay);
    logTrainSchedule(events, mTrainHeat > prevHeat, lr, 1.f - oneMinusDecay);
}

void NeuralRadiosity::logTrainSchedule(uint32_t events, bool heatRaised, float lr, float emaDecay)
{
    if (mTrainHeatLogInterval == 0) return;
    // Events are reported for the whole span since the previous line, not just this frame.
    mTrainHeatLogEvents |= events;
    if (mTrainHeatLogFramesLeft > 0 && !heatRaised)
    {
        mTrainHeatLogFramesLeft--;
        return;
    }

    std::string state = "cooling";
    if (!mEnableTrainSchedule) state = "off";
    else if (mTrainHoldLeft > 0) state = fmt::format("hold {}", mTrainHoldLeft);
    static const std::pair<uint32_t, const char*> kEventNames[] = {
        {kTrainEventCamera, "camera"}, {kTrainEventEdit, "edit"}, {kTrainEventLight, "light"}, {kTrainEventObject, "object"}};
    std::string eventNames;
    for (const auto& [bit, name] : kEventNames)
    {
        if ((mTrainHeatLogEvents & bit) == 0) continue;
        if (!eventNames.empty()) eventNames += '+';
        eventNames += name;
    }
    if (eventNames.empty()) eventNames = "none";
    // With the schedule off the hot values are in use, i.e. an effective heat of 1.
    const float heat = mEnableTrainSchedule ? mTrainHeat : 1.f;
    logInfo("[TrainSchedule] frame {:>6}  heat {:.3f}  {:<9}  lr {:.2e}  ema {:.4f}  events {}{}", mFrameCount, heat, state, lr,
            emaDecay, eventNames, heatRaised ? "  <- raised" : "");

    mTrainHeatLogFramesLeft = mTrainHeatLogInterval - 1;
    mTrainHeatLogEvents = 0;
}

void NeuralRadiosity::render(RenderContext* pRenderContext, const RenderData& renderData)
{
    // submit(false) hands recorded work to the GPU without waiting for it. Falcor
    // otherwise submits only at the CUDA interop syncs, so the GPU would sit idle
    // while the CPU records the whole segment.
    // First smooth pass.
    firstSmoothPass(pRenderContext, renderData);
    pRenderContext->submit(false);
    // Compact pass.
    compactPass(pRenderContext, renderData);
    pRenderContext->submit(false);
    // Cone tracing: either the spatiotemporal cluster reuse pipeline or the
    // original per-frame group-wide K-Means.
    coneTrace(pRenderContext, renderData, mpRenderBatch);
    // Model querying: CUDA kernel.
    modelInferenceCUDA(pRenderContext, mpRenderBatch);
    // Resolve pass.
    resolvePass(pRenderContext, renderData);
    pRenderContext->submit(false);
    // Post-process filtering pass.
    postFilterPass(pRenderContext, renderData);
    // Snapshot the G-Buffer for next frame's reprojection.
    if (useClusterReuse()) savePrevGBuffer(pRenderContext, renderData);
}

void NeuralRadiosity::train(RenderContext* pRenderContext, const RenderData& renderData)
{
    // submit(false) between passes lets the GPU start while the CPU records the
    // rest of the segment (see render()).
    // Sample random smooth points (LHS)
    randomSmooth(pRenderContext, mpTrainLHSBatch);
    pRenderContext->submit(false);
    // Sample RHS
    sampleRHS(pRenderContext);
    pRenderContext->submit(false);

    // Render RHS
    {
        // 1. Compaction
        compactBatch(pRenderContext, mpTrainRHSBatch);
        // 2. Cone tracing (RHS is diffuse-only, so this is a no-op).
        if (mpTrainRHSBatch->needSpec)
            coneTraceBatch(pRenderContext, mpTrainRHSBatch, renderData);
        // 3. Model inference
        modelInferenceCUDA(pRenderContext, mpTrainRHSBatch);
        // 4. Resolve
        resolveBatch(pRenderContext, mpTrainRHSBatch);
        pRenderContext->submit(false);
    }

    // Train LHS with RHS results
    {
        // 1. Compaction
        compactBatch(pRenderContext, mpTrainLHSBatch);
        pRenderContext->submit(false);
        // 2. Cone tracing
        coneTraceBatch(pRenderContext, mpTrainLHSBatch, renderData);
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

    executePass(pRenderContext, mpFirstSmoothPass, "FirstSmoothPass", uint3(mFrameDim, 1));
}

void NeuralRadiosity::compactPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "CompactPass");

    // Prefix sum from active mask to index. The counts stay on the GPU.
    computeBatchCounts(pRenderContext, mpRenderBatch, mFrameDim.x * mFrameDim.y);

    ShaderVar var = mpCompactPass->getRootVar();
    const auto name = "gCompactPass";
    bindScreenData(var, renderData, name);

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["diffInput"] = mpRenderBatch->diffInput;

    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;
    var["specInput"] = mpRenderBatch->specInput;
    var["specSrc"] = mpRenderBatch->specSrc;

    // With shared I/O the diffuse rows start after the padded specular rows. The
    // specular total is only on the GPU here, so the shader reads it from `counts`.
    var[name]["sharedIO"] = mpRenderBatch->sharedIO ? 1u : 0u;
    var["batchCounts"] = mpRenderBatch->counts;

    executePass(pRenderContext, mpCompactPass, "CompactPass", uint3(mFrameDim, 1));

    stageBatchCounts(pRenderContext, mpRenderBatch);
}

void NeuralRadiosity::resolvePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ResolvePass");

    ShaderVar var = mpResolvePass->getRootVar();
    const auto name = "gResolvePass";
    bindScreenData(var, renderData, name);

    var[name]["diffSize"] = mpRenderBatch->diffSize;
    var[name]["specSize"] = mpRenderBatch->specSize;
    // Counts were read back at the inference sync, so the CPU knows the offset.
    var[name]["diffColorOffset"] = mpRenderBatch->diffOutputOffset();

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["diffColor"] = mpRenderBatch->diffColor;

    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;
    var["specColor"] = mpRenderBatch->specColor;

    var["tempColor"] = mpRenderTempColor;

    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
        var["debug"] = mpTrainRHSBatch->emission;

    executePass(pRenderContext, mpResolvePass, "ResolvePass", uint3(mFrameDim, 1));
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

        executePass(pRenderContext, mpPostFilterPass, "PostFilterPass", uint3(mFrameDim, 1));
    }

    pRenderContext->uavBarrier(mpPostFilterTempColor.get());

    {
        ShaderVar var = mpPostFXAAPass->getRootVar();
        const auto name = "gPostFilterPass";
        bindScreenData(var, renderData, name);
        var[name]["enableFXAA"] = mEnableFXAA ? 1u : 0u;
        var["gFxaaInput"] = mpPostFilterTempColor;

        executePass(pRenderContext, mpPostFXAAPass, "PostFXAAPass", uint3(mFrameDim, 1));
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
    var["gRandomSmooth"]["diffTrainOnGlossy"] = mDiffTrainOnGlossy ? 1u : 0u;

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    executePass(pRenderContext, mpRandomSmooth, "RandomSmooth", uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::sampleRHS(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SampleRHS");

    ShaderVar var = mpSampleRHS->getRootVar();
    const auto name = "gSampleRHS";

    // Bind parameters
    var[name]["maxGlossyBounces"] = mRhsGlossyBounces;
    var[name]["queryGlossyFallback"] = (mDiffTrainOnGlossy || mRhsGlossyBounces == 0) ? 1u : 0u;
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

    executePass(pRenderContext, mpSampleRHS, "SampleRHS", uint3(mBatchSize * mNumRHS, 1, 1));
}

void NeuralRadiosity::resolveRHS(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "ResolveRHS");

    ShaderVar var = mpResolveRHS->getRootVar();
    const auto name = "gResolveRHS";
    
    bindRayBatchData(var, mpTrainLHSBatch, name);
    var["rhsColor"] = mpTrainRHSBatch->color;

    executePass(pRenderContext, mpResolveRHS, "ResolveRHS", uint3(mBatchSize, 1, 1));
}

void NeuralRadiosity::compactBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "CompactBatch");

    // Prefix sum from active mask to index. The counts stay on the GPU.
    computeBatchCounts(pRenderContext, pRayBatch, pRayBatch->size);

    ShaderVar var = mpCompactBatch->getRootVar();
    const auto name = "gCompactBatch";
    bindRayBatchData(var, pRayBatch, name);

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    executePass(pRenderContext, mpCompactBatch, "CompactBatch", uint3(pRayBatch->size, 1, 1));

    stageBatchCounts(pRenderContext, pRayBatch);
}

void NeuralRadiosity::computeBatchCounts(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, uint32_t elementCount)
{
    // A disabled group keeps its count at zero.
    pRenderContext->clearUAV(pRayBatch->counts->getUAV().get(), uint4(0));
    bakeHost(fmt::format("clearcounts {}", batchRole(pRayBatch.get())));

    // Passing a CPU pointer here would make PrefixSum read the total back, which
    // flushes and waits for the GPU. The total goes to `counts` instead.
    if (pRayBatch->needDiff)
    {
        mpPrefixSum->execute(pRenderContext, pRayBatch->diffIndex, elementCount, nullptr, pRayBatch->counts, kDiffCountOffset);
        bakeHost(fmt::format("prefixsum {} diff {}", batchRole(pRayBatch.get()), elementCount));
    }
    if (pRayBatch->needSpec)
    {
        mpPrefixSum->execute(pRenderContext, pRayBatch->specIndex, elementCount, nullptr, pRayBatch->counts, kSpecCountOffset);
        bakeHost(fmt::format("prefixsum {} spec {}", batchRole(pRayBatch.get()), elementCount));
    }
}

void NeuralRadiosity::stageBatchCounts(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    // The CPU copy is read after the next CUDA interop sync, which already waits
    // for the GPU, so this costs no extra stall.
    pRenderContext->copyBufferRegion(pRayBatch->countsReadback.get(), 0, pRayBatch->counts.get(), 0, 2 * sizeof(uint32_t));
}

void NeuralRadiosity::readBatchCountsAfterSync(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    // waitForFalcor() blocks the CPU until the GPU is idle only on Vulkan. D3D12
    // uses a shared fence and returns immediately, so wait explicitly there.
    if (mpDevice->getType() != Device::Type::Vulkan)
        pRenderContext->submit(true);
    pRayBatch->readCounts();
}

void NeuralRadiosity::resolveBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ResolveBatch");

    ShaderVar var = mpResolveBatch->getRootVar();
    bindRayBatchData(var, pRayBatch, "gResolveBatch");

    executePass(pRenderContext, mpResolveBatch, "ResolveBatch", uint3(pRayBatch->size, 1, 1));
}

bool NeuralRadiosity::useClusterReuse() const
{
    return mEnableRenderSpecTemporalReuse && mpClusterHistory[0] && mpClusterHistory[1] &&
        mpClusterLockState[0] && mpClusterLockState[1] && mpPrevPos;
}

void NeuralRadiosity::coneTrace(RenderContext* pRenderContext, const RenderData& renderData, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    // Dispatch by batch type: the training batches keep the original T=128
    // group-wide K-Means (their specInput must stay bit-compatible with existing
    // checkpoints), while the render batch goes through the screen-space
    // spatiotemporal cluster reuse pipeline.
    if (pRayBatch == mpRenderBatch && useClusterReuse())
    {
        clusterReuse(pRenderContext, renderData);
        return;
    }

    coneTraceBatch(pRenderContext, pRayBatch, renderData);
}

void NeuralRadiosity::coneTraceBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, const RenderData& renderData)
{
    // This frame's specular count is only known on the GPU here. specSize still
    // holds the previous frame's count and sizes the grid (see specDispatchItems).
    if (!pRayBatch->needSpec) return;

    {
        FALCOR_PROFILE(pRenderContext, "ConeTrace");

        const uint32_t numGroups = specDispatchItems(pRayBatch->specSize, pRayBatch->size);

        ShaderVar var = mpConeTrace->getRootVar();
        const auto name = "gConeTrace";
        // Bind parameters
        var[name]["batchSize"] = pRayBatch->size;
        var[name]["batchCount"] = mFrameCount;
        var[name]["dispatchGroupCount"] = numGroups;
        var[name]["frameDim"] = mFrameDim;
        var[name]["useScreenGBuffer"] = pRayBatch->needAll ? 0u : 1u;

        const AABB& aabb = mpScene->getSceneBounds();
        var[name]["bboxMin"] = aabb.minPoint;
        var[name]["bboxMax"] = aabb.maxPoint;

        var["specSrc"] = pRayBatch->specSrc;
        var["specInput"] = pRayBatch->specInput;
        var["gBatchCounts"] = pRayBatch->counts;

        var["allPos"] = pRayBatch->pos ? pRayBatch->pos : mpBindDummyBatch->pos;
        var["allDir"] = pRayBatch->dir ? pRayBatch->dir : mpBindDummyBatch->dir;
        var["allRoughness"] = pRayBatch->roughness ? pRayBatch->roughness : mpBindDummyBatch->roughness;
        var["allVBuffer"] = pRayBatch->vbuffer ? pRayBatch->vbuffer : mpBindDummyBatch->vbuffer;

        var["gVBuffer"] = renderData.getTexture(kOutputVBuffer);
        var["gPos"] = renderData.getTexture(kOutputPos);
        var["gDir"] = renderData.getTexture(kOutputDir);
        var["gRoughness"] = renderData.getTexture(kOutputRoughness);

        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

        // Groups of NUM_SPEC_RAYS threads, each handling one glossy sample at a time.
        executePass(pRenderContext, mpConeTrace, "ConeTrace", uint3(numGroups * mNumSpecRays, 1, 1));
    }

    // Cluster slots and the envColor slot are different fields of the same element.
    pRenderContext->uavBarrier(pRayBatch->specInput.get());
    specPrimaryPass(pRenderContext, pRayBatch, renderData);
}

void NeuralRadiosity::specPrimaryPass(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "SpecPrimary");

    ShaderVar var = mpSpecPrimary->getRootVar();
    const auto name = "gSpecPrimary";
    const uint32_t numThreads = specDispatchItems(pRayBatch->specSize, pRayBatch->size);
    var[name]["dispatchThreadCount"] = numThreads;
    var[name]["frameDim"] = mFrameDim;
    var[name]["useScreenGBuffer"] = pRayBatch->needAll ? 0u : 1u;
    var[name]["enableMirrorAlbedo"] = mEnableMirrorAlbedo ? 1u : 0u;
    var[name]["envColorEncoding"] = mEnvColorEncoding;

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    var["specSrc"] = pRayBatch->specSrc;
    var["specInput"] = pRayBatch->specInput;
    var["gBatchCounts"] = pRayBatch->counts;

    var["allPos"] = pRayBatch->pos ? pRayBatch->pos : mpBindDummyBatch->pos;
    var["allDir"] = pRayBatch->dir ? pRayBatch->dir : mpBindDummyBatch->dir;
    var["allRoughness"] = pRayBatch->roughness ? pRayBatch->roughness : mpBindDummyBatch->roughness;
    var["allVBuffer"] = pRayBatch->vbuffer ? pRayBatch->vbuffer : mpBindDummyBatch->vbuffer;

    var["gVBuffer"] = renderData.getTexture(kOutputVBuffer);
    var["gPos"] = renderData.getTexture(kOutputPos);
    var["gDir"] = renderData.getTexture(kOutputDir);
    var["gRoughness"] = renderData.getTexture(kOutputRoughness);

    bindPrefiltEnv(var);
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    executePass(pRenderContext, mpSpecPrimary, "SpecPrimary", uint3(numThreads, 1, 1));
}

void NeuralRadiosity::clusterReuse(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ClusterReuse");

    // The cluster passes size their dispatches from specSize on the CPU, so this
    // path still waits for the compaction counts.
    pRenderContext->submit(true);
    mpRenderBatch->readCounts();

    // Ping-pong: the reproject pass reads [1 - write] and seeds [write], the cone
    // trace updates [write] in place, and the spatial pass only reads it.
    mClusterHistoryWriteIdx ^= 1u;

    clusterReprojectPass(pRenderContext, renderData);
    pRenderContext->uavBarrier(mpClusterHistory[mClusterHistoryWriteIdx].get());
    pRenderContext->uavBarrier(mpClusterLockState[mClusterHistoryWriteIdx].get());

    coneTraceScreenPass(pRenderContext, renderData);
    pRenderContext->uavBarrier(mpClusterHistory[mClusterHistoryWriteIdx].get());
    pRenderContext->uavBarrier(mpClusterLockState[mClusterHistoryWriteIdx].get());

    clusterSpatialReusePass(pRenderContext, renderData);

    mClusterHistoryValid = true;
}

void NeuralRadiosity::clusterReprojectPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ClusterReproject");

    const AABB& aabb = mpScene->getSceneBounds();
    const float epsScene = clusterSceneEpsilon(aabb);

    ShaderVar var = mpClusterReproject->getRootVar();
    const auto name = "gClusterReproject";

    var[name]["frameDim"] = mFrameDim;
    var[name]["frameCount"] = mFrameCount;
    var[name]["historyValid"] = mClusterHistoryValid ? 1u : 0u;
    var[name]["epsScene"] = epsScene;
    var[name]["normalThreshold"] = mClusterNormalThreshold;
    var[name]["roughnessThreshold"] = mClusterRoughnessThreshold;
    var[name]["posThresholdRel"] = mClusterPosThresholdRel;
    var[name]["motionDrift"] = mClusterMotionDrift;
    var[name]["motionMemory"] = mClusterMotionMemory;

    var["gPos"] = renderData.getTexture(kOutputPos);
    var["gDir"] = renderData.getTexture(kOutputDir);
    var["gNormal"] = renderData.getTexture(kOutputNormal);
    var["gRoughness"] = renderData.getTexture(kOutputRoughness);
    var["gPrimaryPos"] = renderData.getTexture(kOutputPrimaryPos);

    var["gPrevPos"] = mpPrevPos;
    var["gPrevDir"] = mpPrevDir;
    var["gPrevNormal"] = mpPrevNormal;
    var["gPrevRoughness"] = mpPrevRoughness;
    var["prevSpecActive"] = mpPrevSpecActive;

    var["specActive"] = mpRenderBatch->specActive;
    var["clusterHistoryIn"] = mpClusterHistory[1 - mClusterHistoryWriteIdx];
    var["clusterHistoryOut"] = mpClusterHistory[mClusterHistoryWriteIdx];
    var["clusterLockStateIn"] = mpClusterLockState[1 - mClusterHistoryWriteIdx];
    var["clusterLockStateOut"] = mpClusterLockState[mClusterHistoryWriteIdx];

    mpScene->bindShaderData(var["gScene"]);

    executePass(pRenderContext, mpClusterReproject, "ClusterReproject", uint3(mFrameDim, 1));
}

void NeuralRadiosity::coneTraceScreenPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ConeTraceScreen");

    const AABB& aabb = mpScene->getSceneBounds();
    const float epsScene = clusterSceneEpsilon(aabb);

    ShaderVar var = mpConeTraceScreen->getRootVar();
    const auto name = "gConeTraceScreen";

    var[name]["frameDim"] = mFrameDim;
    var[name]["frameCount"] = mFrameCount;
    var[name]["specSize"] = mpRenderBatch->specSize;
    var[name]["epsScene"] = epsScene;
    var[name]["assignThreshold"] = mClusterAssignThreshold;
    var[name]["maxCount"] = mClusterMaxCount;
    var[name]["weakWeight"] = mClusterWeakWeight;
    var[name]["sigmaPriorScale"] = mClusterSigmaPriorScale;
    var[name]["lockCount"] = mClusterLockCount;
    var[name]["detectThreshold"] = mClusterDetectThreshold;
    var[name]["detectRate"] = mClusterDetectRate;
    var[name]["unlockDrift"] = mClusterUnlockDrift;
    var[name]["unlockOutlierRate"] = mClusterUnlockOutlierRate;
    var[name]["maxRays"] = mClusterMaxRays;

    var["specSrc"] = mpRenderBatch->specSrc;

    var["gVBuffer"] = renderData.getTexture(kOutputVBuffer);
    var["gPos"] = renderData.getTexture(kOutputPos);
    var["gDir"] = renderData.getTexture(kOutputDir);
    var["gNormal"] = renderData.getTexture(kOutputNormal);
    var["gRoughness"] = renderData.getTexture(kOutputRoughness);
    var["clusterHistory"] = mpClusterHistory[mClusterHistoryWriteIdx];
    var["clusterLockState"] = mpClusterLockState[mClusterHistoryWriteIdx];

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    if (mpRenderBatch->specSize == 0) return;

    executePass(pRenderContext, mpConeTraceScreen, "ConeTraceScreen", uint3(mpRenderBatch->specSize, 1, 1));
}

void NeuralRadiosity::clusterSpatialReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ClusterSpatialReuse");

    const AABB& aabb = mpScene->getSceneBounds();
    const float epsScene = clusterSceneEpsilon(aabb);

    ShaderVar var = mpClusterSpatialReuse->getRootVar();
    const auto name = "gClusterSpatialReuse";

    var[name]["frameDim"] = mFrameDim;
    var[name]["specSize"] = mpRenderBatch->specSize;
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;
    var[name]["epsScene"] = epsScene;
    var[name]["assignThreshold"] = mClusterAssignThreshold;
    var[name]["lockCount"] = mClusterLockCount;
    var[name]["enableSpatialReuse"] = mEnableClusterSpatialReuse ? 1u : 0u;
    var[name]["enableMirrorAlbedo"] = mEnableMirrorAlbedo ? 1u : 0u;
    var[name]["envColorEncoding"] = mEnvColorEncoding;
    var[name]["numTaps"] = mClusterNumTaps;
    var[name]["gatherRadiusScale"] = mClusterGatherRadiusScale;
    var[name]["maxGatherRadius"] = mClusterMaxGatherRadius;
    var[name]["gammaMax"] = mClusterGammaMax;
    var[name]["gammaMin"] = mClusterGammaMin;
    var[name]["confidenceCount"] = mClusterConfidenceCount;

    var["specSrc"] = mpRenderBatch->specSrc;
    var["specInput"] = mpRenderBatch->specInput;
    var["specActive"] = mpRenderBatch->specActive;

    var["gPos"] = renderData.getTexture(kOutputPos);
    var["gDir"] = renderData.getTexture(kOutputDir);
    var["gNormal"] = renderData.getTexture(kOutputNormal);
    var["gRoughness"] = renderData.getTexture(kOutputRoughness);
    var["clusterHistory"] = mpClusterHistory[mClusterHistoryWriteIdx];

    bindPrefiltEnv(var);

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    if (mpRenderBatch->specSize == 0) return;

    executePass(pRenderContext, mpClusterSpatialReuse, "ClusterSpatialReuse", uint3(mpRenderBatch->specSize, 1, 1));
}

void NeuralRadiosity::savePrevGBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "SavePrevGBuffer");

    pRenderContext->copyResource(mpPrevPos.get(), renderData.getTexture(kOutputPos).get());
    pRenderContext->copyResource(mpPrevDir.get(), renderData.getTexture(kOutputDir).get());
    pRenderContext->copyResource(mpPrevNormal.get(), renderData.getTexture(kOutputNormal).get());
    pRenderContext->copyResource(mpPrevRoughness.get(), renderData.getTexture(kOutputRoughness).get());

    // RayBatchBuffer never shrinks, so specActive can be larger than one frame
    // of pixels. Copy exactly the region the history is indexed over.
    pRenderContext->copyBufferRegion(
        mpPrevSpecActive.get(), 0, mpRenderBatch->specActive.get(), 0, mpPrevSpecActive->getSize()
    );
}

void NeuralRadiosity::modelInferenceCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ModelInferenceCUDA");
    bakeHost(fmt::format("infer {}", batchRole(pRayBatch.get())));

    pRenderContext->waitForFalcor(mpNRModel->stream(false));
    readBatchCountsAfterSync(pRenderContext, pRayBatch);
    mpNRModel->inference(pRayBatch->getDiffPtrs(), pRayBatch->getSpecPtrs());
    pRenderContext->waitForCuda(mpNRModel->stream(false));
}

void NeuralRadiosity::modelTrainCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "ModelTrainCUDA");
    bakeHost(fmt::format("train {}", batchRole(pRayBatch.get())));

    pRenderContext->waitForFalcor(mpNRModel->stream(true));
    readBatchCountsAfterSync(pRenderContext, pRayBatch);
    mpNRModel->train(pRayBatch->getDiffPtrs(), pRayBatch->getSpecPtrs());
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

    if (!mpSpecPrimary)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSpecPrimaryFile).csEntry("main");

        DefineList primaryDefines = defines;
        primaryDefines.add("SPEC_PRIMARY_GROUP_SIZE", std::to_string(kSpecPrimaryGroupSize));
        mpSpecPrimary = ComputePass::create(mpDevice, desc, primaryDefines, true);

        ShaderVar var = mpSpecPrimary->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpClusterReproject)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kClusterReprojectFile).csEntry("main");

        mpClusterReproject = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpClusterReproject->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpConeTraceScreen)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kConeTraceScreenFile).csEntry("main");

        DefineList screenDefines = defines;
        screenDefines.add("CONE_TRACE_SCREEN_GROUP_SIZE", std::to_string(kClusterGroupSize));
        mpConeTraceScreen = ComputePass::create(mpDevice, desc, screenDefines, true);

        // Bind static resources
        ShaderVar var = mpConeTraceScreen->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpClusterSpatialReuse)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kClusterSpatialReuseFile).csEntry("main");

        DefineList spatialDefines = defines;
        spatialDefines.add("CLUSTER_SPATIAL_REUSE_GROUP_SIZE", std::to_string(kClusterGroupSize));
        mpClusterSpatialReuse = ComputePass::create(mpDevice, desc, spatialDefines, true);

        // Bind static resources
        ShaderVar var = mpClusterSpatialReuse->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    const auto batchEndTime = std::chrono::steady_clock::now();
    const auto batchElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(batchEndTime - passEndTime).count();
    logInfo("Batch passes created: \t{} ms", batchElapsedMs);

    if (!mpNRModel)
    {
        mpNRModel = std::make_unique<NRModel>();
        mpNRModel->setOnline(mRenderMode != RenderMode::Train);

        if (!mLoadModelPath.empty())
        {
            logInfo("NeuralRadiosity: loading model state from '{}'", mLoadModelPath);
            try
            {
                mpNRModel->loadState(mLoadModelPath);
            }
            catch (const std::exception& e)
            {
                logError("NeuralRadiosity: failed to load model state '{}': {}", mLoadModelPath, e.what());
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

    if (!mpBindDummyBatch)
        mpBindDummyBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, 1, mNumClusters, "BindDummy", true, true, true);
    if (!mpRenderBatch)
    {
        // Shared diffuse/specular network I/O binds one buffer as both
        // RWStructuredBuffer<DiffNetInput> and <SpecNetInput>. Vulkan addresses each
        // binding with the shader's stride; D3D12 views use the buffer's struct size
        // (SpecNetInput), which would misplace every diffuse row. D3D12 keeps
        // separate buffers.
        const bool sharedIO = mpDevice->getType() == Device::Type::Vulkan;
        mpRenderBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mFrameDim.x * mFrameDim.y, mNumClusters, "RenderBatch", false, true, true, sharedIO);
    }
    if (!mpTrainLHSBatch)
        mpTrainLHSBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mBatchSize, mNumClusters, "LHSBatch", true, true, true);
    if (!mpTrainRHSBatch)
        mpTrainRHSBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mBatchSize * mNumRHS, mNumClusters, "RHSBatch", true, true, false);

    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        mpRenderBatch->resize(mFrameDim.x * mFrameDim.y, mNumClusters);
    }
    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
    {
        mpTrainLHSBatch->resize(mBatchSize, mNumClusters);
        mpTrainRHSBatch->resize(mBatchSize * mNumRHS, mNumClusters);
    }

    // Spatiotemporal cluster reuse resources. The history is addressed per pixel
    // (not per compacted index) because both the reprojection and the spatial
    // gather need to reach arbitrary neighbouring pixels. With reuse off the
    // render path uses ConeTrace, so these copies are released.
    if (!mEnableRenderSpecTemporalReuse)
    {
        for (uint32_t i = 0; i < 2; ++i)
        {
            mpClusterHistory[i] = nullptr;
            mpClusterLockState[i] = nullptr;
        }
        mpPrevSpecActive = nullptr;
        mpPrevPos = nullptr;
        mpPrevDir = nullptr;
        mpPrevNormal = nullptr;
        mpPrevRoughness = nullptr;
        mClusterHistoryValid = false;
    }
    else
    {
        const uint32_t pixelCount = mFrameDim.x * mFrameDim.y;

        const bool historyResized =
            !mpClusterHistory[0] || mpClusterHistory[0]->getElementCount() != pixelCount ||
            !mpClusterLockState[0] || mpClusterLockState[0]->getElementCount() != pixelCount;

        if (pixelCount > 0 && historyResized)
        {
            ShaderVar reprojectVar = mpClusterReproject->getRootVar();
            for (uint32_t i = 0; i < 2; ++i)
            {
                mpClusterHistory[i] = mpDevice->createStructuredBuffer(
                    reprojectVar["clusterHistoryOut"],
                    pixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    MemoryType::DeviceLocal,
                    nullptr,
                    false
                );
                mpClusterHistory[i]->setName(fmt::format("NeuralRadiosity.ClusterHistory{}", i));

                mpClusterLockState[i] = mpDevice->createStructuredBuffer(
                    reprojectVar["clusterLockStateOut"],
                    pixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    MemoryType::DeviceLocal,
                    nullptr,
                    false
                );
                mpClusterLockState[i]->setName(fmt::format("NeuralRadiosity.ClusterLockState{}", i));
            }

            mpPrevSpecActive = mpDevice->createStructuredBuffer(
                reprojectVar["prevSpecActive"],
                pixelCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
            mpPrevSpecActive->setName("NeuralRadiosity.PrevSpecActive");

            clearRenderTemporalHistory(pRenderContext);
        }

        const bool prevResized =
            !mpPrevPos || mpPrevPos->getWidth() != mFrameDim.x || mpPrevPos->getHeight() != mFrameDim.y;

        if (pixelCount > 0 && prevResized)
        {
            // Exact copies of the render-graph G-Buffer channels: the previous
            // frame's (pos, dir, normal, roughness) is what turns a stored t_k
            // back into a world anchor.
            const auto createPrev = [&](ResourceFormat format, const char* name)
            {
                ref<Texture> pTex = mpDevice->createTexture2D(
                    mFrameDim.x, mFrameDim.y, format, 1, 1, nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
                pTex->setName(name);
                return pTex;
            };

            mpPrevPos = createPrev(ResourceFormat::RGBA32Float, "NeuralRadiosity.PrevPos");
            mpPrevDir = createPrev(ResourceFormat::RGBA32Float, "NeuralRadiosity.PrevDir");
            mpPrevNormal = createPrev(ResourceFormat::RGBA32Float, "NeuralRadiosity.PrevNormal");
            mpPrevRoughness = createPrev(ResourceFormat::R32Float, "NeuralRadiosity.PrevRoughness");

            clearRenderTemporalHistory(pRenderContext);
        }
    }

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

    const auto endTime = std::chrono::steady_clock::now();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    logInfo("NeuralRadiosity::prepareResources() took {} ms", elapsedMs);
}

void NeuralRadiosity::clearRenderTemporalHistory(RenderContext* pRenderContext)
{
    // Note: camera motion deliberately does *not* land here. World-anchored
    // clusters are validated one at a time against the current cone, so a global
    // clear is only needed when the anchors themselves become meaningless
    // (scene geometry moved, resource resize, or an explicit user request).
    if (!pRenderContext) return;

    for (uint32_t i = 0; i < 2; ++i)
    {
        if (mpClusterHistory[i]) pRenderContext->clearUAV(mpClusterHistory[i]->getUAV().get(), uint4(0));
        if (mpClusterLockState[i]) pRenderContext->clearUAV(mpClusterLockState[i]->getUAV().get(), uint4(0));
    }

    if (mpPrevSpecActive) pRenderContext->clearUAV(mpPrevSpecActive->getUAV().get(), uint4(0));

    mClusterHistoryValid = false;
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

void NeuralRadiosity::bindPrefiltEnv(ShaderVar& var)
{
    var["gPrefiltEnv"]["mipCount"] = mpPrefiltEnvCubemap ? mpPrefiltEnvCubemap->getMipCount() : 0u;
    var["gPrefiltEnv"]["faceSize"] = mpPrefiltEnvCubemap ? mpPrefiltEnvCubemap->getWidth() : 0u;
    var["gPrefiltEnv"]["valid"] = mpPrefiltEnvCubemap ? 1u : 0u;
    var["gPrefiltEnv"]["fallbackValid"] = (mpScene && mpScene->useEnvLight()) ? 1u : 0u;
    var["gPrefiltEnv"]["cubemap"].setTexture(mpPrefiltEnvCubemap);
    var["gPrefiltEnv"]["sampler"].setSampler(mpPrefiltEnvSampler);
}

void NeuralRadiosity::bindRayBatchData(ShaderVar& var, std::shared_ptr<RayBatchBuffer> pRayBatch, const std::string &name)
{
    FALCOR_ASSERT(mpBindDummyBatch);

    // Bind parameters
    var[name]["batchSize"] = pRayBatch->size;
    var[name]["batchCount"] = mFrameCount;
    var[name]["diffSize"] = pRayBatch->diffSize;
    var[name]["specSize"] = pRayBatch->specSize;
    var[name]["enableDiff"] = pRayBatch->needDiff ? 1u : 0u;
    var[name]["enableSpec"] = pRayBatch->needSpec ? 1u : 0u;

    const auto bind = [&](const char* field, const ref<Buffer>& buf, const ref<Buffer>& dummy)
    {
        var[field] = buf ? buf : dummy;
    };

    bind("allPos", pRayBatch->pos, mpBindDummyBatch->pos);
    bind("allDir", pRayBatch->dir, mpBindDummyBatch->dir);
    bind("allNormal", pRayBatch->normal, mpBindDummyBatch->normal);
    bind("allAlbedo", pRayBatch->albedo, mpBindDummyBatch->albedo);
    bind("allRoughness", pRayBatch->roughness, mpBindDummyBatch->roughness);
    bind("allVBuffer", pRayBatch->vbuffer, mpBindDummyBatch->vbuffer);
    bind("allColor", pRayBatch->color, mpBindDummyBatch->color);
    bind("allEmission", pRayBatch->emission, mpBindDummyBatch->emission);

    bind("diffActive", pRayBatch->diffActive, mpBindDummyBatch->diffActive);
    bind("diffIndex", pRayBatch->diffIndex, mpBindDummyBatch->diffIndex);
    bind("diffInput", pRayBatch->diffInput, mpBindDummyBatch->diffInput);
    bind("diffColor", pRayBatch->diffColor, mpBindDummyBatch->diffColor);

    bind("specActive", pRayBatch->specActive, mpBindDummyBatch->specActive);
    bind("specIndex", pRayBatch->specIndex, mpBindDummyBatch->specIndex);
    bind("specInput", pRayBatch->specInput, mpBindDummyBatch->specInput);
    bind("specColor", pRayBatch->specColor, mpBindDummyBatch->specColor);
    bind("specSrc", pRayBatch->specSrc, mpBindDummyBatch->specSrc);
}

DefineList NeuralRadiosity::getShaderDefines(const RenderData& renderData) const
{
    DefineList defines;

    // Seed the sample generator without 64-bit integers (same sequence): the OpenHarmony port replays
    // these shaders on a GPU that has none.
    defines.add("FALCOR_SPLITMIX64_32BIT", "1");
    defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
    defines.add("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mUseEnvLight ? "1" : "0");
    defines.add("SPECULAR_ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    // Add cone trace parameters
    defines.add("NUM_KMEANS_ITERS", std::to_string(mNumKMeansIters));
    defines.add("NUM_SPEC_RAYS", std::to_string(mNumSpecRays));
    defines.add("NUM_CLUSTERS", std::to_string(mNumClusters));
    defines.add("N_LEVELS", std::to_string(NCR_N_LEVELS));
    defines.add("N_FEATURES_PER_LEVEL", std::to_string(NCR_N_FEATURES_PER_LEVEL));
    defines.add("N_LEVELS_SPEC", std::to_string(NCR_N_LEVELS_SPEC));
    defines.add("N_FEATURES_PER_LEVEL_SPEC", std::to_string(NCR_N_FEATURES_PER_LEVEL_SPEC));
    defines.add("OUTPUT_DIM", std::to_string(NCR_OUTPUT_DIM));
    defines.add("NR_USE_FP16", std::to_string(NCR_USE_FP16));
    // Add sample RHS parameters
    defines.add("NUM_RHS", std::to_string(mNumRHS));
    // Add random smooth parameters
    defines.add("RS_RR_PROB", std::to_string(mRSRRProb));
    defines.add("RS_RANDOM_WI", mRandomTrainDir ? "1" : "0");
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
    // With cluster reuse enabled, ConeTrace is only reached by the training
    // batches, so it can stay specialized for T=128 even in Render/OnlineTrain
    // mode. Without it the render path still needs the cheaper T=32 variant,
    // unless the user explicitly asked for the T=128 reference.
    const bool useTrainConfig = train || mEnableRenderSpecTemporalReuse || mUseReferenceConeTrace;

    const uint32_t numSpecRays = useTrainConfig ? NCR_NUM_SPEC_RAYS_TRAIN : NCR_NUM_SPEC_RAYS_RENDER;
    const uint32_t numKMeansIters = useTrainConfig ? NCR_NUM_KMEANS_ITERS_TRAIN : NCR_NUM_KMEANS_ITERS_RENDER;

    if (mNumSpecRays != numSpecRays || mNumKMeansIters != numKMeansIters)
    {
        mNumSpecRays = numSpecRays;
        mNumKMeansIters = numKMeansIters;
        mpConeTrace = nullptr;
        mVarsChanged = true;
    }

    if (train) updateAdaptiveRHSState(0, true);
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

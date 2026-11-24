#include "NeuralRadiosity.h"

#include "Utils/Math/VectorJson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
    const char kEnableClusterVisibilityCheck[] = "enableClusterVisibilityCheck";
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
    const char kEnableIsolatedHighlightFilter[] = "enableIsolatedHighlightFilter";
    const char kEnableFXAA[] = "enableFXAA";
    const char kEnableMirrorAlbedo[] = "enableMirrorAlbedo";
    const char kEnvColorEncoding[] = "envColorEncoding";
    const char kUseNEE[] = "useNEE";
    const char kDiffTrainOnGlossy[] = "diffTrainOnGlossy";
    const char kRandomTrainDir[] = "randomTrainDir";
    const char kRhsGlossyBounces[] = "rhsGlossyBounces";
    const char kEnableTrainSchedule[] = "enableTrainSchedule";
    const char kTrainLrHot[] = "trainLrHot";
    const char kTrainLrCold[] = "trainLrCold";
    const char kTrainEmaHot[] = "trainEmaHot";
    const char kTrainEmaCold[] = "trainEmaCold";
    const char kTrainInitialHotFrames[] = "trainInitialHotFrames";
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
    const char kNumSpecRaysTrain[] = "numSpecRaysTrain";
    const char kNumSpecRaysRender[] = "numSpecRaysRender";
    const char kNumKMeansItersTrain[] = "numKMeansItersTrain";
    const char kNumKMeansItersRender[] = "numKMeansItersRender";
    const char kBatchSize[] = "batchSize";
    const char kNumRHS[] = "numRHS";
    const char kRsRRProb[] = "rsRRProb";
    const char kSpecularRoughnessThreshold[] = "specularRoughnessThreshold";
    const char kNormalizeClsScale[] = "normalizeClsScale";
    const char kNLevels[] = "nLevels";
    const char kNFeaturesPerLevel[] = "nFeaturesPerLevel";
    const char kLog2HashmapSize[] = "log2HashmapSize";
    const char kBaseResolution[] = "baseResolution";
    const char kPerLevelScale[] = "perLevelScale";
    const char kNLevelsSpec[] = "nLevelsSpec";
    const char kNFeaturesPerLevelSpec[] = "nFeaturesPerLevelSpec";
    const char kLog2HashmapSizeSpec[] = "log2HashmapSizeSpec";
    const char kBaseResolutionSpec[] = "baseResolutionSpec";
    const char kPerLevelScaleSpec[] = "perLevelScaleSpec";
    const char kInterpRatio[] = "interpRatio";
    const char kInitSeed[] = "initSeed";
    const char kInitGridScale[] = "initGridScale";
    const char kInitDiagnostics[] = "initDiagnostics";
    const char kDiffNNeurons[] = "diffNNeurons";
    const char kDiffNHiddenLayers[] = "diffNHiddenLayers";
    const char kSpecNNeurons[] = "specNNeurons";
    const char kSpecNHiddenLayers[] = "specNHiddenLayers";

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
    pass.def("configureDiagnostics", &NeuralRadiosity::configureDiagnostics);
    pass.def_property_readonly("diagnostics", &NeuralRadiosity::getDiagnostics);
    pass.def("inspectRenderPixel", &NeuralRadiosity::inspectRenderPixel);
    pass.def("setDiagnosticRenderOptions", &NeuralRadiosity::setDiagnosticRenderOptions);
    pass.def("setDiagnosticSpatialReuse", &NeuralRadiosity::setDiagnosticSpatialReuse);
    pass.def("setDiagnosticSpatialReuseLegacy", &NeuralRadiosity::setDiagnosticSpatialReuseLegacy);
    pass.def("setDiagnosticFarModeValidation", &NeuralRadiosity::setDiagnosticFarModeValidation);
    pass.def("setDiagnosticClusterRayBudget", &NeuralRadiosity::setDiagnosticClusterRayBudget);
    pass.def("reapplyDiagnosticPostFilterPlacement", &NeuralRadiosity::reapplyDiagnosticPostFilterPlacement,
        pybind11::arg("placement"), pybind11::arg("minExcess") = 0.1f, pybind11::arg("requireQuietRing") = false);
    pass.def("reapplyDiagnosticPostFilter", &NeuralRadiosity::reapplyDiagnosticPostFilter);
    pass.def("getDiagnosticPostFilterInput", &NeuralRadiosity::getDiagnosticPostFilterInput);
    pass.def("getFarModeValidationStats", &NeuralRadiosity::getFarModeValidationStats);
    pass.def("setDiagnosticHistoryFrozen", &NeuralRadiosity::setDiagnosticHistoryFrozen);
    pass.def_property("trainingPaused", &NeuralRadiosity::getTrainingPaused, &NeuralRadiosity::setTrainingPaused);
    pass.def("requestCameraTrainingHeat", &NeuralRadiosity::requestCameraTrainingHeat);
    pass.def_property_readonly("trainingBatchCount", &NeuralRadiosity::getTrainingBatchCount);
    pass.def_property_readonly("trainDiffuseCount", &NeuralRadiosity::getTrainDiffuseCount);
    pass.def_property_readonly("trainSpecularCount", &NeuralRadiosity::getTrainSpecularCount);
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
        else if (key == kEnableClusterVisibilityCheck)
            mEnableClusterVisibilityCheck = value;
        else if (key == kEnableIsolatedHighlightFilter)
            mEnableIsolatedHighlightFilter = value;
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
        else if (key == kTrainInitialHotFrames)
            mTrainInitialHotFrames = value;
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
        else if (key == kNumSpecRaysTrain)
            mNumSpecRaysTrain = value;
        else if (key == kNumSpecRaysRender)
            mNumSpecRaysRender = value;
        else if (key == kNumKMeansItersTrain)
            mNumKMeansItersTrain = value;
        else if (key == kNumKMeansItersRender)
            mNumKMeansItersRender = value;
        else if (key == kBatchSize)
            mBatchSizeInit = value;
        else if (key == kNumRHS)
            mNumRHSInit = value;
        else if (key == kRsRRProb)
            mRSRRProb = value;
        else if (key == kSpecularRoughnessThreshold)
            mSpecularRoughnessThreshold = value;
        else if (key == kNormalizeClsScale)
            mNormalizeClsScale = value;
        else if (key == kNLevels)
            mModelConfig.nLevels = value;
        else if (key == kNFeaturesPerLevel)
            mModelConfig.nFeaturesPerLevel = value;
        else if (key == kLog2HashmapSize)
            mModelConfig.log2HashmapSize = value;
        else if (key == kBaseResolution)
            mModelConfig.baseResolution = value;
        else if (key == kPerLevelScale)
            mModelConfig.perLevelScale = value;
        else if (key == kNLevelsSpec)
            mModelConfig.nLevelsSpec = value;
        else if (key == kNFeaturesPerLevelSpec)
            mModelConfig.nFeaturesPerLevelSpec = value;
        else if (key == kLog2HashmapSizeSpec)
            mModelConfig.log2HashmapSizeSpec = value;
        else if (key == kBaseResolutionSpec)
            mModelConfig.baseResolutionSpec = value;
        else if (key == kPerLevelScaleSpec)
            mModelConfig.perLevelScaleSpec = value;
        else if (key == kInterpRatio)
            mModelConfig.interpRatio = value;
        else if (key == kInitSeed)
            mModelConfig.initSeed = value;
        else if (key == kInitGridScale)
            mModelConfig.initGridScale = value;
        else if (key == kInitDiagnostics)
            mModelConfig.initDiagnostics = value;
        else if (key == kDiffNNeurons)
            mModelConfig.diffNNeurons = value;
        else if (key == kDiffNHiddenLayers)
            mModelConfig.diffNHiddenLayers = value;
        else if (key == kSpecNNeurons)
            mModelConfig.specNNeurons = value;
        else if (key == kSpecNHiddenLayers)
            mModelConfig.specNHiddenLayers = value;
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
        else if (key == "trainWarmupFrames")
            FALCOR_THROW("trainWarmupFrames was renamed to trainInitialHotFrames (learning-rate heat hold).");
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

    // Reject invalid startup values before allocating buffers or compiling shaders.
    auto require = [](bool valid, const char* message) { if (!valid) FALCOR_THROW("{}", message); };
    auto unit = [](float x) { return std::isfinite(x) && x >= 0.f && x <= 1.f; };
    require(unit(mTrainCameraHeat) && unit(mTrainSceneHeat) && unit(mTrainLightMotionHeat) && unit(mTrainObjectMotionHeat),
        "Training heat must be in [0, 1].");
    require(std::isfinite(mTrainLrHot) && mTrainLrHot > 0.f && std::isfinite(mTrainLrCold) && mTrainLrCold > 0.f,
        "Learning rates must be finite and positive.");
    require(unit(mTrainEmaHot) && mTrainEmaHot < 1.f && unit(mTrainEmaCold) && mTrainEmaCold < 1.f,
        "EMA decay must be in [0, 1).");
    require(std::isfinite(mTrainCoolFrames) && mTrainCoolFrames > 0.f, "Cooling time must be positive.");
    require(mNumSpecRaysTrain >= NCR_NUM_CLUSTERS && mNumSpecRaysTrain <= 1024 &&
        mNumSpecRaysRender >= NCR_NUM_CLUSTERS && mNumSpecRaysRender <= 1024, "Specular ray counts must be in [NUM_CLUSTERS, 1024].");
    require(mNumKMeansItersTrain > 0 && mNumKMeansItersRender > 0, "K-means iteration counts must be positive.");
    require(mBatchSizeInit >= kModelBatchAlign && mBatchSizeInit % kModelBatchAlign == 0 && mNumRHSInit > 0 &&
        uint64_t(mBatchSizeInit) * mNumRHSInit <= uint64_t(UINT32_MAX) - kModelBatchAlign && mNumRHSInit <= UINT32_MAX / 8,
        "Batch size must be a positive multiple of 256, with a non-overflowing RHS count.");
    require(unit(mRSRRProb) && mRSRRProb > 0.f && unit(mSpecularRoughnessThreshold), "Invalid RR probability or roughness threshold.");
    require(mEnvColorEncoding <= 2, "Environment encoding must be 0, 1 or 2.");
    auto checkGrid = [&](uint32_t levels, uint32_t features, uint32_t hash, uint32_t base, float scale) {
        require(levels >= 1 && levels <= 8 && (features == 2 || features == 4 || features == 8), "Grid levels must be 1..8 and features 2, 4 or 8.");
        require(hash >= 1 && hash <= 24 && base >= 1 && std::isfinite(scale) && scale > 1.f,
            "Grid hash size must be 1..24, base resolution positive, and level scale greater than 1.");
        require(double(base) * std::pow(double(scale), levels - 1) <= 1000000., "Grid resolution is too large.");
    };
    checkGrid(mModelConfig.nLevels, mModelConfig.nFeaturesPerLevel, mModelConfig.log2HashmapSize, mModelConfig.baseResolution, mModelConfig.perLevelScale);
    checkGrid(mModelConfig.nLevelsSpec, mModelConfig.nFeaturesPerLevelSpec, mModelConfig.log2HashmapSizeSpec, mModelConfig.baseResolutionSpec, mModelConfig.perLevelScaleSpec);
    auto width = [](uint32_t n) { return n == 16 || n == 32 || n == 64 || n == 128; };
    require(width(mModelConfig.diffNNeurons) && width(mModelConfig.specNNeurons), "Fused MLP width must be 16, 32, 64 or 128.");
    require(mModelConfig.diffNHiddenLayers > 0 && mModelConfig.specNHiddenLayers > 0, "MLPs need at least one hidden layer.");
    require(std::isfinite(mModelConfig.interpRatio) && mModelConfig.interpRatio > 0.f, "Interpolation ratio must be positive.");
    require(std::isfinite(mModelConfig.initGridScale) && mModelConfig.initGridScale > 0.f && mModelConfig.initGridScale <= 1.f,
        "Initialization grid scale must be in (0, 1].");
    mBatchSize = mBatchSizeInit;
    mNumRHS = mNumRHSInit;
    resetTrainSchedule();

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
    props[kEnableClusterVisibilityCheck] = mEnableClusterVisibilityCheck;
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
    props[kEnableIsolatedHighlightFilter] = mEnableIsolatedHighlightFilter;
    props[kEnableFXAA] = mEnableFXAA;
    props[kEnableMirrorAlbedo] = mEnableMirrorAlbedo;
    props[kEnvColorEncoding] = mEnvColorEncoding;
    props[kUseNEE] = mUseNEE;
    props[kDiffTrainOnGlossy] = mDiffTrainOnGlossy;
    props[kRandomTrainDir] = mRandomTrainDir;
    props[kRhsGlossyBounces] = mRhsGlossyBounces;
    props[kEnableTrainSchedule] = mEnableTrainSchedule;
    props[kTrainLrHot] = mTrainLrHot;
    props[kTrainLrCold] = mTrainLrCold;
    props[kTrainEmaHot] = mTrainEmaHot;
    props[kTrainEmaCold] = mTrainEmaCold;
    props[kTrainInitialHotFrames] = mTrainInitialHotFrames;
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
    props[kNumSpecRaysTrain] = mNumSpecRaysTrain;
    props[kNumSpecRaysRender] = mNumSpecRaysRender;
    props[kNumKMeansItersTrain] = mNumKMeansItersTrain;
    props[kNumKMeansItersRender] = mNumKMeansItersRender;
    props[kBatchSize] = mBatchSizeInit;
    props[kNumRHS] = mNumRHSInit;
    props[kRsRRProb] = mRSRRProb;
    props[kSpecularRoughnessThreshold] = mSpecularRoughnessThreshold;
    props[kNormalizeClsScale] = mNormalizeClsScale;
    props[kNLevels] = mModelConfig.nLevels;
    props[kNFeaturesPerLevel] = mModelConfig.nFeaturesPerLevel;
    props[kLog2HashmapSize] = mModelConfig.log2HashmapSize;
    props[kBaseResolution] = mModelConfig.baseResolution;
    props[kPerLevelScale] = mModelConfig.perLevelScale;
    props[kNLevelsSpec] = mModelConfig.nLevelsSpec;
    props[kNFeaturesPerLevelSpec] = mModelConfig.nFeaturesPerLevelSpec;
    props[kLog2HashmapSizeSpec] = mModelConfig.log2HashmapSizeSpec;
    props[kBaseResolutionSpec] = mModelConfig.baseResolutionSpec;
    props[kPerLevelScaleSpec] = mModelConfig.perLevelScaleSpec;
    props[kInterpRatio] = mModelConfig.interpRatio;
    props[kInitSeed] = mModelConfig.initSeed;
    props[kInitGridScale] = mModelConfig.initGridScale;
    props[kInitDiagnostics] = mModelConfig.initDiagnostics;
    props[kDiffNNeurons] = mModelConfig.diffNNeurons;
    props[kDiffNHiddenLayers] = mModelConfig.diffNHiddenLayers;
    props[kSpecNNeurons] = mModelConfig.specNNeurons;
    props[kSpecNHiddenLayers] = mModelConfig.specNHiddenLayers;
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

    // A sudden geometry edit must be checked with the previous direction before
    // replacing its reference. Continuous motion still advances the windows.
    // This clock is independent of the training schedule and works while paused.
    mClusterVisibilityFrameAdvanced = !mClusterGeometryMoved || mClusterGeometryMovedPreviousFrame;
    if (mClusterVisibilityFrameAdvanced) ++mClusterVisibilityFrame;
    mClusterGeometryMovedPreviousFrame = mClusterGeometryMoved;
    mClusterGeometryMoved = false;

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

    const bool diagnose = mDiagnosticInterval && ((mFrameCount + 1) % mDiagnosticInterval == 0 || mFrameCount == 0);
    mpNRModel->setTrainingDiagnostics(diagnose);
    // Train the model. Scripted pauses leave optimizer/EMA/schedule untouched.
    if (!mTrainingPaused && (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain))
    {
        updateTrainSchedule();
        train(pRenderContext, renderData);
        ++mTrainingBatchCount;
    }

    // Render the scene.
    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        render(pRenderContext, renderData);
    }

    bakeExport(pRenderContext);
    if (diagnose) runDiagnostics(pRenderContext, renderData);
    bakeTrack(pRenderContext);
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
    widget.tooltip("Trace one mirror ray: ordinary surfaces use albedo; emissive surfaces use -log(1 + emission).");
    {
        static const Gui::DropdownList kEnvEncodings = {{0, "Linear"}, {1, "Signed log"}, {2, "Log"}};
        widget.dropdown("Env Color Encoding", kEnvEncodings, mEnvColorEncoding);
        widget.tooltip(
            "How a mirror-ray miss enters the shared env/albedo input channels.\n"
            "Signed log: -log(1 + env). Ordinary surfaces keep albedo; emissive surfaces always use -log(1 + emission)."
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
        group.var("Initial hot frames", mTrainInitialHotFrames, 0u, 10000u);
        group.tooltip("Frames held fully hot after the scene loads or the schedule is reset.");
        group.var("Hold frames", mTrainHoldFrames, 0u, 10000u);
        group.tooltip("Minimum frames held after an event raises the heat.");
        group.var("Cool frames", mTrainCoolFrames, 1.f, 10000.f, 1.f);
        group.tooltip("Time constant of the decay toward cold once nothing changes.");
        group.var("Camera heat", mTrainCameraHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Heat while the camera moves. The network is world-space, so this only speeds up newly seen regions.");
        group.var("Scene heat", mTrainSceneHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Heat for scene edits, or scene changes while the camera moves.");
        group.var("Light motion heat", mTrainLightMotionHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Fixed camera: moving analytic lights or emissive geometry.");
        group.var("Object motion heat", mTrainObjectMotionHeat, 0.f, 1.f, 0.05f);
        group.tooltip("Fixed camera: moving non-emissive geometry only.");
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
        if (group.checkbox("Visibility Change Detection", mEnableClusterVisibilityCheck)) clearHistory();
        group.tooltip("Detect reflected geometry changes with a repeated ray (plus an old-direction check every 8 frames). Unchanged pixels stay locked.");

        group.checkbox("Clear On Geometry Moved", mClearHistoryOnGeometryMoved);
        group.tooltip(
            "Off by default. A scene with a playing animation raises GeometryMoved\n"
            "every frame, and clearing on it would keep the history permanently empty\n"
            "(1 unfiltered ray per pixel). Local visibility changes are handled by\n"
            "the visibility probe, cone test and outlier path instead."
        );

        group.var("Max Rays / Pixel", mClusterMaxRays, 1u, 4u);
        group.tooltip("Accumulation/outlier ray budget. Visibility detection adds 1 ray, or 2 on a probe-window change.");

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
    if (mEnablePostFilter) widget.checkbox("Isolated Highlight Filter", mEnableIsolatedHighlightFilter);
    if (mEnablePostFilter && widget.checkbox("FXAA", mEnableFXAA))
    {
        mVarsChanged = true;
    }
}

void NeuralRadiosity::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mDiagnosticHistoryFrozen = false;
    mDiagnosticSpatialReuseLegacy = false;
    mEnableFarModeValidation = true;
    mEnableClusterRayBudget = true;
    mClusterVisibilityFrame = 0;
    mClusterVisibilityFrameAdvanced = true;
    mClusterGeometryMoved = false;
    mClusterGeometryMovedPreviousFrame = false;
    mpFarValidationStats.reset();
    mpDiagnosticLHS.reset();
    mpDiagnosticRHS.reset();
    mpFirstSmoothPass = nullptr;
    mpConeTrace = nullptr;
    mpSpecPrimary = nullptr;
    mpClusterReproject = nullptr;
    mpConeTraceScreen = nullptr;
    mpClusterSpatialReuse = nullptr;
    mpResolvePass = nullptr;
    mpPostFilterPass = nullptr;
    mpIsolatedHighlightPass = nullptr;
    mpPostFilterOutputColor.reset();
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
    using F = IScene::UpdateFlags;
    mClusterGeometryMoved |= is_set(sceneUpdates, F::GeometryMoved) ||
        is_set(sceneUpdates, F::CurvesMoved) || is_set(sceneUpdates, F::CustomPrimitivesMoved);

    // Neither camera nor geometry motion invalidates history wholesale. Clusters
    // carry world anchors that are validated one at a time against the current
    // cone. The shared baseline visibility check detects reflected geometry
    // changes that keep hitting an old depth mode or escape to the environment.
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
    mTrainHoldLeft = mTrainInitialHotFrames;
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

    const F sceneEdit = F::LightIntensityChanged | F::LightPropertiesChanged | F::LightCountChanged |
                        F::MaterialsChanged | F::EmissiveMaterialsChanged | F::EnvMapChanged |
                        F::EnvMapPropertiesChanged | F::GeometryChanged | F::MeshesChanged |
                        F::DisplacementChanged | F::SDFGeometryChanged;
    const F lightMotion = F::LightsMoved | F::LightCollectionChanged;
    const F objectMotion = F::GeometryMoved | F::CurvesMoved | F::CustomPrimitivesMoved;
    const F cameraChange = F::CameraMoved | F::CameraSwitched | F::CameraPropertiesChanged;
    const bool cameraMoved = is_set(updates, cameraChange);
    const bool edited = is_set(updates, sceneEdit);
    const bool lightMoved = is_set(updates, lightMotion);
    const bool objectMoved = is_set(updates, objectMotion);

    float target = cameraMoved ? mTrainCameraHeat : 0.f;
    // Keep the existing fully-hot scene-change rule while the camera moves.
    if (edited || (cameraMoved && (lightMoved || objectMoved))) target = std::max(target, mTrainSceneHeat);
    else
    {
        if (lightMoved) target = std::max(target, mTrainLightMotionHeat);
        if (objectMoved) target = std::max(target, mTrainObjectMotionHeat);
    }

    // The event classes, for the schedule log and the OpenHarmony bake's track.
    uint32_t events = 0;
    if (cameraMoved) events |= kTrainEventCamera;
    if (edited) events |= kTrainEventEdit;
    if (lightMoved) events |= kTrainEventLight;
    if (objectMoved) events |= kTrainEventObject;
    mLastTrainEvents = events;

    const float prevHeat = mTrainHeat;
    // Count initial-hot/hold time even when repeated events equal the current heat.
    // Only an actual increase refreshes the hold; otherwise continuous motion
    // can keep a residual initial-hot counter alive forever.
    if (target > mTrainHeat)
    {
        mTrainHeat = target;
        mTrainHoldLeft = std::max(mTrainHoldLeft, mTrainHoldFrames);
    }
    else if (mTrainHoldLeft > 0)
        mTrainHoldLeft--;
    else
    {
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
    if (!mEnablePostFilter) return;
    mpPostFilterOutputColor = renderData.getTexture(kOutputColor);
    const auto name = "gPostFilterPass";
    {
        ShaderVar var = mpIsolatedHighlightPass->getRootVar();
        bindScreenData(var, renderData, name);
        var["specActive"] = mpRenderBatch->specActive;
        var["tempColor"] = mpRenderTempColor;
    }
    {
        ShaderVar var = mpPostFilterPass->getRootVar();
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
        var["gFilteredColor"] = mpPostFilterTempColor;
    }
    {
        ShaderVar var = mpPostFXAAPass->getRootVar();
        bindScreenData(var, renderData, name);
        var[name]["enableFXAA"] = mEnableFXAA ? 1u : 0u;
        var["gFxaaInput"] = mpPostFilterTempColor;
    }
    dispatchPostFilters(pRenderContext, mEnableIsolatedHighlightFilter);
}

void NeuralRadiosity::dispatchPostFilters(RenderContext* pRenderContext, bool isolatedHighlight, uint32_t placement, float minExcess, bool requireQuietRing)
{
    // Ping-pong existing output/scratch textures. Raw resolve stays immutable for
    // exact A/B replay; each dispatch has distinct input and output resources.
    const auto bilateral = [&](const ref<Texture>& input, const ref<Texture>& output)
    {
        auto var = mpPostFilterPass->getRootVar();
        var["tempColor"] = input;
        var["gFilteredColor"] = output;
        executePass(pRenderContext, mpPostFilterPass, "PostFilterPass", uint3(mFrameDim, 1));
        pRenderContext->uavBarrier(output.get());
    };
    const auto fxaa = [&](const ref<Texture>& input, const ref<Texture>& output)
    {
        auto var = mpPostFXAAPass->getRootVar();
        var["gFxaaInput"] = input;
        var["gColor"] = output;
        executePass(pRenderContext, mpPostFXAAPass, "PostFXAAPass", uint3(mFrameDim, 1));
        pRenderContext->uavBarrier(output.get());
    };
    const auto suppress = [&](const ref<Texture>& input, const ref<Texture>& output)
    {
        FALCOR_PROFILE(pRenderContext, "IsolatedHighlightFilter");
        auto var = mpIsolatedHighlightPass->getRootVar();
        var["gPostFilterPass"]["highlightMinExcess"] = minExcess;
        var["gPostFilterPass"]["highlightRequireQuietRing"] = (placement < 3 || requireQuietRing) ? 1u : 0u;
        var["tempColor"] = input;
        var["gColor"] = output;
        executePass(pRenderContext, mpIsolatedHighlightPass, "IsolatedHighlightPass", uint3(mFrameDim, 1));
        pRenderContext->uavBarrier(output.get());
    };
    if (!isolatedHighlight || placement == 0)
    {
        bilateral(mpRenderTempColor, mpPostFilterTempColor);
        fxaa(mpPostFilterTempColor, mpPostFilterOutputColor);
    }
    else if (placement == 1) // Diagnostic: previous implementation.
    {
        suppress(mpRenderTempColor, mpPostFilterOutputColor);
        bilateral(mpPostFilterOutputColor, mpPostFilterTempColor);
        fxaa(mpPostFilterTempColor, mpPostFilterOutputColor);
    }
    else if (placement == 2) // Diagnostic: before FXAA.
    {
        bilateral(mpRenderTempColor, mpPostFilterTempColor);
        suppress(mpPostFilterTempColor, mpPostFilterOutputColor);
        fxaa(mpPostFilterOutputColor, mpPostFilterTempColor);
        pRenderContext->copyResource(mpPostFilterOutputColor.get(), mpPostFilterTempColor.get());
    }
    else
    {
        // Default: judge the actual post-filtered image, including FXAA.
        bilateral(mpRenderTempColor, mpPostFilterOutputColor);
        fxaa(mpPostFilterOutputColor, mpPostFilterTempColor);
        suppress(mpPostFilterTempColor, mpPostFilterOutputColor);
    }
}

void NeuralRadiosity::reapplyDiagnosticPostFilter(bool isolatedHighlight)
{
    FALCOR_CHECK(mTrainingPaused, "Post-filter comparisons require a frozen network.");
    FALCOR_CHECK(mEnablePostFilter && mpPostFilterOutputColor && mpIsolatedHighlightPass,
        "Render a frame with post filtering enabled before replaying it.");
    dispatchPostFilters(mpDevice->getRenderContext(), isolatedHighlight);
}

void NeuralRadiosity::reapplyDiagnosticPostFilterPlacement(uint32_t placement, float minExcess, bool requireQuietRing)
{
    FALCOR_CHECK(placement <= 3, "Invalid highlight filter placement.");
    FALCOR_CHECK(std::isfinite(minExcess) && minExcess > 0.f, "Invalid highlight excess threshold.");
    FALCOR_CHECK(mTrainingPaused && mEnablePostFilter && mpPostFilterOutputColor && mpIsolatedHighlightPass,
        "Render a post-filtered frame and pause training before replaying it.");
    dispatchPostFilters(mpDevice->getRenderContext(), placement != 0, placement, minExcess, requireQuietRing);
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
    sampleRHSForBatch(pRenderContext, mpTrainLHSBatch, mpTrainRHSBatch, mFrameCount);
}

void NeuralRadiosity::sampleRHSForBatch(RenderContext* pRenderContext, const std::shared_ptr<RayBatchBuffer>& lhs,
    const std::shared_ptr<RayBatchBuffer>& rhs, uint32_t sampleFrame)
{
    FALCOR_PROFILE(pRenderContext, "SampleRHS");

    ShaderVar var = mpSampleRHS->getRootVar();
    const auto name = "gSampleRHS";

    // Bind parameters
    var[name]["maxGlossyBounces"] = mRhsGlossyBounces;
    var[name]["queryGlossyFallback"] = (mDiffTrainOnGlossy || mRhsGlossyBounces == 0) ? 1u : 0u;
    var[name]["batchSize"] = lhs->size;
    var[name]["batchCount"] = sampleFrame;

    // LHS buffers
    {
        var["lhsPos"] = lhs->pos;
        var["lhsDir"] = lhs->dir;
        var["lhsVBuffer"] = lhs->vbuffer;
        var["lhsDiffActive"] = lhs->diffActive;
        var["lhsSpecActive"] = lhs->specActive;
    }

    // RHS buffers
    {
        var["rhsPos"] = rhs->pos;
        var["rhsDir"] = rhs->dir;
        var["rhsNormal"] = rhs->normal;
        var["rhsAlbedo"] = rhs->albedo;
        var["rhsRoughness"] = rhs->roughness;
        var["rhsVBuffer"] = rhs->vbuffer;
        var["rhsColor"] = rhs->color;
        var["rhsEmission"] = rhs->emission;

        var["rhsDiffActive"] = rhs->diffActive;
        var["rhsDiffIndex"] = rhs->diffIndex;
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

    executePass(pRenderContext, mpSampleRHS, "SampleRHS", uint3(lhs->size * mNumRHS, 1, 1));
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

void NeuralRadiosity::coneTraceBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch, const RenderData& renderData, uint32_t sampleFrame)
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
        var[name]["batchCount"] = sampleFrame == UINT32_MAX ? mFrameCount : sampleFrame;
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
    bakeHost("readcounts render");

    // Ping-pong: the reproject pass reads [1 - write] and seeds [write], the cone
    // trace updates [write] in place, and the spatial pass only reads it.
    if (!mDiagnosticHistoryFrozen)
    {
        mClusterHistoryWriteIdx ^= 1u;

        clusterReprojectPass(pRenderContext, renderData);
        pRenderContext->uavBarrier(mpClusterHistory[mClusterHistoryWriteIdx].get());
        pRenderContext->uavBarrier(mpClusterLockState[mClusterHistoryWriteIdx].get());

        coneTraceScreenPass(pRenderContext, renderData);
        pRenderContext->uavBarrier(mpClusterHistory[mClusterHistoryWriteIdx].get());
        pRenderContext->uavBarrier(mpClusterLockState[mClusterHistoryWriteIdx].get());
    }

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

    if (!mpFarValidationStats)
        mpFarValidationStats = mpDevice->createStructuredBuffer(sizeof(uint32_t), 10,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    pRenderContext->clearUAV(mpFarValidationStats->getUAV().get(), uint4(0));

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
    var[name]["enableVisibilityCheck"] = mEnableClusterVisibilityCheck ? 1u : 0u;

    var[name]["visibilityFrame"] = mClusterVisibilityFrame;
    var[name]["visibilityFrameAdvanced"] = mClusterVisibilityFrameAdvanced ? 1u : 0u;
    var[name]["enableRayBudget"] = mEnableClusterRayBudget ? 1u : 0u;
    var[name]["enableFarModeValidation"] = mEnableFarModeValidation ? 1u : 0u;
    var["farValidationStats"] = mpFarValidationStats;
    var["specSrc"] = mpRenderBatch->specSrc;

    var["gVBuffer"] = renderData.getTexture(kOutputVBuffer);
    var["gPos"] = renderData.getTexture(kOutputPos);
    var["gDir"] = renderData.getTexture(kOutputDir);
    var["gNormal"] = renderData.getTexture(kOutputNormal);
    var["gRoughness"] = renderData.getTexture(kOutputRoughness);
    var["clusterHistory"] = mpClusterHistory[mClusterHistoryWriteIdx];
    var["clusterLockState"] = mpClusterLockState[mClusterHistoryWriteIdx];

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    if (mpRenderBatch->specSize == 0)
    {
        bakeSkippedPass(mpConeTraceScreen, "ConeTraceScreen");
        return;
    }

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
    var[name]["diagnosticLegacyReuse"] = mDiagnosticSpatialReuseLegacy ? 1u : 0u;
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

    if (mpRenderBatch->specSize == 0)
    {
        bakeSkippedPass(mpClusterSpatialReuse, "ClusterSpatialReuse");
        return;
    }

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
    bakeHost("saveprev");
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
    const auto diffPtrs = pRayBatch->getDiffPtrs();
    const auto specPtrs = pRayBatch->getSpecPtrs();
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

    if (!mpIsolatedHighlightPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kPostFilterPassFile).csEntry("mainIsolatedHighlight");
        mpIsolatedHighlightPass = ComputePass::create(mpDevice, desc, defines, true);
        ShaderVar var = mpIsolatedHighlightPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
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
        mpNRModel = std::make_unique<NRModel>(mModelConfig);
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
    defines.add("NORMALIZE_CLS_SCALE", mNormalizeClsScale ? "1" : "0");
    defines.add("NUM_KMEANS_ITERS", std::to_string(mNumKMeansIters));
    defines.add("NUM_SPEC_RAYS", std::to_string(mNumSpecRays));
    defines.add("NUM_CLUSTERS", std::to_string(mNumClusters));
    defines.add("N_LEVELS", std::to_string(mModelConfig.nLevels));
    defines.add("N_FEATURES_PER_LEVEL", std::to_string(mModelConfig.nFeaturesPerLevel));
    defines.add("N_LEVELS_SPEC", std::to_string(mModelConfig.nLevelsSpec));
    defines.add("N_FEATURES_PER_LEVEL_SPEC", std::to_string(mModelConfig.nFeaturesPerLevelSpec));
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

    const uint32_t numSpecRays = useTrainConfig ? mNumSpecRaysTrain : mNumSpecRaysRender;
    const uint32_t numKMeansIters = useTrainConfig ? mNumKMeansItersTrain : mNumKMeansItersRender;

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


void NeuralRadiosity::configureDiagnostics(uint32_t interval, uint32_t x, uint32_t y)
{
    mDiagnosticInterval = interval;
    mDiagnosticPixel = uint2(x, y);
    mpDiagnosticLHS.reset();
    mpDiagnosticRHS.reset();
    mLastDiagnostics = "{}";
}

std::string NeuralRadiosity::getFarModeValidationStats()
{
    uint32_t counts[10] = {};
    if (mpFarValidationStats) mpDevice->getRenderContext()->readBuffer(mpFarValidationStats.get(), counts, 0, sizeof(counts));
    const uint32_t n = mpRenderBatch ? mpRenderBatch->specSize : 0u;
    return nlohmann::json({{"validated_pixels", counts[0]}, {"extra_samples", 32ull * counts[0]},
        {"specular_pixels", n}, {"budget_enabled", mEnableClusterRayBudget},
        {"regular_reserved", counts[1]}, {"validation_reserved", counts[2]},
        {"traced_rays", counts[3]}, {"ray_limit", uint64_t(n) + n / 2u},
        {"regular_denied", counts[4]}, {"validation_denied", counts[5]},
        {"visibility_resets", counts[6]}, {"deferred_resets", counts[7]}}).dump();
}

void NeuralRadiosity::setDiagnosticRenderOptions(bool postFilter, bool temporalReuse, bool mirrorAlbedo, uint32_t rays, uint32_t kMeansIters)
{
    FALCOR_CHECK(mTrainingPaused, "Render ablations require a frozen network.");
    FALCOR_CHECK(!mDiagnosticHistoryFrozen, "Release the history pin before changing render-path settings.");
    FALCOR_CHECK(rays >= NCR_NUM_CLUSTERS && rays <= 1024 && kMeansIters > 0, "Invalid diagnostic cone sampling parameters.");
    if (mEnableRenderSpecTemporalReuse != temporalReuse) clearHistory();
    mEnablePostFilter = postFilter;
    mEnableRenderSpecTemporalReuse = temporalReuse;
    mEnableMirrorAlbedo = mirrorAlbedo;
    mUseReferenceConeTrace = false;
    mNumSpecRaysRender = rays;
    mNumKMeansItersRender = kMeansIters;
    setConeParameters(false);
    mVarsChanged = true;
}

std::string NeuralRadiosity::inspectRenderPixel(uint32_t x, uint32_t y)
{
    using Json = nlohmann::json;
    FALCOR_CHECK(mpRenderBatch && mpNRModel && x < mFrameDim.x && y < mFrameDim.y, "No rendered pixel at the requested coordinates.");
    auto* ctx = mpDevice->getRenderContext();
    const size_t pixel = size_t(y) * mFrameDim.x + x;
    uint32_t active = 0, index = 0;
    ctx->readBuffer(mpRenderBatch->specActive.get(), &active, pixel * sizeof(uint32_t), sizeof(uint32_t));
    Json result = {{"pixel", {x, y}}, {"frame", mFrameCount}, {"training_batches", mTrainingBatchCount}, {"spec_active", active}};
    if (!active) return result.dump();
    ctx->readBuffer(mpRenderBatch->specIndex.get(), &index, pixel * sizeof(uint32_t), sizeof(uint32_t));
    FALCOR_CHECK(index < mpRenderBatch->specSize, "Invalid compacted specular index in diagnostic readback.");
    const uint32_t stride = mpRenderBatch->specInput->getStructSize();
    std::vector<uint8_t> input(stride);
    ctx->readBuffer(mpRenderBatch->specInput.get(), input.data(), size_t(index) * stride, stride);
    result["input_bytes"] = input;
    result["input_fp16"] = bool(NCR_USE_FP16);
    result["input_stride"] = stride;
    result["index"] = index;
    result["cone_rays"] = mNumSpecRays;
    result["cone_kmeans_iters"] = mNumKMeansIters;
    result["primary_encoding_dim"] = mModelConfig.nFeaturesPerLevel * mModelConfig.nLevels;
    result["cluster_encoding_dim"] = mModelConfig.nFeaturesPerLevelSpec * mNumClusters;
    result["cluster_count"] = mNumClusters;
    const auto bounds = mpScene->getSceneBounds();
    result["bounds_min"] = {bounds.minPoint.x, bounds.minPoint.y, bounds.minPoint.z};
    result["bounds_max"] = {bounds.maxPoint.x, bounds.maxPoint.y, bounds.maxPoint.z};
    std::vector<float> prediction(mpRenderBatch->specColor->getStructSize() / sizeof(float));
    ctx->readBuffer(mpRenderBatch->specColor.get(), prediction.data(), size_t(index) * mpRenderBatch->specColor->getStructSize(), prediction.size() * sizeof(float));
    result["prediction"] = prediction;
    if (useClusterReuse() && mClusterHistoryValid)
    {
        const auto& history = mpClusterHistory[mClusterHistoryWriteIdx];
        std::vector<uint32_t> packed(history->getStructSize() / sizeof(uint32_t));
        ctx->readBuffer(history.get(), packed.data(), pixel * history->getStructSize(), history->getStructSize());
        uint32_t lock = 0;
        ctx->readBuffer(mpClusterLockState[mClusterHistoryWriteIdx].get(), &lock, pixel * sizeof(uint32_t), sizeof(uint32_t));
        result["packed_cluster_history"] = packed;
        result["packed_lock_state"] = lock;
    }
    return result.dump();
}

void NeuralRadiosity::runDiagnostics(RenderContext* ctx, const RenderData& renderData)
{
    using Json = nlohmann::json;
    constexpr uint32_t kProbeCount = 256;
    constexpr uint32_t kProbeSeed = 104729;
    if (!mpDiagnosticLHS) {
        if (mDiagnosticPixel.x >= mFrameDim.x || mDiagnosticPixel.y >= mFrameDim.y)
            throw std::runtime_error("Diagnostic pixel is outside the image");
        const size_t pixel = size_t(mDiagnosticPixel.y)*mFrameDim.x+mDiagnosticPixel.x;
        uint32_t active = 0;
        ctx->readBuffer(mpRenderBatch->specActive.get(), &active, pixel*sizeof(uint32_t), sizeof(uint32_t));
        if (!active) throw std::runtime_error("Diagnostic pixel is not a valid glossy surface");
        ShaderVar reflection = mpRandomSmooth->getRootVar();
        mpDiagnosticLHS = std::make_shared<RayBatchBuffer>(mpDevice, reflection, kProbeCount, mNumClusters, "DiagnosticLHS", true, true, true);
        mpDiagnosticRHS = std::make_shared<RayBatchBuffer>(mpDevice, reflection, kProbeCount*mNumRHS, mNumClusters, "DiagnosticRHS", true, true, false);
        Json metadata = {{"pixel",{mDiagnosticPixel.x,mDiagnosticPixel.y}}, {"copies",kProbeCount}, {"rhs_samples",kProbeCount*mNumRHS}, {"fixed_rng_seed",kProbeSeed}};
        auto copyPixel = [&](const std::string& channel, const ref<Buffer>& destination, size_t elementSize, bool floats) {
            const auto bytes = ctx->readTextureSubresource(renderData.getTexture(channel).get(), 0);
            const uint8_t* source = bytes.data()+pixel*elementSize;
            std::vector<uint8_t> repeated(kProbeCount*elementSize);
            for (size_t i=0;i<kProbeCount;++i) std::memcpy(repeated.data()+i*elementSize,source,elementSize);
            ctx->updateBuffer(destination.get(),repeated.data(),0,repeated.size());
            if (floats) {
                std::vector<float> value(elementSize/sizeof(float));
                std::memcpy(value.data(),source,elementSize);
                metadata[channel] = value;
            }
        };
        copyPixel(kOutputPos,mpDiagnosticLHS->pos,16,true);
        copyPixel(kOutputDir,mpDiagnosticLHS->dir,16,true);
        copyPixel(kOutputNormal,mpDiagnosticLHS->normal,16,true);
        copyPixel(kOutputAlbedo,mpDiagnosticLHS->albedo,16,true);
        copyPixel(kOutputRoughness,mpDiagnosticLHS->roughness,4,true);
        copyPixel(kOutputVBuffer,mpDiagnosticLHS->vbuffer,16,false);
        ctx->clearUAV(mpDiagnosticLHS->diffActive->getUAV().get(),uint4(0));
        ctx->clearUAV(mpDiagnosticLHS->specActive->getUAV().get(),uint4(1));
        ctx->clearUAV(mpDiagnosticLHS->diffInput->getUAV().get(),uint4(0));
        ctx->clearUAV(mpDiagnosticLHS->specInput->getUAV().get(),uint4(0));
        mProbeMetadata = metadata.dump();
    }
    // PrefixSum consumes the index buffer; restore it from the frozen active mask.
    ctx->copyResource(mpDiagnosticLHS->diffIndex.get(),mpDiagnosticLHS->diffActive.get());
    ctx->copyResource(mpDiagnosticLHS->specIndex.get(),mpDiagnosticLHS->specActive.get());
    sampleRHSForBatch(ctx, mpDiagnosticLHS, mpDiagnosticRHS, kProbeSeed);
    // Preserve the raw throughputs / finished radiance and direct term before resolve.
    auto rawColor = ctx->readBuffer<float4>(mpDiagnosticRHS->color.get());
    auto direct = ctx->readBuffer<float4>(mpDiagnosticRHS->emission.get());
    auto active = ctx->readBuffer<uint32_t>(mpDiagnosticRHS->diffActive.get());
    compactBatch(ctx,mpDiagnosticRHS);
    modelInferenceCUDA(ctx,mpDiagnosticRHS);
    resolveBatch(ctx,mpDiagnosticRHS);
    auto resolved = ctx->readBuffer<float4>(mpDiagnosticRHS->color.get());
    double directLum=0, terminalLum=0, bootstrapLum=0, sum=0, square=0;
    auto luminance=[](float4 v) { return .2126*double(v.x)+.7152*double(v.y)+.0722*double(v.z); };
    size_t nonfinite=0, bootstrapCount=0;
    for(size_t i=0;i<resolved.size();++i) {
        const double d=luminance(direct[i]), r=luminance(resolved[i]);
        if (!std::isfinite(d) || !std::isfinite(r)) { ++nonfinite; continue; }
        directLum+=d; sum+=r; square+=r*r;
        if(active[i]) { bootstrapLum+=r-d; ++bootstrapCount; }
        else terminalLum+=luminance(rawColor[i]);
    }
    const double n=double(resolved.size()), mean=sum/n;
    compactBatch(ctx,mpDiagnosticLHS);
    coneTraceBatch(ctx,mpDiagnosticLHS,renderData,kProbeSeed);
    ctx->waitForFalcor(mpNRModel->stream(false));
    readBatchCountsAfterSync(ctx,mpDiagnosticLHS);
    if (mpDiagnosticLHS->specSize != kProbeCount) throw std::runtime_error("Diagnostic probe count changed");
    const auto predictions=Json::parse(mpNRModel->probeDiagnostics(mpDiagnosticLHS->getSpecPtrs()));
    ctx->waitForCuda(mpNRModel->stream(false));
    Json result = {{"frame",mFrameCount+1},{"training_batches",mTrainingBatchCount}, {"heat",mTrainHeat},
        {"training_paused",mTrainingPaused},
        {"probe",Json::parse(mProbeMetadata)}, {"prediction",predictions},
        {"rhs",{{"mean",mean},{"std",std::sqrt(std::max(0.0,square/n-mean*mean))},
                {"direct",directLum/n},{"terminal",terminalLum/n},{"bootstrap",bootstrapLum/n},
                {"bootstrap_count",bootstrapCount},{"sample_count",resolved.size()},{"nonfinite",nonfinite}}}};
    if (!mTrainingPaused) {
        result["training"] = Json::parse(mpNRModel->trainingDiagnostics());
        result["training"]["sampled_spec_count"] = mpTrainLHSBatch->specSize;
        const auto positions = ctx->readBuffer<float4>(mpTrainLHSBatch->pos.get(),0,mBatchSize);
        const auto directions = ctx->readBuffer<float4>(mpTrainLHSBatch->dir.get(),0,mBatchSize);
        const auto normals = ctx->readBuffer<float4>(mpTrainLHSBatch->normal.get(),0,mBatchSize);
        const auto valid = ctx->readBuffer<uint32_t>(mpTrainLHSBatch->specActive.get(),0,mBatchSize);
        const Json metadata=Json::parse(mProbeMetadata);
        const float3 p(metadata["pos"][0].get<float>(),metadata["pos"][1].get<float>(),metadata["pos"][2].get<float>());
        const float3 d=normalize(float3(metadata["dir"][0].get<float>(),metadata["dir"][1].get<float>(),metadata["dir"][2].get<float>()));
        uint32_t near=0, aligned=0, invalidNormal=0, invalidDirection=0;
        Json firstInvalid=nullptr;
        Json invalidSamples=Json::array();
        const auto hits = ctx->readBuffer<uint4>(mpTrainLHSBatch->vbuffer.get(),0,mBatchSize);
        const auto hitDefines = mpScene->getHitInfo().getDefines();
        const uint32_t instanceBits = std::stoul(hitDefines.at("HIT_INFO_INSTANCE_ID_BITS"));
        const uint32_t primitiveBits = std::stoul(hitDefines.at("HIT_INFO_PRIMITIVE_INDEX_BITS"));
        const uint32_t typeBits = std::stoul(hitDefines.at("HIT_INFO_TYPE_BITS"));
        const bool shortHeader = instanceBits + primitiveBits + typeBits <= 32;
        const auto mask = [](uint32_t bits) { return uint32_t((uint64_t(1) << bits) - 1); };
        const auto viewProj = mpScene->getCamera()->getViewProjMatrixNoJitter();
        auto finite3=[](float4 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); };
        for(uint32_t i=0;i<mBatchSize;++i) {
            if(!valid[i]) continue;
            const bool badN=!finite3(normals[i]), badD=!finite3(directions[i]);
            invalidNormal+=badN; invalidDirection+=badD;
            if ((badN || badD) && invalidSamples.size() < 256)
            {
                const auto hit = hits[i];
                const uint32_t instanceID = (shortHeader ? uint32_t(uint64_t(hit.x) >> primitiveBits) : hit.x) & mask(instanceBits);
                const uint32_t primitiveID = shortHeader ? hit.x & mask(primitiveBits) : hit.y;
                const float4 projected = mul(viewProj, float4(positions[i].x,positions[i].y,positions[i].z,1.f));
                Json sample = {{"position",{positions[i].x,positions[i].y,positions[i].z}},
                    {"raw_index",i},{"instance_id",instanceID},{"primitive_id",primitiveID},
                    {"invalid_normal",badN},{"invalid_direction",badD}};
                if (std::isfinite(projected.w) && projected.w > 0.f)
                    sample["pixel"] = {(projected.x/projected.w*.5f+.5f)*mFrameDim.x,
                        (.5f-projected.y/projected.w*.5f)*mFrameDim.y};
                if (instanceID < mpScene->getGeometryInstanceCount())
                {
                    const auto& instance = mpScene->getGeometryInstance(instanceID);
                    sample["geometry_id"] = instance.geometryID;
                    sample["material"] = mpScene->getMaterial(MaterialID{instance.materialID})->getName();
                    if (instance.getType() == GeometryType::TriangleMesh && mpScene->hasMesh(instance.geometryID))
                        sample["mesh"] = mpScene->getMeshName(instance.geometryID);
                }
                invalidSamples.push_back(sample);
            }
            if((badN || badD) && firstInvalid.is_null()) {
                uint32_t compact=0;
                uint4 hit;
                ctx->readBuffer(mpTrainLHSBatch->specIndex.get(),&compact,size_t(i)*sizeof(uint32_t),sizeof(uint32_t));
                ctx->readBuffer(mpTrainLHSBatch->vbuffer.get(),&hit,size_t(i)*sizeof(uint4),sizeof(uint4));
                firstInvalid={{"raw_index",i},{"compact_index",compact},
                    {"position",{positions[i].x,positions[i].y,positions[i].z}},
                    {"direction",{directions[i].x,directions[i].y,directions[i].z}},
                    {"normal",{normals[i].x,normals[i].y,normals[i].z}},
                    {"packed_hit",{hit.x,hit.y,hit.z,hit.w}}};
            }
        }
        result["training"]["source_attributes"]={{"invalid_normal_rows",invalidNormal},{"invalid_direction_rows",invalidDirection},{"first_invalid",firstInvalid},{"invalid_samples",invalidSamples}};
        for(uint32_t i=0;i<mBatchSize;++i) {
            if(!valid[i] || length(float3(positions[i].x,positions[i].y,positions[i].z)-p)>.25f) continue;
            ++near;
            aligned += dot(normalize(float3(directions[i].x,directions[i].y,directions[i].z)),d)>=.95f;
        }
        result["training"]["near_probe"]={{"radius",.25},{"direction_cos",.95},{"position_count",near},{"position_and_direction_count",aligned}};
    }
    mLastDiagnostics = result.dump();
}

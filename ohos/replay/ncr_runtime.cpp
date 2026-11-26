// The NCR runtime (ncr_runtime.h). The frame is the bake's program; the network steps are done here:
//
// Inference of a batch, --networks tiles or npu:
//   GPU   EncodeInputs: grid lookups (EMA grids) into channel-planar tiles of 65,536 samples
//   NPU   the inference graph, tile by tile (CPU stand-in: CpuTileBackend)
//   GPU   DecodeOutputs: the outputs back as the rows the passes read
// A training step (a frame baked in a training mode has one):
//   GPU   EncodeInputs with the training grids, EncodeTargets          -> input and target tiles
//   NPU   the training graph, tile by tile                             -> dL/dw, dL/d(input)
//   GPU   ScatterGradients into the grids, GridOptimizer (Adam + EMA)  -> the grids' new weights
//   CPU   Adam + EMA for the two MLPs                                  -> the MLPs' new weights
// The prefix sums between the passes are a compute pass as well (shaders/PrefixSum).
//
// The cluster-reuse render path (NeuralRadiosity::clusterReuse) adds two host steps to the program:
//   readcounts  the host waits for the GPU and reads the specular count the next passes are sized by
//   saveprev    this frame's G-buffer and specular mask become the next frame's "previous" copies
//   clearstats  the ray budget's counters are cleared before ConeTraceScreen
// and the calls of its three passes alternate between the two history buffers from frame to frame.
// What the pass counts on the CPU for them (ClusterClock) is counted here and set in their uniforms.
#include "ncr_runtime.h"

#include "replay_bake.h"
#include "replay_nn.h"
#include "replay_nnrt.h"
#include "replay_own.h"

#include <fstream>
#include <map>

namespace replay
{
void writeFile(const std::string& path, const void* data, size_t size)
{
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), (std::streamsize)size);
    if (!f.good()) fail("cannot write " + path);
}

void TrainSchedule::advance(uint32_t events)
{
    if (!enabled)
    {
        learningRate = lrHot;
        emaDecay = emaHot;
        return;
    }
    // NeuralRadiosity::updateTrainSchedule, line by line.
    float target = 0.0f;
    if ((events & kEventCamera) != 0)
    {
        target = cameraHeat;
        if ((events & (kEventEdit | kEventLight | kEventObject)) != 0) target = std::max(target, sceneHeat);
    }
    else
    {
        if ((events & kEventEdit) != 0) target = std::max(target, sceneHeat);
        if ((events & kEventLight) != 0) target = std::max(target, lightMotionHeat);
        if ((events & kEventObject) != 0) target = std::max(target, objectMotionHeat);
    }
    // Only a target above the heat raises it and sets the hold, so a hold counts down while the
    // same events repeat.
    if (target > heat)
    {
        heat = target;
        holdLeft = std::max(holdLeft, holdFrames);
    }
    else if (holdLeft > 0)
    {
        holdLeft--;
    }
    else
    {
        const float k = std::exp(-1.0f / std::max(coolFrames, 1.0f));
        heat = target + (heat - target) * k;
    }
    const float h = std::clamp(heat, 0.0f, 1.0f);
    learningRate = lrCold * std::pow(lrHot / lrCold, h);
    const float oneMinusDecay = (1.0f - emaCold) * std::pow((1.0f - emaHot) / (1.0f - emaCold), h);
    emaDecay = 1.0f - oneMinusDecay;
}

void TrainSchedule::reset()
{
    heat = 1.0f;
    holdLeft = initialHotFrames;
}

void ClusterClock::advance(bool geometryMoved, bool viewChanged, bool historyValid)
{
    // A sudden geometry edit is checked with the previous direction before its reference is replaced;
    // continuous motion still advances the windows.
    visibilityFrameAdvanced = !geometryMoved || geometryMovedPrevious;
    if (visibilityFrameAdvanced) ++visibilityFrame;
    if (viewChanged) motionFarWeightFrames = motionFarWeightFadeFrames + 1;
    else if (motionFarWeightFrames) --motionFarWeightFrames;
    if (!historyValid || viewChanged) frequencyFrames = 0;
    else if (geometryMoved) frequencyFrames = 200;
    else if (frequencyFrames) --frequencyFrames;
    geometryMovedPrevious = geometryMoved;
}

uint32_t ClusterClock::frequencySurveyFrame() const
{
    return checkFrequency && frequencyFrames >= 188 && frequencyFrames < 200 ? 200u - frequencyFrames : 0u;
}

float ClusterClock::motionFarWeightStrength() const
{
    return limitMotionFarWeight ? (float)std::min(motionFarWeightFrames, motionFarWeightFadeFrames) / (float)motionFarWeightFadeFrames : 0.0f;
}

namespace
{
/// Exclusive scan in place (Falcor's PrefixSum); returns the total.
uint32_t prefixSum(uint32_t* data, uint32_t count)
{
    // The buffer is memory shared with the GPU, slow to walk element by element: work on a copy.
    std::vector<uint32_t> local(data, data + count);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; i++)
    {
        const uint32_t value = local[i];
        local[i] = sum;
        sum += value;
    }
    std::memcpy(data, local.data(), (size_t)count * sizeof(uint32_t));
    return sum;
}

/// The weight of every sample of a training batch in the target tiles. The training graph divides the
/// loss by the samples of a tile (the caller rescales to the batch), so a batch much smaller than a
/// tile would get gradients smaller than tiny-cuda-nn's by up to the tile size, down where the NPU's
/// fp16 loses them. Such a batch's samples weigh the largest power of two up to tile / samples
/// instead (exact in fp16), which brings the graph's gradients to within a factor of two of
/// tiny-cuda-nn's; 1 for a batch of half a tile or more.
float sampleWeight(uint32_t samples, uint32_t tile)
{
    float weight = 1.0f;
    while (samples > 0 && 2.0 * weight * samples <= tile) weight *= 2.0f;
    return weight;
}

uint32_t log2Exact(uint32_t value)
{
    uint32_t shift = 0;
    while ((1u << shift) < value) shift++;
    if ((1u << shift) != value) fail(format("%u is not a power of two", value));
    return shift;
}

bool fileExists(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return (bool)f;
}

/// NeuralRadiosity.cpp: how many specular samples ConeTrace and SpecPrimary are dispatched for when
/// this frame's count is still on the GPU. The previous frame's count plus headroom.
uint32_t specDispatchItems(uint32_t previousSpecCount, uint32_t batchSize)
{
    return std::min(batchSize, std::max(8192u, previousSpecCount + previousSpecCount / 8));
}

// Push constants of the runtime's own passes (ohos/replay/shaders).
struct EncodePush { uint32_t count, rowBase, tileShift; };
struct DecodePush { uint32_t count, rowBase; };
struct TargetPush { uint32_t count, rowBase, tileShift, slots; float weight; };
struct ScatterPush { uint32_t count, tileShift; float gradScale; };
struct OptimizerPush { uint32_t count; float learningRate, gradToFloat, emaDecay, emaDebiasOld, emaDebiasNew; };
struct PrefixPush { uint32_t count, stage, countIndex; };
struct DisplayPush { uint32_t width, height, bgra; };
constexpr uint32_t kPrefixBlock = 256;   // elements per block in shaders/PrefixSum

/// A ray batch (render, lhs, rhs) as the host sees it.
struct Batch
{
    std::string role;
    uint32_t size = 0;
    bool needDiff = false, needSpec = false, sharedIO = false;
    // The compacted counts as the host last read them (Falcor: at its sync with CUDA before an
    // inference or a training step). Until then they are the previous frame's.
    uint32_t diffSize = 0, specSize = 0;

    /// With shared I/O the specular rows come first and the diffuse rows follow the padded specular
    /// rows (RayBatchBuffer::diffOutputOffset / diffInputOffset), both in rows of their own kind.
    uint32_t diffOutputOffset() const { return sharedIO ? padUp(specSize, cfg::kBatchAlign) : 0; }
    uint32_t diffInputOffset() const
    {
        if (!sharedIO) return 0;
        const uint64_t specBytes = (uint64_t)diffOutputOffset() * cfg::kSpecInputDim * 2;
        const uint32_t diffRowBytes = cfg::kDiffInputDim * 2;
        return (uint32_t)((specBytes + diffRowBytes - 1) / diffRowBytes);
    }
};

/// What one batch needs to send its samples through the networks as tiles.
struct TilePath
{
    Buffer specX, diffX, specY, diffY;
    OwnPass encodeSpec, encodeDiff, decodeSpec, decodeDiff;
};

/// A hash grid on the GPU. Inference reads `inference` (the fp16 EMA); the rest exists when the
/// replayed frame trains.
struct Grid
{
    uint32_t elements = 0;
    uint32_t steps = 0;   // optimizer steps taken
    Buffer inference, training, master, firstMoment, secondMoment, entrySteps, gradient;
    OwnPass optimizer;
};

/// What one batch needs to train one network on it.
struct TrainNet
{
    Buffer x, target, gradient;          // tiles: inputs, targets, dL/d(input)
    OwnPass encode, targets, scatter;
    std::vector<float> weightGradient;   // dL/dw of the batch
};
struct TrainPath { TrainNet diff, spec; };

/// The prefix sum of one mask of one batch on the GPU.
struct PrefixPath
{
    Buffer blockSums;
    OwnPass pass;
};
}  // namespace

struct Runtime::State
{
    Device& dev;
    RuntimeOptions options;
    Bake bake;
    Checkpoint checkpoint;
    Mlp diffNet, specNet;
    CpuNetworks cpuNetworks;
    std::unique_ptr<NetworkBackend> backend;
    std::map<std::string, Batch> batches;
    Grid primGrid, clsGrid;
    std::map<std::string, TilePath> tilePaths;
    std::map<std::string, TrainPath> trainPaths;
    std::map<std::string, PrefixPath> prefixPaths;
    VkDescriptorPool ownPool = VK_NULL_HANDLE;
    bool useTiles = true, training = false, sharedTiles = false, useSchedule = false;
    uint32_t trainShift = 0, inferShift = 0;
    TrainSchedule schedule;
    uint32_t frame = 0, trainSteps = 0;
    uint32_t outputTexture = 0;

    // What can change between frames (ncr_runtime.h).
    Settings settings;
    bool hasPostFilter = false, hasLights = false, hasRenderPath = false;
    size_t trainSectionEnd = 0;       // the program's steps before this index are the training part of the frame
    double animationTime = 0.0;       // within one loop of the track
    uint32_t sample = 0;              // the track sample the scene is at
    bool reapplySample = false;       // the track must write the scene again (the camera returns to its path)
    float bounds[6] = {};             // the scene bounds of that sample
    CameraPose setPose;               // the caller's camera, used while the camera does not follow the track
    bool poseSet = false;
    CameraPose appliedPose;           // the pose the scene's block holds while the caller's camera is in use
    bool poseApplied = false;
    uint32_t pendingEvents = 0;       // seen since the training schedule last advanced
    uint32_t lastEvents = 0;

    // The cluster-reuse render path's state across frames (NeuralRadiosity::clusterReuse).
    bool clusterReuse = false;         // the bake's render path is that one
    uint32_t clusterRuns = 0;          // frames its passes ran in
    bool clusterAlternate = false;     // this frame's calls bind the history pairs swapped against the baked frame
    bool clusterHistoryValid = false;  // a frame has written the history since it was last cleared (mClusterHistoryValid)
    bool clearClusterHistory = false;  // Runtime::clearHistory: before the next frame
    Bake::CameraHistory lastCamera;    // the camera of the last frame: the next frame's "previous" matrices
    bool haveLastCamera = false;
    ClusterClock clusterClock;         // what the pass counts on the CPU for its passes
    bool geometryMoved = false;        // the animation moved geometry for this frame (IScene::UpdateFlags::GeometryMoved)
    bool objectsMovedPrevious = false; // the track moved objects in the last frame
    bool hasRecovery = false, hasStats = false;   // the bake has the recovery countdown pair / the ray budget's counters

    // Calls are recorded into one command buffer until a host step needs their results.
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::string pendingNames;
    std::vector<std::pair<std::string, double>> timings;   // of the frame being replayed
    double lastFrameMs = 0.0;

    static constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    State(Device& device, const std::string& bakeDir, const std::string& model, const RuntimeOptions& opts)
        : dev(device), options(opts), bake(device, bakeDir), cpuNetworks(checkpoint, diffNet, specNet)
    {
        const auto start = std::chrono::steady_clock::now();
        bake.load();
        std::printf("bake %ux%u (Falcor frame %u) loaded in %.0f ms\n", bake.width, bake.height, bake.frameCount, msSince(start));
        outputTexture = bake.textureBoundAs("gColor");

        // The model: the bake's own state, or a checkpoint loaded the way Falcor loads one.
        if (model == "bake")
        {
            checkpoint.load(bakeDir + "/before.ckpt");
            const std::string working = bakeDir + "/before_working.bin";
            if (fileExists(working))
            {
                checkpoint.loadWorking(working);
                std::printf("model: the bake's (before.ckpt, training copies from before_working.bin)\n");
            }
            else
            {
                std::printf("model: the bake's before.ckpt (the bake has no before_working.bin: training copies rebuilt from the masters)\n");
            }
        }
        else
        {
            checkpoint.load(model);
            // The NPU graphs and the bake belong to one network size: a checkpoint of another does not fit them.
            if (fileExists(bakeDir + "/before.ckpt"))
            {
                Checkpoint baked;
                baked.load(bakeDir + "/before.ckpt");
                if (std::memcmp(baked.counts, checkpoint.counts, sizeof(baked.counts)) != 0)
                    fail(format("%s has another network size than the bake (SpecNet %u hidden layers, the bake %u; DiffNet %u, the bake %u)",
                                model.c_str(), mlpHiddenLayers(cfg::kSpecInputDim, checkpoint.counts[3]), mlpHiddenLayers(cfg::kSpecInputDim, baked.counts[3]),
                                mlpHiddenLayers(cfg::kDiffInputDim, checkpoint.counts[2]), mlpHiddenLayers(cfg::kDiffInputDim, baked.counts[2])));
            }
            std::printf("model: %s (training copies rebuilt from the masters, as Falcor loads a checkpoint)\n", model.c_str());
        }
        diffNet.init(false, checkpoint.diffMaster, checkpoint.diffEma, checkpoint.diffWorking);
        specNet.init(true, checkpoint.specMaster, checkpoint.specEma, checkpoint.specWorking);
        std::printf("MLPs: DiffNet %u hidden layers (%zu weights), SpecNet %u hidden layers (%zu weights)\n", diffNet.hiddenLayers,
                    diffNet.master.size(), specNet.hiddenLayers, specNet.master.size());

        for (const char* role : {"render", "lhs", "rhs"})
        {
            const std::string prefix = std::string(role) + ".";
            if (bake.values.count(prefix + "size") == 0) continue;
            Batch b;
            b.role = role;
            b.size = (uint32_t)bake.value(prefix + "size");
            b.needDiff = bake.value(prefix + "needDiff") != 0;
            b.needSpec = bake.value(prefix + "needSpec") != 0;
            b.sharedIO = bake.value(prefix + "sharedIO") != 0;
            batches[role] = b;
        }

        const std::string& networks = options.networks;
        if (networks != "tiles" && networks != "npu" && networks != "rows") fail("unknown networks mode " + networks);
        useTiles = networks != "rows";
        bool recordedTraining = false;
        for (const Step& step : bake.program) recordedTraining = recordedTraining || (!step.isCall && step.host[0] == "train");
        if (recordedTraining && !useTiles) std::printf("note: networks rows does not train; the recorded training step is skipped\n");
        training = recordedTraining && useTiles && !options.noTrain;
        trainShift = log2Exact(options.trainTile);
        inferShift = log2Exact(cfg::kTile);
        setUpSchedule();
        setUpSettings();
        clusterReuse = bake.hasBuffer("cluster.history0");
        if (clusterReuse) std::printf("render path: cluster reuse (per-pixel cluster history, carried from frame to frame)\n");
        hasRecovery = bake.hasBuffer("cluster.vcdRecovery0") && bake.hasBuffer("cluster.vcdRecovery1");
        hasStats = bake.hasBuffer("cluster.farValidationStats");
        if (clusterReuse)
        {
            ClusterClock& k = clusterClock;
            k.visibilityFrame = (uint32_t)bake.value("cluster.visibilityFrame", 0);
            k.geometryMovedPrevious = bake.value("cluster.geometryMovedPrevious", 0) != 0;
            k.frequencyFrames = (uint32_t)bake.value("cluster.frequencyFrames", 0);
            k.checkFrequency = bake.value("cluster.checkFrequency", 1) != 0;
            k.motionFarWeightFrames = (uint32_t)bake.value("cluster.motionFarWeightFrames", 0);
            k.motionFarWeightFadeFrames = std::max(1u, (uint32_t)bake.value("cluster.motionFarWeightFadeFrames", 4));
            k.limitMotionFarWeight = bake.value("cluster.limitMotionFarWeight", 1) != 0;
            objectsMovedPrevious = k.geometryMovedPrevious;
        }

        ownPool = makeOwnPool(dev, 64, 320);
        sharedTiles = networks == "npu" && !options.copyTiles && dev.sharedMemory;
        if (useTiles)
        {
            createGrid(primGrid, checkpoint.primGrid, checkpoint.primGridMaster, checkpoint.primGridWorking);
            createGrid(clsGrid, checkpoint.clsGrid, checkpoint.clsGridMaster, checkpoint.clsGridWorking);
            if (networks == "npu")
            {
#ifdef __OHOS__
                backend = std::make_unique<NnrtBackend>(diffNet, specNet, options.modelDir);
#else
                fail("networks npu needs the OpenHarmony build");
#endif
            }
            else
            {
                backend = std::make_unique<CpuTileBackend>(cpuNetworks);
            }
        }
        std::printf("networks: %s%s\n", useTiles ? backend->name() : "CPU (float32) on the sample rows, lookups included",
                    networks != "npu" ? "" : sharedTiles ? ", tiles shared with the GPU" : ", tiles copied");
        if (training)
        {
            if (useSchedule)
                std::printf("training: tiles of %u samples, Falcor's schedule (heat %.3f, hold %u; learning rate %g, EMA decay %g in the baked frame)\n",
                            options.trainTile, schedule.heat, schedule.holdLeft, schedule.learningRate, schedule.emaDecay);
            else
                std::printf("training: tiles of %u samples, learning rate %g, EMA decay %g\n", options.trainTile, schedule.learningRate,
                            schedule.emaDecay);
        }
    }

    ~State()
    {
        vkDeviceWaitIdle(dev.device);
        backend.reset();   // its tensors lie over the tile buffers
        for (auto& [role, t] : tilePaths)
        {
            for (Buffer* b : {&t.specX, &t.diffX, &t.specY, &t.diffY}) dev.destroy(*b);
            for (OwnPass* p : {&t.encodeSpec, &t.encodeDiff, &t.decodeSpec, &t.decodeDiff}) p->destroy(dev);
        }
        for (auto& [role, t] : trainPaths)
        {
            for (TrainNet* n : {&t.diff, &t.spec})
            {
                for (Buffer* b : {&n->x, &n->target, &n->gradient}) dev.destroy(*b);
                for (OwnPass* p : {&n->encode, &n->targets, &n->scatter}) p->destroy(dev);
            }
        }
        for (auto& [key, p] : prefixPaths)
        {
            dev.destroy(p.blockSums);
            p.pass.destroy(dev);
        }
        for (Grid* g : {&primGrid, &clsGrid})
        {
            for (Buffer* b : {&g->inference, &g->training, &g->master, &g->firstMoment, &g->secondMoment, &g->entrySteps, &g->gradient}) dev.destroy(*b);
            g->optimizer.destroy(dev);
        }
        if (ownPool) vkDestroyDescriptorPool(dev.device, ownPool, nullptr);
    }

    /// A member of a call's own uniform blocks (not the scene's), or null.
    void* passMember(Call& call, const std::string& member, uint32_t size)
    {
        for (auto& [block, members] : bake.passes.at(call.pass).uniforms)
        {
            if (block.rfind("gScene", 0) == 0) continue;
            if (void* at = bake.uniform(call, block, member, size)) return at;
        }
        return nullptr;
    }

    template <typename T> void setMember(Call& call, const char* member, const T& value)
    {
        if (void* at = passMember(call, member, sizeof(T))) std::memcpy(at, &value, sizeof(T));
    }

    uint32_t memberOr(Call& call, const char* member, uint32_t fallback)
    {
        void* at = passMember(call, member, sizeof(uint32_t));
        if (at) std::memcpy(&fallback, at, sizeof(uint32_t));
        return fallback;
    }

    /// The settings as they were in the baked frame, read from what Falcor set there.
    void setUpSettings()
    {
        for (size_t i = 0; i < bake.program.size(); i++)
        {
            const Step& step = bake.program[i];
            if (!step.isCall)
            {
                if (step.host[0] == "train") trainSectionEnd = i + 1;
                continue;
            }
            Call& call = bake.calls.at(step.step);
            if (call.pass == "SampleRHS")
            {
                settings.useNEE = memberOr(call, "useNEE", 1) != 0;
                settings.rhsGlossyBounces = memberOr(call, "maxGlossyBounces", settings.rhsGlossyBounces);
                hasLights = memberOr(call, "useEmissiveLights", 0) != 0 || memberOr(call, "useAnalyticLights", 0) != 0;
            }
            else if (call.pass == "RandomSmooth")
            {
                settings.diffTrainOnGlossy = memberOr(call, "diffTrainOnGlossy", 0) != 0;
            }
            else if (call.pass == "SpecPrimary" || call.pass == "ClusterSpatialReuse")
            {
                settings.mirrorAlbedo = memberOr(call, "enableMirrorAlbedo", 1) != 0;
                settings.envColorEncoding = memberOr(call, "envColorEncoding", settings.envColorEncoding);
            }
            else if (call.pass == "PostFXAAPass")
            {
                hasPostFilter = true;
                settings.fxaa = memberOr(call, "enableFXAA", 1) != 0;
            }
        }
        hasRenderPath = trainSectionEnd < bake.program.size();
        settings.postFilter = hasPostFilter;
        settings.mode = !hasRenderPath ? RenderMode::Train : trainSectionEnd > 0 ? RenderMode::OnlineTrain : RenderMode::Render;
        settings.animate = settings.cameraAnimated = bake.track.samples > 0;
        if (bake.track.samples > 0) bake.track.bounds(0, bounds);
    }

    /// The scene for the frame that is about to run: the animation's sample for the time, the
    /// camera, and from both the events the training schedule grades. The GPU is idle here.
    void updateScene(double seconds)
    {
        const auto start = std::chrono::steady_clock::now();
        uint32_t events = 0;
        bool moved = false;
        const Track& track = bake.track;
        if (track.samples > 0)
        {
            uint32_t passed = 0;   // samples the animation went through since the last frame
            if (settings.animate && seconds > 0.0)
            {
                animationTime = std::fmod(animationTime + seconds * (double)settings.timeScale, track.loopSeconds);
                if (animationTime < 0.0) animationTime += track.loopSeconds;
            }
            const uint32_t target = std::min((uint32_t)(animationTime * track.rate), track.samples - 1);
            if (target != sample || reapplySample)
            {
                passed = target >= sample ? target - sample : target + track.samples - sample;
                // Falcor raises the events of an animation in every frame it plays: those of the
                // samples this frame skipped are part of what changed since the last frame.
                for (uint32_t i = 0; i < std::min(passed, track.samples); i++) events |= track.events((sample + 1 + i) % track.samples);
                bake.applyTrackSample(target);
                track.bounds(target, bounds);
                sample = target;
                reapplySample = false;
                moved = true;
            }
        }
        // The frame after an animation stops, Falcor still reports moved geometry once: its animation
        // controller updates the previous-frame matrices then (Scene::update, "mTime != mPrevTime").
        const bool objectsMoved = (events & TrainSchedule::kEventObject) != 0;
        if (!objectsMoved && objectsMovedPrevious) events |= TrainSchedule::kEventObject;
        objectsMovedPrevious = objectsMoved;
        // Geometry that moved since the last frame, as the cluster passes' clock sees it.
        geometryMoved = (events & TrainSchedule::kEventObject) != 0;
        const bool followsTrack = track.samples > 0 && settings.cameraAnimated;
        if (!followsTrack)
        {
            events &= ~TrainSchedule::kEventCamera;
            if (!poseApplied)
            {
                // Where the camera is now is where it stays until the caller sets another pose.
                appliedPose = bake.camera();
                poseApplied = true;
                if (!poseSet) setPose = appliedPose;
                poseSet = true;
            }
            if (setPose != appliedPose)
            {
                if (moved) bake.setCamera(appliedPose);   // the track wrote its own camera: first the pose of the last frame, as the previous one
                bake.setCamera(setPose);
                appliedPose = setPose;
                events |= TrainSchedule::kEventCamera;
            }
            else if (moved)
            {
                bake.setCamera(appliedPose);
            }
        }
        else if (poseApplied)
        {
            // Back on the path: the track's camera returns with this frame's sample.
            poseApplied = poseSet = false;
            if (!moved) bake.applyTrackSample(sample);
            events |= TrainSchedule::kEventCamera;
        }
        lastEvents = events;
        pendingEvents |= events;
        if (moved || events != 0) timings.push_back({format("scene update (track sample %u)", sample), msSince(start)});
    }

    /// The learning rate and EMA decay: fixed when asked for, otherwise Falcor's schedule continued
    /// from its state after the baked frame. The baked frame runs with the values Falcor used in it.
    void setUpSchedule()
    {
        schedule.learningRate = (float)bake.value("train.learningRate", 5e-3);
        schedule.emaDecay = (float)bake.value("train.emaDecay", 0.95);
        if (options.learningRate > 0.0f || options.emaDecay > 0.0f)
        {
            if (options.learningRate > 0.0f) schedule.learningRate = options.learningRate;
            if (options.emaDecay > 0.0f) schedule.emaDecay = options.emaDecay;
            return;
        }
        if (bake.values.count("train.schedule") == 0)
        {
            if (training) std::printf("note: the bake has no training schedule; the values of the baked frame are kept\n");
            return;
        }
        useSchedule = true;
        schedule.enabled = bake.value("train.schedule") != 0;
        schedule.heat = (float)bake.value("train.heat");
        schedule.holdLeft = (uint32_t)bake.value("train.holdLeft");
        schedule.lrHot = (float)bake.value("train.lrHot");
        schedule.lrCold = (float)bake.value("train.lrCold");
        schedule.emaHot = (float)bake.value("train.emaHot");
        schedule.emaCold = (float)bake.value("train.emaCold");
        // Bakes made before the stage 4 snapshots call it train.warmupFrames.
        schedule.initialHotFrames = (uint32_t)bake.value("train.initialHotFrames", bake.value("train.warmupFrames", 600));
        schedule.holdFrames = (uint32_t)bake.value("train.holdFrames");
        schedule.coolFrames = (float)bake.value("train.coolFrames");
        schedule.cameraHeat = (float)bake.value("train.cameraHeat");
        schedule.sceneHeat = (float)bake.value("train.sceneHeat");
        schedule.lightMotionHeat = (float)bake.value("train.lightMotionHeat");
        schedule.objectMotionHeat = (float)bake.value("train.objectMotionHeat");
    }

    Buffer zeroed(VkDeviceSize bytes)
    {
        Buffer b = dev.makeBuffer(bytes, kStorage, true);
        std::memset(b.data, 0, (size_t)bytes);
        return b;
    }

    // Tiles: what the GPU passes and the MLPs exchange. With the NPU they are shared memory.
    Buffer tileBuffer(VkDeviceSize bytes)
    {
        Buffer b = sharedTiles ? dev.makeSharedBuffer(bytes, kStorage) : dev.makeBuffer(bytes, kStorage, true);
        std::memset(b.data, 0, (size_t)bytes);
        return b;
    }

    static TileMemory tiles(const Buffer& b) { return TileMemory{b.as<uint16_t>(), b.fd, b.sharedSize}; }

    void createGrid(Grid& g, const std::vector<uint16_t>& ema, const std::vector<float>& master, const std::vector<uint16_t>& working)
    {
        g.elements = (uint32_t)ema.size();
        g.inference = dev.makeBuffer(ema.size() * 2, kStorage, true);
        std::memcpy(g.inference.data, ema.data(), ema.size() * 2);
        if (!training) return;
        g.master = dev.makeBuffer(master.size() * 4, kStorage, true);
        std::memcpy(g.master.data, master.data(), master.size() * 4);
        g.training = dev.makeBuffer(working.size() * 2, kStorage, true);
        std::memcpy(g.training.data, working.data(), working.size() * 2);
        g.firstMoment = zeroed(master.size() * 4);
        g.secondMoment = zeroed(master.size() * 4);
        g.entrySteps = zeroed(master.size() * 4);
        g.gradient = zeroed(master.size() * 4);
        g.optimizer.create(dev, ownPool, options.shaderDir + "/GridOptimizer.spv", 7, sizeof(OptimizerPush), 256);
        const Buffer* bound[7] = {&g.gradient, &g.master, &g.firstMoment, &g.secondMoment, &g.entrySteps, &g.training, &g.inference};
        for (uint32_t i = 0; i < 7; i++) g.optimizer.bind(dev, i, *bound[i]);
    }

    // Tile buffers and the passes around them for one batch, created when the batch first needs them.
    TilePath& tilePath(const Batch& b)
    {
        auto it = tilePaths.find(b.role);
        if (it != tilePaths.end()) return it->second;
        TilePath& t = tilePaths[b.role];
        const uint32_t maxTiles = (b.size + cfg::kTile - 1) / cfg::kTile;
        const VkDeviceSize tile = (VkDeviceSize)cfg::kTile * 2;
        auto create = [&](bool spec, Buffer& x, Buffer& y, OwnPass& encode, OwnPass& decode) {
            x = tileBuffer(maxTiles * (spec ? cfg::kSpecInputDim : cfg::kDiffInputDim) * tile);
            y = tileBuffer(maxTiles * cfg::kOutputChannels * tile);
            encode.create(dev, ownPool, options.shaderDir + (spec ? "/EncodeSpec.spv" : "/EncodeDiff.spv"), 4, sizeof(EncodePush), 64);
            decode.create(dev, ownPool, options.shaderDir + "/DecodeOutputs.spv", 2, sizeof(DecodePush), 64);
            encode.bind(dev, 0, bake.buffer(b.role + (spec ? ".specInput" : ".diffInput")));
            encode.bind(dev, 1, primGrid.inference);
            encode.bind(dev, 2, clsGrid.inference);
            encode.bind(dev, 3, x);
            decode.bind(dev, 0, y);
            decode.bind(dev, 1, bake.buffer(b.role + (spec ? ".specColor" : ".diffColor")));
        };
        if (b.needSpec) create(true, t.specX, t.specY, t.encodeSpec, t.decodeSpec);
        if (b.needDiff) create(false, t.diffX, t.diffY, t.encodeDiff, t.decodeDiff);
        return t;
    }

    // The same for training: input, target and gradient tiles of the training graph's size.
    TrainPath& trainPath(const Batch& b)
    {
        auto it = trainPaths.find(b.role);
        if (it != trainPaths.end()) return it->second;
        TrainPath& t = trainPaths[b.role];
        const uint32_t trainTile = options.trainTile;
        const uint32_t maxTiles = (b.size + trainTile - 1) / trainTile;
        const VkDeviceSize tile = (VkDeviceSize)trainTile * 2;
        auto create = [&](bool spec, TrainNet& n) {
            const uint32_t channels = spec ? cfg::kSpecInputDim : cfg::kDiffInputDim;
            n.x = tileBuffer(maxTiles * channels * tile);
            n.target = tileBuffer(maxTiles * cfg::kOutputChannels * tile);
            n.gradient = tileBuffer(maxTiles * channels * tile);
            n.weightGradient.resize((spec ? specNet : diffNet).master.size());
            n.encode.create(dev, ownPool, options.shaderDir + (spec ? "/EncodeSpec.spv" : "/EncodeDiff.spv"), 4, sizeof(EncodePush), 64);
            n.targets.create(dev, ownPool, options.shaderDir + "/EncodeTargets.spv", 2, sizeof(TargetPush), 64);
            n.scatter.create(dev, ownPool, options.shaderDir + (spec ? "/ScatterSpec.spv" : "/ScatterDiff.spv"), spec ? 4 : 3, sizeof(ScatterPush), 64);
            n.encode.bind(dev, 0, bake.buffer(b.role + (spec ? ".specInput" : ".diffInput")));
            n.encode.bind(dev, 1, primGrid.training);
            n.encode.bind(dev, 2, clsGrid.training);
            n.encode.bind(dev, 3, n.x);
            n.targets.bind(dev, 0, bake.buffer(b.role + (spec ? ".specColor" : ".diffColor")));
            n.targets.bind(dev, 1, n.target);
            n.scatter.bind(dev, 0, n.x);
            n.scatter.bind(dev, 1, n.gradient);
            n.scatter.bind(dev, 2, primGrid.gradient);
            if (spec) n.scatter.bind(dev, 3, clsGrid.gradient);
        };
        if (b.needSpec) create(true, t.spec);
        if (b.needDiff) create(false, t.diff);
        return t;
    }

    void flush()
    {
        if (cb == VK_NULL_HANDLE) return;
        timings.push_back({pendingNames, dev.submit(cb)});
        cb = VK_NULL_HANDLE;
        pendingNames.clear();
    }

    VkCommandBuffer record(const std::string& name)
    {
        if (cb == VK_NULL_HANDLE) cb = dev.begin();
        pendingNames += (pendingNames.empty() ? "" : " + ") + name;
        return cb;
    }

    // Mask -> compacted index, and the count into the batch's counts (NeuralRadiosity::computeBatchCounts).
    void prefixSumGpu(const std::string& role, bool spec, uint32_t count)
    {
        const std::string key = role + (spec ? ".spec" : ".diff");
        auto it = prefixPaths.find(key);
        if (it == prefixPaths.end())
        {
            PrefixPath& created = prefixPaths[key];
            created.blockSums = dev.makeBuffer((VkDeviceSize)(count / kPrefixBlock + 1) * 4, kStorage, false);
            created.pass.create(dev, ownPool, options.shaderDir + "/PrefixSum.spv", 3, sizeof(PrefixPush), 64);
            created.pass.bind(dev, 0, bake.buffer(role + (spec ? ".specIndex" : ".diffIndex")));
            created.pass.bind(dev, 1, created.blockSums);
            created.pass.bind(dev, 2, bake.buffer(role + ".counts"));
            it = prefixPaths.find(key);
        }
        VkCommandBuffer c = record("prefix sum (" + role + (spec ? ", spec)" : ", diff)"));
        const uint32_t index = spec ? 1 : 0;
        it->second.pass.dispatch(c, PrefixPush{count, 0, index}, (count + kPrefixBlock - 1) / kPrefixBlock);
        it->second.pass.dispatch(c, PrefixPush{count, 1, index}, 1);
        it->second.pass.dispatch(c, PrefixPush{count, 2, index}, count);
    }

    /// NeuralRadiosity::savePrevGBuffer: what the next frame's ClusterReproject reads as the previous frame.
    void savePreviousFrame()
    {
        VkCommandBuffer c = record("previous-frame copies");
        static const char* const pairs[4][2] = {
            {"gPos", "cluster.prevPos"}, {"gDir", "cluster.prevDir"}, {"gNormal", "cluster.prevNormal"}, {"gRoughness", "cluster.prevRoughness"}};
        for (const auto& pair : pairs)
        {
            const Texture& from = bake.textures.at(bake.textureBoundAs(pair[0]));
            const Texture& to = bake.texture(pair[1]);
            VkImageCopy region = {};
            region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {std::min(from.width, to.width), std::min(from.height, to.height), 1};
            vkCmdCopyImage(c, from.image, VK_IMAGE_LAYOUT_GENERAL, to.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
        const Buffer& mask = bake.buffer("render.specActive");
        const Buffer& previousMask = bake.buffer("cluster.prevSpecActive");
        const VkBufferCopy copy = {0, 0, std::min(mask.size, previousMask.size)};
        vkCmdCopyBuffer(c, mask.buffer, previousMask.buffer, 1, &copy);
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    /// NeuralRadiosity::clearRenderTemporalHistory.
    void clearHistoryNow()
    {
        VkCommandBuffer c = record("clear cluster history");
        for (const char* role : {"cluster.history0", "cluster.history1", "cluster.lockState0", "cluster.lockState1", "cluster.prevSpecActive"})
            vkCmdFillBuffer(c, bake.buffer(role).buffer, 0, VK_WHOLE_SIZE, 0);
        // The recovery countdown goes with the history; the frequency counters stay, as in Falcor.
        if (hasRecovery)
            for (const char* role : {"cluster.vcdRecovery0", "cluster.vcdRecovery1"}) vkCmdFillBuffer(c, bake.buffer(role).buffer, 0, VK_WHOLE_SIZE, 0);
        clusterClock.motionFarWeightFrames = 0;
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        clusterHistoryValid = false;
        clearClusterHistory = false;
    }

    /// Falcor reads the compacted counts back where it syncs with CUDA.
    void readCounts(Batch& b)
    {
        const uint32_t* counts = bake.buffer(b.role + ".counts").as<uint32_t>();
        b.diffSize = b.needDiff ? counts[0] : 0;
        b.specSize = b.needSpec ? counts[1] : 0;
    }

    // Network inference of one batch: the diffuse rows through DiffNet, the specular rows through SpecNet.
    void infer(Batch& b)
    {
        readCounts(b);
        if (useTiles)
        {
            TilePath& t = tilePath(b);
            VkCommandBuffer c = record("grid lookups to tiles (" + b.role + ")");
            if (b.needSpec) t.encodeSpec.dispatch(c, EncodePush{b.specSize, 0, inferShift}, b.specSize);
            if (b.needDiff) t.encodeDiff.dispatch(c, EncodePush{b.diffSize, b.diffInputOffset() * cfg::kDiffInputDim, inferShift}, b.diffSize);
            flush();
            const auto t0 = std::chrono::steady_clock::now();
            const uint32_t specTiles = (b.specSize + cfg::kTile - 1) / cfg::kTile, diffTiles = (b.diffSize + cfg::kTile - 1) / cfg::kTile;
            if (b.needSpec) backend->infer(true, tiles(t.specX), tiles(t.specY), specTiles);
            if (b.needDiff) backend->infer(false, tiles(t.diffX), tiles(t.diffY), diffTiles);
            timings.push_back({"MLPs (" + b.role + "), " + std::to_string(specTiles) + " + " + std::to_string(diffTiles) + " tiles", msSince(t0)});
            c = record("outputs to rows (" + b.role + ")");
            if (b.needSpec) t.decodeSpec.dispatch(c, DecodePush{b.specSize, 0}, b.specSize);
            if (b.needDiff) t.decodeDiff.dispatch(c, DecodePush{b.diffSize, b.diffOutputOffset()}, b.diffSize);
        }
        else
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (b.needSpec)
                cpuNetworks.inferSpec(bake.buffer(b.role + ".specInput").as<uint16_t>(), b.specSize, bake.buffer(b.role + ".specColor").as<float>());
            if (b.needDiff)
                cpuNetworks.inferDiff(bake.buffer(b.role + ".diffInput").as<uint16_t>() + (size_t)b.diffInputOffset() * cfg::kDiffInputDim, b.diffSize,
                                      bake.buffer(b.role + ".diffColor").as<float>() + (size_t)b.diffOutputOffset() * cfg::kOutputDim);
            timings.push_back({"networks on the CPU (" + b.role + ")", msSince(t0)});
        }
    }

    // One training step on a batch whose color buffers hold the targets (NRModel::train).
    void train(Batch& b)
    {
        readCounts(b);
        TrainPath& t = trainPath(b);
        const uint32_t trainTile = options.trainTile;
        const float learningRate = schedule.learningRate, emaDecay = schedule.emaDecay;
        struct Job
        {
            bool spec;
            uint32_t size, tiles, rows;   // samples, tiles of the training graph, samples the loss is normalised by
            float weight;                 // of every sample in the target tiles (sampleWeight)
            TrainNet* net;
            Mlp* mlp;
            uint32_t inputBase, targetBase;
        };
        std::vector<Job> jobs;
        // Falcor (Model/TrainingBatch.h) pads the batch to 256 rows for the MLP, but the padding has
        // no weight and the loss is divided by the samples of the batch. Here the slots of the last
        // tile beyond the batch have weight zero.
        if (b.needSpec && b.specSize > 0)
            jobs.push_back({true, b.specSize, (b.specSize + trainTile - 1) / trainTile, b.specSize, sampleWeight(b.specSize, trainTile), &t.spec,
                            &specNet, 0, 0});
        if (b.needDiff && b.diffSize > 0)
            jobs.push_back({false, b.diffSize, (b.diffSize + trainTile - 1) / trainTile, b.diffSize, sampleWeight(b.diffSize, trainTile), &t.diff,
                            &diffNet, b.diffInputOffset() * cfg::kDiffInputDim, b.diffOutputOffset()});
        if (jobs.empty()) return;
        const bool dump = !options.dumpTrainDir.empty() && trainSteps == 0;

        VkCommandBuffer c = record("training inputs and targets (" + b.role + ")");
        for (const Job& j : jobs)
        {
            j.net->encode.dispatch(c, EncodePush{j.size, j.inputBase, trainShift}, j.size);
            j.net->targets.dispatch(c, TargetPush{j.size, j.targetBase, trainShift, j.tiles * trainTile, j.weight}, j.tiles * trainTile);
        }
        flush();

        if (options.lossInterval != 0 && frame % options.lossInterval == 0)
        {
            std::printf("frame %u loss:", frame);
            for (const Job& j : jobs)
            {
                const double sum = cpuNetworks.trainTiles(j.spec, trainTile, 1, j.net->x.as<uint16_t>(), j.net->target.as<uint16_t>(), nullptr, nullptr);
                std::printf(" %s %.4f", j.spec ? "specular" : "diffuse", sum / ((double)std::min(j.size, trainTile) * cfg::kOutputDim));
            }
            std::printf("  (learning rate %.4g, EMA decay %.5g)\n", learningRate, emaDecay);
        }

        auto t0 = std::chrono::steady_clock::now();
        std::string tileNote;
        for (const Job& j : jobs)
        {
            std::fill(j.net->weightGradient.begin(), j.net->weightGradient.end(), 0.0f);
            backend->train(j.spec, trainTile, j.tiles, tiles(j.net->x), tiles(j.net->target), j.net->weightGradient.data(), tiles(j.net->gradient));
            // Every tile normalised its loss by its own size and the samples carry their weight: bring
            // the sum to the batch.
            const float toBatch = (float)trainTile / ((float)j.rows * j.weight);
            for (float& g : j.net->weightGradient) g *= toBatch;
            tileNote += (tileNote.empty() ? "" : " + ") + std::to_string(j.tiles);
        }
        timings.push_back({"MLP gradients (" + b.role + "), " + tileNote + " tiles", msSince(t0)});

        c = record("grid gradients (" + b.role + ")");
        bool anySpec = false;
        for (const Job& j : jobs)
        {
            j.net->scatter.dispatch(c, ScatterPush{j.size, trainShift, adam::kGridGradientScale * (float)trainTile / ((float)j.rows * j.weight)}, j.size);
            anySpec = anySpec || j.spec;
        }
        if (dump)
        {
            flush();
            const std::string& dir = options.dumpTrainDir;
            for (const Job& j : jobs)
            {
                const std::string prefix = dir + "/train_" + (j.spec ? "spec" : "diff");
                const size_t channels = j.mlp->inputDim;
                writeFile(prefix + "_x.bin", j.net->x.data, (size_t)j.tiles * channels * trainTile * 2);
                writeFile(prefix + "_target.bin", j.net->target.data, (size_t)j.tiles * cfg::kOutputChannels * trainTile * 2);
                writeFile(prefix + "_gx.bin", j.net->gradient.data, (size_t)j.tiles * channels * trainTile * 2);
                writeFile(prefix + "_gw.bin", j.net->weightGradient.data(), j.net->weightGradient.size() * 4);
            }
            writeFile(dir + "/train_prim_grad.bin", primGrid.gradient.data, (size_t)primGrid.gradient.size);
            writeFile(dir + "/train_cls_grad.bin", clsGrid.gradient.data, (size_t)clsGrid.gradient.size);
            // The training copies the step started from, which train_check.py takes for the reference step.
            const uint16_t* working[4] = {primGrid.training.as<uint16_t>(), clsGrid.training.as<uint16_t>(), diffNet.training.data(), specNet.training.data()};
            std::ofstream f(dir + "/train_working.bin", std::ios::binary);
            Checkpoint::Header header = checkpoint.header();
            std::memcpy(header.magic, "NRWK", 4);
            header.version = 1;
            f.write(reinterpret_cast<const char*>(&header), sizeof(header));
            for (int i = 0; i < 4; i++) f.write(reinterpret_cast<const char*>(working[i]), (std::streamsize)header.counts[i] * 2);
            std::ofstream info(dir + "/train.txt");
            info << "tile " << trainTile << "\nlearning_rate " << learningRate << "\nema_decay " << emaDecay << "\n";
            for (const Job& j : jobs)
                info << (j.spec ? "spec" : "diff") << " " << j.size << " " << j.tiles << " " << j.rows << " " << j.weight << "\n";
        }
        auto optimize = [&](Grid& g, const char* name) {
            g.steps++;
            g.optimizer.dispatch(record(std::string("optimizer (") + name + ")"),
                                 OptimizerPush{g.elements, learningRate, 1.0f / (adam::kGridGradientScale * adam::kLossScale), emaDecay,
                                               adam::emaDebiasOld(emaDecay, g.steps), adam::emaDebiasNew(emaDecay, g.steps)},
                                 g.elements);
        };
        optimize(primGrid, "primary grid");
        if (anySpec) optimize(clsGrid, "cluster grid");
        flush();

        t0 = std::chrono::steady_clock::now();
        for (const Job& j : jobs)
        {
            j.mlp->step(j.net->weightGradient.data(), learningRate, emaDecay);
            backend->weightsChanged(j.spec);
        }
        timings.push_back({"optimizer (MLPs, CPU)", msSince(t0)});
        trainSteps++;
    }

    // What Falcor sets anew every frame in a call's uniform block, for the frames after the baked
    // one (which runs with the values Falcor had): the frame counter that seeds the samplers, and
    // what depends on the sample counts the host knows at that point (NeuralRadiosity.cpp).
    void prepare(Call& call, uint32_t dims[3])
    {
        // The panel's switches, where Falcor puts them (NeuralRadiosity.cpp). Untouched, they are the baked values.
        if (call.pass == "SampleRHS")
        {
            if (hasLights) setMember(call, "useNEE", (uint32_t)(settings.useNEE ? 1 : 0));
            setMember(call, "maxGlossyBounces", settings.rhsGlossyBounces);
            setMember(call, "queryGlossyFallback", (uint32_t)((settings.diffTrainOnGlossy || settings.rhsGlossyBounces == 0) ? 1 : 0));
        }
        else if (call.pass == "RandomSmooth")
        {
            setMember(call, "diffTrainOnGlossy", (uint32_t)(settings.diffTrainOnGlossy ? 1 : 0));
        }
        else if (call.pass == "SpecPrimary" || call.pass == "ClusterSpatialReuse")
        {
            setMember(call, "enableMirrorAlbedo", (uint32_t)(settings.mirrorAlbedo ? 1 : 0));
            setMember(call, "envColorEncoding", settings.envColorEncoding);
        }
        else if (call.pass == "PostFXAAPass")
        {
            setMember(call, "enableFXAA", (uint32_t)(settings.fxaa ? 1 : 0));
        }
        // The scene bounds, which positions are normalised by, follow the animation.
        if (bake.track.samples > 0)
        {
            if (void* at = passMember(call, "bboxMin", 12)) std::memcpy(at, bounds, 12);
            if (void* at = passMember(call, "bboxMax", 12)) std::memcpy(at, bounds + 3, 12);
        }
        const std::string block = "g" + call.pass;
        auto set = [&](const char* member, uint32_t value) {
            if (bake.hasUniform(call, block, member)) bake.setUniform(call, block, member, value);
        };
        // The frame counter seeds the samplers. The baked frame keeps Falcor's own uniforms unless it
        // is replayed as another frame.
        const uint32_t firstCount = options.frameCount >= 0 ? (uint32_t)options.frameCount : bake.frameCount;
        if (frame > 0 || firstCount != bake.frameCount)
        {
            set("batchCount", firstCount + frame);
            set("frameCount", firstCount + frame);
        }
        // Whether there is a history to reproject from (the baked frame keeps what Falcor had).
        if (call.pass == "ClusterReproject" && frame > 0) set("historyValid", clusterHistoryValid ? 1u : 0u);
        // What the pass counts on the CPU (NeuralRadiosity::coneTraceScreenPass, clusterSpatialReusePass).
        if (call.pass == "ConeTraceScreen" && frame > 0)
        {
            const ClusterClock& k = clusterClock;
            set("visibilityFrame", k.visibilityFrame);
            set("visibilityFrameAdvanced", k.visibilityFrameAdvanced ? 1u : 0u);
            set("checkFrequency", k.checkFrequency && k.frequencyFrames > 0 ? 1u : 0u);
            set("frequencySurveyFrame", k.frequencySurveyFrame());
        }
        if (call.pass == "ClusterSpatialReuse" && frame > 0) setMember(call, "motionFarWeightStrength", clusterClock.motionFarWeightStrength());
        if (frame == 0) return;
        auto it = batches.find(call.batch);
        if (it == batches.end()) return;
        const Batch& b = it->second;
        set("diffSize", b.diffSize);
        set("specSize", b.specSize);
        set("diffColorOffset", b.diffOutputOffset());
        if (call.pass == "ConeTrace")
        {
            const uint32_t groups = specDispatchItems(b.specSize, b.size);
            set("dispatchGroupCount", groups);
            dims[0] = groups * bake.passes.at(call.pass).group[0];
        }
        else if (call.pass == "SpecPrimary")
        {
            dims[0] = specDispatchItems(b.specSize, b.size);
            set("dispatchThreadCount", dims[0]);
        }
        else if (call.pass == "ConeTraceScreen" || call.pass == "ClusterSpatialReuse")
        {
            dims[0] = b.specSize;   // read back just before (host step readcounts), as on the desktop
        }
    }

    void runFrame(double seconds)
    {
        timings.clear();
        updateScene(seconds);
        // What the camera was in the last frame is this frame's "previous" camera, whichever way
        // the scene got its camera for this frame (the bake, a track sample, the caller).
        if (frame > 0 && haveLastCamera) bake.setCameraHistory(lastCamera);
        if (clearClusterHistory && clusterReuse) clearHistoryNow();
        // The pass's CPU-side counters for this frame; the baked frame ran with Falcor's.
        if (clusterReuse && frame > 0) clusterClock.advance(geometryMoved, bake.viewChanged(), clusterHistoryValid);
        const RenderMode mode = settings.mode;
        const bool trainPart = mode == RenderMode::Train || mode == RenderMode::OnlineTrain;
        const bool renderPart = mode == RenderMode::Render || mode == RenderMode::OnlineTrain;
        // Falcor updates the schedule before the frame's training step, with the events since the
        // last training frame; the baked frame keeps its values.
        if (trainPart && trainSectionEnd > 0)
        {
            if (useSchedule && frame > 0) schedule.advance(pendingEvents);
            pendingEvents = 0;
        }
        for (size_t index = 0; index < bake.program.size(); index++)
        {
            const Step& step = bake.program[index];
            if (!(index < trainSectionEnd ? trainPart : renderPart)) continue;
            if (step.isCall)
            {
                Call& call = bake.calls.at(step.step);
                if (!settings.postFilter && (call.pass == "PostFilterPass" || call.pass == "PostFXAAPass" || call.pass == "IsolatedHighlightPass"))
                    continue;
                uint32_t dims[3] = {call.dims[0], call.dims[1], call.dims[2]};
                prepare(call, dims);
                if (call.pass == "ConeTrace")
                {
                    // One thread group per specular sample, and a group strides over further samples when
                    // there are fewer groups than samples. The desktop takes any number of groups; the phone
                    // runs 65,535 per dimension at most, so the count is capped and the shader told the stride.
                    const uint32_t groupSize = bake.passes.at(call.pass).group[0];
                    const uint32_t groups = std::min(bake.getUniform<uint32_t>(call, "gConeTrace", "dispatchGroupCount"), dev.maxGroupCount[0]);
                    bake.setUniform(call, "gConeTrace", "dispatchGroupCount", groups);
                    dims[0] = groups * groupSize;
                }
                if (call.pass == "ClusterReproject")
                {
                    // The history pairs swap roles every frame the cluster passes run (mClusterHistoryWriteIdx).
                    clusterAlternate = (clusterRuns & 1u) != 0;
                    clusterRuns++;
                    clusterHistoryValid = true;   // after this frame; prepare() has given the pass the value before it
                }
                if (dims[0] == 0) continue;   // nothing to do (a frame without specular pixels)
                const bool clusterCall = call.pass == "ClusterReproject" || call.pass == "ConeTraceScreen" || call.pass == "ClusterSpatialReuse";
                bake.dispatch(record(call.pass), call, dims[0], dims[1], dims[2], clusterCall && clusterAlternate);
                if (options.perPass) flush();
                continue;
            }
            const std::string& what = step.host[0];
            if (what == "clearcounts" && !options.cpuPrefixSum)
            {
                // On the GPU, in order with the calls around it: nothing has to wait for the host.
                VkCommandBuffer c = record("clear counts (" + step.host[1] + ")");
                vkCmdFillBuffer(c, bake.buffer(step.host[1] + ".counts").buffer, 0, 2 * sizeof(uint32_t), 0);
                VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            }
            else if (what == "prefixsum" && !options.cpuPrefixSum)
            {
                prefixSumGpu(step.host[1], step.host[2] == "spec", (uint32_t)std::stoul(step.host[3]));
            }
            else if (what == "clearcounts")
            {
                flush();
                std::memset(bake.buffer(step.host[1] + ".counts").data, 0, 2 * sizeof(uint32_t));
            }
            else if (what == "prefixsum")
            {
                flush();
                const auto t0 = std::chrono::steady_clock::now();
                const bool spec = step.host[2] == "spec";
                const uint32_t count = (uint32_t)std::stoul(step.host[3]);
                uint32_t* counts = bake.buffer(step.host[1] + ".counts").as<uint32_t>();
                counts[spec ? 1 : 0] = prefixSum(bake.buffer(step.host[1] + (spec ? ".specIndex" : ".diffIndex")).as<uint32_t>(), count);
                timings.push_back({"prefix sum on the CPU (" + step.host[1] + ", " + step.host[2] + ")", msSince(t0)});
            }
            else if (what == "infer")
            {
                flush();   // the host reads the counts
                infer(batches.at(step.host[1]));
            }
            else if (what == "train")
            {
                flush();
                if (training) train(batches.at(step.host[1]));
            }
            else if (what == "readcounts")
            {
                flush();   // the passes that follow are dispatched for this frame's count
                readCounts(batches.at(step.host[1]));
            }
            else if (what == "saveprev")
            {
                savePreviousFrame();
            }
            else if (what == "clearstats")
            {
                // NeuralRadiosity::coneTraceScreenPass clears the ray budget's counters before every dispatch.
                if (hasStats)
                {
                    VkCommandBuffer c = record("clear ray-budget counters");
                    vkCmdFillBuffer(c, bake.buffer("cluster.farValidationStats").buffer, 0, VK_WHOLE_SIZE, 0);
                    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                }
            }
            else
            {
                fail("unknown host step " + what);
            }
        }
        flush();
        lastCamera = bake.cameraHistory();
        haveLastCamera = true;
        lastFrameMs = 0.0;
        for (const auto& [name, ms] : timings) lastFrameMs += ms;
        frame++;
    }
};

Runtime::Runtime(Device& dev, const std::string& bakeDir, const std::string& model, const RuntimeOptions& options)
    : s(std::make_unique<State>(dev, bakeDir, model, options))
{
}

Runtime::~Runtime() = default;

void Runtime::frame(double seconds) { s->runFrame(seconds); }
uint32_t Runtime::framesDone() const { return s->frame; }
uint32_t Runtime::width() const { return s->bake.width; }
uint32_t Runtime::height() const { return s->bake.height; }
uint32_t Runtime::bakedFrame() const { return s->bake.frameCount; }
double Runtime::lastFrameMs() const { return s->lastFrameMs; }
const std::vector<std::pair<std::string, double>>& Runtime::lastTimings() const { return s->timings; }
const Texture& Runtime::outputImage() const { return s->bake.textures.at(s->outputTexture); }
bool Runtime::trains() const { return s->training; }
uint32_t Runtime::trainSteps() const { return s->trainSteps; }
TrainSchedule& Runtime::schedule() { return s->schedule; }
const TrainSchedule& Runtime::schedule() const { return s->schedule; }
uint32_t Runtime::lastEvents() const { return s->lastEvents; }
Settings& Runtime::settings() { return s->settings; }
const Settings& Runtime::settings() const { return s->settings; }
bool Runtime::hasPostFilter() const { return s->hasPostFilter; }
bool Runtime::hasLights() const { return s->hasLights; }
bool Runtime::hasRenderPath() const { return s->hasRenderPath; }
bool Runtime::clusterReuse() const { return s->clusterReuse; }
void Runtime::clearHistory() { s->clearClusterHistory = true; }
const ClusterClock& Runtime::clusterClock() const { return s->clusterClock; }
bool Runtime::animated() const { return s->bake.track.samples > 0; }
double Runtime::animationLength() const { return s->bake.track.loopSeconds; }
double Runtime::animationTime() const { return s->animationTime; }
uint32_t Runtime::trackSamples() const { return s->bake.track.samples; }
double Runtime::trackRate() const { return s->bake.track.rate; }
uint32_t Runtime::trackSample() const { return s->sample; }

void Runtime::setAnimationTime(double seconds)
{
    if (s->bake.track.samples == 0) return;
    s->animationTime = std::fmod(std::max(seconds, 0.0), s->bake.track.loopSeconds);
}

CameraPose Runtime::camera() const { return s->poseApplied ? s->appliedPose : s->bake.camera(); }

double Runtime::bakeValue(const std::string& name, double fallback) const { return s->bake.value(name, fallback); }

bool Runtime::sceneBounds(float minMax[6]) const
{
    if (s->bake.track.samples > 0)
    {
        std::memcpy(minMax, s->bounds, sizeof(s->bounds));
        return true;
    }
    for (auto& [step, call] : s->bake.calls)
    {
        void* low = s->passMember(call, "bboxMin", 12);
        void* high = s->passMember(call, "bboxMax", 12);
        if (!low || !high) continue;
        std::memcpy(minMax, low, 12);
        std::memcpy(minMax + 3, high, 12);
        return true;
    }
    return false;
}

void Runtime::setCamera(const CameraPose& pose)
{
    s->setPose = pose;
    s->poseSet = true;
}

namespace
{
/// The settings by name: where each lives, and how a number becomes its value.
struct Named
{
    const char* name;
    double (*get)(const Settings&, const TrainSchedule&);
    void (*set)(Settings&, TrainSchedule&, double);
};
#define NCR_SETTING(name, expression, type) \
    {name, [](const Settings& s, const TrainSchedule& t) { (void)s; (void)t; return (double)(expression); }, \
     [](Settings& s, TrainSchedule& t, double v) { (void)s; (void)t; expression = (type)v; }}
#define NCR_SWITCH(name, expression) \
    {name, [](const Settings& s, const TrainSchedule& t) { (void)s; (void)t; return (expression) ? 1.0 : 0.0; }, \
     [](Settings& s, TrainSchedule& t, double v) { (void)s; (void)t; expression = v != 0.0; }}
const Named kNamed[] = {
    {"mode", [](const Settings& s, const TrainSchedule&) { return (double)(uint32_t)s.mode; },
     [](Settings& s, TrainSchedule&, double v) { s.mode = (RenderMode)std::min((uint32_t)std::max(v, 0.0), 3u); }},
    NCR_SWITCH("useNEE", s.useNEE),
    NCR_SETTING("rhsGlossyBounces", s.rhsGlossyBounces, uint32_t),
    NCR_SWITCH("diffTrainOnGlossy", s.diffTrainOnGlossy),
    NCR_SWITCH("mirrorAlbedo", s.mirrorAlbedo),
    NCR_SETTING("envColorEncoding", s.envColorEncoding, uint32_t),
    NCR_SWITCH("postFilter", s.postFilter),
    NCR_SWITCH("fxaa", s.fxaa),
    NCR_SWITCH("animate", s.animate),
    NCR_SWITCH("cameraAnimated", s.cameraAnimated),
    NCR_SETTING("timeScale", s.timeScale, float),
    NCR_SWITCH("trainSchedule", t.enabled),
    NCR_SETTING("trainLrHot", t.lrHot, float),
    NCR_SETTING("trainLrCold", t.lrCold, float),
    NCR_SETTING("trainEmaHot", t.emaHot, float),
    NCR_SETTING("trainEmaCold", t.emaCold, float),
    NCR_SETTING("trainInitialHotFrames", t.initialHotFrames, uint32_t),
    NCR_SETTING("trainHoldFrames", t.holdFrames, uint32_t),
    NCR_SETTING("trainCoolFrames", t.coolFrames, float),
    NCR_SETTING("trainCameraHeat", t.cameraHeat, float),
    NCR_SETTING("trainSceneHeat", t.sceneHeat, float),
    NCR_SETTING("trainLightMotionHeat", t.lightMotionHeat, float),
    NCR_SETTING("trainObjectMotionHeat", t.objectMotionHeat, float),
};
#undef NCR_SETTING
#undef NCR_SWITCH
}  // namespace

const std::vector<std::string>& Runtime::settingNames()
{
    static const std::vector<std::string> names = [] {
        std::vector<std::string> list;
        for (const Named& n : kNamed) list.push_back(n.name);
        return list;
    }();
    return names;
}

bool Runtime::set(const std::string& name, double value)
{
    for (const Named& n : kNamed)
    {
        if (name != n.name) continue;
        const bool wasAnimated = s->settings.cameraAnimated;
        n.set(s->settings, s->schedule, value);
        // What the bake cannot do stays as it is.
        if (!s->hasPostFilter) s->settings.postFilter = false;
        if (s->bake.track.samples == 0) s->settings.animate = s->settings.cameraAnimated = false;
        if (!s->hasRenderPath) s->settings.mode = s->settings.mode == RenderMode::Idle ? RenderMode::Idle : RenderMode::Train;
        else if (s->trainSectionEnd == 0 && s->settings.mode != RenderMode::Idle) s->settings.mode = RenderMode::Render;
        if (s->settings.cameraAnimated && !wasAnimated) s->reapplySample = true;
        return true;
    }
    return false;
}

double Runtime::get(const std::string& name) const
{
    for (const Named& n : kNamed)
        if (name == n.name) return n.get(s->settings, s->schedule);
    fail("no setting named " + name);
}

void Runtime::writeResults(const std::string& dir) const
{
    size_t written = 0;
    for (auto& [id, texture] : s->bake.textures)
    {
        if (!texture.storage) continue;
        const std::vector<char> data = s->dev.readTexture(texture);
        writeFile(dir + "/texture" + std::to_string(id) + ".bin", data.data(), data.size());
        written++;
    }
    for (auto& [id, buffer] : s->bake.buffers)
    {
        if (s->bake.bufferLoaded[id]) continue;
        writeFile(dir + "/buffer" + std::to_string(id) + ".bin", buffer.data, (size_t)buffer.size);
        written++;
    }
    std::printf("wrote %zu results to %s\n", written, dir.c_str());
}

void Runtime::saveCheckpoint(const std::string& path) const
{
    const State& st = *s;
    if (!st.training) fail("saveCheckpoint: nothing was trained (the grids' masters live on the GPU only when training)");
    const float* masters[4] = {st.primGrid.master.as<float>(), st.clsGrid.master.as<float>(), st.diffNet.master.data(), st.specNet.master.data()};
    const uint16_t* emas[4] = {st.primGrid.inference.as<uint16_t>(), st.clsGrid.inference.as<uint16_t>(), st.diffNet.inference.data(),
                               st.specNet.inference.data()};
    st.checkpoint.save(path, masters, emas);
}

void Runtime::printSummary() const
{
    const State& st = *s;
    if (st.frame == 1)
    {
        // Single frame: the counts must be Falcor's.
        for (const auto& [role, b] : st.batches)
        {
            const uint32_t bakedDiff = (uint32_t)st.bake.value(role + ".diffSize"), bakedSpec = (uint32_t)st.bake.value(role + ".specSize");
            const uint32_t off = (b.diffSize > bakedDiff ? b.diffSize - bakedDiff : bakedDiff - b.diffSize) +
                                 (b.specSize > bakedSpec ? b.specSize - bakedSpec : bakedSpec - b.specSize);
            std::printf("samples (%s): %u diffuse, %u specular (Falcor: %u, %u) %s\n", role.c_str(), b.diffSize, b.specSize, bakedDiff, bakedSpec,
                        off == 0 ? "OK" : ("differs by " + std::to_string(off)).c_str());
        }
    }
    if (st.training)
        std::printf("training: %u steps; schedule now: heat %.4f, hold %u, learning rate %.6g, EMA decay %.6g\n", st.trainSteps, st.schedule.heat,
                    st.schedule.holdLeft, st.schedule.learningRate, st.schedule.emaDecay);
#ifdef __OHOS__
    if (auto* npu = dynamic_cast<const NnrtBackend*>(st.backend.get()))
    {
        std::printf("NPU inference: %u runs, %.1f ms inside RunSync (%.2f ms per tile), %.1f ms around it\n", npu->runs, npu->runMs,
                    npu->runs ? npu->runMs / npu->runs : 0.0, npu->copyMs);
        if (npu->trainRuns)
            std::printf("NPU training: %u runs, %.1f ms inside RunSync (%.2f ms per tile of %u), %.1f ms around it\n", npu->trainRuns,
                        npu->trainRunMs, npu->trainRunMs / npu->trainRuns, st.options.trainTile, npu->trainCopyMs);
    }
#endif
}

struct Display::State
{
    const Device& dev;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    OwnPass pass;
    Buffer pixels;
    uint32_t width = 0, height = 0, bgra = 0;

    explicit State(const Device& device) : dev(device) {}
};

Display::Display(const Device& dev, const std::string& shaderDir, const Texture& hdr, bool bgra) : s(std::make_unique<State>(dev))
{
    s->width = hdr.width;
    s->height = hdr.height;
    s->bgra = bgra ? 1 : 0;
    s->pool = makeOwnPool(dev, 1, 1, 1);
    s->pass.create(dev, s->pool, shaderDir + "/Display.spv", {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(DisplayPush), 64);
    s->pixels = dev.makeBuffer((VkDeviceSize)s->width * s->height * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    s->pass.bindImage(dev, 0, hdr.storageView);
    s->pass.bind(dev, 1, s->pixels);
}

Display::~Display()
{
    vkDeviceWaitIdle(s->dev.device);
    s->pass.destroy(s->dev);
    vkDestroyDescriptorPool(s->dev.device, s->pool, nullptr);
    s->dev.destroy(s->pixels);
}

void Display::record(VkCommandBuffer cb) const
{
    s->pass.dispatch(cb, DisplayPush{s->width, s->height, s->bgra}, s->width * s->height);
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

const Buffer& Display::pixels() const { return s->pixels; }
uint32_t Display::width() const { return s->width; }
uint32_t Display::height() const { return s->height; }
}  // namespace replay

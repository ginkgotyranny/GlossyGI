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
    if (target >= heat && target > 0.0f)
    {
        heat = target;
        holdLeft = std::max(holdLeft > 0 ? holdLeft - 1 : 0u, holdFrames);
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
struct TargetPush { uint32_t count, rowBase, tileShift, slots; };
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
            std::printf("model: %s (training copies rebuilt from the masters, as Falcor loads a checkpoint)\n", model.c_str());
        }
        diffNet.init(false, checkpoint.diffMaster, checkpoint.diffEma, checkpoint.diffWorking);
        specNet.init(true, checkpoint.specMaster, checkpoint.specEma, checkpoint.specWorking);

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
            uint32_t size, tiles, rows;   // samples, tiles of the training graph, rows the loss is normalised by
            TrainNet* net;
            Mlp* mlp;
            uint32_t inputBase, targetBase;
        };
        std::vector<Job> jobs;
        // tiny-cuda-nn runs on the batch padded to 256 rows and divides the loss by that number. Here
        // the rows beyond the batch have weight zero instead of whatever an earlier frame left there.
        if (b.needSpec && b.specSize > 0)
            jobs.push_back({true, b.specSize, (b.specSize + trainTile - 1) / trainTile, padUp(b.specSize, cfg::kBatchAlign), &t.spec, &specNet, 0, 0});
        if (b.needDiff && b.diffSize > 0)
            jobs.push_back({false, b.diffSize, (b.diffSize + trainTile - 1) / trainTile, padUp(b.diffSize, cfg::kBatchAlign), &t.diff, &diffNet,
                            b.diffInputOffset() * cfg::kDiffInputDim, b.diffOutputOffset()});
        if (jobs.empty()) return;
        const bool dump = !options.dumpTrainDir.empty() && trainSteps == 0;

        VkCommandBuffer c = record("training inputs and targets (" + b.role + ")");
        for (const Job& j : jobs)
        {
            j.net->encode.dispatch(c, EncodePush{j.size, j.inputBase, trainShift}, j.size);
            j.net->targets.dispatch(c, TargetPush{j.size, j.targetBase, trainShift, j.tiles * trainTile}, j.tiles * trainTile);
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
            // Every tile normalised its loss by its own size: bring the sum to the batch.
            const float toBatch = (float)trainTile / (float)j.rows;
            for (float& g : j.net->weightGradient) g *= toBatch;
            tileNote += (tileNote.empty() ? "" : " + ") + std::to_string(j.tiles);
        }
        timings.push_back({"MLP gradients (" + b.role + "), " + tileNote + " tiles", msSince(t0)});

        c = record("grid gradients (" + b.role + ")");
        bool anySpec = false;
        for (const Job& j : jobs)
        {
            j.net->scatter.dispatch(c, ScatterPush{j.size, trainShift, adam::kGridGradientScale * (float)trainTile / (float)j.rows}, j.size);
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
            Checkpoint::Header header = Checkpoint::expectedHeader();
            std::memcpy(header.magic, "NRWK", 4);
            header.version = 1;
            f.write(reinterpret_cast<const char*>(&header), sizeof(header));
            for (int i = 0; i < 4; i++) f.write(reinterpret_cast<const char*>(working[i]), (std::streamsize)header.counts[i] * 2);
            std::ofstream info(dir + "/train.txt");
            info << "tile " << trainTile << "\nlearning_rate " << learningRate << "\nema_decay " << emaDecay << "\n";
            for (const Job& j : jobs) info << (j.spec ? "spec" : "diff") << " " << j.size << " " << j.tiles << " " << j.rows << "\n";
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
        if (frame == 0) return;
        const std::string block = "g" + call.pass;
        auto set = [&](const char* member, uint32_t value) {
            if (bake.hasUniform(call, block, member)) bake.setUniform(call, block, member, value);
        };
        set("batchCount", bake.frameCount + frame);
        set("frameCount", bake.frameCount + frame);
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
    }

    void runFrame()
    {
        timings.clear();
        // Falcor updates the schedule before the frame's training step; the baked frame keeps its values.
        if (useSchedule && frame > 0) schedule.advance(0);   // the camera and the scene stay as baked: no events
        for (const Step& step : bake.program)
        {
            if (step.isCall)
            {
                Call& call = bake.calls.at(step.step);
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
                bake.dispatch(record(call.pass), call, dims[0], dims[1], dims[2]);
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
            else
            {
                fail("unknown host step " + what);
            }
        }
        flush();
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

void Runtime::frame() { s->runFrame(); }
uint32_t Runtime::framesDone() const { return s->frame; }
uint32_t Runtime::width() const { return s->bake.width; }
uint32_t Runtime::height() const { return s->bake.height; }
uint32_t Runtime::bakedFrame() const { return s->bake.frameCount; }
double Runtime::lastFrameMs() const { return s->lastFrameMs; }
const std::vector<std::pair<std::string, double>>& Runtime::lastTimings() const { return s->timings; }
const Texture& Runtime::outputImage() const { return s->bake.textures.at(s->outputTexture); }
bool Runtime::trains() const { return s->training; }
uint32_t Runtime::trainSteps() const { return s->trainSteps; }
const TrainSchedule& Runtime::schedule() const { return s->schedule; }

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
    Checkpoint::save(path, masters, emas);
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

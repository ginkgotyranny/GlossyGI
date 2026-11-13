// NCR replayer: runs frames of the NeuralRadiosity pass from a Falcor bake, without Falcor.
//
//   ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [options]
//
// The bake is a recorded frame: the compute-pass calls in the order Falcor executed them, each with
// its bindings and uniform values, and the steps Falcor did on the host in between (prefix sums,
// network inference, the training step). The replayer executes that program with the same SPIR-V and
// the same scene data and writes every image and ray-batch buffer the passes produced to the output
// directory, where ohos/tools/replay_compare.py compares them with what Falcor produced in that frame.
//
// The networks, --networks:
//   tiles  (default) the way the phone runs them: a GPU pass looks up the hash grids and lays the
//          samples out in NPU tiles (shaders/EncodeInputs), the MLPs run on those tiles, a GPU pass
//          writes the outputs back as rows (shaders/DecodeOutputs). The MLPs run on the CPU.
//   npu    the same with the MLPs on the NPU through NNRt (OpenHarmony only; --models is the folder
//          with the .om files from ohos/npu/export_models.py).
//   rows   the float32 reference on the CPU, lookups included, straight on the sample rows.
//          Inference only: a recorded training step is skipped.
//
// A training step (a frame baked in a training mode has one), with tiles or npu:
//   GPU   EncodeInputs with the training grids, EncodeTargets          -> input and target tiles
//   NPU   the training graph, tile by tile                             -> dL/dw, dL/d(input)
//   GPU   ScatterGradients into the grids, GridOptimizer (Adam + EMA)  -> the grids' new weights
//   CPU   Adam + EMA for the two MLPs                                  -> the MLPs' new weights
// The weights after the last frame are written as <output dir>/trained.ckpt, which Falcor loads.
//
// Options:
//   --frames N        replay the frame N times, as consecutive frames (default 1). The camera and the
//                     scene stay as baked; the frame counter advances, so every frame draws new samples.
//   --train-tile N    samples per tile of the training graph (default 4096; the NPU needs
//                     ncr_<net>_train_<N>_fp16.om)
//   --lr X, --ema X   learning rate and EMA decay (default: what Falcor used in the baked frame)
//   --no-train        skip recorded training steps
//   --loss N          every N frames print the training loss of the first tile, computed on the CPU
//   --dump-train      write the tensors of the first training step for ohos/tools/train_check.py
//   --shaders <dir>   the replayer's own SPIR-V (default: shaders next to the executable)
//   --models <dir>    the NPU models
//   --per-pass        submit every call on its own and print the time of each (slower overall)
//
// The prefix sums still run on the CPU.
// The same source builds for Windows (build_pc.bat) and for OpenHarmony (run_device.sh).
#include "replay_bake.h"
#include "replay_nn.h"
#include "replay_nnrt.h"
#include "replay_own.h"

#include <memory>

using namespace replay;

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

void writeFile(const std::string& path, const void* data, size_t size)
{
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), (std::streamsize)size);
    if (!f.good())
    {
        std::printf("FAIL cannot write %s\n", path.c_str());
        std::exit(2);
    }
}

uint32_t log2Exact(uint32_t value)
{
    uint32_t shift = 0;
    while ((1u << shift) < value) shift++;
    if ((1u << shift) != value)
    {
        std::printf("FAIL %u is not a power of two\n", value);
        std::exit(2);
    }
    return shift;
}

/// NeuralRadiosity.cpp: how many specular samples ConeTrace and SpecPrimary are dispatched for when
/// this frame's count is still on the GPU. The previous frame's count plus headroom.
uint32_t specDispatchItems(uint32_t previousSpecCount, uint32_t batchSize)
{
    return std::min(batchSize, std::max(8192u, previousSpecCount + previousSpecCount / 8));
}

// Push constants of the replayer's own passes (ohos/replay/shaders).
struct EncodePush { uint32_t count, rowBase, tileShift; };
struct DecodePush { uint32_t count, rowBase; };
struct TargetPush { uint32_t count, rowBase, tileShift, slots; };
struct ScatterPush { uint32_t count, tileShift; float gradScale; };
struct OptimizerPush { uint32_t count; float learningRate, gradToFloat, emaDecay, emaDebiasOld, emaDebiasNew; };

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
}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: MSVC rejects line buffering with a null buffer
    if (argc < 4)
    {
        std::printf("usage: ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--networks rows|tiles|npu] [--frames N] [--train-tile N]\n"
                    "                  [--lr X] [--ema X] [--no-train] [--loss N] [--dump-train] [--shaders <dir>] [--models <dir>] [--per-pass]\n");
        return 2;
    }
    const std::string bakeDir = argv[1], checkpointPath = argv[2], outDir = argv[3];
    const std::string executable = argv[0];
    const size_t slash = executable.find_last_of("/\\");
    std::string networksMode = "tiles", modelDir = ".";
    std::string shaderDir = (slash == std::string::npos ? std::string(".") : executable.substr(0, slash)) + "/shaders";
    bool perPass = false, noTrain = false, dumpTrain = false;
    uint32_t frames = 1, trainTile = 4096, lossInterval = 0;
    float learningRate = 0.0f, emaDecay = 0.0f;
    for (int i = 4; i < argc; i++)
    {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--per-pass") perPass = true;
        else if (arg == "--no-train") noTrain = true;
        else if (arg == "--dump-train") dumpTrain = true;
        else if (arg == "--networks" && hasValue) networksMode = argv[++i];
        else if (arg == "--shaders" && hasValue) shaderDir = argv[++i];
        else if (arg == "--models" && hasValue) modelDir = argv[++i];
        else if (arg == "--frames" && hasValue) frames = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--train-tile" && hasValue) trainTile = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--loss" && hasValue) lossInterval = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--lr" && hasValue) learningRate = std::stof(argv[++i]);
        else if (arg == "--ema" && hasValue) emaDecay = std::stof(argv[++i]);
        else
        {
            std::printf("FAIL unknown option %s\n", arg.c_str());
            return 2;
        }
    }
    const uint32_t trainShift = log2Exact(trainTile), inferShift = log2Exact(cfg::kTile);

    Device dev;
    dev.create();
    std::printf("device: %s\n", dev.name);

    auto start = std::chrono::steady_clock::now();
    Bake bake(dev, bakeDir);
    bake.load();
    std::printf("bake %ux%u (Falcor frame %u) loaded in %.0f ms\n", bake.width, bake.height, bake.frameCount, msSince(start));

    Checkpoint checkpoint;
    checkpoint.load(checkpointPath);
    Mlp diffNet, specNet;
    diffNet.init(false, checkpoint.diffMaster, checkpoint.diffEma);
    specNet.init(true, checkpoint.specMaster, checkpoint.specEma);
    const CpuNetworks cpuNetworks(checkpoint, diffNet, specNet);

    std::map<std::string, Batch> batches;
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

    const bool useTiles = networksMode != "rows";
    bool training = false;
    for (const Step& step : bake.program) training = training || (!step.isCall && step.host[0] == "train");
    if (training && !useTiles) std::printf("note: --networks rows does not train; the recorded training step is skipped\n");
    training = training && useTiles && !noTrain;
    if (learningRate == 0.0f) learningRate = (float)bake.value("train.learningRate", 5e-3);
    if (emaDecay == 0.0f) emaDecay = (float)bake.value("train.emaDecay", 0.95);

    // The network path of the phone: grids and tiles on the GPU, the MLPs behind a backend.
    std::unique_ptr<NetworkBackend> backend;
    Grid primGrid, clsGrid;
    std::map<std::string, TilePath> tilePaths;
    std::map<std::string, TrainPath> trainPaths;
    VkDescriptorPool ownPool = VK_NULL_HANDLE;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    auto zeroed = [&](VkDeviceSize bytes) {
        Buffer b = dev.makeBuffer(bytes, storage, true);
        std::memset(b.data, 0, (size_t)bytes);
        return b;
    };
    if (useTiles)
    {
        ownPool = makeOwnPool(dev, 48, 256);
        auto createGrid = [&](Grid& g, const std::vector<uint16_t>& ema, const std::vector<float>& master) {
            g.elements = (uint32_t)ema.size();
            g.inference = dev.makeBuffer(ema.size() * 2, storage, true);
            std::memcpy(g.inference.data, ema.data(), ema.size() * 2);
            if (!training) return;
            g.master = dev.makeBuffer(master.size() * 4, storage, true);
            std::memcpy(g.master.data, master.data(), master.size() * 4);
            // The training copy: the masters in fp16, as loading a checkpoint rebuilds it in Model.cu.
            std::vector<uint16_t> half(master.size());
            for (size_t i = 0; i < master.size(); i++) half[i] = floatToHalf(master[i]);
            g.training = dev.makeBuffer(half.size() * 2, storage, true);
            std::memcpy(g.training.data, half.data(), half.size() * 2);
            g.firstMoment = zeroed(master.size() * 4);
            g.secondMoment = zeroed(master.size() * 4);
            g.entrySteps = zeroed(master.size() * 4);
            g.gradient = zeroed(master.size() * 4);
            g.optimizer.create(dev, ownPool, shaderDir + "/GridOptimizer.spv", 7, sizeof(OptimizerPush), 256);
            const Buffer* bound[7] = {&g.gradient, &g.master, &g.firstMoment, &g.secondMoment, &g.entrySteps, &g.training, &g.inference};
            for (uint32_t i = 0; i < 7; i++) g.optimizer.bind(dev, i, *bound[i]);
        };
        createGrid(primGrid, checkpoint.primGrid, checkpoint.primGridMaster);
        createGrid(clsGrid, checkpoint.clsGrid, checkpoint.clsGridMaster);
        if (networksMode == "npu")
        {
#ifdef __OHOS__
            backend = std::make_unique<NnrtBackend>(diffNet, specNet, modelDir);
#else
            std::printf("FAIL --networks npu needs the OpenHarmony build\n");
            return 2;
#endif
        }
        else if (networksMode == "tiles")
        {
            backend = std::make_unique<CpuTileBackend>(cpuNetworks);
        }
        else
        {
            std::printf("FAIL unknown --networks %s\n", networksMode.c_str());
            return 2;
        }
    }
    std::printf("networks: %s\n", useTiles ? backend->name() : "CPU (float32) on the sample rows, lookups included");
    if (training)
        std::printf("training: tiles of %u samples, learning rate %g, EMA decay %g\n", trainTile, learningRate, emaDecay);

    // Tile buffers and the passes around them for one batch, created when the batch first needs them.
    auto tilePath = [&](const Batch& b) -> TilePath& {
        auto it = tilePaths.find(b.role);
        if (it != tilePaths.end()) return it->second;
        TilePath& t = tilePaths[b.role];
        const uint32_t maxTiles = (b.size + cfg::kTile - 1) / cfg::kTile;
        const VkDeviceSize tile = (VkDeviceSize)cfg::kTile * 2;
        auto create = [&](bool spec, Buffer& x, Buffer& y, OwnPass& encode, OwnPass& decode) {
            x = zeroed(maxTiles * (spec ? cfg::kSpecInputDim : cfg::kDiffInputDim) * tile);
            y = dev.makeBuffer(maxTiles * cfg::kOutputChannels * tile, storage, true);
            encode.create(dev, ownPool, shaderDir + (spec ? "/EncodeSpec.spv" : "/EncodeDiff.spv"), 4, sizeof(EncodePush), 64);
            decode.create(dev, ownPool, shaderDir + "/DecodeOutputs.spv", 2, sizeof(DecodePush), 64);
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
    };

    // The same for training: input, target and gradient tiles of the training graph's size.
    auto trainPath = [&](const Batch& b) -> TrainPath& {
        auto it = trainPaths.find(b.role);
        if (it != trainPaths.end()) return it->second;
        TrainPath& t = trainPaths[b.role];
        const uint32_t maxTiles = (b.size + trainTile - 1) / trainTile;
        const VkDeviceSize tile = (VkDeviceSize)trainTile * 2;
        auto create = [&](bool spec, TrainNet& n) {
            const uint32_t channels = spec ? cfg::kSpecInputDim : cfg::kDiffInputDim;
            n.x = zeroed(maxTiles * channels * tile);
            n.target = zeroed(maxTiles * cfg::kOutputChannels * tile);
            n.gradient = zeroed(maxTiles * channels * tile);
            n.weightGradient.resize((spec ? specNet : diffNet).master.size());
            n.encode.create(dev, ownPool, shaderDir + (spec ? "/EncodeSpec.spv" : "/EncodeDiff.spv"), 4, sizeof(EncodePush), 64);
            n.targets.create(dev, ownPool, shaderDir + "/EncodeTargets.spv", 2, sizeof(TargetPush), 64);
            n.scatter.create(dev, ownPool, shaderDir + (spec ? "/ScatterSpec.spv" : "/ScatterDiff.spv"), spec ? 4 : 3, sizeof(ScatterPush), 64);
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
    };

    // Calls are recorded into one command buffer until a host step needs their results.
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::string pendingNames;
    std::vector<std::pair<std::string, double>> timings;   // of the frame being replayed
    auto flush = [&]() {
        if (cb == VK_NULL_HANDLE) return;
        timings.push_back({pendingNames, dev.submit(cb)});
        cb = VK_NULL_HANDLE;
        pendingNames.clear();
    };
    auto record = [&](const std::string& name) -> VkCommandBuffer {
        if (cb == VK_NULL_HANDLE) cb = dev.begin();
        pendingNames += (pendingNames.empty() ? "" : " + ") + name;
        return cb;
    };

    /// Falcor reads the compacted counts back where it syncs with CUDA.
    auto readCounts = [&](Batch& b) {
        const uint32_t* counts = bake.buffer(b.role + ".counts").as<uint32_t>();
        b.diffSize = b.needDiff ? counts[0] : 0;
        b.specSize = b.needSpec ? counts[1] : 0;
    };

    // Network inference of one batch: the diffuse rows through DiffNet, the specular rows through SpecNet.
    auto infer = [&](Batch& b) {
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
            if (b.needSpec) backend->infer(true, t.specX.as<uint16_t>(), t.specY.as<uint16_t>(), specTiles);
            if (b.needDiff) backend->infer(false, t.diffX.as<uint16_t>(), t.diffY.as<uint16_t>(), diffTiles);
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
    };

    // One training step on a batch whose color buffers hold the targets (NRModel::train).
    uint32_t trainSteps = 0;
    auto train = [&](Batch& b, uint32_t frame) {
        readCounts(b);
        TrainPath& t = trainPath(b);
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
        const bool dump = dumpTrain && trainSteps == 0;

        VkCommandBuffer c = record("training inputs and targets (" + b.role + ")");
        for (const Job& j : jobs)
        {
            j.net->encode.dispatch(c, EncodePush{j.size, j.inputBase, trainShift}, j.size);
            j.net->targets.dispatch(c, TargetPush{j.size, j.targetBase, trainShift, j.tiles * trainTile}, j.tiles * trainTile);
        }
        flush();

        if (lossInterval != 0 && frame % lossInterval == 0)
        {
            std::printf("frame %u loss:", frame);
            for (const Job& j : jobs)
            {
                const double sum = cpuNetworks.trainTiles(j.spec, trainTile, 1, j.net->x.as<uint16_t>(), j.net->target.as<uint16_t>(), nullptr, nullptr);
                std::printf(" %s %.4f", j.spec ? "specular" : "diffuse", sum / ((double)std::min(j.size, trainTile) * cfg::kOutputDim));
            }
            std::printf("\n");
        }

        auto t0 = std::chrono::steady_clock::now();
        std::string tileNote;
        for (const Job& j : jobs)
        {
            std::fill(j.net->weightGradient.begin(), j.net->weightGradient.end(), 0.0f);
            backend->train(j.spec, trainTile, j.tiles, j.net->x.as<uint16_t>(), j.net->target.as<uint16_t>(), j.net->weightGradient.data(),
                           j.net->gradient.as<uint16_t>());
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
            for (const Job& j : jobs)
            {
                const std::string prefix = outDir + "/train_" + (j.spec ? "spec" : "diff");
                const size_t channels = j.mlp->inputDim;
                writeFile(prefix + "_x.bin", j.net->x.data, (size_t)j.tiles * channels * trainTile * 2);
                writeFile(prefix + "_target.bin", j.net->target.data, (size_t)j.tiles * cfg::kOutputChannels * trainTile * 2);
                writeFile(prefix + "_gx.bin", j.net->gradient.data, (size_t)j.tiles * channels * trainTile * 2);
                writeFile(prefix + "_gw.bin", j.net->weightGradient.data(), j.net->weightGradient.size() * 4);
            }
            writeFile(outDir + "/train_prim_grad.bin", primGrid.gradient.data, (size_t)primGrid.gradient.size);
            writeFile(outDir + "/train_cls_grad.bin", clsGrid.gradient.data, (size_t)clsGrid.gradient.size);
            std::ofstream info(outDir + "/train.txt");
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
    };

    // What Falcor sets anew every frame in a call's uniform block, for the frames after the baked
    // one (which runs with the values Falcor had): the frame counter that seeds the samplers, and
    // what depends on the sample counts the host knows at that point (NeuralRadiosity.cpp).
    auto prepare = [&](Call& call, uint32_t frame, uint32_t dims[3]) {
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
    };

    // The recorded frame, `frames` times.
    std::vector<double> frameMs;
    for (uint32_t frame = 0; frame < frames; frame++)
    {
        timings.clear();
        for (const Step& step : bake.program)
        {
            if (step.isCall)
            {
                Call& call = bake.calls.at(step.step);
                uint32_t dims[3] = {call.dims[0], call.dims[1], call.dims[2]};
                prepare(call, frame, dims);
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
                if (perPass) flush();
                continue;
            }
            flush();
            const std::string& what = step.host[0];
            const auto t0 = std::chrono::steady_clock::now();
            if (what == "clearcounts")
            {
                std::memset(bake.buffer(step.host[1] + ".counts").data, 0, 2 * sizeof(uint32_t));
            }
            else if (what == "prefixsum")
            {
                // Mask -> compacted index, and the count (NeuralRadiosity::computeBatchCounts).
                const bool spec = step.host[2] == "spec";
                const uint32_t count = (uint32_t)std::stoul(step.host[3]);
                uint32_t* counts = bake.buffer(step.host[1] + ".counts").as<uint32_t>();
                counts[spec ? 1 : 0] = prefixSum(bake.buffer(step.host[1] + (spec ? ".specIndex" : ".diffIndex")).as<uint32_t>(), count);
                timings.push_back({"prefix sum on the CPU (" + step.host[1] + ", " + step.host[2] + ")", msSince(t0)});
            }
            else if (what == "infer")
            {
                infer(batches.at(step.host[1]));
            }
            else if (what == "train")
            {
                if (training) train(batches.at(step.host[1]), frame);
            }
            else
            {
                std::printf("FAIL unknown host step %s\n", what.c_str());
                return 2;
            }
        }
        flush();
        double total = 0.0;
        for (const auto& [name, ms] : timings) total += ms;
        frameMs.push_back(total);
    }

    for (const auto& [name, ms] : timings) std::printf("  %8.1f ms  %s\n", ms, name.c_str());
    std::printf("frame: %.1f ms in %zu segments\n", frameMs.back(), timings.size());
    if (frames > 1)
    {
        double sum = 0.0;
        for (size_t i = 1; i < frameMs.size(); i++) sum += frameMs[i];
        std::printf("frames: %u, the first %.1f ms, the others %.1f ms on average\n", frames, frameMs[0], sum / (double)(frameMs.size() - 1));
    }
    else
    {
        for (const auto& [role, b] : batches)
        {
            const uint32_t bakedDiff = (uint32_t)bake.value(role + ".diffSize"), bakedSpec = (uint32_t)bake.value(role + ".specSize");
            const uint32_t off = (b.diffSize > bakedDiff ? b.diffSize - bakedDiff : bakedDiff - b.diffSize) +
                                 (b.specSize > bakedSpec ? b.specSize - bakedSpec : bakedSpec - b.specSize);
            std::printf("samples (%s): %u diffuse, %u specular (Falcor: %u, %u) %s\n", role.c_str(), b.diffSize, b.specSize, bakedDiff, bakedSpec,
                        off == 0 ? "OK" : ("differs by " + std::to_string(off)).c_str());
        }
    }
#ifdef __OHOS__
    if (auto* npu = dynamic_cast<NnrtBackend*>(backend.get()))
    {
        std::printf("NPU inference: %u runs, %.1f ms inside RunSync (%.2f ms per tile), %.1f ms copying tiles\n", npu->runs, npu->runMs,
                    npu->runs ? npu->runMs / npu->runs : 0.0, npu->copyMs);
        if (npu->trainRuns)
            std::printf("NPU training: %u runs, %.1f ms inside RunSync (%.2f ms per tile of %u), %.1f ms copying tiles and gradients\n", npu->trainRuns,
                        npu->trainRunMs, npu->trainRunMs / npu->trainRuns, trainTile, npu->trainCopyMs);
    }
#endif

    // Everything the passes wrote, for the comparison with Falcor.
    size_t written = 0;
    for (auto& [id, texture] : bake.textures)
    {
        if (!texture.storage) continue;
        const std::vector<char> data = dev.readTexture(texture);
        writeFile(outDir + "/texture" + std::to_string(id) + ".bin", data.data(), data.size());
        written++;
    }
    for (auto& [id, buffer] : bake.buffers)
    {
        if (bake.bufferLoaded[id]) continue;
        writeFile(outDir + "/buffer" + std::to_string(id) + ".bin", buffer.data, (size_t)buffer.size);
        written++;
    }
    std::printf("wrote %zu results to %s\n", written, outDir.c_str());
    if (trainSteps > 0)
    {
        const float* masters[4] = {primGrid.master.as<float>(), clsGrid.master.as<float>(), diffNet.master.data(), specNet.master.data()};
        const uint16_t* emas[4] = {primGrid.inference.as<uint16_t>(), clsGrid.inference.as<uint16_t>(), diffNet.inference.data(), specNet.inference.data()};
        Checkpoint::save(outDir + "/trained.ckpt", masters, emas);
        std::printf("training: %u steps, weights written to %s/trained.ckpt\n", trainSteps, outDir.c_str());
    }
    vkDeviceWaitIdle(dev.device);
    return 0;
}

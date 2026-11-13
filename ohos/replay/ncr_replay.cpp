// NCR replayer: runs a frame of the NeuralRadiosity pass from a Falcor bake, without Falcor.
//
//   ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--networks rows|tiles|npu] [--shaders <dir>]
//              [--models <dir>] [--per-pass]
//
// The bake is a recorded frame: the compute-pass calls in the order Falcor executed them, each with
// its bindings and uniform values, and the steps Falcor did on the host in between (prefix sums,
// network inference, training). The replayer executes that program with the same SPIR-V and the same
// scene data and writes every image and ray-batch buffer the passes produced to the output directory,
// where ohos/tools/replay_compare.py compares them with what Falcor produced in that frame.
//
// The networks, --networks:
//   tiles  (default) the way the phone runs them: a GPU pass looks up the hash grids and lays the
//          samples out in NPU tiles (shaders/EncodeInputs), the MLPs run on those tiles, a GPU pass
//          writes the outputs back as rows (shaders/DecodeOutputs). The MLPs run on the CPU.
//   npu    the same with the MLPs on the NPU through NNRt (OpenHarmony only; --models is the folder
//          with ncr_{diff,spec}_inf_256x256_fp16.om from ohos/npu/export_models.py).
//   rows   the float32 reference on the CPU, lookups included, straight on the sample rows.
// --shaders is the folder with the replayer's own SPIR-V (default: shaders next to the executable).
// --per-pass submits every call on its own and prints the time of each (slower overall).
//
// The prefix sums still run on the CPU; a training step in the recorded frame is not replayed yet.
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

struct RowsPush { uint32_t count, rowBase; };

/// A ray batch (render, lhs, rhs) as the host sees it.
struct Batch
{
    std::string role;
    uint32_t size = 0;
    bool needDiff = false, needSpec = false, sharedIO = false;
    uint32_t diffSize = 0, specSize = 0;   // compacted counts, known after the counts are read

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
}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: MSVC rejects line buffering with a null buffer
    if (argc < 4)
    {
        std::printf("usage: ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--networks rows|tiles|npu] [--shaders <dir>] [--models <dir>] [--per-pass]\n");
        return 2;
    }
    const std::string bakeDir = argv[1], checkpointPath = argv[2], outDir = argv[3];
    const std::string executable = argv[0];
    const size_t slash = executable.find_last_of("/\\");
    std::string networksMode = "tiles", modelDir = ".";
    std::string shaderDir = (slash == std::string::npos ? std::string(".") : executable.substr(0, slash)) + "/shaders";
    bool perPass = false;
    for (int i = 4; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--per-pass") perPass = true;
        else if (arg == "--networks" && i + 1 < argc) networksMode = argv[++i];
        else if (arg == "--shaders" && i + 1 < argc) shaderDir = argv[++i];
        else if (arg == "--models" && i + 1 < argc) modelDir = argv[++i];
    }

    Device dev;
    dev.create();
    std::printf("device: %s\n", dev.name);

    auto start = std::chrono::steady_clock::now();
    Bake bake(dev, bakeDir);
    bake.load();
    std::printf("bake %ux%u (Falcor frame %u) loaded in %.0f ms\n", bake.width, bake.height, bake.frameCount, msSince(start));

    Checkpoint checkpoint;
    checkpoint.load(checkpointPath);
    const CpuNetworks rowNetworks(checkpoint);

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

    // The network path of the phone: grids and tiles on the GPU, the MLPs behind a backend.
    const bool useTiles = networksMode != "rows";
    std::unique_ptr<NetworkBackend> backend;
    Buffer primGrid, clsGrid;
    std::map<std::string, TilePath> tilePaths;
    VkDescriptorPool ownPool = VK_NULL_HANDLE;
    if (useTiles)
    {
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        primGrid = dev.makeBuffer(checkpoint.primGrid.size() * 2, usage, true);
        clsGrid = dev.makeBuffer(checkpoint.clsGrid.size() * 2, usage, true);
        std::memcpy(primGrid.data, checkpoint.primGrid.data(), checkpoint.primGrid.size() * 2);
        std::memcpy(clsGrid.data, checkpoint.clsGrid.data(), checkpoint.clsGrid.size() * 2);
        ownPool = makeOwnPool(dev, 16, 64);
        if (networksMode == "npu")
        {
#ifdef __OHOS__
            backend = std::make_unique<NnrtBackend>(checkpoint, modelDir);
#else
            std::printf("FAIL --networks npu needs the OpenHarmony build\n");
            return 2;
#endif
        }
        else
        {
            backend = std::make_unique<CpuTileBackend>(checkpoint);
        }
    }
    std::printf("networks: %s\n", useTiles ? backend->name() : "CPU (float32) on the sample rows, lookups included");

    // Tile buffers and the passes around them for one batch, created when the batch first needs them.
    auto tilePath = [&](const Batch& b) -> TilePath& {
        auto it = tilePaths.find(b.role);
        if (it != tilePaths.end()) return it->second;
        TilePath& t = tilePaths[b.role];
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        const uint32_t maxTiles = (b.size + cfg::kTile - 1) / cfg::kTile;
        const VkDeviceSize tile = (VkDeviceSize)cfg::kTile * 2;
        if (b.needSpec)
        {
            t.specX = dev.makeBuffer(maxTiles * cfg::kSpecInputDim * tile, usage, true);
            t.specY = dev.makeBuffer(maxTiles * cfg::kOutputChannels * tile, usage, true);
            std::memset(t.specX.data, 0, (size_t)t.specX.size);
            t.encodeSpec.create(dev, ownPool, shaderDir + "/EncodeSpec.spv", 4, sizeof(RowsPush), 64);
            t.decodeSpec.create(dev, ownPool, shaderDir + "/DecodeOutputs.spv", 2, sizeof(RowsPush), 64);
            t.encodeSpec.bind(dev, 0, bake.buffer(b.role + ".specInput"));
            t.encodeSpec.bind(dev, 1, primGrid);
            t.encodeSpec.bind(dev, 2, clsGrid);
            t.encodeSpec.bind(dev, 3, t.specX);
            t.decodeSpec.bind(dev, 0, t.specY);
            t.decodeSpec.bind(dev, 1, bake.buffer(b.role + ".specColor"));
        }
        if (b.needDiff)
        {
            t.diffX = dev.makeBuffer(maxTiles * cfg::kDiffInputDim * tile, usage, true);
            t.diffY = dev.makeBuffer(maxTiles * cfg::kOutputChannels * tile, usage, true);
            std::memset(t.diffX.data, 0, (size_t)t.diffX.size);
            t.encodeDiff.create(dev, ownPool, shaderDir + "/EncodeDiff.spv", 4, sizeof(RowsPush), 64);
            t.decodeDiff.create(dev, ownPool, shaderDir + "/DecodeOutputs.spv", 2, sizeof(RowsPush), 64);
            t.encodeDiff.bind(dev, 0, bake.buffer(b.role + ".diffInput"));
            t.encodeDiff.bind(dev, 1, primGrid);
            t.encodeDiff.bind(dev, 2, clsGrid);
            t.encodeDiff.bind(dev, 3, t.diffX);
            t.decodeDiff.bind(dev, 0, t.diffY);
            t.decodeDiff.bind(dev, 1, bake.buffer(b.role + ".diffColor"));
        }
        return t;
    };

    // Calls are recorded into one command buffer until a host step needs their results.
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::string pendingNames;
    std::vector<std::pair<std::string, double>> timings;
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

    // Network inference of one batch: the diffuse rows through DiffNet, the specular rows through SpecNet.
    auto infer = [&](Batch& b) {
        // Falcor reads the compacted counts back here, at its CUDA sync.
        const uint32_t* counts = bake.buffer(b.role + ".counts").as<uint32_t>();
        b.diffSize = b.needDiff ? counts[0] : 0;
        b.specSize = b.needSpec ? counts[1] : 0;
        if (useTiles)
        {
            TilePath& t = tilePath(b);
            VkCommandBuffer c = record("grid lookups to tiles (" + b.role + ")");
            if (b.needSpec) t.encodeSpec.dispatch(c, RowsPush{b.specSize, 0}, b.specSize);
            if (b.needDiff) t.encodeDiff.dispatch(c, RowsPush{b.diffSize, b.diffInputOffset() * cfg::kDiffInputDim}, b.diffSize);
            flush();
            const auto t0 = std::chrono::steady_clock::now();
            const uint32_t specTiles = (b.specSize + cfg::kTile - 1) / cfg::kTile, diffTiles = (b.diffSize + cfg::kTile - 1) / cfg::kTile;
            if (b.needSpec) backend->infer(true, t.specX.as<uint16_t>(), t.specY.as<uint16_t>(), specTiles);
            if (b.needDiff) backend->infer(false, t.diffX.as<uint16_t>(), t.diffY.as<uint16_t>(), diffTiles);
            timings.push_back({"MLPs (" + b.role + "), " + std::to_string(specTiles) + " + " + std::to_string(diffTiles) + " tiles", msSince(t0)});
            c = record("outputs to rows (" + b.role + ")");
            if (b.needSpec) t.decodeSpec.dispatch(c, RowsPush{b.specSize, 0}, b.specSize);
            if (b.needDiff) t.decodeDiff.dispatch(c, RowsPush{b.diffSize, b.diffOutputOffset()}, b.diffSize);
        }
        else
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (b.needSpec)
                rowNetworks.inferSpec(bake.buffer(b.role + ".specInput").as<uint16_t>(), b.specSize, bake.buffer(b.role + ".specColor").as<float>());
            if (b.needDiff)
                rowNetworks.inferDiff(bake.buffer(b.role + ".diffInput").as<uint16_t>() + (size_t)b.diffInputOffset() * cfg::kDiffInputDim, b.diffSize,
                                      bake.buffer(b.role + ".diffColor").as<float>() + (size_t)b.diffOutputOffset() * cfg::kOutputDim);
            timings.push_back({"networks on the CPU (" + b.role + ")", msSince(t0)});
        }
    };

    // The recorded frame.
    for (const Step& step : bake.program)
    {
        if (step.isCall)
        {
            Call& call = bake.calls.at(step.step);
            uint32_t dims[3] = {call.dims[0], call.dims[1], call.dims[2]};
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
            std::printf("note: the training step of the recorded frame is not replayed\n");
        }
        else
        {
            std::printf("FAIL unknown host step %s\n", what.c_str());
            return 2;
        }
    }
    flush();

    double total = 0.0;
    for (const auto& [name, ms] : timings)
    {
        std::printf("  %8.1f ms  %s\n", ms, name.c_str());
        total += ms;
    }
    std::printf("frame: %.1f ms in %zu segments\n", total, timings.size());
    for (const auto& [role, b] : batches)
    {
        const uint32_t bakedDiff = (uint32_t)bake.value(role + ".diffSize"), bakedSpec = (uint32_t)bake.value(role + ".specSize");
        std::printf("samples (%s): %u diffuse, %u specular (Falcor: %u, %u) %s\n", role.c_str(), b.diffSize, b.specSize, bakedDiff, bakedSpec,
                    b.diffSize == bakedDiff && b.specSize == bakedSpec ? "OK" : "MISMATCH");
    }
#ifdef __OHOS__
    if (auto* npu = dynamic_cast<NnrtBackend*>(backend.get()))
        std::printf("NPU: %u runs, %.1f ms inside RunSync (%.2f ms per tile), %.1f ms copying tiles\n", npu->runs, npu->runMs,
                    npu->runs ? npu->runMs / npu->runs : 0.0, npu->copyMs);
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
    vkDeviceWaitIdle(dev.device);
    return 0;
}

// NCR replayer: runs the NeuralRadiosity render path from a Falcor bake, without Falcor.
//
//   ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--networks rows|tiles|npu] [--shaders <dir>]
//              [--models <dir>] [--stages]
//
// It replays the baked frame: the same SPIR-V, the same scene data, the same uniform values, in the
// order NeuralRadiosity::render() runs them (the per-frame K-Means cone trace path):
//   FirstSmoothPass -> prefix sums -> CompactPass -> ConeTrace -> SpecPrimary -> networks ->
//   ResolvePass -> PostFilterPass -> PostFXAAPass
// and writes every image and ray-batch buffer the passes produced to the output directory, where
// ohos/tools/replay_compare.py compares them with what Falcor produced in that frame.
//
// The networks, --networks:
//   tiles  (default) the way the phone runs them: a GPU pass looks up the hash grids and lays the
//          samples out in NPU tiles (shaders/EncodeInputs), the MLPs run on those tiles, a GPU pass
//          writes the outputs back as rows (shaders/DecodeOutputs). The MLPs run on the CPU.
//   npu    the same with the MLPs on the NPU through NNRt (OpenHarmony only; --models is the folder
//          with ncr_{diff,spec}_inf_256x256_fp16.om from ohos/npu/export_models.py).
//   rows   the float32 reference on the CPU, lookups included, straight on the sample rows.
// --shaders is the folder with the replayer's own SPIR-V (default: shaders next to the executable).
// With --stages the results are also written after the primary rays and after the cone trace
// (stage1_*, stage3_*): intermediate states Falcor's bake does not have, for comparing two replays.
//
// The prefix sums still run on the CPU. The same source builds for Windows (build_pc.bat) and for
// OpenHarmony (run_device.sh).
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
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; i++)
    {
        const uint32_t value = data[i];
        data[i] = sum;
        sum += value;
    }
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
}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: MSVC rejects line buffering with a null buffer
    if (argc < 4)
    {
        std::printf("usage: ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--networks rows|tiles|npu] [--shaders <dir>] [--models <dir>] [--stages]\n");
        return 2;
    }
    const std::string bakeDir = argv[1], checkpointPath = argv[2], outDir = argv[3];
    const std::string executable = argv[0];
    const size_t slash = executable.find_last_of("/\\");
    std::string networksMode = "tiles", modelDir = ".";
    std::string shaderDir = (slash == std::string::npos ? std::string(".") : executable.substr(0, slash)) + "/shaders";
    bool stages = false;
    for (int i = 4; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--stages") stages = true;
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

    // Everything the passes write: the images they output and the ray-batch buffers.
    auto dump = [&](const std::string& prefix) {
        size_t written = 0;
        for (auto& [id, texture] : bake.textures)
        {
            if (!texture.storage) continue;
            const std::vector<char> data = dev.readTexture(texture);
            writeFile(outDir + "/" + prefix + "texture" + std::to_string(id) + ".bin", data.data(), data.size());
            written++;
        }
        for (auto& [id, buffer] : bake.buffers)
        {
            if (bake.bufferLoaded[id]) continue;
            writeFile(outDir + "/" + prefix + "buffer" + std::to_string(id) + ".bin", buffer.data, (size_t)buffer.size);
            written++;
        }
        return written;
    };

    const uint32_t pixels = bake.width * bake.height;
    const bool sharedIO = bake.values["render.sharedIO"] != 0;
    Pass& firstSmooth = bake.pass("FirstSmoothPass");
    Pass& compact = bake.pass("CompactPass");
    Pass& coneTrace = bake.pass("ConeTrace");
    Pass& specPrimary = bake.pass("SpecPrimary");
    Pass& resolve = bake.pass("ResolvePass");
    Pass& postFilter = bake.pass("PostFilterPass");
    Pass& postFXAA = bake.pass("PostFXAAPass");

    // The network path of the phone: grids and tiles on the GPU, the MLPs behind a backend.
    const bool useTiles = networksMode != "rows";
    std::unique_ptr<NetworkBackend> backend;
    Buffer primGrid, clsGrid, specX, diffX, specY, diffY;
    OwnPass encodeSpec, encodeDiff, decodeSpec, decodeDiff;
    const uint32_t maxTiles = (pixels + cfg::kTile - 1) / cfg::kTile;
    if (useTiles)
    {
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        primGrid = dev.makeBuffer(checkpoint.primGrid.size() * 2, usage, true);
        clsGrid = dev.makeBuffer(checkpoint.clsGrid.size() * 2, usage, true);
        std::memcpy(primGrid.data, checkpoint.primGrid.data(), checkpoint.primGrid.size() * 2);
        std::memcpy(clsGrid.data, checkpoint.clsGrid.data(), checkpoint.clsGrid.size() * 2);
        const VkDeviceSize tile = (VkDeviceSize)cfg::kTile * 2;
        specX = dev.makeBuffer(maxTiles * cfg::kSpecInputDim * tile, usage, true);
        diffX = dev.makeBuffer(maxTiles * cfg::kDiffInputDim * tile, usage, true);
        specY = dev.makeBuffer(maxTiles * cfg::kOutputChannels * tile, usage, true);
        diffY = dev.makeBuffer(maxTiles * cfg::kOutputChannels * tile, usage, true);
        for (Buffer* b : {&specX, &diffX, &specY, &diffY}) std::memset(b->data, 0, (size_t)b->size);

        VkDescriptorPool pool = makeOwnPool(dev, 4, 12);
        encodeSpec.create(dev, pool, shaderDir + "/EncodeSpec.spv", 4, sizeof(RowsPush), 64);
        encodeDiff.create(dev, pool, shaderDir + "/EncodeDiff.spv", 4, sizeof(RowsPush), 64);
        decodeSpec.create(dev, pool, shaderDir + "/DecodeOutputs.spv", 2, sizeof(RowsPush), 64);
        decodeDiff.create(dev, pool, shaderDir + "/DecodeOutputs.spv", 2, sizeof(RowsPush), 64);
        encodeSpec.bind(dev, 0, bake.buffer("render.specInput"));
        encodeSpec.bind(dev, 1, primGrid);
        encodeSpec.bind(dev, 2, clsGrid);
        encodeSpec.bind(dev, 3, specX);
        encodeDiff.bind(dev, 0, bake.buffer("render.diffInput"));
        encodeDiff.bind(dev, 1, primGrid);
        encodeDiff.bind(dev, 2, clsGrid);
        encodeDiff.bind(dev, 3, diffX);
        decodeSpec.bind(dev, 0, specY);
        decodeSpec.bind(dev, 1, bake.buffer("render.specColor"));
        decodeDiff.bind(dev, 0, diffY);
        decodeDiff.bind(dev, 1, bake.buffer("render.diffColor"));

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

    // 1. Primary rays: G-buffer and the diffuse / specular masks.
    VkCommandBuffer cb = dev.begin();
    bake.dispatch(cb, firstSmooth, bake.width, bake.height);
    const double firstSmoothMs = dev.submit(cb);
    if (stages) dump("stage1_");

    // 2. Masks -> compacted indices and counts (computeBatchCounts).
    start = std::chrono::steady_clock::now();
    uint32_t* counts = bake.buffer("render.counts").as<uint32_t>();
    counts[0] = prefixSum(bake.buffer("render.diffIndex").as<uint32_t>(), pixels);
    counts[1] = prefixSum(bake.buffer("render.specIndex").as<uint32_t>(), pixels);
    const uint32_t diffSize = counts[0], specSize = counts[1];
    const double prefixMs = msSince(start);

    // 3. Compaction and the specular samples' cluster inputs. The dispatch sizes are the ones Falcor
    //    used in this frame: it sizes them from the previous frame's count.
    //    ConeTrace runs one thread group per specular sample, and a group strides over further samples
    //    when there are fewer groups than samples. The desktop takes any number of groups; the phone
    //    runs 65,535 per dimension at most, so the count is capped and the shader told the stride.
    const uint32_t bakedConeGroups = bake.getUniform<uint32_t>(coneTrace, "gConeTrace", "dispatchGroupCount");
    const uint32_t coneGroups = std::min(bakedConeGroups, dev.maxGroupCount[0]);
    bake.setUniform(coneTrace, "gConeTrace", "dispatchGroupCount", coneGroups);
    const uint32_t specThreads = bake.getUniform<uint32_t>(specPrimary, "gSpecPrimary", "dispatchThreadCount");
    const uint32_t numSpecRays = (uint32_t)bake.values["numSpecRays"];
    cb = dev.begin();
    bake.dispatch(cb, compact, bake.width, bake.height);
    bake.dispatch(cb, coneTrace, coneGroups * numSpecRays);
    bake.dispatch(cb, specPrimary, specThreads);
    const double coneMs = dev.submit(cb);
    if (stages) dump("stage3_");

    // 4. The networks. With shared I/O the specular rows come first and the diffuse rows follow the
    //    padded specular rows (RayBatchBuffer::diffInputOffset / diffOutputOffset).
    const uint32_t diffOutputOffset = sharedIO ? padUp(specSize, cfg::kBatchAlign) : 0;
    uint32_t diffInputOffset = 0;   // in DiffNetInput rows
    if (sharedIO)
    {
        const uint64_t specBytes = (uint64_t)diffOutputOffset * cfg::kSpecInputDim * 2;
        const uint32_t diffRowBytes = cfg::kDiffInputDim * 2;
        diffInputOffset = (uint32_t)((specBytes + diffRowBytes - 1) / diffRowBytes);
    }
    double encodeMs = 0.0, networkMs = 0.0;
    const uint32_t specTiles = (specSize + cfg::kTile - 1) / cfg::kTile, diffTiles = (diffSize + cfg::kTile - 1) / cfg::kTile;
    if (useTiles)
    {
        cb = dev.begin();
        encodeSpec.dispatch(cb, RowsPush{specSize, 0}, specSize);
        encodeDiff.dispatch(cb, RowsPush{diffSize, diffInputOffset * cfg::kDiffInputDim}, diffSize);
        encodeMs = dev.submit(cb);
        start = std::chrono::steady_clock::now();
        backend->infer(true, specX.as<uint16_t>(), specY.as<uint16_t>(), specTiles);
        backend->infer(false, diffX.as<uint16_t>(), diffY.as<uint16_t>(), diffTiles);
        networkMs = msSince(start);
    }
    else
    {
        start = std::chrono::steady_clock::now();
        rowNetworks.inferSpec(bake.buffer("render.specInput").as<uint16_t>(), specSize, bake.buffer("render.specColor").as<float>());
        rowNetworks.inferDiff(bake.buffer("render.diffInput").as<uint16_t>() + (size_t)diffInputOffset * cfg::kDiffInputDim, diffSize,
                              bake.buffer("render.diffColor").as<float>() + (size_t)diffOutputOffset * cfg::kOutputDim);
        networkMs = msSince(start);
    }

    // 5. Outputs back to rows, compose and filter.
    const uint32_t bakedDiff = bake.getUniform<uint32_t>(resolve, "gResolvePass", "diffSize");
    const uint32_t bakedSpec = bake.getUniform<uint32_t>(resolve, "gResolvePass", "specSize");
    cb = dev.begin();
    if (useTiles)
    {
        decodeSpec.dispatch(cb, RowsPush{specSize, 0}, specSize);
        decodeDiff.dispatch(cb, RowsPush{diffSize, diffOutputOffset}, diffSize);
    }
    bake.dispatch(cb, resolve, bake.width, bake.height);
    bake.dispatch(cb, postFilter, bake.width, bake.height);
    bake.dispatch(cb, postFXAA, bake.width, bake.height);
    const double resolveMs = dev.submit(cb);

    std::printf("samples: %u diffuse, %u specular (Falcor: %u, %u) %s\n", diffSize, specSize, bakedDiff, bakedSpec,
                diffSize == bakedDiff && specSize == bakedSpec ? "OK" : "MISMATCH");
    std::printf("frame: FirstSmoothPass %.1f ms | prefix sums (CPU) %.1f | CompactPass + ConeTrace (%u groups) + SpecPrimary %.1f | "
                "grid lookups to tiles (GPU) %.1f | MLPs, %u + %u tiles %.1f | outputs + ResolvePass + PostFilterPass + PostFXAAPass %.1f\n",
                firstSmoothMs, prefixMs, coneGroups, coneMs, encodeMs, specTiles, diffTiles, networkMs, resolveMs);
    if (coneGroups != bakedConeGroups)
        std::printf("ConeTrace: %u groups instead of Falcor's %u (device limit), each striding over the samples beyond\n", coneGroups, bakedConeGroups);
#ifdef __OHOS__
    if (auto* npu = dynamic_cast<NnrtBackend*>(backend.get()))
        std::printf("NPU: %u runs, %.1f ms inside RunSync (%.2f ms per tile), %.1f ms copying tiles\n", npu->runs, npu->runMs,
                    npu->runs ? npu->runMs / npu->runs : 0.0, npu->copyMs);
#endif

    // The end of the frame, for the comparison with Falcor.
    const size_t written = dump("");
    std::printf("wrote %zu results to %s\n", written, outDir.c_str());
    vkDeviceWaitIdle(dev.device);
    return 0;
}

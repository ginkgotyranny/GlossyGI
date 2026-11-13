// NCR replayer: runs the NeuralRadiosity render path from a Falcor bake, without Falcor.
//
//   ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--stages]
//
// It replays the baked frame: the same SPIR-V, the same scene data, the same uniform values, in the
// order NeuralRadiosity::render() runs them (the per-frame K-Means cone trace path):
//   FirstSmoothPass -> prefix sums -> CompactPass -> ConeTrace -> SpecPrimary -> networks ->
//   ResolvePass -> PostFilterPass -> PostFXAAPass
// and writes every image and ray-batch buffer the passes produced to the output directory, where
// ohos/tools/replay_compare.py compares them with what Falcor produced in that frame.
//
// With --stages the same files are also written after the primary rays and after the cone trace
// (stage1_*, stage3_*): intermediate states Falcor's bake does not have, for comparing two replays.
//
// This first version runs the prefix sums and the networks on the CPU (replay_nn.h). The same source
// builds for Windows (build_pc.bat) and for OpenHarmony.
#include "replay_bake.h"
#include "replay_nn.h"

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
}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: MSVC rejects line buffering with a null buffer
    if (argc < 4)
    {
        std::printf("usage: ncr_replay <bake dir> <checkpoint.ckpt> <output dir> [--stages]\n");
        return 2;
    }
    const std::string bakeDir = argv[1], checkpointPath = argv[2], outDir = argv[3];
    const bool stages = argc > 4 && std::string(argv[4]) == "--stages";

    Device dev;
    dev.create();
    std::printf("device: %s\n", dev.name);

    auto start = std::chrono::steady_clock::now();
    Bake bake(dev, bakeDir);
    bake.load();
    std::printf("bake %ux%u (Falcor frame %u) loaded in %.0f ms\n", bake.width, bake.height, bake.frameCount, msSince(start));

    Checkpoint checkpoint;
    checkpoint.load(checkpointPath);
    const CpuNetworks networks(checkpoint);

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
    start = std::chrono::steady_clock::now();
    uint16_t* specRows = bake.buffer("render.specInput").as<uint16_t>();
    float* specOut = bake.buffer("render.specColor").as<float>();
    uint16_t* diffRows = bake.buffer("render.diffInput").as<uint16_t>();
    float* diffOut = bake.buffer("render.diffColor").as<float>();
    const uint32_t diffOutputOffset = sharedIO ? padUp(specSize, cfg::kBatchAlign) : 0;
    if (sharedIO)
    {
        const uint64_t specBytes = (uint64_t)diffOutputOffset * cfg::kSpecInputDim * 2;
        const uint32_t diffRowBytes = cfg::kDiffInputDim * 2;
        const uint32_t diffInputOffset = (uint32_t)((specBytes + diffRowBytes - 1) / diffRowBytes);
        diffRows += (size_t)diffInputOffset * cfg::kDiffInputDim;
        diffOut += (size_t)diffOutputOffset * cfg::kOutputDim;
    }
    networks.inferSpec(specRows, specSize, specOut);
    networks.inferDiff(diffRows, diffSize, diffOut);
    const double networkMs = msSince(start);

    // 5. Compose and filter.
    const uint32_t bakedDiff = bake.getUniform<uint32_t>(resolve, "gResolvePass", "diffSize");
    const uint32_t bakedSpec = bake.getUniform<uint32_t>(resolve, "gResolvePass", "specSize");
    cb = dev.begin();
    bake.dispatch(cb, resolve, bake.width, bake.height);
    bake.dispatch(cb, postFilter, bake.width, bake.height);
    bake.dispatch(cb, postFXAA, bake.width, bake.height);
    const double resolveMs = dev.submit(cb);

    std::printf("samples: %u diffuse, %u specular (Falcor: %u, %u) %s\n", diffSize, specSize, bakedDiff, bakedSpec,
                diffSize == bakedDiff && specSize == bakedSpec ? "OK" : "MISMATCH");
    std::printf("frame: FirstSmoothPass %.1f ms | prefix sums (CPU) %.1f | CompactPass + ConeTrace (%u groups) + SpecPrimary %.1f | "
                "networks (CPU) %.1f | ResolvePass + PostFilterPass + PostFXAAPass %.1f\n",
                firstSmoothMs, prefixMs, coneGroups, coneMs, networkMs, resolveMs);
    if (coneGroups != bakedConeGroups)
        std::printf("ConeTrace: %u groups instead of Falcor's %u (device limit), each striding over the samples beyond\n", coneGroups, bakedConeGroups);

    // The end of the frame, for the comparison with Falcor.
    const size_t written = dump("");
    std::printf("wrote %zu results to %s\n", written, outDir.c_str());
    vkDeviceWaitIdle(dev.device);
    return 0;
}

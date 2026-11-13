// Bake export for the OpenHarmony port (ohos/README.md).
//
// The phone has neither Falcor nor a shader compiler. It replays the compute passes of this render
// pass from what Falcor compiled and bound on the desktop: with the environment variable
// NCR_BAKE_DIR set, frame NCR_BAKE_FRAME (default 5) writes
//   spirv/<Pass>.spv     the SPIR-V of every compute pass, as compiled for the current scene and
//                        settings (Vulkan backend only)
// into that directory. Without the variable nothing here runs.
#include "NeuralRadiosity.h"
#include "Core/Program/ProgramVersion.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace
{
struct BakeSettings
{
    bool enabled = false;
    std::filesystem::path dir;
    uint32_t frame = 5;
    bool done = false;

    BakeSettings()
    {
        const char* d = std::getenv("NCR_BAKE_DIR");
        enabled = d && *d;
        if (!enabled)
            return;
        dir = d;
        if (const char* f = std::getenv("NCR_BAKE_FRAME"))
            frame = (uint32_t)std::atoi(f);
    }
};
BakeSettings gBake;

void writeFile(const std::filesystem::path& path, const void* data, size_t size)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), (std::streamsize)size);
    if (!f.good())
        FALCOR_THROW("NCR_BAKE_DIR: cannot write {}", path.string());
}
} // namespace

void NeuralRadiosity::bakeExport(RenderContext* pRenderContext)
{
    if (!gBake.enabled || gBake.done || mFrameCount != gBake.frame)
        return;
    gBake.done = true;
    if (mpDevice->getType() != Device::Type::Vulkan)
    {
        logWarning("NCR_BAKE_DIR is set, but the bake needs the Vulkan backend (SPIR-V); nothing written.");
        return;
    }

    const std::pair<const char*, ref<ComputePass>> passes[] = {
        {"FirstSmoothPass", mpFirstSmoothPass},
        {"CompactPass", mpCompactPass},
        {"ResolvePass", mpResolvePass},
        {"PostFilterPass", mpPostFilterPass},
        {"PostFXAAPass", mpPostFXAAPass},
        {"RandomSmooth", mpRandomSmooth},
        {"SampleRHS", mpSampleRHS},
        {"ResolveRHS", mpResolveRHS},
        {"CompactBatch", mpCompactBatch},
        {"ResolveBatch", mpResolveBatch},
        {"ConeTrace", mpConeTrace},
        {"SpecPrimary", mpSpecPrimary},
        {"ClusterReproject", mpClusterReproject},
        {"ConeTraceScreen", mpConeTraceScreen},
        {"ClusterSpatialReuse", mpClusterSpatialReuse},
    };
    uint32_t written = 0;
    for (const auto& [name, pPass] : passes)
    {
        if (!pPass)
            continue;
        const auto pKernels = pPass->getProgram()->getActiveVersion()->getKernels(mpDevice.get(), pPass->getVars().get());
        const EntryPointKernel* pKernel = pKernels->getKernel(ShaderType::Compute);
        const auto blob = pKernel->getBlobData();
        writeFile(gBake.dir / "spirv" / (std::string(name) + ".spv"), blob.data, blob.size);
        ++written;
    }
    logInfo("NCR bake: wrote the SPIR-V of {} compute passes to {}", written, gBake.dir.string());
}

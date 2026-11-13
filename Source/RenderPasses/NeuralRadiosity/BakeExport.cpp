// Bake export for the OpenHarmony port (ohos/README.md).
//
// The phone has neither Falcor nor a shader compiler. It replays the compute passes of this render
// pass from what Falcor compiled and bound on the desktop. With the environment variable
// NCR_BAKE_DIR set (Vulkan backend), frame NCR_BAKE_FRAME (default 5) is recorded as it executes
// and written into that directory:
//
//   spirv/<Pass>.spv   the SPIR-V of every compute pass that ran, as compiled for the loaded scene
//                      and the current settings. The modules name their bindings by shader path.
//   data/*.bin         the contents of every buffer and texture those passes had bound (the scene's
//                      geometry, materials and lights, and this pass's own buffers as they are at the
//                      end of the frame, which serve as reference results), and their uniform blocks.
//   bake.txt           the index, one record per line, fields separated by spaces:
//       frame <width> <height> <frameCount>
//       pass <Pass> <groupSizeX> <groupSizeY> <groupSizeZ>
//       uniform <Pass> <blockPath> <memberPath> <byteOffset> <byteSize>   layout of the uniform blocks
//     the frame, in execution order (<k> counts calls and host steps together):
//       call <k> <Pass> <dimX> <dimY> <dimZ>          one execution; group counts are dims / group size, rounded up
//       cbind <k> <path> <index> buffer|texture|sampler <id>           what that execution had bound; <index> is the array element
//       cbind <k> <path> <index> uniform <bytes> <file>                the uniform block of a parameter block, as it was then
//       cbind <k> <path> <index> accel | null
//       host <k> clearcounts <batch>                  what Falcor does between passes on the CPU or in CUDA:
//       host <k> prefixsum <batch> diff|spec <elementCount>             exclusive scan of the index buffer, total into counts
//       host <k> infer <batch> | train <batch>                          network inference / one training step on that batch
//     resources:
//       buffer <id> <bytes> <structSize> <elementCount>                    contents in data/buffer<id>.bin
//       texture <id> <type> <format> <width> <height> <depth> <mips> <layers>
//                                         contents in data/texture<id>.bin: for each layer, for each mip
//       sampler <id> <min> <mag> <mip> <addressU> <addressV> <addressW> <maxAnisotropy> <lodBias>
//               <minLod> <maxLod> <comparison> <borderR> <borderG> <borderB> <borderA>
//       named <batch>.<member> buffer <id>       the buffers of the ray batches (render, lhs, rhs, dummy) by role
//       value <name> <number>                    sizes and switches at the baked frame
//       geometry <globalInstanceID> <tlasInstance> <geometryIndex> <meshID> <materialID> <vbOffset>
//                <ibOffset> <vertexCount> <indexCount> <index16> <frontFaceCW> <opaque> <12 floats: 3x4 transform>
//
// Without the variable nothing here runs.
#include "NeuralRadiosity.h"
#include "Core/API/ParameterBlock.h"
#include "Core/Program/ProgramReflection.h"
#include "Core/Program/ProgramVersion.h"
#include "Scene/Animation/AnimationController.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>

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

/// The resource bound to a texture or buffer variable, or null. ShaderVar::getBuffer() and
/// getTexture() dereference the view without a check, and an unbound slot holds a null view.
Resource* boundResource(const ShaderVar& var)
{
    try
    {
        ref<ShaderResourceView> pSrv = var.getSrv();   // throws for a read-write variable
        return pSrv ? pSrv->getResource() : nullptr;
    }
    catch (const std::exception&)
    {}
    try
    {
        ref<UnorderedAccessView> pUav = var.getUav();
        return pUav ? pUav->getResource() : nullptr;
    }
    catch (const std::exception&)
    {}
    return nullptr;
}

class Baker
{
public:
    Baker(Device* pDevice, const std::filesystem::path& dir) : mpDevice(pDevice), mDir(dir)
    {
        std::filesystem::create_directories(mDir / "data");
        mIndex.open(mDir / "bake.txt", std::ios::trunc);
    }

    template<typename... Args>
    void record(fmt::format_string<Args...> format, Args&&... args)
    {
        mIndex << fmt::format(format, std::forward<Args>(args)...) << std::endl;
    }

    /// A host-side step between passes.
    void host(const std::string& what) { record("host {} {}", mStep++, what); }

    /// One execution of a compute pass, with everything it had bound.
    void call(const std::string& name, const ref<ComputePass>& pPass, const uint3& dims)
    {
        mPass = name;
        mRecordLayout = mPasses.insert(name).second;
        if (mRecordLayout)
        {
            const auto pKernels = pPass->getProgram()->getActiveVersion()->getKernels(mpDevice, pPass->getVars().get());
            const auto blob = pKernels->getKernel(ShaderType::Compute)->getBlobData();
            writeFile(mDir / "spirv" / (name + ".spv"), blob.data, blob.size);
            const uint3 group = pPass->getThreadGroupSize();
            record("pass {} {} {} {}", name, group.x, group.y, group.z);
        }
        mCall = mStep++;
        record("call {} {} {} {} {}", mCall, name, dims.x, dims.y, dims.z);
        walk("", "", pPass->getRootVar());
    }

    /// Records a buffer under a role name.
    void named(const std::string& name, const ref<Buffer>& pBuffer)
    {
        if (pBuffer)
            record("named {} buffer {}", name, bufferId(pBuffer.get()));
    }

    /// The contents of every resource seen, as they are now.
    void writeResources(RenderContext* pRenderContext)
    {
        for (const auto& [pBuffer, id] : mBuffers)
        {
            std::vector<uint8_t> data(pBuffer->getSize());
            pBuffer->getBlob(data.data(), 0, data.size());
            writeFile(mDir / fmt::format("data/buffer{}.bin", id), data.data(), data.size());
            record("buffer {} {} {} {}", id, pBuffer->getSize(), pBuffer->getStructSize(), pBuffer->getElementCount());
        }
        for (const auto& [pTexture, id] : mTextures)
        {
            const bool cube = pTexture->getType() == Resource::Type::TextureCube;
            const uint32_t layers = pTexture->getArraySize() * (cube ? 6 : 1);
            std::ofstream f(mDir / fmt::format("data/texture{}.bin", id), std::ios::binary);
            for (uint32_t layer = 0; layer < layers; ++layer)
            {
                for (uint32_t mip = 0; mip < pTexture->getMipCount(); ++mip)
                {
                    const std::vector<uint8_t> data = pRenderContext->readTextureSubresource(pTexture, pTexture->getSubresourceIndex(layer, mip));
                    f.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());
                }
            }
            const char* type = cube ? "Cube" : pTexture->getType() == Resource::Type::Texture3D ? "3D" : "2D";
            record(
                "texture {} {} {} {} {} {} {} {}",
                id, type, to_string(pTexture->getFormat()), pTexture->getWidth(), pTexture->getHeight(), pTexture->getDepth(),
                pTexture->getMipCount(), layers
            );
        }
        for (const auto& [pSampler, id] : mSamplers)
        {
            const Sampler::Desc& d = pSampler->getDesc();
            record(
                "sampler {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                id, (int)d.minFilter, (int)d.magFilter, (int)d.mipFilter, (int)d.addressModeU, (int)d.addressModeV, (int)d.addressModeW,
                d.maxAnisotropy, d.lodBias, d.minLod, d.maxLod, (int)d.comparisonFunc, d.borderColor.x, d.borderColor.y, d.borderColor.z,
                d.borderColor.w
            );
        }
    }

private:
    /// Visits every resource and uniform reachable from a shader variable. `path` is the dotted shader
    /// path, which is the name the SPIR-V gives the binding; `block` is the parameter block whose
    /// uniform buffer holds the plain members.
    void walk(const std::string& path, const std::string& block, const ShaderVar& var)
    {
        const ReflectionType* pType = var.getType();
        switch (pType->getKind())
        {
        case ReflectionType::Kind::Struct:
        {
            const ReflectionStructType* pStruct = pType->asStructType();
            for (uint32_t i = 0; i < pStruct->getMemberCount(); ++i)
            {
                const std::string& member = pStruct->getMember(i)->getName();
                walk(path.empty() ? member : path + "." + member, block, var[member]);
            }
            break;
        }
        case ReflectionType::Kind::Array:
        {
            const ReflectionArrayType* pArray = pType->asArrayType();
            const ReflectionType::Kind elementKind = pArray->getElementType()->getKind();
            if (elementKind == ReflectionType::Kind::Resource)
            {
                for (uint32_t i = 0; i < pArray->getElementCount(); ++i)
                    resource(path, i, var[i]);
            }
            else if (elementKind == ReflectionType::Kind::Struct && pArray->getElementCount() <= 16)
            {
                for (uint32_t i = 0; i < pArray->getElementCount(); ++i)
                    walk(fmt::format("{}[{}]", path, i), block, var[i]);
            }
            else if (elementKind == ReflectionType::Kind::Basic)
            {
                uniform(path, block, var, pArray->getElementCount() * pArray->getElementByteStride());
            }
            break;
        }
        case ReflectionType::Kind::Basic:
            uniform(path, block, var, pType->getByteSize());
            break;
        case ReflectionType::Kind::Resource:
            resource(path, 0, var);
            break;
        default:
            break;
        }
    }

    void uniform(const std::string& path, const std::string& block, const ShaderVar& var, size_t size)
    {
        if (block.empty() || !mRecordLayout)
            return;
        // Member path relative to the block.
        const std::string member = path.size() > block.size() ? path.substr(block.size() + 1) : path;
        record("uniform {} {} {} {} {}", mPass, block, member, var.getByteOffset(), size);
    }

    void resource(const std::string& path, uint32_t index, const ShaderVar& var)
    {
        const ReflectionResourceType* pType = var.getType()->asResourceType();
        const std::string where = fmt::format("cbind {} {} {}", mCall, path, index);
        try
        {
            switch (pType->getType())
            {
            case ReflectionResourceType::Type::Texture:
            {
                Resource* pResource = boundResource(var);
                ref<Texture> pTexture = pResource ? pResource->asTexture() : nullptr;
                if (pTexture)
                    record("{} texture {}", where, textureId(pTexture.get()));
                else
                    record("{} null", where);
                break;
            }
            case ReflectionResourceType::Type::StructuredBuffer:
            case ReflectionResourceType::Type::RawBuffer:
            case ReflectionResourceType::Type::TypedBuffer:
            {
                Resource* pResource = boundResource(var);
                ref<Buffer> pBuffer = pResource ? pResource->asBuffer() : nullptr;
                if (pBuffer)
                    record("{} buffer {}", where, bufferId(pBuffer.get()));
                else
                    record("{} null", where);
                break;
            }
            case ReflectionResourceType::Type::Sampler:
            {
                ref<Sampler> pSampler = var.getSampler();
                if (pSampler)
                    record("{} sampler {}", where, samplerId(pSampler.get()));
                else
                    record("{} null", where);
                break;
            }
            case ReflectionResourceType::Type::AccelerationStructure:
                record("{} accel", where);
                break;
            case ReflectionResourceType::Type::ConstantBuffer:
            {
                // ConstantBuffer<T> and ParameterBlock<T>: a block with its own uniform data and members.
                ref<ParameterBlock> pBlock = var.getParameterBlock();
                if (!pBlock)
                {
                    record("{} null", where);
                    break;
                }
                const size_t size = pBlock->getSize();
                if (size > 0 && pBlock->getRawData())
                    record("{} uniform {} {}", where, size, uniformFile(pBlock->getRawData(), size));
                walk(path, path, pBlock->getRootVar());
                break;
            }
            }
        }
        catch (const std::exception& e)
        {
            record("{} null", where);
            logWarning("NCR bake: {} {}: {}", mPass, path, e.what());
        }
    }

    /// Uniform blocks with the same bytes share one file (the scene's block is the same in every call).
    std::string uniformFile(const void* data, size_t size)
    {
        const std::string bytes(static_cast<const char*>(data), size);
        auto it = mUniformFiles.find(bytes);
        if (it != mUniformFiles.end())
            return it->second;
        const std::string file = fmt::format("data/uniform{}.bin", mUniformFiles.size());
        writeFile(mDir / file, data, size);
        mUniformFiles[bytes] = file;
        return file;
    }

    uint32_t bufferId(const Buffer* pBuffer)
    {
        auto it = mBuffers.find(pBuffer);
        if (it != mBuffers.end())
            return it->second;
        const uint32_t id = (uint32_t)mBuffers.size();
        mBuffers[pBuffer] = id;
        mKeepAlive.push_back(ref<const Object>(pBuffer));
        return id;
    }

    uint32_t textureId(const Texture* pTexture)
    {
        auto it = mTextures.find(pTexture);
        if (it != mTextures.end())
            return it->second;
        const uint32_t id = (uint32_t)mTextures.size();
        mTextures[pTexture] = id;
        mKeepAlive.push_back(ref<const Object>(pTexture));
        return id;
    }

    uint32_t samplerId(const Sampler* pSampler)
    {
        auto it = mSamplers.find(pSampler);
        if (it != mSamplers.end())
            return it->second;
        const uint32_t id = (uint32_t)mSamplers.size();
        mSamplers[pSampler] = id;
        mKeepAlive.push_back(ref<const Object>(pSampler));
        return id;
    }

    Device* mpDevice;
    std::filesystem::path mDir;
    std::ofstream mIndex;
    std::string mPass;
    uint32_t mStep = 0;
    uint32_t mCall = 0;
    bool mRecordLayout = false;
    std::set<std::string> mPasses;
    std::map<const Buffer*, uint32_t> mBuffers;
    std::map<const Texture*, uint32_t> mTextures;
    std::map<const Sampler*, uint32_t> mSamplers;
    std::map<std::string, std::string> mUniformFiles;
    std::vector<ref<const Object>> mKeepAlive;
};

std::unique_ptr<Baker> gBaker;
} // namespace

void NeuralRadiosity::bakeBegin()
{
    if (!gBake.enabled || gBake.done || mFrameCount != gBake.frame)
        return;
    if (mpDevice->getType() != Device::Type::Vulkan)
    {
        gBake.done = true;
        logWarning("NCR_BAKE_DIR is set, but the bake needs the Vulkan backend (SPIR-V); nothing written.");
        return;
    }
    gBaker = std::make_unique<Baker>(mpDevice.get(), gBake.dir);
    gBaker->record("frame {} {} {}", mFrameDim.x, mFrameDim.y, mFrameCount);
}

void NeuralRadiosity::executePass(RenderContext* pRenderContext, const ref<ComputePass>& pPass, const char* name, const uint3& dims)
{
    pPass->execute(pRenderContext, dims);
    if (gBaker)
        gBaker->call(name, pPass, dims);
}

const char* NeuralRadiosity::batchRole(const RayBatchBuffer* pBatch) const
{
    return pBatch == mpRenderBatch.get() ? "render" : pBatch == mpTrainLHSBatch.get() ? "lhs" : pBatch == mpTrainRHSBatch.get() ? "rhs" : "dummy";
}

void NeuralRadiosity::bakeHost(const std::string& step)
{
    if (gBaker)
        gBaker->host(step);
}

void NeuralRadiosity::bakeExport(RenderContext* pRenderContext)
{
    if (!gBaker)
        return;
    gBake.done = true;
    // Everything the frame recorded has to be on the GPU before the buffers are read back.
    pRenderContext->submit(true);
    Baker& baker = *gBaker;

    // The ray batches by role: a pass like ConeTrace or CompactBatch runs on several of them per frame.
    const std::pair<const char*, std::shared_ptr<RayBatchBuffer>> batches[] = {
        {"render", mpRenderBatch}, {"lhs", mpTrainLHSBatch}, {"rhs", mpTrainRHSBatch}, {"dummy", mpBindDummyBatch}};
    for (const auto& [name, pBatch] : batches)
    {
        if (!pBatch)
            continue;
        const std::string prefix = std::string(name) + ".";
        baker.named(prefix + "pos", pBatch->pos);
        baker.named(prefix + "dir", pBatch->dir);
        baker.named(prefix + "normal", pBatch->normal);
        baker.named(prefix + "albedo", pBatch->albedo);
        baker.named(prefix + "roughness", pBatch->roughness);
        baker.named(prefix + "vbuffer", pBatch->vbuffer);
        baker.named(prefix + "color", pBatch->color);
        baker.named(prefix + "emission", pBatch->emission);
        baker.named(prefix + "diffActive", pBatch->diffActive);
        baker.named(prefix + "diffIndex", pBatch->diffIndex);
        baker.named(prefix + "diffInput", pBatch->diffInput);
        baker.named(prefix + "diffColor", pBatch->diffColor);
        baker.named(prefix + "specSrc", pBatch->specSrc);
        baker.named(prefix + "specActive", pBatch->specActive);
        baker.named(prefix + "specIndex", pBatch->specIndex);
        baker.named(prefix + "specInput", pBatch->specInput);
        baker.named(prefix + "specColor", pBatch->specColor);
        baker.named(prefix + "counts", pBatch->counts);
        baker.record("value {}size {}", prefix, pBatch->size);
        baker.record("value {}diffSize {}", prefix, pBatch->diffSize);
        baker.record("value {}specSize {}", prefix, pBatch->specSize);
        baker.record("value {}needDiff {}", prefix, pBatch->needDiff ? 1 : 0);
        baker.record("value {}needSpec {}", prefix, pBatch->needSpec ? 1 : 0);
        baker.record("value {}needAll {}", prefix, pBatch->needAll ? 1 : 0);
        baker.record("value {}sharedIO {}", prefix, pBatch->sharedIO ? 1 : 0);
        baker.record("value {}diffStride {}", prefix, pBatch->sharedIO ? pBatch->diffStride : 0);
    }
    baker.record("value renderMode {}", (int)mRenderMode);
    baker.record("value numSpecRays {}", mNumSpecRays);
    baker.record("value numRHS {}", mNumRHS);
    baker.record("value clusterReuse {}", useClusterReuse() ? 1 : 0);

    baker.writeResources(pRenderContext);

    // What the replayer needs to rebuild the acceleration structures as Falcor lays them out:
    // InstanceID() + GeometryIndex() of a hit is the global geometry instance index.
    const auto& matrices = mpScene->getAnimationController()->getGlobalMatrices();
    for (uint32_t i = 0; i < mpScene->getGeometryInstanceCount(); ++i)
    {
        const GeometryInstanceData& instance = mpScene->getGeometryInstance(i);
        if (instance.getType() != GeometryType::TriangleMesh)
            continue;
        const MeshDesc& mesh = mpScene->getMesh(MeshID::fromSlang(instance.geometryID));
        const float4x4& m = matrices[instance.globalMatrixID];
        const bool opaque = mpScene->getMaterial(MaterialID::fromSlang(instance.materialID))->isOpaque();
        baker.record(
            "geometry {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            i, instance.instanceIndex, instance.geometryIndex, instance.geometryID, instance.materialID, mesh.vbOffset, mesh.ibOffset,
            mesh.vertexCount, mesh.indexCount, mesh.use16BitIndices() ? 1 : 0, mesh.isFrontFaceCW() ? 1 : 0, opaque ? 1 : 0,
            m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1], m[2][2], m[2][3]
        );
    }
    logInfo("NCR bake: wrote frame {} to {}", mFrameCount, gBake.dir.string());
    gBaker.reset();
}

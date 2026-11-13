// Bake export for the OpenHarmony port (ohos/README.md).
//
// The phone has neither Falcor nor a shader compiler. It replays the compute passes of this render
// pass from what Falcor compiled and bound on the desktop. With the environment variable
// NCR_BAKE_DIR set (Vulkan backend), frame NCR_BAKE_FRAME (default 5) writes into that directory:
//
//   spirv/<Pass>.spv   the SPIR-V of every compute pass, as compiled for the loaded scene and the
//                      current settings. The modules name their bindings by shader path.
//   data/*.bin         the contents of every buffer and texture those passes have bound (the scene's
//                      geometry, materials and lights, and this pass's own buffers as they are at the
//                      end of the frame, which serve as reference results), and their uniform blocks.
//   bake.txt           the index, one record per line, fields separated by spaces:
//       frame <width> <height> <frameCount>
//       pass <Pass> <groupSizeX> <groupSizeY> <groupSizeZ>
//       buffer <id> <bytes> <structSize> <elementCount>                    contents in data/buffer<id>.bin
//       texture <id> <type> <format> <width> <height> <depth> <mips> <layers>
//                                         contents in data/texture<id>.bin: for each layer, for each mip
//       sampler <id> <min> <mag> <mip> <addressU> <addressV> <addressW> <maxAnisotropy> <lodBias>
//               <minLod> <maxLod> <comparison> <borderR> <borderG> <borderB> <borderA>
//       bind <Pass> <path> <index> buffer|texture|sampler <id>          <index> is the array element
//       bind <Pass> <path> <index> uniform <bytes> <file>               the uniform block of a parameter block
//       bind <Pass> <path> <index> accel | null
//       uniform <Pass> <blockPath> <memberPath> <byteOffset> <byteSize>  layout of the uniform blocks
//       named <name> buffer <id>                 the buffers of the three ray batches, by role: the passes are
//                                                rebound to a different batch at each call site
//       value <name> <number>                    sizes and switches of the batches at the baked frame
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
    Baker(Device* pDevice, RenderContext* pRenderContext, const std::filesystem::path& dir)
        : mpDevice(pDevice), mpRenderContext(pRenderContext), mDir(dir)
    {
        std::filesystem::create_directories(mDir / "data");
        mIndex.open(mDir / "bake.txt", std::ios::trunc);
    }

    template<typename... Args>
    void record(fmt::format_string<Args...> format, Args&&... args)
    {
        mIndex << fmt::format(format, std::forward<Args>(args)...) << std::endl;
    }

    /// Records a buffer under a role name (and dumps it, if no pass had it bound).
    void named(const std::string& name, const ref<Buffer>& pBuffer)
    {
        if (pBuffer)
            record("named {} buffer {}", name, bufferId(pBuffer.get()));
    }

    void addPass(const std::string& name, const ref<ComputePass>& pPass)
    {
        const auto pKernels = pPass->getProgram()->getActiveVersion()->getKernels(mpDevice, pPass->getVars().get());
        const auto blob = pKernels->getKernel(ShaderType::Compute)->getBlobData();
        writeFile(mDir / "spirv" / (name + ".spv"), blob.data, blob.size);
        const uint3 group = pPass->getThreadGroupSize();
        record("pass {} {} {} {}", name, group.x, group.y, group.z);
        mPass = name;
        walk("", "", pPass->getRootVar());
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
                    resource(path, i, block, var[i]);
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
            resource(path, 0, block, var);
            break;
        default:
            break;
        }
    }

    void uniform(const std::string& path, const std::string& block, const ShaderVar& var, size_t size)
    {
        if (block.empty())
            return;
        // Member path relative to the block.
        const std::string member = path.size() > block.size() ? path.substr(block.size() + 1) : path;
        record("uniform {} {} {} {} {}", mPass, block, member, var.getByteOffset(), size);
    }

    void resource(const std::string& path, uint32_t index, const std::string& block, const ShaderVar& var)
    {
        const ReflectionResourceType* pType = var.getType()->asResourceType();
        try
        {
            switch (pType->getType())
            {
            case ReflectionResourceType::Type::Texture:
            {
                Resource* pResource = boundResource(var);
                ref<Texture> pTexture = pResource ? pResource->asTexture() : nullptr;
                if (pTexture)
                    record("bind {} {} {} texture {}", mPass, path, index, textureId(pTexture.get()));
                else
                    record("bind {} {} {} null", mPass, path, index);
                break;
            }
            case ReflectionResourceType::Type::StructuredBuffer:
            case ReflectionResourceType::Type::RawBuffer:
            case ReflectionResourceType::Type::TypedBuffer:
            {
                Resource* pResource = boundResource(var);
                ref<Buffer> pBuffer = pResource ? pResource->asBuffer() : nullptr;
                if (pBuffer)
                    record("bind {} {} {} buffer {}", mPass, path, index, bufferId(pBuffer.get()));
                else
                    record("bind {} {} {} null", mPass, path, index);
                break;
            }
            case ReflectionResourceType::Type::Sampler:
            {
                ref<Sampler> pSampler = var.getSampler();
                if (pSampler)
                    record("bind {} {} {} sampler {}", mPass, path, index, samplerId(pSampler.get()));
                else
                    record("bind {} {} {} null", mPass, path, index);
                break;
            }
            case ReflectionResourceType::Type::AccelerationStructure:
                record("bind {} {} {} accel", mPass, path, index);
                break;
            case ReflectionResourceType::Type::ConstantBuffer:
            {
                // ConstantBuffer<T> and ParameterBlock<T>: a block with its own uniform data and members.
                ref<ParameterBlock> pBlock = var.getParameterBlock();
                if (!pBlock)
                {
                    record("bind {} {} {} null", mPass, path, index);
                    break;
                }
                const size_t size = pBlock->getSize();
                if (size > 0 && pBlock->getRawData())
                {
                    std::string file = fmt::format("data/uniform_{}_{}.bin", mPass, path);
                    writeFile(mDir / file, pBlock->getRawData(), size);
                    record("bind {} {} {} uniform {} {}", mPass, path, index, size, file);
                }
                walk(path, path, pBlock->getRootVar());
                break;
            }
            }
        }
        catch (const std::exception& e)
        {
            record("bind {} {} {} null", mPass, path, index);
            logWarning("NCR bake: {} {}: {}", mPass, path, e.what());
        }
    }

    uint32_t bufferId(const Buffer* pBuffer)
    {
        auto it = mBuffers.find(pBuffer);
        if (it != mBuffers.end())
            return it->second;
        const uint32_t id = (uint32_t)mBuffers.size();
        mBuffers[pBuffer] = id;
        std::vector<uint8_t> data(pBuffer->getSize());
        pBuffer->getBlob(data.data(), 0, data.size());
        writeFile(mDir / fmt::format("data/buffer{}.bin", id), data.data(), data.size());
        record("buffer {} {} {} {}", id, pBuffer->getSize(), pBuffer->getStructSize(), pBuffer->getElementCount());
        return id;
    }

    uint32_t textureId(const Texture* pTexture)
    {
        auto it = mTextures.find(pTexture);
        if (it != mTextures.end())
            return it->second;
        const uint32_t id = (uint32_t)mTextures.size();
        mTextures[pTexture] = id;
        const bool cube = pTexture->getType() == Resource::Type::TextureCube;
        const uint32_t layers = pTexture->getArraySize() * (cube ? 6 : 1);
        std::ofstream f(mDir / fmt::format("data/texture{}.bin", id), std::ios::binary);
        for (uint32_t layer = 0; layer < layers; ++layer)
        {
            for (uint32_t mip = 0; mip < pTexture->getMipCount(); ++mip)
            {
                const std::vector<uint8_t> data = mpRenderContext->readTextureSubresource(pTexture, pTexture->getSubresourceIndex(layer, mip));
                f.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());
            }
        }
        const char* type = cube ? "Cube" : pTexture->getType() == Resource::Type::Texture3D ? "3D" : "2D";
        record(
            "texture {} {} {} {} {} {} {} {}",
            id, type, to_string(pTexture->getFormat()), pTexture->getWidth(), pTexture->getHeight(), pTexture->getDepth(),
            pTexture->getMipCount(), layers
        );
        return id;
    }

    uint32_t samplerId(const Sampler* pSampler)
    {
        auto it = mSamplers.find(pSampler);
        if (it != mSamplers.end())
            return it->second;
        const uint32_t id = (uint32_t)mSamplers.size();
        mSamplers[pSampler] = id;
        const Sampler::Desc& d = pSampler->getDesc();
        record(
            "sampler {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            id, (int)d.minFilter, (int)d.magFilter, (int)d.mipFilter, (int)d.addressModeU, (int)d.addressModeV, (int)d.addressModeW,
            d.maxAnisotropy, d.lodBias, d.minLod, d.maxLod, (int)d.comparisonFunc, d.borderColor.x, d.borderColor.y, d.borderColor.z,
            d.borderColor.w
        );
        return id;
    }

    Device* mpDevice;
    RenderContext* mpRenderContext;
    std::filesystem::path mDir;
    std::ofstream mIndex;
    std::string mPass;
    std::map<const Buffer*, uint32_t> mBuffers;
    std::map<const Texture*, uint32_t> mTextures;
    std::map<const Sampler*, uint32_t> mSamplers;
};
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
    // Everything the frame recorded has to be on the GPU before the buffers are read back.
    pRenderContext->submit(true);

    Baker baker(mpDevice.get(), pRenderContext, gBake.dir);
    baker.record("frame {} {} {}", mFrameDim.x, mFrameDim.y, mFrameCount);

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
        baker.addPass(name, pPass);
        ++written;
    }

    // The ray batches by role. A pass like ConeTrace or CompactBatch runs on several of them per frame;
    // the bindings recorded above are those of its last call.
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
        baker.record("value {}sharedIO {}", prefix, pBatch->sharedIO ? 1 : 0);
        baker.record("value {}diffStride {}", prefix, pBatch->sharedIO ? pBatch->diffStride : 0);
    }
    baker.record("value renderMode {}", (int)mRenderMode);
    baker.record("value numSpecRays {}", mNumSpecRays);
    baker.record("value clusterReuse {}", useClusterReuse() ? 1 : 0);

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
    logInfo("NCR bake: wrote {} compute passes and their resources to {}", written, gBake.dir.string());
}

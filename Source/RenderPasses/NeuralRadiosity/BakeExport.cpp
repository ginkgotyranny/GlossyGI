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
//   before.ckpt, after.ckpt   the model before and after the frame (NRCK checkpoints). A frame that
//                      trains is replayed from before.ckpt; a bake of frame 0 starts, like the replay,
//                      with fresh optimizer moments, so after.ckpt is what the replay should arrive at.
//   before_working.bin the fp16 copies the training step computes with, before the frame (NRWK):
//                      a checkpoint does not hold them, and a fresh model's differ from its masters.
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
//
// Animated scenes: with NCR_BAKE_TRACK_RATE set (samples per second; the clock must step by that
// rate, scripts/ncr_smoke.py does) and a scene that has animations, the frames after the baked one
// are recorded as the bake's track, one loop of the scene's animation (Falcor wraps the scene's
// time at the longest animation). Sample k is the scene k frames after the baked frame:
//
//   track.txt          track <samples> <rate> <recordBytes> <loopSeconds>
//                      trackrange uniform 0 <byteOffset> <bytes>       part of the scene's uniform block (gScene)
//                      trackrange buffer <id> <byteOffset> <bytes>     part of a scene buffer
//                      trackrange geometry 0 <byteOffset> <bytes>      part of the geometry transforms: 12 floats
//                                                                      (as in the geometry records) per global instance
//                      The ranges are the bytes that differ from the baked frame in any sample.
//   track.bin          "NRTK", version, samples, recordBytes, rate (float), then one record per sample:
//                      the training schedule's event classes of that frame (uint32), the scene bounds
//                      (min, max: 6 floats), then the bytes of every range in the order of track.txt.
//
// Vertex and index buffers are not tracked: the scenes animate rigid transforms only, and a scene
// with skinned meshes or vertex caches is refused. A sample holds what is bound at each shader path
// in that frame, not what one buffer object holds: Falcor swaps its current and previous matrix
// buffers every frame (AnimationController::update), so the object bound as worldMatrices in the
// baked frame is prevWorldMatrices in the next one.
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
    float trackRate = 0.f;   // samples per second of the animation track; 0: none

    BakeSettings()
    {
        const char* d = std::getenv("NCR_BAKE_DIR");
        enabled = d && *d;
        if (!enabled)
            return;
        dir = d;
        if (const char* f = std::getenv("NCR_BAKE_FRAME"))
            frame = (uint32_t)std::atoi(f);
        if (const char* r = std::getenv("NCR_BAKE_TRACK_RATE"))
            trackRate = (float)std::atof(r);
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

    /// The scene's buffers an animation can change: everything bound in gScene but the vertices and
    /// indices, by id, with the shader path and array index they were first seen bound at.
    const std::map<uint32_t, std::pair<std::string, uint32_t>>& sceneBuffers() const { return mSceneBuffers; }

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
                {
                    const uint32_t id = bufferId(pBuffer.get());
                    record("{} buffer {}", where, id);
                    if (path.rfind("gScene.", 0) == 0 && path != "gScene.vertices.data" && path != "gScene.indexData.data")
                        mSceneBuffers.emplace(id, std::make_pair(path, index));
                }
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
    std::map<uint32_t, std::pair<std::string, uint32_t>> mSceneBuffers;
    std::map<std::string, std::string> mUniformFiles;
    std::vector<ref<const Object>> mKeepAlive;
};

std::unique_ptr<Baker> gBaker;

/// The 3x4 transform of every geometry instance, as the geometry records give it (zeros for what
/// is not a triangle mesh).
std::vector<uint8_t> geometryTransforms(const Scene& scene)
{
    const auto& matrices = scene.getAnimationController()->getGlobalMatrices();
    std::vector<float> values((size_t)scene.getGeometryInstanceCount() * 12, 0.f);
    for (uint32_t i = 0; i < scene.getGeometryInstanceCount(); ++i)
    {
        const GeometryInstanceData& instance = scene.getGeometryInstance(i);
        if (instance.getType() != GeometryType::TriangleMesh)
            continue;
        const float4x4& m = matrices[instance.globalMatrixID];
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col)
                values[(size_t)i * 12 + row * 4 + col] = m[row][col];
    }
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

/// Records the track of an animated scene (see the top of this file): the first capture is the
/// baked frame, every later one is kept as its differences from it.
class TrackRecorder
{
public:
    TrackRecorder(
        const std::filesystem::path& dir, const ref<Scene>& pScene, const ref<ParameterBlock>& pSceneBlock,
        const std::map<uint32_t, std::pair<std::string, uint32_t>>& buffers, float rate, double loopSeconds
    )
        : mDir(dir), mpScene(pScene), mpSceneBlock(pSceneBlock), mRate(rate), mLoopSeconds(loopSeconds)
    {
        mSamples = std::max(1u, (uint32_t)std::llround(loopSeconds * rate));
        mTargets.push_back({"uniform", 0, "", 0});
        for (const auto& [id, binding] : buffers)
            mTargets.push_back({"buffer", id, binding.first, binding.second});
        mTargets.push_back({"geometry", 0, "", 0});
    }

    uint32_t samples() const { return mSamples; }

    /// One sample; true once the track is complete and written.
    bool capture(RenderContext* pRenderContext, uint32_t events)
    {
        pRenderContext->submit(true);   // the scene updates some of its buffers on the GPU
        Sample sample;
        sample.events = events;
        const AABB& bounds = mpScene->getSceneBounds();
        const float box[6] = {bounds.minPoint.x, bounds.minPoint.y, bounds.minPoint.z, bounds.maxPoint.x, bounds.maxPoint.y, bounds.maxPoint.z};
        std::memcpy(sample.bounds, box, sizeof(box));
        for (size_t t = 0; t < mTargets.size(); ++t)
        {
            Target& target = mTargets[t];
            std::vector<uint8_t> now;
            if (target.kind == "uniform")
            {
                now.resize(mpSceneBlock->getSize());
                std::memcpy(now.data(), mpSceneBlock->getRawData(), now.size());
            }
            else if (target.kind == "buffer")
            {
                const ref<Buffer> pBuffer = boundBuffer(target);
                now.resize(pBuffer->getSize());
                pBuffer->getBlob(now.data(), 0, now.size());
            }
            else
            {
                now = geometryTransforms(*mpScene);
            }
            if (mSampleList.empty())
            {
                target.base = std::move(now);
                target.changed.assign((target.base.size() + kChunk - 1) / kChunk, false);
                continue;
            }
            if (now.size() != target.base.size())
                FALCOR_THROW("NCR bake: a scene buffer changed its size during the animation ({} {})", target.kind, target.id);
            for (size_t chunk = 0; chunk < target.changed.size(); ++chunk)
            {
                const size_t offset = chunk * kChunk, size = std::min(kChunk, now.size() - offset);
                if (std::memcmp(now.data() + offset, target.base.data() + offset, size) == 0)
                    continue;
                target.changed[chunk] = true;
                Difference d;
                d.target = (uint32_t)t;
                d.chunk = (uint32_t)chunk;
                std::memcpy(d.bytes, now.data() + offset, size);
                sample.differences.push_back(d);
                mStoredBytes += sizeof(Difference);
            }
        }
        if (mStoredBytes > (size_t(3) << 30))
            FALCOR_THROW("NCR bake: the animation changes too much scene data per frame for a track (over 3 GB of differences)");
        mSampleList.push_back(std::move(sample));
        if (mSampleList.size() < mSamples)
            return false;
        write();
        return true;
    }

private:
    static constexpr size_t kChunk = 16;
    struct Target
    {
        std::string kind;
        uint32_t id;
        std::string path;   // a buffer: where it is bound, e.g. gScene.worldMatrices, and the array element
        uint32_t index;
        std::vector<uint8_t> base;
        std::vector<bool> changed;   // per chunk: differs from the base in some sample
    };
    struct Difference
    {
        uint32_t target, chunk;
        uint8_t bytes[kChunk];
    };
    struct Sample
    {
        uint32_t events = 0;
        float bounds[6] = {};
        std::vector<Difference> differences;
    };
    struct Range
    {
        uint32_t target, offset, bytes, recordOffset;
    };

    /// The buffer bound at a target's shader path now. The path goes through the scene's block and
    /// the blocks inside it (gScene.materials.materialData).
    ref<Buffer> boundBuffer(const Target& target) const
    {
        ShaderVar var = mpSceneBlock->getRootVar();
        std::string rest = target.path.substr(target.path.find('.') + 1);   // without "gScene."
        while (true)
        {
            const size_t dot = rest.find('.');
            var = member(var, rest.substr(0, dot));
            if (dot == std::string::npos)
                break;
            rest = rest.substr(dot + 1);
        }
        if (var.getType()->getKind() == ReflectionType::Kind::Array)
            var = var[target.index];
        Resource* pResource = boundResource(var);
        ref<Buffer> pBuffer = pResource ? pResource->asBuffer() : nullptr;
        if (!pBuffer)
            FALCOR_THROW("NCR bake: nothing is bound at {} during the animation", target.path);
        return pBuffer;
    }

    static ShaderVar member(ShaderVar var, const std::string& name)
    {
        const ReflectionResourceType* pResource = var.getType()->asResourceType();
        if (pResource && pResource->getType() == ReflectionResourceType::Type::ConstantBuffer)
        {
            ref<ParameterBlock> pBlock = var.getParameterBlock();
            if (!pBlock)
                FALCOR_THROW("NCR bake: no parameter block bound where the track needs one");
            var = pBlock->getRootVar();
        }
        return var[name];
    }

    void write()
    {
        // The ranges: runs of changed chunks.
        std::vector<Range> ranges;
        std::vector<std::vector<int64_t>> chunkAt(mTargets.size());   // target, chunk -> offset in the record, or -1
        uint32_t recordBytes = sizeof(uint32_t) + 6 * sizeof(float);
        for (size_t t = 0; t < mTargets.size(); ++t)
        {
            const Target& target = mTargets[t];
            chunkAt[t].assign(target.changed.size(), -1);
            size_t chunk = 0;
            while (chunk < target.changed.size())
            {
                if (!target.changed[chunk])
                {
                    ++chunk;
                    continue;
                }
                const size_t first = chunk;
                while (chunk < target.changed.size() && target.changed[chunk])
                {
                    chunkAt[t][chunk] = (int64_t)recordBytes + (int64_t)((chunk - first) * kChunk);
                    ++chunk;
                }
                Range r;
                r.target = (uint32_t)t;
                r.offset = (uint32_t)(first * kChunk);
                r.bytes = (uint32_t)(std::min(chunk * kChunk, target.base.size()) - first * kChunk);
                r.recordOffset = recordBytes;
                recordBytes += r.bytes;
                ranges.push_back(r);
            }
        }
        std::ofstream index(mDir / "track.txt", std::ios::trunc);
        index << fmt::format("track {} {} {} {}\n", mSamples, mRate, recordBytes, mLoopSeconds);
        for (const Range& r : ranges)
            index << fmt::format("trackrange {} {} {} {}\n", mTargets[r.target].kind, mTargets[r.target].id, r.offset, r.bytes);

        // The record of the baked frame, then every sample as that record with its differences.
        std::vector<uint8_t> baseRecord(recordBytes, 0);
        for (const Range& r : ranges)
            std::memcpy(baseRecord.data() + r.recordOffset, mTargets[r.target].base.data() + r.offset, r.bytes);
        std::ofstream f(mDir / "track.bin", std::ios::binary | std::ios::trunc);
        const uint32_t header[4] = {0x4B54524Eu /* "NRTK" */, 1u, mSamples, recordBytes};
        f.write(reinterpret_cast<const char*>(header), sizeof(header));
        f.write(reinterpret_cast<const char*>(&mRate), sizeof(float));
        for (size_t s = 0; s < mSampleList.size(); ++s)
        {
            const Sample& sample = mSampleList[s];
            std::vector<uint8_t> record = baseRecord;
            // The baked frame's own events are those of a scene that has just been loaded: take the
            // next frame's, which are the animation's.
            const uint32_t events = (s == 0 && mSampleList.size() > 1) ? mSampleList[1].events : sample.events;
            std::memcpy(record.data(), &events, sizeof(uint32_t));
            std::memcpy(record.data() + sizeof(uint32_t), sample.bounds, sizeof(sample.bounds));
            for (const Difference& d : sample.differences)
            {
                const int64_t at = chunkAt[d.target][d.chunk];
                const size_t size = std::min(kChunk, mTargets[d.target].base.size() - (size_t)d.chunk * kChunk);
                std::memcpy(record.data() + at, d.bytes, size);
            }
            f.write(reinterpret_cast<const char*>(record.data()), (std::streamsize)record.size());
        }
        if (!f.good() || !index.good())
            FALCOR_THROW("NCR bake: cannot write the track to {}", mDir.string());
        logInfo(
            "NCR bake: track of {} samples at {} per second ({} s), {} ranges, {} bytes per sample, written to {}",
            mSamples, mRate, mLoopSeconds, ranges.size(), recordBytes, mDir.string()
        );
    }

    std::filesystem::path mDir;
    ref<Scene> mpScene;
    ref<ParameterBlock> mpSceneBlock;
    float mRate;
    double mLoopSeconds;
    uint32_t mSamples = 0;
    std::vector<Target> mTargets;
    std::vector<Sample> mSampleList;
    size_t mStoredBytes = 0;
};

std::unique_ptr<TrackRecorder> gTrack;
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
    if (mpNRModel)
    {
        mpNRModel->saveState((gBake.dir / "before.ckpt").string());
        mpNRModel->saveWorkingCopies((gBake.dir / "before_working.bin").string());
    }
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
    // How fast the desktop's first-person camera moves (units per second), for the phone's own controls.
    baker.record("value cameraSpeed {}", mpScene->getCameraSpeed());
    baker.record("value numSpecRays {}", mNumSpecRays);
    baker.record("value numRHS {}", mNumRHS);
    baker.record("value clusterReuse {}", useClusterReuse() ? 1 : 0);
    if (mpNRModel)
    {
        // The optimizer settings the frame's training step ran with (the training schedule sets them).
        baker.record("value train.learningRate {}", mpNRModel->learningRate());
        // The training schedule (updateTrainSchedule): its settings and its state after this frame,
        // from which the replay continues frame after frame.
        baker.record("value train.schedule {}", mEnableTrainSchedule ? 1 : 0);
        baker.record("value train.heat {}", mTrainHeat);
        baker.record("value train.holdLeft {}", mTrainHoldLeft);
        baker.record("value train.lrHot {}", mTrainLrHot);
        baker.record("value train.lrCold {}", mTrainLrCold);
        baker.record("value train.emaHot {}", mTrainEmaHot);
        baker.record("value train.emaCold {}", mTrainEmaCold);
        baker.record("value train.warmupFrames {}", mTrainWarmupFrames);
        baker.record("value train.holdFrames {}", mTrainHoldFrames);
        baker.record("value train.coolFrames {}", mTrainCoolFrames);
        baker.record("value train.cameraHeat {}", mTrainCameraHeat);
        baker.record("value train.sceneHeat {}", mTrainSceneHeat);
        baker.record("value train.lightMotionHeat {}", mTrainLightMotionHeat);
        baker.record("value train.objectMotionHeat {}", mTrainObjectMotionHeat);
        baker.record("value train.emaDecay {}", mpNRModel->emaDecay());
        mpNRModel->saveState((gBake.dir / "after.ckpt").string());
    }

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

    // An animated scene: this frame is the first sample of the track, the following frames the rest.
    double loopSeconds = 0.0;
    for (const auto& pAnimation : mpScene->getAnimations())
        loopSeconds = std::max(loopSeconds, pAnimation->getDuration());
    if (gBake.trackRate > 0.f && loopSeconds > 0.0)
    {
        const auto pController = mpScene->getAnimationController();
        if (pController->hasSkinnedMeshes() || pController->hasAnimatedVertexCaches())
            FALCOR_THROW("NCR bake: the track holds rigid animation only; this scene animates vertices");
        const ref<ParameterBlock> pSceneBlock = mpRandomSmooth->getRootVar()["gScene"].getParameterBlock();
        gTrack = std::make_unique<TrackRecorder>(gBake.dir, mpScene, pSceneBlock, baker.sceneBuffers(), gBake.trackRate, loopSeconds);
        gTrack->capture(pRenderContext, 0);
        logInfo("NCR bake: recording a track of {} samples", gTrack->samples());
    }
    gBaker.reset();
}

void NeuralRadiosity::bakeTrack(RenderContext* pRenderContext)
{
    if (!gTrack || mFrameCount == gBake.frame)
        return;
    if (gTrack->capture(pRenderContext, mLastTrainEvents))
        gTrack.reset();
}

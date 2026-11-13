// A packed Falcor bake as live Vulkan objects: see ohos/tools/bake_pack.py for the index format.
//
// The bake is a recorded frame: an ordered list of compute-pass calls, each with the resources it
// had bound, and of the steps Falcor did on the host in between. A pass is a pipeline; every call of
// it has its own descriptor sets and its own copy of the pass's uniform block.
#pragma once
#include "replay_vk.h"

namespace replay
{
struct Binding
{
    uint32_t set = 0, binding = 0, count = 1;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    std::string name;   // shader path, e.g. gScene.materials.materialTextures
};

struct Element
{
    uint32_t set = 0, binding = 0, index = 0;
    std::string kind;   // buffer, texture, sampler, uniform, accel, null
    uint32_t id = 0;    // buffer / texture / sampler id; byte size of a uniform block
    std::string file;   // uniform block contents
};

struct Geometry
{
    uint32_t id, tlasInstance, geometryIndex, vbOffset, ibOffset, vertexCount, indexCount, index16, frontFaceCW, opaque;
    float transform[12];
};

struct Pass
{
    std::string name;
    uint32_t group[3] = {1, 1, 1};
    std::vector<Binding> bindings;
    std::map<std::string, std::map<std::string, std::pair<uint32_t, uint32_t>>> uniforms;   // block -> member -> (offset, size)
    std::vector<VkDescriptorSetLayout> setLayouts;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    double createMs = 0.0;
};

/// One execution of a pass in the recorded frame.
struct Call
{
    uint32_t step = 0;
    std::string pass;
    uint32_t dims[3] = {1, 1, 1};                    // as Falcor dispatched it; group counts are dims / group size, rounded up
    std::string batch;                               // render, lhs, rhs or -: the ray batch whose buffers it has bound
    std::vector<Element> elements;
    std::vector<VkDescriptorSet> sets;
    std::map<std::string, Buffer*> uniformBlocks;    // block path -> its buffer
};

/// One entry of the recorded frame: a call, or a host step with its arguments.
struct Step
{
    uint32_t step = 0;
    bool isCall = false;
    std::vector<std::string> host;   // e.g. {"prefixsum", "render", "diff", "230400"}
};

class Bake
{
public:
    Device& dev;
    std::string dir;
    uint32_t width = 0, height = 0, frameCount = 0;
    std::map<uint32_t, Buffer> buffers;
    std::map<uint32_t, bool> bufferLoaded;      // false: a ray-batch buffer the passes write
    std::map<uint32_t, Texture> textures;
    std::map<uint32_t, VkSampler> samplers;
    std::vector<Geometry> geometry;
    uint32_t vertexBuffer = UINT32_MAX, vertexStride = 0, indexBuffer = UINT32_MAX;
    std::vector<std::string> passOrder;
    std::map<std::string, Pass> passes;
    std::map<uint32_t, Call> calls;             // by step
    std::vector<Step> program;                  // the frame, in execution order
    std::map<std::string, uint32_t> named;      // role, e.g. render.specInput -> buffer id
    std::map<std::string, double> values;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;

    Bake(Device& device, const std::string& folder) : dev(device), dir(folder) {}

    void load()
    {
        parseIndex();
        createResources();
        buildAccelerationStructures();
        createPasses();
    }

    bool hasBuffer(const std::string& role) const { return named.count(role) != 0; }

    Buffer& buffer(const std::string& role)
    {
        auto it = named.find(role);
        if (it == named.end())
        {
            std::printf("FAIL the bake has no buffer named %s\n", role.c_str());
            std::exit(2);
        }
        return buffers.at(it->second);
    }

    double value(const std::string& name, double fallback = 0.0) const
    {
        auto it = values.find(name);
        return it == values.end() ? fallback : it->second;
    }

    /// Location of a uniform member of a call, or null.
    void* uniform(Call& c, const std::string& block, const std::string& member, uint32_t size)
    {
        Pass& p = passes.at(c.pass);
        auto b = p.uniforms.find(block);
        if (b == p.uniforms.end()) return nullptr;
        auto m = b->second.find(member);
        auto buffer = c.uniformBlocks.find(block);
        if (m == b->second.end() || buffer == c.uniformBlocks.end() || m->second.second < size) return nullptr;
        return buffer->second->as<char>() + m->second.first;
    }

    bool hasUniform(Call& c, const std::string& block, const std::string& member) { return uniform(c, block, member, 1) != nullptr; }

    template <typename T> T getUniform(Call& c, const std::string& block, const std::string& member)
    {
        T value = {};
        void* at = uniform(c, block, member, sizeof(T));
        if (!at)
        {
            std::printf("FAIL %s has no uniform %s.%s\n", c.pass.c_str(), block.c_str(), member.c_str());
            std::exit(2);
        }
        std::memcpy(&value, at, sizeof(T));
        return value;
    }

    template <typename T> void setUniform(Call& c, const std::string& block, const std::string& member, const T& value)
    {
        void* at = uniform(c, block, member, sizeof(T));
        if (!at)
        {
            std::printf("FAIL %s has no uniform %s.%s\n", c.pass.c_str(), block.c_str(), member.c_str());
            std::exit(2);
        }
        std::memcpy(at, &value, sizeof(T));
    }

    /// Records one dispatch of a call over `dims` items (Falcor's ComputePass::execute convention:
    /// the group counts are the dimensions divided by the thread group size, rounded up), followed
    /// by a barrier so that the next pass sees what this one wrote.
    void dispatch(VkCommandBuffer cb, Call& c, uint32_t dimX, uint32_t dimY = 1, uint32_t dimZ = 1)
    {
        Pass& p = passes.at(c.pass);
        const uint32_t groups[3] = {(dimX + p.group[0] - 1) / p.group[0], (dimY + p.group[1] - 1) / p.group[1], (dimZ + p.group[2] - 1) / p.group[2]};
        for (int a = 0; a < 3; a++)
        {
            // A driver runs only as many groups as it supports and drops the rest without a word.
            if (groups[a] > dev.maxGroupCount[a])
            {
                std::printf("FAIL %s: %u thread groups in dimension %d, the device runs at most %u\n", p.name.c_str(), groups[a], a, dev.maxGroupCount[a]);
                std::exit(2);
            }
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, (uint32_t)c.sets.size(), c.sets.data(), 0, nullptr);
        vkCmdDispatch(cb, groups[0], groups[1], groups[2]);
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

private:
    Buffer mDummyBuffer;
    Texture mDummyTexture;
    VkSampler mDummySampler = VK_NULL_HANDLE;
    std::map<std::string, Buffer> mUniformBuffers;   // key: see uniformKey()
    std::vector<Buffer> mAccelerationStorage;
    VkDescriptorPool mPool = VK_NULL_HANDLE;

    void parseIndex()
    {
        std::ifstream index(dir + "/replay.txt");
        if (!index)
        {
            std::printf("FAIL cannot read %s/replay.txt (run ohos/tools/bake_pack.py on the bake)\n", dir.c_str());
            std::exit(2);
        }
        std::string line;
        while (std::getline(index, line))
        {
            std::istringstream in(line);
            std::string kind;
            in >> kind;
            if (kind == "frame")
            {
                in >> width >> height >> frameCount;
            }
            else if (kind == "buffer")
            {
                uint32_t id, structSize, load;
                uint64_t bytes;
                in >> id >> bytes >> structSize >> load;
                Buffer b;
                b.size = bytes;
                buffers[id] = b;
                bufferLoaded[id] = load != 0;
            }
            else if (kind == "texture")
            {
                uint32_t id, format, storage;
                std::string type;
                Texture t;
                in >> id >> type >> format >> t.bytesPerPixel >> t.width >> t.height >> t.depth >> t.mips >> t.layers >> storage;
                t.format = (VkFormat)format;
                t.cube = type == "Cube";
                t.storage = storage != 0;
                textures[id] = t;
            }
            else if (kind == "sampler")
            {
                uint32_t id, minFilter, magFilter, mipFilter, addressU, addressV, addressW, anisotropy;
                float lodBias, minLod, maxLod;
                in >> id >> minFilter >> magFilter >> mipFilter >> addressU >> addressV >> addressW >> anisotropy >> lodBias >> minLod >> maxLod;
                // Falcor: filtering Point 0 / Linear 1; addressing Wrap, Mirror, Clamp, Border, MirrorOnce.
                static const VkSamplerAddressMode modes[5] = {VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                              VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE};
                VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                ci.minFilter = minFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                ci.magFilter = magFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                ci.mipmapMode = mipFilter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                ci.addressModeU = modes[std::min(addressU, 4u)];
                ci.addressModeV = modes[std::min(addressV, 4u)];
                ci.addressModeW = modes[std::min(addressW, 4u)];
                ci.anisotropyEnable = anisotropy > 1;
                ci.maxAnisotropy = (float)anisotropy;
                ci.mipLodBias = lodBias;
                ci.minLod = std::max(minLod, 0.0f);
                ci.maxLod = maxLod;
                VK_CHECK(vkCreateSampler(dev.device, &ci, nullptr, &samplers[id]));
            }
            else if (kind == "scene")
            {
                std::string what;
                in >> what;
                if (what == "vertices") in >> vertexBuffer >> vertexStride;
                else if (what == "indices") in >> indexBuffer;
            }
            else if (kind == "geometry")
            {
                Geometry g;
                in >> g.id >> g.tlasInstance >> g.geometryIndex >> g.vbOffset >> g.ibOffset >> g.vertexCount >> g.indexCount >> g.index16 >>
                    g.frontFaceCW >> g.opaque;
                for (float& f : g.transform) in >> f;
                geometry.push_back(g);
            }
            else if (kind == "pass")
            {
                Pass p;
                in >> p.name >> p.group[0] >> p.group[1] >> p.group[2];
                passOrder.push_back(p.name);
                passes[p.name] = p;
            }
            else if (kind == "binding")
            {
                std::string passName;
                uint32_t type;
                Binding b;
                in >> passName >> b.set >> b.binding >> type >> b.count >> b.name;
                b.type = (VkDescriptorType)type;
                passes[passName].bindings.push_back(b);
            }
            else if (kind == "uniform")
            {
                std::string passName, block, member;
                uint32_t offset, size;
                in >> passName >> block >> member >> offset >> size;
                passes[passName].uniforms[block][member] = {offset, size};
            }
            else if (kind == "call")
            {
                Call c;
                in >> c.step >> c.pass >> c.dims[0] >> c.dims[1] >> c.dims[2] >> c.batch;
                calls[c.step] = c;
                Step s;
                s.step = c.step;
                s.isCall = true;
                program.push_back(s);
            }
            else if (kind == "element")
            {
                uint32_t step;
                Element e;
                in >> step >> e.set >> e.binding >> e.index >> e.kind;
                if (e.kind == "uniform") in >> e.id >> e.file;
                else if (e.kind != "accel" && e.kind != "null") in >> e.id;
                calls[step].elements.push_back(e);
            }
            else if (kind == "host")
            {
                Step s;
                in >> s.step;
                std::string word;
                while (in >> word) s.host.push_back(word);
                program.push_back(s);
            }
            else if (kind == "named")
            {
                std::string role;
                uint32_t id;
                in >> role >> id;
                named[role] = id;
            }
            else if (kind == "value")
            {
                std::string name;
                double value;
                in >> name >> value;
                values[name] = value;
            }
        }
    }

    void createResources()
    {
        // Every storage buffer can also feed an acceleration structure build and be copied.
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        size_t loadedBytes = 0, scratchBytes = 0;
        for (auto& [id, b] : buffers)
        {
            const VkDeviceSize size = std::max<VkDeviceSize>(b.size, 16);
            b = dev.makeBuffer(size, usage, true);
            std::memset(b.data, 0, (size_t)size);
            if (bufferLoaded[id])
            {
                const std::vector<char> data = readFile(dir + "/data/buffer" + std::to_string(id) + ".bin");
                std::memcpy(b.data, data.data(), std::min<size_t>(data.size(), (size_t)size));
                loadedBytes += data.size();
            }
            else
            {
                scratchBytes += (size_t)size;
            }
        }
        size_t textureBytes = 0;
        for (auto& [id, t] : textures)
        {
            Texture created = dev.makeTexture(t.format, t.bytesPerPixel, t.width, t.height, t.depth, t.mips, t.layers, t.cube, t.storage);
            if (t.storage)
            {
                dev.initTexture(created, nullptr);   // an output: starts cleared, the passes fill it
            }
            else
            {
                const std::vector<char> data = readFile(dir + "/data/texture" + std::to_string(id) + ".bin");
                dev.initTexture(created, &data);
                textureBytes += data.size();
            }
            t = created;
        }
        mDummyBuffer = dev.makeBuffer(4096, usage | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
        std::memset(mDummyBuffer.data, 0, 4096);
        mDummyTexture = dev.makeTexture(VK_FORMAT_R32G32B32A32_SFLOAT, 16, 1, 1, 1, 1, 1, false, true);
        dev.initTexture(mDummyTexture, nullptr);
        VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        VK_CHECK(vkCreateSampler(dev.device, &sci, nullptr, &mDummySampler));
        std::printf("bake: %zu buffers (%.1f MB loaded, %.1f MB written by the passes), %zu textures (%.1f MB loaded), %zu samplers\n",
                    buffers.size(), loadedBytes / 1e6, scratchBytes / 1e6, textures.size(), textureBytes / 1e6, samplers.size());
    }

    VkAccelerationStructureKHR buildAS(VkAccelerationStructureTypeKHR type, const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
                                       const std::vector<uint32_t>& primitiveCounts)
    {
        VkAccelerationStructureBuildGeometryInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = type;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = (uint32_t)geometries.size();
        info.pGeometries = geometries.data();
        VkAccelerationStructureBuildSizesInfoKHR sizes = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        dev.getBuildSizes(dev.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, primitiveCounts.data(), &sizes);

        mAccelerationStorage.push_back(dev.makeBuffer(
            sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false));
        VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        ci.buffer = mAccelerationStorage.back().buffer;
        ci.size = sizes.accelerationStructureSize;
        ci.type = type;
        VkAccelerationStructureKHR as;
        VK_CHECK(dev.createAS(dev.device, &ci, nullptr, &as));

        Buffer scratch = dev.makeBuffer(sizes.buildScratchSize + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);
        info.dstAccelerationStructure = as;
        info.scratchData.deviceAddress = (dev.address(scratch) + 255) & ~VkDeviceAddress(255);
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geometries.size());
        for (size_t i = 0; i < ranges.size(); i++) ranges[i] = {primitiveCounts[i], 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
        VkCommandBuffer cb = dev.begin();
        dev.cmdBuildAS(cb, 1, &info, &pRanges);
        dev.submit(cb);
        dev.destroy(scratch);
        return as;
    }

    /// Falcor's layout: one bottom-level structure per top-level instance, its geometries in
    /// geometryIndex order, and the instance's custom index the global id of its first geometry,
    /// so that InstanceID() + GeometryIndex() of a hit is the global geometry instance id.
    void buildAccelerationStructures()
    {
        if (geometry.empty() || vertexBuffer == UINT32_MAX)
        {
            std::printf("FAIL the bake has no geometry\n");
            std::exit(2);
        }
        const auto start = std::chrono::steady_clock::now();
        const VkDeviceAddress vertices = dev.address(buffers.at(vertexBuffer));
        const VkDeviceAddress indices = indexBuffer != UINT32_MAX ? dev.address(buffers.at(indexBuffer)) : 0;
        std::map<uint32_t, std::vector<Geometry>> instances;
        for (const Geometry& g : geometry) instances[g.tlasInstance].push_back(g);

        std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
        uint64_t triangles = 0;
        for (auto& [index, list] : instances)
        {
            std::sort(list.begin(), list.end(), [](const Geometry& a, const Geometry& b) { return a.geometryIndex < b.geometryIndex; });
            std::vector<VkAccelerationStructureGeometryKHR> geometries;
            std::vector<uint32_t> counts;
            for (const Geometry& g : list)
            {
                VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
                geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                geo.flags = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
                auto& t = geo.geometry.triangles;
                t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                t.vertexData.deviceAddress = vertices + (VkDeviceAddress)g.vbOffset * vertexStride;
                t.vertexStride = vertexStride;
                t.maxVertex = g.vertexCount - 1;
                if (g.indexCount > 0)
                {
                    t.indexType = g.index16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                    t.indexData.deviceAddress = indices + (VkDeviceAddress)g.ibOffset * 4;
                }
                else
                {
                    t.indexType = VK_INDEX_TYPE_NONE_KHR;
                }
                geometries.push_back(geo);
                counts.push_back((g.indexCount > 0 ? g.indexCount : g.vertexCount) / 3);
                triangles += counts.back();
            }
            VkAccelerationStructureKHR blas = buildAS(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometries, counts);
            VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            addressInfo.accelerationStructure = blas;
            VkAccelerationStructureInstanceKHR instance = {};
            std::memcpy(instance.transform.matrix, list[0].transform, sizeof(float) * 12);
            instance.instanceCustomIndex = list[0].id;
            instance.mask = 0xff;
            instance.flags = list[0].frontFaceCW ? VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR : 0;
            instance.accelerationStructureReference = dev.getASAddress(dev.device, &addressInfo);
            tlasInstances.push_back(instance);
        }
        Buffer instanceBuffer = dev.makeBuffer(
            tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true);
        std::memcpy(instanceBuffer.data, tlasInstances.data(), (size_t)instanceBuffer.size);
        VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geo.geometry.instances.data.deviceAddress = dev.address(instanceBuffer);
        tlas = buildAS(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, {geo}, {(uint32_t)tlasInstances.size()});
        dev.destroy(instanceBuffer);
        std::printf("acceleration structures: %zu geometries, %llu triangles, %zu instances, built in %.1f ms\n", geometry.size(),
                    (unsigned long long)triangles, tlasInstances.size(), msSince(start));
    }

    /// The scene's blocks are the same data in every call: one buffer for all of them, so that the
    /// camera is updated in one place. A pass's own block belongs to the call: the same pass runs
    /// with different values at different points of the frame.
    static std::string uniformKey(const Call& c, const std::string& block)
    {
        return block.rfind("gScene", 0) == 0 ? block : "call" + std::to_string(c.step) + ":" + block;
    }

    void createPasses()
    {
        // One pool for the descriptor sets of every call.
        std::map<VkDescriptorType, uint32_t> totals;
        uint32_t setCount = 0;
        for (auto& [step, c] : calls)
        {
            uint32_t maxSet = 0;
            for (const Binding& b : passes.at(c.pass).bindings)
            {
                totals[b.type] += b.count;
                maxSet = std::max(maxSet, b.set);
            }
            setCount += maxSet + 1;
        }
        std::vector<VkDescriptorPoolSize> sizes;
        for (auto& [type, count] : totals) sizes.push_back({type, count});
        VkDescriptorPoolCreateInfo pci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = setCount;
        pci.poolSizeCount = (uint32_t)sizes.size();
        pci.pPoolSizes = sizes.data();
        VK_CHECK(vkCreateDescriptorPool(dev.device, &pci, nullptr, &mPool));

        double totalMs = 0.0;
        for (const std::string& name : passOrder)
        {
            Pass& p = passes[name];
            uint32_t maxSet = 0;
            for (const Binding& b : p.bindings) maxSet = std::max(maxSet, b.set);
            p.setLayouts.resize(maxSet + 1);
            for (uint32_t set = 0; set <= maxSet; set++)
            {
                std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
                for (const Binding& b : p.bindings)
                    if (b.set == set) layoutBindings.push_back({b.binding, b.type, b.count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
                VkDescriptorSetLayoutCreateInfo sl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                sl.bindingCount = (uint32_t)layoutBindings.size();
                sl.pBindings = layoutBindings.data();
                VK_CHECK(vkCreateDescriptorSetLayout(dev.device, &sl, nullptr, &p.setLayouts[set]));
            }
            VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pl.setLayoutCount = (uint32_t)p.setLayouts.size();
            pl.pSetLayouts = p.setLayouts.data();
            VK_CHECK(vkCreatePipelineLayout(dev.device, &pl, nullptr, &p.layout));

            const std::vector<char> code = readFile(dir + "/spirv/" + name + ".spv");
            VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            sm.codeSize = code.size();
            sm.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            VK_CHECK(vkCreateShaderModule(dev.device, &sm, nullptr, &module));
            VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            ci.layout = p.layout;
            const auto start = std::chrono::steady_clock::now();
            VK_CHECK(vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline));
            p.createMs = msSince(start);
            totalMs += p.createMs;
            vkDestroyShaderModule(dev.device, module, nullptr);
        }
        for (auto& [step, c] : calls)
        {
            Pass& p = passes.at(c.pass);
            c.sets.resize(p.setLayouts.size());
            for (size_t set = 0; set < p.setLayouts.size(); set++)
            {
                VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                da.descriptorPool = mPool;
                da.descriptorSetCount = 1;
                da.pSetLayouts = &p.setLayouts[set];
                VK_CHECK(vkAllocateDescriptorSets(dev.device, &da, &c.sets[set]));
            }
            writeDescriptors(p, c);
        }
        std::printf("pipelines: %zu passes created in %.0f ms; %zu calls in the recorded frame\n", passOrder.size(), totalMs, calls.size());
    }

    void writeDescriptors(Pass& p, Call& c)
    {
        for (const Element& e : c.elements)
        {
            const Binding* binding = nullptr;
            for (const Binding& b : p.bindings)
                if (b.set == e.set && b.binding == e.binding) binding = &b;
            if (!binding) continue;

            VkDescriptorBufferInfo bufferInfo = {mDummyBuffer.buffer, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo imageInfo = {mDummySampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSetAccelerationStructureKHR accelInfo = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            accelInfo.accelerationStructureCount = 1;
            accelInfo.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = c.sets[e.set];
            w.dstBinding = e.binding;
            w.dstArrayElement = e.index;
            w.descriptorCount = 1;
            w.descriptorType = binding->type;

            switch (binding->type)
            {
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                if (e.kind == "buffer") bufferInfo.buffer = buffers.at(e.id).buffer;
                w.pBufferInfo = &bufferInfo;
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                if (e.kind == "uniform")
                {
                    const std::string key = uniformKey(c, binding->name);
                    auto it = mUniformBuffers.find(key);
                    if (it == mUniformBuffers.end())
                    {
                        Buffer b = dev.makeBuffer(std::max<VkDeviceSize>(e.id, 16), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
                        const std::vector<char> data = readFile(dir + "/" + e.file);
                        std::memcpy(b.data, data.data(), std::min<size_t>(data.size(), (size_t)b.size));
                        it = mUniformBuffers.emplace(key, b).first;
                    }
                    c.uniformBlocks[binding->name] = &it->second;
                    bufferInfo.buffer = it->second.buffer;
                }
                w.pBufferInfo = &bufferInfo;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            {
                const bool storage = binding->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                const Texture& t = e.kind == "texture" ? textures.at(e.id) : mDummyTexture;
                imageInfo.sampler = VK_NULL_HANDLE;
                imageInfo.imageView = storage ? t.storageView : t.view;
                w.pImageInfo = &imageInfo;
                break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                if (e.kind == "sampler") imageInfo.sampler = samplers.at(e.id);
                w.pImageInfo = &imageInfo;
                break;
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                w.pNext = &accelInfo;
                break;
            default:
                std::printf("FAIL %s: descriptor type %d of %s is not handled\n", p.name.c_str(), (int)binding->type, binding->name.c_str());
                std::exit(2);
            }
            vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);   // the infos above are alive until here
        }
    }
};
}  // namespace replay

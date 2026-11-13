// Stand-alone Vulkan probe for the GPU side of NCR on the phone, run from a shell (see run.sh).
// It answers, on the device itself:
//   info      which features and limits the driver reports (16-bit storage, fp16, 64-bit ints,
//             subgroups, ray query, external memory)
//   encode    primary and cluster hash-grid lookups in compute: result vs a CPU reference, time
//   scatter   hash-grid gradient scatter with integer atomics: result vs a CPU reference, time
//   adam      one dense optimizer pass over a grid: time
//   rays      inline ray query throughput in a closed 196,608-triangle box (every ray must hit)
//   features  subgroup arithmetic and groupshared memory in a Slang compute shader
//   features64 the same shader with 64-bit integers (needs shaderInt64)
// The shaders are the .slang files next to this one, compiled to SPIR-V by Falcor's slangc.
//
//   gpu_probe [test ...]        default: all of the above, in that order
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s -> %d (line %d)\n", #x, (int)r_, __LINE__); exit(2); } } while (0)

static VkPhysicalDevice gPhysical;
static VkDevice gDev;
static VkQueue gQueue;
static VkCommandPool gPool;
static VkDescriptorPool gDescriptorPool;
static VkPhysicalDeviceMemoryProperties gMem;
static bool gRayQuery = false;

static PFN_vkGetAccelerationStructureBuildSizesKHR pGetBuildSizes;
static PFN_vkCreateAccelerationStructureKHR pCreateAS;
static PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuildAS;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pGetASAddress;

// ---------------------------------------------------------------------------------------------
// Small helpers
static double msSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static uint32_t gSeed = 2024;
static uint32_t nextU32()
{
    gSeed = gSeed * 1664525u + 1013904223u;
    return gSeed;
}
static float rnd()   // [0, 1)
{
    return float(nextU32() >> 8) * (1.0f / 16777216.0f);
}

static float halfToFloat(uint16_t h)
{
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

// Truncating conversion; the probe only needs some fp16 value near f, and reads it back exactly.
static uint16_t floatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000u;
    const int exponent = int((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent <= 0) return sign;
    if (exponent >= 31) return sign | 0x7c00u;
    return sign | uint16_t(exponent << 10) | uint16_t((bits >> 13) & 0x3ffu);
}

static std::vector<uint32_t> loadSpv(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<uint32_t> code(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), code.size() * 4);
    return code;
}

static uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < gMem.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (gMem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* data = nullptr;
    VkDeviceSize size = 0;
    template <typename T> T* as() { return static_cast<T*>(data); }
};

// Host-visible by default, so that the CPU can fill and check the buffer. The host-visible types of
// this driver are DEVICE_LOCAL as well, so this costs the GPU nothing.
static Buffer makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible = true)
{
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    CHK(vkCreateBuffer(gDev, &ci, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gDev, b.buffer, &req);
    VkMemoryAllocateFlagsInfo flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext = &flags;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = hostVisible ? memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                     : memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX) ai.memoryTypeIndex = memoryType(req.memoryTypeBits, 0);
    CHK(vkAllocateMemory(gDev, &ai, nullptr, &b.mem));
    CHK(vkBindBufferMemory(gDev, b.buffer, b.mem, 0));
    if (hostVisible) CHK(vkMapMemory(gDev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.data));
    return b;
}

static void freeBuffer(Buffer& b)
{
    if (b.buffer) vkDestroyBuffer(gDev, b.buffer, nullptr);
    if (b.mem) vkFreeMemory(gDev, b.mem, nullptr);
    b = Buffer();
}

static VkDeviceAddress addressOf(const Buffer& b)
{
    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = b.buffer;
    return vkGetBufferDeviceAddress(gDev, &info);
}

struct Pipeline
{
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool ok = false;
};

// Bindings 0..n-1 in order. Failures are reported, not fatal: an unsupported shader feature must not
// stop the other tests.
static Pipeline makeCompute(const std::string& spvPath, const std::vector<VkDescriptorType>& types, uint32_t pushSize)
{
    Pipeline p;
    std::vector<uint32_t> code = loadSpv(spvPath);
    if (code.empty())
    {
        printf("FAIL cannot read %s\n", spvPath.c_str());
        return p;
    }
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < types.size(); i++) bindings.push_back({i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo sl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = (uint32_t)bindings.size();
    sl.pBindings = bindings.data();
    CHK(vkCreateDescriptorSetLayout(gDev, &sl, nullptr, &p.setLayout));
    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.setLayout;
    pl.pushConstantRangeCount = pushSize ? 1 : 0;
    pl.pPushConstantRanges = &range;
    CHK(vkCreatePipelineLayout(gDev, &pl, nullptr, &p.layout));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = code.size() * 4;
    sm.pCode = code.data();
    VkShaderModule module;
    VkResult r = vkCreateShaderModule(gDev, &sm, nullptr, &module);
    if (r != VK_SUCCESS)
    {
        printf("FAIL vkCreateShaderModule(%s) -> %d\n", spvPath.c_str(), (int)r);
        return p;
    }
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
    ci.layout = p.layout;
    const auto start = std::chrono::steady_clock::now();
    r = vkCreateComputePipelines(gDev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline);
    if (r != VK_SUCCESS)
    {
        printf("FAIL vkCreateComputePipelines(%s) -> %d\n", spvPath.c_str(), (int)r);
        return p;
    }
    printf("  pipeline %s: %zu bytes of SPIR-V, created in %.1f ms\n", spvPath.c_str(), code.size() * 4, msSince(start));
    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = gDescriptorPool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &p.setLayout;
    CHK(vkAllocateDescriptorSets(gDev, &da, &p.set));
    p.ok = true;
    return p;
}

static void bindBuffer(const Pipeline& p, uint32_t binding, const Buffer& b)
{
    VkDescriptorBufferInfo info = {b.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = p.set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(gDev, 1, &w, 0, nullptr);   // info is alive until here
}

static VkCommandBuffer beginCommands()
{
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = gPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHK(vkAllocateCommandBuffers(gDev, &ai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

/// Submits and waits. Returns the wall time from submit to completion.
static double submitAndWait(VkCommandBuffer cb)
{
    CHK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHK(vkCreateFence(gDev, &fi, nullptr, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    const auto start = std::chrono::steady_clock::now();
    CHK(vkQueueSubmit(gQueue, 1, &si, fence));
    CHK(vkWaitForFences(gDev, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000));
    const double ms = msSince(start);
    vkDestroyFence(gDev, fence, nullptr);
    vkFreeCommandBuffers(gDev, gPool, 1, &cb);
    return ms;
}

/// Runs `repeats` dispatches in one submission; returns the time per dispatch.
static double runCompute(const Pipeline& p, const void* push, uint32_t pushSize, uint32_t gx, uint32_t gy, uint32_t repeats)
{
    VkCommandBuffer cb = beginCommands();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, nullptr);
    if (pushSize) vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, push);
    for (uint32_t i = 0; i < repeats; i++) vkCmdDispatch(cb, gx, gy, 1);
    return submitAndWait(cb) / repeats;
}

/// Best of three timed submissions of `repeats` dispatches, after the caller's own first run.
static double timeCompute(const Pipeline& p, const void* push, uint32_t pushSize, uint32_t gx, uint32_t gy, uint32_t repeats)
{
    double best = 1e30;
    for (int i = 0; i < 3; i++) best = std::min(best, runCompute(p, push, pushSize, gx, gy, repeats));
    return best;
}

// ---------------------------------------------------------------------------------------------
// Hash grids on the CPU: the reference, mirroring Model/HashGrid.cu and Model/HashGridInterp.cu.
static const uint32_t kHashMapSize = 1u << 18;
static const uint32_t kPrimLevels = 6, kPrimFeatures = 2, kPrimBaseRes = 32;
static const uint32_t kClsLevels = 6, kClsFeatures = 4, kClsBaseRes = 16, kNumClusters = 4;

struct GridLevel { uint32_t res, size, offset; };

static uint32_t levelEntries(uint32_t baseRes, uint32_t level)
{
    const uint64_t res1 = (uint64_t)(baseRes << level) + 1;
    return (uint32_t)std::min<uint64_t>(res1 * res1 * res1, kHashMapSize);
}

static GridLevel gridLevel(uint32_t baseRes, uint32_t level)
{
    GridLevel l;
    l.res = baseRes << level;
    l.size = levelEntries(baseRes, level);
    l.offset = 0;
    for (uint32_t i = 0; i < level; i++) l.offset += levelEntries(baseRes, i);
    return l;
}

static uint32_t gridEntries(uint32_t baseRes, uint32_t levels)
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < levels; i++) total += levelEntries(baseRes, i);
    return total;
}

// Dense vertex index modulo the level size, for every level: what the CUDA code does (its prime hash
// is never selected, see HashGrid.slang).
static uint32_t gridIndex(uint32_t x, uint32_t y, uint32_t z, const GridLevel& l)
{
    const uint32_t res1 = l.res + 1;
    return ((res1 * x + y) * res1 + z) % l.size;
}

struct Corners { uint32_t entry[8]; float weight[8]; };

static Corners corners(const float* pos, const GridLevel& l)
{
    Corners c;
    float grid[3], frac[3];
    uint32_t base[3];
    for (int a = 0; a < 3; a++)
    {
        grid[a] = pos[a] * (float)l.res;
        base[a] = grid[a] > 0.0f ? (uint32_t)grid[a] : 0u;   // CUDA's saturating float -> unsigned
        frac[a] = grid[a] - (float)base[a];
    }
    for (uint32_t k = 0; k < 8; k++)
    {
        const uint32_t o[3] = {k & 1, (k >> 1) & 1, (k >> 2) & 1};
        c.entry[k] = l.offset + gridIndex(base[0] + o[0], base[1] + o[1], base[2] + o[2], l);
        c.weight[k] = (o[0] ? frac[0] : 1 - frac[0]) * (o[1] ? frac[1] : 1 - frac[1]) * (o[2] ? frac[2] : 1 - frac[2]);
    }
    return c;
}

static void clusterLevels(float scale, uint32_t& upper, uint32_t& lower, float& upperWeight, float& lowerWeight)
{
    auto voxel = [](int i) { return 1.0f / (float)(kClsBaseRes << i); };
    int level = (int)kClsLevels;
    for (int i = 0; i < (int)kClsLevels; ++i)
        if (scale > voxel(i)) { level = i; break; }
    if (level == 0) { upperWeight = 0.0f; upper = 0; }
    else if (level == (int)kClsLevels) { upperWeight = 1.0f; upper = kClsLevels - 1; }
    else { upperWeight = (voxel(level - 1) - scale) / (voxel(level - 1) - voxel(level)); upper = (uint32_t)level; }
    level = -1;
    for (int i = (int)kClsLevels - 1; i >= 0; --i)
        if (scale < voxel(i)) { level = i; break; }
    if (level == -1) { lowerWeight = 1.0f; lower = 0; }
    else if (level == (int)kClsLevels - 1) { lowerWeight = 0.0f; lower = kClsLevels - 1; }
    else { lowerWeight = (scale - voxel(level + 1)) / (voxel(level) - voxel(level + 1)); lower = (uint32_t)level; }
}

static const VkDescriptorType kStorage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
static const VkBufferUsageFlags kStorageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

// ---------------------------------------------------------------------------------------------
static void testEncode(const std::string& dir)
{
    printf("== encode: hash-grid lookups in compute\n");
    const uint32_t primEntries = gridEntries(kPrimBaseRes, kPrimLevels);
    const uint32_t clsEntries = gridEntries(kClsBaseRes, kClsLevels);
    printf("  primary grid %u entries x %u, cluster grid %u entries x %u\n", primEntries, kPrimFeatures, clsEntries, kClsFeatures);

    const uint32_t capacity = 1u << 18;
    Buffer samples = makeBuffer(capacity * 16ull, kStorageUsage);
    Buffer primGrid = makeBuffer(primEntries * kPrimFeatures * 2ull, kStorageUsage);
    Buffer primEnc = makeBuffer(capacity * 2ull * kPrimLevels * kPrimFeatures, kStorageUsage);
    for (uint32_t i = 0; i < capacity * 4; i++) samples.as<float>()[i] = rnd();
    for (uint32_t i = 0; i < primEntries * kPrimFeatures; i++) primGrid.as<uint16_t>()[i] = floatToHalf(rnd() * 2.0f - 1.0f);

    struct { uint32_t count, capacity; } push;
    Pipeline prim = makeCompute(dir + "/PrimEncode.spv", {kStorage, kStorage, kStorage}, sizeof(push));
    if (prim.ok)
    {
        bindBuffer(prim, 0, samples);
        bindBuffer(prim, 1, primGrid);
        bindBuffer(prim, 2, primEnc);
        for (uint32_t count : {1u << 16, 1u << 18})
        {
            push = {count, capacity};
            memset(primEnc.data, 0, primEnc.size);
            const double first = runCompute(prim, &push, sizeof(push), (count + 63) / 64, 1, 1);
            double maxError = 0.0;
            for (uint32_t n = 0; n < count; n += 7)   // every 7th sample
            {
                for (uint32_t level = 0; level < kPrimLevels; level++)
                {
                    const Corners c = corners(samples.as<float>() + n * 4, gridLevel(kPrimBaseRes, level));
                    for (uint32_t f = 0; f < kPrimFeatures; f++)
                    {
                        float ref = 0.0f;
                        for (int k = 0; k < 8; k++) ref += halfToFloat(primGrid.as<uint16_t>()[c.entry[k] * kPrimFeatures + f]) * c.weight[k];
                        const float gpu = halfToFloat(primEnc.as<uint16_t>()[(level * kPrimFeatures + f) * capacity + n]);
                        maxError = std::max(maxError, (double)std::fabs(gpu - ref));
                    }
                }
            }
            const double ms = timeCompute(prim, &push, sizeof(push), (count + 63) / 64, 1, 8);
            printf("RESULT encode primary %u samples: %.2f ms (first %.2f) | max error vs CPU %.2e %s\n", count, ms, first, maxError,
                   maxError < 1.5e-3 ? "OK" : "MISMATCH");
        }
    }

    // Cluster grid: 4 clusters per sample, radius log-uniform over all level brackets and beyond.
    const uint32_t clsCapacity = 1u << 17;
    Buffer clusters = makeBuffer(clsCapacity * kNumClusters * 16ull, kStorageUsage);
    Buffer weights = makeBuffer(clsCapacity * kNumClusters * 4ull, kStorageUsage);
    Buffer clsGrid = makeBuffer(clsEntries * kClsFeatures * 2ull, kStorageUsage);
    Buffer clsEnc = makeBuffer(clsCapacity * 2ull * kNumClusters * kClsFeatures, kStorageUsage);
    for (uint32_t i = 0; i < clsCapacity * kNumClusters; i++)
    {
        float* c = clusters.as<float>() + i * 4;
        // Slightly beyond the unit cube on both sides: cluster positions do leave it, and the lookup
        // then extrapolates from the border cell. Kept small so the random grid stays within fp16.
        c[0] = rnd() * 1.04f - 0.02f; c[1] = rnd() * 1.04f - 0.02f; c[2] = rnd() * 1.04f - 0.02f;
        c[3] = std::exp2(-11.0f + 9.0f * rnd());   // 2^-11 .. 2^-2; the level scales are 2^-4 .. 2^-9
        weights.as<float>()[i] = rnd();
    }
    for (uint32_t i = 0; i < clsEntries * kClsFeatures; i++) clsGrid.as<uint16_t>()[i] = floatToHalf(rnd() * 2.0f - 1.0f);
    Pipeline cls = makeCompute(dir + "/ClsEncode.spv", {kStorage, kStorage, kStorage, kStorage}, sizeof(push));
    if (cls.ok)
    {
        bindBuffer(cls, 0, clusters);
        bindBuffer(cls, 1, weights);
        bindBuffer(cls, 2, clsGrid);
        bindBuffer(cls, 3, clsEnc);
        for (uint32_t count : {1u << 16, 1u << 17})
        {
            push = {count, clsCapacity};
            memset(clsEnc.data, 0, clsEnc.size);
            const double first = runCompute(cls, &push, sizeof(push), (count + 63) / 64, 1, 1);
            double maxError = 0.0;
            for (uint32_t n = 0; n < count; n += 7)
            {
                for (uint32_t k = 0; k < kNumClusters; k++)
                {
                    const float* c = clusters.as<float>() + (n * kNumClusters + k) * 4;
                    uint32_t upper, lower;
                    float upperWeight, lowerWeight;
                    clusterLevels(c[3], upper, lower, upperWeight, lowerWeight);
                    const Corners cu = corners(c, gridLevel(kClsBaseRes, upper));
                    const Corners cl = corners(c, gridLevel(kClsBaseRes, lower));
                    for (uint32_t f = 0; f < kClsFeatures; f++)
                    {
                        float u = 0.0f, l = 0.0f;
                        for (int j = 0; j < 8; j++)
                        {
                            u += halfToFloat(clsGrid.as<uint16_t>()[cu.entry[j] * kClsFeatures + f]) * cu.weight[j];
                            l += halfToFloat(clsGrid.as<uint16_t>()[cl.entry[j] * kClsFeatures + f]) * cl.weight[j];
                        }
                        const float ref = (l * lowerWeight + u * upperWeight) * weights.as<float>()[n * kNumClusters + k];
                        const float gpu = halfToFloat(clsEnc.as<uint16_t>()[(k * kClsFeatures + f) * clsCapacity + n]);
                        maxError = std::max(maxError, (double)std::fabs(gpu - ref) / std::max(1.0f, std::fabs(ref)));   // extrapolated values are large
                    }
                }
            }
            const double ms = timeCompute(cls, &push, sizeof(push), (count + 63) / 64, 1, 8);
            printf("RESULT encode cluster %u samples x 4 clusters: %.2f ms (first %.2f) | max error vs CPU %.2e %s\n", count, ms, first,
                   maxError, maxError < 1.5e-3 ? "OK" : "MISMATCH");
        }
    }
    for (Buffer* b : {&samples, &primGrid, &primEnc, &clusters, &weights, &clsGrid, &clsEnc}) freeBuffer(*b);
}

// ---------------------------------------------------------------------------------------------
static void testScatter(const std::string& dir)
{
    printf("== scatter: hash-grid gradient scatter with integer atomics\n");
    const uint32_t primEntries = gridEntries(kPrimBaseRes, kPrimLevels);
    const uint32_t capacity = 1u << 16;
    const float gradScale = 16777216.0f;   // 2^24, the scale the port uses (GRID_GRADIENT_SCALE in ncr_reference.py)
    Buffer samples = makeBuffer(capacity * 16ull, kStorageUsage);
    Buffer gradIn = makeBuffer(capacity * 2ull * kPrimLevels * kPrimFeatures, kStorageUsage);
    Buffer grad = makeBuffer(primEntries * kPrimFeatures * 4ull, kStorageUsage);
    for (uint32_t i = 0; i < capacity * 4; i++) samples.as<float>()[i] = rnd();
    for (uint32_t i = 0; i < capacity * kPrimLevels * kPrimFeatures; i++) gradIn.as<uint16_t>()[i] = floatToHalf(rnd() * 2.0f - 1.0f);

    struct { uint32_t count, capacity; float gradScale; } push;
    Pipeline p = makeCompute(dir + "/PrimScatter.spv", {kStorage, kStorage, kStorage}, sizeof(push));
    if (!p.ok) return;
    bindBuffer(p, 0, samples);
    bindBuffer(p, 1, gradIn);
    bindBuffer(p, 2, grad);
    for (uint32_t count : {1u << 13, 1u << 15, 1u << 16})
    {
        push = {count, capacity, gradScale};
        memset(grad.data, 0, grad.size);
        const double first = runCompute(p, &push, sizeof(push), (count + 63) / 64, 1, 1);

        std::vector<int64_t> reference(primEntries * kPrimFeatures, 0);
        std::vector<uint32_t> adds(primEntries * kPrimFeatures, 0);
        for (uint32_t n = 0; n < count; n++)
        {
            for (uint32_t level = 0; level < kPrimLevels; level++)
            {
                const Corners c = corners(samples.as<float>() + n * 4, gridLevel(kPrimBaseRes, level));
                for (uint32_t f = 0; f < kPrimFeatures; f++)
                {
                    const float g = halfToFloat(gradIn.as<uint16_t>()[(level * kPrimFeatures + f) * capacity + n]) * gradScale;
                    for (int k = 0; k < 8; k++)
                    {
                        reference[c.entry[k] * kPrimFeatures + f] += std::lround(g * c.weight[k]);
                        adds[c.entry[k] * kPrimFeatures + f]++;
                    }
                }
            }
        }
        // Each add may round the other way on the GPU (different float evaluation order), by one unit.
        int64_t worst = 0;
        size_t beyond = 0, touched = 0;
        uint32_t mostAdds = 0;
        for (size_t i = 0; i < reference.size(); i++)
        {
            const int64_t diff = std::llabs((int64_t)grad.as<int32_t>()[i] - reference[i]);
            worst = std::max(worst, diff);
            beyond += diff > (int64_t)adds[i];
            touched += adds[i] > 0;
            mostAdds = std::max(mostAdds, adds[i]);
        }
        const double ms = timeCompute(p, &push, sizeof(push), (count + 63) / 64, 1, 8);
        printf("RESULT scatter %u samples: %.2f ms (first %.2f) | %zu entries touched, up to %u adds on one entry | worst difference %lld fixed-point units, %zu entries beyond one unit per add %s\n",
               count, ms, first, touched, mostAdds, (long long)worst, beyond, beyond == 0 ? "OK" : "MISMATCH");
    }
    for (Buffer* b : {&samples, &gradIn, &grad}) freeBuffer(*b);
}

// ---------------------------------------------------------------------------------------------
static void testAdam(const std::string& dir)
{
    printf("== adam: dense optimizer pass over a grid\n");
    struct { uint32_t count; float learningRate, gradToFloat, emaDecay, emaDebiasOld, emaDebiasNew; } push;
    Pipeline p = makeCompute(dir + "/GridAdam.spv", {kStorage, kStorage, kStorage, kStorage, kStorage, kStorage}, sizeof(push));
    if (!p.ok) return;
    const uint32_t counts[2] = {gridEntries(kPrimBaseRes, kPrimLevels) * kPrimFeatures, gridEntries(kClsBaseRes, kClsLevels) * kClsFeatures};
    const char* names[2] = {"primary", "cluster"};
    for (int g = 0; g < 2; g++)
    {
        const uint32_t count = counts[g];
        Buffer grad = makeBuffer(count * 4ull, kStorageUsage);
        Buffer master = makeBuffer(count * 4ull, kStorageUsage);
        Buffer m = makeBuffer(count * 4ull, kStorageUsage);
        Buffer v = makeBuffer(count * 4ull, kStorageUsage);
        Buffer weights = makeBuffer(count * 2ull, kStorageUsage);
        Buffer ema = makeBuffer(count * 2ull, kStorageUsage);
        for (uint32_t i = 0; i < count; i++)
        {
            grad.as<int32_t>()[i] = (int32_t)(nextU32() % 2001u) - 1000;
            master.as<float>()[i] = rnd() - 0.5f;
        }
        memset(m.data, 0, m.size);
        memset(v.data, 0, v.size);
        memset(ema.data, 0, ema.size);
        const std::vector<int32_t> grad0(grad.as<int32_t>(), grad.as<int32_t>() + 1024);
        const std::vector<float> master0(master.as<float>(), master.as<float>() + 1024);
        const Buffer* all[6] = {&grad, &master, &m, &v, &weights, &ema};
        for (uint32_t b = 0; b < 6; b++) bindBuffer(p, b, *all[b]);
        push = {count, 5e-3f, 1.0f / (4096.0f * 128.0f), 0.95f, 0.0f, 1.0f / (1.0f - 0.95f)};
        const double first = runCompute(p, &push, sizeof(push), (count + 255) / 256, 1, 1);
        double maxError = 0.0;
        for (uint32_t i = 0; i < 1024; i++)
        {
            const float gradient = (float)grad0[i] * push.gradToFloat + 1e-8f * master0[i];
            const float m1 = 0.1f * gradient, v1 = 0.001f * gradient * gradient;
            const float updated = master0[i] - push.learningRate * m1 / (std::sqrt(v1) + 1e-8f);
            maxError = std::max(maxError, (double)std::fabs(master.as<float>()[i] - updated));
            maxError = std::max(maxError, (double)std::fabs(halfToFloat(ema.as<uint16_t>()[i]) - updated));   // debiased first EMA step
            if (grad.as<int32_t>()[i] != 0) maxError = 1.0;   // the gradient must be cleared
        }
        const double ms = timeCompute(p, &push, sizeof(push), (count + 255) / 256, 1, 4);
        printf("RESULT adam %s grid, %u parameters: %.2f ms (first %.2f) | max error vs CPU on 1024 parameters %.2e %s\n", names[g], count,
               ms, first, maxError, maxError < 1e-3 ? "OK" : "MISMATCH");
        for (Buffer* b : {&grad, &master, &m, &v, &weights, &ema}) freeBuffer(*b);
    }
}

// ---------------------------------------------------------------------------------------------
struct AccelerationStructure
{
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    Buffer storage;
};

static AccelerationStructure buildAS(VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitives, double* buildMs)
{
    VkAccelerationStructureBuildGeometryInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type = type;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pGetBuildSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primitives, &sizes);

    AccelerationStructure as;
    as.storage = makeBuffer(sizes.accelerationStructureSize,
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);
    VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = as.storage.buffer;
    ci.size = sizes.accelerationStructureSize;
    ci.type = type;
    CHK(pCreateAS(gDev, &ci, nullptr, &as.handle));
    Buffer scratch = makeBuffer(sizes.buildScratchSize + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);
    info.dstAccelerationStructure = as.handle;
    info.scratchData.deviceAddress = (addressOf(scratch) + 255) & ~VkDeviceAddress(255);
    VkAccelerationStructureBuildRangeInfoKHR range = {primitives, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    VkCommandBuffer cb = beginCommands();
    pCmdBuildAS(cb, 1, &info, &ranges);
    *buildMs = submitAndWait(cb);
    freeBuffer(scratch);
    printf("  %s: %u primitives, %.1f MB, built in %.1f ms\n", type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ? "BLAS" : "TLAS",
           primitives, sizes.accelerationStructureSize / 1048576.0, *buildMs);
    return as;
}

static void testRays(const std::string& dir)
{
    printf("== rays: inline ray query throughput\n");
    if (!gRayQuery)
    {
        printf("FAIL rays: the device does not report VK_KHR_ray_query with VK_KHR_acceleration_structure\n");
        return;
    }
    // A closed box [-1, 1]^3: six faces of G x G quads, displaced along the normal by a ripple that
    // vanishes on the face borders, so the faces still meet. Origins are inside: every ray hits.
    const uint32_t G = 128;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    for (uint32_t face = 0; face < 6; face++)
    {
        const uint32_t axis = face / 2, b = (axis + 1) % 3, c = (axis + 2) % 3;
        const float sign = (face & 1) ? 1.0f : -1.0f;
        const uint32_t first = (uint32_t)(vertices.size() / 3);
        for (uint32_t j = 0; j <= G; j++)
        {
            for (uint32_t i = 0; i <= G; i++)
            {
                const float u = (float)i / G, v = (float)j / G;
                const float ripple = 0.05f * std::sin(4.0f * 3.14159265f * u) * std::sin(4.0f * 3.14159265f * v);
                float p[3];
                p[axis] = sign * (1.0f - ((i == 0 || i == G || j == 0 || j == G) ? 0.0f : ripple));
                p[b] = 2.0f * u - 1.0f;
                p[c] = 2.0f * v - 1.0f;
                vertices.insert(vertices.end(), p, p + 3);
            }
        }
        for (uint32_t j = 0; j < G; j++)
        {
            for (uint32_t i = 0; i < G; i++)
            {
                const uint32_t v0 = first + j * (G + 1) + i, v1 = v0 + 1, v2 = v0 + G + 1, v3 = v2 + 1;
                const uint32_t quad[6] = {v0, v1, v2, v1, v3, v2};
                indices.insert(indices.end(), quad, quad + 6);
            }
        }
    }
    const uint32_t triangles = (uint32_t)(indices.size() / 3);
    const VkBufferUsageFlags inputUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer vertexBuffer = makeBuffer(vertices.size() * 4, inputUsage);
    Buffer indexBuffer = makeBuffer(indices.size() * 4, inputUsage);
    memcpy(vertexBuffer.data, vertices.data(), vertices.size() * 4);
    memcpy(indexBuffer.data, indices.data(), indices.size() * 4);

    double buildMs = 0.0;
    VkAccelerationStructureGeometryKHR geometry = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = addressOf(vertexBuffer);
    geometry.geometry.triangles.vertexStride = 12;
    geometry.geometry.triangles.maxVertex = (uint32_t)(vertices.size() / 3) - 1;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress = addressOf(indexBuffer);
    AccelerationStructure blas = buildAS(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry, triangles, &buildMs);

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = blas.handle;
    VkAccelerationStructureInstanceKHR instance = {};
    instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
    instance.mask = 0xff;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = pGetASAddress(gDev, &addressInfo);
    Buffer instanceBuffer = makeBuffer(sizeof(instance), inputUsage);
    memcpy(instanceBuffer.data, &instance, sizeof(instance));
    VkAccelerationStructureGeometryKHR instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    instances.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    instances.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    instances.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.geometry.instances.data.deviceAddress = addressOf(instanceBuffer);
    AccelerationStructure tlas = buildAS(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, instances, 1, &buildMs);

    struct { uint32_t width, height, raysPerPixel, mode, seed; } push;
    Pipeline p = makeCompute(dir + "/RayThroughput.spv", {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kStorage}, sizeof(push));
    if (!p.ok) return;
    const uint32_t width = 640, height = 360;
    Buffer out = makeBuffer(width * height * 8ull, kStorageUsage);
    VkWriteDescriptorSetAccelerationStructureKHR asWrite = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas.handle;
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.pNext = &asWrite;
    w.dstSet = p.set;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(gDev, 1, &w, 0, nullptr);
    bindBuffer(p, 1, out);

    const char* modes[2] = {"incoherent", "cone"};
    for (uint32_t mode = 0; mode < 2; mode++)
    {
        for (uint32_t rays : {1u, 8u, 32u})
        {
            push = {width, height, rays, mode, 1};
            memset(out.data, 0, out.size);
            const double first = runCompute(p, &push, sizeof(push), (width + 7) / 8, (height + 7) / 8, 1);
            uint64_t hits = 0, distance = 0;
            for (uint32_t i = 0; i < width * height; i++)
            {
                hits += out.as<uint32_t>()[i * 2];
                distance += out.as<uint32_t>()[i * 2 + 1];
            }
            const uint64_t total = (uint64_t)width * height * rays;
            const double ms = timeCompute(p, &push, sizeof(push), (width + 7) / 8, (height + 7) / 8, 2);
            printf("RESULT rays %s, %ux%u x %u rays (%.2f M): %.2f ms (first %.2f) = %.1f M rays/s | hits %llu of %llu, mean distance %.3f %s\n",
                   modes[mode], width, height, rays, total / 1e6, ms, first, total / ms / 1e3, (unsigned long long)hits,
                   (unsigned long long)total, hits ? distance / 1024.0 / hits : 0.0, hits == total ? "OK" : "MISMATCH");
        }
    }
}

// ---------------------------------------------------------------------------------------------
static void testFeatures(const std::string& dir, bool int64)
{
    const char* name = int64 ? "features64" : "features";
    printf("== %s: subgroup arithmetic and groupshared memory%s (Slang)\n", name, int64 ? ", with 64-bit integers" : "");
    Pipeline p = makeCompute(dir + (int64 ? "/Features64.spv" : "/Features.spv"), {kStorage}, 0);
    if (!p.ok)
    {
        printf("RESULT %s: the pipeline cannot be created on this device\n", name);
        return;
    }
    const uint32_t groups = 64, threads = groups * 32;
    Buffer out = makeBuffer(threads * 8 * 4ull, kStorageUsage);
    memset(out.data, 0, out.size);
    bindBuffer(p, 0, out);
    runCompute(p, nullptr, 0, groups, 1, 1);
    const uint32_t* o = out.as<uint32_t>();
    size_t written = 0, groupSharedOk = 0, wideOk = 0, oneWave = 0, waveInRange = 0;
    uint32_t laneCountMin = UINT32_MAX, laneCountMax = 0;
    for (uint32_t t = 0; t < threads; t++)
    {
        const uint32_t* v = o + t * 8;
        written += v[7] == 0xabcd1234u;
        groupSharedOk += v[3] == 528;                       // 1 + 2 + ... + 32
        wideOk += v[4] == (t + 1) * 3 && v[5] == t + 1;      // (t + 1) * 0x100000003
        oneWave += v[2] == 528;
        waveInRange += v[2] >= v[6] && v[2] <= 528;
        laneCountMin = std::min(laneCountMin, v[1]);
        laneCountMax = std::max(laneCountMax, v[1]);
    }
    printf("  group 0, per thread (lane index / wave sum):");
    for (uint32_t t = 0; t < 32; t++) printf(" %u/%u", o[t * 8], o[t * 8 + 2]);
    printf("\n");
    const bool ok = written == threads && groupSharedOk == threads && waveInRange == threads && (!int64 || wideOk == threads);
    printf("RESULT %s: threads written %zu/%u | groupshared sum %zu/%u | WaveGetLaneCount %u..%u, wave sum equals the group sum on %zu/%u threads, plausible on %zu/%u",
           name, written, threads, groupSharedOk, threads, laneCountMin, laneCountMax, oneWave, threads, waveInRange, threads);
    if (int64) printf(" | uint64 %zu/%u", wideOk, threads);
    printf(" %s\n", ok ? "OK" : "MISMATCH");
    freeBuffer(out);
}

// ---------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string dir = argv[0];
    dir = dir.find('/') == std::string::npos ? "." : dir.substr(0, dir.find_last_of('/'));
    std::vector<std::string> tests;
    for (int i = 1; i < argc; i++) tests.push_back(argv[i]);
    if (tests.empty()) tests = {"encode", "scatter", "adam", "rays", "features", "features64"};

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    CHK(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 1;
    vkEnumeratePhysicalDevices(instance, &count, &gPhysical);
    if (count == 0)
    {
        printf("FAIL no Vulkan device\n");
        return 1;
    }
    vkGetPhysicalDeviceMemoryProperties(gPhysical, &gMem);

    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(gPhysical, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(gPhysical, nullptr, &extensionCount, extensions.data());
    auto has = [&](const char* name) {
        for (const auto& e : extensions) if (!strcmp(e.extensionName, name)) return true;
        return false;
    };

    VkPhysicalDeviceSubgroupProperties subgroup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
    vkGetPhysicalDeviceProperties2(gPhysical, &properties);
    const VkPhysicalDeviceLimits& limits = properties.properties.limits;

    gRayQuery = has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && has(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rayQuery};
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, gRayQuery ? &accel : nullptr};
    VkPhysicalDeviceVulkan11Features v11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(gPhysical, &features);
    gRayQuery = gRayQuery && accel.accelerationStructure && rayQuery.rayQuery;

    printf("== info\n");
    printf("  device %s, Vulkan %u.%u.%u, driver 0x%x\n", properties.properties.deviceName, VK_VERSION_MAJOR(properties.properties.apiVersion),
           VK_VERSION_MINOR(properties.properties.apiVersion), VK_VERSION_PATCH(properties.properties.apiVersion), properties.properties.driverVersion);
    printf("  features: shaderInt64 %u, shaderInt16 %u, shaderFloat16 %u, shaderInt8 %u, storageBuffer16BitAccess %u, uniformAndStorageBuffer16BitAccess %u, bufferDeviceAddress %u, scalarBlockLayout %u, descriptorIndexing %u, runtimeDescriptorArray %u\n",
           features.features.shaderInt64, features.features.shaderInt16, v12.shaderFloat16, v12.shaderInt8, v11.storageBuffer16BitAccess,
           v11.uniformAndStorageBuffer16BitAccess, v12.bufferDeviceAddress, v12.scalarBlockLayout, v12.descriptorIndexing, v12.runtimeDescriptorArray);
    printf("  ray tracing: accelerationStructure %u, rayQuery %u, VK_KHR_ray_tracing_pipeline %s\n", accel.accelerationStructure,
           rayQuery.rayQuery, has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) ? "yes" : "no");
    printf("  subgroups: size %u, stages 0x%x (compute %s), operations 0x%x (arithmetic %s, ballot %s, shuffle %s)\n", subgroup.subgroupSize,
           subgroup.supportedStages, (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "yes" : "no", subgroup.supportedOperations,
           (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? "yes" : "no",
           (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ? "yes" : "no",
           (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? "yes" : "no");
    printf("  limits: maxComputeWorkGroupCount %u/%u/%u, maxComputeWorkGroupSize %u/%u/%u, invocations %u, maxStorageBufferRange %u, maxPushConstantsSize %u, maxPerStageDescriptorStorageBuffers %u\n",
           limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1], limits.maxComputeWorkGroupCount[2],
           limits.maxComputeWorkGroupSize[0], limits.maxComputeWorkGroupSize[1], limits.maxComputeWorkGroupSize[2],
           limits.maxComputeWorkGroupInvocations, limits.maxStorageBufferRange, limits.maxPushConstantsSize,
           limits.maxPerStageDescriptorStorageBuffers);
    printf("  extensions: VK_OHOS_external_memory %s, %u device extensions in total\n", has("VK_OHOS_external_memory") ? "yes" : "no", extensionCount);
    for (uint32_t i = 0; i < gMem.memoryTypeCount; i++)
        printf("  memory type %u: flags 0x%x, heap %u (%llu MB)\n", i, gMem.memoryTypes[i].propertyFlags, gMem.memoryTypes[i].heapIndex,
               (unsigned long long)(gMem.memoryHeaps[gMem.memoryTypes[i].heapIndex].size >> 20));

    std::vector<const char*> enabled;
    if (gRayQuery)
    {
        enabled = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    }
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features;   // everything the driver reports
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enabled.size();
    dci.ppEnabledExtensionNames = enabled.data();
    CHK(vkCreateDevice(gPhysical, &dci, nullptr, &gDev));
    vkGetDeviceQueue(gDev, 0, 0, &gQueue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = 0;
    CHK(vkCreateCommandPool(gDev, &pci, nullptr, &gPool));
    VkDescriptorPoolSize poolSizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 16;
    dpi.poolSizeCount = gRayQuery ? 2 : 1;
    dpi.pPoolSizes = poolSizes;
    CHK(vkCreateDescriptorPool(gDev, &dpi, nullptr, &gDescriptorPool));
    if (gRayQuery)
    {
        pGetBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(gDev, "vkGetAccelerationStructureBuildSizesKHR");
        pCreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(gDev, "vkCreateAccelerationStructureKHR");
        pCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(gDev, "vkCmdBuildAccelerationStructuresKHR");
        pGetASAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(gDev, "vkGetAccelerationStructureDeviceAddressKHR");
        gRayQuery = pGetBuildSizes && pCreateAS && pCmdBuildAS && pGetASAddress;
    }

    for (const std::string& test : tests)
    {
        if (test == "encode") testEncode(dir);
        else if (test == "scatter") testScatter(dir);
        else if (test == "adam") testAdam(dir);
        else if (test == "rays") testRays(dir);
        else if (test == "features") testFeatures(dir, false);
        else if (test == "features64") testFeatures(dir, true);
        else if (test != "info") printf("unknown test %s\n", test.c_str());
    }
    vkDeviceWaitIdle(gDev);
    return 0;
}

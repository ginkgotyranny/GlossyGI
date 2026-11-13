// Vulkan side of the NCR replayer: loads a packed Falcor bake (ohos/tools/bake_pack.py) and creates
// what it describes: buffers, textures, samplers, acceleration structures, and one compute pipeline
// per baked pass with its descriptor sets bound as Falcor had them.
//
// Header-only and platform-neutral: the same code is built for Windows (parity checks against
// Falcor) and for OpenHarmony (the phone). No window: everything is compute.
#pragma once
#ifdef __OHOS__
#define VK_USE_PLATFORM_OHOS   // vulkan_ohos.h: VK_OHOS_external_memory
#include <native_buffer/native_buffer.h>
#include <native_window/buffer_handle.h>
#include <native_window/external_window.h>
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define VK_CHECK(x) \
    do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::printf("FAIL %s -> %d (%s:%d)\n", #x, (int)r_, __FILE__, __LINE__); std::exit(2); } } while (0)

namespace replay
{
inline double msSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

inline std::vector<char> readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        std::printf("FAIL cannot read %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<char> data((size_t)f.tellg());
    f.seekg(0);
    f.read(data.data(), (std::streamsize)data.size());
    return data;
}

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* data = nullptr;   // mapped for the buffer's lifetime, when host-visible
    // A buffer shared with the NPU (Device::makeSharedBuffer): the dma-buf behind it and its size.
    int fd = -1;
    size_t sharedSize = 0;
    template <typename T> T* as() const { return static_cast<T*>(data); }
};

struct Texture
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;        // all mips and layers: sampling
    VkImageView storageView = VK_NULL_HANDLE; // mip 0: storage images
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t bytesPerPixel = 0, width = 0, height = 0, depth = 1, mips = 1, layers = 1;
    bool cube = false, storage = false;
};

struct Device
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory = {};
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};
    uint32_t maxGroupCount[3] = {65535, 65535, 65535};   // maxComputeWorkGroupCount
    bool rayPipeline = false;
    bool sharedMemory = false;   // VK_OHOS_external_memory: buffers shared with the NPU

    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR createAS = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAS = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getASAddress = nullptr;

    void create()
    {
        VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

        // The first device that has ray queries (a desktop may list an integrated GPU as well).
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        std::vector<const char*> enabled;
        for (VkPhysicalDevice candidate : devices)
        {
            uint32_t n = 0;
            vkEnumerateDeviceExtensionProperties(candidate, nullptr, &n, nullptr);
            std::vector<VkExtensionProperties> extensions(n);
            vkEnumerateDeviceExtensionProperties(candidate, nullptr, &n, extensions.data());
            auto has = [&](const char* e) {
                for (const auto& x : extensions) if (!std::strcmp(x.extensionName, e)) return true;
                return false;
            };
            if (has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && has(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME))
            {
                physical = candidate;
                enabled = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                           VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
                // The baked modules declare the RayTracingKHR capability next to RayQueryKHR.
                rayPipeline = has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
                if (rayPipeline) enabled.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
#ifdef __OHOS__
                // Buffers the GPU and the NPU both work on (makeSharedBuffer).
                sharedMemory = has(VK_OHOS_EXTERNAL_MEMORY_EXTENSION_NAME);
                if (sharedMemory) enabled.push_back(VK_OHOS_EXTERNAL_MEMORY_EXTENSION_NAME);
#endif
                break;
            }
        }
        if (!physical)
        {
            std::printf("FAIL no Vulkan device with ray queries\n");
            std::exit(2);
        }
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physical, &properties);
        std::memcpy(name, properties.deviceName, sizeof(name));
        std::memcpy(maxGroupCount, properties.limits.maxComputeWorkGroupCount, sizeof(maxGroupCount));
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);

        // Everything the driver offers: the baked modules use 16-bit types and storage, ray queries,
        // descriptor arrays, storage images without format.
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR pipelineFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        VkPhysicalDeviceRayQueryFeaturesKHR rayQuery = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, rayPipeline ? &pipelineFeatures : nullptr};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accel = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rayQuery};
        VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &accel};
        VkPhysicalDeviceVulkan11Features v11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
        VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
        vkGetPhysicalDeviceFeatures2(physical, &features);

        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, queues.data());
        uint32_t family = 0;
        for (uint32_t i = 0; i < families; i++)
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &features;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)enabled.size();
        dci.ppEnabledExtensionNames = enabled.data();
        VK_CHECK(vkCreateDevice(physical, &dci, nullptr, &device));
        vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = family;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));

        getBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
        createAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        cmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
        getASAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    }

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want) const
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    }

    /// Host-visible buffers stay mapped. On the phone every memory type is device-local, so mapping
    /// costs the GPU nothing; on a desktop the shared buffers live in system memory, which is slower
    /// but keeps one code path.
    Buffer makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible) const
    {
        Buffer b;
        b.size = size;
        VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = size;
        ci.usage = usage;
        VK_CHECK(vkCreateBuffer(device, &ci, nullptr, &b.buffer));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateFlagsInfo flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext = &flags;
        ai.allocationSize = req.size;
        if (hostVisible)
        {
            const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            ai.memoryTypeIndex = memoryType(req.memoryTypeBits, host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (ai.memoryTypeIndex == UINT32_MAX) ai.memoryTypeIndex = memoryType(req.memoryTypeBits, host);
        }
        else
        {
            ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (ai.memoryTypeIndex == UINT32_MAX)
        {
            std::printf("FAIL no memory type for a %s buffer\n", hostVisible ? "host-visible" : "device-local");
            std::exit(2);
        }
        VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &b.memory));
        VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
        if (hostVisible) VK_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.data));
        return b;
    }

    /// A storage buffer the GPU, the CPU and the NPU all work on without copies: an OH_NativeBuffer
    /// (dma-buf) imported into Vulkan through VK_OHOS_external_memory, mapped for the CPU, and handed
    /// to NNRt by its file descriptor (OH_NNTensor_CreateWithFd). Where there is no such memory
    /// (the desktop) it is an ordinary host-visible buffer, with fd -1.
    Buffer makeSharedBuffer(VkDeviceSize size, VkBufferUsageFlags usage) const
    {
#ifdef __OHOS__
        if (sharedMemory)
        {
            Buffer b;
            b.size = size;
            // Whole pages: the memory goes to another device by file descriptor.
            b.sharedSize = (size_t)((size + 4095) / 4096 * 4096);
            OH_NativeBuffer_Config config = {};
            config.width = (int32_t)b.sharedSize;
            config.height = 1;
            config.format = NATIVEBUFFER_PIXEL_FMT_BLOB;
            config.usage = NATIVEBUFFER_USAGE_CPU_READ | NATIVEBUFFER_USAGE_CPU_WRITE | NATIVEBUFFER_USAGE_MEM_DMA | NATIVEBUFFER_USAGE_HW_RENDER |
                           NATIVEBUFFER_USAGE_HW_TEXTURE;
            OH_NativeBuffer* native = OH_NativeBuffer_Alloc(&config);
            if (!native)
            {
                std::printf("FAIL cannot allocate a native buffer of %zu bytes\n", b.sharedSize);
                std::exit(2);
            }
            // The file descriptor is only reachable through the window-buffer view of the buffer.
            OHNativeWindowBuffer* window = OH_NativeWindow_CreateNativeWindowBufferFromNativeBuffer(native);
            BufferHandle* handle = window ? OH_NativeWindow_GetBufferHandleFromNative(window) : nullptr;
            if (!handle || handle->fd < 0)
            {
                std::printf("FAIL a native buffer without a file descriptor\n");
                std::exit(2);
            }
            b.fd = handle->fd;

            VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OHOS_NATIVE_BUFFER_BIT_OHOS;
            VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ci.pNext = &external;
            ci.size = size;
            ci.usage = usage;
            VK_CHECK(vkCreateBuffer(device, &ci, nullptr, &b.buffer));
            VkNativeBufferPropertiesOHOS properties = {VK_STRUCTURE_TYPE_NATIVE_BUFFER_PROPERTIES_OHOS};
            VK_CHECK(vkGetNativeBufferPropertiesOHOS(device, native, &properties));
            VkImportNativeBufferInfoOHOS import = {VK_STRUCTURE_TYPE_IMPORT_NATIVE_BUFFER_INFO_OHOS};
            import.buffer = native;
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &import;
            ai.allocationSize = properties.allocationSize;
            ai.memoryTypeIndex = 0;
            while (ai.memoryTypeIndex < 32 && !(properties.memoryTypeBits & (1u << ai.memoryTypeIndex))) ai.memoryTypeIndex++;
            VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &b.memory));
            VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
            if (OH_NativeBuffer_Map(native, &b.data) != 0 || !b.data)
            {
                std::printf("FAIL cannot map a native buffer\n");
                std::exit(2);
            }
            return b;
        }
#endif
        return makeBuffer(size, usage, true);
    }

    void destroy(Buffer& b) const
    {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = Buffer();
    }

    VkDeviceAddress address(const Buffer& b) const
    {
        VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = b.buffer;
        return vkGetBufferDeviceAddress(device, &info);
    }

    VkCommandBuffer begin() const
    {
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cb));
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        return cb;
    }

    /// Submits, waits for completion and frees the command buffer. Returns the wall time.
    double submit(VkCommandBuffer cb) const
    {
        VK_CHECK(vkEndCommandBuffer(cb));
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        VK_CHECK(vkCreateFence(device, &fi, nullptr, &fence));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        const auto start = std::chrono::steady_clock::now();
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 120ull * 1000 * 1000 * 1000));
        const double ms = msSince(start);
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, pool, 1, &cb);
        return ms;
    }

    Texture makeTexture(VkFormat format, uint32_t bytesPerPixel, uint32_t width, uint32_t height, uint32_t depth, uint32_t mips,
                        uint32_t layers, bool cube, bool storage) const
    {
        Texture t;
        t.format = format; t.bytesPerPixel = bytesPerPixel; t.width = width; t.height = height; t.depth = depth;
        t.mips = mips; t.layers = layers; t.cube = cube; t.storage = storage;
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        ci.imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {width, height, depth};
        ci.mipLevels = mips;
        ci.arrayLayers = layers;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(device, &ci, nullptr, &t.image));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, t.image, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &t.memory));
        VK_CHECK(vkBindImageMemory(device, t.image, t.memory, 0));

        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = t.image;
        vi.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : depth > 1 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers};
        VK_CHECK(vkCreateImageView(device, &vi, nullptr, &t.view));
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
        VK_CHECK(vkCreateImageView(device, &vi, nullptr, &t.storageView));
        return t;
    }

    /// Moves the whole image to GENERAL, the layout it keeps: it is valid for sampling, storage and
    /// copies, so no pass has to track layouts. `data`, if given, holds every mip of every layer in
    /// the bake's order (for each layer, for each mip) and is uploaded; otherwise the image is cleared.
    void initTexture(const Texture& t, const std::vector<char>* data) const
    {
        Buffer staging;
        std::vector<VkBufferImageCopy> regions;
        if (data)
        {
            staging = makeBuffer(data->size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            std::memcpy(staging.data, data->data(), data->size());
            VkDeviceSize offset = 0;
            for (uint32_t layer = 0; layer < t.layers; layer++)
            {
                for (uint32_t mip = 0; mip < t.mips; mip++)
                {
                    const uint32_t w = std::max(1u, t.width >> mip), h = std::max(1u, t.height >> mip), d = std::max(1u, t.depth >> mip);
                    VkBufferImageCopy region = {};
                    region.bufferOffset = offset;
                    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
                    region.imageExtent = {w, h, d};
                    regions.push_back(region);
                    offset += (VkDeviceSize)w * h * d * t.bytesPerPixel;
                }
            }
            if (offset != data->size())
            {
                std::printf("FAIL texture data is %zu bytes, the image needs %llu\n", data->size(), (unsigned long long)offset);
                std::exit(2);
            }
        }
        VkCommandBuffer cb = begin();
        VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = t.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t.mips, 0, t.layers};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        if (data)
        {
            vkCmdCopyBufferToImage(cb, staging.buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)regions.size(), regions.data());
        }
        else
        {
            const VkClearColorValue zero = {};
            vkCmdClearColorImage(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &barrier.subresourceRange);
        }
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        submit(cb);
        if (data) destroy(staging);
    }

    /// Mip 0 of layer 0 as raw texels.
    std::vector<char> readTexture(const Texture& t) const
    {
        const VkDeviceSize size = (VkDeviceSize)t.width * t.height * t.depth * t.bytesPerPixel;
        Buffer staging = makeBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        VkBufferImageCopy region = {};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {t.width, t.height, t.depth};
        VkCommandBuffer cb = begin();
        vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
        submit(cb);
        std::vector<char> data((size_t)size);
        std::memcpy(data.data(), staging.data, data.size());
        destroy(staging);
        return data;
    }
};
}  // namespace replay

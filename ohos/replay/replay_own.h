// Compute passes written for the phone runtime itself (ohos/replay/shaders, compiled to SPIR-V by
// slangc on the PC), as opposed to the passes Falcor bakes: storage buffers (or storage images) at
// bindings 0..n-1 of set 0 and one block of push constants.
#pragma once
#include "replay_vk.h"

namespace replay
{
struct OwnPass
{
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t groupSize = 64;
    std::string name;

    /// A pass whose bindings are all storage buffers.
    void create(const Device& dev, VkDescriptorPool pool, const std::string& spvPath, uint32_t bufferCount, uint32_t pushSize, uint32_t group)
    {
        create(dev, pool, spvPath, std::vector<VkDescriptorType>(bufferCount, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), pushSize, group);
    }

    /// A pass with the given descriptor type at each binding (storage buffers and storage images).
    void create(const Device& dev, VkDescriptorPool pool, const std::string& spvPath, const std::vector<VkDescriptorType>& types, uint32_t pushSize,
                uint32_t group)
    {
        name = spvPath.substr(spvPath.find_last_of("/\\") + 1);
        groupSize = group;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < (uint32_t)types.size(); i++) bindings.push_back({i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo sl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sl.bindingCount = (uint32_t)bindings.size();
        sl.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev.device, &sl, nullptr, &setLayout));
        VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize};
        VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &setLayout;
        pl.pushConstantRangeCount = pushSize ? 1 : 0;
        pl.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(dev.device, &pl, nullptr, &layout));
        const std::vector<char> code = readFile(spvPath);
        VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = code.size();
        sm.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module;
        VK_CHECK(vkCreateShaderModule(dev.device, &sm, nullptr, &module));
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        ci.layout = layout;
        VK_CHECK(vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
        vkDestroyShaderModule(dev.device, module, nullptr);
        VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = pool;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &setLayout;
        VK_CHECK(vkAllocateDescriptorSets(dev.device, &da, &set));
    }

    void bind(const Device& dev, uint32_t binding, const Buffer& buffer)
    {
        VkDescriptorBufferInfo info = {buffer.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);
    }

    /// A storage image, in the GENERAL layout.
    void bindImage(const Device& dev, uint32_t binding, VkImageView view)
    {
        VkDescriptorImageInfo info = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);   // `info` is alive until here
    }

    /// One dispatch over `items` items with the push constants, then a barrier for the next pass.
    template <typename Push> void dispatch(VkCommandBuffer cb, const Push& push, uint32_t items) const
    {
        if (items == 0) return;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdDispatch(cb, (items + groupSize - 1) / groupSize, 1, 1);
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    void destroy(const Device& dev)
    {
        if (pipeline) vkDestroyPipeline(dev.device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(dev.device, layout, nullptr);
        if (setLayout) vkDestroyDescriptorSetLayout(dev.device, setLayout, nullptr);
        pipeline = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        setLayout = VK_NULL_HANDLE;
        set = VK_NULL_HANDLE;   // freed with its pool
    }
};

inline VkDescriptorPool makeOwnPool(const Device& dev, uint32_t passes, uint32_t buffers, uint32_t images = 0)
{
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max(buffers, 1u)}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, std::max(images, 1u)}};
    VkDescriptorPoolCreateInfo pci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = passes;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(dev.device, &pci, nullptr, &pool));
    return pool;
}
}  // namespace replay

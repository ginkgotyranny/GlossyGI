// Compute passes written for the phone runtime itself (ohos/replay/shaders, compiled to SPIR-V by
// slangc on the PC), as opposed to the passes Falcor bakes: storage buffers at bindings 0..n-1 of
// set 0 and one block of push constants.
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

    void create(const Device& dev, VkDescriptorPool pool, const std::string& spvPath, uint32_t bufferCount, uint32_t pushSize, uint32_t group)
    {
        name = spvPath.substr(spvPath.find_last_of("/\\") + 1);
        groupSize = group;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < bufferCount; i++) bindings.push_back({i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo sl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sl.bindingCount = bufferCount;
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

    /// One dispatch over `items` items with the push constants, then a barrier for the next pass.
    template <typename Push> void dispatch(VkCommandBuffer cb, const Push& push, uint32_t items)
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
};

inline VkDescriptorPool makeOwnPool(const Device& dev, uint32_t passes, uint32_t buffers)
{
    VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers};
    VkDescriptorPoolCreateInfo pci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = passes;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &size;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(dev.device, &pci, nullptr, &pool));
    return pool;
}
}  // namespace replay

#pragma once

#include <vector>
#include <stdexcept>
#include <vulkan/vulkan.h>

#include "vulkan_context.h"
#include "vulkan_network.h"
#include "compute_graph.h"
#include "helper/logger.h"


class Graph_Executor
{
private:
    const Vulkan_Context &context;
    const Vulkan_Network &network;

    VkCommandBuffer cmd_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> descriptor_cache[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorPool descriptor_pools[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};

    void initResources()
    {
        VkDevice device = context.getDevice();

        VkCommandBufferAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc_info.commandPool = context.getCommandPool();
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

        vkAllocateCommandBuffers(device, &alloc_info, cmd_buffers);

        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4000}};

        VkDescriptorPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.flags = 0;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = pool_sizes;

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pools[i]);
        }
    }

    void insertMemoryBarrier(VkCommandBuffer cmd_buffer) const
    {
        VkMemoryBarrier memory_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &memory_barrier,
            0, nullptr,
            0, nullptr);
    }

    void updateDescriptorSet(VkDescriptorSet descriptor_set, const std::vector<std::shared_ptr<GVector>> &buffers) const
    {
        VkDevice device = context.getDevice();
        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());

        for (std::size_t i = 0; i < buffers.size(); ++i)
        {
            buffer_infos[i].buffer = buffers[i]->getBuffer();
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

public:
    Graph_Executor(const Vulkan_Context &ctx, const Vulkan_Network &net)
        : context(ctx), network(net)
    {
        initResources();
    }

    ~Graph_Executor()
    {
        VkDevice device = context.getDevice();
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (descriptor_pools[i] != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, descriptor_pools[i], nullptr);
            }
        }
        vkFreeCommandBuffers(device, context.getCommandPool(), MAX_FRAMES_IN_FLIGHT, cmd_buffers);
    }

    void resetFrameState(std::uint32_t frame_index)
    {
        vkResetDescriptorPool(context.getDevice(), descriptor_pools[frame_index], 0);
        descriptor_cache[frame_index].clear();
    }

    void compileAndExecute(const Compute_Graph &graph, const std::vector<Buffer_Transfer_Task> &transfers, std::uint32_t frame_index)
    {
        VkDevice device = context.getDevice();

        context.resetFrameFence(frame_index);

        VkCommandBuffer cmd_buffer = cmd_buffers[frame_index];
        vkResetCommandBuffer(cmd_buffer, 0);

        VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_buffer, &begin_info);

        if (!transfers.empty())
        {
            for (const auto &task : transfers)
            {
                VkBufferCopy copy_region{};
                copy_region.srcOffset = task.src_offset;
                copy_region.dstOffset = 0;
                copy_region.size = task.size;
                vkCmdCopyBuffer(cmd_buffer, task.src_buffer, task.dst_buffer, 1, &copy_region);
            }

            VkMemoryBarrier transfer_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            transfer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            transfer_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                cmd_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &transfer_barrier,
                0, nullptr,
                0, nullptr);
        }

        const std::vector<Compute_Node> &nodes = graph.getNodes();

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const Compute_Node &node = nodes[i];

            for (const auto &vec : node.buffers)
            {
                if (vec)
                {
                    vec->markAsUsedInFrame(frame_index);
                }
            }

            VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
            VkDescriptorSetAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc_info.descriptorPool = descriptor_pools[frame_index];
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &layout;

            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
            vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set);
            descriptor_cache[frame_index].push_back(descriptor_set);

            updateDescriptorSet(descriptor_set, node.buffers);

            vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipeline(node.pipeline_id));
            vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &descriptor_set, 0, nullptr);

            if (!node.push_constants_data.empty())
            {
                vkCmdPushConstants(cmd_buffer, network.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<std::uint32_t>(node.push_constants_data.size()), node.push_constants_data.data());
            }

            vkCmdDispatch(cmd_buffer, node.group_x, node.group_y, node.group_z);

            if (i < nodes.size() - 1)
            {
                insertMemoryBarrier(cmd_buffer);
            }
        }

        vkEndCommandBuffer(cmd_buffer);

        VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer;

        vkQueueSubmit(context.getComputeQueue(), 1, &submit_info, context.getFrameFence(frame_index));
    }
};
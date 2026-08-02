#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

#include "vulkan_context.h"
#include "logger.h"


class GVector
{
private:
    const Vulkan_Context &context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    std::size_t buffer_size = 0;
    std::uint32_t used_frame = 0;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &target_buffer, VkDeviceMemory &target_memory) const
    {
        VkDevice device = context.getDevice();

        VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &buffer_info, nullptr, &target_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("GVector::createBuffer: Failed to create buffer", LOG_ERROR, true);
            throw std::runtime_error("Failed to create buffer");
        }

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device, target_buffer, &mem_requirements);

        VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = context.findMemoryType(mem_requirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &alloc_info, nullptr, &target_memory) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, target_buffer, nullptr);
            target_buffer = VK_NULL_HANDLE;
            Logger::logMessage("GVector::createBuffer: Failed to allocate buffer memory", LOG_ERROR, true);
            throw std::runtime_error("Failed to allocate buffer memory");
        }

        if (vkBindBufferMemory(device, target_buffer, target_memory, 0) != VK_SUCCESS)
        {
            vkFreeMemory(device, target_memory, nullptr);
            target_memory = VK_NULL_HANDLE;
            vkDestroyBuffer(device, target_buffer, nullptr);
            target_buffer = VK_NULL_HANDLE;
            Logger::logMessage("GVector::createBuffer: Failed to bind buffer memory", LOG_ERROR, true);
            throw std::runtime_error("Failed to bind buffer memory");
        }
    }

    void copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) const
    {
        VkDevice device = context.getDevice();
        VkCommandPool command_pool = context.getCommandPool();
        VkQueue compute_queue = context.getComputeQueue();

        VkCommandBufferAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &alloc_info, &cmd_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("GVector::copyBuffer: Failed to allocate command buffer", LOG_ERROR, true);
            throw std::runtime_error("Failed to allocate command buffer");
        }

        VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(cmd_buffer, &begin_info);

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(cmd_buffer, src_buffer, dst_buffer, 1, &copy_region);

        vkEndCommandBuffer(cmd_buffer);

        VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer;

        VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(device, &fence_info, nullptr, &fence);

        vkQueueSubmit(compute_queue, 1, &submit_info, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, command_pool, 1, &cmd_buffer);
    }

public:
    explicit GVector(const Vulkan_Context &ctx)
        : context(ctx), buffer_size(0)
    {
    }

    GVector(const Vulkan_Context &ctx, std::size_t num_elements)
        : context(ctx), buffer_size(0)
    {
        allocateMemory(num_elements);
    }

    GVector(const Vulkan_Context &ctx, const std::vector<float> &host_data)
        : context(ctx), buffer_size(0)
    {
        if (host_data.empty())
        {
            return;
        }
        allocateMemory(host_data.size());
        uploadData(host_data);
    }

    ~GVector()
    {
        freeMemory();
    }

    GVector(const GVector &) = delete;
    GVector &operator=(const GVector &) = delete;

    GVector(GVector &&other) noexcept
        : context(other.context), buffer(other.buffer), buffer_memory(other.buffer_memory), buffer_size(other.buffer_size)
    {
        other.buffer = VK_NULL_HANDLE;
        other.buffer_memory = VK_NULL_HANDLE;
        other.buffer_size = 0;
        other.used_frame = 0;
    }

    GVector &operator=(GVector &&other) noexcept
    {
        if (this != &other)
        {
            freeMemory();

            buffer = other.buffer;
            buffer_memory = other.buffer_memory;
            buffer_size = other.buffer_size;
            used_frame = other.used_frame;

            other.buffer = VK_NULL_HANDLE;
            other.buffer_memory = VK_NULL_HANDLE;
            other.buffer_size = 0;
            other.used_frame = 0;
        }
        return *this;
    }

    void allocateMemory(std::size_t num_elements)
    {
        if (num_elements * sizeof(float) == buffer_size)
        {
            return;
        }

        freeMemory();

        if (num_elements == 0)
        {
            return;
        }

        buffer_size = num_elements * sizeof(float);
        createBuffer(buffer_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     buffer,
                     buffer_memory);
    }

    void freeMemory()
    {
        if (buffer != VK_NULL_HANDLE || buffer_memory != VK_NULL_HANDLE)
        {
            context.deferDestruction(used_frame, buffer, buffer_memory);

            buffer = VK_NULL_HANDLE;
            buffer_memory = VK_NULL_HANDLE;
            buffer_size = 0;
        }
    }

    void uploadData(const std::vector<float> &host_data)
    {
        if (host_data.empty() || buffer_size == 0)
        {
            return;
        }

        if (host_data.size() * sizeof(float) != buffer_size)
        {
            throw std::runtime_error("size mismatch");
        }

        std::uint32_t frame = context.getCurrentFrame();

        context.prepareFrame();

        VkBuffer staging_buf = VK_NULL_HANDLE;
        VkDeviceSize staging_offset = 0;
        void *mapped_ptr = context.allocateStagingSpace(frame, buffer_size, staging_buf, staging_offset);

        std::memcpy(mapped_ptr, host_data.data(), buffer_size);
        context.addTransferTask({staging_buf, staging_offset, buffer, buffer_size});
        markAsUsedInFrame(frame);
    }

    void downloadData(std::vector<float> &host_data) const
    {
        if (buffer_size == 0)
        {
            return;
        }

        context.flush();
        vkDeviceWaitIdle(context.getDevice());

        if (host_data.size() != getSize())
        {
            host_data.resize(getSize());
        }

        VkDevice device = context.getDevice();
        VkBuffer staging_buf = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;

        VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = buffer_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &buffer_info, nullptr, &staging_buf) != VK_SUCCESS)
        {
            throw std::runtime_error("GVector::downloadData: Failed to create staging buffer");
        }

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, staging_buf, &mem_reqs);

        VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = context.findMemoryType(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &alloc_info, nullptr, &staging_memory) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, staging_buf, nullptr);
            throw std::runtime_error("GVector::downloadData: Failed to allocate staging memory");
        }

        if (vkBindBufferMemory(device, staging_buf, staging_memory, 0) != VK_SUCCESS)
        {
            vkFreeMemory(device, staging_memory, nullptr);
            vkDestroyBuffer(device, staging_buf, nullptr);
            throw std::runtime_error("GVector::downloadData: Failed to bind staging memory");
        }

        copyBuffer(buffer, staging_buf, buffer_size);

        void *mapped_ptr = nullptr;
        if (vkMapMemory(device, staging_memory, 0, buffer_size, 0, &mapped_ptr) != VK_SUCCESS)
        {
            vkFreeMemory(device, staging_memory, nullptr);
            vkDestroyBuffer(device, staging_buf, nullptr);
            throw std::runtime_error("GVector::downloadData: Failed to map memory");
        }

        std::memcpy(host_data.data(), mapped_ptr, buffer_size);

        vkUnmapMemory(device, staging_memory);
        vkDestroyBuffer(device, staging_buf, nullptr);
        vkFreeMemory(device, staging_memory, nullptr);
    }

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    std::size_t getSize() const
    {
        return buffer_size / sizeof(float);
    }

    void markAsUsedInFrame(std::uint32_t frame_index)
    {
        used_frame = frame_index;
    }
};
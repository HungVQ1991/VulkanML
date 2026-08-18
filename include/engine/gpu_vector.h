#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "helper/logger.h"
#include "vulkan_context.h"
#include "vulkan_sub_allocator.h"

#ifndef ENABLE_GVECTOR_DEBUG_LOGS
#define ENABLE_GVECTOR_DEBUG_LOGS 0
#endif

#if ENABLE_GVECTOR_DEBUG_LOGS
#define GVECTOR_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define GVECTOR_LOG_DEBUG(msg) ((void)0)
#endif

class GVector
{
private:
    const Vulkan_Context &context;
    VkBuffer buffer = VK_NULL_HANDLE;
    Memory_Allocation allocation{};
    std::size_t buffer_size = 0;
    std::uint32_t used_frame = 0;

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    {
        VkDevice device = context.getDevice();

        VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        {
            Logger::logMessage("GVector::createBuffer: Failed to create buffer", LOG_ERROR, true);
            throw std::runtime_error("Failed to create buffer");
        }

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

        allocation = context.allocateMemory(mem_requirements, properties);

        if (vkBindBufferMemory(device, buffer, allocation.memory, allocation.offset) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            context.deferDestruction(used_frame, VK_NULL_HANDLE, allocation);
            Logger::logMessage("GVector::createBuffer: Failed to bind buffer memory", LOG_ERROR, true);
            throw std::runtime_error("Failed to bind buffer memory");
        }
    }

private:
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
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(device, command_pool, 1, &cmd_buffer);
            Logger::logMessage("GVector::copyBuffer: Failed to create fence", LOG_ERROR, true);
            throw std::runtime_error("Failed to create fence");
        }

        if (vkQueueSubmit(compute_queue, 1, &submit_info, fence) != VK_SUCCESS)
        {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, command_pool, 1, &cmd_buffer);
            Logger::logMessage("GVector::copyBuffer: Failed to submit copy command to queue", LOG_ERROR, true);
            throw std::runtime_error("Failed to submit copy command");
        }

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
            Logger::logMessage("GVector::GVector: Host data vector is empty during construction", LOG_WARNING);
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
        : context(other.context), buffer(other.buffer), allocation(other.allocation), buffer_size(other.buffer_size), used_frame(other.used_frame)
    {
        other.buffer = VK_NULL_HANDLE;
        other.allocation = Memory_Allocation{};
        other.buffer_size = 0;
        other.used_frame = 0;
    }

    GVector &operator=(GVector &&other) noexcept
    {
        if (this != &other)
        {
            freeMemory();

            buffer = other.buffer;
            allocation = other.allocation;
            buffer_size = other.buffer_size;
            used_frame = other.used_frame;

            other.buffer = VK_NULL_HANDLE;
            other.allocation = Memory_Allocation{};
            other.buffer_size = 0;
            other.used_frame = 0;
        }
        return *this;
    }

    void allocateMemory(std::size_t num_elements)
    {
        if (num_elements * sizeof(float) == buffer_size)
        {
            GVECTOR_LOG_DEBUG("GVector::allocateMemory: Requested allocation size matches current buffer size, skipping");
            return;
        }

        freeMemory();

        if (num_elements == 0)
        {
            Logger::logMessage("GVector::allocateMemory: Requested allocation size is 0", LOG_WARNING);
            return;
        }

        buffer_size = num_elements * sizeof(float);
        GVECTOR_LOG_DEBUG("GVector::allocateMemory: Allocating " + std::to_string(num_elements) + " elements (" + std::to_string(buffer_size) + " bytes)");

        createBuffer(buffer_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    void freeMemory()
    {
        if (buffer != VK_NULL_HANDLE || allocation.memory != VK_NULL_HANDLE)
        {
            GVECTOR_LOG_DEBUG("GVector::freeMemory: Deferring destruction for buffer of size " + std::to_string(buffer_size) + " bytes");
            context.deferDestruction(used_frame, buffer, allocation);

            buffer = VK_NULL_HANDLE;
            allocation = Memory_Allocation{};
            buffer_size = 0;
        }
    }

    void uploadData(const std::vector<float> &host_data)
    {
        if (host_data.empty() || buffer_size == 0)
        {
            Logger::logMessage("GVector::uploadData: Empty host data or zero GPU buffer size", LOG_WARNING);
            return;
        }

        if (host_data.size() * sizeof(float) != buffer_size)
        {
            Logger::logMessage("GVector::uploadData: Host data size mismatch with GPU buffer size", LOG_ERROR, true);
            throw std::runtime_error("size mismatch");
        }

        std::uint32_t frame = context.getCurrentFrame();

        VkBuffer staging_buf = VK_NULL_HANDLE;
        VkDeviceSize staging_offset = 0;
        void *mapped_ptr = context.allocateStagingSpace(frame, buffer_size, staging_buf, staging_offset);

        std::memcpy(mapped_ptr, host_data.data(), buffer_size);

        context.addTransferTask({staging_buf, staging_offset, buffer, 0, buffer_size});

        markAsUsedInFrame(frame);
    }

    void downloadData(std::vector<float> &host_data) const
    {
        if (buffer_size == 0)
        {
            Logger::logMessage("GVector::downloadData: Attempted download from zero-sized buffer", LOG_WARNING);
            return;
        }

        context.flush();
        if (context.getDevice() != VK_NULL_HANDLE && context.getComputeQueue() != VK_NULL_HANDLE)
        {
            vkQueueWaitIdle(context.getComputeQueue());
        }

        if (host_data.size() != getSize())
        {
            host_data.resize(getSize());
        }

        VkDevice device = context.getDevice();
        VkBuffer staging_buf = VK_NULL_HANDLE;

        VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = buffer_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &buffer_info, nullptr, &staging_buf) != VK_SUCCESS)
        {
            Logger::logMessage("GVector::downloadData: Failed to create staging buffer", LOG_ERROR, true);
            throw std::runtime_error("GVector::downloadData: Failed to create staging buffer");
        }

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, staging_buf, &mem_reqs);

        Memory_Allocation staging_alloc = context.allocateMemory(mem_reqs, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkBindBufferMemory(device, staging_buf, staging_alloc.memory, staging_alloc.offset) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, staging_buf, nullptr);
            context.deferDestruction(context.getCurrentFrame(), VK_NULL_HANDLE, staging_alloc);
            Logger::logMessage("GVector::downloadData: Failed to bind staging memory", LOG_ERROR, true);
            throw std::runtime_error("GVector::downloadData: Failed to bind staging memory");
        }

        copyBuffer(buffer, staging_buf, buffer_size);

        vkDeviceWaitIdle(device);

        void *mapped_ptr = nullptr;
        if (vkMapMemory(device, staging_alloc.memory, staging_alloc.offset, buffer_size, 0, &mapped_ptr) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, staging_buf, nullptr);
            context.deferDestruction(context.getCurrentFrame(), VK_NULL_HANDLE, staging_alloc);
            Logger::logMessage("GVector::downloadData: Failed to map staging memory", LOG_ERROR, true);
            throw std::runtime_error("GVector::downloadData: Failed to map memory");
        }
        std::memcpy(host_data.data(), mapped_ptr, buffer_size);

        vkUnmapMemory(device, staging_alloc.memory);
        context.deferDestruction(context.getCurrentFrame(), staging_buf, staging_alloc);
    }

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    std::size_t getSize() const
    {
        return buffer_size / sizeof(float);
    }

    size_t getSizeBytes() const { return buffer_size; }

    void markAsUsedInFrame(std::uint32_t frame_index)
    {
        used_frame = frame_index;
    }

    const Vulkan_Context &getContext() { return context; }
    VkBuffer getBuffer() { return buffer; }
    VkDevice getDevice() { return context.getDevice(); }
    size_t getBufferSize() { return buffer_size; }
};
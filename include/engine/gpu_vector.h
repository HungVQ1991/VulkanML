#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"
#include "vulkan_context.h"
#include "vulkan_sub_allocator.h"

namespace gpu
{
    class vector
    {
    private:
        static inline std::atomic<std::uint64_t> global_vector_counter{0};

        const Vulkan_Context &context;
        VkBuffer buffer = VK_NULL_HANDLE;
        Memory_Allocation allocation{};
        std::size_t buffer_size_in_bytes = 0;
        std::uint32_t used_frame_index = 0;
        std::uint64_t vector_id = 0;

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_properties)
        {
            VkDevice device = context.getDevice();

            VkBufferCreateInfo buffer_create_information{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = size,
                .usage = usage_flags,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr};

            if (vkCreateBuffer(device, &buffer_create_information, nullptr, &buffer) != VK_SUCCESS)
            {
                Logger::logMessage("gpu::vector::createBuffer: Failed to create buffer",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                throw std::runtime_error("Failed to create buffer");
            }

            VkMemoryRequirements memory_requirements;
            vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

            allocation = context.allocateMemory(memory_requirements, memory_properties);

            if (vkBindBufferMemory(device, buffer, allocation.memory, allocation.offset) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
                context.deferDestruction(used_frame_index, VK_NULL_HANDLE, allocation);
                Logger::logMessage("gpu::vector::createBuffer: Failed to bind buffer memory",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                throw std::runtime_error("Failed to bind buffer memory");
            }

            Logger::logMessage(std::format("gpu::vector::createBuffer: ID: {}, Buffer: {:p}, Chunk: {}, Offset: {}, Size: {} bytes, Frame: {}",
                                           vector_id, static_cast<void *>(buffer), allocation.chunk_index, allocation.offset, buffer_size_in_bytes, used_frame_index),
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
        }

        void copyBuffer(VkBuffer source_buffer, VkBuffer destination_buffer, VkDeviceSize size) const
        {
            VkDevice device = context.getDevice();
            VkCommandPool command_pool = context.getCommandPool();
            VkQueue compute_queue = context.getComputeQueue();

            VkCommandBufferAllocateInfo allocate_information{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = command_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1};

            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &allocate_information, &command_buffer) != VK_SUCCESS)
            {
                Logger::logMessage("gpu::vector::copyBuffer: Failed to allocate command buffer",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to allocate command buffer");
            }

            VkCommandBufferBeginInfo begin_information{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr};

            if (vkBeginCommandBuffer(command_buffer, &begin_information) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                Logger::logMessage("gpu::vector::copyBuffer: Failed to begin command buffer",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to begin command buffer");
            }

            VkBufferCopy copy_region{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = size};
            vkCmdCopyBuffer(command_buffer, source_buffer, destination_buffer, 1, &copy_region);

            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                Logger::logMessage("gpu::vector::copyBuffer: Failed to end command buffer",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to end command buffer");
            }

            VkFenceCreateInfo fence_create_information{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0};

            VkFence fence = VK_NULL_HANDLE;
            if (vkCreateFence(device, &fence_create_information, nullptr, &fence) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                Logger::logMessage("gpu::vector::copyBuffer: Failed to create fence",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::SYNCHRONIZATION);
                throw std::runtime_error("Failed to create fence");
            }

            VkSubmitInfo submit_information{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &command_buffer,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr};

            if (vkQueueSubmit(compute_queue, 1, &submit_information, fence) != VK_SUCCESS)
            {
                vkDestroyFence(device, fence, nullptr);
                vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
                Logger::logMessage("gpu::vector::copyBuffer: Failed to submit copy command to queue",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to submit copy command");
            }

            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
        }

    public:
        explicit vector(const Vulkan_Context &_context)
            : context(_context), buffer_size_in_bytes(0), used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
        {
        }

        vector(const Vulkan_Context &_context, std::size_t _element_count)
            : context(_context), buffer_size_in_bytes(0), used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
        {
            allocateMemory(_element_count);
        }

        vector(const Vulkan_Context &_context, const std::vector<float> &_host_data)
            : context(_context), buffer_size_in_bytes(0), used_frame_index(_context.getCurrentFrame()), vector_id(++global_vector_counter)
        {
            if (_host_data.empty())
            {
                Logger::logMessage("gpu::vector::vector: Host data vector is empty during construction",
                                   Log_Level::LOG_WARNING,
                                   false,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                return;
            }
            allocateMemory(_host_data.size());
            uploadData(_host_data);
        }

        ~vector()
        {
            freeMemory();
        }

        vector(const vector &) = delete;
        vector &operator=(const vector &) = delete;

        vector(vector &&other) noexcept
            : context(other.context), buffer(other.buffer), allocation(other.allocation),
              buffer_size_in_bytes(other.buffer_size_in_bytes), used_frame_index(other.used_frame_index), vector_id(other.vector_id)
        {
            other.buffer = VK_NULL_HANDLE;
            other.allocation = Memory_Allocation{};
            other.buffer_size_in_bytes = 0;
            other.used_frame_index = 0;
            other.vector_id = 0;
        }

        vector &operator=(vector &&other) noexcept
        {
            if (this != &other)
            {
                freeMemory();

                buffer = other.buffer;
                allocation = other.allocation;
                buffer_size_in_bytes = other.buffer_size_in_bytes;
                used_frame_index = other.used_frame_index;
                vector_id = other.vector_id;

                other.buffer = VK_NULL_HANDLE;
                other.allocation = Memory_Allocation{};
                other.buffer_size_in_bytes = 0;
                other.used_frame_index = 0;
                other.vector_id = 0;
            }
            return *this;
        }

        void allocateMemory(std::size_t element_count)
        {
            VkDeviceSize required_byte_size = element_count * sizeof(float);
            if (required_byte_size == buffer_size_in_bytes && buffer != VK_NULL_HANDLE)
            {
                Logger::logMessage(std::format("gpu::vector::allocateMemory: Requested size matches current buffer size (ID: {}), skipping", vector_id),
                                   Log_Level::LOG_DEBUG,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                return;
            }

            freeMemory();

            if (element_count == 0)
            {
                Logger::logMessage("gpu::vector::allocateMemory: Requested allocation size is 0",
                                   Log_Level::LOG_WARNING,
                                   false,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                return;
            }

            if (vector_id == 0)
            {
                vector_id = ++global_vector_counter;
            }

            buffer_size_in_bytes = required_byte_size;
            used_frame_index = context.getCurrentFrame();

            Logger::logMessage(std::format("gpu::vector::allocateMemory: Allocating {} elements ({} bytes) for ID: {}",
                                           element_count, buffer_size_in_bytes, vector_id),
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);

            createBuffer(buffer_size_in_bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }

        void freeMemory()
        {
            if (buffer != VK_NULL_HANDLE || allocation.memory != VK_NULL_HANDLE)
            {
                Logger::logMessage(std::format("gpu::vector::freeMemory: ID: {}, Buffer: {:p}, Chunk: {}, Offset: {}, Size: {} bytes, Frame: {}",
                                               vector_id, static_cast<void *>(buffer), allocation.chunk_index, allocation.offset, buffer_size_in_bytes, used_frame_index),
                                   Log_Level::LOG_DEBUG,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);

                context.deferDestruction(used_frame_index, buffer, allocation);

                buffer = VK_NULL_HANDLE;
                allocation = Memory_Allocation{};
                buffer_size_in_bytes = 0;
            }
        }

        void uploadData(const std::vector<float> &host_data)
        {
            if (host_data.empty() || buffer_size_in_bytes == 0)
            {
                Logger::logMessage("gpu::vector::uploadData: Empty host data or zero GPU buffer size",
                                   Log_Level::LOG_WARNING,
                                   false,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                return;
            }

            if (host_data.size() * sizeof(float) != buffer_size_in_bytes)
            {
                Logger::logMessage("gpu::vector::uploadData: Host data size mismatch with GPU buffer size",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("size mismatch");
            }

            std::uint32_t frame_index = context.getCurrentFrame();

            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VkDeviceSize staging_offset = 0;
            void *mapped_pointer = context.allocateStagingSpace(frame_index, buffer_size_in_bytes, staging_buffer, staging_offset);

            std::memcpy(mapped_pointer, host_data.data(), buffer_size_in_bytes);

            Buffer_Transfer_Task transfer_task{
                .source_buffer = staging_buffer,
                .source_offset = staging_offset,
                .destination_buffer = buffer,
                .destination_offset = 0,
                .size = buffer_size_in_bytes};

            context.addTransferTask(transfer_task);

            markAsUsedInFrame(frame_index);
        }

        void downloadData(std::vector<float> &host_data) const
        {
            if (buffer_size_in_bytes == 0 || buffer == VK_NULL_HANDLE)
            {
                Logger::logMessage("gpu::vector::downloadData: Attempted download from zero-sized or invalid buffer",
                                   Log_Level::LOG_WARNING,
                                   false,
                                   0,
                                   Log_Feature::MEMORY_TRANSFER);
                return;
            }

            context.flush();
            if (context.getDevice() != VK_NULL_HANDLE && context.getComputeQueue() != VK_NULL_HANDLE)
            {
                vkQueueWaitIdle(context.getComputeQueue());
            }

            if (host_data.size() != getElementCount())
            {
                host_data.resize(getElementCount());
            }

            VkDevice device = context.getDevice();
            VkBuffer staging_buffer = VK_NULL_HANDLE;

            VkBufferCreateInfo buffer_create_information{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = buffer_size_in_bytes,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr};

            if (vkCreateBuffer(device, &buffer_create_information, nullptr, &staging_buffer) != VK_SUCCESS)
            {
                Logger::logMessage("gpu::vector::downloadData: Failed to create staging buffer",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("gpu::vector::downloadData: Failed to create staging buffer");
            }

            VkMemoryRequirements memory_requirements;
            vkGetBufferMemoryRequirements(device, staging_buffer, &memory_requirements);

            Memory_Allocation staging_allocation = context.allocateMemory(memory_requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            if (vkBindBufferMemory(device, staging_buffer, staging_allocation.memory, staging_allocation.offset) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, staging_buffer, nullptr);
                context.deferDestruction(context.getCurrentFrame(), VK_NULL_HANDLE, staging_allocation);
                Logger::logMessage("gpu::vector::downloadData: Failed to bind staging memory",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("gpu::vector::downloadData: Failed to bind staging memory");
            }

            copyBuffer(buffer, staging_buffer, buffer_size_in_bytes);

            void *mapped_pointer = nullptr;
            if (vkMapMemory(device, staging_allocation.memory, staging_allocation.offset, buffer_size_in_bytes, 0, &mapped_pointer) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, staging_buffer, nullptr);
                context.deferDestruction(context.getCurrentFrame(), VK_NULL_HANDLE, staging_allocation);
                Logger::logMessage("gpu::vector::downloadData: Failed to map staging memory",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("gpu::vector::downloadData: Failed to map memory");
            }

            std::memcpy(host_data.data(), mapped_pointer, buffer_size_in_bytes);

            vkUnmapMemory(device, staging_allocation.memory);
            context.deferDestruction(context.getCurrentFrame(), staging_buffer, staging_allocation);
        }

        [[nodiscard]] std::uint64_t getId() const noexcept { return vector_id; }
        [[nodiscard]] std::uint64_t getVectorId() const noexcept { return vector_id; }
        [[nodiscard]] const Memory_Allocation &getAllocation() const noexcept { return allocation; }
        [[nodiscard]] VkBuffer getBuffer() const noexcept { return buffer; }
        [[nodiscard]] std::size_t getSize() const noexcept { return buffer_size_in_bytes / sizeof(float); }
        [[nodiscard]] std::size_t getElementCount() const noexcept { return buffer_size_in_bytes / sizeof(float); }
        [[nodiscard]] std::size_t getSizeBytes() const noexcept { return buffer_size_in_bytes; }
        [[nodiscard]] std::size_t getByteSize() const noexcept { return buffer_size_in_bytes; }
        [[nodiscard]] std::size_t getBufferSizeInBytes() const noexcept { return buffer_size_in_bytes; }
        [[nodiscard]] const Vulkan_Context &getContext() const noexcept { return context; }
        [[nodiscard]] VkDevice getDevice() const noexcept { return context.getDevice(); }
        [[nodiscard]] std::uint32_t getUsedFrameIndex() const noexcept { return used_frame_index; }
        [[nodiscard]] bool isEmpty() const noexcept { return buffer_size_in_bytes == 0 || buffer == VK_NULL_HANDLE; }

        void markAsUsedInFrame(std::uint32_t frame_index) noexcept
        {
            used_frame_index = frame_index;
        }
    };
}
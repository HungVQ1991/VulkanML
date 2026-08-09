#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"

#ifndef ENABLE_ALLOCATOR_DEBUG_LOGS
#define ENABLE_ALLOCATOR_DEBUG_LOGS 0
#endif

#if ENABLE_ALLOCATOR_DEBUG_LOGS
#define ALLOCATOR_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define ALLOCATOR_LOG_DEBUG(msg) ((void)0)
#endif

class Vulkan_Context;

struct Memory_Allocation
{
    VkDeviceMemory memory;
    VkDeviceSize offset;
    VkDeviceSize size;
    std::size_t block_index;
};

struct Free_Block
{
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct Memory_Chunk
{
    VkDeviceMemory device_memory;
    VkDeviceSize chunk_size;
    std::uint32_t memory_type_index;
    std::vector<Free_Block> free_blocks;
};

class Vulkan_Sub_Allocator
{
private:
    VkDevice device;
    const Vulkan_Context &context;
    std::vector<Memory_Chunk> chunks;
    VkDeviceSize default_chunk_size = 256 * 1024 * 1024;
    mutable std::mutex allocator_mutex;

public:
    Vulkan_Sub_Allocator(VkDevice dev, const Vulkan_Context &ctx)
        : device(dev), context(ctx)
    {
        ALLOCATOR_LOG_DEBUG("Vulkan_Sub_Allocator::Vulkan_Sub_Allocator: Initializing sub allocator");
    }

    ~Vulkan_Sub_Allocator()
    {
        ALLOCATOR_LOG_DEBUG("Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator: Freeing memory chunks");
        std::lock_guard<std::mutex> lock(allocator_mutex);
        for (const auto &chunk : chunks)
        {
            if (chunk.device_memory != VK_NULL_HANDLE)
            {
                vkFreeMemory(device, chunk.device_memory, nullptr);
            }
            else
            {
                Logger::logMessage("Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator: Null device_memory handle encountered in chunk", LOG_WARNING);
            }
        }
    }

    Memory_Allocation allocate(const VkMemoryRequirements &reqs, VkMemoryPropertyFlags properties);
    void free(const Memory_Allocation &alloc);
};
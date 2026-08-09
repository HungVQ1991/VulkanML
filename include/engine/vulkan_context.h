#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"
#include "vulkan_sub_allocator.h"

#ifndef ENABLE_CONTEXT_DEBUG_LOGS
#define ENABLE_CONTEXT_DEBUG_LOGS 0
#endif

#if ENABLE_CONTEXT_DEBUG_LOGS
#define CONTEXT_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define CONTEXT_LOG_DEBUG(msg) ((void)0)
#endif

const bool DEBUG_VALIDATION = true;

constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Resource_Garbage
{
    VkBuffer buffer;
    Memory_Allocation allocation;
};

struct Buffer_Transfer_Task
{
    VkBuffer src_buffer;
    VkDeviceSize src_offset;
    VkBuffer dst_buffer;
    VkDeviceSize size;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
    void *p_user_data)
{
    if (p_callback_data && p_callback_data->pMessage)
    {
        Logger::logMessage("Vulkan Validation Layer: " + std::string(p_callback_data->pMessage), LOG_WARNING);
    }
    return VK_FALSE;
}

class Vulkan_Context
{
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::uint32_t compute_queue_family_index = 0;

    std::unique_ptr<Vulkan_Sub_Allocator> allocator;

    mutable VkBuffer staging_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable VkDeviceMemory staging_memories[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable void *staging_mapped_ptrs[MAX_FRAMES_IN_FLIGHT]{nullptr, nullptr};
    mutable VkDeviceSize staging_capacities[MAX_FRAMES_IN_FLIGHT]{0, 0};
    mutable VkDeviceSize current_offsets[MAX_FRAMES_IN_FLIGHT]{0, 0};

    struct Staging_Garbage
    {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };
    mutable std::vector<Staging_Garbage> staging_garbage[MAX_FRAMES_IN_FLIGHT];

    mutable std::uint32_t current_frame = 0;
    mutable std::vector<Buffer_Transfer_Task> pending_transfers;

    mutable VkFence fences[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable std::function<void()> flush_callback = nullptr;

    mutable std::mutex garbage_mutex;
    mutable std::vector<Resource_Garbage> garbage_bins[MAX_FRAMES_IN_FLIGHT];

    mutable bool frame_ready[MAX_FRAMES_IN_FLIGHT]{false, false};

    void initFences()
    {
        VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateFence(device, &fence_info, nullptr, &fences[i]) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::initFences: Failed to create fence for frame " + std::to_string(i), LOG_ERROR, true);
                throw std::runtime_error("Failed to create fence");
            }
        }
    }

    void initInstance()
    {
        VkApplicationInfo app_info{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app_info.pApplicationName = "Matrix_Compute";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No_Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_4;

        const char *validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
        const char *instance_extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

        VkDebugUtilsMessengerCreateInfoEXT debug_info{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_info.pfnUserCallback = debugCallback;

        VkInstanceCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        if (DEBUG_VALIDATION)
            create_info.pNext = &debug_info;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = DEBUG_VALIDATION ? 1 : 0;
        create_info.ppEnabledLayerNames = DEBUG_VALIDATION ? validation_layers : nullptr;
        create_info.enabledExtensionCount = DEBUG_VALIDATION ? 1 : 0;
        create_info.ppEnabledExtensionNames = DEBUG_VALIDATION ? instance_extensions : nullptr;

        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::initInstance: Failed to create Vulkan instance", LOG_ERROR, true);
            throw std::runtime_error("Failed to create Vulkan instance");
        }
    }

    void pickPhysicalDevice()
    {
        std::uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

        if (device_count == 0)
        {
            Logger::logMessage("Vulkan_Context::pickPhysicalDevice: No physical devices with Vulkan support found", LOG_ERROR, true);
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

        constexpr std::array<size_t, 5> priority_order = {2, 1, 3, 4, 0};
        VkPhysicalDevice best_device = VK_NULL_HANDLE;
        std::uint32_t best_compute_family_index = 0;
        size_t best_rank = priority_order.size();

        for (const auto &d : devices)
        {
            VkPhysicalDeviceProperties device_properties;
            vkGetPhysicalDeviceProperties(d, &device_properties);

            std::uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &queue_family_count, nullptr);

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &queue_family_count, queue_families.data());

            for (uint32_t i = 0; i < queue_family_count; ++i)
            {
                if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                {
                    auto it = std::find(priority_order.begin(), priority_order.end(), static_cast<size_t>(device_properties.deviceType));
                    if (it != priority_order.end())
                    {
                        size_t current_rank = static_cast<size_t>(std::distance(priority_order.begin(), it));
                        if (current_rank < best_rank)
                        {
                            best_rank = current_rank;
                            best_device = d;
                            best_compute_family_index = i;
                        }
                    }
                    break;
                }
            }
        }

        if (best_device == VK_NULL_HANDLE)
        {
            Logger::logMessage("Vulkan_Context::pickPhysicalDevice: Failed to find a suitable GPU with compute queue support", LOG_ERROR, true);
            throw std::runtime_error("Failed to find a suitable GPU");
        }

        physical_device = best_device;
        compute_queue_family_index = best_compute_family_index;
    }

    void createLogicalDevice()
    {
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_create_info.queueFamilyIndex = compute_queue_family_index;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        std::vector<const char *> required_extensions = {};
        VkPhysicalDeviceFeatures enabled_features{};

        VkDeviceCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = &queue_create_info;
        create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
        create_info.ppEnabledExtensionNames = required_extensions.empty() ? nullptr : required_extensions.data();
        create_info.pEnabledFeatures = &enabled_features;

        if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::createLogicalDevice: Failed to create logical Vulkan device", LOG_ERROR, true);
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = compute_queue_family_index;
        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::createCommandPool: Failed to create command pool", LOG_ERROR, true);
            throw std::runtime_error("Failed to create command pool");
        }
    }

public:
    Vulkan_Context()
    {
        CONTEXT_LOG_DEBUG("Vulkan_Context::Vulkan_Context: Initializing Vulkan Context");
        initInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        initFences();
        allocator = std::make_unique<Vulkan_Sub_Allocator>(device, *this);
    }

    ~Vulkan_Context()
    {
        CONTEXT_LOG_DEBUG("Vulkan_Context::~Vulkan_Context: Destroying Vulkan Context");
        if (device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);
        }

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            cleanGarbage(i);

            if (fences[i] != VK_NULL_HANDLE)
                vkDestroyFence(device, fences[i], nullptr);

            if (staging_buffers[i] != VK_NULL_HANDLE)
            {
                vkUnmapMemory(device, staging_memories[i]);
                vkDestroyBuffer(device, staging_buffers[i], nullptr);
                vkFreeMemory(device, staging_memories[i], nullptr);
            }
        }

        allocator.reset();

        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, command_pool, nullptr);

        if (device != VK_NULL_HANDLE)
            vkDestroyDevice(device, nullptr);

        if (instance != VK_NULL_HANDLE)
            vkDestroyInstance(instance, nullptr);
    }

    Memory_Allocation allocateMemory(const VkMemoryRequirements &reqs, VkMemoryPropertyFlags properties) const
    {
        return allocator->allocate(reqs, properties);
    }

    void deferDestruction(uint32_t used_frame, VkBuffer buf, const Memory_Allocation &alloc) const
    {
        if (buf != VK_NULL_HANDLE || alloc.memory != VK_NULL_HANDLE)
        {
            std::lock_guard<std::mutex> lock(garbage_mutex);
            garbage_bins[used_frame].push_back({buf, alloc});
        }
        else
        {
            Logger::logMessage("Vulkan_Context::deferDestruction: Both buffer and memory handle are null", LOG_WARNING);
        }
    }

    VkInstance getInstance() const { return instance; }
    VkPhysicalDevice getPhysicalDevice() const { return physical_device; }
    VkDevice getDevice() const { return device; }
    VkQueue getComputeQueue() const { return compute_queue; }
    VkCommandPool getCommandPool() const { return command_pool; }
    std::uint32_t getComputeQueueFamilyIndex() const { return compute_queue_family_index; }

    std::uint32_t findMemoryType(std::uint32_t type_filter, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
        for (std::uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i)
        {
            if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        Logger::logMessage("Vulkan_Context::findMemoryType: Failed to find suitable memory type", LOG_ERROR, true);
        throw std::runtime_error("Failed to find suitable memory type");
    }

    void *allocateStagingSpace(std::uint32_t frame_index, VkDeviceSize size, VkBuffer &out_buffer, VkDeviceSize &out_offset) const
    {
        if (frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage("Vulkan_Context::allocateStagingSpace: frame_index out of bounds (" + std::to_string(frame_index) + ")", LOG_WARNING);
            frame_index = frame_index % MAX_FRAMES_IN_FLIGHT;
        }

        if (current_offsets[frame_index] + size > staging_capacities[frame_index] || staging_buffers[frame_index] == VK_NULL_HANDLE)
        {
            constexpr VkDeviceSize INITIAL_STAGING_CAPACITY = 32 * 1024 * 1024;
            VkDeviceSize calculated_size = std::max(staging_capacities[frame_index] * 2, current_offsets[frame_index] + size + 1024 * 1024);
            VkDeviceSize new_capacity = std::max(INITIAL_STAGING_CAPACITY, calculated_size);

            Logger::logMessage("Vulkan_Context::allocateStagingSpace: Reallocating staging buffer for frame " + std::to_string(frame_index) + " to new capacity " + std::to_string(new_capacity) + " bytes", LOG_WARNING);

            VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer_info.size = new_capacity;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkBuffer new_buffer = VK_NULL_HANDLE;
            if (vkCreateBuffer(device, &buffer_info, nullptr, &new_buffer) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to create new staging buffer", LOG_ERROR, true);
                throw std::runtime_error("Failed to create staging buffer");
            }

            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(device, new_buffer, &mem_reqs);

            VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            alloc_info.allocationSize = mem_reqs.size;
            alloc_info.memoryTypeIndex = findMemoryType(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            VkDeviceMemory new_memory = VK_NULL_HANDLE;
            if (vkAllocateMemory(device, &alloc_info, nullptr, &new_memory) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, new_buffer, nullptr);
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to allocate memory for staging buffer", LOG_ERROR, true);
                throw std::runtime_error("Failed to allocate staging memory");
            }

            if (vkBindBufferMemory(device, new_buffer, new_memory, 0) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, new_buffer, nullptr);
                vkFreeMemory(device, new_memory, nullptr);
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to bind staging buffer memory", LOG_ERROR, true);
                throw std::runtime_error("Failed to bind staging memory");
            }

            void *new_mapped_ptr = nullptr;
            if (vkMapMemory(device, new_memory, 0, new_capacity, 0, &new_mapped_ptr) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, new_buffer, nullptr);
                vkFreeMemory(device, new_memory, nullptr);
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to map staging memory", LOG_ERROR, true);
                throw std::runtime_error("Failed to map staging memory");
            }

            if (staging_buffers[frame_index] != VK_NULL_HANDLE)
            {
                if (current_offsets[frame_index] > 0)
                {
                    std::memcpy(new_mapped_ptr, staging_mapped_ptrs[frame_index], current_offsets[frame_index]);
                }

                vkUnmapMemory(device, staging_memories[frame_index]);
                staging_garbage[frame_index].push_back({staging_buffers[frame_index], staging_memories[frame_index]});

                for (auto &task : pending_transfers)
                {
                    if (task.src_buffer == staging_buffers[frame_index])
                        task.src_buffer = new_buffer;
                }
            }

            staging_buffers[frame_index] = new_buffer;
            staging_memories[frame_index] = new_memory;
            staging_mapped_ptrs[frame_index] = new_mapped_ptr;
            staging_capacities[frame_index] = new_capacity;
        }

        out_buffer = staging_buffers[frame_index];
        out_offset = current_offsets[frame_index];
        void *ptr = static_cast<char *>(staging_mapped_ptrs[frame_index]) + out_offset;
        current_offsets[frame_index] += size;

        CONTEXT_LOG_DEBUG("Vulkan_Context::allocateStagingSpace: Allocated " + std::to_string(size) + " bytes in staging buffer for frame " + std::to_string(frame_index));

        return ptr;
    }

    VkFence getFrameFence(std::uint32_t frame_index) const { return fences[frame_index]; }

    void resetFrameFence(std::uint32_t frame_index) const
    {
        if (frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage("Vulkan_Context::resetFrameFence: frame_index out of bounds (" + std::to_string(frame_index) + ")", LOG_WARNING);
            return;
        }
        vkResetFences(device, 1, &fences[frame_index]);
    }

    void registerFlushCallback(std::function<void()> cb) const { flush_callback = cb; }

    void flush() const
    {
        if (flush_callback)
        {
            CONTEXT_LOG_DEBUG("Vulkan_Context::flush: Executing flush callback");
            flush_callback();
        }
        else
        {
            Logger::logMessage("Vulkan_Context::flush: Flush callback is not registered", LOG_WARNING);
        }
    }

    void resetStagingOffset(std::uint32_t frame_index) const
    {
        if (frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage("Vulkan_Context::resetStagingOffset: frame_index out of bounds (" + std::to_string(frame_index) + ")", LOG_WARNING);
            return;
        }
        current_offsets[frame_index] = 0;
    }

    std::uint32_t getCurrentFrame() const { return current_frame; }

    void addTransferTask(const Buffer_Transfer_Task &task) const
    {
        pending_transfers.push_back(task);
    }

    const std::vector<Buffer_Transfer_Task> &getTransferTasks() const
    {
        return pending_transfers;
    }

    void clearTransferTasks() const
    {
        pending_transfers.clear();
    }

    void cleanGarbage(std::uint32_t frame_index) const
    {
        if (frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage("Vulkan_Context::cleanGarbage: frame_index out of bounds (" + std::to_string(frame_index) + ")", LOG_WARNING);
            return;
        }

        for (const auto &stg : staging_garbage[frame_index])
        {
            if (stg.buffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device, stg.buffer, nullptr);
            if (stg.memory != VK_NULL_HANDLE)
                vkFreeMemory(device, stg.memory, nullptr);
        }
        staging_garbage[frame_index].clear();

        std::vector<Resource_Garbage> local_bin;
        {
            std::lock_guard<std::mutex> lock(garbage_mutex);
            local_bin.swap(garbage_bins[frame_index]);
        }

        if (!local_bin.empty())
        {
            CONTEXT_LOG_DEBUG("Vulkan_Context::cleanGarbage: Cleaning " + std::to_string(local_bin.size()) + " garbage items for frame " + std::to_string(frame_index));
        }

        for (const auto &garbage : local_bin)
        {
            if (garbage.buffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(device, garbage.buffer, nullptr);
            }
            if (garbage.allocation.memory != VK_NULL_HANDLE)
            {
                allocator->free(garbage.allocation);
            }
        }
    }

    void prepareFrame() const
    {
        if (!frame_ready[current_frame])
        {
            if (device != VK_NULL_HANDLE && fences[current_frame] != VK_NULL_HANDLE)
            {
                vkWaitForFences(device, 1, &fences[current_frame], VK_TRUE, UINT64_MAX);

                cleanGarbage(current_frame);
                frame_ready[current_frame] = true;
            }
            else
            {
                Logger::logMessage("Vulkan_Context::prepareFrame: Invalid device or fence handle for frame " + std::to_string(current_frame), LOG_ERROR, true);
                throw std::runtime_error("invalid handle");
            }
        }
    }

    void advanceFrame() const
    {
        frame_ready[current_frame] = false;
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        CONTEXT_LOG_DEBUG("Vulkan_Context::advanceFrame: Advanced current frame to " + std::to_string(current_frame));
    }
};

inline Memory_Allocation Vulkan_Sub_Allocator::allocate(const VkMemoryRequirements &reqs, VkMemoryPropertyFlags properties)
{
    std::lock_guard<std::mutex> lock(allocator_mutex);
    std::uint32_t mem_type_idx = context.findMemoryType(reqs.memoryTypeBits, properties);

    for (std::size_t i = 0; i < chunks.size(); ++i)
    {
        auto &chunk = chunks[i];

        if (chunk.memory_type_index != mem_type_idx)
        {
            continue;
        }

        for (auto it = chunk.free_blocks.begin(); it != chunk.free_blocks.end(); ++it)
        {
            VkDeviceSize aligned_offset = (it->offset + reqs.alignment - 1) & ~(reqs.alignment - 1);
            if (aligned_offset < it->offset)
            {
                continue;
            }

            VkDeviceSize padding = aligned_offset - it->offset;

            if (it->size >= reqs.size + padding)
            {
                Memory_Allocation alloc{chunk.device_memory, aligned_offset, reqs.size, i};

                VkDeviceSize back_remainder_offset = aligned_offset + reqs.size;
                VkDeviceSize back_remainder_size = it->size - (reqs.size + padding);

                if (padding > 0 && back_remainder_size > 0)
                {
                    it->size = padding;
                    chunk.free_blocks.insert(it + 1, {back_remainder_offset, back_remainder_size});
                }
                else if (padding > 0)
                {
                    it->size = padding;
                }
                else if (back_remainder_size > 0)
                {
                    it->offset = back_remainder_offset;
                    it->size = back_remainder_size;
                }
                else
                {
                    chunk.free_blocks.erase(it);
                }

                return alloc;
            }
        }
    }

    VkDeviceSize new_size = std::max(reqs.size, default_chunk_size);
    Logger::logMessage("Vulkan_Sub_Allocator::allocate: Allocating new memory chunk of size " + std::to_string(new_size) + " bytes", LOG_WARNING);

    VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = new_size;
    alloc_info.memoryTypeIndex = mem_type_idx;

    VkDeviceMemory new_memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &alloc_info, nullptr, &new_memory) != VK_SUCCESS)
    {
        Logger::logMessage("Vulkan_Sub_Allocator::allocate: Failed to allocate new memory chunk", LOG_ERROR, true);
        throw std::runtime_error("Failed to allocate new memory chunk");
    }

    Memory_Chunk new_chunk;
    new_chunk.device_memory = new_memory;
    new_chunk.chunk_size = new_size;
    new_chunk.memory_type_index = mem_type_idx;

    if (new_size > reqs.size)
    {
        new_chunk.free_blocks.push_back({reqs.size, new_size - reqs.size});
    }

    chunks.push_back(std::move(new_chunk));

    return {new_memory, 0, reqs.size, chunks.size() - 1};
}

inline void Vulkan_Sub_Allocator::free(const Memory_Allocation &alloc)
{
    std::lock_guard<std::mutex> lock(allocator_mutex);
    if (alloc.block_index >= chunks.size())
    {
        Logger::logMessage("Vulkan_Sub_Allocator::free: Block index out of bounds (" + std::to_string(alloc.block_index) + ")", LOG_WARNING);
        return;
    }

    auto &chunk = chunks[alloc.block_index];

    auto it = std::lower_bound(chunk.free_blocks.begin(), chunk.free_blocks.end(), alloc.offset,
                               [](const Free_Block &block, VkDeviceSize offset)
                               {
                                   return block.offset < offset;
                               });

    it = chunk.free_blocks.insert(it, {alloc.offset, alloc.size});

    auto next_it = it + 1;
    if (next_it != chunk.free_blocks.end() && it->offset + it->size == next_it->offset)
    {
        it->size += next_it->size;
        chunk.free_blocks.erase(next_it);
    }

    if (it != chunk.free_blocks.begin())
    {
        auto prev_it = it - 1;
        if (prev_it->offset + prev_it->size == it->offset)
        {
            prev_it->size += it->size;
            chunk.free_blocks.erase(it);
        }
    }
}
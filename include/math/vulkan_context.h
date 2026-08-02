#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <string>
#include <stdexcept>
#include <functional>
#include <vulkan/vulkan.h>

#include "logger.h"

const bool DEBUG_VALIDATION = false;

constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Resource_Garbage
{
    VkBuffer buffer;
    VkDeviceMemory memory;
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
    std::cerr << "Validation layer: " << p_callback_data->pMessage << std::endl;
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

    mutable VkBuffer staging_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable VkDeviceMemory staging_memories[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable void *staging_mapped_ptrs[MAX_FRAMES_IN_FLIGHT]{nullptr, nullptr};
    mutable VkDeviceSize staging_capacities[MAX_FRAMES_IN_FLIGHT]{0, 0};
    mutable VkDeviceSize current_offsets[MAX_FRAMES_IN_FLIGHT]{0, 0};

    mutable std::uint32_t current_frame = 0;
    mutable std::vector<Buffer_Transfer_Task> pending_transfers;

    mutable VkFence fences[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable std::function<void()> flush_callback = nullptr;

    mutable std::vector<Resource_Garbage> garbage_bins[MAX_FRAMES_IN_FLIGHT];

    mutable bool frame_ready[MAX_FRAMES_IN_FLIGHT]{false, false};

    void initFences()
    {
        VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkCreateFence(device, &fence_info, nullptr, &fences[i]);
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
        if (DEBUG_VALIDATION) create_info.pNext = &debug_info;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = DEBUG_VALIDATION ? 1 : 0;
        create_info.ppEnabledLayerNames = DEBUG_VALIDATION ? validation_layers : nullptr;
        create_info.enabledExtensionCount = DEBUG_VALIDATION ? 1 : 0;
        create_info.ppEnabledExtensionNames = DEBUG_VALIDATION ? instance_extensions : nullptr;

        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan instance");
        }
    }

    void pickPhysicalDevice()
    {
        std::uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

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

        vkCreateDevice(physical_device, &create_info, nullptr, &device);
        vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = compute_queue_family_index;
        vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
    }

public:
    Vulkan_Context()
    {
        initInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        initFences();
    }

    ~Vulkan_Context()
    {
        if (device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);
        }

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            for (const auto &garbage : garbage_bins[i])
            {
                if (garbage.buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, garbage.buffer, nullptr);
                if (garbage.memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, garbage.memory, nullptr);
            }
            garbage_bins[i].clear();

            if (fences[i] != VK_NULL_HANDLE)
                vkDestroyFence(device, fences[i], nullptr);

            if (staging_buffers[i] != VK_NULL_HANDLE)
            {
                vkUnmapMemory(device, staging_memories[i]);
                vkDestroyBuffer(device, staging_buffers[i], nullptr);
                vkFreeMemory(device, staging_memories[i], nullptr);
            }
        }

        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, command_pool, nullptr);

        if (device != VK_NULL_HANDLE)
            vkDestroyDevice(device, nullptr);

        if (instance != VK_NULL_HANDLE)
            vkDestroyInstance(instance, nullptr);
    }

    void deferDestruction(uint32_t used_frame, VkBuffer buf, VkDeviceMemory mem) const
    {
        if (buf != VK_NULL_HANDLE || mem != VK_NULL_HANDLE)
            garbage_bins[used_frame].push_back({buf, mem});
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
        throw std::runtime_error("Failed to find suitable memory type");
    }

    void *allocateStagingSpace(std::uint32_t frame_index, VkDeviceSize size, VkBuffer &out_buffer, VkDeviceSize &out_offset) const
    {
        if (current_offsets[frame_index] + size > staging_capacities[frame_index])
        {
            VkDeviceSize new_capacity = std::max(staging_capacities[frame_index] * 2, current_offsets[frame_index] + size + 1024 * 1024);

            VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer_info.size = new_capacity;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkBuffer new_buffer;
            vkCreateBuffer(device, &buffer_info, nullptr, &new_buffer);

            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(device, new_buffer, &mem_reqs);

            VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            alloc_info.allocationSize = mem_reqs.size;
            alloc_info.memoryTypeIndex = findMemoryType(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            VkDeviceMemory new_memory;
            vkAllocateMemory(device, &alloc_info, nullptr, &new_memory);
            vkBindBufferMemory(device, new_buffer, new_memory, 0);

            void *new_mapped_ptr;
            vkMapMemory(device, new_memory, 0, new_capacity, 0, &new_mapped_ptr);

            if (staging_buffers[frame_index] != VK_NULL_HANDLE)
            {
                if (current_offsets[frame_index] > 0)
                {
                    std::memcpy(new_mapped_ptr, staging_mapped_ptrs[frame_index], current_offsets[frame_index]);
                }

                vkUnmapMemory(device, staging_memories[frame_index]);
                vkDestroyBuffer(device, staging_buffers[frame_index], nullptr);
                vkFreeMemory(device, staging_memories[frame_index], nullptr);

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

        return ptr;
    }

    VkFence getFrameFence(std::uint32_t frame_index) const { return fences[frame_index]; }
    void resetFrameFence(std::uint32_t frame_index) const { vkResetFences(device, 1, &fences[frame_index]); }
    void registerFlushCallback(std::function<void()> cb) const { flush_callback = cb; }
    void flush() const { if (flush_callback) flush_callback(); }

    void resetStagingOffset(std::uint32_t frame_index) const
    {
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
        for (auto [buf, mem] : garbage_bins[frame_index])
        {
            if (buf != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(device, buf, nullptr);
            }
            if (mem != VK_NULL_HANDLE)
            {
                vkFreeMemory(device, mem, nullptr);
            }
        }
        garbage_bins[frame_index].clear();
    }

    void prepareFrame() const
    {
        if (!frame_ready[current_frame])
        {
            if (device != VK_NULL_HANDLE && fences[current_frame] != VK_NULL_HANDLE)
            {
                vkWaitForFences(device, 1, &fences[current_frame], VK_TRUE, UINT64_MAX);

                for (const auto &garbage : garbage_bins[current_frame])
                {
                //     if (garbage.buffer != VK_NULL_HANDLE)
                //         vkDestroyBuffer(device, garbage.buffer, nullptr);
                    if (garbage.memory != VK_NULL_HANDLE)
                        vkFreeMemory(device, garbage.memory, nullptr);
                }
                garbage_bins[current_frame].clear();

                frame_ready[current_frame] = true;
            }
            else
            {
                throw std::runtime_error("invalid handle");
            }
        }
    }

    void advanceFrame() const
    {
        frame_ready[current_frame] = false;
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};
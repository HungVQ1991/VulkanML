#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
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

constexpr bool IS_DEBUG_VALIDATION_ENABLED = false;
constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Resource_Garbage
{
    VkBuffer buffer = VK_NULL_HANDLE;
    Memory_Allocation allocation{};
};

struct Buffer_Transfer_Task
{
    VkBuffer source_buffer = VK_NULL_HANDLE;
    VkDeviceSize source_offset = 0;
    VkBuffer destination_buffer = VK_NULL_HANDLE;
    VkDeviceSize destination_offset = 0;
    VkDeviceSize size = 0;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT _message_severity,
    VkDebugUtilsMessageTypeFlagsEXT _message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *_callback_data,
    void *_user_data)
{
    if (_callback_data && _callback_data->pMessage)
    {
        Logger::logMessage(std::format("debugCallback: Vulkan Validation Layer: {}", _callback_data->pMessage),
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
    }
    return VK_FALSE;
}

class Vulkan_Context
{
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::uint32_t compute_queue_family_index = 0;

    std::unique_ptr<Vulkan_Sub_Allocator> allocator;

    mutable VkBuffer staging_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable Memory_Allocation staging_allocations[MAX_FRAMES_IN_FLIGHT]{};
    mutable void *staging_mapped_pointers[MAX_FRAMES_IN_FLIGHT]{nullptr, nullptr};
    mutable VkDeviceSize staging_capacities[MAX_FRAMES_IN_FLIGHT]{0, 0};
    mutable VkDeviceSize current_offsets[MAX_FRAMES_IN_FLIGHT]{0, 0};

    struct Staging_Garbage
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        Memory_Allocation allocation{};
    };
    mutable std::vector<Staging_Garbage> staging_garbages[MAX_FRAMES_IN_FLIGHT];

    mutable std::uint32_t current_frame = 0;
    mutable std::vector<Buffer_Transfer_Task> pending_transfer_tasks;

    mutable VkFence fences[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable std::function<void(VkFence)> flush_callback = nullptr;

    mutable std::mutex garbage_mutex;
    mutable std::mutex context_mutex;
    mutable std::vector<Resource_Garbage> garbage_bins[MAX_FRAMES_IN_FLIGHT];

    mutable bool is_frame_ready[MAX_FRAMES_IN_FLIGHT]{false, false};

    void initializeFences()
    {
        VkFenceCreateInfo fence_create_information{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT};

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateFence(device, &fence_create_information, nullptr, &fences[i]) != VK_SUCCESS)
            {
                Logger::logMessage(std::format("Vulkan_Context::initializeFences: Failed to create fence for frame {}", i),
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
                throw std::runtime_error("Failed to create fence");
            }
        }
    }

    void initializeInstance()
    {
        VkApplicationInfo application_information{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "Matrix_Compute",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No_Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_4};

        const char *validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
        const char *instance_extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

        VkDebugUtilsMessengerCreateInfoEXT debug_create_information{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = nullptr,
            .flags = 0,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData = nullptr};

        VkInstanceCreateInfo create_information{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = IS_DEBUG_VALIDATION_ENABLED ? &debug_create_information : nullptr,
            .flags = 0,
            .pApplicationInfo = &application_information,
            .enabledLayerCount = IS_DEBUG_VALIDATION_ENABLED ? 1u : 0u,
            .ppEnabledLayerNames = IS_DEBUG_VALIDATION_ENABLED ? validation_layers : nullptr,
            .enabledExtensionCount = IS_DEBUG_VALIDATION_ENABLED ? 1u : 0u,
            .ppEnabledExtensionNames = IS_DEBUG_VALIDATION_ENABLED ? instance_extensions : nullptr};

        if (vkCreateInstance(&create_information, nullptr, &instance) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::initializeInstance: Failed to create Vulkan instance",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            throw std::runtime_error("Failed to create Vulkan instance");
        }
    }

    bool hasRequiredDeviceLimits(VkPhysicalDevice _target_physical_device) const
    {
        VkPhysicalDeviceProperties device_properties;
        vkGetPhysicalDeviceProperties(_target_physical_device, &device_properties);
        const auto &limits = device_properties.limits;

        constexpr std::uint32_t REQUIRED_STORAGE_BUFFERS = 32;
        constexpr std::uint32_t REQUIRED_PUSH_CONSTANTS = 128;

        if (limits.maxPerStageDescriptorStorageBuffers < REQUIRED_STORAGE_BUFFERS)
        {
            Logger::logMessage(std::format("Vulkan_Context::hasRequiredDeviceLimits: Device lacks required maxPerStageDescriptorStorageBuffers ({} < {})",
                                           limits.maxPerStageDescriptorStorageBuffers, REQUIRED_STORAGE_BUFFERS),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            return false;
        }

        if (limits.maxDescriptorSetStorageBuffers < REQUIRED_STORAGE_BUFFERS)
        {
            Logger::logMessage(std::format("Vulkan_Context::hasRequiredDeviceLimits: Device lacks required maxDescriptorSetStorageBuffers ({} < {})",
                                           limits.maxDescriptorSetStorageBuffers, REQUIRED_STORAGE_BUFFERS),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            return false;
        }

        if (limits.maxPushConstantsSize < REQUIRED_PUSH_CONSTANTS)
        {
            Logger::logMessage(std::format("Vulkan_Context::hasRequiredDeviceLimits: Device lacks required maxPushConstantsSize ({} < {})",
                                           limits.maxPushConstantsSize, REQUIRED_PUSH_CONSTANTS),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            return false;
        }

        return true;
    }

    void selectPhysicalDevice()
    {
        std::uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

        if (device_count == 0)
        {
            Logger::logMessage("Vulkan_Context::selectPhysicalDevice: No physical devices with Vulkan support found",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

        constexpr std::array<std::size_t, 5> priority_order = {2, 1, 3, 4, 0};
        VkPhysicalDevice best_device = VK_NULL_HANDLE;
        std::uint32_t best_compute_family_index = 0;
        std::size_t best_rank = priority_order.size();

        for (const auto &device_candidate : devices)
        {
            if (!hasRequiredDeviceLimits(device_candidate))
            {
                continue;
            }

            VkPhysicalDeviceProperties device_properties;
            vkGetPhysicalDeviceProperties(device_candidate, &device_properties);

            std::uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &queue_family_count, nullptr);

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &queue_family_count, queue_families.data());

            for (std::uint32_t i = 0; i < queue_family_count; ++i)
            {
                if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                {
                    auto it = std::find(priority_order.begin(), priority_order.end(), static_cast<std::size_t>(device_properties.deviceType));
                    if (it != priority_order.end())
                    {
                        std::size_t current_rank = static_cast<std::size_t>(std::distance(priority_order.begin(), it));
                        if (current_rank < best_rank)
                        {
                            best_rank = current_rank;
                            best_device = device_candidate;
                            best_compute_family_index = i;
                        }
                    }
                    break;
                }
            }
        }

        if (best_device == VK_NULL_HANDLE)
        {
            Logger::logMessage("Vulkan_Context::selectPhysicalDevice: Failed to find a suitable GPU that satisfies all compute and resource limits",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            throw std::runtime_error("Failed to find a suitable GPU");
        }

        physical_device = best_device;
        compute_queue_family_index = best_compute_family_index;
    }

    void createLogicalDevice()
    {
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_information{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = compute_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority};

        std::vector<const char *> required_extensions = {};
        VkPhysicalDeviceFeatures enabled_features{};

        VkDeviceCreateInfo device_create_information{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_information,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<std::uint32_t>(required_extensions.size()),
            .ppEnabledExtensionNames = required_extensions.empty() ? nullptr : required_extensions.data(),
            .pEnabledFeatures = &enabled_features};

        if (vkCreateDevice(physical_device, &device_create_information, nullptr, &device) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::createLogicalDevice: Failed to create logical Vulkan device",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo pool_create_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = compute_queue_family_index};

        if (vkCreateCommandPool(device, &pool_create_information, nullptr, &command_pool) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::createCommandPool: Failed to create command pool",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            throw std::runtime_error("Failed to create command pool");
        }
    }

    void createPipelineCache()
    {
        VkPipelineCacheCreateInfo cache_create_information{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = 0,
            .pInitialData = nullptr};

        if (vkCreatePipelineCache(device, &cache_create_information, nullptr, &pipeline_cache) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::createPipelineCache: Failed to create pipeline cache",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            throw std::runtime_error("Failed to create pipeline cache");
        }
    }

public:
    Vulkan_Context()
    {
        Logger::logMessage("Vulkan_Context::Vulkan_Context: Initializing Vulkan Context",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        initializeInstance();
        selectPhysicalDevice();
        createLogicalDevice();
        createPipelineCache();
        createCommandPool();
        initializeFences();
        allocator = std::make_unique<Vulkan_Sub_Allocator>(device, *this, physical_device);
    }

    ~Vulkan_Context()
    {
        Logger::logMessage("Vulkan_Context::~Vulkan_Context: Destroying Vulkan Context",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);
        if (device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);
        }

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            cleanGarbage(i);

            if (fences[i] != VK_NULL_HANDLE)
            {
                vkDestroyFence(device, fences[i], nullptr);
            }

            if (staging_buffers[i] != VK_NULL_HANDLE)
            {
                vkUnmapMemory(device, staging_allocations[i].memory);
                vkDestroyBuffer(device, staging_buffers[i], nullptr);
                allocator->free(staging_allocations[i]);
            }
        }

        allocator.reset();

        if (command_pool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device, command_pool, nullptr);
        }

        if (pipeline_cache != VK_NULL_HANDLE)
        {
            vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        }

        if (device != VK_NULL_HANDLE)
        {
            vkDestroyDevice(device, nullptr);
        }

        if (instance != VK_NULL_HANDLE)
        {
            vkDestroyInstance(instance, nullptr);
        }
    }

    [[nodiscard]] Memory_Allocation allocateMemory(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties) const
    {
        return allocator->allocate(_memory_requirements, _memory_properties, physical_device);
    }

    void deferDestruction(std::uint32_t _used_frame, VkBuffer _buffer, const Memory_Allocation &_allocation) const
    {
        if (_buffer != VK_NULL_HANDLE || _allocation.memory != VK_NULL_HANDLE)
        {
            std::lock_guard<std::mutex> lock(garbage_mutex);
            garbage_bins[_used_frame].push_back(Resource_Garbage{
                .buffer = _buffer,
                .allocation = _allocation});
        }
        else
        {
            Logger::logMessage("Vulkan_Context::deferDestruction: Both buffer and memory handle are null",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
        }
    }

    [[nodiscard]] VkInstance getInstance() const noexcept { return instance; }
    [[nodiscard]] VkPhysicalDevice getPhysicalDevice() const noexcept { return physical_device; }
    [[nodiscard]] VkDevice getDevice() const noexcept { return device; }
    [[nodiscard]] VkQueue getComputeQueue() const noexcept { return compute_queue; }
    [[nodiscard]] VkCommandPool getCommandPool() const noexcept { return command_pool; }
    [[nodiscard]] std::uint32_t getComputeQueueFamilyIndex() const noexcept { return compute_queue_family_index; }
    [[nodiscard]] VkPipelineCache getPipelineCache() const noexcept { return pipeline_cache; }
    [[nodiscard]] Vulkan_Sub_Allocator &getAllocator() const noexcept { return *allocator; }
    [[nodiscard]] VkFence getFrameFence(std::uint32_t _frame_index) const noexcept { return fences[_frame_index]; }
    [[nodiscard]] std::uint32_t getCurrentFrame() const noexcept { return current_frame; }
    [[nodiscard]] bool isFrameReady(std::uint32_t _frame_index) const noexcept { return is_frame_ready[_frame_index]; }
    [[nodiscard]] VkBuffer getStagingBuffer(std::uint32_t _frame_index) const noexcept { return staging_buffers[_frame_index]; }
    [[nodiscard]] VkDeviceSize getStagingCapacity(std::uint32_t _frame_index) const noexcept { return staging_capacities[_frame_index]; }
    [[nodiscard]] VkDeviceSize getCurrentStagingOffset(std::uint32_t _frame_index) const noexcept { return current_offsets[_frame_index]; }

    [[nodiscard]] std::uint32_t findMemoryType(std::uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const
    {
        VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
        for (std::uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i)
        {
            if ((_type_filter & (1u << i)) && (physical_device_memory_properties.memoryTypes[i].propertyFlags & _memory_properties) == _memory_properties)
            {
                return i;
            }
        }
        Logger::logMessage("Vulkan_Context::findMemoryType: Failed to find suitable memory type",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        throw std::runtime_error("Failed to find suitable memory type");
    }

    void *allocateStagingSpace(std::uint32_t _frame_index, VkDeviceSize _size, VkBuffer &_out_buffer, VkDeviceSize &_out_offset) const
    {
        std::lock_guard<std::mutex> lock(context_mutex);

        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Vulkan_Context::allocateStagingSpace: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
            _frame_index = _frame_index % MAX_FRAMES_IN_FLIGHT;
        }

        if (current_offsets[_frame_index] + _size > staging_capacities[_frame_index] || staging_buffers[_frame_index] == VK_NULL_HANDLE)
        {
            constexpr VkDeviceSize INITIAL_STAGING_CAPACITY = 32 * 1024 * 1024;
            VkDeviceSize calculated_size = std::max(staging_capacities[_frame_index] * 2, current_offsets[_frame_index] + _size + 1024 * 1024);
            VkDeviceSize new_capacity = std::max(INITIAL_STAGING_CAPACITY, calculated_size);

            Logger::logMessage(std::format("Vulkan_Context::allocateStagingSpace: Reallocating staging buffer for frame {} to new capacity {} bytes", _frame_index, new_capacity),
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);

            VkBufferCreateInfo buffer_create_information{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = new_capacity,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr};

            VkBuffer new_buffer = VK_NULL_HANDLE;
            if (vkCreateBuffer(device, &buffer_create_information, nullptr, &new_buffer) != VK_SUCCESS)
            {
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to create new staging buffer",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to create staging buffer");
            }

            VkMemoryRequirements memory_requirements;
            vkGetBufferMemoryRequirements(device, new_buffer, &memory_requirements);

            Memory_Allocation new_allocation = allocateMemory(memory_requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            if (vkBindBufferMemory(device, new_buffer, new_allocation.memory, new_allocation.offset) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, new_buffer, nullptr);
                allocator->free(new_allocation);
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to bind staging buffer memory",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to bind staging memory");
            }

            void *new_mapped_pointer = nullptr;
            if (vkMapMemory(device, new_allocation.memory, new_allocation.offset, new_capacity, 0, &new_mapped_pointer) != VK_SUCCESS)
            {
                vkDestroyBuffer(device, new_buffer, nullptr);
                allocator->free(new_allocation);
                Logger::logMessage("Vulkan_Context::allocateStagingSpace: Failed to map staging memory",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
                throw std::runtime_error("Failed to map staging memory");
            }

            if (staging_buffers[_frame_index] != VK_NULL_HANDLE)
            {
                if (current_offsets[_frame_index] > 0)
                {
                    std::memcpy(new_mapped_pointer, staging_mapped_pointers[_frame_index], current_offsets[_frame_index]);
                }

                vkUnmapMemory(device, staging_allocations[_frame_index].memory);
                staging_garbages[_frame_index].push_back(Staging_Garbage{
                    .buffer = staging_buffers[_frame_index],
                    .allocation = staging_allocations[_frame_index]});

                for (auto &task : pending_transfer_tasks)
                {
                    if (task.source_buffer == staging_buffers[_frame_index])
                    {
                        task.source_buffer = new_buffer;
                    }
                }
            }

            staging_buffers[_frame_index] = new_buffer;
            staging_allocations[_frame_index] = new_allocation;
            staging_mapped_pointers[_frame_index] = new_mapped_pointer;
            staging_capacities[_frame_index] = new_capacity;
        }

        _out_buffer = staging_buffers[_frame_index];
        _out_offset = current_offsets[_frame_index];
        void *target_pointer = static_cast<char *>(staging_mapped_pointers[_frame_index]) + _out_offset;
        current_offsets[_frame_index] += _size;

        Logger::logMessage(std::format("Vulkan_Context::allocateStagingSpace: Allocated {} bytes in staging buffer for frame {}", _size, _frame_index),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);

        return target_pointer;
    }

    void resetFrameFence(std::uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Vulkan_Context::resetFrameFence: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SYNCHRONIZATION);
            return;
        }
        vkResetFences(device, 1, &fences[_frame_index]);
    }

    void registerFlushCallback(std::function<void(VkFence)> _callback) const
    {
        flush_callback = _callback;
    }

    void flush(VkFence _fence = VK_NULL_HANDLE) const
    {
        if (flush_callback)
        {
            Logger::logMessage("Vulkan_Context::flush: Executing flush callback",
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::SYNCHRONIZATION);
            flush_callback(_fence);
        }
        else
        {
            Logger::logMessage("Vulkan_Context::flush: Flush callback is not registered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::SYNCHRONIZATION);
        }
    }

    void resetStagingOffset(std::uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Vulkan_Context::resetStagingOffset: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION | Log_Feature::MEMORY_TRANSFER);
            return;
        }
        std::lock_guard<std::mutex> lock(context_mutex);
        current_offsets[_frame_index] = 0;
    }

    void addTransferTask(const Buffer_Transfer_Task &_task) const
    {
        std::lock_guard<std::mutex> lock(context_mutex);
        pending_transfer_tasks.push_back(_task);
    }

    [[nodiscard]] std::vector<Buffer_Transfer_Task> getTransferTasks() const
    {
        std::lock_guard<std::mutex> lock(context_mutex);
        return pending_transfer_tasks;
    }

    void clearTransferTasks() const
    {
        std::lock_guard<std::mutex> lock(context_mutex);
        pending_transfer_tasks.clear();
    }

    void executePendingTransfers() const
    {
        std::vector<Buffer_Transfer_Task> transfers_to_execute;
        {
            std::lock_guard<std::mutex> lock(context_mutex);
            if (pending_transfer_tasks.empty())
            {
                return;
            }
            transfers_to_execute = std::move(pending_transfer_tasks);
            pending_transfer_tasks.clear();
        }

        VkCommandBufferAllocateInfo allocate_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &allocate_information, &command_buffer) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to allocate command buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
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
            Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to begin command buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to begin command buffer");
        }

        for (const auto &task : transfers_to_execute)
        {
            VkBufferCopy copy_region{
                .srcOffset = task.source_offset,
                .dstOffset = task.destination_offset,
                .size = task.size};

            vkCmdCopyBuffer(command_buffer, task.source_buffer, task.destination_buffer, 1, &copy_region);

            VkBufferMemoryBarrier memory_barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = task.destination_buffer,
                .offset = task.destination_offset,
                .size = task.size};

            vkCmdPipelineBarrier(command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 1, &memory_barrier, 0, nullptr);
        }

        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to end command buffer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to end command buffer");
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

        if (vkQueueSubmit(compute_queue, 1, &submit_information, VK_NULL_HANDLE) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            Logger::logMessage("Vulkan_Context::executePendingTransfers: Failed to submit transfer queue",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::MEMORY_TRANSFER);
            throw std::runtime_error("Failed to submit transfer queue");
        }

        vkQueueWaitIdle(compute_queue);
        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    }

    void cleanGarbage(std::uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Vulkan_Context::cleanGarbage: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(context_mutex);
            for (const auto &staging_garbage_item : staging_garbages[_frame_index])
            {
                if (staging_garbage_item.buffer != VK_NULL_HANDLE)
                {
                    vkDestroyBuffer(device, staging_garbage_item.buffer, nullptr);
                }
                if (staging_garbage_item.allocation.memory != VK_NULL_HANDLE)
                {
                    allocator->free(staging_garbage_item.allocation);
                }
            }
            staging_garbages[_frame_index].clear();
        }

        std::vector<Resource_Garbage> local_bin;
        {
            std::lock_guard<std::mutex> lock(garbage_mutex);
            local_bin.swap(garbage_bins[_frame_index]);
        }

        if (!local_bin.empty())
        {
            Logger::logMessage(std::format("Vulkan_Context::cleanGarbage: Cleaning {} garbage items for frame {}", local_bin.size(), _frame_index),
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
        }

        for (const auto &garbage_item : local_bin)
        {
            if (garbage_item.buffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(device, garbage_item.buffer, nullptr);
            }
            if (garbage_item.allocation.memory != VK_NULL_HANDLE)
            {
                allocator->free(garbage_item.allocation);
            }
        }
    }

    void prepareFrame() const
    {
        if (!is_frame_ready[current_frame])
        {
            if (device != VK_NULL_HANDLE && fences[current_frame] != VK_NULL_HANDLE)
            {
                vkWaitForFences(device, 1, &fences[current_frame], VK_TRUE, UINT64_MAX);

                cleanGarbage(current_frame);
                is_frame_ready[current_frame] = true;
            }
            else
            {
                Logger::logMessage(std::format("Vulkan_Context::prepareFrame: Invalid device or fence handle for frame {}", current_frame),
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
                throw std::runtime_error("invalid handle");
            }
        }
    }

    void advanceFrame() const
    {
        is_frame_ready[current_frame] = false;
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Logger::logMessage(std::format("Vulkan_Context::advanceFrame: Advanced current frame to {}", current_frame),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SYNCHRONIZATION);
    }

    void copyBuffer(VkBuffer _source_buffer,
                    VkBuffer _destination_buffer,
                    VkDeviceSize _size,
                    VkDeviceSize _source_offset = 0,
                    VkDeviceSize _destination_offset = 0) const
    {
        if (_source_buffer == VK_NULL_HANDLE || _destination_buffer == VK_NULL_HANDLE || _size == 0)
        {
            Logger::logMessage("Vulkan_Context::copyBuffer: Invalid parameters provided for buffer copy",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::MEMORY_TRANSFER);
            return;
        }

        Buffer_Transfer_Task transfer_task{
            .source_buffer = _source_buffer,
            .source_offset = _source_offset,
            .destination_buffer = _destination_buffer,
            .destination_offset = _destination_offset,
            .size = _size};

        addTransferTask(transfer_task);
        Logger::logMessage(std::format("Vulkan_Context::copyBuffer: Enqueued transfer task of size {} bytes", _size),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_TRANSFER);
    }
};
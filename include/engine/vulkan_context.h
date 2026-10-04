#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#include "vulkan_sub_allocator.h"

#ifndef IS_VULKAN_DEBUG_VALIDATION
#define IS_VULKAN_DEBUG_VALIDATION 0
#endif

constexpr bool IS_DEBUG_VALIDATION_ENABLED = false;
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

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

class Vulkan_Context
{
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index = 0;

    bool is_cooperative_matrix_supported = false;
    bool is_cooperative_matrix_enabled = false;
    bool is_float16_supported = false;
    bool is_float16_enabled = false;
    VkCooperativeMatrixPropertiesKHR cooperative_matrix_properties{};

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

    mutable uint32_t current_frame = 0;
    mutable std::vector<Buffer_Transfer_Task> pending_transfer_tasks;

    mutable VkFence fences[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable std::function<void(VkFence)> flush_callback = nullptr;

    bool is_timeline_semaphore_supported = false;
    VkSemaphore timeline_semaphore = VK_NULL_HANDLE;
    mutable std::atomic<uint64_t> current_timeline_value{0};

    mutable VkBuffer readback_staging_buffer = VK_NULL_HANDLE;
    mutable Memory_Allocation readback_staging_allocation{};
    mutable void *readback_mapped_pointer = nullptr;
    mutable VkDeviceSize readback_staging_capacity = 0;

    mutable std::mutex garbage_mutex;
    mutable std::recursive_mutex context_mutex;
    mutable std::vector<Resource_Garbage> garbage_bins[MAX_FRAMES_IN_FLIGHT];

    mutable bool is_frame_ready[MAX_FRAMES_IN_FLIGHT]{true, true};

    void initializeFences();
    void initializeTimelineSemaphore();
    void initializeInstance();
    bool hasRequiredDeviceLimits(VkPhysicalDevice _target_physical_device) const;
    std::pair<VkPhysicalDevice, uint32_t> findComputeQueueFamily() const;
    void selectPhysicalDevice();
    void queryCooperativeMatrixSupport();
    void createLogicalDevice();
    void createCommandPool();
    void createPipelineCache();

public:
    Vulkan_Context();
    ~Vulkan_Context();

    Memory_Allocation allocateMemory(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties) const;
    void deferDestruction(uint32_t _used_frame, VkBuffer _buffer, const Memory_Allocation &_allocation) const;
    uint32_t findMemoryType(uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const;
    void *allocateStagingSpace(uint32_t _frame_index, VkDeviceSize _size, VkBuffer &_out_buffer, VkDeviceSize &_out_offset) const;
    void resetFrameFence(uint32_t _frame_index) const;
    void flush(VkFence _fence = VK_NULL_HANDLE) const;
    void resetStagingOffset(uint32_t _frame_index) const;
    void addTransferTask(const Buffer_Transfer_Task &_task) const;
    void removeTransferTasksForBuffer(VkBuffer _buffer) const;
    void clearTransferTasks() const;
    void executePendingTransfers() const;
    void cleanGarbage(uint32_t _frame_index) const;
    void prepareFrame(uint32_t _frame_index = UINT32_MAX);
    void advanceFrame() const;
    void copyBuffer(VkBuffer _source_buffer,
                    VkBuffer _destination_buffer,
                    VkDeviceSize _size,
                    VkDeviceSize _source_offset = 0,
                    VkDeviceSize _destination_offset = 0) const;
    void fillBuffer(VkBuffer _destination_buffer,
                    VkDeviceSize _size,
                    VkDeviceSize _destination_offset = 0,
                    uint32_t _pattern = 0) const;

    Vulkan_Sub_Allocator &getAllocator() const noexcept { return *allocator; }
    const VkCooperativeMatrixPropertiesKHR &getCooperativeMatrixProperties() const noexcept { return cooperative_matrix_properties; }
    const std::vector<Buffer_Transfer_Task> &getTransferTasks() const
    {
        std::lock_guard lock(context_mutex);
        return pending_transfer_tasks;
    }
    VkDeviceSize getCurrentStagingOffset(uint32_t _frame_index) const noexcept { return current_offsets[_frame_index]; }
    VkDeviceSize getStagingCapacity(uint32_t _frame_index) const noexcept { return staging_capacities[_frame_index]; }
    VkBuffer getStagingBuffer(uint32_t _frame_index) const noexcept { return staging_buffers[_frame_index]; }
    VkFence getFrameFence(uint32_t _frame_index) const noexcept { return fences[_frame_index]; }
    VkPhysicalDevice getPhysicalDevice() const noexcept { return physical_device; }
    VkPipelineCache getPipelineCache() const noexcept { return pipeline_cache; }
    VkCommandPool getCommandPool() const noexcept { return command_pool; }
    VkQueue getComputeQueue() const noexcept { return compute_queue; }
    VkInstance getInstance() const noexcept { return instance; }
    VkDevice getDevice() const noexcept { return device; }
    uint32_t getComputeQueueFamilyIndex() const noexcept { return compute_queue_family_index; }
    uint32_t getCurrentFrame() const noexcept { return current_frame; }
    bool isCooperativeMatrixEnabled() const noexcept { return is_cooperative_matrix_supported && is_cooperative_matrix_enabled; }
    bool isCooperativeMatrixSupported() const noexcept { return is_cooperative_matrix_supported; }
    bool isFloat16Enabled() const noexcept { return is_float16_supported && is_float16_enabled; }
    bool isFloat16Supported() const noexcept { return is_float16_supported; }
    bool isFrameReady(uint32_t _frame_index) const noexcept { return is_frame_ready[_frame_index]; }

    void setCooperativeMatrixProperties(const VkCooperativeMatrixPropertiesKHR &_properties) noexcept { cooperative_matrix_properties = _properties; }
    void registerFlushCallback(std::function<void(VkFence)> _callback) const { flush_callback = _callback; }
    void setComputeQueueFamilyIndex(uint32_t _index) noexcept { compute_queue_family_index = _index; }
    void setCurrentFrame(uint32_t _frame) const noexcept { current_frame = _frame; }
    void setCooperativeMatrixEnabled(bool _enable) noexcept
    {
        if (is_cooperative_matrix_supported)
        {
            is_cooperative_matrix_enabled = _enable;
        }
    }
    void setFloat16Enabled(bool _enable) noexcept
    {
        if (is_float16_supported)
        {
            is_float16_enabled = _enable;
        }
    }

    uint64_t getNextTimelineValue() const noexcept { return ++current_timeline_value; }
    uint64_t getCurrentTimelineValue() const noexcept { return current_timeline_value.load(); }
    VkSemaphore getTimelineSemaphore() const noexcept { return timeline_semaphore; }
    bool isTimelineSemaphoreSupported() const noexcept { return is_timeline_semaphore_supported && (timeline_semaphore != VK_NULL_HANDLE); }

    void waitTimelineSemaphore(uint64_t target_value, uint64_t timeout_ns = UINT64_MAX) const;
    void *ensureReadbackStagingBuffer(VkDeviceSize required_size, VkBuffer &out_buffer) const;
};
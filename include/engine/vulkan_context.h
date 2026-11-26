#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#include "vulkan_sub_allocator.h"

#ifndef IS_VULKAN_DEBUG_VALIDATION
#define IS_VULKAN_DEBUG_VALIDATION 0
#endif

constexpr bool IS_DEBUG_VALIDATION_ENABLED = (IS_VULKAN_DEBUG_VALIDATION != 0);
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct Resource_Garbage
{
    VkBuffer buffer = VK_NULL_HANDLE;
    Memory_Allocation allocation{};
    uint32_t pending_frame_mask = 0;
};

struct Buffer_Transfer_Task
{
    VkBuffer source_buffer = VK_NULL_HANDLE;
    VkDeviceSize source_offset = 0;
    VkBuffer destination_buffer = VK_NULL_HANDLE;
    VkDeviceSize destination_offset = 0;
    VkDeviceSize size = 0;
    bool is_fill = false;
    uint32_t fill_pattern = 0;
};

class Vulkan_Context
{
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
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
    bool is_subgroup_size_control_supported = false;
    bool is_subgroup_size_control_enabled = false;
    bool is_compute_full_subgroups_supported = false;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    uint32_t default_subgroup_size = 0;
    VkShaderStageFlags required_subgroup_size_stages = 0;
    std::array<uint32_t, 3> max_compute_work_group_count{65535u, 65535u, 65535u};
    float timestamp_period_ns = 0.0f;
    uint32_t timestamp_valid_bits = 0;
    bool is_push_descriptor_supported = false;
    bool is_push_descriptor_enabled = false;
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ptr = nullptr;
    VkCooperativeMatrixPropertiesKHR cooperative_matrix_properties{};

    std::unique_ptr<Vulkan_Sub_Allocator> allocator;

    mutable VkCommandBuffer transfer_command_buffer = VK_NULL_HANDLE;
    mutable VkCommandBuffer immediate_copy_command_buffer = VK_NULL_HANDLE;
    mutable VkFence immediate_copy_fence = VK_NULL_HANDLE;

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

    mutable std::atomic<uint32_t> current_frame{0};
    mutable std::vector<Buffer_Transfer_Task> pending_transfer_tasks;

    mutable VkFence fences[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    mutable std::function<void(VkFence)> flush_callback = nullptr;

    bool is_timeline_semaphore_feature_enabled = false;
    bool is_timeline_semaphore_supported = false;
    VkSemaphore timeline_semaphore = VK_NULL_HANDLE;
    mutable std::atomic<uint64_t> current_timeline_value{0};

    mutable VkBuffer readback_staging_buffer = VK_NULL_HANDLE;
    mutable Memory_Allocation readback_staging_allocation{};
    mutable void *readback_mapped_pointer = nullptr;
    mutable VkDeviceSize readback_staging_capacity = 0;

    mutable std::mutex garbage_mutex;
    mutable std::recursive_mutex context_mutex;
    mutable std::vector<Resource_Garbage> deferred_garbage;

    mutable std::atomic<bool> is_frame_ready[MAX_FRAMES_IN_FLIGHT]{{true}, {true}};

    void initializeFences();
    void initializeTimelineSemaphore();
    void initializeInstance();
    bool hasRequiredDeviceLimits(VkPhysicalDevice _target_physical_device) const;
    void selectPhysicalDevice();
    void queryCooperativeMatrixSupport();
    void createLogicalDevice();
    void createCommandPool();
    void createPipelineCache();
    void cleanUp() noexcept;

public:
    Vulkan_Context();
    ~Vulkan_Context();

    Memory_Allocation allocateMemory(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties) const;
    void deferDestruction(uint32_t _used_frame, VkBuffer _buffer, const Memory_Allocation &_allocation) const;
    void deferDestructionMask(uint32_t _used_frame_mask, VkBuffer _buffer, const Memory_Allocation &_allocation) const;
    uint32_t findMemoryType(uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const;
    void *allocateStagingSpace(uint32_t _frame_index, VkDeviceSize _size, VkBuffer &_out_buffer, VkDeviceSize &_out_offset) const;
    void uploadToDevice(uint32_t _frame_index, const void *_source_data, VkDeviceSize _size, VkBuffer _destination_buffer, VkDeviceSize _destination_offset = 0) const;
    void downloadFromDevice(VkBuffer _source_buffer, void *_destination_data, VkDeviceSize _copy_size, VkDeviceSize _buffer_size) const;
    void resetFrameFence(uint32_t _frame_index) const;
    void flush(VkFence _fence = VK_NULL_HANDLE) const;
    void resetStagingOffset(uint32_t _frame_index) const;
    void addTransferTask(const Buffer_Transfer_Task &_task) const;
    void removeTransferTasksForBuffer(VkBuffer _buffer) const;
    void clearTransferTasks() const;
    bool hasPendingTransfers() const;
    std::vector<Buffer_Transfer_Task> takeTransferTasks() const;
    void executePendingTransfers() const;
    void cleanGarbage(uint32_t _frame_index) const;
    void prepareFrame(uint32_t _frame_index = UINT32_MAX) const;
    void advanceFrame() const;
    void copyBufferImmediate(VkBuffer _source_buffer,
                             VkBuffer _destination_buffer,
                             VkDeviceSize _size,
                             VkDeviceSize _source_offset = 0,
                             VkDeviceSize _destination_offset = 0) const;
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
    std::vector<Buffer_Transfer_Task> getTransferTasks() const
    {
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        return pending_transfer_tasks;
    }
    VkDeviceSize getCurrentStagingOffset(uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            return 0;
        }
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        return current_offsets[_frame_index];
    }
    VkDeviceSize getStagingCapacity(uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            return 0;
        }
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        return staging_capacities[_frame_index];
    }
    VkBuffer getStagingBuffer(uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            return VK_NULL_HANDLE;
        }
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        return staging_buffers[_frame_index];
    }
    VkFence getFrameFence(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? fences[_frame_index] : VK_NULL_HANDLE;
    }
    VkPhysicalDevice getPhysicalDevice() const noexcept { return physical_device; }
    VkPipelineCache getPipelineCache() const noexcept { return pipeline_cache; }
    VkCommandPool getCommandPool() const noexcept { return command_pool; }
    VkQueue getComputeQueue() const noexcept { return compute_queue; }
    VkInstance getInstance() const noexcept { return instance; }
    VkDevice getDevice() const noexcept { return device; }
    uint32_t getComputeQueueFamilyIndex() const noexcept { return compute_queue_family_index; }
    uint32_t getCurrentFrame() const noexcept { return current_frame.load(std::memory_order_acquire); }
    bool isCooperativeMatrixEnabled() const noexcept { return is_cooperative_matrix_supported && is_cooperative_matrix_enabled; }
    bool isCooperativeMatrixSupported() const noexcept { return is_cooperative_matrix_supported; }
    bool isFloat16Enabled() const noexcept { return is_float16_supported && is_float16_enabled; }
    bool isFloat16Supported() const noexcept { return is_float16_supported; }
    bool isSubgroupSizeControlSupported() const noexcept { return is_subgroup_size_control_supported; }
    bool isSubgroupSizeControlEnabled() const noexcept { return is_subgroup_size_control_supported && is_subgroup_size_control_enabled; }
    bool isComputeFullSubgroupsSupported() const noexcept { return is_compute_full_subgroups_supported; }
    bool canRequireSubgroupSize(uint32_t _size) const noexcept
    {
        return isSubgroupSizeControlEnabled() &&
               ((required_subgroup_size_stages & VK_SHADER_STAGE_COMPUTE_BIT) != 0) &&
               _size >= min_subgroup_size &&
               _size <= max_subgroup_size;
    }
    uint32_t getMinSubgroupSize() const noexcept { return min_subgroup_size; }
    uint32_t getMaxSubgroupSize() const noexcept { return max_subgroup_size; }
    uint32_t getDefaultSubgroupSize() const noexcept { return default_subgroup_size; }
    bool canGuaranteeSubgroupSize32() const noexcept { return canRequireSubgroupSize(32) || default_subgroup_size == 32; }
    const std::array<uint32_t, 3> &getMaxComputeWorkGroupCount() const noexcept { return max_compute_work_group_count; }
    std::pair<uint32_t, uint32_t> compute2DWorkgroupGrid(uint64_t total_workgroups) const
    {
        const uint32_t max_x = std::max<uint32_t>(1u, max_compute_work_group_count[0]);
        const uint32_t max_y = std::max<uint32_t>(1u, max_compute_work_group_count[1]);
        if (total_workgroups == 0)
        {
            return {0u, 1u};
        }
        if (total_workgroups <= max_x)
        {
            return {static_cast<uint32_t>(total_workgroups), 1u};
        }
        uint64_t wg_y = (total_workgroups + max_x - 1u) / max_x;
        if (wg_y > max_y)
        {
            throw std::overflow_error("Vulkan_Context::compute2DWorkgroupGrid: Dispatch exceeds maxComputeWorkGroupCount");
        }
        uint64_t wg_x = (total_workgroups + wg_y - 1u) / wg_y;
        return {static_cast<uint32_t>(wg_x), static_cast<uint32_t>(wg_y)};
    }
    bool isPushDescriptorSupported() const noexcept { return is_push_descriptor_supported; }
    bool isPushDescriptorEnabled() const noexcept { return is_push_descriptor_supported && is_push_descriptor_enabled; }
    bool isTimestampQuerySupported() const noexcept { return timestamp_valid_bits > 0u && timestamp_period_ns > 0.0f; }
    float getTimestampPeriodNs() const noexcept { return timestamp_period_ns; }
    uint32_t getTimestampValidBits() const noexcept { return timestamp_valid_bits; }
    bool isFrameReady(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) && is_frame_ready[_frame_index].load(std::memory_order_acquire);
    }

    void registerFlushCallback(std::function<void(VkFence)> _callback) const
    {
        std::lock_guard<std::recursive_mutex> lock(context_mutex);
        flush_callback = std::move(_callback);
    }
    void setCurrentFrame(uint32_t _frame) const noexcept { current_frame.store(_frame % MAX_FRAMES_IN_FLIGHT, std::memory_order_release); }
    void setPushDescriptorEnabled(bool _enable) noexcept
    {
        if (is_push_descriptor_supported)
        {
            is_push_descriptor_enabled = _enable;
        }
    }
    void cmdPushDescriptorSet(VkCommandBuffer _command_buffer,
                              VkPipelineBindPoint _pipeline_bind_point,
                              VkPipelineLayout _layout,
                              uint32_t _set,
                              uint32_t _descriptor_write_count,
                              const VkWriteDescriptorSet *_descriptor_writes) const
    {
        if (vkCmdPushDescriptorSetKHR_ptr != nullptr)
        {
            vkCmdPushDescriptorSetKHR_ptr(_command_buffer, _pipeline_bind_point, _layout, _set, _descriptor_write_count, _descriptor_writes);
        }
    }
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

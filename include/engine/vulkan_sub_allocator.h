#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

class Vulkan_Context;

struct Memory_Allocation
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    size_t chunk_index = 0;
    void* mapped_pointer = nullptr;
};

struct Free_Block
{
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

struct Memory_Chunk
{
    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    VkDeviceSize chunk_size = 0;
    uint32_t memory_type_index = 0;
    VkMemoryPropertyFlags memory_properties = 0;
    std::vector<Free_Block> free_blocks;
    bool is_dedicated_arena = false;
    void* mapped_pointer = nullptr;
};

struct Tensor_Lifetime
{
    uint32_t tensor_id = 0;
    VkDeviceSize size = 0;
    VkDeviceSize alignment = 4;
    uint32_t start_node_index = 0;
    uint32_t end_node_index = 0;
    VkDeviceSize allocated_offset = std::numeric_limits<VkDeviceSize>::max();
};

struct Virtual_Block
{
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    uint32_t free_after_node_index = 0;
    std::vector<std::pair<uint32_t, uint32_t>> occupied_intervals;
};

class Memory_Planner
{
private:
    VkDeviceSize total_allocated_size = 0;
    std::vector<Tensor_Lifetime> tensor_lifetimes;

public:
    void registerTensor(uint32_t _tensor_id, VkDeviceSize _size, VkDeviceSize _alignment, uint32_t _birth_node_index);
    void updateLastUsage(uint32_t _tensor_id, uint32_t _current_node_index);
    void planMemoryLayout();
    void reset() noexcept;

    const std::vector<Tensor_Lifetime> &getTensorLifetimes() const noexcept { return tensor_lifetimes; }
    VkDeviceSize getOffset(uint32_t _tensor_id) const;
    VkDeviceSize getTotalMemoryRequired() const noexcept { return total_allocated_size; }
    bool isTensorPlanned(uint32_t _tensor_id) const noexcept
    {
        for (const auto &tensor : tensor_lifetimes)
        {
            if (tensor.tensor_id == _tensor_id)
            {
                return tensor.allocated_offset != std::numeric_limits<VkDeviceSize>::max();
            }
        }
        return false;
    }
    bool hasTensor(uint32_t _tensor_id) const noexcept
    {
        return std::any_of(tensor_lifetimes.begin(), tensor_lifetimes.end(), [_tensor_id](const Tensor_Lifetime &lifetime)
                           { return lifetime.tensor_id == _tensor_id; });
    }

    void setTotalAllocatedSize(VkDeviceSize _size) noexcept { total_allocated_size = _size; }
};

struct Allocator_Stats
{
    uint64_t total_allocated_bytes = 0;
    uint64_t total_used_bytes = 0;
    uint64_t total_free_bytes = 0;
    uint64_t dedicated_arena_bytes = 0;
    size_t active_chunk_count = 0;
    size_t empty_chunk_count = 0;
    size_t reclaimed_chunk_count = 0;
    uint64_t reclaimed_bytes = 0;
};

class Vulkan_Sub_Allocator
{
private:
    VkDevice device = VK_NULL_HANDLE;
    const Vulkan_Context &context;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties{};
    std::vector<Memory_Chunk> memory_chunks;
    VkDeviceSize default_chunk_size = 64 * 1024 * 1024;
    size_t arena_chunk_index = std::numeric_limits<size_t>::max();
    size_t total_reclaimed_chunks = 0;
    uint64_t total_reclaimed_bytes = 0;
    mutable std::recursive_mutex allocator_mutex;
    Memory_Planner memory_planner;

    uint32_t findMemoryType(uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const;
    size_t createChunk(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, bool _is_dedicated_arena = false);
    void insertAndCoalesce(std::vector<Free_Block> &_free_blocks, VkDeviceSize _offset, VkDeviceSize _size);

public:
    explicit Vulkan_Sub_Allocator(VkDevice _device, const Vulkan_Context &_context, VkPhysicalDevice _physical_device = VK_NULL_HANDLE);
    ~Vulkan_Sub_Allocator();

    void free(const Memory_Allocation &_allocation);
    Memory_Allocation allocate(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, VkPhysicalDevice _target_physical_device = VK_NULL_HANDLE);
    size_t allocateDedicatedArena(VkDeviceSize _total_size, uint32_t _memory_type_bits, VkMemoryPropertyFlags _memory_properties);
    Memory_Allocation getArenaAllocation(size_t _chunk_index, VkDeviceSize _offset, VkDeviceSize _size) const;
    void releaseDedicatedArena(size_t _chunk_index);
    uint64_t reclaimEmptyChunks(bool _keep_one_default_chunk_per_type = false);
    Allocator_Stats getStats() const;

    const Vulkan_Context &getContext() const noexcept { return context; }
    const Memory_Planner &getMemoryPlanner() const noexcept { return memory_planner; }
    Memory_Planner &getMemoryPlanner() noexcept { return memory_planner; }
    const Memory_Planner &getPlanner() const noexcept { return memory_planner; }
    Memory_Planner &getPlanner() noexcept { return memory_planner; }
    std::vector<Memory_Chunk> getMemoryChunks() const
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        return memory_chunks;
    }
    size_t getMemoryChunkCount() const
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        return memory_chunks.size();
    }
    size_t getArenaChunkIndex() const
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        return arena_chunk_index;
    }
    VkDeviceSize getDefaultChunkSize() const
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        return default_chunk_size;
    }
    VkPhysicalDevice getPhysicalDevice() const noexcept { return physical_device; }
    VkDevice getDevice() const noexcept { return device; }
    bool hasArenaChunk() const
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        return arena_chunk_index != std::numeric_limits<size_t>::max();
    }

    void setDefaultChunkSize(VkDeviceSize _size)
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        default_chunk_size = _size;
    }
    void setArenaChunkIndex(size_t _index)
    {
        std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
        arena_chunk_index = _index;
    }
};

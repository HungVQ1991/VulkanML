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
    mutable std::mutex allocator_mutex;
    Memory_Planner memory_planner;

    uint32_t findMemoryType(uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const;
    size_t createChunk(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, bool _is_dedicated_arena = false);
    void insertAndCoalesce(std::vector<Free_Block> &_free_blocks, VkDeviceSize _offset, VkDeviceSize _size);

public:
    explicit Vulkan_Sub_Allocator(VkDevice _device, const Vulkan_Context &_context, VkPhysicalDevice _physical_device = VK_NULL_HANDLE);
    ~Vulkan_Sub_Allocator();

    void free(const Memory_Allocation &_allocation);
    Memory_Allocation allocate(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, VkPhysicalDevice _target_physical_device = VK_NULL_HANDLE);

    const Vulkan_Context &getContext() const noexcept { return context; }
    const Memory_Planner &getMemoryPlanner() const noexcept { return memory_planner; }
    Memory_Planner &getMemoryPlanner() noexcept { return memory_planner; }
    const Memory_Planner &getPlanner() const noexcept { return memory_planner; }
    Memory_Planner &getPlanner() noexcept { return memory_planner; }
    const std::vector<Memory_Chunk> &getMemoryChunks() const noexcept { return memory_chunks; }
    size_t getMemoryChunkCount() const noexcept { return memory_chunks.size(); }
    size_t getArenaChunkIndex() const noexcept { return arena_chunk_index; }
    VkDeviceSize getDefaultChunkSize() const noexcept { return default_chunk_size; }
    VkPhysicalDevice getPhysicalDevice() const noexcept { return physical_device; }
    VkDevice getDevice() const noexcept { return device; }
    bool hasArenaChunk() const noexcept { return arena_chunk_index != std::numeric_limits<size_t>::max(); }

    void setDefaultChunkSize(VkDeviceSize _size) noexcept { default_chunk_size = _size; }
    void setArenaChunkIndex(size_t _index) noexcept { arena_chunk_index = _index; }
};
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"

class Vulkan_Context;

struct Memory_Allocation
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    std::size_t chunk_index = 0;
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
    std::uint32_t memory_type_index = 0;
    VkMemoryPropertyFlags memory_properties = 0;
    std::vector<Free_Block> free_blocks;
    bool is_dedicated_arena = false;
};

struct Tensor_Lifetime
{
    std::uint32_t tensor_id = 0;
    VkDeviceSize size = 0;
    VkDeviceSize alignment = 4;
    std::uint32_t start_node_index = 0;
    std::uint32_t end_node_index = 0;
    VkDeviceSize allocated_offset = std::numeric_limits<VkDeviceSize>::max();
};

struct Virtual_Block
{
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    std::uint32_t free_after_node_index = 0;
};

class Memory_Planner
{
private:
    VkDeviceSize total_allocated_size = 0;
    std::vector<Tensor_Lifetime> tensor_lifetimes;

public:
    void registerTensor(std::uint32_t _tensor_id, VkDeviceSize _size, VkDeviceSize _alignment, std::uint32_t _birth_node_index)
    {
        tensor_lifetimes.push_back(Tensor_Lifetime{
            .tensor_id = _tensor_id,
            .size = _size,
            .alignment = _alignment,
            .start_node_index = _birth_node_index,
            .end_node_index = _birth_node_index,
            .allocated_offset = std::numeric_limits<VkDeviceSize>::max()});
    }

    void updateLastUsage(std::uint32_t _tensor_id, std::uint32_t _current_node_index)
    {
        for (auto &lifetime : tensor_lifetimes)
        {
            if (lifetime.tensor_id == _tensor_id)
            {
                lifetime.end_node_index = std::max(lifetime.end_node_index, _current_node_index);
                return;
            }
        }
    }

    void planMemoryLayout()
    {
        std::sort(tensor_lifetimes.begin(), tensor_lifetimes.end(), [](const Tensor_Lifetime &lhs, const Tensor_Lifetime &rhs)
                  {
            if (lhs.size != rhs.size)
            {
                return lhs.size > rhs.size;
            }
            return lhs.start_node_index < rhs.start_node_index; });

        std::vector<Virtual_Block> virtual_blocks;
        total_allocated_size = 0;

        for (auto &tensor : tensor_lifetimes)
        {
            std::size_t best_block_index = std::numeric_limits<std::size_t>::max();
            VkDeviceSize minimum_leftover = std::numeric_limits<VkDeviceSize>::max();
            VkDeviceSize best_aligned_offset = 0;

            for (std::size_t block_index = 0; block_index < virtual_blocks.size(); ++block_index)
            {
                const auto &block = virtual_blocks[block_index];
                if (block.free_after_node_index < tensor.start_node_index)
                {
                    VkDeviceSize alignment = tensor.alignment;
                    VkDeviceSize aligned_offset = (alignment <= 1)
                                                      ? block.offset
                                                      : ((block.offset + alignment - 1) / alignment) * alignment;
                    VkDeviceSize padding = aligned_offset - block.offset;
                    VkDeviceSize total_required = tensor.size + padding;

                    if (block.size >= total_required)
                    {
                        VkDeviceSize leftover = block.size - total_required;
                        if (leftover < minimum_leftover)
                        {
                            minimum_leftover = leftover;
                            best_block_index = block_index;
                            best_aligned_offset = aligned_offset;
                        }
                    }
                }
            }

            if (best_block_index != std::numeric_limits<std::size_t>::max())
            {
                tensor.allocated_offset = best_aligned_offset;
                virtual_blocks[best_block_index].free_after_node_index = tensor.end_node_index;
            }
            else
            {
                VkDeviceSize alignment = tensor.alignment;
                VkDeviceSize aligned_offset = (alignment <= 1)
                                                  ? total_allocated_size
                                                  : ((total_allocated_size + alignment - 1) / alignment) * alignment;

                tensor.allocated_offset = aligned_offset;
                total_allocated_size = aligned_offset + tensor.size;

                virtual_blocks.push_back(Virtual_Block{
                    .offset = aligned_offset,
                    .size = tensor.size,
                    .free_after_node_index = tensor.end_node_index});
            }
        }
    }

    [[nodiscard]] VkDeviceSize getOffset(std::uint32_t _tensor_id) const
    {
        for (const auto &tensor : tensor_lifetimes)
        {
            if (tensor.tensor_id == _tensor_id)
            {
                if (tensor.allocated_offset == std::numeric_limits<VkDeviceSize>::max())
                {
                    Logger::logMessage(std::format("Memory_Planner::getOffset: Tensor ID {} has not been planned", _tensor_id),
                                       Log_Level::LOG_ERROR,
                                       true,
                                       0,
                                       Log_Feature::MEMORY_ALLOCATION);
                    throw std::runtime_error(std::format("Memory_Planner::getOffset: Tensor ID {} has not been planned", _tensor_id));
                }
                return tensor.allocated_offset;
            }
        }
        Logger::logMessage(std::format("Memory_Planner::getOffset: Tensor ID {} not found in registry", _tensor_id),
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        throw std::runtime_error(std::format("Memory_Planner::getOffset: Tensor ID {} not found in registry", _tensor_id));
    }

    [[nodiscard]] VkDeviceSize getTotalMemoryRequired() const noexcept
    {
        return total_allocated_size;
    }

    [[nodiscard]] const std::vector<Tensor_Lifetime> &getTensorLifetimes() const noexcept
    {
        return tensor_lifetimes;
    }

    [[nodiscard]] bool hasTensor(std::uint32_t _tensor_id) const noexcept
    {
        return std::any_of(tensor_lifetimes.begin(), tensor_lifetimes.end(), [_tensor_id](const Tensor_Lifetime &lifetime)
                           { return lifetime.tensor_id == _tensor_id; });
    }

    [[nodiscard]] bool isTensorPlanned(std::uint32_t _tensor_id) const noexcept
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

    void reset() noexcept
    {
        tensor_lifetimes.clear();
        total_allocated_size = 0;
    }
};

class Vulkan_Sub_Allocator
{
private:
    VkDevice device = VK_NULL_HANDLE;
    const Vulkan_Context &context;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties{};
    std::vector<Memory_Chunk> memory_chunks;
    VkDeviceSize default_chunk_size = 256 * 1024 * 1024;
    std::size_t arena_chunk_index = std::numeric_limits<std::size_t>::max();
    mutable std::mutex allocator_mutex;
    Memory_Planner memory_planner;

    std::uint32_t findMemoryType(std::uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const
    {
        for (std::uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i)
        {
            if ((_type_filter & (1u << i)) && (physical_device_memory_properties.memoryTypes[i].propertyFlags & _memory_properties) == _memory_properties)
            {
                return i;
            }
        }
        Logger::logMessage("Vulkan_Sub_Allocator::findMemoryType: Failed to find suitable memory type",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        throw std::runtime_error("Vulkan_Sub_Allocator::findMemoryType: Failed to find suitable memory type");
    }

    std::size_t createChunk(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, bool _is_dedicated_arena = false)
    {
        std::uint32_t memory_type_index = findMemoryType(_memory_requirements.memoryTypeBits, _memory_properties);
        VkDeviceSize allocation_size = std::max(default_chunk_size, _memory_requirements.size);

        Logger::logMessage(std::format("Vulkan_Sub_Allocator::createChunk: Allocating new memory chunk of size {} bytes (memory_type_index={}, memory_properties={}, is_dedicated_arena={})",
                                       allocation_size, memory_type_index, static_cast<std::uint32_t>(_memory_properties), _is_dedicated_arena),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        VkMemoryAllocateInfo allocate_information{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = allocation_size,
            .memoryTypeIndex = memory_type_index};

        VkDeviceMemory device_memory = VK_NULL_HANDLE;
        if (vkAllocateMemory(device, &allocate_information, nullptr, &device_memory) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Sub_Allocator::createChunk: Failed to allocate device memory chunk",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Vulkan_Sub_Allocator::createChunk: Failed to allocate device memory chunk");
        }

        Memory_Chunk new_chunk{
            .device_memory = device_memory,
            .chunk_size = allocation_size,
            .memory_type_index = memory_type_index,
            .memory_properties = _memory_properties,
            .free_blocks = _is_dedicated_arena ? std::vector<Free_Block>{} : std::vector<Free_Block>{Free_Block{.offset = 0, .size = allocation_size}},
            .is_dedicated_arena = _is_dedicated_arena};

        memory_chunks.push_back(std::move(new_chunk));
        return memory_chunks.size() - 1;
    }

    void insertAndCoalesce(std::vector<Free_Block> &_free_blocks, VkDeviceSize _offset, VkDeviceSize _size)
    {
        Free_Block new_block{.offset = _offset, .size = _size};
        auto block_iterator = std::lower_bound(_free_blocks.begin(), _free_blocks.end(), new_block,
                                               [](const Free_Block &lhs, const Free_Block &rhs)
                                               {
                                                   return lhs.offset < rhs.offset;
                                               });

        block_iterator = _free_blocks.insert(block_iterator, new_block);

        if (block_iterator + 1 != _free_blocks.end())
        {
            if (block_iterator->offset + block_iterator->size == (block_iterator + 1)->offset)
            {
                Logger::logMessage(std::format("Vulkan_Sub_Allocator::insertAndCoalesce: Merging with next block at offset {} (new size={})",
                                               (block_iterator + 1)->offset, block_iterator->size + (block_iterator + 1)->size),
                                   Log_Level::LOG_DEBUG,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                block_iterator->size += (block_iterator + 1)->size;
                _free_blocks.erase(block_iterator + 1);
            }
        }

        if (block_iterator != _free_blocks.begin())
        {
            auto previous_block_iterator = block_iterator - 1;
            if (previous_block_iterator->offset + previous_block_iterator->size == block_iterator->offset)
            {
                Logger::logMessage(std::format("Vulkan_Sub_Allocator::insertAndCoalesce: Merging with previous block at offset {} (new size={})",
                                               previous_block_iterator->offset, previous_block_iterator->size + block_iterator->size),
                                   Log_Level::LOG_DEBUG,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                previous_block_iterator->size += block_iterator->size;
                _free_blocks.erase(block_iterator);
            }
        }
    }

public:
    explicit Vulkan_Sub_Allocator(VkDevice _device, const Vulkan_Context &_context, VkPhysicalDevice _physical_device = VK_NULL_HANDLE)
        : device(_device), context(_context), physical_device(_physical_device)
    {
        Logger::logMessage("Vulkan_Sub_Allocator::Vulkan_Sub_Allocator: Initializing sub allocator",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        if (physical_device != VK_NULL_HANDLE)
        {
            vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
        }
    }

    ~Vulkan_Sub_Allocator()
    {
        Logger::logMessage(std::format("Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator: Freeing {} memory chunks", memory_chunks.size()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        std::lock_guard<std::mutex> lock(allocator_mutex);
        for (const auto &chunk : memory_chunks)
        {
            if (chunk.device_memory != VK_NULL_HANDLE)
            {
                vkFreeMemory(device, chunk.device_memory, nullptr);
            }
            else
            {
                Logger::logMessage("Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator: Null device_memory handle encountered in chunk",
                                   Log_Level::LOG_WARNING,
                                   false,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
            }
        }
    }

    [[nodiscard]] Memory_Planner &getPlanner() noexcept
    {
        return memory_planner;
    }

    [[nodiscard]] const Memory_Planner &getPlanner() const noexcept
    {
        return memory_planner;
    }

    [[nodiscard]] Memory_Planner &getMemoryPlanner() noexcept
    {
        return memory_planner;
    }

    [[nodiscard]] const Memory_Planner &getMemoryPlanner() const noexcept
    {
        return memory_planner;
    }

    [[nodiscard]] VkDevice getDevice() const noexcept
    {
        return device;
    }

    [[nodiscard]] const Vulkan_Context &getContext() const noexcept
    {
        return context;
    }

    [[nodiscard]] VkPhysicalDevice getPhysicalDevice() const noexcept
    {
        return physical_device;
    }

    [[nodiscard]] const std::vector<Memory_Chunk> &getMemoryChunks() const noexcept
    {
        return memory_chunks;
    }

    [[nodiscard]] std::size_t getMemoryChunkCount() const noexcept
    {
        return memory_chunks.size();
    }

    [[nodiscard]] VkDeviceSize getDefaultChunkSize() const noexcept
    {
        return default_chunk_size;
    }

    [[nodiscard]] std::size_t getArenaChunkIndex() const noexcept
    {
        return arena_chunk_index;
    }

    [[nodiscard]] bool hasArenaChunk() const noexcept
    {
        return arena_chunk_index != std::numeric_limits<std::size_t>::max();
    }

    void free(const Memory_Allocation &_allocation)
    {
        if (_allocation.memory == VK_NULL_HANDLE || _allocation.size == 0)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(allocator_mutex);
        if (_allocation.chunk_index < memory_chunks.size())
        {
            if (memory_chunks[_allocation.chunk_index].is_dedicated_arena)
            {
                return;
            }

            Logger::logMessage(std::format("Vulkan_Sub_Allocator::free: Freeing block in chunk {} at offset {} of size {} bytes",
                                           _allocation.chunk_index, _allocation.offset, _allocation.size),
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            insertAndCoalesce(memory_chunks[_allocation.chunk_index].free_blocks, _allocation.offset, _allocation.size);
        }
        else
        {
            Logger::logMessage(std::format("Vulkan_Sub_Allocator::free: Invalid chunk_index {} (total chunks: {})",
                                           _allocation.chunk_index, memory_chunks.size()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
        }
    }

    Memory_Allocation allocateAliased(std::uint32_t _tensor_id, VkDeviceSize _size, VkMemoryPropertyFlags _memory_properties, VkPhysicalDevice _target_physical_device = VK_NULL_HANDLE)
    {
        std::lock_guard<std::mutex> lock(allocator_mutex);

        if (_target_physical_device != VK_NULL_HANDLE && physical_device != _target_physical_device)
        {
            physical_device = _target_physical_device;
            vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
        }

        VkDeviceSize planned_offset = memory_planner.getOffset(_tensor_id);
        VkDeviceSize total_required = memory_planner.getTotalMemoryRequired();

        if (arena_chunk_index == std::numeric_limits<std::size_t>::max())
        {
            VkMemoryRequirements memory_requirements{
                .size = total_required,
                .alignment = 16,
                .memoryTypeBits = 0xFFFFFFFF};
            arena_chunk_index = createChunk(memory_requirements, _memory_properties, true);
        }

        auto &arena_chunk = memory_chunks[arena_chunk_index];
        if (planned_offset + _size > arena_chunk.chunk_size)
        {
            Logger::logMessage(std::format("Vulkan_Sub_Allocator::allocateAliased: Out of bounds on arena chunk {} (offset={}, size={}, chunk_size={})",
                                           arena_chunk_index, planned_offset, _size, arena_chunk.chunk_size),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Vulkan_Sub_Allocator::allocateAliased: Requested offset + size exceeds arena capacity");
        }

        return Memory_Allocation{
            .memory = arena_chunk.device_memory,
            .offset = planned_offset,
            .size = _size,
            .chunk_index = arena_chunk_index};
    }

    Memory_Allocation allocate(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, VkPhysicalDevice _target_physical_device = VK_NULL_HANDLE)
    {
        std::lock_guard<std::mutex> lock(allocator_mutex);

        Logger::logMessage(std::format("Vulkan_Sub_Allocator::allocate: Requesting size {} bytes, alignment {} bytes, memory_properties {}",
                                       _memory_requirements.size, _memory_requirements.alignment, static_cast<std::uint32_t>(_memory_properties)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        if (_target_physical_device != VK_NULL_HANDLE && physical_device != _target_physical_device)
        {
            physical_device = _target_physical_device;
            vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
        }

        std::size_t best_chunk_index = std::numeric_limits<std::size_t>::max();
        std::size_t best_block_index = std::numeric_limits<std::size_t>::max();
        VkDeviceSize minimum_leftover_size = std::numeric_limits<VkDeviceSize>::max();
        VkDeviceSize best_aligned_offset = 0;
        VkDeviceSize best_padding = 0;

        for (std::size_t chunk_index = 0; chunk_index < memory_chunks.size(); ++chunk_index)
        {
            auto &chunk = memory_chunks[chunk_index];

            if (chunk.is_dedicated_arena ||
                (_memory_requirements.memoryTypeBits & (1u << chunk.memory_type_index)) == 0 ||
                (chunk.memory_properties & _memory_properties) != _memory_properties)
            {
                continue;
            }

            for (std::size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index)
            {
                const auto &block = chunk.free_blocks[block_index];
                VkDeviceSize alignment = _memory_requirements.alignment;
                VkDeviceSize aligned_offset = (alignment <= 1) ? block.offset : ((block.offset + alignment - 1) / alignment) * alignment;
                VkDeviceSize padding = aligned_offset - block.offset;
                VkDeviceSize total_required = _memory_requirements.size + padding;

                if (block.size >= total_required)
                {
                    VkDeviceSize leftover = block.size - total_required;
                    if (leftover < minimum_leftover_size)
                    {
                        minimum_leftover_size = leftover;
                        best_chunk_index = chunk_index;
                        best_block_index = block_index;
                        best_aligned_offset = aligned_offset;
                        best_padding = padding;

                        if (leftover == 0)
                        {
                            break;
                        }
                    }
                }
            }

            if (minimum_leftover_size == 0)
            {
                break;
            }
        }

        if (best_chunk_index != std::numeric_limits<std::size_t>::max())
        {
            auto &chunk = memory_chunks[best_chunk_index];
            auto it = chunk.free_blocks.begin() + best_block_index;

            VkDeviceSize allocated_offset = best_aligned_offset;
            VkDeviceSize allocation_size = _memory_requirements.size;
            VkDeviceSize trailing_size = minimum_leftover_size;
            VkDeviceSize padding = best_padding;

            if (padding > 0 && trailing_size > 0)
            {
                it->size = padding;
                Free_Block trailing_block{.offset = allocated_offset + allocation_size, .size = trailing_size};
                chunk.free_blocks.insert(it + 1, trailing_block);
            }
            else if (padding > 0 && trailing_size == 0)
            {
                it->size = padding;
            }
            else if (padding == 0 && trailing_size > 0)
            {
                it->offset = allocated_offset + allocation_size;
                it->size = trailing_size;
            }
            else
            {
                chunk.free_blocks.erase(it);
            }

            Logger::logMessage(std::format("Vulkan_Sub_Allocator::allocate: Reusing chunk {} at offset {} (size={} bytes, padding={} bytes, remaining blocks={})",
                                           best_chunk_index, allocated_offset, allocation_size, padding, chunk.free_blocks.size()),
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);

            return Memory_Allocation{
                .memory = chunk.device_memory,
                .offset = allocated_offset,
                .size = allocation_size,
                .chunk_index = best_chunk_index};
        }

        std::size_t new_chunk_index = createChunk(_memory_requirements, _memory_properties, false);
        auto &new_chunk = memory_chunks[new_chunk_index];
        auto &block = new_chunk.free_blocks.front();

        VkDeviceSize alignment = _memory_requirements.alignment;
        VkDeviceSize aligned_offset = (alignment <= 1) ? block.offset : ((block.offset + alignment - 1) / alignment) * alignment;
        VkDeviceSize padding = aligned_offset - block.offset;
        VkDeviceSize total_required = _memory_requirements.size + padding;

        VkDeviceSize allocated_offset = aligned_offset;
        VkDeviceSize allocation_size = _memory_requirements.size;
        VkDeviceSize trailing_size = block.size - total_required;

        if (padding > 0 && trailing_size > 0)
        {
            block.size = padding;
            Free_Block trailing_block{.offset = allocated_offset + allocation_size, .size = trailing_size};
            new_chunk.free_blocks.push_back(trailing_block);
        }
        else if (padding > 0 && trailing_size == 0)
        {
            block.size = padding;
        }
        else if (padding == 0 && trailing_size > 0)
        {
            block.offset = allocated_offset + allocation_size;
            block.size = trailing_size;
        }
        else
        {
            new_chunk.free_blocks.clear();
        }

        Logger::logMessage(std::format("Vulkan_Sub_Allocator::allocate: Allocated in newly created chunk {} at offset {} (size={} bytes, padding={} bytes, remaining blocks={})",
                                       new_chunk_index, allocated_offset, allocation_size, padding, new_chunk.free_blocks.size()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        return Memory_Allocation{
            .memory = new_chunk.device_memory,
            .offset = allocated_offset,
            .size = allocation_size,
            .chunk_index = new_chunk_index};
    }
};
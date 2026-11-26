#include "engine/vulkan_sub_allocator.h"
#include "engine/vulkan_context.h"

#include <algorithm>
#include <bit>
#include <format>
#include <limits>
#include <stdexcept>

#include "helper/logger.h"
#include "helper/user_preferences.h"

void Memory_Planner::registerTensor(uint32_t _tensor_id, VkDeviceSize _size, VkDeviceSize _alignment, uint32_t _birth_node_index)
{
    tensor_lifetimes.push_back(Tensor_Lifetime{
        .tensor_id = _tensor_id,
        .size = _size,
        .alignment = _alignment,
        .start_node_index = _birth_node_index,
        .end_node_index = _birth_node_index,
        .allocated_offset = std::numeric_limits<VkDeviceSize>::max()});
}

void Memory_Planner::updateLastUsage(uint32_t _tensor_id, uint32_t _current_node_index)
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

void Memory_Planner::planMemoryLayout()
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
        size_t best_block_index = std::numeric_limits<size_t>::max();
        VkDeviceSize minimum_leftover = std::numeric_limits<VkDeviceSize>::max();
        VkDeviceSize best_aligned_offset = 0;

        for (size_t block_index = 0; block_index < virtual_blocks.size(); ++block_index)
        {
            const auto &block = virtual_blocks[block_index];
            const bool overlaps = std::any_of(
                block.occupied_intervals.begin(),
                block.occupied_intervals.end(),
                [&](const std::pair<uint32_t, uint32_t> &iv)
                {
                    return !(iv.second < tensor.start_node_index || tensor.end_node_index < iv.first);
                });
            if (!overlaps)
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

        if (best_block_index != std::numeric_limits<size_t>::max())
        {
            tensor.allocated_offset = best_aligned_offset;
            virtual_blocks[best_block_index].free_after_node_index = std::max(virtual_blocks[best_block_index].free_after_node_index, tensor.end_node_index);
            virtual_blocks[best_block_index].occupied_intervals.emplace_back(tensor.start_node_index, tensor.end_node_index);
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
                .free_after_node_index = tensor.end_node_index,
                .occupied_intervals = { { tensor.start_node_index, tensor.end_node_index } } });
        }
    }
}

void Memory_Planner::reset() noexcept
{
    tensor_lifetimes.clear();
    total_allocated_size = 0;
}

VkDeviceSize Memory_Planner::getOffset(uint32_t _tensor_id) const
{
    for (const auto &tensor : tensor_lifetimes)
    {
        if (tensor.tensor_id == _tensor_id)
        {
            if (tensor.allocated_offset == std::numeric_limits<VkDeviceSize>::max())
            {
                Logger::logMessage(Input_Format{"Memory_Planner::getOffset: Tensor ID {} has not been planned", _tensor_id},
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
                throw std::runtime_error(std::format("Memory_Planner::getOffset: Tensor ID {} has not been planned", _tensor_id));
            }
            return tensor.allocated_offset;
        }
    }
    Logger::logMessage(Input_Format{"Memory_Planner::getOffset: Tensor ID {} not found in registry", _tensor_id},
                       Log_Level::LOG_ERROR,
                       true,
                       0,
                       Log_Feature::MEMORY_ALLOCATION);
    throw std::runtime_error(std::format("Memory_Planner::getOffset: Tensor ID {} not found in registry", _tensor_id));
}

uint32_t Vulkan_Sub_Allocator::findMemoryType(uint32_t _type_filter, VkMemoryPropertyFlags _memory_properties) const
{
    const bool wants_host_visible = (_memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const bool wants_device_local = (_memory_properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

    uint32_t best_index = std::numeric_limits<uint32_t>::max();
    int best_score = std::numeric_limits<int>::min();

    for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i)
    {
        if ((_type_filter & (1u << i)) == 0)
        {
            continue;
        }
        const VkMemoryPropertyFlags flags = physical_device_memory_properties.memoryTypes[i].propertyFlags;
        if ((flags & _memory_properties) != _memory_properties)
        {
            continue;
        }

        int score = 0;
        if (wants_host_visible && !wants_device_local)
        {
            if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
            {
                score += 1000;
            }
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
            {
                score -= 500;
            }
        }
        const VkMemoryPropertyFlags extra_flags = flags & ~_memory_properties;
        score -= std::popcount(static_cast<uint32_t>(extra_flags));

        if (best_index == std::numeric_limits<uint32_t>::max() || score > best_score)
        {
            best_score = score;
            best_index = i;
        }
    }

    if (best_index != std::numeric_limits<uint32_t>::max())
    {
        return best_index;
    }

    Logger::logMessage("Vulkan_Sub_Allocator::findMemoryType: Failed to find suitable memory type",
                       Log_Level::LOG_ERROR,
                       true,
                       0,
                       Log_Feature::MEMORY_ALLOCATION);
    throw std::runtime_error("Vulkan_Sub_Allocator::findMemoryType: Failed to find suitable memory type");
}

size_t Vulkan_Sub_Allocator::createChunk(const VkMemoryRequirements& _memory_requirements, VkMemoryPropertyFlags _memory_properties, bool _is_dedicated_arena)
{
    uint32_t memory_type_index = findMemoryType(_memory_requirements.memoryTypeBits, _memory_properties);
    VkDeviceSize allocation_size = _is_dedicated_arena
        ? _memory_requirements.size
        : std::max(default_chunk_size, _memory_requirements.size);

    Logger::logMessage(Input_Format{ "Vulkan_Sub_Allocator::createChunk: Allocating new memory chunk of size {} bytes (memory_type_index={}, memory_properties={}, is_dedicated_arena={})",
                                   allocation_size, memory_type_index, static_cast<uint32_t>(_memory_properties), _is_dedicated_arena },
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::MEMORY_ALLOCATION);

    VkMemoryAllocateInfo allocate_information{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = allocation_size,
        .memoryTypeIndex = memory_type_index };

    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    VkResult alloc_res = vkAllocateMemory(device, &allocate_information, nullptr, &device_memory);

    if (alloc_res != VK_SUCCESS && allocation_size > _memory_requirements.size)
    {
        VkDeviceSize candidate_sizes[] = {
            std::max(allocation_size / 2, _memory_requirements.size),
            std::max(allocation_size / 4, _memory_requirements.size),
            std::max(allocation_size / 8, _memory_requirements.size),
            _memory_requirements.size
        };
        for (VkDeviceSize candidate : candidate_sizes)
        {
            if (candidate >= allocation_size) continue;
            allocate_information.allocationSize = candidate;
            alloc_res = vkAllocateMemory(device, &allocate_information, nullptr, &device_memory);
            if (alloc_res == VK_SUCCESS)
            {
                allocation_size = candidate;
                break;
            }
        }
    }

    if (alloc_res != VK_SUCCESS)
    {
        for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount; ++i)
        {
            if (i == memory_type_index) continue;
            if ((_memory_requirements.memoryTypeBits & (1u << i)) &&
                (physical_device_memory_properties.memoryTypes[i].propertyFlags & _memory_properties) == _memory_properties)
            {
                allocate_information.memoryTypeIndex = i;
                allocate_information.allocationSize = _memory_requirements.size;
                alloc_res = vkAllocateMemory(device, &allocate_information, nullptr, &device_memory);
                if (alloc_res == VK_SUCCESS)
                {
                    memory_type_index = i;
                    allocation_size = _memory_requirements.size;
                    break;
                }
            }
        }
    }

    if (alloc_res != VK_SUCCESS)
    {
        Logger::logMessage(Input_Format{ "Vulkan_Sub_Allocator::createChunk: Failed to allocate device memory chunk (requested size={}, error={})",
                                       allocation_size, static_cast<int>(alloc_res) },
            Log_Level::LOG_ERROR,
            true,
            0,
            Log_Feature::MEMORY_ALLOCATION);
        throw std::runtime_error("Vulkan_Sub_Allocator::createChunk: Failed to allocate device memory chunk");
    }

    void* mapped_ptr = nullptr;
    if ((_memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        if (vkMapMemory(device, device_memory, 0, allocation_size, 0, &mapped_ptr) != VK_SUCCESS)
        {
            vkFreeMemory(device, device_memory, nullptr);
            Logger::logMessage("Vulkan_Sub_Allocator::createChunk: Failed to map host visible memory chunk",
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::MEMORY_ALLOCATION);
            throw std::runtime_error("Vulkan_Sub_Allocator::createChunk: Failed to map host visible memory chunk");
        }
    }

    const VkMemoryPropertyFlags actual_flags =
        (memory_type_index < physical_device_memory_properties.memoryTypeCount)
            ? physical_device_memory_properties.memoryTypes[memory_type_index].propertyFlags
            : _memory_properties;

    Memory_Chunk new_chunk{
        .device_memory = device_memory,
        .chunk_size = allocation_size,
        .memory_type_index = memory_type_index,
        .memory_properties = actual_flags,
        .free_blocks = _is_dedicated_arena ? std::vector<Free_Block>{} : std::vector<Free_Block>{ Free_Block{ .offset = 0, .size = allocation_size } },
        .is_dedicated_arena = _is_dedicated_arena,
        .mapped_pointer = mapped_ptr };

    for (size_t i = 0; i < memory_chunks.size(); ++i)
    {
        if (memory_chunks[i].device_memory == VK_NULL_HANDLE && !memory_chunks[i].is_dedicated_arena)
        {
            memory_chunks[i] = std::move(new_chunk);
            return i;
        }
    }

    memory_chunks.push_back(std::move(new_chunk));
    return memory_chunks.size() - 1;
}

void Vulkan_Sub_Allocator::insertAndCoalesce(std::vector<Free_Block> &_free_blocks, VkDeviceSize _offset, VkDeviceSize _size)
{
    Free_Block new_block{.offset = _offset, .size = _size};
    auto block_iterator = std::lower_bound(_free_blocks.begin(), _free_blocks.end(), new_block,
                                           [](const Free_Block &lhs, const Free_Block &rhs)
                                           {
                                               return lhs.offset < rhs.offset;
                                           });

    if (block_iterator != _free_blocks.end() && (_offset >= block_iterator->offset || _size > block_iterator->offset - _offset))
    {
        Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::insertAndCoalesce: Overlapping or double-free block with next free block (offset={}, size={}, next_offset={})",
                                       _offset, _size, block_iterator->offset},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        return;
    }

    if (block_iterator != _free_blocks.begin())
    {
        auto prev_it = block_iterator - 1;
        if (prev_it->size > _offset - prev_it->offset)
        {
            Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::insertAndCoalesce: Overlapping or double-free block with previous free block (prev_offset={}, prev_size={}, offset={})",
                                           prev_it->offset, prev_it->size, _offset},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }
    }

    block_iterator = _free_blocks.insert(block_iterator, new_block);

    if (block_iterator + 1 != _free_blocks.end())
    {
        if (block_iterator->offset + block_iterator->size == (block_iterator + 1)->offset)
        {
            Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::insertAndCoalesce: Merging with next block at offset {} (new size={})",
                                           (block_iterator + 1)->offset, block_iterator->size + (block_iterator + 1)->size},
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
            Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::insertAndCoalesce: Merging with previous block at offset {} (new size={})",
                                           previous_block_iterator->offset, previous_block_iterator->size + block_iterator->size},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            previous_block_iterator->size += block_iterator->size;
            _free_blocks.erase(block_iterator);
        }
    }
}

Vulkan_Sub_Allocator::Vulkan_Sub_Allocator(VkDevice _device, const Vulkan_Context &_context, VkPhysicalDevice _physical_device)
    : device(_device), context(_context), physical_device(_physical_device)
{
    uint32_t staging_mb = User_Preferences::getInstance().getMemoryPreferences().staging_pool_initial_size_mb;
    if (staging_mb > 0)
    {
        default_chunk_size = static_cast<VkDeviceSize>(staging_mb) * 1024 * 1024;
    }

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

Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator()
{
    Logger::logMessage(Input_Format{ "Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator: Freeing {} memory chunks", memory_chunks.size() },
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::MEMORY_ALLOCATION);
    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
    for (size_t i = 0; i < memory_chunks.size(); ++i)
    {
        auto &chunk = memory_chunks[i];
        if (chunk.device_memory != VK_NULL_HANDLE)
        {
            const bool is_completely_free = chunk.free_blocks.size() == 1 &&
                                            chunk.free_blocks[0].offset == 0 &&
                                            chunk.free_blocks[0].size == chunk.chunk_size;
            if (!chunk.is_dedicated_arena && !is_completely_free)
            {
                Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::~Vulkan_Sub_Allocator: Unfreed allocation leaked in chunk {} (size={} bytes)",
                                                i, chunk.chunk_size},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
            }
            if (chunk.mapped_pointer != nullptr)
            {
                vkUnmapMemory(device, chunk.device_memory);
                chunk.mapped_pointer = nullptr;
            }
            vkFreeMemory(device, chunk.device_memory, nullptr);
            chunk.device_memory = VK_NULL_HANDLE;
        }
    }
}

void Vulkan_Sub_Allocator::free(const Memory_Allocation &_allocation)
{
    if (_allocation.memory == VK_NULL_HANDLE || _allocation.size == 0)
    {
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);
    if (_allocation.chunk_index < memory_chunks.size())
    {
        auto &chunk = memory_chunks[_allocation.chunk_index];
        if (chunk.is_dedicated_arena)
        {
            return;
        }

        if (_allocation.memory != chunk.device_memory ||
            _allocation.offset > chunk.chunk_size ||
            _allocation.size > chunk.chunk_size - _allocation.offset)
        {
            Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::free: Invalid allocation bounds or memory handle in chunk {} (offset={}, size={}, chunk_size={})",
                                           _allocation.chunk_index, _allocation.offset, _allocation.size, chunk.chunk_size},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MEMORY_ALLOCATION);
            return;
        }

        Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::free: Freeing block in chunk {} at offset {} of size {} bytes",
                                       _allocation.chunk_index, _allocation.offset, _allocation.size},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
        insertAndCoalesce(chunk.free_blocks, _allocation.offset, _allocation.size);

        if (chunk.free_blocks.size() == 1 &&
            chunk.free_blocks[0].offset == 0 &&
            chunk.free_blocks[0].size == chunk.chunk_size)
        {
            const uint32_t mtype = chunk.memory_type_index;
            size_t empty_same_type_count = 0;
            for (const auto &c : memory_chunks)
            {
                if (c.device_memory != VK_NULL_HANDLE &&
                    !c.is_dedicated_arena &&
                    c.memory_type_index == mtype &&
                    c.free_blocks.size() == 1 &&
                    c.free_blocks[0].offset == 0 &&
                    c.free_blocks[0].size == c.chunk_size)
                {
                    ++empty_same_type_count;
                }
            }

            if (empty_same_type_count > 1 || chunk.chunk_size > default_chunk_size)
            {
                const VkDeviceSize freed_size = chunk.chunk_size;
                if (chunk.mapped_pointer != nullptr)
                {
                    vkUnmapMemory(device, chunk.device_memory);
                    chunk.mapped_pointer = nullptr;
                }
                vkFreeMemory(device, chunk.device_memory, nullptr);
                chunk.device_memory = VK_NULL_HANDLE;
                chunk.chunk_size = 0;
                chunk.free_blocks.clear();
                ++total_reclaimed_chunks;
                total_reclaimed_bytes += static_cast<uint64_t>(freed_size);
                Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::free: Reclaimed empty chunk {} ({} bytes)",
                                                _allocation.chunk_index, freed_size},
                                   Log_Level::LOG_DEBUG,
                                   true,
                                   0,
                                   Log_Feature::MEMORY_ALLOCATION);
            }
        }
    }
    else
    {
        Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::free: Invalid chunk_index {} (total chunks: {})",
                                       _allocation.chunk_index, memory_chunks.size()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
    }
}

Memory_Allocation Vulkan_Sub_Allocator::allocate(const VkMemoryRequirements &_memory_requirements, VkMemoryPropertyFlags _memory_properties, VkPhysicalDevice _target_physical_device)
{
    if (_memory_requirements.size == 0)
    {
        return Memory_Allocation{};
    }

    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);

    Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::allocate: Requesting size {} bytes, alignment {} bytes, memory_properties {}",
                                   _memory_requirements.size, _memory_requirements.alignment, static_cast<uint32_t>(_memory_properties)},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MEMORY_ALLOCATION);

    if (_target_physical_device != VK_NULL_HANDLE && physical_device != _target_physical_device)
    {
        physical_device = _target_physical_device;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_device_memory_properties);
    }

    const bool wants_host_visible = (_memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const bool wants_device_local = (_memory_properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    const uint32_t preferred_type = findMemoryType(_memory_requirements.memoryTypeBits, _memory_properties);
    const bool preferred_is_device_local =
        (preferred_type < physical_device_memory_properties.memoryTypeCount) &&
        ((physical_device_memory_properties.memoryTypes[preferred_type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0);

    size_t best_chunk_index = std::numeric_limits<size_t>::max();
    size_t best_block_index = std::numeric_limits<size_t>::max();
    VkDeviceSize minimum_leftover_size = std::numeric_limits<VkDeviceSize>::max();
    VkDeviceSize best_aligned_offset = 0;
    VkDeviceSize best_padding = 0;

    for (size_t chunk_index = 0; chunk_index < memory_chunks.size(); ++chunk_index)
    {
        auto &chunk = memory_chunks[chunk_index];

        if (chunk.device_memory == VK_NULL_HANDLE ||
            chunk.is_dedicated_arena ||
            (_memory_requirements.memoryTypeBits & (1u << chunk.memory_type_index)) == 0 ||
            (chunk.memory_properties & _memory_properties) != _memory_properties ||
            (wants_host_visible && chunk.mapped_pointer == nullptr) ||
            (wants_host_visible && !wants_device_local && !preferred_is_device_local &&
             (chunk.memory_properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0))
        {
            continue;
        }

        for (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index)
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

    if (best_chunk_index != std::numeric_limits<size_t>::max())
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

        Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::allocate: Reusing chunk {} at offset {} (size={} bytes, padding={} bytes, remaining blocks={})",
                                       best_chunk_index, allocated_offset, allocation_size, padding, chunk.free_blocks.size()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);

        return Memory_Allocation{
                    .memory = chunk.device_memory,
                    .offset = allocated_offset,
                    .size = allocation_size,
                    .chunk_index = best_chunk_index,
                    .mapped_pointer = (chunk.mapped_pointer != nullptr)
                                          ? static_cast<void*>(static_cast<uint8_t*>(chunk.mapped_pointer) + allocated_offset)
                                          : nullptr };
    }

    size_t new_chunk_index = createChunk(_memory_requirements, _memory_properties, false);
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

    Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::allocate: Allocated in newly created chunk {} at offset {} (size={} bytes, padding={} bytes, remaining blocks={})",
                                   new_chunk_index, allocated_offset, allocation_size, padding, new_chunk.free_blocks.size()},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MEMORY_ALLOCATION);

    return Memory_Allocation{
            .memory = new_chunk.device_memory,
            .offset = allocated_offset,
            .size = allocation_size,
            .chunk_index = new_chunk_index,
            .mapped_pointer = (new_chunk.mapped_pointer != nullptr)
                                  ? static_cast<void*>(static_cast<uint8_t*>(new_chunk.mapped_pointer) + allocated_offset)
                                  : nullptr };
}

size_t Vulkan_Sub_Allocator::allocateDedicatedArena(VkDeviceSize _total_size, uint32_t _memory_type_bits, VkMemoryPropertyFlags _memory_properties)
{
    if (_total_size == 0)
    {
        throw std::invalid_argument("Vulkan_Sub_Allocator::allocateDedicatedArena: Total size must be > 0");
    }

    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);

    VkMemoryRequirements mem_req{
        .size = _total_size,
        .alignment = 256,
        .memoryTypeBits = _memory_type_bits
    };
    size_t chunk_idx = createChunk(mem_req, _memory_properties, true);
    arena_chunk_index = chunk_idx;
    return chunk_idx;
}

Memory_Allocation Vulkan_Sub_Allocator::getArenaAllocation(size_t _chunk_index, VkDeviceSize _offset, VkDeviceSize _size) const
{
    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);

    if (_chunk_index >= memory_chunks.size())
    {
        throw std::out_of_range("Vulkan_Sub_Allocator::getArenaAllocation: chunk index out of range");
    }
    const auto &chunk = memory_chunks[_chunk_index];
    if (!chunk.is_dedicated_arena || chunk.device_memory == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Vulkan_Sub_Allocator::getArenaAllocation: chunk is not an active dedicated arena");
    }
    if (_size == 0 || _offset > chunk.chunk_size || _size > chunk.chunk_size - _offset)
    {
        throw std::out_of_range("Vulkan_Sub_Allocator::getArenaAllocation: offset + size exceeds arena bounds");
    }

    return Memory_Allocation{
        .memory = chunk.device_memory,
        .offset = _offset,
        .size = _size,
        .chunk_index = _chunk_index,
        .mapped_pointer = (chunk.mapped_pointer != nullptr)
                              ? static_cast<void *>(static_cast<uint8_t *>(chunk.mapped_pointer) + _offset)
                              : nullptr
    };
}

void Vulkan_Sub_Allocator::releaseDedicatedArena(size_t _chunk_index)
{
    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);

    if (_chunk_index >= memory_chunks.size())
    {
        return;
    }
    auto &chunk = memory_chunks[_chunk_index];
    if (!chunk.is_dedicated_arena)
    {
        return;
    }
    if (chunk.mapped_pointer != nullptr && chunk.device_memory != VK_NULL_HANDLE)
    {
        vkUnmapMemory(device, chunk.device_memory);
        chunk.mapped_pointer = nullptr;
    }
    if (chunk.device_memory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, chunk.device_memory, nullptr);
        chunk.device_memory = VK_NULL_HANDLE;
    }
    chunk.chunk_size = 0;
    chunk.free_blocks.clear();
    chunk.is_dedicated_arena = false;
    if (arena_chunk_index == _chunk_index)
    {
        arena_chunk_index = std::numeric_limits<size_t>::max();
    }
}

uint64_t Vulkan_Sub_Allocator::reclaimEmptyChunks(bool _keep_one_default_chunk_per_type)
{
    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);

    uint64_t reclaimed_now = 0;
    size_t chunks_reclaimed_now = 0;
    std::vector<uint32_t> kept_memory_types;

    for (size_t i = 0; i < memory_chunks.size(); ++i)
    {
        auto &chunk = memory_chunks[i];
        if (chunk.device_memory == VK_NULL_HANDLE || chunk.is_dedicated_arena)
        {
            continue;
        }
        const bool is_completely_free = chunk.free_blocks.size() == 1 &&
                                        chunk.free_blocks[0].offset == 0 &&
                                        chunk.free_blocks[0].size == chunk.chunk_size;
        if (!is_completely_free)
        {
            continue;
        }

        if (_keep_one_default_chunk_per_type && chunk.chunk_size <= default_chunk_size)
        {
            if (std::find(kept_memory_types.begin(), kept_memory_types.end(), chunk.memory_type_index) == kept_memory_types.end())
            {
                kept_memory_types.push_back(chunk.memory_type_index);
                continue;
            }
        }

        const VkDeviceSize freed_size = chunk.chunk_size;
        if (chunk.mapped_pointer != nullptr)
        {
            vkUnmapMemory(device, chunk.device_memory);
            chunk.mapped_pointer = nullptr;
        }
        vkFreeMemory(device, chunk.device_memory, nullptr);
        chunk.device_memory = VK_NULL_HANDLE;
        chunk.chunk_size = 0;
        chunk.free_blocks.clear();

        reclaimed_now += static_cast<uint64_t>(freed_size);
        ++chunks_reclaimed_now;
    }

    if (chunks_reclaimed_now > 0)
    {
        total_reclaimed_chunks += chunks_reclaimed_now;
        total_reclaimed_bytes += reclaimed_now;
        Logger::logMessage(Input_Format{"Vulkan_Sub_Allocator::reclaimEmptyChunks: Reclaimed {} empty chunk(s) totaling {} bytes",
                                        chunks_reclaimed_now, reclaimed_now},
                           Log_Level::LOG_INFO,
                           false,
                           0,
                           Log_Feature::MEMORY_ALLOCATION);
    }

    return reclaimed_now;
}

Allocator_Stats Vulkan_Sub_Allocator::getStats() const
{
    std::lock_guard<std::recursive_mutex> lock(allocator_mutex);

    Allocator_Stats stats{};
    stats.reclaimed_chunk_count = total_reclaimed_chunks;
    stats.reclaimed_bytes = total_reclaimed_bytes;

    for (const auto &chunk : memory_chunks)
    {
        if (chunk.device_memory == VK_NULL_HANDLE)
        {
            continue;
        }
        ++stats.active_chunk_count;
        const uint64_t cap = static_cast<uint64_t>(chunk.chunk_size);
        stats.total_allocated_bytes += cap;

        if (chunk.is_dedicated_arena)
        {
            stats.dedicated_arena_bytes += cap;
            stats.total_used_bytes += cap;
            continue;
        }

        uint64_t chunk_free = 0;
        for (const auto &fb : chunk.free_blocks)
        {
            chunk_free += static_cast<uint64_t>(fb.size);
        }
        stats.total_free_bytes += chunk_free;
        if (cap >= chunk_free)
        {
            stats.total_used_bytes += (cap - chunk_free);
        }
        if (chunk.free_blocks.size() == 1 &&
            chunk.free_blocks[0].offset == 0 &&
            chunk.free_blocks[0].size == chunk.chunk_size)
        {
            ++stats.empty_chunk_count;
        }
    }

    return stats;
}

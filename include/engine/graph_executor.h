#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "compute_node.h"
#include "gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "pipeline_cache_manager.h"
#include "shader_dictionary.h"
#include "shader_generator.h"
#include "vulkan_context.h"
#include "vulkan_network.h"

extern bool is_coop;

struct Persistent_Descriptor_Entry
{
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    std::vector<VkBuffer> bound_buffers;
    std::vector<std::uint32_t> bound_binding_indices;
};

class Graph_Executor
{
private:
    const Vulkan_Context &context;
    const Vulkan_Network &network;
    Pipeline_Cache_Manager &pipeline_cache_manager;
    const Shader_Dictionary &shader_dictionary;

    mutable std::vector<std::string> printed_terminal_shader_chains;

    VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pools[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};

    std::vector<Persistent_Descriptor_Entry> persistent_descriptor_caches[MAX_FRAMES_IN_FLIGHT];
    std::vector<std::vector<Persistent_Descriptor_Entry>> fallback_descriptor_caches[MAX_FRAMES_IN_FLIGHT];

    mutable std::vector<VkDescriptorBufferInfo> shared_descriptor_buffer_informations;
    mutable std::vector<VkWriteDescriptorSet> shared_write_descriptor_sets;
    mutable std::vector<std::shared_ptr<gpu::vector>> shared_fused_buffers;
    mutable std::vector<std::uint32_t> shared_external_buffer_indices;

    static bool isBuffersMatching(const Persistent_Descriptor_Entry &_entry,
                                  const std::vector<std::uint32_t> &_binding_indices,
                                  const std::vector<std::shared_ptr<gpu::vector>> &_buffers)
    {
        if (_entry.descriptor_set == VK_NULL_HANDLE)
        {
            return false;
        }
        if (_entry.bound_binding_indices.size() != _binding_indices.size() || _entry.bound_buffers.size() != _buffers.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < _binding_indices.size(); ++i)
        {
            if (_entry.bound_binding_indices[i] != _binding_indices[i])
            {
                return false;
            }
        }
        for (std::size_t i = 0; i < _buffers.size(); ++i)
        {
            VkBuffer current_vulkan_buffer = _buffers[i] ? _buffers[i]->getBuffer() : VK_NULL_HANDLE;
            if (_entry.bound_buffers[i] != current_vulkan_buffer)
            {
                return false;
            }
        }
        return true;
    }

    static bool isBuffersMatching(const Persistent_Descriptor_Entry &_entry,
                                  const std::vector<std::shared_ptr<gpu::vector>> &_buffers)
    {
        if (_entry.descriptor_set == VK_NULL_HANDLE)
        {
            return false;
        }
        if (!_entry.bound_binding_indices.empty() || _entry.bound_buffers.size() != _buffers.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < _buffers.size(); ++i)
        {
            VkBuffer current_vulkan_buffer = _buffers[i] ? _buffers[i]->getBuffer() : VK_NULL_HANDLE;
            if (_entry.bound_buffers[i] != current_vulkan_buffer)
            {
                return false;
            }
        }
        return true;
    }

    static std::string replacePlaceholders(
        std::string _text,
        const std::vector<std::string> &_input_identifiers,
        const std::vector<bool> &_is_input_register_flags,
        const std::vector<std::string> &_output_identifiers,
        const std::vector<bool> &_is_output_register_flags,
        const std::vector<std::uint32_t> &_output_buffer_indices,
        const std::unordered_set<std::uint32_t> &_external_buffer_indices_set,
        std::uint32_t _push_constants_word_offset)
    {
        for (std::size_t i = 0; i < _input_identifiers.size(); ++i)
        {
            std::string token_prefix = std::format("{{in_{}}}[", i);
            std::string token_plain = std::format("{{in_{}}}", i);

            if (_is_input_register_flags[i])
            {
                std::size_t position = 0;
                while ((position = _text.find(token_prefix, position)) != std::string::npos)
                {
                    std::size_t end_position = _text.find(']', position);
                    if (end_position != std::string::npos)
                    {
                        _text.replace(position, end_position - position + 1, _input_identifiers[i]);
                        position += _input_identifiers[i].length();
                    }
                    else
                    {
                        break;
                    }
                }
            }

            std::size_t position = 0;
            while ((position = _text.find(token_plain, position)) != std::string::npos)
            {
                _text.replace(position, token_plain.length(), _input_identifiers[i]);
                position += _input_identifiers[i].length();
            }
        }

        for (std::size_t i = 0; i < _output_identifiers.size(); ++i)
        {
            std::string token_prefix = std::format("{{out_{}}}[", i);
            std::string token_plain = std::format("{{out_{}}}", i);

            if (_is_output_register_flags[i])
            {
                std::uint32_t real_buffer_index = (i < _output_buffer_indices.size()) ? _output_buffer_indices[i] : 0;
                bool is_external = _external_buffer_indices_set.contains(real_buffer_index);

                std::size_t position = 0;
                while ((position = _text.find(token_prefix, position)) != std::string::npos)
                {
                    std::size_t end_position = _text.find(']', position);
                    if (end_position != std::string::npos)
                    {
                        std::string index_expression = _text.substr(position + token_prefix.length(), end_position - (position + token_prefix.length()));
                        _text.replace(position, end_position - position + 1, _output_identifiers[i]);
                        position += _output_identifiers[i].length();

                        if (is_external)
                        {
                            std::size_t semicolon_position = _text.find(';', position);
                            if (semicolon_position != std::string::npos)
                            {
                                std::string write_statement = std::format(" buf_{}[{}] = {};", real_buffer_index, index_expression, _output_identifiers[i]);
                                _text.insert(semicolon_position + 1, write_statement);
                                position = semicolon_position + 1 + write_statement.length();
                            }
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            }

            std::size_t position = 0;
            while ((position = _text.find(token_plain, position)) != std::string::npos)
            {
                _text.replace(position, token_plain.length(), _output_identifiers[i]);
                position += _output_identifiers[i].length();
            }
        }

        for (int i = 31; i >= 0; --i)
        {
            std::string token = std::format("{{pc_{}}}", i);
            std::size_t position = 0;
            while ((position = _text.find(token, position)) != std::string::npos)
            {
                std::string replacement = std::format("pc.data[{}]", _push_constants_word_offset + i);
                _text.replace(position, token.length(), replacement);
                position += replacement.length();
            }
        }
        return _text;
    }

    void insertBufferMemoryBarriers(VkCommandBuffer _command_buffer, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
    {
        if (_buffers.empty())
        {
            return;
        }

        std::vector<VkBufferMemoryBarrier> buffer_barriers;
        buffer_barriers.reserve(_buffers.size());

        for (const auto &vector_pointer : _buffers)
        {
            if (vector_pointer && vector_pointer->getBuffer() != VK_NULL_HANDLE)
            {
                buffer_barriers.push_back(VkBufferMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .buffer = vector_pointer->getBuffer(),
                    .offset = 0,
                    .size = VK_WHOLE_SIZE});
            }
        }

        if (!buffer_barriers.empty())
        {
            vkCmdPipelineBarrier(
                _command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                static_cast<std::uint32_t>(buffer_barriers.size()), buffer_barriers.data(),
                0, nullptr);
        }
    }

    void executeFallbackNode(VkCommandBuffer _command_buffer, const Compute_Node &_node, std::size_t _node_index, std::uint32_t _frame_index)
    {
        VkDevice device = context.getDevice();
        VkDescriptorSetLayout layout = network.getDescriptorSetLayout();

        if (_node_index >= fallback_descriptor_caches[_frame_index].size())
        {
            fallback_descriptor_caches[_frame_index].resize(_node_index + 1);
        }

        auto &node_fallback_entries = fallback_descriptor_caches[_frame_index][_node_index];
        if (node_fallback_entries.size() < _node.fused_operations.size())
        {
            node_fallback_entries.resize(_node.fused_operations.size());
        }

        for (std::size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const auto &operation = _node.fused_operations[operation_index];

            shared_fused_buffers.clear();
            for (std::uint32_t index : operation.input_buffer_indices)
            {
                if (index < _node.buffers.size())
                {
                    shared_fused_buffers.push_back(_node.buffers[index]);
                }
            }
            for (std::uint32_t index : operation.output_buffer_indices)
            {
                if (index < _node.buffers.size())
                {
                    shared_fused_buffers.push_back(_node.buffers[index]);
                }
            }

            auto &entry = node_fallback_entries[operation_index];
            if (entry.descriptor_set == VK_NULL_HANDLE)
            {
                VkDescriptorSetAllocateInfo allocate_information{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .pNext = nullptr,
                    .descriptorPool = descriptor_pools[_frame_index],
                    .descriptorSetCount = 1,
                    .pSetLayouts = &layout};

                if (vkAllocateDescriptorSets(device, &allocate_information, &entry.descriptor_set) != VK_SUCCESS)
                {
                    Logger::logMessage(std::format("Graph_Executor::executeFallbackNode: Failed to allocate descriptor set for fallback op {}", operation_index),
                                       Log_Level::LOG_ERROR,
                                       true,
                                       0,
                                       Log_Feature::DISPATCH_EXECUTION);
                    throw std::runtime_error("Failed to allocate descriptor set");
                }
            }

            if (!isBuffersMatching(entry, shared_fused_buffers))
            {
                updateDescriptorSet(entry.descriptor_set, shared_fused_buffers);
                entry.bound_binding_indices.clear();
                entry.bound_buffers.resize(shared_fused_buffers.size());
                for (std::size_t j = 0; j < shared_fused_buffers.size(); ++j)
                {
                    entry.bound_buffers[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getBuffer() : VK_NULL_HANDLE;
                }
            }

            VkPipeline static_pipeline = network.getPipeline(operation.pipeline_id);

            vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, static_pipeline);
            vkCmdBindDescriptorSets(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &entry.descriptor_set, 0, nullptr);

            if (operation.push_constants_size > 0 && operation.push_constants_offset + operation.push_constants_size <= _node.push_constants_data.size())
            {
                vkCmdPushConstants(
                    _command_buffer,
                    network.getPipelineLayout(),
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    operation.push_constants_size,
                    _node.push_constants_data.data() + operation.push_constants_offset);
            }

            vkCmdDispatch(_command_buffer, operation.workgroup_count_x, operation.workgroup_count_y, operation.workgroup_count_z);

            if (operation_index < _node.fused_operations.size() - 1)
            {
                std::vector<std::shared_ptr<gpu::vector>> output_buffers;
                for (std::uint32_t out_index : operation.output_buffer_indices)
                {
                    if (out_index < _node.buffers.size() && _node.buffers[out_index])
                    {
                        output_buffers.push_back(_node.buffers[out_index]);
                    }
                }
                if (!output_buffers.empty())
                {
                    insertBufferMemoryBarriers(_command_buffer, output_buffers);
                }
            }
        }
    }

    void initializeResources()
    {
        Logger::logMessage("Graph_Executor::initializeResources: Allocating compute command buffers and descriptor pools",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);

        VkDevice device = context.getDevice();

        VkCommandBufferAllocateInfo allocate_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = context.getCommandPool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT};

        if (vkAllocateCommandBuffers(device, &allocate_information, command_buffers) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::initializeResources: Failed to allocate command buffers",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to allocate command buffers");
        }

        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16000}};

        VkDescriptorPoolCreateInfo pool_create_information{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = 4000,
            .poolSizeCount = 1,
            .pPoolSizes = pool_sizes};

        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateDescriptorPool(device, &pool_create_information, nullptr, &descriptor_pools[i]) != VK_SUCCESS)
            {
                Logger::logMessage(std::format("Graph_Executor::initializeResources: Failed to create descriptor pool for frame {}", i),
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
                throw std::runtime_error("Failed to create descriptor pool");
            }
        }

        shared_descriptor_buffer_informations.reserve(32);
        shared_write_descriptor_sets.reserve(32);
        shared_fused_buffers.reserve(32);
        shared_external_buffer_indices.reserve(32);
    }

    void updateDescriptorSet(VkDescriptorSet _descriptor_set, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
    {
        if (_buffers.empty())
        {
            Logger::logMessage("Graph_Executor::updateDescriptorSet: Node buffers vector is empty",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }

        VkDevice device = context.getDevice();
        std::size_t count = _buffers.size();

        shared_descriptor_buffer_informations.clear();
        shared_write_descriptor_sets.clear();

        for (std::size_t i = 0; i < count; ++i)
        {
            if (!_buffers[i] || _buffers[i]->getBuffer() == VK_NULL_HANDLE)
            {
                Logger::logMessage(std::format("Graph_Executor::updateDescriptorSet: Null buffer encountered at index {}", i),
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                continue;
            }

            shared_descriptor_buffer_informations.push_back(VkDescriptorBufferInfo{
                .buffer = _buffers[i]->getBuffer(),
                .offset = 0,
                .range = VK_WHOLE_SIZE});

            shared_write_descriptor_sets.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = _descriptor_set,
                .dstBinding = static_cast<std::uint32_t>(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr});
        }

        for (std::size_t i = 0; i < shared_write_descriptor_sets.size(); ++i)
        {
            shared_write_descriptor_sets[i].pBufferInfo = &shared_descriptor_buffer_informations[i];
        }

        if (!shared_write_descriptor_sets.empty())
        {
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(shared_write_descriptor_sets.size()), shared_write_descriptor_sets.data(), 0, nullptr);
        }
    }

    void updateDescriptorSet(VkDescriptorSet _descriptor_set,
                             const std::vector<std::uint32_t> &_binding_indices,
                             const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
    {
        if (_buffers.empty() || _binding_indices.size() != _buffers.size())
        {
            Logger::logMessage("Graph_Executor::updateDescriptorSet: Mismatched or empty binding/buffer list",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }

        VkDevice device = context.getDevice();
        std::size_t count = _buffers.size();

        shared_descriptor_buffer_informations.clear();
        shared_write_descriptor_sets.clear();

        for (std::size_t i = 0; i < count; ++i)
        {
            if (!_buffers[i] || _buffers[i]->getBuffer() == VK_NULL_HANDLE)
            {
                continue;
            }

            shared_descriptor_buffer_informations.push_back(VkDescriptorBufferInfo{
                .buffer = _buffers[i]->getBuffer(),
                .offset = 0,
                .range = VK_WHOLE_SIZE});

            shared_write_descriptor_sets.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = _descriptor_set,
                .dstBinding = _binding_indices[i],
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr});
        }

        for (std::size_t i = 0; i < shared_write_descriptor_sets.size(); ++i)
        {
            shared_write_descriptor_sets[i].pBufferInfo = &shared_descriptor_buffer_informations[i];
        }

        if (!shared_write_descriptor_sets.empty())
        {
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(shared_write_descriptor_sets.size()), shared_write_descriptor_sets.data(), 0, nullptr);
        }
    }

public:
    Graph_Executor(const Vulkan_Context &_context,
                   const Vulkan_Network &_network,
                   Pipeline_Cache_Manager &_pipeline_cache_manager,
                   const Shader_Dictionary &_shader_dictionary)
        : context(_context),
          network(_network),
          pipeline_cache_manager(_pipeline_cache_manager),
          shader_dictionary(_shader_dictionary)
    {
        initializeResources();
    }

    ~Graph_Executor()
    {
        Logger::logMessage("Graph_Executor::~Graph_Executor: Cleaning up Vulkan descriptor pools and command buffers",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
        VkDevice device = context.getDevice();
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (descriptor_pools[i] != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, descriptor_pools[i], nullptr);
            }
        }
        vkFreeCommandBuffers(device, context.getCommandPool(), MAX_FRAMES_IN_FLIGHT, command_buffers);
    }

    void getExternalBufferIndices(const Compute_Node &_node, std::vector<std::uint32_t> &_output_indices) const
    {
        _output_indices.clear();

        std::array<std::int32_t, 64> last_write_operation_indices;
        last_write_operation_indices.fill(-1);

        for (std::size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const auto &operation = _node.fused_operations[operation_index];
            for (std::uint32_t output_index : operation.output_buffer_indices)
            {
                if (output_index < last_write_operation_indices.size())
                {
                    last_write_operation_indices[output_index] = static_cast<std::int32_t>(operation_index);
                }
            }
        }

        std::array<bool, 64> is_added_flags{};

        for (std::size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const auto &operation = _node.fused_operations[operation_index];
            for (std::uint32_t input_index : operation.input_buffer_indices)
            {
                if (input_index < last_write_operation_indices.size())
                {
                    if (last_write_operation_indices[input_index] == -1 && !is_added_flags[input_index])
                    {
                        _output_indices.push_back(input_index);
                        is_added_flags[input_index] = true;
                    }
                }
            }
        }

        for (std::uint32_t external_index : _node.external_output_indices)
        {
            if (external_index < is_added_flags.size() && !is_added_flags[external_index])
            {
                _output_indices.push_back(external_index);
                is_added_flags[external_index] = true;
            }
        }

        for (const auto &operation : _node.fused_operations)
        {
            const auto &metadata = shader_dictionary.getMetadata(operation.pipeline_id);
            for (std::uint32_t persistent_output_local_index : metadata.persistent_output_indices)
            {
                if (persistent_output_local_index < operation.output_buffer_indices.size())
                {
                    std::uint32_t persistent_buffer_index = operation.output_buffer_indices[persistent_output_local_index];
                    if (persistent_buffer_index < is_added_flags.size() && !is_added_flags[persistent_buffer_index])
                    {
                        _output_indices.push_back(persistent_buffer_index);
                        is_added_flags[persistent_buffer_index] = true;
                    }
                }
            }
        }

        if (!_node.fused_operations.empty())
        {
            for (std::uint32_t output_index : _node.fused_operations.back().output_buffer_indices)
            {
                if (output_index < is_added_flags.size() && !is_added_flags[output_index])
                {
                    _output_indices.push_back(output_index);
                    is_added_flags[output_index] = true;
                }
            }
        }
    }

    std::string generateFusedGlsl(const Compute_Node &_node) const
    {
        std::uint32_t local_size_x = 256;
        std::uint32_t local_size_y = 1;
        std::uint32_t local_size_z = 1;

        Operation_Class primary_operation_class = Operation_Class::ELEMENTWISE;
        Compute_Pipeline primary_pipeline = Compute_Pipeline::ADD;

        bool has_reduction = false;
        bool has_shared_memory = false;
        bool has_matrix_tiles = false;
        std::uint32_t max_shared_memory_size = 0;

        if (!_node.fused_operations.empty())
        {
            primary_pipeline = _node.fused_operations[0].pipeline_id;
            primary_operation_class = shader_dictionary.getMetadata(primary_pipeline).operation_class;

            if (primary_operation_class == Operation_Class::MATRIX_2D || primary_operation_class == Operation_Class::TENSOR_3D)
            {
                if (is_coop && shader_dictionary.getMetadata(primary_pipeline).is_cooperative_matrix_support)
                {
                    local_size_x = 32;
                    local_size_y = 1;
                    local_size_z = 1;
                }
                else
                {
                    local_size_x = 16;
                    local_size_y = 16;
                    local_size_z = 1;
                }
            }

            for (const auto &operation : _node.fused_operations)
            {
                const auto &metadata = shader_dictionary.getMetadata(operation.pipeline_id);
                if (metadata.operation_class == Operation_Class::STANDALONE)
                {
                    has_reduction = true;
                }

                std::uint32_t op_shared_memory_size = (is_coop && metadata.is_cooperative_matrix_support)
                                                          ? metadata.cooperative_shared_memory_size
                                                          : metadata.shared_memory_size;

                if (op_shared_memory_size > 0)
                {
                    has_shared_memory = true;
                    has_reduction = true;
                    max_shared_memory_size = std::max(max_shared_memory_size, op_shared_memory_size);
                }

                bool is_using_manual_tiles = metadata.glsl_template.find("tile_a") != std::string::npos;

                if (is_using_manual_tiles && (!is_coop || !metadata.is_cooperative_matrix_support))
                {
                    has_matrix_tiles = true;
                }
            }
        }

        if (has_reduction && (!is_coop || !shader_dictionary.getMetadata(primary_pipeline).is_cooperative_matrix_support))
        {
            local_size_x = 256;
            local_size_y = 1;
            local_size_z = 1;
        }

        Shader_Generator shader_generator(local_size_x, local_size_y, local_size_z, "float");

        if (has_reduction)
        {
            shader_generator.enableSubgroupOperations();
        }

        if (has_shared_memory && max_shared_memory_size > 0)
        {
            shader_generator.addSharedMemory(max_shared_memory_size, "shared_mem_0", "float");
        }

        if (has_matrix_tiles)
        {
            shader_generator.addSharedMemoryRaw("shared float tile_a[16][17];\nshared float tile_b[16][17];");
        }

        std::vector<std::uint32_t> external_indices;
        getExternalBufferIndices(_node, external_indices);
        std::unordered_set<std::uint32_t> external_buffer_set(external_indices.begin(), external_indices.end());

        std::unordered_set<std::uint32_t> read_buffer_indices;
        std::unordered_set<std::uint32_t> written_buffer_indices;

        for (std::size_t op_idx = 0; op_idx < _node.fused_operations.size(); ++op_idx)
        {
            const auto &operation = _node.fused_operations[op_idx];

            for (std::uint32_t input_index : operation.input_buffer_indices)
            {
                bool is_passed_via_register = false;
                for (std::size_t prev_idx = 0; prev_idx < op_idx; ++prev_idx)
                {
                    const auto &prev_op = _node.fused_operations[prev_idx];
                    for (std::uint32_t prev_out : prev_op.output_buffer_indices)
                    {
                        if (prev_out == input_index)
                        {
                            is_passed_via_register = true;
                            break;
                        }
                    }
                    if (is_passed_via_register)
                    {
                        break;
                    }
                }

                if (!is_passed_via_register)
                {
                    read_buffer_indices.insert(input_index);
                }
            }

            for (std::uint32_t output_index : operation.output_buffer_indices)
            {
                written_buffer_indices.insert(output_index);
            }
        }

        for (std::uint32_t buffer_index : external_indices)
        {
            bool is_read = read_buffer_indices.contains(buffer_index);
            bool is_written = written_buffer_indices.contains(buffer_index) || _node.external_output_indices.contains(buffer_index);

            Buffer_Access buffer_access = Buffer_Access::READ_WRITE;
            if (is_read && !is_written)
            {
                buffer_access = Buffer_Access::READ_ONLY;
            }
            else if (!is_read && is_written)
            {
                buffer_access = Buffer_Access::WRITE_ONLY;
            }

            shader_generator.addBuffer(buffer_index, std::format("buf_{}", buffer_index), "float", buffer_access);
        }

        shader_generator.setPushConstants("uint data[32];");

        std::unordered_map<std::uint32_t, std::string> register_map;

        if (primary_operation_class == Operation_Class::MATRIX_2D)
        {
            if (!is_coop || !shader_dictionary.getMetadata(primary_pipeline).is_cooperative_matrix_support)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint r = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint lc = gl_LocalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint lr = gl_LocalInvocationID.y;");
            }
        }
        else if (primary_operation_class == Operation_Class::STANDALONE)
        {
            if (primary_pipeline == Compute_Pipeline::BATCH_NORM_STATS_FORWARD ||
                primary_pipeline == Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD ||
                primary_pipeline == Compute_Pipeline::BATCH_NORM_STATS_BACKWARD ||
                primary_pipeline == Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_WorkGroupID.x;");
                shader_generator.addLogicSnippet("    if (c >= pc.data[1]) return;");
            }
            else
            {
                shader_generator.addLogicSnippet("    uint r = gl_WorkGroupID.x;");
                shader_generator.addLogicSnippet("    if (r >= pc.data[0]) return;");
            }
        }
        else if (primary_operation_class == Operation_Class::TENSOR_3D)
        {
            if (primary_pipeline == Compute_Pipeline::CONV2D_FORWARD_PASS)
            {
                shader_generator.addLogicSnippet("    uint oc = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint ow = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint n_oh = gl_GlobalInvocationID.z;");
                shader_generator.addLogicSnippet("    uint n = n_oh / pc.data[4];");
                shader_generator.addLogicSnippet("    uint oh = n_oh % pc.data[4];");
                shader_generator.addLogicSnippet("    if (oc >= pc.data[6] || ow >= pc.data[5] || oh >= pc.data[4] || n >= pc.data[0]) return;");
            }
            else if (primary_pipeline == Compute_Pipeline::MAXPOOL2D_FORWARD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint ow = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint n_oh = gl_GlobalInvocationID.z;");
                shader_generator.addLogicSnippet("    uint n = n_oh / pc.data[4];");
                shader_generator.addLogicSnippet("    uint oh = n_oh % pc.data[4];");
                shader_generator.addLogicSnippet("    if (c >= pc.data[3] || ow >= pc.data[5] || oh >= pc.data[4] || n >= pc.data[0]) return;");
            }
            else if (primary_pipeline == Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT)
            {
                shader_generator.addLogicSnippet("    uint ic = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint iw = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint n_ih = gl_GlobalInvocationID.z;");
                shader_generator.addLogicSnippet("    uint n = n_ih / pc.data[1];");
                shader_generator.addLogicSnippet("    uint ih = n_ih % pc.data[1];");
                shader_generator.addLogicSnippet("    if (ic >= pc.data[3] || iw >= pc.data[2] || ih >= pc.data[1] || n >= pc.data[0]) return;");
            }
            else if (primary_pipeline == Compute_Pipeline::MAXPOOL2D_BACKWARD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint iw = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint n_ih = gl_GlobalInvocationID.z;");
                shader_generator.addLogicSnippet("    uint n = n_ih / pc.data[1];");
                shader_generator.addLogicSnippet("    uint ih = n_ih % pc.data[1];");
                shader_generator.addLogicSnippet("    if (c >= pc.data[3] || iw >= pc.data[2] || ih >= pc.data[1] || n >= pc.data[0]) return;");
            }
            else if (primary_pipeline == Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint n = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    if (c >= pc.data[3] || n >= pc.data[0]) return;");
            }
            else if (primary_pipeline == Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint iw = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint n_ih = gl_GlobalInvocationID.z;");
                shader_generator.addLogicSnippet("    uint n = n_ih / pc.data[1];");
                shader_generator.addLogicSnippet("    uint ih = n_ih % pc.data[1];");
                shader_generator.addLogicSnippet("    if (c >= pc.data[3] || iw >= pc.data[2] || ih >= pc.data[1] || n >= pc.data[0]) return;");
            }
            else
            {
                shader_generator.addLogicSnippet("    if (gl_GlobalInvocationID.x >= pc.data[0]) return;");
            }
        }
        else
        {
            shader_generator.addLogicSnippet("    if (global_id >= pc.data[0]) return;");
        }

        for (std::size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const Fused_Operation &operation = _node.fused_operations[operation_index];
            const Snippet_Metadata &metadata = shader_dictionary.getMetadata(operation.pipeline_id);

            std::vector<std::string> inputs;
            std::vector<bool> is_input_register_flags;

            for (std::uint32_t input_index : operation.input_buffer_indices)
            {
                if (register_map.contains(input_index))
                {
                    inputs.push_back(register_map[input_index]);
                    is_input_register_flags.push_back(true);
                }
                else
                {
                    inputs.push_back(std::format("buf_{}", input_index));
                    is_input_register_flags.push_back(false);
                }
            }

            std::vector<std::string> outputs;
            std::vector<bool> is_output_register_flags;

            for (std::size_t local_output_index = 0; local_output_index < operation.output_buffer_indices.size(); ++local_output_index)
            {
                std::uint32_t output_index = operation.output_buffer_indices[local_output_index];
                bool is_accumulator = std::find(metadata.accumulator_output_indices.begin(),
                                                metadata.accumulator_output_indices.end(),
                                                static_cast<std::uint32_t>(local_output_index)) != metadata.accumulator_output_indices.end();
                bool is_persistent = std::find(metadata.persistent_output_indices.begin(),
                                               metadata.persistent_output_indices.end(),
                                               static_cast<std::uint32_t>(local_output_index)) != metadata.persistent_output_indices.end();

                if (is_accumulator || is_persistent)
                {
                    outputs.push_back(std::format("buf_{}", output_index));
                    is_output_register_flags.push_back(false);
                }
                else
                {
                    std::string register_name = shader_generator.getUniqueVar("reg");
                    shader_generator.addLogicSnippet(std::format("    float {} = 0.0;", register_name));

                    register_map[output_index] = register_name;
                    outputs.push_back(register_name);
                    is_output_register_flags.push_back(true);
                }
            }

            const std::string &raw_template = (is_coop && metadata.is_cooperative_matrix_support && !metadata.cooperative_glsl_template.empty())
                                                  ? metadata.cooperative_glsl_template
                                                  : metadata.glsl_template;

            std::uint32_t push_constants_word_offset = operation.push_constants_offset / 4;
            std::string snippet = replacePlaceholders(raw_template,
                                                      inputs,
                                                      is_input_register_flags,
                                                      outputs,
                                                      is_output_register_flags,
                                                      operation.output_buffer_indices,
                                                      external_buffer_set,
                                                      push_constants_word_offset);

            auto remove_pattern = [](std::string &source_string, const std::string &prefix, const std::string &suffix)
            {
                std::size_t start_position = source_string.find(prefix);
                if (start_position != std::string::npos)
                {
                    std::size_t end_position = source_string.find(suffix, start_position);
                    if (end_position != std::string::npos)
                    {
                        source_string.erase(start_position, (end_position + suffix.length()) - start_position);
                    }
                }
            };

            auto remove_exact = [](std::string &source_string, const std::string &string_to_remove)
            {
                std::size_t position = 0;
                while ((position = source_string.find(string_to_remove, position)) != std::string::npos)
                {
                    source_string.erase(position, string_to_remove.length());
                }
            };

            if (operation_index == 0)
            {
                if (primary_operation_class == Operation_Class::ELEMENTWISE)
                {
                    if (snippet.starts_with("if (global_id >= pc.data[0]) return;"))
                    {
                        snippet = snippet.substr(37);
                    }
                }
                else if (primary_operation_class == Operation_Class::MATRIX_2D)
                {
                    if (!is_coop || !shader_dictionary.getMetadata(primary_pipeline).is_cooperative_matrix_support)
                    {
                        remove_exact(snippet, "uint r = gl_GlobalInvocationID.y;");
                        remove_exact(snippet, "uint c = gl_GlobalInvocationID.x;");
                        remove_exact(snippet, "uint lc = gl_LocalInvocationID.x;");
                        remove_exact(snippet, "uint lr = gl_LocalInvocationID.y;");
                        remove_pattern(snippet, "if (r >= ", " return;");
                    }
                }
                else if (primary_operation_class == Operation_Class::TENSOR_3D)
                {
                    remove_exact(snippet, "uint oc = gl_GlobalInvocationID.x;");
                    remove_exact(snippet, "uint ic = gl_GlobalInvocationID.x;");
                    remove_exact(snippet, "uint c = gl_GlobalInvocationID.x;");
                    remove_exact(snippet, "uint ow = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint iw = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint n = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint n_oh = gl_GlobalInvocationID.z;");
                    remove_exact(snippet, "uint n_ih = gl_GlobalInvocationID.z;");

                    remove_pattern(snippet, "uint n = n_oh /", ";");
                    remove_pattern(snippet, "uint oh = n_oh %", ";");
                    remove_pattern(snippet, "uint n = n_ih /", ";");
                    remove_pattern(snippet, "uint ih = n_ih %", ";");
                    remove_pattern(snippet, "if (oc >= ", " return;");
                    remove_pattern(snippet, "if (c >= ", " return;");
                    remove_pattern(snippet, "if (ic >= ", " return;");
                }
                else if (primary_operation_class == Operation_Class::STANDALONE)
                {
                    remove_exact(snippet, "uint r = gl_WorkGroupID.x;");
                    remove_exact(snippet, "uint c = gl_WorkGroupID.x;");
                    remove_exact(snippet, "uint d = gl_WorkGroupID.x;");
                    remove_pattern(snippet, "if (c >= ", " return;");
                    remove_pattern(snippet, "if (r >= ", " return;");
                }
            }

            shader_generator.addLogicSnippet(std::format("    {}", snippet));

            if (operation_index == 0 && !metadata.index_expression.empty() && (!is_coop || !metadata.is_cooperative_matrix_support))
            {
                std::string resolved_expression = metadata.index_expression;
                for (int push_constant_index = 31; push_constant_index >= 0; --push_constant_index)
                {
                    std::string token = std::format("{{pc_{}}}", push_constant_index);
                    std::string replacement = std::format("pc.data[{}]", push_constants_word_offset + push_constant_index);
                    std::size_t position = 0;
                    while ((position = resolved_expression.find(token, position)) != std::string::npos)
                    {
                        resolved_expression.replace(position, token.length(), replacement);
                        position += replacement.length();
                    }
                }

                if (has_matrix_tiles)
                {
                    shader_generator.addLogicSnippet("    if (r >= M || c >= N) return;");
                }
                else if (primary_operation_class == Operation_Class::MATRIX_2D)
                {
                    shader_generator.addLogicSnippet(std::format("    if (r >= pc.data[{}] || c >= pc.data[{}]) return;",
                                                                 push_constants_word_offset,
                                                                 push_constants_word_offset + 1));
                }

                shader_generator.addLogicSnippet(std::format("    global_id = {};", resolved_expression));
            }
        }

        std::string glsl_code = shader_generator.build();

        std::string node_chain_name = "[";
        for (std::size_t op_idx = 0; op_idx < _node.fused_operations.size(); ++op_idx)
        {
            node_chain_name += std::string(magic_enum::enum_name(_node.fused_operations[op_idx].pipeline_id));
            if (op_idx + 1 < _node.fused_operations.size())
            {
                node_chain_name += " -> ";
            }
        }
        node_chain_name += "]";

        if (std::find(printed_terminal_shader_chains.begin(), printed_terminal_shader_chains.end(), node_chain_name) == printed_terminal_shader_chains.end())
        {
            printed_terminal_shader_chains.push_back(node_chain_name);
            Logger::logMessage(std::format("Graph_Executor::generateFusedGlsl: Generated GLSL for node {}:\n\n===============\n{}\n\n===============",
                                           node_chain_name, glsl_code),
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION | Log_Feature::OPERATOR_FUSION);
        }
        return glsl_code;
    }

    const Vulkan_Context &getContext() const noexcept
    {
        return context;
    }

    const Vulkan_Network &getNetwork() const noexcept
    {
        return network;
    }

    Pipeline_Cache_Manager &getPipelineCacheManager() noexcept
    {
        return pipeline_cache_manager;
    }

    const Pipeline_Cache_Manager &getPipelineCacheManager() const noexcept
    {
        return pipeline_cache_manager;
    }

    const Shader_Dictionary &getShaderDictionary() const noexcept
    {
        return shader_dictionary;
    }

    VkCommandBuffer getCommandBuffer(std::uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Graph_Executor::getCommandBuffer: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return VK_NULL_HANDLE;
        }
        return command_buffers[_frame_index];
    }

    VkDescriptorPool getDescriptorPool(std::uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Graph_Executor::getDescriptorPool: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return VK_NULL_HANDLE;
        }
        return descriptor_pools[_frame_index];
    }

    void invalidate()
    {
        Logger::logMessage("Graph_Executor::invalidate: Invalidating persistent descriptor caches and resetting pools",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);
        VkDevice device = context.getDevice();
        for (std::uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (descriptor_pools[i] != VK_NULL_HANDLE)
            {
                vkResetDescriptorPool(device, descriptor_pools[i], 0);
            }
            persistent_descriptor_caches[i].clear();
            fallback_descriptor_caches[i].clear();
        }
    }

    void resetFrameState(std::uint32_t _frame_index)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Graph_Executor::resetFrameState: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }
    }

    void compileAndExecute(const Compute_Graph &_graph,
                           const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                           std::uint32_t _frame_index,
                           VkFence _external_fence = VK_NULL_HANDLE)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: frame_index out of bounds ({})", _frame_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }

        const std::vector<Compute_Node> &nodes = _graph.getNodes();
        if (nodes.empty() && _transfer_tasks.empty())
        {
            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Both compute nodes and transfer tasks are empty for frame {}", _frame_index),
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
        }

        Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Frame {}, nodes={}, transfer_tasks={}", _frame_index, nodes.size(), _transfer_tasks.size()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);

        VkDevice device = context.getDevice();
        VkCommandBuffer command_buffer = command_buffers[_frame_index];

        context.resetFrameFence(_frame_index);

        if (vkResetCommandBuffer(command_buffer, 0) != VK_SUCCESS)
        {
            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Failed to reset command buffer for frame {}", _frame_index),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to reset command buffer");
        }

        VkCommandBufferBeginInfo begin_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};

        if (vkBeginCommandBuffer(command_buffer, &begin_information) != VK_SUCCESS)
        {
            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Failed to begin command buffer for frame {}", _frame_index),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to begin command buffer");
        }

        if (!_transfer_tasks.empty())
        {
            for (const auto &task : _transfer_tasks)
            {
                if (task.size == 0 || task.source_buffer == VK_NULL_HANDLE || task.destination_buffer == VK_NULL_HANDLE)
                {
                    continue;
                }

                VkBufferCopy copy_region{
                    .srcOffset = task.source_offset,
                    .dstOffset = task.destination_offset,
                    .size = task.size};

                vkCmdCopyBuffer(command_buffer, task.source_buffer, task.destination_buffer, 1, &copy_region);
            }

            VkMemoryBarrier transfer_memory_barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};

            vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &transfer_memory_barrier,
                0, nullptr,
                0, nullptr);
        }

        if (persistent_descriptor_caches[_frame_index].size() < nodes.size())
        {
            persistent_descriptor_caches[_frame_index].resize(nodes.size());
        }

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const Compute_Node &node = nodes[i];

            for (const auto &vector_ptr : node.buffers)
            {
                if (vector_ptr)
                {
                    vector_ptr->markAsUsedInFrame(_frame_index);
                }
                else
                {
                    Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Null gpu::vector buffer encountered in compute node {}", i),
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::DISPATCH_EXECUTION);
                }
            }

            bool is_fused_successful = false;
            if (node.is_fused && node.fused_operations.size() > 1)
            {
                try
                {
                    VkPipeline target_pipeline = node.cached_pipeline;
                    if (target_pipeline == VK_NULL_HANDLE)
                    {
                        std::string glsl_code = node.fused_glsl_code.empty() ? generateFusedGlsl(node) : node.fused_glsl_code;
                        target_pipeline = pipeline_cache_manager.getOrCreatePipeline(glsl_code);
                    }

                    if (target_pipeline != VK_NULL_HANDLE)
                    {
                        const std::vector<std::uint32_t> &external_indices = !node.cached_external_buffer_indices.empty()
                                                                                 ? node.cached_external_buffer_indices
                                                                                 : (getExternalBufferIndices(node, shared_external_buffer_indices), shared_external_buffer_indices);

                        shared_fused_buffers.clear();
                        for (std::uint32_t buffer_index : external_indices)
                        {
                            if (buffer_index < node.buffers.size())
                            {
                                shared_fused_buffers.push_back(node.buffers[buffer_index]);
                            }
                        }

                        auto &entry = persistent_descriptor_caches[_frame_index][i];
                        if (entry.descriptor_set == VK_NULL_HANDLE)
                        {
                            VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
                            VkDescriptorSetAllocateInfo allocate_information{
                                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                .pNext = nullptr,
                                .descriptorPool = descriptor_pools[_frame_index],
                                .descriptorSetCount = 1,
                                .pSetLayouts = &layout};

                            if (vkAllocateDescriptorSets(device, &allocate_information, &entry.descriptor_set) != VK_SUCCESS)
                            {
                                Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Failed to allocate persistent descriptor set for fused node {}", i),
                                                   Log_Level::LOG_ERROR,
                                                   true,
                                                   0,
                                                   Log_Feature::DISPATCH_EXECUTION);
                                throw std::runtime_error("Failed to allocate descriptor set");
                            }
                        }

                        if (!isBuffersMatching(entry, external_indices, shared_fused_buffers))
                        {
                            updateDescriptorSet(entry.descriptor_set, external_indices, shared_fused_buffers);
                            entry.bound_binding_indices = external_indices;
                            entry.bound_buffers.resize(shared_fused_buffers.size());
                            for (std::size_t j = 0; j < shared_fused_buffers.size(); ++j)
                            {
                                entry.bound_buffers[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getBuffer() : VK_NULL_HANDLE;
                            }
                        }

                        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);
                        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &entry.descriptor_set, 0, nullptr);

                        if (!node.push_constants_data.empty())
                        {
                            std::uint32_t push_constants_size = std::min<std::uint32_t>(static_cast<std::uint32_t>(node.push_constants_data.size()), 128);
                            vkCmdPushConstants(command_buffer, network.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, node.push_constants_data.data());
                        }

                        vkCmdDispatch(command_buffer, node.workgroup_count_x, node.workgroup_count_y, node.workgroup_count_z);
                        is_fused_successful = true;
                    }
                }
                catch (const std::exception &exception)
                {
                    Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Fused shader execution failed for node [{}] ({}), initiating fallback execution", i, exception.what()),
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::DISPATCH_EXECUTION | Log_Feature::OPERATOR_FUSION);
                    is_fused_successful = false;
                }
            }

            if (!is_fused_successful)
            {
                if (node.is_fused && node.fused_operations.size() > 1)
                {
                    executeFallbackNode(command_buffer, node, i, _frame_index);
                }
                else
                {
                    auto &entry = persistent_descriptor_caches[_frame_index][i];
                    if (entry.descriptor_set == VK_NULL_HANDLE)
                    {
                        VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
                        VkDescriptorSetAllocateInfo allocate_information{
                            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                            .pNext = nullptr,
                            .descriptorPool = descriptor_pools[_frame_index],
                            .descriptorSetCount = 1,
                            .pSetLayouts = &layout};

                        if (vkAllocateDescriptorSets(device, &allocate_information, &entry.descriptor_set) != VK_SUCCESS)
                        {
                            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Failed to allocate descriptor set for node {}", i),
                                               Log_Level::LOG_ERROR,
                                               true,
                                               0,
                                               Log_Feature::DISPATCH_EXECUTION);
                            throw std::runtime_error("Failed to allocate descriptor set");
                        }
                    }

                    if (!isBuffersMatching(entry, node.buffers))
                    {
                        updateDescriptorSet(entry.descriptor_set, node.buffers);
                        entry.bound_binding_indices.clear();
                        entry.bound_buffers.resize(node.buffers.size());
                        for (std::size_t j = 0; j < node.buffers.size(); ++j)
                        {
                            entry.bound_buffers[j] = node.buffers[j] ? node.buffers[j]->getBuffer() : VK_NULL_HANDLE;
                        }
                    }

                    VkPipeline target_pipeline = network.getPipeline(node.pipeline_id);

                    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &entry.descriptor_set, 0, nullptr);

                    if (!node.push_constants_data.empty())
                    {
                        std::uint32_t push_constants_size = std::min<std::uint32_t>(static_cast<std::uint32_t>(node.push_constants_data.size()), 128);
                        vkCmdPushConstants(command_buffer, network.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, node.push_constants_data.data());
                    }

                    vkCmdDispatch(command_buffer, node.workgroup_count_x, node.workgroup_count_y, node.workgroup_count_z);
                }
            }

            if (node.is_barrier_required_after)
            {
                insertBufferMemoryBarriers(command_buffer, node.buffers);
            }
        }

        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
        {
            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Failed to end command buffer for frame {}", _frame_index),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
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

        VkFence primary_fence = (_external_fence != VK_NULL_HANDLE) ? _external_fence : context.getFrameFence(_frame_index);

        if (vkQueueSubmit(context.getComputeQueue(), 1, &submit_information, primary_fence) != VK_SUCCESS)
        {
            Logger::logMessage(std::format("Graph_Executor::compileAndExecute: Failed to submit command buffer for frame {}", _frame_index),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to submit command buffer");
        }

        if (_external_fence != VK_NULL_HANDLE && _external_fence != context.getFrameFence(_frame_index))
        {
            vkQueueSubmit(context.getComputeQueue(), 0, nullptr, context.getFrameFence(_frame_index));
        }
    }

    void warmupPipelineCache(Compute_Graph &_graph)
    {
        auto &nodes = _graph.getNodes();
        if (nodes.empty())
        {
            return;
        }

        Logger::logMessage(std::format("Graph_Executor::warmupPipelineCache: Pre-compiling pipelines for {} nodes", nodes.size()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION | Log_Feature::DISPATCH_EXECUTION);

        for (Compute_Node &node : nodes)
        {
            if (node.is_fused && node.fused_operations.size() > 1)
            {
                try
                {
                    if (node.cached_external_buffer_indices.empty())
                    {
                        getExternalBufferIndices(node, node.cached_external_buffer_indices);
                    }
                    if (node.fused_glsl_code.empty())
                    {
                        node.fused_glsl_code = generateFusedGlsl(node);
                    }
                    node.cached_pipeline = pipeline_cache_manager.getOrCreatePipeline(node.fused_glsl_code);
                }
                catch (const std::exception &exception)
                {
                    Logger::logMessage(std::format("Graph_Executor::warmupPipelineCache: Pre-compilation failed ({}), fallback will be used at runtime", exception.what()),
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::SHADER_GENERATION | Log_Feature::DISPATCH_EXECUTION);
                }
            }
            else
            {
                network.getPipeline(node.pipeline_id);
            }
        }

        pipeline_cache_manager.freezeCache(true);
    }
};
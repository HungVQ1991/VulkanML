#include "engine/graph_executor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "engine/pipeline_cache_manager.h"
#include "engine/shader_dictionary.h"
#include "engine/shader_generator.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_network.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/training_profiler.h"

namespace
{
    size_t findMatchingBracket(const std::string &text, size_t open_bracket_pos) noexcept
    {
        if (open_bracket_pos >= text.size() || text[open_bracket_pos] != '[')
        {
            return std::string::npos;
        }
        size_t depth = 1;
        for (size_t pos = open_bracket_pos + 1; pos < text.size(); ++pos)
        {
            if (text[pos] == '[')
            {
                ++depth;
            }
            else if (text[pos] == ']')
            {
                --depth;
                if (depth == 0)
                {
                    return pos;
                }
            }
        }
        return std::string::npos;
    }

    void pushConstantsChecked(VkCommandBuffer command_buffer,
                              VkPipelineLayout layout,
                              const std::uint8_t *data,
                              size_t size_bytes,
                              const char *context_name)
    {
        if (!data || size_bytes == 0)
        {
            return;
        }
        size_t effective_size = size_bytes;
        if (effective_size > 128)
        {
            Logger::logMessage(Input_Format{"{}: Push constant size ({}) exceeds 128 bytes; clamping to 128",
                                            context_name, size_bytes},
                               Log_Level::LOG_WARNING, true, 0, Log_Feature::DISPATCH_EXECUTION);
            effective_size = 128;
        }
        if ((effective_size % 4) != 0)
        {
            Logger::logMessage(Input_Format{"{}: Push constant size ({}) is not 4-byte aligned; rounding down",
                                            context_name, effective_size},
                               Log_Level::LOG_WARNING, true, 0, Log_Feature::DISPATCH_EXECUTION);
            effective_size &= ~size_t{3};
        }
        if (effective_size == 0)
        {
            return;
        }
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(effective_size), data);
    }
}

bool Graph_Executor::isDynamicNode(const Compute_Node &) noexcept
    {
        return false;
    }

bool Graph_Executor::isBuffersMatching(const Persistent_Descriptor_Entry &_entry,
                                  const std::vector<uint32_t> &_binding_indices,
                                  const std::vector<std::shared_ptr<gpu::vector>> &_buffers)
    {
        if (_entry.descriptor_set == VK_NULL_HANDLE)
        {
            return false;
        }
        if (_entry.bound_binding_indices.size() != _binding_indices.size() ||
            _entry.bound_buffers.size() != _buffers.size() ||
            _entry.bound_vector_ids.size() != _buffers.size() ||
            _entry.bound_buffer_sizes.size() != _buffers.size() ||
            _entry.bound_weak_buffers.size() != _buffers.size())
        {
            return false;
        }
        for (size_t i = 0; i < _binding_indices.size(); ++i)
        {
            if (_entry.bound_binding_indices[i] != _binding_indices[i])
            {
                return false;
            }
        }
        for (size_t i = 0; i < _buffers.size(); ++i)
        {
            VkBuffer current_vulkan_buffer = _buffers[i] ? _buffers[i]->getBuffer() : VK_NULL_HANDLE;
            size_t current_id = _buffers[i] ? _buffers[i]->getId() : 0;
            VkDeviceSize current_size = _buffers[i] ? _buffers[i]->getSizeBytes() : 0;
            if (_entry.bound_buffers[i] != current_vulkan_buffer ||
                _entry.bound_vector_ids[i] != current_id ||
                _entry.bound_buffer_sizes[i] != current_size)
            {
                return false;
            }
            if (current_vulkan_buffer != VK_NULL_HANDLE &&
                (_entry.bound_weak_buffers[i].expired() || _entry.bound_weak_buffers[i].lock() != _buffers[i]))
            {
                return false;
            }
        }
        return true;
    }

bool Graph_Executor::isBuffersMatching(const Persistent_Descriptor_Entry &_entry,
                                  const std::vector<std::shared_ptr<gpu::vector>> &_buffers)
    {
        if (_entry.descriptor_set == VK_NULL_HANDLE)
        {
            return false;
        }
        if (!_entry.bound_binding_indices.empty() ||
            _entry.bound_buffers.size() != _buffers.size() ||
            _entry.bound_vector_ids.size() != _buffers.size() ||
            _entry.bound_buffer_sizes.size() != _buffers.size() ||
            _entry.bound_weak_buffers.size() != _buffers.size())
        {
            return false;
        }
        for (size_t i = 0; i < _buffers.size(); ++i)
        {
            VkBuffer current_vulkan_buffer = _buffers[i] ? _buffers[i]->getBuffer() : VK_NULL_HANDLE;
            size_t current_id = _buffers[i] ? _buffers[i]->getId() : 0;
            VkDeviceSize current_size = _buffers[i] ? _buffers[i]->getSizeBytes() : 0;
            if (_entry.bound_buffers[i] != current_vulkan_buffer ||
                _entry.bound_vector_ids[i] != current_id ||
                _entry.bound_buffer_sizes[i] != current_size)
            {
                return false;
            }
            if (current_vulkan_buffer != VK_NULL_HANDLE &&
                (_entry.bound_weak_buffers[i].expired() || _entry.bound_weak_buffers[i].lock() != _buffers[i]))
            {
                return false;
            }
        }
        return true;
    }

std::string Graph_Executor::replacePlaceholders(
    std::string _text,
    const std::vector<std::string>& _input_identifiers,
    const std::vector<bool>& _is_input_register_flags,
    const std::vector<std::string>& _output_identifiers,
    const std::vector<bool>& _is_output_register_flags,
    const std::vector<uint32_t>& _output_buffer_indices,
    const std::unordered_set<uint32_t>& _external_buffer_indices_set,
    uint32_t _push_constants_word_offset,
    const std::vector<std::shared_ptr<gpu::vector>>& _node_buffers)
{
    for (size_t i = 0; i < _input_identifiers.size(); ++i)
    {
        std::string token_prefix = std::format("{{in_{}}}[", i);
        std::string token_plain = std::format("{{in_{}}}", i);

        if (_is_input_register_flags[i])
        {
            size_t position = 0;
            while ((position = _text.find(token_prefix, position)) != std::string::npos)
            {
                const size_t open_bracket_pos = position + token_prefix.length() - 1;
                size_t end_position = findMatchingBracket(_text, open_bracket_pos);
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

        size_t position = 0;
        while ((position = _text.find(token_plain, position)) != std::string::npos)
        {
            _text.replace(position, token_plain.length(), _input_identifiers[i]);
            position += _input_identifiers[i].length();
        }
    }

    for (size_t i = 0; i < _output_identifiers.size(); ++i)
    {
        std::string token_prefix = std::format("{{out_{}}}[", i);
        std::string token_plain = std::format("{{out_{}}}", i);

        if (_is_output_register_flags[i])
        {
            uint32_t real_buffer_index = (i < _output_buffer_indices.size()) ? _output_buffer_indices[i] : 0;
            bool is_external = _external_buffer_indices_set.contains(real_buffer_index);
            std::string target_type = "float";
            if (real_buffer_index < _node_buffers.size() && _node_buffers[real_buffer_index])
            {
                target_type = std::string(getDataTypeGlslName(_node_buffers[real_buffer_index]->getDataType()));
            }

            size_t position = 0;
            while ((position = _text.find(token_prefix, position)) != std::string::npos)
            {
                const size_t open_bracket_pos = position + token_prefix.length() - 1;
                size_t end_position = findMatchingBracket(_text, open_bracket_pos);
                if (end_position != std::string::npos)
                {
                    std::string index_expression = _text.substr(position + token_prefix.length(), end_position - (position + token_prefix.length()));
                    _text.replace(position, end_position - position + 1, _output_identifiers[i]);
                    position += _output_identifiers[i].length();

                    size_t semi_pos = _text.find(';', position);
                    size_t eq_pos = _text.find('=', position);
                    bool is_assignment_lhs = false;

                    if (semi_pos != std::string::npos && eq_pos != std::string::npos && eq_pos < semi_pos)
                    {
                        bool next_is_eq = (eq_pos + 1 < _text.size() && _text[eq_pos + 1] == '=');
                        if (!next_is_eq)
                        {
                            bool valid_lhs_gap = true;
                            for (size_t k = position; k < eq_pos; ++k)
                            {
                                char ch = _text[k];
                                if (ch != ' ' && ch != '\t' && ch != '+' && ch != '-' && ch != '*' && ch != '/')
                                {
                                    valid_lhs_gap = false;
                                    break;
                                }
                            }
                            is_assignment_lhs = valid_lhs_gap;
                        }
                    }

                    if (is_assignment_lhs && target_type == "float16_t")
                    {
                        std::string expr = _text.substr(eq_pos + 1, semi_pos - (eq_pos + 1));
                        std::string wrapped_expr = std::format(" float16_t({})", expr);
                        _text.replace(eq_pos + 1, semi_pos - (eq_pos + 1), wrapped_expr);
                        semi_pos = _text.find(';', eq_pos);
                    }

                    if (is_assignment_lhs && is_external && semi_pos != std::string::npos)
                    {
                        std::string write_statement = std::format(" buf_{}[{}] = {}({});", real_buffer_index, index_expression, target_type, _output_identifiers[i]);
                        _text.insert(semi_pos + 1, write_statement);
                        position = semi_pos + 1 + write_statement.length();
                    }
                }
                else
                {
                    break;
                }
            }
        }

        size_t position = 0;
        while ((position = _text.find(token_plain, position)) != std::string::npos)
        {
            _text.replace(position, token_plain.length(), _output_identifiers[i]);
            position += _output_identifiers[i].length();
        }
    }

    for (int i = 31; i >= 0; --i)
    {
        std::string token = std::format("{{pc_{}}}", i);
        size_t position = 0;
        while ((position = _text.find(token, position)) != std::string::npos)
        {
            std::string replacement = std::format("pc.data[{}]", _push_constants_word_offset + i);
            _text.replace(position, token.length(), replacement);
            position += replacement.length();
        }
    }
    return _text;
}

std::vector<std::shared_ptr<gpu::vector>> Graph_Executor::getNodeWrittenBuffers(const Compute_Node &_node) const
    {
        std::vector<std::shared_ptr<gpu::vector>> written_buffers;
        std::unordered_set<VkBuffer> seen_handles;

        auto addUniqueBuffer = [&](const std::shared_ptr<gpu::vector> &buf)
        {
            if (!buf)
            {
                return;
            }
            VkBuffer handle = buf->getBuffer();
            if (handle != VK_NULL_HANDLE)
            {
                if (!seen_handles.insert(handle).second)
                {
                    return;
                }
            }
            written_buffers.push_back(buf);
        };

        if (_node.fusion_plan && _node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
        {
            const uint32_t dyn_idx = _node.fusion_plan->shared_dyn_params_binding;
            if (dyn_idx < _node.buffers.size())
            {
                addUniqueBuffer(_node.buffers[dyn_idx]);
            }
            for (const auto &lane : _node.fusion_plan->horizontal_lanes)
            {
                for (uint32_t idx : {lane.param_buf_idx, lane.m_buf_idx, lane.v_buf_idx, lane.param_fp16_buf_idx})
                {
                    if (idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[idx]);
                    }
                }
            }
            return written_buffers;
        }

        if (_node.is_fused)
        {
            for (uint32_t out_idx : _node.external_output_indices)
            {
                if (out_idx < _node.buffers.size())
                {
                    addUniqueBuffer(_node.buffers[out_idx]);
                }
            }
            if (!_node.fused_operations.empty())
            {
                for (uint32_t out_idx : _node.fused_operations.back().output_buffer_indices)
                {
                    if (out_idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[out_idx]);
                    }
                }
            }
            for (const auto &op : _node.fused_operations)
            {
                const auto &meta = shader_dictionary.getMetadata(op.pipeline_id);
                for (uint32_t p_idx : meta.persistent_output_indices)
                {
                    if (p_idx < op.output_buffer_indices.size())
                    {
                        uint32_t buf_idx = op.output_buffer_indices[p_idx];
                        if (buf_idx < _node.buffers.size())
                        {
                            addUniqueBuffer(_node.buffers[buf_idx]);
                        }
                    }
                }
                for (uint32_t a_idx : meta.accumulator_output_indices)
                {
                    if (a_idx < op.output_buffer_indices.size())
                    {
                        uint32_t buf_idx = op.output_buffer_indices[a_idx];
                        if (buf_idx < _node.buffers.size())
                        {
                            addUniqueBuffer(_node.buffers[buf_idx]);
                        }
                    }
                }
                if (op.pipeline_id == Compute_Pipeline::ADAM_UPDATE || op.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16)
                {
                    for (size_t idx : {0, 2, 3})
                    {
                        if (idx < op.input_buffer_indices.size())
                        {
                            uint32_t buf_idx = op.input_buffer_indices[idx];
                            if (buf_idx < _node.buffers.size())
                            {
                                addUniqueBuffer(_node.buffers[buf_idx]);
                            }
                        }
                    }
                }
                else if (op.pipeline_id == Compute_Pipeline::SGD_UPDATE || op.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16)
                {
                    if (!op.input_buffer_indices.empty())
                    {
                        uint32_t buf_idx = op.input_buffer_indices[0];
                        if (buf_idx < _node.buffers.size())
                        {
                            addUniqueBuffer(_node.buffers[buf_idx]);
                        }
                    }
                }
                else if (op.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE || op.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16)
                {
                    for (size_t idx : {0, 2})
                    {
                        if (idx < op.input_buffer_indices.size())
                        {
                            uint32_t buf_idx = op.input_buffer_indices[idx];
                            if (buf_idx < _node.buffers.size())
                            {
                                addUniqueBuffer(_node.buffers[buf_idx]);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            const auto &metadata = shader_dictionary.getMetadata(_node.pipeline_id);
            if (!metadata.write_indices.empty())
            {
                for (uint32_t idx : metadata.write_indices)
                {
                    if (idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[idx]);
                    }
                }
            }
            else
            {
                for (uint32_t i = metadata.input_count;
                     i < metadata.input_count + metadata.output_count && i < _node.buffers.size();
                     ++i)
                {
                    addUniqueBuffer(_node.buffers[i]);
                }
            }
            if (_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE || _node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16)
            {
                for (size_t idx : {0, 2, 3, 4})
                {
                    if (idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[idx]);
                    }
                }
            }
            else if (_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16)
            {
                if (!_node.buffers.empty())
                {
                    addUniqueBuffer(_node.buffers[0]);
                }
                for (size_t slot = 0; slot < 4; ++slot)
                {
                    size_t base = 1 + slot * 5;
                    for (size_t off : {size_t{0}, size_t{2}, size_t{3}, size_t{4}})
                    {
                        if (base + off < _node.buffers.size())
                        {
                            addUniqueBuffer(_node.buffers[base + off]);
                        }
                    }
                }
            }
            else if (_node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16 ||
                     _node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16)
            {
                for (size_t idx : {2, 3, 4, 5})
                {
                    if (idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[idx]);
                    }
                }
            }
            else if (_node.pipeline_id == Compute_Pipeline::SGD_UPDATE || _node.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16)
            {
                for (size_t idx : {size_t{0}, size_t{2}})
                {
                    if (idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[idx]);
                    }
                }
            }
            else if (_node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE || _node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16)
            {
                for (size_t idx : {0, 2, 3})
                {
                    if (idx < _node.buffers.size())
                    {
                        addUniqueBuffer(_node.buffers[idx]);
                    }
                }
            }
        }

        if (written_buffers.empty())
        {
            return _node.buffers;
        }
        return written_buffers;
    }

void Graph_Executor::insertBufferMemoryBarriers(VkCommandBuffer _command_buffer, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
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
                static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(),
                0, nullptr);
        }
    }

void Graph_Executor::insertComputeMemoryBarrier(VkCommandBuffer _command_buffer) const
    {
        VkMemoryBarrier memory_barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};

        vkCmdPipelineBarrier(
            _command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memory_barrier,
            0, nullptr,
            0, nullptr);
    }

void Graph_Executor::insertInterSubmissionComputeBarrier(VkCommandBuffer _command_buffer) const
    {
        VkMemoryBarrier memory_barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};

        vkCmdPipelineBarrier(
            _command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memory_barrier,
            0, nullptr,
            0, nullptr);
    }

VkDescriptorSet Graph_Executor::allocateDescriptorSet(uint32_t _frame_index)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            throw std::invalid_argument("Graph_Executor::allocateDescriptorSet: frame_index out of bounds");
        }

        VkDevice device = context.getDevice();
        VkDescriptorSetLayout layout = network.getDescriptorSetLayout();
        VkDescriptorPool active_pool = overflow_descriptor_pools[_frame_index].empty()
                                           ? descriptor_pools[_frame_index]
                                           : overflow_descriptor_pools[_frame_index].back();

        VkDescriptorSetAllocateInfo allocate_information{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = active_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout};

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkResult res = vkAllocateDescriptorSets(device, &allocate_information, &descriptor_set);
        if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL)
        {
            VkDescriptorPoolSize pool_sizes[] = {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4000u * DESCRIPTOR_BINDINGS_COUNT}};
            VkDescriptorPoolCreateInfo pool_create_information{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .maxSets = 4000,
                .poolSizeCount = 1,
                .pPoolSizes = pool_sizes};

            VkDescriptorPool new_pool = VK_NULL_HANDLE;
            if (vkCreateDescriptorPool(device, &pool_create_information, nullptr, &new_pool) != VK_SUCCESS)
            {
                throw std::runtime_error("Graph_Executor::allocateDescriptorSet: Failed to create overflow descriptor pool");
            }
            overflow_descriptor_pools[_frame_index].push_back(new_pool);
            allocate_information.descriptorPool = new_pool;
            res = vkAllocateDescriptorSets(device, &allocate_information, &descriptor_set);
        }

        if (res != VK_SUCCESS || descriptor_set == VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::allocateDescriptorSet: Failed to allocate descriptor set (VkResult={})", static_cast<int>(res)},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        return descriptor_set;
    }

void Graph_Executor::executeFallbackNode(VkCommandBuffer _command_buffer, const Compute_Node &_node, size_t _node_index, uint32_t _frame_index)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || _command_buffer == VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::executeFallbackNode: Invalid frame_index ({}) or null command buffer", _frame_index},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::invalid_argument("Graph_Executor::executeFallbackNode: Invalid frame_index or null command buffer");
        }

        if (_node_index >= fallback_descriptor_caches[_frame_index].size())
        {
            fallback_descriptor_caches[_frame_index].resize(_node_index + 1);
        }

        auto &node_fallback_entries = fallback_descriptor_caches[_frame_index][_node_index];
        if (node_fallback_entries.size() < _node.fused_operations.size())
        {
            node_fallback_entries.resize(_node.fused_operations.size());
        }

        std::vector<std::shared_ptr<gpu::vector>> output_buffers;
        for (size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const auto &operation = _node.fused_operations[operation_index];

            shared_fused_buffers.clear();
            for (uint32_t index : operation.input_buffer_indices)
            {
                if (index < _node.buffers.size())
                {
                    auto resolved = resolveNodeBuffer(_node, index, _frame_index);
                    if (resolved)
                    {
                        resolved->markAsUsedInFrame(_frame_index);
                    }
                    shared_fused_buffers.push_back(std::move(resolved));
                }
            }
            for (uint32_t index : operation.output_buffer_indices)
            {
                if (index < _node.buffers.size())
                {
                    auto resolved = resolveNodeBuffer(_node, index, _frame_index);
                    if (resolved)
                    {
                        resolved->markAsUsedInFrame(_frame_index);
                    }
                    shared_fused_buffers.push_back(std::move(resolved));
                }
            }

            VkPipeline static_pipeline = (operation.spec_constant_u32 == 0u)
                ? network.getPipeline(operation.pipeline_id)
                : network.getSpecializedPipeline(operation.pipeline_id, operation.spec_constant_u32);
            if (static_pipeline == VK_NULL_HANDLE)
            {
                std::string msg = std::format("Graph_Executor::executeFallbackNode: Null pipeline for {}", magic_enum::enum_name(operation.pipeline_id));
                Logger::logMessage(msg, Log_Level::LOG_ERROR, true, 0, Log_Feature::DISPATCH_EXECUTION);
                throw std::runtime_error(msg);
            }
            vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, static_pipeline);

            if (network.isPushDescriptorActive())
            {
                pushDescriptors(_command_buffer, shared_fused_buffers);
            }
            else
            {
                auto &entry = node_fallback_entries[operation_index];
                if (entry.descriptor_set == VK_NULL_HANDLE)
                {
                    entry.descriptor_set = allocateDescriptorSet(_frame_index);
                }

                if (!isBuffersMatching(entry, shared_fused_buffers))
                {
                    updateDescriptorSet(entry.descriptor_set, shared_fused_buffers);
                    entry.bound_binding_indices.clear();
                    entry.bound_buffers.resize(shared_fused_buffers.size());
                    entry.bound_vector_ids.resize(shared_fused_buffers.size());
                    entry.bound_buffer_sizes.resize(shared_fused_buffers.size());
                    entry.bound_weak_buffers.resize(shared_fused_buffers.size());
                    for (size_t j = 0; j < shared_fused_buffers.size(); ++j)
                    {
                        entry.bound_buffers[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getBuffer() : VK_NULL_HANDLE;
                        entry.bound_vector_ids[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getId() : 0;
                        entry.bound_buffer_sizes[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getSizeBytes() : 0;
                        entry.bound_weak_buffers[j] = shared_fused_buffers[j];
                    }
                }

                vkCmdBindDescriptorSets(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &entry.descriptor_set, 0, nullptr);
            }

            if (_node.fusion_plan &&
                _node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE &&
                operation_index < _node.fusion_plan->horizontal_lanes.size())
            {
                const auto &lane = _node.fusion_plan->horizontal_lanes[operation_index];
                struct Adam_Fallback_PC
                {
                    uint32_t total_size;
                    float beta1;
                    float beta2;
                    float eps;
                    float max_grad;
                    float weight_decay;
                } fallback_pc{};
                if (_node.push_constants_data.size() >= 16)
                {
                    std::memcpy(&fallback_pc.beta1, _node.push_constants_data.data(), 16);
                }
                fallback_pc.total_size = lane.element_count;
                fallback_pc.weight_decay = lane.weight_decay;
                if (operation.push_constants_offset + 12 <= _node.push_constants_data.size())
                {
                    std::memcpy(&fallback_pc.total_size, _node.push_constants_data.data() + operation.push_constants_offset, 4);
                    std::memcpy(&fallback_pc.weight_decay, _node.push_constants_data.data() + operation.push_constants_offset + 8, 4);
                }
                pushConstantsChecked(
                    _command_buffer,
                    network.getPipelineLayout(),
                    reinterpret_cast<const std::uint8_t *>(&fallback_pc),
                    sizeof(fallback_pc),
                    "Graph_Executor::executeFallbackNode(horizontal_adam)");
            }
            else
            {
                const uint32_t required_pc_bytes = network.getPipelinePushConstantSize(operation.pipeline_id);
                const size_t provided_pc_bytes =
                    (operation.push_constants_offset + operation.push_constants_size <= _node.push_constants_data.size())
                    ? operation.push_constants_size
                    : 0;
                if (required_pc_bytes > 0u && provided_pc_bytes < required_pc_bytes)
                {
                    std::string msg = std::format(
                        "Graph_Executor::executeFallbackNode: Push constant size mismatch for {} (provided={}B < required={}B)",
                        magic_enum::enum_name(operation.pipeline_id),
                        provided_pc_bytes,
                        required_pc_bytes);
                    Logger::logMessage(msg, Log_Level::LOG_ERROR, true, 0, Log_Feature::DISPATCH_EXECUTION);
                    throw std::runtime_error(msg);
                }
                if (operation.push_constants_size > 0)
                {
                    if (operation.push_constants_offset + operation.push_constants_size <= _node.push_constants_data.size())
                    {
                        pushConstantsChecked(
                            _command_buffer,
                            network.getPipelineLayout(),
                            _node.push_constants_data.data() + operation.push_constants_offset,
                            operation.push_constants_size,
                            "Graph_Executor::executeFallbackNode");
                    }
                    else
                    {
                        Logger::logMessage(Input_Format{"Graph_Executor::executeFallbackNode: Push constants range [{} + {}) exceeds buffer size {}",
                                                        operation.push_constants_offset,
                                                        operation.push_constants_size,
                                                        _node.push_constants_data.size()},
                                           Log_Level::LOG_WARNING, true, 0, Log_Feature::DISPATCH_EXECUTION);
                    }
                }
            }

            vkCmdDispatch(_command_buffer, operation.workgroup_count_x, operation.workgroup_count_y, operation.workgroup_count_z);

            if (operation_index < _node.fused_operations.size() - 1)
            {
                output_buffers.clear();
                for (uint32_t out_index : operation.output_buffer_indices)
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

void Graph_Executor::initializeResources()
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

        if (vkAllocateCommandBuffers(device, &allocate_information, command_buffers) != VK_SUCCESS ||
            vkAllocateCommandBuffers(device, &allocate_information, transfer_command_buffers) != VK_SUCCESS ||
            vkAllocateCommandBuffers(device, &allocate_information, static_command_buffers) != VK_SUCCESS ||
            vkAllocateCommandBuffers(device, &allocate_information, epilogue_command_buffers) != VK_SUCCESS)
        {
            Logger::logMessage("Graph_Executor::initializeResources: Failed to allocate command buffers",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to allocate command buffers");
        }

        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4000u * DESCRIPTOR_BINDINGS_COUNT}};

        VkDescriptorPoolCreateInfo pool_create_information{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = 4000,
            .poolSizeCount = 1,
            .pPoolSizes = pool_sizes};

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateDescriptorPool(device, &pool_create_information, nullptr, &descriptor_pools[i]) != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::initializeResources: Failed to create descriptor pool for frame {}", i},
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
                throw std::runtime_error("Failed to create descriptor pool");
            }
        }

        VkSemaphoreCreateInfo semaphore_create_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0};

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateSemaphore(device, &semaphore_create_info, nullptr, &external_fence_semaphores[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create external fence semaphore");
            }
            dynamic_optimizer_buffers[i] = std::make_shared<gpu::vector>(context);
            dynamic_optimizer_buffers[i]->allocateHostVisible(256);
            Adam_Dynamic_Params init_params{};
            dynamic_optimizer_buffers[i]->writeHostDirect(&init_params, sizeof(init_params));
        }

        dummy_null_buffer = std::make_shared<gpu::vector>(context);
        dummy_null_buffer->allocateMemoryByBytes(64, Data_Type::FLOAT32, 16);

        shared_fused_buffers.reserve(32);
        shared_external_buffer_indices.reserve(32);
    }

void Graph_Executor::cleanupResources() noexcept
    {
        VkDevice device = context.getDevice();
        if (device != VK_NULL_HANDLE)
        {
            VkResult idle_res = vkDeviceWaitIdle(device);
            if (idle_res != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::cleanupResources: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            }
        }
        VkCommandPool pool = context.getCommandPool();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (device != VK_NULL_HANDLE && external_fence_semaphores[i] != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device, external_fence_semaphores[i], nullptr);
                external_fence_semaphores[i] = VK_NULL_HANDLE;
            }
            if (device != VK_NULL_HANDLE && descriptor_pools[i] != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, descriptor_pools[i], nullptr);
                descriptor_pools[i] = VK_NULL_HANDLE;
            }
            for (VkDescriptorPool extra_pool : overflow_descriptor_pools[i])
            {
                if (device != VK_NULL_HANDLE && extra_pool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(device, extra_pool, nullptr);
                }
            }
            overflow_descriptor_pools[i].clear();
            dynamic_optimizer_param_pool[i].clear();
            dynamic_optimizer_pool_values[i].clear();

            if (device != VK_NULL_HANDLE && timestamp_query_pools[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(device, timestamp_query_pools[i], nullptr);
                timestamp_query_pools[i] = VK_NULL_HANDLE;
            }
            timestamp_query_capacities[i] = 0u;
            frame_timestamp_entries[i].clear();
            frame_barrier_count[i] = 0;
            has_pending_timestamps[i] = false;
            static_recorded_entries[i].clear();
            static_recorded_barrier_count[i] = 0;
            static_next_query_index[i] = 0u;
            static_baked_with_profiling[i] = false;

            if (device != VK_NULL_HANDLE && pool != VK_NULL_HANDLE)
            {
                if (command_buffers[i] != VK_NULL_HANDLE)
                {
                    vkFreeCommandBuffers(device, pool, 1, &command_buffers[i]);
                    command_buffers[i] = VK_NULL_HANDLE;
                }
                if (transfer_command_buffers[i] != VK_NULL_HANDLE)
                {
                    vkFreeCommandBuffers(device, pool, 1, &transfer_command_buffers[i]);
                    transfer_command_buffers[i] = VK_NULL_HANDLE;
                }
                if (static_command_buffers[i] != VK_NULL_HANDLE)
                {
                    vkFreeCommandBuffers(device, pool, 1, &static_command_buffers[i]);
                    static_command_buffers[i] = VK_NULL_HANDLE;
                }
                if (epilogue_command_buffers[i] != VK_NULL_HANDLE)
                {
                    vkFreeCommandBuffers(device, pool, 1, &epilogue_command_buffers[i]);
                    epilogue_command_buffers[i] = VK_NULL_HANDLE;
                }
            }
        }
        for (auto &buf : dynamic_optimizer_buffers)
        {
            buf.reset();
        }
        dummy_null_buffer.reset();
    }

void Graph_Executor::updateDescriptorSet(VkDescriptorSet _descriptor_set, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
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
        size_t count = _buffers.size();

        thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
        thread_local std::vector<VkWriteDescriptorSet> write_sets;
        buffer_infos.clear();
        write_sets.clear();
        buffer_infos.reserve(count);
        write_sets.reserve(count);

        for (size_t i = 0; i < count; ++i)
        {
            VkBuffer buf_handle = VK_NULL_HANDLE;
            VkDeviceSize buf_range = VK_WHOLE_SIZE;
            if (_buffers[i] && _buffers[i]->getBuffer() != VK_NULL_HANDLE)
            {
                buf_handle = _buffers[i]->getBuffer();
            }
            else if (dummy_null_buffer && dummy_null_buffer->getBuffer() != VK_NULL_HANDLE)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::updateDescriptorSet: Null buffer encountered at index {}, binding dummy buffer", i},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                buf_handle = dummy_null_buffer->getBuffer();
                buf_range = 64;
            }
            else
            {
                Logger::logMessage(Input_Format{"Graph_Executor::updateDescriptorSet: Null buffer encountered at index {}", i},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                continue;
            }

            buffer_infos.push_back(VkDescriptorBufferInfo{
                .buffer = buf_handle,
                .offset = 0,
                .range = buf_range});

            write_sets.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = _descriptor_set,
                .dstBinding = static_cast<uint32_t>(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr});
        }

        for (size_t i = 0; i < write_sets.size(); ++i)
        {
            write_sets[i].pBufferInfo = &buffer_infos[i];
        }

        if (!write_sets.empty())
        {
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(write_sets.size()), write_sets.data(), 0, nullptr);
        }
    }

void Graph_Executor::updateDescriptorSet(VkDescriptorSet _descriptor_set,
                             const std::vector<uint32_t> &_binding_indices,
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
        size_t count = _buffers.size();

        thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
        thread_local std::vector<VkWriteDescriptorSet> write_sets;
        buffer_infos.clear();
        write_sets.clear();
        buffer_infos.reserve(count);
        write_sets.reserve(count);

        for (size_t i = 0; i < count; ++i)
        {
            VkBuffer buf_handle = VK_NULL_HANDLE;
            VkDeviceSize buf_range = VK_WHOLE_SIZE;
            if (_buffers[i] && _buffers[i]->getBuffer() != VK_NULL_HANDLE)
            {
                buf_handle = _buffers[i]->getBuffer();
            }
            else if (dummy_null_buffer && dummy_null_buffer->getBuffer() != VK_NULL_HANDLE)
            {
                buf_handle = dummy_null_buffer->getBuffer();
                buf_range = 64;
            }
            else
            {
                continue;
            }

            buffer_infos.push_back(VkDescriptorBufferInfo{
                .buffer = buf_handle,
                .offset = 0,
                .range = buf_range});

            write_sets.push_back(VkWriteDescriptorSet{
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

        for (size_t i = 0; i < write_sets.size(); ++i)
        {
            write_sets[i].pBufferInfo = &buffer_infos[i];
        }

        if (!write_sets.empty())
        {
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(write_sets.size()), write_sets.data(), 0, nullptr);
        }
    }

void Graph_Executor::pushDescriptors(VkCommandBuffer _command_buffer, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
{
    if (_buffers.empty())
    {
        return;
    }

    size_t count = _buffers.size();
    thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
    thread_local std::vector<VkWriteDescriptorSet> write_sets;
    buffer_infos.clear();
    write_sets.clear();
    buffer_infos.reserve(count);
    write_sets.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        VkBuffer buf_handle = VK_NULL_HANDLE;
        VkDeviceSize buf_range = VK_WHOLE_SIZE;
        if (_buffers[i] && _buffers[i]->getBuffer() != VK_NULL_HANDLE)
        {
            buf_handle = _buffers[i]->getBuffer();
        }
        else if (dummy_null_buffer && dummy_null_buffer->getBuffer() != VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::pushDescriptors: Null buffer encountered at index {}, binding dummy buffer", i},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            buf_handle = dummy_null_buffer->getBuffer();
            buf_range = 64;
        }
        else
        {
            continue;
        }

        buffer_infos.push_back(VkDescriptorBufferInfo{
            .buffer = buf_handle,
            .offset = 0,
            .range = buf_range});

        write_sets.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = static_cast<uint32_t>(i),
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr});
    }

    for (size_t i = 0; i < write_sets.size(); ++i)
    {
        write_sets[i].pBufferInfo = &buffer_infos[i];
    }

    if (!write_sets.empty())
    {
        context.cmdPushDescriptorSet(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     network.getPipelineLayout(), 0,
                                     static_cast<uint32_t>(write_sets.size()),
                                     write_sets.data());
    }
}

void Graph_Executor::pushDescriptors(VkCommandBuffer _command_buffer,
                                     const std::vector<uint32_t> &_binding_indices,
                                     const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const
{
    if (_buffers.empty() || _binding_indices.size() != _buffers.size())
    {
        return;
    }

    size_t count = _buffers.size();
    thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
    thread_local std::vector<VkWriteDescriptorSet> write_sets;
    buffer_infos.clear();
    write_sets.clear();
    buffer_infos.reserve(count);
    write_sets.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        VkBuffer buf_handle = VK_NULL_HANDLE;
        VkDeviceSize buf_range = VK_WHOLE_SIZE;
        if (_buffers[i] && _buffers[i]->getBuffer() != VK_NULL_HANDLE)
        {
            buf_handle = _buffers[i]->getBuffer();
        }
        else if (dummy_null_buffer && dummy_null_buffer->getBuffer() != VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::pushDescriptors: Null buffer encountered at index {}, binding dummy buffer", i},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            buf_handle = dummy_null_buffer->getBuffer();
            buf_range = 64;
        }
        else
        {
            continue;
        }

        buffer_infos.push_back(VkDescriptorBufferInfo{
            .buffer = buf_handle,
            .offset = 0,
            .range = buf_range});

        write_sets.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = _binding_indices[i],
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr});
    }

    for (size_t i = 0; i < write_sets.size(); ++i)
    {
        write_sets[i].pBufferInfo = &buffer_infos[i];
    }

    if (!write_sets.empty())
    {
        context.cmdPushDescriptorSet(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     network.getPipelineLayout(), 0,
                                     static_cast<uint32_t>(write_sets.size()),
                                     write_sets.data());
    }
}

Graph_Executor::Graph_Executor(const Vulkan_Context &_context,
                   const Vulkan_Network &_network,
                   Pipeline_Cache_Manager &_pipeline_cache_manager,
                   const Shader_Dictionary &_shader_dictionary)
        : context(_context),
          network(_network),
          pipeline_cache_manager(_pipeline_cache_manager),
          shader_dictionary(_shader_dictionary)
    {
        try
        {
            initializeResources();
        }
        catch (...)
        {
            cleanupResources();
            throw;
        }
    }

Graph_Executor::~Graph_Executor()
    {
        Logger::logMessage("Graph_Executor::~Graph_Executor: Cleaning up Vulkan descriptor pools and command buffers",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
        cleanupResources();
    }

void Graph_Executor::updateDynamicOptimizerParams(float _lr, float _inv_bc1, float _inv_sqrt_bc2, float _inv_scale, uint32_t _frame_index)
    {
        if (_frame_index < MAX_FRAMES_IN_FLIGHT && dynamic_optimizer_buffers[_frame_index])
        {
            if (!context.isFrameReady(_frame_index))
            {
                context.prepareFrame(_frame_index);
                resetFrameState(_frame_index);
            }
            Adam_Dynamic_Params params{_lr, _inv_bc1, _inv_sqrt_bc2, _inv_scale, 0u};
            dynamic_optimizer_buffers[_frame_index]->writeHostDirect(&params, sizeof(params));
            if (dynamic_optimizer_pool_values[_frame_index].empty())
            {
                dynamic_optimizer_pool_values[_frame_index].push_back(params);
            }
            else
            {
                dynamic_optimizer_pool_values[_frame_index][0] = params;
            }
            dynamic_optimizer_pool_used[_frame_index] = std::max<size_t>(dynamic_optimizer_pool_used[_frame_index], 1);
        }
    }

std::shared_ptr<gpu::vector> Graph_Executor::allocateDynamicOptimizerParamsBuffer(float _lr, float _inv_bc1, float _inv_sqrt_bc2, float _inv_scale, uint32_t _frame_index)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            return nullptr;
        }
        if (!context.isFrameReady(_frame_index))
        {
            context.prepareFrame(_frame_index);
            resetFrameState(_frame_index);
        }
        else if (dynamic_pool_dirty[_frame_index])
        {
            resetFrameState(_frame_index);
        }

        size_t used = dynamic_optimizer_pool_used[_frame_index];
        for (size_t s = 0; s < used && s < dynamic_optimizer_pool_values[_frame_index].size(); ++s)
        {
            const auto &v = dynamic_optimizer_pool_values[_frame_index][s];
            if (v.learning_rate == _lr &&
                v.inv_bc1 == _inv_bc1 &&
                v.inv_sqrt_bc2 == _inv_sqrt_bc2 &&
                v.inv_scale == _inv_scale)
            {
                return (s == 0) ? dynamic_optimizer_buffers[_frame_index] : dynamic_optimizer_param_pool[_frame_index][s - 1];
            }
        }

        size_t slot_idx = dynamic_optimizer_pool_used[_frame_index]++;
        std::shared_ptr<gpu::vector> target_buf;
        if (slot_idx == 0)
        {
            target_buf = dynamic_optimizer_buffers[_frame_index];
        }
        else
        {
            size_t pool_idx = slot_idx - 1;
            while (dynamic_optimizer_param_pool[_frame_index].size() <= pool_idx)
            {
                auto extra_buf = std::make_shared<gpu::vector>(context);
                extra_buf->allocateHostVisible(256);
                dynamic_optimizer_param_pool[_frame_index].push_back(std::move(extra_buf));
            }
            target_buf = dynamic_optimizer_param_pool[_frame_index][pool_idx];
        }

        Adam_Dynamic_Params params{_lr, _inv_bc1, _inv_sqrt_bc2, _inv_scale, 0u};
        if (target_buf)
        {
            target_buf->writeHostDirect(&params, sizeof(params));
        }
        if (dynamic_optimizer_pool_values[_frame_index].size() <= slot_idx)
        {
            dynamic_optimizer_pool_values[_frame_index].resize(slot_idx + 1);
        }
        dynamic_optimizer_pool_values[_frame_index][slot_idx] = params;
        return target_buf;
    }

void Graph_Executor::setCommandBuffer(uint32_t _frame_index, VkCommandBuffer _command_buffer)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || command_buffers[_frame_index] == _command_buffer)
        {
            return;
        }
        VkDevice device = context.getDevice();
        if (device != VK_NULL_HANDLE)
        {
            VkResult idle_res = vkDeviceWaitIdle(device);
            if (idle_res != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::setCommandBuffer: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                                   Log_Level::LOG_WARNING, true, 0, Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            }
            if (command_buffers[_frame_index] != VK_NULL_HANDLE)
            {
                vkFreeCommandBuffers(device, context.getCommandPool(), 1, &command_buffers[_frame_index]);
            }
        }
        command_buffers[_frame_index] = _command_buffer;
    }

void Graph_Executor::setDescriptorPool(uint32_t _frame_index, VkDescriptorPool _pool)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || descriptor_pools[_frame_index] == _pool)
        {
            return;
        }
        VkDevice device = context.getDevice();
        if (device != VK_NULL_HANDLE)
        {
            VkResult idle_res = vkDeviceWaitIdle(device);
            if (idle_res != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::setDescriptorPool: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                                   Log_Level::LOG_WARNING, true, 0, Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            }
            if (descriptor_pools[_frame_index] != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, descriptor_pools[_frame_index], nullptr);
            }
            for (VkDescriptorPool extra_pool : overflow_descriptor_pools[_frame_index])
            {
                if (extra_pool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(device, extra_pool, nullptr);
                }
            }
            overflow_descriptor_pools[_frame_index].clear();
        }
        persistent_descriptor_caches[_frame_index].clear();
        fallback_descriptor_caches[_frame_index].clear();
        descriptor_pools[_frame_index] = _pool;
    }

void Graph_Executor::getExternalBufferIndices(const Compute_Node &_node, std::vector<uint32_t> &_output_indices) const
    {
        _output_indices.clear();

        if (_node.fusion_plan && _node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
        {
            for (uint32_t i = 0; i < static_cast<uint32_t>(_node.buffers.size()); ++i)
            {
                _output_indices.push_back(i);
            }
            return;
        }

        size_t max_buf_count = _node.buffers.size();
        for (const auto &op : _node.fused_operations)
        {
            for (uint32_t idx : op.input_buffer_indices)
            {
                max_buf_count = std::max<size_t>(max_buf_count, static_cast<size_t>(idx) + 1);
            }
            for (uint32_t idx : op.output_buffer_indices)
            {
                max_buf_count = std::max<size_t>(max_buf_count, static_cast<size_t>(idx) + 1);
            }
        }
        for (uint32_t idx : _node.external_output_indices)
        {
            max_buf_count = std::max<size_t>(max_buf_count, static_cast<size_t>(idx) + 1);
        }

        std::vector<uint8_t> is_written_by_earlier_op(max_buf_count, 0);
        std::vector<uint8_t> is_added_flags(max_buf_count, 0);

        for (size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const auto &operation = _node.fused_operations[operation_index];
            for (uint32_t input_index : operation.input_buffer_indices)
            {
                if (input_index < is_written_by_earlier_op.size())
                {
                    if (!is_written_by_earlier_op[input_index] && !is_added_flags[input_index])
                    {
                        _output_indices.push_back(input_index);
                        is_added_flags[input_index] = 1;
                    }
                }
            }

            for (uint32_t output_index : operation.output_buffer_indices)
            {
                if (output_index < is_written_by_earlier_op.size())
                {
                    is_written_by_earlier_op[output_index] = 1;
                }
            }
        }

        for (uint32_t external_index : _node.external_output_indices)
        {
            if (external_index < is_added_flags.size() && !is_added_flags[external_index])
            {
                _output_indices.push_back(external_index);
                is_added_flags[external_index] = 1;
            }
        }

        for (const auto &operation : _node.fused_operations)
        {
            const auto &metadata = shader_dictionary.getMetadata(operation.pipeline_id);
            for (uint32_t persistent_output_local_index : metadata.persistent_output_indices)
            {
                if (persistent_output_local_index < operation.output_buffer_indices.size())
                {
                    uint32_t persistent_buffer_index = operation.output_buffer_indices[persistent_output_local_index];
                    if (persistent_buffer_index < is_added_flags.size() && !is_added_flags[persistent_buffer_index])
                    {
                        _output_indices.push_back(persistent_buffer_index);
                        is_added_flags[persistent_buffer_index] = 1;
                    }
                }
            }

            for (uint32_t accum_output_local_index : metadata.accumulator_output_indices)
            {
                if (accum_output_local_index < operation.output_buffer_indices.size())
                {
                    uint32_t accum_buffer_index = operation.output_buffer_indices[accum_output_local_index];
                    if (accum_buffer_index < is_added_flags.size() && !is_added_flags[accum_buffer_index])
                    {
                        _output_indices.push_back(accum_buffer_index);
                        is_added_flags[accum_buffer_index] = 1;
                    }
                }
            }
        }

        if (!_node.fused_operations.empty())
        {
            for (uint32_t output_index : _node.fused_operations.back().output_buffer_indices)
            {
                if (output_index < is_added_flags.size() && !is_added_flags[output_index])
                {
                    _output_indices.push_back(output_index);
                    is_added_flags[output_index] = 1;
                }
            }
        }
        if (_node.fusion_plan)
        {
            for (uint32_t mat_idx : _node.fusion_plan->materialized_buffer_indices)
            {
                if (mat_idx < is_added_flags.size() && !is_added_flags[mat_idx])
                {
                    _output_indices.push_back(mat_idx);
                    is_added_flags[mat_idx] = 1;
                }
            }
        }
    }

std::string Graph_Executor::emitHorizontalAdamGlsl(const Compute_Node &_node) const
    {
        const auto &plan = *_node.fusion_plan;
        const size_t lanes = plan.horizontal_lanes.size();
        std::ostringstream os;
        os << "#version 450\n"
           << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
           << "#extension GL_EXT_shader_16bit_storage : require\n\n"
           << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n"
           << "layout(std430, set = 0, binding = 0) coherent restrict buffer DynamicParams {\n"
           << "    float lr;\n"
           << "    float inv_correction1;\n"
           << "    float inv_sqrt_correction2;\n"
           << "    float inv_scale;\n"
           << "    uint overflow_detected;\n"
           << "} dyn;\n\n";

        for (size_t k = 0; k < lanes; ++k)
        {
            const auto &l = plan.horizontal_lanes[k];
            os << std::format("layout(std430, set = 0, binding = {}) buffer ParamBuffer_{} {{ float p_{}[]; }};\n", l.param_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) buffer ParamBuffer4_{} {{ vec4 p4_{}[]; }};\n", l.param_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) readonly buffer GradBuffer_{} {{ float g_{}[]; }};\n", l.grad_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) readonly buffer GradBuffer4_{} {{ vec4 g4_{}[]; }};\n", l.grad_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) buffer MBuffer_{} {{ float m_{}[]; }};\n", l.m_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) buffer MBuffer4_{} {{ vec4 m4_{}[]; }};\n", l.m_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) buffer VBuffer_{} {{ float v_{}[]; }};\n", l.v_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) buffer VBuffer4_{} {{ vec4 v4_{}[]; }};\n", l.v_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) writeonly buffer Fp16ParamBuffer_{} {{ float16_t pf_{}[]; }};\n", l.param_fp16_buf_idx, k, k)
               << std::format("layout(std430, set = 0, binding = {}) writeonly buffer Fp16ParamBuffer4_{} {{ f16vec4 pf4_{}[]; }};\n\n", l.param_fp16_buf_idx, k, k);
        }

        os << "layout(push_constant) uniform PushConstants {\n"
           << "    float beta1;\n"
           << "    float beta2;\n"
           << "    float eps;\n"
           << "    float max_grad;\n"
           << "    uint num_lanes;\n";
        for (size_t k = 0; k < lanes; ++k)
        {
            os << std::format("    uint elem_count_{};\n    uint wg_end_{};\n    float wd_{};\n", k, k, k);
        }
        os << "} pc;\n\n"
           << "void main() {\n"
           << "    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;\n"
           << "    float lr = dyn.lr;\n"
           << "    float inv_c1 = dyn.inv_correction1;\n"
           << "    float inv_sqrt_c2 = dyn.inv_sqrt_correction2;\n"
           << "    float inv_scale = dyn.inv_scale;\n"
           << "    float one_minus_b1 = 1.0 - pc.beta1;\n"
           << "    float one_minus_b2 = 1.0 - pc.beta2;\n"
           << "    bool saw_overflow = false;\n";

        for (size_t k = 0; k < lanes; ++k)
        {
            os << std::format("    if (wg_id < pc.wg_end_{}) {{\n", k);
            if (k == 0)
            {
                os << "        uint vec_idx = wg_id * 256u + gl_LocalInvocationID.x;\n";
            }
            else
            {
                os << std::format("        uint vec_idx = (wg_id - pc.wg_end_{}) * 256u + gl_LocalInvocationID.x;\n", k - 1);
            }
            os << std::format("        uint n = pc.elem_count_{};\n", k)
               << std::format("        float wd = pc.wd_{};\n", k)
               << "        uint base = vec_idx * 4u;\n"
               << "        if (base >= n) return;\n"
               << "        if (base + 3u < n) {\n"
               << std::format("            vec4 g = g4_{}[vec_idx] * inv_scale;\n", k)
               << "            if (any(isnan(g)) || any(isinf(g))) {\n"
               << "                for (uint s = 0u; s < 4u; ++s) {\n"
               << "                    uint idx = base + s;\n"
               << std::format("                    float gs = g_{}[idx] * inv_scale;\n", k)
               << "                    if (isnan(gs) || isinf(gs)) { saw_overflow = true; continue; }\n"
               << "                    if (pc.max_grad > 0.0) gs = clamp(gs, -pc.max_grad, pc.max_grad);\n"
               << std::format("                    float mv = fma(pc.beta1, m_{}[idx], one_minus_b1 * gs);\n", k)
               << std::format("                    float vv = fma(pc.beta2, v_{}[idx], one_minus_b2 * (gs * gs));\n", k)
               << std::format("                    m_{}[idx] = mv; v_{}[idx] = vv;\n", k, k)
               << "                    float st = (mv * inv_c1) / (sqrt(vv) * inv_sqrt_c2 + pc.eps);\n"
               << std::format("                    float pv = p_{}[idx];\n", k)
               << "                    pv -= (wd > 0.0) ? (lr * fma(wd, pv, st)) : (lr * st);\n"
               << std::format("                    p_{}[idx] = pv;\n", k)
               << std::format("                    pf_{}[idx] = float16_t(clamp(pv, -65504.0, 65504.0));\n", k)
               << "                }\n"
               << "            } else {\n"
               << "                if (pc.max_grad > 0.0) g = clamp(g, vec4(-pc.max_grad), vec4(pc.max_grad));\n"
               << std::format("                vec4 mv = fma(vec4(pc.beta1), m4_{}[vec_idx], vec4(one_minus_b1) * g);\n", k)
               << std::format("                vec4 vv = fma(vec4(pc.beta2), v4_{}[vec_idx], vec4(one_minus_b2) * (g * g));\n", k)
               << std::format("                m4_{}[vec_idx] = mv; v4_{}[vec_idx] = vv;\n", k, k)
               << "                vec4 st = (mv * inv_c1) / (sqrt(vv) * inv_sqrt_c2 + vec4(pc.eps));\n"
               << std::format("                vec4 pv = p4_{}[vec_idx];\n", k)
               << "                pv -= (wd > 0.0) ? (lr * fma(vec4(wd), pv, st)) : (lr * st);\n"
               << std::format("                p4_{}[vec_idx] = pv;\n", k)
               << std::format("                pf4_{}[vec_idx] = f16vec4(clamp(pv, vec4(-65504.0), vec4(65504.0)));\n", k)
               << "            }\n"
               << "        } else {\n"
               << "            for (uint s = 0u; s < 4u && (base + s) < n; ++s) {\n"
               << "                uint idx = base + s;\n"
               << std::format("                float gs = g_{}[idx] * inv_scale;\n", k)
               << "                if (isnan(gs) || isinf(gs)) { saw_overflow = true; continue; }\n"
               << "                if (pc.max_grad > 0.0) gs = clamp(gs, -pc.max_grad, pc.max_grad);\n"
               << std::format("                float mv = fma(pc.beta1, m_{}[idx], one_minus_b1 * gs);\n", k)
               << std::format("                float vv = fma(pc.beta2, v_{}[idx], one_minus_b2 * (gs * gs));\n", k)
               << std::format("                m_{}[idx] = mv; v_{}[idx] = vv;\n", k, k)
               << "                float st = (mv * inv_c1) / (sqrt(vv) * inv_sqrt_c2 + pc.eps);\n"
               << std::format("                float pv = p_{}[idx];\n", k)
               << "                pv -= (wd > 0.0) ? (lr * fma(wd, pv, st)) : (lr * st);\n"
               << std::format("                p_{}[idx] = pv;\n", k)
               << std::format("                pf_{}[idx] = float16_t(clamp(pv, -65504.0, 65504.0));\n", k)
               << "            }\n"
               << "        }\n"
               << "        if (saw_overflow) atomicOr(dyn.overflow_detected, 1u);\n"
               << "        return;\n"
               << "    }\n";
        }
        os << "}\n";
        return os.str();
    }

static void emitPointwiseExprScalar(std::ostringstream &os,
                                    const Fused_Op_Step &step,
                                    const std::string &lhs_expr,
                                    const std::string &rhs_expr,
                                    const std::string &out_var,
                                    bool is_fp16)
    {
        const uint32_t pc_w = step.pc_offset_bytes / 4u;
        auto sanitize = [&](const std::string &v)
        {
            if (is_fp16)
            {
                os << std::format("    if (isnan({}) || isinf({})) {} = 0.0;\n", v, v, v);
                os << std::format("    {} = clamp({}, -65504.0, 65504.0);\n", v, v);
            }
        };
        switch (step.op_kind)
        {
        case Pointwise_Op_Kind::ADD:
            os << std::format("    float {} = {} + {};\n", out_var, lhs_expr, rhs_expr);
            sanitize(out_var);
            break;
        case Pointwise_Op_Kind::SUB:
            os << std::format("    float {} = {} - {};\n", out_var, lhs_expr, rhs_expr);
            sanitize(out_var);
            break;
        case Pointwise_Op_Kind::MUL:
            os << std::format("    float {} = {} * {};\n", out_var, lhs_expr, rhs_expr);
            sanitize(out_var);
            break;
        case Pointwise_Op_Kind::DIV:
            os << std::format("    float {}_d = {};\n", out_var, rhs_expr)
               << std::format("    float {} = (abs({}_d) > 1e-7) ? ({} / {}_d) : 0.0;\n", out_var, out_var, lhs_expr, out_var);
            sanitize(out_var);
            break;
        case Pointwise_Op_Kind::MUL_SCALAR:
            os << std::format("    float {} = {} * uintBitsToFloat(pc.data[{}]);\n", out_var, lhs_expr, pc_w + 1u);
            if (is_fp16)
            {
                if (step.pc_size_bytes >= 12)
                {
                    os << std::format("    if (pc.data[{}] == 0u) {{\n", pc_w + 2u)
                       << std::format("        if (isnan({}) || isinf({})) {} = 0.0;\n", out_var, out_var, out_var)
                       << std::format("        {} = clamp({}, -65504.0, 65504.0);\n", out_var, out_var)
                       << "    }\n";
                }
                else
                {
                    sanitize(out_var);
                }
            }
            break;
        case Pointwise_Op_Kind::RELU:
            os << std::format("    float {} = max(0.0, {});\n", out_var, lhs_expr);
            break;
        case Pointwise_Op_Kind::SIGMOID:
            os << std::format("    float {} = 1.0 / (1.0 + exp(-clamp({}, -85.0, 85.0)));\n", out_var, lhs_expr);
            break;
        case Pointwise_Op_Kind::SILU:
        {
            os << std::format("    float {}_s = 1.0 / (1.0 + exp(-clamp({}, -85.0, 85.0)));\n", out_var, lhs_expr)
               << std::format("    float {} = {} * {}_s;\n", out_var, lhs_expr, out_var);
            break;
        }
        case Pointwise_Op_Kind::GELU:
            os << std::format("    float {}_x = {};\n", out_var, lhs_expr)
               << std::format("    float {}_y = clamp(0.79788456 * ({}_x + 0.044715 * {}_x * {}_x * {}_x), -15.0, 15.0);\n", out_var, out_var, out_var, out_var, out_var)
               << std::format("    float {} = 0.5 * {}_x * (1.0 + tanh({}_y));\n", out_var, out_var, out_var);
            break;
        default:
            os << std::format("    float {} = {};\n", out_var, lhs_expr);
            break;
        }
    }

static void emitPointwiseExprVec4(std::ostringstream &os,
                                  const Fused_Op_Step &step,
                                  const std::string &lhs_expr,
                                  const std::string &rhs_expr,
                                  const std::string &out_var)
    {
        const uint32_t pc_w = step.pc_offset_bytes / 4u;
        auto sanitize4 = [&](const std::string &v)
        {
            os << std::format("    {} = clamp(mix(vec4(0.0), {}, lessThanEqual(abs({}), vec4(3.4e38))), vec4(-65504.0), vec4(65504.0));\n", v, v, v);
        };
        switch (step.op_kind)
        {
        case Pointwise_Op_Kind::ADD:
            os << std::format("    vec4 {} = {} + {};\n", out_var, lhs_expr, rhs_expr);
            sanitize4(out_var);
            break;
        case Pointwise_Op_Kind::SUB:
            os << std::format("    vec4 {} = {} - {};\n", out_var, lhs_expr, rhs_expr);
            sanitize4(out_var);
            break;
        case Pointwise_Op_Kind::MUL:
            os << std::format("    vec4 {} = {} * {};\n", out_var, lhs_expr, rhs_expr);
            sanitize4(out_var);
            break;
        case Pointwise_Op_Kind::DIV:
            os << std::format("    vec4 {}_d = {};\n", out_var, rhs_expr)
               << std::format("    bvec4 {}_nz = greaterThan(abs({}_d), vec4(1e-7));\n", out_var, out_var)
               << std::format("    vec4 {} = mix(vec4(0.0), {} / mix(vec4(1.0), {}_d, {}_nz), {}_nz);\n", out_var, lhs_expr, out_var, out_var, out_var);
            sanitize4(out_var);
            break;
        case Pointwise_Op_Kind::MUL_SCALAR:
            os << std::format("    vec4 {} = {} * uintBitsToFloat(pc.data[{}]);\n", out_var, lhs_expr, pc_w + 1u);
            if (step.pc_size_bytes >= 12)
            {
                os << std::format("    if (pc.data[{}] == 0u) {{\n", pc_w + 2u)
                   << std::format("        {} = clamp(mix(vec4(0.0), {}, lessThanEqual(abs({}), vec4(3.4e38))), vec4(-65504.0), vec4(65504.0));\n", out_var, out_var, out_var)
                   << "    }\n";
            }
            else
            {
                sanitize4(out_var);
            }
            break;
        case Pointwise_Op_Kind::RELU:
            os << std::format("    vec4 {} = max(vec4(0.0), {});\n", out_var, lhs_expr);
            break;
        case Pointwise_Op_Kind::SIGMOID:
            os << std::format("    vec4 {} = vec4(1.0) / (vec4(1.0) + exp(-clamp({}, vec4(-85.0), vec4(85.0))));\n", out_var, lhs_expr);
            break;
        case Pointwise_Op_Kind::SILU:
            os << std::format("    vec4 {}_s = vec4(1.0) / (vec4(1.0) + exp(-clamp({}, vec4(-85.0), vec4(85.0))));\n", out_var, lhs_expr)
               << std::format("    vec4 {} = {} * {}_s;\n", out_var, lhs_expr, out_var);
            break;
        case Pointwise_Op_Kind::GELU:
            os << std::format("    vec4 {}_x = {};\n", out_var, lhs_expr)
               << std::format("    vec4 {}_y = clamp(0.79788456 * ({}_x + 0.044715 * {}_x * {}_x * {}_x), vec4(-15.0), vec4(15.0));\n", out_var, out_var, out_var, out_var, out_var)
               << std::format("    vec4 {} = 0.5 * {}_x * (vec4(1.0) + tanh({}_y));\n", out_var, out_var, out_var);
            break;
        default:
            os << std::format("    vec4 {} = {};\n", out_var, lhs_expr);
            break;
        }
    }

static void emitFp16StoreExpr(std::ostringstream &os,
                              const std::string &indent,
                              const Fused_Op_Step &step,
                              const std::string &dst_expr,
                              const std::string &val_expr,
                              bool is_vec4)
    {
        if (step.op_kind == Pointwise_Op_Kind::MUL_SCALAR && step.pc_size_bytes >= 12)
        {
            const uint32_t pc_w = step.pc_offset_bytes / 4u;
            if (is_vec4)
            {
                os << std::format("{}if (pc.data[{}] != 0u) {{ {} = f16vec4({}); }} else {{ {} = f16vec4(clamp({}, vec4(-65504.0), vec4(65504.0))); }}\n",
                                  indent, pc_w + 2u, dst_expr, val_expr, dst_expr, val_expr);
            }
            else
            {
                os << std::format("{}if (pc.data[{}] != 0u) {{ {} = float16_t({}); }} else {{ {} = float16_t(clamp({}, -65504.0, 65504.0)); }}\n",
                                  indent, pc_w + 2u, dst_expr, val_expr, dst_expr, val_expr);
            }
        }
        else if (is_vec4)
        {
            os << std::format("{}{} = f16vec4(clamp({}, vec4(-65504.0), vec4(65504.0)));\n", indent, dst_expr, val_expr);
        }
        else
        {
            os << std::format("{}{} = float16_t(clamp({}, -65504.0, 65504.0));\n", indent, dst_expr, val_expr);
        }
    }

std::string Graph_Executor::emitPointwiseChainGlsl(const Compute_Node &_node) const
    {
        const auto &plan = *_node.fusion_plan;
        std::vector<uint32_t> external_indices;
        getExternalBufferIndices(_node, external_indices);
        std::unordered_set<uint32_t> ext_out_set(_node.external_output_indices.begin(), _node.external_output_indices.end());
        if (!_node.fused_operations.empty())
        {
            for (uint32_t o : _node.fused_operations.back().output_buffer_indices) ext_out_set.insert(o);
        }
        for (uint32_t m : plan.materialized_buffer_indices) ext_out_set.insert(m);

        std::unordered_set<uint32_t> rw_set;
        std::unordered_set<uint32_t> ro_set;
        for (const auto &s : plan.epilogue_steps)
        {
            for (uint32_t ib : s.input_buffer_indices)
            {
                if (ext_out_set.contains(ib)) rw_set.insert(ib);
                else ro_set.insert(ib);
            }
        }

        const std::string scalar_t = plan.is_fp16 ? "float16_t" : "float";
        std::ostringstream os;
        os << "#version 450\n";
        if (plan.is_fp16)
        {
            os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
               << "#extension GL_EXT_shader_16bit_storage : require\n";
        }
        os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";

        for (uint32_t bi : external_indices)
        {
            std::string qual = rw_set.contains(bi) ? "" : (ext_out_set.contains(bi) ? "writeonly " : "readonly ");
            os << std::format("layout(std430, binding = {}) {}buffer Buf_{} {{ {} b_{}[]; }};\n", bi, qual, bi, scalar_t, bi);
            if (plan.use_vec4)
            {
                os << std::format("layout(std430, binding = {}) {}buffer Buf4_{} {{ f16vec4 b4_{}[]; }};\n", bi, qual, bi, bi);
            }
        }
        const size_t pc_words = std::max<size_t>(1, (_node.push_constants_data.size() + 3) / 4);
        os << std::format("\nlayout(push_constant) uniform PushConstants {{ uint data[{}]; }} pc;\n\n", pc_words);
        os << "void main() {\n"
           << "    uint n = pc.data[0];\n";

        if (plan.use_vec4)
        {
            os << "    uint tid = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationID.x;\n"
               << "    uint base = tid * 4u;\n"
               << "    if (base >= n) return;\n"
               << "    if (base + 3u < n) {\n";
            std::unordered_map<uint32_t, std::string> reg4;
            for (size_t i = 0; i < plan.epilogue_steps.size(); ++i)
            {
                const auto &s = plan.epilogue_steps[i];
                auto loadLhs = [&](uint32_t bi) -> std::string
                {
                    if (auto it = reg4.find(bi); it != reg4.end()) return it->second;
                    return std::format("vec4(b4_{}[tid])", bi);
                };
                std::string lhs = s.input_buffer_indices.empty() ? "vec4(0.0)" : loadLhs(s.input_buffer_indices[0]);
                std::string rhs = (s.input_buffer_indices.size() >= 2) ? loadLhs(s.input_buffer_indices[1]) : "";
                std::string outv = std::format("r4_{}", i);
                emitPointwiseExprVec4(os, s, lhs, rhs, outv);
                reg4[s.output_buffer_index] = outv;
                if (ext_out_set.contains(s.output_buffer_index))
                {
                    emitFp16StoreExpr(os, "        ", s, std::format("b4_{}[tid]", s.output_buffer_index), outv, true);
                }
            }
            os << "    } else {\n"
               << "        for (uint tail = 0u; tail < 4u && (base + tail) < n; ++tail) {\n"
               << "            uint idx = base + tail;\n";
            std::unordered_map<uint32_t, std::string> reg1;
            for (size_t i = 0; i < plan.epilogue_steps.size(); ++i)
            {
                const auto &s = plan.epilogue_steps[i];
                auto load1 = [&](uint32_t bi) -> std::string
                {
                    if (auto it = reg1.find(bi); it != reg1.end()) return it->second;
                    return std::format("float(b_{}[idx])", bi);
                };
                std::string lhs = s.input_buffer_indices.empty() ? "0.0" : load1(s.input_buffer_indices[0]);
                std::string rhs = (s.input_buffer_indices.size() >= 2) ? load1(s.input_buffer_indices[1]) : "";
                std::string outv = std::format("rt_{}", i);
                emitPointwiseExprScalar(os, s, lhs, rhs, outv, true);
                reg1[s.output_buffer_index] = outv;
                if (ext_out_set.contains(s.output_buffer_index))
                {
                    emitFp16StoreExpr(os, "            ", s, std::format("b_{}[idx]", s.output_buffer_index), outv, false);
                }
            }
            os << "        }\n    }\n";
        }
        else
        {
            os << "    uint idx = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationID.x;\n"
               << "    if (idx >= n) return;\n";
            std::unordered_map<uint32_t, std::string> reg1;
            for (size_t i = 0; i < plan.epilogue_steps.size(); ++i)
            {
                const auto &s = plan.epilogue_steps[i];
                auto loadLhs = [&](uint32_t bi) -> std::string
                {
                    if (auto it = reg1.find(bi); it != reg1.end()) return it->second;
                    return plan.is_fp16 ? std::format("float(b_{}[idx])", bi) : std::format("b_{}[idx]", bi);
                };
                auto loadRhs = [&](uint32_t bi) -> std::string
                {
                    if (auto it = reg1.find(bi); it != reg1.end()) return it->second;
                    std::string idx_expr = (s.rhs_index_mode == Index_Map_Kind::BROADCAST_ROW_1D)
                        ? std::format("(idx % max(pc.data[{}], 1u))", s.pc_offset_bytes / 4u + 1u)
                        : "idx";
                    return plan.is_fp16 ? std::format("float(b_{}[{}])", bi, idx_expr) : std::format("b_{}[{}]", bi, idx_expr);
                };
                std::string lhs = s.input_buffer_indices.empty() ? "0.0" : loadLhs(s.input_buffer_indices[0]);
                std::string rhs = (s.input_buffer_indices.size() >= 2) ? loadRhs(s.input_buffer_indices[1]) : "";
                std::string outv = std::format("rs_{}", i);
                emitPointwiseExprScalar(os, s, lhs, rhs, outv, plan.is_fp16);
                reg1[s.output_buffer_index] = outv;
                if (ext_out_set.contains(s.output_buffer_index))
                {
                    if (plan.is_fp16)
                    {
                        emitFp16StoreExpr(os, "    ", s, std::format("b_{}[idx]", s.output_buffer_index), outv, false);
                    }
                    else
                    {
                        os << std::format("    b_{}[idx] = {};\n", s.output_buffer_index, outv);
                    }
                }
            }
        }
        os << "}\n";
        return os.str();
    }

std::string Graph_Executor::emitRowReductionGlsl(const Compute_Node &_node) const
    {
        const auto &plan = *_node.fusion_plan;
        const auto &anchor_op = *std::find_if(_node.fused_operations.begin(), _node.fused_operations.end(),
            [](const Fused_Operation &o) {
                return o.pipeline_id == Compute_Pipeline::RMSNORM_FP16 || o.pipeline_id == Compute_Pipeline::RMSNORM;
            });
        const uint32_t rm_pc_w = anchor_op.push_constants_offset / 4u;
        const uint32_t rm_in_buf = anchor_op.input_buffer_indices[0];
        const uint32_t rm_gamma_buf = anchor_op.input_buffer_indices[1];
        const uint32_t rm_out_buf = anchor_op.output_buffer_indices[0];
        const uint32_t rm_inv_buf = anchor_op.output_buffer_indices[1];

        std::vector<uint32_t> external_indices;
        getExternalBufferIndices(_node, external_indices);
        std::unordered_set<uint32_t> ext_out_set(_node.external_output_indices.begin(), _node.external_output_indices.end());
        for (uint32_t o : _node.fused_operations.back().output_buffer_indices) ext_out_set.insert(o);
        for (uint32_t m : plan.materialized_buffer_indices) ext_out_set.insert(m);
        ext_out_set.insert(rm_inv_buf);

        std::unordered_set<uint32_t> rw_set;
        for (const auto &s : plan.prologue_steps) for (uint32_t ib : s.input_buffer_indices) if (ext_out_set.contains(ib)) rw_set.insert(ib);
        for (const auto &s : plan.epilogue_steps) for (uint32_t ib : s.input_buffer_indices) if (ext_out_set.contains(ib)) rw_set.insert(ib);
        if (ext_out_set.contains(rm_in_buf)) rw_set.insert(rm_in_buf);

        const std::string scalar_t = plan.is_fp16 ? "float16_t" : "float";
        std::ostringstream os;
        os << "#version 450\n";
        if (plan.is_fp16)
        {
            os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
               << "#extension GL_EXT_shader_16bit_storage : require\n";
        }
        os << "#extension GL_KHR_shader_subgroup_arithmetic : require\n\n"
           << "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n\n";

        for (uint32_t bi : external_indices)
        {
            const std::string btype = (bi == rm_inv_buf) ? "float" : scalar_t;
            std::string qual = rw_set.contains(bi) ? "" : (ext_out_set.contains(bi) ? "writeonly " : "readonly ");
            os << std::format("layout(std430, binding = {}) {}buffer Buf_{} {{ {} b_{}[]; }};\n", bi, qual, bi, btype, bi);
        }
        const size_t pc_words = std::max<size_t>(4, (_node.push_constants_data.size() + 3) / 4);
        os << std::format("\nlayout(push_constant) uniform PushConstants {{ uint data[{}]; }} pc;\n\n", pc_words)
           << "shared float s_wave_sums[16];\n"
           << "float blockReduceSum(float val) {\n"
           << "    val = subgroupAdd(val);\n"
           << "    if (gl_SubgroupInvocationID == 0u) s_wave_sums[gl_SubgroupID] = val;\n"
           << "    barrier();\n"
           << "    float total = 0.0;\n"
           << "    for (uint i = 0u; i < gl_NumSubgroups; ++i) total += s_wave_sums[i];\n"
           << "    return total;\n"
           << "}\n\n";

        auto emitColumnVal = [&](const std::vector<Fused_Op_Step> &steps,
                                 const std::string &idx_expr,
                                 const std::string &c_expr,
                                 std::unordered_map<uint32_t, std::string> &reg_map,
                                 const std::string &tag,
                                 bool materialize)
        {
            for (size_t si = 0; si < steps.size(); ++si)
            {
                const auto &s = steps[si];
                auto rdLhs = [&](uint32_t bi) -> std::string
                {
                    if (auto it = reg_map.find(bi); it != reg_map.end()) return it->second;
                    return plan.is_fp16 ? std::format("float(b_{}[{}])", bi, idx_expr) : std::format("b_{}[{}]", bi, idx_expr);
                };
                auto rdRhs = [&](uint32_t bi) -> std::string
                {
                    if (auto it = reg_map.find(bi); it != reg_map.end()) return it->second;
                    std::string ie = (s.rhs_index_mode == Index_Map_Kind::BROADCAST_ROW_1D) ? c_expr : idx_expr;
                    return plan.is_fp16 ? std::format("float(b_{}[{}])", bi, ie) : std::format("b_{}[{}]", bi, ie);
                };
                std::string lhs = s.input_buffer_indices.empty() ? "0.0" : rdLhs(s.input_buffer_indices[0]);
                std::string rhs = (s.input_buffer_indices.size() >= 2) ? rdRhs(s.input_buffer_indices[1]) : "";
                std::string outv = std::format("{}_{}", tag, si);
                emitPointwiseExprScalar(os, s, lhs, rhs, outv, plan.is_fp16);
                reg_map[s.output_buffer_index] = outv;
                if (materialize && ext_out_set.contains(s.output_buffer_index))
                {
                    if (plan.is_fp16)
                        emitFp16StoreExpr(os, "            ", s, std::format("b_{}[{}]", s.output_buffer_index, idx_expr), outv, false);
                    else
                        os << std::format("            b_{}[{}] = {};\n", s.output_buffer_index, idx_expr, outv);
                }
            }
        };

        const std::string fallback_load_in = plan.is_fp16
            ? std::format("float(b_{}[gidx])", rm_in_buf)
            : std::format("b_{}[gidx]", rm_in_buf);

        std::unordered_set<uint32_t> prologue_outputs;
        for (const auto &s : plan.prologue_steps)
        {
            prologue_outputs.insert(s.output_buffer_index);
        }
        bool epilogue_reads_extra_prologue = false;
        for (const auto &s : plan.epilogue_steps)
        {
            for (uint32_t ib : s.input_buffer_indices)
            {
                if (ib != rm_in_buf && prologue_outputs.contains(ib))
                {
                    epilogue_reads_extra_prologue = true;
                    break;
                }
            }
            if (epilogue_reads_extra_prologue)
            {
                break;
            }
        }

        os << "void main() {\n"
           << std::format("    uint batch_size = pc.data[{}];\n", rm_pc_w + 0u)
           << std::format("    uint dim = pc.data[{}];\n", rm_pc_w + 1u)
           << std::format("    float eps = uintBitsToFloat(pc.data[{}]);\n", rm_pc_w + 2u)
           << "    if (dim == 0u || batch_size == 0u) return;\n"
           << "    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;\n"
           << "    if (row >= batch_size) return;\n"
           << "    uint lid = gl_LocalInvocationID.x;\n"
           << "    uint row_off = row * dim;\n";
        if (!epilogue_reads_extra_prologue)
        {
            os << "    if (dim <= 256u) {\n"
               << "        float xv[4];\n"
               << "        float sum_sq = 0.0;\n"
               << "        for (uint k = 0u; k < 4u; ++k) {\n"
               << "            uint c = lid + k * 64u;\n"
               << "            xv[k] = 0.0;\n"
               << "            if (c < dim) {\n"
               << "                uint gidx = row_off + c;\n";
            if (plan.prologue_steps.empty())
            {
                os << std::format("                float pval = {};\n", fallback_load_in);
            }
            else
            {
                std::unordered_map<uint32_t, std::string> preg;
                emitColumnVal(plan.prologue_steps, "gidx", "c", preg, "pf", true);
                const std::string pval_expr = preg.contains(rm_in_buf) ? preg[rm_in_buf] : fallback_load_in;
                os << std::format("                float pval = {};\n", pval_expr);
            }
            os << "                xv[k] = pval;\n"
               << "                sum_sq = fma(pval, pval, sum_sq);\n"
               << "            }\n"
               << "        }\n"
               << "        float mean_sq = blockReduceSum(sum_sq) / float(dim);\n"
               << "        float inv_rms = inversesqrt(mean_sq + eps);\n"
               << std::format("        if (lid == 0u) b_{}[row] = inv_rms;\n", rm_inv_buf)
               << "        for (uint k = 0u; k < 4u; ++k) {\n"
               << "            uint c = lid + k * 64u;\n"
               << "            if (c < dim) {\n"
               << "                uint gidx = row_off + c;\n"
               << (plan.is_fp16 ? std::format("                float nval = xv[k] * inv_rms * float(b_{}[c]);\n", rm_gamma_buf)
                                  : std::format("                float nval = xv[k] * inv_rms * b_{}[c];\n", rm_gamma_buf));
            if (ext_out_set.contains(rm_out_buf) && !plan.epilogue_steps.empty())
            {
                os << (plan.is_fp16 ? std::format("                b_{}[gidx] = float16_t(clamp(nval, -65504.0, 65504.0));\n", rm_out_buf)
                                    : std::format("                b_{}[gidx] = nval;\n", rm_out_buf));
            }
            if (plan.epilogue_steps.empty())
            {
                os << (plan.is_fp16 ? std::format("                b_{}[gidx] = float16_t(clamp(nval, -65504.0, 65504.0));\n", rm_out_buf)
                                    : std::format("                b_{}[gidx] = nval;\n", rm_out_buf));
            }
            else
            {
                std::unordered_map<uint32_t, std::string> ereg;
                if (!plan.prologue_steps.empty())
                {
                    ereg[rm_in_buf] = "xv[k]";
                }
                ereg[rm_out_buf] = "nval";
                emitColumnVal(plan.epilogue_steps, "gidx", "c", ereg, "ef", true);
            }
            os << "            }\n"
               << "        }\n"
               << "    } else {\n";
        }
        os << "        float sum_sq = 0.0;\n"
           << "        for (uint c = lid; c < dim; c += 64u) {\n"
           << "            uint gidx = row_off + c;\n";
        if (plan.prologue_steps.empty())
        {
            os << std::format("            float pval = {};\n", fallback_load_in);
        }
        else
        {
            std::unordered_map<uint32_t, std::string> preg;
            emitColumnVal(plan.prologue_steps, "gidx", "c", preg, "pl", false);
            const std::string pval_expr = preg.contains(rm_in_buf) ? preg[rm_in_buf] : fallback_load_in;
            os << std::format("            float pval = {};\n", pval_expr);
        }
        os << "            sum_sq = fma(pval, pval, sum_sq);\n"
           << "        }\n"
           << "        float mean_sq = blockReduceSum(sum_sq) / float(dim);\n"
           << "        float inv_rms = inversesqrt(mean_sq + eps);\n"
           << std::format("        if (lid == 0u) b_{}[row] = inv_rms;\n", rm_inv_buf)
           << "        for (uint c = lid; c < dim; c += 64u) {\n"
           << "            uint gidx = row_off + c;\n";
        std::unordered_map<uint32_t, std::string> preg2;
        if (plan.prologue_steps.empty())
        {
            os << std::format("            float pval = {};\n", fallback_load_in);
        }
        else
        {
            emitColumnVal(plan.prologue_steps, "gidx", "c", preg2, "pl2", true);
            const std::string pval_expr = preg2.contains(rm_in_buf) ? preg2[rm_in_buf] : fallback_load_in;
            os << std::format("            float pval = {};\n", pval_expr);
        }
        os << (plan.is_fp16 ? std::format("            float nval = pval * inv_rms * float(b_{}[c]);\n", rm_gamma_buf)
                              : std::format("            float nval = pval * inv_rms * b_{}[c];\n", rm_gamma_buf));
        if (ext_out_set.contains(rm_out_buf) && !plan.epilogue_steps.empty())
        {
            os << (plan.is_fp16 ? std::format("            b_{}[gidx] = float16_t(clamp(nval, -65504.0, 65504.0));\n", rm_out_buf)
                                : std::format("            b_{}[gidx] = nval;\n", rm_out_buf));
        }
        if (plan.epilogue_steps.empty())
        {
            os << (plan.is_fp16 ? std::format("            b_{}[gidx] = float16_t(clamp(nval, -65504.0, 65504.0));\n", rm_out_buf)
                                : std::format("            b_{}[gidx] = nval;\n", rm_out_buf));
        }
        else
        {
            std::unordered_map<uint32_t, std::string> ereg2 = preg2;
            ereg2[rm_out_buf] = "nval";
            emitColumnVal(plan.epilogue_steps, "gidx", "c", ereg2, "el", true);
        }
        if (!epilogue_reads_extra_prologue)
        {
            os << "        }\n    }\n}\n";
        }
        else
        {
            os << "        }\n}\n";
        }
        return os.str();
    }

std::string Graph_Executor::emitCoopMatEpilogueGlsl(const Compute_Node &_node) const
    {
        const auto &plan = *_node.fusion_plan;
        const auto &anchor = _node.fused_operations[0];
        const Compute_Pipeline anchor_id = anchor.pipeline_id;
        const bool is_bwd_in = (anchor_id == Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16);
        const uint32_t in_a = anchor.input_buffer_indices[0];
        const uint32_t in_b = anchor.input_buffer_indices[1];
        const uint32_t in_bias = (anchor_id == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16) ? anchor.input_buffer_indices[2] : UINT32_MAX;
        const uint32_t gemm_out = anchor.output_buffer_indices[0];

        std::vector<uint32_t> external_indices;
        getExternalBufferIndices(_node, external_indices);
        std::unordered_set<uint32_t> ext_out_set(_node.external_output_indices.begin(), _node.external_output_indices.end());
        ext_out_set.insert(_node.fused_operations.back().output_buffer_indices[0]);
        for (uint32_t m : plan.materialized_buffer_indices) ext_out_set.insert(m);

        std::unordered_set<uint32_t> rw_set;
        for (const auto &s : plan.epilogue_steps) for (uint32_t ib : s.input_buffer_indices) if (ext_out_set.contains(ib)) rw_set.insert(ib);

        std::ostringstream os;
        os << "#version 450\n"
           << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
           << "#extension GL_EXT_shader_16bit_storage : require\n"
           << "#extension GL_KHR_memory_scope_semantics : require\n"
           << "#extension GL_KHR_cooperative_matrix : require\n"
           << "#extension GL_KHR_shader_subgroup_basic : require\n\n"
           << "#define TILE_M 64u\n#define TILE_N 64u\n#define TILE_K 32u\n#define SMEM_A_STRIDE 5u\n#define SMEM_B_STRIDE 9u\n\n"
           << "layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;\n\n";

        for (uint32_t bi : external_indices)
        {
            std::string qual = rw_set.contains(bi) ? "" : (ext_out_set.contains(bi) ? "writeonly " : "readonly ");
            os << std::format("layout(std430, binding = {}) {}buffer Buf_{} {{ float16_t b_{}[]; }};\n", bi, qual, bi, bi);
            if (bi == in_a || bi == in_b)
            {
                os << std::format("layout(std430, binding = {}) readonly buffer BufV4_{} {{ uvec4 bv4_{}[]; }};\n", bi, bi, bi);
            }
        }
        const size_t pc_words = std::max<size_t>(8, (_node.push_constants_data.size() + 3) / 4);
        os << std::format("\nlayout(push_constant) uniform PushConstants {{ uint data[{}]; }} pc;\n\n", pc_words)
           << "shared uvec4 smem_a[TILE_M * SMEM_A_STRIDE];\n"
           << (is_bwd_in ? "shared uvec4 smem_b[TILE_N * SMEM_A_STRIDE];\n"
                         : "shared uvec4 smem_b[TILE_K * SMEM_B_STRIDE];\n")
           << "shared float smem_c[4u * 256u];\n\n"
           << "uvec4 pack8Row(uint gr, uint gc, uint rows, uint cols, uint which) {\n"
           << "    if (gr >= rows || gc >= cols) return uvec4(0u);\n"
           << "    uint roff = gr * cols;\n"
           << "    float h[8];\n"
           << "    for (uint i = 0u; i < 8u; ++i) {\n"
           << "        if (gc + i < cols) {\n"
           << std::format("            h[i] = (which == 0u) ? float(b_{}[roff + gc + i]) : float(b_{}[roff + gc + i]);\n", in_a, in_b)
           << "        } else { h[i] = 0.0; }\n"
           << "    }\n"
           << "    return uvec4(packHalf2x16(vec2(h[0],h[1])), packHalf2x16(vec2(h[2],h[3])), packHalf2x16(vec2(h[4],h[5])), packHalf2x16(vec2(h[6],h[7])));\n"
           << "}\n\n"
           << "void main() {\n"
           << "    uint tile_row = gl_WorkGroupID.y * TILE_M;\n"
           << "    uint tile_col = gl_WorkGroupID.x * TILE_N;\n";
        if (anchor_id == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16)
        {
            os << "    uint M = pc.data[1]; uint K = pc.data[2]; uint N = pc.data[3];\n"
               << "    uint bcast_b = pc.data[5];\n";
        }
        else if (anchor_id == Compute_Pipeline::MATMUL_COOPMAT_FP16)
        {
            os << "    uint M = pc.data[1]; uint K = pc.data[2]; uint N = pc.data[3];\n";
        }
        else
        {
            os << "    uint M = pc.data[0]; uint N = pc.data[1]; uint K = pc.data[2];\n";
        }

        os << "    if (tile_row >= M || tile_col >= N) return;\n"
           << "    uint tid = gl_LocalInvocationID.x;\n"
           << "    uint wave_id = tid >> 5u; uint lane_id = tid & 31u;\n"
           << "    uint wave_m = wave_id >> 1u; uint wave_n = wave_id & 1u;\n"
           << "    uint wave_base = wave_id * 256u;\n"
           << "    uint base_r = tile_row + wave_m * 32u;\n"
           << "    uint base_c = tile_col + wave_n * 32u;\n"
           << "    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc00 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);\n"
           << "    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc01 = acc00;\n"
           << "    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc10 = acc00;\n"
           << "    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc11 = acc00;\n";

        if (anchor_id == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16)
        {
            os << "    if (bcast_b == 1u) {\n"
               << "        uint c_lane = lane_id & 15u; uint r_init = lane_id >> 4u;\n"
               << std::format("        float bv0 = (base_c + c_lane < N) ? float(b_{}[base_c + c_lane]) : 0.0;\n", in_bias)
               << "        for (uint r = r_init; r < 16u; r += 2u) smem_c[wave_base + r * 16u + c_lane] = bv0;\n"
               << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
               << "        coopMatLoad(acc00, smem_c, wave_base, 16u, gl_CooperativeMatrixLayoutRowMajor); acc10 = acc00;\n"
               << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
               << std::format("        float bv1 = (base_c + 16u + c_lane < N) ? float(b_{}[base_c + 16u + c_lane]) : 0.0;\n", in_bias)
               << "        for (uint r = r_init; r < 16u; r += 2u) smem_c[wave_base + r * 16u + c_lane] = bv1;\n"
               << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
               << "        coopMatLoad(acc01, smem_c, wave_base, 16u, gl_CooperativeMatrixLayoutRowMajor); acc11 = acc01;\n"
               << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
               << "    } else {\n";
            for (int st = 0; st < 4; ++st)
            {
                const char *accn[4] = {"acc00", "acc01", "acc10", "acc11"};
                uint32_t dr = (st >> 1) * 16u, dc = (st & 1) * 16u;
                os << "        for (uint i = 0u; i < 8u; ++i) {\n"
                   << "            uint idx = lane_id + i * 32u;\n"
                   << std::format("            uint gr = base_r + {}u + (idx >> 4u); uint gc = base_c + {}u + (idx & 15u);\n", dr, dc)
                   << std::format("            smem_c[wave_base + idx] = (gr < M && gc < N) ? float(b_{}[gr * N + gc]) : 0.0;\n", in_bias)
                   << "        }\n"
                   << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
                   << std::format("        coopMatLoad({}, smem_c, wave_base, 16u, gl_CooperativeMatrixLayoutRowMajor);\n", accn[st])
                   << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n";
            }
            os << "    }\n";
        }

        os << "    bool k_vec = ((K & 7u) == 0u);\n"
           << "    bool n_vec = ((N & 7u) == 0u);\n"
           << "    uint ax_r0 = tid >> 2u; uint ax_r1 = ax_r0 + 32u; uint ax_c4 = tid & 3u;\n"
           << "    uint bw_r0 = tid >> 3u; uint bw_r1 = bw_r0 + 16u; uint bw_c4 = tid & 7u;\n"
           << "    for (uint k = 0u; k < K; k += TILE_K) {\n"
           << "        if (k_vec) {\n"
           << "            uint ks4 = K >> 3u; uint k4 = k >> 3u; bool kin = (k4 + ax_c4 < ks4);\n"
           << "            uint gr0 = tile_row + ax_r0; uint gr1 = tile_row + ax_r1;\n"
           << std::format("            smem_a[ax_r0 * SMEM_A_STRIDE + ax_c4] = (gr0 < M && kin) ? bv4_{}[gr0 * ks4 + k4 + ax_c4] : uvec4(0u);\n", in_a)
           << std::format("            smem_a[ax_r1 * SMEM_A_STRIDE + ax_c4] = (gr1 < M && kin) ? bv4_{}[gr1 * ks4 + k4 + ax_c4] : uvec4(0u);\n", in_a)
           << "        } else {\n"
           << "            smem_a[ax_r0 * SMEM_A_STRIDE + ax_c4] = pack8Row(tile_row + ax_r0, k + ax_c4 * 8u, M, K, 0u);\n"
           << "            smem_a[ax_r1 * SMEM_A_STRIDE + ax_c4] = pack8Row(tile_row + ax_r1, k + ax_c4 * 8u, M, K, 0u);\n"
           << "        }\n";
        if (is_bwd_in)
        {
            os << "        if (k_vec) {\n"
               << "            uint ks4 = K >> 3u; uint k4 = k >> 3u; bool kin = (k4 + ax_c4 < ks4);\n"
               << "            uint gn0 = tile_col + ax_r0; uint gn1 = tile_col + ax_r1;\n"
               << std::format("            smem_b[ax_r0 * SMEM_A_STRIDE + ax_c4] = (gn0 < N && kin) ? bv4_{}[gn0 * ks4 + k4 + ax_c4] : uvec4(0u);\n", in_b)
               << std::format("            smem_b[ax_r1 * SMEM_A_STRIDE + ax_c4] = (gn1 < N && kin) ? bv4_{}[gn1 * ks4 + k4 + ax_c4] : uvec4(0u);\n", in_b)
               << "        } else {\n"
               << "            smem_b[ax_r0 * SMEM_A_STRIDE + ax_c4] = pack8Row(tile_col + ax_r0, k + ax_c4 * 8u, N, K, 1u);\n"
               << "            smem_b[ax_r1 * SMEM_A_STRIDE + ax_c4] = pack8Row(tile_col + ax_r1, k + ax_c4 * 8u, N, K, 1u);\n"
               << "        }\n"
               << "        barrier();\n"
               << "        for (uint k_sub = 0u; k_sub < 2u; ++k_sub) {\n"
               << "            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma0, ma1;\n"
               << "            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb0, mb1;\n"
               << "            coopMatLoad(ma0, smem_a, (wave_m * 32u)       * SMEM_A_STRIDE + k_sub * 2u, SMEM_A_STRIDE, gl_CooperativeMatrixLayoutRowMajor);\n"
               << "            coopMatLoad(ma1, smem_a, (wave_m * 32u + 16u) * SMEM_A_STRIDE + k_sub * 2u, SMEM_A_STRIDE, gl_CooperativeMatrixLayoutRowMajor);\n"
               << "            coopMatLoad(mb0, smem_b, (wave_n * 32u)       * SMEM_A_STRIDE + k_sub * 2u, SMEM_A_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);\n"
               << "            coopMatLoad(mb1, smem_b, (wave_n * 32u + 16u) * SMEM_A_STRIDE + k_sub * 2u, SMEM_A_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);\n"
               << "            acc00 = coopMatMulAdd(ma0, mb0, acc00); acc01 = coopMatMulAdd(ma0, mb1, acc01);\n"
               << "            acc10 = coopMatMulAdd(ma1, mb0, acc10); acc11 = coopMatMulAdd(ma1, mb1, acc11);\n"
               << "        }\n";
        }
        else
        {
            os << "        if (n_vec) {\n"
               << "            uint ns4 = N >> 3u; uint tn4 = tile_col >> 3u; bool nin = (tn4 + bw_c4 < ns4);\n"
               << "            uint gk0 = k + bw_r0; uint gk1 = k + bw_r1;\n"
               << std::format("            smem_b[bw_r0 * SMEM_B_STRIDE + bw_c4] = (gk0 < K && nin) ? bv4_{}[gk0 * ns4 + tn4 + bw_c4] : uvec4(0u);\n", in_b)
               << std::format("            smem_b[bw_r1 * SMEM_B_STRIDE + bw_c4] = (gk1 < K && nin) ? bv4_{}[gk1 * ns4 + tn4 + bw_c4] : uvec4(0u);\n", in_b)
               << "        } else {\n"
               << "            smem_b[bw_r0 * SMEM_B_STRIDE + bw_c4] = pack8Row(k + bw_r0, tile_col + bw_c4 * 8u, K, N, 1u);\n"
               << "            smem_b[bw_r1 * SMEM_B_STRIDE + bw_c4] = pack8Row(k + bw_r1, tile_col + bw_c4 * 8u, K, N, 1u);\n"
               << "        }\n"
               << "        barrier();\n"
               << "        for (uint k_sub = 0u; k_sub < 2u; ++k_sub) {\n"
               << "            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma0, ma1;\n"
               << "            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb0, mb1;\n"
               << "            coopMatLoad(ma0, smem_a, (wave_m * 32u)       * SMEM_A_STRIDE + k_sub * 2u, SMEM_A_STRIDE, gl_CooperativeMatrixLayoutRowMajor);\n"
               << "            coopMatLoad(ma1, smem_a, (wave_m * 32u + 16u) * SMEM_A_STRIDE + k_sub * 2u, SMEM_A_STRIDE, gl_CooperativeMatrixLayoutRowMajor);\n"
               << "            coopMatLoad(mb0, smem_b, (k_sub * 16u) * SMEM_B_STRIDE + ((wave_n * 32u)       >> 3u), SMEM_B_STRIDE, gl_CooperativeMatrixLayoutRowMajor);\n"
               << "            coopMatLoad(mb1, smem_b, (k_sub * 16u) * SMEM_B_STRIDE + ((wave_n * 32u + 16u) >> 3u), SMEM_B_STRIDE, gl_CooperativeMatrixLayoutRowMajor);\n"
               << "            acc00 = coopMatMulAdd(ma0, mb0, acc00); acc01 = coopMatMulAdd(ma0, mb1, acc01);\n"
               << "            acc10 = coopMatMulAdd(ma1, mb0, acc10); acc11 = coopMatMulAdd(ma1, mb1, acc11);\n"
               << "        }\n";
        }
        os << "        barrier();\n    }\n";

        const char *accn[4] = {"acc00", "acc01", "acc10", "acc11"};
        for (int st = 0; st < 4; ++st)
        {
            uint32_t dr = (st >> 1) * 16u, dc = (st & 1) * 16u;
            os << std::format("    if (base_r + {}u < M && base_c + {}u < N) {{\n", dr, dc)
               << std::format("        coopMatStore({}, smem_c, wave_base, 16u, gl_CooperativeMatrixLayoutRowMajor);\n", accn[st])
               << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
               << "        for (uint i = 0u; i < 8u; ++i) {\n"
               << "            uint idx = lane_id + i * 32u;\n"
               << std::format("            uint gr = base_r + {}u + (idx >> 4u);\n", dr)
               << std::format("            uint gc = base_c + {}u + (idx & 15u);\n", dc)
               << "            if (gr < M && gc < N) {\n"
               << "                uint lidx = gr * N + gc;\n"
               << "                float gval = smem_c[wave_base + idx];\n";
            if (ext_out_set.contains(gemm_out) && !plan.epilogue_steps.empty())
            {
                os << std::format("                b_{}[lidx] = float16_t(clamp(gval, -65504.0, 65504.0));\n", gemm_out);
            }
            if (plan.epilogue_steps.empty())
            {
                os << std::format("                b_{}[lidx] = float16_t(clamp(gval, -65504.0, 65504.0));\n", gemm_out);
            }
            else
            {
                std::unordered_map<uint32_t, std::string> ereg;
                ereg[gemm_out] = "gval";
                for (size_t si = 0; si < plan.epilogue_steps.size(); ++si)
                {
                    const auto &s = plan.epilogue_steps[si];
                    auto rdL = [&](uint32_t bi) -> std::string
                    {
                        if (auto it = ereg.find(bi); it != ereg.end()) return it->second;
                        return std::format("float(b_{}[lidx])", bi);
                    };
                    auto rdR = [&](uint32_t bi) -> std::string
                    {
                        if (auto it = ereg.find(bi); it != ereg.end()) return it->second;
                        std::string ie = (s.rhs_index_mode == Index_Map_Kind::BROADCAST_ROW_1D) ? "gc" : "lidx";
                        return std::format("float(b_{}[{}])", bi, ie);
                    };
                    std::string lhs = s.input_buffer_indices.empty() ? "0.0" : rdL(s.input_buffer_indices[0]);
                    std::string rhs = (s.input_buffer_indices.size() >= 2) ? rdR(s.input_buffer_indices[1]) : "";
                    std::string ov = std::format("ev_{}", si);
                    emitPointwiseExprScalar(os, s, lhs, rhs, ov, true);
                    ereg[s.output_buffer_index] = ov;
                    if (ext_out_set.contains(s.output_buffer_index))
                    {
                        emitFp16StoreExpr(os, "                ", s, std::format("b_{}[lidx]", s.output_buffer_index), ov, false);
                    }
                }
            }
            os << "            }\n"
               << "        }\n"
               << "        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);\n"
               << "    }\n";
        }
        os << "}\n";
        return os.str();
    }

std::string Graph_Executor::generateFusedGlsl(const Compute_Node &_node) const
    {
        if (_node.fusion_plan)
        {
            switch (_node.fusion_plan->anchor_role)
            {
            case Fused_Anchor_Role::HORIZONTAL_MULTI_LANE:
                return emitHorizontalAdamGlsl(_node);
            case Fused_Anchor_Role::POINTWISE_CHAIN:
                return emitPointwiseChainGlsl(_node);
            case Fused_Anchor_Role::ROW_REDUCTION:
                return emitRowReductionGlsl(_node);
            case Fused_Anchor_Role::GEMM_COOPMAT:
                return emitCoopMatEpilogueGlsl(_node);
            default:
                break;
            }
        }
        uint32_t local_size_x = 256;
        uint32_t local_size_y = 1;
        uint32_t local_size_z = 1;

        Operation_Class primary_operation_class = Operation_Class::ELEMENTWISE;
        Compute_Pipeline primary_pipeline = Compute_Pipeline::ADD;

        const bool coop_enabled = context.isCooperativeMatrixEnabled();
        bool has_reduction = false;
        bool has_shared_memory = false;
        bool has_matrix_tiles = false;
        uint32_t max_shared_memory_size = 0;

        bool use_primary_coop = false;

        if (!_node.fused_operations.empty())
        {
            primary_pipeline = _node.fused_operations[0].pipeline_id;
            const auto &primary_meta = shader_dictionary.getMetadata(primary_pipeline);
            primary_operation_class = primary_meta.operation_class;
            use_primary_coop = coop_enabled && primary_meta.is_cooperative_matrix_support && !primary_meta.cooperative_glsl_template.empty();

            if (use_primary_coop && _node.fused_operations.size() > 1)
            {
                throw std::runtime_error("Graph_Executor::generateFusedGlsl: Cooperative matrix kernels cannot fuse multiple operations");
            }

            if (primary_operation_class == Operation_Class::MATRIX_2D || primary_operation_class == Operation_Class::TENSOR_3D)
            {
                if (use_primary_coop)
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

                const bool op_use_coop = coop_enabled && metadata.is_cooperative_matrix_support && !metadata.cooperative_glsl_template.empty();
                uint32_t op_shared_memory_size = op_use_coop
                                                          ? metadata.cooperative_shared_memory_size
                                                          : metadata.shared_memory_size;

                if (op_shared_memory_size > 0)
                {
                    has_shared_memory = true;
                    has_reduction = true;
                    max_shared_memory_size = std::max(max_shared_memory_size, op_shared_memory_size);
                }

                bool is_using_manual_tiles = metadata.glsl_template.find("tile_a") != std::string::npos;

                if (is_using_manual_tiles && !op_use_coop)
                {
                    has_matrix_tiles = true;
                }
            }
        }

        if (has_reduction && !use_primary_coop)
        {
            local_size_x = 256;
            local_size_y = 1;
            local_size_z = 1;
        }

        if (has_reduction && has_shared_memory)
        {
            const uint32_t min_sg = (context.getMinSubgroupSize() > 0)
                                        ? context.getMinSubgroupSize()
                                        : ((context.getDefaultSubgroupSize() > 0) ? context.getDefaultSubgroupSize() : 32u);
            max_shared_memory_size = std::max(max_shared_memory_size, (local_size_x + min_sg - 1u) / min_sg);
        }

        Shader_Generator shader_generator(local_size_x, local_size_y, local_size_z, "float");

        if (coop_enabled)
        {
            for (const auto &operation : _node.fused_operations)
            {
                const auto &meta = shader_dictionary.getMetadata(operation.pipeline_id);
                if (meta.is_cooperative_matrix_support && !meta.cooperative_glsl_template.empty())
                {
                    shader_generator.enableCooperativeMatrix();
                    break;
                }
            }
        }

        if (has_reduction)
        {
            shader_generator.enableSubgroupOperations();
        }

        if (has_shared_memory && max_shared_memory_size > 0)
        {
            shader_generator.addSharedMemoryRaw(std::format("shared float shared_mem_0[{}];", max_shared_memory_size));
        }

        if (has_matrix_tiles)
        {
            shader_generator.addSharedMemoryRaw("shared float tile_a[16][17];\nshared float tile_b[16][17];");
        }

        std::vector<uint32_t> external_indices;
        getExternalBufferIndices(_node, external_indices);

        std::unordered_set<uint32_t> external_output_set(_node.external_output_indices.begin(), _node.external_output_indices.end());
        std::unordered_set<uint32_t> read_write_output_set;
        if (!_node.fused_operations.empty())
        {
            for (uint32_t output_index : _node.fused_operations.back().output_buffer_indices)
            {
                external_output_set.insert(output_index);
            }
        }
        for (const auto &operation : _node.fused_operations)
        {
            const auto &metadata = shader_dictionary.getMetadata(operation.pipeline_id);
            for (uint32_t persistent_output_local_index : metadata.persistent_output_indices)
            {
                if (persistent_output_local_index < operation.output_buffer_indices.size())
                {
                    uint32_t buf_idx = operation.output_buffer_indices[persistent_output_local_index];
                    external_output_set.insert(buf_idx);
                    read_write_output_set.insert(buf_idx);
                }
            }
            for (uint32_t accum_output_local_index : metadata.accumulator_output_indices)
            {
                if (accum_output_local_index < operation.output_buffer_indices.size())
                {
                    uint32_t buf_idx = operation.output_buffer_indices[accum_output_local_index];
                    external_output_set.insert(buf_idx);
                    read_write_output_set.insert(buf_idx);
                }
            }
        }

        std::unordered_set<uint32_t> read_buffer_indices;

        for (size_t op_idx = 0; op_idx < _node.fused_operations.size(); ++op_idx)
        {
            const auto &operation = _node.fused_operations[op_idx];

            for (uint32_t input_index : operation.input_buffer_indices)
            {
                bool is_passed_via_register = false;
                for (size_t prev_idx = 0; prev_idx < op_idx; ++prev_idx)
                {
                    const auto &prev_op = _node.fused_operations[prev_idx];
                    for (uint32_t prev_out : prev_op.output_buffer_indices)
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
        }

        for (uint32_t buffer_index : external_indices)
        {
            bool is_read = read_buffer_indices.contains(buffer_index);
            bool is_written = external_output_set.contains(buffer_index);

            Buffer_Access buffer_access = Buffer_Access::READ_WRITE;
            if (read_write_output_set.contains(buffer_index))
            {
                buffer_access = Buffer_Access::READ_WRITE;
            }
            else if (is_read && !is_written)
            {
                buffer_access = Buffer_Access::READ_ONLY;
            }
            else if (!is_read && is_written)
            {
                buffer_access = Buffer_Access::WRITE_ONLY;
            }
            std::string buffer_type = "float";
            if (buffer_index < _node.buffers.size() && _node.buffers[buffer_index])
            {
                buffer_type = getDataTypeGlslName(_node.buffers[buffer_index]->getDataType());
            }

            if (buffer_type == "float16_t")
            {
                shader_generator.enableFloat16();
            }

            shader_generator.addBuffer(buffer_index, std::format("buf_{}", buffer_index), buffer_type, buffer_access);
        }

        shader_generator.setPushConstants("uint data[32];");

        std::unordered_map<uint32_t, std::string> register_map;

        if (primary_operation_class == Operation_Class::MATRIX_2D)
        {
            if (!use_primary_coop)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint r = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint lc = gl_LocalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint lr = gl_LocalInvocationID.y;");
            }
        }
        else if (primary_operation_class == Operation_Class::STANDALONE)
        {
            if (primary_pipeline == Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD ||
                primary_pipeline == Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD ||
                primary_pipeline == Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD_FP16 ||
                primary_pipeline == Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD_FP16)
            {
            }
            else if (primary_pipeline == Compute_Pipeline::BATCH_NORM_STATS_FORWARD ||
                     primary_pipeline == Compute_Pipeline::BATCH_NORM_STATS_BACKWARD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_WorkGroupID.x;");
                shader_generator.addLogicSnippet("    if (c >= pc.data[1]) return;");
            }
            else
            {
                shader_generator.addLogicSnippet("    uint r = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;");
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
            else if (primary_pipeline == Compute_Pipeline::MATMUL ||
                     primary_pipeline == Compute_Pipeline::MATMUL_ADD)
            {
                shader_generator.addLogicSnippet("    uint c = gl_GlobalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint r = gl_GlobalInvocationID.y;");
                shader_generator.addLogicSnippet("    uint b = gl_GlobalInvocationID.z;");
                shader_generator.addLogicSnippet("    uint lc = gl_LocalInvocationID.x;");
                shader_generator.addLogicSnippet("    uint lr = gl_LocalInvocationID.y;");
                shader_generator.addLogicSnippet("    if (b >= pc.data[0]) return;");
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

        for (size_t operation_index = 0; operation_index < _node.fused_operations.size(); ++operation_index)
        {
            const Fused_Operation &operation = _node.fused_operations[operation_index];
            const Snippet_Metadata &metadata = shader_dictionary.getMetadata(operation.pipeline_id);

            std::vector<std::string> inputs;
            std::vector<bool> is_input_register_flags;

            for (uint32_t input_index : operation.input_buffer_indices)
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

            for (size_t local_output_index = 0; local_output_index < operation.output_buffer_indices.size(); ++local_output_index)
            {
                uint32_t output_index = operation.output_buffer_indices[local_output_index];
                bool is_accumulator = std::find(metadata.accumulator_output_indices.begin(),
                                                metadata.accumulator_output_indices.end(),
                                                static_cast<uint32_t>(local_output_index)) != metadata.accumulator_output_indices.end();
                bool is_persistent = std::find(metadata.persistent_output_indices.begin(),
                                               metadata.persistent_output_indices.end(),
                                               static_cast<uint32_t>(local_output_index)) != metadata.persistent_output_indices.end();

                if (is_accumulator || is_persistent)
                {
                    outputs.push_back(std::format("buf_{}", output_index));
                    is_output_register_flags.push_back(false);
                }
                else
                {
                    std::string reg_type = "float";
                    if (output_index < _node.buffers.size() && _node.buffers[output_index])
                    {
                        reg_type = std::string(getDataTypeGlslName(_node.buffers[output_index]->getDataType()));
                    }
                    if (reg_type == "float16_t")
                    {
                        shader_generator.enableFloat16();
                    }
                    std::string register_name = shader_generator.getUniqueVar("reg");
                    shader_generator.addLogicSnippet(std::format("    {} {} = {}(0.0);", reg_type, register_name, reg_type));

                    register_map[output_index] = register_name;
                    outputs.push_back(register_name);
                    is_output_register_flags.push_back(true);
                }
            }

            const bool op_use_coop = coop_enabled && metadata.is_cooperative_matrix_support && !metadata.cooperative_glsl_template.empty();
            const std::string &raw_template = op_use_coop
                                                  ? metadata.cooperative_glsl_template
                                                  : metadata.glsl_template;

            uint32_t push_constants_word_offset = operation.push_constants_offset / 4;
            std::string snippet = replacePlaceholders(raw_template,
                                                      inputs,
                                                      is_input_register_flags,
                                                      outputs,
                                                      is_output_register_flags,
                                                      operation.output_buffer_indices,
                                                      external_output_set,
                                                      push_constants_word_offset,
                                                      _node.buffers);

            auto remove_pattern = [](std::string &source_string, const std::string &prefix, const std::string &suffix)
            {
                size_t start_position = source_string.find(prefix);
                if (start_position != std::string::npos)
                {
                    size_t end_position = source_string.find(suffix, start_position);
                    if (end_position != std::string::npos)
                    {
                        source_string.erase(start_position, (end_position + suffix.length()) - start_position);
                    }
                }
            };

            auto remove_exact = [](std::string &source_string, const std::string &string_to_remove)
            {
                size_t position = 0;
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
                        snippet = snippet.substr(36);
                        if (!snippet.empty() && snippet.front() == ' ')
                        {
                            snippet = snippet.substr(1);
                        }
                    }
                }
                else if (primary_operation_class == Operation_Class::MATRIX_2D)
                {
                    if (!use_primary_coop)
                    {
                        const bool primary_uses_tiles = metadata.glsl_template.find("tile_a") != std::string::npos;
                        remove_exact(snippet, "uint r = gl_GlobalInvocationID.y;");
                        remove_exact(snippet, "uint c = gl_GlobalInvocationID.x;");
                        remove_exact(snippet, "uint lc = gl_LocalInvocationID.x;");
                        remove_exact(snippet, "uint lr = gl_LocalInvocationID.y;");
                        if (primary_uses_tiles)
                        {
                            remove_pattern(snippet, "if (r >= ", " return;");
                        }
                    }
                }
                else if (primary_operation_class == Operation_Class::TENSOR_3D)
                {
                    remove_exact(snippet, "uint oc = gl_GlobalInvocationID.x;");
                    remove_exact(snippet, "uint ic = gl_GlobalInvocationID.x;");
                    remove_exact(snippet, "uint c = gl_GlobalInvocationID.x;");
                    remove_exact(snippet, "uint r = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint b = gl_GlobalInvocationID.z;");
                    remove_exact(snippet, "uint ow = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint iw = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint n = gl_GlobalInvocationID.y;");
                    remove_exact(snippet, "uint n_oh = gl_GlobalInvocationID.z;");
                    remove_exact(snippet, "uint n_ih = gl_GlobalInvocationID.z;");
                    remove_exact(snippet, "uint lc = gl_LocalInvocationID.x;");
                    remove_exact(snippet, "uint lr = gl_LocalInvocationID.y;");

                    remove_pattern(snippet, "uint n = n_oh /", ";");
                    remove_pattern(snippet, "uint oh = n_oh %", ";");
                    remove_pattern(snippet, "uint n = n_ih /", ";");
                    remove_pattern(snippet, "uint ih = n_ih %", ";");
                    remove_pattern(snippet, "if (oc >= ", " return;");
                    remove_pattern(snippet, "if (c >= ", " return;");
                    remove_pattern(snippet, "if (ic >= ", " return;");
                    remove_pattern(snippet, "if (b >= ", " return;");
                    remove_pattern(snippet, "if (r >= ", " return;");
                }
                else if (primary_operation_class == Operation_Class::STANDALONE)
                {
                    remove_exact(snippet, "uint r = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;");
                    remove_exact(snippet, "uint r = gl_WorkGroupID.x;");
                    remove_exact(snippet, "uint c = gl_WorkGroupID.x;");
                    remove_exact(snippet, "uint d = gl_WorkGroupID.x;");
                    remove_pattern(snippet, "if (c >= ", " return;");
                    remove_pattern(snippet, "if (r >= ", " return;");
                }
            }

            const bool wrap_in_block = (operation_index >= 1) ||
                                       (operation_index == 0 && primary_operation_class == Operation_Class::ELEMENTWISE && metadata.index_expression.empty());
            if (wrap_in_block)
            {
                shader_generator.addLogicSnippet(std::format("    {{\n    {}\n    }}", snippet));
            }
            else
            {
                shader_generator.addLogicSnippet(std::format("    {}", snippet));
            }

            if (operation_index == 0 && !metadata.index_expression.empty() && !use_primary_coop)
            {
                std::string resolved_expression = metadata.index_expression;
                for (int push_constant_index = 31; push_constant_index >= 0; --push_constant_index)
                {
                    std::string token = std::format("{{pc_{}}}", push_constant_index);
                    std::string replacement = std::format("pc.data[{}]", push_constants_word_offset + push_constant_index);
                    size_t position = 0;
                    while ((position = resolved_expression.find(token, position)) != std::string::npos)
                    {
                        resolved_expression.replace(position, token.length(), replacement);
                        position += replacement.length();
                    }
                }

                const bool primary_uses_tiles = metadata.glsl_template.find("tile_a") != std::string::npos;
                if (primary_uses_tiles)
                {
                    shader_generator.addLogicSnippet("    if (r >= M || c >= N) return;");
                }

                shader_generator.addLogicSnippet(std::format("    global_id = {};", resolved_expression));
            }
        }

        std::string glsl_code = shader_generator.build();

        std::string node_chain_name = "[";
        for (size_t op_idx = 0; op_idx < _node.fused_operations.size(); ++op_idx)
        {
            node_chain_name += std::string(magic_enum::enum_name(_node.fused_operations[op_idx].pipeline_id));
            if (op_idx + 1 < _node.fused_operations.size())
            {
                node_chain_name += " -> ";
            }
        }
        node_chain_name += "]";

        bool should_log_chain = false;
        {
            std::lock_guard<std::mutex> lock(printed_chains_mutex);
            if (std::find(printed_terminal_shader_chains.begin(), printed_terminal_shader_chains.end(), node_chain_name) == printed_terminal_shader_chains.end())
            {
                if (printed_terminal_shader_chains.size() < 512)
                {
                    printed_terminal_shader_chains.push_back(node_chain_name);
                }
                should_log_chain = true;
            }
        }
        if (should_log_chain)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::generateFusedGlsl: Generated GLSL for node {}:\n\n===============\n{}\n\n===============",
                                           node_chain_name, glsl_code},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION | Log_Feature::OPERATOR_FUSION);
        }
        return glsl_code;
    }

void Graph_Executor::invalidate()
{
        Logger::logMessage("Graph_Executor::invalidate: Invalidating persistent descriptor caches and resetting pools",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);
        VkDevice device = context.getDevice();
        if (device != VK_NULL_HANDLE)
        {
            VkResult idle_res = vkDeviceWaitIdle(device);
            if (idle_res != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::invalidate: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            }
        }
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            resolveFrameTimestamps(i);
            if (descriptor_pools[i] != VK_NULL_HANDLE)
            {
                vkResetDescriptorPool(device, descriptor_pools[i], 0);
            }
            for (VkDescriptorPool extra_pool : overflow_descriptor_pools[i])
            {
                if (device != VK_NULL_HANDLE && extra_pool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(device, extra_pool, nullptr);
                }
            }
            overflow_descriptor_pools[i].clear();
            persistent_descriptor_caches[i].clear();
            fallback_descriptor_caches[i].clear();
            is_static_baked[i] = false;
            static_graph_signatures[i] = 0;
            static_split_indices[i] = 0;
            baked_buffer_handles[i].clear();
            baked_buffer_records[i].clear();
            unique_baked_buffers[i].clear();
            static_validated_buffer_epoch[i] = 0;
            static_baked_pc_snapshots[i].clear();
            static_baked_node_meta[i].clear();
            static_bake_generations[i] = 0;
            dynamic_optimizer_pool_used[i] = 0;
            dynamic_pool_dirty[i] = false;
            frame_timestamp_entries[i].clear();
            frame_barrier_count[i] = 0;
            has_pending_timestamps[i] = false;
            static_recorded_entries[i].clear();
            static_recorded_barrier_count[i] = 0;
            static_next_query_index[i] = 0u;
            static_baked_with_profiling[i] = false;
        }
    }

void Graph_Executor::invalidateStaticGraph() noexcept
    {
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            is_static_baked[i] = false;
            static_graph_signatures[i] = 0;
            static_split_indices[i] = 0;
            baked_buffer_handles[i].clear();
            baked_buffer_records[i].clear();
            unique_baked_buffers[i].clear();
            static_validated_buffer_epoch[i] = 0;
            static_baked_pc_snapshots[i].clear();
            static_baked_node_meta[i].clear();
            static_bake_generations[i] = 0;
            static_recorded_entries[i].clear();
            static_recorded_barrier_count[i] = 0;
            static_next_query_index[i] = 0u;
            static_baked_with_profiling[i] = false;
        }
    }

std::shared_ptr<gpu::vector> Graph_Executor::resolveNodeBuffer(const Compute_Node &_node, size_t _b_idx, uint32_t _frame_index) const
    {
        if (_b_idx >= _node.buffers.size())
        {
            return nullptr;
        }
        auto buf = _node.buffers[_b_idx];
        bool is_dyn_slot =
            ((_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE || _node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16) && _b_idx == 4) ||
            ((_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16) && _b_idx == 0) ||
            ((_node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16 || _node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16) && _b_idx == 5) ||
            ((_node.pipeline_id == Compute_Pipeline::SGD_UPDATE || _node.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16) && _b_idx == 2) ||
            ((_node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE || _node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16) && _b_idx == 3);
        if (is_dyn_slot && _frame_index < MAX_FRAMES_IN_FLIGHT)
        {
            bool is_engine_default = (!buf);
            for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
            {
                if (buf == dynamic_optimizer_buffers[f])
                {
                    is_engine_default = true;
                    break;
                }
            }
            if (is_engine_default)
            {
                return dynamic_optimizer_buffers[_frame_index];
            }
        }
        if (_node.fusion_plan && _node.fusion_plan->shared_dyn_params_binding == _b_idx &&
            _frame_index < MAX_FRAMES_IN_FLIGHT)
        {
            bool is_engine_default = (!buf);
            for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
            {
                if (buf == dynamic_optimizer_buffers[f])
                {
                    is_engine_default = true;
                    break;
                }
            }
            if (is_engine_default)
            {
                return dynamic_optimizer_buffers[_frame_index];
            }
        }
        return buf;
    }

void Graph_Executor::recordTransferTasks(VkCommandBuffer _command_buffer, const std::vector<Buffer_Transfer_Task> &_transfer_tasks) const
    {
        if (_transfer_tasks.empty())
        {
            return;
        }

        VkMemoryBarrier pre_transfer_barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};

        vkCmdPipelineBarrier(
            _command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &pre_transfer_barrier,
            0, nullptr,
            0, nullptr);

        std::unordered_set<VkBuffer> pending_reads;
        std::unordered_set<VkBuffer> pending_writes;
        bool recorded_any = false;
        for (const auto &task : _transfer_tasks)
        {
            if (task.destination_buffer == VK_NULL_HANDLE || task.size == 0 ||
                (!task.is_fill && task.source_buffer != VK_NULL_HANDLE && task.source_buffer == task.destination_buffer && task.source_offset == task.destination_offset))
            {
                continue;
            }

            const bool is_fill = (task.is_fill || task.source_buffer == VK_NULL_HANDLE);
            const VkBuffer src_buf = is_fill ? VK_NULL_HANDLE : task.source_buffer;
            const VkBuffer dst_buf = task.destination_buffer;

            bool needs_barrier = false;
            if (!is_fill && src_buf != VK_NULL_HANDLE && pending_writes.contains(src_buf))
            {
                needs_barrier = true;
            }
            if (!needs_barrier && dst_buf != VK_NULL_HANDLE &&
                (pending_writes.contains(dst_buf) || pending_reads.contains(dst_buf)))
            {
                needs_barrier = true;
            }

            if (recorded_any && needs_barrier)
            {
                VkMemoryBarrier inter_transfer_barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};

                vkCmdPipelineBarrier(
                    _command_buffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 1, &inter_transfer_barrier,
                    0, nullptr,
                    0, nullptr);

                pending_reads.clear();
                pending_writes.clear();
            }

            if (is_fill)
            {
                if ((task.destination_offset % 4) != 0 || (task.size != VK_WHOLE_SIZE && (task.size % 4) != 0))
                {
                    throw std::invalid_argument(std::format(
                        "Graph_Executor::recordTransferTasks: vkCmdFillBuffer requires 4-byte aligned offset ({}) and size ({})",
                        task.destination_offset, task.size));
                }
                const uint32_t pattern = task.is_fill ? task.fill_pattern : static_cast<uint32_t>(task.source_offset);
                vkCmdFillBuffer(_command_buffer, dst_buf, task.destination_offset, task.size, pattern);
            }
            else
            {
                VkBufferCopy copy_region{
                    .srcOffset = task.source_offset,
                    .dstOffset = task.destination_offset,
                    .size = task.size};

                vkCmdCopyBuffer(_command_buffer, src_buf, dst_buf, 1, &copy_region);
                if (src_buf != VK_NULL_HANDLE)
                {
                    pending_reads.insert(src_buf);
                }
            }
            if (dst_buf != VK_NULL_HANDLE)
            {
                pending_writes.insert(dst_buf);
            }
            recorded_any = true;
        }

        VkMemoryBarrier post_transfer_barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};

        vkCmdPipelineBarrier(
            _command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &post_transfer_barrier,
            0, nullptr,
            0, nullptr);
    }

void Graph_Executor::resetFrameState(uint32_t _frame_index)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::resetFrameState: frame_index out of bounds ({})", _frame_index},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }
        resolveFrameTimestamps(_frame_index);
        if (dynamic_pool_dirty[_frame_index])
        {
            dynamic_optimizer_pool_used[_frame_index] = 0;
            dynamic_pool_dirty[_frame_index] = false;
        }
    }

void Graph_Executor::recordComputeNodes(VkCommandBuffer _command_buffer,
                            const std::vector<Compute_Node> &_nodes,
                            size_t _start_index,
                            size_t _end_index,
                            uint32_t _frame_index,
                            size_t _cache_index_offset,
                            uint32_t *_query_cursor,
                            std::vector<Recorded_Timestamp_Entry> *_entries_out,
                            uint64_t *_barriers_out)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || _command_buffer == VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::recordComputeNodes: Invalid frame_index ({}) or null command buffer", _frame_index},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }

        if (_start_index >= _end_index || _start_index >= _nodes.size())
        {
            return;
        }

        size_t end_idx = std::min(_end_index, _nodes.size());

        size_t required_cache_size = end_idx + _cache_index_offset;
        if (persistent_descriptor_caches[_frame_index].size() < required_cache_size)
        {
            persistent_descriptor_caches[_frame_index].resize(required_cache_size);
        }

        const bool fallback_all_barriers = (_nodes.size() > 1) &&
                                           std::none_of(_nodes.begin(), _nodes.end(),
                                                        [](const Compute_Node &n) { return n.is_barrier_required_after; });

        const VkQueryPool active_query_pool =
            (isGpuProfilingActive() && _query_cursor != nullptr && _entries_out != nullptr)
                ? timestamp_query_pools[_frame_index]
                : VK_NULL_HANDLE;
        const uint32_t query_capacity = timestamp_query_capacities[_frame_index];

        for (size_t i = _start_index; i < end_idx; ++i)
        {
            size_t cache_idx = i + _cache_index_offset;
            const Compute_Node &node = _nodes[i];

            for (const auto &vector_ptr : node.buffers)
            {
                if (vector_ptr)
                {
                    vector_ptr->markAsUsedInFrame(_frame_index);
                }
                else
                {
                    Logger::logMessage(Input_Format{"Graph_Executor::recordComputeNodes: Null gpu::vector buffer encountered in compute node {}", i},
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::DISPATCH_EXECUTION);
                }
            }

            const bool write_ts = (active_query_pool != VK_NULL_HANDLE) && ((*_query_cursor + 2u) <= query_capacity);
            uint32_t q_start = 0u;
            uint32_t q_end = 0u;
            if (write_ts)
            {
                q_start = (*_query_cursor)++;
                q_end = (*_query_cursor)++;
                vkCmdWriteTimestamp(_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, active_query_pool, q_start);
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
                        if (node.fusion_plan && !node.fusion_plan->canonical_key.empty())
                        {
                            target_pipeline = pipeline_cache_manager.getOrCreatePipeline(
                                node.fusion_plan->canonical_key, glsl_code, node.fusion_plan->spec_constants);
                        }
                        else
                        {
                            target_pipeline = pipeline_cache_manager.getOrCreatePipeline(glsl_code);
                        }
                    }

                    if (target_pipeline != VK_NULL_HANDLE)
                    {
                        const std::vector<uint32_t> &external_indices = !node.cached_external_buffer_indices.empty()
                                                                                 ? node.cached_external_buffer_indices
                                                                                 : (getExternalBufferIndices(node, shared_external_buffer_indices), shared_external_buffer_indices);

                        shared_fused_buffers.clear();
                        for (uint32_t buffer_index : external_indices)
                        {
                            if (buffer_index < node.buffers.size())
                            {
                                auto resolved_buf = resolveNodeBuffer(node, buffer_index, _frame_index);
                                if (resolved_buf)
                                {
                                    resolved_buf->markAsUsedInFrame(_frame_index);
                                }
                                shared_fused_buffers.push_back(std::move(resolved_buf));
                            }
                        }

                        vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);

                        if (network.isPushDescriptorActive())
                        {
                            pushDescriptors(_command_buffer, external_indices, shared_fused_buffers);
                        }
                        else
                        {
                            auto &entry = persistent_descriptor_caches[_frame_index][cache_idx];
                            if (entry.descriptor_set == VK_NULL_HANDLE)
                            {
                                entry.descriptor_set = allocateDescriptorSet(_frame_index);
                            }

                            if (!isBuffersMatching(entry, external_indices, shared_fused_buffers))
                            {
                                updateDescriptorSet(entry.descriptor_set, external_indices, shared_fused_buffers);
                                entry.bound_binding_indices = external_indices;
                                entry.bound_buffers.resize(shared_fused_buffers.size());
                                entry.bound_vector_ids.resize(shared_fused_buffers.size());
                                entry.bound_buffer_sizes.resize(shared_fused_buffers.size());
                                entry.bound_weak_buffers.resize(shared_fused_buffers.size());
                                for (size_t j = 0; j < shared_fused_buffers.size(); ++j)
                                {
                                    entry.bound_buffers[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getBuffer() : VK_NULL_HANDLE;
                                    entry.bound_vector_ids[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getId() : 0;
                                    entry.bound_buffer_sizes[j] = shared_fused_buffers[j] ? shared_fused_buffers[j]->getSizeBytes() : 0;
                                    entry.bound_weak_buffers[j] = shared_fused_buffers[j];
                                }
                            }

                            vkCmdBindDescriptorSets(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &entry.descriptor_set, 0, nullptr);
                        }

                        if (!node.push_constants_data.empty())
                        {
                            pushConstantsChecked(
                                _command_buffer,
                                network.getPipelineLayout(),
                                node.push_constants_data.data(),
                                node.push_constants_data.size(),
                                "Graph_Executor::recordComputeNodes(fused)");
                        }

                        vkCmdDispatch(_command_buffer, node.workgroup_count_x, node.workgroup_count_y, node.workgroup_count_z);
                        is_fused_successful = true;
                    }
                }
                catch (const std::exception &exception)
                {
                    Logger::logMessage(Input_Format{"Graph_Executor::recordComputeNodes: Fused shader execution failed for node [{}] ({}), initiating fallback execution", i, exception.what()},
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
                    executeFallbackNode(_command_buffer, node, cache_idx, _frame_index);
                }
                else
                {
                    const auto *effective_buffers = &node.buffers;
                    std::vector<std::shared_ptr<gpu::vector>> localized_buffers;
                    if (((node.pipeline_id == Compute_Pipeline::ADAM_UPDATE || node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16) && node.buffers.size() > 4) ||
                        ((node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16) && !node.buffers.empty()) ||
                        ((node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16 || node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16) && node.buffers.size() > 5) ||
                        ((node.pipeline_id == Compute_Pipeline::SGD_UPDATE || node.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16) && node.buffers.size() > 2) ||
                        ((node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE || node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16) && node.buffers.size() > 3))
                    {
                        localized_buffers.resize(node.buffers.size());
                        for (size_t b_idx = 0; b_idx < node.buffers.size(); ++b_idx)
                        {
                            localized_buffers[b_idx] = resolveNodeBuffer(node, b_idx, _frame_index);
                            if (localized_buffers[b_idx])
                            {
                                localized_buffers[b_idx]->markAsUsedInFrame(_frame_index);
                            }
                        }
                        effective_buffers = &localized_buffers;
                    }

                    VkPipeline target_pipeline = (node.spec_constant_u32 == 0u)
                        ? network.getPipeline(node.pipeline_id)
                        : network.getSpecializedPipeline(node.pipeline_id, node.spec_constant_u32);
                    if (target_pipeline == VK_NULL_HANDLE)
                    {
                        std::string msg = std::format("Graph_Executor::recordComputeNodes: Null pipeline for {}", magic_enum::enum_name(node.pipeline_id));
                        Logger::logMessage(msg, Log_Level::LOG_ERROR, true, 0, Log_Feature::DISPATCH_EXECUTION);
                        throw std::runtime_error(msg);
                    }
                    vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);

                    if (network.isPushDescriptorActive())
                    {
                        pushDescriptors(_command_buffer, *effective_buffers);
                    }
                    else
                    {
                        auto &entry = persistent_descriptor_caches[_frame_index][cache_idx];
                        if (entry.descriptor_set == VK_NULL_HANDLE)
                        {
                            entry.descriptor_set = allocateDescriptorSet(_frame_index);
                        }

                        if (!isBuffersMatching(entry, *effective_buffers))
                        {
                            updateDescriptorSet(entry.descriptor_set, *effective_buffers);
                            entry.bound_binding_indices.clear();
                            entry.bound_buffers.resize(effective_buffers->size());
                            entry.bound_vector_ids.resize(effective_buffers->size());
                            entry.bound_buffer_sizes.resize(effective_buffers->size());
                            entry.bound_weak_buffers.resize(effective_buffers->size());
                            for (size_t j = 0; j < effective_buffers->size(); ++j)
                            {
                                entry.bound_buffers[j] = (*effective_buffers)[j] ? (*effective_buffers)[j]->getBuffer() : VK_NULL_HANDLE;
                                entry.bound_vector_ids[j] = (*effective_buffers)[j] ? (*effective_buffers)[j]->getId() : 0;
                                entry.bound_buffer_sizes[j] = (*effective_buffers)[j] ? (*effective_buffers)[j]->getSizeBytes() : 0;
                                entry.bound_weak_buffers[j] = (*effective_buffers)[j];
                            }
                        }

                        vkCmdBindDescriptorSets(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, network.getPipelineLayout(), 0, 1, &entry.descriptor_set, 0, nullptr);
                    }

                    const uint32_t required_pc_bytes = network.getPipelinePushConstantSize(node.pipeline_id);
                    if (required_pc_bytes > 0u && node.push_constants_data.size() < required_pc_bytes)
                    {
                        std::string msg = std::format(
                            "Graph_Executor::recordComputeNodes: Push constant size mismatch for {} (provided={}B < required={}B)",
                            magic_enum::enum_name(node.pipeline_id),
                            node.push_constants_data.size(),
                            required_pc_bytes);
                        Logger::logMessage(msg, Log_Level::LOG_ERROR, true, 0, Log_Feature::DISPATCH_EXECUTION);
                        throw std::runtime_error(msg);
                    }

                    if (!node.push_constants_data.empty())
                    {
                        pushConstantsChecked(
                            _command_buffer,
                            network.getPipelineLayout(),
                            node.push_constants_data.data(),
                            node.push_constants_data.size(),
                            "Graph_Executor::recordComputeNodes");
                    }

                    vkCmdDispatch(_command_buffer, node.workgroup_count_x, node.workgroup_count_y, node.workgroup_count_z);
                }
            }

            if (write_ts)
            {
                vkCmdWriteTimestamp(_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, active_query_pool, q_end);
                _entries_out->push_back(Recorded_Timestamp_Entry{
                    makeNodeProfileLabel(node, is_fused_successful),
                    q_start,
                    q_end});
            }

            if (node.is_barrier_required_after || fallback_all_barriers)
            {
                insertComputeMemoryBarrier(_command_buffer);
                if (_barriers_out != nullptr)
                {
                    ++(*_barriers_out);
                }
            }
        }
        shared_fused_buffers.clear();
        shared_external_buffer_indices.clear();
    }

void Graph_Executor::compileAndExecute(const Compute_Graph &_graph,
                           const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                           uint32_t _frame_index,
                           VkFence _external_fence)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: frame_index out of bounds ({})", _frame_index},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }

        if (is_static_baked[_frame_index])
        {
            is_static_baked[_frame_index] = false;
            static_split_indices[_frame_index] = 0;
            baked_buffer_records[_frame_index].clear();
            unique_baked_buffers[_frame_index].clear();
            static_validated_buffer_epoch[_frame_index] = 0;
            static_baked_pc_snapshots[_frame_index].clear();
            static_baked_node_meta[_frame_index].clear();
            static_recorded_entries[_frame_index].clear();
            static_next_query_index[_frame_index] = 0;
            static_recorded_barrier_count[_frame_index] = 0;
            static_baked_with_profiling[_frame_index] = false;
            static_bake_generations[_frame_index] = 0;
        }

        const std::vector<Compute_Node> &nodes = _graph.getNodes();
        if (nodes.empty() && _transfer_tasks.empty())
        {
            Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: Both compute nodes and transfer tasks are empty for frame {}", _frame_index},
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
        }

        Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: Frame {}, nodes={}, transfer_tasks={}", _frame_index, nodes.size(), _transfer_tasks.size()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);

        VkCommandBuffer command_buffer = command_buffers[_frame_index];

        has_pending_timestamps[_frame_index] = false;
        frame_timestamp_entries[_frame_index].clear();
        frame_barrier_count[_frame_index] = 0;

        uint32_t query_cursor = 0;
        const bool profiling_this_submit =
            isGpuProfilingActive() &&
            !nodes.empty() &&
            ensureTimestampQueryPool(_frame_index, static_cast<uint32_t>(nodes.size() * 2u), true);
        if (profiling_this_submit)
        {
            frame_timestamp_entries[_frame_index].reserve(nodes.size());
        }

        if (vkResetCommandBuffer(command_buffer, 0) != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: Failed to reset command buffer for frame {}", _frame_index},
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
            Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: Failed to begin command buffer for frame {}", _frame_index},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to begin command buffer");
        }

        if (profiling_this_submit)
        {
            vkCmdResetQueryPool(command_buffer,
                                timestamp_query_pools[_frame_index],
                                0u,
                                static_cast<uint32_t>(nodes.size() * 2u));
        }

        if (!_transfer_tasks.empty())
        {
            recordTransferTasks(command_buffer, _transfer_tasks);
        }
        else
        {
            insertInterSubmissionComputeBarrier(command_buffer);
        }

        recordComputeNodes(command_buffer,
                           nodes,
                           0,
                           nodes.size(),
                           _frame_index,
                           0,
                           profiling_this_submit ? &query_cursor : nullptr,
                           profiling_this_submit ? &frame_timestamp_entries[_frame_index] : nullptr,
                           profiling_this_submit ? &frame_barrier_count[_frame_index] : nullptr);

        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: Failed to end command buffer for frame {}", _frame_index},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to end command buffer");
        }

        bool has_separate_external_fence = (_external_fence != VK_NULL_HANDLE && _external_fence != context.getFrameFence(_frame_index));

        VkSubmitInfo submit_information{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = has_separate_external_fence ? 1u : 0u,
            .pSignalSemaphores = has_separate_external_fence ? &external_fence_semaphores[_frame_index] : nullptr};

        VkFence primary_fence = (_external_fence != VK_NULL_HANDLE) ? _external_fence : context.getFrameFence(_frame_index);

        if (has_separate_external_fence)
        {
            if (vkResetFences(context.getDevice(), 1, &_external_fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Graph_Executor::compileAndExecute: Failed to reset external fence");
            }
        }
        else
        {
            context.resetFrameFence(_frame_index);
        }
        VkResult submit_res = vkQueueSubmit(context.getComputeQueue(), 1, &submit_information, primary_fence);
        if (submit_res != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::compileAndExecute: Failed to submit command buffer for frame {}, VkResult={}", _frame_index, static_cast<int>(submit_res)},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to submit command buffer: VkResult=" + std::to_string(static_cast<int>(submit_res)));
        }
        has_pending_timestamps[_frame_index] = !frame_timestamp_entries[_frame_index].empty();
        dynamic_pool_dirty[_frame_index] = true;

        if (has_separate_external_fence)
        {
            context.resetFrameFence(_frame_index);
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkSubmitInfo frame_fence_submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &external_fence_semaphores[_frame_index],
                .pWaitDstStageMask = &wait_stage,
                .commandBufferCount = 0,
                .pCommandBuffers = nullptr,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr};

            VkResult fence_res = vkQueueSubmit(context.getComputeQueue(), 1, &frame_fence_submit, context.getFrameFence(_frame_index));
            if (fence_res != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit frame fence chain: VkResult=" + std::to_string(static_cast<int>(fence_res)));
            }
        }
    }

void Graph_Executor::bakeStaticGraph(const Compute_Graph &_graph, uint32_t _frame_index, size_t _signature)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            return;
        }

        is_static_baked[_frame_index] = false;
        static_recorded_entries[_frame_index].clear();
        static_recorded_barrier_count[_frame_index] = 0;
        static_next_query_index[_frame_index] = 0u;
        static_baked_with_profiling[_frame_index] = false;

        const auto &nodes = _graph.getNodes();
        if (nodes.empty())
        {
            static_graph_signatures[_frame_index] = 0;
            static_split_indices[_frame_index] = 0;
            baked_buffer_handles[_frame_index].clear();
            baked_buffer_records[_frame_index].clear();
            unique_baked_buffers[_frame_index].clear();
            static_validated_buffer_epoch[_frame_index] = 0;
            static_baked_pc_snapshots[_frame_index].clear();
            static_baked_node_meta[_frame_index].clear();
            static_bake_generations[_frame_index] = 0;
            return;
        }

        size_t split_index = nodes.size();
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            if (isDynamicNode(nodes[i]))
            {
                split_index = i;
                break;
            }
        }

        const uint32_t required_static_queries = static_cast<uint32_t>(
            std::max<size_t>(nodes.size() * 2u + 1024u, INITIAL_TIMESTAMP_QUERY_CAPACITY));
        const bool profiling_this_bake =
            isGpuProfilingActive() &&
            ensureTimestampQueryPool(_frame_index, required_static_queries, true);

        VkCommandBuffer static_cmd = static_command_buffers[_frame_index];

        if (vkResetCommandBuffer(static_cmd, 0) != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::bakeStaticGraph: Failed to reset static command buffer for frame {}", _frame_index},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to reset static command buffer");
        }

        VkCommandBufferBeginInfo begin_information{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pInheritanceInfo = nullptr};

        if (vkBeginCommandBuffer(static_cmd, &begin_information) != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::bakeStaticGraph: Failed to begin static command buffer for frame {}", _frame_index},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to begin static command buffer");
        }

        try
        {
            if (profiling_this_bake)
            {
                vkCmdResetQueryPool(static_cmd,
                                    timestamp_query_pools[_frame_index],
                                    0u,
                                    timestamp_query_capacities[_frame_index]);
                static_baked_with_profiling[_frame_index] = true;
                static_recorded_entries[_frame_index].reserve(split_index);
            }

            if (split_index > 0)
            {
                insertInterSubmissionComputeBarrier(static_cmd);
                recordComputeNodes(static_cmd,
                                   nodes,
                                   0,
                                   split_index,
                                   _frame_index,
                                   0,
                                   profiling_this_bake ? &static_next_query_index[_frame_index] : nullptr,
                                   profiling_this_bake ? &static_recorded_entries[_frame_index] : nullptr,
                                   profiling_this_bake ? &static_recorded_barrier_count[_frame_index] : nullptr);
                if (split_index < nodes.size() && !nodes[split_index - 1].is_barrier_required_after)
                {
                    insertComputeMemoryBarrier(static_cmd);
                    if (profiling_this_bake)
                    {
                        ++static_recorded_barrier_count[_frame_index];
                    }
                }
            }

            if (vkEndCommandBuffer(static_cmd) != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::bakeStaticGraph: Failed to end static command buffer for frame {}", _frame_index},
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                throw std::runtime_error("Failed to end static command buffer");
            }
        }
        catch (...)
        {
            vkResetCommandBuffer(static_cmd, 0);
            is_static_baked[_frame_index] = false;
            static_graph_signatures[_frame_index] = 0;
            static_split_indices[_frame_index] = 0;
            baked_buffer_handles[_frame_index].clear();
            baked_buffer_records[_frame_index].clear();
            unique_baked_buffers[_frame_index].clear();
            static_validated_buffer_epoch[_frame_index] = 0;
            static_baked_pc_snapshots[_frame_index].clear();
            static_baked_node_meta[_frame_index].clear();
            static_bake_generations[_frame_index] = 0;
            static_recorded_entries[_frame_index].clear();
            static_recorded_barrier_count[_frame_index] = 0;
            static_next_query_index[_frame_index] = 0u;
            static_baked_with_profiling[_frame_index] = false;
            throw;
        }

        static_split_indices[_frame_index] = split_index;
        static_graph_signatures[_frame_index] = _signature;

        baked_buffer_handles[_frame_index].clear();
        baked_buffer_records[_frame_index].clear();
        unique_baked_buffers[_frame_index].clear();
        static_validated_buffer_epoch[_frame_index] = 0;
        static_baked_pc_snapshots[_frame_index].clear();
        static_baked_pc_snapshots[_frame_index].reserve(split_index);
        static_baked_node_meta[_frame_index].clear();
        static_baked_node_meta[_frame_index].reserve(split_index);

        const uint64_t epoch_before_record = gpu::vector::getGlobalBufferMutationEpoch();
        std::unordered_set<const gpu::vector *> seen_buffers;

        for (size_t i = 0; i < split_index; ++i)
        {
            const auto &n = nodes[i];
            static_baked_pc_snapshots[_frame_index].push_back(n.push_constants_data);
            static_baked_node_meta[_frame_index].push_back(Static_Node_Meta{
                .pipeline_id = n.pipeline_id,
                .is_fused = n.is_fused,
                .wg_x = n.workgroup_count_x,
                .wg_y = n.workgroup_count_y,
                .wg_z = n.workgroup_count_z,
                .spec_constant_u32 = n.spec_constant_u32});
            for (size_t b_idx = 0; b_idx < n.buffers.size(); ++b_idx)
            {
                auto buf = resolveNodeBuffer(n, b_idx, _frame_index);
                if (buf && seen_buffers.insert(buf.get()).second)
                {
                    unique_baked_buffers[_frame_index].push_back(buf);
                }
                VkBuffer handle = buf ? buf->getBuffer() : VK_NULL_HANDLE;
                baked_buffer_handles[_frame_index].push_back(handle);
                baked_buffer_records[_frame_index].push_back(Baked_Buffer_Record{
                    .buffer_ref = buf,
                    .handle = handle,
                    .vector_id = buf ? buf->getId() : 0,
                    .size_bytes = buf ? buf->getSizeBytes() : 0});
            }
        }

        is_static_baked[_frame_index] = true;
        static_bake_generations[_frame_index] = next_bake_generation++;
        static_validated_buffer_epoch[_frame_index] =
            (gpu::vector::getGlobalBufferMutationEpoch() == epoch_before_record) ? epoch_before_record : 0;

        Logger::logMessage(Input_Format{"Graph_Executor::bakeStaticGraph: Successfully baked static graph for frame {} (static nodes: {}, epilogue nodes: {})",
                                        _frame_index, split_index, nodes.size() - split_index},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);
    }

bool Graph_Executor::areStaticBakedBuffersValid(uint32_t _frame_index) const noexcept
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || !is_static_baked[_frame_index])
        {
            return false;
        }
        const uint64_t current_epoch = gpu::vector::getGlobalBufferMutationEpoch();
        if (static_validated_buffer_epoch[_frame_index] == current_epoch)
        {
            return true;
        }
        for (const auto &rec : baked_buffer_records[_frame_index])
        {
            if (rec.handle != VK_NULL_HANDLE)
            {
                if (!rec.buffer_ref ||
                    rec.buffer_ref->getBuffer() != rec.handle ||
                    rec.buffer_ref->getId() != rec.vector_id ||
                    rec.buffer_ref->getSizeBytes() != rec.size_bytes)
                {
                    return false;
                }
            }
        }
        static_validated_buffer_epoch[_frame_index] = current_epoch;
        return true;
    }

bool Graph_Executor::isStaticGraphBuffersMatching(const Compute_Graph &_graph, uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || !is_static_baked[_frame_index])
        {
            return false;
        }
        if (static_baked_with_profiling[_frame_index] != isGpuProfilingActive())
        {
            VML_LOG_DEBUG(Log_Feature::STATIC_GRAPH,
                          "isStaticGraphBuffersMatching: profiling mode changed on frame {}",
                          _frame_index);
            return false;
        }

        const auto &nodes = _graph.getNodes();
        size_t split_index = static_split_indices[_frame_index];
        if (split_index > nodes.size() ||
            static_baked_pc_snapshots[_frame_index].size() != split_index ||
            static_baked_node_meta[_frame_index].size() != split_index)
        {
            VML_LOG_DEBUG(Log_Feature::STATIC_GRAPH,
                          "isStaticGraphBuffersMatching: node count mismatch on frame {} (split={}, current={})",
                          _frame_index, split_index, nodes.size());
            return false;
        }

        const auto &baked_records = baked_buffer_records[_frame_index];
        size_t handle_index = 0;

        for (size_t i = 0; i < split_index; ++i)
        {
            const auto &n = nodes[i];
            const auto &meta = static_baked_node_meta[_frame_index][i];
            if (n.pipeline_id != meta.pipeline_id ||
                n.is_fused != meta.is_fused ||
                n.workgroup_count_x != meta.wg_x ||
                n.workgroup_count_y != meta.wg_y ||
                n.workgroup_count_z != meta.wg_z ||
                n.spec_constant_u32 != meta.spec_constant_u32 ||
                n.push_constants_data != static_baked_pc_snapshots[_frame_index][i])
            {
                VML_LOG_DEBUG(Log_Feature::STATIC_GRAPH,
                              "isStaticGraphBuffersMatching: pipeline/workgroup/push_constants mismatch at node {} on frame {}",
                              i, _frame_index);
                return false;
            }
            for (size_t b_idx = 0; b_idx < n.buffers.size(); ++b_idx)
            {
                if (handle_index >= baked_records.size())
                {
                    return false;
                }
                auto buf = resolveNodeBuffer(n, b_idx, _frame_index);
                VkBuffer current_handle = buf ? buf->getBuffer() : VK_NULL_HANDLE;
                size_t current_id = buf ? buf->getId() : 0;
                VkDeviceSize current_size = buf ? buf->getSizeBytes() : 0;
                const auto &rec = baked_records[handle_index];
                if (current_handle != rec.handle ||
                    current_id != rec.vector_id ||
                    current_size != rec.size_bytes)
                {
                    Logger::logMessage(Input_Format{"isStaticGraphBuffersMatching: Mismatch at node {} (pipeline {}), buffer index {} | current_handle={}, baked_handle={}",
                                                    i, static_cast<int>(nodes[i].pipeline_id), handle_index,
                                                    reinterpret_cast<void*>(current_handle), reinterpret_cast<void*>(rec.handle)},
                                       Log_Level::LOG_DEBUG, true, 2, Log_Feature::DISPATCH_EXECUTION);
                    return false;
                }
                handle_index++;
            }
        }

        return (handle_index == baked_records.size());
    }

void Graph_Executor::executeStaticGraph(const Compute_Graph &_graph,
                            const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                            uint32_t _frame_index,
                            VkFence _external_fence,
                            bool _skip_buffer_match_check)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            return;
        }

        const auto &nodes = _graph.getNodes();
        size_t split_index = static_split_indices[_frame_index];

        if (!_skip_buffer_match_check && split_index > 0 && is_static_baked[_frame_index] && !isStaticGraphBuffersMatching(_graph, _frame_index))
        {
            compileAndExecute(_graph, _transfer_tasks, _frame_index, _external_fence);
            return;
        }

        has_pending_timestamps[_frame_index] = false;
        frame_timestamp_entries[_frame_index].clear();
        frame_barrier_count[_frame_index] = 0;

        std::array<VkCommandBuffer, 3> submit_command_buffers{};
        uint32_t submit_count = 0;
        const bool profile_active = isGpuProfilingActive();
        uint32_t query_cursor = 0u;
        bool static_cmd_submitted = false;

        if (!_transfer_tasks.empty())
        {
            VkCommandBuffer transfer_cmd = transfer_command_buffers[_frame_index];

            if (vkResetCommandBuffer(transfer_cmd, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to reset transfer command buffer");
            }

            VkCommandBufferBeginInfo begin_information{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr};

            if (vkBeginCommandBuffer(transfer_cmd, &begin_information) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin transfer command buffer");
            }

            recordTransferTasks(transfer_cmd, _transfer_tasks);

            if (vkEndCommandBuffer(transfer_cmd) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end transfer command buffer");
            }

            submit_command_buffers[submit_count++] = transfer_cmd;
        }

        if (split_index > 0 && is_static_baked[_frame_index])
        {
            for (const auto &buf : unique_baked_buffers[_frame_index])
            {
                buf->markAsUsedInFrame(_frame_index);
            }
            submit_command_buffers[submit_count++] = static_command_buffers[_frame_index];
            static_cmd_submitted = true;
            if (profile_active && static_baked_with_profiling[_frame_index])
            {
                frame_timestamp_entries[_frame_index] = static_recorded_entries[_frame_index];
                frame_barrier_count[_frame_index] = static_recorded_barrier_count[_frame_index];
                query_cursor = static_next_query_index[_frame_index];
            }
        }

        if (split_index < nodes.size())
        {
            VkCommandBuffer epilogue_cmd = epilogue_command_buffers[_frame_index];
            const uint32_t epilogue_queries = static_cast<uint32_t>((nodes.size() - split_index) * 2u);
            const bool reset_in_epilogue = profile_active && !static_cmd_submitted &&
                                           ensureTimestampQueryPool(_frame_index, epilogue_queries, true);
            const bool profile_epilogue = profile_active &&
                                          (static_cmd_submitted ? static_baked_with_profiling[_frame_index] : reset_in_epilogue);

            if (vkResetCommandBuffer(epilogue_cmd, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to reset epilogue command buffer");
            }

            VkCommandBufferBeginInfo begin_information{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr};

            if (vkBeginCommandBuffer(epilogue_cmd, &begin_information) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin epilogue command buffer");
            }

            if (reset_in_epilogue)
            {
                vkCmdResetQueryPool(epilogue_cmd,
                                    timestamp_query_pools[_frame_index],
                                    0u,
                                    epilogue_queries);
            }

            if (!static_cmd_submitted && _transfer_tasks.empty())
            {
                insertInterSubmissionComputeBarrier(epilogue_cmd);
            }

            recordComputeNodes(epilogue_cmd,
                               nodes,
                               split_index,
                               nodes.size(),
                               _frame_index,
                               0,
                               profile_epilogue ? &query_cursor : nullptr,
                               profile_epilogue ? &frame_timestamp_entries[_frame_index] : nullptr,
                               profile_epilogue ? &frame_barrier_count[_frame_index] : nullptr);

            if (vkEndCommandBuffer(epilogue_cmd) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end epilogue command buffer");
            }

            submit_command_buffers[submit_count++] = epilogue_cmd;
        }

        if (submit_count > 0)
        {
            bool has_separate_external_fence = (_external_fence != VK_NULL_HANDLE && _external_fence != context.getFrameFence(_frame_index));

            VkSubmitInfo submit_information{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = submit_count,
                .pCommandBuffers = submit_command_buffers.data(),
                .signalSemaphoreCount = has_separate_external_fence ? 1u : 0u,
                .pSignalSemaphores = has_separate_external_fence ? &external_fence_semaphores[_frame_index] : nullptr};

            VkFence primary_fence = (_external_fence != VK_NULL_HANDLE) ? _external_fence : context.getFrameFence(_frame_index);

            if (has_separate_external_fence)
            {
                if (vkResetFences(context.getDevice(), 1, &_external_fence) != VK_SUCCESS)
                {
                    throw std::runtime_error("Graph_Executor::executeStaticGraph: Failed to reset external fence");
                }
            }
            else
            {
                context.resetFrameFence(_frame_index);
            }

            if (vkQueueSubmit(context.getComputeQueue(), 1, &submit_information, primary_fence) != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::executeStaticGraph: Failed to submit command buffer for frame {}", _frame_index},
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                throw std::runtime_error("Failed to submit command buffer");
            }
            has_pending_timestamps[_frame_index] = !frame_timestamp_entries[_frame_index].empty();
            dynamic_pool_dirty[_frame_index] = true;

            if (has_separate_external_fence)
            {
                context.resetFrameFence(_frame_index);
                VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                VkSubmitInfo frame_fence_submit{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .waitSemaphoreCount = 1,
                    .pWaitSemaphores = &external_fence_semaphores[_frame_index],
                    .pWaitDstStageMask = &wait_stage,
                    .commandBufferCount = 0,
                    .pCommandBuffers = nullptr,
                    .signalSemaphoreCount = 0,
                    .pSignalSemaphores = nullptr};

                if (vkQueueSubmit(context.getComputeQueue(), 1, &frame_fence_submit, context.getFrameFence(_frame_index)) != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to submit frame fence chain");
                }
            }
        }
    }

void Graph_Executor::executeStaticGraphReplay(const Compute_Graph &_optimizer_graph,
                                  const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                                  uint32_t _frame_index,
                                  VkFence _external_fence)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || !is_static_baked[_frame_index])
        {
            return;
        }

        has_pending_timestamps[_frame_index] = false;
        frame_timestamp_entries[_frame_index].clear();
        frame_barrier_count[_frame_index] = 0;

        std::array<VkCommandBuffer, 3> submit_command_buffers{};
        uint32_t submit_count = 0;
        const bool profile_active = isGpuProfilingActive() && static_baked_with_profiling[_frame_index];
        uint32_t query_cursor = 0u;
        if (profile_active)
        {
            frame_timestamp_entries[_frame_index] = static_recorded_entries[_frame_index];
            frame_barrier_count[_frame_index] = static_recorded_barrier_count[_frame_index];
            query_cursor = static_next_query_index[_frame_index];
        }

        if (!_transfer_tasks.empty())
        {
            VkCommandBuffer transfer_cmd = transfer_command_buffers[_frame_index];

            if (vkResetCommandBuffer(transfer_cmd, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to reset transfer command buffer");
            }

            VkCommandBufferBeginInfo begin_information{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr};

            if (vkBeginCommandBuffer(transfer_cmd, &begin_information) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin transfer command buffer");
            }

            recordTransferTasks(transfer_cmd, _transfer_tasks);

            if (vkEndCommandBuffer(transfer_cmd) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end transfer command buffer");
            }

            submit_command_buffers[submit_count++] = transfer_cmd;
        }

        for (const auto &buf : unique_baked_buffers[_frame_index])
        {
            buf->markAsUsedInFrame(_frame_index);
        }
        submit_command_buffers[submit_count++] = static_command_buffers[_frame_index];

        const auto &opt_nodes = _optimizer_graph.getNodes();
        if (!opt_nodes.empty())
        {
            VkCommandBuffer epilogue_cmd = epilogue_command_buffers[_frame_index];

            if (vkResetCommandBuffer(epilogue_cmd, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to reset epilogue command buffer");
            }

            VkCommandBufferBeginInfo begin_information{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr};

            if (vkBeginCommandBuffer(epilogue_cmd, &begin_information) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin epilogue command buffer");
            }

            recordComputeNodes(epilogue_cmd,
                               opt_nodes,
                               0,
                               opt_nodes.size(),
                               _frame_index,
                               static_split_indices[_frame_index],
                               profile_active ? &query_cursor : nullptr,
                               profile_active ? &frame_timestamp_entries[_frame_index] : nullptr,
                               profile_active ? &frame_barrier_count[_frame_index] : nullptr);

            if (vkEndCommandBuffer(epilogue_cmd) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end epilogue command buffer");
            }

            submit_command_buffers[submit_count++] = epilogue_cmd;
        }

        if (submit_count > 0)
        {
            bool has_separate_external_fence = (_external_fence != VK_NULL_HANDLE && _external_fence != context.getFrameFence(_frame_index));

            VkSubmitInfo submit_information{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = submit_count,
                .pCommandBuffers = submit_command_buffers.data(),
                .signalSemaphoreCount = has_separate_external_fence ? 1u : 0u,
                .pSignalSemaphores = has_separate_external_fence ? &external_fence_semaphores[_frame_index] : nullptr};

            VkFence primary_fence = (_external_fence != VK_NULL_HANDLE) ? _external_fence : context.getFrameFence(_frame_index);

            if (has_separate_external_fence)
            {
                if (vkResetFences(context.getDevice(), 1, &_external_fence) != VK_SUCCESS)
                {
                    throw std::runtime_error("Graph_Executor::executeStaticGraphReplay: Failed to reset external fence");
                }
            }
            else
            {
                context.resetFrameFence(_frame_index);
            }

            if (vkQueueSubmit(context.getComputeQueue(), 1, &submit_information, primary_fence) != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Graph_Executor::executeStaticGraphReplay: Failed to submit command buffer for frame {}", _frame_index},
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                throw std::runtime_error("Failed to submit command buffer");
            }
            has_pending_timestamps[_frame_index] = !frame_timestamp_entries[_frame_index].empty();
            dynamic_pool_dirty[_frame_index] = true;

            if (has_separate_external_fence)
            {
                context.resetFrameFence(_frame_index);
                VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                VkSubmitInfo frame_fence_submit{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .waitSemaphoreCount = 1,
                    .pWaitSemaphores = &external_fence_semaphores[_frame_index],
                    .pWaitDstStageMask = &wait_stage,
                    .commandBufferCount = 0,
                    .pCommandBuffers = nullptr,
                    .signalSemaphoreCount = 0,
                    .pSignalSemaphores = nullptr};

                if (vkQueueSubmit(context.getComputeQueue(), 1, &frame_fence_submit, context.getFrameFence(_frame_index)) != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to submit frame fence chain");
                }
            }
        }
    }

void Graph_Executor::warmupPipelineCache(Compute_Graph &_graph)
    {
        auto &nodes = _graph.getNodes();
        if (nodes.empty())
        {
            return;
        }

        Logger::logMessage(Input_Format{"Graph_Executor::warmupPipelineCache: Pre-compiling pipelines for {} nodes", nodes.size()},
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
                    if (node.fusion_plan && !node.fusion_plan->canonical_key.empty())
                    {
                        node.cached_pipeline = pipeline_cache_manager.getOrCreatePipeline(
                            node.fusion_plan->canonical_key, node.fused_glsl_code, node.fusion_plan->spec_constants);
                    }
                    else
                    {
                        node.cached_pipeline = pipeline_cache_manager.getOrCreatePipeline(node.fused_glsl_code);
                    }
                }
                catch (const std::exception &exception)
                {
                    Logger::logMessage(Input_Format{"Graph_Executor::warmupPipelineCache: Pre-compilation failed ({}), fallback will be used at runtime", exception.what()},
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::SHADER_GENERATION | Log_Feature::DISPATCH_EXECUTION);
                }
            }
            else
            {
                if (node.spec_constant_u32 == 0u)
                {
                    network.getPipeline(node.pipeline_id);
                }
                else
                {
                    network.getSpecializedPipeline(node.pipeline_id, node.spec_constant_u32);
                }
            }
        }
    }

bool Graph_Executor::ensureTimestampQueryPool(uint32_t _frame_index, uint32_t _required_queries, bool _allow_grow)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || !isGpuProfilingActive())
        {
            return false;
        }

        VkDevice device = context.getDevice();
        if (device == VK_NULL_HANDLE)
        {
            return false;
        }

        if (timestamp_query_pools[_frame_index] != VK_NULL_HANDLE &&
            timestamp_query_capacities[_frame_index] >= _required_queries)
        {
            return true;
        }

        if (timestamp_query_pools[_frame_index] != VK_NULL_HANDLE && !_allow_grow)
        {
            return false;
        }

        const uint32_t new_capacity = std::max<uint32_t>({
            INITIAL_TIMESTAMP_QUERY_CAPACITY,
            _required_queries,
            timestamp_query_capacities[_frame_index] * 2u});

        if (timestamp_query_pools[_frame_index] != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);
            is_static_baked[_frame_index] = false;
            static_graph_signatures[_frame_index] = 0;
            static_split_indices[_frame_index] = 0;
            baked_buffer_handles[_frame_index].clear();
            baked_buffer_records[_frame_index].clear();
            unique_baked_buffers[_frame_index].clear();
            static_validated_buffer_epoch[_frame_index] = 0;
            static_baked_pc_snapshots[_frame_index].clear();
            static_baked_node_meta[_frame_index].clear();
            static_bake_generations[_frame_index] = 0;
            static_recorded_entries[_frame_index].clear();
            static_recorded_barrier_count[_frame_index] = 0;
            static_next_query_index[_frame_index] = 0u;
            static_baked_with_profiling[_frame_index] = false;

            vkDestroyQueryPool(device, timestamp_query_pools[_frame_index], nullptr);
            timestamp_query_pools[_frame_index] = VK_NULL_HANDLE;
            timestamp_query_capacities[_frame_index] = 0u;
        }

        VkQueryPoolCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = new_capacity,
            .pipelineStatistics = 0};

        VkResult create_res = vkCreateQueryPool(device, &create_info, nullptr, &timestamp_query_pools[_frame_index]);
        if (create_res != VK_SUCCESS || timestamp_query_pools[_frame_index] == VK_NULL_HANDLE)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::ensureTimestampQueryPool: vkCreateQueryPool failed (VkResult={})", static_cast<int>(create_res)},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            timestamp_query_pools[_frame_index] = VK_NULL_HANDLE;
            timestamp_query_capacities[_frame_index] = 0u;
            return false;
        }

        timestamp_query_capacities[_frame_index] = new_capacity;
        return true;
    }

std::string Graph_Executor::makeNodeProfileLabel(const Compute_Node &_node, bool _is_fused_successful)
    {
        if (_node.is_fused && _node.fused_operations.size() > 1)
        {
            if (_node.fusion_plan)
            {
                if (_node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
                {
                    return _is_fused_successful
                        ? std::format("FUSED_ADAM_MULTI{}_FP16", _node.fusion_plan->horizontal_lanes.size())
                        : std::format("FALLBACK_ADAM_MULTI{}", _node.fusion_plan->horizontal_lanes.size());
                }
                std::string chain;
                for (size_t i = 0; i < _node.fused_operations.size(); ++i)
                {
                    if (i > 0) chain += "+";
                    chain += magic_enum::enum_name(_node.fused_operations[i].pipeline_id);
                }
                if (!_is_fused_successful) return "FALLBACK(" + chain + ")";
                if (_node.fusion_plan->anchor_role == Fused_Anchor_Role::GEMM_COOPMAT) return "FUSED_GEMM(" + chain + ")";
                if (_node.fusion_plan->anchor_role == Fused_Anchor_Role::ROW_REDUCTION) return "FUSED_ROWRED(" + chain + ")";
                if (_node.fusion_plan->anchor_role == Fused_Anchor_Role::POINTWISE_CHAIN) return "FUSED_PW(" + chain + ")";
            }
            std::string label = _is_fused_successful ? "FUSED(" : "FALLBACK(";
            for (size_t i = 0; i < _node.fused_operations.size(); ++i)
            {
                if (i > 0)
                {
                    label += "+";
                }
                label += magic_enum::enum_name(_node.fused_operations[i].pipeline_id);
            }
            label += ")";
            return label;
        }
        return std::string(magic_enum::enum_name(_node.pipeline_id));
    }

void Graph_Executor::setGpuProfilingEnabled(bool _enable)
    {
        if (_enable && !context.isTimestampQuerySupported())
        {
            Logger::logMessage("Graph_Executor::setGpuProfilingEnabled: Device queue family does not support Vulkan timestamp queries",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            is_gpu_profiling_enabled = false;
            return;
        }

        if (is_gpu_profiling_enabled == _enable)
        {
            return;
        }

        VkDevice device = context.getDevice();
        if (device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device);
        }

        resolveAllPendingTimestamps();
        is_gpu_profiling_enabled = _enable;
        invalidateStaticGraph();

        if (_enable)
        {
            for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
            {
                ensureTimestampQueryPool(f, INITIAL_TIMESTAMP_QUERY_CAPACITY, true);
            }
        }
    }

void Graph_Executor::resolveFrameTimestamps(uint32_t _frame_index)
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT || !has_pending_timestamps[_frame_index])
        {
            return;
        }

        has_pending_timestamps[_frame_index] = false;
        std::vector<Recorded_Timestamp_Entry> entries;
        entries.swap(frame_timestamp_entries[_frame_index]);
        const uint64_t barriers = frame_barrier_count[_frame_index];
        frame_barrier_count[_frame_index] = 0;

        if (entries.empty() ||
            timestamp_query_pools[_frame_index] == VK_NULL_HANDLE ||
            !context.isTimestampQuerySupported())
        {
            return;
        }

        uint32_t max_query = 0u;
        for (const auto &entry : entries)
        {
            max_query = std::max<uint32_t>(max_query, std::max(entry.start_query, entry.end_query) + 1u);
        }
        if (max_query == 0u || max_query > timestamp_query_capacities[_frame_index])
        {
            return;
        }

        VkDevice device = context.getDevice();
        if (device == VK_NULL_HANDLE)
        {
            return;
        }

        std::vector<uint64_t> raw_timestamps(max_query, 0ull);
        VkResult query_res = vkGetQueryPoolResults(
            device,
            timestamp_query_pools[_frame_index],
            0u,
            max_query,
            raw_timestamps.size() * sizeof(uint64_t),
            raw_timestamps.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        if (query_res != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::resolveFrameTimestamps: vkGetQueryPoolResults failed (VkResult={})", static_cast<int>(query_res)},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return;
        }

        const uint32_t valid_bits = context.getTimestampValidBits();
        const uint64_t mask = (valid_bits >= 64u) ? UINT64_MAX : ((1ULL << valid_bits) - 1ULL);
        const uint64_t half_range = mask >> 1;
        const double ms_per_tick = static_cast<double>(context.getTimestampPeriodNs()) * 1e-6;

        auto elapsedMs = [&](uint32_t _start_q, uint32_t _end_q) -> double
        {
            uint64_t delta = (raw_timestamps[_end_q] - raw_timestamps[_start_q]) & mask;
            if (delta > half_range)
            {
                delta = 0ull;
            }
            return static_cast<double>(delta) * ms_per_tick;
        };

        const double submit_span_ms = elapsedMs(entries.front().start_query, entries.back().end_query);
        double submit_kernel_ms = 0.0;

        std::lock_guard<std::mutex> lock(gpu_profile_mutex);
        for (const auto &entry : entries)
        {
            const double dur_ms = elapsedMs(entry.start_query, entry.end_query);
            auto &stat = aggregated_kernel_stats[entry.label];
            if (stat.name.empty())
            {
                stat.name = entry.label;
            }
            stat.dispatch_count += 1;
            stat.total_ms += dur_ms;
            stat.max_ms = std::max(stat.max_ms, dur_ms);
            submit_kernel_ms += dur_ms;
        }

        aggregated_total_kernel_ms += submit_kernel_ms;
        aggregated_total_span_ms += submit_span_ms;
        aggregated_total_dispatches += static_cast<uint64_t>(entries.size());
        aggregated_total_barriers += barriers;
        aggregated_profiled_submits += 1;
    }

void Graph_Executor::resolveAllPendingTimestamps()
    {
        for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
        {
            resolveFrameTimestamps(f);
        }
    }

Gpu_Profile_Summary Graph_Executor::getGpuProfileSummary() const
    {
        std::lock_guard<std::mutex> lock(gpu_profile_mutex);
        Gpu_Profile_Summary summary;
        summary.total_kernel_ms = aggregated_total_kernel_ms;
        summary.total_span_ms = aggregated_total_span_ms;
        summary.total_dispatches = aggregated_total_dispatches;
        summary.total_barriers = aggregated_total_barriers;
        summary.profiled_submits = aggregated_profiled_submits;
        summary.timestamp_period_ns = context.getTimestampPeriodNs();
        summary.timestamp_valid_bits = context.getTimestampValidBits();

        summary.kernels.reserve(aggregated_kernel_stats.size());
        for (const auto &[label, stat] : aggregated_kernel_stats)
        {
            summary.kernels.push_back(stat);
        }
        std::sort(summary.kernels.begin(), summary.kernels.end(),
                  [](const Gpu_Kernel_Stat &a, const Gpu_Kernel_Stat &b)
                  {
                      if (a.total_ms != b.total_ms)
                      {
                          return a.total_ms > b.total_ms;
                      }
                      return a.dispatch_count > b.dispatch_count;
                  });
        return summary;
    }

void Graph_Executor::resetGpuProfileStats()
    {
        std::lock_guard<std::mutex> lock(gpu_profile_mutex);
        aggregated_kernel_stats.clear();
        aggregated_total_kernel_ms = 0.0;
        aggregated_total_span_ms = 0.0;
        aggregated_total_dispatches = 0;
        aggregated_total_barriers = 0;
        aggregated_profiled_submits = 0;
    }

std::string Graph_Executor::formatGpuProfileTable(size_t _top_k, size_t _batch_divisor) const
    {
        const Gpu_Profile_Summary summary = getGpuProfileSummary();
        if (summary.profiled_submits == 0 && summary.kernels.empty())
        {
            return "[GPU Timestamp Profile] No completed GPU timestamp samples collected.\n";
        }

        const double div = static_cast<double>(std::max<size_t>(1, _batch_divisor));
        const double denom_ms = (summary.total_kernel_ms > 0.0) ? summary.total_kernel_ms : 1.0;

        auto formatName = [](const std::string &name) -> std::string
        {
            constexpr size_t max_len = 42;
            if (name.size() <= max_len)
            {
                return name;
            }
            return name.substr(0, max_len - 3) + "...";
        };

        std::ostringstream oss;
        oss << "+--------------------------------------------------------------------------------------+\n";
        oss << "| Vulkan GPU Hardware Timestamp Profile (Avg/step over " << _batch_divisor << " step"
            << (_batch_divisor == 1 ? "" : "s") << ", submits=" << summary.profiled_submits
            << ", tick=" << std::fixed << std::setprecision(1) << summary.timestamp_period_ns << " ns)\n";
        oss << "+--------------------------------------------+---------+-----------+----------+--------+\n";
        oss << "| Pipeline / Fused Chain                     | Disp/b  | Time (ms) | Avg (us) |  % GPU |\n";
        oss << "+--------------------------------------------+---------+-----------+----------+--------+\n";

        const size_t limit = (_top_k == 0) ? summary.kernels.size() : std::min(_top_k, summary.kernels.size());
        for (size_t i = 0; i < limit; ++i)
        {
            const auto &k = summary.kernels[i];
            const double disp_per_batch = static_cast<double>(k.dispatch_count) / div;
            const double time_ms = k.total_ms / div;
            const double avg_us = (k.dispatch_count > 0) ? (k.total_ms * 1000.0 / static_cast<double>(k.dispatch_count)) : 0.0;
            const double pct = (k.total_ms / denom_ms) * 100.0;

            oss << "| " << std::left << std::setw(42) << formatName(k.name) << " | "
                << std::right << std::fixed << std::setprecision(1) << std::setw(7) << disp_per_batch << " | "
                << std::setprecision(3) << std::setw(9) << time_ms << " | "
                << std::setprecision(2) << std::setw(8) << avg_us << " | "
                << std::setprecision(1) << std::setw(5) << pct << "% |\n";
        }

        if (summary.kernels.size() > limit)
        {
            uint64_t other_dispatches = 0;
            double other_ms = 0.0;
            for (size_t i = limit; i < summary.kernels.size(); ++i)
            {
                other_dispatches += summary.kernels[i].dispatch_count;
                other_ms += summary.kernels[i].total_ms;
            }
            const double other_disp_per_b = static_cast<double>(other_dispatches) / div;
            const double other_time_ms = other_ms / div;
            const double other_avg_us = (other_dispatches > 0) ? (other_ms * 1000.0 / static_cast<double>(other_dispatches)) : 0.0;
            const double other_pct = (other_ms / denom_ms) * 100.0;
            const std::string other_label = "(Other " + std::to_string(summary.kernels.size() - limit) + " kernels)";

            oss << "| " << std::left << std::setw(42) << formatName(other_label) << " | "
                << std::right << std::fixed << std::setprecision(1) << std::setw(7) << other_disp_per_b << " | "
                << std::setprecision(3) << std::setw(9) << other_time_ms << " | "
                << std::setprecision(2) << std::setw(8) << other_avg_us << " | "
                << std::setprecision(1) << std::setw(5) << other_pct << "% |\n";
        }

        const double avg_kernel_ms = summary.total_kernel_ms / div;
        const double avg_span_ms = summary.total_span_ms / div;
        const double gap_ms = summary.total_span_ms - summary.total_kernel_ms;
        const double avg_gap_ms = gap_ms / div;
        const double avg_dispatches = static_cast<double>(summary.total_dispatches) / div;
        const double avg_barriers = static_cast<double>(summary.total_barriers) / div;

        oss << "+--------------------------------------------+---------+-----------+----------+--------+\n";
        oss << "| " << std::left << std::setw(42) << "Sum of Kernels" << " | "
            << std::right << std::fixed << std::setprecision(1) << std::setw(7) << avg_dispatches << " | "
            << std::setprecision(3) << std::setw(9) << avg_kernel_ms << " | "
            << std::setw(8) << "-" << " | "
            << "100.0% |\n";
        oss << "| " << std::left << std::setw(42) << "Total GPU Active Span" << " | "
            << std::right << std::setw(7) << "-" << " | "
            << std::fixed << std::setprecision(3) << std::setw(9) << avg_span_ms << " | "
            << std::setw(8) << "-" << " | "
            << std::setw(6) << "-" << " |\n";
        oss << "| " << std::left << std::setw(42) << "Inter-Node Barrier / Overlap Gap" << " | "
            << std::right << std::setw(7) << "-" << " | "
            << std::fixed << std::setprecision(3) << std::setw(9) << avg_gap_ms << " | "
            << std::setw(8) << "-" << " | "
            << std::setw(6) << "-" << " |\n";
        oss << "| " << std::left << std::setw(42) << "Compute Barriers Emitted (avg/step)" << " | "
            << std::right << std::fixed << std::setprecision(1) << std::setw(7) << avg_barriers << " | "
            << std::setw(9) << "-" << " | "
            << std::setw(8) << "-" << " | "
            << std::setw(6) << "-" << " |\n";
        oss << "+--------------------------------------------+---------+-----------+----------+--------+\n";

        return oss.str();
    }

void Graph_Executor::logGpuProfileTable(size_t _top_k, size_t _batch_divisor, Log_Level _level, bool _print_to_console) const
    {
        if (!Logger::shouldLog(_level, Log_Feature::PROFILER))
        {
            return;
        }
        std::string table = formatGpuProfileTable(_top_k, _batch_divisor);
        while (!table.empty() && table.back() == '\n')
        {
            table.pop_back();
        }
        if (!table.empty() && table.front() == '+')
        {
            table.insert(table.begin(), '\n');
        }
        Logger::logMessage(table, _level, _print_to_console, 0, Log_Feature::PROFILER);
    }

void Graph_Executor::printGpuProfileTable(size_t _top_k, size_t _batch_divisor) const
    {
        logGpuProfileTable(_top_k, _batch_divisor, Log_Level::LOG_INFO, true);
    }

void Graph_Executor::printGpuProfileTable(std::ostream &_os, size_t _top_k, size_t _batch_divisor) const
    {
        _os << formatGpuProfileTable(_top_k, _batch_divisor);
    }

VkCommandBuffer Graph_Executor::getCommandBuffer(uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::getCommandBuffer: frame_index out of bounds ({})", _frame_index},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return VK_NULL_HANDLE;
        }
        return command_buffers[_frame_index];
    }

VkDescriptorPool Graph_Executor::getDescriptorPool(uint32_t _frame_index) const
    {
        if (_frame_index >= MAX_FRAMES_IN_FLIGHT)
        {
            Logger::logMessage(Input_Format{"Graph_Executor::getDescriptorPool: frame_index out of bounds ({})", _frame_index},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return VK_NULL_HANDLE;
        }
        return descriptor_pools[_frame_index];
    }

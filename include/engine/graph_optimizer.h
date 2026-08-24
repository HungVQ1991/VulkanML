#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "compute_node.h"
#include "gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "shader_dictionary.h"

#ifndef ENABLE_SHADER_FUSION
#define ENABLE_SHADER_FUSION 0
#endif

struct Buffer_Binding_Mapping
{
    std::uint32_t raw_node_index = 0;
    std::uint32_t raw_buffer_index = 0;
    std::uint32_t fused_buffer_index = 0;
};

struct Push_Constant_Mapping
{
    std::uint32_t raw_node_index = 0;
    std::uint32_t fused_push_constants_offset = 0;
    std::uint32_t push_constants_size = 0;
};

struct Cached_Graph_Template
{
    std::vector<Compute_Node> fused_nodes;
    std::vector<std::vector<Buffer_Binding_Mapping>> buffer_mappings;
    std::vector<std::vector<Push_Constant_Mapping>> push_constants_mappings;
    std::vector<std::vector<std::uint32_t>> raw_node_indices;
    bool is_valid = false;

    [[nodiscard]] const std::vector<Compute_Node> &getFusedNodes() const noexcept
    {
        return fused_nodes;
    }

    [[nodiscard]] const std::vector<std::vector<Buffer_Binding_Mapping>> &getBufferMappings() const noexcept
    {
        return buffer_mappings;
    }

    [[nodiscard]] const std::vector<std::vector<Push_Constant_Mapping>> &getPushConstantsMappings() const noexcept
    {
        return push_constants_mappings;
    }

    [[nodiscard]] const std::vector<std::vector<std::uint32_t>> &getRawNodeIndices() const noexcept
    {
        return raw_node_indices;
    }

    [[nodiscard]] bool isValid() const noexcept
    {
        return is_valid;
    }

    void clear() noexcept
    {
        fused_nodes.clear();
        buffer_mappings.clear();
        push_constants_mappings.clear();
        raw_node_indices.clear();
        is_valid = false;
    }
};

class Graph_Optimizer
{
private:
    static constexpr std::size_t MAX_PUSH_CONSTANTS_BYTES = 128;
    static constexpr std::size_t MAX_STORAGE_BUFFER_BINDINGS = 32;
    static constexpr std::size_t MAX_FUSED_OPERATIONS = 8;

    static constexpr std::size_t alignTo4Bytes(std::size_t offset) noexcept
    {
        return (offset + 3) & ~std::size_t(3);
    }

    static bool isOptimizerNode(const Compute_Node &node) noexcept
    {
        if (node.is_fused)
        {
            return false;
        }
        return node.pipeline_id == Compute_Pipeline::ADAM_UPDATE ||
               node.pipeline_id == Compute_Pipeline::SGD_UPDATE;
    }

    static void assignPipelineBarriers(std::vector<Compute_Node> &nodes)
    {
        if (nodes.empty())
        {
            return;
        }

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            if (i == nodes.size() - 1)
            {
                nodes[i].is_barrier_required_after = true;
                continue;
            }

            bool current_is_optimizer = isOptimizerNode(nodes[i]);
            bool next_is_optimizer = isOptimizerNode(nodes[i + 1]);

            if (current_is_optimizer && next_is_optimizer)
            {
                nodes[i].is_barrier_required_after = false;
            }
            else
            {
                nodes[i].is_barrier_required_after = true;
            }
        }
    }

    static bool isFusible(
        Operation_Class producer_class,
        Operation_Class consumer_class,
        Compute_Pipeline producer_pipeline,
        Compute_Pipeline consumer_pipeline)
    {
        if (consumer_class != Operation_Class::ELEMENTWISE)
        {
            return false;
        }

        const auto &shader_dictionary = Shader_Dictionary::getInstance();
        const auto &consumer_metadata = shader_dictionary.getMetadata(consumer_pipeline);
        const auto &producer_metadata = shader_dictionary.getMetadata(producer_pipeline);

        if (consumer_metadata.is_writing_multiple_elements || consumer_metadata.shared_memory_size > 0)
        {
            return false;
        }

        if (producer_metadata.operation_class == Operation_Class::STANDALONE ||
            producer_metadata.is_writing_multiple_elements ||
            producer_metadata.output_count > 1)
        {
            return false;
        }

        return (producer_class == Operation_Class::MATRIX_2D ||
                producer_class == Operation_Class::ELEMENTWISE ||
                producer_class == Operation_Class::TENSOR_3D);
    }

    static bool hasCompatibleDimensions(
        const Compute_Node &producer_node,
        const Compute_Node &consumer_node,
        Operation_Class producer_class,
        Operation_Class consumer_class) noexcept
    {
        if (producer_node.workgroup_count_x == consumer_node.workgroup_count_x &&
            producer_node.workgroup_count_y == consumer_node.workgroup_count_y &&
            producer_node.workgroup_count_z == consumer_node.workgroup_count_z)
        {
            return true;
        }

        if (consumer_class == Operation_Class::ELEMENTWISE)
        {
            if (producer_class == Operation_Class::MATRIX_2D || producer_class == Operation_Class::TENSOR_3D)
            {
                std::uint64_t total_workgroups_producer = static_cast<std::uint64_t>(producer_node.workgroup_count_x) *
                                                          producer_node.workgroup_count_y *
                                                          producer_node.workgroup_count_z;
                std::uint64_t total_workgroups_consumer = static_cast<std::uint64_t>(consumer_node.workgroup_count_x) *
                                                          consumer_node.workgroup_count_y *
                                                          consumer_node.workgroup_count_z;

                return (total_workgroups_producer >= total_workgroups_consumer);
            }
        }

        return false;
    }

    static std::uint32_t findOrAddBuffer(Compute_Node &node, const std::shared_ptr<gpu::vector> &target_buffer)
    {
        for (std::uint32_t index = 0; index < node.buffers.size(); ++index)
        {
            const auto &existing_buffer = node.buffers[index];
            if (target_buffer && existing_buffer &&
                (target_buffer == existing_buffer ||
                 (target_buffer->getBuffer() != VK_NULL_HANDLE &&
                  target_buffer->getBuffer() == existing_buffer->getBuffer())))
            {
                return index;
            }
        }
        node.buffers.push_back(target_buffer);
        return static_cast<std::uint32_t>(node.buffers.size() - 1);
    }

    static void updateDispatchGrid(
        Compute_Node &fused_node,
        const Compute_Node &next_node,
        Operation_Class producer_class,
        Operation_Class consumer_class)
    {
        const auto &shader_dictionary = Shader_Dictionary::getInstance();
        const auto &next_metadata = shader_dictionary.getMetadata(next_node.pipeline_id);

        if (next_metadata.shared_memory_size > 0 || consumer_class == Operation_Class::STANDALONE)
        {
            fused_node.workgroup_count_x = next_node.workgroup_count_x;
            fused_node.workgroup_count_y = 1;
            fused_node.workgroup_count_z = 1;
            return;
        }

        if (producer_class == Operation_Class::MATRIX_2D ||
            producer_class == Operation_Class::TENSOR_3D ||
            producer_class == Operation_Class::STANDALONE)
        {
            return;
        }

        fused_node.workgroup_count_x = std::max(fused_node.workgroup_count_x, next_node.workgroup_count_x);
        fused_node.workgroup_count_y = std::max(fused_node.workgroup_count_y, next_node.workgroup_count_y);
        fused_node.workgroup_count_z = std::max(fused_node.workgroup_count_z, next_node.workgroup_count_z);
    }

    static Fused_Operation buildFusedOperation(
        Compute_Node &fused_node,
        const Compute_Node &source_node,
        std::uint32_t push_constants_offset,
        const Snippet_Metadata &metadata,
        std::uint32_t raw_node_index = 0,
        std::vector<Buffer_Binding_Mapping> *output_buffer_mappings = nullptr)
    {
        Fused_Operation operation{
            .pipeline_id = source_node.pipeline_id,
            .push_constants_offset = push_constants_offset,
            .push_constants_size = static_cast<std::uint32_t>(source_node.push_constants_data.size()),
            .input_buffer_indices = {},
            .output_buffer_indices = {},
            .workgroup_count_x = source_node.workgroup_count_x,
            .workgroup_count_y = source_node.workgroup_count_y,
            .workgroup_count_z = source_node.workgroup_count_z};

        for (std::uint32_t buffer_index = 0; buffer_index < metadata.input_count && buffer_index < source_node.buffers.size(); ++buffer_index)
        {
            std::uint32_t fused_buffer_index = findOrAddBuffer(fused_node, source_node.buffers[buffer_index]);
            operation.input_buffer_indices.push_back(fused_buffer_index);
            if (output_buffer_mappings)
            {
                output_buffer_mappings->push_back(Buffer_Binding_Mapping{
                    .raw_node_index = raw_node_index,
                    .raw_buffer_index = buffer_index,
                    .fused_buffer_index = fused_buffer_index});
            }
        }

        for (std::uint32_t buffer_index = metadata.input_count;
             buffer_index < metadata.input_count + metadata.output_count && buffer_index < source_node.buffers.size();
             ++buffer_index)
        {
            std::uint32_t fused_buffer_index = findOrAddBuffer(fused_node, source_node.buffers[buffer_index]);
            operation.output_buffer_indices.push_back(fused_buffer_index);
            if (output_buffer_mappings)
            {
                output_buffer_mappings->push_back(Buffer_Binding_Mapping{
                    .raw_node_index = raw_node_index,
                    .raw_buffer_index = buffer_index,
                    .fused_buffer_index = fused_buffer_index});
            }

            std::uint32_t local_output_index = buffer_index - metadata.input_count;
            bool is_declared_persistent = std::find(metadata.persistent_output_indices.begin(),
                                                    metadata.persistent_output_indices.end(),
                                                    local_output_index) != metadata.persistent_output_indices.end();

            if (source_node.external_output_indices.contains(buffer_index) || is_declared_persistent)
            {
                fused_node.external_output_indices.insert(fused_buffer_index);
            }
        }

        return operation;
    }

    static bool hasAliasingHazard(
        const Compute_Node &fused_node,
        const Compute_Node &next_node,
        const Shader_Dictionary &shader_dictionary)
    {
        const Snippet_Metadata &next_metadata = shader_dictionary.getMetadata(next_node.pipeline_id);

        std::vector<std::shared_ptr<gpu::vector>> next_output_buffers;
        for (std::uint32_t i = next_metadata.input_count;
             i < next_metadata.input_count + next_metadata.output_count && i < next_node.buffers.size();
             ++i)
        {
            if (next_node.buffers[i])
            {
                next_output_buffers.push_back(next_node.buffers[i]);
            }
        }

        for (const auto &output_buffer : next_output_buffers)
        {
            for (const auto &fused_operation : fused_node.fused_operations)
            {
                for (std::uint32_t input_index : fused_operation.input_buffer_indices)
                {
                    if (input_index < fused_node.buffers.size())
                    {
                        const auto &fused_input_buffer = fused_node.buffers[input_index];
                        if (fused_input_buffer &&
                            (output_buffer == fused_input_buffer ||
                             (output_buffer->getBuffer() != VK_NULL_HANDLE &&
                              output_buffer->getBuffer() == fused_input_buffer->getBuffer())))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        for (const auto &output_buffer : next_output_buffers)
        {
            for (std::size_t operation_index = 0; operation_index < fused_node.fused_operations.size() - 1; ++operation_index)
            {
                const auto &fused_operation = fused_node.fused_operations[operation_index];
                for (std::uint32_t output_index : fused_operation.output_buffer_indices)
                {
                    if (output_index < fused_node.buffers.size())
                    {
                        const auto &fused_output_buffer = fused_node.buffers[output_index];
                        if (fused_output_buffer &&
                            (output_buffer == fused_output_buffer ||
                             (output_buffer->getBuffer() != VK_NULL_HANDLE &&
                              output_buffer->getBuffer() == fused_output_buffer->getBuffer())))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    static void markExternalOutputs(
        Compute_Node &fused_node,
        const std::vector<Compute_Node> &nodes,
        std::size_t next_raw_node_index)
    {
        std::unordered_set<std::uint32_t> internal_consumed_output_indices;
        for (std::size_t i = 0; i < fused_node.fused_operations.size(); ++i)
        {
            for (std::size_t j = i + 1; j < fused_node.fused_operations.size(); ++j)
            {
                for (std::uint32_t input_index : fused_node.fused_operations[j].input_buffer_indices)
                {
                    for (std::uint32_t output_index : fused_node.fused_operations[i].output_buffer_indices)
                    {
                        if (input_index == output_index)
                        {
                            internal_consumed_output_indices.insert(output_index);
                        }
                    }
                }
            }
        }

        for (const auto &operation : fused_node.fused_operations)
        {
            for (std::uint32_t output_index : operation.output_buffer_indices)
            {
                if (!internal_consumed_output_indices.contains(output_index))
                {
                    fused_node.external_output_indices.insert(output_index);
                }
            }
        }

        if (next_raw_node_index >= nodes.size())
        {
            for (std::size_t operation_buffer_index = 0; operation_buffer_index < fused_node.buffers.size(); ++operation_buffer_index)
            {
                if (fused_node.buffers[operation_buffer_index])
                {
                    fused_node.external_output_indices.insert(static_cast<std::uint32_t>(operation_buffer_index));
                }
            }
            return;
        }

        for (std::size_t operation_buffer_index = 0; operation_buffer_index < fused_node.buffers.size(); ++operation_buffer_index)
        {
            const auto &buffer = fused_node.buffers[operation_buffer_index];
            if (!buffer)
            {
                continue;
            }

            for (std::size_t j = next_raw_node_index; j < nodes.size(); ++j)
            {
                for (const auto &future_buffer : nodes[j].buffers)
                {
                    if (future_buffer &&
                        (buffer == future_buffer ||
                         (buffer->getBuffer() != VK_NULL_HANDLE &&
                          buffer->getBuffer() == future_buffer->getBuffer())))
                    {
                        fused_node.external_output_indices.insert(static_cast<std::uint32_t>(operation_buffer_index));
                    }
                }
            }
        }
    }

    static Cached_Graph_Template optimizeInternal(const std::vector<Compute_Node> &original_nodes, bool is_tracking_mappings)
    {
        const Shader_Dictionary &shader_dictionary = Shader_Dictionary::getInstance();
        Cached_Graph_Template graph_template;
        if (original_nodes.empty())
        {
            return graph_template;
        }

        Compute_Node current_fused_node;
        current_fused_node.pipeline_id = original_nodes[0].pipeline_id;
        current_fused_node.push_constants_data = original_nodes[0].push_constants_data;
        current_fused_node.workgroup_count_x = original_nodes[0].workgroup_count_x;
        current_fused_node.workgroup_count_y = original_nodes[0].workgroup_count_y;
        current_fused_node.workgroup_count_z = original_nodes[0].workgroup_count_z;
        current_fused_node.is_fused = false;

        std::vector<Buffer_Binding_Mapping> current_buffer_mappings;
        std::vector<Push_Constant_Mapping> current_push_constants_mappings;
        std::vector<std::uint32_t> current_raw_node_indices;

        const Snippet_Metadata &first_metadata = shader_dictionary.getMetadata(original_nodes[0].pipeline_id);
        current_fused_node.fused_operations.push_back(
            buildFusedOperation(current_fused_node, original_nodes[0], 0, first_metadata, 0, is_tracking_mappings ? &current_buffer_mappings : nullptr));

        if (is_tracking_mappings)
        {
            current_push_constants_mappings.push_back(Push_Constant_Mapping{
                .raw_node_index = 0,
                .fused_push_constants_offset = 0,
                .push_constants_size = static_cast<std::uint32_t>(original_nodes[0].push_constants_data.size())});
            current_raw_node_indices.push_back(0);
        }

        for (std::size_t i = 1; i < original_nodes.size(); ++i)
        {
            const Compute_Node &next_node = original_nodes[i];

            bool is_sharing_buffer = false;
            for (const auto &buffer_a : current_fused_node.buffers)
            {
                if (!buffer_a)
                {
                    continue;
                }
                for (const auto &buffer_b : next_node.buffers)
                {
                    if (!buffer_b)
                    {
                        continue;
                    }
                    if (buffer_a == buffer_b ||
                        (buffer_a->getBuffer() != VK_NULL_HANDLE &&
                         buffer_a->getBuffer() == buffer_b->getBuffer()))
                    {
                        is_sharing_buffer = true;
                        break;
                    }
                }
                if (is_sharing_buffer)
                {
                    break;
                }
            }

            std::vector<std::shared_ptr<gpu::vector>> unique_buffers;
            for (const auto &new_buffer : next_node.buffers)
            {
                bool is_existing = false;
                for (const auto &existing_buffer : current_fused_node.buffers)
                {
                    if (new_buffer && existing_buffer &&
                        (new_buffer == existing_buffer ||
                         (new_buffer->getBuffer() != VK_NULL_HANDLE &&
                          new_buffer->getBuffer() == existing_buffer->getBuffer())))
                    {
                        is_existing = true;
                        break;
                    }
                }
                if (!is_existing)
                {
                    unique_buffers.push_back(new_buffer);
                }
            }

            std::size_t current_push_constants_bytes = current_fused_node.push_constants_data.size();
            std::size_t aligned_push_constants_offset = alignTo4Bytes(current_push_constants_bytes);
            std::size_t total_push_constants_size = aligned_push_constants_offset + next_node.push_constants_data.size();

            bool does_push_constants_fit = (total_push_constants_size <= MAX_PUSH_CONSTANTS_BYTES);
            std::size_t total_bindings = current_fused_node.buffers.size() + unique_buffers.size();
            bool do_bindings_fit = (total_bindings <= MAX_STORAGE_BUFFER_BINDINGS);
            bool does_chain_fit = (current_fused_node.fused_operations.size() < MAX_FUSED_OPERATIONS);

            Compute_Pipeline producer_pipeline = current_fused_node.fused_operations[0].pipeline_id;
            Compute_Pipeline consumer_pipeline = next_node.pipeline_id;

            Operation_Class producer_class = shader_dictionary.getMetadata(producer_pipeline).operation_class;
            Operation_Class consumer_class = shader_dictionary.getMetadata(consumer_pipeline).operation_class;

            bool is_fusible_operation = !current_fused_node.fused_operations.empty() &&
                                        isFusible(producer_class, consumer_class, producer_pipeline, consumer_pipeline);
            bool are_dimensions_matching = hasCompatibleDimensions(current_fused_node, next_node, producer_class, consumer_class);
            bool has_aliasing_hazard = hasAliasingHazard(current_fused_node, next_node, shader_dictionary);

            if (is_sharing_buffer && does_push_constants_fit && do_bindings_fit && does_chain_fit &&
                is_fusible_operation && are_dimensions_matching && !has_aliasing_hazard)
            {
                std::size_t padding_bytes = aligned_push_constants_offset - current_push_constants_bytes;
                if (padding_bytes > 0)
                {
                    current_fused_node.push_constants_data.insert(
                        current_fused_node.push_constants_data.end(),
                        padding_bytes,
                        0);
                }

                std::uint32_t current_push_constants_offset_val = static_cast<std::uint32_t>(aligned_push_constants_offset);
                const Snippet_Metadata &next_metadata = shader_dictionary.getMetadata(next_node.pipeline_id);

                current_fused_node.fused_operations.push_back(
                    buildFusedOperation(current_fused_node, next_node, current_push_constants_offset_val, next_metadata, static_cast<std::uint32_t>(i), is_tracking_mappings ? &current_buffer_mappings : nullptr));

                if (is_tracking_mappings)
                {
                    current_push_constants_mappings.push_back(Push_Constant_Mapping{
                        .raw_node_index = static_cast<std::uint32_t>(i),
                        .fused_push_constants_offset = current_push_constants_offset_val,
                        .push_constants_size = static_cast<std::uint32_t>(next_node.push_constants_data.size())});
                    current_raw_node_indices.push_back(static_cast<std::uint32_t>(i));
                }

                current_fused_node.push_constants_data.insert(
                    current_fused_node.push_constants_data.end(),
                    next_node.push_constants_data.begin(),
                    next_node.push_constants_data.end());

                updateDispatchGrid(current_fused_node, next_node, producer_class, consumer_class);
                current_fused_node.is_fused = true;
            }
            else
            {
                markExternalOutputs(current_fused_node, original_nodes, i);

                graph_template.fused_nodes.push_back(current_fused_node);
                if (is_tracking_mappings)
                {
                    graph_template.buffer_mappings.push_back(std::move(current_buffer_mappings));
                    graph_template.push_constants_mappings.push_back(std::move(current_push_constants_mappings));
                    graph_template.raw_node_indices.push_back(std::move(current_raw_node_indices));
                    current_buffer_mappings.clear();
                    current_push_constants_mappings.clear();
                    current_raw_node_indices.clear();
                }

                current_fused_node = Compute_Node{};
                current_fused_node.pipeline_id = next_node.pipeline_id;
                current_fused_node.push_constants_data = next_node.push_constants_data;
                current_fused_node.workgroup_count_x = next_node.workgroup_count_x;
                current_fused_node.workgroup_count_y = next_node.workgroup_count_y;
                current_fused_node.workgroup_count_z = next_node.workgroup_count_z;
                current_fused_node.is_fused = false;

                const Snippet_Metadata &metadata = shader_dictionary.getMetadata(next_node.pipeline_id);
                current_fused_node.fused_operations.push_back(
                    buildFusedOperation(current_fused_node, next_node, 0, metadata, static_cast<std::uint32_t>(i), is_tracking_mappings ? &current_buffer_mappings : nullptr));

                if (is_tracking_mappings)
                {
                    current_push_constants_mappings.push_back(Push_Constant_Mapping{
                        .raw_node_index = static_cast<std::uint32_t>(i),
                        .fused_push_constants_offset = 0,
                        .push_constants_size = static_cast<std::uint32_t>(next_node.push_constants_data.size())});
                    current_raw_node_indices.push_back(static_cast<std::uint32_t>(i));
                }
            }
        }

        markExternalOutputs(current_fused_node, original_nodes, original_nodes.size());

        graph_template.fused_nodes.push_back(current_fused_node);
        if (is_tracking_mappings)
        {
            graph_template.buffer_mappings.push_back(std::move(current_buffer_mappings));
            graph_template.push_constants_mappings.push_back(std::move(current_push_constants_mappings));
            graph_template.raw_node_indices.push_back(std::move(current_raw_node_indices));
        }

        assignPipelineBarriers(graph_template.fused_nodes);
        graph_template.is_valid = true;

        return graph_template;
    }

public:
    static void optimize(Compute_Graph &graph)
    {
#if !ENABLE_SHADER_FUSION
        auto nodes = graph.getNodes();
        assignPipelineBarriers(nodes);
        graph.clear();
        for (const auto &node : nodes)
        {
            graph.addNode(node);
        }
        return;
#else
        const auto &original_nodes = graph.getNodes();
        if (original_nodes.empty())
        {
            return;
        }

        Cached_Graph_Template graph_template = optimizeInternal(original_nodes, false);
        graph.clear();
        for (const auto &node : graph_template.fused_nodes)
        {
            graph.addNode(node);
        }
#endif
    }

    static Cached_Graph_Template buildCachedTemplate(const Compute_Graph &graph)
    {
#if !ENABLE_SHADER_FUSION
        return Cached_Graph_Template{};
#else
        const auto &original_nodes = graph.getNodes();
        if (original_nodes.empty())
        {
            return Cached_Graph_Template{};
        }

        return optimizeInternal(original_nodes, true);
#endif
    }

    static void applyCachedTemplate(
        const Compute_Graph &_raw_graph,
        const Cached_Graph_Template &_graph_template,
        Compute_Graph &_output_graph)
    {
#if !ENABLE_SHADER_FUSION
        _output_graph = _raw_graph;
        auto nodes = _output_graph.getNodes();
        assignPipelineBarriers(nodes);
        _output_graph.clear();
        for (const auto &node : nodes)
        {
            _output_graph.addNode(node);
        }
#else
        if (!_graph_template.is_valid)
        {
            _output_graph = _raw_graph;
            auto nodes = _output_graph.getNodes();
            assignPipelineBarriers(nodes);
            _output_graph.clear();
            for (const auto &node : nodes)
            {
                _output_graph.addNode(node);
            }
            return;
        }

        const auto &raw_nodes = _raw_graph.getNodes();
        const Shader_Dictionary &shader_dictionary = Shader_Dictionary::getInstance();
        _output_graph.clear();

        for (std::size_t node_index = 0; node_index < _graph_template.fused_nodes.size(); ++node_index)
        {
            Compute_Node node = _graph_template.fused_nodes[node_index];

            for (const auto &buffer_binding_mapping : _graph_template.buffer_mappings[node_index])
            {
                if (buffer_binding_mapping.raw_node_index < raw_nodes.size() &&
                    buffer_binding_mapping.raw_buffer_index < raw_nodes[buffer_binding_mapping.raw_node_index].buffers.size())
                {
                    auto old_buffer = node.buffers[buffer_binding_mapping.fused_buffer_index];
                    auto new_buffer = raw_nodes[buffer_binding_mapping.raw_node_index].buffers[buffer_binding_mapping.raw_buffer_index];

                    Logger::logMessage(std::format(
                                           "Graph_Optimizer::applyCachedTemplate: Fused Node {} | Binding buffer_{} <- Raw Node {}[Buffer {}] (Old ID: {}, New ID: {})",
                                           node_index,
                                           buffer_binding_mapping.fused_buffer_index,
                                           buffer_binding_mapping.raw_node_index,
                                           buffer_binding_mapping.raw_buffer_index,
                                           old_buffer ? old_buffer->getId() : 0,
                                           new_buffer ? new_buffer->getId() : 0),
                                       Log_Level::LOG_DEBUG,
                                       true,
                                       0,
                                       Log_Feature::OPERATOR_FUSION);

                    node.buffers[buffer_binding_mapping.fused_buffer_index] = raw_nodes[buffer_binding_mapping.raw_node_index].buffers[buffer_binding_mapping.raw_buffer_index];
                }
            }

            for (const auto &push_constant_mapping : _graph_template.push_constants_mappings[node_index])
            {
                if (push_constant_mapping.raw_node_index < raw_nodes.size())
                {
                    const auto &source_push_constants = raw_nodes[push_constant_mapping.raw_node_index].push_constants_data;
                    std::size_t copy_bytes = std::min<std::size_t>(source_push_constants.size(), push_constant_mapping.push_constants_size);
                    if (push_constant_mapping.fused_push_constants_offset + copy_bytes <= node.push_constants_data.size())
                    {
                        std::copy_n(source_push_constants.begin(), copy_bytes, node.push_constants_data.begin() + push_constant_mapping.fused_push_constants_offset);
                    }
                }
            }

            if (node_index < _graph_template.raw_node_indices.size())
            {
                const auto &mapped_raw_indices = _graph_template.raw_node_indices[node_index];
                if (!mapped_raw_indices.empty())
                {
                    std::uint32_t first_raw_index = mapped_raw_indices[0];
                    node.workgroup_count_x = raw_nodes[first_raw_index].workgroup_count_x;
                    node.workgroup_count_y = raw_nodes[first_raw_index].workgroup_count_y;
                    node.workgroup_count_z = raw_nodes[first_raw_index].workgroup_count_z;

                    if (!node.fused_operations.empty())
                    {
                        node.fused_operations[0].workgroup_count_x = raw_nodes[first_raw_index].workgroup_count_x;
                        node.fused_operations[0].workgroup_count_y = raw_nodes[first_raw_index].workgroup_count_y;
                        node.fused_operations[0].workgroup_count_z = raw_nodes[first_raw_index].workgroup_count_z;
                    }

                    Operation_Class producer_class = shader_dictionary.getMetadata(node.fused_operations[0].pipeline_id).operation_class;

                    for (std::size_t k = 1; k < mapped_raw_indices.size(); ++k)
                    {
                        std::uint32_t next_raw_index = mapped_raw_indices[k];
                        Operation_Class consumer_class = shader_dictionary.getMetadata(raw_nodes[next_raw_index].pipeline_id).operation_class;

                        if (k < node.fused_operations.size())
                        {
                            node.fused_operations[k].workgroup_count_x = raw_nodes[next_raw_index].workgroup_count_x;
                            node.fused_operations[k].workgroup_count_y = raw_nodes[next_raw_index].workgroup_count_y;
                            node.fused_operations[k].workgroup_count_z = raw_nodes[next_raw_index].workgroup_count_z;
                        }

                        updateDispatchGrid(node, raw_nodes[next_raw_index], producer_class, consumer_class);
                    }
                }
            }

            _output_graph.addNode(node);
        }
#endif
    }
};
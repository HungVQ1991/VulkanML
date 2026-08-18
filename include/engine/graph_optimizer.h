#pragma once

#include <vector>
#include <string>
#include <format>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "compute_node.h"
#include "shader_dictionary.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"

#ifndef ENABLE_GRAPH_OPTIMIZER_DEBUG_LOGS
#define ENABLE_GRAPH_OPTIMIZER_DEBUG_LOGS 0
#endif

#if ENABLE_GRAPH_OPTIMIZER_DEBUG_LOGS
#define GRAPH_OPTIMIZER_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define GRAPH_OPTIMIZER_LOG_DEBUG(msg) ((void)0)
#endif

#ifndef ENABLE_SHADER_FUSION
#define ENABLE_SHADER_FUSION 0
#endif

struct Buffer_Binding_Map
{
    std::uint32_t raw_node_index;
    std::uint32_t raw_buffer_index;
    std::uint32_t fused_buffer_index;
};

struct Push_Constant_Map
{
    std::uint32_t raw_node_index;
    std::uint32_t fused_pc_offset;
    std::uint32_t pc_size;
};

struct Cached_Graph_Template
{
    std::vector<Compute_Node> fused_nodes;
    std::vector<std::vector<Buffer_Binding_Map>> buffer_mappings;
    std::vector<std::vector<Push_Constant_Map>> pc_mappings;
    std::vector<std::vector<std::uint32_t>> raw_node_indices;
    bool is_valid = false;

    void clear()
    {
        fused_nodes.clear();
        buffer_mappings.clear();
        pc_mappings.clear();
        raw_node_indices.clear();
        is_valid = false;
    }
};

class Graph_Optimizer
{
private:
    static constexpr std::size_t MAX_PUSH_CONSTANTS_BYTES = 128;
    static constexpr std::size_t MAX_SSBO_BINDINGS = 32;
    static constexpr std::size_t MAX_FUSED_OPERATIONS = 8;

    static constexpr std::size_t alignTo4Bytes(std::size_t offset)
    {
        return (offset + 3) & ~std::size_t(3);
    }

static bool isFusible(
        Op_Class prod_class, 
        Op_Class cons_class, 
        Compute_Pipeline prod_op, 
        Compute_Pipeline tail_op)
    {
        if (cons_class != Op_Class::ELEMENTWISE)
        {
            return false;
        }

        const auto &shader_dict = Shader_Dictionary::getInstance();
        const auto &tail_meta = shader_dict.getMetadata(tail_op);
        const auto &prod_meta = shader_dict.getMetadata(prod_op);

        if (tail_meta.writes_multiple_elements || tail_meta.shared_mem_size > 0)
        {
            return false;
        }

        if (prod_meta.op_class == Op_Class::STANDALONE || 
            prod_meta.writes_multiple_elements || 
            prod_meta.output_count > 1)
        {
            return false;
        }

        return (prod_class == Op_Class::MATRIX_2D ||
                prod_class == Op_Class::ELEMENTWISE ||
                prod_class == Op_Class::TENSOR_3D);
    }

    static bool checkDimensions(
        const Compute_Node &a, 
        const Compute_Node &b, 
        Op_Class prod_class, 
        Op_Class cons_class)
    {
        if (a.group_x == b.group_x && a.group_y == b.group_y && a.group_z == b.group_z)
        {
            return true;
        }

        if (cons_class == Op_Class::ELEMENTWISE)
        {
            if (prod_class == Op_Class::MATRIX_2D || prod_class == Op_Class::TENSOR_3D)
            {
                std::uint64_t total_workgroups_a = static_cast<std::uint64_t>(a.group_x) * a.group_y * a.group_z;
                std::uint64_t total_workgroups_b = static_cast<std::uint64_t>(b.group_x) * b.group_y * b.group_z;

                return (total_workgroups_a >= total_workgroups_b);
            }
        }

        return false;
    }

    static std::uint32_t findOrAddBuffer(Compute_Node &node, const std::shared_ptr<GVector> &buf)
    {
        for (std::uint32_t idx = 0; idx < node.buffers.size(); ++idx)
        {
            const auto &exist_buf = node.buffers[idx];
            if (buf && exist_buf && (buf == exist_buf || (buf->getBuffer() != VK_NULL_HANDLE && buf->getBuffer() == exist_buf->getBuffer())))
            {
                return idx;
            }
        }
        node.buffers.push_back(buf);
        return static_cast<std::uint32_t>(node.buffers.size() - 1);
    }

    static void updateDispatchGrid(Compute_Node &fused_node, const Compute_Node &next_node, Op_Class prod_class, Op_Class cons_class)
    {
        const auto &shader_dict = Shader_Dictionary::getInstance();
        const auto &next_meta = shader_dict.getMetadata(next_node.pipeline_id);

        if (next_meta.shared_mem_size > 0 || cons_class == Op_Class::STANDALONE)
        {
            fused_node.group_x = next_node.group_x;
            fused_node.group_y = 1;
            fused_node.group_z = 1;
            return;
        }

        if (prod_class == Op_Class::MATRIX_2D || prod_class == Op_Class::TENSOR_3D || prod_class == Op_Class::STANDALONE)
        {
            return;
        }

        fused_node.group_x = std::max(fused_node.group_x, next_node.group_x);
        fused_node.group_y = std::max(fused_node.group_y, next_node.group_y);
        fused_node.group_z = std::max(fused_node.group_z, next_node.group_z);
    }

    static Fused_Operation buildFusedOperation(
        Compute_Node &fused_node,
        const Compute_Node &source_node,
        std::uint32_t pc_offset,
        const Snippet_Metadata &meta,
        std::uint32_t raw_node_index = 0,
        std::vector<Buffer_Binding_Map> *out_buf_map = nullptr)
    {
        Fused_Operation op{
            .pipeline_id = source_node.pipeline_id,
            .pc_offset = pc_offset,
            .pc_size = static_cast<std::uint32_t>(source_node.push_constants_data.size()),
            .input_buffer_indices = {},
            .output_buffer_indices = {},
            .group_x = source_node.group_x,
            .group_y = source_node.group_y,
            .group_z = source_node.group_z};

        for (std::uint32_t b = 0; b < meta.input_count && b < source_node.buffers.size(); ++b)
        {
            std::uint32_t fused_buf_idx = findOrAddBuffer(fused_node, source_node.buffers[b]);
            op.input_buffer_indices.push_back(fused_buf_idx);
            if (out_buf_map)
            {
                out_buf_map->push_back({raw_node_index, b, fused_buf_idx});
            }
        }

        for (std::uint32_t b = meta.input_count; b < meta.input_count + meta.output_count && b < source_node.buffers.size(); ++b)
        {
            std::uint32_t fused_buf_idx = findOrAddBuffer(fused_node, source_node.buffers[b]);
            op.output_buffer_indices.push_back(fused_buf_idx);
            if (out_buf_map)
            {
                out_buf_map->push_back({raw_node_index, b, fused_buf_idx});
            }

            std::uint32_t local_out_idx = b - meta.input_count;
            bool is_declared_persistent = std::find(meta.persistent_outputs.begin(),
                                                    meta.persistent_outputs.end(),
                                                    local_out_idx) != meta.persistent_outputs.end();

            if (source_node.external_output_indices.contains(b) || is_declared_persistent)
            {
                fused_node.external_output_indices.insert(fused_buf_idx);
            }
        }

        return op;
    }

    static bool hasAliasingHazard(const Compute_Node &fused_node, const Compute_Node &next_node, const Shader_Dictionary &shader_dict)
    {
        const Snippet_Metadata &next_meta = shader_dict.getMetadata(next_node.pipeline_id);

        std::vector<std::shared_ptr<GVector>> next_outputs;
        for (std::uint32_t i = next_meta.input_count; i < next_meta.input_count + next_meta.output_count && i < next_node.buffers.size(); ++i)
        {
            if (next_node.buffers[i])
            {
                next_outputs.push_back(next_node.buffers[i]);
            }
        }

        for (const auto &out_buf : next_outputs)
        {
            for (const auto &fused_op : fused_node.fused_operations)
            {
                for (std::uint32_t in_idx : fused_op.input_buffer_indices)
                {
                    if (in_idx < fused_node.buffers.size())
                    {
                        const auto &fused_in_buf = fused_node.buffers[in_idx];
                        if (fused_in_buf && (out_buf == fused_in_buf || (out_buf->getBuffer() != VK_NULL_HANDLE && out_buf->getBuffer() == fused_in_buf->getBuffer())))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        for (const auto &out_buf : next_outputs)
        {
            for (std::size_t op_idx = 0; op_idx < fused_node.fused_operations.size() - 1; ++op_idx)
            {
                const auto &fused_op = fused_node.fused_operations[op_idx];
                for (std::uint32_t out_idx : fused_op.output_buffer_indices)
                {
                    if (out_idx < fused_node.buffers.size())
                    {
                        const auto &fused_out_buf = fused_node.buffers[out_idx];
                        if (fused_out_buf && (out_buf == fused_out_buf || (out_buf->getBuffer() != VK_NULL_HANDLE && out_buf->getBuffer() == fused_out_buf->getBuffer())))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    static void markExternalOutputs(Compute_Node &fused_node, const std::vector<Compute_Node> &nodes, std::size_t next_raw_index)
    {
        std::unordered_set<std::uint32_t> internal_consumed_outputs;
        for (std::size_t i = 0; i < fused_node.fused_operations.size(); ++i)
        {
            for (std::size_t j = i + 1; j < fused_node.fused_operations.size(); ++j)
            {
                for (std::uint32_t in_idx : fused_node.fused_operations[j].input_buffer_indices)
                {
                    for (std::uint32_t out_idx : fused_node.fused_operations[i].output_buffer_indices)
                    {
                        if (in_idx == out_idx)
                        {
                            internal_consumed_outputs.insert(out_idx);
                        }
                    }
                }
            }
        }

        for (const auto &op : fused_node.fused_operations)
        {
            for (std::uint32_t out_idx : op.output_buffer_indices)
            {
                if (!internal_consumed_outputs.contains(out_idx))
                {
                    fused_node.external_output_indices.insert(out_idx);
                }
            }
        }

        if (next_raw_index >= nodes.size())
        {
            for (std::size_t op_buf_idx = 0; op_buf_idx < fused_node.buffers.size(); ++op_buf_idx)
            {
                if (fused_node.buffers[op_buf_idx])
                {
                    fused_node.external_output_indices.insert(static_cast<std::uint32_t>(op_buf_idx));
                }
            }
            return;
        }

        for (std::size_t op_buf_idx = 0; op_buf_idx < fused_node.buffers.size(); ++op_buf_idx)
        {
            const auto &buf = fused_node.buffers[op_buf_idx];
            if (!buf)
            {
                continue;
            }

            for (std::size_t j = next_raw_index; j < nodes.size(); ++j)
            {
                for (const auto &future_buf : nodes[j].buffers)
                {
                    if (future_buf && (buf == future_buf || (buf->getBuffer() != VK_NULL_HANDLE && buf->getBuffer() == future_buf->getBuffer())))
                    {
                        fused_node.external_output_indices.insert(static_cast<std::uint32_t>(op_buf_idx));
                    }
                }
            }
        }
    }

    static Cached_Graph_Template optimizeInternal(const std::vector<Compute_Node> &original_nodes, bool track_mappings)
    {
        const Shader_Dictionary &shader_dict = Shader_Dictionary::getInstance();
        Cached_Graph_Template tmpl;
        if (original_nodes.empty())
        {
            return tmpl;
        }

        Compute_Node current_fused;
        current_fused.pipeline_id = original_nodes[0].pipeline_id;
        current_fused.push_constants_data = original_nodes[0].push_constants_data;
        current_fused.group_x = original_nodes[0].group_x;
        current_fused.group_y = original_nodes[0].group_y;
        current_fused.group_z = original_nodes[0].group_z;
        current_fused.is_fused = false;

        std::vector<Buffer_Binding_Map> current_buf_map;
        std::vector<Push_Constant_Map> current_pc_map;
        std::vector<std::uint32_t> current_raw_indices;

        const Snippet_Metadata &first_meta = shader_dict.getMetadata(original_nodes[0].pipeline_id);
        current_fused.fused_operations.push_back(
            buildFusedOperation(current_fused, original_nodes[0], 0, first_meta, 0, track_mappings ? &current_buf_map : nullptr));

        if (track_mappings)
        {
            current_pc_map.push_back({0, 0, static_cast<std::uint32_t>(original_nodes[0].push_constants_data.size())});
            current_raw_indices.push_back(0);
        }

        for (std::size_t i = 1; i < original_nodes.size(); ++i)
        {
            const Compute_Node &next_node = original_nodes[i];

            bool share_buffer = false;
            for (const auto &buf_a : current_fused.buffers)
            {
                if (!buf_a)
                {
                    continue;
                }
                for (const auto &buf_b : next_node.buffers)
                {
                    if (!buf_b)
                    {
                        continue;
                    }
                    if (buf_a == buf_b || (buf_a->getBuffer() != VK_NULL_HANDLE && buf_a->getBuffer() == buf_b->getBuffer()))
                    {
                        share_buffer = true;
                        break;
                    }
                }
                if (share_buffer)
                {
                    break;
                }
            }

            std::vector<std::shared_ptr<GVector>> unique_buffers;
            for (const auto &new_buf : next_node.buffers)
            {
                bool exists = false;
                for (const auto &exist_buf : current_fused.buffers)
                {
                    if (new_buf && exist_buf && (new_buf == exist_buf || (new_buf->getBuffer() != VK_NULL_HANDLE && new_buf->getBuffer() == exist_buf->getBuffer())))
                    {
                        exists = true;
                        break;
                    }
                }
                if (!exists)
                {
                    unique_buffers.push_back(new_buf);
                }
            }

            std::size_t current_pc_bytes = current_fused.push_constants_data.size();
            std::size_t aligned_pc_offset = alignTo4Bytes(current_pc_bytes);
            std::size_t total_pc_size = aligned_pc_offset + next_node.push_constants_data.size();

            bool pc_fits = (total_pc_size <= MAX_PUSH_CONSTANTS_BYTES);
            if (!pc_fits)
            {
                Logger::logMessage(std::format("Graph_Optimizer::optimize: Push constants limit exceeded ({} > {} bytes) at pair {} -> {}",
                                               total_pc_size, MAX_PUSH_CONSTANTS_BYTES, i - 1, i),
                                   LOG_WARNING);
            }

            std::size_t total_bindings = current_fused.buffers.size() + unique_buffers.size();
            bool bindings_fit = (total_bindings <= MAX_SSBO_BINDINGS);
            if (!bindings_fit)
            {
                Logger::logMessage(std::format("Graph_Optimizer::optimize: SSBO bindings limit exceeded ({} > {}) at pair {} -> {}",
                                               total_bindings, MAX_SSBO_BINDINGS, i - 1, i),
                                   LOG_WARNING);
            }

            bool chain_fits = (current_fused.fused_operations.size() < MAX_FUSED_OPERATIONS);
            if (!chain_fits)
            {
                GRAPH_OPTIMIZER_LOG_DEBUG(std::format("Graph_Optimizer::optimize: Max fused operations chain length reached ({}) at pair {} -> {}",
                                                      MAX_FUSED_OPERATIONS, i - 1, i));
            }

            Compute_Pipeline producer_op = current_fused.fused_operations[0].pipeline_id;
            Compute_Pipeline tail_op = current_fused.fused_operations.back().pipeline_id;
            Compute_Pipeline consumer_op = next_node.pipeline_id;

            Op_Class tail_class = shader_dict.getMetadata(tail_op).op_class;
            Op_Class prod_class = shader_dict.getMetadata(producer_op).op_class;
            Op_Class cons_class = shader_dict.getMetadata(consumer_op).op_class;

            bool is_fusible_flag = !current_fused.fused_operations.empty() && isFusible(prod_class, cons_class, producer_op, tail_op);
            bool check_dims = checkDimensions(current_fused, next_node, prod_class, cons_class);
            bool has_hazard = hasAliasingHazard(current_fused, next_node, shader_dict);

            if (has_hazard)
            {
                GRAPH_OPTIMIZER_LOG_DEBUG(std::format("Graph_Optimizer::optimize: Aliasing hazard detected between group and node {} ({}), preventing fusion",
                                                      i, std::string(magic_enum::enum_name(consumer_op))));
            }

            if (share_buffer && pc_fits && bindings_fit && chain_fits && is_fusible_flag && check_dims && !has_hazard)
            {
                std::size_t padding_bytes = aligned_pc_offset - current_pc_bytes;
                if (padding_bytes > 0)
                {
                    current_fused.push_constants_data.insert(
                        current_fused.push_constants_data.end(),
                        padding_bytes,
                        0);
                }

                std::uint32_t current_pc_offset = static_cast<std::uint32_t>(aligned_pc_offset);
                const Snippet_Metadata &next_meta = shader_dict.getMetadata(next_node.pipeline_id);

                current_fused.fused_operations.push_back(
                    buildFusedOperation(current_fused, next_node, current_pc_offset, next_meta, static_cast<std::uint32_t>(i), track_mappings ? &current_buf_map : nullptr));

                if (track_mappings)
                {
                    current_pc_map.push_back({static_cast<std::uint32_t>(i), current_pc_offset, static_cast<std::uint32_t>(next_node.push_constants_data.size())});
                    current_raw_indices.push_back(static_cast<std::uint32_t>(i)); // Thêm node i vào cụm
                }

                current_fused.push_constants_data.insert(
                    current_fused.push_constants_data.end(),
                    next_node.push_constants_data.begin(),
                    next_node.push_constants_data.end());

                updateDispatchGrid(current_fused, next_node, prod_class, cons_class);
                current_fused.is_fused = true;
            }
            else
            {
                markExternalOutputs(current_fused, original_nodes, i);

                tmpl.fused_nodes.push_back(current_fused);
                if (track_mappings)
                {
                    tmpl.buffer_mappings.push_back(std::move(current_buf_map));
                    tmpl.pc_mappings.push_back(std::move(current_pc_map));
                    tmpl.raw_node_indices.push_back(std::move(current_raw_indices));
                    current_buf_map.clear();
                    current_pc_map.clear();
                    current_raw_indices.clear();
                }

                current_fused = Compute_Node{};
                current_fused.pipeline_id = next_node.pipeline_id;
                current_fused.push_constants_data = next_node.push_constants_data;
                current_fused.group_x = next_node.group_x;
                current_fused.group_y = next_node.group_y;
                current_fused.group_z = next_node.group_z;
                current_fused.is_fused = false;

                const Snippet_Metadata &meta = shader_dict.getMetadata(next_node.pipeline_id);
                current_fused.fused_operations.push_back(
                    buildFusedOperation(current_fused, next_node, 0, meta, static_cast<std::uint32_t>(i), track_mappings ? &current_buf_map : nullptr));

                if (track_mappings)
                {
                    current_pc_map.push_back({static_cast<std::uint32_t>(i), 0, static_cast<std::uint32_t>(next_node.push_constants_data.size())});
                    current_raw_indices.push_back(static_cast<std::uint32_t>(i));
                }
            }
        }

        markExternalOutputs(current_fused, original_nodes, original_nodes.size());

        tmpl.fused_nodes.push_back(current_fused);
        if (track_mappings)
        {
            tmpl.buffer_mappings.push_back(std::move(current_buf_map));
            tmpl.pc_mappings.push_back(std::move(current_pc_map));
            tmpl.raw_node_indices.push_back(std::move(current_raw_indices));
        }
        tmpl.is_valid = true;

        return tmpl;
    }

public:
    static void optimize(Compute_Graph &graph)
    {
#if !ENABLE_SHADER_FUSION
        Logger::logMessage("Graph_Optimizer::optimize: Shader fusion disabled via ENABLE_SHADER_FUSION", LOG_INFO, true);
        return;
#else
        const auto &original_nodes = graph.getNodes();
        if (original_nodes.empty())
        {
            Logger::logMessage("Graph_Optimizer::optimize: Graph contains no nodes to optimize", LOG_WARNING);
            return;
        }

        GRAPH_OPTIMIZER_LOG_DEBUG(std::format("Graph_Optimizer::optimize: Starting graph optimization for {} nodes", original_nodes.size()));

        Cached_Graph_Template tmpl = optimizeInternal(original_nodes, false);

        GRAPH_OPTIMIZER_LOG_DEBUG(std::format("Graph_Optimizer::optimize: Graph optimization complete. Nodes reduced from {} to {}",
                                              original_nodes.size(), tmpl.fused_nodes.size()));

        graph.clear();
        for (const auto &node : tmpl.fused_nodes)
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
        const Compute_Graph &raw_graph,
        const Cached_Graph_Template &tmpl,
        Compute_Graph &out_graph)
    {
#if !ENABLE_SHADER_FUSION
        out_graph = raw_graph;
#else
        if (!tmpl.is_valid)
        {
            out_graph = raw_graph;
            return;
        }

        const auto &raw_nodes = raw_graph.getNodes();
        const Shader_Dictionary &shader_dict = Shader_Dictionary::getInstance();
        out_graph.clear();

        for (std::size_t node_idx = 0; node_idx < tmpl.fused_nodes.size(); ++node_idx)
        {
            Compute_Node node = tmpl.fused_nodes[node_idx];

            for (const auto &b_map : tmpl.buffer_mappings[node_idx])
            {
                if (b_map.raw_node_index < raw_nodes.size() && b_map.raw_buffer_index < raw_nodes[b_map.raw_node_index].buffers.size())
                {
                    node.buffers[b_map.fused_buffer_index] = raw_nodes[b_map.raw_node_index].buffers[b_map.raw_buffer_index];
                }
            }

            for (const auto &pc_map : tmpl.pc_mappings[node_idx])
            {
                if (pc_map.raw_node_index < raw_nodes.size())
                {
                    const auto &src_pc = raw_nodes[pc_map.raw_node_index].push_constants_data;

                    if (src_pc.size() != pc_map.pc_size)
                    {
                        std::string err_msg = std::format("Graph_Optimizer::applyCachedTemplate: Push Constants size mismatch at raw node {}. Expected {}, got {}. Stale data hazard prevented.", pc_map.raw_node_index, pc_map.pc_size, src_pc.size());
                        Logger::logMessage(err_msg, LOG_ERROR, true);
                        throw std::runtime_error(err_msg);
                    }

                    if (pc_map.fused_pc_offset + pc_map.pc_size <= node.push_constants_data.size())
                    {
                        std::copy(src_pc.begin(), src_pc.end(), node.push_constants_data.begin() + pc_map.fused_pc_offset);
                    }
                }
            }

            if (node_idx < tmpl.raw_node_indices.size())
            {
                const auto &mapped_raw_indices = tmpl.raw_node_indices[node_idx];
                if (!mapped_raw_indices.empty())
                {
                    std::uint32_t first_raw_idx = mapped_raw_indices[0];
                    node.group_x = raw_nodes[first_raw_idx].group_x;
                    node.group_y = raw_nodes[first_raw_idx].group_y;
                    node.group_z = raw_nodes[first_raw_idx].group_z;

                    if (!node.fused_operations.empty())
                    {
                        node.fused_operations[0].group_x = raw_nodes[first_raw_idx].group_x;
                        node.fused_operations[0].group_y = raw_nodes[first_raw_idx].group_y;
                        node.fused_operations[0].group_z = raw_nodes[first_raw_idx].group_z;
                    }

                    Op_Class prod_class = shader_dict.getMetadata(node.fused_operations[0].pipeline_id).op_class;

                    for (std::size_t k = 1; k < mapped_raw_indices.size(); ++k)
                    {
                        std::uint32_t next_raw_idx = mapped_raw_indices[k];
                        Op_Class cons_class = shader_dict.getMetadata(raw_nodes[next_raw_idx].pipeline_id).op_class;

                        if (k < node.fused_operations.size())
                        {
                            node.fused_operations[k].group_x = raw_nodes[next_raw_idx].group_x;
                            node.fused_operations[k].group_y = raw_nodes[next_raw_idx].group_y;
                            node.fused_operations[k].group_z = raw_nodes[next_raw_idx].group_z;
                        }

                        updateDispatchGrid(node, raw_nodes[next_raw_idx], prod_class, cons_class);
                    }
                }
            }

            out_graph.addNode(node);
        }
#endif
    }
};
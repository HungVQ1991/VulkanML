#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "compute_node.h"
#include "gpu_vector.h"
#include "shader_dictionary.h"
#include "vulkan_context.h"

extern bool is_coop;

#ifndef ENABLE_SHADER_FUSION
#define ENABLE_SHADER_FUSION 1
#endif

struct Buffer_Binding_Mapping
{
    uint32_t raw_node_index = 0;
    uint32_t raw_buffer_index = 0;
    uint32_t fused_buffer_index = 0;

    uint32_t getFusedBufferIndex() const noexcept { return fused_buffer_index; }
    uint32_t getRawBufferIndex() const noexcept { return raw_buffer_index; }
    uint32_t getRawNodeIndex() const noexcept { return raw_node_index; }

    void setFusedBufferIndex(uint32_t _index) noexcept { fused_buffer_index = _index; }
    void setRawBufferIndex(uint32_t _index) noexcept { raw_buffer_index = _index; }
    void setRawNodeIndex(uint32_t _index) noexcept { raw_node_index = _index; }
};

struct Push_Constant_Mapping
{
    uint32_t raw_node_index = 0;
    uint32_t fused_push_constants_offset = 0;
    uint32_t push_constants_size = 0;

    uint32_t getFusedPushConstantsOffset() const noexcept { return fused_push_constants_offset; }
    uint32_t getPushConstantsSize() const noexcept { return push_constants_size; }
    uint32_t getRawNodeIndex() const noexcept { return raw_node_index; }

    void setFusedPushConstantsOffset(uint32_t _offset) noexcept { fused_push_constants_offset = _offset; }
    void setPushConstantsSize(uint32_t _size) noexcept { push_constants_size = _size; }
    void setRawNodeIndex(uint32_t _index) noexcept { raw_node_index = _index; }
};

struct Cached_Graph_Template
{
    std::vector<Compute_Node> fused_nodes;
    std::vector<std::vector<Buffer_Binding_Mapping>> buffer_mappings;
    std::vector<std::vector<Push_Constant_Mapping>> push_constants_mappings;
    std::vector<std::vector<uint32_t>> raw_node_indices;
    mutable std::array<Compute_Graph, MAX_FRAMES_IN_FLIGHT> instantiated_graphs;
    size_t total_buffer_mappings = 0;
    mutable bool is_mapping_log_enabled = true;
    bool is_valid = false;

    void clear() noexcept
    {
        fused_nodes.clear();
        buffer_mappings.clear();
        push_constants_mappings.clear();
        raw_node_indices.clear();
        for (auto &graph : instantiated_graphs)
        {
            graph.clear();
        }
        total_buffer_mappings = 0;
        is_mapping_log_enabled = true;
        is_valid = false;
    }

    const std::vector<std::vector<Push_Constant_Mapping>> &getPushConstantsMappings() const noexcept { return push_constants_mappings; }
    const std::vector<std::vector<Buffer_Binding_Mapping>> &getBufferMappings() const noexcept { return buffer_mappings; }
    const std::array<Compute_Graph, MAX_FRAMES_IN_FLIGHT> &getInstantiatedGraphs() const noexcept { return instantiated_graphs; }
    std::array<Compute_Graph, MAX_FRAMES_IN_FLIGHT> &getInstantiatedGraphs() noexcept { return instantiated_graphs; }
    const std::vector<std::vector<uint32_t>> &getRawNodeIndices() const noexcept { return raw_node_indices; }
    const std::vector<Compute_Node> &getFusedNodes() const noexcept { return fused_nodes; }
    size_t getTotalBufferMappings() const noexcept { return total_buffer_mappings; }
    bool isMappingLogEnabled() const noexcept { return is_mapping_log_enabled; }
    bool isValid() const noexcept { return is_valid; }

    void setInstantiatedGraphs(const std::array<Compute_Graph, MAX_FRAMES_IN_FLIGHT> &_graphs) { instantiated_graphs = _graphs; }
    void setPushConstantsMappings(const std::vector<std::vector<Push_Constant_Mapping>> &_mappings) { push_constants_mappings = _mappings; }
    void setBufferMappings(const std::vector<std::vector<Buffer_Binding_Mapping>> &_mappings) { buffer_mappings = _mappings; }
    void setRawNodeIndices(const std::vector<std::vector<uint32_t>> &_indices) { raw_node_indices = _indices; }
    void setFusedNodes(const std::vector<Compute_Node> &_nodes) { fused_nodes = _nodes; }
    void setTotalBufferMappings(size_t _total) noexcept { total_buffer_mappings = _total; }
    void setMappingLogEnabled(bool _enabled) noexcept { is_mapping_log_enabled = _enabled; }
    void setIsValid(bool _valid) noexcept { is_valid = _valid; }
};

class Graph_Optimizer
{
private:
    static bool is_fused_gemm_adam_enabled;

    static constexpr size_t MAX_PUSH_CONSTANTS_BYTES = 128;
    static constexpr size_t MAX_STORAGE_BUFFER_BINDINGS = 32;
    static constexpr size_t MAX_FUSED_OPERATIONS = 8;

    static constexpr size_t alignTo4Bytes(size_t _offset) noexcept
    {
        return (_offset + 3) & ~size_t(3);
    }

    static void getNodeBufferAccess(
        const Compute_Node &_node,
        const Shader_Dictionary &_shader_dictionary,
        std::vector<VkBuffer> &_reads,
        std::vector<VkBuffer> &_writes);

    static void assignPipelineBarriers(std::vector<Compute_Node> &_nodes);

    static bool isFusible(
        Operation_Class _producer_class,
        Operation_Class _consumer_class,
        Compute_Pipeline _producer_pipeline,
        Compute_Pipeline _consumer_pipeline);

    static bool hasCompatibleDimensions(
        const Compute_Node &_producer_node,
        const Compute_Node &_consumer_node,
        Operation_Class _producer_class,
        Operation_Class _consumer_class) noexcept;

    static uint32_t findOrAddBuffer(Compute_Node &_node, const std::shared_ptr<gpu::vector> &_target_buffer);

    static void updateDispatchGrid(
        Compute_Node &_fused_node,
        const Compute_Node &_next_node,
        Operation_Class _producer_class,
        Operation_Class _consumer_class);

    static Fused_Operation buildFusedOperation(
        Compute_Node &_fused_node,
        const Compute_Node &_source_node,
        uint32_t _push_constants_offset,
        const Snippet_Metadata &_metadata,
        uint32_t _raw_node_index = 0,
        std::vector<Buffer_Binding_Mapping> *_output_buffer_mappings = nullptr);

    static bool hasAliasingHazard(
        const Compute_Node &_fused_node,
        const Compute_Node &_next_node,
        const Shader_Dictionary &_shader_dictionary);

    static void markExternalOutputs(
        Compute_Node &_fused_node,
        const std::vector<Compute_Node> &_nodes,
        size_t _next_raw_node_index);

    static std::vector<Compute_Node> fuseLinearBackwardAdamPass(const std::vector<Compute_Node> &_nodes);

    static Cached_Graph_Template optimizeInternal(const std::vector<Compute_Node> &_original_nodes, bool _is_tracking_mappings);

public:
    static void optimize(Compute_Graph &_graph);
    static Cached_Graph_Template buildCachedTemplate(const Compute_Graph &_graph);
    static void applyCachedTemplateInPlace(
        const Compute_Graph &_raw_graph,
        const Cached_Graph_Template &_graph_template,
        Compute_Graph &_cached_graph);
    static void applyCachedTemplate(
        const Compute_Graph &_raw_graph,
        const Cached_Graph_Template &_graph_template,
        Compute_Graph &_output_graph);

    static constexpr size_t getMaxPushConstantsBytes() noexcept { return MAX_PUSH_CONSTANTS_BYTES; }
    static constexpr size_t getMaxStorageBufferBindings() noexcept { return MAX_STORAGE_BUFFER_BINDINGS; }
    static constexpr size_t getMaxFusedOperations() noexcept { return MAX_FUSED_OPERATIONS; }

    static void setFusedGemmAdamEnabled(bool _enable) noexcept { is_fused_gemm_adam_enabled = _enable; }
    static bool isFusedGemmAdamEnabled() noexcept { return is_fused_gemm_adam_enabled; }
};
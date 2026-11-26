#include "engine/graph_optimizer.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <unordered_set>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "engine/shader_dictionary.h"

std::atomic<bool> Graph_Optimizer::is_fused_gemm_adam_enabled{false};
std::atomic<bool> Graph_Optimizer::is_coop_enabled{false};

void Graph_Optimizer::setFusedGemmAdamEnabled(bool _enable) noexcept
{
    is_fused_gemm_adam_enabled.store(_enable, std::memory_order_release);
}

bool Graph_Optimizer::isFusedGemmAdamEnabled() noexcept
{
    return is_fused_gemm_adam_enabled.load(std::memory_order_acquire);
}

void Graph_Optimizer::setCooperativeMatrixEnabled(bool _enable) noexcept
{
    is_coop_enabled.store(_enable, std::memory_order_release);
}

bool Graph_Optimizer::isCooperativeMatrixEnabled() noexcept
{
    return is_coop_enabled.load(std::memory_order_acquire);
}

namespace
{
VkBuffer getOptimizerDynamicParamsBuffer(const Compute_Node &_node) noexcept
{
    if (_node.fusion_plan && _node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
    {
        const uint32_t dyn_idx = _node.fusion_plan->shared_dyn_params_binding;
        if (dyn_idx < _node.buffers.size() && _node.buffers[dyn_idx])
        {
            return _node.buffers[dyn_idx]->getBuffer();
        }
        return VK_NULL_HANDLE;
    }
    if (_node.is_fused)
    {
        return VK_NULL_HANDLE;
    }
    size_t dyn_idx = SIZE_MAX;
    switch (_node.pipeline_id)
    {
    case Compute_Pipeline::ADAM_UPDATE:
    case Compute_Pipeline::ADAM_UPDATE_FP16:
        dyn_idx = 4;
        break;
    case Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16:
        dyn_idx = 0;
        break;
    case Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16:
    case Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16:
        dyn_idx = 5;
        break;
    case Compute_Pipeline::SGD_UPDATE:
    case Compute_Pipeline::SGD_UPDATE_FP16:
        dyn_idx = 2;
        break;
    case Compute_Pipeline::SGD_MOMENTUM_UPDATE:
    case Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16:
        dyn_idx = 3;
        break;
    default:
        return VK_NULL_HANDLE;
    }
    if (dyn_idx < _node.buffers.size() && _node.buffers[dyn_idx])
    {
        return _node.buffers[dyn_idx]->getBuffer();
    }
    return VK_NULL_HANDLE;
}
}

void Graph_Optimizer::getNodeBufferAccess(
    const Compute_Node &_node,
    const Shader_Dictionary &_shader_dictionary,
    std::vector<VkBuffer> &_reads,
    std::vector<VkBuffer> &_writes)
{
    _reads.clear();
    _writes.clear();

    if (_node.fusion_plan && _node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
    {
        const uint32_t dyn_idx = _node.fusion_plan->shared_dyn_params_binding;
        if (dyn_idx < _node.buffers.size() && _node.buffers[dyn_idx] &&
            _node.buffers[dyn_idx]->getBuffer() != VK_NULL_HANDLE)
        {
            VkBuffer dyn_buf = _node.buffers[dyn_idx]->getBuffer();
            _reads.push_back(dyn_buf);
            _writes.push_back(dyn_buf);
        }
        for (const auto &lane : _node.fusion_plan->horizontal_lanes)
        {
            for (uint32_t idx : {lane.param_buf_idx, lane.grad_buf_idx, lane.m_buf_idx, lane.v_buf_idx})
            {
                if (idx < _node.buffers.size() && _node.buffers[idx] &&
                    _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
                {
                    _reads.push_back(_node.buffers[idx]->getBuffer());
                }
            }
            for (uint32_t idx : {lane.param_buf_idx, lane.m_buf_idx, lane.v_buf_idx, lane.param_fp16_buf_idx})
            {
                if (idx < _node.buffers.size() && _node.buffers[idx] &&
                    _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
                {
                    _writes.push_back(_node.buffers[idx]->getBuffer());
                }
            }
        }
        return;
    }

    if (_node.is_fused && !_node.fused_operations.empty())
    {
        for (const auto &op : _node.fused_operations)
        {
            for (uint32_t in_idx : op.input_buffer_indices)
            {
                if (in_idx < _node.buffers.size() && _node.buffers[in_idx])
                {
                    VkBuffer buf = _node.buffers[in_idx]->getBuffer();
                    if (buf != VK_NULL_HANDLE)
                    {
                        _reads.push_back(buf);
                    }
                }
            }
            for (uint32_t out_idx : op.output_buffer_indices)
            {
                if (out_idx < _node.buffers.size() && _node.buffers[out_idx])
                {
                    VkBuffer buf = _node.buffers[out_idx]->getBuffer();
                    if (buf != VK_NULL_HANDLE)
                    {
                        _writes.push_back(buf);
                    }
                }
            }
        }
        return;
    }

    if (_node.pipeline_id == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16)
    {
        for (size_t idx : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{8}})
        {
            if (idx < _node.buffers.size() && _node.buffers[idx] && _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
            {
                _reads.push_back(_node.buffers[idx]->getBuffer());
            }
        }
        for (size_t idx : {size_t{5}, size_t{9}})
        {
            if (idx < _node.buffers.size() && _node.buffers[idx] && _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
            {
                _writes.push_back(_node.buffers[idx]->getBuffer());
            }
        }
        return;
    }
    if (_node.pipeline_id == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16)
    {
        for (size_t idx : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{8}, size_t{9}})
        {
            if (idx < _node.buffers.size() && _node.buffers[idx] && _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
            {
                _reads.push_back(_node.buffers[idx]->getBuffer());
            }
        }
        for (size_t idx : {size_t{6}, size_t{7}})
        {
            if (idx < _node.buffers.size() && _node.buffers[idx] && _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
            {
                _writes.push_back(_node.buffers[idx]->getBuffer());
            }
        }
        return;
    }

    const auto &metadata = _shader_dictionary.getMetadata(_node.pipeline_id);

    if (!metadata.read_indices.empty() || !metadata.write_indices.empty())
    {
        for (uint32_t idx : metadata.read_indices)
        {
            if (idx < _node.buffers.size() && _node.buffers[idx] && _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
            {
                _reads.push_back(_node.buffers[idx]->getBuffer());
            }
        }
        for (uint32_t idx : metadata.write_indices)
        {
            if (idx < _node.buffers.size() && _node.buffers[idx] && _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
            {
                _writes.push_back(_node.buffers[idx]->getBuffer());
            }
        }
        return;
    }

    for (uint32_t i = 0; i < metadata.input_count && i < _node.buffers.size(); ++i)
    {
        if (_node.buffers[i] && _node.buffers[i]->getBuffer() != VK_NULL_HANDLE)
        {
            _reads.push_back(_node.buffers[i]->getBuffer());
        }
    }

    for (uint32_t i = metadata.input_count;
         i < metadata.input_count + metadata.output_count && i < _node.buffers.size();
         ++i)
    {
        if (_node.buffers[i] && _node.buffers[i]->getBuffer() != VK_NULL_HANDLE)
        {
            _writes.push_back(_node.buffers[i]->getBuffer());
        }
    }

    auto addInPlaceWrite = [&](size_t idx)
    {
        if ((idx < metadata.input_count || idx >= metadata.input_count + metadata.output_count) &&
            idx < _node.buffers.size() &&
            _node.buffers[idx] &&
            _node.buffers[idx]->getBuffer() != VK_NULL_HANDLE)
        {
            _writes.push_back(_node.buffers[idx]->getBuffer());
        }
    };

    if (_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE || _node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16)
    {
        for (size_t idx : {size_t{0}, size_t{2}, size_t{3}, size_t{4}})
        {
            addInPlaceWrite(idx);
        }
        if (_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16)
        {
            addInPlaceWrite(5);
        }
    }
    else if (_node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16)
    {
        addInPlaceWrite(0);
        for (size_t slot = 0; slot < 4; ++slot)
        {
            size_t base = 1 + slot * 5;
            addInPlaceWrite(base + 0);
            addInPlaceWrite(base + 2);
            addInPlaceWrite(base + 3);
            addInPlaceWrite(base + 4);
        }
    }
    else if (_node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16 ||
             _node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16)
    {
        for (size_t idx : {size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{6}, size_t{7}})
        {
            addInPlaceWrite(idx);
        }
    }
    else if (_node.pipeline_id == Compute_Pipeline::SGD_UPDATE || _node.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16)
    {
        addInPlaceWrite(0);
        addInPlaceWrite(2);
        if (_node.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16)
        {
            addInPlaceWrite(3);
        }
    }
    else if (_node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE || _node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16)
    {
        for (size_t idx : {size_t{0}, size_t{2}, size_t{3}})
        {
            addInPlaceWrite(idx);
        }
        if (_node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16)
        {
            addInPlaceWrite(4);
        }
    }
}

void Graph_Optimizer::assignPipelineBarriers(std::vector<Compute_Node> &_nodes, const Shader_Dictionary &_shader_dictionary)
{
    if (_nodes.empty())
    {
        return;
    }

    for (auto &node : _nodes)
    {
        node.is_barrier_required_after = false;
    }

    std::unordered_set<VkBuffer> pending_writes;
    std::unordered_set<VkBuffer> pending_reads;
    std::unordered_set<VkBuffer> pending_non_opt_writes;
    std::unordered_set<VkBuffer> pending_non_opt_reads;

    std::vector<VkBuffer> current_reads;
    std::vector<VkBuffer> current_writes;

    for (size_t i = 0; i < _nodes.size(); ++i)
    {
        getNodeBufferAccess(_nodes[i], _shader_dictionary, current_reads, current_writes);
        const VkBuffer current_dyn = getOptimizerDynamicParamsBuffer(_nodes[i]);

        bool has_hazard = false;

        for (VkBuffer read_buffer : current_reads)
        {
            const bool is_shared_opt_dyn = (current_dyn != VK_NULL_HANDLE && read_buffer == current_dyn);
            const auto &check_writes = is_shared_opt_dyn ? pending_non_opt_writes : pending_writes;
            if (check_writes.contains(read_buffer))
            {
                has_hazard = true;
                break;
            }
        }

        if (!has_hazard)
        {
            for (VkBuffer write_buffer : current_writes)
            {
                const bool is_shared_opt_dyn = (current_dyn != VK_NULL_HANDLE && write_buffer == current_dyn);
                const auto &check_writes = is_shared_opt_dyn ? pending_non_opt_writes : pending_writes;
                const auto &check_reads = is_shared_opt_dyn ? pending_non_opt_reads : pending_reads;
                if (check_writes.contains(write_buffer) || check_reads.contains(write_buffer))
                {
                    has_hazard = true;
                    break;
                }
            }
        }

        if (has_hazard && i > 0)
        {
            _nodes[i - 1].is_barrier_required_after = true;
            pending_writes.clear();
            pending_reads.clear();
            pending_non_opt_writes.clear();
            pending_non_opt_reads.clear();
        }

        for (VkBuffer read_buffer : current_reads)
        {
            pending_reads.insert(read_buffer);
            if (current_dyn == VK_NULL_HANDLE || read_buffer != current_dyn)
            {
                pending_non_opt_reads.insert(read_buffer);
            }
        }
        for (VkBuffer write_buffer : current_writes)
        {
            pending_writes.insert(write_buffer);
            if (current_dyn == VK_NULL_HANDLE || write_buffer != current_dyn)
            {
                pending_non_opt_writes.insert(write_buffer);
            }
        }
    }

    _nodes.back().is_barrier_required_after = true;
}

namespace
{
void scheduleNodesByDependencyLevelImpl(
    std::vector<Compute_Node> &_nodes,
    Cached_Graph_Template *_template,
    bool _is_tracking_mappings,
    const Shader_Dictionary &_shader_dictionary,
    void (*_access_fn)(const Compute_Node &, const Shader_Dictionary &, std::vector<VkBuffer> &, std::vector<VkBuffer> &))
{
    const size_t n = _nodes.size();
    if (n <= 2)
    {
        return;
    }

    std::vector<uint32_t> levels(n, 0);
    std::vector<VkBuffer> raw_reads;
    std::vector<VkBuffer> raw_writes;
    std::unordered_map<VkBuffer, uint32_t> last_writer_level;
    std::unordered_map<VkBuffer, uint32_t> max_reader_level_since_write;
    std::unordered_map<VkBuffer, uint32_t> last_non_opt_writer_level;
    std::unordered_map<VkBuffer, uint32_t> max_non_opt_reader_level_since_write;
    int max_opaque_level = -1;
    uint32_t running_max_level = 0;

    for (size_t i = 0; i < n; ++i)
    {
        _access_fn(_nodes[i], _shader_dictionary, raw_reads, raw_writes);
        const VkBuffer current_dyn = getOptimizerDynamicParamsBuffer(_nodes[i]);

        bool has_null_handle = false;
        for (const auto &buf : _nodes[i].buffers)
        {
            if (buf && buf->getBuffer() == VK_NULL_HANDLE)
            {
                has_null_handle = true;
                break;
            }
        }

        const bool is_opaque = (raw_reads.empty() && raw_writes.empty()) || has_null_handle;
        if (is_opaque)
        {
            const uint32_t lvl = (i == 0) ? 0u : (running_max_level + 1u);
            levels[i] = lvl;
            max_opaque_level = static_cast<int>(lvl);
            last_writer_level.clear();
            max_reader_level_since_write.clear();
            last_non_opt_writer_level.clear();
            max_non_opt_reader_level_since_write.clear();
        }
        else
        {
            uint32_t lvl = (max_opaque_level >= 0) ? static_cast<uint32_t>(max_opaque_level + 1) : 0u;
            for (VkBuffer buf : raw_reads)
            {
                const bool is_shared_opt_dyn = (current_dyn != VK_NULL_HANDLE && buf == current_dyn);
                const auto &writers = is_shared_opt_dyn ? last_non_opt_writer_level : last_writer_level;
                if (auto it = writers.find(buf); it != writers.end())
                {
                    lvl = std::max(lvl, it->second + 1u);
                }
            }
            for (VkBuffer buf : raw_writes)
            {
                const bool is_shared_opt_dyn = (current_dyn != VK_NULL_HANDLE && buf == current_dyn);
                const auto &writers = is_shared_opt_dyn ? last_non_opt_writer_level : last_writer_level;
                const auto &readers = is_shared_opt_dyn ? max_non_opt_reader_level_since_write : max_reader_level_since_write;
                if (auto it = writers.find(buf); it != writers.end())
                {
                    lvl = std::max(lvl, it->second + 1u);
                }
                if (auto it = readers.find(buf); it != readers.end())
                {
                    lvl = std::max(lvl, it->second + 1u);
                }
            }
            levels[i] = lvl;
            for (VkBuffer buf : raw_reads)
            {
                auto &mr = max_reader_level_since_write[buf];
                mr = std::max(mr, lvl);
                if (current_dyn == VK_NULL_HANDLE || buf != current_dyn)
                {
                    auto &non_opt_mr = max_non_opt_reader_level_since_write[buf];
                    non_opt_mr = std::max(non_opt_mr, lvl);
                }
            }
            for (VkBuffer buf : raw_writes)
            {
                last_writer_level[buf] = lvl;
                max_reader_level_since_write.erase(buf);
                if (current_dyn == VK_NULL_HANDLE || buf != current_dyn)
                {
                    last_non_opt_writer_level[buf] = lvl;
                    max_non_opt_reader_level_since_write.erase(buf);
                }
            }
        }
        running_max_level = std::max(running_max_level, levels[i]);
    }

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i)
    {
        order[i] = i;
    }

    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b)
    {
        if (levels[a] != levels[b])
        {
            return levels[a] < levels[b];
        }
        const int fused_a = _nodes[a].is_fused ? 1 : 0;
        const int fused_b = _nodes[b].is_fused ? 1 : 0;
        if (fused_a != fused_b)
        {
            return fused_a < fused_b;
        }
        if (_nodes[a].pipeline_id != _nodes[b].pipeline_id)
        {
            return static_cast<uint32_t>(_nodes[a].pipeline_id) < static_cast<uint32_t>(_nodes[b].pipeline_id);
        }
        return a < b;
    });

    bool changed = false;
    for (size_t i = 0; i < n; ++i)
    {
        if (order[i] != i)
        {
            changed = true;
            break;
        }
    }
    if (!changed)
    {
        return;
    }

    auto applyPerm = [&]<typename Vec>(Vec &vec)
    {
        if (vec.size() != n)
        {
            return;
        }
        Vec reordered;
        reordered.reserve(n);
        for (size_t idx : order)
        {
            reordered.push_back(std::move(vec[idx]));
        }
        vec = std::move(reordered);
    };

    applyPerm(_nodes);
    if (_template != nullptr && _is_tracking_mappings)
    {
        applyPerm(_template->buffer_mappings);
        applyPerm(_template->push_constants_mappings);
        applyPerm(_template->raw_node_indices);
    }
}
}

void Graph_Optimizer::scheduleNodesByDependencyLevel(
    Cached_Graph_Template &_graph_template,
    bool _is_tracking_mappings,
    const Shader_Dictionary &_shader_dictionary)
{
    scheduleNodesByDependencyLevelImpl(
        _graph_template.fused_nodes,
        &_graph_template,
        _is_tracking_mappings,
        _shader_dictionary,
        &Graph_Optimizer::getNodeBufferAccess);
}

void Graph_Optimizer::scheduleNodesByDependencyLevel(
    std::vector<Compute_Node> &_nodes,
    const Shader_Dictionary &_shader_dictionary)
{
    scheduleNodesByDependencyLevelImpl(
        _nodes,
        nullptr,
        false,
        _shader_dictionary,
        &Graph_Optimizer::getNodeBufferAccess);
}

static bool isFp16Pipeline(Compute_Pipeline _pipeline) noexcept
{
    return magic_enum::enum_name(_pipeline).find("FP16") != std::string_view::npos;
}

bool Graph_Optimizer::isCoopMatAnchorPipeline(Compute_Pipeline _pipeline) noexcept
{
    if (!isCooperativeMatrixEnabled())
    {
        return false;
    }
    return (_pipeline == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16 ||
            _pipeline == Compute_Pipeline::MATMUL_COOPMAT_FP16 ||
            _pipeline == Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16);
}

bool Graph_Optimizer::isRowReductionAnchorPipeline(Compute_Pipeline _pipeline) noexcept
{
    return (_pipeline == Compute_Pipeline::RMSNORM_FP16 ||
            _pipeline == Compute_Pipeline::RMSNORM);
}

Pointwise_Op_Kind Graph_Optimizer::classifyPointwiseOp(Compute_Pipeline _pipeline) noexcept
{
    switch (_pipeline)
    {
    case Compute_Pipeline::ADD:
    case Compute_Pipeline::ADD_FP16:
        return Pointwise_Op_Kind::ADD;
    case Compute_Pipeline::SUB:
    case Compute_Pipeline::SUB_FP16:
        return Pointwise_Op_Kind::SUB;
    case Compute_Pipeline::HADAMARD_MUL:
    case Compute_Pipeline::HADAMARD_MUL_FP16:
        return Pointwise_Op_Kind::MUL;
    case Compute_Pipeline::HADAMARD_DIV:
    case Compute_Pipeline::HADAMARD_DIV_FP16:
        return Pointwise_Op_Kind::DIV;
    case Compute_Pipeline::MUL_SCALAR:
    case Compute_Pipeline::MUL_SCALAR_FP16:
        return Pointwise_Op_Kind::MUL_SCALAR;
    case Compute_Pipeline::RELU:
    case Compute_Pipeline::RELU_FP16:
        return Pointwise_Op_Kind::RELU;
    case Compute_Pipeline::SIGMOID_FORWARD:
    case Compute_Pipeline::SIGMOID_FORWARD_FP16:
        return Pointwise_Op_Kind::SIGMOID;
    case Compute_Pipeline::GELU:
    case Compute_Pipeline::GELU_FP16:
        return Pointwise_Op_Kind::GELU;
    default:
        return Pointwise_Op_Kind::NONE;
    }
}

bool Graph_Optimizer::isFusible(
    Operation_Class _producer_class,
    Operation_Class _consumer_class,
    Compute_Pipeline _producer_pipeline,
    Compute_Pipeline _consumer_pipeline,
    const Shader_Dictionary &_shader_dictionary)
{
    if (isRowReductionAnchorPipeline(_consumer_pipeline))
    {
        return (classifyPointwiseOp(_producer_pipeline) != Pointwise_Op_Kind::NONE &&
                isFp16Pipeline(_producer_pipeline) == isFp16Pipeline(_consumer_pipeline));
    }

    if (isRowReductionAnchorPipeline(_producer_pipeline))
    {
        return (classifyPointwiseOp(_consumer_pipeline) != Pointwise_Op_Kind::NONE &&
                isFp16Pipeline(_producer_pipeline) == isFp16Pipeline(_consumer_pipeline));
    }

    if (isCoopMatAnchorPipeline(_producer_pipeline))
    {
        return (classifyPointwiseOp(_consumer_pipeline) != Pointwise_Op_Kind::NONE &&
                isFp16Pipeline(_consumer_pipeline));
    }

    if (classifyPointwiseOp(_producer_pipeline) != Pointwise_Op_Kind::NONE &&
        classifyPointwiseOp(_consumer_pipeline) != Pointwise_Op_Kind::NONE &&
        isFp16Pipeline(_producer_pipeline) == isFp16Pipeline(_consumer_pipeline))
    {
        return true;
    }

    if (_consumer_class != Operation_Class::ELEMENTWISE)
    {
        return false;
    }

    const bool coop_enabled = isCooperativeMatrixEnabled();
    const auto &consumer_metadata = _shader_dictionary.getMetadata(_consumer_pipeline);
    const auto &producer_metadata = _shader_dictionary.getMetadata(_producer_pipeline);

    if (consumer_metadata.is_cooperative_matrix_support)
    {
        return false;
    }

    if (coop_enabled && producer_metadata.is_cooperative_matrix_support)
    {
        return false;
    }

    uint32_t consumer_shared_memory = coop_enabled ? consumer_metadata.cooperative_shared_memory_size : consumer_metadata.shared_memory_size;
    if (consumer_metadata.is_writing_multiple_elements || consumer_shared_memory > 0)
    {
        return false;
    }

    if (producer_metadata.operation_class == Operation_Class::STANDALONE ||
        producer_metadata.is_writing_multiple_elements)
    {
        return false;
    }

    if (producer_metadata.output_count > 1)
    {
        if (producer_metadata.persistent_output_indices.size() >= producer_metadata.output_count)
        {
            return false;
        }
    }

    return (_producer_class == Operation_Class::MATRIX_2D ||
            _producer_class == Operation_Class::ELEMENTWISE ||
            _producer_class == Operation_Class::TENSOR_3D);
}

bool Graph_Optimizer::hasCompatibleDimensions(
    const Compute_Node &_producer_node,
    const Compute_Node &_consumer_node,
    Operation_Class _producer_class,
    Operation_Class _consumer_class,
    const Shader_Dictionary &_shader_dictionary) noexcept
{
    if (_producer_node.fusion_plan && _producer_node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
    {
        return false;
    }

    Compute_Pipeline anchor_pipe = _producer_node.fused_operations.empty()
        ? _producer_node.pipeline_id
        : _producer_node.fused_operations[0].pipeline_id;
    for (const auto &op : _producer_node.fused_operations)
    {
        if (isRowReductionAnchorPipeline(op.pipeline_id))
        {
            anchor_pipe = op.pipeline_id;
            break;
        }
    }

    auto buffersAliasLocal = [](const std::shared_ptr<gpu::vector> &a, const std::shared_ptr<gpu::vector> &b) noexcept -> bool
    {
        return a && b && (a == b || (a->getBuffer() != VK_NULL_HANDLE && a->getBuffer() == b->getBuffer()));
    };

    if (isRowReductionAnchorPipeline(_consumer_node.pipeline_id))
    {
        if (isRowReductionAnchorPipeline(anchor_pipe) || isCoopMatAnchorPipeline(anchor_pipe))
        {
            return false;
        }
        if (_producer_node.push_constants_data.size() < 4 || _consumer_node.push_constants_data.size() < 12)
        {
            return false;
        }
        if (_producer_node.fused_operations.empty() ||
            _producer_node.fused_operations.back().output_buffer_indices.empty() ||
            _consumer_node.buffers.empty())
        {
            return false;
        }
        const uint32_t last_out_idx = _producer_node.fused_operations.back().output_buffer_indices[0];
        if (last_out_idx >= _producer_node.buffers.size() ||
            !buffersAliasLocal(_producer_node.buffers[last_out_idx], _consumer_node.buffers[0]))
        {
            return false;
        }
        const uint32_t rm_rows = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 0);
        const uint32_t rm_cols = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 1);
        const uint64_t rm_elems = static_cast<uint64_t>(rm_rows) * static_cast<uint64_t>(rm_cols);
        const uint32_t prod_elems = readPushConstantWord<uint32_t>(_producer_node.push_constants_data, 0);
        return (rm_rows > 0 && rm_cols > 0 && rm_elems == static_cast<uint64_t>(prod_elems));
    }

    if (isRowReductionAnchorPipeline(anchor_pipe))
    {
        size_t anchor_pc_off = 0;
        const Fused_Operation *anchor_op_ptr = nullptr;
        for (const auto &op : _producer_node.fused_operations)
        {
            if (isRowReductionAnchorPipeline(op.pipeline_id))
            {
                anchor_pc_off = op.push_constants_offset;
                anchor_op_ptr = &op;
                break;
            }
        }
        if (anchor_op_ptr && anchor_op_ptr->output_buffer_indices.size() >= 2)
        {
            uint32_t inv_idx = anchor_op_ptr->output_buffer_indices[1];
            if (inv_idx < _producer_node.buffers.size() && _producer_node.buffers[inv_idx])
            {
                const auto &cons_meta = _shader_dictionary.getMetadata(_consumer_node.pipeline_id);
                const size_t cons_in_limit = std::min<size_t>(cons_meta.input_count, _consumer_node.buffers.size());
                for (size_t in_i = 0; in_i < cons_in_limit; ++in_i)
                {
                    if (buffersAliasLocal(_producer_node.buffers[inv_idx], _consumer_node.buffers[in_i]))
                    {
                        return false;
                    }
                }
            }
        }
        if (anchor_pc_off + 12 > _producer_node.push_constants_data.size() || _consumer_node.push_constants_data.size() < 4)
        {
            return false;
        }
        const uint32_t rm_rows = readPushConstantWord<uint32_t>(_producer_node.push_constants_data, 0, anchor_pc_off);
        const uint32_t rm_cols = readPushConstantWord<uint32_t>(_producer_node.push_constants_data, 1, anchor_pc_off);
        const uint64_t rm_elems = static_cast<uint64_t>(rm_rows) * static_cast<uint64_t>(rm_cols);
        const uint32_t cons_elems = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 0);
        if (rm_rows == 0 || rm_cols == 0 || rm_elems != static_cast<uint64_t>(cons_elems))
        {
            return false;
        }
        if (_consumer_node.push_constants_data.size() >= 12)
        {
            Pointwise_Op_Kind kind = classifyPointwiseOp(_consumer_node.pipeline_id);
            if (kind == Pointwise_Op_Kind::ADD || kind == Pointwise_Op_Kind::SUB ||
                kind == Pointwise_Op_Kind::MUL || kind == Pointwise_Op_Kind::DIV)
            {
                const uint32_t c_cols = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 1);
                const uint32_t c_bcast = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 2);
                if (c_bcast != 0u && c_cols != rm_cols)
                {
                    return false;
                }
            }
        }
        return true;
    }

    if (isCoopMatAnchorPipeline(anchor_pipe))
    {
        if (_consumer_node.push_constants_data.size() < 4)
        {
            return false;
        }
        const uint32_t cons_elems = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 0);
        const uint32_t g_w0 = readPushConstantWord<uint32_t>(_producer_node.push_constants_data, 0);
        const uint32_t g_w1 = readPushConstantWord<uint32_t>(_producer_node.push_constants_data, 1);
        const uint32_t g_w3 = readPushConstantWord<uint32_t>(_producer_node.push_constants_data, 3);
        uint64_t out_rows = 0, out_cols = 0;
        if (anchor_pipe == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16)
        {
            if (_producer_node.push_constants_data.size() < 24 || g_w0 != 1u) return false;
            out_rows = g_w1;
            out_cols = g_w3;
        }
        else if (anchor_pipe == Compute_Pipeline::MATMUL_COOPMAT_FP16)
        {
            if (_producer_node.push_constants_data.size() < 20 || g_w0 != 1u) return false;
            out_rows = g_w1;
            out_cols = g_w3;
        }
        else if (anchor_pipe == Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16)
        {
            if (_producer_node.push_constants_data.size() < 12) return false;
            out_rows = g_w0;
            out_cols = g_w1;
        }
        if (out_rows == 0 || out_cols == 0 || out_rows * out_cols != static_cast<uint64_t>(cons_elems))
        {
            return false;
        }
        if (_consumer_node.push_constants_data.size() >= 12)
        {
            Pointwise_Op_Kind kind = classifyPointwiseOp(_consumer_node.pipeline_id);
            if (kind == Pointwise_Op_Kind::ADD || kind == Pointwise_Op_Kind::SUB ||
                kind == Pointwise_Op_Kind::MUL || kind == Pointwise_Op_Kind::DIV)
            {
                const uint32_t c_cols = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 1);
                const uint32_t c_bcast = readPushConstantWord<uint32_t>(_consumer_node.push_constants_data, 2);
                if (c_bcast != 0u && c_cols != static_cast<uint32_t>(out_cols))
                {
                    return false;
                }
            }
        }
        return true;
    }

    if (_producer_node.workgroup_count_x == _consumer_node.workgroup_count_x &&
        _producer_node.workgroup_count_y == _consumer_node.workgroup_count_y &&
        _producer_node.workgroup_count_z == _consumer_node.workgroup_count_z)
    {
        return true;
    }

    if (_consumer_class == Operation_Class::ELEMENTWISE)
    {
        if (_producer_class == Operation_Class::MATRIX_2D)
        {
            const auto &prod_meta = _shader_dictionary.getMetadata(_producer_node.pipeline_id);
            bool use_coop_grid = isCooperativeMatrixEnabled() && prod_meta.is_cooperative_matrix_support && !prod_meta.cooperative_glsl_template.empty();
            uint64_t total_threads_producer = static_cast<uint64_t>(_producer_node.workgroup_count_x) * (use_coop_grid ? 32 : 16) *
                                                   static_cast<uint64_t>(_producer_node.workgroup_count_y) * (use_coop_grid ? 1 : 16);
            uint64_t total_threads_consumer = static_cast<uint64_t>(_consumer_node.workgroup_count_x) * 256;
            return (total_threads_producer >= total_threads_consumer);
        }
        if (_producer_class == Operation_Class::TENSOR_3D)
        {
            uint64_t total_threads_producer = static_cast<uint64_t>(_producer_node.workgroup_count_x) * 16 *
                                                   static_cast<uint64_t>(_producer_node.workgroup_count_y) * 16 *
                                                   static_cast<uint64_t>(_producer_node.workgroup_count_z);
            uint64_t total_threads_consumer = static_cast<uint64_t>(_consumer_node.workgroup_count_x) * 256;
            return (total_threads_producer >= total_threads_consumer);
        }
    }

    return false;
}

uint32_t Graph_Optimizer::findOrAddBuffer(Compute_Node &_node, const std::shared_ptr<gpu::vector> &_target_buffer)
{
    for (uint32_t index = 0; index < _node.buffers.size(); ++index)
    {
        const auto &existing_buffer = _node.buffers[index];
        if (_target_buffer && existing_buffer &&
            (_target_buffer == existing_buffer ||
             (_target_buffer->getBuffer() != VK_NULL_HANDLE &&
              _target_buffer->getBuffer() == existing_buffer->getBuffer())))
        {
            return index;
        }
    }
    _node.buffers.push_back(_target_buffer);
    return static_cast<uint32_t>(_node.buffers.size() - 1);
}

void Graph_Optimizer::updateDispatchGrid(
    Compute_Node &_fused_node,
    const Compute_Node &_next_node,
    Operation_Class _producer_class,
    Operation_Class _consumer_class,
    const Shader_Dictionary &_shader_dictionary)
{
    if (isRowReductionAnchorPipeline(_next_node.pipeline_id))
    {
        _fused_node.workgroup_count_x = _next_node.workgroup_count_x;
        _fused_node.workgroup_count_y = _next_node.workgroup_count_y;
        _fused_node.workgroup_count_z = _next_node.workgroup_count_z;
        return;
    }

    for (const auto &op : _fused_node.fused_operations)
    {
        if (isRowReductionAnchorPipeline(op.pipeline_id) || isCoopMatAnchorPipeline(op.pipeline_id))
        {
            return;
        }
    }

    const bool coop_enabled = isCooperativeMatrixEnabled();
    const auto &next_metadata = _shader_dictionary.getMetadata(_next_node.pipeline_id);
    uint32_t next_shared_memory = coop_enabled ? next_metadata.cooperative_shared_memory_size : next_metadata.shared_memory_size;
    const bool next_is_pointwise = (classifyPointwiseOp(_next_node.pipeline_id) != Pointwise_Op_Kind::NONE);

    if (!next_is_pointwise &&
        (next_shared_memory > 0 || _consumer_class == Operation_Class::STANDALONE || (coop_enabled && next_metadata.is_cooperative_matrix_support)))
    {
        _fused_node.workgroup_count_x = _next_node.workgroup_count_x;
        _fused_node.workgroup_count_y = 1;
        _fused_node.workgroup_count_z = 1;
        return;
    }

    const bool prod_is_pointwise = !_fused_node.fused_operations.empty() &&
        (classifyPointwiseOp(_fused_node.fused_operations[0].pipeline_id) != Pointwise_Op_Kind::NONE);

    if (_producer_class == Operation_Class::MATRIX_2D ||
        _producer_class == Operation_Class::TENSOR_3D ||
        (_producer_class == Operation_Class::STANDALONE && !prod_is_pointwise))
    {
        return;
    }

    _fused_node.workgroup_count_x = std::max(_fused_node.workgroup_count_x, _next_node.workgroup_count_x);
    _fused_node.workgroup_count_y = std::max(_fused_node.workgroup_count_y, _next_node.workgroup_count_y);
    _fused_node.workgroup_count_z = std::max(_fused_node.workgroup_count_z, _next_node.workgroup_count_z);
}

Fused_Operation Graph_Optimizer::buildFusedOperation(
    Compute_Node &_fused_node,
    const Compute_Node &_source_node,
    uint32_t _push_constants_offset,
    const Snippet_Metadata &_metadata,
    uint32_t _raw_node_index,
    std::vector<Buffer_Binding_Mapping> *_output_buffer_mappings)
{
    Fused_Operation operation{
        .pipeline_id = _source_node.pipeline_id,
        .push_constants_offset = _push_constants_offset,
        .push_constants_size = static_cast<uint32_t>(_source_node.push_constants_data.size()),
        .input_buffer_indices = {},
        .output_buffer_indices = {},
        .workgroup_count_x = _source_node.workgroup_count_x,
        .workgroup_count_y = _source_node.workgroup_count_y,
        .workgroup_count_z = _source_node.workgroup_count_z,
        .spec_constant_u32 = _source_node.spec_constant_u32};

    for (uint32_t buffer_index = 0; buffer_index < _metadata.input_count && buffer_index < _source_node.buffers.size(); ++buffer_index)
    {
        uint32_t fused_buffer_index = findOrAddBuffer(_fused_node, _source_node.buffers[buffer_index]);
        operation.input_buffer_indices.push_back(fused_buffer_index);
        if (_output_buffer_mappings)
        {
            _output_buffer_mappings->push_back(Buffer_Binding_Mapping{
                .raw_node_index = _raw_node_index,
                .raw_buffer_index = buffer_index,
                .fused_buffer_index = fused_buffer_index});
        }
    }

    for (uint32_t buffer_index = _metadata.input_count;
         buffer_index < _metadata.input_count + _metadata.output_count && buffer_index < _source_node.buffers.size();
         ++buffer_index)
    {
        uint32_t fused_buffer_index = findOrAddBuffer(_fused_node, _source_node.buffers[buffer_index]);
        operation.output_buffer_indices.push_back(fused_buffer_index);
        if (_output_buffer_mappings)
        {
            _output_buffer_mappings->push_back(Buffer_Binding_Mapping{
                .raw_node_index = _raw_node_index,
                .raw_buffer_index = buffer_index,
                .fused_buffer_index = fused_buffer_index});
        }

        if (_source_node.external_output_indices.contains(buffer_index))
        {
            _fused_node.external_output_indices.insert(fused_buffer_index);
        }
    }

    return operation;
}

bool Graph_Optimizer::hasAliasingHazard(
    const Compute_Node &_fused_node,
    const Compute_Node &_next_node,
    const Shader_Dictionary &_shader_dictionary)
{
    if (_fused_node.fused_operations.empty())
    {
        return false;
    }

    auto buffersAlias = [](const std::shared_ptr<gpu::vector> &lhs, const std::shared_ptr<gpu::vector> &rhs) -> bool
    {
        return lhs && rhs &&
               (lhs == rhs || (lhs->getBuffer() != VK_NULL_HANDLE && lhs->getBuffer() == rhs->getBuffer()));
    };

    auto isPersistentOrAccum = [](const Snippet_Metadata &meta, uint32_t local_out_idx) -> bool
    {
        return std::find(meta.persistent_output_indices.begin(), meta.persistent_output_indices.end(), local_out_idx) !=
                   meta.persistent_output_indices.end() ||
               std::find(meta.accumulator_output_indices.begin(), meta.accumulator_output_indices.end(), local_out_idx) !=
                   meta.accumulator_output_indices.end();
    };

    const Snippet_Metadata &next_metadata = _shader_dictionary.getMetadata(_next_node.pipeline_id);

    std::vector<std::shared_ptr<gpu::vector>> next_input_buffers;
    for (uint32_t i = 0; i < next_metadata.input_count && i < _next_node.buffers.size(); ++i)
    {
        if (_next_node.buffers[i])
        {
            next_input_buffers.push_back(_next_node.buffers[i]);
        }
    }

    std::vector<std::shared_ptr<gpu::vector>> next_output_buffers;
    for (uint32_t i = next_metadata.input_count;
         i < next_metadata.input_count + next_metadata.output_count && i < _next_node.buffers.size();
         ++i)
    {
        if (_next_node.buffers[i])
        {
            next_output_buffers.push_back(_next_node.buffers[i]);
        }
    }

    for (const auto &output_buffer : next_output_buffers)
    {
        for (const auto &fused_operation : _fused_node.fused_operations)
        {
            for (uint32_t input_index : fused_operation.input_buffer_indices)
            {
                if (input_index < _fused_node.buffers.size() &&
                    buffersAlias(output_buffer, _fused_node.buffers[input_index]))
                {
                    return true;
                }
            }
        }
    }

    for (const auto &output_buffer : next_output_buffers)
    {
        for (size_t operation_index = 0; operation_index + 1 < _fused_node.fused_operations.size(); ++operation_index)
        {
            const auto &fused_operation = _fused_node.fused_operations[operation_index];
            for (uint32_t output_index : fused_operation.output_buffer_indices)
            {
                if (output_index < _fused_node.buffers.size() &&
                    buffersAlias(output_buffer, _fused_node.buffers[output_index]))
                {
                    return true;
                }
            }
        }

        const auto &back_op = _fused_node.fused_operations.back();
        const auto &back_meta = _shader_dictionary.getMetadata(back_op.pipeline_id);
        for (uint32_t local_out_idx = 0; local_out_idx < static_cast<uint32_t>(back_op.output_buffer_indices.size()); ++local_out_idx)
        {
            if (!isPersistentOrAccum(back_meta, local_out_idx))
            {
                continue;
            }
            const uint32_t output_index = back_op.output_buffer_indices[local_out_idx];
            if (output_index < _fused_node.buffers.size() &&
                buffersAlias(output_buffer, _fused_node.buffers[output_index]))
            {
                return true;
            }
        }
    }

    for (const auto &fused_operation : _fused_node.fused_operations)
    {
        const auto &op_meta = _shader_dictionary.getMetadata(fused_operation.pipeline_id);
        for (uint32_t local_out_idx = 0; local_out_idx < static_cast<uint32_t>(fused_operation.output_buffer_indices.size()); ++local_out_idx)
        {
            if (!isPersistentOrAccum(op_meta, local_out_idx))
            {
                continue;
            }
            const uint32_t output_index = fused_operation.output_buffer_indices[local_out_idx];
            if (output_index >= _fused_node.buffers.size())
            {
                continue;
            }
            for (const auto &input_buffer : next_input_buffers)
            {
                if (buffersAlias(input_buffer, _fused_node.buffers[output_index]))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

void Graph_Optimizer::markExternalOutputs(
    Compute_Node &_fused_node,
    const std::vector<Compute_Node> &_nodes,
    size_t _next_raw_node_index,
    const std::vector<Compute_Node> &_original_nodes,
    const std::vector<Compute_Node> &_built_fused_nodes)
{
    std::unordered_set<uint32_t> all_output_indices;
    for (const auto &operation : _fused_node.fused_operations)
    {
        for (uint32_t output_index : operation.output_buffer_indices)
        {
            all_output_indices.insert(output_index);
        }
    }

    if (!_fused_node.fused_operations.empty())
    {
        for (uint32_t output_index : _fused_node.fused_operations.back().output_buffer_indices)
        {
            _fused_node.external_output_indices.insert(output_index);
        }
    }

    auto countInternalRefs = [&](const gpu::vector *ptr) -> long
    {
        long count = 0;
        for (const auto &n : _original_nodes)
        {
            for (const auto &b : n.buffers)
            {
                if (b.get() == ptr) ++count;
            }
        }
        for (const auto &n : _nodes)
        {
            for (const auto &b : n.buffers)
            {
                if (b.get() == ptr) ++count;
            }
        }
        for (const auto &n : _built_fused_nodes)
        {
            for (const auto &b : n.buffers)
            {
                if (b.get() == ptr) ++count;
            }
        }
        for (const auto &b : _fused_node.buffers)
        {
            if (b.get() == ptr) ++count;
        }
        return count;
    };

    for (uint32_t output_buffer_index : all_output_indices)
    {
        if (output_buffer_index >= _fused_node.buffers.size())
        {
            continue;
        }

        const auto &buffer = _fused_node.buffers[output_buffer_index];
        if (!buffer)
        {
            continue;
        }

        if (buffer.use_count() > countInternalRefs(buffer.get()))
        {
            _fused_node.external_output_indices.insert(output_buffer_index);
            continue;
        }

        for (size_t j = _next_raw_node_index; j < _nodes.size(); ++j)
        {
            bool found_in_future = false;
            for (const auto &future_buffer : _nodes[j].buffers)
            {
                if (future_buffer &&
                    (buffer == future_buffer ||
                     (buffer->getBuffer() != VK_NULL_HANDLE &&
                      buffer->getBuffer() == future_buffer->getBuffer())))
                {
                    _fused_node.external_output_indices.insert(output_buffer_index);
                    found_in_future = true;
                    break;
                }
            }
            if (found_in_future)
            {
                break;
            }
        }
    }
}

std::vector<Compute_Node> Graph_Optimizer::fuseLinearBackwardAdamPass(
    const std::vector<Compute_Node> &_nodes,
    std::vector<Effective_Node_Provenance> *_out_provenance)
{
    if (_out_provenance)
    {
        _out_provenance->clear();
    }

    if (!is_fused_gemm_adam_enabled.load(std::memory_order_acquire) || _nodes.size() < 2)
    {
        if (_out_provenance)
        {
            _out_provenance->reserve(_nodes.size());
            for (size_t i = 0; i < _nodes.size(); ++i)
            {
                _out_provenance->push_back(Effective_Node_Provenance{
                    .primary_raw_index = static_cast<uint32_t>(i),
                    .fused_adam_raw_index = -1,
                    .horizontal_raw_indices = {}});
            }
        }
        return _nodes;
    }

    std::vector<Compute_Node> result = _nodes;
    std::vector<bool> consumed(result.size(), false);
    std::vector<int32_t> fused_adam_indices(result.size(), -1);

    auto nodeUsesBuffer = [](const Compute_Node &node, VkBuffer handle) -> bool
    {
        if (handle == VK_NULL_HANDLE)
        {
            return false;
        }
        for (const auto &buf : node.buffers)
        {
            if (buf && buf->getBuffer() == handle)
            {
                return true;
            }
        }
        return false;
    };

    for (size_t i = 0; i < result.size(); ++i)
    {
        if (consumed[i])
        {
            continue;
        }

        auto &node_bwd = result[i];
        bool is_bwd_weight = (node_bwd.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16 ||
                              node_bwd.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_FP16);
        if (!is_bwd_weight || node_bwd.buffers.size() < 4 || node_bwd.push_constants_data.size() < 16)
        {
            continue;
        }

        uint32_t batch_size = readPushConstantWord<uint32_t>(node_bwd.push_constants_data, 0);
        uint32_t in_dim = readPushConstantWord<uint32_t>(node_bwd.push_constants_data, 1);
        uint32_t out_dim = readPushConstantWord<uint32_t>(node_bwd.push_constants_data, 2);
        uint32_t accumulate = readPushConstantWord<uint32_t>(node_bwd.push_constants_data, 3);

        if (accumulate != 0)
        {
            continue;
        }
        if (node_bwd.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16 &&
            node_bwd.push_constants_data.size() >= 24 &&
            readPushConstantWord<uint32_t>(node_bwd.push_constants_data, 5) != 0u)
        {
            continue;
        }

        auto dw_buf = node_bwd.buffers[2];
        if (!dw_buf || dw_buf->getBuffer() == VK_NULL_HANDLE)
        {
            continue;
        }

        VkBuffer dw_handle = dw_buf->getBuffer();

        int matching_adam_index = -1;
        for (size_t j = i + 1; j < result.size(); ++j)
        {
            if (consumed[j])
            {
                continue;
            }

            auto &node_adam = result[j];
            bool is_adam = (node_adam.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16);
            if (is_adam && node_adam.buffers.size() >= 6 && node_adam.push_constants_data.size() >= 24)
            {
                if (node_adam.buffers[1] && node_adam.buffers[1]->getBuffer() == dw_handle)
                {
                    uint32_t total_elements = readPushConstantWord<uint32_t>(node_adam.push_constants_data, 0);
                    if (static_cast<uint64_t>(total_elements) == static_cast<uint64_t>(in_dim) * static_cast<uint64_t>(out_dim))
                    {
                        bool has_intervening_hazard = false;
                        VkBuffer w_handle = node_adam.buffers[0] ? node_adam.buffers[0]->getBuffer() : VK_NULL_HANDLE;
                        VkBuffer m_handle = node_adam.buffers[2] ? node_adam.buffers[2]->getBuffer() : VK_NULL_HANDLE;
                        VkBuffer v_handle = node_adam.buffers[3] ? node_adam.buffers[3]->getBuffer() : VK_NULL_HANDLE;
                        VkBuffer w_fp16_handle = node_adam.buffers[5] ? node_adam.buffers[5]->getBuffer() : VK_NULL_HANDLE;

                        if (dw_handle == w_handle || dw_handle == m_handle ||
                            dw_handle == v_handle || dw_handle == w_fp16_handle)
                        {
                            has_intervening_hazard = true;
                        }

                        for (size_t k = i + 1; !has_intervening_hazard && k < result.size(); ++k)
                        {
                            if (k == j || consumed[k])
                            {
                                continue;
                            }
                            if (nodeUsesBuffer(result[k], dw_handle))
                            {
                                has_intervening_hazard = true;
                                break;
                            }
                            if (k < j)
                            {
                                if (nodeUsesBuffer(result[k], w_handle) ||
                                    nodeUsesBuffer(result[k], m_handle) ||
                                    nodeUsesBuffer(result[k], v_handle) ||
                                    nodeUsesBuffer(result[k], w_fp16_handle))
                                {
                                    has_intervening_hazard = true;
                                    break;
                                }
                            }
                        }

                        if (!has_intervening_hazard)
                        {
                            matching_adam_index = static_cast<int>(j);
                        }
                        break;
                    }
                }
            }
        }

        if (matching_adam_index >= 0)
        {
            auto &node_adam = result[matching_adam_index];
            float beta1 = readPushConstantWord<float>(node_adam.push_constants_data, 1);
            float beta2 = readPushConstantWord<float>(node_adam.push_constants_data, 2);
            float epsilon = readPushConstantWord<float>(node_adam.push_constants_data, 3);
            float max_grad = readPushConstantWord<float>(node_adam.push_constants_data, 4);
            float weight_decay = readPushConstantWord<float>(node_adam.push_constants_data, 5);

            struct Fused_PC
            {
                uint32_t batch_size;
                uint32_t in_dim;
                uint32_t out_dim;
                float beta1;
                float beta2;
                float eps;
                float max_grad;
                float weight_decay;
            } fused_pc{batch_size, in_dim, out_dim, beta1, beta2, epsilon, max_grad, weight_decay};

            bool use_coop = (node_bwd.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16);
            node_bwd.pipeline_id = use_coop
                ? Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16
                : Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16;

            node_bwd.buffers = {
                node_bwd.buffers[0],
                node_bwd.buffers[1],
                node_adam.buffers[0],
                node_adam.buffers[2],
                node_adam.buffers[3],
                node_adam.buffers[4],
                node_adam.buffers[5],
                node_bwd.buffers[3]
            };

            node_bwd.push_constants_data.resize(sizeof(Fused_PC));
            std::memcpy(node_bwd.push_constants_data.data(), &fused_pc, sizeof(Fused_PC));
            uint32_t tile = use_coop ? 64u : 16u;
            node_bwd.workgroup_count_x = (out_dim + tile - 1u) / tile;
            node_bwd.workgroup_count_y = (in_dim + tile - 1u) / tile;
            node_bwd.workgroup_count_z = 1;

            consumed[matching_adam_index] = true;
            fused_adam_indices[i] = matching_adam_index;
        }
    }

    std::vector<Compute_Node> compacted;
    compacted.reserve(result.size());
    if (_out_provenance)
    {
        _out_provenance->reserve(result.size());
    }
    size_t fused_count = 0;
    for (size_t i = 0; i < result.size(); ++i)
    {
        if (!consumed[i])
        {
            compacted.push_back(std::move(result[i]));
            if (_out_provenance)
            {
                _out_provenance->push_back(Effective_Node_Provenance{
                    .primary_raw_index = static_cast<uint32_t>(i),
                    .fused_adam_raw_index = fused_adam_indices[i],
                    .horizontal_raw_indices = {}});
            }
        }
        else
        {
            ++fused_count;
        }
    }
    if (fused_count > 0)
    {
        VML_LOG_DEBUG(Log_Feature::OPERATOR_FUSION,
                      "Graph_Optimizer::fuseLinearBackwardAdamPass: fused_adam_pairs={}, before={}, after={}",
                      fused_count, result.size(), compacted.size());
    }
    return compacted;
}

void Graph_Optimizer::fuseHorizontalAdamPass(
    std::vector<Compute_Node> &_nodes,
    std::vector<Effective_Node_Provenance> &_provenance,
    const Shader_Dictionary &_shader_dictionary)
{
    if (_nodes.size() < 2)
    {
        return;
    }

    constexpr size_t MAX_LANES = 6;
    std::vector<bool> consumed(_nodes.size(), false);
    std::vector<Compute_Node> out_nodes;
    std::vector<Effective_Node_Provenance> out_prov;
    out_nodes.reserve(_nodes.size());
    out_prov.reserve(_nodes.size());

    auto buffersOverlap = [&](size_t node_a, size_t node_b, VkBuffer ignore_shared_buf = VK_NULL_HANDLE) -> bool
    {
        std::vector<VkBuffer> ra, wa, rb, wb;
        getNodeBufferAccess(_nodes[node_a], _shader_dictionary, ra, wa);
        getNodeBufferAccess(_nodes[node_b], _shader_dictionary, rb, wb);
        if ((ra.empty() && wa.empty()) || (rb.empty() && wb.empty()))
        {
            return true;
        }
        std::unordered_set<VkBuffer> touched_b;
        for (VkBuffer h : rb) if (h != VK_NULL_HANDLE && h != ignore_shared_buf) touched_b.insert(h);
        for (VkBuffer h : wb) if (h != VK_NULL_HANDLE && h != ignore_shared_buf) touched_b.insert(h);
        for (VkBuffer h : wa)
        {
            if (h != VK_NULL_HANDLE && h != ignore_shared_buf && touched_b.contains(h)) return true;
        }
        std::unordered_set<VkBuffer> writes_b;
        for (VkBuffer h : wb) if (h != VK_NULL_HANDLE && h != ignore_shared_buf) writes_b.insert(h);
        for (VkBuffer h : ra)
        {
            if (h != VK_NULL_HANDLE && h != ignore_shared_buf && writes_b.contains(h)) return true;
        }
        return false;
    };

    for (size_t i = 0; i < _nodes.size(); ++i)
    {
        if (consumed[i])
        {
            continue;
        }

        const Compute_Node &anchor = _nodes[i];
        bool is_adam_lane = (!anchor.is_fused &&
                             anchor.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16 &&
                             anchor.buffers.size() >= 6 &&
                             anchor.push_constants_data.size() >= 24 &&
                             (i >= _provenance.size() || _provenance[i].fused_adam_raw_index < 0));
        if (!is_adam_lane)
        {
            out_nodes.push_back(_nodes[i]);
            if (i < _provenance.size()) out_prov.push_back(_provenance[i]);
            continue;
        }

        const uint32_t a_w1 = readPushConstantWord<uint32_t>(anchor.push_constants_data, 1);
        const uint32_t a_w2 = readPushConstantWord<uint32_t>(anchor.push_constants_data, 2);
        const uint32_t a_w3 = readPushConstantWord<uint32_t>(anchor.push_constants_data, 3);
        const uint32_t a_w4 = readPushConstantWord<uint32_t>(anchor.push_constants_data, 4);
        VkBuffer a_dyn = anchor.buffers[4] ? anchor.buffers[4]->getBuffer() : VK_NULL_HANDLE;
        std::vector<size_t> group = {i};

        for (size_t j = i + 1; j < _nodes.size() && group.size() < MAX_LANES; ++j)
        {
            if (consumed[j])
            {
                continue;
            }
            const Compute_Node &cand = _nodes[j];
            bool is_cand = (!cand.is_fused &&
                            cand.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16 &&
                            cand.buffers.size() >= 6 &&
                            cand.push_constants_data.size() >= 24 &&
                            (j >= _provenance.size() || _provenance[j].fused_adam_raw_index < 0));
            if (!is_cand)
            {
                continue;
            }

            VkBuffer c_dyn = cand.buffers[4] ? cand.buffers[4]->getBuffer() : VK_NULL_HANDLE;
            if (c_dyn != a_dyn)
            {
                continue;
            }
            if (a_w1 != readPushConstantWord<uint32_t>(cand.push_constants_data, 1) ||
                a_w2 != readPushConstantWord<uint32_t>(cand.push_constants_data, 2) ||
                a_w3 != readPushConstantWord<uint32_t>(cand.push_constants_data, 3) ||
                a_w4 != readPushConstantWord<uint32_t>(cand.push_constants_data, 4))
            {
                continue;
            }

            bool disjoint_from_group = true;
            for (size_t g_idx : group)
            {
                if (buffersOverlap(g_idx, j, a_dyn))
                {
                    disjoint_from_group = false;
                    break;
                }
            }
            if (!disjoint_from_group)
            {
                break;
            }

            bool blocked = false;
            for (size_t k = i + 1; k < j; ++k)
            {
                if (consumed[k]) continue;
                bool k_in_group = false;
                for (size_t g_idx : group)
                {
                    if (g_idx == k) { k_in_group = true; break; }
                }
                if (k_in_group) continue;
                const VkBuffer k_dyn = getOptimizerDynamicParamsBuffer(_nodes[k]);
                const VkBuffer ignore_k = (k_dyn != VK_NULL_HANDLE && k_dyn == a_dyn) ? a_dyn : VK_NULL_HANDLE;
                if (buffersOverlap(k, j, ignore_k))
                {
                    blocked = true;
                    break;
                }
            }
            if (blocked)
            {
                break;
            }

            group.push_back(j);
        }

        if (group.size() < 2)
        {
            out_nodes.push_back(_nodes[i]);
            if (i < _provenance.size()) out_prov.push_back(_provenance[i]);
            continue;
        }

        Compute_Node merged;
        merged.pipeline_id = Compute_Pipeline::ADAM_UPDATE_FP16;
        merged.is_fused = true;
        merged.buffers.push_back(anchor.buffers[4]);

        auto plan = std::make_shared<Fused_Subgraph_Plan>();
        plan->anchor_role = Fused_Anchor_Role::HORIZONTAL_MULTI_LANE;
        plan->is_fp16 = true;
        plan->use_vec4 = true;
        plan->local_size_x = 256;
        plan->shared_dyn_params_binding = 0;

        Effective_Node_Provenance prov;
        prov.primary_raw_index = (i < _provenance.size()) ? _provenance[i].primary_raw_index : static_cast<uint32_t>(i);

        const uint32_t num_lanes = static_cast<uint32_t>(group.size());
        merged.push_constants_data.resize(20 + 12 * num_lanes, 0);
        std::memcpy(merged.push_constants_data.data(), anchor.push_constants_data.data() + 4, 16);
        std::memcpy(merged.push_constants_data.data() + 16, &num_lanes, 4);

        uint32_t wg_cursor = 0;
        for (size_t lane_idx = 0; lane_idx < group.size(); ++lane_idx)
        {
            size_t src_i = group[lane_idx];
            consumed[src_i] = true;
            const Compute_Node &src = _nodes[src_i];
            uint32_t raw_idx = (src_i < _provenance.size()) ? _provenance[src_i].primary_raw_index : static_cast<uint32_t>(src_i);
            prov.horizontal_raw_indices.push_back(raw_idx);

            const uint32_t b_base = static_cast<uint32_t>(merged.buffers.size());
            merged.buffers.push_back(src.buffers[0]);
            merged.buffers.push_back(src.buffers[1]);
            merged.buffers.push_back(src.buffers[2]);
            merged.buffers.push_back(src.buffers[3]);
            merged.buffers.push_back(src.buffers[5]);

            uint32_t elem_count = 0;
            float wd = 0.0f;
            std::memcpy(&elem_count, src.push_constants_data.data(), 4);
            std::memcpy(&wd, src.push_constants_data.data() + 20, 4);

            const uint32_t lane_wgs = std::max<uint32_t>(1u, (elem_count + 1023u) / 1024u);
            wg_cursor += lane_wgs;

            Horizontal_Lane_Spec lspec{};
            lspec.param_buf_idx = b_base + 0;
            lspec.grad_buf_idx = b_base + 1;
            lspec.m_buf_idx = b_base + 2;
            lspec.v_buf_idx = b_base + 3;
            lspec.param_fp16_buf_idx = b_base + 4;
            lspec.element_count = elem_count;
            lspec.wg_end = wg_cursor;
            lspec.weight_decay = wd;
            plan->horizontal_lanes.push_back(lspec);

            const size_t pc_off = 20 + 12 * lane_idx;
            std::memcpy(merged.push_constants_data.data() + pc_off + 0, &elem_count, 4);
            std::memcpy(merged.push_constants_data.data() + pc_off + 4, &wg_cursor, 4);
            std::memcpy(merged.push_constants_data.data() + pc_off + 8, &wd, 4);

            merged.external_output_indices.insert(b_base + 0);
            merged.external_output_indices.insert(b_base + 2);
            merged.external_output_indices.insert(b_base + 3);
            merged.external_output_indices.insert(b_base + 4);

            merged.fused_operations.push_back(Fused_Operation{
                .pipeline_id = Compute_Pipeline::ADAM_UPDATE_FP16,
                .push_constants_offset = static_cast<uint32_t>(pc_off),
                .push_constants_size = 24,
                .input_buffer_indices = {b_base + 0, b_base + 1, b_base + 2, b_base + 3, 0},
                .output_buffer_indices = {b_base + 4},
                .workgroup_count_x = src.workgroup_count_x,
                .workgroup_count_y = src.workgroup_count_y,
                .workgroup_count_z = src.workgroup_count_z});
        }

        uint32_t eff_wgs = std::max<uint32_t>(1u, wg_cursor);
        if (eff_wgs <= 65535u)
        {
            merged.workgroup_count_x = eff_wgs;
            merged.workgroup_count_y = 1;
        }
        else
        {
            merged.workgroup_count_x = 65535u;
            merged.workgroup_count_y = (eff_wgs + 65534u) / 65535u;
        }
        merged.workgroup_count_z = 1;
        plan->canonical_key = std::format("h_adam_fp16_k{}_g2d{}", num_lanes, (merged.workgroup_count_y > 1u ? 1u : 0u));
        merged.fusion_plan = std::move(plan);

        out_nodes.push_back(std::move(merged));
        out_prov.push_back(std::move(prov));
    }

    _nodes = std::move(out_nodes);
    _provenance = std::move(out_prov);
}

void Graph_Optimizer::finalizeFusedNodePlan(Compute_Node &_fused_node, const Shader_Dictionary &_shader_dictionary)
{
    (void)_shader_dictionary;
    if (!_fused_node.is_fused || _fused_node.fused_operations.size() <= 1 || _fused_node.fusion_plan)
    {
        return;
    }

    auto collectMaterialized = [&](size_t skip_last_op)
    {
        std::vector<uint32_t> mats;
        std::unordered_set<uint32_t> seen;
        for (size_t i = 0; i + skip_last_op < _fused_node.fused_operations.size(); ++i)
        {
            for (uint32_t out_idx : _fused_node.fused_operations[i].output_buffer_indices)
            {
                if (_fused_node.external_output_indices.contains(out_idx) && seen.insert(out_idx).second)
                {
                    mats.push_back(out_idx);
                }
            }
        }
        return mats;
    };

    auto makeStep = [](const Fused_Operation &op, const Compute_Node &node) -> Fused_Op_Step
    {
        Fused_Op_Step step;
        step.pipeline_id = op.pipeline_id;
        step.op_kind = classifyPointwiseOp(op.pipeline_id);
        step.input_buffer_indices = op.input_buffer_indices;
        step.output_buffer_index = op.output_buffer_indices.empty() ? 0u : op.output_buffer_indices[0];
        step.pc_offset_bytes = op.push_constants_offset;
        step.pc_size_bytes = op.push_constants_size;
        if ((step.op_kind == Pointwise_Op_Kind::ADD || step.op_kind == Pointwise_Op_Kind::SUB ||
             step.op_kind == Pointwise_Op_Kind::MUL || step.op_kind == Pointwise_Op_Kind::DIV) &&
            op.push_constants_size >= 12 &&
            op.push_constants_offset + 12 <= node.push_constants_data.size())
        {
            if (readPushConstantWord<uint32_t>(node.push_constants_data, 2, op.push_constants_offset) != 0u)
            {
                step.rhs_index_mode = Index_Map_Kind::BROADCAST_ROW_1D;
                step.num_columns = readPushConstantWord<uint32_t>(node.push_constants_data, 1, op.push_constants_offset);
            }
        }
        return step;
    };

    auto appendStructuralKey = [&](std::string &key, const Fused_Subgraph_Plan &plan)
    {
        const size_t pc_words = (_fused_node.push_constants_data.size() + 3) / 4;
        key += std::format("_w{}", pc_words);
        for (const auto &op : _fused_node.fused_operations)
        {
            key += std::format("_op{}:{}:{}[", static_cast<int>(op.pipeline_id), op.push_constants_offset, op.push_constants_size);
            for (uint32_t ib : op.input_buffer_indices) key += std::format("{},", ib);
            key += "->";
            for (uint32_t ob : op.output_buffer_indices) key += std::format("{},", ob);
            key += "]";
        }
        for (const auto &s : plan.prologue_steps)
        {
            key += std::format("_p{}_{}_{}_{}", static_cast<int>(s.op_kind), static_cast<int>(s.rhs_index_mode), s.pc_offset_bytes, s.pc_size_bytes);
        }
        for (const auto &s : plan.epilogue_steps)
        {
            key += std::format("_e{}_{}_{}_{}", static_cast<int>(s.op_kind), static_cast<int>(s.rhs_index_mode), s.pc_offset_bytes, s.pc_size_bytes);
        }
        std::vector<uint32_t> ext_sorted(_fused_node.external_output_indices.begin(), _fused_node.external_output_indices.end());
        std::sort(ext_sorted.begin(), ext_sorted.end());
        for (uint32_t ex : ext_sorted) key += std::format("_x{}", ex);
        for (uint32_t m : plan.materialized_buffer_indices) key += std::format("_m{}", m);
    };

    size_t row_anchor_idx = SIZE_MAX;
    for (size_t i = 0; i < _fused_node.fused_operations.size(); ++i)
    {
        if (isRowReductionAnchorPipeline(_fused_node.fused_operations[i].pipeline_id))
        {
            row_anchor_idx = i;
            break;
        }
    }

    if (row_anchor_idx != SIZE_MAX)
    {
        for (size_t i = 0; i < _fused_node.fused_operations.size(); ++i)
        {
            if (i != row_anchor_idx && classifyPointwiseOp(_fused_node.fused_operations[i].pipeline_id) == Pointwise_Op_Kind::NONE)
            {
                return;
            }
        }
        const auto &anchor_op = _fused_node.fused_operations[row_anchor_idx];
        if (anchor_op.output_buffer_indices.size() >= 2)
        {
            _fused_node.external_output_indices.insert(anchor_op.output_buffer_indices[1]);
        }
        auto plan = std::make_shared<Fused_Subgraph_Plan>();
        plan->anchor_role = Fused_Anchor_Role::ROW_REDUCTION;
        plan->is_fp16 = (anchor_op.pipeline_id == Compute_Pipeline::RMSNORM_FP16);
        plan->use_vec4 = false;
        plan->local_size_x = 64;
        for (size_t i = 0; i < row_anchor_idx; ++i)
        {
            plan->prologue_steps.push_back(makeStep(_fused_node.fused_operations[i], _fused_node));
        }
        for (size_t i = row_anchor_idx + 1; i < _fused_node.fused_operations.size(); ++i)
        {
            plan->epilogue_steps.push_back(makeStep(_fused_node.fused_operations[i], _fused_node));
        }
        plan->materialized_buffer_indices = collectMaterialized(1);
        if (anchor_op.output_buffer_indices.size() >= 2)
        {
            uint32_t inv_buf = anchor_op.output_buffer_indices[1];
            if (std::find(plan->materialized_buffer_indices.begin(), plan->materialized_buffer_indices.end(), inv_buf) == plan->materialized_buffer_indices.end())
            {
                plan->materialized_buffer_indices.push_back(inv_buf);
            }
        }
        std::sort(plan->materialized_buffer_indices.begin(), plan->materialized_buffer_indices.end());
        std::string key = std::format("rowred_{}", static_cast<int>(anchor_op.pipeline_id));
        appendStructuralKey(key, *plan);
        plan->canonical_key = std::move(key);
        _fused_node.fusion_plan = std::move(plan);
        return;
    }

    if (isCoopMatAnchorPipeline(_fused_node.fused_operations[0].pipeline_id))
    {
        for (size_t i = 1; i < _fused_node.fused_operations.size(); ++i)
        {
            if (classifyPointwiseOp(_fused_node.fused_operations[i].pipeline_id) == Pointwise_Op_Kind::NONE)
            {
                return;
            }
        }
        auto plan = std::make_shared<Fused_Subgraph_Plan>();
        plan->anchor_role = Fused_Anchor_Role::GEMM_COOPMAT;
        plan->is_fp16 = true;
        plan->local_size_x = 128;
        for (size_t i = 1; i < _fused_node.fused_operations.size(); ++i)
        {
            plan->epilogue_steps.push_back(makeStep(_fused_node.fused_operations[i], _fused_node));
        }
        plan->materialized_buffer_indices = collectMaterialized(1);
        std::sort(plan->materialized_buffer_indices.begin(), plan->materialized_buffer_indices.end());
        std::string key = std::format("gemm_{}", static_cast<int>(_fused_node.fused_operations[0].pipeline_id));
        appendStructuralKey(key, *plan);
        plan->canonical_key = std::move(key);
        _fused_node.fusion_plan = std::move(plan);
        return;
    }

    bool all_pointwise = true;
    bool all_fp16 = true;
    bool any_broadcast = false;
    for (const auto &op : _fused_node.fused_operations)
    {
        Pointwise_Op_Kind k = classifyPointwiseOp(op.pipeline_id);
        if (k == Pointwise_Op_Kind::NONE)
        {
            all_pointwise = false;
            break;
        }
        if (magic_enum::enum_name(op.pipeline_id).find("FP16") == std::string_view::npos)
        {
            all_fp16 = false;
        }
        Fused_Op_Step st = makeStep(op, _fused_node);
        if (st.rhs_index_mode != Index_Map_Kind::FLAT_1D)
        {
            any_broadcast = true;
        }
    }
    if (all_pointwise)
    {
        auto plan = std::make_shared<Fused_Subgraph_Plan>();
        plan->anchor_role = Fused_Anchor_Role::POINTWISE_CHAIN;
        plan->is_fp16 = all_fp16;
        plan->use_vec4 = all_fp16 && !any_broadcast;
        plan->local_size_x = 256;
        for (const auto &op : _fused_node.fused_operations)
        {
            plan->epilogue_steps.push_back(makeStep(op, _fused_node));
        }
        plan->materialized_buffer_indices = collectMaterialized(1);
        std::sort(plan->materialized_buffer_indices.begin(), plan->materialized_buffer_indices.end());
        std::string key = std::format("pw_fp{}_v{}", all_fp16 ? 16 : 32, plan->use_vec4 ? 4 : 1);
        appendStructuralKey(key, *plan);
        plan->canonical_key = std::move(key);
        if (plan->use_vec4 && _fused_node.push_constants_data.size() >= 4)
        {
            uint32_t total_elems = readPushConstantWord<uint32_t>(_fused_node.push_constants_data, 0);
            uint32_t vec_wgs = std::max<uint32_t>(1u, (total_elems + 1023u) / 1024u);
            if (vec_wgs <= 65535u)
            {
                _fused_node.workgroup_count_x = vec_wgs;
                _fused_node.workgroup_count_y = 1;
            }
            else
            {
                _fused_node.workgroup_count_x = 65535u;
                _fused_node.workgroup_count_y = (vec_wgs + 65534u) / 65535u;
            }
            _fused_node.workgroup_count_z = 1;
        }
        _fused_node.fusion_plan = std::move(plan);
    }
}

Cached_Graph_Template Graph_Optimizer::optimizeInternal(
    const std::vector<Compute_Node>& _original_nodes,
    bool _is_tracking_mappings,
    const Shader_Dictionary& shader_dictionary)
{
    std::vector<Effective_Node_Provenance> provenance;
    std::vector<Compute_Node> effective_nodes = fuseLinearBackwardAdamPass(_original_nodes, &provenance);
    fuseHorizontalAdamPass(effective_nodes, provenance, shader_dictionary);
    Cached_Graph_Template graph_template;
    if (effective_nodes.empty())
    {
        return graph_template;
    }

    auto translateBufferMappings = [&](size_t eff_idx,
                                       const std::vector<Buffer_Binding_Mapping> &local_mappings,
                                       std::vector<Buffer_Binding_Mapping> &out_mappings)
    {
        const Effective_Node_Provenance prov = (eff_idx < provenance.size())
            ? provenance[eff_idx]
            : Effective_Node_Provenance{.primary_raw_index = static_cast<uint32_t>(eff_idx), .fused_adam_raw_index = -1, .horizontal_raw_indices = {}};

        if (!prov.horizontal_raw_indices.empty())
        {
            if (!prov.horizontal_raw_indices.empty())
            {
                out_mappings.push_back(Buffer_Binding_Mapping{
                    .raw_node_index = prov.horizontal_raw_indices[0],
                    .raw_buffer_index = 4,
                    .fused_buffer_index = 0});
                for (size_t k = 0; k < prov.horizontal_raw_indices.size(); ++k)
                {
                    const uint32_t rk = prov.horizontal_raw_indices[k];
                    const uint32_t fbase = static_cast<uint32_t>(1 + k * 5);
                    out_mappings.push_back(Buffer_Binding_Mapping{.raw_node_index = rk, .raw_buffer_index = 0, .fused_buffer_index = fbase + 0});
                    out_mappings.push_back(Buffer_Binding_Mapping{.raw_node_index = rk, .raw_buffer_index = 1, .fused_buffer_index = fbase + 1});
                    out_mappings.push_back(Buffer_Binding_Mapping{.raw_node_index = rk, .raw_buffer_index = 2, .fused_buffer_index = fbase + 2});
                    out_mappings.push_back(Buffer_Binding_Mapping{.raw_node_index = rk, .raw_buffer_index = 3, .fused_buffer_index = fbase + 3});
                    out_mappings.push_back(Buffer_Binding_Mapping{.raw_node_index = rk, .raw_buffer_index = 5, .fused_buffer_index = fbase + 4});
                }
            }
            return;
        }

        for (const auto &m : local_mappings)
        {
            if (prov.fused_adam_raw_index < 0)
            {
                out_mappings.push_back(Buffer_Binding_Mapping{
                    .raw_node_index = prov.primary_raw_index,
                    .raw_buffer_index = m.raw_buffer_index,
                    .fused_buffer_index = m.fused_buffer_index});
            }
            else
            {
                uint32_t bwd_raw = prov.primary_raw_index;
                uint32_t adam_raw = static_cast<uint32_t>(prov.fused_adam_raw_index);
                uint32_t src_node = bwd_raw;
                uint32_t src_buf = 0;
                switch (m.raw_buffer_index)
                {
                case 0: src_node = bwd_raw; src_buf = 0; break;
                case 1: src_node = bwd_raw; src_buf = 1; break;
                case 2: src_node = adam_raw; src_buf = 0; break;
                case 3: src_node = adam_raw; src_buf = 2; break;
                case 4: src_node = adam_raw; src_buf = 3; break;
                case 5: src_node = adam_raw; src_buf = 4; break;
                case 6: src_node = adam_raw; src_buf = 5; break;
                case 7: src_node = bwd_raw; src_buf = 3; break;
                default: src_node = bwd_raw; src_buf = m.raw_buffer_index; break;
                }
                out_mappings.push_back(Buffer_Binding_Mapping{
                    .raw_node_index = src_node,
                    .raw_buffer_index = src_buf,
                    .fused_buffer_index = m.fused_buffer_index});
            }
        }
    };

    auto appendPcAndRawIndices = [&](size_t eff_idx,
                                     uint32_t fused_pc_offset,
                                     uint32_t pc_size,
                                     std::vector<Push_Constant_Mapping> &out_pc_mappings,
                                     std::vector<uint32_t> &out_raw_indices)
    {
        const Effective_Node_Provenance prov = (eff_idx < provenance.size())
            ? provenance[eff_idx]
            : Effective_Node_Provenance{.primary_raw_index = static_cast<uint32_t>(eff_idx), .fused_adam_raw_index = -1, .horizontal_raw_indices = {}};

        if (!prov.horizontal_raw_indices.empty())
        {
            out_pc_mappings.push_back(Push_Constant_Mapping{
                .raw_node_index = prov.horizontal_raw_indices[0],
                .fused_push_constants_offset = fused_pc_offset,
                .push_constants_size = 16,
                .raw_push_constants_offset = 4});
            for (size_t k = 0; k < prov.horizontal_raw_indices.size(); ++k)
            {
                const uint32_t rk = prov.horizontal_raw_indices[k];
                const uint32_t lane_off = fused_pc_offset + 20u + static_cast<uint32_t>(k * 12u);
                out_pc_mappings.push_back(Push_Constant_Mapping{
                    .raw_node_index = rk,
                    .fused_push_constants_offset = lane_off,
                    .push_constants_size = 4,
                    .raw_push_constants_offset = 0});
                out_pc_mappings.push_back(Push_Constant_Mapping{
                    .raw_node_index = rk,
                    .fused_push_constants_offset = lane_off + 8u,
                    .push_constants_size = 4,
                    .raw_push_constants_offset = 20});
                out_raw_indices.push_back(rk);
            }
            return;
        }

        if (prov.fused_adam_raw_index < 0)
        {
            out_pc_mappings.push_back(Push_Constant_Mapping{
                .raw_node_index = prov.primary_raw_index,
                .fused_push_constants_offset = fused_pc_offset,
                .push_constants_size = pc_size,
                .raw_push_constants_offset = 0});
            out_raw_indices.push_back(prov.primary_raw_index);
        }
        else
        {
            uint32_t bwd_raw = prov.primary_raw_index;
            uint32_t adam_raw = static_cast<uint32_t>(prov.fused_adam_raw_index);
            out_pc_mappings.push_back(Push_Constant_Mapping{
                .raw_node_index = bwd_raw,
                .fused_push_constants_offset = fused_pc_offset,
                .push_constants_size = 12,
                .raw_push_constants_offset = 0});
            out_pc_mappings.push_back(Push_Constant_Mapping{
                .raw_node_index = adam_raw,
                .fused_push_constants_offset = fused_pc_offset + 12,
                .push_constants_size = 20,
                .raw_push_constants_offset = 4});
            out_raw_indices.push_back(bwd_raw);
        }
    };

    auto restoreUnfusedNodeBindings = [&](Compute_Node& node,
        uint32_t eff_idx,
        std::vector<Buffer_Binding_Mapping>& mappings)
        {
            if (!node.is_fused && eff_idx < effective_nodes.size())
            {
                const auto& orig_buffers = effective_nodes[eff_idx].buffers;
                node.buffers = orig_buffers;
                if (_is_tracking_mappings)
                {
                    mappings.clear();
                    std::vector<Buffer_Binding_Mapping> local_mappings;
                    local_mappings.reserve(orig_buffers.size());
                    for (uint32_t b_idx = 0; b_idx < static_cast<uint32_t>(orig_buffers.size()); ++b_idx)
                    {
                        local_mappings.push_back(Buffer_Binding_Mapping{
                            .raw_node_index = eff_idx,
                            .raw_buffer_index = b_idx,
                            .fused_buffer_index = b_idx });
                    }
                    translateBufferMappings(eff_idx, local_mappings, mappings);
                }
            }
        };

    Compute_Node current_fused_node{};
    current_fused_node.pipeline_id = effective_nodes[0].pipeline_id;
    current_fused_node.push_constants_data = effective_nodes[0].push_constants_data;
    current_fused_node.workgroup_count_x = effective_nodes[0].workgroup_count_x;
    current_fused_node.workgroup_count_y = effective_nodes[0].workgroup_count_y;
    current_fused_node.workgroup_count_z = effective_nodes[0].workgroup_count_z;
    current_fused_node.spec_constant_u32 = effective_nodes[0].spec_constant_u32;
    current_fused_node.is_fused = effective_nodes[0].is_fused;
    current_fused_node.buffers = effective_nodes[0].buffers;
    current_fused_node.external_output_indices = effective_nodes[0].external_output_indices;
    current_fused_node.fusion_plan = effective_nodes[0].fusion_plan;

    uint32_t current_start_raw_index = 0;
    std::vector<Buffer_Binding_Mapping> current_buffer_mappings;
    std::vector<Push_Constant_Mapping> current_push_constants_mappings;
    std::vector<uint32_t> current_raw_node_indices;

    const Snippet_Metadata& first_metadata = shader_dictionary.getMetadata(effective_nodes[0].pipeline_id);
    if (effective_nodes[0].fusion_plan && effective_nodes[0].fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
    {
        current_fused_node.fused_operations = effective_nodes[0].fused_operations;
        if (_is_tracking_mappings)
        {
            translateBufferMappings(0, {}, current_buffer_mappings);
            appendPcAndRawIndices(0, 0, 0, current_push_constants_mappings, current_raw_node_indices);
        }
    }
    else
    {
        std::vector<Buffer_Binding_Mapping> first_local_mappings;
        current_fused_node.fused_operations.push_back(
            buildFusedOperation(current_fused_node, effective_nodes[0], 0, first_metadata, 0, _is_tracking_mappings ? &first_local_mappings : nullptr));

        if (_is_tracking_mappings)
        {
            translateBufferMappings(0, first_local_mappings, current_buffer_mappings);
            appendPcAndRawIndices(0, 0, static_cast<uint32_t>(effective_nodes[0].push_constants_data.size()), current_push_constants_mappings, current_raw_node_indices);
        }
    }

    for (size_t i = 1; i < effective_nodes.size(); ++i)
    {
        const Compute_Node& next_node = effective_nodes[i];
        const auto& next_meta = shader_dictionary.getMetadata(next_node.pipeline_id);

        std::shared_ptr<gpu::vector> consumed_producer_output;
        bool is_consuming_non_persistent_output = false;
        size_t next_in_limit = std::min<size_t>(next_meta.input_count, next_node.buffers.size());
        for (const auto& op : current_fused_node.fused_operations)
        {
            const auto& op_meta = shader_dictionary.getMetadata(op.pipeline_id);
            for (size_t local_out_idx = 0; local_out_idx < op.output_buffer_indices.size(); ++local_out_idx)
            {
                bool is_persistent = std::find(op_meta.persistent_output_indices.begin(),
                                               op_meta.persistent_output_indices.end(),
                                               static_cast<uint32_t>(local_out_idx)) != op_meta.persistent_output_indices.end();
                bool is_accum = std::find(op_meta.accumulator_output_indices.begin(),
                                          op_meta.accumulator_output_indices.end(),
                                          static_cast<uint32_t>(local_out_idx)) != op_meta.accumulator_output_indices.end();
                if (is_persistent || is_accum)
                {
                    continue;
                }
                uint32_t out_buf_idx = op.output_buffer_indices[local_out_idx];
                if (out_buf_idx >= current_fused_node.buffers.size() || !current_fused_node.buffers[out_buf_idx])
                {
                    continue;
                }
                const auto& prod_buf = current_fused_node.buffers[out_buf_idx];
                for (size_t in_idx = 0; in_idx < next_in_limit; ++in_idx)
                {
                    const auto& in_buf = next_node.buffers[in_idx];
                    if (in_buf &&
                        (prod_buf == in_buf ||
                         (prod_buf->getBuffer() != VK_NULL_HANDLE &&
                          prod_buf->getBuffer() == in_buf->getBuffer())))
                    {
                        is_consuming_non_persistent_output = true;
                        consumed_producer_output = prod_buf;
                        break;
                    }
                }
                if (is_consuming_non_persistent_output)
                {
                    break;
                }
            }
            if (is_consuming_non_persistent_output)
            {
                break;
            }
        }

        std::vector<std::shared_ptr<gpu::vector>> unique_buffers;
        for (const auto& new_buffer : next_node.buffers)
        {
            bool is_existing = false;
            for (const auto& existing_buffer : current_fused_node.buffers)
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

        size_t current_push_constants_bytes = current_fused_node.push_constants_data.size();
        size_t aligned_push_constants_offset = alignTo4Bytes(current_push_constants_bytes);
        size_t total_push_constants_size = aligned_push_constants_offset + next_node.push_constants_data.size();

        bool does_push_constants_fit = (total_push_constants_size <= MAX_PUSH_CONSTANTS_BYTES);
        size_t total_bindings = current_fused_node.buffers.size() + unique_buffers.size();
        bool do_bindings_fit = (total_bindings <= MAX_STORAGE_BUFFER_BINDINGS);
        bool does_chain_fit = (current_fused_node.fused_operations.size() < MAX_FUSED_OPERATIONS);

        Compute_Pipeline producer_pipeline = current_fused_node.fused_operations[0].pipeline_id;
        for (const auto &op : current_fused_node.fused_operations)
        {
            if (isRowReductionAnchorPipeline(op.pipeline_id))
            {
                producer_pipeline = op.pipeline_id;
                break;
            }
        }
        Compute_Pipeline consumer_pipeline = next_node.pipeline_id;

        Operation_Class producer_class = shader_dictionary.getMetadata(producer_pipeline).operation_class;
        Operation_Class consumer_class = shader_dictionary.getMetadata(consumer_pipeline).operation_class;

        bool is_fusible_operation = !next_node.is_fused && !current_fused_node.fused_operations.empty() &&
            !(current_fused_node.fusion_plan && current_fused_node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE) &&
            isFusible(producer_class, consumer_class, producer_pipeline, consumer_pipeline, shader_dictionary);
        bool are_dimensions_matching = hasCompatibleDimensions(current_fused_node, next_node, producer_class, consumer_class, shader_dictionary);
        bool has_aliasing_hazard = hasAliasingHazard(current_fused_node, next_node, shader_dictionary);

        const bool prod_elem_or_pw = (producer_class == Operation_Class::ELEMENTWISE ||
                                      classifyPointwiseOp(producer_pipeline) != Pointwise_Op_Kind::NONE);
        const bool cons_elem_or_pw = (consumer_class == Operation_Class::ELEMENTWISE ||
                                      classifyPointwiseOp(consumer_pipeline) != Pointwise_Op_Kind::NONE);
        if (are_dimensions_matching && prod_elem_or_pw && cons_elem_or_pw)
        {
            if (current_fused_node.push_constants_data.size() >= 4 && next_node.push_constants_data.size() >= 4)
            {
                uint32_t prod_total = readPushConstantWord<uint32_t>(current_fused_node.push_constants_data, 0);
                uint32_t cons_total = readPushConstantWord<uint32_t>(next_node.push_constants_data, 0);
                if (prod_total != cons_total)
                {
                    are_dimensions_matching = false;
                }
            }
        }
        else if (are_dimensions_matching &&
                 (producer_class == Operation_Class::MATRIX_2D || producer_class == Operation_Class::TENSOR_3D) &&
                 consumer_class == Operation_Class::ELEMENTWISE)
        {
            if (!consumed_producer_output || next_node.push_constants_data.size() < 4)
            {
                are_dimensions_matching = false;
            }
            else
            {
                uint32_t cons_total = readPushConstantWord<uint32_t>(next_node.push_constants_data, 0);
                if (consumed_producer_output->getSize() != static_cast<size_t>(cons_total))
                {
                    are_dimensions_matching = false;
                }
            }
        }

        if (is_fusible_operation && are_dimensions_matching && !has_aliasing_hazard)
        {
            auto isProducedByCurrentFusedNode = [&](const std::shared_ptr<gpu::vector> &target_buf) -> bool
            {
                if (!target_buf)
                {
                    return false;
                }
                for (const auto &op : current_fused_node.fused_operations)
                {
                    for (uint32_t out_idx : op.output_buffer_indices)
                    {
                        if (out_idx < current_fused_node.buffers.size())
                        {
                            const auto &prod_buf = current_fused_node.buffers[out_idx];
                            if (prod_buf &&
                                (prod_buf == target_buf ||
                                 (prod_buf->getBuffer() != VK_NULL_HANDLE &&
                                  prod_buf->getBuffer() == target_buf->getBuffer())))
                            {
                                return true;
                            }
                        }
                    }
                }
                return false;
            };

            if (consumer_pipeline == Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD ||
                consumer_pipeline == Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD)
            {
                for (size_t in_slot = 1; in_slot < 5 && in_slot < next_node.buffers.size(); ++in_slot)
                {
                    if (isProducedByCurrentFusedNode(next_node.buffers[in_slot]))
                    {
                        is_fusible_operation = false;
                        break;
                    }
                }
            }
            else if (consumer_pipeline == Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD ||
                     consumer_pipeline == Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD)
            {
                for (size_t in_slot = 2; in_slot < 6 && in_slot < next_node.buffers.size(); ++in_slot)
                {
                    if (isProducedByCurrentFusedNode(next_node.buffers[in_slot]))
                    {
                        is_fusible_operation = false;
                        break;
                    }
                }
            }
            else if (consumer_pipeline == Compute_Pipeline::ADD ||
                     consumer_pipeline == Compute_Pipeline::SUB ||
                     consumer_pipeline == Compute_Pipeline::HADAMARD_MUL ||
                     consumer_pipeline == Compute_Pipeline::HADAMARD_DIV ||
                     consumer_pipeline == Compute_Pipeline::ADD_FP16 ||
                     consumer_pipeline == Compute_Pipeline::SUB_FP16 ||
                     consumer_pipeline == Compute_Pipeline::HADAMARD_MUL_FP16 ||
                     consumer_pipeline == Compute_Pipeline::HADAMARD_DIV_FP16)
            {
                if (next_node.push_constants_data.size() >= 12 && next_node.buffers.size() >= 2)
                {
                    uint32_t is_broadcast = readPushConstantWord<uint32_t>(next_node.push_constants_data, 2);
                    if (is_broadcast != 0 && isProducedByCurrentFusedNode(next_node.buffers[1]))
                    {
                        is_fusible_operation = false;
                    }
                }
            }
        }

        if (is_consuming_non_persistent_output && does_push_constants_fit && do_bindings_fit && does_chain_fit &&
            is_fusible_operation && are_dimensions_matching && !has_aliasing_hazard)
        {
            size_t padding_bytes = aligned_push_constants_offset - current_push_constants_bytes;
            if (padding_bytes > 0)
            {
                current_fused_node.push_constants_data.insert(
                    current_fused_node.push_constants_data.end(),
                    padding_bytes,
                    0);
            }

            uint32_t current_push_constants_offset_val = static_cast<uint32_t>(aligned_push_constants_offset);
            const Snippet_Metadata& next_metadata = shader_dictionary.getMetadata(next_node.pipeline_id);

            std::vector<Buffer_Binding_Mapping> next_local_mappings;
            current_fused_node.fused_operations.push_back(
                buildFusedOperation(current_fused_node, next_node, current_push_constants_offset_val, next_metadata, static_cast<uint32_t>(i), _is_tracking_mappings ? &next_local_mappings : nullptr));

            if (_is_tracking_mappings)
            {
                translateBufferMappings(i, next_local_mappings, current_buffer_mappings);
                appendPcAndRawIndices(i, current_push_constants_offset_val, static_cast<uint32_t>(next_node.push_constants_data.size()), current_push_constants_mappings, current_raw_node_indices);
            }

            current_fused_node.push_constants_data.insert(
                current_fused_node.push_constants_data.end(),
                next_node.push_constants_data.begin(),
                next_node.push_constants_data.end());

            updateDispatchGrid(current_fused_node, next_node, producer_class, consumer_class, shader_dictionary);
            current_fused_node.is_fused = true;
        }
        else
        {
            restoreUnfusedNodeBindings(current_fused_node, current_start_raw_index, current_buffer_mappings);
            if (current_fused_node.is_fused && !(current_fused_node.fusion_plan && current_fused_node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE))
            {
                markExternalOutputs(current_fused_node, effective_nodes, i, _original_nodes, graph_template.fused_nodes);
            }
            finalizeFusedNodePlan(current_fused_node, shader_dictionary);

            graph_template.fused_nodes.push_back(current_fused_node);
            if (_is_tracking_mappings)
            {
                graph_template.buffer_mappings.push_back(std::move(current_buffer_mappings));
                graph_template.push_constants_mappings.push_back(std::move(current_push_constants_mappings));
                graph_template.raw_node_indices.push_back(std::move(current_raw_node_indices));
                current_buffer_mappings.clear();
                current_push_constants_mappings.clear();
                current_raw_node_indices.clear();
            }

            current_start_raw_index = static_cast<uint32_t>(i);
            current_fused_node = Compute_Node{};
            current_fused_node.pipeline_id = next_node.pipeline_id;
            current_fused_node.push_constants_data = next_node.push_constants_data;
            current_fused_node.workgroup_count_x = next_node.workgroup_count_x;
            current_fused_node.workgroup_count_y = next_node.workgroup_count_y;
            current_fused_node.workgroup_count_z = next_node.workgroup_count_z;
            current_fused_node.spec_constant_u32 = next_node.spec_constant_u32;
            current_fused_node.is_fused = next_node.is_fused;
            current_fused_node.buffers = next_node.buffers;
            current_fused_node.external_output_indices = next_node.external_output_indices;
            current_fused_node.fusion_plan = next_node.fusion_plan;

            const Snippet_Metadata& metadata = shader_dictionary.getMetadata(next_node.pipeline_id);
            if (next_node.fusion_plan && next_node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
            {
                current_fused_node.fused_operations = next_node.fused_operations;
                if (_is_tracking_mappings)
                {
                    translateBufferMappings(i, {}, current_buffer_mappings);
                    appendPcAndRawIndices(i, 0, 0, current_push_constants_mappings, current_raw_node_indices);
                }
            }
            else
            {
                std::vector<Buffer_Binding_Mapping> local_mappings;
                current_fused_node.fused_operations.push_back(
                    buildFusedOperation(current_fused_node, next_node, 0, metadata, static_cast<uint32_t>(i), _is_tracking_mappings ? &local_mappings : nullptr));

                if (_is_tracking_mappings)
                {
                    translateBufferMappings(i, local_mappings, current_buffer_mappings);
                    appendPcAndRawIndices(i, 0, static_cast<uint32_t>(next_node.push_constants_data.size()), current_push_constants_mappings, current_raw_node_indices);
                }
            }
        }
    }

    restoreUnfusedNodeBindings(current_fused_node, current_start_raw_index, current_buffer_mappings);
    if (current_fused_node.is_fused && !(current_fused_node.fusion_plan && current_fused_node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE))
    {
        markExternalOutputs(current_fused_node, effective_nodes, effective_nodes.size(), _original_nodes, graph_template.fused_nodes);
    }
    finalizeFusedNodePlan(current_fused_node, shader_dictionary);

    graph_template.fused_nodes.push_back(current_fused_node);
    if (_is_tracking_mappings)
    {
        graph_template.buffer_mappings.push_back(std::move(current_buffer_mappings));
        graph_template.push_constants_mappings.push_back(std::move(current_push_constants_mappings));
        graph_template.raw_node_indices.push_back(std::move(current_raw_node_indices));

        graph_template.total_buffer_mappings = 0;
        for (const auto& mappings : graph_template.buffer_mappings)
        {
            graph_template.total_buffer_mappings += mappings.size();
        }
    }

    scheduleNodesByDependencyLevel(graph_template, _is_tracking_mappings, shader_dictionary);
    assignPipelineBarriers(graph_template.fused_nodes, shader_dictionary);
    graph_template.is_valid = true;

    VML_LOG_DEBUG(Log_Feature::OPERATOR_FUSION,
                  "Graph_Optimizer::optimizeInternal: raw_nodes={}, effective_nodes={}, fused_nodes={}",
                  _original_nodes.size(), effective_nodes.size(), graph_template.fused_nodes.size());

    return graph_template;
}

void Graph_Optimizer::optimize(Compute_Graph &_graph, const Shader_Dictionary *_shader_dictionary)
{
    const Shader_Dictionary &dict = (_shader_dictionary != nullptr) ? *_shader_dictionary : Shader_Dictionary::getInstance();
#if !ENABLE_SHADER_FUSION
    auto nodes = fuseLinearBackwardAdamPass(_graph.getNodes());
    scheduleNodesByDependencyLevel(nodes, dict);
    assignPipelineBarriers(nodes, dict);
    _graph.clear();
    for (const auto &node : nodes)
    {
        _graph.addNode(node);
    }
    return;
#else
    const std::vector<Compute_Node> &original_nodes = _graph.getNodes();
    if (original_nodes.empty())
    {
        return;
    }

    Cached_Graph_Template graph_template = optimizeInternal(original_nodes, false, dict);
    _graph.clear();
    for (const auto &node : graph_template.fused_nodes)
    {
        _graph.addNode(node);
    }
#endif
}

Cached_Graph_Template Graph_Optimizer::buildCachedTemplate(const Compute_Graph &_graph, const Shader_Dictionary *_shader_dictionary)
{
    const Shader_Dictionary &dict = (_shader_dictionary != nullptr) ? *_shader_dictionary : Shader_Dictionary::getInstance();
#if !ENABLE_SHADER_FUSION
    (void)dict;
    return Cached_Graph_Template{};
#else
    const auto &original_nodes = _graph.getNodes();
    if (original_nodes.empty())
    {
        return Cached_Graph_Template{};
    }

    Cached_Graph_Template graph_template = optimizeInternal(original_nodes, true, dict);
    for (auto &node : graph_template.fused_nodes)
    {
        for (auto &buf : node.buffers)
        {
            if (buf)
            {
                Data_Type dt = buf->getDataType();
                auto dummy = std::make_shared<gpu::vector>(buf->getContext());
                dummy->setDataType(dt);
                buf = dummy;
            }
        }
    }
    return graph_template;
#endif
}

void Graph_Optimizer::applyCachedTemplateInPlace(
    const Compute_Graph &_raw_graph,
    const Cached_Graph_Template &_graph_template,
    Compute_Graph &_cached_graph,
    const Shader_Dictionary *_shader_dictionary)
{
    const Shader_Dictionary &shader_dictionary = (_shader_dictionary != nullptr) ? *_shader_dictionary : Shader_Dictionary::getInstance();
#if !ENABLE_SHADER_FUSION
    auto nodes = fuseLinearBackwardAdamPass(_raw_graph.getNodes());
    scheduleNodesByDependencyLevel(nodes, shader_dictionary);
    assignPipelineBarriers(nodes, shader_dictionary);
    _cached_graph.setNodes(std::move(nodes));
    return;
#else
    if (!_graph_template.is_valid)
    {
        auto nodes = fuseLinearBackwardAdamPass(_raw_graph.getNodes());
        scheduleNodesByDependencyLevel(nodes, shader_dictionary);
        assignPipelineBarriers(nodes, shader_dictionary);
        _cached_graph.setNodes(std::move(nodes));
        return;
    }

    bool needs_init = (_cached_graph.getNodeCount() != _graph_template.fused_nodes.size());
    if (!needs_init)
    {
        for (size_t i = 0; i < _graph_template.fused_nodes.size(); ++i)
        {
            if (_cached_graph.getNodes()[i].pipeline_id != _graph_template.fused_nodes[i].pipeline_id ||
                _cached_graph.getNodes()[i].spec_constant_u32 != _graph_template.fused_nodes[i].spec_constant_u32 ||
                _cached_graph.getNodes()[i].is_fused != _graph_template.fused_nodes[i].is_fused ||
                _cached_graph.getNodes()[i].buffers.size() != _graph_template.fused_nodes[i].buffers.size())
            {
                needs_init = true;
                break;
            }
        }
    }

    if (needs_init)
    {
        _cached_graph.setNodes(_graph_template.fused_nodes);
    }

    const auto &raw_nodes = _raw_graph.getNodes();
    auto &cached_nodes = _cached_graph.getNodes();
    const size_t node_count = std::min(_graph_template.fused_nodes.size(), cached_nodes.size());

    for (size_t node_index = 0; node_index < node_count; ++node_index)
    {
        Compute_Node &node = cached_nodes[node_index];

        if (node_index < _graph_template.buffer_mappings.size())
        {
            for (const auto &buffer_binding_mapping : _graph_template.buffer_mappings[node_index])
            {
                if (buffer_binding_mapping.raw_node_index < raw_nodes.size() &&
                    buffer_binding_mapping.raw_buffer_index < raw_nodes[buffer_binding_mapping.raw_node_index].buffers.size() &&
                    buffer_binding_mapping.fused_buffer_index < node.buffers.size())
                {
                    auto old_buffer = node.buffers[buffer_binding_mapping.fused_buffer_index];
                    auto new_buffer = raw_nodes[buffer_binding_mapping.raw_node_index].buffers[buffer_binding_mapping.raw_buffer_index];

                    if (_graph_template.is_mapping_log_enabled.load(std::memory_order_acquire))
                    {
                        bool keep_logging = Logger::logMessage(
                            Input_Format{
                                "Graph_Optimizer::applyCachedTemplate: Fused Node {} | Binding buffer_{} <- Raw Node {}[Buffer {}] (Old ID: {}, New ID: {})",
                                node_index,
                                buffer_binding_mapping.fused_buffer_index,
                                buffer_binding_mapping.raw_node_index,
                                buffer_binding_mapping.raw_buffer_index,
                                old_buffer ? old_buffer->getId() : 0,
                                new_buffer ? new_buffer->getId() : 0},
                            Log_Level::LOG_DEBUG,
                            true,
                            _graph_template.total_buffer_mappings,
                            Log_Feature::OPERATOR_FUSION);
                        if (!keep_logging)
                        {
                            _graph_template.is_mapping_log_enabled.store(false, std::memory_order_release);
                        }
                    }

                    node.buffers[buffer_binding_mapping.fused_buffer_index] = new_buffer;
                }
            }
        }

        if (node_index < _graph_template.push_constants_mappings.size())
        {
            for (const auto &push_constant_mapping : _graph_template.push_constants_mappings[node_index])
            {
                if (push_constant_mapping.raw_node_index < raw_nodes.size())
                {
                    const auto &source_push_constants = raw_nodes[push_constant_mapping.raw_node_index].push_constants_data;
                    size_t src_off = push_constant_mapping.raw_push_constants_offset;
                    if (src_off < source_push_constants.size())
                    {
                        size_t copy_bytes = std::min<size_t>(source_push_constants.size() - src_off, push_constant_mapping.push_constants_size);
                        if (push_constant_mapping.fused_push_constants_offset + copy_bytes <= node.push_constants_data.size())
                        {
                            std::copy_n(source_push_constants.begin() + src_off, copy_bytes, node.push_constants_data.begin() + push_constant_mapping.fused_push_constants_offset);
                        }
                    }
                }
            }
        }

        if (node_index < _graph_template.raw_node_indices.size())
        {
            const auto &mapped_raw_indices = _graph_template.raw_node_indices[node_index];
            if (!mapped_raw_indices.empty())
            {
                uint32_t first_raw_index = mapped_raw_indices[0];
                if (node.fusion_plan && node.fusion_plan->anchor_role == Fused_Anchor_Role::HORIZONTAL_MULTI_LANE)
                {
                    const uint32_t lanes = static_cast<uint32_t>(node.fusion_plan->horizontal_lanes.size());
                    uint32_t wg_cursor = 0;
                    for (uint32_t l = 0; l < lanes; ++l)
                    {
                        const size_t pc_off = 20 + 12 * l;
                        uint32_t elem_count = 0;
                        if (pc_off + 8 <= node.push_constants_data.size())
                        {
                            std::memcpy(&elem_count, node.push_constants_data.data() + pc_off, 4);
                            wg_cursor += std::max<uint32_t>(1u, (elem_count + 1023u) / 1024u);
                            std::memcpy(node.push_constants_data.data() + pc_off + 4, &wg_cursor, 4);
                        }
                    }
                    uint32_t eff_wgs = std::max<uint32_t>(1u, wg_cursor);
                    if (eff_wgs <= 65535u)
                    {
                        node.workgroup_count_x = eff_wgs;
                        node.workgroup_count_y = 1;
                    }
                    else
                    {
                        node.workgroup_count_x = 65535u;
                        node.workgroup_count_y = (eff_wgs + 65534u) / 65535u;
                    }
                    node.workgroup_count_z = 1;
                    continue;
                }
                if (node.fusion_plan && node.fusion_plan->anchor_role == Fused_Anchor_Role::POINTWISE_CHAIN &&
                    node.fusion_plan->use_vec4 && node.push_constants_data.size() >= 4)
                {
                    uint32_t total_elems = readPushConstantWord<uint32_t>(node.push_constants_data, 0);
                    uint32_t vec_wgs = std::max<uint32_t>(1u, (total_elems + 1023u) / 1024u);
                    if (vec_wgs <= 65535u)
                    {
                        node.workgroup_count_x = vec_wgs;
                        node.workgroup_count_y = 1;
                    }
                    else
                    {
                        node.workgroup_count_x = 65535u;
                        node.workgroup_count_y = (vec_wgs + 65534u) / 65535u;
                    }
                    node.workgroup_count_z = 1;
                    continue;
                }
                if (node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16 ||
                    node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16)
                {
                    if (node.push_constants_data.size() >= 12)
                    {
                        uint32_t in_dim = readPushConstantWord<uint32_t>(node.push_constants_data, 1);
                        uint32_t out_dim = readPushConstantWord<uint32_t>(node.push_constants_data, 2);
                        uint32_t tile = (node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16) ? 64u : 16u;
                        node.workgroup_count_x = (out_dim + tile - 1u) / tile;
                        node.workgroup_count_y = (in_dim + tile - 1u) / tile;
                        node.workgroup_count_z = 1;
                    }
                }
                else if (first_raw_index < raw_nodes.size())
                {
                    node.workgroup_count_x = raw_nodes[first_raw_index].workgroup_count_x;
                    node.workgroup_count_y = raw_nodes[first_raw_index].workgroup_count_y;
                    node.workgroup_count_z = raw_nodes[first_raw_index].workgroup_count_z;
                    node.spec_constant_u32 = raw_nodes[first_raw_index].spec_constant_u32;
                }

                Operation_Class producer_class = shader_dictionary.getMetadata(node.pipeline_id).operation_class;
                if (!node.fused_operations.empty())
                {
                    node.fused_operations[0].workgroup_count_x = node.workgroup_count_x;
                    node.fused_operations[0].workgroup_count_y = node.workgroup_count_y;
                    node.fused_operations[0].workgroup_count_z = node.workgroup_count_z;
                    producer_class = shader_dictionary.getMetadata(node.fused_operations[0].pipeline_id).operation_class;
                }

                for (size_t k = 1; k < mapped_raw_indices.size(); ++k)
                {
                    uint32_t next_raw_index = mapped_raw_indices[k];
                    if (next_raw_index >= raw_nodes.size())
                    {
                        continue;
                    }
                    Operation_Class consumer_class = shader_dictionary.getMetadata(raw_nodes[next_raw_index].pipeline_id).operation_class;

                    if (k < node.fused_operations.size())
                    {
                        node.fused_operations[k].workgroup_count_x = raw_nodes[next_raw_index].workgroup_count_x;
                        node.fused_operations[k].workgroup_count_y = raw_nodes[next_raw_index].workgroup_count_y;
                        node.fused_operations[k].workgroup_count_z = raw_nodes[next_raw_index].workgroup_count_z;
                    }

                    updateDispatchGrid(node, raw_nodes[next_raw_index], producer_class, consumer_class, shader_dictionary);
                }
            }
        }
    }
#endif
}

void Graph_Optimizer::applyCachedTemplate(
    const Compute_Graph &_raw_graph,
    const Cached_Graph_Template &_graph_template,
    Compute_Graph &_output_graph,
    const Shader_Dictionary *_shader_dictionary)
{
    applyCachedTemplateInPlace(_raw_graph, _graph_template, _output_graph, _shader_dictionary);
}

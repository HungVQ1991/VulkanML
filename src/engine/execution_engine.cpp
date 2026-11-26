#include "engine/execution_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "engine/graph_executor.h"
#include "engine/graph_optimizer.h"
#include "engine/pipeline_cache_manager.h"
#include "engine/shader_dictionary.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_network.h"
#include "helper/logger.h"
#include "helper/resource_path.h"
#include "helper/training_profiler.h"

namespace
{
void scrubCachedGraphBuffers(
    Compute_Graph &cached_graph,
    std::unordered_map<Data_Type, std::shared_ptr<gpu::vector>> &dummy_pool)
{
    for (auto &node : cached_graph.getNodes())
    {
        for (auto &buf : node.buffers)
        {
            if (buf)
            {
                Data_Type dt = buf->getDataType();
                auto &dummy = dummy_pool[dt];
                if (!dummy)
                {
                    dummy = std::make_shared<gpu::vector>(buf->getContext());
                    dummy->setDataType(dt);
                }
                buf = dummy;
            }
        }
    }
}

struct CachedGraphScrubGuard
{
    Compute_Graph &graph;
    std::unordered_map<Data_Type, std::shared_ptr<gpu::vector>> &dummy_pool;
    ~CachedGraphScrubGuard()
    {
        try
        {
            scrubCachedGraphBuffers(graph, dummy_pool);
        }
        catch (...)
        {
        }
    }
};

struct GraphExecutionCleanupGuard
{
    Compute_Graph &graph;
    Vulkan_Context *context;
    bool active = true;
    ~GraphExecutionCleanupGuard()
    {
        if (active)
        {
            graph.clear();
            if (context)
            {
                context->clearTransferTasks();
            }
        }
    }
};
}

Execution_Engine& Execution_Engine::getInstance()
{
    static Execution_Engine* instance = new Execution_Engine();
    return *instance;
}

void Execution_Engine::processPendingLossReadback(uint32_t frame_index)
{
    if (frame_index < loss_slots.size() &&
        loss_slots[frame_index].has_pending_read &&
        loss_slots[frame_index].is_submitted)
    {
        auto read_start = std::chrono::high_resolution_clock::now();
        auto& slot = loss_slots[frame_index];
        if (slot.buffer && slot.buffer->isHostMapped())
        {
            const float* ptr = static_cast<const float*>(slot.buffer->getHostMappedPointer());
            if (ptr)
            {
                float sum = 0.0f;
                bool has_non_finite = false;
                for (size_t i = 0; i < slot.total_tokens; ++i)
                {
                    float val = ptr[i];
                    if (std::isnan(val) || std::isinf(val))
                    {
                        has_non_finite = true;
                        break;
                    }
                    sum += val;
                }
                if (has_non_finite)
                {
                    latest_loss = std::numeric_limits<float>::quiet_NaN();
                }
                else
                {
                    float denom = (slot.valid_tokens > 0)
                        ? static_cast<float>(slot.valid_tokens)
                        : ptr[slot.total_tokens];
                    if (denom > 0.0f && !std::isnan(denom) && !std::isinf(denom))
                    {
                        latest_loss = sum / denom;
                    }
                    else
                    {
                        latest_loss = 0.0f;
                    }
                }
            }
        }
        slot.has_pending_read = false;
        slot.is_submitted = false;
        auto read_end = std::chrono::high_resolution_clock::now();
        double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
        Step_Timings::getInstance().recordDuration(Timing_Stage::LOSS_READBACK, read_ms);
    }
}

Graph_Topology_Key Execution_Engine::computeGraphTopologyKey(const Compute_Graph& graph) const
{
    const auto& nodes = graph.getNodes();
    Graph_Topology_Key key;
    key.is_coop_enabled = isCooperativeMatrixEnabled();
    key.is_fused_gemm_adam = is_fused_gemm_adam_enabled;
    key.nodes.reserve(nodes.size());

    std::unordered_map<const gpu::vector*, size_t> raw_ptr_counts;
    raw_ptr_counts.reserve(nodes.size() * 3);
    for (const auto& node : nodes)
    {
        for (const auto& buffer : node.buffers)
        {
            if (buffer)
            {
                ++raw_ptr_counts[buffer.get()];
            }
        }
    }

    std::unordered_map<VkBuffer, size_t> buffer_to_id;
    buffer_to_id.reserve(nodes.size() * 2);
    size_t next_id = 0;

    auto getCanonicalBufferId = [&](const std::shared_ptr<gpu::vector>& buf) -> size_t {
        if (!buf)
        {
            return static_cast<size_t>(-1);
        }
        VkBuffer handle = buf->getBuffer();
        if (handle == VK_NULL_HANDLE)
        {
            handle = reinterpret_cast<VkBuffer>(buf.get());
        }
        auto [it, inserted] = buffer_to_id.try_emplace(handle, next_id);
        if (inserted)
        {
            ++next_id;
        }
        return it->second;
    };

    const auto& dict = shader_dictionary ? *shader_dictionary : Shader_Dictionary::getInstance();

    for (const auto& node : nodes)
    {
        Graph_Topology_Node_Key nk;
        nk.pipeline_id = node.pipeline_id;
        nk.workgroup_count_x = node.workgroup_count_x;
        nk.workgroup_count_y = node.workgroup_count_y;
        nk.workgroup_count_z = node.workgroup_count_z;
        nk.spec_constant_u32 = node.spec_constant_u32;
        if (node.fusion_plan && !node.fusion_plan->canonical_key.empty())
        {
            nk.fusion_plan_hash = std::hash<std::string>{}(node.fusion_plan->canonical_key);
        }
        nk.push_constants_size = static_cast<uint32_t>(node.push_constants_data.size());

        const auto& meta = dict.getMetadata(node.pipeline_id);
        if (meta.operation_class == Operation_Class::ELEMENTWISE ||
            Graph_Optimizer::classifyPointwiseOp(node.pipeline_id) != Pointwise_Op_Kind::NONE ||
            node.pipeline_id == Compute_Pipeline::ADAM_UPDATE ||
            node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16 ||
            node.pipeline_id == Compute_Pipeline::SGD_UPDATE ||
            node.pipeline_id == Compute_Pipeline::SGD_UPDATE_FP16 ||
            node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE ||
            node.pipeline_id == Compute_Pipeline::SGD_MOMENTUM_UPDATE_FP16)
        {
            if (node.push_constants_data.size() >= 4)
            {
                nk.structural_pc_word0 = readPushConstantWord<uint32_t>(node.push_constants_data, 0);
            }
        }

        if (node.pipeline_id == Compute_Pipeline::ADD ||
            node.pipeline_id == Compute_Pipeline::SUB ||
            node.pipeline_id == Compute_Pipeline::HADAMARD_MUL ||
            node.pipeline_id == Compute_Pipeline::HADAMARD_DIV ||
            node.pipeline_id == Compute_Pipeline::ADD_FP16 ||
            node.pipeline_id == Compute_Pipeline::SUB_FP16 ||
            node.pipeline_id == Compute_Pipeline::HADAMARD_MUL_FP16 ||
            node.pipeline_id == Compute_Pipeline::HADAMARD_DIV_FP16)
        {
            if (node.push_constants_data.size() >= 12)
            {
                nk.structural_pc_word1 = readPushConstantWord<uint32_t>(node.push_constants_data, 1);
                nk.structural_pc_word2 = readPushConstantWord<uint32_t>(node.push_constants_data, 2);
            }
        }
        else if (node.pipeline_id == Compute_Pipeline::RMSNORM ||
                 node.pipeline_id == Compute_Pipeline::RMSNORM_FP16)
        {
            if (node.push_constants_data.size() >= 8)
            {
                nk.structural_pc_word0 = readPushConstantWord<uint32_t>(node.push_constants_data, 0);
                nk.structural_pc_word1 = readPushConstantWord<uint32_t>(node.push_constants_data, 1);
            }
        }
        else if (node.pipeline_id == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16 ||
                 node.pipeline_id == Compute_Pipeline::MATMUL_COOPMAT_FP16 ||
                 node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16)
        {
            if (node.push_constants_data.size() >= 12)
            {
                nk.structural_pc_word0 = readPushConstantWord<uint32_t>(node.push_constants_data, 0);
                nk.structural_pc_word1 = readPushConstantWord<uint32_t>(node.push_constants_data, 1);
                nk.structural_pc_word2 = readPushConstantWord<uint32_t>(node.push_constants_data, 2);
            }
            if (node.push_constants_data.size() >= 16)
            {
                nk.structural_pc_word3 = readPushConstantWord<uint32_t>(node.push_constants_data, 3);
            }
            if (node.push_constants_data.size() >= 20)
            {
                nk.structural_pc_word4 = readPushConstantWord<uint32_t>(node.push_constants_data, 4);
            }
            if (node.push_constants_data.size() >= 24)
            {
                nk.structural_pc_word5 = readPushConstantWord<uint32_t>(node.push_constants_data, 5);
            }
        }
        else if (node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_FP16)
        {
            if (node.push_constants_data.size() >= 20)
            {
                nk.structural_pc_word1 = readPushConstantWord<uint32_t>(node.push_constants_data, 1);
                nk.structural_pc_word2 = readPushConstantWord<uint32_t>(node.push_constants_data, 2);
                nk.structural_pc_word3 = readPushConstantWord<uint32_t>(node.push_constants_data, 3);
                nk.structural_pc_word4 = readPushConstantWord<uint32_t>(node.push_constants_data, 4);
            }
        }
        else if (node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_FP16 ||
                 node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16 ||
                 node.pipeline_id == Compute_Pipeline::ADAM_UPDATE_MULTI4_FP16)
        {
            if (node.push_constants_data.size() >= 16)
            {
                nk.structural_pc_word0 = readPushConstantWord<uint32_t>(node.push_constants_data, 0);
                nk.structural_pc_word1 = readPushConstantWord<uint32_t>(node.push_constants_data, 1);
                nk.structural_pc_word2 = readPushConstantWord<uint32_t>(node.push_constants_data, 2);
                nk.structural_pc_word3 = readPushConstantWord<uint32_t>(node.push_constants_data, 3);
            }
            if (node.pipeline_id == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16 &&
                node.push_constants_data.size() >= 24)
            {
                nk.structural_pc_word4 = readPushConstantWord<uint32_t>(node.push_constants_data, 4);
                nk.structural_pc_word5 = readPushConstantWord<uint32_t>(node.push_constants_data, 5);
            }
        }

        nk.canonical_buffer_ids.reserve(node.buffers.size());
        nk.buffer_data_types.reserve(node.buffers.size());
        for (size_t b_idx = 0; b_idx < node.buffers.size(); ++b_idx)
        {
            const auto& buffer = node.buffers[b_idx];
            nk.canonical_buffer_ids.push_back(getCanonicalBufferId(buffer));
            nk.buffer_data_types.push_back(buffer ? static_cast<uint8_t>(buffer->getDataType()) : 0xFF);
            if (b_idx < 32 && buffer && buffer.use_count() > static_cast<long>(raw_ptr_counts[buffer.get()]))
            {
                nk.external_owner_mask |= (1u << static_cast<uint32_t>(b_idx));
            }
        }

        if (!node.external_output_indices.empty())
        {
            nk.external_outputs.assign(node.external_output_indices.begin(), node.external_output_indices.end());
            std::sort(nk.external_outputs.begin(), nk.external_outputs.end());
        }

        key.nodes.push_back(std::move(nk));
    }
    return key;
}

size_t Execution_Engine::hashGraphTopologyKey(const Graph_Topology_Key& key) noexcept
{
    size_t graph_signature_hash = key.nodes.size();
    auto hashCombine = [&](size_t val) {
        graph_signature_hash ^= val + 0x9e3779b97f4a7c15ULL + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
    };

    hashCombine(static_cast<size_t>(key.is_coop_enabled ? 1 : 0));
    hashCombine(static_cast<size_t>(key.is_fused_gemm_adam ? 1 : 0));

    for (const auto& nk : key.nodes)
    {
        hashCombine(static_cast<size_t>(nk.pipeline_id));
        hashCombine(static_cast<size_t>(nk.workgroup_count_x));
        hashCombine(static_cast<size_t>(nk.workgroup_count_y));
        hashCombine(static_cast<size_t>(nk.workgroup_count_z));
        hashCombine(static_cast<size_t>(nk.spec_constant_u32));
        hashCombine(static_cast<size_t>(nk.push_constants_size));
        hashCombine(static_cast<size_t>(nk.structural_pc_word0));
        hashCombine(static_cast<size_t>(nk.structural_pc_word1));
        hashCombine(static_cast<size_t>(nk.structural_pc_word2));
        hashCombine(static_cast<size_t>(nk.structural_pc_word3));
        hashCombine(static_cast<size_t>(nk.structural_pc_word4));
        hashCombine(static_cast<size_t>(nk.structural_pc_word5));
        hashCombine(nk.fusion_plan_hash);
        hashCombine(static_cast<size_t>(nk.external_owner_mask));

        for (size_t i = 0; i < nk.canonical_buffer_ids.size(); ++i)
        {
            hashCombine(nk.canonical_buffer_ids[i]);
            hashCombine(static_cast<size_t>(nk.buffer_data_types[i]));
        }
        for (uint32_t ext_idx : nk.external_outputs)
        {
            hashCombine(static_cast<size_t>(ext_idx));
        }
    }
    return graph_signature_hash;
}

size_t Execution_Engine::computeGraphSignature(const Compute_Graph& graph) const
{
    return hashGraphTopologyKey(computeGraphTopologyKey(graph));
}

void Execution_Engine::precompileTemplatePipelines(Cached_Graph_Template& _template)
{
    for (auto& fused_node : _template.fused_nodes)
    {
        if (fused_node.is_fused && fused_node.fused_operations.size() > 1)
        {
            try
            {
                graph_executor->getExternalBufferIndices(fused_node, fused_node.cached_external_buffer_indices);
                fused_node.fused_glsl_code = graph_executor->generateFusedGlsl(fused_node);
                if (fused_node.fusion_plan && !fused_node.fusion_plan->canonical_key.empty())
                {
                    fused_node.cached_pipeline = pipeline_cache_manager->getOrCreatePipeline(
                        fused_node.fusion_plan->canonical_key,
                        fused_node.fused_glsl_code,
                        fused_node.fusion_plan->spec_constants);
                }
                else
                {
                    fused_node.cached_pipeline = pipeline_cache_manager->getOrCreatePipeline(fused_node.fused_glsl_code);
                }
            }
            catch (const std::exception& e)
            {
                Logger::logMessage(Input_Format{ "Execution_Engine::precompileTemplatePipelines: Fused pipeline compilation failed: {}. Fallback will be used.", e.what() },
                    Log_Level::LOG_WARNING,
                    true,
                    0,
                    Log_Feature::SHADER_GENERATION);
                fused_node.cached_pipeline = VK_NULL_HANDLE;
            }
        }
    }
    for (uint32_t frame_index = 0; frame_index < MAX_FRAMES_IN_FLIGHT; ++frame_index)
    {
        _template.instantiated_graphs[frame_index].setNodes(_template.fused_nodes);
    }
}

Execution_Engine::Execution_Engine()
{
    Logger::logMessage("Execution_Engine::Execution_Engine: Initializing execution engine",
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);

    context = std::make_unique<Vulkan_Context>();
    Graph_Optimizer::setCooperativeMatrixEnabled(context->isCooperativeMatrixEnabled());
    Graph_Optimizer::setFusedGemmAdamEnabled(is_fused_gemm_adam_enabled);
    shader_folder_path = Resource_Path::resolveDirectory(shader_folder_path, "VULKANML_SHADER_DIR", "compute_shader/spv");
    network = std::make_unique<Vulkan_Network>(*context, shader_folder_path);
    pipeline_cache_manager = std::make_unique<Pipeline_Cache_Manager>(*context, network->getPipelineLayout());
    network->setPipelineCacheManager(pipeline_cache_manager.get());
    shader_dictionary = std::make_unique<Shader_Dictionary>(
        Resource_Path::resolveFile("compute_shader/shader_dictionary.json",
                                   "VULKANML_SHADER_DICTIONARY",
                                   "compute_shader/shader_dictionary.json"));
    graph_executor = std::make_unique<Graph_Executor>(*context, *network, *pipeline_cache_manager, *shader_dictionary);

    uint32_t initial_frame_index = context->getCurrentFrame();
    context->prepareFrame();
    graph_executor->resetFrameState(initial_frame_index);
    frame_state_reset_done[initial_frame_index] = true;

    context->registerFlushCallback([this](VkFence _fence)
        {
            std::lock_guard<std::recursive_mutex> lock(engine_mutex);
            if (!current_graph.getNodes().empty() || context->hasPendingTransfers())
            {
                Logger::logMessage("Execution_Engine::flushCallback: Triggering executeGraph via flush callback",
                    Log_Level::LOG_DEBUG,
                    true,
                    0,
                    Log_Feature::DISPATCH_EXECUTION);
                this->executeGraph(_fence);
            }
        });
}

Execution_Engine::~Execution_Engine()
{
    Logger::logMessage("Execution_Engine::~Execution_Engine: Destroying execution engine",
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::DEVICE_MANAGEMENT);
    if (context)
    {
        context->registerFlushCallback(nullptr);
        if (context->getDevice() != VK_NULL_HANDLE)
        {
            VkResult idle_res = vkDeviceWaitIdle(context->getDevice());
            if (idle_res != VK_SUCCESS)
            {
                Logger::logMessage(Input_Format{"Execution_Engine::~Execution_Engine: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::DEVICE_MANAGEMENT);
            }
        }
    }
    current_graph.clear();
    cached_graph_templates.clear();
    dummy_scrub_buffers.clear();
    reusable_scratch_buffers.clear();
    graph_executor.reset();
    if (network)
    {
        network->setPipelineCacheManager(nullptr);
    }
    pipeline_cache_manager.reset();
    shader_dictionary.reset();
    network.reset();
    context.reset();
}

void Execution_Engine::invalidateGraphCache()
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    cached_graph_templates.clear();
    if (graph_executor)
    {
        graph_executor->invalidate();
        graph_executor->invalidateStaticGraph();
    }
}

void Execution_Engine::invalidateStaticGraph() noexcept
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (graph_executor)
    {
        graph_executor->invalidateStaticGraph();
    }
}

void Execution_Engine::warmCache(const Compute_Graph& _raw_graph)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (_raw_graph.getNodes().empty())
    {
        return;
    }

    if (!is_graph_cache_enabled)
    {
        Compute_Graph local_graph = _raw_graph;
        Graph_Optimizer::optimize(local_graph, shader_dictionary.get());
        graph_executor->warmupPipelineCache(local_graph);
        pipeline_cache_manager->savePipelineCache();
        return;
    }

    Graph_Topology_Key topology_key = computeGraphTopologyKey(_raw_graph);
    size_t graph_signature = hashGraphTopologyKey(topology_key);
    auto template_iterator = cached_graph_templates.find(topology_key);
    if (template_iterator == cached_graph_templates.end())
    {
        Cached_Graph_Template built_template = Graph_Optimizer::buildCachedTemplate(_raw_graph, shader_dictionary.get());
        if (!built_template.is_valid)
        {
            Compute_Graph local_graph = _raw_graph;
            Graph_Optimizer::optimize(local_graph, shader_dictionary.get());
            graph_executor->warmupPipelineCache(local_graph);
            pipeline_cache_manager->savePipelineCache();
            return;
        }
        built_template.topology_key = topology_key;
        auto [inserted_iterator, is_inserted] = cached_graph_templates.insert_or_assign(std::move(topology_key), std::move(built_template));
        template_iterator = inserted_iterator;
        precompileTemplatePipelines(template_iterator->second);
    }

    uint32_t current_frame_index = context->getCurrentFrame();
    auto& cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
    {
        CachedGraphScrubGuard scrub_guard{cached_graph, dummy_scrub_buffers};
        Graph_Optimizer::applyCachedTemplateInPlace(_raw_graph, template_iterator->second, cached_graph, shader_dictionary.get());
        graph_executor->warmupPipelineCache(cached_graph);
    }

    pipeline_cache_manager->savePipelineCache();
    Logger::logMessage(Input_Format{ "Execution_Engine::warmCache: Warmed cache for signature {}", graph_signature },
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::SHADER_GENERATION | Log_Feature::DISPATCH_EXECUTION);
}

void Execution_Engine::prepareCurrentFrame(Execution_Stage _stage)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    uint32_t current_frame_index = context->getCurrentFrame();
    if (!context->isFrameReady(current_frame_index))
    {
        auto wait_start_time = std::chrono::high_resolution_clock::now();
        context->prepareFrame(current_frame_index);
        auto wait_end_time = std::chrono::high_resolution_clock::now();
        last_fence_wait_ms = std::chrono::duration<double, std::milli>(wait_end_time - wait_start_time).count();

        graph_executor->resetFrameState(current_frame_index);
        frame_state_reset_done[current_frame_index] = true;
        processPendingLossReadback(current_frame_index);

        Execution_Stage waited_stage = (current_frame_index < frame_stages.size() && frame_stages[current_frame_index] != Execution_Stage::NONE)
            ? frame_stages[current_frame_index]
            : _stage;
        switch (waited_stage)
        {
        case Execution_Stage::FORWARD:
            Step_Timings::getInstance().recordDuration(Timing_Stage::FWD_LOSS_FENCE_WAIT, last_fence_wait_ms);
            break;
        case Execution_Stage::BACKWARD:
        case Execution_Stage::BACKWARD_OPTIMIZER:
            Step_Timings::getInstance().recordDuration(Timing_Stage::BWD_FENCE_WAIT, last_fence_wait_ms);
            break;
        case Execution_Stage::OPTIMIZER:
            Step_Timings::getInstance().recordDuration(Timing_Stage::OPT_FENCE_WAIT, last_fence_wait_ms);
            break;
        default:
            break;
        }
    }
    else
    {
        last_fence_wait_ms = 0.0;
        if (!frame_state_reset_done[current_frame_index])
        {
            graph_executor->resetFrameState(current_frame_index);
            frame_state_reset_done[current_frame_index] = true;
        }
        processPendingLossReadback(current_frame_index);
    }
}

void Execution_Engine::executeGraph(VkFence _external_fence, Execution_Stage _stage)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    prepareCurrentFrame(_stage);

    GraphExecutionCleanupGuard exec_cleanup{current_graph, context.get()};

    auto submit_start_time = std::chrono::high_resolution_clock::now();
    last_executed_node_count = current_graph.getNodes().size();
    uint32_t current_frame_index = context->getCurrentFrame();
    std::vector<Buffer_Transfer_Task> transfer_tasks = context->takeTransferTasks();

    if (current_graph.getNodes().empty() && transfer_tasks.empty())
    {
        Logger::logMessage("Execution_Engine::executeGraph: Executing empty compute graph and transfer task queue",
            Log_Level::LOG_WARNING,
            false,
            0,
            Log_Feature::DISPATCH_EXECUTION);
    }

    Logger::logMessage(Input_Format{ "Execution_Engine::executeGraph: Executing compute graph for frame {}", current_frame_index },
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::DISPATCH_EXECUTION);

    if (is_static_graph_enabled)
    {
        auto template_iterator = cached_graph_templates.end();
        size_t graph_signature = 0;
        if (is_graph_cache_enabled && !current_graph.getNodes().empty())
        {
            Graph_Topology_Key topology_key = computeGraphTopologyKey(current_graph);
            graph_signature = hashGraphTopologyKey(topology_key);
            template_iterator = cached_graph_templates.find(topology_key);
            if (template_iterator == cached_graph_templates.end())
            {
                Cached_Graph_Template built_template = Graph_Optimizer::buildCachedTemplate(current_graph, shader_dictionary.get());
                if (built_template.is_valid)
                {
                    built_template.topology_key = topology_key;
                    auto [inserted_iterator, is_inserted] = cached_graph_templates.insert_or_assign(std::move(topology_key), std::move(built_template));
                    template_iterator = inserted_iterator;
                    precompileTemplatePipelines(template_iterator->second);
                }
            }
        }

        if (template_iterator != cached_graph_templates.end())
        {
            auto& cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
            {
                CachedGraphScrubGuard scrub_guard{cached_graph, dummy_scrub_buffers};
                Graph_Optimizer::applyCachedTemplateInPlace(current_graph, template_iterator->second, cached_graph, shader_dictionary.get());

                if (!graph_executor->isStaticBaked(current_frame_index) ||
                    graph_executor->getStaticGraphSignature(current_frame_index) != graph_signature ||
                    !graph_executor->isStaticGraphBuffersMatching(cached_graph, current_frame_index))
                {
                    graph_executor->bakeStaticGraph(cached_graph, current_frame_index, graph_signature);
                }

                last_executed_node_count = cached_graph.getNodes().size();
                graph_executor->executeStaticGraph(cached_graph, transfer_tasks, current_frame_index, _external_fence, true);
            }
        }
        else if (!current_graph.getNodes().empty())
        {
            Graph_Optimizer::optimize(current_graph, shader_dictionary.get());
            size_t opt_signature = computeGraphSignature(current_graph);
            if (!graph_executor->isStaticBaked(current_frame_index) ||
                graph_executor->getStaticGraphSignature(current_frame_index) != opt_signature ||
                !graph_executor->isStaticGraphBuffersMatching(current_graph, current_frame_index))
            {
                graph_executor->bakeStaticGraph(current_graph, current_frame_index, opt_signature);
            }
            last_executed_node_count = current_graph.getNodes().size();
            graph_executor->executeStaticGraph(current_graph, transfer_tasks, current_frame_index, _external_fence, true);
        }
        else
        {
            last_executed_node_count = current_graph.getNodes().size();
            graph_executor->compileAndExecute(current_graph, transfer_tasks, current_frame_index, _external_fence);
        }
    }
    else
    {
        auto template_iterator = cached_graph_templates.end();
        if (is_graph_cache_enabled && !current_graph.getNodes().empty())
        {
            Graph_Topology_Key topology_key = computeGraphTopologyKey(current_graph);
            template_iterator = cached_graph_templates.find(topology_key);
            if (template_iterator == cached_graph_templates.end())
            {
                Cached_Graph_Template built_template = Graph_Optimizer::buildCachedTemplate(current_graph, shader_dictionary.get());
                if (built_template.is_valid)
                {
                    built_template.topology_key = topology_key;
                    auto [inserted_iterator, is_inserted] = cached_graph_templates.insert_or_assign(std::move(topology_key), std::move(built_template));
                    template_iterator = inserted_iterator;
                    precompileTemplatePipelines(template_iterator->second);
                }
            }
        }

        if (template_iterator != cached_graph_templates.end())
        {
            auto& cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
            {
                CachedGraphScrubGuard scrub_guard{cached_graph, dummy_scrub_buffers};
                Graph_Optimizer::applyCachedTemplateInPlace(current_graph, template_iterator->second, cached_graph, shader_dictionary.get());
                last_executed_node_count = cached_graph.getNodes().size();
                graph_executor->compileAndExecute(cached_graph, transfer_tasks, current_frame_index, _external_fence);
            }
        }
        else
        {
            Graph_Optimizer::optimize(current_graph, shader_dictionary.get());
            last_executed_node_count = current_graph.getNodes().size();
            graph_executor->compileAndExecute(current_graph, transfer_tasks, current_frame_index, _external_fence);
        }
    }

    auto submit_end_time = std::chrono::high_resolution_clock::now();
    last_submit_time_ms = std::chrono::duration<double, std::milli>(submit_end_time - submit_start_time).count();

    if (current_frame_index < loss_slots.size() && loss_slots[current_frame_index].has_pending_read)
    {
        loss_slots[current_frame_index].is_submitted = true;
    }

    current_graph.clear();
    exec_cleanup.active = false;

    Execution_Stage effective_stage = (_stage != Execution_Stage::NONE) ? _stage : current_stage;
    switch (effective_stage)
    {
    case Execution_Stage::FORWARD:
        Step_Timings::getInstance().recordDuration(Timing_Stage::FWD_LOSS_GPU_SUBMIT, last_submit_time_ms);
        break;
    case Execution_Stage::BACKWARD:
        Step_Timings::getInstance().recordDuration(Timing_Stage::BWD_GPU_SUBMIT, last_submit_time_ms);
        break;
    case Execution_Stage::BACKWARD_OPTIMIZER:
        Step_Timings::getInstance().recordDuration(Timing_Stage::BWD_GPU_SUBMIT, last_submit_time_ms);
        Step_Timings::getInstance().setChainedBwdOpt(true);
        break;
    case Execution_Stage::OPTIMIZER:
        Step_Timings::getInstance().recordDuration(Timing_Stage::OPT_GPU_SUBMIT, last_submit_time_ms);
        break;
    default:
        break;
    }

    if (current_frame_index < frame_stages.size())
    {
        frame_stages[current_frame_index] = effective_stage;
    }
    frame_state_reset_done[current_frame_index] = false;
    context->advanceFrame();

    last_execution_time_ms = last_submit_time_ms + last_fence_wait_ms;
}

void Execution_Engine::executeStaticReplay(VkFence _external_fence, Execution_Stage _stage)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    prepareCurrentFrame(_stage);
    GraphExecutionCleanupGuard exec_cleanup{current_graph, context.get()};

    uint32_t current_frame_index = context->getCurrentFrame();
    if (!graph_executor || !graph_executor->isStaticBaked(current_frame_index))
    {
        Logger::logMessage(Input_Format{"Execution_Engine::executeStaticReplay: Frame {} is not statically baked", current_frame_index},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);
        throw std::runtime_error("Execution_Engine::executeStaticReplay: Frame is not statically baked");
    }

    auto submit_start_time = std::chrono::high_resolution_clock::now();
    last_executed_node_count = graph_executor->getStaticSplitIndex(current_frame_index) + current_graph.getNodes().size();
    std::vector<Buffer_Transfer_Task> transfer_tasks = context->takeTransferTasks();
    graph_executor->executeStaticGraphReplay(current_graph, transfer_tasks, current_frame_index, _external_fence);
    auto submit_end_time = std::chrono::high_resolution_clock::now();
    last_submit_time_ms = std::chrono::duration<double, std::milli>(submit_end_time - submit_start_time).count();
    last_execution_time_ms = last_submit_time_ms + last_fence_wait_ms;

    Execution_Stage effective_stage = (_stage != Execution_Stage::NONE) ? _stage : current_stage;
    switch (effective_stage)
    {
    case Execution_Stage::FORWARD:
        Step_Timings::getInstance().recordDuration(Timing_Stage::FWD_LOSS_GPU_SUBMIT, last_submit_time_ms);
        break;
    case Execution_Stage::BACKWARD:
        Step_Timings::getInstance().recordDuration(Timing_Stage::BWD_GPU_SUBMIT, last_submit_time_ms);
        break;
    case Execution_Stage::BACKWARD_OPTIMIZER:
        Step_Timings::getInstance().recordDuration(Timing_Stage::BWD_GPU_SUBMIT, last_submit_time_ms);
        Step_Timings::getInstance().setChainedBwdOpt(true);
        break;
    case Execution_Stage::OPTIMIZER:
        Step_Timings::getInstance().recordDuration(Timing_Stage::OPT_GPU_SUBMIT, last_submit_time_ms);
        break;
    default:
        break;
    }

    if (current_frame_index < frame_stages.size())
    {
        frame_stages[current_frame_index] = effective_stage;
    }
    frame_state_reset_done[current_frame_index] = false;

    if (current_frame_index < loss_slots.size() && loss_slots[current_frame_index].has_pending_read)
    {
        loss_slots[current_frame_index].is_submitted = true;
        last_loss_frame_index = current_frame_index;
    }

    current_graph.clear();
    exec_cleanup.active = false;

    context->advanceFrame();
}

void Execution_Engine::waitIdleNoThrow() const noexcept
{
    if (context && context->getDevice() != VK_NULL_HANDLE)
    {
        VkResult idle_res = vkDeviceWaitIdle(context->getDevice());
        if (idle_res != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Execution_Engine::waitIdleNoThrow: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
        }
    }
}

void Execution_Engine::waitIdle() const
{
    Logger::logMessage("Execution_Engine::waitIdle: Waiting for device idle",
        Log_Level::LOG_DEBUG,
        true,
        0,
        Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
    if (context)
    {
        VkDevice device = context->getDevice();
        VkResult idle_res = vkDeviceWaitIdle(device);
        if (idle_res != VK_SUCCESS)
        {
            Logger::logMessage(Input_Format{"Execution_Engine::waitIdle: vkDeviceWaitIdle returned {}", static_cast<int>(idle_res)},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SYNCHRONIZATION);
            throw std::runtime_error("Execution_Engine::waitIdle: vkDeviceWaitIdle failed");
        }
        if (device != VK_NULL_HANDLE)
        {
            for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
            {
                VkFence fence = context->getFrameFence(f);
                if (fence != VK_NULL_HANDLE && vkGetFenceStatus(device, fence) == VK_NOT_READY)
                {
                    (void)vkWaitForFences(device, 1, &fence, VK_TRUE, 1'000'000'000ULL);
                }
            }
        }
    }
    else
    {
        Logger::logMessage("Execution_Engine::waitIdle: Attempted waitIdle on null Vulkan context",
            Log_Level::LOG_ERROR,
            true,
            0,
            Log_Feature::DEVICE_MANAGEMENT);
        throw std::runtime_error("Attempted waitIdle on null Vulkan context");
    }
}

void Execution_Engine::warmupPipelineCache()
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    graph_executor->warmupPipelineCache(current_graph);
}

void Execution_Engine::optimize()
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    Graph_Optimizer::optimize(current_graph, shader_dictionary.get());
}

std::shared_ptr<gpu::vector> Execution_Engine::acquireReusableScratchBuffer(
    size_t slot_id,
    size_t required_bytes,
    Data_Type dtype,
    size_t logical_elements)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (!context || context->getDevice() == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Execution_Engine::acquireReusableScratchBuffer: Vulkan context is not initialized");
    }
    const size_t alloc_bytes = std::max<size_t>(required_bytes, 4);
    auto it = reusable_scratch_buffers.find(slot_id);
    if (it != reusable_scratch_buffers.end() && it->second)
    {
        const auto &existing = it->second;
        if (!existing->isEmpty() &&
            existing->getDataType() == dtype &&
            existing->getBufferSizeInBytes() >= alloc_bytes)
        {
            return existing;
        }
    }
    auto buf = std::make_shared<gpu::vector>(*context);
    const size_t elem_size = std::max<size_t>(getDataTypeSize(dtype), 1);
    const size_t logical = (logical_elements > 0) ? logical_elements : std::max<size_t>(1, alloc_bytes / elem_size);
    buf->allocateMemoryByBytes(alloc_bytes, dtype, logical);
    reusable_scratch_buffers[slot_id] = buf;
    return buf;
}

bool Execution_Engine::isCooperativeMatrixEnabled() const noexcept
{
    return (context != nullptr) && context->isCooperativeMatrixEnabled();
}

void Execution_Engine::setCooperativeMatrixEnabled(bool _enable)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (_enable && (!context || !context->isCooperativeMatrixSupported()))
    {
        Logger::logMessage("Execution_Engine::setCooperativeMatrixEnabled: Device does not support Cooperative Tensor",
            Log_Level::LOG_WARNING,
            true,
            0,
            Log_Feature::DEVICE_MANAGEMENT);
        return;
    }

    if (isCooperativeMatrixEnabled() == _enable)
    {
        return;
    }

    waitIdle();
    if (context)
    {
        context->setCooperativeMatrixEnabled(_enable);
    }
    Graph_Optimizer::setCooperativeMatrixEnabled(isCooperativeMatrixEnabled());
    invalidateGraphCache();
}

void Execution_Engine::enableGraphCaching(bool _is_enabled)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    is_graph_cache_enabled = _is_enabled;
    if (!_is_enabled)
    {
        waitIdle();
        cached_graph_templates.clear();
        invalidateStaticGraph();
    }
}

void Execution_Engine::enableStaticGraph(bool _enable) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (is_static_graph_enabled != _enable)
    {
        waitIdleNoThrow();
        is_static_graph_enabled = _enable;
        invalidateStaticGraph();
    }
}

void Execution_Engine::setSingleSubmitEnabled(bool _enable) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (is_single_submit_enabled != _enable)
    {
        waitIdleNoThrow();
        is_single_submit_enabled = _enable;
        invalidateStaticGraph();
    }
}

void Execution_Engine::enableFusedGemmAdam(bool _enable) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (is_fused_gemm_adam_enabled != _enable)
    {
        waitIdleNoThrow();
        is_fused_gemm_adam_enabled = _enable;
        Graph_Optimizer::setFusedGemmAdamEnabled(_enable);
        invalidateGraphCache();
        invalidateStaticGraph();
    }
}

void Execution_Engine::setGpuProfilingEnabled(bool _enable)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (graph_executor)
    {
        graph_executor->setGpuProfilingEnabled(_enable);
    }
}

void Execution_Engine::flushGpuProfileResults()
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    waitIdle();
    if (graph_executor)
    {
        graph_executor->resolveAllPendingTimestamps();
    }
}

Gpu_Profile_Summary Execution_Engine::getGpuProfileSummary() const
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    return graph_executor ? graph_executor->getGpuProfileSummary() : Gpu_Profile_Summary{};
}

void Execution_Engine::resetGpuProfileStats()
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (graph_executor)
    {
        graph_executor->resetGpuProfileStats();
    }
}

std::string Execution_Engine::formatGpuProfileTable(size_t _top_k, size_t _batch_divisor) const
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (graph_executor)
    {
        return graph_executor->formatGpuProfileTable(_top_k, _batch_divisor);
    }
    return "[GPU Timestamp Profile] No completed GPU timestamp samples collected.\n";
}

void Execution_Engine::logGpuProfileTable(size_t _top_k, size_t _batch_divisor, Log_Level _level, bool _print_to_console) const
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (graph_executor)
    {
        graph_executor->logGpuProfileTable(_top_k, _batch_divisor, _level, _print_to_console);
        return;
    }
    if (Logger::shouldLog(_level, Log_Feature::PROFILER))
    {
        Logger::logMessage("[GPU Timestamp Profile] No completed GPU timestamp samples collected.",
                           _level, _print_to_console, 0, Log_Feature::PROFILER);
    }
}

void Execution_Engine::printGpuProfileTable(size_t _top_k, size_t _batch_divisor) const
{
    logGpuProfileTable(_top_k, _batch_divisor, Log_Level::LOG_INFO, true);
}

void Execution_Engine::printGpuProfileTable(std::ostream &_os, size_t _top_k, size_t _batch_divisor) const
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    if (graph_executor)
    {
        graph_executor->printGpuProfileTable(_os, _top_k, _batch_divisor);
    }
}

float Execution_Engine::readPendingLoss(uint32_t frame_index)
{
    std::lock_guard<std::recursive_mutex> lock(engine_mutex);
    auto drainSlot = [&](uint32_t idx)
    {
        if (idx < loss_slots.size() && loss_slots[idx].has_pending_read)
        {
            if (context && !context->isFrameReady(idx))
            {
                context->prepareFrame(idx);
                if (graph_executor)
                {
                    graph_executor->resetFrameState(idx);
                }
                if (idx < frame_state_reset_done.size())
                {
                    frame_state_reset_done[idx] = true;
                }
            }
            processPendingLossReadback(idx);
        }
    };

    if (frame_index == UINT32_MAX)
    {
        for (uint32_t idx = 0; idx < MAX_FRAMES_IN_FLIGHT; ++idx)
        {
            if (idx != last_loss_frame_index)
            {
                drainSlot(idx);
            }
        }
        if (last_loss_frame_index < MAX_FRAMES_IN_FLIGHT)
        {
            drainSlot(last_loss_frame_index);
        }
    }
    else
    {
        drainSlot(frame_index % MAX_FRAMES_IN_FLIGHT);
    }
    return latest_loss;
}

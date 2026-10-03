#include "engine/execution_engine.h"

#include <chrono>
#include <cmath>
#include <format>
#include <stdexcept>

#include "engine/graph_executor.h"
#include "engine/graph_optimizer.h"
#include "engine/pipeline_cache_manager.h"
#include "engine/shader_dictionary.h"
#include "engine/vulkan_context.h"
#include "engine/vulkan_network.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"

Execution_Engine &Execution_Engine::getInstance()
{
    static Execution_Engine instance;
    return instance;
}


void Execution_Engine::processPendingLossReadback(uint32_t frame_index)
    {
        if (frame_index < loss_slots.size() && loss_slots[frame_index].has_pending_read)
        {
            auto read_start = std::chrono::high_resolution_clock::now();
            auto& slot = loss_slots[frame_index];
            if (slot.buffer && slot.buffer->isHostMapped())
            {
                const float* ptr = static_cast<const float*>(slot.buffer->getHostMappedPointer());
                if (ptr)
                {
                    float sum = 0.0f;
                    for (size_t i = 0; i < slot.total_tokens; ++i)
                    {
                        float val = ptr[i];
                        if (!std::isnan(val) && !std::isinf(val))
                        {
                            sum += val;
                        }
                    }
                    if (slot.valid_tokens > 0)
                    {
                        latest_loss = sum / static_cast<float>(slot.valid_tokens);
                    }
                }
            }
            slot.has_pending_read = false;
            auto read_end = std::chrono::high_resolution_clock::now();
            double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
            Step_Timings::getInstance().loss_read_ms += read_ms;
            Step_Timings::getInstance().loss_readback_ms += read_ms;
        }
    }

size_t Execution_Engine::computeGraphSignature(const Compute_Graph &graph) const
    {
        const auto &nodes = graph.getNodes();
        size_t graph_signature_hash = nodes.size();
        graph_signature_hash ^= static_cast<size_t>(is_coop ? 1 : 0) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
        graph_signature_hash ^= static_cast<size_t>(is_fused_gemm_adam_enabled ? 1 : 0) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);

        std::unordered_map<VkBuffer, size_t> buffer_to_id;
        buffer_to_id.reserve(nodes.size() * 2);
        size_t next_id = 0;

        auto getCanonicalBufferId = [&](const std::shared_ptr<gpu::vector> &buf) -> size_t {
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

        for (size_t i = 0; i < nodes.size(); ++i)
        {
            graph_signature_hash ^= static_cast<size_t>(nodes[i].pipeline_id) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<size_t>(nodes[i].workgroup_count_x) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<size_t>(nodes[i].workgroup_count_y) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<size_t>(nodes[i].workgroup_count_z) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= nodes[i].push_constants_data.size() + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);

            for (const auto &buffer : nodes[i].buffers)
            {
                size_t canon_id = getCanonicalBufferId(buffer);
                graph_signature_hash ^= canon_id + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            }
        }
        return graph_signature_hash;
    }

void Execution_Engine::precompileTemplatePipelines(Cached_Graph_Template &_template)
    {
        for (auto &fused_node : _template.fused_nodes)
        {
            if (fused_node.is_fused && fused_node.fused_operations.size() > 1)
            {
                try
                {
                    graph_executor->getExternalBufferIndices(fused_node, fused_node.cached_external_buffer_indices);
                    fused_node.fused_glsl_code = graph_executor->generateFusedGlsl(fused_node);
                    fused_node.cached_pipeline = pipeline_cache_manager->getOrCreatePipeline(fused_node.fused_glsl_code);
                }
                catch (const std::exception &e)
                {
                    Logger::logMessage(Input_Format{"Execution_Engine::precompileTemplatePipelines: Fused pipeline compilation failed: {}. Fallback will be used.", e.what()},
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
        is_coop = context->isCooperativeMatrixEnabled();
        Graph_Optimizer::setFusedGemmAdamEnabled(is_fused_gemm_adam_enabled);
        network = std::make_unique<Vulkan_Network>(*context, shader_folder_path);
        pipeline_cache_manager = std::make_unique<Pipeline_Cache_Manager>(*context, network->getPipelineLayout());
        shader_dictionary = std::make_unique<Shader_Dictionary>("compute_shader/shader_dictionary.json");
        graph_executor = std::make_unique<Graph_Executor>(*context, *network, *pipeline_cache_manager, *shader_dictionary);

        uint32_t initial_frame_index = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(initial_frame_index);
        graph_executor->resetFrameState(initial_frame_index);

        context->registerFlushCallback([this](VkFence _fence)
                                       {
            if (!current_graph.getNodes().empty() || !context->getTransferTasks().empty())
            {
                Logger::logMessage("Execution_Engine::flushCallback: Triggering executeGraph via flush callback",
                                   Log_Level::LOG_DEBUG,
                                   true,
                                   0,
                                   Log_Feature::DISPATCH_EXECUTION);
                this->executeGraph(_fence);
            } });
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
            vkDeviceWaitIdle(context->getDevice());
        }
    }

void Execution_Engine::invalidateGraphCache()
    {
        cached_graph_templates.clear();
        if (graph_executor)
        {
            graph_executor->invalidate();
            graph_executor->invalidateStaticGraph();
        }
    }

void Execution_Engine::invalidateStaticGraph() noexcept
    {
        if (graph_executor)
        {
            graph_executor->invalidateStaticGraph();
        }
    }

void Execution_Engine::warmCache(const Compute_Graph &_raw_graph)
    {
        if (_raw_graph.getNodes().empty())
        {
            return;
        }

        size_t graph_signature = computeGraphSignature(_raw_graph);
        auto template_iterator = cached_graph_templates.find(graph_signature);
        if (template_iterator == cached_graph_templates.end())
        {
            auto [inserted_iterator, is_inserted] = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(_raw_graph));
            template_iterator = inserted_iterator;
            precompileTemplatePipelines(template_iterator->second);
        }

        uint32_t current_frame_index = context->getCurrentFrame();
        auto &cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
        Graph_Optimizer::applyCachedTemplateInPlace(_raw_graph, template_iterator->second, cached_graph);
        graph_executor->warmupPipelineCache(cached_graph);

        pipeline_cache_manager->savePipelineCache();
        Logger::logMessage(Input_Format{"Execution_Engine::warmCache: Warmed cache for signature {}", graph_signature},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION | Log_Feature::DISPATCH_EXECUTION);
    }

void Execution_Engine::prepareCurrentFrame(Execution_Stage _stage)
    {
        uint32_t current_frame_index = context->getCurrentFrame();
        if (!context->isFrameReady(current_frame_index))
        {
            auto wait_start_time = std::chrono::high_resolution_clock::now();
            context->prepareFrame(current_frame_index);
            auto wait_end_time = std::chrono::high_resolution_clock::now();
            last_fence_wait_ms = std::chrono::duration<double, std::milli>(wait_end_time - wait_start_time).count();

            context->cleanGarbage(current_frame_index);
            graph_executor->resetFrameState(current_frame_index);
            processPendingLossReadback(current_frame_index);

            Execution_Stage waited_stage = (current_frame_index < frame_stages.size()) ? frame_stages[current_frame_index] : Execution_Stage::NONE;
            switch (waited_stage)
            {
            case Execution_Stage::FORWARD:
                Step_Timings::getInstance().fwd_loss_fence_wait_ms += last_fence_wait_ms;
                break;
            case Execution_Stage::BACKWARD:
            case Execution_Stage::BACKWARD_OPTIMIZER:
                Step_Timings::getInstance().bwd_fence_wait_ms += last_fence_wait_ms;
                break;
            case Execution_Stage::OPTIMIZER:
                Step_Timings::getInstance().opt_fence_wait_ms += last_fence_wait_ms;
                break;
            default:
                break;
            }
        }
        else
        {
            last_fence_wait_ms = 0.0;
            processPendingLossReadback(current_frame_index);
        }
    }

void Execution_Engine::executeGraph(VkFence _external_fence, Execution_Stage _stage)
    {
        prepareCurrentFrame(_stage);

        auto submit_start_time = std::chrono::high_resolution_clock::now();
        last_executed_node_count = current_graph.getNodes().size();
        uint32_t current_frame_index = context->getCurrentFrame();

        if (current_graph.getNodes().empty() && context->getTransferTasks().empty())
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
            if (is_graph_cache_enabled && !current_graph.getNodes().empty())
            {
                size_t graph_signature = computeGraphSignature(current_graph);
                auto template_iterator = cached_graph_templates.find(graph_signature);
                if (template_iterator == cached_graph_templates.end())
                {
                    auto [inserted_iterator, is_inserted] = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(current_graph));
                    template_iterator = inserted_iterator;
                    precompileTemplatePipelines(template_iterator->second);
                }

                auto& cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
                Graph_Optimizer::applyCachedTemplateInPlace(current_graph, template_iterator->second, cached_graph);

                if (!graph_executor->isStaticBaked(current_frame_index) ||
                    graph_executor->getStaticGraphSignature(current_frame_index) != graph_signature ||
                    !graph_executor->isStaticGraphBuffersMatching(cached_graph, current_frame_index))
                {
                    graph_executor->bakeStaticGraph(cached_graph, current_frame_index, graph_signature);
                }

                last_executed_node_count = cached_graph.getNodes().size();
                graph_executor->executeStaticGraph(cached_graph, context->getTransferTasks(), current_frame_index, _external_fence);
            }
            else if (!current_graph.getNodes().empty())
            {
                Graph_Optimizer::optimize(current_graph);
                size_t graph_signature = computeGraphSignature(current_graph);
                if (!graph_executor->isStaticBaked(current_frame_index) ||
                    graph_executor->getStaticGraphSignature(current_frame_index) != graph_signature ||
                    !graph_executor->isStaticGraphBuffersMatching(current_graph, current_frame_index))
                {
                    graph_executor->bakeStaticGraph(current_graph, current_frame_index, graph_signature);
                }
                last_executed_node_count = current_graph.getNodes().size();
                graph_executor->executeStaticGraph(current_graph, context->getTransferTasks(), current_frame_index, _external_fence);
            }
            else
            {
                last_executed_node_count = current_graph.getNodes().size();
                graph_executor->compileAndExecute(current_graph, context->getTransferTasks(), current_frame_index, _external_fence);
            }
        }
        else if (is_graph_cache_enabled && !current_graph.getNodes().empty())
        {
            size_t graph_signature = computeGraphSignature(current_graph);
            auto template_iterator = cached_graph_templates.find(graph_signature);
            if (template_iterator == cached_graph_templates.end())
            {
                auto [inserted_iterator, is_inserted] = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(current_graph));
                template_iterator = inserted_iterator;
                precompileTemplatePipelines(template_iterator->second);
            }

            auto& cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
            Graph_Optimizer::applyCachedTemplateInPlace(current_graph, template_iterator->second, cached_graph);
            last_executed_node_count = cached_graph.getNodes().size();
            graph_executor->compileAndExecute(cached_graph, context->getTransferTasks(), current_frame_index, _external_fence);
        }
        else
        {
            Graph_Optimizer::optimize(current_graph);
            last_executed_node_count = current_graph.getNodes().size();
            graph_executor->compileAndExecute(current_graph, context->getTransferTasks(), current_frame_index, _external_fence);
        }

        auto submit_end_time = std::chrono::high_resolution_clock::now();
        last_submit_time_ms = std::chrono::duration<double, std::milli>(submit_end_time - submit_start_time).count();

        context->clearTransferTasks();
        current_graph.clear();

        Execution_Stage effective_stage = (_stage != Execution_Stage::NONE) ? _stage : current_stage;
        switch (effective_stage)
        {
        case Execution_Stage::FORWARD:
            Step_Timings::getInstance().fwd_loss_gpu_submit_ms += last_submit_time_ms;
            break;
        case Execution_Stage::BACKWARD:
            Step_Timings::getInstance().bwd_gpu_submit_ms += last_submit_time_ms;
            break;
        case Execution_Stage::BACKWARD_OPTIMIZER:
            Step_Timings::getInstance().bwd_gpu_submit_ms += last_submit_time_ms;
            Step_Timings::getInstance().is_chained_bwd_opt = true;
            break;
        case Execution_Stage::OPTIMIZER:
            Step_Timings::getInstance().opt_gpu_submit_ms += last_submit_time_ms;
            break;
        default:
            break;
        }

        if (current_frame_index < frame_stages.size())
        {
            frame_stages[current_frame_index] = effective_stage;
        }
        context->advanceFrame();

        last_execution_time_ms = last_submit_time_ms + last_fence_wait_ms;
        
    }

void Execution_Engine::executeStaticReplay(VkFence _external_fence)
    {
        prepareCurrentFrame();
        uint32_t current_frame_index = context->getCurrentFrame();
        graph_executor->executeStaticGraphReplay(current_graph, context->getTransferTasks(), current_frame_index, _external_fence);

        context->clearTransferTasks();
        current_graph.clear();

        context->advanceFrame();
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
            vkDeviceWaitIdle(context->getDevice());
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
        graph_executor->warmupPipelineCache(current_graph);
    }

void Execution_Engine::optimize()
{
        Graph_Optimizer::optimize(current_graph);
    }

void Execution_Engine::setCooperativeMatrixEnabled(bool _enable)
    {
        if (_enable && (!context || !context->isCooperativeMatrixSupported()))
        {
            Logger::logMessage("Execution_Engine::setCooperativeMatrixEnabled: Device does not support Cooperative Tensor",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT);
            return;
        }

        if (is_coop == _enable)
        {
            return;
        }

        waitIdle();
        is_coop = _enable;
        if (context)
        {
            context->setCooperativeMatrixEnabled(_enable);
        }
        invalidateGraphCache();
    }

void Execution_Engine::setGraphCachingEnabled(bool _is_enabled)
    {
        is_graph_cache_enabled = _is_enabled;
        if (!_is_enabled)
        {
            cached_graph_templates.clear();
        }
    }

void Execution_Engine::setStaticGraphEnabled(bool _enable) noexcept
    {
        if (is_static_graph_enabled != _enable)
        {
            is_static_graph_enabled = _enable;
            invalidateStaticGraph();
        }
    }

void Execution_Engine::setFusedGemmAdamEnabled(bool _enable) noexcept
    {
        if (is_fused_gemm_adam_enabled != _enable)
        {
            waitIdle();
            is_fused_gemm_adam_enabled = _enable;
            Graph_Optimizer::setFusedGemmAdamEnabled(_enable);
            invalidateGraphCache();
            invalidateStaticGraph();
        }
    }

float Execution_Engine::readPendingLoss(uint32_t frame_index)
    {
        uint32_t idx = frame_index % MAX_FRAMES_IN_FLIGHT;
        if (loss_slots[idx].has_pending_read)
        {
            if (context)
            {
                vkDeviceWaitIdle(context->getDevice());
            }
            processPendingLossReadback(idx);
        }
        return latest_loss;
    }

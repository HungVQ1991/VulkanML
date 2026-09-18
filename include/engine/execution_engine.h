#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "graph_executor.h"
#include "graph_optimizer.h"
#include "helper/logger.h"
#include "pipeline_cache_manager.h"
#include "shader_dictionary.h"
#include "vulkan_context.h"
#include "vulkan_network.h"

extern bool is_coop;

class Execution_Engine
{
private:
    std::unique_ptr<Vulkan_Context> context;
    std::unique_ptr<Vulkan_Network> network;
    std::unique_ptr<Pipeline_Cache_Manager> pipeline_cache_manager;
    std::unique_ptr<Shader_Dictionary> shader_dictionary;
    std::string shader_folder_path = "compute_shader";

    Compute_Graph current_graph;
    std::unique_ptr<Graph_Executor> graph_executor;

    std::unordered_map<std::size_t, Cached_Graph_Template> cached_graph_templates;
    bool is_graph_cache_enabled = true;

    std::size_t computeGraphSignature(const Compute_Graph &graph) const
    {
        const auto &nodes = graph.getNodes();
        std::size_t graph_signature_hash = nodes.size();
        graph_signature_hash ^= static_cast<std::size_t>(is_coop ? 1 : 0) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);

        std::unordered_map<VkBuffer, std::size_t> buffer_to_id;
        buffer_to_id.reserve(nodes.size() * 2);
        std::size_t next_id = 0;

        auto getCanonicalBufferId = [&](const std::shared_ptr<gpu::vector> &buf) -> std::size_t {
            if (!buf)
            {
                return static_cast<std::size_t>(-1);
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

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].pipeline_id) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].workgroup_count_x) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].workgroup_count_y) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].workgroup_count_z) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= nodes[i].push_constants_data.size() + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);

            for (const auto &buffer : nodes[i].buffers)
            {
                std::size_t canon_id = getCanonicalBufferId(buffer);
                graph_signature_hash ^= canon_id + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            }
        }
        return graph_signature_hash;
    }

    void precompileTemplatePipelines(Cached_Graph_Template &_template)
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
        for (std::uint32_t frame_index = 0; frame_index < MAX_FRAMES_IN_FLIGHT; ++frame_index)
        {
            _template.instantiated_graphs[frame_index].setNodes(_template.fused_nodes);
        }
    }

    Execution_Engine()
    {
        Logger::logMessage("Execution_Engine::Execution_Engine: Initializing execution engine",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);

        context = std::make_unique<Vulkan_Context>();
        is_coop = context->isCooperativeMatrixEnabled();
        network = std::make_unique<Vulkan_Network>(*context, shader_folder_path);
        pipeline_cache_manager = std::make_unique<Pipeline_Cache_Manager>(*context, network->getPipelineLayout());
        shader_dictionary = std::make_unique<Shader_Dictionary>("compute_shader/shader_dictionary.json");
        graph_executor = std::make_unique<Graph_Executor>(*context, *network, *pipeline_cache_manager, *shader_dictionary);

        std::uint32_t initial_frame_index = context->getCurrentFrame();
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

public:
    ~Execution_Engine()
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

    Execution_Engine(const Execution_Engine &) = delete;
    Execution_Engine &operator=(const Execution_Engine &) = delete;
    Execution_Engine(Execution_Engine &&) = delete;
    Execution_Engine &operator=(Execution_Engine &&) = delete;

    static Execution_Engine &getInstance()
    {
        static Execution_Engine instance;
        return instance;
    }


    void invalidateGraphCache()
    {
        cached_graph_templates.clear();
        if (graph_executor)
        {
            graph_executor->invalidate();
        }
    }

    void warmCache(const Compute_Graph &_raw_graph)
    {
        if (_raw_graph.getNodes().empty())
        {
            return;
        }

        std::size_t graph_signature = computeGraphSignature(_raw_graph);
        auto template_iterator = cached_graph_templates.find(graph_signature);
        if (template_iterator == cached_graph_templates.end())
        {
            auto [inserted_iterator, is_inserted] = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(_raw_graph));
            template_iterator = inserted_iterator;
            precompileTemplatePipelines(template_iterator->second);
        }

        std::uint32_t current_frame_index = context->getCurrentFrame();
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

    void executeGraph(VkFence _external_fence = VK_NULL_HANDLE)
    {
        std::uint32_t current_frame_index = context->getCurrentFrame();

        if (current_graph.getNodes().empty() && context->getTransferTasks().empty())
        {
            Logger::logMessage("Execution_Engine::executeGraph: Executing empty compute graph and transfer task queue",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
        }

        Logger::logMessage(Input_Format{"Execution_Engine::executeGraph: Executing compute graph for frame {}", current_frame_index},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);

        if (is_graph_cache_enabled && !current_graph.getNodes().empty())
        {
            std::size_t graph_signature = computeGraphSignature(current_graph);
            auto template_iterator = cached_graph_templates.find(graph_signature);
            if (template_iterator == cached_graph_templates.end())
            {
                auto [inserted_iterator, is_inserted] = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(current_graph));
                template_iterator = inserted_iterator;
                precompileTemplatePipelines(template_iterator->second);
            }

            auto &cached_graph = template_iterator->second.instantiated_graphs[current_frame_index];
            Graph_Optimizer::applyCachedTemplateInPlace(current_graph, template_iterator->second, cached_graph);
            graph_executor->compileAndExecute(cached_graph, context->getTransferTasks(), current_frame_index, _external_fence);
        }
        else
        {
            Graph_Optimizer::optimize(current_graph);
            graph_executor->compileAndExecute(current_graph, context->getTransferTasks(), current_frame_index, _external_fence);
        }

        context->resetStagingOffset(current_frame_index);
        context->clearTransferTasks();
        current_graph.clear();

        context->advanceFrame();

        std::uint32_t next_frame_index = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(next_frame_index);
        graph_executor->resetFrameState(next_frame_index);
    }

    void waitIdle() const
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

    void warmupPipelineCache()
    {
        graph_executor->warmupPipelineCache(current_graph);
    }

    void optimize()
    {
        Graph_Optimizer::optimize(current_graph);
    }

    const std::unordered_map<std::size_t, Cached_Graph_Template> &getCachedGraphTemplates() const noexcept { return cached_graph_templates; }
    const std::string &getShaderFolderPath() const noexcept { return shader_folder_path; }
    const Pipeline_Cache_Manager &getPipelineCacheManager() const noexcept { return *pipeline_cache_manager; }
    Pipeline_Cache_Manager &getPipelineCacheManager() noexcept { return *pipeline_cache_manager; }
    const Shader_Dictionary &getShaderDictionary() const noexcept { return *shader_dictionary; }
    Shader_Dictionary &getShaderDictionary() noexcept { return *shader_dictionary; }
    const Graph_Executor &getGraphExecutor() const noexcept { return *graph_executor; }
    Graph_Executor &getGraphExecutor() noexcept { return *graph_executor; }
    const Vulkan_Network &getNetwork() const noexcept { return *network; }
    Vulkan_Network &getNetwork() noexcept { return *network; }
    const Vulkan_Context &getContext() const noexcept { return *context; }
    Vulkan_Context &getContext() noexcept { return *context; }
    const Compute_Graph &getCurrentGraph() const noexcept { return current_graph; }
    Compute_Graph &getCurrentGraph() noexcept { return current_graph; }
    bool isCooperativeMatrixSupported() const noexcept { return context && context->isCooperativeMatrixSupported(); }
    bool isCooperativeMatrixEnabled() const noexcept { return is_coop; }
    bool isGraphCacheEnabled() const noexcept { return is_graph_cache_enabled; }

    void setShaderFolderPath(const std::string &_path) { shader_folder_path = _path; }
    void setCooperativeMatrixEnabled(bool _enable)
    {
        if (_enable && (!context || !context->isCooperativeMatrixSupported()))
        {
            Logger::logMessage("Execution_Engine::setCooperativeMatrixEnabled: Device does not support Cooperative Matrix",
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
    void setGraphCachingEnabled(bool _is_enabled)
    {
        is_graph_cache_enabled = _is_enabled;
        if (!_is_enabled)
        {
            cached_graph_templates.clear();
        }
    }
    void enableGraphCaching(bool _is_enabled) { setGraphCachingEnabled(_is_enabled); }
};
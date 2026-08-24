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
    bool is_graph_cache_enabled = false;

    std::size_t computeGraphSignature(const Compute_Graph &graph) const
    {
        const auto &nodes = graph.getNodes();
        std::size_t graph_signature_hash = nodes.size();

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].pipeline_id) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].workgroup_count_x) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].workgroup_count_y) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= static_cast<std::size_t>(nodes[i].workgroup_count_z) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            graph_signature_hash ^= nodes[i].push_constants_data.size() + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);

            if (i > 0)
            {
                bool is_sharing_buffer = false;
                for (const auto &buffer_a : nodes[i - 1].buffers)
                {
                    if (!buffer_a)
                    {
                        continue;
                    }
                    for (const auto &buffer_b : nodes[i].buffers)
                    {
                        if (!buffer_b)
                        {
                            continue;
                        }
                        if (buffer_a == buffer_b || (buffer_a->getBuffer() != VK_NULL_HANDLE && buffer_a->getBuffer() == buffer_b->getBuffer()))
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
                graph_signature_hash ^= static_cast<std::size_t>(is_sharing_buffer ? 1 : 0) + 0x9e3779b9 + (graph_signature_hash << 6) + (graph_signature_hash >> 2);
            }
        }
        return graph_signature_hash;
    }

    Execution_Engine()
    {
        Logger::logMessage("Execution_Engine::Execution_Engine: Initializing execution engine",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);

        context = std::make_unique<Vulkan_Context>();
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

    [[nodiscard]] const Vulkan_Context &getContext() const noexcept
    {
        return *context;
    }

    [[nodiscard]] Vulkan_Context &getContext() noexcept
    {
        return *context;
    }

    [[nodiscard]] const Vulkan_Network &getNetwork() const noexcept
    {
        return *network;
    }

    [[nodiscard]] Vulkan_Network &getNetwork() noexcept
    {
        return *network;
    }

    [[nodiscard]] const Pipeline_Cache_Manager &getPipelineCacheManager() const noexcept
    {
        return *pipeline_cache_manager;
    }

    [[nodiscard]] Pipeline_Cache_Manager &getPipelineCacheManager() noexcept
    {
        return *pipeline_cache_manager;
    }

    [[nodiscard]] const Shader_Dictionary &getShaderDictionary() const noexcept
    {
        return *shader_dictionary;
    }

    [[nodiscard]] Shader_Dictionary &getShaderDictionary() noexcept
    {
        return *shader_dictionary;
    }

    [[nodiscard]] const Graph_Executor &getGraphExecutor() const noexcept
    {
        return *graph_executor;
    }

    [[nodiscard]] Graph_Executor &getGraphExecutor() noexcept
    {
        return *graph_executor;
    }

    [[nodiscard]] const Compute_Graph &getCurrentGraph() const noexcept
    {
        return current_graph;
    }

    [[nodiscard]] Compute_Graph &getCurrentGraph() noexcept
    {
        return current_graph;
    }

    [[nodiscard]] const std::string &getShaderFolderPath() const noexcept
    {
        return shader_folder_path;
    }

    [[nodiscard]] const std::unordered_map<std::size_t, Cached_Graph_Template> &getCachedGraphTemplates() const noexcept
    {
        return cached_graph_templates;
    }

    [[nodiscard]] bool isGraphCacheEnabled() const noexcept
    {
        return is_graph_cache_enabled;
    }

    void setGraphCachingEnabled(bool _is_enabled)
    {
        is_graph_cache_enabled = _is_enabled;
        if (!_is_enabled)
        {
            cached_graph_templates.clear();
        }
    }

    void enableGraphCaching(bool _is_enabled)
    {
        setGraphCachingEnabled(_is_enabled);
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
            template_iterator = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(_raw_graph)).first;
        }

        Compute_Graph optimized_graph;
        Graph_Optimizer::applyCachedTemplate(_raw_graph, template_iterator->second, optimized_graph);
        graph_executor->warmupPipelineCache(optimized_graph);

        pipeline_cache_manager->savePipelineCache();
        Logger::logMessage(std::format("Execution_Engine::warmCache: Warmed cache for signature {}", graph_signature),
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

        Logger::logMessage(std::format("Execution_Engine::executeGraph: Executing compute graph for frame {}", current_frame_index),
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
                template_iterator = cached_graph_templates.emplace(graph_signature, Graph_Optimizer::buildCachedTemplate(current_graph)).first;
            }

            Compute_Graph optimized_graph;
            Graph_Optimizer::applyCachedTemplate(current_graph, template_iterator->second, optimized_graph);
            graph_executor->compileAndExecute(optimized_graph, context->getTransferTasks(), current_frame_index, _external_fence);
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
};
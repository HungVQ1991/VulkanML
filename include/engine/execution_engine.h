#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "graph_executor.h"
#include "helper/logger.h"
#include "vulkan_context.h"
#include "vulkan_network.h"
#include "pipeline_cache_manager.h"
#include "graph_optimizer.h"
#include "shader_dictionary.h"

#ifndef ENABLE_ENGINE_DEBUG_LOGS
#define ENABLE_ENGINE_DEBUG_LOGS 0
#endif

#if ENABLE_ENGINE_DEBUG_LOGS
#define ENGINE_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define ENGINE_LOG_DEBUG(msg) ((void)0)
#endif

class Execution_Engine
{
private:
    std::unique_ptr<Vulkan_Context> context;
    std::unique_ptr<Vulkan_Network> network;
    std::unique_ptr<Pipeline_Cache_Manager> cache_manager;
    std::unique_ptr<Shader_Dictionary> shader_dict;
    std::string shader_folder = "compute_shader";

    Compute_Graph current_graph;
    std::unique_ptr<Graph_Executor> executor;

    Compute_Graph optimized_graph;
    bool is_graph_compiled = false;

    std::shared_ptr<GVector> compiled_input_buffer;
    std::shared_ptr<GVector> compiled_target_buffer;

    std::unordered_map<std::size_t, Cached_Graph_Template> graph_cache_map;
    bool is_graph_cache_enabled = false;

    std::size_t computeGraphSignature(const Compute_Graph &graph) const
    {
        const auto &nodes = graph.getNodes();
        std::size_t seed = nodes.size();

        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            // Hash Pipeline ID
            seed ^= static_cast<std::size_t>(nodes[i].pipeline_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash Dispatch Grid (Group X, Y, Z)
            seed ^= static_cast<std::size_t>(nodes[i].group_x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= static_cast<std::size_t>(nodes[i].group_y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= static_cast<std::size_t>(nodes[i].group_z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash Push Constants Size
            seed ^= nodes[i].push_constants_data.size() + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            if (i > 0)
            {
                bool share_buffer = false;
                for (const auto &buf_a : nodes[i - 1].buffers)
                {
                    if (!buf_a)
                        continue;
                    for (const auto &buf_b : nodes[i].buffers)
                    {
                        if (!buf_b)
                            continue;
                        if (buf_a == buf_b || (buf_a->getBuffer() != VK_NULL_HANDLE && buf_a->getBuffer() == buf_b->getBuffer()))
                        {
                            share_buffer = true;
                            break;
                        }
                    }
                    if (share_buffer)
                        break;
                }
                seed ^= static_cast<std::size_t>(share_buffer ? 1 : 0) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }

    Execution_Engine()
    {
        ENGINE_LOG_DEBUG("Execution_Engine::Execution_Engine: Initializing execution engine");

        context = std::make_unique<Vulkan_Context>();
        network = std::make_unique<Vulkan_Network>(*context, shader_folder);
        cache_manager = std::make_unique<Pipeline_Cache_Manager>(*context, network->getPipelineLayout());
        shader_dict = std::make_unique<Shader_Dictionary>("compute_shader/shader_dictionary.json");
        executor = std::make_unique<Graph_Executor>(*context, *network, *cache_manager, *shader_dict);

        std::uint32_t initial_frame = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(initial_frame);
        executor->resetFrameState(initial_frame);

        context->registerFlushCallback([this](VkFence fence)
                                       {
            if (!current_graph.getNodes().empty() || !context->getTransferTasks().empty())
            {
                ENGINE_LOG_DEBUG("Execution_Engine::flushCallback: Triggering executeGraph via flush callback");
                this->executeGraph(fence);
            } });
    }

public:
    ~Execution_Engine()
    {
        ENGINE_LOG_DEBUG("Execution_Engine::~Execution_Engine: Destroying execution engine");
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

    const Vulkan_Context &getContext() const { return *context; }
    const Vulkan_Network &getNetwork() const { return *network; }
    Compute_Graph &getCurrentGraph() { return current_graph; }

    void enableGraphCaching(bool enable)
    {
        is_graph_cache_enabled = enable;
        if (!enable)
        {
            graph_cache_map.clear();
        }
    }

    bool isGraphCacheEnabled() const { return is_graph_cache_enabled; }

    void compileGraph(const Compute_Graph &raw_graph,
                      std::shared_ptr<GVector> input_buf,
                      std::shared_ptr<GVector> target_buf)
    {
        compiled_input_buffer = input_buf;
        compiled_target_buffer = target_buf;

        optimized_graph = raw_graph;
        Graph_Optimizer::optimize(optimized_graph);
        executor->warmupPipelineCache(optimized_graph);

        cache_manager->savePipelineCache();

        is_graph_compiled = true;
        ENGINE_LOG_DEBUG("Execution_Engine::compileGraph: Successfully compiled graph and persistent pipeline cache updated");
    }

    void executeCompiledGraph(std::shared_ptr<GVector> input_buf, std::shared_ptr<GVector> target_buf)
    {
        if (!is_graph_compiled)
        {
            Logger::logMessage("Execution_Engine::executeCompiledGraph: Attempted to execute uncompiled graph", LOG_ERROR, true);
            throw std::runtime_error("Attempted to execute uncompiled graph");
        }

        if (compiled_input_buffer && input_buf && compiled_input_buffer != input_buf)
        {
            context->copyBuffer(input_buf->getBuffer(), compiled_input_buffer->getBuffer(), input_buf->getSizeBytes());
        }
        if (compiled_target_buffer && target_buf && compiled_target_buffer != target_buf)
        {
            context->copyBuffer(target_buf->getBuffer(), compiled_target_buffer->getBuffer(), target_buf->getSizeBytes());
        }

        std::uint32_t frame = context->getCurrentFrame();
        ENGINE_LOG_DEBUG("Execution_Engine::executeCompiledGraph: Executing compiled compute graph for frame " + std::to_string(frame));

        executor->compileAndExecute(optimized_graph, context->getTransferTasks(), frame);

        context->resetStagingOffset(frame);
        context->clearTransferTasks();

        context->advanceFrame();

        std::uint32_t next_frame = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(next_frame);
        executor->resetFrameState(next_frame);
    }

    void executeGraph(VkFence fence = VK_NULL_HANDLE)
    {
        std::uint32_t frame = context->getCurrentFrame();

        if (current_graph.getNodes().empty() && context->getTransferTasks().empty())
        {
            Logger::logMessage("Execution_Engine::executeGraph: Executing empty compute graph and transfer task queue", LOG_WARNING);
        }

        ENGINE_LOG_DEBUG("Execution_Engine::executeGraph: Executing compute graph for frame " + std::to_string(frame));

        if (is_graph_cache_enabled && !current_graph.getNodes().empty())
        {
            std::size_t signature = computeGraphSignature(current_graph);
            auto it = graph_cache_map.find(signature);
            if (it == graph_cache_map.end())
            {
                it = graph_cache_map.emplace(signature, Graph_Optimizer::buildCachedTemplate(current_graph)).first;
            }

            Compute_Graph opt_graph;
            Graph_Optimizer::applyCachedTemplate(current_graph, it->second, opt_graph);
            executor->compileAndExecute(opt_graph, context->getTransferTasks(), frame, fence);
        }
        else
        {
            Graph_Optimizer::optimize(current_graph);
            executor->compileAndExecute(current_graph, context->getTransferTasks(), frame, fence);
        }

        context->resetStagingOffset(frame);
        context->clearTransferTasks();
        current_graph.clear();

        context->advanceFrame();

        std::uint32_t next_frame = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(next_frame);
        executor->resetFrameState(next_frame);
    }

    void waitIdle() const
    {
        ENGINE_LOG_DEBUG("Execution_Engine::waitIdle: Waiting for device idle");
        if (context)
        {
            vkDeviceWaitIdle(context->getDevice());
        }
        else
        {
            Logger::logMessage("Execution_Engine::waitIdle: Attempted waitIdle on null Vulkan context", LOG_ERROR, true);
            throw std::runtime_error("Attempted waitIdle on null Vulkan context");
        }
    }

    Pipeline_Cache_Manager &getPipelineCacheManager() { return *cache_manager; }
    Graph_Executor &getGraphExecutor() { return *executor; }
    void warmupPipelineCache() { executor->warmupPipelineCache(current_graph); }
    void optimize() { Graph_Optimizer::optimize(current_graph); }
    bool isCompiled() const { return is_graph_compiled; }
    void resetCompiledGraph()
    {
        optimized_graph.clear();
        compiled_input_buffer.reset();
        compiled_target_buffer.reset();
        is_graph_compiled = false;
        graph_cache_map.clear();
    }
};
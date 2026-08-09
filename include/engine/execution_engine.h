#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "graph_executor.h"
#include "helper/logger.h"
#include "vulkan_context.h"
#include "vulkan_network.h"

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
    std::string shader_folder = "compute_shader";

    Compute_Graph current_graph;
    std::unique_ptr<Graph_Executor> executor;

    Execution_Engine()
    {
        ENGINE_LOG_DEBUG("Execution_Engine::Execution_Engine: Initializing execution engine");

        context = std::make_unique<Vulkan_Context>();
        network = std::make_unique<Vulkan_Network>(*context, shader_folder);
        executor = std::make_unique<Graph_Executor>(*context, *network);

        std::uint32_t initial_frame = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(initial_frame);
        executor->resetFrameState(initial_frame);

        context->registerFlushCallback([this]()
                                       {
            if (!current_graph.getNodes().empty() || !context->getTransferTasks().empty())
            {
                ENGINE_LOG_DEBUG("Execution_Engine::flushCallback: Triggering executeGraph via flush callback");
                this->executeGraph();
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

    void executeGraph()
    {
        std::uint32_t frame = context->getCurrentFrame();

        if (current_graph.getNodes().empty() && context->getTransferTasks().empty())
        {
            Logger::logMessage("Execution_Engine::executeGraph: Executing empty compute graph and transfer task queue", LOG_WARNING);
        }

        ENGINE_LOG_DEBUG("Execution_Engine::executeGraph: Executing compute graph for frame " + std::to_string(frame));

        executor->compileAndExecute(current_graph, context->getTransferTasks(), frame);

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
};
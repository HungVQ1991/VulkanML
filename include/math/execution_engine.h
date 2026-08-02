#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

#include "vulkan_context.h"
#include "vulkan_network.h"
#include "logger.h"
#include "compute_graph.h"
#include "graph_executor.h"


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
        context = std::make_unique<Vulkan_Context>();
        network = std::make_unique<Vulkan_Network>(*context, shader_folder);
        executor = std::make_unique<Graph_Executor>(*context, *network);

        std::uint32_t initial_frame = context->getCurrentFrame();
        context->prepareFrame();
        context->cleanGarbage(initial_frame);
        executor->resetFrameState(initial_frame);

        context->registerFlushCallback([this]()
                                       { 
            if (!current_graph.getNodes().empty() || !context->getTransferTasks().empty()) {
                this->executeGraph(); 
            } });
    }

public:
    ~Execution_Engine()
    {
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
        vkDeviceWaitIdle(context->getDevice());
    }
};
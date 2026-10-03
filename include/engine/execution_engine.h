#pragma once

#include <array>
#include <chrono>
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
#include "gpu_vector.h"
#include "graph_executor.h"
#include "graph_optimizer.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"
#include "pipeline_cache_manager.h"
#include "shader_dictionary.h"
#include "vulkan_context.h"
#include "vulkan_network.h"

extern bool is_coop;

enum class Execution_Stage
{
    NONE = 0,
    FORWARD,
    BACKWARD,
    OPTIMIZER,
    BACKWARD_OPTIMIZER
};

class Execution_Engine
{
public:
    struct Loss_Readback_Slot
    {
        std::shared_ptr<gpu::vector> buffer;
        uint32_t valid_tokens = 0;
        size_t total_tokens = 0;
        bool has_pending_read = false;
    };

    struct Adam_Dynamic_Params
    {
        float learning_rate = 0.001f;
        float inv_bc1 = 1.0f;
        float inv_sqrt_bc2 = 1.0f;
        float inv_scale = 1.0f;
    };

private:
    Execution_Stage current_stage = Execution_Stage::NONE;
    double last_submit_time_ms = 0.0;
    double last_fence_wait_ms = 0.0;
    std::unique_ptr<Vulkan_Context> context;
    std::unique_ptr<Vulkan_Network> network;
    std::unique_ptr<Pipeline_Cache_Manager> pipeline_cache_manager;
    std::unique_ptr<Shader_Dictionary> shader_dictionary;
    std::string shader_folder_path = "compute_shader/spv";
    Compute_Graph current_graph;
    std::unique_ptr<Graph_Executor> graph_executor;

    std::unordered_map<size_t, Cached_Graph_Template> cached_graph_templates;
    bool is_graph_cache_enabled = true;
    bool is_static_graph_enabled = false;
    bool is_fused_gemm_adam_enabled = false;
    double last_execution_time_ms = 0.0;
    size_t last_executed_node_count = 0;
    std::array<Execution_Stage, MAX_FRAMES_IN_FLIGHT> frame_stages{Execution_Stage::NONE, Execution_Stage::NONE};
    std::array<Loss_Readback_Slot, MAX_FRAMES_IN_FLIGHT> loss_slots;
    float latest_loss = 0.0f;
    bool is_async_loss_enabled = false;

    void processPendingLossReadback(uint32_t frame_index);
    size_t computeGraphSignature(const Compute_Graph &graph) const;
    void precompileTemplatePipelines(Cached_Graph_Template &_template);

    Execution_Engine();

public:
    ~Execution_Engine();

    Execution_Engine(const Execution_Engine &) = delete;
    Execution_Engine &operator=(const Execution_Engine &) = delete;
    Execution_Engine(Execution_Engine &&) = delete;
    Execution_Engine &operator=(Execution_Engine &&) = delete;

    static Execution_Engine &getInstance();

    void setExecutionStage(Execution_Stage stage) noexcept { current_stage = stage; }
    Execution_Stage getExecutionStage() const noexcept { return current_stage; }
    double getLastSubmitTimeMs() const noexcept { return last_submit_time_ms; }
    double getLastFenceWaitMs() const noexcept { return last_fence_wait_ms; }

    void invalidateGraphCache();
    void invalidateStaticGraph() noexcept;
    void warmCache(const Compute_Graph &_raw_graph);
    void prepareCurrentFrame(Execution_Stage _stage = Execution_Stage::NONE);
    void executeGraph(VkFence _external_fence = VK_NULL_HANDLE, Execution_Stage _stage = Execution_Stage::NONE);
    void executeStaticReplay(VkFence _external_fence = VK_NULL_HANDLE);
    void waitIdle() const;
    void warmupPipelineCache();
    void optimize();

    const std::unordered_map<size_t, Cached_Graph_Template> &getCachedGraphTemplates() const noexcept { return cached_graph_templates; }
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

    void updateDynamicOptimizerParams(float _lr, float _inv_bc1, float _inv_sqrt_bc2, float _inv_scale, uint32_t _frame_index)
    {
        if (graph_executor)
        {
            graph_executor->updateDynamicOptimizerParams(_lr, _inv_bc1, _inv_sqrt_bc2, _inv_scale, _frame_index);
        }
    }

    std::shared_ptr<gpu::vector> getDynamicOptimizerBuffer(uint32_t _frame_index) const noexcept
    {
        return graph_executor ? graph_executor->getDynamicOptimizerBuffer(_frame_index) : nullptr;
    }

    void setShaderFolderPath(const std::string &_path) { shader_folder_path = _path; }
    void setCooperativeMatrixEnabled(bool _enable);
    void setGraphCachingEnabled(bool _is_enabled);
    void enableGraphCaching(bool _is_enabled) { setGraphCachingEnabled(_is_enabled); }

    bool isStaticGraphEnabled() const noexcept { return is_static_graph_enabled; }
    void setStaticGraphEnabled(bool _enable) noexcept;
    void enableStaticGraph(bool _enable = true) noexcept { setStaticGraphEnabled(_enable); }

    bool isFusedGemmAdamEnabled() const noexcept { return is_fused_gemm_adam_enabled; }
    void setFusedGemmAdamEnabled(bool _enable) noexcept;
    void enableFusedGemmAdam(bool _enable = true) noexcept { setFusedGemmAdamEnabled(_enable); }

    double getLastExecutionTimeMs() const noexcept { return last_execution_time_ms; }
    size_t getLastExecutedNodeCount() const noexcept { return last_executed_node_count; }

    Loss_Readback_Slot& getLossSlot(uint32_t frame_index) { return loss_slots[frame_index % MAX_FRAMES_IN_FLIGHT]; }
    float getLatestLoss() const noexcept { return latest_loss; }
    void setLatestLoss(float loss) noexcept { latest_loss = loss; }
    void setAsyncLossEnabled(bool enable) noexcept { is_async_loss_enabled = enable; }
    bool isAsyncLossEnabled() const noexcept { return is_async_loss_enabled; }

    float readPendingLoss(uint32_t frame_index);
};

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "compute_node.h"
#include "gpu_vector.h"
#include "helper/training_profiler.h"
#include "pipeline_cache_manager.h"
#include "shader_dictionary.h"
#include "vulkan_context.h"
#include "vulkan_network.h"

struct Persistent_Descriptor_Entry
{
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    std::vector<VkBuffer> bound_buffers;
    std::vector<uint32_t> bound_binding_indices;
    std::vector<size_t> bound_vector_ids;
    std::vector<VkDeviceSize> bound_buffer_sizes;
    std::vector<std::weak_ptr<gpu::vector>> bound_weak_buffers;
};

struct Baked_Buffer_Record
{
    std::shared_ptr<gpu::vector> buffer_ref;
    VkBuffer handle = VK_NULL_HANDLE;
    size_t vector_id = 0;
    VkDeviceSize size_bytes = 0;
};

class Graph_Executor
{
private:
    const Vulkan_Context &context;
    const Vulkan_Network &network;
    Pipeline_Cache_Manager &pipeline_cache_manager;
    const Shader_Dictionary &shader_dictionary;

    mutable std::vector<std::string> printed_terminal_shader_chains;
    mutable std::mutex printed_chains_mutex;

    VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer transfer_command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer static_command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer epilogue_command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pools[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkDescriptorPool> overflow_descriptor_pools[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore external_fence_semaphores[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};

    struct Static_Node_Meta
    {
        Compute_Pipeline pipeline_id = Compute_Pipeline::ADD;
        bool is_fused = false;
        uint32_t wg_x = 1;
        uint32_t wg_y = 1;
        uint32_t wg_z = 1;
        uint32_t spec_constant_u32 = 0;
    };

    bool is_static_baked[MAX_FRAMES_IN_FLIGHT]{false, false};
    size_t static_graph_signatures[MAX_FRAMES_IN_FLIGHT]{0, 0};
    size_t static_split_indices[MAX_FRAMES_IN_FLIGHT]{0, 0};
    std::array<std::vector<VkBuffer>, MAX_FRAMES_IN_FLIGHT> baked_buffer_handles{};
    std::array<std::vector<Baked_Buffer_Record>, MAX_FRAMES_IN_FLIGHT> baked_buffer_records{};
    std::array<std::vector<std::shared_ptr<gpu::vector>>, MAX_FRAMES_IN_FLIGHT> unique_baked_buffers{};
    mutable std::array<uint64_t, MAX_FRAMES_IN_FLIGHT> static_validated_buffer_epoch{0, 0};
    std::array<std::vector<std::vector<uint8_t>>, MAX_FRAMES_IN_FLIGHT> static_baked_pc_snapshots{};
    std::array<std::vector<Static_Node_Meta>, MAX_FRAMES_IN_FLIGHT> static_baked_node_meta{};
    std::array<uint64_t, MAX_FRAMES_IN_FLIGHT> static_bake_generations{0, 0};
    uint64_t next_bake_generation = 1;

    static constexpr uint32_t INITIAL_TIMESTAMP_QUERY_CAPACITY = 4096u;
    bool is_gpu_profiling_enabled = false;
    VkQueryPool timestamp_query_pools[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    uint32_t timestamp_query_capacities[MAX_FRAMES_IN_FLIGHT]{0u, 0u};

    struct Recorded_Timestamp_Entry
    {
        std::string label;
        uint32_t start_query = 0;
        uint32_t end_query = 0;
    };

    std::vector<Recorded_Timestamp_Entry> frame_timestamp_entries[MAX_FRAMES_IN_FLIGHT];
    uint64_t frame_barrier_count[MAX_FRAMES_IN_FLIGHT]{0, 0};
    bool has_pending_timestamps[MAX_FRAMES_IN_FLIGHT]{false, false};
    std::vector<Recorded_Timestamp_Entry> static_recorded_entries[MAX_FRAMES_IN_FLIGHT];
    uint64_t static_recorded_barrier_count[MAX_FRAMES_IN_FLIGHT]{0, 0};
    uint32_t static_next_query_index[MAX_FRAMES_IN_FLIGHT]{0u, 0u};
    bool static_baked_with_profiling[MAX_FRAMES_IN_FLIGHT]{false, false};

    mutable std::mutex gpu_profile_mutex;
    std::unordered_map<std::string, Gpu_Kernel_Stat> aggregated_kernel_stats;
    double aggregated_total_kernel_ms = 0.0;
    double aggregated_total_span_ms = 0.0;
    uint64_t aggregated_total_dispatches = 0;
    uint64_t aggregated_total_barriers = 0;
    uint64_t aggregated_profiled_submits = 0;

    std::vector<Persistent_Descriptor_Entry> persistent_descriptor_caches[MAX_FRAMES_IN_FLIGHT];
    std::vector<std::vector<Persistent_Descriptor_Entry>> fallback_descriptor_caches[MAX_FRAMES_IN_FLIGHT];
    std::array<std::shared_ptr<gpu::vector>, MAX_FRAMES_IN_FLIGHT> dynamic_optimizer_buffers;
    std::vector<std::shared_ptr<gpu::vector>> dynamic_optimizer_param_pool[MAX_FRAMES_IN_FLIGHT];
    std::shared_ptr<gpu::vector> dummy_null_buffer;

    VkDescriptorSet allocateDescriptorSet(uint32_t _frame_index);

    static bool isDynamicNode(const Compute_Node &_node) noexcept;

    mutable std::vector<std::shared_ptr<gpu::vector>> shared_fused_buffers;
    mutable std::vector<uint32_t> shared_external_buffer_indices;

    static bool isBuffersMatching(const Persistent_Descriptor_Entry &_entry,
                                  const std::vector<uint32_t> &_binding_indices,
                                  const std::vector<std::shared_ptr<gpu::vector>> &_buffers);

    static bool isBuffersMatching(const Persistent_Descriptor_Entry &_entry,
                                  const std::vector<std::shared_ptr<gpu::vector>> &_buffers);

    static std::string replacePlaceholders(
        std::string _text,
        const std::vector<std::string> &_input_identifiers,
        const std::vector<bool> &_is_input_register_flags,
        const std::vector<std::string> &_output_identifiers,
        const std::vector<bool> &_is_output_register_flags,
        const std::vector<uint32_t> &_output_buffer_indices,
        const std::unordered_set<uint32_t> &_external_buffer_indices_set,
        uint32_t _push_constants_word_offset,
        const std::vector<std::shared_ptr<gpu::vector>> &_node_buffers = {});

    std::shared_ptr<gpu::vector> resolveNodeBuffer(const Compute_Node &_node, size_t _b_idx, uint32_t _frame_index) const;
    void recordTransferTasks(VkCommandBuffer _command_buffer, const std::vector<Buffer_Transfer_Task> &_transfer_tasks) const;
    std::vector<std::shared_ptr<gpu::vector>> getNodeWrittenBuffers(const Compute_Node &_node) const;
    void insertBufferMemoryBarriers(VkCommandBuffer _command_buffer, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;
    void insertComputeMemoryBarrier(VkCommandBuffer _command_buffer) const;
    void insertInterSubmissionComputeBarrier(VkCommandBuffer _command_buffer) const;
    void executeFallbackNode(VkCommandBuffer _command_buffer, const Compute_Node &_node, size_t _node_index, uint32_t _frame_index);
    bool isGpuProfilingActive() const noexcept { return is_gpu_profiling_enabled && context.isTimestampQuerySupported(); }
    bool ensureTimestampQueryPool(uint32_t _frame_index, uint32_t _required_queries, bool _allow_grow);
    static std::string makeNodeProfileLabel(const Compute_Node &_node, bool _is_fused_successful);
    void initializeResources();
    void cleanupResources() noexcept;
    void updateDescriptorSet(VkDescriptorSet _descriptor_set, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;
    void updateDescriptorSet(VkDescriptorSet _descriptor_set,
                             const std::vector<uint32_t> &_binding_indices,
                             const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;
    void pushDescriptors(VkCommandBuffer _command_buffer, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;
    void pushDescriptors(VkCommandBuffer _command_buffer,
                         const std::vector<uint32_t> &_binding_indices,
                         const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;

public:
    Graph_Executor(const Vulkan_Context &_context,
                   const Vulkan_Network &_network,
                   Pipeline_Cache_Manager &_pipeline_cache_manager,
                   const Shader_Dictionary &_shader_dictionary);
    ~Graph_Executor();

    Graph_Executor(const Graph_Executor &) = delete;
    Graph_Executor &operator=(const Graph_Executor &) = delete;
    Graph_Executor(Graph_Executor &&) = delete;
    Graph_Executor &operator=(Graph_Executor &&) = delete;

    struct Adam_Dynamic_Params
    {
        float learning_rate = 0.001f;
        float inv_bc1 = 1.0f;
        float inv_sqrt_bc2 = 1.0f;
        float inv_scale = 1.0f;
        uint32_t overflow_detected = 0u;
    };

private:
    std::vector<Adam_Dynamic_Params> dynamic_optimizer_pool_values[MAX_FRAMES_IN_FLIGHT];
    size_t dynamic_optimizer_pool_used[MAX_FRAMES_IN_FLIGHT]{0, 0};
    bool dynamic_pool_dirty[MAX_FRAMES_IN_FLIGHT]{false, false};

public:
    void updateDynamicOptimizerParams(float _lr, float _inv_bc1, float _inv_sqrt_bc2, float _inv_scale, uint32_t _frame_index);
    std::shared_ptr<gpu::vector> allocateDynamicOptimizerParamsBuffer(float _lr, float _inv_bc1, float _inv_sqrt_bc2, float _inv_scale, uint32_t _frame_index);
    std::shared_ptr<gpu::vector> getDynamicOptimizerBuffer(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? dynamic_optimizer_buffers[_frame_index] : nullptr;
    }

    void getExternalBufferIndices(const Compute_Node &_node, std::vector<uint32_t> &_output_indices) const;
    std::string generateFusedGlsl(const Compute_Node &_node) const;
    std::string emitPointwiseChainGlsl(const Compute_Node &_node) const;
    std::string emitCoopMatEpilogueGlsl(const Compute_Node &_node) const;
    std::string emitRowReductionGlsl(const Compute_Node &_node) const;
    std::string emitHorizontalAdamGlsl(const Compute_Node &_node) const;
    void invalidate();
    void invalidateStaticGraph() noexcept;
    void resetFrameState(uint32_t _frame_index);

    void recordComputeNodes(VkCommandBuffer _command_buffer,
                            const std::vector<Compute_Node> &_nodes,
                            size_t _start_index,
                            size_t _end_index,
                            uint32_t _frame_index,
                            size_t _cache_index_offset = 0,
                            uint32_t *_query_cursor = nullptr,
                            std::vector<Recorded_Timestamp_Entry> *_entries_out = nullptr,
                            uint64_t *_barriers_out = nullptr);

    void setGpuProfilingEnabled(bool _enable);
    bool isGpuProfilingEnabled() const noexcept { return isGpuProfilingActive(); }
    void resolveFrameTimestamps(uint32_t _frame_index);
    void resolveAllPendingTimestamps();
    Gpu_Profile_Summary getGpuProfileSummary() const;
    void resetGpuProfileStats();
    std::string formatGpuProfileTable(size_t _top_k = 20, size_t _batch_divisor = 1) const;
    void logGpuProfileTable(size_t _top_k = 20, size_t _batch_divisor = 1, Log_Level _level = Log_Level::LOG_INFO, bool _print_to_console = true) const;
    void printGpuProfileTable(size_t _top_k = 20, size_t _batch_divisor = 1) const;
    void printGpuProfileTable(std::ostream &_os, size_t _top_k = 20, size_t _batch_divisor = 1) const;

    void compileAndExecute(const Compute_Graph &_graph,
                           const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                           uint32_t _frame_index,
                           VkFence _external_fence = VK_NULL_HANDLE);

    void bakeStaticGraph(const Compute_Graph &_graph, uint32_t _frame_index, size_t _signature);
    bool isStaticGraphBuffersMatching(const Compute_Graph &_graph, uint32_t _frame_index) const;

    void executeStaticGraph(const Compute_Graph &_graph,
                            const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                            uint32_t _frame_index,
                            VkFence _external_fence = VK_NULL_HANDLE,
                            bool _skip_buffer_match_check = false);

    void executeStaticGraphReplay(const Compute_Graph &_optimizer_graph,
                                  const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                                  uint32_t _frame_index,
                                  VkFence _external_fence = VK_NULL_HANDLE);

    void warmupPipelineCache(Compute_Graph &_graph);

    const Pipeline_Cache_Manager &getPipelineCacheManager() const noexcept { return pipeline_cache_manager; }
    Pipeline_Cache_Manager &getPipelineCacheManager() noexcept { return pipeline_cache_manager; }
    const Shader_Dictionary &getShaderDictionary() const noexcept { return shader_dictionary; }
    const Vulkan_Network &getNetwork() const noexcept { return network; }
    const Vulkan_Context &getContext() const noexcept { return context; }
    std::vector<std::string> getPrintedTerminalShaderChains() const
    {
        std::lock_guard<std::mutex> lock(printed_chains_mutex);
        return printed_terminal_shader_chains;
    }

    VkCommandBuffer getCommandBuffer(uint32_t _frame_index) const;
    VkDescriptorPool getDescriptorPool(uint32_t _frame_index) const;
    VkCommandBuffer getTransferCommandBuffer(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? transfer_command_buffers[_frame_index] : VK_NULL_HANDLE;
    }
    VkCommandBuffer getStaticCommandBuffer(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_command_buffers[_frame_index] : VK_NULL_HANDLE;
    }
    VkCommandBuffer getEpilogueCommandBuffer(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? epilogue_command_buffers[_frame_index] : VK_NULL_HANDLE;
    }

    bool isStaticBaked(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? is_static_baked[_frame_index] : false;
    }
    bool isStaticBakedWithProfiling(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_baked_with_profiling[_frame_index] : false;
    }
    uint64_t getStaticBakeGeneration(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_bake_generations[_frame_index] : 0;
    }
    bool areStaticBakedBuffersValid(uint32_t _frame_index) const noexcept;
    size_t getStaticGraphSignature(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_graph_signatures[_frame_index] : 0;
    }
    size_t getStaticSplitIndex(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_split_indices[_frame_index] : 0;
    }

    void setPrintedTerminalShaderChains(const std::vector<std::string> &_chains)
    {
        std::lock_guard<std::mutex> lock(printed_chains_mutex);
        printed_terminal_shader_chains = _chains;
    }
    void setCommandBuffer(uint32_t _frame_index, VkCommandBuffer _command_buffer);
    void setDescriptorPool(uint32_t _frame_index, VkDescriptorPool _pool);
};

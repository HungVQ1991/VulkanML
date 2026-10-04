#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_graph.h"
#include "compute_node.h"
#include "gpu_vector.h"
#include "pipeline_cache_manager.h"
#include "shader_dictionary.h"
#include "vulkan_context.h"
#include "vulkan_network.h"

extern bool is_coop;

struct Persistent_Descriptor_Entry
{
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    std::vector<VkBuffer> bound_buffers;
    std::vector<uint32_t> bound_binding_indices;

    const std::vector<VkBuffer> &getBoundBuffers() const noexcept { return bound_buffers; }
    const std::vector<uint32_t> &getBoundBindingIndices() const noexcept { return bound_binding_indices; }
    VkDescriptorSet getDescriptorSet() const noexcept { return descriptor_set; }

    void setBoundBuffers(const std::vector<VkBuffer> &_buffers) { bound_buffers = _buffers; }
    void setBoundBindingIndices(const std::vector<uint32_t> &_indices) { bound_binding_indices = _indices; }
    void setDescriptorSet(VkDescriptorSet _set) noexcept { descriptor_set = _set; }
};

class Graph_Executor
{
private:
    const Vulkan_Context &context;
    const Vulkan_Network &network;
    Pipeline_Cache_Manager &pipeline_cache_manager;
    const Shader_Dictionary &shader_dictionary;

    mutable std::vector<std::string> printed_terminal_shader_chains;

    VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer transfer_command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer static_command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer epilogue_command_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pools[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};

    bool is_static_baked[MAX_FRAMES_IN_FLIGHT]{false, false};
    size_t static_graph_signatures[MAX_FRAMES_IN_FLIGHT]{0, 0};
    size_t static_split_indices[MAX_FRAMES_IN_FLIGHT]{0, 0};
    std::array<std::vector<VkBuffer>, MAX_FRAMES_IN_FLIGHT> baked_buffer_handles{};

    std::vector<Persistent_Descriptor_Entry> persistent_descriptor_caches[MAX_FRAMES_IN_FLIGHT];
    std::vector<std::vector<Persistent_Descriptor_Entry>> fallback_descriptor_caches[MAX_FRAMES_IN_FLIGHT];
    std::array<std::shared_ptr<gpu::vector>, MAX_FRAMES_IN_FLIGHT> dynamic_optimizer_buffers;

    static bool isDynamicNode(const Compute_Node &_node) noexcept;

    mutable std::vector<VkDescriptorBufferInfo> shared_descriptor_buffer_informations;
    mutable std::vector<VkWriteDescriptorSet> shared_write_descriptor_sets;
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

    std::vector<std::shared_ptr<gpu::vector>> getNodeWrittenBuffers(const Compute_Node &_node) const;
    void insertBufferMemoryBarriers(VkCommandBuffer _command_buffer, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;
    void insertComputeMemoryBarrier(VkCommandBuffer _command_buffer) const;
    void executeFallbackNode(VkCommandBuffer _command_buffer, const Compute_Node &_node, size_t _node_index, uint32_t _frame_index);
    void initializeResources();
    void updateDescriptorSet(VkDescriptorSet _descriptor_set, const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;
    void updateDescriptorSet(VkDescriptorSet _descriptor_set,
                             const std::vector<uint32_t> &_binding_indices,
                             const std::vector<std::shared_ptr<gpu::vector>> &_buffers) const;

public:
    Graph_Executor(const Vulkan_Context &_context,
                   const Vulkan_Network &_network,
                   Pipeline_Cache_Manager &_pipeline_cache_manager,
                   const Shader_Dictionary &_shader_dictionary);
    ~Graph_Executor();

    struct Adam_Dynamic_Params
    {
        float learning_rate = 0.001f;
        float inv_bc1 = 1.0f;
        float inv_sqrt_bc2 = 1.0f;
        float inv_scale = 1.0f;
    };

    void updateDynamicOptimizerParams(float _lr, float _inv_bc1, float _inv_sqrt_bc2, float _inv_scale, uint32_t _frame_index);
    std::shared_ptr<gpu::vector> getDynamicOptimizerBuffer(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? dynamic_optimizer_buffers[_frame_index] : nullptr;
    }

    void getExternalBufferIndices(const Compute_Node &_node, std::vector<uint32_t> &_output_indices) const;
    std::string generateFusedGlsl(const Compute_Node &_node) const;
    void invalidate();
    void invalidateStaticGraph() noexcept;
    void resetFrameState(uint32_t _frame_index);

    void recordComputeNodes(VkCommandBuffer _command_buffer,
                            const std::vector<Compute_Node> &_nodes,
                            size_t _start_index,
                            size_t _end_index,
                            uint32_t _frame_index,
                            size_t _cache_index_offset = 0);

    void compileAndExecute(const Compute_Graph &_graph,
                           const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                           uint32_t _frame_index,
                           VkFence _external_fence = VK_NULL_HANDLE);

    void bakeStaticGraph(const Compute_Graph &_graph, uint32_t _frame_index, size_t _signature);
    bool isStaticGraphBuffersMatching(const Compute_Graph &_graph, uint32_t _frame_index) const;

    void executeStaticGraph(const Compute_Graph &_graph,
                            const std::vector<Buffer_Transfer_Task> &_transfer_tasks,
                            uint32_t _frame_index,
                            VkFence _external_fence = VK_NULL_HANDLE);

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
    const std::vector<std::string> &getPrintedTerminalShaderChains() const noexcept { return printed_terminal_shader_chains; }

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
    size_t getStaticGraphSignature(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_graph_signatures[_frame_index] : 0;
    }
    size_t getStaticSplitIndex(uint32_t _frame_index) const noexcept
    {
        return (_frame_index < MAX_FRAMES_IN_FLIGHT) ? static_split_indices[_frame_index] : 0;
    }

    void setPrintedTerminalShaderChains(const std::vector<std::string> &_chains) { printed_terminal_shader_chains = _chains; }
    void setCommandBuffer(uint32_t _frame_index, VkCommandBuffer _command_buffer) noexcept { if (_frame_index < MAX_FRAMES_IN_FLIGHT) command_buffers[_frame_index] = _command_buffer; }
    void setDescriptorPool(uint32_t _frame_index, VkDescriptorPool _pool) noexcept { if (_frame_index < MAX_FRAMES_IN_FLIGHT) descriptor_pools[_frame_index] = _pool; }
};

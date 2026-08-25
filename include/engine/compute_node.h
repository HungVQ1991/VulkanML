#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "gpu_vector.h"
#include "vulkan_network.h"

struct Fused_Operation
{
    Compute_Pipeline pipeline_id = Compute_Pipeline::ADD;
    std::uint32_t push_constants_offset = 0;
    std::uint32_t push_constants_size = 0;
    std::vector<std::uint32_t> input_buffer_indices;
    std::vector<std::uint32_t> output_buffer_indices;
    std::uint32_t workgroup_count_x = 1;
    std::uint32_t workgroup_count_y = 1;
    std::uint32_t workgroup_count_z = 1;
};

struct Compute_Node
{
    Compute_Pipeline pipeline_id = Compute_Pipeline::ADD;
    std::vector<std::shared_ptr<gpu::vector>> buffers;
    std::vector<std::uint8_t> push_constants_data;
    std::unordered_set<std::uint32_t> external_output_indices;

    std::uint32_t workgroup_count_x = 1;
    std::uint32_t workgroup_count_y = 1;
    std::uint32_t workgroup_count_z = 1;

    bool is_fused = false;
    bool is_barrier_required_after = false;
    std::vector<Fused_Operation> fused_operations;

    std::string fused_glsl_code;
    std::vector<std::uint32_t> cached_external_buffer_indices;
    VkPipeline cached_pipeline = VK_NULL_HANDLE;
};
#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include "vulkan_network.h"
#include "gpu_vector.h"

struct Fused_Operation
{
    Compute_Pipeline pipeline_id;
    std::uint32_t pc_offset;
    std::uint32_t pc_size;
    std::vector<std::uint32_t> input_buffer_indices;  
    std::vector<std::uint32_t> output_buffer_indices;
    std::uint32_t group_x = 1;
    std::uint32_t group_y = 1;
    std::uint32_t group_z = 1;
};

struct Compute_Node
{
    Compute_Pipeline pipeline_id;
    std::vector<std::shared_ptr<GVector>> buffers;
    std::vector<std::uint8_t> push_constants_data;
    std::unordered_set<std::uint32_t> external_output_indices;

    std::uint32_t group_x;
    std::uint32_t group_y;
    std::uint32_t group_z;

    bool is_fused = false;
    std::vector<Fused_Operation> fused_operations;
};
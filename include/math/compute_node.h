#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "vulkan_network.h"
#include "gpu_vector.h"

struct Compute_Node
{
    Compute_Pipeline pipeline_id;
    std::vector<std::shared_ptr<GVector>> buffers;
    std::vector<std::uint8_t> push_constants_data;

    std::uint32_t group_x;
    std::uint32_t group_y;
    std::uint32_t group_z;
};
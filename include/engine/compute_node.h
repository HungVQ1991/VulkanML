#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "fusion_ir.h"
#include "gpu_vector.h"
#include "vulkan_network.h"

struct Fused_Operation
{
    Compute_Pipeline pipeline_id = Compute_Pipeline::ADD;
    uint32_t push_constants_offset = 0;
    uint32_t push_constants_size = 0;
    std::vector<uint32_t> input_buffer_indices;
    std::vector<uint32_t> output_buffer_indices;
    uint32_t workgroup_count_x = 1;
    uint32_t workgroup_count_y = 1;
    uint32_t workgroup_count_z = 1;
    uint32_t spec_constant_u32 = 0;
};

struct Compute_Node
{
    Compute_Pipeline pipeline_id = Compute_Pipeline::ADD;
    std::vector<std::shared_ptr<gpu::vector>> buffers;
    std::vector<std::uint8_t> push_constants_data;
    std::unordered_set<uint32_t> external_output_indices;

    uint32_t workgroup_count_x = 1;
    uint32_t workgroup_count_y = 1;
    uint32_t workgroup_count_z = 1;
    uint32_t spec_constant_u32 = 0;

    bool is_fused = false;
    bool is_barrier_required_after = false;
    std::vector<Fused_Operation> fused_operations;

    std::string fused_glsl_code;
    std::vector<uint32_t> cached_external_buffer_indices;
    VkPipeline cached_pipeline = VK_NULL_HANDLE;
    std::shared_ptr<Fused_Subgraph_Plan> fusion_plan;
};

template <typename T>
inline T readPushConstant(const std::vector<std::uint8_t> &_data, size_t _byte_offset) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    if (_byte_offset + sizeof(T) <= _data.size())
    {
        std::memcpy(&value, _data.data() + _byte_offset, sizeof(T));
    }
    return value;
}

template <typename T>
inline T readPushConstantWord(const std::vector<std::uint8_t> &_data, size_t _word_index, size_t _base_byte_offset = 0) noexcept
{
    return readPushConstant<T>(_data, _base_byte_offset + _word_index * sizeof(T));
}

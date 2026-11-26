#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vulkan_network.h"

enum class Fused_Anchor_Role : uint8_t
{
    NONE = 0,
    POINTWISE_CHAIN,
    GEMM_COOPMAT,
    ROW_REDUCTION,
    HORIZONTAL_MULTI_LANE
};

enum class Pointwise_Op_Kind : uint8_t
{
    NONE = 0,
    ADD,
    SUB,
    MUL,
    DIV,
    MUL_SCALAR,
    ADD_SCALAR,
    RELU,
    SILU,
    GELU,
    SIGMOID,
    TANH
};

enum class Index_Map_Kind : uint8_t
{
    FLAT_1D = 0,
    BROADCAST_ROW_1D,
    ROW_STRIDED_2D
};

struct Fused_Op_Step
{
    Compute_Pipeline pipeline_id = Compute_Pipeline::ADD;
    Pointwise_Op_Kind op_kind = Pointwise_Op_Kind::NONE;
    Index_Map_Kind rhs_index_mode = Index_Map_Kind::FLAT_1D;
    std::vector<uint32_t> input_buffer_indices;
    uint32_t output_buffer_index = 0;
    uint32_t pc_offset_bytes = 0;
    uint32_t pc_size_bytes = 0;
    uint32_t num_columns = 0;
};

struct Horizontal_Lane_Spec
{
    uint32_t param_buf_idx = 0;
    uint32_t grad_buf_idx = 0;
    uint32_t m_buf_idx = 0;
    uint32_t v_buf_idx = 0;
    uint32_t param_fp16_buf_idx = 0;
    uint32_t element_count = 0;
    uint32_t wg_end = 0;
    float weight_decay = 0.0f;
};

struct Fused_Subgraph_Plan
{
    Fused_Anchor_Role anchor_role = Fused_Anchor_Role::NONE;
    std::string canonical_key;
    std::vector<uint32_t> spec_constants;
    uint32_t shared_dyn_params_binding = UINT32_MAX;
    std::vector<Fused_Op_Step> prologue_steps;
    std::vector<Fused_Op_Step> epilogue_steps;
    std::vector<uint32_t> materialized_buffer_indices;
    std::vector<Horizontal_Lane_Spec> horizontal_lanes;
    bool is_fp16 = false;
    bool use_vec4 = false;
    uint32_t local_size_x = 256;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
};

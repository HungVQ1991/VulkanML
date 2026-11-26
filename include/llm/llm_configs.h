#pragma once

#include <cstddef>
#include <cstdint>

#include "math/tensor.h"

class Execution_Engine;

namespace ops::llm
{

enum class RoPE_Mode : uint32_t
{
    STANDARD = 0,
    FUSED_PERMUTE = 1,
    FUSED_UNPERMUTE = 2,
    BSHD_INPLACE = 3
};

enum class RoPE_Direction : int32_t
{
    FORWARD = 1,
    BACKWARD = -1
};

struct YaRN_Config
{
    bool enabled = false;
    float factor = 1.0f;
    uint32_t original_max_position_embeddings = 32768u;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float attention_factor = 0.0f;
};

struct RoPE_Config
{
    uint32_t seq_len = 0;
    uint32_t head_dim = 0;
    RoPE_Direction direction = RoPE_Direction::FORWARD;
    float base = 10000.0f;
    uint32_t num_heads = 1;
    RoPE_Mode mode = RoPE_Mode::STANDARD;
    uint32_t position_offset = 0;
    uint32_t rotary_dim = 0;
    float freq_scale = 1.0f;
    Execution_Engine *engine = nullptr;
    const Tensor *position_ids = nullptr;
    YaRN_Config yarn{};
};

enum class Attention_Mask_Mode : uint32_t
{
    NONE = 0,
    CAUSAL = 1,
    DENSE_BOOL = 2
};

struct Attention_Config
{
    uint32_t num_heads = 1;
    uint32_t num_kv_heads = 0;
    uint32_t seq_len = 0;
    uint32_t head_dim = 0;
    bool is_causal = false;
    float scale = 0.0f;
    uint32_t kv_seq_len = 0;
    uint32_t causal_offset = 0;
    bool is_bshd = false;
    Execution_Engine *engine = nullptr;
    Attention_Mask_Mode mask_mode = Attention_Mask_Mode::NONE;
    const Tensor *attn_mask = nullptr;
    const Tensor *kv_index = nullptr;
};

}

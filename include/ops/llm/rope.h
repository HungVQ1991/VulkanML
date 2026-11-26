#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include "llm/llm_configs.h"
#include "math/tensor.h"

namespace ops::llm
{

std::pair<std::vector<float>, float> computeRoPEInvFreq(uint32_t rotary_dim, float base, float freq_scale, const YaRN_Config &yarn);

void applyRoPE(const Tensor &input, Tensor &output, const RoPE_Config &config);

void applyRoPE(const Tensor &input, Tensor &output, uint32_t seq_len, uint32_t head_dim,
               int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0,
               uint32_t position_offset = 0);

void qkvSplitRopeForward(const Tensor &qkv, Tensor &q_out, Tensor &k_out, Tensor &v_out,
                         uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t num_kv_heads,
                         uint32_t head_dim, float rope_base = 10000.0f, bool output_bhsd = false,
                         uint32_t position_offset = 0, Execution_Engine *engine = nullptr);

void qkvPackUnropeBackward(const Tensor &dq, const Tensor &dk, const Tensor &dv, Tensor &dqkv_out,
                           uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t num_kv_heads,
                           uint32_t head_dim, float rope_base = 10000.0f, bool input_bhsd = false,
                           uint32_t position_offset = 0, Execution_Engine *engine = nullptr);

}

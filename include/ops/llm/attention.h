#pragma once

#include <cstddef>
#include <cstdint>
#include "math/tensor.h"

namespace ops::llm
{

void flashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                           uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                           bool is_causal = false, float scale = 0.0f, Tensor *l_stats = nullptr);

void flashAttentionBackward(const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &o, const Tensor &do_grad,
                            Tensor &dq, Tensor &dk, Tensor &dv,
                            uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                            bool is_causal = false, float scale = 0.0f, const Tensor *l_stats = nullptr);

void singleTokenAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                                 size_t num_heads, size_t head_dim, size_t total_seq_len);

} // namespace ops::llm

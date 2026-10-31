#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::llm
{

void applyRoPE(const Tensor &input, Tensor &output, uint32_t seq_len, uint32_t head_dim,
               int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0);

} // namespace ops::llm

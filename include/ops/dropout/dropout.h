#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::dropout
{

void dropout(const Tensor &input, Tensor &output, Tensor &mask, float dropout_rate, bool is_training, uint64_t seed = 0);

void dropoutBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient, float dropout_rate);

} // namespace ops::dropout

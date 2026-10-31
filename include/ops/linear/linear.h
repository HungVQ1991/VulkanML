#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::linear
{

void linearForward(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output);

void linearBackwardInput(const Tensor &output_gradient, const Tensor &weights, Tensor &input_gradient);

void linearBackwardWeightBias(const Tensor &input, const Tensor &output_gradient,
                             Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate = false);

} // namespace ops::linear

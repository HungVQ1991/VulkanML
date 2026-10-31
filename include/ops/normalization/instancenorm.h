#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::normalization
{

void instanceNorm2dForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                           Tensor &saved_mean, Tensor &saved_inv_std,
                           Tensor &normalized_input, Tensor &output,
                           uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                           float epsilon);

void instanceNorm2dBackward(const Tensor &output_gradient, const Tensor &normalized_input,
                            const Tensor &gamma, const Tensor &saved_inv_std,
                            Tensor &gamma_gradient, Tensor &beta_gradient,
                            Tensor &input_gradient,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon);

} // namespace ops::normalization

namespace ops::norm
{
using ops::normalization::instanceNorm2dForward;
using ops::normalization::instanceNorm2dBackward;
} // namespace ops::norm

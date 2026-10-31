#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::vision
{

void conv2d(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output,
            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
            uint32_t output_channels, uint32_t kernel_size,
            uint32_t stride, uint32_t padding,
            Tensor *scratch = nullptr);

void conv2dBackwardInput(const Tensor &output_gradient, const Tensor &weights, Tensor &input_gradient,
                         uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                         uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                         uint32_t kernel_size, uint32_t stride, uint32_t padding);

void conv2dBackwardWeight(const Tensor &input, const Tensor &output_gradient,
                          Tensor &weight_gradient, Tensor &bias_gradient,
                          uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_size, uint32_t stride, uint32_t padding,
                          Tensor *im2col_scratch = nullptr);

} // namespace ops::vision

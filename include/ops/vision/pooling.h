#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::vision
{

void maxpool2d(const Tensor &input, Tensor &output, Tensor &output_mask,
               uint32_t input_height, uint32_t input_width, uint32_t channels,
               uint32_t kernel_size, uint32_t stride, uint32_t padding);

void maxpool2dBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient,
                       uint32_t input_height, uint32_t input_width, uint32_t channels,
                       uint32_t output_height, uint32_t output_width,
                       uint32_t kernel_size, uint32_t stride, uint32_t padding);

void avgpool2d(const Tensor &input, Tensor &output,
               uint32_t input_height, uint32_t input_width, uint32_t channels,
               uint32_t kernel_size, uint32_t stride, uint32_t padding);

void avgpool2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                       uint32_t input_height, uint32_t input_width, uint32_t channels,
                       uint32_t output_height, uint32_t output_width,
                       uint32_t kernel_size, uint32_t stride, uint32_t padding);

void globalAvgPool2d(const Tensor &input, Tensor &output,
                     uint32_t input_height, uint32_t input_width, uint32_t channels);

void globalAvgPool2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t channels);

void upsample2d(const Tensor &input, Tensor &output,
                uint32_t input_height, uint32_t input_width, uint32_t channels,
                uint32_t scale_factor);

void upsample2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                        uint32_t input_height, uint32_t input_width, uint32_t channels,
                        uint32_t scale_factor);

} // namespace ops::vision

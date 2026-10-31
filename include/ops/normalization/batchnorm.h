#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::normalization
{

void batchNormForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                      Tensor &running_mean, Tensor &running_variance,
                      Tensor &batch_mean, Tensor &batch_variance,
                      Tensor &normalized_input, Tensor &output,
                      float epsilon, float momentum, bool is_training);

void batchNormBackward(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                       const Tensor &batch_variance, const Tensor &normalized_input,
                       Tensor &gamma_gradient, Tensor &beta_gradient,
                       Tensor &input_gradient, float epsilon,
                       bool is_training = true, const Tensor &running_variance = Tensor());

void batchNorm2dForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                        Tensor &running_mean, Tensor &running_variance,
                        Tensor &batch_mean, Tensor &batch_variance,
                        Tensor &normalized_input, Tensor &output,
                        uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                        float epsilon, float momentum, bool is_training);

void batchNorm2dBackward(const Tensor &output_gradient, const Tensor &gamma,
                         const Tensor &batch_variance, const Tensor &normalized_input,
                         Tensor &gamma_gradient, Tensor &beta_gradient,
                         Tensor &input_gradient,
                         uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                         float epsilon,
                         bool is_training = true, const Tensor &running_variance = Tensor());

} // namespace ops::normalization

namespace ops::norm
{
using ops::normalization::batchNormForward;
using ops::normalization::batchNormBackward;
using ops::normalization::batchNorm2dForward;
using ops::normalization::batchNorm2dBackward;
} // namespace ops::norm

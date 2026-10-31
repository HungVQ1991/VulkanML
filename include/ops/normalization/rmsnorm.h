#pragma once

#include "math/tensor.h"

namespace ops::normalization
{

void rmsNormForward(const Tensor &input, const Tensor &gamma, Tensor &inv_rms,
                    Tensor &output, float epsilon = 1e-5f);

void rmsNormBackward(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                     const Tensor &inv_rms, Tensor &gamma_gradient,
                     Tensor &input_gradient, bool accumulate_gamma = false);

} // namespace ops::normalization

namespace ops::norm
{
using ops::normalization::rmsNormForward;
using ops::normalization::rmsNormBackward;
} // namespace ops::norm

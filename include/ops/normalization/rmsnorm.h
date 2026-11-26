#pragma once

#include "math/tensor.h"

namespace ops::normalization
{

void rmsNormForward(const Tensor &input, const Tensor &gamma, Tensor &inv_rms,
                    Tensor &output, float epsilon = 1e-5f);

void rmsNormBackward(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                     const Tensor &inv_rms, Tensor &gamma_gradient,
                     Tensor &input_gradient, bool accumulate_gamma = false);

void residualScaleRmsNormForward(const Tensor &residual, const Tensor &branch, const Tensor &gamma,
                                 float scale, Tensor &out_residual, Tensor &inv_rms,
                                 Tensor &out_norm, float epsilon = 1e-5f);

void rmsNormBackwardResidual(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                             const Tensor &inv_rms, const Tensor &residual_grad,
                             Tensor &gamma_gradient, Tensor &out_sum,
                             Tensor *out_scaled = nullptr, float scale = 1.0f,
                             bool accumulate_gamma = false);

}

namespace ops::norm
{
using ops::normalization::rmsNormForward;
using ops::normalization::rmsNormBackward;
using ops::normalization::residualScaleRmsNormForward;
using ops::normalization::rmsNormBackwardResidual;
}

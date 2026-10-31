#pragma once

#include <cstdint>
#include "math/tensor.h"

namespace ops::loss
{

void diceBce(const Tensor &prediction, const Tensor &target, Tensor &output,
             float dice_weight = 1.0f, float bce_weight = 1.0f, float epsilon = 1e-7f);

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight = 1.0f, float bce_weight = 1.0f, float epsilon = 1e-7f,
                     const Tensor *stats_tensor = nullptr);

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight, float bce_weight, float epsilon,
                     float intersection, float pred_sum, float target_sum);

float diceBceLossAndGradient(const Tensor &prediction, const Tensor &target, Tensor &gradient_out,
                             float dice_weight = 1.0f, float bce_weight = 1.0f, float epsilon = 1e-7f);

} // namespace ops::loss

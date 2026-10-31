#pragma once

#include <cstddef>
#include "math/tensor.h"

namespace ops::optimizer
{

void sgdStep(Tensor &parameter,
             const Tensor &gradient,
             float learning_rate,
             float max_gradient = 0.0f,
             float inv_scale = 1.0f);

} // namespace ops::optimizer

namespace ops::optim
{
using ops::optimizer::sgdStep;
} // namespace ops::optim

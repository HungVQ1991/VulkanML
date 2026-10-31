#pragma once

#include <cstddef>
#include "math/tensor.h"

namespace ops::optimizer
{

void adamStep(Tensor &parameter,
              const Tensor &gradient,
              Tensor &first_moment,
              Tensor &second_moment,
              float learning_rate,
              float beta1,
              float beta2,
              float epsilon,
              size_t timestep,
              float max_gradient = 1.0f,
              float inv_scale = 1.0f,
              float weight_decay = 0.0f);

} // namespace ops::optimizer

namespace ops::optim
{
using ops::optimizer::adamStep;
} // namespace ops::optim

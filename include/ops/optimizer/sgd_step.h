#pragma once

#include <cstddef>
#include <memory>
#include "engine/gpu_vector.h"
#include "math/tensor.h"

namespace ops::optimizer
{

void sgdStep(Tensor &parameter,
             const Tensor &gradient,
             float learning_rate,
             float max_gradient = 0.0f,
             float inv_scale = 1.0f,
             const std::shared_ptr<gpu::vector> &dynamic_params_buffer = nullptr,
             float weight_decay = 0.0f,
             float momentum = 0.0f,
             Tensor *velocity = nullptr);

}

namespace ops::optim
{
using ops::optimizer::sgdStep;
}

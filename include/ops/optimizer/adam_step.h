#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include "engine/gpu_vector.h"
#include "math/tensor.h"

namespace ops::optimizer
{

struct Adam_Multi_Item
{
    Tensor *parameter = nullptr;
    const Tensor *gradient = nullptr;
    Tensor *first_moment = nullptr;
    Tensor *second_moment = nullptr;
};

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
              float weight_decay = 0.0f,
              const std::shared_ptr<gpu::vector> &dynamic_params_buffer = nullptr);

void adamStepMulti4Fp16(const std::array<Adam_Multi_Item, 4> &items,
                        size_t count,
                        float learning_rate,
                        float beta1,
                        float beta2,
                        float epsilon,
                        size_t timestep,
                        float max_gradient,
                        float inv_scale,
                        float weight_decay,
                        const std::shared_ptr<gpu::vector> &dynamic_params_buffer);

}

namespace ops::optim
{
using ops::optimizer::Adam_Multi_Item;
using ops::optimizer::adamStep;
using ops::optimizer::adamStepMulti4Fp16;
}

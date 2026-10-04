#include "layer/swiglu_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"


SwiGLU_Layer::SwiGLU_Layer(Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
          tensor_a(0, 0, _execution_target),
          tensor_b(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          grad_a(0, 0, _execution_target),
          grad_b(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{
    }

Tensor SwiGLU_Layer::forward(const Tensor &_input_tensor)
{
        size_t total_cols = _input_tensor.getColumns();
        if (total_cols % 2 != 0)
        {
            throw std::invalid_argument("SwiGLU_Layer::forward: Input columns must be even to split into gate and up projections");
        }

        input_tensor = _input_tensor;
        input_tensor.fusedSwiGLUForward(output_tensor);

        is_forward_completed = true;
        return output_tensor;
    }

Tensor SwiGLU_Layer::forward(const Tensor &a, const Tensor &b)
{
        tensor_a = a;
        tensor_b = b;
        tensor_a.swigluForward(tensor_b, output_tensor);

        is_forward_completed = true;
        return output_tensor;
    }

Tensor SwiGLU_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            throw std::logic_error("SwiGLU_Layer::backward called before forward");
        }

        input_tensor.fusedSwiGLUBackward(_output_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

std::unique_ptr<ILayer> SwiGLU_Layer::clone() const
{
        auto copy = std::make_unique<SwiGLU_Layer>(execution_target);
        copy->setMixedPrecision(is_mixed_precision_enabled);
        return copy;
    }

void SwiGLU_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void SwiGLU_Layer::saveConfiguration(std::ofstream &) const
{}

void SwiGLU_Layer::saveInference(std::ofstream &) const
{}

void SwiGLU_Layer::loadInference(std::ifstream &)
{}

void SwiGLU_Layer::saveCheckpoint(std::ofstream &) const
{}

void SwiGLU_Layer::loadCheckpoint(std::ifstream &)
{}

const Tensor & SwiGLU_Layer::getGradA() const noexcept
{ return grad_a; }

const Tensor & SwiGLU_Layer::getGradB() const noexcept
{ return grad_b; }

void SwiGLU_Layer::setExecutionTarget(Execution_Target _new_target)
{
        if (execution_target == _new_target) return;
        execution_target = _new_target;
        input_tensor.setExecutionTarget(_new_target);
        tensor_a.setExecutionTarget(_new_target);
        tensor_b.setExecutionTarget(_new_target);
        output_tensor.setExecutionTarget(_new_target);
        grad_a.setExecutionTarget(_new_target);
        grad_b.setExecutionTarget(_new_target);
        input_gradient_tensor.setExecutionTarget(_new_target);
    }

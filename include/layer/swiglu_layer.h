#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class SwiGLU_Layer : public ILayer
{
private:
    Tensor input_tensor;
    Tensor tensor_a;
    Tensor tensor_b;

    Tensor output_tensor;

    Tensor grad_a;
    Tensor grad_b;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;

    explicit SwiGLU_Layer(Execution_Target _execution_target = Execution_Target::CPU)
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

    ~SwiGLU_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override
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

    Tensor forward(const Tensor &a, const Tensor &b)
    {
        tensor_a = a;
        tensor_b = b;
        tensor_a.swigluForward(tensor_b, output_tensor);

        is_forward_completed = true;
        return output_tensor;
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            throw std::logic_error("SwiGLU_Layer::backward called before forward");
        }

        input_tensor.fusedSwiGLUBackward(_output_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        auto copy = std::make_unique<SwiGLU_Layer>(execution_target);
        copy->setMixedPrecision(is_mixed_precision_enabled);
        return copy;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    void saveConfiguration(std::ofstream &) const override {}
    void saveInference(std::ofstream &) const override {}
    void loadInference(std::ifstream &) override {}
    void saveCheckpoint(std::ofstream &) const override {}
    void loadCheckpoint(std::ifstream &) override {}

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getGradA() const noexcept { return grad_a; }
    const Tensor &getGradB() const noexcept { return grad_b; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::SWIGLU; }

    void setExecutionTarget(Execution_Target _new_target) override
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
};

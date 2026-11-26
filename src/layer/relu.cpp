#include "layer/relu.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_internal_utils.h"
#include "ops/activation/activation.h"

Relu_Layer::Relu_Layer(Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor Relu_Layer::forward(const Tensor &_input_tensor)
{
    return layer_internal::runStatelessUnaryForward(
        _input_tensor, input_tensor, output_tensor, is_forward_completed, execution_target,
        "Relu_Layer::forward", Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION,
        [](const Tensor &in, Tensor &out) { ops::activation::relu(in, out); });
}

Tensor Relu_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    return layer_internal::runStatelessUnaryBatchedForward(
        _batched_input, _batched_params, "Relu_Layer::forward", "Relu_Layer",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION,
        [](const Tensor &in, Tensor &out) { ops::activation::relu(in, out); });
}

Tensor Relu_Layer::backward(const Tensor &_output_gradient)
{
    return layer_internal::runStatelessUnaryBackward(
        _output_gradient, output_tensor, output_tensor, input_gradient_tensor,
        is_forward_completed, execution_target, "Relu_Layer::backward",
        "Relu backward called before forward", "Relu gradient shape must match output shape",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION,
        [](const Tensor &out, const Tensor &grad, Tensor &in_grad) {
            ops::activation::reluBackward(out, grad, in_grad);
        });
}

void Relu_Layer::resetGradient() { is_forward_completed = false; }

std::unique_ptr<ILayer> Relu_Layer::clone() const
{
    auto cloned = std::make_unique<Relu_Layer>(execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

void Relu_Layer::saveConfiguration(std::ofstream &) const {}
void Relu_Layer::saveInference(std::ofstream &) const {}
void Relu_Layer::loadInference(std::ifstream &) {}
void Relu_Layer::saveCheckpoint(std::ofstream &) const {}
void Relu_Layer::loadCheckpoint(std::ifstream &) {}

void Relu_Layer::setPopulationParameter(size_t, std::vector<float>)
{
    throw std::out_of_range("Relu_Layer::setPopulationParameter: Layer has no parameters");
}

void Relu_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, input_gradient_tensor);
}

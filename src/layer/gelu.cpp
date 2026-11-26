#include "layer/gelu.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_internal_utils.h"
#include "ops/activation/activation.h"

Gelu_Layer::Gelu_Layer(Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor Gelu_Layer::forward(const Tensor &_input_tensor)
{
    return layer_internal::runStatelessUnaryForward(
        _input_tensor, input_tensor, output_tensor, is_forward_completed, execution_target,
        "Gelu_Layer::forward", Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION,
        [](const Tensor &in, Tensor &out) { ops::activation::gelu(in, out); });
}

Tensor Gelu_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    return layer_internal::runStatelessUnaryBatchedForward(
        _batched_input, _batched_params, "Gelu_Layer::forward", "Gelu_Layer",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION,
        [](const Tensor &in, Tensor &out) { ops::activation::gelu(in, out); });
}

Tensor Gelu_Layer::backward(const Tensor &_output_gradient)
{
    return layer_internal::runStatelessUnaryBackward(
        _output_gradient, input_tensor, output_tensor, input_gradient_tensor,
        is_forward_completed, execution_target, "Gelu_Layer::backward",
        "GeLU backward called before forward", "GeLU gradient shape must match output shape",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION,
        [](const Tensor &in, const Tensor &grad, Tensor &in_grad) {
            ops::activation::geluBackward(in, grad, in_grad);
        });
}

void Gelu_Layer::resetGradient() { is_forward_completed = false; }

std::unique_ptr<ILayer> Gelu_Layer::clone() const
{
    auto cloned = std::make_unique<Gelu_Layer>(execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

void Gelu_Layer::saveConfiguration(std::ofstream &) const {}
void Gelu_Layer::saveInference(std::ofstream &) const {}
void Gelu_Layer::loadInference(std::ifstream &) {}
void Gelu_Layer::saveCheckpoint(std::ofstream &) const {}
void Gelu_Layer::loadCheckpoint(std::ifstream &) {}

void Gelu_Layer::setPopulationParameter(size_t, std::vector<float>)
{
    throw std::out_of_range("Gelu_Layer::setPopulationParameter: Layer has no parameters");
}

void Gelu_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, input_gradient_tensor);
}

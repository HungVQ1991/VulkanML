#include "layer/leaky_relu_layer.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helper/logger.h"
#include "layer_internal_utils.h"
#include "ops/activation/activation.h"

Leaky_Relu_Layer::Leaky_Relu_Layer(float _negative_slope, Execution_Target _execution_target)
    : negative_slope(_negative_slope),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
    if (!std::isfinite(_negative_slope))
    {
        throw std::invalid_argument("Leaky_Relu_Layer: negative_slope must be finite");
    }
}

Tensor Leaky_Relu_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateForwardInput(_input_tensor, execution_target, "Leaky_Relu_Layer::forward");
    Logger::logMessage(Input_Format{"Leaky_Relu_Layer::forward: rows={}, columns={}, negative_slope={}",
                                    _input_tensor.getRows(), _input_tensor.getColumns(), negative_slope},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    input_tensor = _input_tensor;
    ops::activation::leakyRelu(input_tensor, output_tensor, negative_slope);
    is_forward_completed = true;
    return output_tensor;
}

Tensor Leaky_Relu_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    return layer_internal::runStatelessUnaryBatchedForward(
        _batched_input, _batched_params, "Leaky_Relu_Layer::forward", "Leaky_Relu_Layer",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION,
        [this](const Tensor &in, Tensor &out) { ops::activation::leakyRelu(in, out, negative_slope); });
}

Tensor Leaky_Relu_Layer::backward(const Tensor &_output_gradient)
{
    return layer_internal::runStatelessUnaryBackward(
        _output_gradient, input_tensor, output_tensor, input_gradient_tensor,
        is_forward_completed, execution_target, "Leaky_Relu_Layer::backward",
        "LeakyRelu backward called before forward", "Gradient shape must match output shape",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION,
        [this](const Tensor &in, const Tensor &grad, Tensor &in_grad) {
            ops::activation::leakyReluBackward(in, grad, in_grad, negative_slope);
        });
}

void Leaky_Relu_Layer::resetGradient() { is_forward_completed = false; }

std::unique_ptr<ILayer> Leaky_Relu_Layer::clone() const
{
    auto cloned = std::make_unique<Leaky_Relu_Layer>(negative_slope, execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

void Leaky_Relu_Layer::setNegativeSlope(float _negative_slope)
{
    if (!std::isfinite(_negative_slope))
    {
        throw std::invalid_argument("Leaky_Relu_Layer::setNegativeSlope: negative_slope must be finite");
    }
    negative_slope = _negative_slope;
}

void Leaky_Relu_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Leaky_Relu_Layer::saveConfiguration",
                              "Failed writing negative_slope", negative_slope);
}

void Leaky_Relu_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Leaky_Relu_Layer::saveInference",
                              "Failed writing negative_slope", negative_slope);
}

void Leaky_Relu_Layer::loadInference(std::ifstream &_input_file_stream)
{
    layer_internal::requireInStream(_input_file_stream, "Leaky_Relu_Layer::loadInference");
    float loaded_slope = 0.0f;
    layer_internal::readPods(_input_file_stream, "Leaky_Relu_Layer::loadInference: Invalid or corrupted negative_slope", loaded_slope);
    if (!std::isfinite(loaded_slope))
    {
        throw std::runtime_error("Leaky_Relu_Layer::loadInference: Invalid or corrupted negative_slope");
    }
    negative_slope = loaded_slope;
}

void Leaky_Relu_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Leaky_Relu_Layer::saveCheckpoint",
                              "Failed writing negative_slope", negative_slope);
}

void Leaky_Relu_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    layer_internal::requireInStream(_input_file_stream, "Leaky_Relu_Layer::loadCheckpoint");
    float loaded_slope = 0.0f;
    layer_internal::readPods(_input_file_stream, "Leaky_Relu_Layer::loadCheckpoint: Invalid or corrupted negative_slope", loaded_slope);
    if (!std::isfinite(loaded_slope))
    {
        throw std::runtime_error("Leaky_Relu_Layer::loadCheckpoint: Invalid or corrupted negative_slope");
    }
    negative_slope = loaded_slope;
}

void Leaky_Relu_Layer::setPopulationParameter(size_t, std::vector<float>)
{
    throw std::out_of_range("Leaky_Relu_Layer::setPopulationParameter: Layer has no parameters");
}

void Leaky_Relu_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, input_gradient_tensor);
}

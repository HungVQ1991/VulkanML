#include "layer/sigmoid.h"

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
#include "ops/activation/activation.h"

Sigmoid_Layer::Sigmoid_Layer(Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor Sigmoid_Layer::forward(const Tensor &_input_tensor)
{
    Logger::logMessage(Input_Format{"Sigmoid_Layer::forward: rows={}, columns={}",
                                    _input_tensor.getRows(),
                                    _input_tensor.getColumns()},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::activation::sigmoid(input_tensor, output_tensor);
    is_forward_completed = true;
    return output_tensor;
}

Tensor Sigmoid_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
    if (!_batched_params.empty())
    {
        Logger::logMessage(Input_Format{"Sigmoid_Layer::forward: expected 0 batched parameter tensors, got {}",
                                        _batched_params.size()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument("Invalid input params size: Sigmoid_Layer expects 0 parameters");
    }

    Tensor output_tensor_result(_batched_input.getExecutionTarget());
    ops::activation::sigmoid(_batched_input, output_tensor_result);
    return output_tensor_result;
}

Tensor Sigmoid_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Sigmoid_Layer::backward: Sigmoid backward called before forward"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("Sigmoid backward called before forward");
    }

    if (_output_gradient.getTotalElements() != output_tensor.getTotalElements())
    {
        Logger::logMessage(Input_Format{"Sigmoid_Layer::backward: Sigmoid gradient total elements must match output total elements ({} vs {})",
                                        _output_gradient.getTotalElements(), output_tensor.getTotalElements()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::invalid_argument("Sigmoid gradient total elements must match output total elements");
    }

    Logger::logMessage(Input_Format{"Sigmoid_Layer::backward: output_gradient rows={}, columns={}",
                                    _output_gradient.getRows(),
                                    _output_gradient.getColumns()},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    ops::activation::sigmoidBackward(output_tensor, _output_gradient, input_gradient_tensor);
    return input_gradient_tensor;
}

void Sigmoid_Layer::resetGradient()
{
    is_forward_completed = false;
    input_tensor = Tensor(0, 0, execution_target);
    output_tensor = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);
}

std::unique_ptr<ILayer> Sigmoid_Layer::clone() const
{
    return std::make_unique<Sigmoid_Layer>(execution_target);
}

void Sigmoid_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
}

void Sigmoid_Layer::saveInference(std::ofstream &_output_file_stream) const
{
}

void Sigmoid_Layer::loadInference(std::ifstream &_input_file_stream)
{
}

void Sigmoid_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
}

void Sigmoid_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
}

void Sigmoid_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
    throw std::out_of_range("Sigmoid_Layer::setPopulationParameter: Layer has no parameters");
}

void Sigmoid_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target)
    {
        return;
    }

    logChangeExecutionTarget(_new_execution_target);

    execution_target = _new_execution_target;
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

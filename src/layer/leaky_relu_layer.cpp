#include "layer/leaky_relu_layer.h"

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

Leaky_Relu_Layer::Leaky_Relu_Layer(float _negative_slope, Execution_Target _execution_target)
    : negative_slope(_negative_slope),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor Leaky_Relu_Layer::forward(const Tensor &_input_tensor)
{
    Logger::logMessage(Input_Format{"Leaky_Relu_Layer::forward: rows={}, columns={}, negative_slope={}",
                                    _input_tensor.getRows(),
                                    _input_tensor.getColumns(),
                                    negative_slope},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::activation::leakyRelu(input_tensor, output_tensor, negative_slope);
    is_forward_completed = true;
    return output_tensor;
}

Tensor Leaky_Relu_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    if (!_batched_params.empty())
    {
        Logger::logMessage(Input_Format{"Leaky_Relu_Layer::forward: expected 0 batched parameter tensors, got {}",
                                        _batched_params.size()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument("Invalid input params size: Leaky_Relu_Layer expects 0 parameters");
    }

    Tensor output_tensor_result(_batched_input.getExecutionTarget());
    ops::activation::leakyRelu(_batched_input, output_tensor_result, negative_slope);
    return output_tensor_result;
}

Tensor Leaky_Relu_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Leaky_Relu_Layer::backward: LeakyRelu backward called before forward"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("LeakyRelu backward called before forward");
    }

    if (_output_gradient.getRows() != output_tensor.getRows() || _output_gradient.getColumns() != output_tensor.getColumns())
    {
        Logger::logMessage(Input_Format{"Leaky_Relu_Layer::backward: Gradient dimensions must match output dimensions"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::invalid_argument("Gradient dimensions must match output dimensions");
    }

    Logger::logMessage(Input_Format{"Leaky_Relu_Layer::backward: output_gradient rows={}, columns={}",
                                    _output_gradient.getRows(),
                                    _output_gradient.getColumns()},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    ops::activation::leakyReluBackward(input_tensor, _output_gradient, input_gradient_tensor, negative_slope);
    return input_gradient_tensor;
}

void Leaky_Relu_Layer::resetGradient()
{
    is_forward_completed = false;
}

std::unique_ptr<ILayer> Leaky_Relu_Layer::clone() const
{
    return std::make_unique<Leaky_Relu_Layer>(negative_slope, execution_target);
}

void Leaky_Relu_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&negative_slope), sizeof(negative_slope));
}

void Leaky_Relu_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&negative_slope), sizeof(negative_slope));
}

void Leaky_Relu_Layer::loadInference(std::ifstream &_input_file_stream)
{
    _input_file_stream.read(reinterpret_cast<char *>(&negative_slope), sizeof(negative_slope));
}

void Leaky_Relu_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&negative_slope), sizeof(negative_slope));
}

void Leaky_Relu_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    _input_file_stream.read(reinterpret_cast<char *>(&negative_slope), sizeof(negative_slope));
}

void Leaky_Relu_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
    (void)param_index;
    (void)flat_data;
    throw std::out_of_range("Leaky_Relu_Layer::setPopulationParameter: Layer has no parameters");
}

void Leaky_Relu_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

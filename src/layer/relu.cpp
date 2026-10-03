#include "layer/relu.h"

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
        Logger::logMessage(Input_Format{"Relu_Layer::forward: rows={}, columns={}",
                                        _input_tensor.getRows(),
                                        _input_tensor.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.relu(output_tensor);
        is_forward_completed = true;
        return output_tensor;
    }

Tensor Relu_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Relu_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Relu_Layer expects 0 parameters");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.relu(output_tensor_result);
        return output_tensor_result;
    }

Tensor Relu_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Relu_Layer::backward: Relu backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Relu backward called before forward");
        }

        if (_output_gradient.getRows() != output_tensor.getRows() || _output_gradient.getColumns() != output_tensor.getColumns())
        {
            Logger::logMessage(Input_Format{"Relu_Layer::backward: Relu gradient dimensions must match output dimensions"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Relu gradient dimensions must match output dimensions");
        }

        Logger::logMessage(Input_Format{"Relu_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        output_tensor.reluBackward(_output_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

void Relu_Layer::resetGradient()
{
        is_forward_completed = false;
    }

std::unique_ptr<ILayer> Relu_Layer::clone() const
{
        return std::make_unique<Relu_Layer>(execution_target);
    }

void Relu_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{}

void Relu_Layer::saveInference(std::ofstream &_output_file_stream) const
{}

void Relu_Layer::loadInference(std::ifstream &_input_file_stream)
{}

void Relu_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{}

void Relu_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{}

void Relu_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        throw std::out_of_range("Relu_Layer::setPopulationParameter: Layer has no parameters");
    }

void Relu_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

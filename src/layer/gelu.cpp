#include "layer/gelu.h"

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
        Logger::logMessage(Input_Format{"Gelu_Layer::forward: rows={}, columns={}",
                                        _input_tensor.getRows(),
                                        _input_tensor.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.gelu(output_tensor);
        is_forward_completed = true;
        return output_tensor;
    }

Tensor Gelu_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Gelu_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Gelu_Layer expects 0 parameters");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.gelu(output_tensor_result);
        return output_tensor_result;
    }

Tensor Gelu_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Gelu_Layer::backward: GeLU backward called before forward"},
                               Log_Level::LOG_ERROR,
                                true,
                                0,
                                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("GeLU backward called before forward");
        }

        if (_output_gradient.getRows() != output_tensor.getRows() || _output_gradient.getColumns() != output_tensor.getColumns())
        {
            Logger::logMessage(Input_Format{"Gelu_Layer::backward: GeLU gradient dimensions must match output dimensions"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("GeLU gradient dimensions must match output dimensions");
        }

        Logger::logMessage(Input_Format{"Gelu_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        logBufferAddress(&input_tensor, "input_tensor (Backward)");

        input_tensor.geluBackward(_output_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

void Gelu_Layer::resetGradient()
{
        is_forward_completed = false;
    }

std::unique_ptr<ILayer> Gelu_Layer::clone() const
{
        return std::make_unique<Gelu_Layer>(execution_target);
    }

void Gelu_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{}

void Gelu_Layer::saveInference(std::ofstream &_output_file_stream) const
{}

void Gelu_Layer::loadInference(std::ifstream &_input_file_stream)
{}

void Gelu_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{}

void Gelu_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{}

void Gelu_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        throw std::out_of_range("Gelu_Layer::setPopulationParameter: Layer has no parameters");
    }

void Gelu_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

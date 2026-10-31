#include "layer/dropout.h"

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "ops/dropout/dropout.h"

Dropout_Layer::Dropout_Layer(float _dropout_rate, Execution_Target _execution_target, uint64_t _seed)
    : dropout_rate(_dropout_rate),
      is_training(true),
      seed(_seed),
      mask_tensor(0, 0, _execution_target),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
    if (_dropout_rate < 0.0f || _dropout_rate >= 1.0f)
    {
        Logger::logMessage(Input_Format{"Dropout_Layer: invalid dropout_rate {}, must be in [0.0, 1.0)", _dropout_rate},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE);
        throw std::invalid_argument("Dropout rate must be in the range [0.0, 1.0)");
    }
}

Tensor Dropout_Layer::forward(const Tensor &_input_tensor)
{
    Logger::logMessage(Input_Format{"Dropout_Layer::forward: rows={}, columns={}, is_training={}",
                                    _input_tensor.getRows(),
                                    _input_tensor.getColumns(),
                                    is_training},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::dropout::dropout(input_tensor, output_tensor, mask_tensor, dropout_rate, is_training, seed);
    is_forward_completed = true;
    return output_tensor;
}

Tensor Dropout_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    if (!_batched_params.empty())
    {
        Logger::logMessage(Input_Format{"Dropout_Layer::forward: expected 0 batched parameter tensors, got {}",
                                        _batched_params.size()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument("Invalid input params size: Dropout_Layer expects 0 parameters");
    }

    Tensor output_tensor_result(_batched_input.getExecutionTarget());
    Tensor dummy_mask(0, 0, _batched_input.getExecutionTarget());
    ops::dropout::dropout(_batched_input, output_tensor_result, dummy_mask, dropout_rate, is_training, seed);
    return output_tensor_result;
}

Tensor Dropout_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage("Dropout_Layer::backward: Forward pass has not been completed",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::runtime_error("Forward pass must be completed before backward pass in Dropout_Layer");
    }

    Logger::logMessage(Input_Format{"Dropout_Layer::backward: rows={}, columns={}",
                                    _output_gradient.getRows(),
                                    _output_gradient.getColumns()},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    if (!is_training || dropout_rate <= 0.0f)
    {
        input_gradient_tensor = _output_gradient;
    }
    else
    {
        ops::dropout::dropoutBackward(_output_gradient, mask_tensor, input_gradient_tensor, dropout_rate);
    }

    return input_gradient_tensor;
}

void Dropout_Layer::resetGradient()
{
    input_gradient_tensor = Tensor(0, 0, execution_target);
    mask_tensor = Tensor(0, 0, execution_target);
    is_forward_completed = false;
}

std::unique_ptr<ILayer> Dropout_Layer::clone() const
{
    auto cloned = std::make_unique<Dropout_Layer>(dropout_rate, execution_target, seed);
    cloned->setTrainingMode(is_training);
    return cloned;
}

void Dropout_Layer::setDropoutRate(float _rate)
{
    if (_rate < 0.0f || _rate >= 1.0f)
    {
        throw std::invalid_argument("Dropout rate must be in the range [0.0, 1.0)");
    }
    dropout_rate = _rate;
}

void Dropout_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    mask_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

void Dropout_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open())
    {
        throw std::runtime_error("Dropout_Layer::saveConfiguration: Output stream is not open");
    }
    _output_file_stream.write(reinterpret_cast<const char *>(&dropout_rate), sizeof(dropout_rate));
}

void Dropout_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    saveConfiguration(_output_file_stream);
}

void Dropout_Layer::loadInference(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open())
    {
        throw std::runtime_error("Dropout_Layer::loadInference: Input stream is not open");
    }
    _input_file_stream.read(reinterpret_cast<char *>(&dropout_rate), sizeof(dropout_rate));
}

void Dropout_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    saveConfiguration(_output_file_stream);
}

void Dropout_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    loadInference(_input_file_stream);
}

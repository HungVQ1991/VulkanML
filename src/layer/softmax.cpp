#include "layer/softmax.h"

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


Softmax_Layer::Softmax_Layer(bool _is_fused_with_loss, Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          input_fp32(0, 0, _execution_target),
          is_fused_with_loss(_is_fused_with_loss),
          is_forward_completed(false),
          execution_target(_execution_target)
{
    }

Tensor Softmax_Layer::forward(const Tensor &_input_tensor)
{
        Logger::logMessage(Input_Format{"Softmax_Layer::forward: rows={}, columns={}",
                                        _input_tensor.getRows(),
                                        _input_tensor.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        if (input_tensor.getDataType() == Data_Type::FLOAT16)
        {
            input_tensor.to(Data_Type::FLOAT32, input_fp32);
            input_fp32.softmax(output_tensor);
        }
        else
        {
            input_tensor.softmax(output_tensor);
        }
        is_forward_completed = true;
        return output_tensor;
    }

Tensor Softmax_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Softmax_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Softmax_Layer expects 0 parameters");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.softmax(output_tensor_result);
        return output_tensor_result;
    }

std::unique_ptr<ILayer> Softmax_Layer::clone() const
{
        return std::make_unique<Softmax_Layer>(is_fused_with_loss, execution_target);
    }

Tensor Softmax_Layer::backward(const Tensor& _output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{ "Softmax_Layer::backward: Backward called before forward" },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{ "Softmax_Layer::backward: output_gradient rows={}, columns={}, is_fused_with_loss={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns(),
                                        is_fused_with_loss },
            Log_Level::LOG_DEBUG,
            true,
            1,
            Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        if (is_fused_with_loss)
        {
            return _output_gradient;
        }

        output_tensor.softmaxBackward(_output_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

void Softmax_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Softmax_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        std::uint8_t fused_value = is_fused_with_loss ? 1 : 0;
        _output_file_stream.write(reinterpret_cast<const char *>(&fused_value), sizeof(fused_value));
    }

void Softmax_Layer::saveInference(std::ofstream &_output_file_stream) const
{}

void Softmax_Layer::loadInference(std::ifstream &_input_file_stream)
{}

void Softmax_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{}

void Softmax_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{}

bool Softmax_Layer::isFusedWithLoss() const noexcept
{ return is_fused_with_loss; }

void Softmax_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        throw std::out_of_range("Softmax_Layer::setPopulationParameter: Layer has no parameters");
    }

void Softmax_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

void Softmax_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

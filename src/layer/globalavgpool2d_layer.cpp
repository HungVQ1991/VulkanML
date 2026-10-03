#include "layer/globalavgpool2d_layer.h"

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


Global_Avg_Pool_2d_Layer::Global_Avg_Pool_2d_Layer(uint32_t _height, uint32_t _width, uint32_t _channels, Execution_Target _execution_target)
    : input_height(_height),
          input_width(_width),
          channels(_channels),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{
    }

Tensor Global_Avg_Pool_2d_Layer::forward(const Tensor &_input_tensor)
{
        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                        input_height, input_width, channels},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.globalAvgPool2d(output_tensor, input_height, input_width, channels);
        is_forward_completed = true;
        logBufferAddress(&input_tensor, "input_tensor (Forward)");
        logBufferAddress(&output_tensor, "output_tensor (Forward)");

        return output_tensor;
    }

Tensor Global_Avg_Pool_2d_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Global_Avg_Pool_2d_Layer::forward: expected {} batched parameter tensors, got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.globalAvgPool2d(output_tensor_result, input_height, input_width, channels);
        return output_tensor_result;
    }

std::unique_ptr<ILayer> Global_Avg_Pool_2d_Layer::clone() const
{
        return std::make_unique<Global_Avg_Pool_2d_Layer>(
            input_height, input_width, channels, execution_target);
    }

Tensor Global_Avg_Pool_2d_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        _output_gradient.globalAvgPool2dBackward(input_gradient_tensor, input_height, input_width, channels);

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        logBufferAddress(&input_gradient_tensor, "input_gradient_tensor (Backward)");
        logBufferAddress(const_cast<Tensor *>(&_output_gradient), "output_gradient (Backward)");

        return input_gradient_tensor;
    }

void Global_Avg_Pool_2d_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Global_Avg_Pool_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    }

void Global_Avg_Pool_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{}

void Global_Avg_Pool_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{}

void Global_Avg_Pool_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{}

void Global_Avg_Pool_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{}

uint32_t Global_Avg_Pool_2d_Layer::getChannels() const noexcept
{ return channels; }

void Global_Avg_Pool_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        throw std::out_of_range("Global_Avg_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
    }

void Global_Avg_Pool_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

void Global_Avg_Pool_2d_Layer::setInputHeight(uint32_t _height) noexcept
{ input_height = _height; }

void Global_Avg_Pool_2d_Layer::setInputWidth(uint32_t _width) noexcept
{ input_width = _width; }

void Global_Avg_Pool_2d_Layer::setChannels(uint32_t _channels) noexcept
{ channels = _channels; }

#include "layer/maxpool2d_layer.h"

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


Max_Pool_2d_Layer::Max_Pool_2d_Layer(uint32_t _height, uint32_t _width, uint32_t _channels, uint32_t _kernel_size, uint32_t _stride, uint32_t _padding, Execution_Target _execution_target)
    : input_height(_height),
          input_width(_width),
          channels(_channels),
          kernel_size(_kernel_size),
          stride(_stride),
          padding(_padding),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          mask_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{
        output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    }

Tensor Max_Pool_2d_Layer::forward(const Tensor &_input_tensor)
{
        Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                        input_height,
                                        input_width,
                                        channels},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.maxpool2d(output_tensor, mask_tensor, input_height, input_width, channels, kernel_size, stride, padding);
        is_forward_completed = true;
        logBufferAddress(&mask_tensor, "mask_tensor");
        return output_tensor;
    }

Tensor Max_Pool_2d_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Max_Pool_2d_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Max_Pool_2d_Layer expects 0 parameters");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        Tensor mask_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.maxpool2d(output_tensor_result, mask_tensor_result, input_height, input_width, channels, kernel_size, stride, padding);
        return output_tensor_result;
    }

std::unique_ptr<ILayer> Max_Pool_2d_Layer::clone() const
{
        return std::make_unique<Max_Pool_2d_Layer>(
            input_height, input_width, channels, kernel_size, stride, padding, execution_target);
    }

Tensor Max_Pool_2d_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                            Log_Level::LOG_DEBUG,
                            true,
                            1,
                            Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        _output_gradient.maxpool2dBackward(mask_tensor, input_gradient_tensor, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
        logBufferAddress(&mask_tensor, "mask_tensor (Backward)");
        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        logBufferAddress(&input_gradient_tensor, "input_gradient_tensor (Backward)");
        logBufferAddress(const_cast<Tensor *>(&_output_gradient), "output_gradient (Backward)");
        return input_gradient_tensor;
    }

void Max_Pool_2d_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Max_Pool_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
        _output_file_stream.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
    }

void Max_Pool_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{}

void Max_Pool_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{}

void Max_Pool_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{}

void Max_Pool_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{}

const Tensor & Max_Pool_2d_Layer::getMask() const noexcept
{ return mask_tensor; }

uint32_t Max_Pool_2d_Layer::getChannels() const noexcept
{ return channels; }

void Max_Pool_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        throw std::out_of_range("Max_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
    }

void Max_Pool_2d_Layer::setMask(const Tensor &_tensor)
{ mask_tensor = _tensor; }

void Max_Pool_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        mask_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }

void Max_Pool_2d_Layer::setOutputHeight(uint32_t _height) noexcept
{ output_height = _height; }

void Max_Pool_2d_Layer::setOutputWidth(uint32_t _width) noexcept
{ output_width = _width; }

void Max_Pool_2d_Layer::setInputHeight(uint32_t _height) noexcept
{ input_height = _height; }

void Max_Pool_2d_Layer::setKernelSize(uint32_t _size) noexcept
{ kernel_size = _size; }

void Max_Pool_2d_Layer::setInputWidth(uint32_t _width) noexcept
{ input_width = _width; }

void Max_Pool_2d_Layer::setChannels(uint32_t _channels) noexcept
{ channels = _channels; }

void Max_Pool_2d_Layer::setPadding(uint32_t _padding) noexcept
{ padding = _padding; }

void Max_Pool_2d_Layer::setStride(uint32_t _stride) noexcept
{ stride = _stride; }

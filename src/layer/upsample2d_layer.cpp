#include "layer/upsample2d_layer.h"

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
#include "ops/vision/pooling.h"

Up_Sample_2d_Layer::Up_Sample_2d_Layer(
    uint32_t _height,
    uint32_t _width,
    uint32_t _channels,
    uint32_t _scale_factor,
    Execution_Target _execution_target)
    : input_height(_height),
      input_width(_width),
      channels(_channels),
      scale_factor(_scale_factor),
      output_height(_height * _scale_factor),
      output_width(_width * _scale_factor),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor Up_Sample_2d_Layer::forward(const Tensor &_input_tensor)
{
    Logger::logMessage(Input_Format{"Up_Sample_2d_Layer::forward: in_h={}, in_w={}, ch={}, scale={}",
                                    input_height,
                                    input_width,
                                    channels,
                                    scale_factor},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::vision::upsample2d(input_tensor, output_tensor, input_height, input_width, channels, scale_factor);
    is_forward_completed = true;
    return output_tensor;
}

std::unique_ptr<ILayer> Up_Sample_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Up_Sample_2d_Layer>(input_height, input_width, channels, scale_factor, execution_target);
    cloned->output_height = output_height;
    cloned->output_width = output_width;
    return cloned;
}

Tensor Up_Sample_2d_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage("Up_Sample_2d_Layer::backward: Forward must be called before backward",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::BACKWARD_PROPAGATION);
        throw std::runtime_error("Forward must be called before backward");
    }

    Logger::logMessage(Input_Format{"Up_Sample_2d_Layer::backward: in_h={}, in_w={}, ch={}",
                                    input_height,
                                    input_width,
                                    channels},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::BACKWARD_PROPAGATION);

    ops::vision::upsample2dBackward(_output_gradient, input_gradient_tensor, input_height, input_width, channels, scale_factor);
    return input_gradient_tensor;
}

void Up_Sample_2d_Layer::resetGradient()
{
    is_forward_completed = false;
    input_tensor = Tensor(0, 0, execution_target);
    output_tensor = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);
}

void Up_Sample_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    _output_file_stream.write(reinterpret_cast<const char *>(&scale_factor), sizeof(scale_factor));
}

void Up_Sample_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    saveConfiguration(_output_file_stream);
}

void Up_Sample_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
    _input_file_stream.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
    _input_file_stream.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
    _input_file_stream.read(reinterpret_cast<char *>(&channels), sizeof(channels));
    _input_file_stream.read(reinterpret_cast<char *>(&scale_factor), sizeof(scale_factor));
    output_height = input_height * scale_factor;
    output_width = input_width * scale_factor;
}

void Up_Sample_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    saveConfiguration(_output_file_stream);
}

void Up_Sample_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    loadInference(_input_file_stream);
}

void Up_Sample_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

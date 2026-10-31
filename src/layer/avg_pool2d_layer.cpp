#include "layer/avg_pool2d_layer.h"

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

Avg_Pool_2d_Layer::Avg_Pool_2d_Layer(
    uint32_t _height,
    uint32_t _width,
    uint32_t _channels,
    uint32_t _kernel_size,
    uint32_t _stride,
    uint32_t _padding,
    Execution_Target _execution_target)
    : input_height(_height),
      input_width(_width),
      channels(_channels),
      kernel_size(_kernel_size),
      stride(_stride),
      padding(_padding),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
    output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
}

Tensor Avg_Pool_2d_Layer::forward(const Tensor &_input_tensor)
{
    input_tensor = _input_tensor;
    ops::vision::avgpool2d(input_tensor, output_tensor, input_height, input_width, channels, kernel_size, stride, padding);
    is_forward_completed = true;
    return output_tensor;
}

std::unique_ptr<ILayer> Avg_Pool_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Avg_Pool_2d_Layer>(
        input_height, input_width, channels, kernel_size, stride, padding, execution_target);
    cloned->output_height = output_height;
    cloned->output_width = output_width;
    return cloned;
}

Tensor Avg_Pool_2d_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::runtime_error("Avg_Pool_2d_Layer::backward: Backward called before forward");
    }

    ops::vision::avgpool2dBackward(
        _output_gradient, input_gradient_tensor,
        input_height, input_width, channels,
        output_height, output_width,
        kernel_size, stride, padding);

    return input_gradient_tensor;
}

void Avg_Pool_2d_Layer::resetGradient()
{
    is_forward_completed = false;
    input_tensor = Tensor(0, 0, execution_target);
    output_tensor = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);
}

void Avg_Pool_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    _output_file_stream.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
    _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
    _output_file_stream.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
}

void Avg_Pool_2d_Layer::saveInference(std::ofstream &) const
{
}

void Avg_Pool_2d_Layer::loadInference(std::ifstream &)
{
}

void Avg_Pool_2d_Layer::saveCheckpoint(std::ofstream &) const
{
}

void Avg_Pool_2d_Layer::loadCheckpoint(std::ifstream &)
{
}

void Avg_Pool_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

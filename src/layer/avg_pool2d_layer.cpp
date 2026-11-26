#include "layer/avg_pool2d_layer.h"

#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_internal_utils.h"
#include "ops/vision/pooling.h"

std::pair<uint32_t, uint32_t> Avg_Pool_2d_Layer::validateAndComputeOutput(
    uint32_t _height, uint32_t _width, uint32_t _channels,
    uint32_t _kernel_size, uint32_t _stride, uint32_t _padding)
{
    return layer_internal::validatePool2dOutputDims(
        _height, _width, _channels, _kernel_size, _stride, _padding, "Avg_Pool_2d_Layer");
}

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
    auto [out_h, out_w] = validateAndComputeOutput(_height, _width, _channels, _kernel_size, _stride, _padding);
    output_height = out_h;
    output_width = out_w;
}

Tensor Avg_Pool_2d_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateNhwcSpatialInput(
        _input_tensor, execution_target, input_height, input_width, channels, "Avg_Pool_2d_Layer::forward");
    input_tensor = _input_tensor;
    ops::vision::avgpool2d(input_tensor, output_tensor, input_height, input_width, channels, kernel_size, stride, padding);
    is_forward_completed = true;
    return output_tensor;
}

std::unique_ptr<ILayer> Avg_Pool_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Avg_Pool_2d_Layer>(
        input_height, input_width, channels, kernel_size, stride, padding, execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

Tensor Avg_Pool_2d_Layer::backward(const Tensor &_output_gradient)
{
    layer_internal::validateNhwcSpatialGrad(
        is_forward_completed, input_tensor, _output_gradient, output_height, output_width, channels,
        "Avg_Pool_2d_Layer::backward", "Avg_Pool_2d_Layer::backward: Backward called before forward");

    ops::vision::avgpool2dBackward(
        _output_gradient, input_gradient_tensor,
        input_height, input_width, channels,
        output_height, output_width,
        kernel_size, stride, padding);
    if (input_gradient_tensor.getShape() != input_tensor.getShape())
    {
        input_gradient_tensor.reshape(input_tensor.getShape());
    }
    return input_gradient_tensor;
}

void Avg_Pool_2d_Layer::resetGradient() { is_forward_completed = false; }

void Avg_Pool_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float>)
{
    throw std::out_of_range(std::format("Avg_Pool_2d_Layer has no evolvable population parameters (index={})", param_index));
}

void Avg_Pool_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Avg_Pool_2d_Layer::saveConfiguration",
                              "Failed writing configuration",
                              input_height, input_width, channels, kernel_size, stride, padding);
}

void Avg_Pool_2d_Layer::saveInference(std::ofstream &) const {}
void Avg_Pool_2d_Layer::loadInference(std::ifstream &) {}
void Avg_Pool_2d_Layer::saveCheckpoint(std::ofstream &) const {}
void Avg_Pool_2d_Layer::loadCheckpoint(std::ifstream &) {}

void Avg_Pool_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, input_gradient_tensor);
}

void Avg_Pool_2d_Layer::reconfigure(uint32_t _height, uint32_t _width, uint32_t _channels,
                                    uint32_t _kernel_size, uint32_t _stride, uint32_t _padding)
{
    auto [out_h, out_w] = validateAndComputeOutput(_height, _width, _channels, _kernel_size, _stride, _padding);
    input_height = _height;
    input_width = _width;
    channels = _channels;
    kernel_size = _kernel_size;
    stride = _stride;
    padding = _padding;
    output_height = out_h;
    output_width = out_w;
    is_forward_completed = false;
}

void Avg_Pool_2d_Layer::setInputHeight(uint32_t _height) { reconfigure(_height, input_width, channels, kernel_size, stride, padding); }
void Avg_Pool_2d_Layer::setKernelSize(uint32_t _size) { reconfigure(input_height, input_width, channels, _size, stride, padding); }
void Avg_Pool_2d_Layer::setInputWidth(uint32_t _width) { reconfigure(input_height, _width, channels, kernel_size, stride, padding); }
void Avg_Pool_2d_Layer::setChannels(uint32_t _channels) { reconfigure(input_height, input_width, _channels, kernel_size, stride, padding); }
void Avg_Pool_2d_Layer::setPadding(uint32_t _padding) { reconfigure(input_height, input_width, channels, kernel_size, stride, _padding); }
void Avg_Pool_2d_Layer::setStride(uint32_t _stride) { reconfigure(input_height, input_width, channels, kernel_size, _stride, padding); }

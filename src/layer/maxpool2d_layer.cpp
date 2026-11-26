#include "layer/maxpool2d_layer.h"

#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helper/logger.h"
#include "layer_internal_utils.h"
#include "ops/vision/pooling.h"

std::pair<uint32_t, uint32_t> Max_Pool_2d_Layer::validateAndComputeOutput(
    uint32_t _height, uint32_t _width, uint32_t _channels,
    uint32_t _kernel_size, uint32_t _stride, uint32_t _padding)
{
    return layer_internal::validatePool2dOutputDims(
        _height, _width, _channels, _kernel_size, _stride, _padding, "Max_Pool_2d_Layer");
}

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
    auto [out_h, out_w] = validateAndComputeOutput(_height, _width, _channels, _kernel_size, _stride, _padding);
    output_height = out_h;
    output_width = out_w;
}

Tensor Max_Pool_2d_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateNhwcSpatialInput(
        _input_tensor, execution_target, input_height, input_width, channels, "Max_Pool_2d_Layer::forward");
    Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                    input_height, input_width, channels},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::vision::maxpool2d(input_tensor, output_tensor, mask_tensor, input_height, input_width, channels, kernel_size, stride, padding);
    is_forward_completed = true;
    return output_tensor;
}

Tensor Max_Pool_2d_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    layer_internal::validateBatchedParamsEmpty(
        _batched_params, "Max_Pool_2d_Layer::forward", "Max_Pool_2d_Layer",
        Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    if (!_batched_input.getImplementation())
    {
        throw std::invalid_argument("Max_Pool_2d_Layer::forward: Batched input tensor is uninitialized");
    }
    if (_batched_input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Max_Pool_2d_Layer::forward: TERNARY_INT2 is not supported");
    }

    const Shape &in_shape = _batched_input.getShape();
    const size_t expected_in_cols = static_cast<size_t>(input_height) * input_width * channels;
    const size_t expected_out_cols = static_cast<size_t>(output_height) * output_width * channels;
    Tensor output_tensor_result(_batched_input.getExecutionTarget());
    Tensor mask_tensor_result(_batched_input.getExecutionTarget());

    if (in_shape.getRank() == 3)
    {
        if (in_shape[2] != expected_in_cols)
        {
            throw std::invalid_argument("Max_Pool_2d_Layer::forward: Rank-3 input last dimension does not match H*W*C");
        }
        const size_t p_dim = in_shape[0], s_dim = in_shape[1];
        Tensor flat_input = _batched_input.view(Shape{p_dim * s_dim, expected_in_cols});
        ops::vision::maxpool2d(flat_input, output_tensor_result, mask_tensor_result, input_height, input_width, channels, kernel_size, stride, padding);
        output_tensor_result.reshape(Shape{p_dim, s_dim, expected_out_cols});
        return output_tensor_result;
    }

    layer_internal::validateNhwcSpatialInput(
        _batched_input, _batched_input.getExecutionTarget(), input_height, input_width, channels, "Max_Pool_2d_Layer::forward");
    ops::vision::maxpool2d(_batched_input, output_tensor_result, mask_tensor_result, input_height, input_width, channels, kernel_size, stride, padding);
    return output_tensor_result;
}

std::unique_ptr<ILayer> Max_Pool_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Max_Pool_2d_Layer>(
        input_height, input_width, channels, kernel_size, stride, padding, execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

Tensor Max_Pool_2d_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::backward: Backward called before forward"},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    layer_internal::validateNhwcSpatialGrad(
        is_forward_completed, input_tensor, _output_gradient, output_height, output_width, channels,
        "Max_Pool_2d_Layer::backward", "Backward called before forward");

    const size_t expected_batch = input_tensor.getShape()[0];
    const size_t expected_out_cols = static_cast<size_t>(output_height) * output_width * channels;
    if (mask_tensor.getTotalElements() != expected_batch * expected_out_cols)
    {
        throw std::invalid_argument("Max_Pool_2d_Layer::backward: Mask element count does not match output dimensions");
    }

    Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                    _output_gradient.getRows(), _output_gradient.getColumns()},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    ops::vision::maxpool2dBackward(_output_gradient, mask_tensor, input_gradient_tensor, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
    if (input_gradient_tensor.getShape() != input_tensor.getShape())
    {
        input_gradient_tensor.reshape(input_tensor.getShape());
    }
    return input_gradient_tensor;
}

void Max_Pool_2d_Layer::resetGradient() { is_forward_completed = false; }

void Max_Pool_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Max_Pool_2d_Layer::saveConfiguration",
                              "Failed writing configuration",
                              input_height, input_width, channels, kernel_size, stride, padding);
}

void Max_Pool_2d_Layer::saveInference(std::ofstream &) const {}
void Max_Pool_2d_Layer::loadInference(std::ifstream &) {}
void Max_Pool_2d_Layer::saveCheckpoint(std::ofstream &) const {}
void Max_Pool_2d_Layer::loadCheckpoint(std::ifstream &) {}

const Tensor &Max_Pool_2d_Layer::getMask() const noexcept { return mask_tensor; }
uint32_t Max_Pool_2d_Layer::getChannels() const noexcept { return channels; }

void Max_Pool_2d_Layer::setPopulationParameter(size_t, std::vector<float>)
{
    throw std::out_of_range("Max_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
}

void Max_Pool_2d_Layer::setMask(const Tensor &_tensor) { mask_tensor = _tensor; }

void Max_Pool_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, mask_tensor, input_gradient_tensor);
}

void Max_Pool_2d_Layer::reconfigure(uint32_t _height, uint32_t _width, uint32_t _channels,
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

void Max_Pool_2d_Layer::setInputHeight(uint32_t _height) { reconfigure(_height, input_width, channels, kernel_size, stride, padding); }
void Max_Pool_2d_Layer::setKernelSize(uint32_t _size) { reconfigure(input_height, input_width, channels, _size, stride, padding); }
void Max_Pool_2d_Layer::setInputWidth(uint32_t _width) { reconfigure(input_height, _width, channels, kernel_size, stride, padding); }
void Max_Pool_2d_Layer::setChannels(uint32_t _channels) { reconfigure(input_height, input_width, _channels, kernel_size, stride, padding); }
void Max_Pool_2d_Layer::setPadding(uint32_t _padding) { reconfigure(input_height, input_width, channels, kernel_size, stride, _padding); }
void Max_Pool_2d_Layer::setStride(uint32_t _stride) { reconfigure(input_height, input_width, channels, kernel_size, _stride, padding); }

#include "layer/globalavgpool2d_layer.h"

#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helper/logger.h"
#include "layer_internal_utils.h"
#include "ops/vision/pooling.h"

void Global_Avg_Pool_2d_Layer::validateDimensions(uint32_t _height, uint32_t _width, uint32_t _channels)
{
    if (_height == 0 || _width == 0 || _channels == 0)
    {
        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer: Invalid non-positive parameter (h={}, w={}, c={})",
                                        _height, _width, _channels},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE);
        throw std::invalid_argument("Global_Avg_Pool_2d_Layer: input dimensions and channels must be > 0");
    }
    const uint64_t spatial = static_cast<uint64_t>(_height) * static_cast<uint64_t>(_width);
    if (spatial > std::numeric_limits<size_t>::max() / _channels)
    {
        throw std::overflow_error("Global_Avg_Pool_2d_Layer: Per-sample feature size overflows size_t");
    }
}

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
    validateDimensions(_height, _width, _channels);
}

Tensor Global_Avg_Pool_2d_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateNhwcSpatialInput(
        _input_tensor, execution_target, input_height, input_width, channels, "Global_Avg_Pool_2d_Layer::forward");
    Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                    input_height, input_width, channels},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::vision::globalAvgPool2d(input_tensor, output_tensor, input_height, input_width, channels);
    is_forward_completed = true;
    return output_tensor;
}

Tensor Global_Avg_Pool_2d_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    if (_batched_params.size() != getPopulationParameterDims().size())
    {
        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::forward: expected {} batched parameter tensors, got {}",
                                        getPopulationParameterDims().size(), _batched_params.size()},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument("Invalid input params size");
    }
    if (!_batched_input.getImplementation())
    {
        throw std::invalid_argument("Global_Avg_Pool_2d_Layer::forward: Batched input tensor is uninitialized");
    }
    if (_batched_input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Global_Avg_Pool_2d_Layer::forward: TERNARY_INT2 is not supported");
    }

    const Shape &in_shape = _batched_input.getShape();
    const size_t expected_in_cols = static_cast<size_t>(input_height) * input_width * channels;
    Tensor output_tensor_result(_batched_input.getExecutionTarget());

    if (in_shape.getRank() == 3)
    {
        if (in_shape[2] != expected_in_cols)
        {
            throw std::invalid_argument("Global_Avg_Pool_2d_Layer::forward: Rank-3 input last dimension does not match H*W*C");
        }
        const size_t p_dim = in_shape[0], s_dim = in_shape[1];
        Tensor flat_input = _batched_input.view(Shape{p_dim * s_dim, expected_in_cols});
        ops::vision::globalAvgPool2d(flat_input, output_tensor_result, input_height, input_width, channels);
        output_tensor_result.reshape(Shape{p_dim, s_dim, channels});
        return output_tensor_result;
    }

    layer_internal::validateNhwcSpatialInput(
        _batched_input, _batched_input.getExecutionTarget(), input_height, input_width, channels, "Global_Avg_Pool_2d_Layer::forward");
    ops::vision::globalAvgPool2d(_batched_input, output_tensor_result, input_height, input_width, channels);
    return output_tensor_result;
}

std::unique_ptr<ILayer> Global_Avg_Pool_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Global_Avg_Pool_2d_Layer>(input_height, input_width, channels, execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

Tensor Global_Avg_Pool_2d_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::backward: Backward called before forward"},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    layer_internal::validateNhwcSpatialGrad(
        is_forward_completed, input_tensor, _output_gradient, 1, 1, channels,
        "Global_Avg_Pool_2d_Layer::backward", "Backward called before forward");

    Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                    _output_gradient.getRows(), _output_gradient.getColumns()},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    ops::vision::globalAvgPool2dBackward(_output_gradient, input_gradient_tensor, input_height, input_width, channels);
    if (input_gradient_tensor.getShape() != input_tensor.getShape())
    {
        input_gradient_tensor.reshape(input_tensor.getShape());
    }
    return input_gradient_tensor;
}

void Global_Avg_Pool_2d_Layer::resetGradient() { is_forward_completed = false; }

void Global_Avg_Pool_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Global_Avg_Pool_2d_Layer::saveConfiguration",
                              "Failed writing configuration", input_height, input_width, channels);
}

void Global_Avg_Pool_2d_Layer::saveInference(std::ofstream &) const {}
void Global_Avg_Pool_2d_Layer::loadInference(std::ifstream &) {}
void Global_Avg_Pool_2d_Layer::saveCheckpoint(std::ofstream &) const {}
void Global_Avg_Pool_2d_Layer::loadCheckpoint(std::ifstream &) {}

uint32_t Global_Avg_Pool_2d_Layer::getChannels() const noexcept { return channels; }

void Global_Avg_Pool_2d_Layer::setPopulationParameter(size_t, std::vector<float>)
{
    throw std::out_of_range("Global_Avg_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
}

void Global_Avg_Pool_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, input_gradient_tensor);
}

void Global_Avg_Pool_2d_Layer::setInputHeight(uint32_t _height)
{
    validateDimensions(_height, input_width, channels);
    input_height = _height;
    is_forward_completed = false;
}

void Global_Avg_Pool_2d_Layer::setInputWidth(uint32_t _width)
{
    validateDimensions(input_height, _width, channels);
    input_width = _width;
    is_forward_completed = false;
}

void Global_Avg_Pool_2d_Layer::setChannels(uint32_t _channels)
{
    validateDimensions(input_height, input_width, _channels);
    channels = _channels;
    is_forward_completed = false;
}

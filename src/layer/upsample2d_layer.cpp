#include "layer/upsample2d_layer.h"

#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "layer_internal_utils.h"
#include "ops/vision/pooling.h"

std::pair<uint32_t, uint32_t> Up_Sample_2d_Layer::validateAndComputeOutput(
    uint32_t _height, uint32_t _width, uint32_t _channels, uint32_t _scale_factor)
{
    if (_height == 0 || _width == 0 || _channels == 0 || _scale_factor == 0)
    {
        throw std::invalid_argument("Up_Sample_2d_Layer: input dimensions, channels, and scale_factor must be > 0");
    }
    const uint64_t out_h = static_cast<uint64_t>(_height) * static_cast<uint64_t>(_scale_factor);
    const uint64_t out_w = static_cast<uint64_t>(_width) * static_cast<uint64_t>(_scale_factor);
    if (out_h > std::numeric_limits<uint32_t>::max() || out_w > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Up_Sample_2d_Layer: Output dimensions exceed uint32_t range");
    }
    const uint64_t in_spatial = static_cast<uint64_t>(_height) * static_cast<uint64_t>(_width);
    const uint64_t out_spatial = out_h * out_w;
    if (in_spatial > std::numeric_limits<size_t>::max() / _channels ||
        out_spatial > std::numeric_limits<size_t>::max() / _channels)
    {
        throw std::overflow_error("Up_Sample_2d_Layer: Per-sample feature size overflows size_t");
    }
    return {static_cast<uint32_t>(out_h), static_cast<uint32_t>(out_w)};
}

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
      output_height(0),
      output_width(0),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
    auto [out_h, out_w] = validateAndComputeOutput(_height, _width, _channels, _scale_factor);
    output_height = out_h;
    output_width = out_w;
}

Tensor Up_Sample_2d_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateNhwcSpatialInput(
        _input_tensor, execution_target, input_height, input_width, channels, "Up_Sample_2d_Layer::forward");
    Logger::logMessage(Input_Format{"Up_Sample_2d_Layer::forward: in_h={}, in_w={}, ch={}, scale={}",
                                    input_height, input_width, channels, scale_factor},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    ops::vision::upsample2d(input_tensor, output_tensor, input_height, input_width, channels, scale_factor);
    is_forward_completed = true;
    return output_tensor;
}

std::unique_ptr<ILayer> Up_Sample_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Up_Sample_2d_Layer>(input_height, input_width, channels, scale_factor, execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

Tensor Up_Sample_2d_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage("Up_Sample_2d_Layer::backward: Forward must be called before backward",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::BACKWARD_PROPAGATION);
    }
    layer_internal::validateNhwcSpatialGrad(
        is_forward_completed, input_tensor, _output_gradient, output_height, output_width, channels,
        "Up_Sample_2d_Layer::backward", "Forward must be called before backward");

    Logger::logMessage(Input_Format{"Up_Sample_2d_Layer::backward: in_h={}, in_w={}, ch={}",
                                    input_height, input_width, channels},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::BACKWARD_PROPAGATION);

    ops::vision::upsample2dBackward(_output_gradient, input_gradient_tensor, input_height, input_width, channels, scale_factor);
    if (input_gradient_tensor.getShape() != input_tensor.getShape())
    {
        input_gradient_tensor.reshape(input_tensor.getShape());
    }
    return input_gradient_tensor;
}

void Up_Sample_2d_Layer::resetGradient() { is_forward_completed = false; }

void Up_Sample_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float>)
{
    throw std::out_of_range(std::format("Up_Sample_2d_Layer has no evolvable population parameters (index={})", param_index));
}

void Up_Sample_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Up_Sample_2d_Layer::saveConfiguration",
                              "Failed writing configuration",
                              input_height, input_width, channels, scale_factor);
}

void Up_Sample_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    saveConfiguration(_output_file_stream);
}

void Up_Sample_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
    layer_internal::requireInStream(_input_file_stream, "Up_Sample_2d_Layer::loadInference");
    uint32_t h = 0, w = 0, c = 0, sf = 0;
    layer_internal::readPods(_input_file_stream, "Up_Sample_2d_Layer::loadInference: Failed reading configuration from stream",
                             h, w, c, sf);
    std::pair<uint32_t, uint32_t> out_dims;
    try
    {
        out_dims = validateAndComputeOutput(h, w, c, sf);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("Up_Sample_2d_Layer::loadInference: Corrupted dimensions: ") + e.what());
    }
    input_height = h;
    input_width = w;
    channels = c;
    scale_factor = sf;
    output_height = out_dims.first;
    output_width = out_dims.second;
    is_forward_completed = false;
}

void Up_Sample_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const { saveConfiguration(_output_file_stream); }
void Up_Sample_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream) { loadInference(_input_file_stream); }

void Up_Sample_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, input_gradient_tensor);
}

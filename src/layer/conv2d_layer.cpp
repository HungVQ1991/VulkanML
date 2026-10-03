#include "layer/conv2d_layer.h"

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


void Conv2d_Layer::initializeWeights()
{
        size_t weight_count = kernel_size * kernel_size * input_channels * output_channels;
        std::vector<float> host_weights(weight_count);
        std::vector<float> host_biases(output_channels, 0.0f);

        float fan_in = static_cast<float>(kernel_size * kernel_size * input_channels);
        float standard_deviation = std::sqrt(2.0f / fan_in);

        std::mt19937 generator(std::random_device{}());
        std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

        for (size_t i = 0; i < weight_count; ++i)
        {
            host_weights[i] = normal_distribution(generator);
        }

        weights = Tensor(1, weight_count, host_weights, execution_target);
        biases = Tensor(1, output_channels, host_biases, execution_target);
        weights_gradient_tensor = Tensor(1, weight_count, execution_target);
        biases_gradient_tensor = Tensor(1, output_channels, execution_target);
    }

Conv2d_Layer::Conv2d_Layer(uint32_t _height, uint32_t _width, uint32_t _input_channels, uint32_t _output_channels, uint32_t _kernel_size, uint32_t _stride, uint32_t _padding, Execution_Target _execution_target)
    : input_height(_height),
          input_width(_width),
          input_channels(_input_channels),
          output_channels(_output_channels),
          kernel_size(_kernel_size),
          stride(_stride),
          padding(_padding),
          execution_target(_execution_target),
          is_forward_completed(false),
          weights(0, 0, _execution_target),
          biases(0, 0, _execution_target),
          weights_gradient_tensor(0, 0, _execution_target),
          biases_gradient_tensor(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          weights_fp16(0, 0, _execution_target),
          biases_fp16(0, 0, _execution_target),
          input_tensor_fp16(0, 0, _execution_target),
          output_tensor_fp16(0, 0, _execution_target),
          input_gradient_tensor_fp16(0, 0, _execution_target),
          output_gradient_tensor_fp16(0, 0, _execution_target),
          im2col_scratch(0, 0, _execution_target),
          im2col_scratch_fp16(0, 0, _execution_target)
{
        output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        initializeWeights();
    }

Tensor Conv2d_Layer::forward(const Tensor &_input_tensor)
{
        Logger::logMessage(Input_Format{"Conv2d_Layer::forward: input_height={}, input_width={}, input_channels={}, output_channels={}",
                                        input_height, input_width, input_channels, output_channels},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU)
        {
            if (input_tensor.getDataType() != Data_Type::FLOAT16)
            {
                input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
            }
            else
            {
                input_tensor_fp16 = input_tensor;
            }

            if (is_weights_fp16_dirty || weights_fp16.isEmpty() || biases_fp16.isEmpty())
            {
                weights.to(Data_Type::FLOAT16, weights_fp16);
                biases.to(Data_Type::FLOAT16, biases_fp16);
                is_weights_fp16_dirty = false;
            }

            input_tensor_fp16.conv2d(weights_fp16, biases_fp16, output_tensor_fp16, input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16);

            output_tensor = output_tensor_fp16;
        }
        else
        {
            input_tensor.conv2d(weights, biases, output_tensor, input_height, input_width, input_channels, output_channels, kernel_size, stride, padding);
        }
        is_forward_completed = true;
        return output_tensor;
    }

Tensor Conv2d_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Conv2d_Layer::forward: expected {} batched parameter tensors (weights, biases), got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::CONV2D_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        Tensor output(_batched_input.getExecutionTarget());
        _batched_input.conv2d(_batched_params[0],
            _batched_params[1],
            output,
            input_height,
            input_width,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding);
        return output;
    }

Tensor Conv2d_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Conv2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Conv2d_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU)
        {
            if (_output_gradient.getDataType() != Data_Type::FLOAT16)
            {
                _output_gradient.to(Data_Type::FLOAT16, output_gradient_tensor_fp16);
            }
            else
            {
                output_gradient_tensor_fp16 = _output_gradient;
            }

            if (!is_accumulated)
            {
                input_tensor_fp16.conv2dBackwardWeight(output_gradient_tensor_fp16, weights_gradient_tensor, biases_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16);
            }
            else
            {
                Tensor step_weights_grad(weights_gradient_tensor.getShape(), execution_target);
                Tensor step_biases_grad(biases_gradient_tensor.getShape(), execution_target);
                input_tensor_fp16.conv2dBackwardWeight(output_gradient_tensor_fp16, step_weights_grad, step_biases_grad, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16);
                weights_gradient_tensor = weights_gradient_tensor + step_weights_grad;
                biases_gradient_tensor = biases_gradient_tensor + step_biases_grad;
            }
            logBufferAddress(&input_tensor_fp16, "input_tensor_fp16 (Backward)");
            output_gradient_tensor_fp16.conv2dBackwardInput(weights_fp16, input_gradient_tensor_fp16, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);

            if (input_tensor.getDataType() == Data_Type::FLOAT16)
            {
                input_gradient_tensor = input_gradient_tensor_fp16;
            }
            else
            {
                input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
            }
            return input_gradient_tensor;
        }
        else
        {
            if (!is_accumulated)
            {
                input_tensor.conv2dBackwardWeight(_output_gradient, weights_gradient_tensor, biases_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch);
            }
            else
            {
                Tensor step_weights_grad(weights_gradient_tensor.getShape(), execution_target);
                Tensor step_biases_grad(biases_gradient_tensor.getShape(), execution_target);
                input_tensor.conv2dBackwardWeight(_output_gradient, step_weights_grad, step_biases_grad, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch);
                weights_gradient_tensor = weights_gradient_tensor + step_weights_grad;
                biases_gradient_tensor = biases_gradient_tensor + step_biases_grad;
            }
            logBufferAddress(&input_tensor, "input_tensor (Backward)");
            _output_gradient.conv2dBackwardInput(weights, input_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
            return input_gradient_tensor;
        }
    }

void Conv2d_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Conv2d_Layer::resetGradients()
{
        resetGradient();
        weights_gradient_tensor.zero();
        biases_gradient_tensor.zero();
    }

std::unique_ptr<ILayer> Conv2d_Layer::clone() const
{
        auto cloned = std::make_unique<Conv2d_Layer>(
            input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, execution_target);
        cloned->setMixedPrecision(is_mixed_precision_enabled);
        return cloned;
    }

void Conv2d_Layer::invalidateWeightCache() noexcept
{
        is_weights_fp16_dirty = true;
    }

void Conv2d_Layer::setMixedPrecision(bool _enable) noexcept
{
        ILayer::setMixedPrecision(_enable);
        is_weights_fp16_dirty = true;
    }

void Conv2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_channels), sizeof(input_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&output_channels), sizeof(output_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
        _output_file_stream.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
    }

void Conv2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
    }

void Conv2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
        weights = Tensor::loadTensor(_input_file_stream, execution_target);
        biases = Tensor::loadTensor(_input_file_stream, execution_target);
        is_weights_fp16_dirty = true;
    }

void Conv2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
        weights_gradient_tensor.saveTensor(_output_file_stream);
        biases_gradient_tensor.saveTensor(_output_file_stream);
    }

void Conv2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        weights = Tensor::loadTensor(_input_file_stream, execution_target);
        biases = Tensor::loadTensor(_input_file_stream, execution_target);
        weights_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        biases_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        is_weights_fp16_dirty = true;
    }

std::function<float(std::mt19937&)> Conv2d_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        if (param_index == 0)
        {
            float fan_in = static_cast<float>(kernel_size * kernel_size * input_channels);
            float standard_deviation = (fan_in > 0.0f) ? std::sqrt(2.0f / fan_in) : 0.0f;
            return [standard_deviation](std::mt19937& generator)
                {
                    std::normal_distribution<float> distribution(0.0f, standard_deviation);
                    return distribution(generator);
                };
        }
        else if (param_index == 1)
        {
            return [](std::mt19937&)
                {
                    return 0.0f;
                };
        }

        return [](std::mt19937&)
            {
                return 0.0f;
            };
    }

std::vector<float> Conv2d_Layer::getPopulationParameter(size_t param_index) const
{
        if (param_index == 0)
        {
            return weights.getData();
        }
        if (param_index == 1)
        {
            return biases.getData();
        }
        throw std::out_of_range("Conv2d_Layer::getPopulationParameter: Parameter index out of range");
    }

std::vector<std::pair<Tensor *, Tensor *>> Conv2d_Layer::getParametersAndGradients()
{ return {{&weights, &weights_gradient_tensor}, {&biases, &biases_gradient_tensor}}; }

void Conv2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        size_t weight_count = kernel_size * kernel_size * input_channels * output_channels;
        if (param_index == 0)
        {
            if (flat_data.size() != weight_count)
            {
                throw std::invalid_argument("Conv2d_Layer::setPopulationParameter: Weight size mismatch");
            }
            weights = Tensor(1, weight_count, std::move(flat_data), execution_target);
            is_weights_fp16_dirty = true;
        }
        else if (param_index == 1)
        {
            if (flat_data.size() != output_channels)
            {
                throw std::invalid_argument("Conv2d_Layer::setPopulationParameter: Bias size mismatch");
            }
            biases = Tensor(1, output_channels, std::move(flat_data), execution_target);
            is_weights_fp16_dirty = true;
        }
        else
        {
            throw std::out_of_range("Conv2d_Layer::setPopulationParameter: Parameter index out of range");
        }
    }

void Conv2d_Layer::setWeightsGradient(const Tensor &_tensor)
{ weights_gradient_tensor = _tensor; }

void Conv2d_Layer::setBiasesGradient(const Tensor &_tensor)
{ biases_gradient_tensor = _tensor; }

void Conv2d_Layer::setWeights(const Tensor &_new_weights)
{
        weights = _new_weights;
        is_weights_fp16_dirty = true;
    }

void Conv2d_Layer::setBiases(const Tensor &_new_biases)
{
        biases = _new_biases;
        is_weights_fp16_dirty = true;
    }

void Conv2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        weights.setExecutionTarget(_new_execution_target);
        biases.setExecutionTarget(_new_execution_target);
        weights_gradient_tensor.setExecutionTarget(_new_execution_target);
        biases_gradient_tensor.setExecutionTarget(_new_execution_target);
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
        weights_fp16.setExecutionTarget(_new_execution_target);
        biases_fp16.setExecutionTarget(_new_execution_target);
        input_tensor_fp16.setExecutionTarget(_new_execution_target);
        output_tensor_fp16.setExecutionTarget(_new_execution_target);
        input_gradient_tensor_fp16.setExecutionTarget(_new_execution_target);
        output_gradient_tensor_fp16.setExecutionTarget(_new_execution_target);
        im2col_scratch.setExecutionTarget(_new_execution_target);
        im2col_scratch_fp16.setExecutionTarget(_new_execution_target);
    }

void Conv2d_Layer::setOutputChannels(uint32_t _channels) noexcept
{ output_channels = _channels; }

void Conv2d_Layer::setInputChannels(uint32_t _channels) noexcept
{ input_channels = _channels; }

void Conv2d_Layer::setOutputHeight(uint32_t _height) noexcept
{ output_height = _height; }

void Conv2d_Layer::setOutputWidth(uint32_t _width) noexcept
{ output_width = _width; }

void Conv2d_Layer::setInputHeight(uint32_t _height) noexcept
{ input_height = _height; }

void Conv2d_Layer::setKernelSize(uint32_t _size) noexcept
{ kernel_size = _size; }

void Conv2d_Layer::setInputWidth(uint32_t _width) noexcept
{ input_width = _width; }

void Conv2d_Layer::setPadding(uint32_t _padding) noexcept
{ padding = _padding; }

void Conv2d_Layer::setStride(uint32_t _stride) noexcept
{ stride = _stride; }

void Conv2d_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

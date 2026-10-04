#include "layer/linear_layer.h"

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


Linear_Layer::Linear_Layer()
    : weights(0, 0),
          biases(0, 0),
          input_tensor(0, 0),
          output_tensor(0, 0),
          input_gradient_tensor(0, 0),
          weights_gradient_tensor(0, 0),
          biases_gradient_tensor(0, 0),
          is_forward_completed(false)
{}

Linear_Layer::Linear_Layer(size_t _input_dimension, size_t _output_dimension, Execution_Target _execution_target, float _initialization_gain, Data_Type _data_type)
    : weights(0, 0, _execution_target),
          biases(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          weights_gradient_tensor(_input_dimension, _output_dimension, _execution_target),
          biases_gradient_tensor(1, _output_dimension, _execution_target),
          input_dimension(_input_dimension),
          output_dimension(_output_dimension),
          initialization_gain(_initialization_gain),
          is_forward_completed(false),
          execution_target(_execution_target)
{
        if (_data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }

        std::vector<float> weight_data(_input_dimension * _output_dimension);
        std::vector<float> bias_data(_output_dimension, 0.0f);

        float standard_deviation = std::sqrt(_initialization_gain / static_cast<float>(_input_dimension));
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

        float weight_sum = 0.0f;
        for (float &weight_value : weight_data)
        {
            weight_value = normal_distribution(generator);
            weight_sum += std::abs(weight_value);
        }

        float mean_absolute_weight = weight_data.empty() ? 0.0f : weight_sum / static_cast<float>(weight_data.size());
        Logger::logMessage(Input_Format{"Linear_Layer::Linear_Layer: Layer weights mean_abs = {:.8f}", mean_absolute_weight},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::DENSE_COMPUTE);

        weights = Tensor(_input_dimension, _output_dimension, std::move(weight_data), execution_target);
        biases = Tensor(1, _output_dimension, std::move(bias_data), execution_target);
    }

Tensor Linear_Layer::forward(const Tensor &_input_tensor)
{
        size_t in_features = (_input_tensor.getShape().getRank() >= 2)
                             ? _input_tensor.getShape()[_input_tensor.getShape().getRank() - 1]
                             : _input_tensor.getColumns();
        if (in_features != input_dimension)
        {
            Logger::logMessage(Input_Format{"Linear_Layer::forward: Input dimension mismatch (expected {}, got {})",
                                            input_dimension, in_features},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument(std::format("Linear_Layer input dimension mismatch (expected {}, got {})", input_dimension, in_features));
        }

        cache_orig_shape = _input_tensor.getShape();
        cache_is_3d = (cache_orig_shape.getRank() == 3);

        Tensor eff_input = _input_tensor;
        if (cache_is_3d)
        {
            eff_input.reshape(Shape{ cache_orig_shape[0] * cache_orig_shape[1], input_dimension });
        }

        Logger::logMessage(Input_Format{"Linear_Layer::forward: batch_size={}, input_dimension={}, output_dimension={}",
                                         eff_input.getRows(),
                                         input_dimension,
                                         output_dimension},
                            Log_Level::LOG_DEBUG,
                            true,
                            1,
                            Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = eff_input;
        input_tensor.linearForward(weights, biases, output_tensor);

        if (cache_is_3d)
        {
            output_tensor.reshape(Shape{ cache_orig_shape[0], cache_orig_shape[1], output_dimension });
        }

        logBufferAddress(&weights, "weights (Forward)");
        logBufferAddress(&biases, "biases (Forward)");
        is_forward_completed = true;
        return output_tensor;
    }

Tensor Linear_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Linear_Layer::forward: expected {} batched parameter tensors, got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }
        return _batched_input.matmulAdd(_batched_params[0], _batched_params[1]);
    }

Tensor Linear_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Linear_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        size_t out_features = (_output_gradient.getShape().getRank() >= 2)
                              ? _output_gradient.getShape()[_output_gradient.getShape().getRank() - 1]
                              : _output_gradient.getColumns();
        if (out_features != output_dimension)
        {
            Logger::logMessage(Input_Format{"Linear_Layer::backward: Gradient output dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument(std::format("Linear_Layer gradient output dimension mismatch (expected {}, got {})", output_dimension, out_features));
        }

        Tensor eff_grad = _output_gradient;
        if (cache_is_3d && eff_grad.getShape().getRank() == 3)
        {
            eff_grad.reshape(Shape{ cache_orig_shape[0] * cache_orig_shape[1], output_dimension });
        }

        Logger::logMessage(Input_Format{"Linear_Layer::backward: output_gradient rows={}, columns={}",
                                        eff_grad.getRows(),
                                        eff_grad.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        eff_grad.linearBackwardInput(weights, input_gradient_tensor);
        input_tensor.linearBackwardWeightBias(eff_grad, weights_gradient_tensor, biases_gradient_tensor, is_accumulated);

        if (cache_is_3d)
        {
            input_gradient_tensor.reshape(cache_orig_shape);
        }

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        logBufferAddress(&output_tensor, "output_tensor (Backward)");
        logBufferAddress(const_cast<Tensor *>(&_output_gradient), "output_gradient (Backward)");

        return input_gradient_tensor;
    }

std::unique_ptr<ILayer> Linear_Layer::clone() const
{
        auto cloned = std::make_unique<Linear_Layer>(input_dimension, output_dimension, execution_target, initialization_gain);
        cloned->setMixedPrecision(is_mixed_precision_enabled);
        return cloned;
    }

void Linear_Layer::invalidateWeightCache() noexcept
{
        weights.invalidateFp16Cache();
        biases.invalidateFp16Cache();
    }

void Linear_Layer::setMixedPrecision(bool _enable) noexcept
{
        ILayer::setMixedPrecision(_enable);
        weights.invalidateFp16Cache();
        biases.invalidateFp16Cache();
    }

void Linear_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Linear_Layer::resetGradients()
{
        resetGradient();
        weights_gradient_tensor.zero();
        biases_gradient_tensor.zero();
    }

void Linear_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        uint32_t input_dimension_value = static_cast<uint32_t>(input_dimension);
        uint32_t output_dimension_value = static_cast<uint32_t>(output_dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&input_dimension_value), sizeof(input_dimension_value));
        _output_file_stream.write(reinterpret_cast<const char *>(&output_dimension_value), sizeof(output_dimension_value));
    }

void Linear_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
    }

void Linear_Layer::loadInference(std::ifstream &_input_file_stream)
{
        weights = Tensor::loadTensor(_input_file_stream, execution_target);
        biases = Tensor::loadTensor(_input_file_stream, execution_target);
        input_dimension = weights.getRows();
        output_dimension = weights.getColumns();
        invalidateWeightCache();
    }

void Linear_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
        weights_gradient_tensor.saveTensor(_output_file_stream);
        biases_gradient_tensor.saveTensor(_output_file_stream);
    }

void Linear_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        weights = Tensor::loadTensor(_input_file_stream, execution_target);
        biases = Tensor::loadTensor(_input_file_stream, execution_target);
        weights_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        biases_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        input_dimension = weights.getRows();
        output_dimension = weights.getColumns();
        invalidateWeightCache();
    }

std::function<float(std::mt19937&)> Linear_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        if (param_index == 0)
        {
            float standard_deviation = (input_dimension > 0) ? std::sqrt(initialization_gain / static_cast<float>(input_dimension)) : 0.0f;
            return [standard_deviation](std::mt19937& generator)
            {
                std::normal_distribution<float> distribution(0.0f, standard_deviation);
                return distribution(generator);
            };
        }
        return [](std::mt19937&)
        {
            return 0.0f;
        };
    }

std::vector<float> Linear_Layer::getPopulationParameter(size_t param_index) const
{
        if (param_index == 0)
        {
            return weights.getData();
        }
        if (param_index == 1)
        {
            return biases.getData();
        }
        throw std::out_of_range("Linear_Layer::getPopulationParameter: Parameter index out of range");
    }

std::vector<std::pair<Tensor *, Tensor *>> Linear_Layer::getParametersAndGradients()
{ return {{&weights, &weights_gradient_tensor}, {&biases, &biases_gradient_tensor}}; }

float Linear_Layer::getInitializationGain() const noexcept
{ return initialization_gain; }

void Linear_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        if (param_index == 0)
        {
            if (flat_data.size() != input_dimension * output_dimension)
            {
                throw std::invalid_argument("Linear_Layer::setPopulationParameter: Weight size mismatch");
            }
            weights = Tensor(input_dimension, output_dimension, std::move(flat_data), execution_target);
            invalidateWeightCache();
        }
        else if (param_index == 1)
        {
            if (flat_data.size() != output_dimension)
            {
                throw std::invalid_argument("Linear_Layer::setPopulationParameter: Bias size mismatch");
            }
            biases = Tensor(1, output_dimension, std::move(flat_data), execution_target);
            invalidateWeightCache();
        }
        else
        {
            throw std::out_of_range("Linear_Layer::setPopulationParameter: Parameter index out of range");
        }
    }

void Linear_Layer::setWeights(const Tensor &_new_weights)
{
        if (_new_weights.getRows() != input_dimension || _new_weights.getColumns() != output_dimension)
        {
            Logger::logMessage(Input_Format{"Linear_Layer::setWeights: Dimension size of weight must match"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Dimension size of weight must match");
        }
        weights = _new_weights;
        invalidateWeightCache();
    }

void Linear_Layer::setBiases(const Tensor &_new_biases)
{
        if (_new_biases.getRows() != 1 || _new_biases.getColumns() != output_dimension)
        {
            Logger::logMessage(Input_Format{"Linear_Layer::setBiases: Dimension size of bias must match"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Dimension size of bias must match");
        }
        biases = _new_biases;
        invalidateWeightCache();
    }

void Linear_Layer::setWeightsGradient(const Tensor &_tensor)
{ weights_gradient_tensor = _tensor; }

void Linear_Layer::setBiasesGradient(const Tensor &_tensor)
{ biases_gradient_tensor = _tensor; }

void Linear_Layer::setOutputDimension(size_t _output_dimension) noexcept
{ output_dimension = _output_dimension; }

void Linear_Layer::setInputDimension(size_t _input_dimension) noexcept
{ input_dimension = _input_dimension; }

void Linear_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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
    }

void Linear_Layer::setInitializationGain(float _initialization_gain) noexcept
{ initialization_gain = _initialization_gain; }

void Linear_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

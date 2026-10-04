#include "layer/batch_norm_layer.h"

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


void Batch_Norm_Layer::initializeParameters()
{
        std::vector<float> gamma_data(input_dimension, 1.0f);
        std::vector<float> beta_data(input_dimension, 0.0f);
        std::vector<float> mean_data(input_dimension, 0.0f);
        std::vector<float> variance_data(input_dimension, 1.0f);

        gamma = Tensor(1, input_dimension, std::move(gamma_data), execution_target);
        beta = Tensor(1, input_dimension, std::move(beta_data), execution_target);
        gamma_gradient_tensor = Tensor(1, input_dimension, execution_target);
        beta_gradient_tensor = Tensor(1, input_dimension, execution_target);

        running_mean = Tensor(1, input_dimension, std::move(mean_data), execution_target);
        running_variance = Tensor(1, input_dimension, std::move(variance_data), execution_target);

        batch_mean = Tensor(1, input_dimension, execution_target);
        batch_variance = Tensor(1, input_dimension, execution_target);
    }

Batch_Norm_Layer::Batch_Norm_Layer(size_t _dimension, float _epsilon, float _momentum, Execution_Target _execution_target)
    : input_dimension(_dimension),
          epsilon(_epsilon),
          momentum(_momentum),
          is_training(true),
          gamma(0, 0, _execution_target),
          beta(0, 0, _execution_target),
          gamma_gradient_tensor(0, 0, _execution_target),
          beta_gradient_tensor(0, 0, _execution_target),
          running_mean(0, 0, _execution_target),
          running_variance(0, 0, _execution_target),
          batch_mean(0, 0, _execution_target),
          batch_variance(0, 0, _execution_target),
          normalized_input(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{
        initializeParameters();
    }

Tensor Batch_Norm_Layer::forward(const Tensor &_input_tensor)
{
        if (_input_tensor.getColumns() != input_dimension)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_Layer::forward: Input dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Input dimension mismatch");
        }

        Logger::logMessage(Input_Format{"Batch_Norm_Layer::forward: dimension={}, mode={}",
                                        input_dimension,
                                        is_training ? "Train" : "Eval"},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        if (input_tensor.getExecutionTarget() != execution_target)
        {
            input_tensor.setExecutionTarget(execution_target);
        }

        input_tensor.batchNormForward(
            gamma,
            beta,
            running_mean,
            running_variance,
            batch_mean,
            batch_variance,
            normalized_input,
            output_tensor,
            epsilon,
            momentum,
            is_training);

        is_forward_completed = true;
        return output_tensor;
    }

Tensor Batch_Norm_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Batch_Norm_Layer::forward: expected {} batched parameter tensors (gamma, beta, running_mean, running_variance), got {}",
                                             getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }
        Tensor batch_mean_tensor(getExecutionTarget());
        Tensor batch_variance_tensor(getExecutionTarget());
        Tensor normalized_input_tensor(getExecutionTarget());
        Tensor output_tensor_result(getExecutionTarget());

        _batched_input.batchNormForward(
            _batched_params[0],
            _batched_params[1],
            const_cast<Tensor &>(_batched_params[2]),
            const_cast<Tensor &>(_batched_params[3]),
            batch_mean_tensor,
            batch_variance_tensor,
            normalized_input_tensor,
            output_tensor_result,
            epsilon,
            momentum,
            is_training);

        return output_tensor_result;
    }

Tensor Batch_Norm_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Batch_Norm_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        if (!is_accumulated)
        {
            input_tensor.batchNormBackward(
                _output_gradient,
                gamma,
                batch_variance,
                normalized_input,
                gamma_gradient_tensor,
                beta_gradient_tensor,
                input_gradient_tensor,
                epsilon);
        }
        else
        {
            Tensor step_gamma_grad(gamma_gradient_tensor.getShape(), execution_target);
            Tensor step_beta_grad(beta_gradient_tensor.getShape(), execution_target);
            input_tensor.batchNormBackward(
                _output_gradient,
                gamma,
                batch_variance,
                normalized_input,
                step_gamma_grad,
                step_beta_grad,
                input_gradient_tensor,
                epsilon);
            gamma_gradient_tensor = gamma_gradient_tensor + step_gamma_grad;
            beta_gradient_tensor = beta_gradient_tensor + step_beta_grad;
        }

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        return input_gradient_tensor;
    }

std::unique_ptr<ILayer> Batch_Norm_Layer::clone() const
{
        return std::make_unique<Batch_Norm_Layer>(input_dimension, epsilon, momentum, execution_target);
    }

void Batch_Norm_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Batch_Norm_Layer::resetGradients()
{
        resetGradient();
        gamma_gradient_tensor.zero();
        beta_gradient_tensor.zero();
    }

void Batch_Norm_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        uint32_t feature_count = static_cast<uint32_t>(input_dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
    }

void Batch_Norm_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
    }

void Batch_Norm_Layer::loadInference(std::ifstream &_input_file_stream)
{
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        beta = Tensor::loadTensor(_input_file_stream, execution_target);
        running_mean = Tensor::loadTensor(_input_file_stream, execution_target);
        running_variance = Tensor::loadTensor(_input_file_stream, execution_target);
    }

void Batch_Norm_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
        gamma_gradient_tensor.saveTensor(_output_file_stream);
        beta_gradient_tensor.saveTensor(_output_file_stream);
    }

void Batch_Norm_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        beta = Tensor::loadTensor(_input_file_stream, execution_target);
        running_mean = Tensor::loadTensor(_input_file_stream, execution_target);
        running_variance = Tensor::loadTensor(_input_file_stream, execution_target);
        gamma_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        beta_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
    }

std::function<float(std::mt19937&)> Batch_Norm_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        if (param_index == 0 || param_index == 3)
        {
            return [](std::mt19937&)
                {
                    return 1.0f;
                };
        }
        else if (param_index == 1 || param_index == 2)
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

std::vector<float> Batch_Norm_Layer::getPopulationParameter(size_t param_index) const
{
        switch (param_index)
        {
        case 0:
            return gamma.getData();
        case 1:
            return beta.getData();
        case 2:
            return running_mean.getData();
        case 3:
            return running_variance.getData();
        default:
            throw std::out_of_range("Batch_Norm_Layer::getPopulationParameter: Parameter index out of range");
        }
    }

std::vector<std::pair<Tensor *, Tensor *>> Batch_Norm_Layer::getParametersAndGradients()
{ return {{&gamma, &gamma_gradient_tensor}, {&beta, &beta_gradient_tensor}}; }

const Tensor & Batch_Norm_Layer::getNormalizedInput() const noexcept
{ return normalized_input; }

const Tensor & Batch_Norm_Layer::getBatchVariance() const noexcept
{ return batch_variance; }

const Tensor & Batch_Norm_Layer::getBatchMean() const noexcept
{ return batch_mean; }

bool Batch_Norm_Layer::isTraining() const noexcept
{ return is_training; }

void Batch_Norm_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        if (flat_data.size() != input_dimension)
        {
            throw std::invalid_argument("Batch_Norm_Layer::setPopulationParameter: Dimension mismatch");
        }

        switch (param_index)
        {
        case 0:
            gamma = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 1:
            beta = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 2:
            running_mean = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 3:
            running_variance = Tensor(1, input_dimension, std::move(flat_data), execution_target);
            break;
        default:
            throw std::out_of_range("Batch_Norm_Layer::setPopulationParameter: Parameter index out of range");
        }
    }

void Batch_Norm_Layer::setGammaGradient(const Tensor &_tensor)
{ gamma_gradient_tensor = _tensor; }

void Batch_Norm_Layer::setBetaGradient(const Tensor &_tensor)
{ beta_gradient_tensor = _tensor; }

void Batch_Norm_Layer::setRunningVariance(const Tensor &_tensor)
{ running_variance = _tensor; }

void Batch_Norm_Layer::setNormalizedInput(const Tensor &_tensor)
{ normalized_input = _tensor; }

void Batch_Norm_Layer::setBatchVariance(const Tensor &_tensor)
{ batch_variance = _tensor; }

void Batch_Norm_Layer::setRunningMean(const Tensor &_tensor)
{ running_mean = _tensor; }

void Batch_Norm_Layer::setBatchMean(const Tensor &_tensor)
{ batch_mean = _tensor; }

void Batch_Norm_Layer::setGamma(const Tensor &_tensor)
{ gamma = _tensor; }

void Batch_Norm_Layer::setBeta(const Tensor &_tensor)
{ beta = _tensor; }

void Batch_Norm_Layer::setInputDimension(size_t _dimension) noexcept
{ input_dimension = _dimension; }

void Batch_Norm_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        gamma.setExecutionTarget(_new_execution_target);
        beta.setExecutionTarget(_new_execution_target);
        gamma_gradient_tensor.setExecutionTarget(_new_execution_target);
        beta_gradient_tensor.setExecutionTarget(_new_execution_target);
        running_mean.setExecutionTarget(_new_execution_target);
        running_variance.setExecutionTarget(_new_execution_target);
        batch_mean.setExecutionTarget(_new_execution_target);
        batch_variance.setExecutionTarget(_new_execution_target);
        normalized_input.setExecutionTarget(_new_execution_target);
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }

void Batch_Norm_Layer::setMomentum(float _momentum) noexcept
{ momentum = _momentum; }

void Batch_Norm_Layer::setEpsilon(float _epsilon) noexcept
{ epsilon = _epsilon; }

void Batch_Norm_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

void Batch_Norm_Layer::setTrainingMode(bool _is_training)
{ is_training = _is_training; }

void Batch_Norm_Layer::setIsTraining(bool _is_training) noexcept
{ is_training = _is_training; }

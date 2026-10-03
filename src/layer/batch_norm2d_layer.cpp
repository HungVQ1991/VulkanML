#include "layer/batch_norm2d_layer.h"

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


Batch_Norm_2d_Layer::Batch_Norm_2d_Layer(uint32_t _height, uint32_t _width, uint32_t _channels, float _epsilon, float _momentum, Execution_Target _execution_target)
    : input_height(_height),
          input_width(_width),
          channels(_channels),
          epsilon(_epsilon),
          momentum(_momentum),
          is_training(true),
          execution_target(_execution_target),
          gamma(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target),
          beta(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target),
          gamma_gradient_tensor(1, _channels, _execution_target),
          beta_gradient_tensor(1, _channels, _execution_target),
          running_mean(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target),
          running_variance(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target),
          batch_mean(1, _channels, _execution_target),
          batch_variance(1, _channels, _execution_target),
          normalized_input(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false)
{
    }

Tensor Batch_Norm_2d_Layer::forward(const Tensor &_input_tensor)
{
        input_tensor = _input_tensor;

        input_tensor.batchNorm2dForward(
            gamma,
            beta,
            running_mean,
            running_variance,
            batch_mean,
            batch_variance,
            normalized_input,
            output_tensor,
            input_height,
            input_width,
            channels,
            epsilon,
            momentum,
            is_training);

        is_forward_completed = true;
        logBufferAddress(&input_tensor, "input_tensor (Forward)");
        logBufferAddress(&output_tensor, "output_tensor (Forward)");
        return output_tensor;
    }

Tensor Batch_Norm_2d_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Batch_Norm_2d_Layer::forward: expected {} batched parameter tensors (gamma, beta, running_mean, running_variance), got {}",
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

        _batched_input.batchNorm2dForward(
            _batched_params[0],
            _batched_params[1],
            const_cast<Tensor&>(_batched_params[2]),
            const_cast<Tensor&>(_batched_params[3]),
            batch_mean_tensor,
            batch_variance_tensor,
            normalized_input_tensor,
            output_tensor_result,
            input_height,
            input_width,
            channels,
            epsilon,
            momentum,
            is_training);

        return output_tensor_result;
    }

std::unique_ptr<ILayer> Batch_Norm_2d_Layer::clone() const
{
        return std::make_unique<Batch_Norm_2d_Layer>(
            input_height, input_width, channels, epsilon, momentum, execution_target);
    }

Tensor Batch_Norm_2d_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        if (!is_accumulated)
        {
            _output_gradient.batchNorm2dBackward(
                gamma,
                batch_variance,
                normalized_input,
                gamma_gradient_tensor,
                beta_gradient_tensor,
                input_gradient_tensor,
                input_height,
                input_width,
                channels,
                epsilon);
        }
        else
        {
            Tensor step_gamma_grad(gamma_gradient_tensor.getShape(), execution_target);
            Tensor step_beta_grad(beta_gradient_tensor.getShape(), execution_target);
            _output_gradient.batchNorm2dBackward(
                gamma,
                batch_variance,
                normalized_input,
                step_gamma_grad,
                step_beta_grad,
                input_gradient_tensor,
                input_height,
                input_width,
                channels,
                epsilon);
            gamma_gradient_tensor = gamma_gradient_tensor + step_gamma_grad;
            beta_gradient_tensor = beta_gradient_tensor + step_beta_grad;
        }

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        return input_gradient_tensor;
    }

void Batch_Norm_2d_Layer::resetGradient()
{
        is_forward_completed = false;
    }

void Batch_Norm_2d_Layer::resetGradients()
{
        resetGradient();
        gamma_gradient_tensor.zero();
        beta_gradient_tensor.zero();
    }

void Batch_Norm_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
    }

void Batch_Norm_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
        gamma_gradient_tensor.saveTensor(_output_file_stream);
        beta_gradient_tensor.saveTensor(_output_file_stream);
    }

void Batch_Norm_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        beta = Tensor::loadTensor(_input_file_stream, execution_target);
        running_mean = Tensor::loadTensor(_input_file_stream, execution_target);
        running_variance = Tensor::loadTensor(_input_file_stream, execution_target);
        gamma_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        beta_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
    }

void Batch_Norm_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
    }

void Batch_Norm_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        beta = Tensor::loadTensor(_input_file_stream, execution_target);
        running_mean = Tensor::loadTensor(_input_file_stream, execution_target);
        running_variance = Tensor::loadTensor(_input_file_stream, execution_target);
    }

std::function<float(std::mt19937&)> Batch_Norm_2d_Layer::getPopulationParameterInitializer(size_t param_index) const
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

std::vector<float> Batch_Norm_2d_Layer::getPopulationParameter(size_t param_index) const
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
            throw std::out_of_range("Batch_Norm_2d_Layer::getPopulationParameter: Parameter index out of range");
        }
    }

std::vector<std::pair<Tensor *, Tensor *>> Batch_Norm_2d_Layer::getParametersAndGradients()
{ return {{&gamma, &gamma_gradient_tensor}, {&beta, &beta_gradient_tensor}}; }

const Tensor & Batch_Norm_2d_Layer::getNormalizedInput() const noexcept
{ return normalized_input; }

const Tensor & Batch_Norm_2d_Layer::getBatchVariance() const noexcept
{ return batch_variance; }

const Tensor & Batch_Norm_2d_Layer::getBatchMean() const noexcept
{ return batch_mean; }

uint32_t Batch_Norm_2d_Layer::getChannels() const noexcept
{ return channels; }

bool Batch_Norm_2d_Layer::isTraining() const noexcept
{ return is_training; }

void Batch_Norm_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        if (flat_data.size() != channels)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::setPopulationParameter: Dimension mismatch");
        }

        switch (param_index)
        {
        case 0:
            gamma = Tensor(1, channels, std::move(flat_data), execution_target);
            break;
        case 1:
            beta = Tensor(1, channels, std::move(flat_data), execution_target);
            break;
        case 2:
            running_mean = Tensor(1, channels, std::move(flat_data), execution_target);
            break;
        case 3:
            running_variance = Tensor(1, channels, std::move(flat_data), execution_target);
            break;
        default:
            throw std::out_of_range("Batch_Norm_2d_Layer::setPopulationParameter: Parameter index out of range");
        }
    }

void Batch_Norm_2d_Layer::setGammaGradient(const Tensor &_tensor)
{ gamma_gradient_tensor = _tensor; }

void Batch_Norm_2d_Layer::setBetaGradient(const Tensor &_tensor)
{ beta_gradient_tensor = _tensor; }

void Batch_Norm_2d_Layer::setRunningVariance(const Tensor &_tensor)
{ running_variance = _tensor; }

void Batch_Norm_2d_Layer::setNormalizedInput(const Tensor &_tensor)
{ normalized_input = _tensor; }

void Batch_Norm_2d_Layer::setBatchVariance(const Tensor &_tensor)
{ batch_variance = _tensor; }

void Batch_Norm_2d_Layer::setRunningMean(const Tensor &_tensor)
{ running_mean = _tensor; }

void Batch_Norm_2d_Layer::setBatchMean(const Tensor &_tensor)
{ batch_mean = _tensor; }

void Batch_Norm_2d_Layer::setGamma(const Tensor &_tensor)
{ gamma = _tensor; }

void Batch_Norm_2d_Layer::setBeta(const Tensor &_tensor)
{ beta = _tensor; }

void Batch_Norm_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

void Batch_Norm_2d_Layer::setInputHeight(uint32_t _height) noexcept
{ input_height = _height; }

void Batch_Norm_2d_Layer::setInputWidth(uint32_t _width) noexcept
{ input_width = _width; }

void Batch_Norm_2d_Layer::setChannels(uint32_t _channels) noexcept
{ channels = _channels; }

void Batch_Norm_2d_Layer::setMomentum(float _momentum) noexcept
{ momentum = _momentum; }

void Batch_Norm_2d_Layer::setEpsilon(float _epsilon) noexcept
{ epsilon = _epsilon; }

void Batch_Norm_2d_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

void Batch_Norm_2d_Layer::setTrainingMode(bool _is_training)
{ is_training = _is_training; }

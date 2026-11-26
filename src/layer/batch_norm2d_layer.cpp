#include "layer/batch_norm2d_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "math/tensor_impl.h"
#include "ops/normalization/batchnorm.h"

Batch_Norm_2d_Layer::Batch_Norm_2d_Layer(uint32_t _height, uint32_t _width, uint32_t _channels, float _epsilon, float _momentum, Execution_Target _execution_target)
    : input_height(_height),
          input_width(_width),
          channels(_channels),
          epsilon(_epsilon),
          momentum(_momentum),
          is_training(true),
          execution_target(_execution_target),
          gamma(0, 0, _execution_target),
          beta(0, 0, _execution_target),
          gamma_gradient_tensor(0, 0, _execution_target),
          beta_gradient_tensor(0, 0, _execution_target),
          step_gamma_grad_scratch(0, 0, _execution_target),
          step_beta_grad_scratch(0, 0, _execution_target),
          running_mean(0, 0, _execution_target),
          running_variance(0, 0, _execution_target),
          batch_mean(0, 0, _execution_target),
          batch_variance(0, 0, _execution_target),
          normalized_input(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          input_tensor_fp16(0, 0, _execution_target),
          output_gradient_fp16(0, 0, _execution_target),
          input_gradient_tensor_fp16(0, 0, _execution_target),
          is_forward_completed(false)
{
        if (_height == 0 || _width == 0 || _channels == 0)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer: height, width, and channels must be > 0");
        }
        if (!std::isfinite(_epsilon) || _epsilon <= 0.0f)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer: epsilon must be finite and > 0");
        }
        if (!std::isfinite(_momentum) || _momentum < 0.0f || _momentum > 1.0f)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer: momentum must be finite and in [0, 1]");
        }
        if (static_cast<uint64_t>(_height) * _width * _channels > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Batch_Norm_2d_Layer: H * W * C exceeds uint32_t limit");
        }

        gamma = Tensor(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target);
        beta = Tensor(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target);
        gamma_gradient_tensor = Tensor(1, _channels, _execution_target);
        beta_gradient_tensor = Tensor(1, _channels, _execution_target);
        gamma_gradient_tensor.zero();
        beta_gradient_tensor.zero();
        running_mean = Tensor(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target);
        running_variance = Tensor(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target);
        batch_mean = Tensor(1, _channels, _execution_target);
        batch_variance = Tensor(1, _channels, _execution_target);
    }

Tensor Batch_Norm_2d_Layer::forward(const Tensor &_input_tensor)
{
        if (input_height == 0 || input_width == 0 || channels == 0)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::forward: height, width, and channels must be > 0");
        }
        if (!std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::forward: epsilon must be finite and > 0");
        }
        uint64_t tot_feat = static_cast<uint64_t>(input_height) * input_width * channels;
        if (tot_feat > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Batch_Norm_2d_Layer::forward: H * W * C exceeds uint32_t limit");
        }
        if (!_input_tensor.getImplementation())
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::forward: Input tensor has null implementation");
        }
        if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::forward: TERNARY_INT2 input is not supported");
        }
        if (_input_tensor.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::forward: Execution_Target mismatch");
        }

        const Shape in_shape = _input_tensor.getShape();
        size_t rank = in_shape.getRank();
        bool valid_shape = false;
        if (rank == 2)
        {
            valid_shape = (in_shape[0] > 0 && in_shape[1] == tot_feat);
        }
        else if (rank == 4)
        {
            valid_shape = (in_shape[0] > 0 && in_shape[1] == input_height && in_shape[2] == input_width && in_shape[3] == channels);
        }
        if (!valid_shape)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::forward: Input shape must be {N, H*W*C} or {N, H, W, C} with N > 0");
        }
        uint64_t total_elems = static_cast<uint64_t>(in_shape[0]) * tot_feat;
        if (total_elems > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Batch_Norm_2d_Layer::forward: Input element count exceeds uint32_t limit");
        }

        cached_input_shape = in_shape;
        input_tensor = _input_tensor;

        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *eff_input = &input_tensor;
        if (use_amp)
        {
            if (input_tensor.getDataType() != Data_Type::FLOAT16)
            {
                input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                eff_input = &input_tensor_fp16;
            }
            else
            {
                input_tensor_fp16 = input_tensor;
            }
        }

        ops::normalization::batchNorm2dForward(
            *eff_input,
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

        if (cached_input_shape.getRank() == 4)
        {
            output_tensor.reshape(cached_input_shape);
        }

        last_forward_training = is_training;
        is_forward_completed = true;
        logBufferAddress(&input_tensor, "input_tensor (Forward)");
        logBufferAddress(&output_tensor, "output_tensor (Forward)");
        return output_tensor;
    }

Tensor Batch_Norm_2d_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
        (void)_batched_input;
        (void)_batched_params;
        throw std::logic_error("Batch_Norm_2d_Layer::forward: Population batch execution is not supported");
    }

std::unique_ptr<ILayer> Batch_Norm_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Batch_Norm_2d_Layer>(
        input_height, input_width, channels, epsilon, momentum, execution_target);
    cloned->gamma = gamma.clone();
    cloned->beta = beta.clone();
    cloned->running_mean = running_mean.clone();
    cloned->running_variance = running_variance.clone();
    cloned->is_training = is_training;
    cloned->setAccumulated(is_accumulated);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    return cloned;
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
        if (input_height == 0 || input_width == 0 || channels == 0 || !std::isfinite(epsilon) || epsilon <= 0.0f)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::backward: Invalid layer geometry or epsilon");
        }
        if (!_output_gradient.getImplementation())
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::backward: Output gradient has null implementation");
        }
        if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::backward: TERNARY_INT2 gradient is not supported");
        }
        if (_output_gradient.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::backward: Execution_Target mismatch");
        }
        if (_output_gradient.getShape() != output_tensor.getShape())
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::backward: Output gradient shape mismatch");
        }

        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *eff_grad = &_output_gradient;
        if (use_amp)
        {
            if (_output_gradient.getDataType() != Data_Type::FLOAT16)
            {
                _output_gradient.to(Data_Type::FLOAT16, output_gradient_fp16);
                eff_grad = &output_gradient_fp16;
            }
        }
        Tensor &target_in_grad = use_amp ? input_gradient_tensor_fp16 : input_gradient_tensor;
        auto finalize_in_grad = [&]() -> Tensor {
            if (use_amp)
            {
                if (input_tensor.getDataType() == Data_Type::FLOAT16)
                {
                    input_gradient_tensor = input_gradient_tensor_fp16;
                }
                else
                {
                    input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
                }
            }
            if (cached_input_shape.getRank() == 4)
            {
                input_gradient_tensor.reshape(cached_input_shape);
            }
            logBufferAddress(&input_tensor, "input_tensor (Backward)");
            return input_gradient_tensor;
        };

        const bool train_mode_for_bwd = last_forward_training;
        if (!is_accumulated)
        {
            ops::normalization::batchNorm2dBackward(
                *eff_grad,
                gamma,
                batch_variance,
                normalized_input,
                gamma_gradient_tensor,
                beta_gradient_tensor,
                target_in_grad,
                input_height,
                input_width,
                channels,
                epsilon,
                train_mode_for_bwd,
                running_variance);
        }
        else
        {
            ops::normalization::batchNorm2dBackward(
                *eff_grad,
                gamma,
                batch_variance,
                normalized_input,
                step_gamma_grad_scratch,
                step_beta_grad_scratch,
                target_in_grad,
                input_height,
                input_width,
                channels,
                epsilon,
                train_mode_for_bwd,
                running_variance);
            if (!gamma_gradient_tensor.getImplementation() ||
                gamma_gradient_tensor.getExecutionTarget() != execution_target ||
                gamma_gradient_tensor.getDataType() != Data_Type::FLOAT32 ||
                gamma_gradient_tensor.getShape() != step_gamma_grad_scratch.getShape())
            {
                gamma_gradient_tensor = Tensor(step_gamma_grad_scratch.getShape(), Data_Type::FLOAT32, execution_target);
                gamma_gradient_tensor.zero();
            }
            if (!beta_gradient_tensor.getImplementation() ||
                beta_gradient_tensor.getExecutionTarget() != execution_target ||
                beta_gradient_tensor.getDataType() != Data_Type::FLOAT32 ||
                beta_gradient_tensor.getShape() != step_beta_grad_scratch.getShape())
            {
                beta_gradient_tensor = Tensor(step_beta_grad_scratch.getShape(), Data_Type::FLOAT32, execution_target);
                beta_gradient_tensor.zero();
            }
            gamma_gradient_tensor.add(step_gamma_grad_scratch, gamma_gradient_tensor);
            beta_gradient_tensor.add(step_beta_grad_scratch, beta_gradient_tensor);
        }

        return finalize_in_grad();
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
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::saveConfiguration: Output stream is not open or writable");
        }
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::saveConfiguration: Failed writing configuration");
        }
    }

void Batch_Norm_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::saveCheckpoint: Output stream is not open or writable");
        }
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
        gamma_gradient_tensor.saveTensor(_output_file_stream);
        beta_gradient_tensor.saveTensor(_output_file_stream);
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::saveCheckpoint: Failed writing checkpoint state");
        }
    }

void Batch_Norm_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::loadCheckpoint: Input stream is not open or readable");
        }
        Tensor loaded_g = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_rm = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_rv = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_gg = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_bg = Tensor::loadTensor(_input_file_stream, execution_target);
        if (!_input_file_stream)
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::loadCheckpoint: Failed reading checkpoint state");
        }
        validateChannelTensor(loaded_g, "loadCheckpoint(gamma)");
        validateChannelTensor(loaded_b, "loadCheckpoint(beta)");
        validateChannelTensor(loaded_rm, "loadCheckpoint(running_mean)");
        validateChannelTensor(loaded_rv, "loadCheckpoint(running_variance)");
        validateChannelTensor(loaded_gg, "loadCheckpoint(gamma_gradient)");
        validateChannelTensor(loaded_bg, "loadCheckpoint(beta_gradient)");
        gamma = std::move(loaded_g);
        beta = std::move(loaded_b);
        running_mean = std::move(loaded_rm);
        running_variance = std::move(loaded_rv);
        gamma_gradient_tensor = std::move(loaded_gg);
        beta_gradient_tensor = std::move(loaded_bg);
        invalidateWeightCache();
        sanitizeRunningStats();
    }

void Batch_Norm_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open() || !_output_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::saveInference: Output stream is not open or writable");
        }
        gamma.saveTensor(_output_file_stream);
        beta.saveTensor(_output_file_stream);
        running_mean.saveTensor(_output_file_stream);
        running_variance.saveTensor(_output_file_stream);
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::saveInference: Failed writing inference state");
        }
    }

void Batch_Norm_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
        if (!_input_file_stream.is_open() || !_input_file_stream.good())
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::loadInference: Input stream is not open or readable");
        }
        Tensor loaded_g = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_rm = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_rv = Tensor::loadTensor(_input_file_stream, execution_target);
        if (!_input_file_stream)
        {
            throw std::runtime_error("Batch_Norm_2d_Layer::loadInference: Failed reading inference state");
        }
        validateChannelTensor(loaded_g, "loadInference(gamma)");
        validateChannelTensor(loaded_b, "loadInference(beta)");
        validateChannelTensor(loaded_rm, "loadInference(running_mean)");
        validateChannelTensor(loaded_rv, "loadInference(running_variance)");
        gamma = std::move(loaded_g);
        beta = std::move(loaded_b);
        running_mean = std::move(loaded_rm);
        running_variance = std::move(loaded_rv);
        invalidateWeightCache();
        sanitizeRunningStats();
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

void Batch_Norm_2d_Layer::validateChannelTensor(const Tensor &_tensor, const char *_setter_name) const
{
        if (!_tensor.getImplementation())
        {
            throw std::invalid_argument(std::format("Batch_Norm_2d_Layer::{}: Tensor has null implementation", _setter_name));
        }
        if (_tensor.getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument(std::format("Batch_Norm_2d_Layer::{}: Execution_Target mismatch", _setter_name));
        }
        if (_tensor.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument(std::format("Batch_Norm_2d_Layer::{}: TERNARY_INT2 is not supported", _setter_name));
        }
        if (_tensor.getTotalElements() != channels)
        {
            throw std::invalid_argument(std::format(
                "Batch_Norm_2d_Layer::{}: Element count mismatch (expected {}, got {})",
                _setter_name, channels, _tensor.getTotalElements()));
        }
    }

void Batch_Norm_2d_Layer::setGammaGradient(const Tensor &_tensor)
{
        validateChannelTensor(_tensor, "setGammaGradient");
        gamma_gradient_tensor = _tensor;
    }

void Batch_Norm_2d_Layer::setBetaGradient(const Tensor &_tensor)
{
        validateChannelTensor(_tensor, "setBetaGradient");
        beta_gradient_tensor = _tensor;
    }

void Batch_Norm_2d_Layer::setRunningVariance(const Tensor &_tensor)
{
        validateChannelTensor(_tensor, "setRunningVariance");
        running_variance = _tensor;
    }

void Batch_Norm_2d_Layer::setNormalizedInput(const Tensor &_tensor)
{ normalized_input = _tensor; }

void Batch_Norm_2d_Layer::setBatchVariance(const Tensor &_tensor)
{ batch_variance = _tensor; }

void Batch_Norm_2d_Layer::setRunningMean(const Tensor &_tensor)
{
        validateChannelTensor(_tensor, "setRunningMean");
        running_mean = _tensor;
    }

void Batch_Norm_2d_Layer::setBatchMean(const Tensor &_tensor)
{ batch_mean = _tensor; }

void Batch_Norm_2d_Layer::setGamma(const Tensor &_tensor)
{
        validateChannelTensor(_tensor, "setGamma");
        gamma = _tensor;
        gamma.invalidateFp16Cache();
    }

void Batch_Norm_2d_Layer::setBeta(const Tensor &_tensor)
{
        validateChannelTensor(_tensor, "setBeta");
        beta = _tensor;
        beta.invalidateFp16Cache();
    }

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
        step_gamma_grad_scratch.setExecutionTarget(_new_execution_target);
        step_beta_grad_scratch.setExecutionTarget(_new_execution_target);
        running_mean.setExecutionTarget(_new_execution_target);
        running_variance.setExecutionTarget(_new_execution_target);
        batch_mean.setExecutionTarget(_new_execution_target);
        batch_variance.setExecutionTarget(_new_execution_target);
        normalized_input.setExecutionTarget(_new_execution_target);
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
        input_tensor_fp16 = Tensor(0, 0, _new_execution_target);
        output_gradient_fp16 = Tensor(0, 0, _new_execution_target);
        input_gradient_tensor_fp16 = Tensor(0, 0, _new_execution_target);
        invalidateWeightCache();
    }

void Batch_Norm_2d_Layer::setMixedPrecision(bool _enable) noexcept
{
        ILayer::setMixedPrecision(_enable);
        invalidateWeightCache();
    }

void Batch_Norm_2d_Layer::invalidateWeightCache() noexcept
{
        gamma.invalidateFp16Cache();
        beta.invalidateFp16Cache();
    }

void Batch_Norm_2d_Layer::setInputHeight(uint32_t _height)
{
        if (_height == 0 || input_width == 0 || channels == 0)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::setInputHeight: dimensions must be > 0");
        }
        if (static_cast<uint64_t>(_height) * input_width * channels > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Batch_Norm_2d_Layer::setInputHeight: H * W * C exceeds uint32_t limit");
        }
        input_height = _height;
        is_forward_completed = false;
    }

void Batch_Norm_2d_Layer::setInputWidth(uint32_t _width)
{
        if (input_height == 0 || _width == 0 || channels == 0)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::setInputWidth: dimensions must be > 0");
        }
        if (static_cast<uint64_t>(input_height) * _width * channels > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Batch_Norm_2d_Layer::setInputWidth: H * W * C exceeds uint32_t limit");
        }
        input_width = _width;
        is_forward_completed = false;
    }

void Batch_Norm_2d_Layer::setMomentum(float _momentum)
{
        if (!std::isfinite(_momentum) || _momentum < 0.0f || _momentum > 1.0f)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::setMomentum: momentum must be finite and in [0, 1]");
        }
        momentum = _momentum;
    }

void Batch_Norm_2d_Layer::setEpsilon(float _epsilon)
{
        if (!std::isfinite(_epsilon) || _epsilon <= 0.0f)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::setEpsilon: epsilon must be finite and > 0");
        }
        epsilon = _epsilon;
    }

void Batch_Norm_2d_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

void Batch_Norm_2d_Layer::setTrainingMode(bool _is_training)
{
    is_training = _is_training;
    if (!_is_training)
    {
        sanitizeRunningStats();
    }
}

void Batch_Norm_2d_Layer::sanitizeRunningStats()
{
    if (channels == 0) return;

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        engine.waitIdle();
    }

    auto rm_data = running_mean.getData();
    auto rv_data = running_variance.getData();
    bool rm_bad = (rm_data.size() != channels);
    bool rv_bad = (rv_data.size() != channels);

    if (!rm_bad)
    {
        for (float v : rm_data)
        {
            if (std::isnan(v) || std::isinf(v)) { rm_bad = true; break; }
        }
    }
    if (!rv_bad)
    {
        for (float v : rv_data)
        {
            if (std::isnan(v) || std::isinf(v) || v < 0.0f) { rv_bad = true; break; }
        }
    }

    if (rm_bad)
    {
        std::vector<float> clean_rm(channels, 0.0f);
        for (size_t c = 0; c < rm_data.size() && c < channels; ++c)
        {
            clean_rm[c] = (std::isnan(rm_data[c]) || std::isinf(rm_data[c])) ? 0.0f : rm_data[c];
        }
        auto rm_impl = running_mean.getImplementation();
        if (rm_impl &&
            running_mean.getExecutionTarget() == execution_target &&
            running_mean.getDataType() == Data_Type::FLOAT32 &&
            running_mean.getTotalElements() == channels &&
            rm_impl->isContiguous() &&
            rm_impl->getByteOffset() == 0)
        {
            running_mean.uploadData(clean_rm);
        }
        else
        {
            running_mean = Tensor(1, channels, clean_rm, execution_target);
        }
    }
    if (rv_bad)
    {
        std::vector<float> clean_rv(channels, 1.0f);
        for (size_t c = 0; c < rv_data.size() && c < channels; ++c)
        {
            clean_rv[c] = (std::isnan(rv_data[c]) || std::isinf(rv_data[c]) || rv_data[c] < 0.0f) ? 1.0f : rv_data[c];
        }
        auto rv_impl = running_variance.getImplementation();
        if (rv_impl &&
            running_variance.getExecutionTarget() == execution_target &&
            running_variance.getDataType() == Data_Type::FLOAT32 &&
            running_variance.getTotalElements() == channels &&
            rv_impl->isContiguous() &&
            rv_impl->getByteOffset() == 0)
        {
            running_variance.uploadData(clean_rv);
        }
        else
        {
            running_variance = Tensor(1, channels, clean_rv, execution_target);
        }
    }
}

void Batch_Norm_2d_Layer::snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        engine.waitIdle();
    }
    _out.push_back(running_mean.getData());
    _out.push_back(running_variance.getData());
}

void Batch_Norm_2d_Layer::restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor)
{
    if (_cursor + 2 > _in.size())
    {
        return;
    }
    const auto &rm = _in[_cursor++];
    const auto &rv = _in[_cursor++];
    if (!rm.empty() && running_mean.getTotalElements() == rm.size())
    {
        running_mean.uploadData(rm);
    }
    if (!rv.empty() && running_variance.getTotalElements() == rv.size())
    {
        running_variance.uploadData(rv);
    }
}

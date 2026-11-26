#include "layer/instance_norm2d_layer.h"

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
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "ops/normalization/instancenorm.h"

Instance_Norm_2d_Layer::Instance_Norm_2d_Layer(
    uint32_t _height,
    uint32_t _width,
    uint32_t _channels,
    float _epsilon,
    Execution_Target _execution_target)
    : input_height(_height),
      input_width(_width),
      channels(_channels),
      epsilon(_epsilon),
      execution_target(_execution_target),
      gamma(0, 0, _execution_target),
      beta(0, 0, _execution_target),
      gamma_gradient_tensor(0, 0, _execution_target),
      beta_gradient_tensor(0, 0, _execution_target),
      step_gamma_grad_scratch(0, 0, _execution_target),
      step_beta_grad_scratch(0, 0, _execution_target),
      saved_mean(0, 0, _execution_target),
      saved_inv_std(0, 0, _execution_target),
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
        throw std::invalid_argument("Instance_Norm_2d_Layer: height, width, and channels must be > 0");
    }
    if (!std::isfinite(_epsilon) || _epsilon <= 0.0f)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer: epsilon must be finite and > 0");
    }
    if (static_cast<uint64_t>(_height) * _width * _channels > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Instance_Norm_2d_Layer: H * W * C exceeds uint32_t limit");
    }

    gamma = Tensor(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target);
    beta = Tensor(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target);
    gamma_gradient_tensor = Tensor(1, _channels, _execution_target);
    beta_gradient_tensor = Tensor(1, _channels, _execution_target);
    gamma_gradient_tensor.zero();
    beta_gradient_tensor.zero();
}

Tensor Instance_Norm_2d_Layer::forward(const Tensor &_input_tensor)
{
    if (input_height == 0 || input_width == 0 || channels == 0)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::forward: height, width, and channels must be > 0");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::forward: epsilon must be finite and > 0");
    }
    if (!_input_tensor.getImplementation())
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::forward: Input tensor has null implementation");
    }
    if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::forward: TERNARY_INT2 input is not supported");
    }
    if (_input_tensor.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::forward: Execution_Target mismatch");
    }

    uint64_t tot_spatial = static_cast<uint64_t>(input_height) * input_width;
    uint64_t tot_feat = tot_spatial * channels;
    if (tot_feat > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Instance_Norm_2d_Layer::forward: H * W * C exceeds uint32_t limit");
    }

    const Shape &in_shape = _input_tensor.getShape();
    size_t rank = in_shape.getRank();
    bool valid_shape = false;
    if (rank == 2)
    {
        size_t rows = in_shape[0];
        size_t cols = in_shape[1];
        valid_shape = (rows > 0) && (cols == tot_feat || (cols == channels && rows % tot_spatial == 0));
    }
    else if (rank == 4)
    {
        valid_shape = (in_shape[0] > 0 && in_shape[1] == input_height && in_shape[2] == input_width && in_shape[3] == channels);
    }
    if (!valid_shape)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::forward: Input shape mismatch (expected {N, H*W*C}, {N*H*W, C}, or {N, H, W, C})");
    }
    if (_input_tensor.getTotalElements() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Instance_Norm_2d_Layer::forward: Input element count exceeds uint32_t limit");
    }

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

    ops::normalization::instanceNorm2dForward(
        *eff_input,
        gamma,
        beta,
        saved_mean,
        saved_inv_std,
        normalized_input,
        output_tensor,
        input_height,
        input_width,
        channels,
        epsilon);

    is_forward_completed = true;
    logBufferAddress(&input_tensor, "input_tensor (Forward)");
    logBufferAddress(&output_tensor, "output_tensor (Forward)");
    return output_tensor;
}

Tensor Instance_Norm_2d_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    (void)_batched_input;
    (void)_batched_params;
    throw std::logic_error("Instance_Norm_2d_Layer::forward: Population batch execution is not supported");
}

std::unique_ptr<ILayer> Instance_Norm_2d_Layer::clone() const
{
    auto cloned = std::make_unique<Instance_Norm_2d_Layer>(
        input_height, input_width, channels, epsilon, execution_target);
    cloned->gamma = gamma.clone();
    cloned->beta = beta.clone();
    cloned->setAccumulated(is_accumulated);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    return cloned;
}

Tensor Instance_Norm_2d_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Instance_Norm_2d_Layer::backward: Backward called before forward"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("Backward called before forward");
    }
    if (input_height == 0 || input_width == 0 || channels == 0 || !std::isfinite(epsilon) || epsilon <= 0.0f)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::backward: Invalid layer geometry or epsilon");
    }
    if (!_output_gradient.getImplementation())
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::backward: Output gradient has null implementation");
    }
    if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::backward: TERNARY_INT2 gradient is not supported");
    }
    if (_output_gradient.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::backward: Execution_Target mismatch");
    }
    if (_output_gradient.getShape() != output_tensor.getShape())
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::backward: Output gradient shape mismatch");
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

    if (!is_accumulated)
    {
        ops::normalization::instanceNorm2dBackward(
            *eff_grad,
            normalized_input,
            gamma,
            saved_inv_std,
            gamma_gradient_tensor,
            beta_gradient_tensor,
            target_in_grad,
            input_height,
            input_width,
            channels,
            epsilon);
    }
    else
    {
        ops::normalization::instanceNorm2dBackward(
            *eff_grad,
            normalized_input,
            gamma,
            saved_inv_std,
            step_gamma_grad_scratch,
            step_beta_grad_scratch,
            target_in_grad,
            input_height,
            input_width,
            channels,
            epsilon);
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

    logBufferAddress(&input_tensor, "input_tensor (Backward)");
    return input_gradient_tensor;
}

void Instance_Norm_2d_Layer::resetGradient()
{
    is_forward_completed = false;
}

void Instance_Norm_2d_Layer::resetGradients()
{
    resetGradient();
    gamma_gradient_tensor.zero();
    beta_gradient_tensor.zero();
}

void Instance_Norm_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::saveConfiguration: Output stream is not open or writable");
    }
    _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::saveConfiguration: Failed writing configuration");
    }
}

void Instance_Norm_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::saveCheckpoint: Output stream is not open or writable");
    }
    gamma.saveTensor(_output_file_stream);
    beta.saveTensor(_output_file_stream);
    gamma_gradient_tensor.saveTensor(_output_file_stream);
    beta_gradient_tensor.saveTensor(_output_file_stream);
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::saveCheckpoint: Failed writing checkpoint state");
    }
}

void Instance_Norm_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open() || !_input_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::loadCheckpoint: Input stream is not open or readable");
    }
    Tensor loaded_g = Tensor::loadTensor(_input_file_stream, execution_target);
    Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
    Tensor loaded_gg = Tensor::loadTensor(_input_file_stream, execution_target);
    Tensor loaded_bg = Tensor::loadTensor(_input_file_stream, execution_target);
    if (!_input_file_stream)
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::loadCheckpoint: Failed reading checkpoint state");
    }
    validateChannelTensor(loaded_g, "loadCheckpoint(gamma)");
    validateChannelTensor(loaded_b, "loadCheckpoint(beta)");
    validateChannelTensor(loaded_gg, "loadCheckpoint(gamma_gradient)");
    validateChannelTensor(loaded_bg, "loadCheckpoint(beta_gradient)");
    gamma = std::move(loaded_g);
    beta = std::move(loaded_b);
    gamma_gradient_tensor = std::move(loaded_gg);
    beta_gradient_tensor = std::move(loaded_bg);
    invalidateWeightCache();
}

void Instance_Norm_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::saveInference: Output stream is not open or writable");
    }
    gamma.saveTensor(_output_file_stream);
    beta.saveTensor(_output_file_stream);
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::saveInference: Failed writing inference state");
    }
}

void Instance_Norm_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open() || !_input_file_stream.good())
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::loadInference: Input stream is not open or readable");
    }
    Tensor loaded_g = Tensor::loadTensor(_input_file_stream, execution_target);
    Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
    if (!_input_file_stream)
    {
        throw std::runtime_error("Instance_Norm_2d_Layer::loadInference: Failed reading inference state");
    }
    validateChannelTensor(loaded_g, "loadInference(gamma)");
    validateChannelTensor(loaded_b, "loadInference(beta)");
    gamma = std::move(loaded_g);
    beta = std::move(loaded_b);
    invalidateWeightCache();
}

std::function<float(std::mt19937&)> Instance_Norm_2d_Layer::getPopulationParameterInitializer(size_t param_index) const
{
    if (param_index == 0)
    {
        return [](std::mt19937&) { return 1.0f; };
    }
    return [](std::mt19937&) { return 0.0f; };
}

std::vector<float> Instance_Norm_2d_Layer::getPopulationParameter(size_t param_index) const
{
    switch (param_index)
    {
    case 0:
        return gamma.getData();
    case 1:
        return beta.getData();
    default:
        throw std::out_of_range("Invalid parameter index for Instance_Norm_2d_Layer");
    }
}

void Instance_Norm_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
    if (flat_data.size() != channels)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::setPopulationParameter: Dimension mismatch");
    }

    switch (param_index)
    {
    case 0:
        gamma = Tensor(1, channels, std::move(flat_data), execution_target);
        break;
    case 1:
        beta = Tensor(1, channels, std::move(flat_data), execution_target);
        break;
    default:
        throw std::out_of_range("Invalid parameter index for Instance_Norm_2d_Layer");
    }
}

std::vector<std::pair<Tensor *, Tensor *>> Instance_Norm_2d_Layer::getParametersAndGradients()
{
    return {
        {&gamma, &gamma_gradient_tensor},
        {&beta, &beta_gradient_tensor}
    };
}

void Instance_Norm_2d_Layer::validateChannelTensor(const Tensor &_tensor, const char *_setter_name) const
{
    if (!_tensor.getImplementation())
    {
        throw std::invalid_argument(std::format("Instance_Norm_2d_Layer::{}: Tensor has null implementation", _setter_name));
    }
    if (_tensor.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument(std::format("Instance_Norm_2d_Layer::{}: Execution_Target mismatch", _setter_name));
    }
    if (_tensor.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::format("Instance_Norm_2d_Layer::{}: TERNARY_INT2 is not supported", _setter_name));
    }
    if (_tensor.getTotalElements() != channels)
    {
        throw std::invalid_argument(std::format(
            "Instance_Norm_2d_Layer::{}: Element count mismatch (expected {}, got {})",
            _setter_name, channels, _tensor.getTotalElements()));
    }
}

void Instance_Norm_2d_Layer::setGammaGradient(const Tensor &_tensor)
{
    validateChannelTensor(_tensor, "setGammaGradient");
    gamma_gradient_tensor = _tensor;
}

void Instance_Norm_2d_Layer::setBetaGradient(const Tensor &_tensor)
{
    validateChannelTensor(_tensor, "setBetaGradient");
    beta_gradient_tensor = _tensor;
}

void Instance_Norm_2d_Layer::setGamma(const Tensor &_tensor)
{
    validateChannelTensor(_tensor, "setGamma");
    gamma = _tensor;
    gamma.invalidateFp16Cache();
}

void Instance_Norm_2d_Layer::setBeta(const Tensor &_tensor)
{
    validateChannelTensor(_tensor, "setBeta");
    beta = _tensor;
    beta.invalidateFp16Cache();
}

void Instance_Norm_2d_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    invalidateWeightCache();
}

void Instance_Norm_2d_Layer::invalidateWeightCache() noexcept
{
    gamma.invalidateFp16Cache();
    beta.invalidateFp16Cache();
}

void Instance_Norm_2d_Layer::setInputHeight(uint32_t _height)
{
    if (_height == 0 || input_width == 0 || channels == 0)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::setInputHeight: dimensions must be > 0");
    }
    if (static_cast<uint64_t>(_height) * input_width * channels > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Instance_Norm_2d_Layer::setInputHeight: H * W * C exceeds uint32_t limit");
    }
    input_height = _height;
    is_forward_completed = false;
}

void Instance_Norm_2d_Layer::setInputWidth(uint32_t _width)
{
    if (input_height == 0 || _width == 0 || channels == 0)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::setInputWidth: dimensions must be > 0");
    }
    if (static_cast<uint64_t>(input_height) * _width * channels > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Instance_Norm_2d_Layer::setInputWidth: H * W * C exceeds uint32_t limit");
    }
    input_width = _width;
    is_forward_completed = false;
}

void Instance_Norm_2d_Layer::setEpsilon(float _epsilon)
{
    if (!std::isfinite(_epsilon) || _epsilon <= 0.0f)
    {
        throw std::invalid_argument("Instance_Norm_2d_Layer::setEpsilon: epsilon must be finite and > 0");
    }
    epsilon = _epsilon;
}

void Instance_Norm_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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
    saved_mean.setExecutionTarget(_new_execution_target);
    saved_inv_std.setExecutionTarget(_new_execution_target);
    normalized_input.setExecutionTarget(_new_execution_target);
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
    input_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    output_gradient_fp16 = Tensor(0, 0, _new_execution_target);
    input_gradient_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    invalidateWeightCache();
}

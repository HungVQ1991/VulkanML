#include "layer/instance_norm2d_layer.h"

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
      gamma(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target),
      beta(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target),
      gamma_gradient_tensor(1, _channels, _execution_target),
      beta_gradient_tensor(1, _channels, _execution_target),
      saved_mean(0, 0, _execution_target),
      saved_inv_std(0, 0, _execution_target),
      normalized_input(0, 0, _execution_target),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false)
{
}

Tensor Instance_Norm_2d_Layer::forward(const Tensor &_input_tensor)
{
    input_tensor = _input_tensor;

    ops::normalization::instanceNorm2dForward(
        input_tensor,
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
    if (_batched_params.size() != getPopulationParameterDims().size())
    {
        Logger::logMessage(Input_Format{"Instance_Norm_2d_Layer::forward: expected {} batched parameter tensors (gamma, beta), got {}",
                                        getPopulationParameterDims().size(), _batched_params.size()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument("Invalid input params size");
    }

    Tensor saved_mean_tensor(getExecutionTarget());
    Tensor saved_inv_std_tensor(getExecutionTarget());
    Tensor normalized_input_tensor(getExecutionTarget());
    Tensor output_tensor_result(getExecutionTarget());

    ops::normalization::instanceNorm2dForward(
        _batched_input,
        _batched_params[0],
        _batched_params[1],
        saved_mean_tensor,
        saved_inv_std_tensor,
        normalized_input_tensor,
        output_tensor_result,
        input_height,
        input_width,
        channels,
        epsilon);

    return output_tensor_result;
}

std::unique_ptr<ILayer> Instance_Norm_2d_Layer::clone() const
{
    return std::make_unique<Instance_Norm_2d_Layer>(
        input_height, input_width, channels, epsilon, execution_target);
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

    if (!is_accumulated)
    {
        ops::normalization::instanceNorm2dBackward(
            _output_gradient,
            normalized_input,
            gamma,
            saved_inv_std,
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
        ops::normalization::instanceNorm2dBackward(
            _output_gradient,
            normalized_input,
            gamma,
            saved_inv_std,
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

void Instance_Norm_2d_Layer::resetGradient()
{
    is_forward_completed = false;
    input_tensor = Tensor(0, 0, execution_target);
    output_tensor = Tensor(0, 0, execution_target);
    normalized_input = Tensor(0, 0, execution_target);
    saved_mean = Tensor(0, 0, execution_target);
    saved_inv_std = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);
}

void Instance_Norm_2d_Layer::resetGradients()
{
    resetGradient();
    gamma_gradient_tensor.zero();
    beta_gradient_tensor.zero();
}

void Instance_Norm_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
}

void Instance_Norm_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    gamma.saveTensor(_output_file_stream);
    beta.saveTensor(_output_file_stream);
    gamma_gradient_tensor.saveTensor(_output_file_stream);
    beta_gradient_tensor.saveTensor(_output_file_stream);
}

void Instance_Norm_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    gamma = Tensor::loadTensor(_input_file_stream, execution_target);
    beta = Tensor::loadTensor(_input_file_stream, execution_target);
    gamma_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
    beta_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
}

void Instance_Norm_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    gamma.saveTensor(_output_file_stream);
    beta.saveTensor(_output_file_stream);
}

void Instance_Norm_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
    gamma = Tensor::loadTensor(_input_file_stream, execution_target);
    beta = Tensor::loadTensor(_input_file_stream, execution_target);
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
    switch (param_index)
    {
    case 0:
        gamma = Tensor(1, channels, flat_data, execution_target);
        break;
    case 1:
        beta = Tensor(1, channels, flat_data, execution_target);
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

void Instance_Norm_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    gamma.setExecutionTarget(_new_execution_target);
    beta.setExecutionTarget(_new_execution_target);
    gamma_gradient_tensor.setExecutionTarget(_new_execution_target);
    beta_gradient_tensor.setExecutionTarget(_new_execution_target);
    saved_mean.setExecutionTarget(_new_execution_target);
    saved_inv_std.setExecutionTarget(_new_execution_target);
    normalized_input.setExecutionTarget(_new_execution_target);
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

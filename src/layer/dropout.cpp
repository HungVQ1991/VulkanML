#include "layer/dropout.h"

#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helper/logger.h"
#include "layer_internal_utils.h"
#include "ops/dropout/dropout.h"

namespace
{

void validateDropoutRate(float rate, const char *log_ctx = nullptr)
{
    if (!std::isfinite(rate) || rate < 0.0f || rate >= 1.0f)
    {
        if (log_ctx != nullptr)
        {
            Logger::logMessage(Input_Format{"{}: invalid dropout_rate {}, must be finite in [0.0, 1.0)", log_ctx, rate},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::ACTIVATION_COMPUTE);
        }
        throw std::invalid_argument("Dropout rate must be finite and in the range [0.0, 1.0)");
    }
}

}

Dropout_Layer::Dropout_Layer(float _dropout_rate, Execution_Target _execution_target, uint64_t _seed)
    : dropout_rate(_dropout_rate),
      is_training(true),
      seed(_seed),
      mask_tensor(0, 0, _execution_target),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
    validateDropoutRate(_dropout_rate, "Dropout_Layer");
}

uint64_t Dropout_Layer::nextStepSeed() const noexcept
{
    const uint64_t s = step_counter++;
    if (seed == 0)
    {
        return 0;
    }
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (s + 1ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (z == 0ULL) ? 1ULL : z;
}

Tensor Dropout_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateForwardInput(_input_tensor, execution_target, "Dropout_Layer::forward");
    const bool run_training = is_training;
    const float run_rate = dropout_rate;
    Logger::logMessage(Input_Format{"Dropout_Layer::forward: rows={}, columns={}, is_training={}",
                                    _input_tensor.getRows(), _input_tensor.getColumns(), run_training},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    const uint64_t step_seed = (run_training && run_rate > 0.0f) ? nextStepSeed() : seed;
    ops::dropout::dropout(input_tensor, output_tensor, mask_tensor, run_rate, run_training, step_seed);
    was_training_in_forward = run_training;
    last_forward_dropout_rate = run_rate;
    is_forward_completed = true;
    return output_tensor;
}

Tensor Dropout_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    layer_internal::validateBatchedParamsEmpty(
        _batched_params, "Dropout_Layer::forward", "Dropout_Layer",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    if (!_batched_input.getImplementation())
    {
        throw std::invalid_argument("Dropout_Layer::forward: Batched input tensor has null implementation");
    }
    if (_batched_input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Dropout_Layer::forward: TERNARY_INT2 batched input is not supported");
    }
    if (!is_training || dropout_rate <= 0.0f)
    {
        return _batched_input;
    }

    Tensor output_tensor_result(_batched_input.getExecutionTarget());
    Tensor dummy_mask(0, 0, _batched_input.getExecutionTarget());
    ops::dropout::dropout(_batched_input, output_tensor_result, dummy_mask, dropout_rate, is_training, nextStepSeed());
    return output_tensor_result;
}

Tensor Dropout_Layer::backward(const Tensor &_output_gradient)
{
    layer_internal::validateBackwardInput(
        is_forward_completed, output_tensor, _output_gradient, output_tensor.getExecutionTarget(),
        "Dropout_Layer::backward", "Forward pass must be completed before backward pass in Dropout_Layer",
        "Dropout_Layer::backward: Gradient shape must match output shape",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    Logger::logMessage(Input_Format{"Dropout_Layer::backward: rows={}, columns={}",
                                    _output_gradient.getRows(), _output_gradient.getColumns()},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    if (!was_training_in_forward || last_forward_dropout_rate <= 0.0f)
    {
        input_gradient_tensor = _output_gradient;
    }
    else
    {
        ops::dropout::dropoutBackward(_output_gradient, mask_tensor, input_gradient_tensor, last_forward_dropout_rate);
    }
    return input_gradient_tensor;
}

void Dropout_Layer::resetGradient()
{
    is_forward_completed = false;
    was_training_in_forward = false;
    last_forward_dropout_rate = 0.0f;
}

std::unique_ptr<ILayer> Dropout_Layer::clone() const
{
    auto cloned = std::make_unique<Dropout_Layer>(dropout_rate, execution_target, seed);
    cloned->setTrainingMode(is_training);
    cloned->setStepCounter(step_counter);
    copyBaseStateTo(*cloned);
    return cloned;
}

void Dropout_Layer::setPopulationParameter(size_t param_index, std::vector<float>)
{
    throw std::out_of_range(std::format("Dropout_Layer has no evolvable population parameters (index={})", param_index));
}

void Dropout_Layer::setDropoutRate(float _rate)
{
    validateDropoutRate(_rate);
    dropout_rate = _rate;
}

void Dropout_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(_new_execution_target, input_tensor, output_tensor, mask_tensor, input_gradient_tensor);
}

void Dropout_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    layer_internal::writePods(_output_file_stream, "Dropout_Layer::saveConfiguration",
                              "Failed writing dropout_rate", dropout_rate);
}

void Dropout_Layer::saveInference(std::ofstream &_output_file_stream) const { saveConfiguration(_output_file_stream); }

void Dropout_Layer::loadInference(std::ifstream &_input_file_stream)
{
    layer_internal::requireInStream(_input_file_stream, "Dropout_Layer::loadInference");
    float loaded_rate = 0.0f;
    layer_internal::readPods(_input_file_stream, "Dropout_Layer::loadInference: Invalid or corrupted dropout_rate", loaded_rate);
    if (!std::isfinite(loaded_rate) || loaded_rate < 0.0f || loaded_rate >= 1.0f)
    {
        throw std::runtime_error("Dropout_Layer::loadInference: Invalid or corrupted dropout_rate");
    }
    dropout_rate = loaded_rate;
}

void Dropout_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    constexpr float DROPOUT_CKPT_V2_SENTINEL = -1.0f;
    layer_internal::writePods(_output_file_stream, "Dropout_Layer::saveCheckpoint",
                              "Failed writing dropout checkpoint",
                              DROPOUT_CKPT_V2_SENTINEL, dropout_rate, seed, step_counter);
}

void Dropout_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    layer_internal::requireInStream(_input_file_stream, "Dropout_Layer::loadCheckpoint");
    constexpr float DROPOUT_CKPT_V2_SENTINEL = -1.0f;
    float first_val = 0.0f;
    layer_internal::readPods(_input_file_stream, "Dropout_Layer::loadCheckpoint: Failed to read dropout header", first_val);

    float loaded_rate = 0.0f;
    uint64_t loaded_seed = seed;
    uint64_t loaded_step = 0;

    if (first_val == DROPOUT_CKPT_V2_SENTINEL)
    {
        layer_internal::readPods(_input_file_stream, "Dropout_Layer::loadCheckpoint: Failed to read V2 dropout state",
                                 loaded_rate, loaded_seed, loaded_step);
    }
    else
    {
        loaded_rate = first_val;
    }

    if (!std::isfinite(loaded_rate) || loaded_rate < 0.0f || loaded_rate >= 1.0f)
    {
        throw std::runtime_error("Dropout_Layer::loadCheckpoint: Invalid dropout_rate in checkpoint");
    }

    dropout_rate = loaded_rate;
    seed = loaded_seed;
    step_counter = loaded_step;
}

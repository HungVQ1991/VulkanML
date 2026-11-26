#include "layer/softmax.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helper/logger.h"
#include "layer_internal_utils.h"
#include "ops/activation/activation.h"

Softmax_Layer::Softmax_Layer(bool _is_fused_with_loss, Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      input_fp32(0, 0, _execution_target),
      grad_fp32(0, 0, _execution_target),
      input_grad_fp32(0, 0, _execution_target),
      is_fused_with_loss(_is_fused_with_loss),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor Softmax_Layer::forward(const Tensor &_input_tensor)
{
    layer_internal::validateForwardInput(_input_tensor, execution_target, "Softmax_Layer::forward");
    Logger::logMessage(Input_Format{"Softmax_Layer::forward: rows={}, columns={}",
                                    _input_tensor.getRows(), _input_tensor.getColumns()},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

    input_tensor = _input_tensor;
    if (input_tensor.getDataType() == Data_Type::FLOAT16)
    {
        input_tensor.to(Data_Type::FLOAT32, input_fp32);
        ops::activation::softmax(input_fp32, output_tensor);
    }
    else
    {
        ops::activation::softmax(input_tensor, output_tensor);
    }
    is_forward_completed = true;
    return output_tensor;
}

Tensor Softmax_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    return layer_internal::runStatelessUnaryBatchedForward(
        _batched_input, _batched_params, "Softmax_Layer::forward", "Softmax_Layer",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION,
        [](const Tensor &in, Tensor &out) {
            if (in.getDataType() == Data_Type::FLOAT16)
            {
                Tensor batched_fp32(in.getExecutionTarget());
                in.to(Data_Type::FLOAT32, batched_fp32);
                ops::activation::softmax(batched_fp32, out);
            }
            else
            {
                ops::activation::softmax(in, out);
            }
        });
}

std::unique_ptr<ILayer> Softmax_Layer::clone() const
{
    auto cloned = std::make_unique<Softmax_Layer>(is_fused_with_loss, execution_target);
    copyBaseStateTo(*cloned);
    return cloned;
}

Tensor Softmax_Layer::backward(const Tensor &_output_gradient)
{
    layer_internal::validateBackwardInput(
        is_forward_completed, output_tensor, _output_gradient, output_tensor.getExecutionTarget(),
        "Softmax_Layer::backward", "Backward called before forward",
        "Softmax_Layer::backward: Gradient shape must match output shape",
        Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Softmax_Layer::backward: TERNARY_INT2 gradient is not supported");
    }

    Logger::logMessage(Input_Format{"Softmax_Layer::backward: output_gradient rows={}, columns={}, is_fused_with_loss={}",
                                    _output_gradient.getRows(), _output_gradient.getColumns(), is_fused_with_loss},
                       Log_Level::LOG_DEBUG, true, 1, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

    if (is_fused_with_loss)
    {
        const Data_Type in_dtype = input_tensor.getDataType();
        if (_output_gradient.getDataType() != in_dtype &&
            (in_dtype == Data_Type::FLOAT16 || in_dtype == Data_Type::FLOAT32))
        {
            _output_gradient.to(in_dtype, input_gradient_tensor);
        }
        else
        {
            input_gradient_tensor = _output_gradient;
        }
        return input_gradient_tensor;
    }

    const Tensor *grad_src = &_output_gradient;
    if (output_tensor.getDataType() == Data_Type::FLOAT32 &&
        _output_gradient.getDataType() == Data_Type::FLOAT16)
    {
        _output_gradient.to(Data_Type::FLOAT32, grad_fp32);
        grad_src = &grad_fp32;
    }

    if (input_tensor.getDataType() == Data_Type::FLOAT16)
    {
        ops::activation::softmaxBackward(output_tensor, *grad_src, input_grad_fp32);
        input_grad_fp32.to(Data_Type::FLOAT16, input_gradient_tensor);
    }
    else
    {
        ops::activation::softmaxBackward(output_tensor, *grad_src, input_gradient_tensor);
    }
    return input_gradient_tensor;
}

void Softmax_Layer::resetGradient() { is_forward_completed = false; }

void Softmax_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    const std::uint8_t fused_value = is_fused_with_loss ? 1 : 0;
    layer_internal::writePods(_output_file_stream, "Softmax_Layer::saveConfiguration",
                              "Failed writing configuration", fused_value);
}

void Softmax_Layer::saveInference(std::ofstream &) const {}
void Softmax_Layer::loadInference(std::ifstream &) {}
void Softmax_Layer::saveCheckpoint(std::ofstream &) const {}
void Softmax_Layer::loadCheckpoint(std::ifstream &) {}

bool Softmax_Layer::isFusedWithLoss() const noexcept { return is_fused_with_loss; }

void Softmax_Layer::setPopulationParameter(size_t, std::vector<float>)
{
    throw std::out_of_range("Softmax_Layer::setPopulationParameter: Layer has no parameters");
}

void Softmax_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target) return;
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    layer_internal::moveTensorsToTarget(
        _new_execution_target, input_tensor, output_tensor, input_gradient_tensor,
        input_fp32, grad_fp32, input_grad_fp32);
}

void Softmax_Layer::setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }

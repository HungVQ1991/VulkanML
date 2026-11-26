#include "layer/swiglu_layer.h"

#include <format>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_internal_utils.h"
#include "ops/llm/swiglu.h"

namespace
{

void validateFusedSwigluInput(const Tensor &t, Execution_Target expected_target, const char *ctx)
{
    layer_internal::validateForwardInput(t, expected_target, ctx);
    if (t.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(ctx) + ": TERNARY_INT2 input is not supported");
    }
    const Shape &in_shape = t.getShape();
    const size_t rank = in_shape.getRank();
    if (rank < 2 || in_shape[rank - 1] == 0 || in_shape[rank - 1] % 2 != 0 || t.getTotalElements() == 0)
    {
        throw std::invalid_argument(std::string(ctx) + ": Input must have rank >= 2, non-zero elements, and an even non-zero last dimension");
    }
}

}

SwiGLU_Layer::SwiGLU_Layer(Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
      tensor_a(0, 0, _execution_target),
      tensor_b(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      grad_a(0, 0, _execution_target),
      grad_b(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false),
      execution_target(_execution_target)
{
}

Tensor SwiGLU_Layer::forward(const Tensor &_input_tensor)
{
    validateFusedSwigluInput(_input_tensor, execution_target, "SwiGLU_Layer::forward");
    input_tensor = _input_tensor;
    ops::llm::fusedSwiGLUForward(input_tensor, output_tensor);
    used_split_forward = false;
    is_forward_completed = true;
    return output_tensor;
}

Tensor SwiGLU_Layer::forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const
{
    if (!_batched_params.empty())
    {
        throw std::invalid_argument(std::format(
            "SwiGLU_Layer::forward: expected 0 batched parameter tensors, got {}", _batched_params.size()));
    }
    validateFusedSwigluInput(_batched_input, execution_target, "SwiGLU_Layer::forward");
    Tensor output_tensor_result(_batched_input.getExecutionTarget());
    ops::llm::fusedSwiGLUForward(_batched_input, output_tensor_result);
    return output_tensor_result;
}

Tensor SwiGLU_Layer::forward(const Tensor &a, const Tensor &b)
{
    if (!a.getImplementation() || !b.getImplementation())
    {
        throw std::invalid_argument("SwiGLU_Layer::forward: Input tensor has null implementation");
    }
    if (a.getDataType() == Data_Type::TERNARY_INT2 || b.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("SwiGLU_Layer::forward: TERNARY_INT2 input is not supported");
    }
    if (a.getExecutionTarget() != execution_target || b.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("SwiGLU_Layer::forward: Execution_Target mismatch");
    }
    if (a.getShape() != b.getShape() || a.getShape().getRank() < 1 || a.getTotalElements() == 0)
    {
        throw std::invalid_argument("SwiGLU_Layer::forward: Operands a and b must have identical non-empty shapes with rank >= 1");
    }

    tensor_a = a;
    tensor_b = b;
    ops::llm::swigluForward(tensor_a, tensor_b, output_tensor);
    used_split_forward = true;
    is_forward_completed = true;
    return output_tensor;
}

Tensor SwiGLU_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::logic_error("SwiGLU_Layer::backward called before forward");
    }
    layer_internal::validateForwardInput(_output_gradient, execution_target, "SwiGLU_Layer::backward");
    if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("SwiGLU_Layer::backward: TERNARY_INT2 gradient is not supported");
    }
    if (_output_gradient.getShape() != output_tensor.getShape())
    {
        throw std::invalid_argument("SwiGLU_Layer::backward: Output gradient shape mismatch");
    }

    if (used_split_forward)
    {
        ops::llm::swigluBackward(tensor_a, tensor_b, _output_gradient, grad_a, grad_b, ops::llm::SwiGLU_Config{});
        return grad_a;
    }
    ops::llm::fusedSwiGLUBackward(input_tensor, _output_gradient, input_gradient_tensor, ops::llm::SwiGLU_Config{});
    return input_gradient_tensor;
}

std::unique_ptr<ILayer> SwiGLU_Layer::clone() const
{
    auto copy = std::make_unique<SwiGLU_Layer>(execution_target);
    copyBaseStateTo(*copy);
    return copy;
}

void SwiGLU_Layer::setPopulationParameter(size_t param_index, std::vector<float>)
{
    throw std::out_of_range(std::format("SwiGLU_Layer has no evolvable population parameters (index={})", param_index));
}

void SwiGLU_Layer::resetGradient()
{
    is_forward_completed = false;
    used_split_forward = false;
}

void SwiGLU_Layer::saveConfiguration(std::ofstream &) const {}
void SwiGLU_Layer::saveInference(std::ofstream &) const {}
void SwiGLU_Layer::loadInference(std::ifstream &) {}
void SwiGLU_Layer::saveCheckpoint(std::ofstream &) const {}
void SwiGLU_Layer::loadCheckpoint(std::ifstream &) {}

const Tensor &SwiGLU_Layer::getGradA() const noexcept { return grad_a; }
const Tensor &SwiGLU_Layer::getGradB() const noexcept { return grad_b; }

void SwiGLU_Layer::setExecutionTarget(Execution_Target _new_target)
{
    if (execution_target == _new_target) return;
    execution_target = _new_target;
    layer_internal::moveTensorsToTarget(
        _new_target, input_tensor, tensor_a, tensor_b, output_tensor, grad_a, grad_b, input_gradient_tensor);
}

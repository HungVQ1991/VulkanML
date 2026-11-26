#include "cost_function/fused_cce_cost.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ops/loss/loss_ops.h"

namespace
{

void zeroGradientOutput(const Tensor &pred, const Shape &shape, Tensor &gradient_out)
{
    if (gradient_out.getShape() != shape ||
        gradient_out.getExecutionTarget() != pred.getExecutionTarget() ||
        gradient_out.getDataType() != pred.getDataType())
    {
        gradient_out = Tensor(shape, pred.getDataType(), pred.getExecutionTarget());
    }
    gradient_out.zero();
}

}

Fused_Cross_Entropy::Fused_Cross_Entropy(int32_t ignore_idx) noexcept
    : ignore_index(ignore_idx)
{
}

float Fused_Cross_Entropy::computeLoss(const Tensor &prediction_tensor, const Tensor &target_tensor) const
{
    Tensor dummy_grad;
    return ops::loss::fusedCrossEntropy(prediction_tensor, target_tensor, dummy_grad, 0, ignore_index, false, true);
}

void Fused_Cross_Entropy::computeGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    ops::loss::fusedCrossEntropy(prediction_tensor, target_tensor, gradient_out, 0, ignore_index, true, false);
}

Tensor Fused_Cross_Entropy::computeGradient(const Tensor &prediction_tensor, const Tensor &target_tensor) const
{
    Tensor gradient_out;
    computeGradient(prediction_tensor, target_tensor, gradient_out);
    return gradient_out;
}

float Fused_Cross_Entropy::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    return ops::loss::fusedCrossEntropy(prediction_tensor, target_tensor, gradient_out, 0, ignore_index);
}

float Fused_Cross_Entropy::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out, uint32_t valid_tokens) const
{
    return ops::loss::fusedCrossEntropy(prediction_tensor, target_tensor, gradient_out, valid_tokens, ignore_index);
}

float Fused_Cross_Entropy::computeLossAndGradient(const Tensor &prediction_tensor, const std::vector<int32_t> &target_indices, Tensor &gradient_out) const
{
    if (prediction_tensor.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Fused_Cross_Entropy::computeLossAndGradient: TERNARY_INT2 is not supported");
    }

    const Shape &shape = prediction_tensor.getShape();
    if (shape.getRank() < 2)
    {
        throw std::invalid_argument("Fused_Cross_Entropy::computeLossAndGradient: prediction logits rank must be >= 2");
    }

    const size_t vocab_size = shape[shape.getRank() - 1];
    const size_t total_tokens = (vocab_size > 0) ? (prediction_tensor.getTotalElements() / vocab_size) : 0;

    if (target_indices.empty() && total_tokens == 0)
    {
        zeroGradientOutput(prediction_tensor, shape, gradient_out);
        return 0.0f;
    }

    if (target_indices.size() != total_tokens)
    {
        throw std::invalid_argument("Fused_Cross_Entropy::computeLossAndGradient: target_indices size does not match total tokens");
    }

    uint32_t valid_tokens = 0;
    for (int32_t id : target_indices)
    {
        if (id >= 0 && id != ignore_index && static_cast<size_t>(id) < vocab_size)
        {
            valid_tokens++;
        }
    }

    if (valid_tokens == 0)
    {
        zeroGradientOutput(prediction_tensor, shape, gradient_out);
        return 0.0f;
    }

    return ops::loss::fusedCrossEntropy(prediction_tensor, target_indices, gradient_out, valid_tokens, ignore_index);
}

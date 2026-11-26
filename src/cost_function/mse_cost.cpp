#include "cost_function/mse_cost.h"

#include "cost_internal_utils.h"
#include "ops/loss/loss_ops.h"

Mse_Cost::Mse_Cost(Execution_Target _execution_target)
    : loss_matrix(0, 0, _execution_target),
      synced_target_matrix(0, 0, _execution_target),
      difference_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
}

float Mse_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Mse_Cost::computeLoss", true, true))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::mse(_prediction_matrix, _target_matrix, loss_matrix);
    return loss_matrix.getScalar() / static_cast<float>(_prediction_matrix.getTotalElements());
}

Tensor Mse_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Mse_Cost::computeGradient", true, true))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    const float scale = 2.0f / static_cast<float>(_prediction_matrix.getTotalElements());
    cost_internal::computeScaledDiffGradient(
        _prediction_matrix, _target_matrix, scale, synced_target_matrix, difference_matrix, gradient_matrix);
    return gradient_matrix;
}

float Mse_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Mse_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    const size_t total_elements = prediction_tensor.getTotalElements();
    const float scale = 2.0f / static_cast<float>(total_elements);
    cost_internal::computeScaledDiffGradientOut(
        prediction_tensor, target_tensor, scale, synced_target_matrix, difference_matrix, gradient_out);
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::mse(prediction_tensor, target_tensor, loss_matrix);
    return loss_matrix.getScalar() / static_cast<float>(total_elements);
}

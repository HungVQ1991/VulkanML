#include "cost_function/mae_cost.h"

#include "cost_internal_utils.h"
#include "ops/loss/loss_ops.h"

Mae_Cost::Mae_Cost(Execution_Target _execution_target)
    : loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
}

float Mae_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Mae_Cost::computeLoss", true, true))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::mae(_prediction_matrix, _target_matrix, loss_matrix);
    return loss_matrix.getScalar() / static_cast<float>(_prediction_matrix.getTotalElements());
}

Tensor Mae_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Mae_Cost::computeGradient", true, true))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    cost_internal::syncScratchTarget(gradient_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::maeGradient(_prediction_matrix, _target_matrix, gradient_matrix);
    return gradient_matrix;
}

float Mae_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Mae_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    ops::loss::maeGradient(prediction_tensor, target_tensor, gradient_out);
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::mae(prediction_tensor, target_tensor, loss_matrix);
    return loss_matrix.getScalar() / static_cast<float>(prediction_tensor.getTotalElements());
}

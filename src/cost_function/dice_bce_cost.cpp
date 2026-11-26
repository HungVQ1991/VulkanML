#include "cost_function/dice_bce_cost.h"

#include <vector>

#include "cost_internal_utils.h"
#include "ops/loss/dice_bce.h"

Dice_BCE_Cost::Dice_BCE_Cost(float _dice_weight, float _bce_weight, float _epsilon, Execution_Target _execution_target)
    : dice_weight(_dice_weight),
      bce_weight(_bce_weight),
      epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target),
      grad_stats_matrix(0, 0, _execution_target)
{
    validateParameters(_dice_weight, _bce_weight, _epsilon);
}

float Dice_BCE_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Dice_BCE_Cost::computeLoss", false, false))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::diceBce(_prediction_matrix, _target_matrix, loss_matrix, dice_weight, bce_weight, epsilon);
    const auto &d = loss_matrix.getData();
    return d.empty() ? 0.0f : d[0];
}

Tensor Dice_BCE_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Dice_BCE_Cost::computeGradient", false, false))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    const Execution_Target target_dev = _prediction_matrix.getExecutionTarget();
    cost_internal::syncScratchTarget(grad_stats_matrix, target_dev);
    cost_internal::syncScratchTarget(gradient_matrix, target_dev);
    ops::loss::diceBce(_prediction_matrix, _target_matrix, grad_stats_matrix, dice_weight, bce_weight, epsilon);
    ops::loss::diceBceGradient(_prediction_matrix, _target_matrix, gradient_matrix, dice_weight, bce_weight, epsilon, &grad_stats_matrix);
    return gradient_matrix;
}

float Dice_BCE_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Dice_BCE_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::diceBce(prediction_tensor, target_tensor, loss_matrix, dice_weight, bce_weight, epsilon);
    ops::loss::diceBceGradient(prediction_tensor, target_tensor, gradient_out, dice_weight, bce_weight, epsilon, &loss_matrix);
    const auto &d = loss_matrix.getData();
    return d.empty() ? 0.0f : d[0];
}

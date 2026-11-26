#include "cost_function/dice_cost.h"

#include <vector>

#include "cost_internal_utils.h"
#include "ops/loss/loss_ops.h"

Dice_Cost::Dice_Cost(float _epsilon, Execution_Target _execution_target)
    : epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target),
      grad_stats_matrix(0, 0, _execution_target)
{
    validateEpsilon(_epsilon);
}

float Dice_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Dice_Cost::computeLoss", false, false))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::dice(_prediction_matrix, _target_matrix, loss_matrix, epsilon);
    return loss_matrix.getScalar();
}

Tensor Dice_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Dice_Cost::computeGradient", false, false))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    const Execution_Target target_dev = _prediction_matrix.getExecutionTarget();
    cost_internal::syncScratchTarget(grad_stats_matrix, target_dev);
    cost_internal::syncScratchTarget(gradient_matrix, target_dev);
    ops::loss::dice(_prediction_matrix, _target_matrix, grad_stats_matrix, epsilon, true);
    ops::loss::diceGradient(_prediction_matrix, _target_matrix, gradient_matrix, epsilon, &grad_stats_matrix);
    return gradient_matrix;
}

float Dice_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Dice_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::dice(prediction_tensor, target_tensor, loss_matrix, epsilon, true);
    ops::loss::diceGradient(prediction_tensor, target_tensor, gradient_out, epsilon, &loss_matrix);
    const auto stats_data = loss_matrix.getData();
    return stats_data.empty() ? 0.0f : stats_data[0];
}

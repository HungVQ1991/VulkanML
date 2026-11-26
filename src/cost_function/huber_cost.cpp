#include "cost_function/huber_cost.h"

#include "cost_internal_utils.h"
#include "ops/loss/loss_ops.h"

Huber_Cost::Huber_Cost(float _delta, Execution_Target _execution_target)
    : delta(_delta),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
    validateDelta(_delta);
}

float Huber_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Huber_Cost::computeLoss", true, true))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::huber(_prediction_matrix, _target_matrix, loss_matrix, delta);
    return loss_matrix.getScalar() / static_cast<float>(_prediction_matrix.getTotalElements());
}

Tensor Huber_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Huber_Cost::computeGradient", true, true))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    cost_internal::syncScratchTarget(gradient_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::huberGradient(_prediction_matrix, _target_matrix, gradient_matrix, delta);
    return gradient_matrix;
}

float Huber_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Huber_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    ops::loss::huberGradient(prediction_tensor, target_tensor, gradient_out, delta);
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::huber(prediction_tensor, target_tensor, loss_matrix, delta);
    return loss_matrix.getScalar() / static_cast<float>(prediction_tensor.getTotalElements());
}

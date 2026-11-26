#include "cost_function/bce_cost.h"

#include "cost_internal_utils.h"
#include "ops/loss/loss_ops.h"

Bce_Cost::Bce_Cost(float _epsilon, Execution_Target _execution_target)
    : epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
    validateEpsilon(_epsilon);
}

float Bce_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Bce_Cost::computeLoss", true, true))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::bce(_prediction_matrix, _target_matrix, loss_matrix, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(_prediction_matrix.getTotalElements());
}

Tensor Bce_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Bce_Cost::computeGradient", true, true))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    cost_internal::syncScratchTarget(gradient_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::bceGradient(_prediction_matrix, _target_matrix, gradient_matrix, epsilon);
    return gradient_matrix;
}

float Bce_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Bce_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    ops::loss::bceGradient(prediction_tensor, target_tensor, gradient_out, epsilon);
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::bce(prediction_tensor, target_tensor, loss_matrix, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(prediction_tensor.getTotalElements());
}

#include "cost_function/asl_cost.h"

#include "cost_internal_utils.h"
#include "ops/loss/loss_ops.h"

Asl_Cost::Asl_Cost(
    float _gamma_pos,
    float _gamma_neg,
    float _clip_margin,
    float _epsilon,
    Execution_Target _execution_target)
    : gamma_pos(_gamma_pos),
      gamma_neg(_gamma_neg),
      clip_margin(_clip_margin),
      epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
    validateParameters(_gamma_pos, _gamma_neg, _clip_margin, _epsilon);
}

float Asl_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Asl_Cost::computeLoss", false, false))
    {
        return 0.0f;
    }
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::asl(_prediction_matrix, _target_matrix, loss_matrix, gamma_pos, gamma_neg, clip_margin, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(_prediction_matrix.getTotalElements());
}

Tensor Asl_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Asl_Cost::computeGradient", false, false))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    cost_internal::syncScratchTarget(gradient_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::aslGradient(_prediction_matrix, _target_matrix, gradient_matrix, gamma_pos, gamma_neg, clip_margin, epsilon);
    return gradient_matrix;
}

float Asl_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Asl_Cost::computeLossAndGradient", false, false))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    ops::loss::aslGradient(prediction_tensor, target_tensor, gradient_out, gamma_pos, gamma_neg, clip_margin, epsilon);
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::asl(prediction_tensor, target_tensor, loss_matrix, gamma_pos, gamma_neg, clip_margin, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(prediction_tensor.getTotalElements());
}

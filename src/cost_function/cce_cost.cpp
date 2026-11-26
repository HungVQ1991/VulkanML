#include "cost_function/cce_cost.h"

#include "cost_internal_utils.h"
#include "helper/logger.h"
#include "ops/loss/loss_ops.h"

namespace
{

size_t computeSampleCount(const Tensor &t, size_t &out_last_dim) noexcept
{
    const Shape &shape = t.getShape();
    out_last_dim = (shape.getRank() > 0) ? shape[shape.getRank() - 1] : 0;
    return (out_last_dim > 0) ? (t.getTotalElements() / out_last_dim) : t.getRows();
}

}

Cce_Cost::Cce_Cost(float _epsilon, Execution_Target _execution_target)
    : epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      synced_target_matrix(0, 0, _execution_target),
      difference_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
    validateEpsilon(_epsilon);
}

float Cce_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Cce_Cost::computeLoss", true, false))
    {
        return 0.0f;
    }
    size_t last_dim = 0;
    const size_t sample_count = computeSampleCount(_prediction_matrix, last_dim);
    if (sample_count == 0)
    {
        return 0.0f;
    }
    Logger::logMessage(Input_Format{"Cce_Cost::computeLoss: batch_size={}, columns={}", sample_count, last_dim},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LOSS_COMPUTE);
    cost_internal::syncScratchTarget(loss_matrix, _prediction_matrix.getExecutionTarget());
    ops::loss::cce(_prediction_matrix, _target_matrix, loss_matrix, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(sample_count);
}

Tensor Cce_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (!cost_internal::validateDenseInputs(_prediction_matrix, _target_matrix, "Cce_Cost::computeGradient", true, false, true))
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    size_t last_dim = 0;
    const size_t sample_count = computeSampleCount(_prediction_matrix, last_dim);
    if (sample_count == 0)
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }
    Logger::logMessage(Input_Format{"Cce_Cost::computeGradient: batch_size={}, columns={}", sample_count, last_dim},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LOSS_COMPUTE);
    const float inv_batch = 1.0f / static_cast<float>(sample_count);
    cost_internal::computeScaledDiffGradient(
        _prediction_matrix, _target_matrix, inv_batch, synced_target_matrix, difference_matrix, gradient_matrix);
    return gradient_matrix;
}

float Cce_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (!cost_internal::validateDenseInputs(prediction_tensor, target_tensor, "Cce_Cost::computeLossAndGradient", false, false, true))
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    size_t last_dim = 0;
    const size_t sample_count = computeSampleCount(prediction_tensor, last_dim);
    if (sample_count == 0)
    {
        gradient_out = Tensor(0, 0, prediction_tensor.getExecutionTarget());
        return 0.0f;
    }
    const float inv_batch = 1.0f / static_cast<float>(sample_count);
    cost_internal::computeScaledDiffGradientOut(
        prediction_tensor, target_tensor, inv_batch, synced_target_matrix, difference_matrix, gradient_out);
    cost_internal::syncScratchTarget(loss_matrix, prediction_tensor.getExecutionTarget());
    ops::loss::cce(prediction_tensor, target_tensor, loss_matrix, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(sample_count);
}

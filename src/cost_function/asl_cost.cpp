#include "cost_function/asl_cost.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
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
}

float Asl_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
    {
        Logger::logMessage("Asl_Cost::computeLoss: Dimensions mismatch",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
        throw std::invalid_argument("Dimensions mismatch");
    }

    if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
    {
        return 0.0f;
    }

    if (loss_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
    {
        loss_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
    }

    size_t total_elements = _prediction_matrix.getRows() * _prediction_matrix.getColumns();
    ops::loss::asl(_prediction_matrix, _target_matrix, loss_matrix, gamma_pos, gamma_neg, clip_margin, epsilon);
    return loss_matrix.getScalar() / static_cast<float>(total_elements);
}

Tensor Asl_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
    {
        Logger::logMessage("Asl_Cost::computeGradient: Dimensions mismatch",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
        throw std::invalid_argument("Dimensions mismatch");
    }

    if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
    {
        return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
    }

    if (gradient_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
    {
        gradient_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
    }

    ops::loss::aslGradient(_prediction_matrix, _target_matrix, gradient_matrix, gamma_pos, gamma_neg, clip_margin, epsilon);
    return gradient_matrix;
}

float Asl_Cost::computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const
{
    gradient_out = computeGradient(prediction_tensor, target_tensor);
    return computeLoss(prediction_tensor, target_tensor);
}

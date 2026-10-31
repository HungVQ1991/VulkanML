#include "cost_function/dice_bce_cost.h"

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
#include "ops/loss/dice_bce.h"

Dice_BCE_Cost::Dice_BCE_Cost(float _dice_weight, float _bce_weight, float _epsilon, Execution_Target _execution_target)
    : dice_weight(_dice_weight),
      bce_weight(_bce_weight),
      epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
}

float Dice_BCE_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
    {
        Logger::logMessage("Dice_BCE_Cost::computeLoss: Dimensions mismatch",
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

    ops::loss::diceBce(_prediction_matrix, _target_matrix, loss_matrix, dice_weight, bce_weight, epsilon);
    const auto &d = loss_matrix.getData();
    return d.empty() ? 0.0f : d[0];
}

Tensor Dice_BCE_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
    {
        Logger::logMessage("Dice_BCE_Cost::computeGradient: Dimensions mismatch",
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

    if (loss_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
    {
        loss_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
    }
    if (gradient_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
    {
        gradient_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
    }

    ops::loss::diceBce(_prediction_matrix, _target_matrix, loss_matrix, dice_weight, bce_weight, epsilon);
    ops::loss::diceBceGradient(_prediction_matrix, _target_matrix, gradient_matrix, dice_weight, bce_weight, epsilon, &loss_matrix);
    return gradient_matrix;
}

float Dice_BCE_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    if (prediction_tensor.getRows() != target_tensor.getRows() || prediction_tensor.getColumns() != target_tensor.getColumns())
    {
        Logger::logMessage("Dice_BCE_Cost::computeLossAndGradient: Dimensions mismatch",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
        throw std::invalid_argument("Dimensions mismatch");
    }

    if (prediction_tensor.getRows() == 0 || prediction_tensor.getColumns() == 0)
    {
        return 0.0f;
    }

    if (loss_matrix.getExecutionTarget() != prediction_tensor.getExecutionTarget())
    {
        loss_matrix.setExecutionTarget(prediction_tensor.getExecutionTarget());
    }

    ops::loss::diceBce(prediction_tensor, target_tensor, loss_matrix, dice_weight, bce_weight, epsilon);
    ops::loss::diceBceGradient(prediction_tensor, target_tensor, gradient_out, dice_weight, bce_weight, epsilon, &loss_matrix);
    const auto &d = loss_matrix.getData();
    return d.empty() ? 0.0f : d[0];
}

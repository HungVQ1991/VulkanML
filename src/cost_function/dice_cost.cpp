#include "cost_function/dice_cost.h"

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

Dice_Cost::Dice_Cost(float _epsilon, Execution_Target _execution_target)
    : epsilon(_epsilon),
      loss_matrix(0, 0, _execution_target),
      gradient_matrix(0, 0, _execution_target)
{
}

float Dice_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
    {
        Logger::logMessage("Dice_Cost::computeLoss: Dimensions mismatch",
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

    ops::loss::dice(_prediction_matrix, _target_matrix, loss_matrix, epsilon);
    return loss_matrix.getScalar();
}

Tensor Dice_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
    if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
    {
        Logger::logMessage("Dice_Cost::computeGradient: Dimensions mismatch",
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

    ops::loss::diceGradient(_prediction_matrix, _target_matrix, gradient_matrix, epsilon);
    return gradient_matrix;
}

float Dice_Cost::computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const
{
    gradient_out = computeGradient(prediction_tensor, target_tensor);
    return computeLoss(prediction_tensor, target_tensor);
}

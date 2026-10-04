#include "cost_function/cce_cost.h"

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


Cce_Cost::Cce_Cost(float _epsilon, Execution_Target _execution_target)
    : epsilon(_epsilon),
          loss_matrix(0, 0, _execution_target),
          synced_target_matrix(0, 0, _execution_target),
          difference_matrix(0, 0, _execution_target),
          gradient_matrix(0, 0, _execution_target)
{
    }

float Cce_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Cce_Cost::computeLoss: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        size_t batch_size = _prediction_matrix.getRows();
        if (batch_size == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Cce_Cost::computeLoss: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return 0.0f;
        }

        Logger::logMessage(Input_Format{"Cce_Cost::computeLoss: batch_size={}, columns={}",
                                        batch_size,
                                        _prediction_matrix.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        if (loss_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
        {
            loss_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
        }

        _prediction_matrix.cceLoss(_target_matrix, loss_matrix, epsilon);
        return loss_matrix.getScalar() / static_cast<float>(batch_size);
    }

Tensor Cce_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage(Input_Format{"Cce_Cost::computeGradient: Dimensions mismatch! Prediction: ({}x{}), Target: ({}x{})",
                                            _prediction_matrix.getRows(),
                                            _prediction_matrix.getColumns(),
                                            _target_matrix.getRows(),
                                            _target_matrix.getColumns()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Cce_Cost::computeGradient: Dimensions mismatch");
        }

        size_t batch_size = _prediction_matrix.getRows();
        if (batch_size == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Cce_Cost::computeGradient: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
        }

        Logger::logMessage(Input_Format{"Cce_Cost::computeGradient: batch_size={}, columns={}",
                                        batch_size,
                                        _prediction_matrix.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        Execution_Target execution_target = _prediction_matrix.getExecutionTarget();

        synced_target_matrix = _target_matrix;
        if (synced_target_matrix.getExecutionTarget() != execution_target)
        {
            synced_target_matrix.setExecutionTarget(execution_target);
        }

        if (difference_matrix.getExecutionTarget() != execution_target)
        {
            difference_matrix.setExecutionTarget(execution_target);
        }

        if (gradient_matrix.getExecutionTarget() != execution_target)
        {
            gradient_matrix.setExecutionTarget(execution_target);
        }

        float inverse_batch_size = 1.0f / static_cast<float>(batch_size);

        _prediction_matrix.sub(synced_target_matrix, difference_matrix);
        difference_matrix.mulScalar(inverse_batch_size, gradient_matrix);

        return gradient_matrix;
    }

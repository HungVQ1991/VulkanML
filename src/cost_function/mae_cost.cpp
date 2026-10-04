#include "cost_function/mae_cost.h"

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


Mae_Cost::Mae_Cost(Execution_Target _execution_target)
    : loss_matrix(0, 0, _execution_target),
          gradient_matrix(0, 0, _execution_target)
{
    }

float Mae_Cost::computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Mae_Cost::computeLoss: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (_prediction_matrix.getExecutionTarget() != _target_matrix.getExecutionTarget())
        {
            Logger::logMessage("Mae_Cost::computeLoss: Execution target mismatch between prediction and target matrix",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
        }

        if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Mae_Cost::computeLoss: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return 0.0f;
        }

        Logger::logMessage(Input_Format{"Mae_Cost::computeLoss: rows={}, columns={}",
                                        _prediction_matrix.getRows(),
                                        _prediction_matrix.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        if (loss_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
        {
            loss_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
        }

        size_t total_elements = _prediction_matrix.getRows() * _prediction_matrix.getColumns();
        _prediction_matrix.maeLoss(_target_matrix, loss_matrix);
        return loss_matrix.getScalar() / static_cast<float>(total_elements);
    }

Tensor Mae_Cost::computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const
{
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Mae_Cost::computeGradient: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (_prediction_matrix.getExecutionTarget() != _target_matrix.getExecutionTarget())
        {
            Logger::logMessage("Mae_Cost::computeGradient: Execution target mismatch between prediction and target matrix",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
        }

        if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Mae_Cost::computeGradient: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return Tensor(0, 0, _prediction_matrix.getExecutionTarget());
        }

        Logger::logMessage(Input_Format{"Mae_Cost::computeGradient: rows={}, columns={}",
                                        _prediction_matrix.getRows(),
                                        _prediction_matrix.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        std::vector<float> prediction_data = _prediction_matrix.getData();
        std::vector<float> target_data = _target_matrix.getData();
        size_t total_elements = prediction_data.size();
        std::vector<float> gradient_data(total_elements);
        float inverse_total_elements = 1.0f / static_cast<float>(total_elements);

        for (size_t i = 0; i < total_elements; ++i)
        {
            float difference_value = prediction_data[i] - target_data[i];
            if (difference_value > 0.0f)
            {
                gradient_data[i] = inverse_total_elements;
            }
            else if (difference_value < 0.0f)
            {
                gradient_data[i] = -inverse_total_elements;
            }
            else
            {
                gradient_data[i] = 0.0f;
            }
        }

        gradient_matrix.initializeShape(_prediction_matrix.getRows(), _prediction_matrix.getColumns());
        if (gradient_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
        {
            gradient_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
        }
        gradient_matrix.uploadData(gradient_data);

        return gradient_matrix;
    }

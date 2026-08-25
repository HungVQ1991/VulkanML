#pragma once

#include <cstddef>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "icost_function.h"
#include "math/matrix.h"

class Mse_Cost : public ICost_Function
{
private:
    mutable Matrix loss_matrix;
    mutable Matrix synced_target_matrix;
    mutable Matrix difference_matrix;
    mutable Matrix gradient_matrix;

public:
    explicit Mse_Cost(Execution_Target _execution_target = Execution_Target::CPU)
        : loss_matrix(0, 0, _execution_target),
          synced_target_matrix(0, 0, _execution_target),
          difference_matrix(0, 0, _execution_target),
          gradient_matrix(0, 0, _execution_target)
    {
    }

    ~Mse_Cost() noexcept override = default;

     float computeLoss(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const override
    {
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Mse_Cost::computeLoss: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (_prediction_matrix.getExecutionTarget() != _target_matrix.getExecutionTarget())
        {
            Logger::logMessage("Mse_Cost::computeLoss: Execution target mismatch between prediction and target matrix",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
        }

        if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Mse_Cost::computeLoss: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return 0.0f;
        }

        Logger::logMessage(std::format("Mse_Cost::computeLoss: rows={}, columns={}",
                                       _prediction_matrix.getRows(),
                                       _prediction_matrix.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        if (loss_matrix.getExecutionTarget() != _prediction_matrix.getExecutionTarget())
        {
            loss_matrix.setExecutionTarget(_prediction_matrix.getExecutionTarget());
        }

        std::size_t total_elements = _prediction_matrix.getRows() * _prediction_matrix.getColumns();
        _prediction_matrix.mseLoss(_target_matrix, loss_matrix);
        return loss_matrix.getScalar() / static_cast<float>(total_elements);
    }

     Matrix computeGradient(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const override
    {
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Mse_Cost::computeGradient: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Mse_Cost::computeGradient: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return Matrix(0, 0, _prediction_matrix.getExecutionTarget());
        }

        Logger::logMessage(std::format("Mse_Cost::computeGradient: rows={}, columns={}",
                                       _prediction_matrix.getRows(),
                                       _prediction_matrix.getColumns()),
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

        std::size_t total_elements = _prediction_matrix.getRows() * _prediction_matrix.getColumns();
        float scaling_factor = 2.0f / static_cast<float>(total_elements);

        _prediction_matrix.sub(synced_target_matrix, difference_matrix);
        difference_matrix.mulScalar(scaling_factor, gradient_matrix);

        return gradient_matrix;
    }

     Cost_Type getType() const noexcept override
    {
        return Cost_Type::MSE;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override {}
    void loadCheckpoint(std::ifstream &_input_file_stream) override {}
};

using MSE_Cost = Mse_Cost;
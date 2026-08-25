#pragma once

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
#include "icost_function.h"
#include "math/matrix.h"

class Cce_Cost : public ICost_Function
{
private:
    float epsilon = 1e-7f;
    mutable Matrix loss_matrix;
    mutable Matrix synced_target_matrix;
    mutable Matrix difference_matrix;
    mutable Matrix gradient_matrix;

public:
    explicit Cce_Cost(float _epsilon = 1e-7f, Execution_Target _execution_target = Execution_Target::CPU)
        : epsilon(_epsilon),
          loss_matrix(0, 0, _execution_target),
          synced_target_matrix(0, 0, _execution_target),
          difference_matrix(0, 0, _execution_target),
          gradient_matrix(0, 0, _execution_target)
    {
    }

    ~Cce_Cost() noexcept override = default;

     float computeLoss(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const override
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

        std::size_t batch_size = _prediction_matrix.getRows();
        if (batch_size == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Cce_Cost::computeLoss: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return 0.0f;
        }

        Logger::logMessage(std::format("Cce_Cost::computeLoss: batch_size={}, columns={}",
                                       batch_size,
                                       _prediction_matrix.getColumns()),
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

     Matrix computeGradient(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const override
    {
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            std::string error_message = std::format(
                "Cce_Cost::computeGradient: Dimensions mismatch! Prediction: ({}x{}), Target: ({}x{})",
                _prediction_matrix.getRows(),
                _prediction_matrix.getColumns(),
                _target_matrix.getRows(),
                _target_matrix.getColumns());
            Logger::logMessage(error_message, Log_Level::LOG_ERROR, true, 0, Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument(error_message);
        }

        std::size_t batch_size = _prediction_matrix.getRows();
        if (batch_size == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Cce_Cost::computeGradient: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return Matrix(0, 0, _prediction_matrix.getExecutionTarget());
        }

        Logger::logMessage(std::format("Cce_Cost::computeGradient: batch_size={}, columns={}",
                                       batch_size,
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

        float inverse_batch_size = 1.0f / static_cast<float>(batch_size);

        _prediction_matrix.sub(synced_target_matrix, difference_matrix);
        difference_matrix.mulScalar(inverse_batch_size, gradient_matrix);

        return gradient_matrix;
    }

     Cost_Type getType() const noexcept override
    {
        return Cost_Type::CCE;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
    }
};

using CCE_Cost = Cce_Cost;
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

class Huber_Cost : public ICost_Function
{
private:
    float delta = 1.0f;
    mutable Matrix loss_matrix;
    mutable Matrix gradient_matrix;

public:
    explicit Huber_Cost(float _delta = 1.0f, Execution_Target _execution_target = Execution_Target::CPU)
        : delta(_delta),
          loss_matrix(0, 0, _execution_target),
          gradient_matrix(0, 0, _execution_target)
    {
    }

    ~Huber_Cost() noexcept override = default;

    float computeLoss(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const override
    {
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Huber_Cost::computeLoss: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (_prediction_matrix.getExecutionTarget() != _target_matrix.getExecutionTarget())
        {
            Logger::logMessage("Huber_Cost::computeLoss: Execution target mismatch between prediction and target matrix",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
        }

        if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Huber_Cost::computeLoss: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return 0.0f;
        }

        Logger::logMessage(std::format("Huber_Cost::computeLoss: rows={}, columns={}",
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
        _prediction_matrix.huberLoss(_target_matrix, loss_matrix, delta);
        return loss_matrix.getScalar() / static_cast<float>(total_elements);
    }

    Matrix computeGradient(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const override
    {
        if (_prediction_matrix.getRows() != _target_matrix.getRows() || _prediction_matrix.getColumns() != _target_matrix.getColumns())
        {
            Logger::logMessage("Huber_Cost::computeGradient: Dimensions mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (_prediction_matrix.getExecutionTarget() != _target_matrix.getExecutionTarget())
        {
            Logger::logMessage("Huber_Cost::computeGradient: Execution target mismatch between prediction and target matrix",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LOSS_COMPUTE);
        }

        if (_prediction_matrix.getRows() == 0 || _prediction_matrix.getColumns() == 0)
        {
            Logger::logMessage("Huber_Cost::computeGradient: Empty input matrix encountered",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LOSS_COMPUTE);
            return Matrix(0, 0, _prediction_matrix.getExecutionTarget());
        }

        Logger::logMessage(std::format("Huber_Cost::computeGradient: rows={}, columns={}",
                                       _prediction_matrix.getRows(),
                                       _prediction_matrix.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        std::vector<float> prediction_data = _prediction_matrix.getData();
        std::vector<float> target_data = _target_matrix.getData();
        std::size_t total_elements = prediction_data.size();
        std::vector<float> gradient_data(total_elements);
        float inverse_total_elements = 1.0f / static_cast<float>(total_elements);

        for (std::size_t i = 0; i < total_elements; ++i)
        {
            float difference = prediction_data[i] - target_data[i];
            float abs_diff = std::abs(difference);
            if (abs_diff <= delta)
            {
                gradient_data[i] = difference * inverse_total_elements;
            }
            else
            {
                gradient_data[i] = (difference > 0.0f ? delta : -delta) * inverse_total_elements;
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

    Cost_Type getType() const noexcept override
    {
        return Cost_Type::HUBER;
    }

    float getDelta() const noexcept
    {
        return delta;
    }

    void setDelta(float _delta) noexcept
    {
        delta = _delta;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&delta), sizeof(delta));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&delta), sizeof(delta));
    }
};

using Huber_Loss = Huber_Cost;
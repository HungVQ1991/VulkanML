#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/matrix.h"

class Batch_Norm_Layer : public ILayer
{
private:
    std::size_t input_dimension = 0;
    float epsilon = 1e-5f;
    float momentum = 0.1f;
    bool is_training = true;

    Matrix gamma;
    Matrix beta;
    Matrix gamma_gradient;
    Matrix beta_gradient;

    Matrix running_mean;
    Matrix running_variance;
    Matrix batch_mean;
    Matrix batch_variance;
    Matrix normalized_input;

    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

    void initializeParameters()
    {
        std::vector<float> gamma_data(input_dimension, 1.0f);
        std::vector<float> beta_data(input_dimension, 0.0f);
        std::vector<float> mean_data(input_dimension, 0.0f);
        std::vector<float> variance_data(input_dimension, 1.0f);

        gamma = Matrix(1, input_dimension, std::move(gamma_data), execution_target);
        beta = Matrix(1, input_dimension, std::move(beta_data), execution_target);
        gamma_gradient = Matrix(1, input_dimension, execution_target);
        beta_gradient = Matrix(1, input_dimension, execution_target);

        running_mean = Matrix(1, input_dimension, std::move(mean_data), execution_target);
        running_variance = Matrix(1, input_dimension, std::move(variance_data), execution_target);

        batch_mean = Matrix(1, input_dimension, execution_target);
        batch_variance = Matrix(1, input_dimension, execution_target);
    }

public:
    explicit Batch_Norm_Layer(std::size_t _dimension,
                              float _epsilon = 1e-5f,
                              float _momentum = 0.1f,
                              Execution_Target _execution_target = Execution_Target::CPU)
        : input_dimension(_dimension),
          epsilon(_epsilon),
          momentum(_momentum),
          is_training(true),
          gamma(0, 0, _execution_target),
          beta(0, 0, _execution_target),
          gamma_gradient(0, 0, _execution_target),
          beta_gradient(0, 0, _execution_target),
          running_mean(0, 0, _execution_target),
          running_variance(0, 0, _execution_target),
          batch_mean(0, 0, _execution_target),
          batch_variance(0, 0, _execution_target),
          normalized_input(0, 0, _execution_target),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
        initializeParameters();
    }

    ~Batch_Norm_Layer() noexcept override = default;

    void setTrainingMode(bool _is_training) override
    {
        is_training = _is_training;
    }

    Matrix forward(const Matrix &_input_matrix) override
    {
        if (_input_matrix.getColumns() != input_dimension)
        {
            Logger::logMessage("Batch_Norm_Layer::forward: Input dimension mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Input dimension mismatch");
        }

        Logger::logMessage(std::format("Batch_Norm_Layer::forward: dimension={}, mode={}",
                                       input_dimension,
                                       is_training ? "Train" : "Eval"),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        if (input_matrix.getExecutionTarget() != execution_target)
        {
            input_matrix.setExecutionTarget(execution_target);
        }

        input_matrix.batchNormForward(
            gamma,
            beta,
            running_mean,
            running_variance,
            batch_mean,
            batch_variance,
            normalized_input,
            output_matrix,
            epsilon,
            momentum,
            is_training);

        is_forward_completed = true;
        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage("Batch_Norm_Layer::backward: Backward called before forward",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(std::format("Batch_Norm_Layer::backward: output_gradient rows={}, columns={}",
                                       _output_gradient.getRows(),
                                       _output_gradient.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        input_matrix.batchNormBackward(
            _output_gradient,
            gamma,
            batch_variance,
            normalized_input,
            gamma_gradient,
            beta_gradient,
            input_gradient,
            epsilon);

        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

     Matrix getWeights() const override
    {
        return gamma;
    }

     Matrix getBiases() const override
    {
        return beta;
    }

     Matrix getWeightsGradient() override
    {
        return gamma_gradient;
    }

     Matrix getInput() override
    {
        return input_matrix;
    }

     Matrix getOutput() override
    {
        return output_matrix;
    }

     bool hasParameters() const noexcept override
    {
        return true;
    }

     Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::BATCH_NORM;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        std::uint32_t feature_count = static_cast<std::uint32_t>(input_dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        gamma.saveMatrix(_output_file_stream);
        beta.saveMatrix(_output_file_stream);
        running_mean.saveMatrix(_output_file_stream);
        running_variance.saveMatrix(_output_file_stream);
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        gamma = Matrix::loadMatrix(_input_file_stream, execution_target);
        beta = Matrix::loadMatrix(_input_file_stream, execution_target);
        running_mean = Matrix::loadMatrix(_input_file_stream, execution_target);
        running_variance = Matrix::loadMatrix(_input_file_stream, execution_target);
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        gamma.saveMatrix(_output_file_stream);
        beta.saveMatrix(_output_file_stream);
        running_mean.saveMatrix(_output_file_stream);
        running_variance.saveMatrix(_output_file_stream);
        gamma_gradient.saveMatrix(_output_file_stream);
        beta_gradient.saveMatrix(_output_file_stream);
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        gamma = Matrix::loadMatrix(_input_file_stream, execution_target);
        beta = Matrix::loadMatrix(_input_file_stream, execution_target);
        running_mean = Matrix::loadMatrix(_input_file_stream, execution_target);
        running_variance = Matrix::loadMatrix(_input_file_stream, execution_target);
        gamma_gradient = Matrix::loadMatrix(_input_file_stream, execution_target);
        beta_gradient = Matrix::loadMatrix(_input_file_stream, execution_target);
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        return {{&gamma, &gamma_gradient}, {&beta, &beta_gradient}};
    }

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        Logger::logMessage(std::format("Batch_Norm_Layer::setExecutionTarget: Changing execution target from {} to {}",
                                       magic_enum::enum_name(execution_target),
                                       magic_enum::enum_name(_new_execution_target)),
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);

        execution_target = _new_execution_target;
        gamma.setExecutionTarget(_new_execution_target);
        beta.setExecutionTarget(_new_execution_target);
        gamma_gradient.setExecutionTarget(_new_execution_target);
        beta_gradient.setExecutionTarget(_new_execution_target);
        running_mean.setExecutionTarget(_new_execution_target);
        running_variance.setExecutionTarget(_new_execution_target);
        batch_mean.setExecutionTarget(_new_execution_target);
        batch_variance.setExecutionTarget(_new_execution_target);
        normalized_input.setExecutionTarget(_new_execution_target);
        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};
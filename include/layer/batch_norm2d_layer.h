#pragma once

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

class Batch_Norm_2d_Layer : public ILayer
{
private:
    std::uint32_t input_height = 0;
    std::uint32_t input_width = 0;
    std::uint32_t channels = 0;
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

public:
    Batch_Norm_2d_Layer(
        std::uint32_t _height,
        std::uint32_t _width,
        std::uint32_t _channels,
        float _epsilon = 1e-5f,
        float _momentum = 0.1f,
        Execution_Target _execution_target = Execution_Target::CPU)
        : input_height(_height),
          input_width(_width),
          channels(_channels),
          epsilon(_epsilon),
          momentum(_momentum),
          execution_target(_execution_target),
          gamma(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target),
          beta(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target),
          gamma_gradient(1, _channels, _execution_target),
          beta_gradient(1, _channels, _execution_target),
          running_mean(1, _channels, std::vector<float>(_channels, 0.0f), _execution_target),
          running_variance(1, _channels, std::vector<float>(_channels, 1.0f), _execution_target),
          batch_mean(1, _channels, _execution_target),
          batch_variance(1, _channels, _execution_target),
          normalized_input(0, 0, _execution_target),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target)
    {
    }

    ~Batch_Norm_2d_Layer() noexcept override = default;

    void setTrainingMode(bool _is_training) override
    {
        is_training = _is_training;
    }

    Matrix forward(const Matrix &_input_matrix) override
    {
        input_matrix = _input_matrix;

        input_matrix.batchNorm2dForward(
            gamma,
            beta,
            running_mean,
            running_variance,
            batch_mean,
            batch_variance,
            normalized_input,
            output_matrix,
            input_height,
            input_width,
            channels,
            epsilon,
            momentum,
            is_training);

        is_forward_completed = true;
        logBufferAddress(&input_matrix, "input_matrix (Forward)");
        logBufferAddress(&output_matrix, "output_matrix (Forward)");
        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage("Batch_Norm_2d_Layer::backward: Backward called before forward",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        _output_gradient.batchNorm2dBackward(
            gamma,
            batch_variance,
            normalized_input,
            gamma_gradient,
            beta_gradient,
            input_gradient,
            input_height,
            input_width,
            channels,
            epsilon);

        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    [[nodiscard]] bool hasParameters() const noexcept override
    {
        return true;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        return {{&gamma, &gamma_gradient}, {&beta, &beta_gradient}};
    }

    [[nodiscard]] Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::BATCH_NORM_2D;
    }

    [[nodiscard]] Matrix getOutput() override
    {
        return output_matrix;
    }

    [[nodiscard]] Matrix getInput() override
    {
        return input_matrix;
    }

    [[nodiscard]] Matrix getWeights() const override
    {
        return gamma;
    }

    [[nodiscard]] Matrix getBiases() const override
    {
        return beta;
    }

    [[nodiscard]] Matrix getWeightsGradient() override
    {
        return gamma_gradient;
    }

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        Logger::logMessage(std::format("Batch_Norm_2d_Layer::setExecutionTarget: Changing execution target from {} to {}",
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

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&momentum), sizeof(momentum));
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
};

using Batch_Norm2d_Layer = Batch_Norm_2d_Layer;
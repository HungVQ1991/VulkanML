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
#include "math/tensor.h"

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
    bool is_accumulated = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
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
          is_accumulated(false),
          is_forward_completed(false),
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

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Batch_Norm_2d_Layer::forward: expected {} batched parameter tensors (gamma, beta, running_mean, running_variance), got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        Tensor batch_mean_tensor(getExecutionTarget());
        Tensor batch_variance_tensor(getExecutionTarget());
        Tensor normalized_input_tensor(getExecutionTarget());
        Tensor output_tensor(getExecutionTarget());

        _batched_input.batchNorm2dForward(
            _batched_params[0],
            _batched_params[1],
            const_cast<Tensor&>(_batched_params[2]),
            const_cast<Tensor&>(_batched_params[3]),
            batch_mean_tensor,
            batch_variance_tensor,
            normalized_input_tensor,
            output_tensor,
            input_height,
            input_width,
            channels,
            epsilon,
            momentum,
            is_training);

        return output_tensor;
    }

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        return { true, true, false, false };
    }

    std::vector<Shape> getPopulationParameterDims() const noexcept override
    {
        return { Shape{1, channels}, Shape{1, channels}, Shape{1, channels}, Shape{1, channels}};
    }

    bool supportsPopulationBatch() const override { return true; }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Batch_Norm_2d_Layer>(
            input_height, input_width, channels, epsilon, momentum, execution_target);
    }

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override
    {
        if (param_index == 0 || param_index == 3)
        {
            return [](std::mt19937&)
                {
                    return 1.0f;
                };
        }
        else if (param_index == 1 || param_index == 2)
        {
            return [](std::mt19937&)
                {
                    return 0.0f;
                };
        }

        return [](std::mt19937&)
            {
                return 0.0f;
            };
    }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        if (flat_data.size() != channels)
        {
            throw std::invalid_argument("Batch_Norm_2d_Layer::setPopulationParameter: Dimension mismatch");
        }

        switch (param_index)
        {
        case 0:
            gamma = Matrix(1, channels, std::move(flat_data), execution_target);
            break;
        case 1:
            beta = Matrix(1, channels, std::move(flat_data), execution_target);
            break;
        case 2:
            running_mean = Matrix(1, channels, std::move(flat_data), execution_target);
            break;
        case 3:
            running_variance = Matrix(1, channels, std::move(flat_data), execution_target);
            break;
        default:
            throw std::out_of_range("Batch_Norm_2d_Layer::setPopulationParameter: Parameter index out of range");
        }
    }

    bool isAccumulated() const noexcept
    {
        return is_accumulated;
    }

    void setAccumulated(bool _is_accumulated) noexcept
    {
        is_accumulated = _is_accumulated;
    }

    std::uint32_t getInputHeight() const noexcept
    {
        return input_height;
    }

    std::uint32_t getInputWidth() const noexcept
    {
        return input_width;
    }

    std::uint32_t getChannels() const noexcept
    {
        return channels;
    }

    float getEpsilon() const noexcept
    {
        return epsilon;
    }

    float getMomentum() const noexcept
    {
        return momentum;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_2d_Layer::backward: Backward called before forward"},
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

    bool hasParameters() const noexcept override
    {
        return true;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        return {{&gamma, &gamma_gradient}, {&beta, &beta_gradient}};
    }

    Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::BATCH_NORM_2D;
    }

    const Matrix &getOutput() const override
    {
        return output_matrix;
    }

    const Matrix &getInput() const override
    {
        return input_matrix;
    }

    const Matrix &getWeights() const override
    {
        return gamma;
    }

    const Matrix &getBiases() const override
    {
        return beta;
    }

    Execution_Target getExecutionTarget() const override { return execution_target; }

    const Matrix &getWeightsGradient() const override
    {
        return gamma_gradient;
    }

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

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
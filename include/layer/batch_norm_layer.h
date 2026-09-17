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
#include "math/tensor.h"

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
    bool is_accumulated = false;
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
    using ILayer::forward;
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
          is_accumulated(false),
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
            Logger::logMessage(Input_Format{"Batch_Norm_Layer::forward: Input dimension mismatch"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Input dimension mismatch");
        }

        Logger::logMessage(Input_Format{"Batch_Norm_Layer::forward: dimension={}, mode={}",
                                        input_dimension,
                                        is_training ? "Train" : "Eval"},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
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

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Batch_Norm_Layer::forward: expected {} batched parameter tensors (gamma, beta, running_mean, running_variance), got {}",
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

        _batched_input.batchNormForward(
            _batched_params[0],
            _batched_params[1],
            const_cast<Tensor &>(_batched_params[2]),
            const_cast<Tensor &>(_batched_params[3]),
            batch_mean_tensor,
            batch_variance_tensor,
            normalized_input_tensor,
            output_tensor,
            epsilon,
            momentum,
            is_training);

        return output_tensor;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Batch_Norm_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Batch_Norm_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
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

    const Matrix &getWeights() const override
    {
        return gamma;
    }

    const Matrix &getBiases() const override
    {
        return beta;
    }

    const Matrix &getWeightsGradient() const override
    {
        return gamma_gradient;
    }

    const Matrix &getInput() const override
    {
        return input_matrix;
    }

    const Matrix &getOutput() const override
    {
        return output_matrix;
    }

    Execution_Target getExecutionTarget() const override { return execution_target; }

    bool hasParameters() const noexcept override
    {
        return true;
    }

    bool supportsPopulationBatch() const noexcept override
    {
        return true;
    }

    Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::BATCH_NORM;
    }

    std::vector<Shape> getPopulationParameterDims() const override
    {
        return { Shape{ 1, input_dimension }, Shape{ 1, input_dimension }, Shape{ 1, input_dimension }, Shape{ 1, input_dimension } };
    }

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        return {true, true, false, false};
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Batch_Norm_Layer>(input_dimension, epsilon, momentum, execution_target);
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
        if (flat_data.size() != input_dimension)
        {
            throw std::invalid_argument("Batch_Norm_Layer::setPopulationParameter: Dimension mismatch");
        }

        switch (param_index)
        {
        case 0:
            gamma = Matrix(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 1:
            beta = Matrix(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 2:
            running_mean = Matrix(1, input_dimension, std::move(flat_data), execution_target);
            break;
        case 3:
            running_variance = Matrix(1, input_dimension, std::move(flat_data), execution_target);
            break;
        default:
            throw std::out_of_range("Batch_Norm_Layer::setPopulationParameter: Parameter index out of range");
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

    std::size_t getInputDimension() const noexcept
    {
        return input_dimension;
    }

    float getEpsilon() const noexcept
    {
        return epsilon;
    }

    float getMomentum() const noexcept
    {
        return momentum;
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
};
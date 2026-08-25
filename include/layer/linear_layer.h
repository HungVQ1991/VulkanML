#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/matrix.h"

class Linear_Layer : public ILayer
{
private:
    Matrix weights;
    Matrix biases;
    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;

    Matrix weights_gradient;
    Matrix biases_gradient;

    std::size_t input_dimension = 0;
    std::size_t output_dimension = 0;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    Linear_Layer()
        : weights(0, 0),
          biases(0, 0),
          input_matrix(0, 0),
          output_matrix(0, 0),
          input_gradient(0, 0),
          weights_gradient(0, 0),
          biases_gradient(0, 0)
    {
    }

    Linear_Layer(std::size_t _input_dimension,
                 std::size_t _output_dimension,
                 Execution_Target _execution_target = Execution_Target::CPU,
                 float _initialization_gain = 2.0f)
        : weights(0, 0, _execution_target),
          biases(0, 0, _execution_target),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          weights_gradient(_input_dimension, _output_dimension, _execution_target),
          biases_gradient(1, _output_dimension, _execution_target),
          input_dimension(_input_dimension),
          output_dimension(_output_dimension),
          execution_target(_execution_target)
    {
        std::vector<float> weight_data(_input_dimension * _output_dimension);
        std::vector<float> bias_data(_output_dimension, 0.0f);

        float standard_deviation = std::sqrt(_initialization_gain / static_cast<float>(_input_dimension));
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

        float weight_sum = 0.0f;
        for (float &weight_value : weight_data)
        {
            weight_value = normal_distribution(generator);
            weight_sum += std::abs(weight_value);
        }

        float mean_absolute_weight = weight_data.empty() ? 0.0f : weight_sum / static_cast<float>(weight_data.size());
        Logger::logMessage(std::format("Linear_Layer::Linear_Layer: Layer weights mean_abs = {:.8f}", mean_absolute_weight),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        weights = Matrix(_input_dimension, _output_dimension, std::move(weight_data), execution_target);
        biases = Matrix(1, _output_dimension, std::move(bias_data), execution_target);
    }

    ~Linear_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        if (_input_matrix.getColumns() != input_dimension)
        {
            Logger::logMessage("Linear_Layer::forward: Input dimension mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Input dimension mismatch");
        }

        Logger::logMessage(std::format("Linear_Layer::forward: batch_size={}, input_dimension={}, output_dimension={}",
                                       _input_matrix.getRows(),
                                       input_dimension,
                                       output_dimension),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.linearForward(weights, biases, output_matrix);

        logBufferAddress(&weights, "weights (Forward)");
        logBufferAddress(&biases, "biases (Forward)");

        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (_output_gradient.getColumns() != output_dimension || _output_gradient.getRows() != input_matrix.getRows())
        {
            Logger::logMessage("Linear_Layer::backward: Gradient output dimension mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Gradient output dimension mismatch");
        }

        Logger::logMessage(std::format("Linear_Layer::backward: output_gradient rows={}, columns={}",
                                       _output_gradient.getRows(),
                                       _output_gradient.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        input_matrix.linearBackwardWeightBias(_output_gradient, weights_gradient, biases_gradient);
        _output_gradient.linearBackwardInput(weights, input_gradient);

        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        logBufferAddress(&output_matrix, "output_matrix (Backward)");
        logBufferAddress(const_cast<Matrix *>(&_output_gradient), "output_gradient (Backward)");

        return input_gradient;
    }

     Matrix getWeights() const override
    {
        return weights;
    }

     Matrix getBiases() const override
    {
        return biases;
    }

     Matrix getWeightsGradient() override
    {
        return weights_gradient;
    }

     Matrix getBiasesGradient()
    {
        return biases_gradient;
    }

     Matrix getInput() override
    {
        return input_matrix;
    }

     Matrix getOutput() override
    {
        return output_matrix;
    }

     std::size_t getInputDimension() const noexcept
    {
        return input_dimension;
    }

     std::size_t getOutputDimension() const noexcept
    {
        return output_dimension;
    }

    void setWeights(const Matrix &_new_weights)
    {
        if (_new_weights.getRows() != input_dimension || _new_weights.getColumns() != output_dimension)
        {
            Logger::logMessage("Linear_Layer::setWeights: Dimension size of weight must match",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Dimension size of weight must match");
        }
        weights = _new_weights;
    }

    void setBiases(const Matrix &_new_biases)
    {
        if (_new_biases.getRows() != 1 || _new_biases.getColumns() != output_dimension)
        {
            Logger::logMessage("Linear_Layer::setBiases: Dimension size of bias must match",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Dimension size of bias must match");
        }
        biases = _new_biases;
    }

    void resetGradient() override
    {
    }

     bool hasParameters() const noexcept override
    {
        return true;
    }

     Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::LINEAR;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        std::uint32_t input_dimension_value = static_cast<std::uint32_t>(input_dimension);
        std::uint32_t output_dimension_value = static_cast<std::uint32_t>(output_dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&input_dimension_value), sizeof(input_dimension_value));
        _output_file_stream.write(reinterpret_cast<const char *>(&output_dimension_value), sizeof(output_dimension_value));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        weights.saveMatrix(_output_file_stream);
        biases.saveMatrix(_output_file_stream);
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        weights = Matrix::loadMatrix(_input_file_stream, execution_target);
        biases = Matrix::loadMatrix(_input_file_stream, execution_target);
        input_dimension = weights.getRows();
        output_dimension = weights.getColumns();
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        weights.saveMatrix(_output_file_stream);
        biases.saveMatrix(_output_file_stream);
        weights_gradient.saveMatrix(_output_file_stream);
        biases_gradient.saveMatrix(_output_file_stream);
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        weights = Matrix::loadMatrix(_input_file_stream, execution_target);
        biases = Matrix::loadMatrix(_input_file_stream, execution_target);
        weights_gradient = Matrix::loadMatrix(_input_file_stream, execution_target);
        biases_gradient = Matrix::loadMatrix(_input_file_stream, execution_target);
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        return {{&weights, &weights_gradient}, {&biases, &biases_gradient}};
    }

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        Logger::logMessage(std::format("Linear_Layer::setExecutionTarget: Changing execution target from {} to {}",
                                       magic_enum::enum_name(execution_target),
                                       magic_enum::enum_name(_new_execution_target)),
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);

        execution_target = _new_execution_target;
        weights.setExecutionTarget(_new_execution_target);
        biases.setExecutionTarget(_new_execution_target);
        weights_gradient.setExecutionTarget(_new_execution_target);
        biases_gradient.setExecutionTarget(_new_execution_target);
        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};
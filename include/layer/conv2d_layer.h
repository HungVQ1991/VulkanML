#pragma once

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

class Conv2d_Layer : public ILayer
{
private:
    std::uint32_t input_height = 0;
    std::uint32_t input_width = 0;
    std::uint32_t input_channels = 0;
    std::uint32_t output_channels = 0;
    std::uint32_t kernel_size = 0;
    std::uint32_t stride = 1;
    std::uint32_t padding = 0;
    std::uint32_t output_height = 0;
    std::uint32_t output_width = 0;

    Matrix weights;
    Matrix biases;
    Matrix weights_gradient;
    Matrix biases_gradient;
    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

    void initializeWeights()
    {
        std::size_t weight_count = kernel_size * kernel_size * input_channels * output_channels;
        std::vector<float> host_weights(weight_count);
        std::vector<float> host_biases(output_channels, 0.0f);

        float fan_in = static_cast<float>(kernel_size * kernel_size * input_channels);
        float standard_deviation = std::sqrt(2.0f / fan_in);

        std::mt19937 generator(std::random_device{}());
        std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

        for (std::size_t i = 0; i < weight_count; ++i)
        {
            host_weights[i] = normal_distribution(generator);
        }

        weights = Matrix(1, weight_count, host_weights, execution_target);
        biases = Matrix(1, output_channels, host_biases, execution_target);
        weights_gradient = Matrix(1, weight_count, execution_target);
        biases_gradient = Matrix(1, output_channels, execution_target);
    }

public:
    Conv2d_Layer(std::uint32_t _height,
                 std::uint32_t _width,
                 std::uint32_t _input_channels,
                 std::uint32_t _output_channels,
                 std::uint32_t _kernel_size,
                 std::uint32_t _stride,
                 std::uint32_t _padding,
                 Execution_Target _execution_target = Execution_Target::CPU)
        : input_height(_height),
          input_width(_width),
          input_channels(_input_channels),
          output_channels(_output_channels),
          kernel_size(_kernel_size),
          stride(_stride),
          padding(_padding),
          execution_target(_execution_target),
          weights(0, 0, _execution_target),
          biases(0, 0, _execution_target),
          weights_gradient(0, 0, _execution_target),
          biases_gradient(0, 0, _execution_target),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_forward_completed(false)
    {
        output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        initializeWeights();
    }

    ~Conv2d_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(std::format("Conv2d_Layer::forward: input_height={}, input_width={}, input_channels={}, output_channels={}",
                                       input_height, input_width, input_channels, output_channels),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.conv2d(weights, biases, output_matrix, input_height, input_width, input_channels, output_channels, kernel_size, stride, padding);
        is_forward_completed = true;
        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage("Conv2d_Layer::backward: Backward called before forward",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(std::format("Conv2d_Layer::backward: output_gradient rows={}, columns={}",
                                       _output_gradient.getRows(),
                                       _output_gradient.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        input_matrix.conv2dBackwardWeight(_output_gradient, weights_gradient, biases_gradient, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        _output_gradient.conv2dBackwardInput(weights, input_gradient, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    [[nodiscard]] Matrix getWeights() const override
    {
        return weights;
    }

    [[nodiscard]] Matrix getBiases() const override
    {
        return biases;
    }

    [[nodiscard]] Matrix getWeightsGradient() override
    {
        return weights_gradient;
    }

    [[nodiscard]] Matrix getInput() override
    {
        return input_matrix;
    }

    [[nodiscard]] Matrix getOutput() override
    {
        return output_matrix;
    }

    [[nodiscard]] bool hasParameters() const noexcept override
    {
        return true;
    }

    [[nodiscard]] Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::CONV2D;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_channels), sizeof(input_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&output_channels), sizeof(output_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
        _output_file_stream.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
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

        Logger::logMessage(std::format("Conv2d_Layer::setExecutionTarget: Changing execution target from {} to {}",
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
#pragma once

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/matrix.h"

class Relu_Layer : public ILayer
{
private:
    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    explicit Relu_Layer(Execution_Target _execution_target = Execution_Target::CPU)
        : input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
    }

    ~Relu_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(std::format("Relu_Layer::forward: rows={}, columns={}",
                                       _input_matrix.getRows(),
                                       _input_matrix.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.relu(output_matrix);
        is_forward_completed = true;
        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage("Relu_Layer::backward: Relu backward called before forward",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Relu backward called before forward");
        }

        if (_output_gradient.getRows() != output_matrix.getRows() || _output_gradient.getColumns() != output_matrix.getColumns())
        {
            Logger::logMessage("Relu_Layer::backward: Relu gradient dimensions must match output dimensions",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Relu gradient dimensions must match output dimensions");
        }

        Logger::logMessage(std::format("Relu_Layer::backward: output_gradient rows={}, columns={}",
                                       _output_gradient.getRows(),
                                       _output_gradient.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        output_matrix.reluBackward(_output_gradient, input_gradient);
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    [[nodiscard]] bool hasParameters() const noexcept override
    {
        return false;
    }

    [[nodiscard]] Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::RELU;
    }

    [[nodiscard]] Matrix getInput() override
    {
        return input_matrix;
    }

    [[nodiscard]] Matrix getOutput() override
    {
        return output_matrix;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override {}
    void saveInference(std::ofstream &_output_file_stream) const override {}
    void loadInference(std::ifstream &_input_file_stream) override {}
    void saveCheckpoint(std::ofstream &_output_file_stream) const override {}
    void loadCheckpoint(std::ifstream &_input_file_stream) override {}

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        Logger::logMessage(std::format("Relu_Layer::setExecutionTarget: Changing execution target from {} to {}",
                                       magic_enum::enum_name(execution_target),
                                       magic_enum::enum_name(_new_execution_target)),
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);

        execution_target = _new_execution_target;
        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};

using ReLU = Relu_Layer;
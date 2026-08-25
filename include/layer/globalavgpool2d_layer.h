#pragma once

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

class Global_Avg_Pool_2d_Layer : public ILayer
{
private:
    std::uint32_t input_height = 0;
    std::uint32_t input_width = 0;
    std::uint32_t channels = 0;

    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    Global_Avg_Pool_2d_Layer(std::uint32_t _height,
                             std::uint32_t _width,
                             std::uint32_t _channels,
                             Execution_Target _execution_target = Execution_Target::CPU)
        : input_height(_height),
          input_width(_width),
          channels(_channels),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
    }

    ~Global_Avg_Pool_2d_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(std::format("Global_Avg_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                       input_height, input_width, channels),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.globalAvgPool2d(output_matrix, input_height, input_width, channels);
        is_forward_completed = true;
        logBufferAddress(&input_matrix, "input_matrix (Forward)");
        logBufferAddress(&output_matrix, "output_matrix (Forward)");

        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage("Global_Avg_Pool_2d_Layer::backward: Backward called before forward",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(std::format("Global_Avg_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                       _output_gradient.getRows(),
                                       _output_gradient.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        _output_gradient.globalAvgPool2dBackward(input_gradient, input_height, input_width, channels);

        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        logBufferAddress(&input_gradient, "input_gradient (Backward)");
        logBufferAddress(const_cast<Matrix *>(&_output_gradient), "output_gradient (Backward)");

        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

     bool hasParameters() const noexcept override
    {
        return false;
    }

     Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::GLOBAL_AVG_POOL_2D;
    }

     Matrix getInput() override
    {
        return input_matrix;
    }

     Matrix getOutput() override
    {
        return output_matrix;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    }

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

        Logger::logMessage(std::format("Global_Avg_Pool_2d_Layer::setExecutionTarget: Changing execution target from {} to {}",
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

using GlobalAvgPool2d_Layer = Global_Avg_Pool_2d_Layer;
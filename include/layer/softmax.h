#pragma once

#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/matrix.h"

class Softmax_Layer : public ILayer
{
private:
    Matrix input_matrix;
    Matrix cached_output_matrix;
    Matrix input_gradient;
    bool is_fused_with_loss = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    explicit Softmax_Layer(bool _is_fused_with_loss = false, Execution_Target _execution_target = Execution_Target::CPU)
        : input_matrix(0, 0, _execution_target),
          cached_output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_fused_with_loss(_is_fused_with_loss),
          execution_target(_execution_target)
    {
    }

    ~Softmax_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(std::format("Softmax_Layer::forward: rows={}, columns={}",
                                       _input_matrix.getRows(),
                                       _input_matrix.getColumns()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.softmax(cached_output_matrix);
        return cached_output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        Logger::logMessage(std::format("Softmax_Layer::backward: output_gradient rows={}, columns={}, is_fused_with_loss={}",
                                       _output_gradient.getRows(),
                                       _output_gradient.getColumns(),
                                       is_fused_with_loss),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        if (is_fused_with_loss)
        {
            return _output_gradient;
        }

        cached_output_matrix.softmaxBackward(_output_gradient, input_gradient);
        return input_gradient;
    }

    void resetGradient() override
    {
    }

     bool hasParameters() const noexcept override
    {
        return false;
    }

     Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::SOFTMAX;
    }

     Matrix getInput() override
    {
        return input_matrix;
    }

     Matrix getOutput() override
    {
        return cached_output_matrix;
    }

     bool isFusedWithLoss() const noexcept
    {
        return is_fused_with_loss;
    }

    void setFusedWithLoss(bool _is_fused_with_loss) noexcept
    {
        is_fused_with_loss = _is_fused_with_loss;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        std::uint8_t fused_value = is_fused_with_loss ? 1 : 0;
        _output_file_stream.write(reinterpret_cast<const char *>(&fused_value), sizeof(fused_value));
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

        Logger::logMessage(std::format("Softmax_Layer::setExecutionTarget: Changing execution target from {} to {}",
                                       magic_enum::enum_name(execution_target),
                                       magic_enum::enum_name(_new_execution_target)),
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT);

        execution_target = _new_execution_target;
        input_matrix.setExecutionTarget(_new_execution_target);
        cached_output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};

using Softmax = Softmax_Layer;
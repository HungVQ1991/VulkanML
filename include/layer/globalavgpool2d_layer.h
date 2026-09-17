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
#include "math/tensor.h"

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
    bool is_accumulated = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
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
          is_accumulated(false),
          execution_target(_execution_target)
    {
    }

    ~Global_Avg_Pool_2d_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                        input_height, input_width, channels},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.globalAvgPool2d(output_matrix, input_height, input_width, channels);
        is_forward_completed = true;
        logBufferAddress(&input_matrix, "input_matrix (Forward)");
        logBufferAddress(&output_matrix, "output_matrix (Forward)");

        return output_matrix;
    }

    Tensor forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const override
    {
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Conv2d_Layer::forward: expected {} batched parameter tensors (weights, biases), got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        Tensor output_matrix(execution_target);
        _batched_input.globalAvgPool2d(output_matrix, input_height, input_width, channels);
        return output_matrix;
    }

    std::vector<Shape> getPopulationParameterDims() const override
    {
        return {};
    }

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        return {};
    }

    bool supportsPopulationBatch() const override
    {
        return true;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Global_Avg_Pool_2d_Layer>(
            input_height, input_width, channels, execution_target);
    }

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override
    {
        return [](std::mt19937&)
            {
                return 0.0f;
            };
    }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        throw std::out_of_range("Global_Avg_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
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


    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
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

    const Matrix &getInput() const override
    {
        return input_matrix;
    }

    const Matrix &getOutput() const override
    {
        return output_matrix;
    }

    Execution_Target getExecutionTarget() const override { return execution_target; }

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

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};

using GlobalAvgPool2d_Layer = Global_Avg_Pool_2d_Layer;
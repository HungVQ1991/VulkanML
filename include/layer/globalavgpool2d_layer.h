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

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
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
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
    }

    ~Global_Avg_Pool_2d_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override
    {
        Logger::logMessage(Input_Format{"Global_Avg_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                        input_height, input_width, channels},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.globalAvgPool2d(output_tensor, input_height, input_width, channels);
        is_forward_completed = true;
        logBufferAddress(&input_tensor, "input_tensor (Forward)");
        logBufferAddress(&output_tensor, "output_tensor (Forward)");

        return output_tensor;
    }

    Tensor forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const override
    {
        if (_batched_params.size() != getPopulationParameterDims().size())
        {
            Logger::logMessage(Input_Format{ "Global_Avg_Pool_2d_Layer::forward: expected {} batched parameter tensors, got {}",
                                            getPopulationParameterDims().size(), _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.globalAvgPool2d(output_tensor_result, input_height, input_width, channels);
        return output_tensor_result;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Global_Avg_Pool_2d_Layer>(
            input_height, input_width, channels, execution_target);
    }

    Tensor backward(const Tensor &_output_gradient) override
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

        _output_gradient.globalAvgPool2dBackward(input_gradient_tensor, input_height, input_width, channels);

        logBufferAddress(&input_tensor, "input_tensor (Backward)");
        logBufferAddress(&input_gradient_tensor, "input_gradient_tensor (Backward)");
        logBufferAddress(const_cast<Tensor *>(&_output_gradient), "output_gradient (Backward)");

        return input_gradient_tensor;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
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

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override { return [](std::mt19937&) { return 0.0f; }; }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    std::uint32_t getInputHeight() const noexcept { return input_height; }
    std::uint32_t getInputWidth() const noexcept { return input_width; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::GLOBAL_AVG_POOL_2D; }
    std::uint32_t getChannels() const noexcept { return channels; }
    bool supportsPopulationBatch() const override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        throw std::out_of_range("Global_Avg_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
    }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    void setInputHeight(std::uint32_t _height) noexcept { input_height = _height; }
    void setInputWidth(std::uint32_t _width) noexcept { input_width = _width; }
    void setChannels(std::uint32_t _channels) noexcept { channels = _channels; }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
};

using GlobalAvgPool2d_Layer = Global_Avg_Pool_2d_Layer;
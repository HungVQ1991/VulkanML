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

class Max_Pool_2d_Layer : public ILayer
{
private:
    uint32_t input_height = 0;
    uint32_t input_width = 0;
    uint32_t channels = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;
    uint32_t kernel_size = 0;
    uint32_t stride = 1;
    uint32_t padding = 0;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor mask_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    Max_Pool_2d_Layer(
        uint32_t _height,
        uint32_t _width,
        uint32_t _channels,
        uint32_t _kernel_size,
        uint32_t _stride,
        uint32_t _padding,
        Execution_Target _execution_target = Execution_Target::CPU)
        : input_height(_height),
          input_width(_width),
          channels(_channels),
          kernel_size(_kernel_size),
          stride(_stride),
          padding(_padding),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          mask_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
        output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    }

    ~Max_Pool_2d_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override
    {
        Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::forward: input_height={}, input_width={}, channels={}",
                                        input_height,
                                        input_width,
                                        channels},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.maxpool2d(output_tensor, mask_tensor, input_height, input_width, channels, kernel_size, stride, padding);
        is_forward_completed = true;
        logBufferAddress(&mask_tensor, "mask_tensor");
        return output_tensor;
    }

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Max_Pool_2d_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Max_Pool_2d_Layer expects 0 parameters");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        Tensor mask_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.maxpool2d(output_tensor_result, mask_tensor_result, input_height, input_width, channels, kernel_size, stride, padding);
        return output_tensor_result;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Max_Pool_2d_Layer>(
            input_height, input_width, channels, kernel_size, stride, padding, execution_target);
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{"Max_Pool_2d_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                            Log_Level::LOG_DEBUG,
                            true,
                            1,
                            Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        _output_gradient.maxpool2dBackward(mask_tensor, input_gradient_tensor, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
        logBufferAddress(&mask_tensor, "mask_tensor (Backward)");
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
        _output_file_stream.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
        _output_file_stream.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
    }

    void saveInference(std::ofstream &_output_file_stream) const override {}
    void loadInference(std::ifstream &_input_file_stream) override {}

    void saveCheckpoint(std::ofstream &_output_file_stream) const override {}
    void loadCheckpoint(std::ifstream &_input_file_stream) override {}

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override { return [](std::mt19937&) { return 0.0f; }; }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getMask() const noexcept { return mask_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    uint32_t getOutputHeight() const noexcept { return output_height; }
    uint32_t getOutputWidth() const noexcept { return output_width; }
    uint32_t getInputHeight() const noexcept { return input_height; }
    uint32_t getInputWidth() const noexcept { return input_width; }
    uint32_t getKernelSize() const noexcept { return kernel_size; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::MAX_POOL_2D; }
    uint32_t getChannels() const noexcept { return channels; }
    uint32_t getPadding() const noexcept { return padding; }
    uint32_t getStride() const noexcept { return stride; }
    bool supportsPopulationBatch() const override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }

    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override
    {
        throw std::out_of_range("Max_Pool_2d_Layer::setPopulationParameter: Layer has no parameters");
    }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setMask(const Tensor &_tensor) { mask_tensor = _tensor; }
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
        mask_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    void setOutputHeight(uint32_t _height) noexcept { output_height = _height; }
    void setOutputWidth(uint32_t _width) noexcept { output_width = _width; }
    void setInputHeight(uint32_t _height) noexcept { input_height = _height; }
    void setKernelSize(uint32_t _size) noexcept { kernel_size = _size; }
    void setInputWidth(uint32_t _width) noexcept { input_width = _width; }
    void setChannels(uint32_t _channels) noexcept { channels = _channels; }
    void setPadding(uint32_t _padding) noexcept { padding = _padding; }
    void setStride(uint32_t _stride) noexcept { stride = _stride; }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
};

using MaxPool2d_Layer = Max_Pool_2d_Layer;
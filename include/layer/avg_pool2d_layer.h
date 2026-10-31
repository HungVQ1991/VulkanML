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

class Avg_Pool_2d_Layer : public ILayer
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
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    Avg_Pool_2d_Layer(
        uint32_t _height,
        uint32_t _width,
        uint32_t _channels,
        uint32_t _kernel_size,
        uint32_t _stride,
        uint32_t _padding,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~Avg_Pool_2d_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    std::unique_ptr<ILayer> clone() const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t) const override { return [](std::mt19937&) { return 0.0f; }; }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    uint32_t getOutputHeight() const noexcept { return output_height; }
    uint32_t getOutputWidth() const noexcept { return output_width; }
    uint32_t getInputHeight() const noexcept { return input_height; }
    uint32_t getInputWidth() const noexcept { return input_width; }
    uint32_t getKernelSize() const noexcept { return kernel_size; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::AVG_POOL_2D; }
    uint32_t getChannels() const noexcept { return channels; }

    uint32_t getPadding() const noexcept { return padding; }
    uint32_t getStride() const noexcept { return stride; }
    bool supportsPopulationBatch() const override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }
    void setPopulationParameter(size_t, std::vector<float>) override {}

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setOutputHeight(uint32_t _height) noexcept { output_height = _height; }
    void setOutputWidth(uint32_t _width) noexcept { output_width = _width; }
    void setInputHeight(uint32_t _height) noexcept { input_height = _height; }
    void setKernelSize(uint32_t _size) noexcept { kernel_size = _size; }
    void setInputWidth(uint32_t _width) noexcept { input_width = _width; }
    void setChannels(uint32_t _channels) noexcept { channels = _channels; }
    void setPadding(uint32_t _padding) noexcept { padding = _padding; }
    void setStride(uint32_t _stride) noexcept { stride = _stride; }
};

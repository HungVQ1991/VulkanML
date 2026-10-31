#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class Up_Sample_2d_Layer : public ILayer
{
private:
    uint32_t input_height = 0;
    uint32_t input_width = 0;
    uint32_t channels = 0;
    uint32_t scale_factor = 2;
    uint32_t output_height = 0;
    uint32_t output_width = 0;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;

    Up_Sample_2d_Layer(
        uint32_t _height,
        uint32_t _width,
        uint32_t _channels,
        uint32_t _scale_factor = 2,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~Up_Sample_2d_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    std::unique_ptr<ILayer> clone() const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t) const override 
    { 
        return [](std::mt19937&) { return 0.0f; }; 
    }

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override { return {}; }
    std::vector<float> getPopulationParameter(size_t) const override { return {}; }
    void setPopulationParameter(size_t, std::vector<float>) override {}

    uint32_t getInputHeight() const noexcept { return input_height; }
    uint32_t getInputWidth() const noexcept { return input_width; }
    uint32_t getChannels() const noexcept { return channels; }
    uint32_t getScaleFactor() const noexcept { return scale_factor; }
    uint32_t getOutputHeight() const noexcept { return output_height; }
    uint32_t getOutputWidth() const noexcept { return output_width; }

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }

    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }

    Execution_Target getExecutionTarget() const override { return execution_target; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;

    Layer_Type getLayerType() const noexcept override { return Layer_Type::UP_SAMPLE_2D; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }
};

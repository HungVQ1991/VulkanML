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
#include "math/tensor.h"

class Conv2d_Layer : public ILayer
{
private:
    uint32_t input_height = 0;
    uint32_t input_width = 0;
    uint32_t input_channels = 0;
    uint32_t output_channels = 0;
    uint32_t kernel_size = 0;
    uint32_t stride = 1;
    uint32_t padding = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;

    Execution_Target execution_target = Execution_Target::CPU;
    bool is_forward_completed = false;

    Tensor weights;
    Tensor biases;
    Tensor weights_gradient_tensor;
    Tensor biases_gradient_tensor;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    Tensor weights_fp16;
    Tensor biases_fp16;
    Tensor input_tensor_fp16;
    Tensor output_tensor_fp16;
    Tensor input_gradient_tensor_fp16;
    Tensor output_gradient_tensor_fp16;
    Tensor im2col_scratch;
    Tensor im2col_scratch_fp16;
    bool is_weights_fp16_dirty = true;    void initializeWeights();


public:
    using ILayer::forward;    Conv2d_Layer(uint32_t _height,
                 uint32_t _width,
                 uint32_t _input_channels,
                 uint32_t _output_channels,
                 uint32_t _kernel_size,
                 uint32_t _stride,
                 uint32_t _padding,
                 Execution_Target _execution_target = Execution_Target::CPU);


    ~Conv2d_Layer() noexcept override = default;    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    std::unique_ptr<ILayer> clone() const override;
    void invalidateWeightCache() noexcept override;
    void setMixedPrecision(bool _enable) noexcept override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;

    std::vector<Shape> getPopulationParameterDims() const override { return { Shape{ output_channels, input_channels, kernel_size, kernel_size }, Shape{ 1, output_channels } }; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return { true, true }; }    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    const Tensor &getWeightsGradient() const override { return weights_gradient_tensor; }
    const Tensor &getBiasesGradient() const noexcept { return biases_gradient_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getWeights() const override { return weights; }
    const Tensor &getBiases() const override { return biases; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    uint32_t getOutputChannels() const noexcept { return output_channels; }
    uint32_t getInputChannels() const noexcept { return input_channels; }
    uint32_t getOutputHeight() const noexcept { return output_height; }
    uint32_t getOutputWidth() const noexcept { return output_width; }
    uint32_t getInputHeight() const noexcept { return input_height; }
    uint32_t getInputWidth() const noexcept { return input_width; }
    uint32_t getKernelSize() const noexcept { return kernel_size; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    uint32_t getPadding() const noexcept { return padding; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::CONV2D; }
    uint32_t getStride() const noexcept { return stride; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;
    void setWeightsGradient(const Tensor &_tensor);
    void setBiasesGradient(const Tensor &_tensor);

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }    void setWeights(const Tensor &_new_weights);
    void setBiases(const Tensor &_new_biases);

    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setOutputChannels(uint32_t _channels) noexcept;
    void setInputChannels(uint32_t _channels) noexcept;
    void setOutputHeight(uint32_t _height) noexcept;
    void setOutputWidth(uint32_t _width) noexcept;
    void setInputHeight(uint32_t _height) noexcept;
    void setKernelSize(uint32_t _size) noexcept;
    void setInputWidth(uint32_t _width) noexcept;
    void setPadding(uint32_t _padding) noexcept;
    void setStride(uint32_t _stride) noexcept;
    void setIsForwardCompleted(bool _is_completed) noexcept;

};;
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "batch_norm2d_layer.h"
#include "conv2d_layer.h"
#include "gelu.h"
#include "globalavgpool2d_layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "linear_layer.h"
#include "math/tensor.h"
#include "res_net_block_2d_layer.h"

class Res_Net_20_Layer : public ILayer
{
private:
    uint32_t input_height = 32;
    uint32_t input_width = 32;
    uint32_t input_channels = 3;
    uint32_t num_classes = 100;

    Execution_Target execution_target = Execution_Target::CPU;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    bool is_training = true;
    bool is_accumulated = false;

    std::vector<std::unique_ptr<ILayer>> layers;    void buildNetwork();


public:
    using ILayer::forward;    Res_Net_20_Layer(
        uint32_t _height = 32,
        uint32_t _width = 32,
        uint32_t _input_channels = 3,
        uint32_t _num_classes = 100,
        Execution_Target _execution_target = Execution_Target::CPU);


    ~Res_Net_20_Layer() noexcept override = default;    std::unique_ptr<ILayer> clone() const override;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;

    std::vector<Shape> getPopulationParameterDims() const override
    {
        std::vector<Shape> total_dims;
        for (const auto& layer : layers)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        return total_dims;
    }
    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        std::vector<bool> total_evolvable;
        for (const auto& layer : layers)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        return total_evolvable;
    }    const std::vector<std::unique_ptr<ILayer>> &getLayers() const noexcept;

    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    uint32_t getInputChannels() const noexcept { return input_channels; }
    uint32_t getInputHeight() const noexcept { return input_height; }    uint32_t getNumClasses() const noexcept;

    uint32_t getInputWidth() const noexcept { return input_width; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::RES_NET_20; }
    bool supportsPopulationBatch() const noexcept override
    {
        for (const auto& layer : layers)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        return true;
    }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }    bool isAccumulated() const noexcept;

    bool hasParameters() const noexcept override { return true; }    bool isTraining() const noexcept;
    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setInputChannels(uint32_t _channels) noexcept;
    void setInputHeight(uint32_t _height) noexcept;
    void setNumClasses(uint32_t _classes) noexcept;
    void setInputWidth(uint32_t _width) noexcept;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setIsForwardCompleted(bool _is_completed) noexcept;

};;
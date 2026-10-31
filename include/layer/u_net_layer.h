#pragma once

#include <concepts>
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
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"
#include "maxpool2d_layer.h"
#include "sigmoid.h"
#include "upsample2d_layer.h"

class U_Net_Layer : public ILayer
{
private:
    uint32_t input_height = 320;
    uint32_t input_width = 320;
    uint32_t in_channels = 1;
    uint32_t out_channels = 1;
    uint32_t base_channels = 32;

    Execution_Target execution_target = Execution_Target::CPU;
    bool is_forward_completed = false;
    bool is_training = true;
    bool is_accumulated = false;

    // Các tầng Encoder & Pool
    std::vector<std::unique_ptr<ILayer>> enc1_layers; // in_channels -> base (Conv-BN-GELU x2)
    std::unique_ptr<Max_Pool_2d_Layer> pool1;         // 320 -> 160

    std::vector<std::unique_ptr<ILayer>> enc2_layers; // base -> base*2 (Conv-BN-GELU x2)
    std::unique_ptr<Max_Pool_2d_Layer> pool2;         // 160 -> 80

    std::vector<std::unique_ptr<ILayer>> enc3_layers; // base*2 -> base*4 (Conv-BN-GELU x2)
    std::unique_ptr<Max_Pool_2d_Layer> pool3;         // 80 -> 40

    // Bottleneck
    std::vector<std::unique_ptr<ILayer>> bottleneck_layers; // base*4 -> base*8 (Conv-BN-GELU x2)

    // Decoder 3: 40 -> 80
    std::unique_ptr<Up_Sample_2d_Layer> up3;
    std::vector<std::unique_ptr<ILayer>> dec3_layers; // (base*8 + base*4) -> base*4

    // Decoder 2: 80 -> 160
    std::unique_ptr<Up_Sample_2d_Layer> up2;
    std::vector<std::unique_ptr<ILayer>> dec2_layers; // (base*4 + base*2) -> base*2

    // Decoder 1: 160 -> 320
    std::unique_ptr<Up_Sample_2d_Layer> up1;
    std::vector<std::unique_ptr<ILayer>> dec1_layers; // (base*2 + base) -> base

    // Head
    std::unique_ptr<Conv2d_Layer> final_conv;         // base -> out_channels (1x1 conv)
    std::unique_ptr<Sigmoid_Layer> final_sigmoid;

    // Cache cho activations phục vụ Backward pass
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    Tensor skip1_act;
    Tensor skip2_act;
    Tensor skip3_act;
    Tensor dec3_concat;
    Tensor dec2_concat;
    Tensor dec1_concat;

    void buildNetwork();

public:
    using ILayer::forward;

    U_Net_Layer(
        uint32_t _height = 320,
        uint32_t _width = 320,
        uint32_t _in_channels = 1,
        uint32_t _out_channels = 1,
        uint32_t _base_channels = 32,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~U_Net_Layer() noexcept override = default;

    std::unique_ptr<ILayer> clone() const override;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t) const override 
    { 
        return [](std::mt19937&) { return 0.0f; }; 
    }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<float> getPopulationParameter(size_t) const override { return {}; }
    void setPopulationParameter(size_t, std::vector<float>) override {}

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }

    Execution_Target getExecutionTarget() const override { return execution_target; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;

    void inspectAllBatchNorms() const;
    bool hasNanInRunningStats() const;
    void resetRunningStats();
    void resetHeadBias(float bias = -2.0f);
    Layer_Type getLayerType() const noexcept override { return Layer_Type::U_NET; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }
};

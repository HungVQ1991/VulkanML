#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "conv2d_layer.h"
#include "instance_norm2d_layer.h"
#include "leaky_relu_layer.h"
#include "maxpool2d_layer.h"
#include "upsample2d_layer.h"
#include "sigmoid.h"
#include "ilayer.h"
#include "math/tensor.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"

class NN_UNet_Layer : public ILayer
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

    std::vector<std::unique_ptr<ILayer>> enc1_layers;
    std::unique_ptr<Max_Pool_2d_Layer> pool1;

    std::vector<std::unique_ptr<ILayer>> enc2_layers;
    std::unique_ptr<Max_Pool_2d_Layer> pool2;

    std::vector<std::unique_ptr<ILayer>> enc3_layers;
    std::unique_ptr<Max_Pool_2d_Layer> pool3;

    std::vector<std::unique_ptr<ILayer>> enc4_layers;
    std::unique_ptr<Max_Pool_2d_Layer> pool4;

    std::vector<std::unique_ptr<ILayer>> bottleneck_layers;

    std::unique_ptr<Up_Sample_2d_Layer> up4;
    std::vector<std::unique_ptr<ILayer>> dec4_layers;

    std::unique_ptr<Up_Sample_2d_Layer> up3;
    std::vector<std::unique_ptr<ILayer>> dec3_layers;

    std::unique_ptr<Up_Sample_2d_Layer> up2;
    std::vector<std::unique_ptr<ILayer>> dec2_layers;

    std::unique_ptr<Up_Sample_2d_Layer> up1;
    std::vector<std::unique_ptr<ILayer>> dec1_layers;

    std::unique_ptr<Conv2d_Layer> final_conv;
    std::unique_ptr<Sigmoid_Layer> final_sigmoid;

    // Deep Supervision Heads
    std::unique_ptr<Conv2d_Layer> ds3_conv;
    std::unique_ptr<Sigmoid_Layer> ds3_sigmoid;
    std::unique_ptr<Conv2d_Layer> ds2_conv;
    std::unique_ptr<Sigmoid_Layer> ds2_sigmoid;

    bool enable_deep_supervision = true;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    Tensor skip1_act;
    Tensor skip2_act;
    Tensor skip3_act;
    Tensor skip4_act;

    Tensor dec4_concat;
    Tensor dec3_concat;
    Tensor dec2_concat;
    Tensor dec1_concat;

    // Deep Supervision Output and Gradient Tensors
    Tensor ds3_output;
    Tensor ds2_output;
    Tensor ds3_grad;
    Tensor ds2_grad;

    // Residual intermediate activations for each stage
    Tensor res_enc1_act;
    Tensor res_enc2_act;
    Tensor res_enc3_act;
    Tensor res_enc4_act;
    Tensor res_bn_act;
    Tensor res_dec4_act;
    Tensor res_dec3_act;
    Tensor res_dec2_act;
    Tensor res_dec1_act;

    void buildNetwork();

public:
    using ILayer::forward;

    NN_UNet_Layer(
        uint32_t _height = 160,
        uint32_t _width = 160,
        uint32_t _in_channels = 1,
        uint32_t _out_channels = 1,
        uint32_t _base_channels = 32,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~NN_UNet_Layer() noexcept override = default;

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

    void setEnableDeepSupervision(bool _enable) noexcept { enable_deep_supervision = _enable; }
    [[nodiscard]] bool isDeepSupervisionEnabled() const noexcept { return enable_deep_supervision; }

    const Tensor &getDeepSupervisionOutput2() const noexcept { return ds2_output; }
    const Tensor &getDeepSupervisionOutput3() const noexcept { return ds3_output; }

    void setDeepSupervisionGradients(const Tensor &_grad2, const Tensor &_grad3)
    {
        ds2_grad = _grad2;
        ds3_grad = _grad3;
    }
    void clearDeepSupervisionGradients()
    {
        ds2_grad = Tensor(0, 0, execution_target);
        ds3_grad = Tensor(0, 0, execution_target);
    }

    Execution_Target getExecutionTarget() const override { return execution_target; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;

    void resetHeadBias(float bias = -1.5f);
    Layer_Type getLayerType() const noexcept override { return Layer_Type::NN_UNET; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }

    uint32_t getHeight() const noexcept { return input_height; }
    uint32_t getWidth() const noexcept { return input_width; }
    uint32_t getInChannels() const noexcept { return in_channels; }
    uint32_t getOutChannels() const noexcept { return out_channels; }
    uint32_t getBaseChannels() const noexcept { return base_channels; }
};

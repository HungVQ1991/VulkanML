#pragma once

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
    uint32_t input_height = 160;
    uint32_t input_width = 160;
    uint32_t in_channels = 1;
    uint32_t out_channels = 1;
    uint32_t base_channels = 32;

    Execution_Target execution_target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT32;
    bool is_forward_completed = false;
    bool is_training = true;

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

    std::unique_ptr<Conv2d_Layer> ds3_conv;
    std::unique_ptr<Sigmoid_Layer> ds3_sigmoid;
    std::unique_ptr<Conv2d_Layer> ds2_conv;
    std::unique_ptr<Sigmoid_Layer> ds2_sigmoid;

    bool enable_deep_supervision = true;
    bool ds_forward_executed = false;

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

    Tensor ds3_output;
    Tensor ds2_output;
    Tensor ds3_grad;
    Tensor ds2_grad;

    Tensor res_enc1_act;
    Tensor res_enc2_act;
    Tensor res_enc3_act;
    Tensor res_enc4_act;
    Tensor res_bn_act;
    Tensor res_dec4_act;
    Tensor res_dec3_act;
    Tensor res_dec2_act;
    Tensor res_dec1_act;

    Tensor in_nhwc_scratch;
    Tensor out_nchw_scratch;
    Tensor ds2_out_nchw_scratch;
    Tensor ds3_out_nchw_scratch;
    Tensor grad_nhwc_scratch;
    Tensor in_grad_nchw_scratch;
    Tensor ds2_grad_nhwc_scratch;
    Tensor ds3_grad_nhwc_scratch;
    Tensor sig_grad_cast_scratch;
    Tensor ds2_sig_grad_cast;
    Tensor ds3_sig_grad_cast;
    Tensor ds2_add_cast;
    Tensor ds2_add_sum;
    Tensor ds3_add_cast;
    Tensor ds3_add_sum;
    Tensor up4_cast_scratch;
    Tensor up3_cast_scratch;
    Tensor up2_cast_scratch;
    Tensor up1_cast_scratch;
    Tensor dec4_concat_2d_scratch;
    Tensor dec3_concat_2d_scratch;
    Tensor dec2_concat_2d_scratch;
    Tensor dec1_concat_2d_scratch;
    Tensor d_up4_scratch;
    Tensor d_skip4_scratch;
    Tensor d_skip4_cast_scratch;
    Tensor enc4_grad_sum;
    Tensor d_up3_scratch;
    Tensor d_skip3_scratch;
    Tensor d_skip3_cast_scratch;
    Tensor enc3_grad_sum;
    Tensor d_up2_scratch;
    Tensor d_skip2_scratch;
    Tensor d_skip2_cast_scratch;
    Tensor enc2_grad_sum;
    Tensor d_up1_scratch;
    Tensor d_skip1_scratch;
    Tensor d_skip1_cast_scratch;
    Tensor enc1_grad_sum;
    Tensor res_fwd_cast[9];
    Tensor res_fwd_sum[9];
    Tensor res_bwd_cast[9];
    Tensor res_bwd_sum[9];

    void buildNetwork();

public:
    using ILayer::forward;

    NN_UNet_Layer(
        uint32_t _height = 160,
        uint32_t _width = 160,
        uint32_t _in_channels = 1,
        uint32_t _out_channels = 1,
        uint32_t _base_channels = 32,
        Execution_Target _execution_target = Execution_Target::CPU,
        Data_Type _data_type = Data_Type::FLOAT32);

    ~NN_UNet_Layer() noexcept override = default;

    std::unique_ptr<ILayer> clone() const override;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    bool supportsStaticGraph() const noexcept override
    {
        auto ok_vec = [](const auto &vec) noexcept {
            for (const auto &l : vec)
            {
                if (l && !l->supportsStaticGraph()) return false;
            }
            return true;
        };
        auto ok = [](const auto &ptr) noexcept { return !ptr || ptr->supportsStaticGraph(); };
        return ok_vec(enc1_layers) && ok(pool1) &&
               ok_vec(enc2_layers) && ok(pool2) &&
               ok_vec(enc3_layers) && ok(pool3) &&
               ok_vec(enc4_layers) && ok(pool4) &&
               ok_vec(bottleneck_layers) &&
               ok(up4) && ok_vec(dec4_layers) &&
               ok(up3) && ok_vec(dec3_layers) &&
               ok(up2) && ok_vec(dec2_layers) &&
               ok(up1) && ok_vec(dec1_layers) &&
               ok(final_conv) && ok(final_sigmoid) &&
               ok(ds3_conv) && ok(ds3_sigmoid) &&
               ok(ds2_conv) && ok(ds2_sigmoid);
    }
    void prepareInference() override;
    void quantizeToTernary() override;
    void dequantizeFromTernary() override;
    [[nodiscard]] bool isTernary() const noexcept override { return data_type == Data_Type::TERNARY_INT2; }
    [[nodiscard]] Data_Type getDataType() const noexcept { return data_type; }
    void snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const override;
    void restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor) override;

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
    std::vector<const Tensor *> getNoDecayParameters() const override;
    std::vector<float> getPopulationParameter(size_t) const override { return {}; }
    void setPopulationParameter(size_t, std::vector<float>) override {}

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }

    static constexpr uint32_t NN_UNET_DS_DISABLED_FLAG = 0x80000000u;

    void setEnableDeepSupervision(bool _enable);
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

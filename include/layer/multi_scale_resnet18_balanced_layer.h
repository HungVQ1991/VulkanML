#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "layer/conv2d_layer.h"
#include "layer/batch_norm2d_layer.h"
#include "layer/gelu.h"
#include "layer/maxpool2d_layer.h"
#include "layer/globalavgpool2d_layer.h"
#include "layer/res_net_block_2d_layer.h"
#include "math/tensor.h"

class Multi_Scale_ResNet18_Balanced_Layer : public ILayer
{
private:
    uint32_t input_height = 320;
    uint32_t input_width = 320;
    uint32_t in_channels = 2;
    Execution_Target execution_target = Execution_Target::CPU;
    bool use_resnet_d = true;

    std::unique_ptr<Conv2d_Layer> stem_conv;
    std::unique_ptr<Batch_Norm_2d_Layer> stem_bn;
    std::unique_ptr<Gelu_Layer> stem_act;
    std::unique_ptr<Conv2d_Layer> stem_downsample_conv;
    std::unique_ptr<Batch_Norm_2d_Layer> stem_downsample_bn;
    std::unique_ptr<Gelu_Layer> stem_downsample_act;

    std::unique_ptr<Res_Net_Block_2d_Layer> stage1_block1;
    std::unique_ptr<Res_Net_Block_2d_Layer> stage1_block2;

    std::unique_ptr<Res_Net_Block_2d_Layer> stage2_block1;
    std::unique_ptr<Res_Net_Block_2d_Layer> stage2_block2;
    std::unique_ptr<Global_Avg_Pool_2d_Layer> gap2;
    std::unique_ptr<Batch_Norm_2d_Layer> bn_v2;

    std::unique_ptr<Res_Net_Block_2d_Layer> stage3_block1;
    std::unique_ptr<Res_Net_Block_2d_Layer> stage3_block2;
    std::unique_ptr<Global_Avg_Pool_2d_Layer> gap3;
    std::unique_ptr<Batch_Norm_2d_Layer> bn_v3;

    std::unique_ptr<Res_Net_Block_2d_Layer> stage4_block1;
    std::unique_ptr<Res_Net_Block_2d_Layer> stage4_block2;
    std::unique_ptr<Global_Avg_Pool_2d_Layer> gap4;
    std::unique_ptr<Batch_Norm_2d_Layer> bn_v4;

    uint32_t h_stage2_dim = 0;
    uint32_t w_stage2_dim = 0;
    uint32_t h_stage3_dim = 0;
    uint32_t w_stage3_dim = 0;
    uint32_t h_stage4_dim = 0;
    uint32_t w_stage4_dim = 0;

    float lambda_gmp = 0.2f;
    float last_forward_lambda_gmp = 0.2f;
    bool is_training = true;

    Tensor input_tensor;
    Tensor stage1_output;
    Tensor stage2_output;
    Tensor stage3_output;
    Tensor stage4_output;
    Tensor final_output;

    Tensor gmp2_out;
    Tensor gmp2_mask;
    Tensor gmp2_grad;
    Tensor gmp3_out;
    Tensor gmp3_mask;
    Tensor gmp3_grad;
    Tensor gmp4_out;
    Tensor gmp4_mask;
    Tensor gmp4_grad;
    Tensor v2_scaled;
    Tensor v2_sum;
    Tensor v3_scaled;
    Tensor v3_sum;
    Tensor v4_scaled;
    Tensor v4_sum;
    Tensor v23_concat;
    Tensor g23_split;
    Tensor g4_split;
    Tensor g2_split;
    Tensor g3_split;
    Tensor dO4_gmp_scaled;
    Tensor dO3_gmp_scaled;
    Tensor dO3_sum_tmp;
    Tensor dO2_gmp_scaled;
    Tensor dO2_sum_tmp;

    Tensor stage4_gradient;
    Tensor stage3_gradient;
    Tensor stage2_gradient;
    Tensor stage1_gradient;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;

    void buildNetwork();

public:
    static constexpr uint32_t STAGE1_CHANNELS = 48;
    static constexpr uint32_t STAGE2_CHANNELS = 80;
    static constexpr uint32_t STAGE3_CHANNELS = 160;
    static constexpr uint32_t STAGE4_CHANNELS = 320;
    static constexpr uint32_t STAGE23_SPLIT_DIM = STAGE2_CHANNELS + STAGE3_CHANNELS;
    static constexpr uint32_t OUTPUT_DIM = STAGE2_CHANNELS + STAGE3_CHANNELS + STAGE4_CHANNELS;

    using ILayer::forward;

    Multi_Scale_ResNet18_Balanced_Layer(
        uint32_t _height = 320,
        uint32_t _width = 320,
        uint32_t _in_channels = 2,
        Execution_Target _execution_target = Execution_Target::CPU,
        bool _use_resnet_d = true);

    ~Multi_Scale_ResNet18_Balanced_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;
    std::unique_ptr<ILayer> clone() const override;
    void resetGradient() override;
    void resetGradients() override;
    bool supportsStaticGraph() const noexcept override
    {
        auto ok = [](const auto &ptr) noexcept { return !ptr || ptr->supportsStaticGraph(); };
        return ok(stem_conv) && ok(stem_bn) && ok(stem_act) &&
               ok(stem_downsample_conv) && ok(stem_downsample_bn) && ok(stem_downsample_act) &&
               ok(stage1_block1) && ok(stage1_block2) &&
               ok(stage2_block1) && ok(stage2_block2) && ok(gap2) && ok(bn_v2) &&
               ok(stage3_block1) && ok(stage3_block2) && ok(gap3) && ok(bn_v3) &&
               ok(stage4_block1) && ok(stage4_block2) && ok(gap4) && ok(bn_v4);
    }
    void prepareInference() override;
    void quantizeToTernary() override;
    void dequantizeFromTernary() override;
    bool isTernary() const noexcept override;
    void snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const override;
    void restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor) override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<const Tensor *> getNoDecayParameters() const override;
    void zeroInitLastBatchNorm();

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t) const override { return [](std::mt19937&) { return 0.0f; }; }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }

    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return final_output; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }

    const Tensor &getStage1Output() const noexcept { return stage1_output; }
    const Tensor &getStage2Output() const noexcept { return stage2_output; }
    const Tensor &getStage3Output() const noexcept { return stage3_output; }
    const Tensor &getStage4Output() const noexcept { return stage4_output; }
    const Tensor &getStage1Gradient() const noexcept { return stage1_gradient; }
    const Tensor &getStage2Gradient() const noexcept { return stage2_gradient; }
    const Tensor &getStage3Gradient() const noexcept { return stage3_gradient; }
    const Tensor &getStage4Gradient() const noexcept { return stage4_gradient; }

    uint32_t getInputChannels() const noexcept { return in_channels; }
    bool expandStemFrom1To2Channels();
    uint32_t getInputHeight() const noexcept { return input_height; }
    uint32_t getInputWidth() const noexcept { return input_width; }
    uint32_t getStage1Height() const noexcept { return ((input_height + 1) / 2 + 1) / 2; }
    uint32_t getStage1Width() const noexcept { return ((input_width + 1) / 2 + 1) / 2; }
    uint32_t getStage2Height() const noexcept { return h_stage2_dim; }
    uint32_t getStage2Width() const noexcept { return w_stage2_dim; }
    uint32_t getStage3Height() const noexcept { return h_stage3_dim; }
    uint32_t getStage3Width() const noexcept { return w_stage3_dim; }
    uint32_t getStage4Height() const noexcept { return h_stage4_dim; }
    uint32_t getStage4Width() const noexcept { return w_stage4_dim; }
    bool getUseResNetD() const noexcept { return use_resnet_d; }
    uint32_t getOutputDimension() const noexcept { return OUTPUT_DIM; }
    void setLambdaGmp(float _val) noexcept { lambda_gmp = _val; }
    float getLambdaGmp() const noexcept { return lambda_gmp; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::MULTI_SCALE_RESNET_18; }
    bool hasParameters() const noexcept override { return true; }
    bool supportsPopulationBatch() const override { return false; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }

    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void invalidateWeightCache() noexcept override;
};

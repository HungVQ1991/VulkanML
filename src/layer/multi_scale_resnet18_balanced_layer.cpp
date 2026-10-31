#include "layer/multi_scale_resnet18_balanced_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "math/shape.h"
#include "math/tensor.h"

Multi_Scale_ResNet18_Balanced_Layer::Multi_Scale_ResNet18_Balanced_Layer(
    uint32_t _height,
    uint32_t _width,
    uint32_t _in_channels,
    Execution_Target _execution_target,
    bool _use_resnet_d)
    : input_height(_height),
      input_width(_width),
      in_channels(_in_channels),
      execution_target(_execution_target),
      use_resnet_d(_use_resnet_d),
      lambda_gmp(0.2f),
      input_tensor(0, 0, _execution_target),
      stage1_output(0, 0, _execution_target),
      stage2_output(0, 0, _execution_target),
      stage3_output(0, 0, _execution_target),
      stage4_output(0, 0, _execution_target),
      final_output(0, 0, _execution_target),
      stage4_gradient(0, 0, _execution_target),
      stage3_gradient(0, 0, _execution_target),
      stage2_gradient(0, 0, _execution_target),
      stage1_gradient(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false)
{
    buildNetwork();
}

void Multi_Scale_ResNet18_Balanced_Layer::buildNetwork()
{
    uint32_t h = input_height;
    uint32_t w = input_width;
    uint32_t h_stem = h / 2;
    uint32_t w_stem = w / 2;
    uint32_t h_stage1 = h_stem / 2;
    uint32_t w_stage1 = w_stem / 2;

    stem_conv = std::make_unique<Conv2d_Layer>(h, w, in_channels, 48, 3, 2, 1, execution_target);
    stem_bn = std::make_unique<Batch_Norm_2d_Layer>(h_stem, w_stem, 48, 1e-5f, 0.1f, execution_target);
    stem_act = std::make_unique<Gelu_Layer>(execution_target);

    stem_downsample_conv = std::make_unique<Conv2d_Layer>(h_stem, w_stem, 48, 48, 3, 2, 1, execution_target);
    stem_downsample_bn = std::make_unique<Batch_Norm_2d_Layer>(h_stage1, w_stage1, 48, 1e-5f, 0.1f, execution_target);
    stem_downsample_act = std::make_unique<Gelu_Layer>(execution_target);

    stage1_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage1, w_stage1, 48, 48, 1, execution_target, use_resnet_d);
    stage1_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage1, w_stage1, 48, 48, 1, execution_target, use_resnet_d);

    uint32_t h_stage2 = h_stage1 / 2;
    uint32_t w_stage2 = w_stage1 / 2;
    stage2_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage1, w_stage1, 48, 80, 2, execution_target, use_resnet_d);
    stage2_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage2, w_stage2, 80, 80, 1, execution_target, use_resnet_d);
    gap2 = std::make_unique<Global_Avg_Pool_2d_Layer>(h_stage2, w_stage2, 80, execution_target);
    gmp2 = std::make_unique<Max_Pool_2d_Layer>(h_stage2, w_stage2, 80, h_stage2, 1, 0, execution_target);
    bn_v2 = std::make_unique<Batch_Norm_2d_Layer>(1, 1, 80, 1e-5f, 0.1f, execution_target);

    uint32_t h_stage3 = h_stage2 / 2;
    uint32_t w_stage3 = w_stage2 / 2;
    stage3_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage2, w_stage2, 80, 160, 2, execution_target, use_resnet_d);
    stage3_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage3, w_stage3, 160, 160, 1, execution_target, use_resnet_d);
    gap3 = std::make_unique<Global_Avg_Pool_2d_Layer>(h_stage3, w_stage3, 160, execution_target);
    gmp3 = std::make_unique<Max_Pool_2d_Layer>(h_stage3, w_stage3, 160, h_stage3, 1, 0, execution_target);
    bn_v3 = std::make_unique<Batch_Norm_2d_Layer>(1, 1, 160, 1e-5f, 0.1f, execution_target);

    uint32_t h_stage4 = h_stage3 / 2;
    uint32_t w_stage4 = w_stage3 / 2;
    stage4_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage3, w_stage3, 160, 320, 2, execution_target, use_resnet_d);
    stage4_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage4, w_stage4, 320, 320, 1, execution_target, use_resnet_d);
    gap4 = std::make_unique<Global_Avg_Pool_2d_Layer>(h_stage4, w_stage4, 320, execution_target);
    gmp4 = std::make_unique<Max_Pool_2d_Layer>(h_stage4, w_stage4, 320, h_stage4, 1, 0, execution_target);
    bn_v4 = std::make_unique<Batch_Norm_2d_Layer>(1, 1, 320, 1e-5f, 0.1f, execution_target);

    zeroInitLastBatchNorm();
}

Tensor Multi_Scale_ResNet18_Balanced_Layer::forward(const Tensor &_input_tensor)
{
    input_tensor = _input_tensor;

    Tensor x = stem_conv->forward(input_tensor);
    x = stem_bn->forward(x);
    x = stem_act->forward(x);
    x = stem_downsample_conv->forward(x);
    x = stem_downsample_bn->forward(x);
    x = stem_downsample_act->forward(x);

    x = stage1_block1->forward(x);
    stage1_output = stage1_block2->forward(x);

    x = stage2_block1->forward(stage1_output);
    stage2_output = stage2_block2->forward(x);
    Tensor v2 = gap2->forward(stage2_output); // (B, 80)
    if (lambda_gmp > 0.0f)
    {
        Tensor v2_max = gmp2->forward(stage2_output);
        size_t b_rows = v2.getRows();
        if (v2_max.getTotalElements() == b_rows * 80)
        {
            v2_max.reshape(Shape{b_rows, 80});
        }
        else
        {
            size_t spatial = v2_max.getTotalElements() / (b_rows * 80);
            std::vector<float> reduced(b_rows * 80, -std::numeric_limits<float>::infinity());
            const auto& raw_max = v2_max.getData();
            for (size_t b = 0; b < b_rows; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < 80; ++c)
                    {
                        float val = raw_max[(b * spatial + s) * 80 + c];
                        if (val > reduced[b * 80 + c]) reduced[b * 80 + c] = val;
                    }
                }
            }
            v2_max = Tensor(Shape{b_rows, 80}, std::move(reduced), v2.getExecutionTarget());
        }
        v2 = v2 + v2_max * lambda_gmp;
    }
    v2 = bn_v2->forward(v2);

    x = stage3_block1->forward(stage2_output);
    stage3_output = stage3_block2->forward(x);
    Tensor v3 = gap3->forward(stage3_output); 
    if (lambda_gmp > 0.0f)
    {
        Tensor v3_max = gmp3->forward(stage3_output);
        size_t b_rows = v3.getRows();
        if (v3_max.getTotalElements() == b_rows * 160)
        {
            v3_max.reshape(Shape{b_rows, 160});
        }
        else
        {
            size_t spatial = v3_max.getTotalElements() / (b_rows * 160);
            std::vector<float> reduced(b_rows * 160, -std::numeric_limits<float>::infinity());
            const auto& raw_max = v3_max.getData();
            for (size_t b = 0; b < b_rows; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < 160; ++c)
                    {
                        float val = raw_max[(b * spatial + s) * 160 + c];
                        if (val > reduced[b * 160 + c]) reduced[b * 160 + c] = val;
                    }
                }
            }
            v3_max = Tensor(Shape{b_rows, 160}, std::move(reduced), v3.getExecutionTarget());
        }
        v3 = v3 + v3_max * lambda_gmp;
    }
    v3 = bn_v3->forward(v3);

    // Stage 4
    x = stage4_block1->forward(stage3_output);
    stage4_output = stage4_block2->forward(x);
    Tensor v4 = gap4->forward(stage4_output); // (B, 320)
    if (lambda_gmp > 0.0f)
    {
        Tensor v4_max = gmp4->forward(stage4_output);
        size_t b_rows = v4.getRows();
        if (v4_max.getTotalElements() == b_rows * 320)
        {
            v4_max.reshape(Shape{b_rows, 320});
        }
        else
        {
            size_t spatial = v4_max.getTotalElements() / (b_rows * 320);
            std::vector<float> reduced(b_rows * 320, -std::numeric_limits<float>::infinity());
            const auto& raw_max = v4_max.getData();
            for (size_t b = 0; b < b_rows; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < 320; ++c)
                    {
                        float val = raw_max[(b * spatial + s) * 320 + c];
                        if (val > reduced[b * 320 + c]) reduced[b * 320 + c] = val;
                    }
                }
            }
            v4_max = Tensor(Shape{b_rows, 320}, std::move(reduced), v4.getExecutionTarget());
        }
        v4 = v4 + v4_max * lambda_gmp;
    }
    v4 = bn_v4->forward(v4);

    // Multi-Scale Feature Aggregation
    Tensor v23 = v2.concatenateColumns(v3); // (B, 240)
    final_output = v23.concatenateColumns(v4); // (B, 560)

    is_forward_completed = true;
    return final_output;
}

Tensor Multi_Scale_ResNet18_Balanced_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::runtime_error("Multi_Scale_ResNet18_Balanced_Layer::backward: Forward must be called before backward");
    }

    // Split 560 into (240, 320), then 240 into (80, 160)
    auto [g23, g4_concat] = _output_gradient.splitColumns(240);
    auto [g2_concat, g3_concat] = g23.splitColumns(80);

    // Backward through the 3 batch norms
    Tensor g4 = bn_v4->backward(g4_concat);
    Tensor g3 = bn_v3->backward(g3_concat);
    Tensor g2 = bn_v2->backward(g2_concat);

    // Stage 4 Backward + Accumulate GAP + GMP gradient
    Tensor dO4_gap = gap4->backward(g4);
    if (lambda_gmp > 0.0f)
    {
        Tensor dO4_gmp = gmp4->backward(g4) * lambda_gmp;
        stage4_gradient = dO4_gap + dO4_gmp;
    }
    else
    {
        stage4_gradient = dO4_gap;
    }
    Tensor dO3_s4 = stage4_block2->backward(stage4_gradient);
    dO3_s4 = stage4_block1->backward(dO3_s4);

    // Stage 3 Backward + Accumulate GAP + GMP gradient
    Tensor dO3_gap = gap3->backward(g3);
    if (lambda_gmp > 0.0f)
    {
        Tensor dO3_gmp = gmp3->backward(g3) * lambda_gmp;
        stage3_gradient = dO3_s4 + dO3_gap + dO3_gmp;
    }
    else
    {
        stage3_gradient = dO3_s4 + dO3_gap;
    }
    Tensor dO2_s3 = stage3_block2->backward(stage3_gradient);
    dO2_s3 = stage3_block1->backward(dO2_s3);

    // Stage 2 Backward + Accumulate GAP + GMP gradient
    Tensor dO2_gap = gap2->backward(g2);
    if (lambda_gmp > 0.0f)
    {
        Tensor dO2_gmp = gmp2->backward(g2) * lambda_gmp;
        stage2_gradient = dO2_s3 + dO2_gap + dO2_gmp;
    }
    else
    {
        stage2_gradient = dO2_s3 + dO2_gap;
    }
    Tensor dO1 = stage2_block2->backward(stage2_gradient);
    dO1 = stage2_block1->backward(dO1);

    // Stage 1 Backward
    stage1_gradient = dO1;
    Tensor dO0 = stage1_block2->backward(dO1);
    dO0 = stage1_block1->backward(dO0);

    // Stem Backward
    Tensor stem_g = stem_downsample_act->backward(dO0);
    stem_g = stem_downsample_bn->backward(stem_g);
    stem_g = stem_downsample_conv->backward(stem_g);

    stem_g = stem_act->backward(stem_g);
    stem_g = stem_bn->backward(stem_g);
    input_gradient_tensor = stem_conv->backward(stem_g);

    return input_gradient_tensor;
}

std::unique_ptr<ILayer> Multi_Scale_ResNet18_Balanced_Layer::clone() const
{
    auto cloned = std::make_unique<Multi_Scale_ResNet18_Balanced_Layer>(
        input_height, input_width, in_channels, execution_target, use_resnet_d);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setAccumulated(is_accumulated);
    cloned->setLambdaGmp(lambda_gmp);
    return cloned;
}

void Multi_Scale_ResNet18_Balanced_Layer::resetGradient()
{
    is_forward_completed = false;
    input_tensor = Tensor(0, 0, execution_target);
    stage1_output = Tensor(0, 0, execution_target);
    stage2_output = Tensor(0, 0, execution_target);
    stage3_output = Tensor(0, 0, execution_target);
    stage4_output = Tensor(0, 0, execution_target);
    final_output = Tensor(0, 0, execution_target);
    stage4_gradient = Tensor(0, 0, execution_target);
    stage3_gradient = Tensor(0, 0, execution_target);
    stage2_gradient = Tensor(0, 0, execution_target);
    stage1_gradient = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);

    stem_conv->resetGradient();
    stem_bn->resetGradient();
    stem_act->resetGradient();
    stem_downsample_conv->resetGradient();
    stem_downsample_bn->resetGradient();
    stem_downsample_act->resetGradient();

    stage1_block1->resetGradient();
    stage1_block2->resetGradient();

    stage2_block1->resetGradient();
    stage2_block2->resetGradient();
    gap2->resetGradient();
    gmp2->resetGradient();
    bn_v2->resetGradient();

    stage3_block1->resetGradient();
    stage3_block2->resetGradient();
    gap3->resetGradient();
    gmp3->resetGradient();
    bn_v3->resetGradient();

    stage4_block1->resetGradient();
    stage4_block2->resetGradient();
    gap4->resetGradient();
    gmp4->resetGradient();
    bn_v4->resetGradient();
}

void Multi_Scale_ResNet18_Balanced_Layer::resetGradients()
{
    resetGradient();

    stem_conv->resetGradients();
    stem_bn->resetGradients();
    stem_act->resetGradients();
    stem_downsample_conv->resetGradients();
    stem_downsample_bn->resetGradients();
    stem_downsample_act->resetGradients();

    stage1_block1->resetGradients();
    stage1_block2->resetGradients();

    stage2_block1->resetGradients();
    stage2_block2->resetGradients();
    gap2->resetGradients();
    gmp2->resetGradients();
    bn_v2->resetGradients();

    stage3_block1->resetGradients();
    stage3_block2->resetGradients();
    gap3->resetGradients();
    gmp3->resetGradients();
    bn_v3->resetGradients();

    stage4_block1->resetGradients();
    stage4_block2->resetGradients();
    gap4->resetGradients();
    gmp4->resetGradients();
    bn_v4->resetGradients();
}

std::vector<std::pair<Tensor *, Tensor *>> Multi_Scale_ResNet18_Balanced_Layer::getParametersAndGradients()
{
    std::vector<std::pair<Tensor *, Tensor *>> total_params;

    auto appendParams = [&total_params](ILayer *layer) {
        if (layer)
        {
            auto params = layer->getParametersAndGradients();
            total_params.insert(total_params.end(), params.begin(), params.end());
        }
    };

    appendParams(stem_conv.get());
    appendParams(stem_bn.get());
    appendParams(stem_downsample_conv.get());
    appendParams(stem_downsample_bn.get());

    appendParams(stage1_block1.get());
    appendParams(stage1_block2.get());

    appendParams(stage2_block1.get());
    appendParams(stage2_block2.get());
    appendParams(bn_v2.get());

    appendParams(stage3_block1.get());
    appendParams(stage3_block2.get());
    appendParams(bn_v3.get());

    appendParams(stage4_block1.get());
    appendParams(stage4_block2.get());
    appendParams(bn_v4.get());

    return total_params;
}

std::vector<const Tensor *> Multi_Scale_ResNet18_Balanced_Layer::getNoDecayParameters() const
{
    std::vector<const Tensor *> no_decay;

    auto appendNoDecay = [&no_decay](const ILayer *layer) {
        if (layer)
        {
            auto params = layer->getNoDecayParameters();
            no_decay.insert(no_decay.end(), params.begin(), params.end());
        }
    };

    appendNoDecay(stem_conv.get());
    appendNoDecay(stem_bn.get());
    appendNoDecay(stem_downsample_conv.get());
    appendNoDecay(stem_downsample_bn.get());

    appendNoDecay(stage1_block1.get());
    appendNoDecay(stage1_block2.get());

    appendNoDecay(stage2_block1.get());
    appendNoDecay(stage2_block2.get());
    appendNoDecay(bn_v2.get());

    appendNoDecay(stage3_block1.get());
    appendNoDecay(stage3_block2.get());
    appendNoDecay(bn_v3.get());

    appendNoDecay(stage4_block1.get());
    appendNoDecay(stage4_block2.get());
    appendNoDecay(bn_v4.get());

    return no_decay;
}

void Multi_Scale_ResNet18_Balanced_Layer::zeroInitLastBatchNorm()
{
    stage1_block1->zeroInitLastBatchNorm();
    stage1_block2->zeroInitLastBatchNorm();
    stage2_block1->zeroInitLastBatchNorm();
    stage2_block2->zeroInitLastBatchNorm();
    stage3_block1->zeroInitLastBatchNorm();
    stage3_block2->zeroInitLastBatchNorm();
    stage4_block1->zeroInitLastBatchNorm();
    stage4_block2->zeroInitLastBatchNorm();
}

void Multi_Scale_ResNet18_Balanced_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _output_file_stream.write(reinterpret_cast<const char *>(&in_channels), sizeof(in_channels));
    uint8_t d_flag = (use_resnet_d ? 1 : 0) | (lambda_gmp > 0.0f ? 0x80 : 0);
    _output_file_stream.write(reinterpret_cast<const char *>(&d_flag), sizeof(d_flag));
    if (d_flag & 0x80)
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&lambda_gmp), sizeof(lambda_gmp));
    }
}

void Multi_Scale_ResNet18_Balanced_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    stem_conv->saveInference(_output_file_stream);
    stem_bn->saveInference(_output_file_stream);
    stem_downsample_conv->saveInference(_output_file_stream);
    stem_downsample_bn->saveInference(_output_file_stream);

    stage1_block1->saveInference(_output_file_stream);
    stage1_block2->saveInference(_output_file_stream);

    stage2_block1->saveInference(_output_file_stream);
    stage2_block2->saveInference(_output_file_stream);
    bn_v2->saveInference(_output_file_stream);

    stage3_block1->saveInference(_output_file_stream);
    stage3_block2->saveInference(_output_file_stream);
    bn_v3->saveInference(_output_file_stream);

    stage4_block1->saveInference(_output_file_stream);
    stage4_block2->saveInference(_output_file_stream);
    bn_v4->saveInference(_output_file_stream);
}

void Multi_Scale_ResNet18_Balanced_Layer::loadInference(std::ifstream &_input_file_stream)
{
    stem_conv->loadInference(_input_file_stream);
    stem_bn->loadInference(_input_file_stream);
    stem_downsample_conv->loadInference(_input_file_stream);
    stem_downsample_bn->loadInference(_input_file_stream);

    stage1_block1->loadInference(_input_file_stream);
    stage1_block2->loadInference(_input_file_stream);

    stage2_block1->loadInference(_input_file_stream);
    stage2_block2->loadInference(_input_file_stream);
    bn_v2->loadInference(_input_file_stream);

    stage3_block1->loadInference(_input_file_stream);
    stage3_block2->loadInference(_input_file_stream);
    bn_v3->loadInference(_input_file_stream);

    stage4_block1->loadInference(_input_file_stream);
    stage4_block2->loadInference(_input_file_stream);
    bn_v4->loadInference(_input_file_stream);
}

void Multi_Scale_ResNet18_Balanced_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    stem_conv->saveCheckpoint(_output_file_stream);
    stem_bn->saveCheckpoint(_output_file_stream);
    stem_downsample_conv->saveCheckpoint(_output_file_stream);
    stem_downsample_bn->saveCheckpoint(_output_file_stream);

    stage1_block1->saveCheckpoint(_output_file_stream);
    stage1_block2->saveCheckpoint(_output_file_stream);

    stage2_block1->saveCheckpoint(_output_file_stream);
    stage2_block2->saveCheckpoint(_output_file_stream);
    bn_v2->saveCheckpoint(_output_file_stream);

    stage3_block1->saveCheckpoint(_output_file_stream);
    stage3_block2->saveCheckpoint(_output_file_stream);
    bn_v3->saveCheckpoint(_output_file_stream);

    stage4_block1->saveCheckpoint(_output_file_stream);
    stage4_block2->saveCheckpoint(_output_file_stream);
    bn_v4->saveCheckpoint(_output_file_stream);
}

void Multi_Scale_ResNet18_Balanced_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    stem_conv->loadCheckpoint(_input_file_stream);
    stem_bn->loadCheckpoint(_input_file_stream);
    stem_downsample_conv->loadCheckpoint(_input_file_stream);
    stem_downsample_bn->loadCheckpoint(_input_file_stream);

    stage1_block1->loadCheckpoint(_input_file_stream);
    stage1_block2->loadCheckpoint(_input_file_stream);

    stage2_block1->loadCheckpoint(_input_file_stream);
    stage2_block2->loadCheckpoint(_input_file_stream);
    bn_v2->loadCheckpoint(_input_file_stream);

    stage3_block1->loadCheckpoint(_input_file_stream);
    stage3_block2->loadCheckpoint(_input_file_stream);
    bn_v3->loadCheckpoint(_input_file_stream);

    stage4_block1->loadCheckpoint(_input_file_stream);
    stage4_block2->loadCheckpoint(_input_file_stream);
    bn_v4->loadCheckpoint(_input_file_stream);
}

void Multi_Scale_ResNet18_Balanced_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    input_tensor.setExecutionTarget(_new_execution_target);
    stage1_output.setExecutionTarget(_new_execution_target);
    stage2_output.setExecutionTarget(_new_execution_target);
    stage3_output.setExecutionTarget(_new_execution_target);
    stage4_output.setExecutionTarget(_new_execution_target);
    final_output.setExecutionTarget(_new_execution_target);
    stage4_gradient.setExecutionTarget(_new_execution_target);
    stage3_gradient.setExecutionTarget(_new_execution_target);
    stage2_gradient.setExecutionTarget(_new_execution_target);
    stage1_gradient.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);

    stem_conv->setExecutionTarget(_new_execution_target);
    stem_bn->setExecutionTarget(_new_execution_target);
    stem_act->setExecutionTarget(_new_execution_target);
    stem_downsample_conv->setExecutionTarget(_new_execution_target);
    stem_downsample_bn->setExecutionTarget(_new_execution_target);
    stem_downsample_act->setExecutionTarget(_new_execution_target);

    stage1_block1->setExecutionTarget(_new_execution_target);
    stage1_block2->setExecutionTarget(_new_execution_target);

    stage2_block1->setExecutionTarget(_new_execution_target);
    stage2_block2->setExecutionTarget(_new_execution_target);
    gap2->setExecutionTarget(_new_execution_target);
    gmp2->setExecutionTarget(_new_execution_target);
    bn_v2->setExecutionTarget(_new_execution_target);

    stage3_block1->setExecutionTarget(_new_execution_target);
    stage3_block2->setExecutionTarget(_new_execution_target);
    gap3->setExecutionTarget(_new_execution_target);
    gmp3->setExecutionTarget(_new_execution_target);
    bn_v3->setExecutionTarget(_new_execution_target);

    stage4_block1->setExecutionTarget(_new_execution_target);
    stage4_block2->setExecutionTarget(_new_execution_target);
    gap4->setExecutionTarget(_new_execution_target);
    gmp4->setExecutionTarget(_new_execution_target);
    bn_v4->setExecutionTarget(_new_execution_target);
}

void Multi_Scale_ResNet18_Balanced_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);

    stem_conv->setMixedPrecision(_enable);
    stem_bn->setMixedPrecision(_enable);
    stem_act->setMixedPrecision(_enable);
    stem_downsample_conv->setMixedPrecision(_enable);
    stem_downsample_bn->setMixedPrecision(_enable);
    stem_downsample_act->setMixedPrecision(_enable);

    stage1_block1->setMixedPrecision(_enable);
    stage1_block2->setMixedPrecision(_enable);

    stage2_block1->setMixedPrecision(_enable);
    stage2_block2->setMixedPrecision(_enable);
    gap2->setMixedPrecision(_enable);
    gmp2->setMixedPrecision(_enable);
    bn_v2->setMixedPrecision(_enable);

    stage3_block1->setMixedPrecision(_enable);
    stage3_block2->setMixedPrecision(_enable);
    gap3->setMixedPrecision(_enable);
    gmp3->setMixedPrecision(_enable);
    bn_v3->setMixedPrecision(_enable);

    stage4_block1->setMixedPrecision(_enable);
    stage4_block2->setMixedPrecision(_enable);
    gap4->setMixedPrecision(_enable);
    gmp4->setMixedPrecision(_enable);
    bn_v4->setMixedPrecision(_enable);
}

void Multi_Scale_ResNet18_Balanced_Layer::setTrainingMode(bool _is_training)
{
    ILayer::setTrainingMode(_is_training);

    stem_conv->setTrainingMode(_is_training);
    stem_bn->setTrainingMode(_is_training);
    stem_act->setTrainingMode(_is_training);
    stem_downsample_conv->setTrainingMode(_is_training);
    stem_downsample_bn->setTrainingMode(_is_training);
    stem_downsample_act->setTrainingMode(_is_training);

    stage1_block1->setTrainingMode(_is_training);
    stage1_block2->setTrainingMode(_is_training);

    stage2_block1->setTrainingMode(_is_training);
    stage2_block2->setTrainingMode(_is_training);
    gap2->setTrainingMode(_is_training);
    gmp2->setTrainingMode(_is_training);
    bn_v2->setTrainingMode(_is_training);

    stage3_block1->setTrainingMode(_is_training);
    stage3_block2->setTrainingMode(_is_training);
    gap3->setTrainingMode(_is_training);
    gmp3->setTrainingMode(_is_training);
    bn_v3->setTrainingMode(_is_training);

    stage4_block1->setTrainingMode(_is_training);
    stage4_block2->setTrainingMode(_is_training);
    gap4->setTrainingMode(_is_training);
    gmp4->setTrainingMode(_is_training);
    bn_v4->setTrainingMode(_is_training);
}

void Multi_Scale_ResNet18_Balanced_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    ILayer::setAccumulated(_is_accumulated);

    stem_conv->setAccumulated(_is_accumulated);
    stem_bn->setAccumulated(_is_accumulated);
    stem_act->setAccumulated(_is_accumulated);
    stem_downsample_conv->setAccumulated(_is_accumulated);
    stem_downsample_bn->setAccumulated(_is_accumulated);
    stem_downsample_act->setAccumulated(_is_accumulated);

    stage1_block1->setAccumulated(_is_accumulated);
    stage1_block2->setAccumulated(_is_accumulated);

    stage2_block1->setAccumulated(_is_accumulated);
    stage2_block2->setAccumulated(_is_accumulated);
    gap2->setAccumulated(_is_accumulated);
    gmp2->setAccumulated(_is_accumulated);
    bn_v2->setAccumulated(_is_accumulated);

    stage3_block1->setAccumulated(_is_accumulated);
    stage3_block2->setAccumulated(_is_accumulated);
    gap3->setAccumulated(_is_accumulated);
    gmp3->setAccumulated(_is_accumulated);
    bn_v3->setAccumulated(_is_accumulated);

    stage4_block1->setAccumulated(_is_accumulated);
    stage4_block2->setAccumulated(_is_accumulated);
    gap4->setAccumulated(_is_accumulated);
    gmp4->setAccumulated(_is_accumulated);
    bn_v4->setAccumulated(_is_accumulated);
}

void Multi_Scale_ResNet18_Balanced_Layer::invalidateWeightCache() noexcept
{
    stem_conv->invalidateWeightCache();
    stem_bn->invalidateWeightCache();
    stem_downsample_conv->invalidateWeightCache();
    stem_downsample_bn->invalidateWeightCache();

    stage1_block1->invalidateWeightCache();
    stage1_block2->invalidateWeightCache();

    stage2_block1->invalidateWeightCache();
    stage2_block2->invalidateWeightCache();
    bn_v2->invalidateWeightCache();

    stage3_block1->invalidateWeightCache();
    stage3_block2->invalidateWeightCache();
    bn_v3->invalidateWeightCache();

    stage4_block1->invalidateWeightCache();
    stage4_block2->invalidateWeightCache();
    bn_v4->invalidateWeightCache();
}

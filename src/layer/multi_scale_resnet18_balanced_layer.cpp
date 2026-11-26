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

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/shape.h"
#include "math/tensor.h"
#include "ops/vision/pooling.h"

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
      last_forward_lambda_gmp(0.2f),
      is_training(true),
      input_tensor(0, 0, _execution_target),
      stage1_output(0, 0, _execution_target),
      stage2_output(0, 0, _execution_target),
      stage3_output(0, 0, _execution_target),
      stage4_output(0, 0, _execution_target),
      final_output(0, 0, _execution_target),
      gmp2_out(0, 0, _execution_target),
      gmp2_mask(0, 0, _execution_target),
      gmp2_grad(0, 0, _execution_target),
      gmp3_out(0, 0, _execution_target),
      gmp3_mask(0, 0, _execution_target),
      gmp3_grad(0, 0, _execution_target),
      gmp4_out(0, 0, _execution_target),
      gmp4_mask(0, 0, _execution_target),
      gmp4_grad(0, 0, _execution_target),
      v2_scaled(0, 0, _execution_target),
      v2_sum(0, 0, _execution_target),
      v3_scaled(0, 0, _execution_target),
      v3_sum(0, 0, _execution_target),
      v4_scaled(0, 0, _execution_target),
      v4_sum(0, 0, _execution_target),
      v23_concat(0, 0, _execution_target),
      g23_split(0, 0, _execution_target),
      g4_split(0, 0, _execution_target),
      g2_split(0, 0, _execution_target),
      g3_split(0, 0, _execution_target),
      dO4_gmp_scaled(0, 0, _execution_target),
      dO3_gmp_scaled(0, 0, _execution_target),
      dO3_sum_tmp(0, 0, _execution_target),
      dO2_gmp_scaled(0, 0, _execution_target),
      dO2_sum_tmp(0, 0, _execution_target),
      stage4_gradient(0, 0, _execution_target),
      stage3_gradient(0, 0, _execution_target),
      stage2_gradient(0, 0, _execution_target),
      stage1_gradient(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      is_forward_completed(false)
{
    if (input_height == 0 || input_width == 0 || in_channels == 0)
    {
        throw std::invalid_argument("Multi_Scale_ResNet18_Balanced_Layer: input_height, input_width, and in_channels must be > 0");
    }
    buildNetwork();
}

void Multi_Scale_ResNet18_Balanced_Layer::buildNetwork()
{
    uint32_t h = input_height;
    uint32_t w = input_width;
    uint32_t h_stem = (h + 1) / 2;
    uint32_t w_stem = (w + 1) / 2;
    uint32_t h_stage1 = (h_stem + 1) / 2;
    uint32_t w_stage1 = (w_stem + 1) / 2;

    stem_conv = std::make_unique<Conv2d_Layer>(h, w, in_channels, STAGE1_CHANNELS, 3, 2, 1, execution_target);
    stem_bn = std::make_unique<Batch_Norm_2d_Layer>(h_stem, w_stem, STAGE1_CHANNELS, 1e-5f, 0.1f, execution_target);
    stem_act = std::make_unique<Gelu_Layer>(execution_target);

    stem_downsample_conv = std::make_unique<Conv2d_Layer>(h_stem, w_stem, STAGE1_CHANNELS, STAGE1_CHANNELS, 3, 2, 1, execution_target);
    stem_downsample_bn = std::make_unique<Batch_Norm_2d_Layer>(h_stage1, w_stage1, STAGE1_CHANNELS, 1e-5f, 0.1f, execution_target);
    stem_downsample_act = std::make_unique<Gelu_Layer>(execution_target);

    stage1_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage1, w_stage1, STAGE1_CHANNELS, STAGE1_CHANNELS, 1, execution_target, use_resnet_d);
    stage1_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage1, w_stage1, STAGE1_CHANNELS, STAGE1_CHANNELS, 1, execution_target, use_resnet_d);

    uint32_t h_stage2 = (h_stage1 + 1) / 2;
    uint32_t w_stage2 = (w_stage1 + 1) / 2;
    h_stage2_dim = h_stage2;
    w_stage2_dim = w_stage2;
    stage2_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage1, w_stage1, STAGE1_CHANNELS, STAGE2_CHANNELS, 2, execution_target, use_resnet_d);
    stage2_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage2, w_stage2, STAGE2_CHANNELS, STAGE2_CHANNELS, 1, execution_target, use_resnet_d);
    gap2 = std::make_unique<Global_Avg_Pool_2d_Layer>(h_stage2, w_stage2, STAGE2_CHANNELS, execution_target);
    bn_v2 = std::make_unique<Batch_Norm_2d_Layer>(1, 1, STAGE2_CHANNELS, 1e-5f, 0.1f, execution_target);

    uint32_t h_stage3 = (h_stage2 + 1) / 2;
    uint32_t w_stage3 = (w_stage2 + 1) / 2;
    h_stage3_dim = h_stage3;
    w_stage3_dim = w_stage3;
    stage3_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage2, w_stage2, STAGE2_CHANNELS, STAGE3_CHANNELS, 2, execution_target, use_resnet_d);
    stage3_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage3, w_stage3, STAGE3_CHANNELS, STAGE3_CHANNELS, 1, execution_target, use_resnet_d);
    gap3 = std::make_unique<Global_Avg_Pool_2d_Layer>(h_stage3, w_stage3, STAGE3_CHANNELS, execution_target);
    bn_v3 = std::make_unique<Batch_Norm_2d_Layer>(1, 1, STAGE3_CHANNELS, 1e-5f, 0.1f, execution_target);

    uint32_t h_stage4 = (h_stage3 + 1) / 2;
    uint32_t w_stage4 = (w_stage3 + 1) / 2;
    h_stage4_dim = h_stage4;
    w_stage4_dim = w_stage4;
    stage4_block1 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage3, w_stage3, STAGE3_CHANNELS, STAGE4_CHANNELS, 2, execution_target, use_resnet_d);
    stage4_block2 = std::make_unique<Res_Net_Block_2d_Layer>(h_stage4, w_stage4, STAGE4_CHANNELS, STAGE4_CHANNELS, 1, execution_target, use_resnet_d);
    gap4 = std::make_unique<Global_Avg_Pool_2d_Layer>(h_stage4, w_stage4, STAGE4_CHANNELS, execution_target);
    bn_v4 = std::make_unique<Batch_Norm_2d_Layer>(1, 1, STAGE4_CHANNELS, 1e-5f, 0.1f, execution_target);

    zeroInitLastBatchNorm();
}

Tensor Multi_Scale_ResNet18_Balanced_Layer::forward(const Tensor &_input_tensor)
{
    input_tensor = _input_tensor;
    last_forward_lambda_gmp = lambda_gmp;

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
    Tensor v2 = gap2->forward(stage2_output);
    if (last_forward_lambda_gmp > 0.0f)
    {
        ops::vision::globalMaxPool2d(stage2_output, gmp2_out, gmp2_mask, h_stage2_dim, w_stage2_dim, STAGE2_CHANNELS);
        gmp2_out.mulScalar(last_forward_lambda_gmp, v2_scaled);
        v2.add(v2_scaled, v2_sum);
        v2 = v2_sum;
    }
    v2 = bn_v2->forward(v2);

    x = stage3_block1->forward(stage2_output);
    stage3_output = stage3_block2->forward(x);
    Tensor v3 = gap3->forward(stage3_output);
    if (last_forward_lambda_gmp > 0.0f)
    {
        ops::vision::globalMaxPool2d(stage3_output, gmp3_out, gmp3_mask, h_stage3_dim, w_stage3_dim, STAGE3_CHANNELS);
        gmp3_out.mulScalar(last_forward_lambda_gmp, v3_scaled);
        v3.add(v3_scaled, v3_sum);
        v3 = v3_sum;
    }
    v3 = bn_v3->forward(v3);

    x = stage4_block1->forward(stage3_output);
    stage4_output = stage4_block2->forward(x);
    Tensor v4 = gap4->forward(stage4_output);
    if (last_forward_lambda_gmp > 0.0f)
    {
        ops::vision::globalMaxPool2d(stage4_output, gmp4_out, gmp4_mask, h_stage4_dim, w_stage4_dim, STAGE4_CHANNELS);
        gmp4_out.mulScalar(last_forward_lambda_gmp, v4_scaled);
        v4.add(v4_scaled, v4_sum);
        v4 = v4_sum;
    }
    v4 = bn_v4->forward(v4);

    v2.concatenateColumns(v3, v23_concat);
    v23_concat.concatenateColumns(v4, final_output);

    is_forward_completed = true;
    return final_output;
}

Tensor Multi_Scale_ResNet18_Balanced_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage("Multi_Scale_ResNet18_Balanced_Layer::backward: Forward must be called before backward",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::BACKWARD_PROPAGATION);
        throw std::runtime_error("Multi_Scale_ResNet18_Balanced_Layer::backward: Forward must be called before backward");
    }
    if (_output_gradient.getColumns() != OUTPUT_DIM)
    {
        throw std::invalid_argument("Multi_Scale_ResNet18_Balanced_Layer::backward: Output gradient column count mismatch");
    }

    _output_gradient.splitColumns(STAGE23_SPLIT_DIM, g23_split, g4_split);
    g23_split.splitColumns(STAGE2_CHANNELS, g2_split, g3_split);

    Tensor g4 = bn_v4->backward(g4_split);
    Tensor g3 = bn_v3->backward(g3_split);
    Tensor g2 = bn_v2->backward(g2_split);

    Tensor dO4_gap = gap4->backward(g4);
    if (last_forward_lambda_gmp > 0.0f)
    {
        ops::vision::globalMaxPool2dBackward(g4, gmp4_mask, gmp4_grad, h_stage4_dim, w_stage4_dim, STAGE4_CHANNELS);
        if (gmp4_grad.getShape() != stage4_output.getShape())
        {
            gmp4_grad.reshape(stage4_output.getShape());
        }
        gmp4_grad.mulScalar(last_forward_lambda_gmp, dO4_gmp_scaled);
        dO4_gap.add(dO4_gmp_scaled, stage4_gradient);
    }
    else
    {
        stage4_gradient = dO4_gap;
    }
    Tensor dO3_s4 = stage4_block2->backward(stage4_gradient);
    dO3_s4 = stage4_block1->backward(dO3_s4);

    Tensor dO3_gap = gap3->backward(g3);
    if (last_forward_lambda_gmp > 0.0f)
    {
        ops::vision::globalMaxPool2dBackward(g3, gmp3_mask, gmp3_grad, h_stage3_dim, w_stage3_dim, STAGE3_CHANNELS);
        if (gmp3_grad.getShape() != stage3_output.getShape())
        {
            gmp3_grad.reshape(stage3_output.getShape());
        }
        gmp3_grad.mulScalar(last_forward_lambda_gmp, dO3_gmp_scaled);
        dO3_s4.add(dO3_gap, dO3_sum_tmp);
        dO3_sum_tmp.add(dO3_gmp_scaled, stage3_gradient);
    }
    else
    {
        dO3_s4.add(dO3_gap, stage3_gradient);
    }
    Tensor dO2_s3 = stage3_block2->backward(stage3_gradient);
    dO2_s3 = stage3_block1->backward(dO2_s3);

    Tensor dO2_gap = gap2->backward(g2);
    if (last_forward_lambda_gmp > 0.0f)
    {
        ops::vision::globalMaxPool2dBackward(g2, gmp2_mask, gmp2_grad, h_stage2_dim, w_stage2_dim, STAGE2_CHANNELS);
        if (gmp2_grad.getShape() != stage2_output.getShape())
        {
            gmp2_grad.reshape(stage2_output.getShape());
        }
        gmp2_grad.mulScalar(last_forward_lambda_gmp, dO2_gmp_scaled);
        dO2_s3.add(dO2_gap, dO2_sum_tmp);
        dO2_sum_tmp.add(dO2_gmp_scaled, stage2_gradient);
    }
    else
    {
        dO2_s3.add(dO2_gap, stage2_gradient);
    }
    Tensor dO1 = stage2_block2->backward(stage2_gradient);
    dO1 = stage2_block1->backward(dO1);

    stage1_gradient = dO1;
    Tensor dO0 = stage1_block2->backward(dO1);
    dO0 = stage1_block1->backward(dO0);

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
    auto clone_as = [](const auto &src_ptr, auto &dst_ptr) {
        using LayerT = typename std::remove_reference_t<decltype(dst_ptr)>::element_type;
        if (src_ptr)
        {
            auto base_clone = src_ptr->clone();
            dst_ptr.reset(static_cast<LayerT *>(base_clone.release()));
        }
    };
    clone_as(stem_conv, cloned->stem_conv);
    clone_as(stem_bn, cloned->stem_bn);
    clone_as(stem_act, cloned->stem_act);
    clone_as(stem_downsample_conv, cloned->stem_downsample_conv);
    clone_as(stem_downsample_bn, cloned->stem_downsample_bn);
    clone_as(stem_downsample_act, cloned->stem_downsample_act);
    clone_as(stage1_block1, cloned->stage1_block1);
    clone_as(stage1_block2, cloned->stage1_block2);
    clone_as(stage2_block1, cloned->stage2_block1);
    clone_as(stage2_block2, cloned->stage2_block2);
    clone_as(gap2, cloned->gap2);
    clone_as(bn_v2, cloned->bn_v2);
    clone_as(stage3_block1, cloned->stage3_block1);
    clone_as(stage3_block2, cloned->stage3_block2);
    clone_as(gap3, cloned->gap3);
    clone_as(bn_v3, cloned->bn_v3);
    clone_as(stage4_block1, cloned->stage4_block1);
    clone_as(stage4_block2, cloned->stage4_block2);
    clone_as(gap4, cloned->gap4);
    clone_as(bn_v4, cloned->bn_v4);
    cloned->setLambdaGmp(lambda_gmp);
    cloned->setTrainingMode(is_training);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setAccumulated(is_accumulated);
    return cloned;
}

void Multi_Scale_ResNet18_Balanced_Layer::prepareInference()
{
    stem_conv->prepareInference();
    stem_bn->prepareInference();
    stem_act->prepareInference();
    stem_downsample_conv->prepareInference();
    stem_downsample_bn->prepareInference();
    stem_downsample_act->prepareInference();
    stage1_block1->prepareInference();
    stage1_block2->prepareInference();
    stage2_block1->prepareInference();
    stage2_block2->prepareInference();
    gap2->prepareInference();
    bn_v2->prepareInference();
    stage3_block1->prepareInference();
    stage3_block2->prepareInference();
    gap3->prepareInference();
    bn_v3->prepareInference();
    stage4_block1->prepareInference();
    stage4_block2->prepareInference();
    gap4->prepareInference();
    bn_v4->prepareInference();
}

void Multi_Scale_ResNet18_Balanced_Layer::quantizeToTernary()
{
    if (stem_conv) stem_conv->quantizeToTernary();
    if (stem_downsample_conv) stem_downsample_conv->quantizeToTernary();
    if (stage1_block1) stage1_block1->quantizeToTernary();
    if (stage1_block2) stage1_block2->quantizeToTernary();
    if (stage2_block1) stage2_block1->quantizeToTernary();
    if (stage2_block2) stage2_block2->quantizeToTernary();
    if (stage3_block1) stage3_block1->quantizeToTernary();
    if (stage3_block2) stage3_block2->quantizeToTernary();
    if (stage4_block1) stage4_block1->quantizeToTernary();
    if (stage4_block2) stage4_block2->quantizeToTernary();
}

void Multi_Scale_ResNet18_Balanced_Layer::dequantizeFromTernary()
{
    if (stem_conv) stem_conv->dequantizeFromTernary();
    if (stem_downsample_conv) stem_downsample_conv->dequantizeFromTernary();
    if (stage1_block1) stage1_block1->dequantizeFromTernary();
    if (stage1_block2) stage1_block2->dequantizeFromTernary();
    if (stage2_block1) stage2_block1->dequantizeFromTernary();
    if (stage2_block2) stage2_block2->dequantizeFromTernary();
    if (stage3_block1) stage3_block1->dequantizeFromTernary();
    if (stage3_block2) stage3_block2->dequantizeFromTernary();
    if (stage4_block1) stage4_block1->dequantizeFromTernary();
    if (stage4_block2) stage4_block2->dequantizeFromTernary();
}

void Multi_Scale_ResNet18_Balanced_Layer::snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const
{
    auto snap = [&](const ILayer *l) { if (l) l->snapshotNonTrainableState(_out); };
    snap(stem_bn.get());
    snap(stem_downsample_bn.get());
    snap(stage1_block1.get());
    snap(stage1_block2.get());
    snap(stage2_block1.get());
    snap(stage2_block2.get());
    snap(bn_v2.get());
    snap(stage3_block1.get());
    snap(stage3_block2.get());
    snap(bn_v3.get());
    snap(stage4_block1.get());
    snap(stage4_block2.get());
    snap(bn_v4.get());
}

void Multi_Scale_ResNet18_Balanced_Layer::restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor)
{
    auto rest = [&](ILayer *l) { if (l) l->restoreNonTrainableState(_in, _cursor); };
    rest(stem_bn.get());
    rest(stem_downsample_bn.get());
    rest(stage1_block1.get());
    rest(stage1_block2.get());
    rest(stage2_block1.get());
    rest(stage2_block2.get());
    rest(bn_v2.get());
    rest(stage3_block1.get());
    rest(stage3_block2.get());
    rest(bn_v3.get());
    rest(stage4_block1.get());
    rest(stage4_block2.get());
    rest(bn_v4.get());
}

bool Multi_Scale_ResNet18_Balanced_Layer::isTernary() const noexcept
{
    return (stem_conv && stem_conv->isTernary()) ||
           (stem_downsample_conv && stem_downsample_conv->isTernary()) ||
           (stage1_block1 && stage1_block1->isTernary()) ||
           (stage1_block2 && stage1_block2->isTernary()) ||
           (stage2_block1 && stage2_block1->isTernary()) ||
           (stage2_block2 && stage2_block2->isTernary()) ||
           (stage3_block1 && stage3_block1->isTernary()) ||
           (stage3_block2 && stage3_block2->isTernary()) ||
           (stage4_block1 && stage4_block1->isTernary()) ||
           (stage4_block2 && stage4_block2->isTernary());
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
    gmp2_out = Tensor(0, 0, execution_target);
    gmp2_mask = Tensor(0, 0, execution_target);
    gmp2_grad = Tensor(0, 0, execution_target);
    gmp3_out = Tensor(0, 0, execution_target);
    gmp3_mask = Tensor(0, 0, execution_target);
    gmp3_grad = Tensor(0, 0, execution_target);
    gmp4_out = Tensor(0, 0, execution_target);
    gmp4_mask = Tensor(0, 0, execution_target);
    gmp4_grad = Tensor(0, 0, execution_target);
    v2_scaled = Tensor(0, 0, execution_target);
    v2_sum = Tensor(0, 0, execution_target);
    v3_scaled = Tensor(0, 0, execution_target);
    v3_sum = Tensor(0, 0, execution_target);
    v4_scaled = Tensor(0, 0, execution_target);
    v4_sum = Tensor(0, 0, execution_target);
    v23_concat = Tensor(0, 0, execution_target);
    g23_split = Tensor(0, 0, execution_target);
    g4_split = Tensor(0, 0, execution_target);
    g2_split = Tensor(0, 0, execution_target);
    g3_split = Tensor(0, 0, execution_target);
    dO4_gmp_scaled = Tensor(0, 0, execution_target);
    dO3_gmp_scaled = Tensor(0, 0, execution_target);
    dO3_sum_tmp = Tensor(0, 0, execution_target);
    dO2_gmp_scaled = Tensor(0, 0, execution_target);
    dO2_sum_tmp = Tensor(0, 0, execution_target);
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
    bn_v2->resetGradient();

    stage3_block1->resetGradient();
    stage3_block2->resetGradient();
    gap3->resetGradient();
    bn_v3->resetGradient();

    stage4_block1->resetGradient();
    stage4_block2->resetGradient();
    gap4->resetGradient();
    bn_v4->resetGradient();
}

void Multi_Scale_ResNet18_Balanced_Layer::resetGradients()
{
    is_forward_completed = false;

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
    bn_v2->resetGradients();

    stage3_block1->resetGradients();
    stage3_block2->resetGradients();
    gap3->resetGradients();
    bn_v3->resetGradients();

    stage4_block1->resetGradients();
    stage4_block2->resetGradients();
    gap4->resetGradients();
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

bool Multi_Scale_ResNet18_Balanced_Layer::expandStemFrom1To2Channels()
{
    if (in_channels == 2)
    {
        return true;
    }
    if (in_channels != 1 || !stem_conv || stem_conv->isTernary())
    {
        return false;
    }
    if (stem_conv->getInputChannels() != 1 || stem_conv->getOutputChannels() != STAGE1_CHANNELS || stem_conv->getKernelSize() != 3)
    {
        return false;
    }

    std::vector<float> old_w = stem_conv->getWeights().getData();
    std::vector<float> old_b = stem_conv->getBiases().getData();
    constexpr size_t k_sq = 3 * 3;
    constexpr size_t oc_count = STAGE1_CHANNELS;
    if (old_w.size() != k_sq * 1 * oc_count || old_b.size() != oc_count)
    {
        return false;
    }

    std::vector<float> new_w(k_sq * 2 * oc_count, 0.0f);
    for (size_t k = 0; k < k_sq; ++k)
    {
        for (size_t oc = 0; oc < oc_count; ++oc)
        {
            new_w[(k * 2 + 0) * oc_count + oc] = old_w[k * oc_count + oc];
        }
    }

    auto new_stem = std::make_unique<Conv2d_Layer>(input_height, input_width, 2, STAGE1_CHANNELS, 3, 2, 1, execution_target);
    new_stem->setMixedPrecision(stem_conv->isMixedPrecisionEnabled());
    new_stem->setAccumulated(stem_conv->isAccumulated());
    new_stem->setWeights(Tensor(1, new_w.size(), std::move(new_w), execution_target));
    new_stem->setBiases(Tensor(1, old_b.size(), old_b, execution_target));

    stem_conv = std::move(new_stem);
    in_channels = 2;
    invalidateWeightCache();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    return true;
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
    if (execution_target == _new_execution_target)
    {
        return;
    }
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    input_tensor.setExecutionTarget(_new_execution_target);
    stage1_output.setExecutionTarget(_new_execution_target);
    stage2_output.setExecutionTarget(_new_execution_target);
    stage3_output.setExecutionTarget(_new_execution_target);
    stage4_output.setExecutionTarget(_new_execution_target);
    final_output.setExecutionTarget(_new_execution_target);
    gmp2_out = Tensor(0, 0, _new_execution_target);
    gmp2_mask = Tensor(0, 0, _new_execution_target);
    gmp2_grad = Tensor(0, 0, _new_execution_target);
    gmp3_out = Tensor(0, 0, _new_execution_target);
    gmp3_mask = Tensor(0, 0, _new_execution_target);
    gmp3_grad = Tensor(0, 0, _new_execution_target);
    gmp4_out = Tensor(0, 0, _new_execution_target);
    gmp4_mask = Tensor(0, 0, _new_execution_target);
    gmp4_grad = Tensor(0, 0, _new_execution_target);
    v2_scaled = Tensor(0, 0, _new_execution_target);
    v2_sum = Tensor(0, 0, _new_execution_target);
    v3_scaled = Tensor(0, 0, _new_execution_target);
    v3_sum = Tensor(0, 0, _new_execution_target);
    v4_scaled = Tensor(0, 0, _new_execution_target);
    v4_sum = Tensor(0, 0, _new_execution_target);
    v23_concat = Tensor(0, 0, _new_execution_target);
    g23_split = Tensor(0, 0, _new_execution_target);
    g4_split = Tensor(0, 0, _new_execution_target);
    g2_split = Tensor(0, 0, _new_execution_target);
    g3_split = Tensor(0, 0, _new_execution_target);
    dO4_gmp_scaled = Tensor(0, 0, _new_execution_target);
    dO3_gmp_scaled = Tensor(0, 0, _new_execution_target);
    dO3_sum_tmp = Tensor(0, 0, _new_execution_target);
    dO2_gmp_scaled = Tensor(0, 0, _new_execution_target);
    dO2_sum_tmp = Tensor(0, 0, _new_execution_target);
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
    bn_v2->setExecutionTarget(_new_execution_target);

    stage3_block1->setExecutionTarget(_new_execution_target);
    stage3_block2->setExecutionTarget(_new_execution_target);
    gap3->setExecutionTarget(_new_execution_target);
    bn_v3->setExecutionTarget(_new_execution_target);

    stage4_block1->setExecutionTarget(_new_execution_target);
    stage4_block2->setExecutionTarget(_new_execution_target);
    gap4->setExecutionTarget(_new_execution_target);
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
    bn_v2->setMixedPrecision(_enable);

    stage3_block1->setMixedPrecision(_enable);
    stage3_block2->setMixedPrecision(_enable);
    gap3->setMixedPrecision(_enable);
    bn_v3->setMixedPrecision(_enable);

    stage4_block1->setMixedPrecision(_enable);
    stage4_block2->setMixedPrecision(_enable);
    gap4->setMixedPrecision(_enable);
    bn_v4->setMixedPrecision(_enable);
}

void Multi_Scale_ResNet18_Balanced_Layer::setTrainingMode(bool _is_training)
{
    is_training = _is_training;
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
    bn_v2->setTrainingMode(_is_training);

    stage3_block1->setTrainingMode(_is_training);
    stage3_block2->setTrainingMode(_is_training);
    gap3->setTrainingMode(_is_training);
    bn_v3->setTrainingMode(_is_training);

    stage4_block1->setTrainingMode(_is_training);
    stage4_block2->setTrainingMode(_is_training);
    gap4->setTrainingMode(_is_training);
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
    bn_v2->setAccumulated(_is_accumulated);

    stage3_block1->setAccumulated(_is_accumulated);
    stage3_block2->setAccumulated(_is_accumulated);
    gap3->setAccumulated(_is_accumulated);
    bn_v3->setAccumulated(_is_accumulated);

    stage4_block1->setAccumulated(_is_accumulated);
    stage4_block2->setAccumulated(_is_accumulated);
    gap4->setAccumulated(_is_accumulated);
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

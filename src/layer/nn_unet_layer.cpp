#include "layer/nn_unet_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"

namespace
{
static Tensor addTensors(const Tensor &a, const Tensor &b)
{
    Tensor b_cast = b;
    if (b_cast.getDataType() != a.getDataType())
    {
        b_cast = b_cast.to(a.getDataType());
    }
    Tensor out(a.getShape(), a.getDataType(), a.getExecutionTarget());
    a.add(b_cast, out);
    return out;
}
} // namespace

NN_UNet_Layer::NN_UNet_Layer(
    uint32_t _height,
    uint32_t _width,
    uint32_t _in_channels,
    uint32_t _out_channels,
    uint32_t _base_channels,
    Execution_Target _execution_target)
    : input_height(_height),
      input_width(_width),
      in_channels(_in_channels),
      out_channels(_out_channels),
      base_channels(_base_channels),
      execution_target(_execution_target),
      is_forward_completed(false),
      is_training(true),
      is_accumulated(false),
      enable_deep_supervision(true),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target)
{
    buildNetwork();
    resetHeadBias(-1.5f);
}

void NN_UNet_Layer::buildNetwork()
{
    enc1_layers.clear();
    enc2_layers.clear();
    enc3_layers.clear();
    enc4_layers.clear();
    bottleneck_layers.clear();
    dec4_layers.clear();
    dec3_layers.clear();
    dec2_layers.clear();
    dec1_layers.clear();

    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;
    uint32_t c5 = base_channels * 16;

    uint32_t h = input_height;
    uint32_t w = input_width;

    // Stage 1 (160x160)
    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, in_channels, c1, 3, 1, 1, execution_target));
    enc1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    enc1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target));
    enc1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    enc1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool1 = std::make_unique<Max_Pool_2d_Layer>(h, w, c1, 2, 2, 0, execution_target);

    // Stage 2 (80x80)
    h /= 2; w /= 2;
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c2, 3, 1, 1, execution_target));
    enc2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    enc2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target));
    enc2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    enc2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool2 = std::make_unique<Max_Pool_2d_Layer>(h, w, c2, 2, 2, 0, execution_target);

    // Stage 3 (40x40)
    h /= 2; w /= 2;
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c3, 3, 1, 1, execution_target));
    enc3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    enc3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target));
    enc3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    enc3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool3 = std::make_unique<Max_Pool_2d_Layer>(h, w, c3, 2, 2, 0, execution_target);

    // Stage 4 (20x20)
    h /= 2; w /= 2;
    enc4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c4, 3, 1, 1, execution_target));
    enc4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    enc4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c4, 3, 1, 1, execution_target));
    enc4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    enc4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool4 = std::make_unique<Max_Pool_2d_Layer>(h, w, c4, 2, 2, 0, execution_target);

    // Bottleneck (10x10)
    h /= 2; w /= 2;
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c5, 3, 1, 1, execution_target));
    bottleneck_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c5, 1e-4f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c5, c5, 3, 1, 1, execution_target));
    bottleneck_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c5, 1e-4f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    // Decoder 4 (10x10 -> 20x20)
    up4 = std::make_unique<Up_Sample_2d_Layer>(h, w, c5, 2, execution_target);
    h *= 2; w *= 2;
    dec4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c5 + c4, c4, 3, 1, 1, execution_target));
    dec4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    dec4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c4, 3, 1, 1, execution_target));
    dec4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    dec4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    // Decoder 3 (20x20 -> 40x40)
    up3 = std::make_unique<Up_Sample_2d_Layer>(h, w, c4, 2, execution_target);
    h *= 2; w *= 2;
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4 + c3, c3, 3, 1, 1, execution_target));
    dec3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    dec3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target));
    dec3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    dec3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    // Deep Supervision Head at Stage 3 (40x40)
    ds3_conv = std::make_unique<Conv2d_Layer>(h, w, c3, out_channels, 1, 1, 0, execution_target);
    ds3_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);

    // Decoder 2 (40x40 -> 80x80)
    up2 = std::make_unique<Up_Sample_2d_Layer>(h, w, c3, 2, execution_target);
    h *= 2; w *= 2;
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3 + c2, c2, 3, 1, 1, execution_target));
    dec2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    dec2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target));
    dec2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    dec2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    // Deep Supervision Head at Stage 2 (80x80)
    ds2_conv = std::make_unique<Conv2d_Layer>(h, w, c2, out_channels, 1, 1, 0, execution_target);
    ds2_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);

    // Decoder 1 (80x80 -> 160x160)
    up1 = std::make_unique<Up_Sample_2d_Layer>(h, w, c2, 2, execution_target);
    h *= 2; w *= 2;
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2 + c1, c1, 3, 1, 1, execution_target));
    dec1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    dec1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target));
    dec1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    dec1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    // Final Output Head (160x160)
    final_conv = std::make_unique<Conv2d_Layer>(h, w, c1, out_channels, 1, 1, 0, execution_target);
    final_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);
}

std::unique_ptr<ILayer> NN_UNet_Layer::clone() const
{
    auto cloned = std::make_unique<NN_UNet_Layer>(input_height, input_width, in_channels, out_channels, base_channels, execution_target);
    auto this_ptr = const_cast<NN_UNet_Layer*>(this);
    auto src_params = this_ptr->getParametersAndGradients();
    auto dst_params = cloned->getParametersAndGradients();
    if (src_params.size() == dst_params.size())
    {
        for (size_t i = 0; i < src_params.size(); ++i)
        {
            if (src_params[i].first && dst_params[i].first)
            {
                dst_params[i].first->uploadData(src_params[i].first->getData());
            }
        }
    }
    cloned->invalidateWeightCache();
    cloned->setTrainingMode(is_training);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setEnableDeepSupervision(enable_deep_supervision);
    return cloned;
}

Tensor NN_UNet_Layer::forward(const Tensor &_input_tensor)
{
    input_tensor = _input_tensor;
    Tensor cur = _input_tensor;

    auto forward_res_block = [](std::vector<std::unique_ptr<ILayer>> &layers, const Tensor &in, Tensor &saved_xa) -> Tensor {
        Tensor c = in;
        c = layers[0]->forward(c);
        c = layers[1]->forward(c);
        c = layers[2]->forward(c);
        saved_xa = c;
        c = layers[3]->forward(c);
        c = layers[4]->forward(c);
        c = layers[5]->forward(c);
        return addTensors(c, saved_xa);
    };

    // Stage 1
    cur = forward_res_block(enc1_layers, cur, res_enc1_act);
    skip1_act = cur;
    cur = pool1->forward(cur);

    // Stage 2
    cur = forward_res_block(enc2_layers, cur, res_enc2_act);
    skip2_act = cur;
    cur = pool2->forward(cur);

    // Stage 3
    cur = forward_res_block(enc3_layers, cur, res_enc3_act);
    skip3_act = cur;
    cur = pool3->forward(cur);

    // Stage 4
    cur = forward_res_block(enc4_layers, cur, res_enc4_act);
    skip4_act = cur;
    cur = pool4->forward(cur);

    // Bottleneck
    cur = forward_res_block(bottleneck_layers, cur, res_bn_act);

    uint32_t b = static_cast<uint32_t>(_input_tensor.getRows());
    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;
    uint32_t c5 = base_channels * 16;

    uint32_t h4 = input_height / 8;
    uint32_t w4 = input_width / 8;
    uint32_t h3 = input_height / 4;
    uint32_t w3 = input_width / 4;
    uint32_t h2 = input_height / 2;
    uint32_t w2 = input_width / 2;
    uint32_t h1 = input_height;
    uint32_t w1 = input_width;

    // Decoder 4 (concat with skip4)
    Tensor up_feat4 = up4->forward(cur);
    if (up_feat4.getDataType() != skip4_act.getDataType())
    {
        up_feat4 = up_feat4.to(skip4_act.getDataType());
    }
    up_feat4.reshape(Shape{b * h4 * w4, c5});
    skip4_act.reshape(Shape{b * h4 * w4, c4});
    dec4_concat = up_feat4.concatenateColumns(skip4_act);
    dec4_concat.reshape(Shape{b, h4 * w4 * (c5 + c4)});
    skip4_act.reshape(Shape{b, h4 * w4 * c4});
    cur = dec4_concat;
    cur = forward_res_block(dec4_layers, cur, res_dec4_act);

    // Decoder 3 (concat with skip3)
    Tensor up_feat3 = up3->forward(cur);
    if (up_feat3.getDataType() != skip3_act.getDataType())
    {
        up_feat3 = up_feat3.to(skip3_act.getDataType());
    }
    up_feat3.reshape(Shape{b * h3 * w3, c4});
    skip3_act.reshape(Shape{b * h3 * w3, c3});
    dec3_concat = up_feat3.concatenateColumns(skip3_act);
    dec3_concat.reshape(Shape{b, h3 * w3 * (c4 + c3)});
    skip3_act.reshape(Shape{b, h3 * w3 * c3});
    cur = dec3_concat;
    cur = forward_res_block(dec3_layers, cur, res_dec3_act);

    // Deep Supervision 3 (40x40)
    if (enable_deep_supervision && is_training)
    {
        Tensor ds3_before = ds3_conv->forward(cur);
        ds3_output = ds3_sigmoid->forward(ds3_before);
        if (_input_tensor.getShape().getRank() == 4)
        {
            ds3_output.reshape(Shape{b, out_channels, h3, w3});
        }
    }
    else
    {
        ds3_output = Tensor(0, 0, execution_target);
    }

    // Decoder 2 (concat with skip2)
    Tensor up_feat2 = up2->forward(cur);
    if (up_feat2.getDataType() != skip2_act.getDataType())
    {
        up_feat2 = up_feat2.to(skip2_act.getDataType());
    }
    up_feat2.reshape(Shape{b * h2 * w2, c3});
    skip2_act.reshape(Shape{b * h2 * w2, c2});
    dec2_concat = up_feat2.concatenateColumns(skip2_act);
    dec2_concat.reshape(Shape{b, h2 * w2 * (c3 + c2)});
    skip2_act.reshape(Shape{b, h2 * w2 * c2});
    cur = dec2_concat;
    cur = forward_res_block(dec2_layers, cur, res_dec2_act);

    // Deep Supervision 2 (80x80)
    if (enable_deep_supervision && is_training)
    {
        Tensor ds2_before = ds2_conv->forward(cur);
        ds2_output = ds2_sigmoid->forward(ds2_before);
        if (_input_tensor.getShape().getRank() == 4)
        {
            ds2_output.reshape(Shape{b, out_channels, h2, w2});
        }
    }
    else
    {
        ds2_output = Tensor(0, 0, execution_target);
    }

    // Decoder 1 (concat with skip1)
    Tensor up_feat1 = up1->forward(cur);
    if (up_feat1.getDataType() != skip1_act.getDataType())
    {
        up_feat1 = up_feat1.to(skip1_act.getDataType());
    }
    up_feat1.reshape(Shape{b * h1 * w1, c2});
    skip1_act.reshape(Shape{b * h1 * w1, c1});
    dec1_concat = up_feat1.concatenateColumns(skip1_act);
    dec1_concat.reshape(Shape{b, h1 * w1 * (c2 + c1)});
    skip1_act.reshape(Shape{b, h1 * w1 * c1});
    cur = dec1_concat;
    cur = forward_res_block(dec1_layers, cur, res_dec1_act);

    // Final Sigmoid Head (160x160)
    Tensor before_sigmoid = final_conv->forward(cur);
    output_tensor = final_sigmoid->forward(before_sigmoid);

    if (_input_tensor.getShape().getRank() == 4)
    {
        output_tensor.reshape(Shape{_input_tensor.getShape()[0], out_channels, input_height, input_width});
    }

    is_forward_completed = true;
    return output_tensor;
}

Tensor NN_UNet_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::runtime_error("NN_UNet_Layer::backward: Forward must be called before backward");
    }

    Tensor grad = final_sigmoid->backward(_output_gradient);
    if (grad.getShape().getRank() != 2)
    {
        grad.reshape(Shape{grad.getShape()[0], grad.getTotalElements() / grad.getShape()[0]});
    }

    uint32_t b = static_cast<uint32_t>(grad.getRows());
    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;
    uint32_t c5 = base_channels * 16;

    uint32_t h1 = input_height;
    uint32_t w1 = input_width;
    uint32_t h2 = input_height / 2;
    uint32_t w2 = input_width / 2;
    uint32_t h3 = input_height / 4;
    uint32_t w3 = input_width / 4;
    uint32_t h4 = input_height / 8;
    uint32_t w4 = input_width / 8;

    auto backward_res_block = [](std::vector<std::unique_ptr<ILayer>> &layers, const Tensor &grad_sum) -> Tensor {
        Tensor cur_grad = grad_sum;
        cur_grad = layers[5]->backward(cur_grad);
        cur_grad = layers[4]->backward(cur_grad);
        cur_grad = layers[3]->backward(cur_grad);
        Tensor grad_xa = addTensors(cur_grad, grad_sum);
        grad_xa = layers[2]->backward(grad_xa);
        grad_xa = layers[1]->backward(grad_xa);
        return layers[0]->backward(grad_xa);
    };

    // Backward Final Conv
    grad = final_conv->backward(grad);

    // Backward Decoder 1
    grad = backward_res_block(dec1_layers, grad);
    grad.reshape(Shape{b * h1 * w1, c2 + c1});
    auto [d_up1, d_skip1] = grad.splitColumns(c2);
    d_up1.reshape(Shape{b, h1 * w1 * c2});
    d_skip1.reshape(Shape{b, h1 * w1 * c1});
    grad = up1->backward(d_up1);

    // Deep Supervision 2 gradient accumulation
    if (enable_deep_supervision && !ds2_grad.isEmpty())
    {
        Tensor ds2_g = ds2_grad;
        if (ds2_g.getShape().getRank() != 2)
        {
            ds2_g.reshape(Shape{ds2_g.getShape()[0], ds2_g.getTotalElements() / ds2_g.getShape()[0]});
        }
        Tensor d_ds2 = ds2_sigmoid->backward(ds2_g);
        d_ds2 = ds2_conv->backward(d_ds2);
        grad = addTensors(grad, d_ds2);
    }

    // Backward Decoder 2
    grad = backward_res_block(dec2_layers, grad);
    grad.reshape(Shape{b * h2 * w2, c3 + c2});
    auto [d_up2, d_skip2] = grad.splitColumns(c3);
    d_up2.reshape(Shape{b, h2 * w2 * c3});
    d_skip2.reshape(Shape{b, h2 * w2 * c2});
    grad = up2->backward(d_up2);

    // Deep Supervision 3 gradient accumulation
    if (enable_deep_supervision && !ds3_grad.isEmpty())
    {
        Tensor ds3_g = ds3_grad;
        if (ds3_g.getShape().getRank() != 2)
        {
            ds3_g.reshape(Shape{ds3_g.getShape()[0], ds3_g.getTotalElements() / ds3_g.getShape()[0]});
        }
        Tensor d_ds3 = ds3_sigmoid->backward(ds3_g);
        d_ds3 = ds3_conv->backward(d_ds3);
        grad = addTensors(grad, d_ds3);
    }

    // Backward Decoder 3
    grad = backward_res_block(dec3_layers, grad);
    grad.reshape(Shape{b * h3 * w3, c4 + c3});
    auto [d_up3, d_skip3] = grad.splitColumns(c4);
    d_up3.reshape(Shape{b, h3 * w3 * c4});
    d_skip3.reshape(Shape{b, h3 * w3 * c3});
    grad = up3->backward(d_up3);

    // Backward Decoder 4
    grad = backward_res_block(dec4_layers, grad);
    grad.reshape(Shape{b * h4 * w4, c5 + c4});
    auto [d_up4, d_skip4] = grad.splitColumns(c5);
    d_up4.reshape(Shape{b, h4 * w4 * c5});
    d_skip4.reshape(Shape{b, h4 * w4 * c4});
    grad = up4->backward(d_up4);

    // Backward Bottleneck
    grad = backward_res_block(bottleneck_layers, grad);

    // Backward Encoder 4
    grad = pool4->backward(grad);
    grad = addTensors(grad, d_skip4);
    grad = backward_res_block(enc4_layers, grad);

    // Backward Encoder 3
    grad = pool3->backward(grad);
    grad = addTensors(grad, d_skip3);
    grad = backward_res_block(enc3_layers, grad);

    // Backward Encoder 2
    grad = pool2->backward(grad);
    grad = addTensors(grad, d_skip2);
    grad = backward_res_block(enc2_layers, grad);

    // Backward Encoder 1
    grad = pool1->backward(grad);
    grad = addTensors(grad, d_skip1);
    grad = backward_res_block(enc1_layers, grad);

    if (input_tensor.getShape().getRank() == 4)
    {
        grad.reshape(input_tensor.getShape());
    }

    input_gradient_tensor = grad;
    return input_gradient_tensor;
}

void NN_UNet_Layer::resetGradient()
{
    auto reset_vec = [](auto &vec) {
        for (auto &l : vec) l->resetGradient();
    };
    reset_vec(enc1_layers);
    if (pool1) pool1->resetGradient();
    reset_vec(enc2_layers);
    if (pool2) pool2->resetGradient();
    reset_vec(enc3_layers);
    if (pool3) pool3->resetGradient();
    reset_vec(enc4_layers);
    if (pool4) pool4->resetGradient();
    reset_vec(bottleneck_layers);
    if (up4) up4->resetGradient();
    reset_vec(dec4_layers);
    if (up3) up3->resetGradient();
    reset_vec(dec3_layers);
    if (up2) up2->resetGradient();
    reset_vec(dec2_layers);
    if (up1) up1->resetGradient();
    reset_vec(dec1_layers);
    if (final_conv) final_conv->resetGradient();
    if (final_sigmoid) final_sigmoid->resetGradient();
    if (ds3_conv) ds3_conv->resetGradient();
    if (ds3_sigmoid) ds3_sigmoid->resetGradient();
    if (ds2_conv) ds2_conv->resetGradient();
    if (ds2_sigmoid) ds2_sigmoid->resetGradient();

    skip1_act = Tensor(0, 0, execution_target);
    skip2_act = Tensor(0, 0, execution_target);
    skip3_act = Tensor(0, 0, execution_target);
    skip4_act = Tensor(0, 0, execution_target);
    dec4_concat = Tensor(0, 0, execution_target);
    dec3_concat = Tensor(0, 0, execution_target);
    dec2_concat = Tensor(0, 0, execution_target);
    dec1_concat = Tensor(0, 0, execution_target);

    res_enc1_act = Tensor(0, 0, execution_target);
    res_enc2_act = Tensor(0, 0, execution_target);
    res_enc3_act = Tensor(0, 0, execution_target);
    res_enc4_act = Tensor(0, 0, execution_target);
    res_bn_act = Tensor(0, 0, execution_target);
    res_dec4_act = Tensor(0, 0, execution_target);
    res_dec3_act = Tensor(0, 0, execution_target);
    res_dec2_act = Tensor(0, 0, execution_target);
    res_dec1_act = Tensor(0, 0, execution_target);

    ds3_output = Tensor(0, 0, execution_target);
    ds2_output = Tensor(0, 0, execution_target);
    ds3_grad = Tensor(0, 0, execution_target);
    ds2_grad = Tensor(0, 0, execution_target);

    input_tensor = Tensor(0, 0, execution_target);
    output_tensor = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);
    is_forward_completed = false;
}

void NN_UNet_Layer::resetGradients()
{
    resetGradient();
    auto reset_vec = [](auto &vec) {
        for (auto &l : vec) l->resetGradients();
    };
    reset_vec(enc1_layers);
    if (pool1) pool1->resetGradients();
    reset_vec(enc2_layers);
    if (pool2) pool2->resetGradients();
    reset_vec(enc3_layers);
    if (pool3) pool3->resetGradients();
    reset_vec(enc4_layers);
    if (pool4) pool4->resetGradients();
    reset_vec(bottleneck_layers);
    if (up4) up4->resetGradients();
    reset_vec(dec4_layers);
    if (up3) up3->resetGradients();
    reset_vec(dec3_layers);
    if (up2) up2->resetGradients();
    reset_vec(dec2_layers);
    if (up1) up1->resetGradients();
    reset_vec(dec1_layers);
    if (final_conv) final_conv->resetGradients();
    if (final_sigmoid) final_sigmoid->resetGradients();
    if (ds3_conv) ds3_conv->resetGradients();
    if (ds3_sigmoid) ds3_sigmoid->resetGradients();
    if (ds2_conv) ds2_conv->resetGradients();
    if (ds2_sigmoid) ds2_sigmoid->resetGradients();
}

std::vector<std::pair<Tensor *, Tensor *>> NN_UNet_Layer::getParametersAndGradients()
{
    std::vector<std::pair<Tensor *, Tensor *>> total_params;
    auto append_params = [&](auto &layer_ptr) {
        if (layer_ptr)
        {
            auto p = layer_ptr->getParametersAndGradients();
            total_params.insert(total_params.end(), p.begin(), p.end());
        }
    };
    auto append_vec = [&](auto &vec) {
        for (auto &l : vec) append_params(l);
    };

    append_vec(enc1_layers);
    append_params(pool1);
    append_vec(enc2_layers);
    append_params(pool2);
    append_vec(enc3_layers);
    append_params(pool3);
    append_vec(enc4_layers);
    append_params(pool4);
    append_vec(bottleneck_layers);
    append_params(up4);
    append_vec(dec4_layers);
    append_params(up3);
    append_vec(dec3_layers);
    append_params(ds3_conv);
    append_params(ds3_sigmoid);
    append_params(up2);
    append_vec(dec2_layers);
    append_params(ds2_conv);
    append_params(ds2_sigmoid);
    append_params(up1);
    append_vec(dec1_layers);
    append_params(final_conv);
    append_params(final_sigmoid);

    return total_params;
}

void NN_UNet_Layer::saveConfiguration(std::ofstream &_out) const
{
    _out.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _out.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _out.write(reinterpret_cast<const char *>(&in_channels), sizeof(in_channels));
    _out.write(reinterpret_cast<const char *>(&out_channels), sizeof(out_channels));
    _out.write(reinterpret_cast<const char *>(&base_channels), sizeof(base_channels));
}

void NN_UNet_Layer::saveInference(std::ofstream &_out) const
{
    saveConfiguration(_out);
    auto save_vec = [&](const auto &vec) {
        for (const auto &l : vec) l->saveInference(_out);
    };
    save_vec(enc1_layers);
    if (pool1) pool1->saveInference(_out);
    save_vec(enc2_layers);
    if (pool2) pool2->saveInference(_out);
    save_vec(enc3_layers);
    if (pool3) pool3->saveInference(_out);
    save_vec(enc4_layers);
    if (pool4) pool4->saveInference(_out);
    save_vec(bottleneck_layers);
    if (up4) up4->saveInference(_out);
    save_vec(dec4_layers);
    if (up3) up3->saveInference(_out);
    save_vec(dec3_layers);
    if (ds3_conv) ds3_conv->saveInference(_out);
    if (ds3_sigmoid) ds3_sigmoid->saveInference(_out);
    if (up2) up2->saveInference(_out);
    save_vec(dec2_layers);
    if (ds2_conv) ds2_conv->saveInference(_out);
    if (ds2_sigmoid) ds2_sigmoid->saveInference(_out);
    if (up1) up1->saveInference(_out);
    save_vec(dec1_layers);
    if (final_conv) final_conv->saveInference(_out);
    if (final_sigmoid) final_sigmoid->saveInference(_out);
}

void NN_UNet_Layer::loadInference(std::ifstream &_in)
{
    _in.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
    _in.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
    _in.read(reinterpret_cast<char *>(&in_channels), sizeof(in_channels));
    _in.read(reinterpret_cast<char *>(&out_channels), sizeof(out_channels));
    _in.read(reinterpret_cast<char *>(&base_channels), sizeof(base_channels));

    auto load_vec = [&](auto &vec) {
        for (auto &l : vec) l->loadInference(_in);
    };
    load_vec(enc1_layers);
    if (pool1) pool1->loadInference(_in);
    load_vec(enc2_layers);
    if (pool2) pool2->loadInference(_in);
    load_vec(enc3_layers);
    if (pool3) pool3->loadInference(_in);
    load_vec(enc4_layers);
    if (pool4) pool4->loadInference(_in);
    load_vec(bottleneck_layers);
    if (up4) up4->loadInference(_in);
    load_vec(dec4_layers);
    if (up3) up3->loadInference(_in);
    load_vec(dec3_layers);
    if (ds3_conv) ds3_conv->loadInference(_in);
    if (ds3_sigmoid) ds3_sigmoid->loadInference(_in);
    if (up2) up2->loadInference(_in);
    load_vec(dec2_layers);
    if (ds2_conv) ds2_conv->loadInference(_in);
    if (ds2_sigmoid) ds2_sigmoid->loadInference(_in);
    if (up1) up1->loadInference(_in);
    load_vec(dec1_layers);
    if (final_conv) final_conv->loadInference(_in);
    if (final_sigmoid) final_sigmoid->loadInference(_in);
}

void NN_UNet_Layer::saveCheckpoint(std::ofstream &_out) const
{
    saveConfiguration(_out);
    auto save_vec = [&](const auto &vec) {
        for (const auto &l : vec) l->saveCheckpoint(_out);
    };
    save_vec(enc1_layers);
    if (pool1) pool1->saveCheckpoint(_out);
    save_vec(enc2_layers);
    if (pool2) pool2->saveCheckpoint(_out);
    save_vec(enc3_layers);
    if (pool3) pool3->saveCheckpoint(_out);
    save_vec(enc4_layers);
    if (pool4) pool4->saveCheckpoint(_out);
    save_vec(bottleneck_layers);
    if (up4) up4->saveCheckpoint(_out);
    save_vec(dec4_layers);
    if (up3) up3->saveCheckpoint(_out);
    save_vec(dec3_layers);
    if (ds3_conv) ds3_conv->saveCheckpoint(_out);
    if (ds3_sigmoid) ds3_sigmoid->saveCheckpoint(_out);
    if (up2) up2->saveCheckpoint(_out);
    save_vec(dec2_layers);
    if (ds2_conv) ds2_conv->saveCheckpoint(_out);
    if (ds2_sigmoid) ds2_sigmoid->saveCheckpoint(_out);
    if (up1) up1->saveCheckpoint(_out);
    save_vec(dec1_layers);
    if (final_conv) final_conv->saveCheckpoint(_out);
    if (final_sigmoid) final_sigmoid->saveCheckpoint(_out);
}

void NN_UNet_Layer::loadCheckpoint(std::ifstream &_in)
{
    _in.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
    _in.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
    _in.read(reinterpret_cast<char *>(&in_channels), sizeof(in_channels));
    _in.read(reinterpret_cast<char *>(&out_channels), sizeof(out_channels));
    _in.read(reinterpret_cast<char *>(&base_channels), sizeof(base_channels));

    auto load_vec = [&](auto &vec) {
        for (auto &l : vec) l->loadCheckpoint(_in);
    };
    load_vec(enc1_layers);
    if (pool1) pool1->loadCheckpoint(_in);
    load_vec(enc2_layers);
    if (pool2) pool2->loadCheckpoint(_in);
    load_vec(enc3_layers);
    if (pool3) pool3->loadCheckpoint(_in);
    load_vec(enc4_layers);
    if (pool4) pool4->loadCheckpoint(_in);
    load_vec(bottleneck_layers);
    if (up4) up4->loadCheckpoint(_in);
    load_vec(dec4_layers);
    if (up3) up3->loadCheckpoint(_in);
    load_vec(dec3_layers);
    if (ds3_conv) ds3_conv->loadCheckpoint(_in);
    if (ds3_sigmoid) ds3_sigmoid->loadCheckpoint(_in);
    if (up2) up2->loadCheckpoint(_in);
    load_vec(dec2_layers);
    if (ds2_conv) ds2_conv->loadCheckpoint(_in);
    if (ds2_sigmoid) ds2_sigmoid->loadCheckpoint(_in);
    if (up1) up1->loadCheckpoint(_in);
    load_vec(dec1_layers);
    if (final_conv) final_conv->loadCheckpoint(_in);
    if (final_sigmoid) final_sigmoid->loadCheckpoint(_in);
}

void NN_UNet_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    auto set_t = [&](auto &l) { if (l) l->setExecutionTarget(_new_execution_target); };
    auto set_vec = [&](auto &v) { for (auto &l : v) set_t(l); };

    set_vec(enc1_layers);
    set_t(pool1);
    set_vec(enc2_layers);
    set_t(pool2);
    set_vec(enc3_layers);
    set_t(pool3);
    set_vec(enc4_layers);
    set_t(pool4);
    set_vec(bottleneck_layers);
    set_t(up4);
    set_vec(dec4_layers);
    set_t(up3);
    set_vec(dec3_layers);
    set_t(ds3_conv);
    set_t(ds3_sigmoid);
    set_t(up2);
    set_vec(dec2_layers);
    set_t(ds2_conv);
    set_t(ds2_sigmoid);
    set_t(up1);
    set_vec(dec1_layers);
    set_t(final_conv);
    set_t(final_sigmoid);

    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

void NN_UNet_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    is_accumulated = _is_accumulated;
    auto set_acc = [&](auto &v) { for (auto &l : v) l->setAccumulated(_is_accumulated); };
    set_acc(enc1_layers);
    if (pool1) pool1->setAccumulated(_is_accumulated);
    set_acc(enc2_layers);
    if (pool2) pool2->setAccumulated(_is_accumulated);
    set_acc(enc3_layers);
    if (pool3) pool3->setAccumulated(_is_accumulated);
    set_acc(enc4_layers);
    if (pool4) pool4->setAccumulated(_is_accumulated);
    set_acc(bottleneck_layers);
    if (up4) up4->setAccumulated(_is_accumulated);
    set_acc(dec4_layers);
    if (up3) up3->setAccumulated(_is_accumulated);
    set_acc(dec3_layers);
    if (ds3_conv) ds3_conv->setAccumulated(_is_accumulated);
    if (ds3_sigmoid) ds3_sigmoid->setAccumulated(_is_accumulated);
    if (up2) up2->setAccumulated(_is_accumulated);
    set_acc(dec2_layers);
    if (ds2_conv) ds2_conv->setAccumulated(_is_accumulated);
    if (ds2_sigmoid) ds2_sigmoid->setAccumulated(_is_accumulated);
    if (up1) up1->setAccumulated(_is_accumulated);
    set_acc(dec1_layers);
    if (final_conv) final_conv->setAccumulated(_is_accumulated);
    if (final_sigmoid) final_sigmoid->setAccumulated(_is_accumulated);
}

void NN_UNet_Layer::setTrainingMode(bool _training)
{
    is_training = _training;
    auto set_tm = [&](auto &v) { for (auto &l : v) l->setTrainingMode(_training); };
    set_tm(enc1_layers);
    if (pool1) pool1->setTrainingMode(_training);
    set_tm(enc2_layers);
    if (pool2) pool2->setTrainingMode(_training);
    set_tm(enc3_layers);
    if (pool3) pool3->setTrainingMode(_training);
    set_tm(enc4_layers);
    if (pool4) pool4->setTrainingMode(_training);
    set_tm(bottleneck_layers);
    if (up4) up4->setTrainingMode(_training);
    set_tm(dec4_layers);
    if (up3) up3->setTrainingMode(_training);
    set_tm(dec3_layers);
    if (ds3_conv) ds3_conv->setTrainingMode(_training);
    if (ds3_sigmoid) ds3_sigmoid->setTrainingMode(_training);
    if (up2) up2->setTrainingMode(_training);
    set_tm(dec2_layers);
    if (ds2_conv) ds2_conv->setTrainingMode(_training);
    if (ds2_sigmoid) ds2_sigmoid->setTrainingMode(_training);
    if (up1) up1->setTrainingMode(_training);
    set_tm(dec1_layers);
    if (final_conv) final_conv->setTrainingMode(_training);
    if (final_sigmoid) final_sigmoid->setTrainingMode(_training);
}

void NN_UNet_Layer::setMixedPrecision(bool _enable) noexcept
{
    auto set_mp = [&](auto &v) { for (auto &l : v) l->setMixedPrecision(_enable); };
    set_mp(enc1_layers);
    if (pool1) pool1->setMixedPrecision(_enable);
    set_mp(enc2_layers);
    if (pool2) pool2->setMixedPrecision(_enable);
    set_mp(enc3_layers);
    if (pool3) pool3->setMixedPrecision(_enable);
    set_mp(enc4_layers);
    if (pool4) pool4->setMixedPrecision(_enable);
    set_mp(bottleneck_layers);
    if (up4) up4->setMixedPrecision(_enable);
    set_mp(dec4_layers);
    if (up3) up3->setMixedPrecision(_enable);
    set_mp(dec3_layers);
    if (ds3_conv) ds3_conv->setMixedPrecision(_enable);
    if (ds3_sigmoid) ds3_sigmoid->setMixedPrecision(_enable);
    if (up2) up2->setMixedPrecision(_enable);
    set_mp(dec2_layers);
    if (ds2_conv) ds2_conv->setMixedPrecision(_enable);
    if (ds2_sigmoid) ds2_sigmoid->setMixedPrecision(_enable);
    if (up1) up1->setMixedPrecision(_enable);
    set_mp(dec1_layers);
    if (final_conv) final_conv->setMixedPrecision(_enable);
    if (final_sigmoid) final_sigmoid->setMixedPrecision(_enable);
}

void NN_UNet_Layer::invalidateWeightCache() noexcept
{
    auto inv = [&](auto &v) { for (auto &l : v) l->invalidateWeightCache(); };
    inv(enc1_layers);
    if (pool1) pool1->invalidateWeightCache();
    inv(enc2_layers);
    if (pool2) pool2->invalidateWeightCache();
    inv(enc3_layers);
    if (pool3) pool3->invalidateWeightCache();
    inv(enc4_layers);
    if (pool4) pool4->invalidateWeightCache();
    inv(bottleneck_layers);
    if (up4) up4->invalidateWeightCache();
    inv(dec4_layers);
    if (up3) up3->invalidateWeightCache();
    inv(dec3_layers);
    if (ds3_conv) ds3_conv->invalidateWeightCache();
    if (ds3_sigmoid) ds3_sigmoid->invalidateWeightCache();
    if (up2) up2->invalidateWeightCache();
    inv(dec2_layers);
    if (ds2_conv) ds2_conv->invalidateWeightCache();
    if (ds2_sigmoid) ds2_sigmoid->invalidateWeightCache();
    if (up1) up1->invalidateWeightCache();
    inv(dec1_layers);
    if (final_conv) final_conv->invalidateWeightCache();
    if (final_sigmoid) final_sigmoid->invalidateWeightCache();
}

void NN_UNet_Layer::resetHeadBias(float bias)
{
    auto set_head_bias = [bias](std::unique_ptr<Conv2d_Layer> &head) {
        if (head)
        {
            uint32_t c = head->getOutputChannels();
            head->setBiases(Tensor(1, c, std::vector<float>(c, bias), head->getExecutionTarget()));
            head->invalidateWeightCache();
        }
    };
    set_head_bias(final_conv);
    set_head_bias(ds2_conv);
    set_head_bias(ds3_conv);
}

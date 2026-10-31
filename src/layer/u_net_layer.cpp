#include "layer/u_net_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
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

U_Net_Layer::U_Net_Layer(
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
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target)
{
    buildNetwork();
}

void U_Net_Layer::buildNetwork()
{
    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;

    uint32_t h = input_height;
    uint32_t w = input_width;

    // --- ENCODER 1: (H, W, in_channels) -> (H, W, c1) ---
    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, in_channels, c1, 3, 1, 1, execution_target));
    enc1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    enc1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target));
    enc1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    enc1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    pool1 = std::make_unique<Max_Pool_2d_Layer>(h, w, c1, 2, 2, 0, execution_target);

    // --- ENCODER 2: (H/2, W/2, c1) -> (H/2, W/2, c2) ---
    h /= 2; w /= 2;
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c2, 3, 1, 1, execution_target));
    enc2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    enc2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target));
    enc2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    enc2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    pool2 = std::make_unique<Max_Pool_2d_Layer>(h, w, c2, 2, 2, 0, execution_target);

    // --- ENCODER 3: (H/4, W/4, c2) -> (H/4, W/4, c3) ---
    h /= 2; w /= 2;
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c3, 3, 1, 1, execution_target));
    enc3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    enc3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target));
    enc3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    enc3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    pool3 = std::make_unique<Max_Pool_2d_Layer>(h, w, c3, 2, 2, 0, execution_target);

    // --- BOTTLENECK: (H/8, W/8, c3) -> (H/8, W/8, c4) ---
    h /= 2; w /= 2;
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c4, 3, 1, 1, execution_target));
    bottleneck_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c4, 1e-5f, 0.1f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c4, 3, 1, 1, execution_target));
    bottleneck_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c4, 1e-5f, 0.1f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    // --- DECODER 3: (H/8, W/8) -> (H/4, W/4), concat (c4 + c3) -> c3 ---
    up3 = std::make_unique<Up_Sample_2d_Layer>(h, w, c4, 2, execution_target);
    h *= 2; w *= 2;
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4 + c3, c3, 3, 1, 1, execution_target));
    dec3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    dec3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target));
    dec3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    dec3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    // --- DECODER 2: (H/4, W/4) -> (H/2, W/2), concat (c3 + c2) -> c2 ---
    up2 = std::make_unique<Up_Sample_2d_Layer>(h, w, c3, 2, execution_target);
    h *= 2; w *= 2;
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3 + c2, c2, 3, 1, 1, execution_target));
    dec2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    dec2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target));
    dec2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    dec2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    // --- DECODER 1: (H/2, W/2) -> (H, W), concat (c2 + c1) -> c1 ---
    up1 = std::make_unique<Up_Sample_2d_Layer>(h, w, c2, 2, execution_target);
    h *= 2; w *= 2;
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2 + c1, c1, 3, 1, 1, execution_target));
    dec1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    dec1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target));
    dec1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    dec1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    // --- HEAD: 1x1 Conv + Sigmoid ---
    final_conv = std::make_unique<Conv2d_Layer>(h, w, c1, out_channels, 1, 1, 0, execution_target);
    final_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);
}

std::unique_ptr<ILayer> U_Net_Layer::clone() const
{
    auto cloned = std::make_unique<U_Net_Layer>(input_height, input_width, in_channels, out_channels, base_channels, execution_target);
    return cloned;
}

Tensor U_Net_Layer::forward(const Tensor &_input_tensor)
{
    input_tensor = _input_tensor;
    Tensor cur = _input_tensor;

    // 1. Encoder 1
    for (auto &layer : enc1_layers) cur = layer->forward(cur);
    skip1_act = cur;
    cur = pool1->forward(cur);

    // 2. Encoder 2
    for (auto &layer : enc2_layers) cur = layer->forward(cur);
    skip2_act = cur;
    cur = pool2->forward(cur);

    // 3. Encoder 3
    for (auto &layer : enc3_layers) cur = layer->forward(cur);
    skip3_act = cur;
    cur = pool3->forward(cur);

    // 4. Bottleneck
    for (auto &layer : bottleneck_layers) cur = layer->forward(cur);

    uint32_t b = static_cast<uint32_t>(_input_tensor.getRows());
    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;

    uint32_t h3 = input_height / 4;
    uint32_t w3 = input_width / 4;
    uint32_t h2 = input_height / 2;
    uint32_t w2 = input_width / 2;
    uint32_t h1 = input_height;
    uint32_t w1 = input_width;

    // 5. Decoder 3
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
    for (auto &layer : dec3_layers) cur = layer->forward(cur);

    // 6. Decoder 2
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
    for (auto &layer : dec2_layers) cur = layer->forward(cur);

    // 7. Decoder 1
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
    for (auto &layer : dec1_layers) cur = layer->forward(cur);

    // 8. Head
    Tensor before_sigmoid = final_conv->forward(cur);
    output_tensor = final_sigmoid->forward(before_sigmoid);

    if (_input_tensor.getShape().getRank() == 4)
    {
        output_tensor.reshape(Shape{_input_tensor.getShape()[0], out_channels, input_height, input_width});
    }

    // {
    //     std::string mode_str = is_training ? "[TRAIN TRACE] " : "[EVAL TRACE] ";
    //     auto check = [&](const std::string& name, const Tensor& t) {
    //         auto d = t.getData();
    //         float mi = 1e9f, ma = -1e9f, sum = 0.0f;
    //         int z = 0, nan_c = 0;
    //         for (float v : d) {
    //             if (std::isnan(v)) nan_c++;
    //             else { mi = std::min(mi, v); ma = std::max(ma, v); sum += v; if (v == 0.0f) z++; }
    //         }
    //         std::cout << mode_str << name << ": min=" << mi << " max=" << ma << " avg=" << sum / d.size() << " zeros=" << z << "/" << d.size() << " nans=" << nan_c << std::endl;
    //     };
    //     check("input", _input_tensor);
    //     check("skip1", skip1_act);
    //     check("skip2", skip2_act);
    //     check("skip3", skip3_act);
    //     check("dec1_concat", dec1_concat);
    //     check("dec1_out(cur)", cur);
    //     check("final_conv_out", before_sigmoid);
    //     check("final_sigmoid_out", output_tensor);
    // }

    is_forward_completed = true;
    return output_tensor;
}

Tensor U_Net_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::runtime_error("U_Net_Layer::backward: Forward must be called before backward");
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

    uint32_t h1 = input_height;
    uint32_t w1 = input_width;
    uint32_t h2 = input_height / 2;
    uint32_t w2 = input_width / 2;
    uint32_t h3 = input_height / 4;
    uint32_t w3 = input_width / 4;

    grad = final_conv->backward(grad);

    // Backward Decoder 1
    for (auto it = dec1_layers.rbegin(); it != dec1_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    // Tách gradient dec1_concat theo kênh per-pixel: nửa trái up1 (c2), nửa phải skip1 (c1)
    grad.reshape(Shape{b * h1 * w1, c2 + c1});
    auto [d_up1, d_skip1] = grad.splitColumns(c2);
    d_up1.reshape(Shape{b, h1 * w1 * c2});
    d_skip1.reshape(Shape{b, h1 * w1 * c1});
    grad = up1->backward(d_up1);

    // Backward Decoder 2
    for (auto it = dec2_layers.rbegin(); it != dec2_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    // Tách gradient dec2_concat theo kênh per-pixel: nửa trái up2 (c3), nửa phải skip2 (c2)
    grad.reshape(Shape{b * h2 * w2, c3 + c2});
    auto [d_up2, d_skip2] = grad.splitColumns(c3);
    d_up2.reshape(Shape{b, h2 * w2 * c3});
    d_skip2.reshape(Shape{b, h2 * w2 * c2});
    grad = up2->backward(d_up2);

    // Backward Decoder 3
    for (auto it = dec3_layers.rbegin(); it != dec3_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    // Tách gradient dec3_concat theo kênh per-pixel: nửa trái up3 (c4), nửa phải skip3 (c3)
    grad.reshape(Shape{b * h3 * w3, c4 + c3});
    auto [d_up3, d_skip3] = grad.splitColumns(c4);
    d_up3.reshape(Shape{b, h3 * w3 * c4});
    d_skip3.reshape(Shape{b, h3 * w3 * c3});
    grad = up3->backward(d_up3);

    // Backward Bottleneck
    for (auto it = bottleneck_layers.rbegin(); it != bottleneck_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    // Backward Pool 3 + cộng skip 3
    grad = pool3->backward(grad);
    if (d_skip3.getDataType() != grad.getDataType())
    {
        d_skip3 = d_skip3.to(grad.getDataType());
    }
    Tensor temp_grad3(grad.getShape(), grad.getDataType(), grad.getExecutionTarget());
    grad.add(d_skip3, temp_grad3);
    grad = temp_grad3;
    for (auto it = enc3_layers.rbegin(); it != enc3_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    // Backward Pool 2 + cộng skip 2
    grad = pool2->backward(grad);
    if (d_skip2.getDataType() != grad.getDataType())
    {
        d_skip2 = d_skip2.to(grad.getDataType());
    }
    Tensor temp_grad2(grad.getShape(), grad.getDataType(), grad.getExecutionTarget());
    grad.add(d_skip2, temp_grad2);
    grad = temp_grad2;
    for (auto it = enc2_layers.rbegin(); it != enc2_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    // Backward Pool 1 + cộng skip 1
    grad = pool1->backward(grad);
    if (d_skip1.getDataType() != grad.getDataType())
    {
        d_skip1 = d_skip1.to(grad.getDataType());
    }
    Tensor temp_grad1(grad.getShape(), grad.getDataType(), grad.getExecutionTarget());
    grad.add(d_skip1, temp_grad1);
    grad = temp_grad1;
    for (auto it = enc1_layers.rbegin(); it != enc1_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    if (input_tensor.getShape().getRank() == 4)
    {
        grad.reshape(input_tensor.getShape());
    }

    input_gradient_tensor = grad;
    return input_gradient_tensor;
}

void U_Net_Layer::resetGradient()
{
    auto reset_vec = [](auto &vec) {
        for (auto &l : vec) l->resetGradient();
    };
    reset_vec(enc1_layers);
    pool1->resetGradient();
    reset_vec(enc2_layers);
    pool2->resetGradient();
    reset_vec(enc3_layers);
    pool3->resetGradient();
    reset_vec(bottleneck_layers);
    up3->resetGradient();
    reset_vec(dec3_layers);
    up2->resetGradient();
    reset_vec(dec2_layers);
    up1->resetGradient();
    reset_vec(dec1_layers);
    final_conv->resetGradient();
    final_sigmoid->resetGradient();

    skip1_act = Tensor(0, 0, execution_target);
    skip2_act = Tensor(0, 0, execution_target);
    skip3_act = Tensor(0, 0, execution_target);
    dec3_concat = Tensor(0, 0, execution_target);
    dec2_concat = Tensor(0, 0, execution_target);
    dec1_concat = Tensor(0, 0, execution_target);
    input_tensor = Tensor(0, 0, execution_target);
    output_tensor = Tensor(0, 0, execution_target);
    input_gradient_tensor = Tensor(0, 0, execution_target);
    is_forward_completed = false;
}

void U_Net_Layer::resetGradients()
{
    resetGradient();
    auto reset_vec = [](auto &vec) {
        for (auto &l : vec) l->resetGradients();
    };
    reset_vec(enc1_layers);
    pool1->resetGradients();
    reset_vec(enc2_layers);
    pool2->resetGradients();
    reset_vec(enc3_layers);
    pool3->resetGradients();
    reset_vec(bottleneck_layers);
    up3->resetGradients();
    reset_vec(dec3_layers);
    up2->resetGradients();
    reset_vec(dec2_layers);
    up1->resetGradients();
    reset_vec(dec1_layers);
    final_conv->resetGradients();
    final_sigmoid->resetGradients();
}

std::vector<std::pair<Tensor *, Tensor *>> U_Net_Layer::getParametersAndGradients()
{
    std::vector<std::pair<Tensor *, Tensor *>> total_params;
    auto append_params = [&](auto &layer_ptr) {
        auto p = layer_ptr->getParametersAndGradients();
        total_params.insert(total_params.end(), p.begin(), p.end());
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
    append_vec(bottleneck_layers);
    append_params(up3);
    append_vec(dec3_layers);
    append_params(up2);
    append_vec(dec2_layers);
    append_params(up1);
    append_vec(dec1_layers);
    append_params(final_conv);
    append_params(final_sigmoid);

    return total_params;
}

void U_Net_Layer::saveConfiguration(std::ofstream &_out) const
{
    _out.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _out.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _out.write(reinterpret_cast<const char *>(&in_channels), sizeof(in_channels));
    _out.write(reinterpret_cast<const char *>(&out_channels), sizeof(out_channels));
    _out.write(reinterpret_cast<const char *>(&base_channels), sizeof(base_channels));
}

void U_Net_Layer::saveInference(std::ofstream &_out) const
{
    saveConfiguration(_out);
    auto save_vec = [&](const auto &vec) {
        for (const auto &l : vec) l->saveInference(_out);
    };
    save_vec(enc1_layers);
    pool1->saveInference(_out);
    save_vec(enc2_layers);
    pool2->saveInference(_out);
    save_vec(enc3_layers);
    pool3->saveInference(_out);
    save_vec(bottleneck_layers);
    up3->saveInference(_out);
    save_vec(dec3_layers);
    up2->saveInference(_out);
    save_vec(dec2_layers);
    up1->saveInference(_out);
    save_vec(dec1_layers);
    final_conv->saveInference(_out);
    final_sigmoid->saveInference(_out);
}

void U_Net_Layer::loadInference(std::ifstream &_in)
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
    pool1->loadInference(_in);
    load_vec(enc2_layers);
    pool2->loadInference(_in);
    load_vec(enc3_layers);
    pool3->loadInference(_in);
    load_vec(bottleneck_layers);
    up3->loadInference(_in);
    load_vec(dec3_layers);
    up2->loadInference(_in);
    load_vec(dec2_layers);
    up1->loadInference(_in);
    load_vec(dec1_layers);
    final_conv->loadInference(_in);
    final_sigmoid->loadInference(_in);
}

void U_Net_Layer::saveCheckpoint(std::ofstream &_out) const
{
    saveConfiguration(_out);
    auto save_vec = [&](const auto &vec) {
        for (const auto &l : vec) l->saveCheckpoint(_out);
    };
    save_vec(enc1_layers);
    pool1->saveCheckpoint(_out);
    save_vec(enc2_layers);
    pool2->saveCheckpoint(_out);
    save_vec(enc3_layers);
    pool3->saveCheckpoint(_out);
    save_vec(bottleneck_layers);
    up3->saveCheckpoint(_out);
    save_vec(dec3_layers);
    up2->saveCheckpoint(_out);
    save_vec(dec2_layers);
    up1->saveCheckpoint(_out);
    save_vec(dec1_layers);
    final_conv->saveCheckpoint(_out);
    final_sigmoid->saveCheckpoint(_out);
}

void U_Net_Layer::loadCheckpoint(std::ifstream &_in)
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
    pool1->loadCheckpoint(_in);
    load_vec(enc2_layers);
    pool2->loadCheckpoint(_in);
    load_vec(enc3_layers);
    pool3->loadCheckpoint(_in);
    load_vec(bottleneck_layers);
    up3->loadCheckpoint(_in);
    load_vec(dec3_layers);
    up2->loadCheckpoint(_in);
    load_vec(dec2_layers);
    up1->loadCheckpoint(_in);
    load_vec(dec1_layers);
    final_conv->loadCheckpoint(_in);
    final_sigmoid->loadCheckpoint(_in);
}

void U_Net_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    execution_target = _new_execution_target;
    auto set_t = [&](auto &l) { l->setExecutionTarget(_new_execution_target); };
    auto set_vec = [&](auto &v) { for (auto &l : v) set_t(l); };

    set_vec(enc1_layers);
    set_t(pool1);
    set_vec(enc2_layers);
    set_t(pool2);
    set_vec(enc3_layers);
    set_t(pool3);
    set_vec(bottleneck_layers);
    set_t(up3);
    set_vec(dec3_layers);
    set_t(up2);
    set_vec(dec2_layers);
    set_t(up1);
    set_vec(dec1_layers);
    set_t(final_conv);
    set_t(final_sigmoid);

    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
}

void U_Net_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    is_accumulated = _is_accumulated;
    auto set_acc = [&](auto &v) { for (auto &l : v) l->setAccumulated(_is_accumulated); };
    set_acc(enc1_layers);
    pool1->setAccumulated(_is_accumulated);
    set_acc(enc2_layers);
    pool2->setAccumulated(_is_accumulated);
    set_acc(enc3_layers);
    pool3->setAccumulated(_is_accumulated);
    set_acc(bottleneck_layers);
    up3->setAccumulated(_is_accumulated);
    set_acc(dec3_layers);
    up2->setAccumulated(_is_accumulated);
    set_acc(dec2_layers);
    up1->setAccumulated(_is_accumulated);
    set_acc(dec1_layers);
    final_conv->setAccumulated(_is_accumulated);
    final_sigmoid->setAccumulated(_is_accumulated);
}

void U_Net_Layer::setTrainingMode(bool _training)
{
    is_training = _training;
    auto set_tm = [&](auto &v) { for (auto &l : v) l->setTrainingMode(_training); };
    set_tm(enc1_layers);
    pool1->setTrainingMode(_training);
    set_tm(enc2_layers);
    pool2->setTrainingMode(_training);
    set_tm(enc3_layers);
    pool3->setTrainingMode(_training);
    set_tm(bottleneck_layers);
    up3->setTrainingMode(_training);
    set_tm(dec3_layers);
    up2->setTrainingMode(_training);
    set_tm(dec2_layers);
    up1->setTrainingMode(_training);
    set_tm(dec1_layers);
    final_conv->setTrainingMode(_training);
    final_sigmoid->setTrainingMode(_training);
}

void U_Net_Layer::setMixedPrecision(bool _enable) noexcept
{
    auto set_mp = [&](auto &v) { for (auto &l : v) l->setMixedPrecision(_enable); };
    set_mp(enc1_layers);
    pool1->setMixedPrecision(_enable);
    set_mp(enc2_layers);
    pool2->setMixedPrecision(_enable);
    set_mp(enc3_layers);
    pool3->setMixedPrecision(_enable);
    set_mp(bottleneck_layers);
    up3->setMixedPrecision(_enable);
    set_mp(dec3_layers);
    up2->setMixedPrecision(_enable);
    set_mp(dec2_layers);
    up1->setMixedPrecision(_enable);
    set_mp(dec1_layers);
    final_conv->setMixedPrecision(_enable);
    final_sigmoid->setMixedPrecision(_enable);
}

void U_Net_Layer::invalidateWeightCache() noexcept
{
    auto inv = [&](auto &v) { for (auto &l : v) l->invalidateWeightCache(); };
    inv(enc1_layers);
    pool1->invalidateWeightCache();
    inv(enc2_layers);
    pool2->invalidateWeightCache();
    inv(enc3_layers);
    pool3->invalidateWeightCache();
    inv(bottleneck_layers);
    up3->invalidateWeightCache();
    inv(dec3_layers);
    up2->invalidateWeightCache();
    inv(dec2_layers);
    up1->invalidateWeightCache();
    inv(dec1_layers);
    final_conv->invalidateWeightCache();
    final_sigmoid->invalidateWeightCache();
}

void U_Net_Layer::inspectAllBatchNorms() const
{
    auto check_list = [&](const std::string &prefix, const auto &vec)
    {
        for (size_t i = 0; i < vec.size(); ++i)
        {
            auto *bn = dynamic_cast<Batch_Norm_2d_Layer*>(vec[i].get());
            if (bn)
            {
                auto rm = bn->getRunningMean().getData();
                auto rv = bn->getRunningVariance().getData();
                int rm_nans = 0, rv_nans = 0;
                for (float v : rm) if (std::isnan(v)) rm_nans++;
                for (float v : rv) if (std::isnan(v)) rv_nans++;
                float rm_min = 1e9f, rm_max = -1e9f, rm_sum = 0.0f;
                float rv_min = 1e9f, rv_max = -1e9f, rv_sum = 0.0f;
                for (float v : rm) { if (!std::isnan(v)) { rm_min = std::min(rm_min, v); rm_max = std::max(rm_max, v); rm_sum += v; } }
                for (float v : rv) { if (!std::isnan(v)) { rv_min = std::min(rv_min, v); rv_max = std::max(rv_max, v); rv_sum += v; } }
                auto bm = bn->getBatchMean().getData();
                auto bv = bn->getBatchVariance().getData();
                float bm_min = 1e9f, bm_max = -1e9f, bm_sum = 0.0f;
                float bv_min = 1e9f, bv_max = -1e9f, bv_sum = 0.0f;
                for (float v : bm) { if (!std::isnan(v)) { bm_min = std::min(bm_min, v); bm_max = std::max(bm_max, v); bm_sum += v; } }
                for (float v : bv) { if (!std::isnan(v)) { bv_min = std::min(bv_min, v); bv_max = std::max(bv_max, v); bv_sum += v; } }
                std::cout << prefix << "[" << i << "] (C=" << rm.size() << "): "
                          << "RM[" << rm_min << ".." << rm_max << " avg=" << rm_sum / rm.size() << "], "
                          << "RV[" << rv_min << ".." << rv_max << " avg=" << rv_sum / rv.size() << "]"
                          << (bm.empty() ? "" : std::format(" | BM[{:.2f}..{:.2f} avg={:.2f}], BV[{:.2f}..{:.2f} avg={:.2f}]", bm_min, bm_max, bm_sum / bm.size(), bv_min, bv_max, bv_sum / bv.size()));
                if (rm_nans > 0 || rv_nans > 0)
                {
                    std::cout << " (NaNs: RM=" << rm_nans << ", RV=" << rv_nans << ")";
                }
                std::cout << std::endl;
            }
            auto *conv = dynamic_cast<Conv2d_Layer*>(vec[i].get());
            if (conv)
            {
                auto w = conv->getWeights().getData();
                auto b = conv->getBiases().getData();
                int w_nans = 0, b_nans = 0;
                for (float v : w) if (std::isnan(v)) w_nans++;
                for (float v : b) if (std::isnan(v)) b_nans++;
                if (w_nans > 0 || b_nans > 0)
                {
                    std::cout << prefix << "[" << i << "] CONV2D NaNs! W: " << w_nans << "/" << w.size()
                              << ", B: " << b_nans << "/" << b.size() << std::endl;
                }
            }
        }
    };

    check_list("enc1", enc1_layers);
    check_list("enc2", enc2_layers);
    check_list("enc3", enc3_layers);
    check_list("bottleneck", bottleneck_layers);
    check_list("dec3", dec3_layers);
    check_list("dec2", dec2_layers);
    check_list("dec1", dec1_layers);

    if (final_conv)
    {
        auto b = final_conv->getBiases().getData();
        auto w = final_conv->getWeights().getData();
        float w_mean = 0.0f;
        for (float v : w) w_mean += v;
        if (!w.empty()) w_mean /= w.size();
        std::cout << "final_conv: bias=" << (b.empty() ? 0.0f : b[0]) << ", w_mean=" << w_mean << std::endl;
    }
}

void U_Net_Layer::resetRunningStats()
{
    auto reset_list = [&](auto &vec)
    {
        for (auto &layer : vec)
        {
            auto *bn = dynamic_cast<Batch_Norm_2d_Layer*>(layer.get());
            if (bn)
            {
                uint32_t c = bn->getChannels();
                bn->setRunningMean(Tensor(1, c, std::vector<float>(c, 0.0f), bn->getExecutionTarget()));
                bn->setRunningVariance(Tensor(1, c, std::vector<float>(c, 1.0f), bn->getExecutionTarget()));
            }
        }
    };

    reset_list(enc1_layers);
    reset_list(enc2_layers);
    reset_list(enc3_layers);
    reset_list(bottleneck_layers);
    reset_list(dec3_layers);
    reset_list(dec2_layers);
    reset_list(dec1_layers);
}

bool U_Net_Layer::hasNanInRunningStats() const
{
    auto check_list = [](const auto &vec) -> bool
    {
        for (const auto &layer : vec)
        {
            auto *bn = dynamic_cast<Batch_Norm_2d_Layer*>(layer.get());
            if (bn)
            {
                auto rm = bn->getRunningMean().getData();
                auto rv = bn->getRunningVariance().getData();
                for (float v : rm) if (std::isnan(v)) return true;
                for (float v : rv) if (std::isnan(v)) return true;
            }
        }
        return false;
    };

    return check_list(enc1_layers) || check_list(enc2_layers) || check_list(enc3_layers) ||
           check_list(bottleneck_layers) || check_list(dec3_layers) || check_list(dec2_layers) ||
           check_list(dec1_layers);
}

void U_Net_Layer::resetHeadBias(float bias)
{
    if (final_conv)
    {
        uint32_t c = final_conv->getOutputChannels();
        final_conv->setBiases(Tensor(1, c, std::vector<float>(c, bias), final_conv->getExecutionTarget()));
        final_conv->invalidateWeightCache();
    }
}

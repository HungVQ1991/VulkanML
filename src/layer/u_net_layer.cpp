#include "layer/u_net_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "math/tensor_impl.h"

namespace
{
static uint32_t validateUNetIoTensor(
    const Tensor &t,
    Execution_Target expected_target,
    uint32_t expected_channels,
    uint32_t expected_height,
    uint32_t expected_width,
    const char *fn_name,
    const size_t *expected_batch = nullptr)
{
    if (!t.getImplementation())
    {
        throw std::invalid_argument(std::format("U_Net_Layer::{}: Tensor has null implementation", fn_name));
    }
    if (t.getExecutionTarget() != expected_target)
    {
        throw std::invalid_argument(std::format("U_Net_Layer::{}: Execution_Target mismatch", fn_name));
    }
    if (t.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::format("U_Net_Layer::{}: TERNARY_INT2 tensor is not supported", fn_name));
    }
    if (t.isEmpty())
    {
        throw std::invalid_argument(std::format("U_Net_Layer::{}: Tensor is empty", fn_name));
    }
    const Shape &sh = t.getShape();
    const size_t rank = sh.getRank();
    if (rank != 2 && rank != 4)
    {
        throw std::invalid_argument(std::format("U_Net_Layer::{}: Expected rank-2 or rank-4 tensor, got rank {}", fn_name, rank));
    }
    const size_t batch = sh[0];
    if (batch == 0 || batch > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(std::format("U_Net_Layer::{}: Invalid batch size {}", fn_name, batch));
    }
    if (expected_batch && batch != *expected_batch)
    {
        throw std::invalid_argument(std::format(
            "U_Net_Layer::{}: Batch size {} mismatch (expected {})", fn_name, batch, *expected_batch));
    }
    const size_t expected_cols = static_cast<size_t>(expected_channels) * expected_height * expected_width;
    if (rank == 2)
    {
        if (sh[1] != expected_cols)
        {
            throw std::invalid_argument(std::format(
                "U_Net_Layer::{}: Rank-2 column count {} mismatch (expected {})", fn_name, sh[1], expected_cols));
        }
    }
    else
    {
        if (sh[1] != expected_channels || sh[2] != expected_height || sh[3] != expected_width)
        {
            throw std::invalid_argument(std::format(
                "U_Net_Layer::{}: Rank-4 shape must be [B, {}, {}, {}]",
                fn_name, expected_channels, expected_height, expected_width));
        }
    }
    return static_cast<uint32_t>(batch);
}
}

U_Net_Layer::U_Net_Layer(
    uint32_t _height,
    uint32_t _width,
    uint32_t _in_channels,
    uint32_t _out_channels,
    uint32_t _base_channels,
    Execution_Target _execution_target,
    Data_Type _data_type)
    : input_height(_height),
      input_width(_width),
      in_channels(_in_channels),
      out_channels(_out_channels),
      base_channels(_base_channels),
      execution_target(_execution_target),
      data_type(_data_type),
      is_forward_completed(false),
      is_training(true),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      skip1_act(0, 0, _execution_target),
      skip2_act(0, 0, _execution_target),
      skip3_act(0, 0, _execution_target),
      dec3_concat(0, 0, _execution_target),
      dec2_concat(0, 0, _execution_target),
      dec1_concat(0, 0, _execution_target),
      in_nhwc_scratch(0, 0, _execution_target),
      out_nchw_scratch(0, 0, _execution_target),
      grad_nhwc_scratch(0, 0, _execution_target),
      in_grad_nchw_scratch(0, 0, _execution_target),
      up3_cast_scratch(0, 0, _execution_target),
      up2_cast_scratch(0, 0, _execution_target),
      up1_cast_scratch(0, 0, _execution_target),
      dec3_concat_2d_scratch(0, 0, _execution_target),
      dec2_concat_2d_scratch(0, 0, _execution_target),
      dec1_concat_2d_scratch(0, 0, _execution_target),
      sig_grad_cast_scratch(0, 0, _execution_target),
      d_up1_scratch(0, 0, _execution_target),
      d_skip1_scratch(0, 0, _execution_target),
      d_skip1_cast_scratch(0, 0, _execution_target),
      enc1_grad_sum(0, 0, _execution_target),
      d_up2_scratch(0, 0, _execution_target),
      d_skip2_scratch(0, 0, _execution_target),
      d_skip2_cast_scratch(0, 0, _execution_target),
      enc2_grad_sum(0, 0, _execution_target),
      d_up3_scratch(0, 0, _execution_target),
      d_skip3_scratch(0, 0, _execution_target),
      d_skip3_cast_scratch(0, 0, _execution_target),
      enc3_grad_sum(0, 0, _execution_target)
{
    if (in_channels == 0 || out_channels == 0 || base_channels == 0 ||
        input_height < 8 || input_width < 8 || (input_height % 8) != 0 || (input_width % 8) != 0)
    {
        throw std::invalid_argument("U_Net_Layer: height and width must be >= 8 and divisible by 8, and channels > 0");
    }
    if (base_channels > std::numeric_limits<uint32_t>::max() / 8)
    {
        throw std::overflow_error("U_Net_Layer: base_channels * 8 exceeds uint32_t limit");
    }
    buildNetwork();
}

void U_Net_Layer::buildNetwork()
{
    enc1_layers.clear();
    enc2_layers.clear();
    enc3_layers.clear();
    bottleneck_layers.clear();
    dec3_layers.clear();
    dec2_layers.clear();
    dec1_layers.clear();

    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;

    uint32_t h = input_height;
    uint32_t w = input_width;

    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, in_channels, c1, 3, 1, 1, execution_target, data_type));
    enc1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    enc1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target, data_type));
    enc1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    enc1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    pool1 = std::make_unique<Max_Pool_2d_Layer>(h, w, c1, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c2, 3, 1, 1, execution_target, data_type));
    enc2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    enc2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target, data_type));
    enc2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    enc2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    pool2 = std::make_unique<Max_Pool_2d_Layer>(h, w, c2, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c3, 3, 1, 1, execution_target, data_type));
    enc3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    enc3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target, data_type));
    enc3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    enc3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    pool3 = std::make_unique<Max_Pool_2d_Layer>(h, w, c3, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c4, 3, 1, 1, execution_target, data_type));
    bottleneck_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c4, 1e-5f, 0.1f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c4, 3, 1, 1, execution_target, data_type));
    bottleneck_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c4, 1e-5f, 0.1f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    up3 = std::make_unique<Up_Sample_2d_Layer>(h, w, c4, 2, execution_target);
    h *= 2; w *= 2;
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4 + c3, c3, 3, 1, 1, execution_target, data_type));
    dec3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    dec3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target, data_type));
    dec3_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c3, 1e-5f, 0.1f, execution_target));
    dec3_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    up2 = std::make_unique<Up_Sample_2d_Layer>(h, w, c3, 2, execution_target);
    h *= 2; w *= 2;
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3 + c2, c2, 3, 1, 1, execution_target, data_type));
    dec2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    dec2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target, data_type));
    dec2_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c2, 1e-5f, 0.1f, execution_target));
    dec2_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    up1 = std::make_unique<Up_Sample_2d_Layer>(h, w, c2, 2, execution_target);
    h *= 2; w *= 2;
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2 + c1, c1, 3, 1, 1, execution_target, data_type));
    dec1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    dec1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target, data_type));
    dec1_layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(h, w, c1, 1e-5f, 0.1f, execution_target));
    dec1_layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

    final_conv = std::make_unique<Conv2d_Layer>(h, w, c1, out_channels, 1, 1, 0, execution_target, data_type);
    final_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);
}

void U_Net_Layer::prepareInference()
{
    auto prep_vec = [](auto &vec) {
        for (auto &l : vec) l->prepareInference();
    };
    prep_vec(enc1_layers);
    if (pool1) pool1->prepareInference();
    prep_vec(enc2_layers);
    if (pool2) pool2->prepareInference();
    prep_vec(enc3_layers);
    if (pool3) pool3->prepareInference();
    prep_vec(bottleneck_layers);
    if (up3) up3->prepareInference();
    prep_vec(dec3_layers);
    if (up2) up2->prepareInference();
    prep_vec(dec2_layers);
    if (up1) up1->prepareInference();
    prep_vec(dec1_layers);
    if (final_conv) final_conv->prepareInference();
    if (final_sigmoid) final_sigmoid->prepareInference();
}

void U_Net_Layer::quantizeToTernary()
{
    data_type = Data_Type::TERNARY_INT2;
    auto quant_vec = [](auto &vec) {
        for (auto &l : vec)
        {
            if (auto conv = dynamic_cast<Conv2d_Layer*>(l.get())) conv->quantizeToTernary();
        }
    };
    quant_vec(enc1_layers);
    quant_vec(enc2_layers);
    quant_vec(enc3_layers);
    quant_vec(bottleneck_layers);
    quant_vec(dec3_layers);
    quant_vec(dec2_layers);
    quant_vec(dec1_layers);
    if (final_conv) final_conv->quantizeToTernary();
}

void U_Net_Layer::dequantizeFromTernary()
{
    data_type = is_mixed_precision_enabled ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    auto dequant_vec = [](auto &vec) {
        for (auto &l : vec)
        {
            if (l) l->dequantizeFromTernary();
        }
    };
    dequant_vec(enc1_layers);
    dequant_vec(enc2_layers);
    dequant_vec(enc3_layers);
    dequant_vec(bottleneck_layers);
    dequant_vec(dec3_layers);
    dequant_vec(dec2_layers);
    dequant_vec(dec1_layers);
    if (final_conv) final_conv->dequantizeFromTernary();
}

void U_Net_Layer::snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const
{
    auto snap_vec = [&](const auto &vec) {
        for (const auto &l : vec)
        {
            if (l) l->snapshotNonTrainableState(_out);
        }
    };
    snap_vec(enc1_layers);
    snap_vec(enc2_layers);
    snap_vec(enc3_layers);
    snap_vec(bottleneck_layers);
    snap_vec(dec3_layers);
    snap_vec(dec2_layers);
    snap_vec(dec1_layers);
}

void U_Net_Layer::restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor)
{
    auto rest_vec = [&](auto &vec) {
        for (auto &l : vec)
        {
            if (l) l->restoreNonTrainableState(_in, _cursor);
        }
    };
    rest_vec(enc1_layers);
    rest_vec(enc2_layers);
    rest_vec(enc3_layers);
    rest_vec(bottleneck_layers);
    rest_vec(dec3_layers);
    rest_vec(dec2_layers);
    rest_vec(dec1_layers);
}

std::unique_ptr<ILayer> U_Net_Layer::clone() const
{
    auto cloned = std::make_unique<U_Net_Layer>(input_height, input_width, in_channels, out_channels, base_channels, execution_target, data_type);
    auto clone_vec = [](const auto &src_vec, auto &dst_vec) {
        dst_vec.clear();
        dst_vec.reserve(src_vec.size());
        for (const auto &l : src_vec)
        {
            dst_vec.push_back(l ? l->clone() : nullptr);
        }
    };
    auto clone_as = [](const auto &src_ptr, auto &dst_ptr) {
        using LayerT = typename std::remove_reference_t<decltype(dst_ptr)>::element_type;
        if (src_ptr)
        {
            auto base_clone = src_ptr->clone();
            dst_ptr.reset(static_cast<LayerT *>(base_clone.release()));
        }
    };

    clone_vec(enc1_layers, cloned->enc1_layers);
    clone_as(pool1, cloned->pool1);
    clone_vec(enc2_layers, cloned->enc2_layers);
    clone_as(pool2, cloned->pool2);
    clone_vec(enc3_layers, cloned->enc3_layers);
    clone_as(pool3, cloned->pool3);
    clone_vec(bottleneck_layers, cloned->bottleneck_layers);
    clone_as(up3, cloned->up3);
    clone_vec(dec3_layers, cloned->dec3_layers);
    clone_as(up2, cloned->up2);
    clone_vec(dec2_layers, cloned->dec2_layers);
    clone_as(up1, cloned->up1);
    clone_vec(dec1_layers, cloned->dec1_layers);
    clone_as(final_conv, cloned->final_conv);
    clone_as(final_sigmoid, cloned->final_sigmoid);

    cloned->setTrainingMode(is_training);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setAccumulated(is_accumulated);
    return cloned;
}

Tensor U_Net_Layer::forward(const Tensor &_input_tensor)
{
    uint32_t b = validateUNetIoTensor(
        _input_tensor, execution_target, in_channels, input_height, input_width, "forward");

    input_tensor = _input_tensor;
    Tensor cur = _input_tensor;

    if (_input_tensor.getShape().getRank() == 4)
    {
        if (in_channels > 1)
        {
            _input_tensor.permute({0, 2, 3, 1}).contiguous(in_nhwc_scratch);
            cur = in_nhwc_scratch.view(Shape{b, static_cast<size_t>(input_height) * input_width * in_channels});
        }
        else
        {
            cur = _input_tensor.view(Shape{b, static_cast<size_t>(input_height) * input_width});
        }
    }

    for (auto &layer : enc1_layers) cur = layer->forward(cur);
    skip1_act = cur;
    cur = pool1->forward(cur);

    for (auto &layer : enc2_layers) cur = layer->forward(cur);
    skip2_act = cur;
    cur = pool2->forward(cur);

    for (auto &layer : enc3_layers) cur = layer->forward(cur);
    skip3_act = cur;
    cur = pool3->forward(cur);

    for (auto &layer : bottleneck_layers) cur = layer->forward(cur);

    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;

    const size_t b_sz = b;
    const size_t h3_sz = input_height / 4;
    const size_t w3_sz = input_width / 4;
    const size_t h2_sz = input_height / 2;
    const size_t w2_sz = input_width / 2;
    const size_t h1_sz = input_height;
    const size_t w1_sz = input_width;

    Tensor up_feat3 = up3->forward(cur);
    if (up_feat3.getDataType() != skip3_act.getDataType())
    {
        up_feat3.to(skip3_act.getDataType(), up3_cast_scratch);
        up_feat3 = up3_cast_scratch;
    }
    up_feat3.view(Shape{b_sz * h3_sz * w3_sz, c4}).concatenateColumns(skip3_act.view(Shape{b_sz * h3_sz * w3_sz, c3}), dec3_concat_2d_scratch);
    dec3_concat = dec3_concat_2d_scratch.view(Shape{b_sz, h3_sz * w3_sz * (static_cast<size_t>(c4) + c3)});
    cur = dec3_concat;
    for (auto &layer : dec3_layers) cur = layer->forward(cur);

    Tensor up_feat2 = up2->forward(cur);
    if (up_feat2.getDataType() != skip2_act.getDataType())
    {
        up_feat2.to(skip2_act.getDataType(), up2_cast_scratch);
        up_feat2 = up2_cast_scratch;
    }
    up_feat2.view(Shape{b_sz * h2_sz * w2_sz, c3}).concatenateColumns(skip2_act.view(Shape{b_sz * h2_sz * w2_sz, c2}), dec2_concat_2d_scratch);
    dec2_concat = dec2_concat_2d_scratch.view(Shape{b_sz, h2_sz * w2_sz * (static_cast<size_t>(c3) + c2)});
    cur = dec2_concat;
    for (auto &layer : dec2_layers) cur = layer->forward(cur);

    Tensor up_feat1 = up1->forward(cur);
    if (up_feat1.getDataType() != skip1_act.getDataType())
    {
        up_feat1.to(skip1_act.getDataType(), up1_cast_scratch);
        up_feat1 = up1_cast_scratch;
    }
    up_feat1.view(Shape{b_sz * h1_sz * w1_sz, c2}).concatenateColumns(skip1_act.view(Shape{b_sz * h1_sz * w1_sz, c1}), dec1_concat_2d_scratch);
    dec1_concat = dec1_concat_2d_scratch.view(Shape{b_sz, h1_sz * w1_sz * (static_cast<size_t>(c2) + c1)});
    cur = dec1_concat;
    for (auto &layer : dec1_layers) cur = layer->forward(cur);

    Tensor before_sigmoid = final_conv->forward(cur);
    Tensor sig_out = final_sigmoid->forward(before_sigmoid);

    if (_input_tensor.getShape().getRank() == 4)
    {
        if (out_channels > 1)
        {
            sig_out.view(Shape{b, input_height, input_width, out_channels}).permute({0, 3, 1, 2}).contiguous(out_nchw_scratch);
            output_tensor = out_nchw_scratch;
        }
        else
        {
            output_tensor = sig_out.view(Shape{b, out_channels, input_height, input_width});
        }
    }
    else
    {
        output_tensor = sig_out;
    }

    is_forward_completed = true;
    return output_tensor;
}

Tensor U_Net_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::runtime_error("U_Net_Layer::backward: Forward must be called before backward");
    }
    const size_t fwd_batch = (input_tensor.getShape().getRank() >= 1) ? input_tensor.getShape()[0] : 0;
    uint32_t b = validateUNetIoTensor(
        _output_gradient, execution_target, out_channels, input_height, input_width, "backward", &fwd_batch);
    const size_t b_sz = b;
    Tensor grad_in = _output_gradient;
    if (grad_in.getShape().getRank() == 4)
    {
        if (out_channels > 1)
        {
            grad_in.permute({0, 2, 3, 1}).contiguous(grad_nhwc_scratch);
            grad_in = grad_nhwc_scratch.view(Shape{b, static_cast<size_t>(input_height) * input_width * out_channels});
        }
        else
        {
            grad_in = grad_in.view(Shape{b, static_cast<size_t>(input_height) * input_width});
        }
    }

    Data_Type sig_dtype = final_sigmoid->getOutput().getDataType();
    if (grad_in.getDataType() != sig_dtype)
    {
        grad_in.to(sig_dtype, sig_grad_cast_scratch);
        grad_in = sig_grad_cast_scratch;
    }

    Tensor grad = final_sigmoid->backward(grad_in);
    if (grad.getShape().getRank() != 2)
    {
        grad = grad.view(Shape{b, grad.getTotalElements() / b});
    }

    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;

    const size_t h1_sz = input_height;
    const size_t w1_sz = input_width;
    const size_t h2_sz = input_height / 2;
    const size_t w2_sz = input_width / 2;
    const size_t h3_sz = input_height / 4;
    const size_t w3_sz = input_width / 4;

    grad = final_conv->backward(grad);

    for (auto it = dec1_layers.rbegin(); it != dec1_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    grad.view(Shape{b_sz * h1_sz * w1_sz, static_cast<size_t>(c2) + c1}).splitColumns(c2, d_up1_scratch, d_skip1_scratch);
    Tensor d_up1 = d_up1_scratch.view(Shape{b_sz, h1_sz * w1_sz * c2});
    Tensor d_skip1 = d_skip1_scratch.view(Shape{b_sz, h1_sz * w1_sz * c1});
    grad = up1->backward(d_up1);

    for (auto it = dec2_layers.rbegin(); it != dec2_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    grad.view(Shape{b_sz * h2_sz * w2_sz, static_cast<size_t>(c3) + c2}).splitColumns(c3, d_up2_scratch, d_skip2_scratch);
    Tensor d_up2 = d_up2_scratch.view(Shape{b_sz, h2_sz * w2_sz * c3});
    Tensor d_skip2 = d_skip2_scratch.view(Shape{b_sz, h2_sz * w2_sz * c2});
    grad = up2->backward(d_up2);

    for (auto it = dec3_layers.rbegin(); it != dec3_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }
    grad.view(Shape{b_sz * h3_sz * w3_sz, static_cast<size_t>(c4) + c3}).splitColumns(c4, d_up3_scratch, d_skip3_scratch);
    Tensor d_up3 = d_up3_scratch.view(Shape{b_sz, h3_sz * w3_sz * c4});
    Tensor d_skip3 = d_skip3_scratch.view(Shape{b_sz, h3_sz * w3_sz * c3});
    grad = up3->backward(d_up3);

    for (auto it = bottleneck_layers.rbegin(); it != bottleneck_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    grad = pool3->backward(grad);
    if (d_skip3.getDataType() != grad.getDataType())
    {
        d_skip3.to(grad.getDataType(), d_skip3_cast_scratch);
        d_skip3 = d_skip3_cast_scratch;
    }
    grad.add(d_skip3, enc3_grad_sum);
    grad = enc3_grad_sum;
    for (auto it = enc3_layers.rbegin(); it != enc3_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    grad = pool2->backward(grad);
    if (d_skip2.getDataType() != grad.getDataType())
    {
        d_skip2.to(grad.getDataType(), d_skip2_cast_scratch);
        d_skip2 = d_skip2_cast_scratch;
    }
    grad.add(d_skip2, enc2_grad_sum);
    grad = enc2_grad_sum;
    for (auto it = enc2_layers.rbegin(); it != enc2_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    grad = pool1->backward(grad);
    if (d_skip1.getDataType() != grad.getDataType())
    {
        d_skip1.to(grad.getDataType(), d_skip1_cast_scratch);
        d_skip1 = d_skip1_cast_scratch;
    }
    grad.add(d_skip1, enc1_grad_sum);
    grad = enc1_grad_sum;
    for (auto it = enc1_layers.rbegin(); it != enc1_layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    if (input_tensor.getShape().getRank() == 4)
    {
        if (in_channels > 1)
        {
            grad.view(Shape{b, input_height, input_width, in_channels}).permute({0, 3, 1, 2}).contiguous(in_grad_nchw_scratch);
            input_gradient_tensor = in_grad_nchw_scratch;
        }
        else
        {
            input_gradient_tensor = grad.view(input_tensor.getShape());
        }
    }
    else
    {
        input_gradient_tensor = grad;
    }

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
    is_forward_completed = false;
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

std::vector<const Tensor *> U_Net_Layer::getNoDecayParameters() const
{
    std::vector<const Tensor *> no_decay;
    auto append_nd = [&](const auto &layer_ptr) {
        if (layer_ptr)
        {
            auto p = layer_ptr->getNoDecayParameters();
            no_decay.insert(no_decay.end(), p.begin(), p.end());
        }
    };
    auto append_vec = [&](const auto &vec) {
        for (const auto &l : vec) append_nd(l);
    };
    append_vec(enc1_layers);
    append_vec(enc2_layers);
    append_vec(enc3_layers);
    append_vec(bottleneck_layers);
    append_vec(dec3_layers);
    append_vec(dec2_layers);
    append_vec(dec1_layers);
    append_nd(final_conv);
    return no_decay;
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
    uint32_t h = 0, w = 0, ic = 0, oc = 0, bc = 0;
    _in.read(reinterpret_cast<char *>(&h), sizeof(h));
    _in.read(reinterpret_cast<char *>(&w), sizeof(w));
    _in.read(reinterpret_cast<char *>(&ic), sizeof(ic));
    _in.read(reinterpret_cast<char *>(&oc), sizeof(oc));
    _in.read(reinterpret_cast<char *>(&bc), sizeof(bc));
    if (!_in.good() || ic == 0 || oc == 0 || bc == 0 || bc > (std::numeric_limits<uint32_t>::max() / 8) || h < 8 || w < 8 || (h % 8) != 0 || (w % 8) != 0)
    {
        throw std::runtime_error("U_Net_Layer::loadInference: Invalid or corrupted U-Net configuration header");
    }
    if (h != input_height || w != input_width || ic != in_channels || oc != out_channels || bc != base_channels)
    {
        input_height = h;
        input_width = w;
        in_channels = ic;
        out_channels = oc;
        base_channels = bc;
        buildNetwork();
        setMixedPrecision(is_mixed_precision_enabled);
        setTrainingMode(is_training);
        setAccumulated(is_accumulated);
    }

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
    if (final_conv) data_type = final_conv->getDataType();
    resetGradient();
    invalidateWeightCache();
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
    uint32_t h = 0, w = 0, ic = 0, oc = 0, bc = 0;
    _in.read(reinterpret_cast<char *>(&h), sizeof(h));
    _in.read(reinterpret_cast<char *>(&w), sizeof(w));
    _in.read(reinterpret_cast<char *>(&ic), sizeof(ic));
    _in.read(reinterpret_cast<char *>(&oc), sizeof(oc));
    _in.read(reinterpret_cast<char *>(&bc), sizeof(bc));
    if (!_in.good() || ic == 0 || oc == 0 || bc == 0 || bc > (std::numeric_limits<uint32_t>::max() / 8) || h < 8 || w < 8 || (h % 8) != 0 || (w % 8) != 0)
    {
        throw std::runtime_error("U_Net_Layer::loadCheckpoint: Invalid or corrupted U-Net configuration header");
    }
    if (h != input_height || w != input_width || ic != in_channels || oc != out_channels || bc != base_channels)
    {
        input_height = h;
        input_width = w;
        in_channels = ic;
        out_channels = oc;
        base_channels = bc;
        buildNetwork();
        setMixedPrecision(is_mixed_precision_enabled);
        setTrainingMode(is_training);
        setAccumulated(is_accumulated);
    }

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
    if (final_conv) data_type = final_conv->getDataType();
    resetGradient();
    invalidateWeightCache();
}

void U_Net_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target)
    {
        return;
    }
    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;
    auto set_t = [&](auto &l) { if (l) l->setExecutionTarget(_new_execution_target); };
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
    skip1_act.setExecutionTarget(_new_execution_target);
    skip2_act.setExecutionTarget(_new_execution_target);
    skip3_act.setExecutionTarget(_new_execution_target);
    dec3_concat.setExecutionTarget(_new_execution_target);
    dec2_concat.setExecutionTarget(_new_execution_target);
    dec1_concat.setExecutionTarget(_new_execution_target);
    in_nhwc_scratch = Tensor(0, 0, _new_execution_target);
    out_nchw_scratch = Tensor(0, 0, _new_execution_target);
    grad_nhwc_scratch = Tensor(0, 0, _new_execution_target);
    in_grad_nchw_scratch = Tensor(0, 0, _new_execution_target);
    up3_cast_scratch = Tensor(0, 0, _new_execution_target);
    up2_cast_scratch = Tensor(0, 0, _new_execution_target);
    up1_cast_scratch = Tensor(0, 0, _new_execution_target);
    dec3_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    dec2_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    dec1_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    sig_grad_cast_scratch = Tensor(0, 0, _new_execution_target);
    d_up1_scratch = Tensor(0, 0, _new_execution_target);
    d_skip1_scratch = Tensor(0, 0, _new_execution_target);
    d_skip1_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc1_grad_sum = Tensor(0, 0, _new_execution_target);
    d_up2_scratch = Tensor(0, 0, _new_execution_target);
    d_skip2_scratch = Tensor(0, 0, _new_execution_target);
    d_skip2_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc2_grad_sum = Tensor(0, 0, _new_execution_target);
    d_up3_scratch = Tensor(0, 0, _new_execution_target);
    d_skip3_scratch = Tensor(0, 0, _new_execution_target);
    d_skip3_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc3_grad_sum = Tensor(0, 0, _new_execution_target);
    invalidateWeightCache();
}

void U_Net_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    ILayer::setAccumulated(_is_accumulated);
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
    ILayer::setTrainingMode(_training);
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
    ILayer::setMixedPrecision(_enable);
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
    auto summarize = [](const std::vector<float> &vals, bool check_neg) {
        size_t finite_cnt = 0;
        size_t non_finite_cnt = 0;
        size_t neg_cnt = 0;
        float mn = 0.0f;
        float mx = 0.0f;
        double sum = 0.0;
        for (float v : vals)
        {
            if (!std::isfinite(v))
            {
                ++non_finite_cnt;
                continue;
            }
            if (check_neg && v < 0.0f)
            {
                ++neg_cnt;
            }
            if (finite_cnt == 0)
            {
                mn = mx = v;
            }
            else
            {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            sum += static_cast<double>(v);
            ++finite_cnt;
        }
        double avg = (finite_cnt > 0) ? (sum / static_cast<double>(finite_cnt)) : 0.0;
        return std::tuple<size_t, size_t, size_t, float, float, double>{
            finite_cnt, non_finite_cnt, neg_cnt, mn, mx, avg
        };
    };

    auto fmt_stat = [](const char *label, const std::vector<float> &vals,
                       const std::tuple<size_t, size_t, size_t, float, float, double> &s) -> std::string {
        auto [fin, nonfin, neg, mn, mx, avg] = s;
        if (vals.empty())
        {
            return std::format("{}[empty]", label);
        }
        if (fin == 0)
        {
            return std::format("{}[none finite (nonfinite={})]", label, nonfin);
        }
        if (neg > 0)
        {
            return std::format("{}[{:.4g}..{:.4g} avg={:.4g} neg={}]", label, mn, mx, avg, neg);
        }
        return std::format("{}[{:.4g}..{:.4g} avg={:.4g}]", label, mn, mx, avg);
    };

    std::ostringstream report;
    auto check_list = [&](const std::string &prefix, const auto &vec)
    {
        for (size_t i = 0; i < vec.size(); ++i)
        {
            auto *bn = dynamic_cast<Batch_Norm_2d_Layer*>(vec[i].get());
            if (bn)
            {
                auto rm = bn->getRunningMean().getData();
                auto rv = bn->getRunningVariance().getData();
                auto rm_s = summarize(rm, false);
                auto rv_s = summarize(rv, true);
                auto bm = bn->getBatchMean().getData();
                auto bv = bn->getBatchVariance().getData();
                report << prefix << "[" << i << "] (C=" << rm.size() << "): "
                       << fmt_stat("RM", rm, rm_s) << ", "
                       << fmt_stat("RV", rv, rv_s);
                if (!bm.empty())
                {
                    report << " | " << fmt_stat("BM", bm, summarize(bm, false));
                }
                if (!bv.empty())
                {
                    report << (bm.empty() ? " | " : ", ") << fmt_stat("BV", bv, summarize(bv, true));
                }
                const size_t rm_nonfin = std::get<1>(rm_s);
                const size_t rv_nonfin = std::get<1>(rv_s);
                const size_t rv_neg = std::get<2>(rv_s);
                if (rm_nonfin > 0 || rv_nonfin > 0 || rv_neg > 0)
                {
                    report << " (NonFinite: RM=" << rm_nonfin << ", RV=" << rv_nonfin << ", RV<0=" << rv_neg << ")";
                }
                report << "\n";
            }
            auto *conv = dynamic_cast<Conv2d_Layer*>(vec[i].get());
            if (conv)
            {
                auto w = conv->getWeights().getData();
                auto b = conv->getBiases().getData();
                size_t w_bad = 0, b_bad = 0;
                for (float v : w) if (!std::isfinite(v)) ++w_bad;
                for (float v : b) if (!std::isfinite(v)) ++b_bad;
                if (w_bad > 0 || b_bad > 0)
                {
                    report << prefix << "[" << i << "] CONV2D NonFinite! W: " << w_bad << "/" << w.size()
                           << ", B: " << b_bad << "/" << b.size() << "\n";
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
        auto w_s = summarize(w, false);
        const size_t w_fin = std::get<0>(w_s);
        const size_t w_bad = std::get<1>(w_s);
        const double w_mean = std::get<5>(w_s);
        report << "final_conv: bias=" << (b.empty() ? 0.0f : b[0])
               << ", w_mean=" << (w_fin > 0 ? std::format("{:.6g}", w_mean) : std::string("none"))
               << (w_bad > 0 ? std::format(" (nonfinite={})", w_bad) : std::string(""))
               << "\n";
    }

    const std::string out = report.str();
    if (!out.empty())
    {
        Logger::info(out, Log_Feature::LAYER_INSPECTION);
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
                for (float v : rm) if (!std::isfinite(v)) return true;
                for (float v : rv) if (!std::isfinite(v) || v < 0.0f) return true;
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

#include "layer/nn_unet_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
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
#include "math/tensor_impl.h"

namespace
{
static uint32_t validateNnUNetIoTensor(
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
        throw std::invalid_argument(std::format("NN_UNet_Layer::{}: Tensor has null implementation", fn_name));
    }
    if (t.getExecutionTarget() != expected_target)
    {
        throw std::invalid_argument(std::format("NN_UNet_Layer::{}: Execution_Target mismatch", fn_name));
    }
    if (t.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::format("NN_UNet_Layer::{}: TERNARY_INT2 tensor is not supported", fn_name));
    }
    if (t.isEmpty())
    {
        throw std::invalid_argument(std::format("NN_UNet_Layer::{}: Tensor is empty", fn_name));
    }
    const Shape &sh = t.getShape();
    const size_t rank = sh.getRank();
    if (rank != 2 && rank != 4)
    {
        throw std::invalid_argument(std::format("NN_UNet_Layer::{}: Expected rank-2 or rank-4 tensor, got rank {}", fn_name, rank));
    }
    const size_t batch = sh[0];
    if (batch == 0 || batch > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(std::format("NN_UNet_Layer::{}: Invalid batch size {}", fn_name, batch));
    }
    if (expected_batch && batch != *expected_batch)
    {
        throw std::invalid_argument(std::format(
            "NN_UNet_Layer::{}: Batch size {} mismatch (expected {})", fn_name, batch, *expected_batch));
    }
    const size_t expected_cols = static_cast<size_t>(expected_channels) * expected_height * expected_width;
    if (rank == 2)
    {
        if (sh[1] != expected_cols)
        {
            throw std::invalid_argument(std::format(
                "NN_UNet_Layer::{}: Rank-2 column count {} mismatch (expected {})", fn_name, sh[1], expected_cols));
        }
    }
    else
    {
        if (sh[1] != expected_channels || sh[2] != expected_height || sh[3] != expected_width)
        {
            throw std::invalid_argument(std::format(
                "NN_UNet_Layer::{}: Rank-4 shape must be [B, {}, {}, {}]",
                fn_name, expected_channels, expected_height, expected_width));
        }
    }
    return static_cast<uint32_t>(batch);
}

static const Tensor &addTensorsPersistent(const Tensor &a, const Tensor &b, Tensor &b_cast_scratch, Tensor &out_sum_scratch)
{
    const Tensor *b_eff = &b;
    if (b.getDataType() != a.getDataType())
    {
        b.to(a.getDataType(), b_cast_scratch);
        b_eff = &b_cast_scratch;
    }
    a.add(*b_eff, out_sum_scratch);
    return out_sum_scratch;
}
}

NN_UNet_Layer::NN_UNet_Layer(
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
      enable_deep_supervision(true),
      ds_forward_executed(false),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      skip1_act(0, 0, _execution_target),
      skip2_act(0, 0, _execution_target),
      skip3_act(0, 0, _execution_target),
      skip4_act(0, 0, _execution_target),
      dec4_concat(0, 0, _execution_target),
      dec3_concat(0, 0, _execution_target),
      dec2_concat(0, 0, _execution_target),
      dec1_concat(0, 0, _execution_target),
      ds3_output(0, 0, _execution_target),
      ds2_output(0, 0, _execution_target),
      ds3_grad(0, 0, _execution_target),
      ds2_grad(0, 0, _execution_target),
      res_enc1_act(0, 0, _execution_target),
      res_enc2_act(0, 0, _execution_target),
      res_enc3_act(0, 0, _execution_target),
      res_enc4_act(0, 0, _execution_target),
      res_bn_act(0, 0, _execution_target),
      res_dec4_act(0, 0, _execution_target),
      res_dec3_act(0, 0, _execution_target),
      res_dec2_act(0, 0, _execution_target),
      res_dec1_act(0, 0, _execution_target),
      in_nhwc_scratch(0, 0, _execution_target),
      out_nchw_scratch(0, 0, _execution_target),
      ds2_out_nchw_scratch(0, 0, _execution_target),
      ds3_out_nchw_scratch(0, 0, _execution_target),
      grad_nhwc_scratch(0, 0, _execution_target),
      in_grad_nchw_scratch(0, 0, _execution_target),
      ds2_grad_nhwc_scratch(0, 0, _execution_target),
      ds3_grad_nhwc_scratch(0, 0, _execution_target),
      sig_grad_cast_scratch(0, 0, _execution_target),
      ds2_sig_grad_cast(0, 0, _execution_target),
      ds3_sig_grad_cast(0, 0, _execution_target),
      ds2_add_cast(0, 0, _execution_target),
      ds2_add_sum(0, 0, _execution_target),
      ds3_add_cast(0, 0, _execution_target),
      ds3_add_sum(0, 0, _execution_target),
      up4_cast_scratch(0, 0, _execution_target),
      up3_cast_scratch(0, 0, _execution_target),
      up2_cast_scratch(0, 0, _execution_target),
      up1_cast_scratch(0, 0, _execution_target),
      dec4_concat_2d_scratch(0, 0, _execution_target),
      dec3_concat_2d_scratch(0, 0, _execution_target),
      dec2_concat_2d_scratch(0, 0, _execution_target),
      dec1_concat_2d_scratch(0, 0, _execution_target),
      d_up4_scratch(0, 0, _execution_target),
      d_skip4_scratch(0, 0, _execution_target),
      d_skip4_cast_scratch(0, 0, _execution_target),
      enc4_grad_sum(0, 0, _execution_target),
      d_up3_scratch(0, 0, _execution_target),
      d_skip3_scratch(0, 0, _execution_target),
      d_skip3_cast_scratch(0, 0, _execution_target),
      enc3_grad_sum(0, 0, _execution_target),
      d_up2_scratch(0, 0, _execution_target),
      d_skip2_scratch(0, 0, _execution_target),
      d_skip2_cast_scratch(0, 0, _execution_target),
      enc2_grad_sum(0, 0, _execution_target),
      d_up1_scratch(0, 0, _execution_target),
      d_skip1_scratch(0, 0, _execution_target),
      d_skip1_cast_scratch(0, 0, _execution_target),
      enc1_grad_sum(0, 0, _execution_target)
{
    if (in_channels == 0 || out_channels == 0 || base_channels == 0 ||
        input_height < 16 || input_width < 16 || (input_height % 16) != 0 || (input_width % 16) != 0)
    {
        throw std::invalid_argument("NN_UNet_Layer: height and width must be >= 16 and divisible by 16, and channels > 0");
    }
    if (base_channels > std::numeric_limits<uint32_t>::max() / 16)
    {
        throw std::overflow_error("NN_UNet_Layer: base_channels * 16 exceeds uint32_t limit");
    }
    for (int i = 0; i < 9; ++i)
    {
        res_fwd_cast[i] = Tensor(0, 0, _execution_target);
        res_fwd_sum[i] = Tensor(0, 0, _execution_target);
        res_bwd_cast[i] = Tensor(0, 0, _execution_target);
        res_bwd_sum[i] = Tensor(0, 0, _execution_target);
    }
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

    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, in_channels, c1, 3, 1, 1, execution_target, data_type));
    enc1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    enc1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target, data_type));
    enc1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    enc1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool1 = std::make_unique<Max_Pool_2d_Layer>(h, w, c1, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c2, 3, 1, 1, execution_target, data_type));
    enc2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    enc2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target, data_type));
    enc2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    enc2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool2 = std::make_unique<Max_Pool_2d_Layer>(h, w, c2, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c3, 3, 1, 1, execution_target, data_type));
    enc3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    enc3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target, data_type));
    enc3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    enc3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool3 = std::make_unique<Max_Pool_2d_Layer>(h, w, c3, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    enc4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c4, 3, 1, 1, execution_target, data_type));
    enc4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    enc4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    enc4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c4, 3, 1, 1, execution_target, data_type));
    enc4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    enc4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    pool4 = std::make_unique<Max_Pool_2d_Layer>(h, w, c4, 2, 2, 0, execution_target);

    h /= 2; w /= 2;
    const bool use_bottleneck_inorm = (static_cast<uint64_t>(h) * w > 1);
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c5, 3, 1, 1, execution_target, data_type));
    bottleneck_layers.push_back(use_bottleneck_inorm
        ? std::make_unique<Instance_Norm_2d_Layer>(h, w, c5, 1e-4f, execution_target)
        : nullptr);
    bottleneck_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    bottleneck_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c5, c5, 3, 1, 1, execution_target, data_type));
    bottleneck_layers.push_back(use_bottleneck_inorm
        ? std::make_unique<Instance_Norm_2d_Layer>(h, w, c5, 1e-4f, execution_target)
        : nullptr);
    bottleneck_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    up4 = std::make_unique<Up_Sample_2d_Layer>(h, w, c5, 2, execution_target);
    h *= 2; w *= 2;
    dec4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c5 + c4, c4, 3, 1, 1, execution_target, data_type));
    dec4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    dec4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec4_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4, c4, 3, 1, 1, execution_target, data_type));
    dec4_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c4, 1e-4f, execution_target));
    dec4_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    up3 = std::make_unique<Up_Sample_2d_Layer>(h, w, c4, 2, execution_target);
    h *= 2; w *= 2;
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c4 + c3, c3, 3, 1, 1, execution_target, data_type));
    dec3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    dec3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec3_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3, c3, 3, 1, 1, execution_target, data_type));
    dec3_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c3, 1e-4f, execution_target));
    dec3_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    ds3_conv = std::make_unique<Conv2d_Layer>(h, w, c3, out_channels, 1, 1, 0, execution_target, data_type);
    ds3_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);

    up2 = std::make_unique<Up_Sample_2d_Layer>(h, w, c3, 2, execution_target);
    h *= 2; w *= 2;
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c3 + c2, c2, 3, 1, 1, execution_target, data_type));
    dec2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    dec2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec2_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2, c2, 3, 1, 1, execution_target, data_type));
    dec2_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c2, 1e-4f, execution_target));
    dec2_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    ds2_conv = std::make_unique<Conv2d_Layer>(h, w, c2, out_channels, 1, 1, 0, execution_target, data_type);
    ds2_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);

    up1 = std::make_unique<Up_Sample_2d_Layer>(h, w, c2, 2, execution_target);
    h *= 2; w *= 2;
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c2 + c1, c1, 3, 1, 1, execution_target, data_type));
    dec1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    dec1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));
    dec1_layers.push_back(std::make_unique<Conv2d_Layer>(h, w, c1, c1, 3, 1, 1, execution_target, data_type));
    dec1_layers.push_back(std::make_unique<Instance_Norm_2d_Layer>(h, w, c1, 1e-4f, execution_target));
    dec1_layers.push_back(std::make_unique<Leaky_Relu_Layer>(0.01f, execution_target));

    final_conv = std::make_unique<Conv2d_Layer>(h, w, c1, out_channels, 1, 1, 0, execution_target, data_type);
    final_sigmoid = std::make_unique<Sigmoid_Layer>(execution_target);
}

void NN_UNet_Layer::prepareInference()
{
    auto prep_vec = [](auto &vec) {
        for (auto &l : vec) { if (l) l->prepareInference(); }
    };
    prep_vec(enc1_layers);
    if (pool1) pool1->prepareInference();
    prep_vec(enc2_layers);
    if (pool2) pool2->prepareInference();
    prep_vec(enc3_layers);
    if (pool3) pool3->prepareInference();
    prep_vec(enc4_layers);
    if (pool4) pool4->prepareInference();
    prep_vec(bottleneck_layers);
    if (up4) up4->prepareInference();
    prep_vec(dec4_layers);
    if (up3) up3->prepareInference();
    prep_vec(dec3_layers);
    if (ds3_conv) ds3_conv->prepareInference();
    if (ds3_sigmoid) ds3_sigmoid->prepareInference();
    if (up2) up2->prepareInference();
    prep_vec(dec2_layers);
    if (ds2_conv) ds2_conv->prepareInference();
    if (ds2_sigmoid) ds2_sigmoid->prepareInference();
    if (up1) up1->prepareInference();
    prep_vec(dec1_layers);
    if (final_conv) final_conv->prepareInference();
    if (final_sigmoid) final_sigmoid->prepareInference();
}

void NN_UNet_Layer::quantizeToTernary()
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
    quant_vec(enc4_layers);
    quant_vec(bottleneck_layers);
    quant_vec(dec4_layers);
    quant_vec(dec3_layers);
    quant_vec(dec2_layers);
    quant_vec(dec1_layers);
    if (ds3_conv) ds3_conv->quantizeToTernary();
    if (ds2_conv) ds2_conv->quantizeToTernary();
    if (final_conv) final_conv->quantizeToTernary();
}

void NN_UNet_Layer::dequantizeFromTernary()
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
    dequant_vec(enc4_layers);
    dequant_vec(bottleneck_layers);
    dequant_vec(dec4_layers);
    dequant_vec(dec3_layers);
    dequant_vec(dec2_layers);
    dequant_vec(dec1_layers);
    if (ds3_conv) ds3_conv->dequantizeFromTernary();
    if (ds2_conv) ds2_conv->dequantizeFromTernary();
    if (final_conv) final_conv->dequantizeFromTernary();
}

void NN_UNet_Layer::snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const
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
    snap_vec(enc4_layers);
    snap_vec(bottleneck_layers);
    snap_vec(dec4_layers);
    snap_vec(dec3_layers);
    snap_vec(dec2_layers);
    snap_vec(dec1_layers);
}

void NN_UNet_Layer::restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor)
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
    rest_vec(enc4_layers);
    rest_vec(bottleneck_layers);
    rest_vec(dec4_layers);
    rest_vec(dec3_layers);
    rest_vec(dec2_layers);
    rest_vec(dec1_layers);
}

std::unique_ptr<ILayer> NN_UNet_Layer::clone() const
{
    auto cloned = std::make_unique<NN_UNet_Layer>(input_height, input_width, in_channels, out_channels, base_channels, execution_target, data_type);
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
    clone_vec(enc4_layers, cloned->enc4_layers);
    clone_as(pool4, cloned->pool4);
    clone_vec(bottleneck_layers, cloned->bottleneck_layers);
    clone_as(up4, cloned->up4);
    clone_vec(dec4_layers, cloned->dec4_layers);
    clone_as(up3, cloned->up3);
    clone_vec(dec3_layers, cloned->dec3_layers);
    clone_as(ds3_conv, cloned->ds3_conv);
    clone_as(ds3_sigmoid, cloned->ds3_sigmoid);
    clone_as(up2, cloned->up2);
    clone_vec(dec2_layers, cloned->dec2_layers);
    clone_as(ds2_conv, cloned->ds2_conv);
    clone_as(ds2_sigmoid, cloned->ds2_sigmoid);
    clone_as(up1, cloned->up1);
    clone_vec(dec1_layers, cloned->dec1_layers);
    clone_as(final_conv, cloned->final_conv);
    clone_as(final_sigmoid, cloned->final_sigmoid);

    cloned->invalidateWeightCache();
    cloned->setTrainingMode(is_training);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setAccumulated(is_accumulated);
    cloned->setEnableDeepSupervision(enable_deep_supervision);
    return cloned;
}

Tensor NN_UNet_Layer::forward(const Tensor &_input_tensor)
{
    uint32_t b = validateNnUNetIoTensor(
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

    auto forward_res_block = [this](std::vector<std::unique_ptr<ILayer>> &layers, const Tensor &in, Tensor &saved_xa, int idx) -> Tensor {
        Tensor c = in;
        c = layers[0]->forward(c);
        if (layers[1]) c = layers[1]->forward(c);
        c = layers[2]->forward(c);
        saved_xa = c;
        c = layers[3]->forward(c);
        if (layers[4]) c = layers[4]->forward(c);
        c = layers[5]->forward(c);
        return addTensorsPersistent(c, saved_xa, res_fwd_cast[idx], res_fwd_sum[idx]);
    };

    cur = forward_res_block(enc1_layers, cur, res_enc1_act, 0);
    skip1_act = cur;
    cur = pool1->forward(cur);

    cur = forward_res_block(enc2_layers, cur, res_enc2_act, 1);
    skip2_act = cur;
    cur = pool2->forward(cur);

    cur = forward_res_block(enc3_layers, cur, res_enc3_act, 2);
    skip3_act = cur;
    cur = pool3->forward(cur);

    cur = forward_res_block(enc4_layers, cur, res_enc4_act, 3);
    skip4_act = cur;
    cur = pool4->forward(cur);

    cur = forward_res_block(bottleneck_layers, cur, res_bn_act, 4);

    uint32_t c1 = base_channels;
    uint32_t c2 = base_channels * 2;
    uint32_t c3 = base_channels * 4;
    uint32_t c4 = base_channels * 8;
    uint32_t c5 = base_channels * 16;

    const size_t b_sz = b;
    const size_t h4_sz = input_height / 8;
    const size_t w4_sz = input_width / 8;
    const size_t h3_sz = input_height / 4;
    const size_t w3_sz = input_width / 4;
    const size_t h2_sz = input_height / 2;
    const size_t w2_sz = input_width / 2;
    const size_t h1_sz = input_height;
    const size_t w1_sz = input_width;
    uint32_t h3 = static_cast<uint32_t>(h3_sz);
    uint32_t w3 = static_cast<uint32_t>(w3_sz);
    uint32_t h2 = static_cast<uint32_t>(h2_sz);
    uint32_t w2 = static_cast<uint32_t>(w2_sz);

    Tensor up_feat4 = up4->forward(cur);
    if (up_feat4.getDataType() != skip4_act.getDataType())
    {
        up_feat4.to(skip4_act.getDataType(), up4_cast_scratch);
        up_feat4 = up4_cast_scratch;
    }
    up_feat4.view(Shape{b_sz * h4_sz * w4_sz, c5}).concatenateColumns(skip4_act.view(Shape{b_sz * h4_sz * w4_sz, c4}), dec4_concat_2d_scratch);
    dec4_concat = dec4_concat_2d_scratch.view(Shape{b_sz, h4_sz * w4_sz * (static_cast<size_t>(c5) + c4)});
    cur = dec4_concat;
    cur = forward_res_block(dec4_layers, cur, res_dec4_act, 5);

    Tensor up_feat3 = up3->forward(cur);
    if (up_feat3.getDataType() != skip3_act.getDataType())
    {
        up_feat3.to(skip3_act.getDataType(), up3_cast_scratch);
        up_feat3 = up3_cast_scratch;
    }
    up_feat3.view(Shape{b_sz * h3_sz * w3_sz, c4}).concatenateColumns(skip3_act.view(Shape{b_sz * h3_sz * w3_sz, c3}), dec3_concat_2d_scratch);
    dec3_concat = dec3_concat_2d_scratch.view(Shape{b_sz, h3_sz * w3_sz * (static_cast<size_t>(c4) + c3)});
    cur = dec3_concat;
    cur = forward_res_block(dec3_layers, cur, res_dec3_act, 6);

    ds_forward_executed = (enable_deep_supervision && is_training);

    if (ds_forward_executed)
    {
        Tensor ds3_before = ds3_conv->forward(cur);
        Tensor ds3_sig_out = ds3_sigmoid->forward(ds3_before);
        if (_input_tensor.getShape().getRank() == 4)
        {
            if (out_channels > 1)
            {
                ds3_sig_out.view(Shape{b, h3, w3, out_channels}).permute({0, 3, 1, 2}).contiguous(ds3_out_nchw_scratch);
                ds3_output = ds3_out_nchw_scratch;
            }
            else
            {
                ds3_output = ds3_sig_out.view(Shape{b, out_channels, h3, w3});
            }
        }
        else
        {
            ds3_output = ds3_sig_out;
        }
    }
    else
    {
        ds3_output = Tensor(0, 0, execution_target);
    }

    Tensor up_feat2 = up2->forward(cur);
    if (up_feat2.getDataType() != skip2_act.getDataType())
    {
        up_feat2.to(skip2_act.getDataType(), up2_cast_scratch);
        up_feat2 = up2_cast_scratch;
    }
    up_feat2.view(Shape{b_sz * h2_sz * w2_sz, c3}).concatenateColumns(skip2_act.view(Shape{b_sz * h2_sz * w2_sz, c2}), dec2_concat_2d_scratch);
    dec2_concat = dec2_concat_2d_scratch.view(Shape{b_sz, h2_sz * w2_sz * (static_cast<size_t>(c3) + c2)});
    cur = dec2_concat;
    cur = forward_res_block(dec2_layers, cur, res_dec2_act, 7);

    if (ds_forward_executed)
    {
        Tensor ds2_before = ds2_conv->forward(cur);
        Tensor ds2_sig_out = ds2_sigmoid->forward(ds2_before);
        if (_input_tensor.getShape().getRank() == 4)
        {
            if (out_channels > 1)
            {
                ds2_sig_out.view(Shape{b, h2, w2, out_channels}).permute({0, 3, 1, 2}).contiguous(ds2_out_nchw_scratch);
                ds2_output = ds2_out_nchw_scratch;
            }
            else
            {
                ds2_output = ds2_sig_out.view(Shape{b, out_channels, h2, w2});
            }
        }
        else
        {
            ds2_output = ds2_sig_out;
        }
    }
    else
    {
        ds2_output = Tensor(0, 0, execution_target);
    }

    Tensor up_feat1 = up1->forward(cur);
    if (up_feat1.getDataType() != skip1_act.getDataType())
    {
        up_feat1.to(skip1_act.getDataType(), up1_cast_scratch);
        up_feat1 = up1_cast_scratch;
    }
    up_feat1.view(Shape{b_sz * h1_sz * w1_sz, c2}).concatenateColumns(skip1_act.view(Shape{b_sz * h1_sz * w1_sz, c1}), dec1_concat_2d_scratch);
    dec1_concat = dec1_concat_2d_scratch.view(Shape{b_sz, h1_sz * w1_sz * (static_cast<size_t>(c2) + c1)});
    cur = dec1_concat;
    cur = forward_res_block(dec1_layers, cur, res_dec1_act, 8);

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

Tensor NN_UNet_Layer::backward(const Tensor &_output_gradient)
{
    if (!is_forward_completed)
    {
        throw std::runtime_error("NN_UNet_Layer::backward: Forward must be called before backward");
    }
    const size_t fwd_batch = (input_tensor.getShape().getRank() >= 1) ? input_tensor.getShape()[0] : 0;
    uint32_t b = validateNnUNetIoTensor(
        _output_gradient, execution_target, out_channels, input_height, input_width, "backward", &fwd_batch);
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
    uint32_t c5 = base_channels * 16;

    uint32_t h1 = input_height;
    uint32_t w1 = input_width;
    uint32_t h2 = input_height / 2;
    uint32_t w2 = input_width / 2;
    uint32_t h3 = input_height / 4;
    uint32_t w3 = input_width / 4;
    uint32_t h4 = input_height / 8;
    uint32_t w4 = input_width / 8;

    const size_t b_sz = b;
    const size_t c1_sz = c1;
    const size_t c2_sz = c2;
    const size_t c3_sz = c3;
    const size_t c4_sz = c4;
    const size_t c5_sz = c5;
    const size_t h1_sz = h1;
    const size_t w1_sz = w1;
    const size_t h2_sz = h2;
    const size_t w2_sz = w2;
    const size_t h3_sz = h3;
    const size_t w3_sz = w3;
    const size_t h4_sz = h4;
    const size_t w4_sz = w4;

    auto backward_res_block = [this](std::vector<std::unique_ptr<ILayer>> &layers, const Tensor &grad_sum, int idx) -> Tensor {
        Tensor cur_grad = grad_sum;
        cur_grad = layers[5]->backward(cur_grad);
        if (layers[4]) cur_grad = layers[4]->backward(cur_grad);
        cur_grad = layers[3]->backward(cur_grad);
        Tensor grad_xa = addTensorsPersistent(cur_grad, grad_sum, res_bwd_cast[idx], res_bwd_sum[idx]);
        grad_xa = layers[2]->backward(grad_xa);
        if (layers[1]) grad_xa = layers[1]->backward(grad_xa);
        return layers[0]->backward(grad_xa);
    };

    grad = final_conv->backward(grad);

    grad = backward_res_block(dec1_layers, grad, 8);
    grad.view(Shape{b_sz * h1_sz * w1_sz, c2_sz + c1_sz}).splitColumns(c2, d_up1_scratch, d_skip1_scratch);
    Tensor d_up1 = d_up1_scratch.view(Shape{b_sz, h1_sz * w1_sz * c2_sz});
    Tensor d_skip1 = d_skip1_scratch.view(Shape{b_sz, h1_sz * w1_sz * c1_sz});
    grad = up1->backward(d_up1);

    if (ds_forward_executed && !ds2_grad.isEmpty())
    {
        validateNnUNetIoTensor(ds2_grad, execution_target, out_channels, h2, w2, "backward(ds2_grad)", &b_sz);
        Tensor ds2_g = ds2_grad;
        if (ds2_g.getShape().getRank() == 4)
        {
            if (out_channels > 1)
            {
                ds2_g.permute({0, 2, 3, 1}).contiguous(ds2_grad_nhwc_scratch);
                ds2_g = ds2_grad_nhwc_scratch.view(Shape{b, static_cast<size_t>(h2) * w2 * out_channels});
            }
            else
            {
                ds2_g = ds2_g.view(Shape{b, static_cast<size_t>(h2) * w2});
            }
        }
        Data_Type ds2_dtype = ds2_sigmoid->getOutput().getDataType();
        if (ds2_g.getDataType() != ds2_dtype)
        {
            ds2_g.to(ds2_dtype, ds2_sig_grad_cast);
            ds2_g = ds2_sig_grad_cast;
        }
        Tensor d_ds2 = ds2_sigmoid->backward(ds2_g);
        d_ds2 = ds2_conv->backward(d_ds2);
        grad = addTensorsPersistent(grad, d_ds2, ds2_add_cast, ds2_add_sum);
        ds2_grad = Tensor(0, 0, execution_target);
    }

    grad = backward_res_block(dec2_layers, grad, 7);
    grad.view(Shape{b_sz * h2_sz * w2_sz, c3_sz + c2_sz}).splitColumns(c3, d_up2_scratch, d_skip2_scratch);
    Tensor d_up2 = d_up2_scratch.view(Shape{b_sz, h2_sz * w2_sz * c3_sz});
    Tensor d_skip2 = d_skip2_scratch.view(Shape{b_sz, h2_sz * w2_sz * c2_sz});
    grad = up2->backward(d_up2);

    if (ds_forward_executed && !ds3_grad.isEmpty())
    {
        validateNnUNetIoTensor(ds3_grad, execution_target, out_channels, h3, w3, "backward(ds3_grad)", &b_sz);
        Tensor ds3_g = ds3_grad;
        if (ds3_g.getShape().getRank() == 4)
        {
            if (out_channels > 1)
            {
                ds3_g.permute({0, 2, 3, 1}).contiguous(ds3_grad_nhwc_scratch);
                ds3_g = ds3_grad_nhwc_scratch.view(Shape{b, static_cast<size_t>(h3) * w3 * out_channels});
            }
            else
            {
                ds3_g = ds3_g.view(Shape{b, static_cast<size_t>(h3) * w3});
            }
        }
        Data_Type ds3_dtype = ds3_sigmoid->getOutput().getDataType();
        if (ds3_g.getDataType() != ds3_dtype)
        {
            ds3_g.to(ds3_dtype, ds3_sig_grad_cast);
            ds3_g = ds3_sig_grad_cast;
        }
        Tensor d_ds3 = ds3_sigmoid->backward(ds3_g);
        d_ds3 = ds3_conv->backward(d_ds3);
        grad = addTensorsPersistent(grad, d_ds3, ds3_add_cast, ds3_add_sum);
        ds3_grad = Tensor(0, 0, execution_target);
    }

    grad = backward_res_block(dec3_layers, grad, 6);
    grad.view(Shape{b_sz * h3_sz * w3_sz, c4_sz + c3_sz}).splitColumns(c4, d_up3_scratch, d_skip3_scratch);
    Tensor d_up3 = d_up3_scratch.view(Shape{b_sz, h3_sz * w3_sz * c4_sz});
    Tensor d_skip3 = d_skip3_scratch.view(Shape{b_sz, h3_sz * w3_sz * c3_sz});
    grad = up3->backward(d_up3);

    grad = backward_res_block(dec4_layers, grad, 5);
    grad.view(Shape{b_sz * h4_sz * w4_sz, c5_sz + c4_sz}).splitColumns(c5, d_up4_scratch, d_skip4_scratch);
    Tensor d_up4 = d_up4_scratch.view(Shape{b_sz, h4_sz * w4_sz * c5_sz});
    Tensor d_skip4 = d_skip4_scratch.view(Shape{b_sz, h4_sz * w4_sz * c4_sz});
    grad = up4->backward(d_up4);

    grad = backward_res_block(bottleneck_layers, grad, 4);

    grad = pool4->backward(grad);
    grad = addTensorsPersistent(grad, d_skip4, d_skip4_cast_scratch, enc4_grad_sum);
    grad = backward_res_block(enc4_layers, grad, 3);

    grad = pool3->backward(grad);
    grad = addTensorsPersistent(grad, d_skip3, d_skip3_cast_scratch, enc3_grad_sum);
    grad = backward_res_block(enc3_layers, grad, 2);

    grad = pool2->backward(grad);
    grad = addTensorsPersistent(grad, d_skip2, d_skip2_cast_scratch, enc2_grad_sum);
    grad = backward_res_block(enc2_layers, grad, 1);

    grad = pool1->backward(grad);
    grad = addTensorsPersistent(grad, d_skip1, d_skip1_cast_scratch, enc1_grad_sum);
    grad = backward_res_block(enc1_layers, grad, 0);

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

void NN_UNet_Layer::resetGradient()
{
    auto reset_vec = [](auto &vec) {
        for (auto &l : vec) if (l) l->resetGradient();
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
    ds_forward_executed = false;
}

void NN_UNet_Layer::resetGradients()
{
    is_forward_completed = false;
    ds_forward_executed = false;
    auto reset_vec = [](auto &vec) {
        for (auto &l : vec) if (l) l->resetGradients();
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

std::vector<const Tensor *> NN_UNet_Layer::getNoDecayParameters() const
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
    append_vec(enc4_layers);
    append_vec(bottleneck_layers);
    append_vec(dec4_layers);
    append_vec(dec3_layers);
    append_nd(ds3_conv);
    append_vec(dec2_layers);
    append_nd(ds2_conv);
    append_vec(dec1_layers);
    append_nd(final_conv);
    return no_decay;
}

void NN_UNet_Layer::saveConfiguration(std::ofstream &_out) const
{
    if (!_out.is_open() || !_out.good())
    {
        throw std::runtime_error("NN_UNet_Layer::saveConfiguration: Output stream is not open or writable");
    }
    const uint32_t encoded_bc = base_channels | (enable_deep_supervision ? 0u : NN_UNET_DS_DISABLED_FLAG);
    _out.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _out.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _out.write(reinterpret_cast<const char *>(&in_channels), sizeof(in_channels));
    _out.write(reinterpret_cast<const char *>(&out_channels), sizeof(out_channels));
    _out.write(reinterpret_cast<const char *>(&encoded_bc), sizeof(encoded_bc));
    if (!_out.good())
    {
        throw std::runtime_error("NN_UNet_Layer::saveConfiguration: Failed writing configuration");
    }
}

void NN_UNet_Layer::saveInference(std::ofstream &_out) const
{
    saveConfiguration(_out);
    auto save_vec = [&](const auto &vec) {
        for (const auto &l : vec) if (l) l->saveInference(_out);
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
    if (!_in.is_open() || !_in.good())
    {
        throw std::runtime_error("NN_UNet_Layer::loadInference: Input stream is not open or readable");
    }
    uint32_t h = 0, w = 0, ic = 0, oc = 0, raw_bc = 0;
    _in.read(reinterpret_cast<char *>(&h), sizeof(h));
    _in.read(reinterpret_cast<char *>(&w), sizeof(w));
    _in.read(reinterpret_cast<char *>(&ic), sizeof(ic));
    _in.read(reinterpret_cast<char *>(&oc), sizeof(oc));
    _in.read(reinterpret_cast<char *>(&raw_bc), sizeof(raw_bc));
    const bool ds_enabled = (raw_bc & NN_UNET_DS_DISABLED_FLAG) == 0u;
    const uint32_t bc = raw_bc & ~NN_UNET_DS_DISABLED_FLAG;
    if (!_in.good() || ic == 0 || oc == 0 || bc == 0 || bc > (std::numeric_limits<uint32_t>::max() / 16) || h < 16 || w < 16 || (h % 16) != 0 || (w % 16) != 0)
    {
        throw std::runtime_error("NN_UNet_Layer::loadInference: Invalid or corrupted nnU-Net configuration header");
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
    setEnableDeepSupervision(ds_enabled);

    auto load_vec = [&](auto &vec) {
        for (auto &l : vec) if (l) l->loadInference(_in);
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
    if (final_conv) data_type = final_conv->getDataType();
    resetGradient();
    invalidateWeightCache();
}

void NN_UNet_Layer::saveCheckpoint(std::ofstream &_out) const
{
    saveConfiguration(_out);
    auto save_vec = [&](const auto &vec) {
        for (const auto &l : vec) if (l) l->saveCheckpoint(_out);
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
    if (!_in.is_open() || !_in.good())
    {
        throw std::runtime_error("NN_UNet_Layer::loadCheckpoint: Input stream is not open or readable");
    }
    uint32_t h = 0, w = 0, ic = 0, oc = 0, raw_bc = 0;
    _in.read(reinterpret_cast<char *>(&h), sizeof(h));
    _in.read(reinterpret_cast<char *>(&w), sizeof(w));
    _in.read(reinterpret_cast<char *>(&ic), sizeof(ic));
    _in.read(reinterpret_cast<char *>(&oc), sizeof(oc));
    _in.read(reinterpret_cast<char *>(&raw_bc), sizeof(raw_bc));
    const bool ds_enabled = (raw_bc & NN_UNET_DS_DISABLED_FLAG) == 0u;
    const uint32_t bc = raw_bc & ~NN_UNET_DS_DISABLED_FLAG;
    if (!_in.good() || ic == 0 || oc == 0 || bc == 0 || bc > (std::numeric_limits<uint32_t>::max() / 16) || h < 16 || w < 16 || (h % 16) != 0 || (w % 16) != 0)
    {
        throw std::runtime_error("NN_UNet_Layer::loadCheckpoint: Invalid or corrupted nnU-Net configuration header");
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
    setEnableDeepSupervision(ds_enabled);

    auto load_vec = [&](auto &vec) {
        for (auto &l : vec) if (l) l->loadCheckpoint(_in);
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
    if (final_conv) data_type = final_conv->getDataType();
    resetGradient();
    invalidateWeightCache();
}

void NN_UNet_Layer::setEnableDeepSupervision(bool _enable)
{
    enable_deep_supervision = _enable;
    if (!_enable)
    {
        ds_forward_executed = false;
        clearDeepSupervisionGradients();
        if (ds2_conv) ds2_conv->resetGradients();
        if (ds3_conv) ds3_conv->resetGradients();
    }
}

void NN_UNet_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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
    skip1_act.setExecutionTarget(_new_execution_target);
    skip2_act.setExecutionTarget(_new_execution_target);
    skip3_act.setExecutionTarget(_new_execution_target);
    skip4_act.setExecutionTarget(_new_execution_target);
    dec4_concat.setExecutionTarget(_new_execution_target);
    dec3_concat.setExecutionTarget(_new_execution_target);
    dec2_concat.setExecutionTarget(_new_execution_target);
    dec1_concat.setExecutionTarget(_new_execution_target);
    ds3_output.setExecutionTarget(_new_execution_target);
    ds2_output.setExecutionTarget(_new_execution_target);
    ds3_grad.setExecutionTarget(_new_execution_target);
    ds2_grad.setExecutionTarget(_new_execution_target);
    res_enc1_act.setExecutionTarget(_new_execution_target);
    res_enc2_act.setExecutionTarget(_new_execution_target);
    res_enc3_act.setExecutionTarget(_new_execution_target);
    res_enc4_act.setExecutionTarget(_new_execution_target);
    res_bn_act.setExecutionTarget(_new_execution_target);
    res_dec4_act.setExecutionTarget(_new_execution_target);
    res_dec3_act.setExecutionTarget(_new_execution_target);
    res_dec2_act.setExecutionTarget(_new_execution_target);
    res_dec1_act.setExecutionTarget(_new_execution_target);

    in_nhwc_scratch = Tensor(0, 0, _new_execution_target);
    out_nchw_scratch = Tensor(0, 0, _new_execution_target);
    ds2_out_nchw_scratch = Tensor(0, 0, _new_execution_target);
    ds3_out_nchw_scratch = Tensor(0, 0, _new_execution_target);
    grad_nhwc_scratch = Tensor(0, 0, _new_execution_target);
    in_grad_nchw_scratch = Tensor(0, 0, _new_execution_target);
    ds2_grad_nhwc_scratch = Tensor(0, 0, _new_execution_target);
    ds3_grad_nhwc_scratch = Tensor(0, 0, _new_execution_target);
    sig_grad_cast_scratch = Tensor(0, 0, _new_execution_target);
    ds2_sig_grad_cast = Tensor(0, 0, _new_execution_target);
    ds3_sig_grad_cast = Tensor(0, 0, _new_execution_target);
    ds2_add_cast = Tensor(0, 0, _new_execution_target);
    ds2_add_sum = Tensor(0, 0, _new_execution_target);
    ds3_add_cast = Tensor(0, 0, _new_execution_target);
    ds3_add_sum = Tensor(0, 0, _new_execution_target);
    up4_cast_scratch = Tensor(0, 0, _new_execution_target);
    up3_cast_scratch = Tensor(0, 0, _new_execution_target);
    up2_cast_scratch = Tensor(0, 0, _new_execution_target);
    up1_cast_scratch = Tensor(0, 0, _new_execution_target);
    dec4_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    dec3_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    dec2_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    dec1_concat_2d_scratch = Tensor(0, 0, _new_execution_target);
    d_up4_scratch = Tensor(0, 0, _new_execution_target);
    d_skip4_scratch = Tensor(0, 0, _new_execution_target);
    d_skip4_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc4_grad_sum = Tensor(0, 0, _new_execution_target);
    d_up3_scratch = Tensor(0, 0, _new_execution_target);
    d_skip3_scratch = Tensor(0, 0, _new_execution_target);
    d_skip3_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc3_grad_sum = Tensor(0, 0, _new_execution_target);
    d_up2_scratch = Tensor(0, 0, _new_execution_target);
    d_skip2_scratch = Tensor(0, 0, _new_execution_target);
    d_skip2_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc2_grad_sum = Tensor(0, 0, _new_execution_target);
    d_up1_scratch = Tensor(0, 0, _new_execution_target);
    d_skip1_scratch = Tensor(0, 0, _new_execution_target);
    d_skip1_cast_scratch = Tensor(0, 0, _new_execution_target);
    enc1_grad_sum = Tensor(0, 0, _new_execution_target);
    for (int i = 0; i < 9; ++i)
    {
        res_fwd_cast[i] = Tensor(0, 0, _new_execution_target);
        res_fwd_sum[i] = Tensor(0, 0, _new_execution_target);
        res_bwd_cast[i] = Tensor(0, 0, _new_execution_target);
        res_bwd_sum[i] = Tensor(0, 0, _new_execution_target);
    }
    invalidateWeightCache();
}

void NN_UNet_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    ILayer::setAccumulated(_is_accumulated);
    auto set_acc = [&](auto &v) { for (auto &l : v) if (l) l->setAccumulated(_is_accumulated); };
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
    ILayer::setTrainingMode(_training);
    auto set_tm = [&](auto &v) { for (auto &l : v) if (l) l->setTrainingMode(_training); };
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
    ILayer::setMixedPrecision(_enable);
    auto set_mp = [&](auto &v) { for (auto &l : v) if (l) l->setMixedPrecision(_enable); };
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
    auto inv = [&](auto &v) { for (auto &l : v) if (l) l->invalidateWeightCache(); };
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

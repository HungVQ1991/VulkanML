#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "math/tensor.h"

namespace layer_internal
{

inline void validateForwardInput(const Tensor &input, Execution_Target expected_target, const char *ctx)
{
    if (!input.getImplementation())
    {
        throw std::invalid_argument(std::string(ctx) + ": Input tensor has null implementation");
    }
    if (input.getExecutionTarget() != expected_target)
    {
        throw std::invalid_argument(std::string(ctx) + ": Execution_Target mismatch");
    }
}

inline void validateBatchedParamsEmpty(const std::vector<Tensor> &params, const char *ctx, const char *layer_name, Log_Feature log_feature)
{
    if (!params.empty())
    {
        Logger::logMessage(Input_Format{"{}: expected 0 batched parameter tensors, got {}", ctx, params.size()},
                           Log_Level::LOG_ERROR, true, 0, log_feature);
        throw std::invalid_argument(std::string("Invalid input params size: ") + layer_name + " expects 0 parameters");
    }
}

inline void validateBackwardInput(
    bool is_forward_completed,
    const Tensor &expected_output,
    const Tensor &output_gradient,
    Execution_Target expected_target,
    const char *ctx,
    const char *before_forward_msg,
    const char *shape_mismatch_msg,
    Log_Feature log_feature)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"{}: {}", ctx, before_forward_msg},
                           Log_Level::LOG_ERROR, true, 0, log_feature);
        throw std::logic_error(before_forward_msg);
    }
    if (!output_gradient.getImplementation())
    {
        throw std::invalid_argument(std::string(ctx) + ": Output gradient has null implementation");
    }
    if (output_gradient.getExecutionTarget() != expected_target)
    {
        throw std::invalid_argument(std::string(ctx) + ": Execution_Target mismatch");
    }
    if (output_gradient.getShape() != expected_output.getShape())
    {
        Logger::logMessage(Input_Format{"{}: {} (grad {} vs expected {})",
                                        ctx, shape_mismatch_msg,
                                        output_gradient.getShape().toString(),
                                        expected_output.getShape().toString()},
                           Log_Level::LOG_ERROR, true, 0, log_feature);
        throw std::invalid_argument(shape_mismatch_msg);
    }
}

template <typename Op>
inline Tensor runStatelessUnaryForward(
    const Tensor &in,
    Tensor &input_tensor,
    Tensor &output_tensor,
    bool &is_forward_completed,
    Execution_Target target,
    const char *ctx,
    Log_Feature log_feature,
    Op &&op)
{
    validateForwardInput(in, target, ctx);
    Logger::logMessage(Input_Format{"{}: rows={}, columns={}", ctx, in.getRows(), in.getColumns()},
                       Log_Level::LOG_DEBUG, true, 1, log_feature);
    input_tensor = in;
    op(input_tensor, output_tensor);
    is_forward_completed = true;
    return output_tensor;
}

template <typename Op>
inline Tensor runStatelessUnaryBatchedForward(
    const Tensor &batched_in,
    const std::vector<Tensor> &batched_params,
    const char *ctx,
    const char *layer_name,
    Log_Feature log_feature,
    Op &&op)
{
    validateBatchedParamsEmpty(batched_params, ctx, layer_name, log_feature);
    Tensor output_result(batched_in.getExecutionTarget());
    op(batched_in, output_result);
    return output_result;
}

template <typename Op>
inline Tensor runStatelessUnaryBackward(
    const Tensor &out_grad,
    const Tensor &saved_tensor,
    const Tensor &cached_output,
    Tensor &input_gradient_tensor,
    bool is_forward_completed,
    Execution_Target target,
    const char *ctx,
    const char *before_forward_msg,
    const char *shape_mismatch_msg,
    Log_Feature log_feature,
    Op &&op)
{
    validateBackwardInput(is_forward_completed, cached_output, out_grad, target, ctx,
                          before_forward_msg, shape_mismatch_msg, log_feature);
    Logger::logMessage(Input_Format{"{}: output_gradient rows={}, columns={}",
                                    ctx, out_grad.getRows(), out_grad.getColumns()},
                       Log_Level::LOG_DEBUG, true, 1, log_feature);
    op(saved_tensor, out_grad, input_gradient_tensor);
    return input_gradient_tensor;
}

template <typename... Tensors>
inline void moveTensorsToTarget(Execution_Target new_target, Tensors &...tensors)
{
    (tensors.setExecutionTarget(new_target), ...);
}

inline std::pair<uint32_t, uint32_t> validatePool2dOutputDims(
    uint32_t height, uint32_t width, uint32_t channels,
    uint32_t kernel_size, uint32_t stride, uint32_t padding,
    const char *layer_name)
{
    if (height == 0 || width == 0 || channels == 0 || kernel_size == 0 || stride == 0)
    {
        Logger::logMessage(Input_Format{"{}: Invalid non-positive parameter (h={}, w={}, c={}, k={}, s={})",
                                        layer_name, height, width, channels, kernel_size, stride},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE);
        throw std::invalid_argument(std::string(layer_name) + ": input dimensions, channels, kernel_size, and stride must be > 0");
    }
    const uint64_t padded_h = static_cast<uint64_t>(height) + 2ULL * static_cast<uint64_t>(padding);
    const uint64_t padded_w = static_cast<uint64_t>(width) + 2ULL * static_cast<uint64_t>(padding);
    if (padded_h < static_cast<uint64_t>(kernel_size) || padded_w < static_cast<uint64_t>(kernel_size))
    {
        Logger::logMessage(Input_Format{"{}: Padded input ({}x{}) smaller than kernel_size ({})",
                                        layer_name, padded_h, padded_w, kernel_size},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE);
        throw std::invalid_argument(std::string(layer_name) + ": Padded input dimensions must be >= kernel_size");
    }
    const uint64_t out_h = (padded_h - static_cast<uint64_t>(kernel_size)) / static_cast<uint64_t>(stride) + 1ULL;
    const uint64_t out_w = (padded_w - static_cast<uint64_t>(kernel_size)) / static_cast<uint64_t>(stride) + 1ULL;
    if (out_h > std::numeric_limits<uint32_t>::max() || out_w > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(std::string(layer_name) + ": Output dimensions exceed uint32_t range");
    }
    const uint64_t in_spatial = static_cast<uint64_t>(height) * static_cast<uint64_t>(width);
    const uint64_t out_spatial = out_h * out_w;
    if (in_spatial > std::numeric_limits<size_t>::max() / channels ||
        out_spatial > std::numeric_limits<size_t>::max() / channels)
    {
        throw std::overflow_error(std::string(layer_name) + ": Per-sample feature size overflows size_t");
    }
    return {static_cast<uint32_t>(out_h), static_cast<uint32_t>(out_w)};
}

inline void validateNhwcSpatialInput(
    const Tensor &input, Execution_Target expected_target,
    uint32_t height, uint32_t width, uint32_t channels, const char *ctx)
{
    if (!input.getImplementation())
    {
        throw std::invalid_argument(std::string(ctx) + ": Input tensor is uninitialized");
    }
    if (input.getExecutionTarget() != expected_target)
    {
        throw std::invalid_argument(std::string(ctx) + ": Execution_Target mismatch");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(ctx) + ": TERNARY_INT2 is not supported");
    }
    const Shape &in_shape = input.getShape();
    const size_t expected_cols = static_cast<size_t>(height) * width * channels;
    if (in_shape.getRank() == 2)
    {
        if (in_shape[1] != expected_cols)
        {
            throw std::invalid_argument(std::string(ctx) + ": Rank-2 input column count does not match H*W*C");
        }
    }
    else if (in_shape.getRank() == 4)
    {
        if (in_shape[1] != height || in_shape[2] != width || in_shape[3] != channels)
        {
            throw std::invalid_argument(std::string(ctx) + ": Rank-4 NHWC input dimensions do not match [N, H, W, C]");
        }
    }
    else
    {
        throw std::invalid_argument(std::string(ctx) + ": Expected rank-2 [N, H*W*C] or rank-4 [N, H, W, C] input");
    }
}

inline void validateNhwcSpatialGrad(
    bool is_forward_completed, const Tensor &input_tensor, const Tensor &out_grad,
    uint32_t exp_h, uint32_t exp_w, uint32_t channels,
    const char *ctx, const char *before_forward_msg)
{
    if (!is_forward_completed)
    {
        throw std::logic_error(before_forward_msg);
    }
    if (!out_grad.getImplementation() || !input_tensor.getImplementation())
    {
        throw std::invalid_argument(std::string(ctx) + ": Uninitialized tensor state");
    }
    if (out_grad.getExecutionTarget() != input_tensor.getExecutionTarget())
    {
        throw std::invalid_argument(std::string(ctx) + ": Execution_Target mismatch");
    }
    if (out_grad.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(ctx) + ": TERNARY_INT2 is not supported");
    }
    const size_t expected_batch = input_tensor.getShape()[0];
    const size_t expected_cols = static_cast<size_t>(exp_h) * exp_w * channels;
    const Shape &grad_shape = out_grad.getShape();
    if (grad_shape.getRank() == 2)
    {
        if (grad_shape[0] != expected_batch || grad_shape[1] != expected_cols)
        {
            throw std::invalid_argument(std::string(ctx) + ": Rank-2 gradient shape does not match output dimensions");
        }
    }
    else if (grad_shape.getRank() == 4)
    {
        if (grad_shape[0] != expected_batch || grad_shape[1] != exp_h ||
            grad_shape[2] != exp_w || grad_shape[3] != channels)
        {
            throw std::invalid_argument(std::string(ctx) + ": Rank-4 NHWC gradient shape does not match output dimensions");
        }
    }
    else
    {
        throw std::invalid_argument(std::string(ctx) + ": Expected rank-2 or rank-4 output gradient");
    }
}

inline void requireOutStream(const std::ofstream &os, const char *ctx)
{
    if (!os.is_open() || !os.good())
    {
        throw std::runtime_error(std::string(ctx) + ": Output stream is not open or writable");
    }
}

template <typename... Ts>
inline void writePods(std::ofstream &os, const char *ctx, const char *fail_msg, const Ts &...vals)
{
    requireOutStream(os, ctx);
    (os.write(reinterpret_cast<const char *>(&vals), sizeof(vals)), ...);
    if (!os.good())
    {
        throw std::runtime_error(std::string(ctx) + ": " + fail_msg);
    }
}

inline void requireInStream(const std::ifstream &is, const char *ctx)
{
    if (!is.is_open())
    {
        throw std::runtime_error(std::string(ctx) + ": Input stream is not open");
    }
}

template <typename... Ts>
inline void readPods(std::ifstream &is, const char *fail_msg, Ts &...vals)
{
    (is.read(reinterpret_cast<char *>(&vals), sizeof(vals)), ...);
    if (!is)
    {
        throw std::runtime_error(fail_msg);
    }
}

}

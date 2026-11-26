#include "ops/activation/activation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::activation
{

using ops::internal::dispatch2D;
using ops::internal::validateU32Count;
using ops::internal::writeCpuOutput;

namespace
{

template <typename CpuFn, typename PipeFn, typename PcFn>
void runElementwiseUnaryActivation(
    const Tensor &input,
    Tensor &output,
    const char *op_name,
    bool log_cpu,
    CpuFn &&cpu_fn,
    PipeFn &&pipe_fn,
    PcFn &&pc_fn)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op_name) + ": TERNARY_INT2 is not supported");
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }
    const bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
    const Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    const size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        out_impl->setDataType(out_dtype);
        output.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        std::vector<float> result(total_elements);
        cpu_fn(in_data, result, total_elements);
        writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*out_impl), input.getShape(), out_dtype, std::move(result), false);
        if (log_cpu)
        {
            Logger::logMessage(Input_Format{"{} (CPU): elements={}", op_name, total_elements},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
        }
        return;
    }

    const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
    auto contig_self = in_gpu.ensureContiguousSelf();
    const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
    const Shape out_shape = in_gpu.getShape();
    auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

    auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
    output_gpu.setDataType(out_dtype);
    output_gpu.reshape(out_shape);

    const uint32_t total_u32 = validateU32Count(total_elements, op_name);
    const auto [wg_x, wg_y] = dispatch2D((total_elements + 255ull) / 256ull);
    const Compute_Pipeline pipe = pipe_fn(is_fp16);
    const auto pc = pc_fn(total_u32);
    in_gpu.pushToGraph(pipe, {buf_self, output_gpu.getStorageBuffer()}, pc, wg_x, wg_y, 1);
    output_gpu.clearHostCache();
    output_gpu.invalidateFp16Cache();
}

template <typename CpuFn, typename PipeFn, typename PcFn>
void runElementwiseBinaryActivationBackward(
    const Tensor &first,
    const Tensor &output_gradient,
    Tensor &input_gradient,
    const char *op_name,
    bool log_cpu,
    CpuFn &&cpu_fn,
    PipeFn &&pipe_fn,
    PcFn &&pc_fn)
{
    auto first_impl = first.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!first_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (output_gradient.getExecutionTarget() != first.getExecutionTarget())
    {
        throw std::invalid_argument(std::string(op_name) + ": Execution_Target mismatch");
    }
    if (first.getDataType() == Data_Type::TERNARY_INT2 || output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op_name) + ": TERNARY_INT2 is not supported");
    }
    if (input_gradient.getExecutionTarget() != first.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(first.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    first_impl->validateSameDimensions(*grad_impl);
    const bool is_fp16 = (first.getDataType() == Data_Type::FLOAT16 || output_gradient.getDataType() == Data_Type::FLOAT16);
    const Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    const size_t total_elements = first.getTotalElements();
    if (total_elements == 0)
    {
        in_grad_impl->setDataType(out_dtype);
        input_gradient.reshape(first.getShape());
        return;
    }

    if (first.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = first.getData();
        const auto &grad_data = output_gradient.getData();
        std::vector<float> result(total_elements);
        cpu_fn(a_data, grad_data, result, total_elements, is_fp16);
        writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*in_grad_impl), first.getShape(), out_dtype, std::move(result), false);
        if (log_cpu)
        {
            Logger::logMessage(Input_Format{"{} (CPU): elements={}", op_name, total_elements},
                               Log_Level::LOG_DEBUG, true, 0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        }
        return;
    }

    const auto &first_gpu = Gpu_Tensor_Impl::castToGpu(*first_impl);
    auto contig_self = first_gpu.ensureContiguousSelf();
    const auto *effective_self = contig_self ? contig_self.get() : &first_gpu;

    const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
    auto contig_grad = grad_gpu.ensureContiguousSelf();
    const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

    const Shape out_shape = first_gpu.getShape();
    auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
    auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

    auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
    in_grad_gpu.setDataType(out_dtype);
    in_grad_gpu.reshape(out_shape);

    const uint32_t total_u32 = validateU32Count(total_elements, op_name);
    const auto [wg_x, wg_y] = dispatch2D((total_elements + 255ull) / 256ull);
    const Compute_Pipeline pipe = pipe_fn(is_fp16);
    const auto pc = pc_fn(total_u32);
    first_gpu.pushToGraph(pipe, {buf_self, buf_grad, in_grad_gpu.getStorageBuffer()}, pc, wg_x, wg_y, 1);
    in_grad_gpu.clearHostCache();
    in_grad_gpu.invalidateFp16Cache();
}

}

void relu(const Tensor &input, Tensor &output)
{
    runElementwiseUnaryActivation(
        input, output, "ops::activation::relu", true,
        [](const std::vector<float> &a, std::vector<float> &res, size_t n) {
            for (size_t i = 0; i < n; ++i)
            {
                res[i] = std::max(0.0F, a[i]);
            }
        },
        [](bool is_fp16) { return is_fp16 ? Compute_Pipeline::RELU_FP16 : Compute_Pipeline::RELU; },
        [](uint32_t total_u32) { return total_u32; });
}

void reluBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient)
{
    runElementwiseBinaryActivationBackward(
        output, output_gradient, input_gradient, "ops::activation::reluBackward", true,
        [](const std::vector<float> &a, const std::vector<float> &g, std::vector<float> &res, size_t n, bool) {
            for (size_t i = 0; i < n; ++i)
            {
                res[i] = (a[i] > 0.0F) ? g[i] : 0.0F;
            }
        },
        [](bool is_fp16) { return is_fp16 ? Compute_Pipeline::RELU_BACKWARD_FP16 : Compute_Pipeline::RELU_BACKWARD; },
        [](uint32_t total_u32) { return total_u32; });
}

void leakyRelu(const Tensor &input, Tensor &output, float negative_slope)
{
    struct PushConstants
    {
        uint32_t total_elements;
        float negative_slope;
    };
    runElementwiseUnaryActivation(
        input, output, "ops::activation::leakyRelu", false,
        [negative_slope](const std::vector<float> &a, std::vector<float> &res, size_t n) {
            for (size_t i = 0; i < n; ++i)
            {
                const float val = a[i];
                res[i] = (val >= 0.0f) ? val : (val * negative_slope);
            }
        },
        [](bool is_fp16) {
            if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled())
            {
                return Compute_Pipeline::LEAKY_RELU_FORWARD_COOPMAT_FP16;
            }
            return is_fp16 ? Compute_Pipeline::LEAKY_RELU_FORWARD_FP16 : Compute_Pipeline::LEAKY_RELU_FORWARD;
        },
        [negative_slope](uint32_t total_u32) { return PushConstants{total_u32, negative_slope}; });
}

void leakyReluBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient, float negative_slope)
{
    struct PushConstants
    {
        uint32_t total_elements;
        float negative_slope;
    };
    runElementwiseBinaryActivationBackward(
        input, output_gradient, input_gradient, "ops::activation::leakyReluBackward", false,
        [negative_slope](const std::vector<float> &a, const std::vector<float> &g, std::vector<float> &res, size_t n, bool) {
            for (size_t i = 0; i < n; ++i)
            {
                res[i] = (a[i] >= 0.0f) ? g[i] : (g[i] * negative_slope);
            }
        },
        [](bool is_fp16) {
            if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled())
            {
                return Compute_Pipeline::LEAKY_RELU_BACKWARD_COOPMAT_FP16;
            }
            return is_fp16 ? Compute_Pipeline::LEAKY_RELU_BACKWARD_FP16 : Compute_Pipeline::LEAKY_RELU_BACKWARD;
        },
        [negative_slope](uint32_t total_u32) { return PushConstants{total_u32, negative_slope}; });
}

void gelu(const Tensor &input, Tensor &output)
{
    runElementwiseUnaryActivation(
        input, output, "ops::activation::gelu", true,
        [](const std::vector<float> &a, std::vector<float> &res, size_t n) {
            constexpr float sqrt_2_over_pi = 0.7978845608F;
            constexpr float coeff = 0.044715F;
            for (size_t i = 0; i < n; ++i)
            {
                const float x = a[i];
                const float cube = x * x * x;
                const float inner = std::clamp(sqrt_2_over_pi * (x + coeff * cube), -15.0F, 15.0F);
                res[i] = 0.5F * x * (1.0F + std::tanh(inner));
            }
        },
        [](bool is_fp16) { return is_fp16 ? Compute_Pipeline::GELU_FP16 : Compute_Pipeline::GELU; },
        [](uint32_t total_u32) { return total_u32; });
}

void geluBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient)
{
    runElementwiseBinaryActivationBackward(
        input, output_gradient, input_gradient, "ops::activation::geluBackward", true,
        [](const std::vector<float> &a, const std::vector<float> &g, std::vector<float> &res, size_t n, bool) {
            constexpr float sqrt_2_over_pi = 0.7978845608F;
            constexpr float coeff = 0.044715F;
            for (size_t i = 0; i < n; ++i)
            {
                const float x = a[i];
                const float x2 = x * x;
                const float cube = x2 * x;
                const float inner = std::clamp(sqrt_2_over_pi * (x + coeff * cube), -15.0F, 15.0F);
                const float tanh_val = std::tanh(inner);
                const float sech2_val = 1.0F - tanh_val * tanh_val;
                const float d_inner = sqrt_2_over_pi * (1.0F + 3.0F * coeff * x2);
                const float term = (std::abs(inner) >= 15.0F) ? 0.0F : (0.5F * x * sech2_val * d_inner);
                const float d_gelu = 0.5F * (1.0F + tanh_val) + term;
                res[i] = g[i] * d_gelu;
            }
        },
        [](bool is_fp16) { return is_fp16 ? Compute_Pipeline::GELU_BACKWARD_FP16 : Compute_Pipeline::GELU_BACKWARD; },
        [](uint32_t total_u32) { return total_u32; });
}

void softmax(const Tensor &input, Tensor &output)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::activation::softmax: TERNARY_INT2 is not supported");
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }
    const bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
    const Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    const Shape &in_shape = input.getShape();
    const size_t rank = in_shape.getRank();
    const size_t total_elements = input.getTotalElements();
    if (rank == 0 || total_elements == 0)
    {
        out_impl->setDataType(out_dtype);
        output.reshape(in_shape);
        return;
    }
    const size_t cols_count = in_shape[rank - 1];
    const size_t rows_count = total_elements / cols_count;

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        std::vector<float> result(total_elements);
        for (size_t r = 0; r < rows_count; ++r)
        {
            const size_t base = r * cols_count;
            float max_val = a_data[base];
            for (size_t c = 1; c < cols_count; ++c)
            {
                max_val = std::max(max_val, a_data[base + c]);
            }
            float sum_exp = 0.0F;
            for (size_t c = 0; c < cols_count; ++c)
            {
                const float exp_val = std::exp(a_data[base + c] - max_val);
                result[base + c] = exp_val;
                sum_exp += exp_val;
            }
            const float inv_sum = 1.0F / std::max(sum_exp, 1e-12F);
            for (size_t c = 0; c < cols_count; ++c)
            {
                result[base + c] *= inv_sum;
            }
        }
        writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*out_impl), in_shape, out_dtype, std::move(result), false);
        Logger::logMessage(Input_Format{"ops::activation::softmax (CPU): shape={} (rows={}, cols={})",
                                        in_shape.toString(), rows_count, cols_count},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        output_gpu.setDataType(out_dtype);
        output_gpu.reshape(in_shape);

        struct Softmax_Constants
        {
            uint32_t rows;
            uint32_t columns;
        } constants{validateU32Count(rows_count, "ops::activation::softmax"),
                    validateU32Count(cols_count, "ops::activation::softmax")};

        const auto [wg_x, wg_y] = dispatch2D(rows_count);
        const Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SOFTMAX_FP16 : Compute_Pipeline::SOFTMAX;
        in_gpu.pushToGraph(pipe, {buf_self, output_gpu.getStorageBuffer()}, constants, wg_x, wg_y, 1);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void softmaxBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient)
{
    auto out_impl = output.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!out_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (output_gradient.getExecutionTarget() != output.getExecutionTarget())
    {
        throw std::invalid_argument("ops::activation::softmaxBackward: Execution_Target mismatch");
    }
    if (output.getDataType() == Data_Type::TERNARY_INT2 || output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::activation::softmaxBackward: TERNARY_INT2 is not supported");
    }
    if (input_gradient.getExecutionTarget() != output.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(output.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    out_impl->validateSameDimensions(*grad_impl);
    const bool is_fp16 = (output.getDataType() == Data_Type::FLOAT16 || output_gradient.getDataType() == Data_Type::FLOAT16);
    const Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    const Shape &out_shape = output.getShape();
    const size_t rank = out_shape.getRank();
    const size_t total_elements = output.getTotalElements();
    if (rank == 0 || total_elements == 0)
    {
        in_grad_impl->setDataType(out_dtype);
        input_gradient.reshape(out_shape);
        return;
    }
    const size_t cols_count = out_shape[rank - 1];
    const size_t rows_count = total_elements / cols_count;

    if (output.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = output.getData();
        const auto &grad_data = output_gradient.getData();
        std::vector<float> result(total_elements);
        for (size_t r = 0; r < rows_count; ++r)
        {
            const size_t base = r * cols_count;
            float dot = 0.0F;
            for (size_t c = 0; c < cols_count; ++c)
            {
                dot += a_data[base + c] * grad_data[base + c];
            }
            for (size_t c = 0; c < cols_count; ++c)
            {
                result[base + c] = a_data[base + c] * (grad_data[base + c] - dot);
            }
        }
        writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*in_grad_impl), out_shape, out_dtype, std::move(result), false);
        Logger::logMessage(Input_Format{"ops::activation::softmaxBackward (CPU): shape={} (rows={}, cols={})",
                                        out_shape.toString(), rows_count, cols_count},
                           Log_Level::LOG_DEBUG, true, 0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    else
    {
        const auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        auto contig_self = out_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_gpu;

        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
        in_grad_gpu.setDataType(out_dtype);
        in_grad_gpu.reshape(out_shape);

        struct Softmax_Constants
        {
            uint32_t rows;
            uint32_t columns;
        } constants{validateU32Count(rows_count, "ops::activation::softmaxBackward"),
                    validateU32Count(cols_count, "ops::activation::softmaxBackward")};

        const auto [wg_x, wg_y] = dispatch2D(rows_count);
        const Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SOFTMAX_BACKWARD_FP16 : Compute_Pipeline::SOFTMAX_BACKWARD;
        out_gpu.pushToGraph(pipe, {buf_self, buf_grad, in_grad_gpu.getStorageBuffer()}, constants, wg_x, wg_y, 1);
        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
    }
}

void sigmoid(const Tensor &input, Tensor &output)
{
    struct PushConstants
    {
        uint32_t total_elements;
    };
    runElementwiseUnaryActivation(
        input, output, "ops::activation::sigmoid", true,
        [](const std::vector<float> &a, std::vector<float> &res, size_t n) {
            for (size_t i = 0; i < n; ++i)
            {
                res[i] = 1.0f / (1.0f + std::exp(-a[i]));
            }
        },
        [](bool is_fp16) {
            if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled())
            {
                return Compute_Pipeline::SIGMOID_FORWARD_COOPMAT_FP16;
            }
            return is_fp16 ? Compute_Pipeline::SIGMOID_FORWARD_FP16 : Compute_Pipeline::SIGMOID_FORWARD;
        },
        [](uint32_t total_u32) { return PushConstants{total_u32}; });
}

void sigmoidBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient)
{
    struct PushConstants
    {
        uint32_t total_elements;
    };
    runElementwiseBinaryActivationBackward(
        output, output_gradient, input_gradient, "ops::activation::sigmoidBackward", true,
        [](const std::vector<float> &out_data, const std::vector<float> &g, std::vector<float> &res, size_t n, bool is_fp16) {
            const float max_val = is_fp16 ? 65504.0f : 1e6f;
            for (size_t i = 0; i < n; ++i)
            {
                const float s = out_data[i];
                float dx = g[i] * s * (1.0f - s);
                dx = std::isfinite(dx) ? std::clamp(dx, -max_val, max_val) : 0.0f;
                res[i] = dx;
            }
        },
        [](bool is_fp16) {
            if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled())
            {
                return Compute_Pipeline::SIGMOID_BACKWARD_COOPMAT_FP16;
            }
            return is_fp16 ? Compute_Pipeline::SIGMOID_BACKWARD_FP16 : Compute_Pipeline::SIGMOID_BACKWARD;
        },
        [](uint32_t total_u32) { return PushConstants{total_u32}; });
}

}

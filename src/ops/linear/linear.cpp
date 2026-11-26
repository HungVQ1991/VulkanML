#include "ops/linear/linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::linear
{

static void ensureTensorTarget(Tensor &t, Execution_Target target, const Shape &shape, Data_Type dtype)
{
    auto impl = t.getImplementation();
    if (!impl || t.getExecutionTarget() != target || t.getDataType() != dtype ||
        !impl->isContiguous() || impl->getByteOffset() != 0)
    {
        t = Tensor(shape, dtype, target);
        return;
    }
    if (target == Execution_Target::VULKAN_GPU && shape.getTotalElements() > 0)
    {
        const auto &g = Gpu_Tensor_Impl::castToGpu(*impl);
        size_t req_bytes = shape.getTotalElements() * getDataTypeSize(dtype);
        if (!g.getStorageBuffer() || g.getStorageBuffer()->getSizeBytes() < req_bytes)
        {
            t = Tensor(shape, dtype, target);
            return;
        }
    }
    t.reshape(shape);
}

void linearForward(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output)
{
    input.matmulAdd(weights, biases, output);
}

void linearBackwardInput(const Tensor &output_gradient, const Tensor &weights, Tensor &input_gradient)
{
    auto og_impl = output_gradient.getImplementation();
    auto w_impl = weights.getImplementation();
    if (!og_impl || !w_impl) return;

    Execution_Target target = output_gradient.getExecutionTarget();
    if (weights.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::linear::linearBackwardInput: Execution_Target mismatch");
    }
    if (output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        weights.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::linear::linearBackwardInput: TERNARY_INT2 is not supported");
    }
    const Shape &og_shape = output_gradient.getShape();
    const Shape &w_shape = weights.getShape();
    if (og_shape.getRank() < 2 || w_shape.getRank() != 2)
    {
        throw std::invalid_argument("ops::linear::linearBackwardInput: Expected rank >= 2 output_gradient and rank 2 weights");
    }
    const size_t rank = og_shape.getRank();
    const size_t out_dim = og_shape[rank - 1];
    size_t b_size = 1;
    for (size_t i = 0; i + 1 < rank; ++i)
    {
        b_size *= og_shape[i];
    }
    const size_t in_dim = w_shape[0];
    if (out_dim != w_shape[1])
    {
        throw std::invalid_argument("ops::linear::linearBackwardInput: Dimension mismatch");
    }
    Shape in_grad_shape = og_shape;
    in_grad_shape[rank - 1] = in_dim;

    if (target == Execution_Target::CPU)
    {
        Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        std::vector<float> result(b_size * in_dim, 0.0f);
        if (b_size > 0 && in_dim > 0 && out_dim > 0)
        {
            const auto &w_data = weights.getData();
            const auto &out_grad_data = output_gradient.getData();
            for (size_t i = 0; i < b_size; ++i)
            {
                for (size_t j = 0; j < in_dim; ++j)
                {
                    float sum = 0.0F;
                    for (size_t k = 0; k < out_dim; ++k)
                    {
                        sum += out_grad_data[i * out_dim + k] * w_data[j * out_dim + k];
                    }
                    result[i * in_dim + j] = sum;
                }
            }
        }

        ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
        in_grad_cpu.setByteOffset(0);
        if (out_dtype == Data_Type::FLOAT16)
        {
            auto &fp16_buf = in_grad_cpu.getStorageBufferFp16();
            if (!fp16_buf || fp16_buf.use_count() > 1 || fp16_buf->size() != result.size())
            {
                fp16_buf = std::make_shared<std::vector<float16_t>>(result.size());
            }
            if (!result.empty())
            {
                convertFp32ToFp16(result.data(), fp16_buf->data(), result.size());
            }
            in_grad_cpu.getStorageBuffer().reset();
            in_grad_cpu.getStorageBufferU32().reset();
        }
        else
        {
            auto &fp32_buf = in_grad_cpu.getStorageBuffer();
            if (!fp32_buf || fp32_buf.use_count() > 1 || fp32_buf->size() != result.size())
            {
                fp32_buf = std::make_shared<std::vector<float>>(std::move(result));
            }
            else
            {
                *fp32_buf = std::move(result);
            }
            in_grad_cpu.getStorageBufferFp16().reset();
            in_grad_cpu.getStorageBufferU32().reset();
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*weights.getImplementation());

        bool use_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16 || w_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        Data_Type out_dtype = use_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_size == 0 || in_dim == 0 || out_dim == 0)
        {
            ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
            if (out_dim == 0 && b_size > 0 && in_dim > 0)
            {
                in_grad_gpu.zero();
            }
            in_grad_gpu.clearHostCache();
            in_grad_gpu.invalidateFp16Cache();
            return;
        }

        if (b_size > std::numeric_limits<uint32_t>::max() ||
            in_dim > std::numeric_limits<uint32_t>::max() ||
            out_dim > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::linear::linearBackwardInput: Dimension exceeds UINT32_MAX");
        }

        struct Constants
        {
            uint32_t batch_size;
            uint32_t input_dimension;
            uint32_t output_dimension;
        } c{static_cast<uint32_t>(b_size), static_cast<uint32_t>(in_dim), static_cast<uint32_t>(out_dim)};

        const bool use_coop = use_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled();
        Logger::logMessage(Input_Format{"ops::linear::linearBackwardInput: dispatch fp16={}, coopmat={}, B={}, in={}, out={}",
                                        use_fp16, use_coop, b_size, in_dim, out_dim},
                           Log_Level::LOG_DEBUG,
                           false,
                           1,
                           Log_Feature::DENSE_COMPUTE);
        const uint32_t tile = use_coop ? 64u : 16u;
        const uint32_t wg_x = (c.input_dimension + tile - 1u) / tile;
        const uint32_t wg_y = (c.batch_size + tile - 1u) / tile;
        const auto &max_wg = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount();
        if (wg_x > max_wg[0] || wg_y > max_wg[1])
        {
            throw std::overflow_error("ops::linear::linearBackwardInput: Dispatch exceeds maxComputeWorkGroupCount");
        }

        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        auto buf_self = use_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
        auto buf_w = use_fp16 ? effective_w->getEffectiveFp16Storage() : effective_w->getStorageBuffer();

        ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
        if (in_grad_gpu.getStorageBuffer().get() == buf_self.get() ||
            in_grad_gpu.getStorageBuffer().get() == buf_w.get())
        {
            in_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), in_grad_shape.getTotalElements(), out_dtype);
            in_grad_gpu.markGpuMutated();
        }

        if (use_fp16)
        {
            Compute_Pipeline pipeline = use_coop
                ? Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16
                : Compute_Pipeline::LINEAR_BACKWARD_INPUT_FP16;
            effective_grad->pushToGraph(pipeline,
                        {buf_self, buf_w, in_grad_gpu.getStorageBuffer()}, c,
                        wg_x, wg_y, 1);
        }
        else
        {
            effective_grad->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_INPUT,
                        {buf_self, buf_w, in_grad_gpu.getStorageBuffer()}, c,
                        wg_x, wg_y, 1);
        }
        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
    }
}

void linearBackwardWeightBias(const Tensor &input, const Tensor &output_gradient,
                             Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate)
{
    auto in_impl = input.getImplementation();
    auto og_impl = output_gradient.getImplementation();
    if (!in_impl || !og_impl) return;

    Execution_Target target = input.getExecutionTarget();
    if (output_gradient.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::linear::linearBackwardWeightBias: Execution_Target mismatch");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::linear::linearBackwardWeightBias: TERNARY_INT2 is not supported");
    }
    const Shape &in_shape = input.getShape();
    const Shape &og_shape = output_gradient.getShape();
    const size_t rank = in_shape.getRank();
    if (rank < 2 || og_shape.getRank() != rank)
    {
        throw std::invalid_argument("ops::linear::linearBackwardWeightBias: Expected matching rank >= 2 tensors");
    }
    for (size_t i = 0; i + 1 < rank; ++i)
    {
        if (in_shape[i] != og_shape[i])
        {
            throw std::invalid_argument("ops::linear::linearBackwardWeightBias: Batch dimension mismatch");
        }
    }
    size_t b_size = 1;
    for (size_t i = 0; i + 1 < rank; ++i)
    {
        b_size *= in_shape[i];
    }
    const size_t in_dim = in_shape[rank - 1];
    const size_t out_dim = og_shape[rank - 1];

    if (target == Execution_Target::CPU)
    {
        std::vector<float> in_data = (b_size > 0 && in_dim > 0) ? input.getData() : std::vector<float>{};
        std::vector<float> out_grad_data = (b_size > 0 && out_dim > 0) ? output_gradient.getData() : std::vector<float>{};

        ensureTensorTarget(weight_gradient, target, Shape{ in_dim, out_dim }, Data_Type::FLOAT32);
        ensureTensorTarget(bias_gradient, target, Shape{ 1, out_dim }, Data_Type::FLOAT32);

        auto &w_grad_cpu = Cpu_Tensor_Impl::castToCpu(*weight_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*bias_gradient.getImplementation());

        auto prepareWritableFp32 = [accumulate](Cpu_Tensor_Impl &cpu_t, size_t count)
        {
            auto &buf = cpu_t.getStorageBuffer();
            if (!buf || buf->size() != count)
            {
                buf = std::make_shared<std::vector<float>>(count, 0.0f);
            }
            else if (buf.use_count() > 1)
            {
                buf = accumulate ? std::make_shared<std::vector<float>>(*buf)
                                 : std::make_shared<std::vector<float>>(count, 0.0f);
            }
            else if (!accumulate)
            {
                std::fill(buf->begin(), buf->end(), 0.0f);
            }
            cpu_t.getStorageBufferFp16().reset();
            cpu_t.getStorageBufferU32().reset();
        };

        prepareWritableFp32(w_grad_cpu, in_dim * out_dim);
        prepareWritableFp32(b_grad_cpu, out_dim);

        for (size_t i = 0; i < b_size; ++i)
        {
            for (size_t j = 0; j < out_dim; ++j)
            {
                float grad_val = out_grad_data[i * out_dim + j];
                (*b_grad_cpu.getStorageBuffer())[j] += grad_val;
                for (size_t k = 0; k < in_dim; ++k)
                {
                    (*w_grad_cpu.getStorageBuffer())[k * out_dim + j] += in_data[i * in_dim + k] * grad_val;
                }
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_in = in_gpu.ensureContiguousSelf();
        const auto *effective_in = contig_in ? contig_in.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        Shape w_shape{ in_dim, out_dim };
        Shape b_shape{ 1, out_dim };

        bool use_fp16 = (input.getDataType() == Data_Type::FLOAT16 || out_grad_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        auto buf_in = use_fp16 ? effective_in->getEffectiveFp16Storage() : effective_in->getStorageBuffer();
        auto buf_grad = use_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

        auto needsFreshGpuFloat32 = [&](const Tensor &t, const Shape &expected_shape) -> bool
        {
            auto impl = t.getImplementation();
            if (!impl || t.getExecutionTarget() != target || t.getDataType() != Data_Type::FLOAT32 ||
                t.getShape() != expected_shape || !impl->isContiguous() || impl->getByteOffset() != 0)
            {
                return true;
            }
            const auto &g = Gpu_Tensor_Impl::castToGpu(*impl);
            size_t req_bytes = expected_shape.getTotalElements() * sizeof(float);
            return (req_bytes > 0 && (!g.getStorageBuffer() || g.getStorageBuffer()->getSizeBytes() < req_bytes));
        };

        bool w_reallocated = needsFreshGpuFloat32(weight_gradient, w_shape);
        bool b_reallocated = needsFreshGpuFloat32(bias_gradient, b_shape);

        ensureTensorTarget(weight_gradient, target, w_shape, Data_Type::FLOAT32);
        ensureTensorTarget(bias_gradient, target, b_shape, Data_Type::FLOAT32);

        auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*bias_gradient.getImplementation());
        if (w_shape.getTotalElements() > 0 &&
            (w_grad_gpu.getStorageBuffer().get() == buf_in.get() ||
             w_grad_gpu.getStorageBuffer().get() == buf_grad.get()))
        {
            w_reallocated = true;
            w_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), w_shape.getTotalElements(), Data_Type::FLOAT32);
            w_grad_gpu.markGpuMutated();
        }
        if (b_shape.getTotalElements() > 0 &&
            (b_grad_gpu.getStorageBuffer().get() == buf_in.get() ||
             b_grad_gpu.getStorageBuffer().get() == buf_grad.get() ||
             b_grad_gpu.getStorageBuffer().get() == w_grad_gpu.getStorageBuffer().get()))
        {
            b_reallocated = true;
            b_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), b_shape.getTotalElements(), Data_Type::FLOAT32);
            b_grad_gpu.markGpuMutated();
        }

        constexpr size_t elem_size = sizeof(float);
        const size_t w_elems = w_shape.getTotalElements();

        if (in_dim == 0 || out_dim == 0)
        {
            if (out_dim > 0 && b_grad_gpu.getStorageBuffer() && (!accumulate || b_reallocated))
            {
                Execution_Engine::getInstance().getContext().fillBuffer(b_grad_gpu.getStorageBuffer()->getBuffer(), out_dim * elem_size, 0, 0);
            }
            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            return;
        }

        if (b_size == 0)
        {
            auto &ctx = Execution_Engine::getInstance().getContext();
            if ((!accumulate || w_reallocated) && w_grad_gpu.getStorageBuffer())
            {
                ctx.fillBuffer(w_grad_gpu.getStorageBuffer()->getBuffer(), w_elems * elem_size, 0, 0);
            }
            if ((!accumulate || b_reallocated) && b_grad_gpu.getStorageBuffer())
            {
                ctx.fillBuffer(b_grad_gpu.getStorageBuffer()->getBuffer(), out_dim * elem_size, 0, 0);
            }
            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            return;
        }

        bool effective_accumulate = accumulate;
        if (accumulate && w_reallocated && b_reallocated)
        {
            effective_accumulate = false;
        }
        else if (accumulate && w_reallocated && w_grad_gpu.getStorageBuffer())
        {
            Execution_Engine::getInstance().getContext().fillBuffer(w_grad_gpu.getStorageBuffer()->getBuffer(), w_elems * elem_size, 0, 0);
        }
        else if (accumulate && b_reallocated && b_grad_gpu.getStorageBuffer())
        {
            Execution_Engine::getInstance().getContext().fillBuffer(b_grad_gpu.getStorageBuffer()->getBuffer(), out_dim * elem_size, 0, 0);
        }

        if (b_size > std::numeric_limits<uint32_t>::max() ||
            in_dim > std::numeric_limits<uint32_t>::max() ||
            out_dim > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::linear::linearBackwardWeightBias: Dimension exceeds UINT32_MAX");
        }

        struct Constants
        {
            uint32_t batch_size;
            uint32_t input_dimension;
            uint32_t output_dimension;
            uint32_t accumulate;
        } c{static_cast<uint32_t>(b_size), static_cast<uint32_t>(in_dim), static_cast<uint32_t>(out_dim), effective_accumulate ? 1u : 0u};

        const bool use_coop = use_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled();
        Logger::logMessage(Input_Format{"ops::linear::linearBackwardWeightBias: dispatch fp16={}, coopmat={}, B={}, in={}, out={}",
                                        use_fp16, use_coop, b_size, in_dim, out_dim},
                           Log_Level::LOG_DEBUG,
                           false,
                           1,
                           Log_Feature::DENSE_COMPUTE);
        const uint32_t tile = use_coop ? 64u : 16u;
        const uint32_t wg_x = (c.output_dimension + tile - 1u) / tile;
        const uint32_t wg_y = (c.input_dimension + tile - 1u) / tile;
        const auto &max_wg = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount();
        if (wg_x > max_wg[0] || wg_y > max_wg[1])
        {
            throw std::overflow_error("ops::linear::linearBackwardWeightBias: Dispatch exceeds maxComputeWorkGroupCount");
        }

        if (use_fp16)
        {
            if (use_coop)
            {
                const uint32_t total_tiles = wg_x * wg_y;
                const bool splitk_eligible = !Execution_Engine::getInstance().isFusedGemmAdamEnabled() &&
                                             (b_size >= 512u) &&
                                             ((in_dim & 7u) == 0u) &&
                                             ((out_dim & 7u) == 0u) &&
                                             (total_tiles <= 64u);
                const uint32_t split_k = splitk_eligible ? std::min<uint32_t>((total_tiles <= 32u) ? 4u : 2u, max_wg[2]) : 1u;

                struct Coopmat_Constants
                {
                    uint32_t batch_size;
                    uint32_t input_dimension;
                    uint32_t output_dimension;
                    uint32_t accumulate;
                    uint32_t split_k;
                    uint32_t pass_mode;
                };
                static_assert(sizeof(Coopmat_Constants) <= 128);

                if (split_k > 1u)
                {
                    const size_t total_reduce_elems = w_elems + out_dim;
                    const size_t partials_elems = static_cast<size_t>(split_k) * total_reduce_elems;
                    auto partials_buf = Execution_Engine::getInstance().acquireReusableScratchBuffer(
                        Execution_Engine::SCRATCH_GEMM_SPLITK,
                        partials_elems * sizeof(float),
                        Data_Type::FLOAT32,
                        partials_elems);

                    Coopmat_Constants pass1_c{c.batch_size, c.input_dimension, c.output_dimension, c.accumulate, split_k, 1u};
                    effective_in->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
                                              {buf_in, buf_grad, partials_buf, partials_buf}, pass1_c,
                                              wg_x, wg_y, split_k);

                    const uint64_t red_wg_x_u64 = (total_reduce_elems + 127ULL) / 128ULL;
                    if (red_wg_x_u64 > max_wg[0])
                    {
                        throw std::overflow_error("ops::linear::linearBackwardWeightBias: Reduce dispatch exceeds maxComputeWorkGroupCount[0]");
                    }
                    Coopmat_Constants pass2_c{c.batch_size, c.input_dimension, c.output_dimension, c.accumulate, split_k, 2u};
                    effective_in->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
                                              {partials_buf, partials_buf, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                              pass2_c, static_cast<uint32_t>(red_wg_x_u64), 1, 1);
                }
                else
                {
                    Coopmat_Constants single_c{c.batch_size, c.input_dimension, c.output_dimension, c.accumulate, 1u, 0u};
                    effective_in->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
                                              {buf_in, buf_grad, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()}, single_c,
                                              wg_x, wg_y, 1);
                }
            }
            else
            {
                effective_in->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_FP16,
                                          {buf_in, buf_grad, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()}, c,
                                          wg_x, wg_y, 1);
            }
        }
        else
        {
            effective_in->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS,
                        {buf_in, buf_grad, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()}, c,
                        wg_x, wg_y, 1);
        }
        w_grad_gpu.clearHostCache();
        w_grad_gpu.invalidateFp16Cache();
        b_grad_gpu.clearHostCache();
        b_grad_gpu.invalidateFp16Cache();
    }
}

}

#include "ops/llm/swiglu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::llm
{

static void ensureTensorTarget(Tensor &t, Execution_Target target, const Shape &shape, Data_Type dtype)
{
    if (t.getExecutionTarget() != target || t.getDataType() != dtype || t.getShape() != shape)
    {
        t = Tensor(shape, dtype, target);
    }
    else
    {
        t.reshape(shape);
    }
}

void swigluForward(const Tensor &a, const Tensor &b, Tensor &output)
{
    auto a_impl = a.getImplementation();
    auto b_impl = b.getImplementation();
    if (!a_impl || !b_impl) return;

    Execution_Target target = a.getExecutionTarget();
    Shape shape = a.getShape();

    bool is_fp16 = (a.getDataType() == Data_Type::FLOAT16 || b.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type out_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : a.getDataType();

    ensureTensorTarget(output, target, shape, out_dtype);

    if (target == Execution_Target::CPU)
    {
        const auto &a_data = a.getData();
        const auto &b_data = b.getData();
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        size_t total = a.getTotalElements();
        for (size_t idx = 0; idx < total; ++idx)
        {
            float a_val = a_data[idx];
            float b_val = b_data[idx];
            float sig_a = 1.0f / (1.0f + std::exp(-a_val));
            (*out_cpu.getStorageBuffer())[idx] = (a_val * sig_a) * b_val;
        }
    }
    else
    {
        const auto &a_gpu = Gpu_Tensor_Impl::castToGpu(*a.getImplementation());
        auto contig_a = a_gpu.ensureContiguousSelf();
        const auto *effective_a = contig_a ? contig_a.get() : &a_gpu;

        const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*b.getImplementation());
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        uint32_t total = static_cast<uint32_t>(a.getTotalElements());
        struct Swiglu_Constants
        {
            uint32_t total_elements;
        } pcs{total};

        if (is_fp16)
        {
            auto buf_a = effective_a->getEffectiveFp16Storage();
            auto buf_b = effective_b->getEffectiveFp16Storage();
            effective_a->pushToGraph(Compute_Pipeline::SWIGLU_FP16,
                                     {buf_a, buf_b, out_gpu.getStorageBuffer()},
                                     pcs, (total + 255) / 256, 1, 1);
        }
        else
        {
            effective_a->pushToGraph(Compute_Pipeline::SWIGLU,
                                     {effective_a->getStorageBuffer(), effective_b->getStorageBuffer(), out_gpu.getStorageBuffer()},
                                     pcs, (total + 255) / 256, 1, 1);
        }
    }
}

void swigluBackward(const Tensor &output_gradient, const Tensor &a, const Tensor &b,
                    Tensor &grad_a, Tensor &grad_b)
{
    auto dy_impl = output_gradient.getImplementation();
    auto a_impl = a.getImplementation();
    auto b_impl = b.getImplementation();
    if (!dy_impl || !a_impl || !b_impl) return;

    Execution_Target target = a.getExecutionTarget();
    Shape shape = a.getShape();

    bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16 || a.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type grad_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    ensureTensorTarget(grad_a, target, shape, grad_dtype);
    ensureTensorTarget(grad_b, target, shape, grad_dtype);

    if (target == Execution_Target::CPU)
    {
        const auto &dy_data = output_gradient.getData();
        const auto &a_data = a.getData();
        const auto &b_data = b.getData();

        auto &da_cpu = Cpu_Tensor_Impl::castToCpu(*grad_a.getImplementation());
        auto &db_cpu = Cpu_Tensor_Impl::castToCpu(*grad_b.getImplementation());

        size_t total = a.getTotalElements();
        for (size_t idx = 0; idx < total; ++idx)
        {
            float dy = dy_data[idx];
            float a_val = a_data[idx];
            float b_val = b_data[idx];

            float sig_a = 1.0f / (1.0f + std::exp(-a_val));
            float silu_a = a_val * sig_a;
            float d_silu_a = sig_a * (1.0f + a_val * (1.0f - sig_a));

            (*db_cpu.getStorageBuffer())[idx] = dy * silu_a;
            (*da_cpu.getStorageBuffer())[idx] = dy * b_val * d_silu_a;
        }
    }
    else
    {
        const auto &a_gpu = Gpu_Tensor_Impl::castToGpu(*a.getImplementation());
        auto contig_a = a_gpu.ensureContiguousSelf();
        const auto *effective_a = contig_a ? contig_a.get() : &a_gpu;

        const auto &dy_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_dy = dy_gpu.ensureContiguousSelf();
        const auto *effective_dy = contig_dy ? contig_dy.get() : &dy_gpu;

        const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*b.getImplementation());
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        auto &da_gpu = Gpu_Tensor_Impl::castToGpu(*grad_a.getImplementation());
        auto &db_gpu = Gpu_Tensor_Impl::castToGpu(*grad_b.getImplementation());

        uint32_t total = static_cast<uint32_t>(a.getTotalElements());
        struct Swiglu_Constants
        {
            uint32_t total_elements;
        } pcs{total};

        if (is_fp16)
        {
            auto buf_dy = effective_dy->getEffectiveFp16Storage();
            auto buf_a = effective_a->getEffectiveFp16Storage();
            auto buf_b = effective_b->getEffectiveFp16Storage();
            effective_a->pushToGraph(Compute_Pipeline::SWIGLU_BACKWARD_FP16,
                                     {buf_dy, buf_a, buf_b, da_gpu.getStorageBuffer(), db_gpu.getStorageBuffer()},
                                     pcs, (total + 255) / 256, 1, 1);
        }
        else
        {
            effective_a->pushToGraph(Compute_Pipeline::SWIGLU_BACKWARD,
                                     {effective_dy->getStorageBuffer(), effective_a->getStorageBuffer(), effective_b->getStorageBuffer(), da_gpu.getStorageBuffer(), db_gpu.getStorageBuffer()},
                                     pcs, (total + 255) / 256, 1, 1);
        }
    }
}

void fusedSwiGLUForward(const Tensor &input, Tensor &output)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    Execution_Target target = input.getExecutionTarget();
    size_t rows = input.getShape()[0];
    size_t total_cols = input.getShape()[1];
    size_t half_dim = total_cols / 2;

    bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type out_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    ensureTensorTarget(output, target, Shape{rows, half_dim}, out_dtype);

    if (target == Execution_Target::CPU)
    {
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        const auto &in_data = input.getData();

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < half_dim; ++c)
            {
                float a = in_data[r * total_cols + c];
                float b = in_data[r * total_cols + half_dim + c];
                float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a, -85.0f, 85.0f)));
                float silu_a = a * sig_a;
                (*out_cpu.getStorageBuffer())[r * half_dim + c] = silu_a * b;
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        uint32_t total = static_cast<uint32_t>(rows * half_dim);
        struct Fused_Swiglu_Constants
        {
            uint32_t total_elements;
            uint32_t half_dim;
        } pcs{total, static_cast<uint32_t>(half_dim)};

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::FUSED_SWIGLU_FORWARD_FP16 : Compute_Pipeline::FUSED_SWIGLU_FORWARD,
                                    {buf_self, out_gpu.getStorageBuffer()},
                                    pcs, (total + 255) / 256, 1, 1);
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
    }
}

void fusedSwiGLUBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient)
{
    auto in_impl = input.getImplementation();
    auto dy_impl = output_gradient.getImplementation();
    if (!in_impl || !dy_impl) return;

    Execution_Target target = input.getExecutionTarget();
    size_t rows = input.getShape()[0];
    size_t total_cols = input.getShape()[1];
    size_t half_dim = total_cols / 2;

    bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || output_gradient.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type grad_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    ensureTensorTarget(input_gradient, target, Shape{rows, total_cols}, grad_dtype);

    if (target == Execution_Target::CPU)
    {
        auto &dx_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
        const auto &in_data = input.getData();
        const auto &dy_data = output_gradient.getData();

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < half_dim; ++c)
            {
                float dy = dy_data[r * half_dim + c];
                float a = in_data[r * total_cols + c];
                float b = in_data[r * total_cols + half_dim + c];
                float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a, -85.0f, 85.0f)));
                float silu_a = a * sig_a;
                float d_silu_a = sig_a * (1.0f + a * (1.0f - sig_a));

                (*dx_cpu.getStorageBuffer())[r * total_cols + c] = dy * b * d_silu_a;
                (*dx_cpu.getStorageBuffer())[r * total_cols + half_dim + c] = dy * silu_a;
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &dy_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_dy = dy_gpu.ensureContiguousSelf();
        const auto *effective_dy = contig_dy ? contig_dy.get() : &dy_gpu;

        auto &dx_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        uint32_t total = static_cast<uint32_t>(rows * half_dim);
        struct Fused_Swiglu_Constants
        {
            uint32_t total_elements;
            uint32_t half_dim;
        } pcs{total, static_cast<uint32_t>(half_dim)};

        auto buf_dy = is_fp16 ? effective_dy->getEffectiveFp16Storage() : effective_dy->getStorageBuffer();
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::FUSED_SWIGLU_BACKWARD_FP16 : Compute_Pipeline::FUSED_SWIGLU_BACKWARD,
                                    {buf_dy, buf_self, dx_gpu.getStorageBuffer()},
                                    pcs, (total + 255) / 256, 1, 1);
        dx_gpu.clearHostCache();
        dx_gpu.invalidateFp16Cache();
    }
}

} // namespace ops::llm

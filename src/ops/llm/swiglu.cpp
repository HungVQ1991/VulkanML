#include "ops/llm/swiglu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::llm
{

using ops::internal::ensureDisjointGpuStorage;
using ops::internal::ensureDisjointTensorTarget;
using ops::internal::ensureUniqueTensorTarget;
using ops::internal::resolveExecutionEngine;
using ops::internal::validatePushConstants;

static void validateOperand(const Tensor &t, Execution_Target expected, const char *op, const char *name)
{
    if (!t.getImplementation())
    {
        throw std::invalid_argument(std::string(op) + ": " + name + " has null implementation");
    }
    if (t.getExecutionTarget() != expected)
    {
        throw std::invalid_argument(std::string(op) + ": Execution_Target mismatch for " + name);
    }
    if (t.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op) + ": TERNARY_INT2 is not supported for " + name);
    }
}

static void writeCpuOutput(Tensor &out, const Shape &shape, Data_Type dtype, const std::vector<float> &vals)
{
    ensureUniqueTensorTarget(out, Execution_Target::CPU, shape, dtype);
    auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*out.getImplementation());
    if (dtype == Data_Type::FLOAT16)
    {
        std::vector<float> clamped(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
        {
            clamped[i] = std::clamp(vals[i], -65504.0f, 65504.0f);
        }
        out_cpu.uploadData(clamped);
    }
    else
    {
        out_cpu.setStorageBufferFp16(nullptr);
        out_cpu.setDataType(Data_Type::FLOAT32);
        out_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(vals);
    }
}

void swigluForward(const Tensor &a, const Tensor &b, Tensor &output, const SwiGLU_Config &config)
{
    auto a_impl = a.getImplementation();
    auto b_impl = b.getImplementation();
    if (!a_impl || !b_impl) return;

    Execution_Target target = a.getExecutionTarget();
    validateOperand(a, target, "ops::llm::swigluForward", "a");
    validateOperand(b, target, "ops::llm::swigluForward", "b");
    if (a.getShape() != b.getShape())
    {
        throw std::invalid_argument("ops::llm::swigluForward: shape mismatch between a and b");
    }

    Shape shape = a.getShape();
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;

    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (a.getDataType() == Data_Type::FLOAT16 || b.getDataType() == Data_Type::FLOAT16 || resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type out_dtype = (target != Execution_Target::CPU)
        ? (is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
        : ((a.getDataType() == Data_Type::FLOAT16 || b.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : a.getDataType());

    size_t total_elements = a.getTotalElements();
    if (total_elements == 0)
    {
        ensureDisjointTensorTarget(output, target, shape, out_dtype, {&a, &b});
        if (target != Execution_Target::CPU)
        {
            auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            out_gpu.clearHostCache();
            out_gpu.invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto a_data = a.getData();
        const auto b_data = b.getData();
        ensureDisjointTensorTarget(output, target, shape, out_dtype, {&a, &b});
        std::vector<float> fp32_out(total_elements, 0.0f);

        for (size_t idx = 0; idx < total_elements; ++idx)
        {
            float a_val = a_data[idx];
            float b_val = b_data[idx];
            float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a_val, -85.0f, 85.0f)));
            fp32_out[idx] = (a_val * sig_a) * b_val;
        }
        writeCpuOutput(output, shape, out_dtype, fp32_out);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::llm::swigluForward: total_elements exceeds uint32_t limit");
        }
        const auto &a_gpu = Gpu_Tensor_Impl::castToGpu(*a_impl);
        auto contig_a = a_gpu.ensureContiguousSelf();
        const auto *effective_a = contig_a ? contig_a.get() : &a_gpu;

        const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*b_impl);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        uint32_t total = static_cast<uint32_t>(total_elements);
        struct Swiglu_Constants
        {
            uint32_t total_elements;
        } pcs{total};
        validatePushConstants<Swiglu_Constants>();
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid(
            is_fp16 ? ((total_elements + 1023ull) / 1024ull) : ((total_elements + 255ull) / 256ull));

        std::shared_ptr<Gpu_Tensor_Impl> casted_a, casted_b;
        std::shared_ptr<gpu::vector> buf_a, buf_b;
        if (is_fp16)
        {
            buf_a = effective_a->getEffectiveFp16Storage();
            buf_b = effective_b->getEffectiveFp16Storage();
        }
        else
        {
            if (effective_a->getDataType() == Data_Type::FLOAT16)
            {
                casted_a = std::make_shared<Gpu_Tensor_Impl>(effective_a->getShape(), Data_Type::FLOAT32);
                effective_a->to(Data_Type::FLOAT32, *casted_a);
                effective_a = casted_a.get();
            }
            if (effective_b->getDataType() == Data_Type::FLOAT16)
            {
                casted_b = std::make_shared<Gpu_Tensor_Impl>(effective_b->getShape(), Data_Type::FLOAT32);
                effective_b->to(Data_Type::FLOAT32, *casted_b);
                effective_b = casted_b.get();
            }
            buf_a = effective_a->getStorageBuffer();
            buf_b = effective_b->getStorageBuffer();
        }

        ensureDisjointTensorTarget(output, target, shape, out_dtype, {&a, &b});
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        ensureDisjointGpuStorage(out_gpu, total_elements, out_dtype, {buf_a.get(), buf_b.get()});

        effective_a->pushToGraph(is_fp16 ? Compute_Pipeline::SWIGLU_FP16 : Compute_Pipeline::SWIGLU,
                                 {buf_a, buf_b, out_gpu.getStorageBuffer()},
                                 pcs, wg_x, wg_y, 1, &engine);
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
    }
}

void swigluForward(const Tensor &a, const Tensor &b, Tensor &output)
{
    swigluForward(a, b, output, SwiGLU_Config{});
}

void swigluBackward(const Tensor &a, const Tensor &b, const Tensor &output_gradient,
                    Tensor &grad_a, Tensor &grad_b, const SwiGLU_Config &config)
{
    auto dy_impl = output_gradient.getImplementation();
    auto a_impl = a.getImplementation();
    auto b_impl = b.getImplementation();
    if (!dy_impl || !a_impl || !b_impl) return;

    Execution_Target target = a.getExecutionTarget();
    validateOperand(a, target, "ops::llm::swigluBackward", "a");
    validateOperand(b, target, "ops::llm::swigluBackward", "b");
    validateOperand(output_gradient, target, "ops::llm::swigluBackward", "output_gradient");
    if (a.getShape() != b.getShape() || output_gradient.getShape() != a.getShape())
    {
        throw std::invalid_argument("ops::llm::swigluBackward: shape mismatch between output_gradient, a, and b");
    }

    Shape shape = a.getShape();
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;

    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (output_gradient.getDataType() == Data_Type::FLOAT16 ||
                   a.getDataType() == Data_Type::FLOAT16 ||
                   b.getDataType() == Data_Type::FLOAT16 ||
                   resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type grad_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    size_t total_elements = a.getTotalElements();
    if (total_elements == 0)
    {
        ensureDisjointTensorTarget(grad_a, target, shape, grad_dtype, {&a, &b, &output_gradient});
        ensureDisjointTensorTarget(grad_b, target, shape, grad_dtype, {&a, &b, &output_gradient, &grad_a});
        if (target != Execution_Target::CPU)
        {
            auto &da_gpu = Gpu_Tensor_Impl::castToGpu(*grad_a.getImplementation());
            auto &db_gpu = Gpu_Tensor_Impl::castToGpu(*grad_b.getImplementation());
            da_gpu.clearHostCache();
            da_gpu.invalidateFp16Cache();
            db_gpu.clearHostCache();
            db_gpu.invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto dy_data = output_gradient.getData();
        const auto a_data = a.getData();
        const auto b_data = b.getData();
        ensureDisjointTensorTarget(grad_a, target, shape, grad_dtype, {&a, &b, &output_gradient});
        ensureDisjointTensorTarget(grad_b, target, shape, grad_dtype, {&a, &b, &output_gradient, &grad_a});

        std::vector<float> da_vals(total_elements, 0.0f);
        std::vector<float> db_vals(total_elements, 0.0f);
        for (size_t idx = 0; idx < total_elements; ++idx)
        {
            float dy = dy_data[idx];
            float a_val = a_data[idx];
            float b_val = b_data[idx];

            float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a_val, -85.0f, 85.0f)));
            float silu_a = a_val * sig_a;
            float d_silu_a = sig_a * (1.0f + a_val * (1.0f - sig_a));

            db_vals[idx] = dy * silu_a;
            da_vals[idx] = dy * b_val * d_silu_a;
        }
        writeCpuOutput(grad_a, shape, grad_dtype, da_vals);
        writeCpuOutput(grad_b, shape, grad_dtype, db_vals);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::llm::swigluBackward: total_elements exceeds uint32_t limit");
        }
        const auto &a_gpu = Gpu_Tensor_Impl::castToGpu(*a_impl);
        auto contig_a = a_gpu.ensureContiguousSelf();
        const auto *effective_a = contig_a ? contig_a.get() : &a_gpu;

        const auto &dy_gpu = Gpu_Tensor_Impl::castToGpu(*dy_impl);
        auto contig_dy = dy_gpu.ensureContiguousSelf();
        const auto *effective_dy = contig_dy ? contig_dy.get() : &dy_gpu;

        const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*b_impl);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        uint32_t total = static_cast<uint32_t>(total_elements);
        struct Swiglu_Constants
        {
            uint32_t total_elements;
        } pcs{total};
        validatePushConstants<Swiglu_Constants>();
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid(
            is_fp16 ? ((total_elements + 1023ull) / 1024ull) : ((total_elements + 255ull) / 256ull));

        std::shared_ptr<Gpu_Tensor_Impl> casted_dy, casted_a, casted_b;
        std::shared_ptr<gpu::vector> buf_dy, buf_a, buf_b;
        if (is_fp16)
        {
            buf_dy = effective_dy->getEffectiveFp16Storage();
            buf_a = effective_a->getEffectiveFp16Storage();
            buf_b = effective_b->getEffectiveFp16Storage();
        }
        else
        {
            if (effective_dy->getDataType() == Data_Type::FLOAT16)
            {
                casted_dy = std::make_shared<Gpu_Tensor_Impl>(effective_dy->getShape(), Data_Type::FLOAT32);
                effective_dy->to(Data_Type::FLOAT32, *casted_dy);
                effective_dy = casted_dy.get();
            }
            if (effective_a->getDataType() == Data_Type::FLOAT16)
            {
                casted_a = std::make_shared<Gpu_Tensor_Impl>(effective_a->getShape(), Data_Type::FLOAT32);
                effective_a->to(Data_Type::FLOAT32, *casted_a);
                effective_a = casted_a.get();
            }
            if (effective_b->getDataType() == Data_Type::FLOAT16)
            {
                casted_b = std::make_shared<Gpu_Tensor_Impl>(effective_b->getShape(), Data_Type::FLOAT32);
                effective_b->to(Data_Type::FLOAT32, *casted_b);
                effective_b = casted_b.get();
            }
            buf_dy = effective_dy->getStorageBuffer();
            buf_a = effective_a->getStorageBuffer();
            buf_b = effective_b->getStorageBuffer();
        }

        ensureDisjointTensorTarget(grad_a, target, shape, grad_dtype, {&a, &b, &output_gradient});
        ensureDisjointTensorTarget(grad_b, target, shape, grad_dtype, {&a, &b, &output_gradient, &grad_a});
        auto &da_gpu = Gpu_Tensor_Impl::castToGpu(*grad_a.getImplementation());
        auto &db_gpu = Gpu_Tensor_Impl::castToGpu(*grad_b.getImplementation());
        ensureDisjointGpuStorage(da_gpu, total_elements, grad_dtype, {buf_dy.get(), buf_a.get(), buf_b.get()});
        ensureDisjointGpuStorage(db_gpu, total_elements, grad_dtype, {buf_dy.get(), buf_a.get(), buf_b.get(), da_gpu.getStorageBuffer().get()});

        effective_a->pushToGraph(is_fp16 ? Compute_Pipeline::SWIGLU_BACKWARD_FP16 : Compute_Pipeline::SWIGLU_BACKWARD,
                                 {buf_dy, buf_a, buf_b, da_gpu.getStorageBuffer(), db_gpu.getStorageBuffer()},
                                 pcs, wg_x, wg_y, 1, &engine);
        da_gpu.clearHostCache();
        da_gpu.invalidateFp16Cache();
        db_gpu.clearHostCache();
        db_gpu.invalidateFp16Cache();
    }
}

void fusedSwiGLUForward(const Tensor &input, Tensor &output, const SwiGLU_Config &config)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    Execution_Target target = input.getExecutionTarget();
    validateOperand(input, target, "ops::llm::fusedSwiGLUForward", "input");

    const Shape &in_shape = input.getShape();
    size_t rank = in_shape.getRank();
    if (rank == 0)
    {
        throw std::invalid_argument("ops::llm::fusedSwiGLUForward: input rank must be >= 1");
    }
    size_t total_cols = in_shape[rank - 1];
    if (total_cols % 2 != 0)
    {
        throw std::invalid_argument("ops::llm::fusedSwiGLUForward: last dimension must be even");
    }
    size_t half_dim = total_cols / 2;
    auto in_dims = in_shape.getDimensions();
    std::vector<size_t> out_dims(in_dims.begin(), in_dims.end());
    out_dims.back() = half_dim;
    Shape out_shape(std::move(out_dims));

    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;
    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (input.getDataType() == Data_Type::FLOAT16 || resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type out_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    size_t total_in = input.getTotalElements();
    if (total_in == 0 || half_dim == 0)
    {
        ensureDisjointTensorTarget(output, target, out_shape, out_dtype, {&input});
        if (target != Execution_Target::CPU)
        {
            auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            out_gpu.clearHostCache();
            out_gpu.invalidateFp16Cache();
        }
        return;
    }

    size_t rows = total_in / total_cols;

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        ensureDisjointTensorTarget(output, target, out_shape, out_dtype, {&input});
        std::vector<float> out_vals(rows * half_dim, 0.0f);

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < half_dim; ++c)
            {
                float a = in_data[r * total_cols + c];
                float b = in_data[r * total_cols + half_dim + c];
                float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a, -85.0f, 85.0f)));
                float silu_a = a * sig_a;
                out_vals[r * half_dim + c] = silu_a * b;
            }
        }
        writeCpuOutput(output, out_shape, out_dtype, out_vals);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (rows * half_dim > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::llm::fusedSwiGLUForward: output size exceeds uint32_t limit");
        }
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
        std::shared_ptr<Gpu_Tensor_Impl> casted_self;
        if (!is_fp16 && effective_self->getDataType() == Data_Type::FLOAT16)
        {
            casted_self = std::make_shared<Gpu_Tensor_Impl>(effective_self->getShape(), Data_Type::FLOAT32);
            effective_self->to(Data_Type::FLOAT32, *casted_self);
            effective_self = casted_self.get();
        }
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        uint32_t total = static_cast<uint32_t>(rows * half_dim);
        struct Fused_Swiglu_Constants
        {
            uint32_t total_elements;
            uint32_t half_dim;
        } pcs{total, static_cast<uint32_t>(half_dim)};
        validatePushConstants<Fused_Swiglu_Constants>();

        ensureDisjointTensorTarget(output, target, out_shape, out_dtype, {&input});
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        ensureDisjointGpuStorage(out_gpu, total, out_dtype, {buf_self.get()});
        const size_t work_items = (is_fp16 && (half_dim % 4 == 0)) ? ((rows * half_dim) / 4) : (rows * half_dim);
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((work_items + 255) / 256);

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::FUSED_SWIGLU_FORWARD_FP16 : Compute_Pipeline::FUSED_SWIGLU_FORWARD,
                                    {buf_self, out_gpu.getStorageBuffer()},
                                    pcs, wg_x, wg_y, 1, &engine);
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
    }
}

void fusedSwiGLUForward(const Tensor &input, Tensor &output)
{
    fusedSwiGLUForward(input, output, SwiGLU_Config{});
}

void fusedSwiGLUBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient,
                         const SwiGLU_Config &config)
{
    auto in_impl = input.getImplementation();
    auto dy_impl = output_gradient.getImplementation();
    if (!in_impl || !dy_impl) return;

    Execution_Target target = input.getExecutionTarget();
    validateOperand(input, target, "ops::llm::fusedSwiGLUBackward", "input");
    validateOperand(output_gradient, target, "ops::llm::fusedSwiGLUBackward", "output_gradient");

    const Shape &in_shape = input.getShape();
    size_t rank = in_shape.getRank();
    if (rank == 0)
    {
        throw std::invalid_argument("ops::llm::fusedSwiGLUBackward: input rank must be >= 1");
    }
    size_t total_cols = in_shape[rank - 1];
    if (total_cols % 2 != 0)
    {
        throw std::invalid_argument("ops::llm::fusedSwiGLUBackward: last dimension must be even");
    }
    size_t half_dim = total_cols / 2;
    auto in_dims = in_shape.getDimensions();
    std::vector<size_t> expected_dy_dims(in_dims.begin(), in_dims.end());
    expected_dy_dims.back() = half_dim;
    Shape expected_dy_shape(std::move(expected_dy_dims));
    if (output_gradient.getShape() != expected_dy_shape)
    {
        throw std::invalid_argument("ops::llm::fusedSwiGLUBackward: output_gradient shape mismatch");
    }

    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;
    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (input.getDataType() == Data_Type::FLOAT16 || output_gradient.getDataType() == Data_Type::FLOAT16 || resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type grad_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    size_t total_in = input.getTotalElements();
    if (total_in == 0 || half_dim == 0)
    {
        ensureDisjointTensorTarget(input_gradient, target, in_shape, grad_dtype, {&input, &output_gradient});
        if (target != Execution_Target::CPU)
        {
            auto &dx_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
            dx_gpu.clearHostCache();
            dx_gpu.invalidateFp16Cache();
        }
        return;
    }

    size_t rows = total_in / total_cols;

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        const auto dy_data = output_gradient.getData();
        ensureDisjointTensorTarget(input_gradient, target, in_shape, grad_dtype, {&input, &output_gradient});
        std::vector<float> dx_vals(total_in, 0.0f);

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

                dx_vals[r * total_cols + c] = dy * b * d_silu_a;
                dx_vals[r * total_cols + half_dim + c] = dy * silu_a;
            }
        }
        writeCpuOutput(input_gradient, in_shape, grad_dtype, dx_vals);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (total_in > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::llm::fusedSwiGLUBackward: total_in exceeds uint32_t limit");
        }
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &dy_gpu = Gpu_Tensor_Impl::castToGpu(*dy_impl);
        auto contig_dy = dy_gpu.ensureContiguousSelf();
        const auto *effective_dy = contig_dy ? contig_dy.get() : &dy_gpu;

        std::shared_ptr<Gpu_Tensor_Impl> casted_self, casted_dy;
        if (!is_fp16)
        {
            if (effective_self->getDataType() == Data_Type::FLOAT16)
            {
                casted_self = std::make_shared<Gpu_Tensor_Impl>(effective_self->getShape(), Data_Type::FLOAT32);
                effective_self->to(Data_Type::FLOAT32, *casted_self);
                effective_self = casted_self.get();
            }
            if (effective_dy->getDataType() == Data_Type::FLOAT16)
            {
                casted_dy = std::make_shared<Gpu_Tensor_Impl>(effective_dy->getShape(), Data_Type::FLOAT32);
                effective_dy->to(Data_Type::FLOAT32, *casted_dy);
                effective_dy = casted_dy.get();
            }
        }

        uint32_t total = static_cast<uint32_t>(rows * half_dim);
        struct Fused_Swiglu_Constants
        {
            uint32_t total_elements;
            uint32_t half_dim;
        } pcs{total, static_cast<uint32_t>(half_dim)};
        validatePushConstants<Fused_Swiglu_Constants>();

        auto buf_dy = is_fp16 ? effective_dy->getEffectiveFp16Storage() : effective_dy->getStorageBuffer();
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        ensureDisjointTensorTarget(input_gradient, target, in_shape, grad_dtype, {&input, &output_gradient});
        auto &dx_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
        ensureDisjointGpuStorage(dx_gpu, total_in, grad_dtype, {buf_self.get(), buf_dy.get()});
        const size_t work_items = (is_fp16 && (half_dim % 8 == 0))
            ? ((rows * half_dim) / 8)
            : ((is_fp16 && (half_dim % 4 == 0)) ? ((rows * half_dim) / 4) : (rows * half_dim));
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((work_items + 255) / 256);

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::FUSED_SWIGLU_BACKWARD_FP16 : Compute_Pipeline::FUSED_SWIGLU_BACKWARD,
                                    {buf_dy, buf_self, dx_gpu.getStorageBuffer()},
                                    pcs, wg_x, wg_y, 1, &engine);
        dx_gpu.clearHostCache();
        dx_gpu.invalidateFp16Cache();
    }
}

void fusedSwiGLUBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient)
{
    fusedSwiGLUBackward(input, output_gradient, input_gradient, SwiGLU_Config{});
}

}

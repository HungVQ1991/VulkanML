#include "ops/normalization/rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::normalization
{

using ops::internal::ensureDisjointGpuStorage;
using ops::internal::ensureTensorTarget;
using ops::internal::pushGpuFill;
using ops::internal::validatePushConstants;

static std::vector<float> *prepareWritableFp32(Cpu_Tensor_Impl &cpu_t, size_t count, bool preserve_existing = false)
{
    cpu_t.setByteOffset(0);
    cpu_t.setDataType(Data_Type::FLOAT32);
    auto &buf = cpu_t.getStorageBuffer();
    if (!buf)
    {
        buf = std::make_shared<std::vector<float>>(count, 0.0f);
    }
    else if (buf.use_count() > 1)
    {
        buf = preserve_existing ? std::make_shared<std::vector<float>>(*buf)
                                : std::make_shared<std::vector<float>>(count, 0.0f);
    }
    if (buf->size() < count)
    {
        buf->resize(count, 0.0f);
    }
    cpu_t.getStorageBufferFp16().reset();
    cpu_t.getStorageBufferU32().reset();
    return buf.get();
}

static void writeCpuOutput(Cpu_Tensor_Impl &out_cpu, const Shape &out_shape, Data_Type dtype, std::vector<float> values)
{
    out_cpu.setShape(out_shape);
    out_cpu.setByteOffset(0);
    out_cpu.setDataType(dtype);
    const size_t total = values.size();
    if (dtype == Data_Type::FLOAT16)
    {
        std::vector<float> clamped(total);
        for (size_t i = 0; i < total; ++i)
        {
            clamped[i] = std::clamp(values[i], -65504.0f, 65504.0f);
        }
        auto &fp16_buf = out_cpu.getStorageBufferFp16();
        if (!fp16_buf || fp16_buf.use_count() > 1 || fp16_buf->size() != total)
        {
            fp16_buf = std::make_shared<std::vector<float16_t>>(total);
        }
        if (total > 0)
        {
            convertFp32ToFp16(clamped.data(), fp16_buf->data(), total);
        }
        out_cpu.getStorageBuffer().reset();
        out_cpu.getStorageBufferU32().reset();
    }
    else
    {
        auto &fp32_buf = out_cpu.getStorageBuffer();
        if (!fp32_buf || fp32_buf.use_count() > 1 || fp32_buf->size() != total)
        {
            fp32_buf = std::make_shared<std::vector<float>>(std::move(values));
        }
        else
        {
            *fp32_buf = std::move(values);
        }
        out_cpu.getStorageBufferFp16().reset();
        out_cpu.getStorageBufferU32().reset();
    }
}

static void dispatchRmsNormGammaGrad(const Gpu_Tensor_Impl &anchor,
                                     const std::shared_ptr<gpu::vector> &buf_grad,
                                     const std::shared_ptr<gpu::vector> &buf_self,
                                     const std::shared_ptr<gpu::vector> &buf_inv_rms,
                                     const std::shared_ptr<gpu::vector> &buf_g_grad,
                                     uint32_t b_count,
                                     uint32_t f_dim,
                                     bool effective_accumulate,
                                     bool is_fp16)
{
    const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
    if (!is_fp16)
    {
        const uint64_t gamma_wg_x_u64 = (static_cast<uint64_t>(f_dim) + 15ULL) / 16ULL;
        if (gamma_wg_x_u64 > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::dispatchRmsNormGammaGrad: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }
        struct RMSNorm_Backward_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            uint32_t accumulate;
        } pcs{b_count, f_dim, effective_accumulate ? 1u : 0u};
        validatePushConstants<RMSNorm_Backward_Constants>();
        anchor.pushToGraph(Compute_Pipeline::RMSNORM_GAMMA_GRAD,
                           {buf_grad, buf_self, buf_inv_rms, buf_g_grad},
                           pcs, static_cast<uint32_t>(gamma_wg_x_u64), 1, 1);
        return;
    }

    struct RMSNorm_Gamma_Grad_Fp16_Constants
    {
        uint32_t batch_size;
        uint32_t dim;
        uint32_t accumulate;
        uint32_t split_count;
        uint32_t pass_mode;
    };
    validatePushConstants<RMSNorm_Gamma_Grad_Fp16_Constants>();

    const bool can_split = ((f_dim & 7u) == 0u) && (b_count >= 128u);
    if (can_split)
    {
        const uint64_t wg_x_v4 = (static_cast<uint64_t>(f_dim) + 127ULL) / 128ULL;
        const uint64_t wg_x_red = (static_cast<uint64_t>(f_dim) + 15ULL) / 16ULL;
        if (wg_x_v4 > max_wg_x || wg_x_red > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::dispatchRmsNormGammaGrad: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }
        const uint32_t max_wg_y = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[1];
        const uint32_t target_splits = std::max<uint32_t>(1u, static_cast<uint32_t>((64ULL + wg_x_v4 - 1ULL) / wg_x_v4));
        const uint32_t split_count = std::min<uint32_t>({target_splits, 64u, std::max<uint32_t>(1u, b_count / 16u), max_wg_y});
        if (split_count > 1u)
        {
            const size_t partial_elems = static_cast<size_t>(split_count) * static_cast<size_t>(f_dim);
            auto partials_buf = Execution_Engine::getInstance().acquireReusableScratchBuffer(
                Execution_Engine::SCRATCH_RMSNORM_GAMMA_PARTIALS,
                partial_elems * sizeof(float),
                Data_Type::FLOAT32,
                partial_elems);

            RMSNorm_Gamma_Grad_Fp16_Constants pass1_pcs{b_count, f_dim, 0u, split_count, 1u};
            anchor.pushToGraph(Compute_Pipeline::RMSNORM_GAMMA_GRAD_FP16,
                               {buf_grad, buf_self, buf_inv_rms, partials_buf},
                               pass1_pcs, static_cast<uint32_t>(wg_x_v4), split_count, 1);

            RMSNorm_Gamma_Grad_Fp16_Constants pass2_pcs{b_count, f_dim, effective_accumulate ? 1u : 0u, split_count, 2u};
            anchor.pushToGraph(Compute_Pipeline::RMSNORM_GAMMA_GRAD_FP16,
                               {partials_buf, partials_buf, partials_buf, buf_g_grad},
                               pass2_pcs, static_cast<uint32_t>(wg_x_red), 1, 1);
            return;
        }
    }

    const uint64_t gamma_wg_x_u64 = ((f_dim & 1u) == 0u)
        ? ((static_cast<uint64_t>(f_dim) + 31ULL) / 32ULL)
        : ((static_cast<uint64_t>(f_dim) + 15ULL) / 16ULL);
    if (gamma_wg_x_u64 > max_wg_x)
    {
        throw std::overflow_error("ops::normalization::dispatchRmsNormGammaGrad: Workgroup count exceeds maxComputeWorkGroupCount[0]");
    }
    RMSNorm_Gamma_Grad_Fp16_Constants single_pcs{b_count, f_dim, effective_accumulate ? 1u : 0u, 1u, 0u};
    anchor.pushToGraph(Compute_Pipeline::RMSNORM_GAMMA_GRAD_FP16,
                       {buf_grad, buf_self, buf_inv_rms, buf_g_grad},
                       single_pcs, static_cast<uint32_t>(gamma_wg_x_u64), 1, 1);
}

static std::shared_ptr<gpu::vector> getGpuFloat32ReadBuffer(const Gpu_Tensor_Impl &src_gpu, const Gpu_Tensor_Impl &anchor)
{
    if (src_gpu.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization: TERNARY_INT2 is not supported");
    }
    auto contig = src_gpu.ensureContiguousSelf();
    const auto *effective = contig ? contig.get() : &src_gpu;
    size_t total = effective->getTotalElements();
    if (total == 0)
    {
        return effective->getStorageBuffer();
    }
    if (!effective->getStorageBuffer())
    {
        throw std::runtime_error("ops::normalization: Missing GPU storage buffer");
    }
    if (total > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization: Tensor element count exceeds uint32_t dispatch limit");
    }
    if (effective->getDataType() == Data_Type::FLOAT16)
    {
        auto &ctx = Execution_Engine::getInstance().getContext();
        auto tmp = std::make_shared<gpu::vector>(ctx, total, Data_Type::FLOAT32);
        uint32_t total_u32 = static_cast<uint32_t>(total);
        auto [wg_x, wg_y] = ctx.compute2DWorkgroupGrid((static_cast<uint64_t>(total_u32) + 255ULL) / 256ULL);
        anchor.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {effective->getStorageBuffer(), tmp}, total_u32, wg_x, wg_y, 1);
        return tmp;
    }
    return effective->getStorageBuffer();
}

void rmsNormForward(const Tensor &input, const Tensor &gamma, Tensor &inv_rms,
                    Tensor &output, float epsilon)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    if (!in_impl || !gamma_impl)
    {
        Logger::logMessage("ops::normalization::rmsNormForward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (gamma.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::rmsNormForward: Execution_Target mismatch");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 || gamma.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::rmsNormForward: TERNARY_INT2 is not supported");
    }

    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());
    if (gamma.getTotalElements() < f_dim)
    {
        throw std::invalid_argument("ops::normalization::rmsNormForward: gamma size smaller than feature dimension");
    }

    if (target == Execution_Target::CPU)
    {
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(output, target, Shape{b_count, f_dim}, out_dtype);
            ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);
            return;
        }

        std::vector<float> in_data = input.getData();
        std::vector<float> gamma_data = gamma.getData();

        ensureTensorTarget(output, target, Shape{b_count, f_dim}, out_dtype);
        ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);

        auto &inv_rms_cpu = Cpu_Tensor_Impl::castToCpu(*inv_rms.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        auto *inv_rms_vec = prepareWritableFp32(inv_rms_cpu, b_count, false);

        std::vector<float> out_vals(static_cast<size_t>(b_count) * f_dim, 0.0f);
        for (size_t i = 0; i < b_count; ++i)
        {
            float sum_sq = 0.0f;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float val = in_data[i * f_dim + j];
                sum_sq += val * val;
            }
            float mean_sq = sum_sq / static_cast<float>(f_dim);
            float inv_val = 1.0f / std::sqrt(mean_sq + epsilon);
            (*inv_rms_vec)[i] = inv_val;

            for (size_t j = 0; j < f_dim; ++j)
            {
                out_vals[i * f_dim + j] = in_data[i * f_dim + j] * inv_val * gamma_data[j];
            }
        }
        writeCpuOutput(out_cpu, Shape{b_count, f_dim}, out_dtype, std::move(out_vals));
    }
    else
    {
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || gamma.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(output, target, Shape{b_count, f_dim}, out_dtype);
            ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);
            Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).invalidateFp16Cache();
            Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation()).invalidateFp16Cache();
            return;
        }

        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        auto contig_gamma = gamma_gpu.ensureContiguousSelf();
        const auto *effective_gamma = contig_gamma ? contig_gamma.get() : &gamma_gpu;

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_gamma = is_fp16 ? effective_gamma->getEffectiveFp16Storage() : effective_gamma->getStorageBuffer();

        ensureTensorTarget(output, target, Shape{b_count, f_dim}, out_dtype);
        ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);

        auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        ensureDisjointGpuStorage(out_gpu, static_cast<size_t>(b_count) * f_dim, out_dtype, {buf_gamma.get()});
        ensureDisjointGpuStorage(inv_rms_gpu, static_cast<size_t>(b_count), Data_Type::FLOAT32, {buf_self.get(), buf_gamma.get(), out_gpu.getStorageBuffer().get()});

        struct RmsNorm_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            float epsilon;
        } pcs{b_count, f_dim, epsilon};
        validatePushConstants<RmsNorm_Constants>();

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::RMSNORM_FP16 : Compute_Pipeline::RMSNORM,
                                    {buf_self, buf_gamma, out_gpu.getStorageBuffer(), inv_rms_gpu.getStorageBuffer()},
                                    pcs, b_count, 1, 1);

        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
        inv_rms_gpu.clearHostCache();
        inv_rms_gpu.invalidateFp16Cache();
    }
}

void rmsNormBackward(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                     const Tensor &inv_rms, Tensor &gamma_gradient,
                     Tensor &input_gradient, bool accumulate_gamma)
{
    auto in_impl = input.getImplementation();
    auto out_grad_impl = output_gradient.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto inv_rms_impl = inv_rms.getImplementation();
    if (!in_impl || !out_grad_impl || !gamma_impl || !inv_rms_impl)
    {
        Logger::logMessage("ops::normalization::rmsNormBackward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (output_gradient.getExecutionTarget() != target ||
        gamma.getExecutionTarget() != target ||
        inv_rms.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::rmsNormBackward: Execution_Target mismatch");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        inv_rms.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::rmsNormBackward: TERNARY_INT2 is not supported");
    }

    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());
    if (gamma.getTotalElements() < f_dim || inv_rms.getTotalElements() < b_count ||
        output_gradient.getTotalElements() < static_cast<size_t>(b_count) * f_dim)
    {
        throw std::invalid_argument("ops::normalization::rmsNormBackward: Dimension mismatch");
    }

    if (target == Execution_Target::CPU)
    {
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        Data_Type in_grad_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);
            if (f_dim > 0 && !accumulate_gamma)
            {
                auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), f_dim, false);
                std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
            }
            return;
        }

        std::vector<float> in_data = input.getData();
        std::vector<float> out_grad = output_gradient.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> inv_rms_data = inv_rms.getData();

        ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        auto *dg_vec = prepareWritableFp32(g_grad_cpu, f_dim, accumulate_gamma);
        if (!accumulate_gamma)
        {
            std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
        }

        for (size_t j = 0; j < f_dim; ++j)
        {
            float sum_dg = 0.0f;
            for (size_t i = 0; i < b_count; ++i)
            {
                sum_dg += out_grad[i * f_dim + j] * in_data[i * f_dim + j] * inv_rms_data[i];
            }
            (*dg_vec)[j] += sum_dg;
        }

        std::vector<float> dx_vals(static_cast<size_t>(b_count) * f_dim, 0.0f);
        float inv_dim = 1.0f / static_cast<float>(f_dim);
        for (size_t i = 0; i < b_count; ++i)
        {
            float inv_r = inv_rms_data[i];
            float s = 0.0f;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float x_hat = in_data[i * f_dim + j] * inv_r;
                s += out_grad[i * f_dim + j] * gamma_data[j] * x_hat;
            }
            for (size_t j = 0; j < f_dim; ++j)
            {
                float x_hat = in_data[i * f_dim + j] * inv_r;
                dx_vals[i * f_dim + j] = inv_r * (out_grad[i * f_dim + j] * gamma_data[j] - x_hat * (s * inv_dim));
            }
        }
        writeCpuOutput(in_grad_cpu, Shape{b_count, f_dim}, in_grad_dtype, std::move(dx_vals));
    }
    else
    {
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || output_gradient.getDataType() == Data_Type::FLOAT16 || gamma.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        Data_Type in_grad_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);
            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            if (f_dim > 0 && !accumulate_gamma)
            {
                const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
                pushGpuFill(in_gpu, g_grad_gpu.getStorageBuffer(), f_dim, 0.0f);
            }
            Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).invalidateFp16Cache();
            g_grad_gpu.clearHostCache();
            g_grad_gpu.invalidateFp16Cache();
            return;
        }

        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        auto contig_gamma = gamma_gpu.ensureContiguousSelf();
        const auto *effective_gamma = contig_gamma ? contig_gamma.get() : &gamma_gpu;

        const auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation());

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
        auto buf_gamma = is_fp16 ? effective_gamma->getEffectiveFp16Storage() : effective_gamma->getStorageBuffer();
        auto buf_inv_rms = getGpuFloat32ReadBuffer(inv_rms_gpu, *effective_self);

        Shape g_shape{1, f_dim};
        bool g_reallocated = false;
        {
            auto g_impl = gamma_gradient.getImplementation();
            if (!g_impl || gamma_gradient.getExecutionTarget() != target ||
                gamma_gradient.getDataType() != Data_Type::FLOAT32 || gamma_gradient.getShape() != g_shape ||
                !g_impl->isContiguous() || g_impl->getByteOffset() != 0)
            {
                g_reallocated = true;
            }
            else
            {
                const auto &g = Gpu_Tensor_Impl::castToGpu(*g_impl);
                size_t req_bytes = static_cast<size_t>(f_dim) * sizeof(float);
                if (req_bytes > 0 && (!g.getStorageBuffer() || g.getStorageBuffer()->getSizeBytes() < req_bytes))
                {
                    g_reallocated = true;
                }
            }
        }

        ensureTensorTarget(gamma_gradient, target, g_shape, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        ensureDisjointGpuStorage(
            in_grad_gpu,
            static_cast<size_t>(b_count) * f_dim,
            in_grad_dtype,
            {buf_grad.get(), buf_self.get(), buf_gamma.get(), buf_inv_rms.get()});
        if (g_grad_gpu.getStorageBuffer().get() == buf_grad.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_self.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_gamma.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_inv_rms.get() ||
            g_grad_gpu.getStorageBuffer().get() == in_grad_gpu.getStorageBuffer().get())
        {
            g_reallocated = true;
            g_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), static_cast<size_t>(f_dim), Data_Type::FLOAT32);
            g_grad_gpu.markGpuMutated();
        }

        bool effective_accumulate = (accumulate_gamma && !g_reallocated);
        struct RMSNorm_Backward_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            uint32_t accumulate;
        } pcs{b_count, f_dim, effective_accumulate ? 1u : 0u};
        validatePushConstants<RMSNorm_Backward_Constants>();

        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        if (b_count > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::rmsNormBackward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::RMSNORM_BACKWARD_FP16 : Compute_Pipeline::RMSNORM_BACKWARD,
                                    {buf_grad, buf_self, buf_gamma, buf_inv_rms, in_grad_gpu.getStorageBuffer()},
                                    pcs, b_count, 1, 1);
        dispatchRmsNormGammaGrad(*effective_self, buf_grad, buf_self, buf_inv_rms, g_grad_gpu.getStorageBuffer(),
                                 b_count, f_dim, effective_accumulate, is_fp16);

        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
        g_grad_gpu.clearHostCache();
        g_grad_gpu.invalidateFp16Cache();
    }
}

void residualScaleRmsNormForward(const Tensor &residual, const Tensor &branch, const Tensor &gamma,
                                 float scale, Tensor &out_residual, Tensor &inv_rms,
                                 Tensor &out_norm, float epsilon)
{
    auto res_impl = residual.getImplementation();
    auto br_impl = branch.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    if (!res_impl || !br_impl || !gamma_impl)
    {
        Logger::logMessage("ops::normalization::residualScaleRmsNormForward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = residual.getExecutionTarget();
    if (branch.getExecutionTarget() != target || gamma.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::residualScaleRmsNormForward: Execution_Target mismatch");
    }
    if (residual.getDataType() == Data_Type::TERNARY_INT2 ||
        branch.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::residualScaleRmsNormForward: TERNARY_INT2 is not supported");
    }

    uint32_t b_count = static_cast<uint32_t>(residual.getRows());
    uint32_t f_dim = static_cast<uint32_t>(residual.getColumns());
    if (branch.getRows() != b_count || branch.getColumns() != f_dim ||
        residual.getTotalElements() != branch.getTotalElements())
    {
        throw std::invalid_argument("ops::normalization::residualScaleRmsNormForward: residual and branch shapes must match");
    }
    if (gamma.getTotalElements() < f_dim)
    {
        throw std::invalid_argument("ops::normalization::residualScaleRmsNormForward: gamma size smaller than feature dimension");
    }

    if (target == Execution_Target::CPU)
    {
        bool is_fp16 = (residual.getDataType() == Data_Type::FLOAT16 || branch.getDataType() == Data_Type::FLOAT16);
        Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(out_residual, target, Shape{b_count, f_dim}, out_dtype);
            ensureTensorTarget(out_norm, target, Shape{b_count, f_dim}, out_dtype);
            ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);
            return;
        }

        std::vector<float> res_data = residual.getData();
        std::vector<float> br_data = branch.getData();
        std::vector<float> gamma_data = gamma.getData();

        ensureTensorTarget(out_residual, target, Shape{b_count, f_dim}, out_dtype);
        ensureTensorTarget(out_norm, target, Shape{b_count, f_dim}, out_dtype);
        ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);

        auto &out_res_cpu = Cpu_Tensor_Impl::castToCpu(*out_residual.getImplementation());
        auto &inv_rms_cpu = Cpu_Tensor_Impl::castToCpu(*inv_rms.getImplementation());
        auto &out_norm_cpu = Cpu_Tensor_Impl::castToCpu(*out_norm.getImplementation());
        auto *inv_rms_vec = prepareWritableFp32(inv_rms_cpu, b_count, false);

        const size_t total = static_cast<size_t>(b_count) * f_dim;
        std::vector<float> res_vals(total, 0.0f);
        std::vector<float> norm_vals(total, 0.0f);
        for (size_t i = 0; i < b_count; ++i)
        {
            float sum_sq = 0.0f;
            const size_t row_off = i * f_dim;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float h = res_data[row_off + j] + scale * br_data[row_off + j];
                if (is_fp16)
                {
                    h = std::clamp(h, -65504.0f, 65504.0f);
                }
                res_vals[row_off + j] = h;
                sum_sq += h * h;
            }
            float mean_sq = sum_sq / static_cast<float>(f_dim);
            float inv_val = 1.0f / std::sqrt(mean_sq + epsilon);
            (*inv_rms_vec)[i] = inv_val;

            for (size_t j = 0; j < f_dim; ++j)
            {
                norm_vals[row_off + j] = res_vals[row_off + j] * inv_val * gamma_data[j];
            }
        }
        writeCpuOutput(out_res_cpu, Shape{b_count, f_dim}, out_dtype, std::move(res_vals));
        writeCpuOutput(out_norm_cpu, Shape{b_count, f_dim}, out_dtype, std::move(norm_vals));
    }
    else
    {
        bool is_fp16 = (residual.getDataType() == Data_Type::FLOAT16 ||
                        branch.getDataType() == Data_Type::FLOAT16 ||
                        gamma.getDataType() == Data_Type::FLOAT16 ||
                        Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(out_residual, target, Shape{b_count, f_dim}, out_dtype);
            ensureTensorTarget(out_norm, target, Shape{b_count, f_dim}, out_dtype);
            ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);
            auto &out_res_gpu = Gpu_Tensor_Impl::castToGpu(*out_residual.getImplementation());
            auto &out_norm_gpu = Gpu_Tensor_Impl::castToGpu(*out_norm.getImplementation());
            auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation());
            out_res_gpu.clearHostCache();
            out_res_gpu.invalidateFp16Cache();
            out_norm_gpu.clearHostCache();
            out_norm_gpu.invalidateFp16Cache();
            inv_rms_gpu.clearHostCache();
            inv_rms_gpu.invalidateFp16Cache();
            return;
        }

        const auto &res_gpu = Gpu_Tensor_Impl::castToGpu(*res_impl);
        auto contig_res = res_gpu.ensureContiguousSelf();
        const auto *effective_res = contig_res ? contig_res.get() : &res_gpu;

        const auto &br_gpu = Gpu_Tensor_Impl::castToGpu(*br_impl);
        auto contig_br = br_gpu.ensureContiguousSelf();
        const auto *effective_br = contig_br ? contig_br.get() : &br_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_impl);
        auto contig_gamma = gamma_gpu.ensureContiguousSelf();
        const auto *effective_gamma = contig_gamma ? contig_gamma.get() : &gamma_gpu;

        auto buf_res = is_fp16 ? effective_res->getEffectiveFp16Storage() : effective_res->getStorageBuffer();
        auto buf_br = is_fp16 ? effective_br->getEffectiveFp16Storage() : effective_br->getStorageBuffer();
        auto buf_gamma = is_fp16 ? effective_gamma->getEffectiveFp16Storage() : effective_gamma->getStorageBuffer();

        ensureTensorTarget(out_residual, target, Shape{b_count, f_dim}, out_dtype);
        ensureTensorTarget(out_norm, target, Shape{b_count, f_dim}, out_dtype);
        ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);

        auto &out_res_gpu = Gpu_Tensor_Impl::castToGpu(*out_residual.getImplementation());
        auto &out_norm_gpu = Gpu_Tensor_Impl::castToGpu(*out_norm.getImplementation());
        auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation());
        ensureDisjointGpuStorage(out_res_gpu, static_cast<size_t>(b_count) * f_dim, out_dtype, {buf_gamma.get()});
        ensureDisjointGpuStorage(out_norm_gpu, static_cast<size_t>(b_count) * f_dim, out_dtype, {buf_gamma.get(), out_res_gpu.getStorageBuffer().get()});
        ensureDisjointGpuStorage(inv_rms_gpu, static_cast<size_t>(b_count), Data_Type::FLOAT32, {buf_res.get(), buf_br.get(), buf_gamma.get(), out_res_gpu.getStorageBuffer().get(), out_norm_gpu.getStorageBuffer().get()});

        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        if (b_count > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::residualScaleRmsNormForward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        struct Residual_Scale_RmsNorm_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            float scale;
            float epsilon;
        } pcs{b_count, f_dim, scale, epsilon};
        validatePushConstants<Residual_Scale_RmsNorm_Constants>();

        effective_res->pushToGraph(is_fp16 ? Compute_Pipeline::RESIDUAL_SCALE_RMSNORM_FP16 : Compute_Pipeline::RESIDUAL_SCALE_RMSNORM,
                                   {buf_res, buf_br, buf_gamma, out_res_gpu.getStorageBuffer(), out_norm_gpu.getStorageBuffer(), inv_rms_gpu.getStorageBuffer()},
                                   pcs, b_count, 1, 1);

        out_res_gpu.clearHostCache();
        out_res_gpu.invalidateFp16Cache();
        out_norm_gpu.clearHostCache();
        out_norm_gpu.invalidateFp16Cache();
        inv_rms_gpu.clearHostCache();
        inv_rms_gpu.invalidateFp16Cache();
    }
}

void rmsNormBackwardResidual(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                             const Tensor &inv_rms, const Tensor &residual_grad,
                             Tensor &gamma_gradient, Tensor &out_sum,
                             Tensor *out_scaled, float scale,
                             bool accumulate_gamma)
{
    auto in_impl = input.getImplementation();
    auto out_grad_impl = output_gradient.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto inv_rms_impl = inv_rms.getImplementation();
    auto res_grad_impl = residual_grad.getImplementation();
    if (!in_impl || !out_grad_impl || !gamma_impl || !inv_rms_impl || !res_grad_impl)
    {
        Logger::logMessage("ops::normalization::rmsNormBackwardResidual: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (output_gradient.getExecutionTarget() != target ||
        gamma.getExecutionTarget() != target ||
        inv_rms.getExecutionTarget() != target ||
        residual_grad.getExecutionTarget() != target ||
        (out_scaled != nullptr && out_scaled->getImplementation() && out_scaled->getExecutionTarget() != target))
    {
        throw std::invalid_argument("ops::normalization::rmsNormBackwardResidual: Execution_Target mismatch");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        inv_rms.getDataType() == Data_Type::TERNARY_INT2 ||
        residual_grad.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::rmsNormBackwardResidual: TERNARY_INT2 is not supported");
    }

    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());
    const size_t expected_elems = static_cast<size_t>(b_count) * f_dim;
    if (gamma.getTotalElements() < f_dim || inv_rms.getTotalElements() < b_count ||
        output_gradient.getRows() != b_count || output_gradient.getColumns() != f_dim ||
        output_gradient.getTotalElements() != expected_elems ||
        residual_grad.getRows() != b_count || residual_grad.getColumns() != f_dim ||
        residual_grad.getTotalElements() != expected_elems)
    {
        throw std::invalid_argument("ops::normalization::rmsNormBackwardResidual: Dimension mismatch");
    }

    if (target == Execution_Target::CPU)
    {
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 ||
                        output_gradient.getDataType() == Data_Type::FLOAT16 ||
                        residual_grad.getDataType() == Data_Type::FLOAT16);
        Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(out_sum, target, Shape{b_count, f_dim}, out_dtype);
            if (out_scaled != nullptr)
            {
                ensureTensorTarget(*out_scaled, target, Shape{b_count, f_dim}, out_dtype);
            }
            if (f_dim > 0 && !accumulate_gamma)
            {
                auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), f_dim, false);
                std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
            }
            return;
        }

        std::vector<float> in_data = input.getData();
        std::vector<float> out_grad = output_gradient.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> inv_rms_data = inv_rms.getData();
        std::vector<float> res_grad_data = residual_grad.getData();

        ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(out_sum, target, Shape{b_count, f_dim}, out_dtype);
        if (out_scaled != nullptr)
        {
            ensureTensorTarget(*out_scaled, target, Shape{b_count, f_dim}, out_dtype);
        }

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &out_sum_cpu = Cpu_Tensor_Impl::castToCpu(*out_sum.getImplementation());

        auto *dg_vec = prepareWritableFp32(g_grad_cpu, f_dim, accumulate_gamma);
        if (!accumulate_gamma)
        {
            std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
        }

        for (size_t j = 0; j < f_dim; ++j)
        {
            float sum_dg = 0.0f;
            for (size_t i = 0; i < b_count; ++i)
            {
                sum_dg += out_grad[i * f_dim + j] * in_data[i * f_dim + j] * inv_rms_data[i];
            }
            (*dg_vec)[j] += sum_dg;
        }

        std::vector<float> sum_vals(expected_elems, 0.0f);
        std::vector<float> scaled_vals;
        if (out_scaled != nullptr)
        {
            scaled_vals.resize(expected_elems, 0.0f);
        }
        float inv_dim = 1.0f / static_cast<float>(f_dim);
        for (size_t i = 0; i < b_count; ++i)
        {
            float inv_r = inv_rms_data[i];
            const size_t row_off = i * f_dim;
            float s_dot = 0.0f;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float x_hat = in_data[row_off + j] * inv_r;
                s_dot += out_grad[row_off + j] * gamma_data[j] * x_hat;
            }
            float coeff = s_dot * inv_dim;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float x_hat = in_data[row_off + j] * inv_r;
                float dx = inv_r * (out_grad[row_off + j] * gamma_data[j] - x_hat * coeff);
                float total = res_grad_data[row_off + j] + dx;
                if (is_fp16)
                {
                    total = std::clamp(total, -65504.0f, 65504.0f);
                }
                sum_vals[row_off + j] = total;
                if (out_scaled != nullptr)
                {
                    scaled_vals[row_off + j] = is_fp16 ? std::clamp(total * scale, -65504.0f, 65504.0f) : (total * scale);
                }
            }
        }
        writeCpuOutput(out_sum_cpu, Shape{b_count, f_dim}, out_dtype, std::move(sum_vals));
        if (out_scaled != nullptr)
        {
            auto &out_scaled_cpu = Cpu_Tensor_Impl::castToCpu(*out_scaled->getImplementation());
            writeCpuOutput(out_scaled_cpu, Shape{b_count, f_dim}, out_dtype, std::move(scaled_vals));
        }
    }
    else
    {
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 ||
                        output_gradient.getDataType() == Data_Type::FLOAT16 ||
                        gamma.getDataType() == Data_Type::FLOAT16 ||
                        residual_grad.getDataType() == Data_Type::FLOAT16 ||
                        Execution_Engine::getInstance().isCooperativeMatrixEnabled());

        if (!is_fp16)
        {
            Tensor tmp_dx(target);
            rmsNormBackward(input, output_gradient, gamma, inv_rms, gamma_gradient, tmp_dx, accumulate_gamma);
            residual_grad.add(tmp_dx, out_sum);
            if (out_scaled != nullptr)
            {
                out_sum.mulScalar(scale, *out_scaled);
            }
            return;
        }

        if (b_count == 0 || f_dim == 0)
        {
            ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(out_sum, target, Shape{b_count, f_dim}, Data_Type::FLOAT16);
            if (out_scaled != nullptr)
            {
                ensureTensorTarget(*out_scaled, target, Shape{b_count, f_dim}, Data_Type::FLOAT16);
                Gpu_Tensor_Impl::castToGpu(*out_scaled->getImplementation()).clearHostCache();
                Gpu_Tensor_Impl::castToGpu(*out_scaled->getImplementation()).invalidateFp16Cache();
            }
            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            if (f_dim > 0 && !accumulate_gamma)
            {
                const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
                pushGpuFill(in_gpu, g_grad_gpu.getStorageBuffer(), f_dim, 0.0f);
            }
            Gpu_Tensor_Impl::castToGpu(*out_sum.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*out_sum.getImplementation()).invalidateFp16Cache();
            g_grad_gpu.clearHostCache();
            g_grad_gpu.invalidateFp16Cache();
            return;
        }

        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_impl);
        auto contig_gamma = gamma_gpu.ensureContiguousSelf();
        const auto *effective_gamma = contig_gamma ? contig_gamma.get() : &gamma_gpu;

        const auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms_impl);

        const auto &res_grad_gpu = Gpu_Tensor_Impl::castToGpu(*res_grad_impl);
        auto contig_res_grad = res_grad_gpu.ensureContiguousSelf();
        const auto *effective_res_grad = contig_res_grad ? contig_res_grad.get() : &res_grad_gpu;

        auto buf_self = effective_self->getEffectiveFp16Storage();
        auto buf_grad = effective_grad->getEffectiveFp16Storage();
        auto buf_gamma = effective_gamma->getEffectiveFp16Storage();
        auto buf_inv_rms = getGpuFloat32ReadBuffer(inv_rms_gpu, *effective_self);
        auto buf_res_grad = effective_res_grad->getEffectiveFp16Storage();

        Shape g_shape{1, f_dim};
        bool g_reallocated = false;
        {
            auto g_impl = gamma_gradient.getImplementation();
            if (!g_impl || gamma_gradient.getExecutionTarget() != target ||
                gamma_gradient.getDataType() != Data_Type::FLOAT32 || gamma_gradient.getShape() != g_shape ||
                !g_impl->isContiguous() || g_impl->getByteOffset() != 0)
            {
                g_reallocated = true;
            }
            else
            {
                const auto &g = Gpu_Tensor_Impl::castToGpu(*g_impl);
                size_t req_bytes = static_cast<size_t>(f_dim) * sizeof(float);
                if (req_bytes > 0 && (!g.getStorageBuffer() || g.getStorageBuffer()->getSizeBytes() < req_bytes))
                {
                    g_reallocated = true;
                }
            }
        }

        ensureTensorTarget(gamma_gradient, target, g_shape, Data_Type::FLOAT32);
        ensureTensorTarget(out_sum, target, Shape{b_count, f_dim}, Data_Type::FLOAT16);
        if (out_scaled != nullptr)
        {
            ensureTensorTarget(*out_scaled, target, Shape{b_count, f_dim}, Data_Type::FLOAT16);
        }

        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &out_sum_gpu = Gpu_Tensor_Impl::castToGpu(*out_sum.getImplementation());
        ensureDisjointGpuStorage(
            out_sum_gpu,
            static_cast<size_t>(b_count) * f_dim,
            Data_Type::FLOAT16,
            {buf_grad.get(), buf_self.get(), buf_gamma.get(), buf_inv_rms.get()});
        if (out_scaled != nullptr)
        {
            auto &out_scaled_gpu = Gpu_Tensor_Impl::castToGpu(*out_scaled->getImplementation());
            ensureDisjointGpuStorage(
                out_scaled_gpu,
                static_cast<size_t>(b_count) * f_dim,
                Data_Type::FLOAT16,
                {buf_grad.get(), buf_self.get(), buf_gamma.get(), buf_inv_rms.get(), buf_res_grad.get(), out_sum_gpu.getStorageBuffer().get()});
        }
        auto out_scaled_buf = (out_scaled != nullptr)
            ? Gpu_Tensor_Impl::castToGpu(*out_scaled->getImplementation()).getStorageBuffer()
            : out_sum_gpu.getStorageBuffer();
        if (g_grad_gpu.getStorageBuffer().get() == buf_grad.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_self.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_gamma.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_inv_rms.get() ||
            g_grad_gpu.getStorageBuffer().get() == buf_res_grad.get() ||
            g_grad_gpu.getStorageBuffer().get() == out_sum_gpu.getStorageBuffer().get() ||
            g_grad_gpu.getStorageBuffer().get() == out_scaled_buf.get())
        {
            g_reallocated = true;
            g_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), static_cast<size_t>(f_dim), Data_Type::FLOAT32);
            g_grad_gpu.markGpuMutated();
        }

        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        if (b_count > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::rmsNormBackwardResidual: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        struct RMSNorm_Backward_Residual_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            float scale;
            uint32_t write_scaled;
        } bwd_res_pcs{b_count, f_dim, scale, (out_scaled != nullptr) ? 1u : 0u};
        validatePushConstants<RMSNorm_Backward_Residual_Constants>();

        bool effective_accumulate = (accumulate_gamma && !g_reallocated);

        effective_self->pushToGraph(Compute_Pipeline::RMSNORM_BACKWARD_RESIDUAL_FP16,
                                    {buf_grad, buf_self, buf_gamma, buf_inv_rms, buf_res_grad, out_sum_gpu.getStorageBuffer(), out_scaled_buf},
                                    bwd_res_pcs, b_count, 1, 1);
        dispatchRmsNormGammaGrad(*effective_self, buf_grad, buf_self, buf_inv_rms, g_grad_gpu.getStorageBuffer(),
                                 b_count, f_dim, effective_accumulate, true);

        out_sum_gpu.clearHostCache();
        out_sum_gpu.invalidateFp16Cache();
        if (out_scaled != nullptr)
        {
            auto &out_scaled_gpu = Gpu_Tensor_Impl::castToGpu(*out_scaled->getImplementation());
            out_scaled_gpu.clearHostCache();
            out_scaled_gpu.invalidateFp16Cache();
        }
        g_grad_gpu.clearHostCache();
        g_grad_gpu.invalidateFp16Cache();
    }
}

}

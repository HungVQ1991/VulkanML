#include "ops/normalization/batchnorm.h"

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

void batchNormForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                      Tensor &running_mean, Tensor &running_variance,
                      Tensor &batch_mean, Tensor &batch_variance,
                      Tensor &normalized_input, Tensor &output,
                      float epsilon, float momentum, bool is_training)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto beta_impl = beta.getImplementation();
    if (!in_impl || !gamma_impl || !beta_impl)
    {
        Logger::logMessage("ops::normalization::batchNormForward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (gamma.getExecutionTarget() != target || beta.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::batchNormForward: Execution_Target mismatch");
    }
    if (!running_mean.getImplementation() || !running_variance.getImplementation())
    {
        throw std::invalid_argument("ops::normalization::batchNormForward: Null running stats implementation");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        beta.getDataType() == Data_Type::TERNARY_INT2 ||
        running_mean.getDataType() == Data_Type::TERNARY_INT2 ||
        running_variance.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::batchNormForward: TERNARY_INT2 is not supported");
    }

    if (running_mean.getExecutionTarget() != target) running_mean.setExecutionTarget(target);
    if (running_variance.getExecutionTarget() != target) running_variance.setExecutionTarget(target);

    if (input.getRows() > std::numeric_limits<uint32_t>::max() ||
        input.getColumns() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNormForward: Input dimensions exceed uint32_t");
    }
    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());
    if (running_mean.getTotalElements() < f_dim || running_variance.getTotalElements() < f_dim ||
        gamma.getTotalElements() < f_dim || beta.getTotalElements() < f_dim)
    {
        throw std::invalid_argument("ops::normalization::batchNormForward: Parameter or running stats size is smaller than feature dimension");
    }

    Shape feat_shape{b_count, f_dim};
    Shape stat_shape{1, f_dim};
    Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (b_count == 0 || f_dim == 0)
    {
        ensureTensorTarget(output, target, feat_shape, out_dtype);
        ensureTensorTarget(normalized_input, target, feat_shape, Data_Type::FLOAT32);
        if (is_training)
        {
            ensureTensorTarget(batch_mean, target, stat_shape, Data_Type::FLOAT32);
            ensureTensorTarget(batch_variance, target, stat_shape, Data_Type::FLOAT32);
        }
        if (target == Execution_Target::VULKAN_GPU)
        {
            Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).invalidateFp16Cache();
            Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation()).invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        std::vector<float> in_data = input.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> beta_data = beta.getData();
        std::vector<float> rm_vals = running_mean.getData();
        std::vector<float> rv_vals = running_variance.getData();

        ensureTensorTarget(output, target, feat_shape, out_dtype);
        ensureTensorTarget(normalized_input, target, feat_shape, Data_Type::FLOAT32);
        if (is_training)
        {
            ensureTensorTarget(batch_mean, target, stat_shape, Data_Type::FLOAT32);
            ensureTensorTarget(batch_variance, target, stat_shape, Data_Type::FLOAT32);
        }

        auto &rm_cpu = Cpu_Tensor_Impl::castToCpu(*running_mean.getImplementation());
        auto &rv_cpu = Cpu_Tensor_Impl::castToCpu(*running_variance.getImplementation());
        auto &norm_in_cpu = Cpu_Tensor_Impl::castToCpu(*normalized_input.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        std::vector<float> norm_vals(static_cast<size_t>(b_count) * f_dim, 0.0f);
        std::vector<float> out_vals(static_cast<size_t>(b_count) * f_dim, 0.0f);

        if (is_training)
        {
            auto &bm_cpu = Cpu_Tensor_Impl::castToCpu(*batch_mean.getImplementation());
            auto &bv_cpu = Cpu_Tensor_Impl::castToCpu(*batch_variance.getImplementation());
            auto *bm_vec = prepareWritableFp32(bm_cpu, f_dim, false);
            auto *bv_vec = prepareWritableFp32(bv_cpu, f_dim, false);
            std::fill(bm_vec->begin(), bm_vec->begin() + f_dim, 0.0f);
            std::fill(bv_vec->begin(), bv_vec->begin() + f_dim, 0.0f);

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    (*bm_vec)[j] += in_data[i * f_dim + j];
                }
            }
            float inv_b = 1.0f / static_cast<float>(b_count);
            for (size_t j = 0; j < f_dim; ++j)
            {
                (*bm_vec)[j] *= inv_b;
            }

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float diff = in_data[i * f_dim + j] - (*bm_vec)[j];
                    (*bv_vec)[j] += diff * diff;
                }
            }
            for (size_t j = 0; j < f_dim; ++j)
            {
                (*bv_vec)[j] *= inv_b;
                rm_vals[j] = (1.0f - momentum) * rm_vals[j] + momentum * (*bm_vec)[j];
                rv_vals[j] = (1.0f - momentum) * rv_vals[j] + momentum * (*bv_vec)[j];
            }
            writeCpuOutput(rm_cpu, running_mean.getShape(), running_mean.getDataType(), std::move(rm_vals));
            writeCpuOutput(rv_cpu, running_variance.getShape(), running_variance.getDataType(), std::move(rv_vals));

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float norm_val = (in_data[i * f_dim + j] - (*bm_vec)[j]) / std::sqrt((*bv_vec)[j] + epsilon);
                    norm_vals[i * f_dim + j] = norm_val;
                    out_vals[i * f_dim + j] = gamma_data[j] * norm_val + beta_data[j];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float norm_val = (in_data[i * f_dim + j] - rm_vals[j]) / std::sqrt(rv_vals[j] + epsilon);
                    norm_vals[i * f_dim + j] = norm_val;
                    out_vals[i * f_dim + j] = gamma_data[j] * norm_val + beta_data[j];
                }
            }
        }

        writeCpuOutput(norm_in_cpu, feat_shape, Data_Type::FLOAT32, std::move(norm_vals));
        writeCpuOutput(out_cpu, feat_shape, out_dtype, std::move(out_vals));
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_in = in_gpu.ensureContiguousSelf();
        const auto *effective_in = contig_in ? contig_in.get() : &in_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &beta_gpu = Gpu_Tensor_Impl::castToGpu(*beta.getImplementation());
        auto &rm_gpu = Gpu_Tensor_Impl::castToGpu(*running_mean.getImplementation());
        auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());

        if (is_training &&
            (running_mean.getDataType() != Data_Type::FLOAT32 || running_variance.getDataType() != Data_Type::FLOAT32 ||
             !rm_gpu.isContiguous() || rm_gpu.getByteOffset() != 0 ||
             !rv_gpu.isContiguous() || rv_gpu.getByteOffset() != 0))
        {
            throw std::invalid_argument("ops::normalization::batchNormForward: Training running stats on GPU must be contiguous FLOAT32");
        }

        Data_Type data_type = effective_in->getDataType();
        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        auto buf_in = is_fp16 ? effective_in->getEffectiveFp16Storage() : effective_in->getStorageBuffer();
        auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_in);
        auto buf_beta = getGpuFloat32ReadBuffer(beta_gpu, *effective_in);
        std::shared_ptr<gpu::vector> buf_rm;
        std::shared_ptr<gpu::vector> buf_rv;
        if (!is_training)
        {
            buf_rm = getGpuFloat32ReadBuffer(rm_gpu, *effective_in);
            buf_rv = getGpuFloat32ReadBuffer(rv_gpu, *effective_in);
        }

        ensureTensorTarget(output, target, feat_shape, out_dtype);
        ensureTensorTarget(normalized_input, target, feat_shape, Data_Type::FLOAT32);
        if (is_training)
        {
            ensureTensorTarget(batch_mean, target, stat_shape, Data_Type::FLOAT32);
            ensureTensorTarget(batch_variance, target, stat_shape, Data_Type::FLOAT32);
        }

        auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        const uint64_t total_elem_u64 = static_cast<uint64_t>(b_count) * f_dim;
        if (total_elem_u64 > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::normalization::batchNormForward: Total element count exceeds uint32_t");
        }
        const uint32_t total_elem_u32 = static_cast<uint32_t>(total_elem_u64);
        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        const uint64_t tf_wg_x_u64 = (total_elem_u64 + 255ULL) / 256ULL;
        if (f_dim > max_wg_x || tf_wg_x_u64 > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::batchNormForward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }
        const uint32_t tf_wg_x = static_cast<uint32_t>(tf_wg_x_u64);

        if (is_training)
        {
            auto &bm_gpu = Gpu_Tensor_Impl::castToGpu(*batch_mean.getImplementation());
            auto &bv_gpu = Gpu_Tensor_Impl::castToGpu(*batch_variance.getImplementation());

            struct Stats_Constants
            {
                uint32_t batch_size;
                uint32_t feature_dimension;
                float momentum;
            } stats{b_count, f_dim, momentum};
            validatePushConstants<Stats_Constants>();

            effective_in->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_STATS_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_STATS_FORWARD,
                                      {buf_in, bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), rm_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer()},
                                      stats, f_dim, 1, 1);

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t feature_dimension;
                float epsilon;
            } tf{total_elem_u32, f_dim, epsilon};
            validatePushConstants<Transform_Constants>();

            effective_in->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                                      {buf_in, bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), buf_gamma, buf_beta, out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                      tf, tf_wg_x, 1, 1);

            bm_gpu.clearHostCache();
            bm_gpu.invalidateFp16Cache();
            bv_gpu.clearHostCache();
            bv_gpu.invalidateFp16Cache();
            rm_gpu.clearHostCache();
            rm_gpu.invalidateFp16Cache();
            rv_gpu.clearHostCache();
            rv_gpu.invalidateFp16Cache();
        }
        else
        {
            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t feature_dimension;
                float epsilon;
            } tf{total_elem_u32, f_dim, epsilon};
            validatePushConstants<Transform_Constants>();

            effective_in->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                                      {buf_in, buf_rm, buf_rv, buf_gamma, buf_beta, out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                      tf, tf_wg_x, 1, 1);
        }

        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
        norm_in_gpu.clearHostCache();
        norm_in_gpu.invalidateFp16Cache();
    }
}

void batchNormBackward(const Tensor &input, const Tensor &output_gradient, const Tensor &gamma,
                       const Tensor &batch_variance, const Tensor &normalized_input,
                       Tensor &gamma_gradient, Tensor &beta_gradient,
                       Tensor &input_gradient, float epsilon,
                       bool is_training, const Tensor &running_variance)
{
    auto in_impl = input.getImplementation();
    auto out_grad_impl = output_gradient.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto norm_in_impl = normalized_input.getImplementation();
    if (!in_impl || !out_grad_impl || !gamma_impl || !norm_in_impl)
    {
        Logger::logMessage("ops::normalization::batchNormBackward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (input.getRows() > std::numeric_limits<uint32_t>::max() ||
        input.getColumns() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNormBackward: Input dimensions exceed uint32_t");
    }
    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());

    if (output_gradient.getRows() != b_count || output_gradient.getColumns() != f_dim)
    {
        throw std::invalid_argument("ops::normalization::batchNormBackward: output_gradient shape mismatch");
    }
    if (is_training && b_count > 0 && f_dim > 0 &&
        normalized_input.getTotalElements() < static_cast<size_t>(b_count) * f_dim)
    {
        throw std::invalid_argument("ops::normalization::batchNormBackward: normalized_input size smaller than batch*features");
    }

    if (output_gradient.getExecutionTarget() != target ||
        gamma.getExecutionTarget() != target ||
        normalized_input.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::batchNormBackward: Execution_Target mismatch");
    }
    if (is_training)
    {
        if (!batch_variance.getImplementation() || (f_dim > 0 && batch_variance.isEmpty()) ||
            batch_variance.getTotalElements() < f_dim || batch_variance.getExecutionTarget() != target)
        {
            throw std::invalid_argument("ops::normalization::batchNormBackward: Missing or invalid batch_variance for training backward");
        }
    }
    else
    {
        if (!running_variance.getImplementation() || (f_dim > 0 && running_variance.isEmpty()) ||
            running_variance.getTotalElements() < f_dim || running_variance.getExecutionTarget() != target)
        {
            throw std::invalid_argument("ops::normalization::batchNormBackward: Missing or invalid running_variance for eval backward");
        }
    }

    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        normalized_input.getDataType() == Data_Type::TERNARY_INT2 ||
        (is_training ? batch_variance.getDataType() : running_variance.getDataType()) == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::batchNormBackward: TERNARY_INT2 is not supported");
    }

    if (gamma.getTotalElements() < f_dim)
    {
        throw std::invalid_argument("ops::normalization::batchNormBackward: gamma size smaller than feature dimension");
    }

    bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16 || input.getDataType() == Data_Type::FLOAT16);
    Data_Type in_grad_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (b_count == 0 || f_dim == 0)
    {
        ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);
        if (target == Execution_Target::CPU)
        {
            if (f_dim > 0)
            {
                auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), f_dim, false);
                auto *db_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation()), f_dim, false);
                std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
                std::fill(db_vec->begin(), db_vec->begin() + f_dim, 0.0f);
            }
        }
        else
        {
            if (f_dim > 0)
            {
                const auto &anchor = Gpu_Tensor_Impl::castToGpu(*in_impl);
                auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
                auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
                pushGpuFill(anchor, g_grad_gpu.getStorageBuffer(), f_dim, 0.0f);
                pushGpuFill(anchor, b_grad_gpu.getStorageBuffer(), f_dim, 0.0f);
                g_grad_gpu.clearHostCache();
                g_grad_gpu.invalidateFp16Cache();
                b_grad_gpu.clearHostCache();
                b_grad_gpu.invalidateFp16Cache();
            }
            Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).invalidateFp16Cache();
        }
        return;
    }

    if (!is_training)
    {
        if (target == Execution_Target::CPU)
        {
            std::vector<float> out_grad_data = output_gradient.getData();
            std::vector<float> gamma_data = gamma.getData();
            std::vector<float> rv_data = running_variance.getData();

            ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(beta_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

            auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), f_dim, false);
            auto *db_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation()), f_dim, false);
            std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
            std::fill(db_vec->begin(), db_vec->begin() + f_dim, 0.0f);

            std::vector<float> dx_vals(static_cast<size_t>(b_count) * f_dim, 0.0f);
            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    size_t idx = i * f_dim + j;
                    float v = rv_data[j];
                    if (std::isnan(v) || std::isinf(v) || v < 0.0f) v = 1.0f;
                    float g = gamma_data[j];
                    if (std::isnan(g) || std::isinf(g)) g = 1.0f;
                    float inv_std = 1.0f / std::sqrt(v + epsilon);
                    float dy = out_grad_data[idx];
                    if (std::isnan(dy) || std::isinf(dy)) dy = 0.0f;
                    dx_vals[idx] = dy * (g * inv_std);
                }
            }
            auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
            writeCpuOutput(in_grad_cpu, Shape{b_count, f_dim}, in_grad_dtype, std::move(dx_vals));
        }
        else
        {
            const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
            auto contig_grad = out_grad_gpu.ensureContiguousSelf();
            const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

            const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
            const auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());
            const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());

            auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
            auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_grad);
            auto buf_rv = getGpuFloat32ReadBuffer(rv_gpu, *effective_grad);
            auto buf_norm_in = normalized_input.isEmpty() ? buf_grad : getGpuFloat32ReadBuffer(norm_in_gpu, *effective_grad);

            ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(beta_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
            ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

            pushGpuFill(*effective_grad, g_grad_gpu.getStorageBuffer(), f_dim, 0.0f);
            pushGpuFill(*effective_grad, b_grad_gpu.getStorageBuffer(), f_dim, 0.0f);

            const uint64_t total_elem_u64 = static_cast<uint64_t>(b_count) * f_dim;
            if (total_elem_u64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("ops::normalization::batchNormBackward: Total element count exceeds uint32_t");
            }
            const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
            const uint64_t tf_wg_x_u64 = (total_elem_u64 + 255ULL) / 256ULL;
            if (tf_wg_x_u64 > max_wg_x)
            {
                throw std::overflow_error("ops::normalization::batchNormBackward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
            }

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t batch_size;
                uint32_t feature_dimension;
                float epsilon;
            } tf{static_cast<uint32_t>(total_elem_u64), 0, f_dim, epsilon};
            validatePushConstants<Transform_Constants>();

            effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
                                        {buf_grad, buf_norm_in, buf_gamma, g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), buf_rv, in_grad_gpu.getStorageBuffer()},
                                        tf, static_cast<uint32_t>(tf_wg_x_u64), 1, 1);

            in_grad_gpu.clearHostCache();
            in_grad_gpu.invalidateFp16Cache();
            g_grad_gpu.clearHostCache();
            g_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        std::vector<float> out_grad_data = output_gradient.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> bv_data = batch_variance.getData();
        std::vector<float> norm_in_data = normalized_input.getData();

        ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        auto *dg_vec = prepareWritableFp32(g_grad_cpu, f_dim, false);
        auto *db_vec = prepareWritableFp32(b_grad_cpu, f_dim, false);
        std::fill(dg_vec->begin(), dg_vec->begin() + f_dim, 0.0f);
        std::fill(db_vec->begin(), db_vec->begin() + f_dim, 0.0f);

        for (size_t i = 0; i < b_count; ++i)
        {
            for (size_t j = 0; j < f_dim; ++j)
            {
                size_t idx = i * f_dim + j;
                float go = out_grad_data[idx];
                (*dg_vec)[j] += go * norm_in_data[idx];
                (*db_vec)[j] += go;
            }
        }

        std::vector<float> dx_vals(static_cast<size_t>(b_count) * f_dim, 0.0f);
        float inv_b = 1.0f / static_cast<float>(b_count);
        for (size_t j = 0; j < f_dim; ++j)
        {
            float inv_std = 1.0f / std::sqrt(bv_data[j] + epsilon);
            float coeff = gamma_data[j] * inv_std * inv_b;

            for (size_t i = 0; i < b_count; ++i)
            {
                size_t idx = i * f_dim + j;
                dx_vals[idx] = coeff * (static_cast<float>(b_count) * out_grad_data[idx] - (*db_vec)[j] - norm_in_data[idx] * (*dg_vec)[j]);
            }
        }
        writeCpuOutput(in_grad_cpu, Shape{b_count, f_dim}, in_grad_dtype, std::move(dx_vals));
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &bv_gpu = Gpu_Tensor_Impl::castToGpu(*batch_variance.getImplementation());
        const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());

        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
        auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_grad);
        auto buf_bv = getGpuFloat32ReadBuffer(bv_gpu, *effective_grad);
        auto buf_norm_in = getGpuFloat32ReadBuffer(norm_in_gpu, *effective_grad);

        ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

        const uint64_t total_elem_u64 = static_cast<uint64_t>(b_count) * f_dim;
        if (total_elem_u64 > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::normalization::batchNormBackward: Total element count exceeds uint32_t");
        }
        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        const uint64_t tf_wg_x_u64 = (total_elem_u64 + 255ULL) / 256ULL;
        if (f_dim > max_wg_x || tf_wg_x_u64 > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::batchNormBackward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Stats_Constants
        {
            uint32_t batch_size;
            uint32_t feature_dimension;
        } stats{b_count, f_dim};
        validatePushConstants<Stats_Constants>();

        effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_STATS_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_STATS_BACKWARD,
                                    {buf_grad, buf_norm_in, g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                    stats, f_dim, 1, 1);

        struct Transform_Constants
        {
            uint32_t total_elements;
            uint32_t batch_size;
            uint32_t feature_dimension;
            float epsilon;
        } tf{static_cast<uint32_t>(total_elem_u64), b_count, f_dim, epsilon};
        validatePushConstants<Transform_Constants>();

        effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
                                    {buf_grad, buf_norm_in, buf_gamma, g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), buf_bv, in_grad_gpu.getStorageBuffer()},
                                    tf, static_cast<uint32_t>(tf_wg_x_u64), 1, 1);

        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
        g_grad_gpu.clearHostCache();
        g_grad_gpu.invalidateFp16Cache();
        b_grad_gpu.clearHostCache();
        b_grad_gpu.invalidateFp16Cache();
    }
}

void batchNorm2dForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                        Tensor &running_mean, Tensor &running_variance,
                        Tensor &batch_mean, Tensor &batch_variance,
                        Tensor &normalized_input, Tensor &output,
                        uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                        float epsilon, float momentum, bool is_training)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto beta_impl = beta.getImplementation();
    if (!in_impl || !gamma_impl || !beta_impl)
    {
        Logger::logMessage("ops::normalization::batchNorm2dForward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (gamma.getExecutionTarget() != target || beta.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dForward: Execution_Target mismatch");
    }
    if (!running_mean.getImplementation() || !running_variance.getImplementation())
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dForward: Null running stats implementation");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        beta.getDataType() == Data_Type::TERNARY_INT2 ||
        running_mean.getDataType() == Data_Type::TERNARY_INT2 ||
        running_variance.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dForward: TERNARY_INT2 is not supported");
    }

    if (running_mean.getExecutionTarget() != target) running_mean.setExecutionTarget(target);
    if (running_variance.getExecutionTarget() != target) running_variance.setExecutionTarget(target);

    if (running_mean.getTotalElements() < input_channels || running_variance.getTotalElements() < input_channels ||
        gamma.getTotalElements() < input_channels || beta.getTotalElements() < input_channels)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dForward: Parameter or running stats size is smaller than input_channels");
    }

    if (input.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNorm2dForward: Batch size exceeds uint32_t");
    }
    uint32_t b_size = static_cast<uint32_t>(input.getRows());
    uint64_t tot_feat_u64 = static_cast<uint64_t>(input_height) * input_width * input_channels;
    if (tot_feat_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNorm2dForward: Feature size H*W*C exceeds uint32_t");
    }
    uint32_t tot_feat = static_cast<uint32_t>(tot_feat_u64);
    uint64_t sp_count_u64 = static_cast<uint64_t>(b_size) * input_height * input_width;
    if (sp_count_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNorm2dForward: Spatial count B*H*W exceeds uint32_t");
    }
    uint32_t sp_count = static_cast<uint32_t>(sp_count_u64);
    if (sp_count > 0 && input_channels > 0 && input.getColumns() != tot_feat)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dForward: Input columns do not match H*W*C");
    }
    Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (sp_count == 0 || input_channels == 0)
    {
        ensureTensorTarget(output, target, Shape{b_size, tot_feat}, out_dtype);
        ensureTensorTarget(normalized_input, target, Shape{b_size, tot_feat}, Data_Type::FLOAT32);
        if (is_training)
        {
            ensureTensorTarget(batch_mean, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(batch_variance, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        }
        if (target == Execution_Target::VULKAN_GPU)
        {
            Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).invalidateFp16Cache();
            Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation()).invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        std::vector<float> in_data = input.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> beta_data = beta.getData();
        std::vector<float> rm_vals = running_mean.getData();
        std::vector<float> rv_vals = running_variance.getData();

        ensureTensorTarget(output, target, Shape{b_size, tot_feat}, out_dtype);
        ensureTensorTarget(normalized_input, target, Shape{b_size, tot_feat}, Data_Type::FLOAT32);
        if (is_training)
        {
            ensureTensorTarget(batch_mean, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(batch_variance, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        }

        auto &rm_cpu = Cpu_Tensor_Impl::castToCpu(*running_mean.getImplementation());
        auto &rv_cpu = Cpu_Tensor_Impl::castToCpu(*running_variance.getImplementation());
        auto &norm_in_cpu = Cpu_Tensor_Impl::castToCpu(*normalized_input.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        std::vector<float> norm_vals(static_cast<size_t>(b_size) * tot_feat, 0.0f);
        std::vector<float> out_vals(static_cast<size_t>(b_size) * tot_feat, 0.0f);

        if (is_training)
        {
            auto &bm_cpu = Cpu_Tensor_Impl::castToCpu(*batch_mean.getImplementation());
            auto &bv_cpu = Cpu_Tensor_Impl::castToCpu(*batch_variance.getImplementation());
            auto *bm_vec = prepareWritableFp32(bm_cpu, input_channels, false);
            auto *bv_vec = prepareWritableFp32(bv_cpu, input_channels, false);
            std::fill(bm_vec->begin(), bm_vec->begin() + input_channels, 0.0f);
            std::fill(bv_vec->begin(), bv_vec->begin() + input_channels, 0.0f);

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    (*bm_vec)[c] += in_data[i * input_channels + c];
                }
            }
            float inv_sp = 1.0f / static_cast<float>(sp_count);
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                (*bm_vec)[c] *= inv_sp;
            }

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    float diff = in_data[i * input_channels + c] - (*bm_vec)[c];
                    (*bv_vec)[c] += diff * diff;
                }
            }
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                float mean_val = (*bm_vec)[c];
                float var_val = (*bv_vec)[c] * inv_sp;

                bool valid_stats = !std::isnan(mean_val) && !std::isinf(mean_val) &&
                                   !std::isnan(var_val) && !std::isinf(var_val) && (var_val >= 0.0f);
                if (valid_stats)
                {
                    var_val = std::min(var_val, 1e5f);
                    float cur_rm = rm_vals[c];
                    float cur_rv = rv_vals[c];
                    if (std::isnan(cur_rm) || std::isinf(cur_rm)) cur_rm = 0.0f;
                    if (std::isnan(cur_rv) || std::isinf(cur_rv) || cur_rv < 0.0f) cur_rv = 1.0f;
                    rm_vals[c] = (1.0f - momentum) * cur_rm + momentum * mean_val;
                    rv_vals[c] = (1.0f - momentum) * cur_rv + momentum * var_val;
                    (*bm_vec)[c] = mean_val;
                    (*bv_vec)[c] = var_val;
                }
                else
                {
                    float rm = rm_vals[c];
                    float rv = rv_vals[c];
                    if (std::isnan(rm) || std::isinf(rm)) rm = 0.0f;
                    if (std::isnan(rv) || std::isinf(rv) || rv < 0.0f) rv = 1.0f;
                    (*bm_vec)[c] = rm;
                    (*bv_vec)[c] = rv;
                }
            }
            writeCpuOutput(rm_cpu, running_mean.getShape(), running_mean.getDataType(), std::move(rm_vals));
            writeCpuOutput(rv_cpu, running_variance.getShape(), running_variance.getDataType(), std::move(rv_vals));

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float x_val = in_data[idx];
                    float m = (*bm_vec)[c];
                    float v = (*bv_vec)[c];
                    if (std::isnan(x_val) || std::isinf(x_val)) x_val = m;
                    float norm_val = (x_val - m) / std::sqrt(v + epsilon);
                    if (std::isnan(norm_val) || std::isinf(norm_val)) norm_val = 0.0f;
                    norm_val = std::clamp(norm_val, -100.0f, 100.0f);
                    norm_vals[idx] = norm_val;
                    float g = gamma_data[c];
                    float b = beta_data[c];
                    if (std::isnan(g) || std::isinf(g)) g = 1.0f;
                    if (std::isnan(b) || std::isinf(b)) b = 0.0f;
                    float y_val = g * norm_val + b;
                    if (std::isnan(y_val) || std::isinf(y_val)) y_val = b;
                    out_vals[idx] = y_val;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float m = rm_vals[c];
                    float v = rv_vals[c];
                    if (std::isnan(m) || std::isinf(m)) m = 0.0f;
                    if (std::isnan(v) || std::isinf(v) || v < 0.0f) v = 1.0f;

                    float x_val = in_data[idx];
                    if (std::isnan(x_val) || std::isinf(x_val)) x_val = m;
                    float norm_val = (x_val - m) / std::sqrt(v + epsilon);
                    if (std::isnan(norm_val) || std::isinf(norm_val)) norm_val = 0.0f;
                    norm_val = std::clamp(norm_val, -100.0f, 100.0f);
                    norm_vals[idx] = norm_val;
                    float g = gamma_data[c];
                    float b = beta_data[c];
                    if (std::isnan(g) || std::isinf(g)) g = 1.0f;
                    if (std::isnan(b) || std::isinf(b)) b = 0.0f;
                    float y_val = g * norm_val + b;
                    if (std::isnan(y_val) || std::isinf(y_val)) y_val = b;
                    out_vals[idx] = y_val;
                }
            }
        }

        writeCpuOutput(norm_in_cpu, Shape{b_size, tot_feat}, Data_Type::FLOAT32, std::move(norm_vals));
        writeCpuOutput(out_cpu, Shape{b_size, tot_feat}, out_dtype, std::move(out_vals));
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &beta_gpu = Gpu_Tensor_Impl::castToGpu(*beta.getImplementation());
        auto &rm_gpu = Gpu_Tensor_Impl::castToGpu(*running_mean.getImplementation());
        auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());

        if (is_training &&
            (running_mean.getDataType() != Data_Type::FLOAT32 || running_variance.getDataType() != Data_Type::FLOAT32 ||
             !rm_gpu.isContiguous() || rm_gpu.getByteOffset() != 0 ||
             !rv_gpu.isContiguous() || rv_gpu.getByteOffset() != 0))
        {
            throw std::invalid_argument("ops::normalization::batchNorm2dForward: Training running stats on GPU must be contiguous FLOAT32");
        }

        Data_Type data_type = effective_self->getDataType();
        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        auto buf_in = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_self);
        auto buf_beta = getGpuFloat32ReadBuffer(beta_gpu, *effective_self);
        std::shared_ptr<gpu::vector> buf_rm;
        std::shared_ptr<gpu::vector> buf_rv;
        if (!is_training)
        {
            buf_rm = getGpuFloat32ReadBuffer(rm_gpu, *effective_self);
            buf_rv = getGpuFloat32ReadBuffer(rv_gpu, *effective_self);
        }

        ensureTensorTarget(output, target, Shape{b_size, tot_feat}, out_dtype);
        ensureTensorTarget(normalized_input, target, Shape{b_size, tot_feat}, Data_Type::FLOAT32);
        if (is_training)
        {
            ensureTensorTarget(batch_mean, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(batch_variance, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        }

        auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        const uint64_t total_elem_u64 = static_cast<uint64_t>(b_size) * tot_feat;
        if (total_elem_u64 > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::normalization::batchNorm2dForward: Total element count exceeds uint32_t");
        }
        const uint32_t total_elem_u32 = static_cast<uint32_t>(total_elem_u64);
        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        const uint32_t stats_wg_x = input_channels;
        const uint64_t tf_wg_x_u64 = (total_elem_u64 + 255ULL) / 256ULL;
        if (stats_wg_x > max_wg_x || tf_wg_x_u64 > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::batchNorm2dForward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }
        const uint32_t tf_wg_x = static_cast<uint32_t>(tf_wg_x_u64);

        if (is_training)
        {
            auto &bm_gpu = Gpu_Tensor_Impl::castToGpu(*batch_mean.getImplementation());
            auto &bv_gpu = Gpu_Tensor_Impl::castToGpu(*batch_variance.getImplementation());

            struct Stats_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                uint32_t spatial_count;
                float momentum;
            } stats{total_elem_u32, input_channels, sp_count, momentum};
            validatePushConstants<Stats_Constants>();

            effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD,
                                        {buf_in, bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), rm_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer()},
                                        stats, stats_wg_x, 1, 1);

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                float epsilon;
            } tf{total_elem_u32, input_channels, epsilon};
            validatePushConstants<Transform_Constants>();

            effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                                        {buf_in, bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), buf_gamma, buf_beta, out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                        tf, tf_wg_x, 1, 1);

            bm_gpu.clearHostCache();
            bm_gpu.invalidateFp16Cache();
            bv_gpu.clearHostCache();
            bv_gpu.invalidateFp16Cache();
            rm_gpu.clearHostCache();
            rm_gpu.invalidateFp16Cache();
            rv_gpu.clearHostCache();
            rv_gpu.invalidateFp16Cache();
        }
        else
        {
            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                float epsilon;
            } tf{total_elem_u32, input_channels, epsilon};
            validatePushConstants<Transform_Constants>();

            effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                                        {buf_in, buf_rm, buf_rv, buf_gamma, buf_beta, out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                        tf, tf_wg_x, 1, 1);
        }

        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
        norm_in_gpu.clearHostCache();
        norm_in_gpu.invalidateFp16Cache();
    }
}

void batchNorm2dBackward(const Tensor &output_gradient, const Tensor &gamma,
                         const Tensor &batch_variance, const Tensor &normalized_input,
                         Tensor &gamma_gradient, Tensor &beta_gradient,
                         Tensor &input_gradient,
                         uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                         float epsilon,
                         bool is_training, const Tensor &running_variance)
{
    auto out_grad_impl = output_gradient.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto norm_in_impl = normalized_input.getImplementation();
    if (!out_grad_impl || !gamma_impl || !norm_in_impl)
    {
        Logger::logMessage("ops::normalization::batchNorm2dBackward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = output_gradient.getExecutionTarget();
    if (output_gradient.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNorm2dBackward: Batch size exceeds uint32_t");
    }
    uint32_t b_size = static_cast<uint32_t>(output_gradient.getRows());
    uint64_t tot_feat_u64 = static_cast<uint64_t>(input_height) * input_width * input_channels;
    if (tot_feat_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNorm2dBackward: Feature size H*W*C exceeds uint32_t");
    }
    uint32_t tot_feat = static_cast<uint32_t>(tot_feat_u64);
    uint64_t sp_count_u64 = static_cast<uint64_t>(b_size) * input_height * input_width;
    if (sp_count_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::batchNorm2dBackward: Spatial count B*H*W exceeds uint32_t");
    }
    uint32_t sp_count = static_cast<uint32_t>(sp_count_u64);
    if (sp_count > 0 && input_channels > 0)
    {
        if (output_gradient.getColumns() != tot_feat)
        {
            throw std::invalid_argument("ops::normalization::batchNorm2dBackward: output_gradient columns do not match H*W*C");
        }
        if (is_training && normalized_input.getTotalElements() < static_cast<size_t>(b_size) * tot_feat)
        {
            throw std::invalid_argument("ops::normalization::batchNorm2dBackward: normalized_input size smaller than B*H*W*C");
        }
    }

    if (gamma.getExecutionTarget() != target || normalized_input.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dBackward: Execution_Target mismatch");
    }
    if (is_training)
    {
        if (!batch_variance.getImplementation() || (input_channels > 0 && batch_variance.isEmpty()) ||
            batch_variance.getTotalElements() < input_channels || batch_variance.getExecutionTarget() != target)
        {
            throw std::invalid_argument("ops::normalization::batchNorm2dBackward: Missing or invalid batch_variance for training backward");
        }
    }
    else
    {
        if (!running_variance.getImplementation() || (input_channels > 0 && running_variance.isEmpty()) ||
            running_variance.getTotalElements() < input_channels || running_variance.getExecutionTarget() != target)
        {
            throw std::invalid_argument("ops::normalization::batchNorm2dBackward: Missing or invalid running_variance for eval backward");
        }
    }

    if (output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        normalized_input.getDataType() == Data_Type::TERNARY_INT2 ||
        (is_training ? batch_variance.getDataType() : running_variance.getDataType()) == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dBackward: TERNARY_INT2 is not supported");
    }

    if (gamma.getTotalElements() < input_channels)
    {
        throw std::invalid_argument("ops::normalization::batchNorm2dBackward: gamma size smaller than input_channels");
    }

    bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16);
    Data_Type in_grad_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (sp_count == 0 || input_channels == 0)
    {
        ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_size, tot_feat}, in_grad_dtype);
        if (target == Execution_Target::CPU)
        {
            if (input_channels > 0)
            {
                auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), input_channels, false);
                auto *db_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation()), input_channels, false);
                std::fill(dg_vec->begin(), dg_vec->begin() + input_channels, 0.0f);
                std::fill(db_vec->begin(), db_vec->begin() + input_channels, 0.0f);
            }
        }
        else
        {
            if (input_channels > 0)
            {
                const auto &anchor = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
                auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
                auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
                pushGpuFill(anchor, g_grad_gpu.getStorageBuffer(), input_channels, 0.0f);
                pushGpuFill(anchor, b_grad_gpu.getStorageBuffer(), input_channels, 0.0f);
                g_grad_gpu.clearHostCache();
                g_grad_gpu.invalidateFp16Cache();
                b_grad_gpu.clearHostCache();
                b_grad_gpu.invalidateFp16Cache();
            }
            Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).clearHostCache();
            Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).invalidateFp16Cache();
        }
        return;
    }

    if (!is_training)
    {
        if (target == Execution_Target::CPU)
        {
            std::vector<float> out_grad_data = output_gradient.getData();
            std::vector<float> gamma_data = gamma.getData();
            std::vector<float> rv_data = running_variance.getData();

            ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(input_gradient, target, Shape{b_size, tot_feat}, in_grad_dtype);

            auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), input_channels, false);
            auto *db_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation()), input_channels, false);
            std::fill(dg_vec->begin(), dg_vec->begin() + input_channels, 0.0f);
            std::fill(db_vec->begin(), db_vec->begin() + input_channels, 0.0f);

            std::vector<float> dx_vals(static_cast<size_t>(b_size) * tot_feat, 0.0f);
            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float v = rv_data[c];
                    if (std::isnan(v) || std::isinf(v) || v < 0.0f) v = 1.0f;
                    float g = gamma_data[c];
                    if (std::isnan(g) || std::isinf(g)) g = 1.0f;
                    float inv_std = 1.0f / std::sqrt(v + epsilon);
                    float dy = out_grad_data[idx];
                    if (std::isnan(dy) || std::isinf(dy)) dy = 0.0f;
                    dx_vals[idx] = dy * (g * inv_std);
                }
            }
            auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
            writeCpuOutput(in_grad_cpu, Shape{b_size, tot_feat}, in_grad_dtype, std::move(dx_vals));
        }
        else
        {
            const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
            auto contig_grad = out_grad_gpu.ensureContiguousSelf();
            const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

            const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
            const auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());
            const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());

            auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
            auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_grad);
            auto buf_rv = getGpuFloat32ReadBuffer(rv_gpu, *effective_grad);
            auto buf_norm_in = normalized_input.isEmpty() ? buf_grad : getGpuFloat32ReadBuffer(norm_in_gpu, *effective_grad);

            ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
            ensureTensorTarget(input_gradient, target, Shape{b_size, tot_feat}, in_grad_dtype);

            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

            pushGpuFill(*effective_grad, g_grad_gpu.getStorageBuffer(), input_channels, 0.0f);
            pushGpuFill(*effective_grad, b_grad_gpu.getStorageBuffer(), input_channels, 0.0f);

            const uint64_t total_elem_u64 = static_cast<uint64_t>(b_size) * tot_feat;
            if (total_elem_u64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("ops::normalization::batchNorm2dBackward: Total element count exceeds uint32_t");
            }
            const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
            const uint64_t tf_wg_x_u64 = (total_elem_u64 + 255ULL) / 256ULL;
            if (tf_wg_x_u64 > max_wg_x)
            {
                throw std::overflow_error("ops::normalization::batchNorm2dBackward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
            }

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                uint32_t spatial_count;
                float epsilon;
            } tf{static_cast<uint32_t>(total_elem_u64), input_channels, 0, epsilon};
            validatePushConstants<Transform_Constants>();

            effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
                                        {buf_grad, buf_norm_in, buf_gamma, g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), buf_rv, in_grad_gpu.getStorageBuffer()},
                                        tf, static_cast<uint32_t>(tf_wg_x_u64), 1, 1);

            in_grad_gpu.clearHostCache();
            in_grad_gpu.invalidateFp16Cache();
            g_grad_gpu.clearHostCache();
            g_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        std::vector<float> out_grad_data = output_gradient.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> bv_data = batch_variance.getData();
        std::vector<float> norm_in_data = normalized_input.getData();

        ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_size, tot_feat}, in_grad_dtype);

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        auto *dg_vec = prepareWritableFp32(g_grad_cpu, input_channels, false);
        auto *db_vec = prepareWritableFp32(b_grad_cpu, input_channels, false);
        std::fill(dg_vec->begin(), dg_vec->begin() + input_channels, 0.0f);
        std::fill(db_vec->begin(), db_vec->begin() + input_channels, 0.0f);

        for (size_t i = 0; i < sp_count; ++i)
        {
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                size_t idx = i * input_channels + c;
                float go = out_grad_data[idx];
                (*dg_vec)[c] += go * norm_in_data[idx];
                (*db_vec)[c] += go;
            }
        }

        for (uint32_t c = 0; c < input_channels; ++c)
        {
            float dg = (*dg_vec)[c];
            float db = (*db_vec)[c];
            if (std::isnan(dg) || std::isinf(dg)) dg = 0.0f;
            if (std::isnan(db) || std::isinf(db)) db = 0.0f;
            dg = std::clamp(dg, -1e5f, 1e5f);
            db = std::clamp(db, -1e5f, 1e5f);
            (*dg_vec)[c] = dg;
            (*db_vec)[c] = db;
        }

        std::vector<float> dx_vals(static_cast<size_t>(b_size) * tot_feat, 0.0f);
        float inv_sp = 1.0f / static_cast<float>(sp_count);
        for (uint32_t c = 0; c < input_channels; ++c)
        {
            float g = gamma_data[c];
            float v = bv_data[c];
            if (std::isnan(g) || std::isinf(g)) g = 1.0f;
            if (std::isnan(v) || std::isinf(v) || v < 0.0f) v = 1.0f;
            float inv_std = 1.0f / std::sqrt(v + epsilon);
            float coeff = g * inv_std * inv_sp;
            float dg = (*dg_vec)[c];
            float db = (*db_vec)[c];

            for (size_t i = 0; i < sp_count; ++i)
            {
                size_t idx = i * input_channels + c;
                float go = out_grad_data[idx];
                float xh = norm_in_data[idx];
                if (std::isnan(go) || std::isinf(go)) go = 0.0f;
                if (std::isnan(xh) || std::isinf(xh)) xh = 0.0f;
                float dx_val = coeff * (static_cast<float>(sp_count) * go - db - xh * dg);
                if (std::isnan(dx_val) || std::isinf(dx_val)) dx_val = 0.0f;
                dx_vals[idx] = dx_val;
            }
        }
        writeCpuOutput(in_grad_cpu, Shape{b_size, tot_feat}, in_grad_dtype, std::move(dx_vals));
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &bv_gpu = Gpu_Tensor_Impl::castToGpu(*batch_variance.getImplementation());
        const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());

        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
        auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_grad);
        auto buf_bv = getGpuFloat32ReadBuffer(bv_gpu, *effective_grad);
        auto buf_norm_in = getGpuFloat32ReadBuffer(norm_in_gpu, *effective_grad);

        ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, Shape{b_size, tot_feat}, in_grad_dtype);

        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        const uint64_t total_elem_u64 = static_cast<uint64_t>(b_size) * tot_feat;
        if (total_elem_u64 > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::normalization::batchNorm2dBackward: Total element count exceeds uint32_t");
        }
        const uint32_t total_elem_u32 = static_cast<uint32_t>(total_elem_u64);
        const uint32_t max_wg_x = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount()[0];
        const uint32_t stats_wg_x = input_channels;
        const uint64_t tf_wg_x_u64 = (total_elem_u64 + 255ULL) / 256ULL;
        if (stats_wg_x > max_wg_x || tf_wg_x_u64 > max_wg_x)
        {
            throw std::overflow_error("ops::normalization::batchNorm2dBackward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        struct Stats_Constants
        {
            uint32_t total_elements;
            uint32_t channels;
            uint32_t spatial_count;
        } stats{total_elem_u32, input_channels, sp_count};
        validatePushConstants<Stats_Constants>();

        effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD,
                                    {buf_grad, buf_norm_in, g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                    stats, stats_wg_x, 1, 1);

        struct Transform_Constants
        {
            uint32_t total_elements;
            uint32_t channels;
            uint32_t spatial_count;
            float epsilon;
        } tf{total_elem_u32, input_channels, sp_count, epsilon};
        validatePushConstants<Transform_Constants>();

        effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
                                    {buf_grad, buf_norm_in, buf_gamma, g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), buf_bv, in_grad_gpu.getStorageBuffer()},
                                    tf, static_cast<uint32_t>(tf_wg_x_u64), 1, 1);

        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
        g_grad_gpu.clearHostCache();
        g_grad_gpu.invalidateFp16Cache();
        b_grad_gpu.clearHostCache();
        b_grad_gpu.invalidateFp16Cache();
    }
}

}

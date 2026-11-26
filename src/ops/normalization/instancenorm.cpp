#include "ops/normalization/instancenorm.h"

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

void instanceNorm2dForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                           Tensor &saved_mean, Tensor &saved_inv_std,
                           Tensor &normalized_input, Tensor &output,
                           uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                           float epsilon)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto beta_impl = beta.getImplementation();
    if (!in_impl || !gamma_impl || !beta_impl)
    {
        Logger::logMessage("ops::normalization::instanceNorm2dForward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (gamma.getExecutionTarget() != target || beta.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::instanceNorm2dForward: Execution_Target mismatch");
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        beta.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::instanceNorm2dForward: TERNARY_INT2 is not supported");
    }
    if (gamma.getTotalElements() < input_channels || beta.getTotalElements() < input_channels)
    {
        throw std::invalid_argument("ops::normalization::instanceNorm2dForward: Parameter size smaller than input_channels");
    }

    uint64_t tot_spatial_u64 = static_cast<uint64_t>(input_height) * input_width;
    if (tot_spatial_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::instanceNorm2dForward: Spatial size exceeds uint32_t");
    }
    uint32_t tot_spatial = static_cast<uint32_t>(tot_spatial_u64);
    if (input.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::instanceNorm2dForward: Input row count exceeds uint32_t");
    }
    uint32_t b_size = 0;
    if (tot_spatial > 0 && input_channels > 0)
    {
        if (input.getColumns() == input_channels)
        {
            if (input.getRows() % tot_spatial != 0)
            {
                throw std::invalid_argument("ops::normalization::instanceNorm2dForward: Input rows not divisible by H*W");
            }
            size_t b_calc = input.getRows() / tot_spatial;
            if (b_calc > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("ops::normalization::instanceNorm2dForward: Batch size exceeds uint32_t");
            }
            b_size = static_cast<uint32_t>(b_calc);
        }
        else
        {
            uint64_t expected_cols = tot_spatial_u64 * input_channels;
            if (input.getColumns() != expected_cols)
            {
                throw std::invalid_argument("ops::normalization::instanceNorm2dForward: Input shape mismatch for H*W*C");
            }
            b_size = static_cast<uint32_t>(input.getRows());
        }
        if (static_cast<uint64_t>(b_size) * tot_spatial_u64 * input_channels > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::normalization::instanceNorm2dForward: Total element count exceeds uint32_t");
        }
    }
    else
    {
        b_size = static_cast<uint32_t>(input.getRows());
    }

    Shape in_shape = input.getShape();
    Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (b_size == 0 || tot_spatial == 0 || input_channels == 0)
    {
        ensureTensorTarget(output, target, in_shape, out_dtype);
        ensureTensorTarget(normalized_input, target, in_shape, Data_Type::FLOAT32);
        ensureTensorTarget(saved_mean, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(saved_inv_std, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);
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

        ensureTensorTarget(output, target, in_shape, out_dtype);
        ensureTensorTarget(normalized_input, target, in_shape, Data_Type::FLOAT32);
        ensureTensorTarget(saved_mean, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(saved_inv_std, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);

        auto &mean_cpu = Cpu_Tensor_Impl::castToCpu(*saved_mean.getImplementation());
        auto &inv_std_cpu = Cpu_Tensor_Impl::castToCpu(*saved_inv_std.getImplementation());
        auto &norm_in_cpu = Cpu_Tensor_Impl::castToCpu(*normalized_input.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        auto *mean_vec = prepareWritableFp32(mean_cpu, static_cast<size_t>(b_size) * input_channels, false);
        auto *inv_std_vec = prepareWritableFp32(inv_std_cpu, static_cast<size_t>(b_size) * input_channels, false);

        uint32_t M = tot_spatial;
        uint32_t C = input_channels;
        float inv_M = 1.0f / static_cast<float>(M);
        size_t total = static_cast<size_t>(b_size) * M * C;
        std::vector<float> norm_vals(total, 0.0f);
        std::vector<float> out_vals(total, 0.0f);

        for (uint32_t b = 0; b < b_size; ++b)
        {
            size_t batch_offset = static_cast<size_t>(b) * M * C;
            for (uint32_t c = 0; c < C; ++c)
            {
                float sum = 0.0f;
                for (uint32_t m = 0; m < M; ++m)
                {
                    sum += in_data[batch_offset + m * C + c];
                }
                float mean_val = sum * inv_M;
                if (std::isnan(mean_val) || std::isinf(mean_val)) mean_val = 0.0f;

                float sq_sum = 0.0f;
                for (uint32_t m = 0; m < M; ++m)
                {
                    float diff = in_data[batch_offset + m * C + c] - mean_val;
                    sq_sum += diff * diff;
                }
                float var_val = sq_sum * inv_M;
                if (std::isnan(var_val) || std::isinf(var_val) || var_val < 0.0f) var_val = 0.0f;
                float inv_std = 1.0f / std::sqrt(var_val + epsilon);
                if (std::isnan(inv_std) || std::isinf(inv_std)) inv_std = 0.0f;
                inv_std = std::min(inv_std, 100.0f);

                (*mean_vec)[b * C + c] = mean_val;
                (*inv_std_vec)[b * C + c] = inv_std;

                float g = gamma_data[c];
                float bt = beta_data[c];
                for (uint32_t m = 0; m < M; ++m)
                {
                    size_t idx = batch_offset + m * C + c;
                    float x = in_data[idx];
                    float norm = (x - mean_val) * inv_std;
                    norm_vals[idx] = norm;
                    out_vals[idx] = g * norm + bt;
                }
            }
        }

        writeCpuOutput(norm_in_cpu, in_shape, Data_Type::FLOAT32, std::move(norm_vals));
        writeCpuOutput(out_cpu, in_shape, out_dtype, std::move(out_vals));
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &beta_gpu = Gpu_Tensor_Impl::castToGpu(*beta.getImplementation());

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16);
        auto buf_in = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_self);
        auto buf_beta = getGpuFloat32ReadBuffer(beta_gpu, *effective_self);

        ensureTensorTarget(output, target, in_shape, out_dtype);
        ensureTensorTarget(normalized_input, target, in_shape, Data_Type::FLOAT32);
        ensureTensorTarget(saved_mean, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(saved_inv_std, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);

        auto &mean_gpu = Gpu_Tensor_Impl::castToGpu(*saved_mean.getImplementation());
        auto &inv_std_gpu = Gpu_Tensor_Impl::castToGpu(*saved_inv_std.getImplementation());
        auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        auto &engine = Execution_Engine::getInstance();
        bool use_coopmat = is_fp16 && (input_channels % 16 == 0) && engine.isCooperativeMatrixEnabled();

        struct Push_Constants
        {
            uint32_t batch_size;
            uint32_t spatial_count;
            uint32_t channels;
            float epsilon;
        } pc{b_size, tot_spatial, input_channels, epsilon};
        validatePushConstants<Push_Constants>();

        Compute_Pipeline pipeline = Compute_Pipeline::INSTANCE_NORM2D_FORWARD;
        uint32_t wg_x = input_channels;
        uint32_t wg_y = b_size;
        uint32_t wg_z = 1;

        if (use_coopmat)
        {
            pipeline = Compute_Pipeline::INSTANCE_NORM2D_FORWARD_COOPMAT_FP16;
            wg_x = (input_channels + 15) / 16;
        }
        else if (is_fp16)
        {
            pipeline = Compute_Pipeline::INSTANCE_NORM2D_FORWARD_FP16;
        }

        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (wg_x > max_wg[0] || wg_y > max_wg[1])
        {
            throw std::overflow_error("ops::normalization::instanceNorm2dForward: Workgroup grid exceeds hardware limits");
        }

        effective_self->pushToGraph(pipeline,
                                    {buf_in,
                                     buf_gamma,
                                     buf_beta,
                                     mean_gpu.getStorageBuffer(),
                                     inv_std_gpu.getStorageBuffer(),
                                     out_gpu.getStorageBuffer(),
                                     norm_in_gpu.getStorageBuffer()},
                                    pc, wg_x, wg_y, wg_z);

        mean_gpu.clearHostCache();
        mean_gpu.invalidateFp16Cache();
        inv_std_gpu.clearHostCache();
        inv_std_gpu.invalidateFp16Cache();
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
        norm_in_gpu.clearHostCache();
        norm_in_gpu.invalidateFp16Cache();
    }
}

void instanceNorm2dBackward(const Tensor &output_gradient, const Tensor &normalized_input,
                            const Tensor &gamma, const Tensor &saved_inv_std,
                            Tensor &gamma_gradient, Tensor &beta_gradient,
                            Tensor &input_gradient,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon)
{
    auto out_grad_impl = output_gradient.getImplementation();
    auto norm_in_impl = normalized_input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto inv_std_impl = saved_inv_std.getImplementation();
    if (!out_grad_impl || !norm_in_impl || !gamma_impl || !inv_std_impl)
    {
        Logger::logMessage("ops::normalization::instanceNorm2dBackward: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        return;
    }

    Execution_Target target = output_gradient.getExecutionTarget();
    if (normalized_input.getExecutionTarget() != target ||
        gamma.getExecutionTarget() != target ||
        saved_inv_std.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::normalization::instanceNorm2dBackward: Execution_Target mismatch");
    }
    if (output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        normalized_input.getDataType() == Data_Type::TERNARY_INT2 ||
        gamma.getDataType() == Data_Type::TERNARY_INT2 ||
        saved_inv_std.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::normalization::instanceNorm2dBackward: TERNARY_INT2 is not supported");
    }

    uint64_t tot_spatial_u64 = static_cast<uint64_t>(input_height) * input_width;
    if (tot_spatial_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::instanceNorm2dBackward: Spatial size exceeds uint32_t");
    }
    uint32_t tot_spatial = static_cast<uint32_t>(tot_spatial_u64);
    if (output_gradient.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::normalization::instanceNorm2dBackward: output_gradient row count exceeds uint32_t");
    }
    uint32_t b_size = 0;
    if (tot_spatial > 0 && input_channels > 0)
    {
        if (output_gradient.getColumns() == input_channels)
        {
            if (output_gradient.getRows() % tot_spatial != 0)
            {
                throw std::invalid_argument("ops::normalization::instanceNorm2dBackward: output_gradient rows not divisible by H*W");
            }
            size_t b_calc = output_gradient.getRows() / tot_spatial;
            if (b_calc > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("ops::normalization::instanceNorm2dBackward: Batch size exceeds uint32_t");
            }
            b_size = static_cast<uint32_t>(b_calc);
        }
        else
        {
            uint64_t expected_cols = tot_spatial_u64 * input_channels;
            if (output_gradient.getColumns() != expected_cols)
            {
                throw std::invalid_argument("ops::normalization::instanceNorm2dBackward: output_gradient shape mismatch for H*W*C");
            }
            b_size = static_cast<uint32_t>(output_gradient.getRows());
        }
        if (static_cast<uint64_t>(b_size) * tot_spatial_u64 * input_channels > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::normalization::instanceNorm2dBackward: Total element count exceeds uint32_t");
        }
        if (b_size > 0 && normalized_input.getTotalElements() != output_gradient.getTotalElements())
        {
            throw std::invalid_argument("ops::normalization::instanceNorm2dBackward: normalized_input size mismatch");
        }
    }
    else
    {
        b_size = static_cast<uint32_t>(output_gradient.getRows());
    }

    if (gamma.getTotalElements() < input_channels ||
        ((b_size > 0 && tot_spatial > 0 && input_channels > 0) &&
         saved_inv_std.getTotalElements() < static_cast<size_t>(b_size) * input_channels))
    {
        throw std::invalid_argument("ops::normalization::instanceNorm2dBackward: Parameter or saved_inv_std size mismatch");
    }

    Shape grad_shape = output_gradient.getShape();
    Data_Type in_grad_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (b_size == 0 || tot_spatial == 0 || input_channels == 0)
    {
        ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, grad_shape, in_grad_dtype);
        if (target == Execution_Target::CPU)
        {
            if (input_channels > 0)
            {
                auto *dg_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation()), input_channels, false);
                auto *db_vec = prepareWritableFp32(Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation()), input_channels, false);
                std::fill(dg_vec->begin(), dg_vec->begin() + input_channels, 0.0f);
                std::fill(db_vec->begin(), db_vec->begin() + input_channels, 0.0f);
            }
            if (grad_shape.getTotalElements() > 0)
            {
                writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation()), grad_shape, in_grad_dtype,
                               std::vector<float>(grad_shape.getTotalElements(), 0.0f));
            }
        }
        else
        {
            const auto &anchor = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
            if (input_channels > 0)
            {
                pushGpuFill(anchor, g_grad_gpu.getStorageBuffer(), input_channels, 0.0f);
                pushGpuFill(anchor, b_grad_gpu.getStorageBuffer(), input_channels, 0.0f);
            }
            if (grad_shape.getTotalElements() > 0)
            {
                if (grad_shape.getTotalElements() > std::numeric_limits<uint32_t>::max())
                {
                    throw std::overflow_error("ops::normalization::instanceNorm2dBackward: Output fill count exceeds uint32_t");
                }
                const auto &out_grad_gpu = anchor;
                auto contig_grad = out_grad_gpu.ensureContiguousSelf();
                const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;
                if (in_grad_dtype == Data_Type::FLOAT16)
                {
                    struct Fill_FP16_Const { uint32_t total_elements; float fill_value; } pc{static_cast<uint32_t>(grad_shape.getTotalElements()), 0.0f};
                    validatePushConstants<Fill_FP16_Const>();
                    auto [wg_x, wg_y] = Execution_Engine::getInstance().getContext().compute2DWorkgroupGrid((static_cast<uint64_t>(pc.total_elements) + 255ULL) / 256ULL);
                    effective_grad->pushToGraph(Compute_Pipeline::FILL_FP16, {in_grad_gpu.getStorageBuffer()}, pc, wg_x, wg_y, 1);
                }
                else
                {
                    pushGpuFill(*effective_grad, in_grad_gpu.getStorageBuffer(), grad_shape.getTotalElements(), 0.0f);
                }
            }
            g_grad_gpu.clearHostCache();
            g_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            in_grad_gpu.clearHostCache();
            in_grad_gpu.invalidateFp16Cache();
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        std::vector<float> out_grad_data = output_gradient.getData();
        std::vector<float> norm_in_data = normalized_input.getData();
        std::vector<float> gamma_data = gamma.getData();
        std::vector<float> inv_std_data = saved_inv_std.getData();

        ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, grad_shape, in_grad_dtype);

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        auto *dg_vec = prepareWritableFp32(g_grad_cpu, input_channels, false);
        auto *db_vec = prepareWritableFp32(b_grad_cpu, input_channels, false);
        std::fill(dg_vec->begin(), dg_vec->begin() + input_channels, 0.0f);
        std::fill(db_vec->begin(), db_vec->begin() + input_channels, 0.0f);

        uint32_t M = tot_spatial;
        uint32_t C = input_channels;
        float inv_M = 1.0f / static_cast<float>(M);
        std::vector<float> dx_vals(static_cast<size_t>(b_size) * M * C, 0.0f);

        for (uint32_t b = 0; b < b_size; ++b)
        {
            size_t batch_offset = static_cast<size_t>(b) * M * C;
            for (uint32_t c = 0; c < C; ++c)
            {
                float s1 = 0.0f;
                float s2 = 0.0f;
                for (uint32_t m = 0; m < M; ++m)
                {
                    size_t idx = batch_offset + m * C + c;
                    float dy = out_grad_data[idx];
                    float x_hat = norm_in_data[idx];
                    s1 += dy;
                    s2 += dy * x_hat;
                }
                (*db_vec)[c] += s1;
                (*dg_vec)[c] += s2;

                float inv_std = inv_std_data[b * C + c];
                float g = gamma_data[c];
                float g_inv_std = std::clamp(g * inv_std, -100.0f, 100.0f);
                float s1_over_m = s1 * inv_M;
                float s2_over_m = s2 * inv_M;

                for (uint32_t m = 0; m < M; ++m)
                {
                    size_t idx = batch_offset + m * C + c;
                    float dy = out_grad_data[idx];
                    float x_hat = norm_in_data[idx];
                    float dx = g_inv_std * (dy - s1_over_m - x_hat * s2_over_m);
                    if (std::isnan(dx) || std::isinf(dx)) dx = 0.0f;
                    dx_vals[idx] = dx;
                }
            }
        }

        writeCpuOutput(in_grad_cpu, grad_shape, in_grad_dtype, std::move(dx_vals));
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &inv_std_gpu = Gpu_Tensor_Impl::castToGpu(*saved_inv_std.getImplementation());

        bool is_fp16 = (effective_grad->getDataType() == Data_Type::FLOAT16);
        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();
        auto buf_norm_in = getGpuFloat32ReadBuffer(norm_in_gpu, *effective_grad);
        auto buf_gamma = getGpuFloat32ReadBuffer(gamma_gpu, *effective_grad);
        auto buf_inv_std = getGpuFloat32ReadBuffer(inv_std_gpu, *effective_grad);

        ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(input_gradient, target, grad_shape, in_grad_dtype);

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());

        pushGpuFill(*effective_grad, g_grad_gpu.getStorageBuffer(), input_channels, 0.0f);
        pushGpuFill(*effective_grad, b_grad_gpu.getStorageBuffer(), input_channels, 0.0f);

        auto &engine = Execution_Engine::getInstance();
        bool use_coopmat = is_fp16 && (input_channels % 16 == 0) && engine.isCooperativeMatrixEnabled();

        struct Push_Constants
        {
            uint32_t batch_size;
            uint32_t spatial_count;
            uint32_t channels;
            float epsilon;
        } pc{b_size, tot_spatial, input_channels, epsilon};
        validatePushConstants<Push_Constants>();

        Compute_Pipeline pipeline = Compute_Pipeline::INSTANCE_NORM2D_BACKWARD;
        uint32_t wg_x = input_channels;
        uint32_t wg_y = b_size;
        uint32_t wg_z = 1;

        if (use_coopmat)
        {
            pipeline = Compute_Pipeline::INSTANCE_NORM2D_BACKWARD_COOPMAT_FP16;
            wg_x = (input_channels + 15) / 16;
        }
        else if (is_fp16)
        {
            pipeline = Compute_Pipeline::INSTANCE_NORM2D_BACKWARD_FP16;
        }

        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (wg_x > max_wg[0] || wg_y > max_wg[1])
        {
            throw std::overflow_error("ops::normalization::instanceNorm2dBackward: Workgroup grid exceeds hardware limits");
        }

        effective_grad->pushToGraph(pipeline,
                                    {buf_grad,
                                     buf_norm_in,
                                     buf_gamma,
                                     buf_inv_std,
                                     in_grad_gpu.getStorageBuffer(),
                                     g_grad_gpu.getStorageBuffer(),
                                     b_grad_gpu.getStorageBuffer()},
                                    pc, wg_x, wg_y, wg_z);

        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
        g_grad_gpu.clearHostCache();
        g_grad_gpu.invalidateFp16Cache();
        b_grad_gpu.clearHostCache();
        b_grad_gpu.invalidateFp16Cache();
    }
}

}

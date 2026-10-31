#include "ops/normalization/instancenorm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::normalization
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

void instanceNorm2dForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                           Tensor &saved_mean, Tensor &saved_inv_std,
                           Tensor &normalized_input, Tensor &output,
                           uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                           float epsilon)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto beta_impl = beta.getImplementation();
    if (!in_impl || !gamma_impl || !beta_impl) return;

    Execution_Target target = input.getExecutionTarget();
    uint32_t tot_spatial = input_height * input_width;
    uint32_t b_size = 1;
    if (input.getColumns() == input_channels)
    {
        b_size = static_cast<uint32_t>(input.getRows()) / tot_spatial;
    }
    else
    {
        b_size = static_cast<uint32_t>(input.getRows());
    }

    ensureTensorTarget(output, target, input.getShape(), input.getDataType());
    ensureTensorTarget(normalized_input, target, input.getShape(), Data_Type::FLOAT32);
    ensureTensorTarget(saved_mean, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);
    ensureTensorTarget(saved_inv_std, target, Shape{b_size, input_channels}, Data_Type::FLOAT32);

    if (target == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        const auto &gamma_data = gamma.getData();
        const auto &beta_data = beta.getData();
        auto &mean_cpu = Cpu_Tensor_Impl::castToCpu(*saved_mean.getImplementation());
        auto &inv_std_cpu = Cpu_Tensor_Impl::castToCpu(*saved_inv_std.getImplementation());
        auto &norm_in_cpu = Cpu_Tensor_Impl::castToCpu(*normalized_input.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        uint32_t M = tot_spatial;
        uint32_t C = input_channels;
        float inv_M = 1.0f / static_cast<float>(M);

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

                (*mean_cpu.getStorageBuffer())[b * C + c] = mean_val;
                (*inv_std_cpu.getStorageBuffer())[b * C + c] = inv_std;

                float g = gamma_data[c];
                float bt = beta_data[c];
                for (uint32_t m = 0; m < M; ++m)
                {
                    size_t idx = batch_offset + m * C + c;
                    float x = in_data[idx];
                    float norm = (x - mean_val) * inv_std;
                    (*norm_in_cpu.getStorageBuffer())[idx] = norm;
                    (*out_cpu.getStorageBuffer())[idx] = g * norm + bt;
                }
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &beta_gpu = Gpu_Tensor_Impl::castToGpu(*beta.getImplementation());
        auto &mean_gpu = Gpu_Tensor_Impl::castToGpu(*saved_mean.getImplementation());
        auto &inv_std_gpu = Gpu_Tensor_Impl::castToGpu(*saved_inv_std.getImplementation());
        auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16);
        auto &engine = Execution_Engine::getInstance();
        bool use_coopmat = is_fp16 && (input_channels % 16 == 0) && engine.isCooperativeMatrixEnabled();

        struct Push_Constants
        {
            uint32_t batch_size;
            uint32_t spatial_count;
            uint32_t channels;
            float epsilon;
        } pc{b_size, tot_spatial, input_channels, epsilon};

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

        effective_self->pushToGraph(pipeline,
                                    {effective_self->getStorageBuffer(),
                                     gamma_gpu.getStorageBuffer(),
                                     beta_gpu.getStorageBuffer(),
                                     mean_gpu.getStorageBuffer(),
                                     inv_std_gpu.getStorageBuffer(),
                                     out_gpu.getStorageBuffer(),
                                     norm_in_gpu.getStorageBuffer()},
                                    pc, wg_x, wg_y, wg_z);
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
    if (!out_grad_impl || !norm_in_impl || !gamma_impl || !inv_std_impl) return;

    Execution_Target target = output_gradient.getExecutionTarget();
    uint32_t tot_spatial = input_height * input_width;
    uint32_t b_size = 1;
    if (output_gradient.getColumns() == input_channels)
    {
        b_size = static_cast<uint32_t>(output_gradient.getRows()) / tot_spatial;
    }
    else
    {
        b_size = static_cast<uint32_t>(output_gradient.getRows());
    }

    ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
    ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
    ensureTensorTarget(input_gradient, target, output_gradient.getShape(), output_gradient.getDataType());

    if (target == Execution_Target::CPU)
    {
        const auto &out_grad_data = output_gradient.getData();
        const auto &norm_in_data = normalized_input.getData();
        const auto &gamma_data = gamma.getData();
        const auto &inv_std_data = saved_inv_std.getData();
        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        std::fill(g_grad_cpu.getStorageBuffer()->begin(), g_grad_cpu.getStorageBuffer()->end(), 0.0f);
        std::fill(b_grad_cpu.getStorageBuffer()->begin(), b_grad_cpu.getStorageBuffer()->end(), 0.0f);

        uint32_t M = tot_spatial;
        uint32_t C = input_channels;
        float inv_M = 1.0f / static_cast<float>(M);

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
                (*b_grad_cpu.getStorageBuffer())[c] += s1;
                (*g_grad_cpu.getStorageBuffer())[c] += s2;

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
                    (*in_grad_cpu.getStorageBuffer())[idx] = dx;
                }
            }
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &inv_std_gpu = Gpu_Tensor_Impl::castToGpu(*saved_inv_std.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());

        gamma_gradient.fill(0.0f);
        beta_gradient.fill(0.0f);

        bool is_fp16 = (effective_grad->getDataType() == Data_Type::FLOAT16);
        auto &engine = Execution_Engine::getInstance();
        bool use_coopmat = is_fp16 && (input_channels % 16 == 0) && engine.isCooperativeMatrixEnabled();

        struct Push_Constants
        {
            uint32_t batch_size;
            uint32_t spatial_count;
            uint32_t channels;
            float epsilon;
        } pc{b_size, tot_spatial, input_channels, epsilon};

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

        effective_grad->pushToGraph(pipeline,
                                    {effective_grad->getStorageBuffer(),
                                     norm_in_gpu.getStorageBuffer(),
                                     gamma_gpu.getStorageBuffer(),
                                     inv_std_gpu.getStorageBuffer(),
                                     in_grad_gpu.getStorageBuffer(),
                                     g_grad_gpu.getStorageBuffer(),
                                     b_grad_gpu.getStorageBuffer()},
                                    pc, wg_x, wg_y, wg_z);
    }
}

} // namespace ops::normalization

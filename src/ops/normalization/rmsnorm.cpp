#include "ops/normalization/rmsnorm.h"

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

void rmsNormForward(const Tensor &input, const Tensor &gamma, Tensor &inv_rms,
                    Tensor &output, float epsilon)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    if (!in_impl || !gamma_impl) return;

    Execution_Target target = input.getExecutionTarget();
    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());

    bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || gamma.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type out_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : input.getDataType();

    ensureTensorTarget(output, target, Shape{b_count, f_dim}, out_dtype);
    ensureTensorTarget(inv_rms, target, Shape{b_count, 1}, Data_Type::FLOAT32);

    if (target == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        const auto &gamma_data = gamma.getData();

        auto &inv_rms_cpu = Cpu_Tensor_Impl::castToCpu(*inv_rms.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

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
            (*inv_rms_cpu.getStorageBuffer())[i] = inv_val;

            for (size_t j = 0; j < f_dim; ++j)
            {
                (*out_cpu.getStorageBuffer())[i * f_dim + j] = in_data[i * f_dim + j] * inv_val * gamma_data[j];
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        struct RmsNorm_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            float epsilon;
        } pcs{b_count, f_dim, epsilon};

        if (is_fp16)
        {
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_gamma = gamma_gpu.getEffectiveFp16Storage();
            effective_self->pushToGraph(Compute_Pipeline::RMSNORM_FP16,
                                        {buf_self, buf_gamma, out_gpu.getStorageBuffer(), inv_rms_gpu.getStorageBuffer()},
                                        pcs, b_count, 1, 1);
        }
        else
        {
            effective_self->pushToGraph(Compute_Pipeline::RMSNORM,
                                        {effective_self->getStorageBuffer(), gamma_gpu.getStorageBuffer(), out_gpu.getStorageBuffer(), inv_rms_gpu.getStorageBuffer()},
                                        pcs, b_count, 1, 1);
        }
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
    if (!in_impl || !out_grad_impl || !gamma_impl || !inv_rms_impl) return;

    Execution_Target target = input.getExecutionTarget();
    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());

    bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16 || output_gradient.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type in_grad_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : input.getDataType();

    ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
    ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

    if (target == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        const auto &out_grad = output_gradient.getData();
        const auto &gamma_data = gamma.getData();
        const auto &inv_rms_data = inv_rms.getData();

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        if (!accumulate_gamma)
        {
            std::fill(g_grad_cpu.getStorageBuffer()->begin(), g_grad_cpu.getStorageBuffer()->end(), 0.0f);
        }

        for (size_t j = 0; j < f_dim; ++j)
        {
            float sum_dg = 0.0f;
            for (size_t i = 0; i < b_count; ++i)
            {
                sum_dg += out_grad[i * f_dim + j] * in_data[i * f_dim + j] * inv_rms_data[i];
            }
            (*g_grad_cpu.getStorageBuffer())[j] += sum_dg;
        }

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
                float dx = inv_r * (out_grad[i * f_dim + j] * gamma_data[j] - x_hat * (s * inv_dim));
                (*in_grad_cpu.getStorageBuffer())[i * f_dim + j] = dx;
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &inv_rms_gpu = Gpu_Tensor_Impl::castToGpu(*inv_rms.getImplementation());
        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct RMSNorm_Backward_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            uint32_t accumulate;
        } pcs{b_count, f_dim, accumulate_gamma ? 1u : 0u};

        if (is_fp16)
        {
            auto buf_grad = effective_grad->getEffectiveFp16Storage();
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_gamma = gamma_gpu.getEffectiveFp16Storage();

            effective_self->pushToGraph(Compute_Pipeline::RMSNORM_BACKWARD_FP16,
                                        {buf_grad, buf_self, buf_gamma, inv_rms_gpu.getStorageBuffer(), in_grad_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer()},
                                        pcs, b_count, 1, 1);
        }
        else
        {
            effective_self->pushToGraph(Compute_Pipeline::RMSNORM_BACKWARD,
                                        {effective_grad->getStorageBuffer(), effective_self->getStorageBuffer(), gamma_gpu.getStorageBuffer(), inv_rms_gpu.getStorageBuffer(), in_grad_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer()},
                                        pcs, b_count, 1, 1);
        }
    }
}

} // namespace ops::normalization

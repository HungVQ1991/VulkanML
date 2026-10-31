#include "ops/normalization/batchnorm.h"

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

void batchNormForward(const Tensor &input, const Tensor &gamma, const Tensor &beta,
                      Tensor &running_mean, Tensor &running_variance,
                      Tensor &batch_mean, Tensor &batch_variance,
                      Tensor &normalized_input, Tensor &output,
                      float epsilon, float momentum, bool is_training)
{
    auto in_impl = input.getImplementation();
    auto gamma_impl = gamma.getImplementation();
    auto beta_impl = beta.getImplementation();
    if (!in_impl || !gamma_impl || !beta_impl) return;

    Execution_Target target = input.getExecutionTarget();
    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());
    Shape feat_shape{b_count, f_dim};
    Shape stat_shape{1, f_dim};

    if (running_mean.getExecutionTarget() != target) running_mean.setExecutionTarget(target);
    if (running_variance.getExecutionTarget() != target) running_variance.setExecutionTarget(target);

    ensureTensorTarget(output, target, feat_shape, input.getDataType());
    ensureTensorTarget(normalized_input, target, feat_shape, Data_Type::FLOAT32);

    if (is_training)
    {
        ensureTensorTarget(batch_mean, target, stat_shape, Data_Type::FLOAT32);
        ensureTensorTarget(batch_variance, target, stat_shape, Data_Type::FLOAT32);
    }

    if (target == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        const auto &gamma_data = gamma.getData();
        const auto &beta_data = beta.getData();

        auto &rm_cpu = Cpu_Tensor_Impl::castToCpu(*running_mean.getImplementation());
        auto &rv_cpu = Cpu_Tensor_Impl::castToCpu(*running_variance.getImplementation());
        auto &norm_in_cpu = Cpu_Tensor_Impl::castToCpu(*normalized_input.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        if (is_training)
        {
            auto &bm_cpu = Cpu_Tensor_Impl::castToCpu(*batch_mean.getImplementation());
            auto &bv_cpu = Cpu_Tensor_Impl::castToCpu(*batch_variance.getImplementation());
            std::fill(bm_cpu.getStorageBuffer()->begin(), bm_cpu.getStorageBuffer()->end(), 0.0f);
            std::fill(bv_cpu.getStorageBuffer()->begin(), bv_cpu.getStorageBuffer()->end(), 0.0f);

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    (*bm_cpu.getStorageBuffer())[j] += in_data[i * f_dim + j];
                }
            }
            float inv_b = 1.0f / static_cast<float>(b_count);
            for (size_t j = 0; j < f_dim; ++j)
            {
                (*bm_cpu.getStorageBuffer())[j] *= inv_b;
            }

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float diff = in_data[i * f_dim + j] - (*bm_cpu.getStorageBuffer())[j];
                    (*bv_cpu.getStorageBuffer())[j] += diff * diff;
                }
            }
            for (size_t j = 0; j < f_dim; ++j)
            {
                (*bv_cpu.getStorageBuffer())[j] *= inv_b;
                (*rm_cpu.getStorageBuffer())[j] = (1.0f - momentum) * (*rm_cpu.getStorageBuffer())[j] + momentum * (*bm_cpu.getStorageBuffer())[j];
                (*rv_cpu.getStorageBuffer())[j] = (1.0f - momentum) * (*rv_cpu.getStorageBuffer())[j] + momentum * (*bv_cpu.getStorageBuffer())[j];
            }

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float norm_val = (in_data[i * f_dim + j] - (*bm_cpu.getStorageBuffer())[j]) / std::sqrt((*bv_cpu.getStorageBuffer())[j] + epsilon);
                    (*norm_in_cpu.getStorageBuffer())[i * f_dim + j] = norm_val;
                    (*out_cpu.getStorageBuffer())[i * f_dim + j] = gamma_data[j] * norm_val + beta_data[j];
                }
            }
        }
        else
        {
            const auto &rm_data = running_mean.getData();
            const auto &rv_data = running_variance.getData();

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float norm_val = (in_data[i * f_dim + j] - rm_data[j]) / std::sqrt(rv_data[j] + epsilon);
                    (*norm_in_cpu.getStorageBuffer())[i * f_dim + j] = norm_val;
                    (*out_cpu.getStorageBuffer())[i * f_dim + j] = gamma_data[j] * norm_val + beta_data[j];
                }
            }
        }
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
        auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        Data_Type data_type = effective_in->getDataType();

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

            effective_in->pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM_STATS_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_STATS_FORWARD,
                                      {effective_in->getStorageBuffer(), bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), rm_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer()},
                                      stats, f_dim, 1, 1);

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, f_dim, epsilon};

            effective_in->pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                                      {effective_in->getStorageBuffer(), bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), beta_gpu.getStorageBuffer(), out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                      tf, (b_count * f_dim + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, f_dim, epsilon};

            effective_in->pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                                      {effective_in->getStorageBuffer(), rm_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), beta_gpu.getStorageBuffer(), out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                      tf, (b_count * f_dim + 255) / 256, 1, 1);
        }
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
    if (!in_impl || !out_grad_impl || !gamma_impl || !norm_in_impl) return;
    if (is_training && !batch_variance.getImplementation()) return;
    if (!is_training && !running_variance.getImplementation()) return;

    Execution_Target target = input.getExecutionTarget();
    uint32_t b_count = static_cast<uint32_t>(input.getRows());
    uint32_t f_dim = static_cast<uint32_t>(input.getColumns());

    bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16 || input.getDataType() == Data_Type::FLOAT16);
    Data_Type in_grad_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    ensureTensorTarget(gamma_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
    ensureTensorTarget(beta_gradient, target, Shape{1, f_dim}, Data_Type::FLOAT32);
    ensureTensorTarget(input_gradient, target, Shape{b_count, f_dim}, in_grad_dtype);

    if (!is_training)
    {
        if (target == Execution_Target::CPU)
        {
            const auto &out_grad_data = output_gradient.getData();
            const auto &gamma_data = gamma.getData();
            const auto &rv_data = running_variance.getData();
            auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    size_t idx = i * f_dim + j;
                    float v = rv_data[j];
                    if (std::isnan(v) || std::isinf(v) || v <= 0.0f) v = 1.0f;
                    float g = gamma_data[j];
                    if (std::isnan(g) || std::isinf(g)) g = 1.0f;
                    float inv_std = 1.0f / std::sqrt(v + epsilon);
                    float dy = out_grad_data[idx];
                    if (std::isnan(dy) || std::isinf(dy)) dy = 0.0f;
                    (*in_grad_cpu.getStorageBuffer())[idx] = dy * (g * inv_std);
                }
            }
        }
        else
        {
            const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
            auto contig_grad = out_grad_gpu.ensureContiguousSelf();
            const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

            const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
            const auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());
            const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t batch_size;
                uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, 0, f_dim, epsilon};

            effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
                                        {effective_grad->getStorageBuffer(), norm_in_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                        tf, (b_count * f_dim + 255) / 256, 1, 1);
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto &out_grad_data = output_gradient.getData();
        const auto &gamma_data = gamma.getData();
        const auto &bv_data = batch_variance.getData();
        const auto &norm_in_data = normalized_input.getData();

        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        std::fill(g_grad_cpu.getStorageBuffer()->begin(), g_grad_cpu.getStorageBuffer()->end(), 0.0f);
        std::fill(b_grad_cpu.getStorageBuffer()->begin(), b_grad_cpu.getStorageBuffer()->end(), 0.0f);

        for (size_t i = 0; i < b_count; ++i)
        {
            for (size_t j = 0; j < f_dim; ++j)
            {
                size_t idx = i * f_dim + j;
                float go = out_grad_data[idx];
                (*g_grad_cpu.getStorageBuffer())[j] += go * norm_in_data[idx];
                (*b_grad_cpu.getStorageBuffer())[j] += go;
            }
        }

        float inv_b = 1.0f / static_cast<float>(b_count);
        for (size_t j = 0; j < f_dim; ++j)
        {
            float inv_std = 1.0f / std::sqrt(bv_data[j] + epsilon);
            float coeff = gamma_data[j] * inv_std * inv_b;

            for (size_t i = 0; i < b_count; ++i)
            {
                size_t idx = i * f_dim + j;
                (*in_grad_cpu.getStorageBuffer())[idx] = coeff * (static_cast<float>(b_count) * out_grad_data[idx] - (*b_grad_cpu.getStorageBuffer())[j] - norm_in_data[idx] * (*g_grad_cpu.getStorageBuffer())[j]);
            }
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &bv_gpu = Gpu_Tensor_Impl::castToGpu(*batch_variance.getImplementation());
        const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Stats_Constants
        {
            uint32_t batch_size;
            uint32_t feature_dimension;
        } stats{b_count, f_dim};

        effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_STATS_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_STATS_BACKWARD,
                                    {effective_grad->getStorageBuffer(), norm_in_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                    stats, f_dim, 1, 1);

        struct Transform_Constants
        {
            uint32_t total_elements;
            uint32_t batch_size;
            uint32_t feature_dimension;
            float epsilon;
        } tf{b_count * f_dim, b_count, f_dim, epsilon};

        effective_grad->pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
                                    {effective_grad->getStorageBuffer(), norm_in_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                    tf, (b_count * f_dim + 255) / 256, 1, 1);
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
    if (!in_impl || !gamma_impl || !beta_impl) return;

    Execution_Target target = input.getExecutionTarget();
    uint32_t b_size = static_cast<uint32_t>(input.getRows());
    uint32_t tot_feat = input_height * input_width * input_channels;
    uint32_t sp_count = b_size * input_height * input_width;

    if (running_mean.getExecutionTarget() != target) running_mean.setExecutionTarget(target);
    if (running_variance.getExecutionTarget() != target) running_variance.setExecutionTarget(target);

    ensureTensorTarget(output, target, Shape{b_size, tot_feat}, input.getDataType());
    ensureTensorTarget(normalized_input, target, Shape{b_size, tot_feat}, Data_Type::FLOAT32);

    if (is_training)
    {
        ensureTensorTarget(batch_mean, target, Shape{1, input_channels}, Data_Type::FLOAT32);
        ensureTensorTarget(batch_variance, target, Shape{1, input_channels}, Data_Type::FLOAT32);
    }

    if (target == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        const auto &gamma_data = gamma.getData();
        const auto &beta_data = beta.getData();
        auto &rm_cpu = Cpu_Tensor_Impl::castToCpu(*running_mean.getImplementation());
        auto &rv_cpu = Cpu_Tensor_Impl::castToCpu(*running_variance.getImplementation());
        auto &norm_in_cpu = Cpu_Tensor_Impl::castToCpu(*normalized_input.getImplementation());
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        if (is_training)
        {
            auto &bm_cpu = Cpu_Tensor_Impl::castToCpu(*batch_mean.getImplementation());
            auto &bv_cpu = Cpu_Tensor_Impl::castToCpu(*batch_variance.getImplementation());
            std::fill(bm_cpu.getStorageBuffer()->begin(), bm_cpu.getStorageBuffer()->end(), 0.0f);
            std::fill(bv_cpu.getStorageBuffer()->begin(), bv_cpu.getStorageBuffer()->end(), 0.0f);

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    (*bm_cpu.getStorageBuffer())[c] += in_data[i * input_channels + c];
                }
            }
            float inv_sp = 1.0f / static_cast<float>(sp_count);
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                (*bm_cpu.getStorageBuffer())[c] *= inv_sp;
            }

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    float diff = in_data[i * input_channels + c] - (*bm_cpu.getStorageBuffer())[c];
                    (*bv_cpu.getStorageBuffer())[c] += diff * diff;
                }
            }
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                float mean_val = (*bm_cpu.getStorageBuffer())[c];
                float var_val = (*bv_cpu.getStorageBuffer())[c] * inv_sp;

                bool valid_stats = !std::isnan(mean_val) && !std::isinf(mean_val) &&
                                   !std::isnan(var_val) && !std::isinf(var_val) && (var_val >= 0.0f);
                if (valid_stats)
                {
                    var_val = std::min(var_val, 1e5f);
                    float cur_rm = (*rm_cpu.getStorageBuffer())[c];
                    float cur_rv = (*rv_cpu.getStorageBuffer())[c];
                    if (std::isnan(cur_rm) || std::isinf(cur_rm)) cur_rm = 0.0f;
                    if (std::isnan(cur_rv) || std::isinf(cur_rv) || cur_rv <= 0.0f) cur_rv = 1.0f;
                    (*rm_cpu.getStorageBuffer())[c] = (1.0f - momentum) * cur_rm + momentum * mean_val;
                    (*rv_cpu.getStorageBuffer())[c] = (1.0f - momentum) * cur_rv + momentum * var_val;
                    (*bm_cpu.getStorageBuffer())[c] = mean_val;
                    (*bv_cpu.getStorageBuffer())[c] = var_val;
                }
                else
                {
                    float rm = (*rm_cpu.getStorageBuffer())[c];
                    float rv = (*rv_cpu.getStorageBuffer())[c];
                    if (std::isnan(rm) || std::isinf(rm)) rm = 0.0f;
                    if (std::isnan(rv) || std::isinf(rv) || rv <= 0.0f) rv = 1.0f;
                    (*bm_cpu.getStorageBuffer())[c] = rm;
                    (*bv_cpu.getStorageBuffer())[c] = rv;
                }
            }
            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float x_val = in_data[idx];
                    float m = (*bm_cpu.getStorageBuffer())[c];
                    float v = (*bv_cpu.getStorageBuffer())[c];
                    if (std::isnan(x_val) || std::isinf(x_val)) x_val = m;
                    float norm_val = (x_val - m) / std::sqrt(v + epsilon);
                    if (std::isnan(norm_val) || std::isinf(norm_val)) norm_val = 0.0f;
                    norm_val = std::clamp(norm_val, -100.0f, 100.0f);
                    (*norm_in_cpu.getStorageBuffer())[idx] = norm_val;
                    float y_val = gamma_data[c] * norm_val + beta_data[c];
                    if (std::isnan(y_val) || std::isinf(y_val)) y_val = beta_data[c];
                    (*out_cpu.getStorageBuffer())[idx] = y_val;
                }
            }
        }
        else
        {
            const auto &rm_data = running_mean.getData();
            const auto &rv_data = running_variance.getData();

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float m = rm_data[c];
                    float v = rv_data[c];
                    if (std::isnan(m) || std::isinf(m)) m = 0.0f;
                    if (std::isnan(v) || std::isinf(v) || v <= 0.0f) v = 1.0f;

                    float x_val = in_data[idx];
                    if (std::isnan(x_val) || std::isinf(x_val)) x_val = m;
                    float norm_val = (x_val - m) / std::sqrt(v + epsilon);
                    if (std::isnan(norm_val) || std::isinf(norm_val)) norm_val = 0.0f;
                    norm_val = std::clamp(norm_val, -100.0f, 100.0f);
                    (*norm_in_cpu.getStorageBuffer())[idx] = norm_val;
                    float y_val = gamma_data[c] * norm_val + beta_data[c];
                    if (std::isnan(y_val) || std::isinf(y_val)) y_val = beta_data[c];
                    (*out_cpu.getStorageBuffer())[idx] = y_val;
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
        auto &rm_gpu = Gpu_Tensor_Impl::castToGpu(*running_mean.getImplementation());
        auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());
        auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        Data_Type data_type = effective_self->getDataType();

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
            } stats{b_size * tot_feat, input_channels, sp_count, momentum};

            effective_self->pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD,
                                        {effective_self->getStorageBuffer(), bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), rm_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer()},
                                        stats, 1, 1, 1);

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, epsilon};

            effective_self->pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                                        {effective_self->getStorageBuffer(), bm_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), beta_gpu.getStorageBuffer(), out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, epsilon};

            effective_self->pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                                        {effective_self->getStorageBuffer(), rm_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), beta_gpu.getStorageBuffer(), out_gpu.getStorageBuffer(), norm_in_gpu.getStorageBuffer()},
                                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
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
    if (!out_grad_impl || !gamma_impl || !norm_in_impl) return;
    if (is_training && !batch_variance.getImplementation()) return;
    if (!is_training && !running_variance.getImplementation()) return;

    Execution_Target target = output_gradient.getExecutionTarget();
    uint32_t b_size = static_cast<uint32_t>(output_gradient.getRows());
    uint32_t tot_feat = input_height * input_width * input_channels;
    uint32_t sp_count = b_size * input_height * input_width;

    bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16);
    Data_Type in_grad_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    ensureTensorTarget(gamma_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
    ensureTensorTarget(beta_gradient, target, Shape{1, input_channels}, Data_Type::FLOAT32);
    ensureTensorTarget(input_gradient, target, Shape{b_size, tot_feat}, in_grad_dtype);

    if (!is_training)
    {
        if (target == Execution_Target::CPU)
        {
            const auto &out_grad_data = output_gradient.getData();
            const auto &gamma_data = gamma.getData();
            const auto &rv_data = running_variance.getData();
            auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float v = rv_data[c];
                    if (std::isnan(v) || std::isinf(v) || v <= 0.0f) v = 1.0f;
                    float g = gamma_data[c];
                    if (std::isnan(g) || std::isinf(g)) g = 1.0f;
                    float inv_std = 1.0f / std::sqrt(v + epsilon);
                    float dy = out_grad_data[idx];
                    if (std::isnan(dy) || std::isinf(dy)) dy = 0.0f;
                    (*in_grad_cpu.getStorageBuffer())[idx] = dy * (g * inv_std);
                }
            }
        }
        else
        {
            const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
            auto contig_grad = out_grad_gpu.ensureContiguousSelf();
            const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

            const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
            const auto &rv_gpu = Gpu_Tensor_Impl::castToGpu(*running_variance.getImplementation());
            const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
            auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                uint32_t spatial_count;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, 0, epsilon};

            effective_grad->pushToGraph(effective_grad->getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
                                        {effective_grad->getStorageBuffer(), norm_in_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), rv_gpu.getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto &out_grad_data = output_gradient.getData();
        const auto &gamma_data = gamma.getData();
        const auto &bv_data = batch_variance.getData();
        const auto &norm_in_data = normalized_input.getData();
        auto &g_grad_cpu = Cpu_Tensor_Impl::castToCpu(*gamma_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*beta_gradient.getImplementation());
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        std::fill(g_grad_cpu.getStorageBuffer()->begin(), g_grad_cpu.getStorageBuffer()->end(), 0.0f);
        std::fill(b_grad_cpu.getStorageBuffer()->begin(), b_grad_cpu.getStorageBuffer()->end(), 0.0f);

        for (size_t i = 0; i < sp_count; ++i)
        {
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                size_t idx = i * input_channels + c;
                float go = out_grad_data[idx];
                (*g_grad_cpu.getStorageBuffer())[c] += go * norm_in_data[idx];
                (*b_grad_cpu.getStorageBuffer())[c] += go;
            }
        }

        float inv_sp = 1.0f / static_cast<float>(sp_count);
        for (uint32_t c = 0; c < input_channels; ++c)
        {
            float inv_std = 1.0f / std::sqrt(bv_data[c] + epsilon);
            float coeff = gamma_data[c] * inv_std * inv_sp;

            for (size_t i = 0; i < sp_count; ++i)
            {
                size_t idx = i * input_channels + c;
                (*in_grad_cpu.getStorageBuffer())[idx] = coeff * (static_cast<float>(sp_count) * out_grad_data[idx] - (*b_grad_cpu.getStorageBuffer())[c] - norm_in_data[idx] * (*g_grad_cpu.getStorageBuffer())[c]);
            }
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = Gpu_Tensor_Impl::castToGpu(*gamma.getImplementation());
        const auto &bv_gpu = Gpu_Tensor_Impl::castToGpu(*batch_variance.getImplementation());
        const auto &norm_in_gpu = Gpu_Tensor_Impl::castToGpu(*normalized_input.getImplementation());
        auto &g_grad_gpu = Gpu_Tensor_Impl::castToGpu(*gamma_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*beta_gradient.getImplementation());
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Stats_Constants
        {
            uint32_t total_elements;
            uint32_t channels;
            uint32_t spatial_count;
        } stats{b_size * tot_feat, input_channels, sp_count};

        effective_grad->pushToGraph(effective_grad->getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD,
                                    {effective_grad->getStorageBuffer(), norm_in_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                    stats, 1, 1, 1);

        struct Transform_Constants
        {
            uint32_t total_elements;
            uint32_t channels;
            uint32_t spatial_count;
            float epsilon;
        } tf{b_size * tot_feat, input_channels, sp_count, epsilon};

        effective_grad->pushToGraph(effective_grad->getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
                                    {effective_grad->getStorageBuffer(), norm_in_gpu.getStorageBuffer(), gamma_gpu.getStorageBuffer(), g_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer(), bv_gpu.getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                    tf, (b_size * tot_feat + 255) / 256, 1, 1);
    }
}

} // namespace ops::normalization

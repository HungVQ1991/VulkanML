#include "ops/loss/dice_bce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/vulkan_context.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::loss
{

void diceBce(const Tensor &prediction, const Tensor &target, Tensor &output,
             float dice_weight, float bce_weight, float epsilon)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0) return;

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 4);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(4, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() != 4)
        {
            output_cpu.getStorageBuffer()->resize(4, 0.0f);
        }

        float total_inter = 0.0f;
        float total_pred = 0.0f;
        float total_target = 0.0f;
        float total_bce = 0.0f;

        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0f - epsilon);
            float y = t_data[i];
            total_inter += p * y;
            total_pred += p;
            total_target += y;
            total_bce += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }

        float dice_denom = total_pred + total_target + epsilon;
        float dice_numer = 2.0f * total_inter + epsilon;
        float dice_loss = 1.0f - (dice_numer / dice_denom);
        float bce_loss = total_bce / static_cast<float>(total_elements);

        float total_loss = dice_weight * dice_loss + bce_weight * bce_loss;
        auto &loss_buf = *output_cpu.getStorageBuffer();
        loss_buf[0] = total_loss;
        loss_buf[1] = total_inter;
        loss_buf[2] = total_pred;
        loss_buf[3] = total_target;

        if (prediction.getDataType() == Data_Type::FLOAT16)
        {
            if (!output_cpu.getStorageBufferFp16() || output_cpu.getStorageBufferFp16()->size() != 4)
            {
                output_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(4));
            }
            convertFp32ToFp16(loss_buf.data(), output_cpu.getStorageBufferFp16()->data(), 4);
            output_cpu.getStorageBuffer().reset();
        }
    }
    else
    {
        auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        auto contig_self = pred_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &pred_gpu;

        const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        output_gpu.setDataType(Data_Type::FLOAT32);
        output_gpu.reshape(1, 4);

        uint32_t total = static_cast<uint32_t>(total_elements);
        struct PushConstants
        {
            uint32_t total_elements;
            float dice_weight;
            float bce_weight;
            float epsilon;
        } pc{total, dice_weight, bce_weight, epsilon};

        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16);
        bool use_coopmat = false;
        const auto &ctx = Execution_Engine::getInstance().getContext();
        if (is_fp16 && ctx.isCooperativeMatrixEnabled())
        {
            use_coopmat = true;
        }

        Compute_Pipeline pipe = Compute_Pipeline::DICE_BCE_LOSS;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::DICE_BCE_LOSS_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::DICE_BCE_LOSS_FP16;
        }

        auto buf_pred = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_target = is_fp16 ? effective_target->getEffectiveFp16Storage() : effective_target->getStorageBuffer();

        pred_gpu.pushToGraph(pipe, {buf_pred, buf_target, output_gpu.getStorageBuffer()}, pc, 1, 1, 1);
        output_gpu.clearHostCache();
    }
}

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight, float bce_weight, float epsilon,
                     const Tensor *stats_tensor)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0) return;

    // Ensure we have a valid 4-element stats tensor (loss, inter, pred_sum, target_sum)
    static thread_local Tensor s_fallback_stats;
    const Tensor *effective_stats = stats_tensor;
    if (!effective_stats || effective_stats->getTotalElements() < 4)
    {
        if (s_fallback_stats.getExecutionTarget() != prediction.getExecutionTarget())
        {
            s_fallback_stats = Tensor(Shape{1, 4}, prediction.getExecutionTarget());
        }
        diceBce(prediction, target, s_fallback_stats, dice_weight, bce_weight, epsilon);
        effective_stats = &s_fallback_stats;
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, total_elements);
        output.reshape(prediction.getShape());
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() != total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements);
        }

        const auto &stats_data = effective_stats->getData();
        float total_inter = stats_data[1];
        float total_pred = stats_data[2];
        float total_target = stats_data[3];

        float denom = total_pred + total_target + epsilon;
        float numer = 2.0f * total_inter + epsilon;
        float inv_denom_sq = 1.0f / (denom * denom);
        float inv_N = 1.0f / static_cast<float>(total_elements);

        auto &grad_out = *output_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0f - epsilon);
            float y = t_data[i];
            float dice_grad = (numer - 2.0f * y * denom) * inv_denom_sq;
            float p_safe = std::clamp(p, 1e-4f, 1.0f - 1e-4f);
            float bce_grad = (p - y) / (p_safe * (1.0f - p_safe)) * inv_N;
            grad_out[i] = dice_weight * dice_grad + bce_weight * bce_grad;
        }

        if (prediction.getDataType() == Data_Type::FLOAT16)
        {
            if (!output_cpu.getStorageBufferFp16() || output_cpu.getStorageBufferFp16()->size() < total_elements)
            {
                output_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(total_elements));
            }
            convertFp32ToFp16(grad_out.data(), output_cpu.getStorageBufferFp16()->data(), total_elements);
            output_cpu.getStorageBuffer().reset();
        }
    }
    else
    {
        auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        auto contig_self = pred_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &pred_gpu;

        const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        const auto &stats_gpu = Gpu_Tensor_Impl::castToGpu(*effective_stats->getImplementation());

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
        }
        output_gpu.reshape(1, total_elements);
        output.reshape(prediction.getShape());

        uint32_t total = static_cast<uint32_t>(total_elements);

        struct PushConstants
        {
            uint32_t total_elements;
            float dice_weight;
            float bce_weight;
            float epsilon;
        } pc{total, dice_weight, bce_weight, epsilon};

        bool use_coopmat = false;
        const auto &ctx = Execution_Engine::getInstance().getContext();
        if (is_fp16 && ctx.isCooperativeMatrixEnabled())
        {
            use_coopmat = true;
        }

        Compute_Pipeline pipe = Compute_Pipeline::DICE_BCE_GRADIENT;
        uint32_t wg_x = (total + 255) / 256;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::DICE_BCE_GRADIENT_COOPMAT_FP16;
            wg_x = (total + 255) / 256;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::DICE_BCE_GRADIENT_FP16;
        }

        auto buf_pred = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_target = is_fp16 ? effective_target->getEffectiveFp16Storage() : effective_target->getStorageBuffer();

        pred_gpu.pushToGraph(pipe,
                             {buf_pred, buf_target, output_gpu.getStorageBuffer(), stats_gpu.getStorageBuffer()},
                             pc, wg_x, 1, 1);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight, float bce_weight, float epsilon,
                     float intersection, float pred_sum, float target_sum)
{
    if (intersection != 0.0f || pred_sum != 0.0f || target_sum != 0.0f)
    {
        std::vector<float> stats = {0.0f, intersection, pred_sum, target_sum};
        Tensor stats_t(Shape{1, 4}, stats, prediction.getExecutionTarget());
        diceBceGradient(prediction, target, output, dice_weight, bce_weight, epsilon, &stats_t);
    }
    else
    {
        diceBceGradient(prediction, target, output, dice_weight, bce_weight, epsilon, nullptr);
    }
}

float diceBceLossAndGradient(const Tensor &prediction, const Tensor &target, Tensor &gradient_out,
                             float dice_weight, float bce_weight, float epsilon)
{
    Tensor loss_tensor(Shape{1, 4}, prediction.getExecutionTarget());
    diceBce(prediction, target, loss_tensor, dice_weight, bce_weight, epsilon);
    diceBceGradient(prediction, target, gradient_out, dice_weight, bce_weight, epsilon, &loss_tensor);
    return loss_tensor.getData()[0];
}

} // namespace ops::loss

#include "ops/loss/dice_bce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/data_type.h"
#include "engine/execution_engine.h"
#include "engine/vulkan_context.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::loss
{

using ops::internal::ensureUniqueTensorTarget;
using ops::internal::pushGpuFill;
using ops::internal::resolveExecutionEngine;
using ops::internal::validatePushConstants;

namespace
{

void validateDiceBceOperands(const Tensor &prediction, const Tensor &target, const char *op_name)
{
    if (prediction.getExecutionTarget() != target.getExecutionTarget())
    {
        throw std::invalid_argument(std::string(op_name) + ": Execution_Target mismatch between prediction and target");
    }
    if (prediction.getDataType() == Data_Type::TERNARY_INT2 ||
        target.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op_name) + ": TERNARY_INT2 is not supported");
    }
}

void validateDiceBceParams(float dice_weight, float bce_weight, float epsilon, const char *op_name)
{
    if (!std::isfinite(dice_weight) || dice_weight < 0.0f)
    {
        throw std::invalid_argument(std::string(op_name) + ": dice_weight must be finite and >= 0");
    }
    if (!std::isfinite(bce_weight) || bce_weight < 0.0f)
    {
        throw std::invalid_argument(std::string(op_name) + ": bce_weight must be finite and >= 0");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f || epsilon >= 0.5f)
    {
        throw std::invalid_argument(std::string(op_name) + ": epsilon must be finite and in (0, 0.5)");
    }
}

void writeCpuDiceBceOutput(Tensor &output, const Shape &shape, Data_Type dtype, const std::vector<float> &vals)
{
    ensureUniqueTensorTarget(output, Execution_Target::CPU, shape, dtype);
    auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
    const size_t n = vals.size();
    if (dtype == Data_Type::FLOAT16)
    {
        std::vector<float> clamped(n);
        for (size_t i = 0; i < n; ++i)
        {
            clamped[i] = std::isnan(vals[i]) ? vals[i] : std::clamp(vals[i], -65504.0f, 65504.0f);
        }
        output_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(n));
        if (n > 0)
        {
            convertFp32ToFp16(clamped.data(), output_cpu.getStorageBufferFp16()->data(), n);
        }
        output_cpu.setDataType(Data_Type::FLOAT16);
        output_cpu.getStorageBuffer().reset();
    }
    else
    {
        output_cpu.setStorageBufferFp16(nullptr);
        output_cpu.setDataType(Data_Type::FLOAT32);
        output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(vals);
    }
}

std::shared_ptr<gpu::vector> getDiceBceFloat32Buffer(
    const Gpu_Tensor_Impl &gpu_impl,
    std::vector<std::shared_ptr<Gpu_Tensor_Impl>> &temp_impls,
    std::vector<std::shared_ptr<gpu::vector>> &temp_buffers,
    Execution_Engine &engine)
{
    auto contig = gpu_impl.ensureContiguousSelf();
    const Gpu_Tensor_Impl *effective = contig ? contig.get() : &gpu_impl;
    if (contig)
    {
        temp_impls.push_back(contig);
    }
    if (effective->getDataType() == Data_Type::FLOAT32)
    {
        return effective->getStorageBuffer();
    }
    if (effective->getDataType() == Data_Type::FLOAT16)
    {
        const size_t total = effective->getTotalElements();
        if (total > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::loss::diceBce: total_elements exceeds uint32_t limit");
        }
        auto fp32_buf = std::make_shared<gpu::vector>(engine.getContext());
        fp32_buf->allocateMemory(total, Data_Type::FLOAT32);
        struct Cast_PC
        {
            uint32_t total_elements;
        } pc{static_cast<uint32_t>(total)};
        validatePushConstants<Cast_PC>();
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total + 255) / 256);
        gpu_impl.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {effective->getStorageBuffer(), fp32_buf}, pc, wg_x, wg_y, 1, &engine);
        temp_buffers.push_back(fp32_buf);
        return fp32_buf;
    }
    throw std::invalid_argument("ops::loss::diceBce: Unsupported Data_Type for GPU buffer");
}

}

void diceBce(const Tensor &prediction, const Tensor &target, Tensor &output,
             const Dice_Bce_Config &config)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    const float dice_weight = config.dice_weight;
    const float bce_weight = config.bce_weight;
    const float epsilon = config.epsilon;

    validateDiceBceOperands(prediction, target, "ops::loss::diceBce");
    validateDiceBceParams(dice_weight, bce_weight, epsilon, "ops::loss::diceBce");
    pred_impl->validateSameDimensions(*target_impl);

    constexpr size_t stats_count = static_cast<size_t>(Dice_Bce_Stats_Index::COUNT);
    const Shape stats_shape{1, stats_count};

    size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        if (prediction.getExecutionTarget() == Execution_Target::CPU)
        {
            writeCpuDiceBceOutput(output, stats_shape, Data_Type::FLOAT32, {0.0f, 0.0f, 0.0f, 0.0f});
        }
        else
        {
            Execution_Engine &engine = resolveExecutionEngine(config.engine);
            ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, stats_shape, Data_Type::FLOAT32);
            auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            pushGpuFill(output_gpu, stats_count, 0.0f, engine);
        }
        return;
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto a_data = prediction.getData();
        const auto t_data = target.getData();

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

        std::vector<float> stats(stats_count, 0.0f);
        stats[static_cast<size_t>(Dice_Bce_Stats_Index::LOSS)] = total_loss;
        stats[static_cast<size_t>(Dice_Bce_Stats_Index::INTERSECTION)] = total_inter;
        stats[static_cast<size_t>(Dice_Bce_Stats_Index::PRED_SUM)] = total_pred;
        stats[static_cast<size_t>(Dice_Bce_Stats_Index::TARGET_SUM)] = total_target;
        writeCpuDiceBceOutput(output, stats_shape, Data_Type::FLOAT32, stats);
    }
    else
    {
        Execution_Engine &engine = resolveExecutionEngine(config.engine);
        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::loss::diceBce: total_elements exceeds uint32_t limit");
        }
        ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, stats_shape, Data_Type::FLOAT32);

        auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        std::shared_ptr<Gpu_Tensor_Impl> contig_self;
        std::shared_ptr<Gpu_Tensor_Impl> contig_target;
        std::vector<std::shared_ptr<Gpu_Tensor_Impl>> temp_impls;
        std::vector<std::shared_ptr<gpu::vector>> temp_buffers;

        const auto &ctx = engine.getContext();
        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16) && ctx.isFloat16Enabled();
        std::shared_ptr<gpu::vector> buf_pred;
        std::shared_ptr<gpu::vector> buf_target;
        if (is_fp16)
        {
            contig_self = pred_gpu.ensureContiguousSelf();
            contig_target = target_gpu.ensureContiguousSelf();
            const auto *effective_self = contig_self ? contig_self.get() : &pred_gpu;
            const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;
            buf_pred = effective_self->getEffectiveFp16Storage();
            buf_target = effective_target->getEffectiveFp16Storage();
        }
        else
        {
            buf_pred = getDiceBceFloat32Buffer(pred_gpu, temp_impls, temp_buffers, engine);
            buf_target = getDiceBceFloat32Buffer(target_gpu, temp_impls, temp_buffers, engine);
        }

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        output_gpu.setDataType(Data_Type::FLOAT32);
        output_gpu.reshape(1, stats_count);

        uint32_t total = static_cast<uint32_t>(total_elements);
        struct PushConstants
        {
            uint32_t total_elements;
            float dice_weight;
            float bce_weight;
            float epsilon;
        } pc{total, dice_weight, bce_weight, epsilon};
        validatePushConstants<PushConstants>();

        bool use_coopmat = is_fp16 && engine.isCooperativeMatrixEnabled();

        Compute_Pipeline pipe = Compute_Pipeline::DICE_BCE_LOSS;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::DICE_BCE_LOSS_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::DICE_BCE_LOSS_FP16;
        }

        uint32_t wg_x = static_cast<uint32_t>(std::min<uint64_t>((static_cast<uint64_t>(total) + 255ULL) / 256ULL, 256ULL));
        if (wg_x > ctx.getMaxComputeWorkGroupCount()[0])
        {
            throw std::overflow_error("ops::loss::diceBce: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        if (wg_x <= 1u)
        {
            pred_gpu.pushToGraph(pipe, {buf_pred, buf_target, output_gpu.getStorageBuffer()}, pc, 1, 1, 1, &engine);
        }
        else
        {
            auto partials_buf = engine.acquireReusableScratchBuffer(
                Execution_Engine::SCRATCH_LOSS_PARTIALS,
                static_cast<size_t>(wg_x) * stats_count * sizeof(float),
                Data_Type::FLOAT32,
                static_cast<size_t>(wg_x) * stats_count);
            pred_gpu.pushToGraph(pipe, {buf_pred, buf_target, partials_buf}, pc, wg_x, 1, 1, &engine);

            struct ReduceConstants
            {
                uint32_t num_partials;
                uint32_t total_elements;
                float dice_weight;
                float bce_weight;
                float epsilon;
            } reduce_pc{wg_x, total, dice_weight, bce_weight, epsilon};
            validatePushConstants<ReduceConstants>();
            pred_gpu.pushToGraph(Compute_Pipeline::DICE_BCE_REDUCE_FINAL,
                                 {partials_buf, output_gpu.getStorageBuffer()},
                                 reduce_pc, 1, 1, 1, &engine);
        }
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void diceBce(const Tensor &prediction, const Tensor &target, Tensor &output,
             float dice_weight, float bce_weight, float epsilon)
{
    diceBce(prediction, target, output, Dice_Bce_Config{dice_weight, bce_weight, epsilon, nullptr});
}

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     const Dice_Bce_Config &config, const Tensor *stats_tensor)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    const float dice_weight = config.dice_weight;
    const float bce_weight = config.bce_weight;
    const float epsilon = config.epsilon;

    validateDiceBceOperands(prediction, target, "ops::loss::diceBceGradient");
    validateDiceBceParams(dice_weight, bce_weight, epsilon, "ops::loss::diceBceGradient");
    pred_impl->validateSameDimensions(*target_impl);

    constexpr size_t stats_count = static_cast<size_t>(Dice_Bce_Stats_Index::COUNT);
    size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        if (prediction.getExecutionTarget() == Execution_Target::CPU)
        {
            ensureUniqueTensorTarget(output, Execution_Target::CPU, prediction.getShape(), prediction.getDataType());
        }
        else
        {
            Execution_Engine &engine = resolveExecutionEngine(config.engine);
            bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16) &&
                           engine.getContext().isFloat16Enabled();
            ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, prediction.getShape(), is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            output_gpu.clearHostCache();
            output_gpu.invalidateFp16Cache();
        }
        return;
    }

    Tensor fallback_stats;
    const Tensor *effective_stats = stats_tensor;
    if (!effective_stats || effective_stats->getTotalElements() < stats_count)
    {
        fallback_stats = Tensor(Shape{1, stats_count}, Data_Type::FLOAT32, prediction.getExecutionTarget());
        diceBce(prediction, target, fallback_stats, config);
        effective_stats = &fallback_stats;
    }

    if (effective_stats->getExecutionTarget() != prediction.getExecutionTarget())
    {
        throw std::invalid_argument("ops::loss::diceBceGradient: stats_tensor Execution_Target mismatch");
    }
    if (effective_stats->getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::loss::diceBceGradient: stats_tensor cannot be TERNARY_INT2");
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto a_data = prediction.getData();
        const auto t_data = target.getData();
        const auto stats_data = effective_stats->getData();
        float total_inter = stats_data[static_cast<size_t>(Dice_Bce_Stats_Index::INTERSECTION)];
        float total_pred = stats_data[static_cast<size_t>(Dice_Bce_Stats_Index::PRED_SUM)];
        float total_target = stats_data[static_cast<size_t>(Dice_Bce_Stats_Index::TARGET_SUM)];

        float denom = std::max(total_pred + total_target + epsilon, 1e-4f);
        float numer = 2.0f * total_inter + epsilon;
        float inv_denom_sq = 1.0f / (denom * denom);
        float inv_N = 1.0f / static_cast<float>(total_elements);

        std::vector<float> grad_vec(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            float raw_p = a_data[i];
            float y = t_data[i];
            if (std::isnan(raw_p) || std::isnan(y))
            {
                grad_vec[i] = raw_p + y;
                continue;
            }
            float p = std::clamp(raw_p, epsilon, 1.0f - epsilon);
            float dice_grad = (numer - 2.0f * y * denom) * inv_denom_sq;
            float p_safe = std::clamp(p, 1e-4f, 1.0f - 1e-4f);
            float bce_grad = (p - y) / (p_safe * (1.0f - p_safe)) * inv_N;
            float g = dice_weight * dice_grad + bce_weight * bce_grad;
            if (std::isinf(g)) g = 0.0f;
            grad_vec[i] = g;
        }

        writeCpuDiceBceOutput(output, prediction.getShape(), prediction.getDataType(), grad_vec);
    }
    else
    {
        Execution_Engine &engine = resolveExecutionEngine(config.engine);
        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::loss::diceBceGradient: total_elements exceeds uint32_t limit");
        }
        const auto &ctx = engine.getContext();
        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16) && ctx.isFloat16Enabled();
        Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, prediction.getShape(), out_dtype);

        auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        const auto &stats_gpu = Gpu_Tensor_Impl::castToGpu(*effective_stats->getImplementation());
        std::shared_ptr<Gpu_Tensor_Impl> contig_self;
        std::shared_ptr<Gpu_Tensor_Impl> contig_target;
        std::vector<std::shared_ptr<Gpu_Tensor_Impl>> temp_impls;
        std::vector<std::shared_ptr<gpu::vector>> temp_buffers;

        std::shared_ptr<gpu::vector> buf_pred;
        std::shared_ptr<gpu::vector> buf_target;
        if (is_fp16)
        {
            contig_self = pred_gpu.ensureContiguousSelf();
            contig_target = target_gpu.ensureContiguousSelf();
            const auto *effective_self = contig_self ? contig_self.get() : &pred_gpu;
            const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;
            buf_pred = effective_self->getEffectiveFp16Storage();
            buf_target = effective_target->getEffectiveFp16Storage();
        }
        else
        {
            buf_pred = getDiceBceFloat32Buffer(pred_gpu, temp_impls, temp_buffers, engine);
            buf_target = getDiceBceFloat32Buffer(target_gpu, temp_impls, temp_buffers, engine);
        }
        auto buf_stats = getDiceBceFloat32Buffer(stats_gpu, temp_impls, temp_buffers, engine);

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        output_gpu.setDataType(out_dtype);
        output.reshape(prediction.getShape());

        uint32_t total = static_cast<uint32_t>(total_elements);

        struct PushConstants
        {
            uint32_t total_elements;
            float dice_weight;
            float bce_weight;
            float epsilon;
        } pc{total, dice_weight, bce_weight, epsilon};
        validatePushConstants<PushConstants>();

        bool use_coopmat = is_fp16 && engine.isCooperativeMatrixEnabled();

        Compute_Pipeline pipe = Compute_Pipeline::DICE_BCE_GRADIENT;
        uint32_t wg_x = (total + 255) / 256;
        if (wg_x > ctx.getMaxComputeWorkGroupCount()[0])
        {
            throw std::overflow_error("ops::loss::diceBceGradient: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::DICE_BCE_GRADIENT_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::DICE_BCE_GRADIENT_FP16;
        }

        pred_gpu.pushToGraph(pipe,
                             {buf_pred, buf_target, buf_stats, output_gpu.getStorageBuffer()},
                             pc, wg_x, 1, 1, &engine);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     const Dice_Bce_Config &config,
                     float intersection, float pred_sum, float target_sum)
{
    if (!prediction.getImplementation() || !target.getImplementation())
    {
        return;
    }
    std::vector<float> stats(static_cast<size_t>(Dice_Bce_Stats_Index::COUNT), 0.0f);
    stats[static_cast<size_t>(Dice_Bce_Stats_Index::INTERSECTION)] = intersection;
    stats[static_cast<size_t>(Dice_Bce_Stats_Index::PRED_SUM)] = pred_sum;
    stats[static_cast<size_t>(Dice_Bce_Stats_Index::TARGET_SUM)] = target_sum;
    Tensor stats_t(Shape{1, stats.size()}, stats, prediction.getExecutionTarget());
    diceBceGradient(prediction, target, output, config, &stats_t);
}

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight, float bce_weight, float epsilon,
                     const Tensor *stats_tensor)
{
    diceBceGradient(prediction, target, output, Dice_Bce_Config{dice_weight, bce_weight, epsilon, nullptr}, stats_tensor);
}

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight, float bce_weight, float epsilon,
                     float intersection, float pred_sum, float target_sum)
{
    diceBceGradient(prediction, target, output, Dice_Bce_Config{dice_weight, bce_weight, epsilon, nullptr},
                    intersection, pred_sum, target_sum);
}

float diceBceLossAndGradient(const Tensor &prediction, const Tensor &target, Tensor &gradient_out,
                             const Dice_Bce_Config &config)
{
    if (!prediction.getImplementation() || !target.getImplementation())
    {
        return 0.0f;
    }
    constexpr size_t stats_count = static_cast<size_t>(Dice_Bce_Stats_Index::COUNT);
    if (prediction.getTotalElements() == 0)
    {
        Tensor loss_tensor(Shape{1, stats_count}, Data_Type::FLOAT32, prediction.getExecutionTarget());
        diceBce(prediction, target, loss_tensor, config);
        diceBceGradient(prediction, target, gradient_out, config, &loss_tensor);
        return 0.0f;
    }
    Tensor loss_tensor(Shape{1, stats_count}, Data_Type::FLOAT32, prediction.getExecutionTarget());
    diceBce(prediction, target, loss_tensor, config);
    diceBceGradient(prediction, target, gradient_out, config, &loss_tensor);
    return loss_tensor.getData()[static_cast<size_t>(Dice_Bce_Stats_Index::LOSS)];
}

float diceBceLossAndGradient(const Tensor &prediction, const Tensor &target, Tensor &gradient_out,
                             float dice_weight, float bce_weight, float epsilon)
{
    return diceBceLossAndGradient(prediction, target, gradient_out,
                                  Dice_Bce_Config{dice_weight, bce_weight, epsilon, nullptr});
}

}

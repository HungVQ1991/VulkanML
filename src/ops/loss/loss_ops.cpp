#include "ops/loss/loss_ops.h"

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
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::loss
{

using ops::internal::ensureUniqueTensorTarget;
using ops::internal::pushGpuFill;
using ops::internal::resolveExecutionEngine;
using ops::internal::validatePushConstants;
using ops::internal::validateU32Count;

namespace
{

void validateLossOperands(const Tensor &prediction, const Tensor &target, const char *op_name)
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

void validateProbEpsilon(float epsilon, const char *op_name)
{
    if (!std::isfinite(epsilon) || epsilon <= 0.0f || epsilon >= 0.5f)
    {
        throw std::invalid_argument(std::string(op_name) + ": epsilon must be finite and in (0, 0.5)");
    }
}

void validatePositiveFinite(float val, const char *name, const char *op_name)
{
    if (!std::isfinite(val) || val <= 0.0f)
    {
        throw std::invalid_argument(std::string(op_name) + ": " + name + " must be finite and > 0");
    }
}

void validateAslParams(float gamma_pos, float gamma_neg, float clip_margin, float epsilon, const char *op_name)
{
    if (!std::isfinite(gamma_pos) || gamma_pos < 0.0f)
    {
        throw std::invalid_argument(std::string(op_name) + ": gamma_pos must be finite and >= 0");
    }
    if (!std::isfinite(gamma_neg) || gamma_neg < 0.0f)
    {
        throw std::invalid_argument(std::string(op_name) + ": gamma_neg must be finite and >= 0");
    }
    if (!std::isfinite(clip_margin) || clip_margin < 0.0f || clip_margin >= 1.0f)
    {
        throw std::invalid_argument(std::string(op_name) + ": clip_margin must be finite and in [0, 1)");
    }
    validateProbEpsilon(epsilon, op_name);
}

void writeCpuScalarFloat32(Tensor &output, float value)
{
    ensureUniqueTensorTarget(output, Execution_Target::CPU, Shape{1, 1}, Data_Type::FLOAT32);
    auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
    ops::internal::writeCpuOutput(output_cpu, Shape{1, 1}, Data_Type::FLOAT32, std::vector<float>{value}, false);
}

void writeCpuTensorPreserveDtype(Tensor &output, const Shape &out_shape, Data_Type dtype, std::vector<float> &&values)
{
    ensureUniqueTensorTarget(output, Execution_Target::CPU, out_shape, dtype);
    auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
    ops::internal::writeCpuOutput(output_cpu, out_shape, dtype, std::move(values), true);
}

std::shared_ptr<gpu::vector> getGpuFloat32Buffer(
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
        const uint32_t total = validateU32Count(effective->getTotalElements(), "ops::loss");
        auto fp32_buf = std::make_shared<gpu::vector>(engine.getContext());
        fp32_buf->allocateMemory(total, Data_Type::FLOAT32);
        struct Cast_PC
        {
            uint32_t total_elements;
        } pc{total};
        validatePushConstants<Cast_PC>();
        auto [wg_x, wg_y] = ops::internal::dispatch2D((static_cast<size_t>(total) + 255) / 256, &engine);
        gpu_impl.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {effective->getStorageBuffer(), fp32_buf}, pc, wg_x, wg_y, 1, &engine);
        temp_buffers.push_back(fp32_buf);
        return fp32_buf;
    }
    throw std::invalid_argument("ops::loss: Unsupported Data_Type for GPU loss buffer");
}

struct Gpu_Loss_Inputs
{
    bool is_fp16 = false;
    Data_Type grad_dtype = Data_Type::FLOAT32;
    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    std::shared_ptr<Gpu_Tensor_Impl> contig_target;
    std::vector<std::shared_ptr<Gpu_Tensor_Impl>> temp_impls;
    std::vector<std::shared_ptr<gpu::vector>> temp_buffers;
    std::shared_ptr<gpu::vector> buf_pred;
    std::shared_ptr<gpu::vector> buf_target;
};

Gpu_Loss_Inputs resolveGpuLossInputs(
    const Tensor &prediction,
    Gpu_Tensor_Impl &pred_gpu,
    const Gpu_Tensor_Impl &target_gpu,
    bool allow_fp16,
    Execution_Engine &engine)
{
    Gpu_Loss_Inputs inputs;
    inputs.is_fp16 = allow_fp16 &&
                     (prediction.getDataType() == Data_Type::FLOAT16) &&
                     engine.getContext().isFloat16Enabled();
    inputs.grad_dtype = inputs.is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    if (inputs.is_fp16)
    {
        inputs.contig_self = pred_gpu.ensureContiguousSelf();
        inputs.contig_target = target_gpu.ensureContiguousSelf();
        const auto *eff_self = inputs.contig_self ? inputs.contig_self.get() : &pred_gpu;
        const auto *eff_target = inputs.contig_target ? inputs.contig_target.get() : &target_gpu;
        inputs.buf_pred = eff_self->getEffectiveFp16Storage();
        inputs.buf_target = eff_target->getEffectiveFp16Storage();
    }
    else
    {
        inputs.buf_pred = getGpuFloat32Buffer(pred_gpu, inputs.temp_impls, inputs.temp_buffers, engine);
        inputs.buf_target = getGpuFloat32Buffer(target_gpu, inputs.temp_impls, inputs.temp_buffers, engine);
    }
    return inputs;
}

void handleEmptyLossGradient(const Tensor &prediction, Tensor &output, Execution_Engine *engine_ptr)
{
    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        ensureUniqueTensorTarget(output, Execution_Target::CPU, prediction.getShape(), prediction.getDataType());
        return;
    }
    Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
    const bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16) &&
                         engine.getContext().isFloat16Enabled();
    ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, prediction.getShape(),
                             is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
    auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
    output_gpu.clearHostCache();
    output_gpu.invalidateFp16Cache();
}

bool tryFloatToTokenIndex(float raw, int32_t &out_idx) noexcept
{
    if (!std::isfinite(raw) || raw < -2147483648.0f || raw > 2147483520.0f)
    {
        return false;
    }
    out_idx = static_cast<int32_t>(std::round(raw));
    return true;
}

template <typename PushConst>
void dispatchSumLoss2Pass(const Gpu_Tensor_Impl &anchor,
                          Compute_Pipeline pass1_pipe,
                          const std::shared_ptr<gpu::vector> &buf_pred,
                          const std::shared_ptr<gpu::vector> &buf_target,
                          Gpu_Tensor_Impl &output_gpu,
                          const PushConst &pc,
                          uint32_t total,
                          const char *op_name,
                          Execution_Engine &engine)
{
    validatePushConstants<PushConst>();
    const uint64_t wg_x_u64 = (static_cast<uint64_t>(total) + 255ULL) / 256ULL;
    if (wg_x_u64 > engine.getContext().getMaxComputeWorkGroupCount()[0])
    {
        throw std::overflow_error(std::string(op_name) + ": Workgroup count exceeds maxComputeWorkGroupCount[0]");
    }
    const uint32_t wg_x = static_cast<uint32_t>(wg_x_u64);
    if (wg_x <= 1u)
    {
        anchor.pushToGraph(pass1_pipe, {buf_pred, buf_target, output_gpu.getStorageBuffer()}, pc, 1, 1, 1, &engine);
    }
    else
    {
        auto partials_buf = engine.acquireReusableScratchBuffer(
            Execution_Engine::SCRATCH_LOSS_PARTIALS,
            static_cast<size_t>(wg_x) * sizeof(float),
            Data_Type::FLOAT32,
            wg_x);
        anchor.pushToGraph(pass1_pipe, {buf_pred, buf_target, partials_buf}, pc, wg_x, 1, 1, &engine);
        struct Sum_Reduce_Constants
        {
            uint32_t num_partials;
        } reduce_pc{wg_x};
        validatePushConstants<Sum_Reduce_Constants>();
        anchor.pushToGraph(Compute_Pipeline::SUM_REDUCE_FINAL,
                           {partials_buf, output_gpu.getStorageBuffer()},
                           reduce_pc, 1, 1, 1, &engine);
    }
}

void dispatchDiceReduce2Pass(const Gpu_Tensor_Impl &anchor,
                             Compute_Pipeline loss_pipe,
                             const std::shared_ptr<gpu::vector> &buf_pred,
                             const std::shared_ptr<gpu::vector> &buf_target,
                             const std::shared_ptr<gpu::vector> &out_buf,
                             uint32_t total,
                             float epsilon,
                             uint32_t write_stats,
                             const char *op_name,
                             Execution_Engine &engine)
{
    const uint32_t wg_x = static_cast<uint32_t>(std::min<uint64_t>((static_cast<uint64_t>(total) + 255ULL) / 256ULL, 256ULL));
    if (wg_x > engine.getContext().getMaxComputeWorkGroupCount()[0])
    {
        throw std::overflow_error(std::string(op_name) + ": Workgroup count exceeds maxComputeWorkGroupCount[0]");
    }
    struct Loss_Constants
    {
        uint32_t total_elements;
        float epsilon_val;
        uint32_t write_stats;
    } loss_pc{total, epsilon, write_stats};
    validatePushConstants<Loss_Constants>();

    if (wg_x <= 1u)
    {
        anchor.pushToGraph(loss_pipe, {buf_pred, buf_target, out_buf}, loss_pc, 1, 1, 1, &engine);
    }
    else
    {
        auto partials_buf = engine.acquireReusableScratchBuffer(
            Execution_Engine::SCRATCH_LOSS_PARTIALS,
            static_cast<size_t>(wg_x) * 3u * sizeof(float),
            Data_Type::FLOAT32,
            static_cast<size_t>(wg_x) * 3u);
        anchor.pushToGraph(loss_pipe, {buf_pred, buf_target, partials_buf}, loss_pc, wg_x, 1, 1, &engine);
        struct Dice_Reduce_Constants
        {
            uint32_t num_partials;
            float epsilon_val;
            uint32_t write_stats;
        } reduce_pc{wg_x, epsilon, write_stats};
        validatePushConstants<Dice_Reduce_Constants>();
        anchor.pushToGraph(Compute_Pipeline::DICE_REDUCE_FINAL, {partials_buf, out_buf}, reduce_pc, 1, 1, 1, &engine);
    }
}

template <typename CpuElemLossFn, typename PushConstFn>
void runTwoPassSumLoss(
    const Tensor &prediction,
    const Tensor &target,
    Tensor &output,
    const char *op_name,
    bool allow_fp16,
    Compute_Pipeline fp32_pipe,
    Compute_Pipeline fp16_pipe,
    CpuElemLossFn &&cpu_elem_loss,
    PushConstFn &&make_push_constants,
    Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    pred_impl->validateSameDimensions(*target_impl);

    const size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        if (prediction.getExecutionTarget() == Execution_Target::CPU)
        {
            writeCpuScalarFloat32(output, 0.0f);
        }
        else
        {
            Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
            ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, Shape{1, 1}, Data_Type::FLOAT32);
            auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            pushGpuFill(output_gpu, 1, 0.0f, engine);
        }
        return;
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto a_data = prediction.getData();
        const auto t_data = target.getData();
        float loss = 0.0f;
        for (size_t i = 0; i < total_elements; ++i)
        {
            loss += cpu_elem_loss(a_data[i], t_data[i]);
        }
        writeCpuScalarFloat32(output, loss);
        return;
    }

    const uint32_t total = validateU32Count(total_elements, op_name);
    Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
    ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, Shape{1, 1}, Data_Type::FLOAT32);

    auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
    const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
    auto gpu_inputs = resolveGpuLossInputs(prediction, pred_gpu, target_gpu, allow_fp16, engine);

    auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
    output_gpu.setDataType(Data_Type::FLOAT32);
    output_gpu.reshape(1, 1);

    const auto pc = make_push_constants(total);
    const Compute_Pipeline pipe = gpu_inputs.is_fp16 ? fp16_pipe : fp32_pipe;
    dispatchSumLoss2Pass(pred_gpu, pipe, gpu_inputs.buf_pred, gpu_inputs.buf_target, output_gpu, pc, total, op_name, engine);
    output_gpu.clearHostCache();
    output_gpu.invalidateFp16Cache();
}

template <typename CpuElemGradFn, typename PushConstFn>
void runElementwiseLossGradient(
    const Tensor &prediction,
    const Tensor &target,
    Tensor &output,
    const char *op_name,
    Compute_Pipeline fp32_pipe,
    Compute_Pipeline fp16_pipe,
    CpuElemGradFn &&cpu_elem_grad,
    PushConstFn &&make_push_constants,
    Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    pred_impl->validateSameDimensions(*target_impl);

    const size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        handleEmptyLossGradient(prediction, output, engine_ptr);
        return;
    }

    const float inv_total = 1.0f / static_cast<float>(total_elements);
    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto a_data = prediction.getData();
        const auto t_data = target.getData();
        std::vector<float> grad_out(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            grad_out[i] = cpu_elem_grad(a_data[i], t_data[i], inv_total);
        }
        writeCpuTensorPreserveDtype(output, prediction.getShape(), prediction.getDataType(), std::move(grad_out));
        return;
    }

    const uint32_t total = validateU32Count(total_elements, op_name);
    Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
    auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
    const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
    auto gpu_inputs = resolveGpuLossInputs(prediction, pred_gpu, target_gpu, true, engine);
    ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, prediction.getShape(), gpu_inputs.grad_dtype);

    auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
    output_gpu.setDataType(gpu_inputs.grad_dtype);
    output.reshape(prediction.getShape());

    auto [wg_x, wg_y] = ops::internal::dispatch2D((total_elements + 255) / 256, &engine);
    const auto pc = make_push_constants(total, inv_total);
    validatePushConstants<std::decay_t<decltype(pc)>>();

    const Compute_Pipeline pipe = gpu_inputs.is_fp16 ? fp16_pipe : fp32_pipe;
    pred_gpu.pushToGraph(pipe, {gpu_inputs.buf_pred, gpu_inputs.buf_target, output_gpu.getStorageBuffer()}, pc, wg_x, wg_y, 1, &engine);
    output_gpu.clearHostCache();
    output_gpu.invalidateFp16Cache();
}

void dispatchScaledDiffGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                                float scale, const char *op_name, Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    const size_t total_elements = prediction.getTotalElements();
    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto a_data = prediction.getData();
        const auto t_data = target.getData();
        std::vector<float> grad_out(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            grad_out[i] = (a_data[i] - t_data[i]) * scale;
        }
        writeCpuTensorPreserveDtype(output, prediction.getShape(), prediction.getDataType(), std::move(grad_out));
        return;
    }

    const uint32_t total = validateU32Count(total_elements, op_name);
    Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
    auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
    const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
    auto gpu_inputs = resolveGpuLossInputs(prediction, pred_gpu, target_gpu, true, engine);
    ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, prediction.getShape(), gpu_inputs.grad_dtype);

    auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
    output_gpu.setDataType(gpu_inputs.grad_dtype);
    output.reshape(prediction.getShape());

    const uint64_t wg_x_u64 = (static_cast<uint64_t>(total) + 255ULL) / 256ULL;
    if (wg_x_u64 > engine.getContext().getMaxComputeWorkGroupCount()[0])
    {
        throw std::overflow_error(std::string(op_name) + ": Workgroup count exceeds maxComputeWorkGroupCount[0]");
    }
    const uint32_t wg_x = static_cast<uint32_t>(wg_x_u64);

    auto diff_buf = std::make_shared<gpu::vector>(engine.getContext());
    diff_buf->allocateMemory(total_elements, gpu_inputs.grad_dtype);

    struct Sub_Constants
    {
        uint32_t total_elements;
        uint32_t cols;
        uint32_t is_broadcast;
    } sub_pc{total, 1u, 0u};
    validatePushConstants<Sub_Constants>();
    pred_gpu.pushToGraph(gpu_inputs.is_fp16 ? Compute_Pipeline::SUB_FP16 : Compute_Pipeline::SUB,
                         {gpu_inputs.buf_pred, gpu_inputs.buf_target, diff_buf},
                         sub_pc, wg_x, 1, 1, &engine);

    if (gpu_inputs.is_fp16)
    {
        struct Scalar_Fp16_Constants
        {
            uint32_t total_elements;
            float scalar;
            uint32_t preserve_nonfinite;
        } mul_pc{total, scale, 0u};
        validatePushConstants<Scalar_Fp16_Constants>();
        pred_gpu.pushToGraph(Compute_Pipeline::MUL_SCALAR_FP16,
                             {diff_buf, output_gpu.getStorageBuffer()},
                             mul_pc, wg_x, 1, 1, &engine);
    }
    else
    {
        struct Scalar_Constants
        {
            uint32_t total_elements;
            float scalar;
        } mul_pc{total, scale};
        validatePushConstants<Scalar_Constants>();
        pred_gpu.pushToGraph(Compute_Pipeline::MUL_SCALAR,
                             {diff_buf, output_gpu.getStorageBuffer()},
                             mul_pc, wg_x, 1, 1, &engine);
    }

    output_gpu.clearHostCache();
    output_gpu.invalidateFp16Cache();
}

}

void mse(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::mse");
    struct Constants
    {
        uint32_t total_elements;
    };
    runTwoPassSumLoss(
        prediction, target, output, "ops::loss::mse", false,
        Compute_Pipeline::MSE_LOSS, Compute_Pipeline::MSE_LOSS,
        [](float a, float t) {
            const float diff = a - t;
            return diff * diff;
        },
        [](uint32_t total) { return Constants{total}; },
        engine_ptr);
}

void mseGradient(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    validateLossOperands(prediction, target, "ops::loss::mseGradient");
    pred_impl->validateSameDimensions(*target_impl);

    const size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        handleEmptyLossGradient(prediction, output, engine_ptr);
        return;
    }

    const float scale = 2.0f / static_cast<float>(total_elements);
    dispatchScaledDiffGradient(prediction, target, output, scale, "ops::loss::mseGradient", engine_ptr);
}

void mae(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::mae");
    struct Constants
    {
        uint32_t total_elements;
    };
    runTwoPassSumLoss(
        prediction, target, output, "ops::loss::mae", false,
        Compute_Pipeline::MAE_LOSS, Compute_Pipeline::MAE_LOSS,
        [](float a, float t) { return std::abs(a - t); },
        [](uint32_t total) { return Constants{total}; },
        engine_ptr);
}

void maeGradient(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::maeGradient");
    struct Constants
    {
        uint32_t total_elements;
        float inv_total_elements;
    };
    runElementwiseLossGradient(
        prediction, target, output, "ops::loss::maeGradient",
        Compute_Pipeline::MAE_GRADIENT, Compute_Pipeline::MAE_GRADIENT_FP16,
        [](float raw_p, float y, float inv_total) {
            const float diff = raw_p - y;
            if (std::isnan(diff))
            {
                return diff;
            }
            return (diff > 0.0f) ? inv_total : ((diff < 0.0f) ? -inv_total : 0.0f);
        },
        [](uint32_t total, float inv_total) { return Constants{total, inv_total}; },
        engine_ptr);
}

void bce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::bce");
    validateProbEpsilon(epsilon, "ops::loss::bce");
    struct Constants
    {
        uint32_t total_elements;
        float epsilon;
    };
    runTwoPassSumLoss(
        prediction, target, output, "ops::loss::bce", false,
        Compute_Pipeline::BCE_LOSS, Compute_Pipeline::BCE_LOSS,
        [epsilon](float a, float y) {
            const float p = std::clamp(a, epsilon, 1.0f - epsilon);
            return -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        },
        [epsilon](uint32_t total) { return Constants{total, epsilon}; },
        engine_ptr);
}

void bceGradient(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::bceGradient");
    validateProbEpsilon(epsilon, "ops::loss::bceGradient");
    const float eff_eps = (prediction.getDataType() == Data_Type::FLOAT16) ? std::max(epsilon, 1e-4f) : epsilon;
    struct Constants
    {
        uint32_t total_elements;
        float epsilon_val;
        float inv_total_elements;
    };
    runElementwiseLossGradient(
        prediction, target, output, "ops::loss::bceGradient",
        Compute_Pipeline::BCE_GRADIENT, Compute_Pipeline::BCE_GRADIENT_FP16,
        [eff_eps](float raw_p, float y, float inv_total) {
            if (std::isnan(raw_p) || std::isnan(y))
            {
                return raw_p + y;
            }
            const float p = std::clamp(raw_p, eff_eps, 1.0f - eff_eps);
            return ((p - y) / (p * (1.0f - p))) * inv_total;
        },
        [epsilon](uint32_t total, float inv_total) { return Constants{total, epsilon, inv_total}; },
        engine_ptr);
}

void cce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::cce");
    validateProbEpsilon(epsilon, "ops::loss::cce");
    struct Constants
    {
        uint32_t total_elements;
        float epsilon;
    };
    runTwoPassSumLoss(
        prediction, target, output, "ops::loss::cce", false,
        Compute_Pipeline::CCE_LOSS, Compute_Pipeline::CCE_LOSS,
        [epsilon](float a, float t) {
            const float p = std::clamp(a, epsilon, 1.0f - epsilon);
            return -t * std::log(p);
        },
        [epsilon](uint32_t total) { return Constants{total, epsilon}; },
        engine_ptr);
}

void cceGradient(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    validateLossOperands(prediction, target, "ops::loss::cceGradient");
    pred_impl->validateSameDimensions(*target_impl);

    const size_t total_elements = prediction.getTotalElements();
    const Shape &shape = prediction.getShape();
    const size_t last_dim = (shape.getRank() > 0) ? shape[shape.getRank() - 1] : 0;
    const size_t sample_count = (last_dim > 0) ? (total_elements / last_dim) : prediction.getRows();

    if (total_elements == 0 || sample_count == 0)
    {
        handleEmptyLossGradient(prediction, output, engine_ptr);
        return;
    }

    const float scale = 1.0f / static_cast<float>(sample_count);
    dispatchScaledDiffGradient(prediction, target, output, scale, "ops::loss::cceGradient", engine_ptr);
}

void huber(const Tensor &prediction, const Tensor &target, Tensor &output, float delta, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::huber");
    validatePositiveFinite(delta, "delta", "ops::loss::huber");
    struct Constants
    {
        uint32_t total_elements;
        float delta;
    };
    runTwoPassSumLoss(
        prediction, target, output, "ops::loss::huber", false,
        Compute_Pipeline::HUBER_LOSS, Compute_Pipeline::HUBER_LOSS,
        [delta](float a, float t) {
            const float diff = std::abs(a - t);
            return (diff <= delta) ? (0.5f * diff * diff) : (delta * (diff - 0.5f * delta));
        },
        [delta](uint32_t total) { return Constants{total, delta}; },
        engine_ptr);
}

void huberGradient(const Tensor &prediction, const Tensor &target, Tensor &output, float delta, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;
    validateLossOperands(prediction, target, "ops::loss::huberGradient");
    validatePositiveFinite(delta, "delta", "ops::loss::huberGradient");
    struct Constants
    {
        uint32_t total_elements;
        float delta;
        float inv_total_elements;
    };
    runElementwiseLossGradient(
        prediction, target, output, "ops::loss::huberGradient",
        Compute_Pipeline::HUBER_GRADIENT, Compute_Pipeline::HUBER_GRADIENT_FP16,
        [delta](float raw_p, float y, float inv_total) {
            const float diff = raw_p - y;
            if (std::isnan(diff))
            {
                return diff;
            }
            const float abs_diff = std::abs(diff);
            return (abs_diff <= delta) ? (diff * inv_total) : ((diff > 0.0f ? delta : -delta) * inv_total);
        },
        [delta](uint32_t total, float inv_total) { return Constants{total, delta, inv_total}; },
        engine_ptr);
}

float fusedCrossEntropy(const Tensor &logits, const Tensor &targets, Tensor &d_logits, const Cross_Entropy_Config &config)
{
    if (!logits.getImplementation() || !targets.getImplementation())
    {
        return 0.0f;
    }
    if (!std::isfinite(config.label_smoothing) || config.label_smoothing < 0.0f || config.label_smoothing >= 1.0f)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: label_smoothing must be finite and in [0, 1)");
    }
    if (logits.getDataType() == Data_Type::TERNARY_INT2 ||
        targets.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: TERNARY_INT2 is not supported");
    }

    const Shape &shape = logits.getShape();
    if (shape.getRank() == 0)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: logits rank must be >= 1");
    }
    size_t V = shape[shape.getRank() - 1];
    size_t total_tokens = (V > 0) ? (logits.getTotalElements() / V) : 0;

    uint32_t valid_tokens = config.valid_tokens;
    const int32_t ignore_index = config.ignore_index;
    const float smooth_eps = config.label_smoothing;
    const bool compute_grad = config.compute_grad;
    const bool readback_loss = config.readback_loss;

    if (total_tokens == 0 || V == 0)
    {
        if (compute_grad)
        {
            ensureUniqueTensorTarget(d_logits, logits.getExecutionTarget(), shape, logits.getDataType());
        }
        return 0.0f;
    }

    if (logits.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto logits_data = logits.getData();
        const auto targets_data = targets.getData();

        if (compute_grad)
        {
            ensureUniqueTensorTarget(d_logits, logits.getExecutionTarget(), shape, logits.getDataType());
        }

        if (valid_tokens == 0)
        {
            for (size_t s = 0; s < total_tokens; ++s)
            {
                if (s < targets_data.size())
                {
                    int32_t target = 0;
                    if (tryFloatToTokenIndex(targets_data[s], target) &&
                        target >= 0 && target != ignore_index && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
            }
        }

        if (valid_tokens == 0)
        {
            if (compute_grad)
            {
                std::vector<float> zero_grad(total_tokens * V, 0.0f);
                writeCpuTensorPreserveDtype(d_logits, shape, logits.getDataType(), std::move(zero_grad));
            }
            return 0.0f;
        }

        float scale = 1.0f / static_cast<float>(valid_tokens);
        float total_loss = 0.0f;
        std::vector<float> grad_result;
        if (compute_grad)
        {
            grad_result.assign(total_tokens * V, 0.0f);
        }

        for (size_t s = 0; s < total_tokens; ++s)
        {
            int32_t target = -100;
            if (s >= targets_data.size() || !tryFloatToTokenIndex(targets_data[s], target) ||
                target < 0 || target == ignore_index || static_cast<size_t>(target) >= V)
            {
                continue;
            }

            const float *row_logits = logits_data.data() + s * V;

            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < V; ++v)
            {
                float val = row_logits[v];
                if (!std::isnan(val) && !std::isinf(val))
                {
                    max_val = std::max(max_val, val);
                }
            }

            float sum_exp = 0.0f;
            if (std::isfinite(max_val))
            {
                for (size_t v = 0; v < V; ++v)
                {
                    float val = row_logits[v];
                    if (!std::isnan(val) && !std::isinf(val))
                    {
                        sum_exp += std::exp(val - max_val);
                    }
                }
            }

            float mean_logit = 0.0f;
            if (smooth_eps > 0.0f)
            {
                float logit_sum = 0.0f;
                for (size_t v = 0; v < V; ++v)
                {
                    float val = row_logits[v];
                    if (!std::isnan(val) && !std::isinf(val))
                    {
                        logit_sum += val;
                    }
                }
                mean_logit = logit_sum / static_cast<float>(V);
            }

            float lse = max_val + std::log(std::max(sum_exp, 1e-12f));
            float target_logit = row_logits[target];
            bool non_finite_loss = !std::isfinite(target_logit) || !std::isfinite(sum_exp) || sum_exp <= 0.0f;
            float token_loss = non_finite_loss
                ? std::numeric_limits<float>::quiet_NaN()
                : (smooth_eps > 0.0f
                       ? ((1.0f - smooth_eps) * (lse - target_logit) + smooth_eps * (lse - mean_logit))
                       : (lse - target_logit));
            total_loss += token_loss;

            if (compute_grad)
            {
                float *row_grad = grad_result.data() + s * V;
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                float uniform_q = (smooth_eps > 0.0f) ? (smooth_eps / static_cast<float>(V)) : 0.0f;
                for (size_t v = 0; v < V; ++v)
                {
                    float logit_val = row_logits[v];
                    float prob = 0.0f;
                    if (!std::isnan(logit_val) && !std::isinf(logit_val))
                    {
                        prob = std::exp(logit_val - max_val) * inv_sum;
                    }
                    float q_target = (smooth_eps > 0.0f)
                        ? (((v == static_cast<size_t>(target)) ? (1.0f - smooth_eps) : 0.0f) + uniform_q)
                        : ((v == static_cast<size_t>(target)) ? 1.0f : 0.0f);
                    row_grad[v] = (prob - q_target) * scale;
                }
            }
        }

        if (compute_grad)
        {
            writeCpuTensorPreserveDtype(d_logits, shape, logits.getDataType(), std::move(grad_result));
        }
        return total_loss * scale;
    }
    else
    {
        if (total_tokens > std::numeric_limits<uint32_t>::max() ||
            V > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::loss::fusedCrossEntropy: logits dimensions exceed uint32_t");
        }
        Execution_Engine &engine = resolveExecutionEngine(config.engine);
        if (total_tokens > engine.getContext().getMaxComputeWorkGroupCount()[0])
        {
            throw std::overflow_error("ops::loss::fusedCrossEntropy: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        auto &logits_gpu = Gpu_Tensor_Impl::castToGpu(*logits.getImplementation());
        auto contig_logits = logits_gpu.ensureContiguousSelf();
        const auto *effective_logits = contig_logits ? contig_logits.get() : &logits_gpu;

        bool is_fp16 = (effective_logits->getDataType() == Data_Type::FLOAT16) &&
                       engine.getContext().isFloat16Enabled();
        std::shared_ptr<gpu::vector> logits_buf;
        std::vector<std::shared_ptr<Gpu_Tensor_Impl>> temp_impls;
        std::vector<std::shared_ptr<gpu::vector>> temp_buffers;
        if (is_fp16)
        {
            logits_buf = effective_logits->getEffectiveFp16Storage();
        }
        else
        {
            logits_buf = getGpuFloat32Buffer(*effective_logits, temp_impls, temp_buffers, engine);
        }

        std::shared_ptr<gpu::vector> targets_buf;
        bool known_zero_valid = false;
        const auto *targets_gpu_ptr = (targets.getExecutionTarget() == Execution_Target::VULKAN_GPU)
            ? dynamic_cast<const Gpu_Tensor_Impl *>(targets.getImplementation().get())
            : nullptr;

        if (targets_gpu_ptr &&
            targets_gpu_ptr->getDataType() == Data_Type::FLOAT32 &&
            targets_gpu_ptr->isContiguous() &&
            targets_gpu_ptr->getByteOffset() == 0 &&
            targets_gpu_ptr->getStorageBuffer() &&
            targets_gpu_ptr->getStorageBuffer()->getSizeBytes() >= total_tokens * sizeof(float))
        {
            targets_buf = targets_gpu_ptr->getStorageBuffer();
            if (valid_tokens == 0 && !engine.isStaticGraphEnabled() && targets_gpu_ptr->getHostCache().size() >= total_tokens)
            {
                for (size_t s = 0; s < total_tokens; ++s)
                {
                    int32_t target = 0;
                    if (tryFloatToTokenIndex(targets_gpu_ptr->getHostCache()[s], target) &&
                        target >= 0 && target != ignore_index && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
                if (valid_tokens == 0)
                {
                    known_zero_valid = true;
                }
            }
        }
        else
        {
            const auto targets_data = targets.getData();
            if (valid_tokens == 0)
            {
                for (size_t s = 0; s < total_tokens && s < targets_data.size(); ++s)
                {
                    int32_t target = 0;
                    if (tryFloatToTokenIndex(targets_data[s], target) &&
                        target >= 0 && target != ignore_index && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
                if (valid_tokens == 0)
                {
                    known_zero_valid = true;
                }
            }
            std::vector<float> tgt_float(total_tokens, -100.0f);
            for (size_t s = 0; s < std::min(total_tokens, targets_data.size()); ++s)
            {
                tgt_float[s] = targets_data[s];
            }
            targets_buf = std::make_shared<gpu::vector>(engine.getContext(), tgt_float);
        }

        std::shared_ptr<gpu::vector> grad_out_buf;
        if (compute_grad)
        {
            Data_Type d_logits_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
            ensureUniqueTensorTarget(d_logits, logits.getExecutionTarget(), shape, d_logits_dtype);

            auto &d_logits_gpu = Gpu_Tensor_Impl::castToGpu(*d_logits.getImplementation());
            size_t elem_size = getDataTypeSize(d_logits_dtype);

            if (!d_logits_gpu.getStorageBuffer() || d_logits_gpu.getStorageBuffer()->getSizeBytes() < total_tokens * V * elem_size)
            {
                d_logits_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext());
                d_logits_gpu.getStorageBuffer()->allocateMemory(total_tokens * V, d_logits_dtype);
            }
            grad_out_buf = d_logits_gpu.getStorageBuffer();
        }
        else
        {
            grad_out_buf = engine.acquireReusableScratchBuffer(
                Execution_Engine::SCRATCH_LOSS_PARTIALS,
                sizeof(float),
                Data_Type::FLOAT32,
                1);
        }

        engine.prepareCurrentFrame(Execution_Stage::FORWARD);
        uint32_t cur_frame = engine.getContext().getCurrentFrame();
        auto &slot = engine.getLossSlot(cur_frame);
        size_t required_bytes = (total_tokens + 1) * sizeof(float);
        if (!slot.buffer || slot.buffer->getSizeBytes() < required_bytes)
        {
            slot.buffer = std::make_shared<gpu::vector>(engine.getContext());
            slot.buffer->allocateHostVisible(required_bytes);
        }
        if (!slot.valid_count_buffer || slot.valid_count_buffer->getSizeBytes() < sizeof(uint32_t))
        {
            slot.valid_count_buffer = std::make_shared<gpu::vector>(engine.getContext());
            slot.valid_count_buffer->allocateMemory(1, Data_Type::FLOAT32);
        }

        if (valid_tokens == 0)
        {
            struct Count_Valid_Constants
            {
                uint32_t seq_len;
                uint32_t vocab_size;
                int32_t ignore_index;
            } count_pcs{ static_cast<uint32_t>(total_tokens), static_cast<uint32_t>(V), ignore_index };
            validatePushConstants<Count_Valid_Constants>();

            logits_gpu.pushToGraph(Compute_Pipeline::COUNT_VALID_TARGETS,
                                   { targets_buf, slot.valid_count_buffer },
                                   count_pcs,
                                   1, 1, 1, &engine);
        }

        struct Fused_CE_Constants
        {
            uint32_t seq_len;
            uint32_t vocab_size;
            uint32_t valid_tokens;
            int32_t ignore_index;
            uint32_t compute_grad;
            float label_smoothing;
        } pcs{ static_cast<uint32_t>(total_tokens), static_cast<uint32_t>(V), valid_tokens, ignore_index, compute_grad ? 1u : 0u, smooth_eps };
        validatePushConstants<Fused_CE_Constants>();

        Compute_Pipeline pipe = is_fp16
            ? Compute_Pipeline::FUSED_CROSS_ENTROPY_FP16
            : Compute_Pipeline::FUSED_CROSS_ENTROPY;

        logits_gpu.pushToGraph(pipe,
                               { logits_buf, targets_buf, slot.valid_count_buffer, grad_out_buf, slot.buffer },
                               pcs,
                               static_cast<uint32_t>(total_tokens), 1, 1, &engine);
        if (compute_grad)
        {
            auto &d_logits_gpu = Gpu_Tensor_Impl::castToGpu(*d_logits.getImplementation());
            d_logits_gpu.clearHostCache();
            d_logits_gpu.invalidateFp16Cache();
        }

        const bool defer_exec = config.defer_execution || engine.isDeferForwardLossActive();
        if (!defer_exec)
        {
            engine.executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
        }

        slot.valid_tokens = valid_tokens;
        slot.total_tokens = total_tokens;
        slot.has_pending_read = readback_loss && !known_zero_valid;
        slot.is_submitted = !defer_exec;

        if (!readback_loss || known_zero_valid)
        {
            return 0.0f;
        }

        if (defer_exec || engine.isAsyncLossEnabled())
        {
            return engine.getLatestLoss();
        }
        else
        {
            return engine.readPendingLoss(cur_frame);
        }
    }
}

float fusedCrossEntropy(const Tensor &logits, const std::vector<int32_t> &targets, Tensor &d_logits, const Cross_Entropy_Config &config)
{
    if (!logits.getImplementation())
    {
        return 0.0f;
    }
    if (!std::isfinite(config.label_smoothing) || config.label_smoothing < 0.0f || config.label_smoothing >= 1.0f)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: label_smoothing must be finite and in [0, 1)");
    }
    if (logits.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: TERNARY_INT2 logits are not supported");
    }
    const Shape &shape = logits.getShape();
    if (shape.getRank() == 0)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: logits rank must be >= 1");
    }
    size_t V = shape[shape.getRank() - 1];
    size_t S = (V > 0) ? (logits.getTotalElements() / V) : 0;

    if (!(S == 0 && targets.empty()) && targets.size() != S)
    {
        throw std::invalid_argument("ops::loss::fusedCrossEntropy: targets size does not match logits token count");
    }

    if (config.compute_grad)
    {
        ensureUniqueTensorTarget(d_logits, logits.getExecutionTarget(), shape, logits.getDataType());
    }
    if (S == 0 || V == 0)
    {
        return 0.0f;
    }

    std::vector<float> tgt_float(S, -100.0f);
    uint32_t counted_valid = 0;
    for (size_t i = 0; i < std::min(S, targets.size()); ++i)
    {
        tgt_float[i] = static_cast<float>(targets[i]);
        if (targets[i] >= 0 && targets[i] != config.ignore_index && static_cast<size_t>(targets[i]) < V)
        {
            counted_valid++;
        }
    }
    if (config.valid_tokens == 0 && counted_valid == 0)
    {
        if (config.compute_grad)
        {
            if (logits.getExecutionTarget() == Execution_Target::CPU)
            {
                writeCpuTensorPreserveDtype(d_logits, shape, logits.getDataType(), std::vector<float>(S * V, 0.0f));
            }
            else
            {
                Execution_Engine &engine = resolveExecutionEngine(config.engine);
                auto &d_logits_gpu = Gpu_Tensor_Impl::castToGpu(*d_logits.getImplementation());
                pushGpuFill(d_logits_gpu, S * V, 0.0f, engine);
            }
        }
        return 0.0f;
    }
    Cross_Entropy_Config fwd_config = config;
    fwd_config.valid_tokens = (config.valid_tokens > 0) ? config.valid_tokens : counted_valid;
    Tensor tgt_tensor(Shape{ S }, std::move(tgt_float), logits.getExecutionTarget());
    return fusedCrossEntropy(logits, tgt_tensor, d_logits, fwd_config);
}

float fusedCrossEntropy(const Tensor &logits, const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens, int32_t ignore_index, bool compute_grad, bool readback_loss)
{
    Cross_Entropy_Config config;
    config.valid_tokens = valid_tokens;
    config.ignore_index = ignore_index;
    config.compute_grad = compute_grad;
    config.readback_loss = readback_loss;
    return fusedCrossEntropy(logits, targets, d_logits, config);
}

float fusedCrossEntropy(const Tensor &logits, const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens, int32_t ignore_index, bool compute_grad, bool readback_loss)
{
    Cross_Entropy_Config config;
    config.valid_tokens = valid_tokens;
    config.ignore_index = ignore_index;
    config.compute_grad = compute_grad;
    config.readback_loss = readback_loss;
    return fusedCrossEntropy(logits, targets, d_logits, config);
}

void asl(const Tensor &prediction, const Tensor &target, Tensor &output,
         float gamma_pos, float gamma_neg, float clip_margin, float epsilon, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;

    validateLossOperands(prediction, target, "ops::loss::asl");
    validateAslParams(gamma_pos, gamma_neg, clip_margin, epsilon, "ops::loss::asl");

    struct Constants
    {
        uint32_t total_elements;
        float gamma_pos;
        float gamma_neg;
        float clip_margin;
        float epsilon_val;
    };
    runTwoPassSumLoss(
        prediction, target, output, "ops::loss::asl", true,
        Compute_Pipeline::ASL_LOSS, Compute_Pipeline::ASL_LOSS_FP16,
        [gamma_pos, gamma_neg, clip_margin, epsilon](float a, float y) {
            const float p = std::clamp(a, epsilon, 1.0f - epsilon);
            float pos_loss = 0.0f;
            if (y > 0.0f)
            {
                const float pt = 1.0f - p;
                const float mod_factor = (gamma_pos > 0.0f) ? std::pow(pt, gamma_pos) : 1.0f;
                pos_loss = -y * mod_factor * std::log(p);
            }
            float neg_loss = 0.0f;
            if (y < 1.0f)
            {
                float p_m = std::max(p - clip_margin, 0.0f);
                p_m = std::clamp(p_m, 0.0f, 1.0f - epsilon);
                if (p_m > 0.0f)
                {
                    const float mod_factor = (gamma_neg > 0.0f) ? std::pow(p_m, gamma_neg) : 1.0f;
                    neg_loss = -(1.0f - y) * mod_factor * std::log(1.0f - p_m);
                }
            }
            return pos_loss + neg_loss;
        },
        [gamma_pos, gamma_neg, clip_margin, epsilon](uint32_t total) {
            return Constants{total, gamma_pos, gamma_neg, clip_margin, epsilon};
        },
        engine_ptr);
}

void aslGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                 float gamma_pos, float gamma_neg, float clip_margin, float epsilon, Execution_Engine *engine_ptr)
{
    if (!prediction.getImplementation() || !target.getImplementation()) return;

    validateLossOperands(prediction, target, "ops::loss::aslGradient");
    validateAslParams(gamma_pos, gamma_neg, clip_margin, epsilon, "ops::loss::aslGradient");

    const float eff_eps = (prediction.getDataType() == Data_Type::FLOAT16) ? std::max(epsilon, 1e-4f) : epsilon;
    struct Constants
    {
        uint32_t total_elements;
        float gamma_pos;
        float gamma_neg;
        float clip_margin;
        float epsilon_val;
        float inv_total_elements;
    };
    runElementwiseLossGradient(
        prediction, target, output, "ops::loss::aslGradient",
        Compute_Pipeline::ASL_GRADIENT, Compute_Pipeline::ASL_GRADIENT_FP16,
        [gamma_pos, gamma_neg, clip_margin, eff_eps](float raw_p, float y, float inv_total_elements) {
            if (std::isnan(raw_p) || std::isnan(y))
            {
                return raw_p + y;
            }
            const float p = std::clamp(raw_p, eff_eps, 1.0f - eff_eps);

            float dL_pos = 0.0f;
            if (y > 0.0f)
            {
                const float pt = 1.0f - p;
                if (gamma_pos <= 0.0f)
                {
                    dL_pos = -1.0f / p;
                }
                else
                {
                    const float pt_gamma_minus_1 = (gamma_pos == 1.0f) ? 1.0f : std::pow(pt, gamma_pos - 1.0f);
                    const float pt_gamma = pt_gamma_minus_1 * pt;
                    dL_pos = gamma_pos * pt_gamma_minus_1 * std::log(p) - (pt_gamma / p);
                }
            }

            float dL_neg = 0.0f;
            if (y < 1.0f)
            {
                float p_m = std::max(p - clip_margin, 0.0f);
                p_m = std::clamp(p_m, 0.0f, 1.0f - eff_eps);
                if (p_m > 0.0f)
                {
                    const float q_m = 1.0f - p_m;
                    if (gamma_neg <= 0.0f)
                    {
                        dL_neg = 1.0f / q_m;
                    }
                    else
                    {
                        const float pm_safe = (gamma_neg < 1.0f) ? std::max(p_m, eff_eps) : p_m;
                        const float pm_gamma_minus_1 = (gamma_neg == 1.0f) ? 1.0f : std::pow(pm_safe, gamma_neg - 1.0f);
                        const float pm_gamma = pm_gamma_minus_1 * p_m;
                        dL_neg = -gamma_neg * pm_gamma_minus_1 * std::log(q_m) + (pm_gamma / q_m);
                    }
                }
            }

            return (y * dL_pos + (1.0f - y) * dL_neg) * inv_total_elements;
        },
        [gamma_pos, gamma_neg, clip_margin, epsilon](uint32_t total, float inv_total_elements) {
            return Constants{total, gamma_pos, gamma_neg, clip_margin, epsilon, inv_total_elements};
        },
        engine_ptr);
}

void dice(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon, bool write_stats, Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    validateLossOperands(prediction, target, "ops::loss::dice");
    validatePositiveFinite(epsilon, "epsilon", "ops::loss::dice");
    pred_impl->validateSameDimensions(*target_impl);

    size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        if (prediction.getExecutionTarget() == Execution_Target::CPU)
        {
            if (write_stats)
            {
                writeCpuTensorPreserveDtype(output, Shape{1, 4}, Data_Type::FLOAT32, {0.0f, 0.0f, 0.0f, 0.0f});
            }
            else
            {
                writeCpuScalarFloat32(output, 0.0f);
            }
        }
        else
        {
            Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
            const Shape out_shape = write_stats ? Shape{1, 4} : Shape{1, 1};
            ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, out_shape, Data_Type::FLOAT32);
            auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            pushGpuFill(output_gpu, write_stats ? 4 : 1, 0.0f, engine);
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
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], 0.0f, 1.0f);
            float y = t_data[i];
            total_inter += p * y;
            total_pred += p;
            total_target += y;
        }

        float dice_score = (2.0f * total_inter + epsilon) / (total_pred + total_target + epsilon);
        float loss_val = 1.0f - dice_score;

        if (write_stats)
        {
            writeCpuTensorPreserveDtype(output, Shape{1, 4}, Data_Type::FLOAT32,
                                        {loss_val, total_inter, total_pred, total_target});
        }
        else
        {
            writeCpuScalarFloat32(output, loss_val);
        }
    }
    else
    {
        const uint32_t total = validateU32Count(total_elements, "ops::loss::dice");
        Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
        const Shape out_shape = write_stats ? Shape{1, 4} : Shape{1, 1};
        ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, out_shape, Data_Type::FLOAT32);

        auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        auto gpu_inputs = resolveGpuLossInputs(prediction, pred_gpu, target_gpu, true, engine);

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        output_gpu.setDataType(Data_Type::FLOAT32);
        output.reshape(out_shape);

        Compute_Pipeline pipe = gpu_inputs.is_fp16 ? Compute_Pipeline::DICE_LOSS_FP16 : Compute_Pipeline::DICE_LOSS;
        dispatchDiceReduce2Pass(pred_gpu, pipe, gpu_inputs.buf_pred, gpu_inputs.buf_target, output_gpu.getStorageBuffer(), total, epsilon, write_stats ? 1u : 0u, "ops::loss::dice", engine);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void diceGradient(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon, const Tensor* stats_tensor, Execution_Engine *engine_ptr)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    validateLossOperands(prediction, target, "ops::loss::diceGradient");
    validatePositiveFinite(epsilon, "epsilon", "ops::loss::diceGradient");
    pred_impl->validateSameDimensions(*target_impl);

    if (stats_tensor != nullptr)
    {
        if (!stats_tensor->getImplementation())
        {
            throw std::invalid_argument("ops::loss::diceGradient: stats_tensor has null implementation");
        }
        if (stats_tensor->getExecutionTarget() != prediction.getExecutionTarget())
        {
            throw std::invalid_argument("ops::loss::diceGradient: stats_tensor Execution_Target mismatch");
        }
        if (stats_tensor->getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::loss::diceGradient: stats_tensor cannot be TERNARY_INT2");
        }
        if (stats_tensor->getTotalElements() < 4)
        {
            throw std::invalid_argument("ops::loss::diceGradient: stats_tensor must contain at least 4 elements");
        }
    }

    size_t total_elements = prediction.getTotalElements();
    if (total_elements == 0)
    {
        handleEmptyLossGradient(prediction, output, engine_ptr);
        return;
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto a_data = prediction.getData();
        const auto t_data = target.getData();

        float total_inter = 0.0f;
        float total_pred = 0.0f;
        float total_target = 0.0f;
        if (stats_tensor != nullptr)
        {
            const auto stats_data = stats_tensor->getData();
            total_inter = stats_data[1];
            total_pred = stats_data[2];
            total_target = stats_data[3];
        }
        else
        {
            for (size_t i = 0; i < total_elements; ++i)
            {
                float p = std::clamp(a_data[i], 0.0f, 1.0f);
                float y = t_data[i];
                total_inter += p * y;
                total_pred += p;
                total_target += y;
            }
        }

        float denom = total_pred + total_target + epsilon;
        float numer = 2.0f * total_inter + epsilon;
        float denom_sq = denom * denom;
        float inv_denom_sq = (denom_sq > 0.0f) ? (1.0f / denom_sq) : 0.0f;

        std::vector<float> grad_out(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            float a = a_data[i];
            if (a < 0.0f || a > 1.0f)
            {
                grad_out[i] = 0.0f;
            }
            else
            {
                float y = t_data[i];
                float g = (denom_sq > 0.0f) ? ((numer - 2.0f * y * denom) * inv_denom_sq) : 0.0f;
                if (std::isnan(g) || std::isinf(g)) g = 0.0f;
                grad_out[i] = g;
            }
        }
        writeCpuTensorPreserveDtype(output, prediction.getShape(), prediction.getDataType(), std::move(grad_out));
    }
    else
    {
        const uint32_t total = validateU32Count(total_elements, "ops::loss::diceGradient");
        Execution_Engine &engine = resolveExecutionEngine(engine_ptr);
        auto &pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        const auto &target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        auto gpu_inputs = resolveGpuLossInputs(prediction, pred_gpu, target_gpu, true, engine);
        ensureUniqueTensorTarget(output, Execution_Target::VULKAN_GPU, prediction.getShape(), gpu_inputs.grad_dtype);

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        output_gpu.setDataType(gpu_inputs.grad_dtype);
        output.reshape(prediction.getShape());

        std::shared_ptr<gpu::vector> stats_buf;
        if (stats_tensor != nullptr)
        {
            if (stats_tensor->getDataType() != Data_Type::FLOAT32)
            {
                throw std::invalid_argument("ops::loss::diceGradient: GPU stats_tensor must use FLOAT32 Data_Type");
            }
            const auto &stats_gpu = Gpu_Tensor_Impl::castToGpu(*stats_tensor->getImplementation());
            stats_buf = getGpuFloat32Buffer(stats_gpu, gpu_inputs.temp_impls, gpu_inputs.temp_buffers, engine);
        }
        else
        {
            stats_buf = std::make_shared<gpu::vector>(engine.getContext());
            stats_buf->allocateMemory(4, Data_Type::FLOAT32);

            Compute_Pipeline loss_pipe = gpu_inputs.is_fp16 ? Compute_Pipeline::DICE_LOSS_FP16 : Compute_Pipeline::DICE_LOSS;
            dispatchDiceReduce2Pass(pred_gpu, loss_pipe, gpu_inputs.buf_pred, gpu_inputs.buf_target, stats_buf, total, epsilon, 1u, "ops::loss::diceGradient", engine);
        }

        struct Grad_Constants
        {
            uint32_t total_elements;
            float epsilon_val;
        } grad_pc{ total, epsilon };
        validatePushConstants<Grad_Constants>();

        auto [wg_x, wg_y] = ops::internal::dispatch2D((total_elements + 255) / 256, &engine);
        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (wg_x > max_wg[0] || wg_y > max_wg[1])
        {
            throw std::overflow_error("ops::loss::diceGradient: Workgroup grid exceeds hardware limits");
        }
        Compute_Pipeline grad_pipe = gpu_inputs.is_fp16 ? Compute_Pipeline::DICE_GRADIENT_FP16 : Compute_Pipeline::DICE_GRADIENT;

        pred_gpu.pushToGraph(grad_pipe, { gpu_inputs.buf_pred, gpu_inputs.buf_target, stats_buf, output_gpu.getStorageBuffer() }, grad_pc, wg_x, wg_y, 1, &engine);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

}

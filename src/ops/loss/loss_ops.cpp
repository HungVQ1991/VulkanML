#include "ops/loss/loss_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "engine/data_type.h"
#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::loss
{

void mse(const Tensor &prediction, const Tensor &target, Tensor &output)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }

        float loss = 0.0f;
        size_t total_elements = prediction.getTotalElements();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float diff = a_data[i] - t_data[i];
            loss += diff * diff;
        }
        (*output_cpu.getStorageBuffer())[0] = loss;
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

        uint32_t total = static_cast<uint32_t>(prediction.getTotalElements());
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
        } c{total};

        pred_gpu.pushToGraph(Compute_Pipeline::MSE_LOSS, {effective_self->getStorageBuffer(), effective_target->getStorageBuffer(), output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
    }
}

void mae(const Tensor &prediction, const Tensor &target, Tensor &output)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }

        float loss = 0.0f;
        size_t total_elements = prediction.getTotalElements();
        for (size_t i = 0; i < total_elements; ++i)
        {
            loss += std::abs(a_data[i] - t_data[i]);
        }
        (*output_cpu.getStorageBuffer())[0] = loss;
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

        uint32_t total = static_cast<uint32_t>(prediction.getTotalElements());
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
        } c{total};

        pred_gpu.pushToGraph(Compute_Pipeline::MAE_LOSS, {effective_self->getStorageBuffer(), effective_target->getStorageBuffer(), output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
    }
}

void bce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }

        float loss = 0.0f;
        size_t total_elements = prediction.getTotalElements();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0f - epsilon);
            float y = t_data[i];
            loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }
        (*output_cpu.getStorageBuffer())[0] = loss;
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

        uint32_t total = static_cast<uint32_t>(prediction.getTotalElements());
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float epsilon;
        } c{total, epsilon};

        pred_gpu.pushToGraph(Compute_Pipeline::BCE_LOSS, {effective_self->getStorageBuffer(), effective_target->getStorageBuffer(), output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
    }
}

void cce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }

        float loss = 0.0f;
        size_t total_elements = prediction.getTotalElements();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0f - epsilon);
            loss += -t_data[i] * std::log(p);
        }
        (*output_cpu.getStorageBuffer())[0] = loss;
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

        uint32_t total = static_cast<uint32_t>(prediction.getTotalElements());
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float epsilon;
        } c{total, epsilon};

        pred_gpu.pushToGraph(Compute_Pipeline::CCE_LOSS, {effective_self->getStorageBuffer(), effective_target->getStorageBuffer(), output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
    }
}

void huber(const Tensor &prediction, const Tensor &target, Tensor &output, float delta)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }

        float loss = 0.0f;
        size_t total_elements = prediction.getTotalElements();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float diff = std::abs(a_data[i] - t_data[i]);
            loss += (diff <= delta) ? 0.5f * diff * diff : delta * (diff - 0.5f * delta);
        }
        (*output_cpu.getStorageBuffer())[0] = loss;
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

        uint32_t total = static_cast<uint32_t>(prediction.getTotalElements());
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float delta;
        } c{total, delta};

        pred_gpu.pushToGraph(Compute_Pipeline::HUBER_LOSS, {effective_self->getStorageBuffer(), effective_target->getStorageBuffer(), output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
    }
}


float fusedCrossEntropy(const Tensor &logits, const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens)
{
    const Shape &shape = logits.getShape();
    size_t total_tokens = (shape.getRank() == 3) ? (shape[0] * shape[1]) : logits.getRows();
    size_t V = (shape.getRank() == 3) ? shape[2] : logits.getColumns();

    if (d_logits.getExecutionTarget() != logits.getExecutionTarget() ||
        d_logits.getDataType() != logits.getDataType() ||
        d_logits.getShape() != shape)
    {
        d_logits = Tensor(shape, logits.getDataType(), logits.getExecutionTarget());
    }
    else
    {
        d_logits.reshape(shape);
    }

    if (total_tokens == 0 || V == 0)
    {
        return 0.0f;
    }

    if (logits.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &logits_data = logits.getData();
        const auto &targets_data = targets.getData();
        auto &d_logits_cpu = Cpu_Tensor_Impl::castToCpu(*d_logits.getImplementation());
        auto &grad_data = d_logits_cpu.getStorageBuffer();
        if (!grad_data || grad_data->size() < total_tokens * V)
        {
            grad_data = std::make_shared<std::vector<float>>(total_tokens * V, 0.0f);
        }
        else
        {
            std::fill(grad_data->begin(), grad_data->begin() + total_tokens * V, 0.0f);
        }

        if (valid_tokens == 0)
        {
            for (size_t s = 0; s < total_tokens; ++s)
            {
                if (s < targets_data.size())
                {
                    int32_t target = static_cast<int32_t>(std::round(targets_data[s]));
                    if (target >= 0 && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
            }
        }

        if (valid_tokens == 0)
        {
            return 0.0f;
        }

        float scale = 1.0f / static_cast<float>(valid_tokens);
        float total_loss = 0.0f;

        for (size_t s = 0; s < total_tokens; ++s)
        {
            int32_t target = (s < targets_data.size()) ? static_cast<int32_t>(std::round(targets_data[s])) : -100;
            if (target < 0 || static_cast<size_t>(target) >= V)
            {
                continue;
            }

            const float *row_logits = logits_data.data() + s * V;
            float *row_grad = grad_data->data() + s * V;

            float max_val = -1e20f;
            for (size_t v = 0; v < V; ++v)
            {
                float val = row_logits[v];
                if (!std::isnan(val) && !std::isinf(val))
                {
                    max_val = std::max(max_val, val);
                }
            }

            float sum_exp = 0.0f;
            for (size_t v = 0; v < V; ++v)
            {
                float val = row_logits[v];
                if (!std::isnan(val) && !std::isinf(val))
                {
                    sum_exp += std::exp(val - max_val);
                }
            }

            float lse = max_val + std::log(std::max(sum_exp, 1e-12f));
            float target_logit = row_logits[target];
            float token_loss = lse - target_logit;
            total_loss += token_loss;

            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (size_t v = 0; v < V; ++v)
            {
                float logit_val = row_logits[v];
                float prob = 0.0f;
                if (!std::isnan(logit_val) && !std::isinf(logit_val))
                {
                    prob = std::exp(logit_val - max_val) * inv_sum;
                }
                float delta = (v == static_cast<size_t>(target)) ? 1.0f : 0.0f;
                row_grad[v] = (prob - delta) * scale;
            }
        }

        if (logits.getDataType() == Data_Type::FLOAT16)
        {
            if (!d_logits_cpu.getStorageBufferFp16() || d_logits_cpu.getStorageBufferFp16()->size() < total_tokens * V)
            {
                d_logits_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(total_tokens * V));
            }
            convertFp32ToFp16(grad_data->data(), d_logits_cpu.getStorageBufferFp16()->data(), total_tokens * V);
            d_logits_cpu.getStorageBuffer().reset();
        }

        return total_loss * scale;
    }
    else
    {
        auto &logits_gpu = Gpu_Tensor_Impl::castToGpu(*logits.getImplementation());
        auto &d_logits_gpu = Gpu_Tensor_Impl::castToGpu(*d_logits.getImplementation());
        size_t elem_size = getDataTypeSize(logits.getDataType());

        if (!d_logits_gpu.getStorageBuffer() || d_logits_gpu.getStorageBuffer()->getSizeBytes() < total_tokens * V * elem_size)
        {
            d_logits_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            d_logits_gpu.getStorageBuffer()->allocateMemory(total_tokens * V, logits.getDataType());
        }

        std::shared_ptr<gpu::vector> targets_buf;
        const auto *targets_gpu_ptr = dynamic_cast<const Gpu_Tensor_Impl *>(targets.getImplementation().get());
        if (targets_gpu_ptr && targets_gpu_ptr->getStorageBuffer() && targets_gpu_ptr->getStorageBuffer()->getSizeBytes() >= total_tokens * sizeof(float))
        {
            targets_buf = targets_gpu_ptr->getStorageBuffer();
            if (valid_tokens == 0)
            {
                if (targets_gpu_ptr->getHostCache().size() >= total_tokens)
                {
                    for (size_t s = 0; s < total_tokens; ++s)
                    {
                        int32_t target = static_cast<int32_t>(std::round(targets_gpu_ptr->getHostCache()[s]));
                        if (target >= 0 && static_cast<size_t>(target) < V)
                        {
                            ++valid_tokens;
                        }
                    }
                }
                else
                {
                    valid_tokens = static_cast<uint32_t>(total_tokens);
                }
            }
        }
        else
        {
            const auto &targets_data = targets.getData();
            if (valid_tokens == 0)
            {
                for (size_t s = 0; s < total_tokens && s < targets_data.size(); ++s)
                {
                    int32_t target = static_cast<int32_t>(std::round(targets_data[s]));
                    if (target >= 0 && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
            }
            std::vector<float> tgt_float(total_tokens, -100.0f);
            for (size_t s = 0; s < std::min(total_tokens, targets_data.size()); ++s)
            {
                tgt_float[s] = targets_data[s];
            }
            targets_buf = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), tgt_float);
        }

        uint32_t cur_frame = Execution_Engine::getInstance().getContext().getCurrentFrame();
        auto &slot = Execution_Engine::getInstance().getLossSlot(cur_frame);
        size_t required_bytes = total_tokens * sizeof(float);
        if (!slot.buffer || slot.buffer->getSizeBytes() < required_bytes)
        {
            slot.buffer = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            slot.buffer->allocateHostVisible(required_bytes);
        }

        struct Fused_CE_Constants
        {
            uint32_t seq_len;
            uint32_t vocab_size;
            uint32_t valid_tokens;
        } pcs{ static_cast<uint32_t>(total_tokens), static_cast<uint32_t>(V), valid_tokens };

        Compute_Pipeline pipe = (logits.getDataType() == Data_Type::FLOAT16)
            ? Compute_Pipeline::FUSED_CROSS_ENTROPY_FP16
            : Compute_Pipeline::FUSED_CROSS_ENTROPY;

        logits_gpu.pushToGraph(pipe,
                               { logits_gpu.getStorageBuffer(), targets_buf, d_logits_gpu.getStorageBuffer(), slot.buffer },
                               pcs,
                               static_cast<uint32_t>(total_tokens), 1, 1);
        d_logits_gpu.clearHostCache();

        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);

        slot.valid_tokens = valid_tokens;
        slot.total_tokens = total_tokens;
        slot.has_pending_read = true;

        if (valid_tokens == 0)
        {
            return 0.0f;
        }

        if (Execution_Engine::getInstance().isAsyncLossEnabled())
        {
            return Execution_Engine::getInstance().getLatestLoss();
        }
        else
        {
            return Execution_Engine::getInstance().readPendingLoss(cur_frame);
        }
    }
}

float fusedCrossEntropy(const Tensor &logits, const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens)
{
    if (d_logits.getExecutionTarget() != logits.getExecutionTarget() || d_logits.getDataType() != logits.getDataType())
    {
        d_logits = Tensor(logits.getShape(), logits.getDataType(), logits.getExecutionTarget());
    }
    size_t S = (logits.getShape().getRank() == 3) ? (logits.getShape()[0] * logits.getShape()[1]) : logits.getRows();
    std::vector<float> tgt_float(S, -100.0f);
    uint32_t counted_valid = 0;
    size_t V = (logits.getShape().getRank() == 3) ? logits.getShape()[2] : logits.getColumns();
    for (size_t i = 0; i < std::min(S, targets.size()); ++i)
    {
        tgt_float[i] = static_cast<float>(targets[i]);
        if (targets[i] >= 0 && static_cast<size_t>(targets[i]) < V)
        {
            counted_valid++;
        }
    }
    uint32_t effective_valid = (valid_tokens > 0) ? valid_tokens : counted_valid;
    Tensor tgt_tensor(Shape{ S }, std::move(tgt_float), logits.getExecutionTarget());
    return fusedCrossEntropy(logits, tgt_tensor, d_logits, effective_valid);
}

void asl(const Tensor &prediction, const Tensor &target, Tensor &output,
         float gamma_pos, float gamma_neg, float clip_margin, float epsilon)
{
    auto pred_impl = prediction.getImplementation();
    auto target_impl = target.getImplementation();
    if (!pred_impl || !target_impl) return;

    pred_impl->validateSameDimensions(*target_impl);
    if (output.getExecutionTarget() != prediction.getExecutionTarget())
    {
        output.setExecutionTarget(prediction.getExecutionTarget());
    }

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }

        float total_loss = 0.0f;
        size_t total_elements = prediction.getTotalElements();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0f - epsilon);
            float y = t_data[i];

            float pos_loss = 0.0f;
            if (y > 0.0f)
            {
                float pt = 1.0f - p;
                float mod_factor = (gamma_pos > 0.0f) ? std::pow(pt, gamma_pos) : 1.0f;
                pos_loss = -y * mod_factor * std::log(p);
            }

            float neg_loss = 0.0f;
            if (y < 1.0f)
            {
                float p_m = std::max(p - clip_margin, 0.0f);
                p_m = std::clamp(p_m, 0.0f, 1.0f - epsilon);
                if (p_m > 0.0f)
                {
                    float mod_factor = (gamma_neg > 0.0f) ? std::pow(p_m, gamma_neg) : 1.0f;
                    neg_loss = -(1.0f - y) * mod_factor * std::log(1.0f - p_m);
                }
            }
            total_loss += (pos_loss + neg_loss);
        }
        (*output_cpu.getStorageBuffer())[0] = total_loss;
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

        uint32_t total = static_cast<uint32_t>(prediction.getTotalElements());
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float gamma_pos;
            float gamma_neg;
            float clip_margin;
            float epsilon_val;
        } c{total, gamma_pos, gamma_neg, clip_margin, epsilon};

        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ASL_LOSS_FP16 : Compute_Pipeline::ASL_LOSS;

        auto buf_pred = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_target = is_fp16 ? effective_target->getEffectiveFp16Storage() : effective_target->getStorageBuffer();

        pred_gpu.pushToGraph(pipe, {buf_pred, buf_target, output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
        output_gpu.clearHostCache();
        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    }
}

void aslGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                 float gamma_pos, float gamma_neg, float clip_margin, float epsilon)
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
    float inv_total_elements = 1.0f / static_cast<float>(total_elements);

    if (prediction.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = prediction.getData();
        const auto &t_data = target.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(prediction.getRows(), prediction.getColumns());
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() != total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements);
        }

        auto &grad_out = *output_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0f - epsilon);
            float y = t_data[i];

            float dL_pos = 0.0f;
            if (y > 0.0f)
            {
                float pt = 1.0f - p;
                if (gamma_pos <= 0.0f)
                {
                    dL_pos = -1.0f / p;
                }
                else
                {
                    float pt_gamma_minus_1 = (gamma_pos == 1.0f) ? 1.0f : std::pow(pt, gamma_pos - 1.0f);
                    float pt_gamma = pt_gamma_minus_1 * pt;
                    dL_pos = gamma_pos * pt_gamma_minus_1 * std::log(p) - (pt_gamma / p);
                }
            }

            float dL_neg = 0.0f;
            if (y < 1.0f)
            {
                float p_m = std::max(p - clip_margin, 0.0f);
                p_m = std::clamp(p_m, 0.0f, 1.0f - epsilon);
                if (p_m > 0.0f)
                {
                    float q_m = 1.0f - p_m;
                    if (gamma_neg <= 0.0f)
                    {
                        dL_neg = 1.0f / q_m;
                    }
                    else
                    {
                        float pm_gamma_minus_1 = (gamma_neg == 1.0f) ? 1.0f : std::pow(p_m, gamma_neg - 1.0f);
                        float pm_gamma = pm_gamma_minus_1 * p_m;
                        dL_neg = -gamma_neg * pm_gamma_minus_1 * std::log(q_m) + (pm_gamma / q_m);
                    }
                }
            }

            float total_grad = y * dL_pos + (1.0f - y) * dL_neg;
            grad_out[i] = total_grad * inv_total_elements;
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
        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
        }
        output_gpu.reshape(prediction.getRows(), prediction.getColumns());

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = (total + 255) / 256;

        struct Constants
        {
            uint32_t total_elements;
            float gamma_pos;
            float gamma_neg;
            float clip_margin;
            float epsilon_val;
            float inv_total_elements;
        } c{total, gamma_pos, gamma_neg, clip_margin, epsilon, inv_total_elements};

        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ASL_GRADIENT_FP16 : Compute_Pipeline::ASL_GRADIENT;

        auto buf_pred = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_target = is_fp16 ? effective_target->getEffectiveFp16Storage() : effective_target->getStorageBuffer();

        pred_gpu.pushToGraph(pipe, {buf_pred, buf_target, output_gpu.getStorageBuffer()}, c, wg_x, 1, 1);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void dice(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon)
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
        const auto& a_data = prediction.getData();
        const auto& t_data = target.getData();
        auto& output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        output_cpu.reshape(1, 1);
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(1, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->empty())
        {
            output_cpu.getStorageBuffer()->resize(1, 0.0f);
        }

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
        (*output_cpu.getStorageBuffer())[0] = loss_val;

        if (prediction.getDataType() == Data_Type::FLOAT16)
        {
            if (!output_cpu.getStorageBufferFp16() || output_cpu.getStorageBufferFp16()->empty())
            {
                output_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(1));
            }
            convertFp32ToFp16(&loss_val, output_cpu.getStorageBufferFp16()->data(), 1);
            output_cpu.getStorageBuffer().reset();
        }
    }
    else
    {
        auto& pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        auto contig_self = pred_gpu.ensureContiguousSelf();
        const auto* effective_self = contig_self ? contig_self.get() : &pred_gpu;

        const auto& target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto* effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto& output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = 1;
        output_gpu.setDataType(Data_Type::FLOAT32);
        output_gpu.reshape(1, 1);

        struct Constants
        {
            uint32_t total_elements;
            float epsilon_val;
        } c{ total, epsilon };

        bool is_fp16 = (prediction.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::DICE_LOSS_FP16 : Compute_Pipeline::DICE_LOSS;

        auto buf_pred = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_target = is_fp16 ? effective_target->getEffectiveFp16Storage() : effective_target->getStorageBuffer();

        pred_gpu.pushToGraph(pipe, { buf_pred, buf_target, output_gpu.getStorageBuffer() }, c, wg_x, 1, 1);
        output_gpu.clearHostCache();
        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);
    }
}

void diceGradient(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon)
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
        const auto& a_data = prediction.getData();
        const auto& t_data = target.getData();
        auto& output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
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

        float denom = total_pred + total_target + epsilon;
        float numer = 2.0f * total_inter + epsilon;
        float inv_denom_sq = 1.0f / (denom * denom);

        auto& grad_out = *output_cpu.getStorageBuffer();
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
                float dice_grad = (numer - 2.0f * y * denom) * inv_denom_sq;
                float p_safe = std::clamp(a, 1e-4f, 1.0f - 1e-4f);
                float bce_grad = (a - y) / (p_safe * (1.0f - p_safe) * static_cast<float>(total_elements));
                grad_out[i] = 0.5f * dice_grad + 0.5f * bce_grad;
            }
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
        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);

        const auto& a_data = prediction.getData();
        const auto& t_data = target.getData();
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

        auto& pred_gpu = Gpu_Tensor_Impl::castToGpu(*pred_impl);
        auto contig_self = pred_gpu.ensureContiguousSelf();
        const auto* effective_self = contig_self ? contig_self.get() : &pred_gpu;

        const auto& target_gpu = Gpu_Tensor_Impl::castToGpu(*target_impl);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto* effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto& output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
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
        uint32_t wg_x = (total + 255) / 256;

        struct Constants
        {
            uint32_t total_elements;
            float epsilon_val;
            float intersection;
            float pred_sum;
            float target_sum;
        } c{ total, epsilon, total_inter, total_pred, total_target };

        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::DICE_GRADIENT_FP16 : Compute_Pipeline::DICE_GRADIENT;

        auto buf_pred = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_target = is_fp16 ? effective_target->getEffectiveFp16Storage() : effective_target->getStorageBuffer();

        pred_gpu.pushToGraph(pipe, { buf_pred, buf_target, output_gpu.getStorageBuffer() }, c, wg_x, 1, 1);
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

} // namespace ops::loss

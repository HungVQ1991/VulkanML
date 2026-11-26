#include "ops/llm/rope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::llm
{

using ops::internal::ensureUniqueTensorTarget;
using ops::internal::resolveExecutionEngine;
using ops::internal::validatePushConstants;

std::pair<std::vector<float>, float> computeRoPEInvFreq(uint32_t rotary_dim, float base, float freq_scale, const YaRN_Config &yarn)
{
    const uint32_t half_dim = rotary_dim / 2u;
    std::vector<float> inv_freq(half_dim, 0.0f);
    const float eff_scale = (freq_scale > 0.0f) ? freq_scale : 1.0f;

    if (!yarn.enabled || yarn.factor <= 1.0f)
    {
        for (uint32_t j = 0; j < half_dim; ++j)
        {
            inv_freq[j] = eff_scale * std::pow(base, -2.0f * static_cast<float>(j) / static_cast<float>(rotary_dim));
        }
        const float att = (yarn.enabled && yarn.attention_factor > 0.0f) ? yarn.attention_factor : 1.0f;
        return { std::move(inv_freq), att };
    }

    const float factor = yarn.factor;
    const float orig = static_cast<float>(std::max<uint32_t>(1u, yarn.original_max_position_embeddings));
    const float att = (yarn.attention_factor > 0.0f)
        ? yarn.attention_factor
        : (0.1f * std::log(factor) + 1.0f);
    const float bfast = (yarn.beta_fast > 0.0f) ? yarn.beta_fast : 32.0f;
    const float bslow = (yarn.beta_slow > 0.0f) ? yarn.beta_slow : 1.0f;

    auto corr = [&](float n_rot) -> float {
        return (static_cast<float>(rotary_dim) * std::log(orig / (n_rot * 2.0f * std::numbers::pi_v<float>))) /
               (2.0f * std::log(base));
    };

    const float low = std::max(std::floor(corr(bfast)), 0.0f);
    float high = std::min(std::ceil(corr(bslow)), static_cast<float>(rotary_dim - 1u));
    if (std::abs(high - low) < 1e-6f)
    {
        high += 0.001f;
    }

    for (uint32_t j = 0; j < half_dim; ++j)
    {
        const float base_inv = std::pow(base, -2.0f * static_cast<float>(j) / static_cast<float>(rotary_dim));
        const float inter = base_inv / factor;
        const float extra = base_inv;
        const float ramp = std::clamp((static_cast<float>(j) - low) / (high - low), 0.0f, 1.0f);
        const float extra_factor = 1.0f - ramp;
        inv_freq[j] = eff_scale * (inter * (1.0f - extra_factor) + extra * extra_factor);
    }
    return { std::move(inv_freq), att };
}

static void writeCpuOutput(Tensor &out, const Shape &shape, Data_Type dtype, const std::vector<float> &vals)
{
    ensureUniqueTensorTarget(out, Execution_Target::CPU, shape, dtype);
    auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*out.getImplementation());
    if (dtype == Data_Type::FLOAT16)
    {
        std::vector<float> clamped(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
        {
            clamped[i] = std::clamp(vals[i], -65504.0f, 65504.0f);
        }
        out_cpu.uploadData(clamped);
    }
    else
    {
        out_cpu.setStorageBufferFp16(nullptr);
        out_cpu.setDataType(Data_Type::FLOAT32);
        out_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(vals);
    }
}

void applyRoPE(const Tensor &input, Tensor &output, const RoPE_Config &config)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    const uint32_t seq_len = config.seq_len;
    const uint32_t head_dim = config.head_dim;
    const int direction = static_cast<int>(config.direction);
    const float base = config.base;
    const uint32_t num_heads = config.num_heads;
    const uint32_t mode = static_cast<uint32_t>(config.mode);
    const uint32_t position_offset = config.position_offset;
    const uint32_t rotary_dim = config.rotary_dim;
    const float freq_scale = config.freq_scale;

    if (input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: TERNARY_INT2 is not supported");
    }
    if (head_dim == 0 || (head_dim % 2) != 0)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: head_dim must be positive and even");
    }
    if (rotary_dim > head_dim || (rotary_dim % 2) != 0)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: rotary_dim must be even and <= head_dim");
    }
    if (!std::isfinite(base) || base <= 0.0f)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: base must be finite and > 0");
    }
    if (!std::isfinite(freq_scale) || freq_scale <= 0.0f)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: freq_scale must be finite and > 0");
    }
    if (seq_len == 0)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: seq_len must be > 0");
    }
    if (mode > 3)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: mode must be 0, 1, 2, or 3");
    }
    if ((mode == 1 || mode == 2 || mode == 3) && num_heads == 0)
    {
        throw std::invalid_argument("ops::llm::applyRoPE: num_heads must be > 0 for mode 1, 2, and 3");
    }

    const uint32_t eff_rotary_dim = (rotary_dim > 0) ? rotary_dim : head_dim;
    const bool use_indexed = (config.position_ids != nullptr || config.yarn.enabled);

    const Shape &in_shape = input.getShape();
    size_t rank = in_shape.getRank();
    Shape out_shape;
    if (mode == 1)
    {
        if (rank != 4 || in_shape[1] != seq_len || in_shape[2] != num_heads || in_shape[3] != head_dim)
        {
            throw std::invalid_argument("ops::llm::applyRoPE: mode 1 expects [B, seq_len, num_heads, head_dim]");
        }
        out_shape = Shape{in_shape[0], num_heads, seq_len, head_dim};
    }
    else if (mode == 2)
    {
        if (rank != 4 || in_shape[1] != num_heads || in_shape[2] != seq_len || in_shape[3] != head_dim)
        {
            throw std::invalid_argument("ops::llm::applyRoPE: mode 2 expects [B, num_heads, seq_len, head_dim]");
        }
        out_shape = Shape{in_shape[0], seq_len, num_heads, head_dim};
    }
    else if (mode == 3)
    {
        if (rank != 4 || in_shape[1] != seq_len || in_shape[2] != num_heads || in_shape[3] != head_dim)
        {
            throw std::invalid_argument("ops::llm::applyRoPE: mode 3 expects [B, seq_len, num_heads, head_dim]");
        }
        out_shape = in_shape;
    }
    else
    {
        if (rank < 2 || in_shape[rank - 1] != head_dim || in_shape[rank - 2] != seq_len)
        {
            throw std::invalid_argument("ops::llm::applyRoPE: mode 0 expects rank >= 2 with trailing dimensions [seq_len, head_dim]");
        }
        out_shape = in_shape;
    }

    const size_t eff_batch = (rank >= 3) ? in_shape[0] : 1u;
    const uint32_t eff_mode0_heads = (mode == 0)
        ? ((rank >= 3 && (eff_batch * seq_len * head_dim) > 0)
               ? static_cast<uint32_t>(input.getTotalElements() / (eff_batch * seq_len * head_dim))
               : 1u)
        : num_heads;

    uint32_t pos_stride_batch = 0u;
    if (config.position_ids != nullptr)
    {
        if (!config.position_ids->getImplementation() ||
            config.position_ids->getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::llm::applyRoPE: Invalid position_ids tensor");
        }
        const Shape &ps = config.position_ids->getShape();
        if (ps.getRank() == 1)
        {
            if (ps[0] != seq_len)
            {
                throw std::invalid_argument("ops::llm::applyRoPE: 1D position_ids length must match seq_len");
            }
            pos_stride_batch = 0u;
        }
        else if (ps.getRank() == 2)
        {
            if (ps[0] != eff_batch || ps[1] != seq_len)
            {
                throw std::invalid_argument("ops::llm::applyRoPE: 2D position_ids must match [B, seq_len]");
            }
            pos_stride_batch = seq_len;
        }
        else
        {
            throw std::invalid_argument("ops::llm::applyRoPE: position_ids must have rank 1 or 2");
        }
    }

    Execution_Target target = input.getExecutionTarget();
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;
    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (input.getDataType() == Data_Type::FLOAT16 || resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type out_dtype = (target != Execution_Target::CPU)
        ? (is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
        : input.getDataType();

    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        ensureUniqueTensorTarget(output, target, out_shape, out_dtype);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        std::vector<float> fp32_out(total_elements, 0.0f);

        size_t total_pairs = total_elements / 2;
        size_t D = head_dim;
        size_t R = eff_rotary_dim;
        size_t S = seq_len;
        size_t H = eff_mode0_heads;

        std::vector<float> inv_freq;
        float attn_scale = 1.0f;
        std::vector<float> pos_data;
        if (use_indexed)
        {
            auto [freqs, scale_val] = computeRoPEInvFreq(eff_rotary_dim, base, freq_scale, config.yarn);
            inv_freq = std::move(freqs);
            attn_scale = scale_val;
            if (config.position_ids != nullptr)
            {
                pos_data = config.position_ids->getData();
            }
        }

        for (size_t pair_id = 0; pair_id < total_pairs; ++pair_id)
        {
            size_t base_idx = pair_id * 2;
            size_t c = base_idx % D;
            size_t pair_idx = c / 2;
            size_t token_pos = 0;
            size_t batch_idx = 0;
            size_t in_idx = base_idx;
            size_t out_idx = base_idx;

            if (mode == 1)
            {
                token_pos = (base_idx / D) % S;
                size_t h = (base_idx / (D * S)) % H;
                batch_idx = base_idx / (D * S * H);
                in_idx = batch_idx * (S * H * D) + token_pos * (H * D) + h * D + c;
                out_idx = base_idx;
            }
            else if (mode == 2)
            {
                token_pos = (base_idx / D) % S;
                size_t h = (base_idx / (D * S)) % H;
                batch_idx = base_idx / (D * S * H);
                in_idx = base_idx;
                out_idx = batch_idx * (S * H * D) + token_pos * (H * D) + h * D + c;
            }
            else if (mode == 3)
            {
                token_pos = (base_idx / (D * H)) % S;
                batch_idx = base_idx / (D * H * S);
                in_idx = base_idx;
                out_idx = base_idx;
            }
            else
            {
                token_pos = (base_idx / D) % S;
                size_t outer = base_idx / (D * S);
                batch_idx = (H > 1) ? (outer / H) : outer;
                in_idx = base_idx;
                out_idx = base_idx;
            }

            float x0 = in_data[in_idx];
            float x1 = in_data[in_idx + 1];

            if (c >= R)
            {
                fp32_out[out_idx] = x0;
                fp32_out[out_idx + 1] = x1;
                continue;
            }

            float cos_a = 1.0f;
            float sin_a = 0.0f;
            if (use_indexed)
            {
                const float p = (config.position_ids != nullptr)
                    ? pos_data[batch_idx * pos_stride_batch + token_pos]
                    : static_cast<float>(token_pos + position_offset);
                const float alpha = static_cast<float>(direction) * p * inv_freq[pair_idx];
                cos_a = std::cos(alpha) * attn_scale;
                sin_a = std::sin(alpha) * attn_scale;
            }
            else
            {
                float theta = std::pow(base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(R));
                float alpha = static_cast<float>(direction) * (static_cast<float>(token_pos + position_offset) * freq_scale) * theta;
                cos_a = std::cos(alpha);
                sin_a = std::sin(alpha);
            }

            fp32_out[out_idx]     = x0 * cos_a - x1 * sin_a;
            fp32_out[out_idx + 1] = x0 * sin_a + x1 * cos_a;
        }

        writeCpuOutput(output, out_shape, out_dtype, fp32_out);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::llm::applyRoPE: total_elements exceeds uint32_t limit");
        }

        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
        std::shared_ptr<Gpu_Tensor_Impl> casted_self;
        if (!is_fp16 && effective_self->getDataType() == Data_Type::FLOAT16)
        {
            casted_self = std::make_shared<Gpu_Tensor_Impl>(effective_self->getShape(), Data_Type::FLOAT32);
            effective_self->to(Data_Type::FLOAT32, *casted_self);
            effective_self = casted_self.get();
        }
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        auto existing_out_impl = output.getImplementation();
        bool need_temp_out = (&input == &output) ||
            (existing_out_impl && existing_out_impl.get() == in_impl.get()) ||
            (existing_out_impl && output.getExecutionTarget() == Execution_Target::VULKAN_GPU &&
             Gpu_Tensor_Impl::castToGpu(*existing_out_impl).getStorageBuffer() == buf_self);

        Tensor staged_out;
        Tensor *dst_tensor = &output;
        if (need_temp_out && (mode == 1 || mode == 2))
        {
            staged_out = Tensor(out_shape, out_dtype, target);
            dst_tensor = &staged_out;
        }
        else
        {
            ensureUniqueTensorTarget(output, target, out_shape, out_dtype);
        }

        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*dst_tensor->getImplementation());
        size_t elem_size = getDataTypeSize(out_dtype);
        if (!out_gpu.getStorageBuffer() || out_gpu.getStorageBuffer()->getSizeBytes() < total_elements * elem_size)
        {
            out_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext());
            out_gpu.getStorageBuffer()->allocateMemory(total_elements, out_dtype);
        }

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t total_pairs = total / 2;
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total_pairs + 255) / 256);

        if (use_indexed)
        {
            auto [inv_freq_vec, attn_scale] = computeRoPEInvFreq(eff_rotary_dim, base, freq_scale, config.yarn);
            Tensor inv_freq_tensor(Shape{ inv_freq_vec.size() }, inv_freq_vec, Execution_Target::VULKAN_GPU);

            Tensor pos_dev(Execution_Target::VULKAN_GPU);
            if (config.position_ids != nullptr)
            {
                if (config.position_ids->getExecutionTarget() == Execution_Target::VULKAN_GPU &&
                    config.position_ids->getDataType() == Data_Type::FLOAT32)
                {
                    pos_dev = config.position_ids->contiguous();
                }
                else
                {
                    pos_dev = Tensor(config.position_ids->getShape(), config.position_ids->getData(), Execution_Target::VULKAN_GPU);
                }
            }
            else
            {
                pos_dev = Tensor(Shape{ 1 }, std::vector<float>{ 0.0f }, Execution_Target::VULKAN_GPU);
            }
            engine.getContext().executePendingTransfers();

            auto &pos_gpu = Gpu_Tensor_Impl::castToGpu(*pos_dev.getImplementation());
            auto &freq_gpu = Gpu_Tensor_Impl::castToGpu(*inv_freq_tensor.getImplementation());

            struct RoPE_Indexed_Constants
            {
                uint32_t total_pairs;
                uint32_t seq_len;
                uint32_t head_dim;
                int32_t direction;
                uint32_t num_heads;
                uint32_t mode;
                uint32_t rotary_dim;
                uint32_t pos_stride_batch;
                uint32_t has_pos_ids;
                uint32_t position_offset;
                float attn_scale;
            } pcs{
                total_pairs,
                seq_len,
                head_dim,
                direction,
                eff_mode0_heads,
                mode,
                eff_rotary_dim,
                pos_stride_batch,
                (config.position_ids != nullptr) ? 1u : 0u,
                position_offset,
                attn_scale
            };
            validatePushConstants<RoPE_Indexed_Constants>();

            effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::ROPE_INDEXED_FP16 : Compute_Pipeline::ROPE_INDEXED,
                                        {buf_self, out_gpu.getStorageBuffer(), pos_gpu.getStorageBuffer(), freq_gpu.getStorageBuffer()},
                                        pcs, wg_x, wg_y, 1, &engine);
        }
        else
        {
            struct RoPE_Constants
            {
                uint32_t total_pairs;
                uint32_t seq_len;
                uint32_t head_dim;
                int32_t direction;
                float base;
                uint32_t num_heads;
                uint32_t mode;
                uint32_t position_offset;
                uint32_t rotary_dim;
                float freq_scale;
            } pcs{total_pairs, seq_len, head_dim, direction, base, num_heads, mode, position_offset, eff_rotary_dim, freq_scale};
            validatePushConstants<RoPE_Constants>();

            effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::ROPE_FP16 : Compute_Pipeline::ROPE,
                                        {buf_self, out_gpu.getStorageBuffer()},
                                        pcs, wg_x, wg_y, 1, &engine);
        }
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();

        if (dst_tensor == &staged_out)
        {
            output = std::move(staged_out);
        }
    }
}

void applyRoPE(const Tensor &input, Tensor &output, uint32_t seq_len, uint32_t head_dim,
               int direction, float base, uint32_t num_heads, uint32_t mode, uint32_t position_offset)
{
    RoPE_Config config{
        seq_len,
        head_dim,
        static_cast<RoPE_Direction>(direction),
        base,
        num_heads,
        static_cast<RoPE_Mode>(mode),
        position_offset,
        0u,
        1.0f,
        nullptr,
        nullptr,
        YaRN_Config{}
    };
    applyRoPE(input, output, config);
}

void qkvSplitRopeForward(const Tensor &qkv, Tensor &q_out, Tensor &k_out, Tensor &v_out,
                         uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t num_kv_heads,
                         uint32_t head_dim, float rope_base, bool output_bhsd,
                         uint32_t position_offset, Execution_Engine *engine_ptr)
{
    auto qkv_impl = qkv.getImplementation();
    if (!qkv_impl) return;

    if (qkv.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::qkvSplitRopeForward: TERNARY_INT2 is not supported");
    }
    if (head_dim == 0 || (head_dim % 2) != 0 || batch_size == 0 || seq_len == 0 || num_heads == 0 || num_kv_heads == 0)
    {
        throw std::invalid_argument("ops::llm::qkvSplitRopeForward: Invalid dimensions");
    }
    if (!std::isfinite(rope_base) || rope_base <= 0.0f)
    {
        throw std::invalid_argument("ops::llm::qkvSplitRopeForward: rope_base must be finite and > 0");
    }

    const size_t total_elems = qkv.getTotalElements();
    const size_t expected_elems = static_cast<size_t>(batch_size) * seq_len * (num_heads + 2u * num_kv_heads) * head_dim;
    if (total_elems != expected_elems)
    {
        throw std::invalid_argument("ops::llm::qkvSplitRopeForward: QKV element count mismatch");
    }
    if (total_elems > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::llm::qkvSplitRopeForward: Element count exceeds uint32_t limit");
    }

    const Shape q_shape = output_bhsd
        ? Shape{batch_size, num_heads, seq_len, head_dim}
        : Shape{batch_size, seq_len, num_heads, head_dim};
    const Shape kv_shape = output_bhsd
        ? Shape{batch_size, num_kv_heads, seq_len, head_dim}
        : Shape{batch_size, seq_len, num_kv_heads, head_dim};

    Execution_Target target = qkv.getExecutionTarget();
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(engine_ptr) : nullptr;
    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (qkv.getDataType() == Data_Type::FLOAT16 || resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type out_dtype = (target != Execution_Target::CPU)
        ? (is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
        : qkv.getDataType();

    if (target == Execution_Target::CPU)
    {
        const auto in_data = qkv.getData();
        const size_t d_half = head_dim / 2;
        const size_t q_pairs = static_cast<size_t>(num_heads) * d_half;
        const size_t kv_pairs = static_cast<size_t>(num_kv_heads) * d_half;
        const size_t tok_pairs = q_pairs + 2 * kv_pairs;
        const size_t total_pairs = total_elems / 2;

        std::vector<float> q_vals(static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim, 0.0f);
        std::vector<float> k_vals(static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim, 0.0f);
        std::vector<float> v_vals(static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim, 0.0f);

        for (size_t pair_id = 0; pair_id < total_pairs; ++pair_id)
        {
            const size_t row = pair_id / tok_pairs;
            const size_t lp = pair_id % tok_pairs;
            const size_t token_pos = row % seq_len;
            const size_t b = row / seq_len;
            const float x0 = in_data[pair_id * 2];
            const float x1 = in_data[pair_id * 2 + 1];

            if (lp < q_pairs)
            {
                const size_t h = lp / d_half;
                const size_t pair_idx = lp % d_half;
                const float theta = std::pow(rope_base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim));
                const float alpha = static_cast<float>(token_pos + position_offset) * theta;
                const float cos_a = std::cos(alpha);
                const float sin_a = std::sin(alpha);
                const size_t out_base = 2 * (output_bhsd
                    ? (((b * num_heads + h) * seq_len + token_pos) * d_half + pair_idx)
                    : (row * q_pairs + lp));
                q_vals[out_base]     = x0 * cos_a - x1 * sin_a;
                q_vals[out_base + 1] = x0 * sin_a + x1 * cos_a;
            }
            else if (lp < q_pairs + kv_pairs)
            {
                const size_t kp = lp - q_pairs;
                const size_t h = kp / d_half;
                const size_t pair_idx = kp % d_half;
                const float theta = std::pow(rope_base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim));
                const float alpha = static_cast<float>(token_pos + position_offset) * theta;
                const float cos_a = std::cos(alpha);
                const float sin_a = std::sin(alpha);
                const size_t out_base = 2 * (output_bhsd
                    ? (((b * num_kv_heads + h) * seq_len + token_pos) * d_half + pair_idx)
                    : (row * kv_pairs + kp));
                k_vals[out_base]     = x0 * cos_a - x1 * sin_a;
                k_vals[out_base + 1] = x0 * sin_a + x1 * cos_a;
            }
            else
            {
                const size_t vp = lp - q_pairs - kv_pairs;
                const size_t h = vp / d_half;
                const size_t pair_idx = vp % d_half;
                const size_t out_base = 2 * (output_bhsd
                    ? (((b * num_kv_heads + h) * seq_len + token_pos) * d_half + pair_idx)
                    : (row * kv_pairs + vp));
                v_vals[out_base]     = x0;
                v_vals[out_base + 1] = x1;
            }
        }
        writeCpuOutput(q_out, q_shape, out_dtype, q_vals);
        writeCpuOutput(k_out, kv_shape, out_dtype, k_vals);
        writeCpuOutput(v_out, kv_shape, out_dtype, v_vals);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*qkv_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;
        std::shared_ptr<Gpu_Tensor_Impl> casted_self;
        if (!is_fp16 && effective_self->getDataType() == Data_Type::FLOAT16)
        {
            casted_self = std::make_shared<Gpu_Tensor_Impl>(effective_self->getShape(), Data_Type::FLOAT32);
            effective_self->to(Data_Type::FLOAT32, *casted_self);
            effective_self = casted_self.get();
        }
        auto buf_qkv = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        ensureUniqueTensorTarget(q_out, target, q_shape, out_dtype);
        ensureUniqueTensorTarget(k_out, target, kv_shape, out_dtype);
        ensureUniqueTensorTarget(v_out, target, kv_shape, out_dtype);

        auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q_out.getImplementation());
        auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k_out.getImplementation());
        auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v_out.getImplementation());

        uint32_t total_pairs = static_cast<uint32_t>(total_elems / 2);
        struct Qkv_Rope_Constants
        {
            uint32_t total_pairs;
            uint32_t seq_len;
            uint32_t head_dim;
            uint32_t num_heads;
            uint32_t num_kv_heads;
            uint32_t mode;
            uint32_t position_offset;
            float base;
        } pcs{total_pairs, seq_len, head_dim, num_heads, num_kv_heads, output_bhsd ? 1u : 0u, position_offset, rope_base};
        validatePushConstants<Qkv_Rope_Constants>();

        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((static_cast<uint64_t>(total_pairs) + 255ULL) / 256ULL);
        effective_self->pushToGraph(is_fp16 ? Compute_Pipeline::QKV_SPLIT_ROPE_FP16 : Compute_Pipeline::QKV_SPLIT_ROPE,
                                    {buf_qkv, q_gpu.getStorageBuffer(), k_gpu.getStorageBuffer(), v_gpu.getStorageBuffer()},
                                    pcs, wg_x, wg_y, 1, &engine);
        q_gpu.clearHostCache();
        q_gpu.invalidateFp16Cache();
        k_gpu.clearHostCache();
        k_gpu.invalidateFp16Cache();
        v_gpu.clearHostCache();
        v_gpu.invalidateFp16Cache();
    }
}

void qkvPackUnropeBackward(const Tensor &dq, const Tensor &dk, const Tensor &dv, Tensor &dqkv_out,
                           uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t num_kv_heads,
                           uint32_t head_dim, float rope_base, bool input_bhsd,
                           uint32_t position_offset, Execution_Engine *engine_ptr)
{
    auto dq_impl = dq.getImplementation();
    auto dk_impl = dk.getImplementation();
    auto dv_impl = dv.getImplementation();
    if (!dq_impl || !dk_impl || !dv_impl) return;

    Execution_Target target = dq.getExecutionTarget();
    if (dk.getExecutionTarget() != target || dv.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::llm::qkvPackUnropeBackward: Execution_Target mismatch");
    }
    if (dq.getDataType() == Data_Type::TERNARY_INT2 ||
        dk.getDataType() == Data_Type::TERNARY_INT2 ||
        dv.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::qkvPackUnropeBackward: TERNARY_INT2 is not supported");
    }
    if (head_dim == 0 || (head_dim % 2) != 0 || batch_size == 0 || seq_len == 0 || num_heads == 0 || num_kv_heads == 0)
    {
        throw std::invalid_argument("ops::llm::qkvPackUnropeBackward: Invalid dimensions");
    }
    if (!std::isfinite(rope_base) || rope_base <= 0.0f)
    {
        throw std::invalid_argument("ops::llm::qkvPackUnropeBackward: rope_base must be finite and > 0");
    }

    const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
    const size_t kv_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
    if (dq.getTotalElements() != q_elems || dk.getTotalElements() != kv_elems || dv.getTotalElements() != kv_elems)
    {
        throw std::invalid_argument("ops::llm::qkvPackUnropeBackward: Input tensor element count mismatch");
    }
    const size_t total_elems = q_elems + 2 * kv_elems;
    if (total_elems > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::llm::qkvPackUnropeBackward: Element count exceeds uint32_t limit");
    }

    const Shape out_shape{static_cast<size_t>(batch_size) * seq_len, static_cast<size_t>(num_heads + 2u * num_kv_heads) * head_dim};
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(engine_ptr) : nullptr;
    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (dq.getDataType() == Data_Type::FLOAT16 ||
                   dk.getDataType() == Data_Type::FLOAT16 ||
                   dv.getDataType() == Data_Type::FLOAT16 ||
                   resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type out_dtype = (target != Execution_Target::CPU)
        ? (is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
        : dq.getDataType();

    if (target == Execution_Target::CPU)
    {
        const auto dq_data = dq.getData();
        const auto dk_data = dk.getData();
        const auto dv_data = dv.getData();
        const size_t d_half = head_dim / 2;
        const size_t q_pairs = static_cast<size_t>(num_heads) * d_half;
        const size_t kv_pairs = static_cast<size_t>(num_kv_heads) * d_half;
        const size_t tok_pairs = q_pairs + 2 * kv_pairs;
        const size_t total_pairs = total_elems / 2;

        std::vector<float> dqkv_vals(total_elems, 0.0f);
        for (size_t pair_id = 0; pair_id < total_pairs; ++pair_id)
        {
            const size_t row = pair_id / tok_pairs;
            const size_t lp = pair_id % tok_pairs;
            const size_t token_pos = row % seq_len;
            const size_t b = row / seq_len;

            if (lp < q_pairs)
            {
                const size_t h = lp / d_half;
                const size_t pair_idx = lp % d_half;
                const size_t in_base = 2 * (input_bhsd
                    ? (((b * num_heads + h) * seq_len + token_pos) * d_half + pair_idx)
                    : (row * q_pairs + lp));
                const float x0 = dq_data[in_base];
                const float x1 = dq_data[in_base + 1];
                const float theta = std::pow(rope_base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim));
                const float alpha = -static_cast<float>(token_pos + position_offset) * theta;
                const float cos_a = std::cos(alpha);
                const float sin_a = std::sin(alpha);
                dqkv_vals[pair_id * 2]     = x0 * cos_a - x1 * sin_a;
                dqkv_vals[pair_id * 2 + 1] = x0 * sin_a + x1 * cos_a;
            }
            else if (lp < q_pairs + kv_pairs)
            {
                const size_t kp = lp - q_pairs;
                const size_t h = kp / d_half;
                const size_t pair_idx = kp % d_half;
                const size_t in_base = 2 * (input_bhsd
                    ? (((b * num_kv_heads + h) * seq_len + token_pos) * d_half + pair_idx)
                    : (row * kv_pairs + kp));
                const float x0 = dk_data[in_base];
                const float x1 = dk_data[in_base + 1];
                const float theta = std::pow(rope_base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(head_dim));
                const float alpha = -static_cast<float>(token_pos + position_offset) * theta;
                const float cos_a = std::cos(alpha);
                const float sin_a = std::sin(alpha);
                dqkv_vals[pair_id * 2]     = x0 * cos_a - x1 * sin_a;
                dqkv_vals[pair_id * 2 + 1] = x0 * sin_a + x1 * cos_a;
            }
            else
            {
                const size_t vp = lp - q_pairs - kv_pairs;
                const size_t h = vp / d_half;
                const size_t pair_idx = vp % d_half;
                const size_t in_base = 2 * (input_bhsd
                    ? (((b * num_kv_heads + h) * seq_len + token_pos) * d_half + pair_idx)
                    : (row * kv_pairs + vp));
                dqkv_vals[pair_id * 2]     = dv_data[in_base];
                dqkv_vals[pair_id * 2 + 1] = dv_data[in_base + 1];
            }
        }
        writeCpuOutput(dqkv_out, out_shape, out_dtype, dqkv_vals);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        const auto &dq_gpu = Gpu_Tensor_Impl::castToGpu(*dq_impl);
        auto contig_dq = dq_gpu.ensureContiguousSelf();
        const auto *eff_dq = contig_dq ? contig_dq.get() : &dq_gpu;

        const auto &dk_gpu = Gpu_Tensor_Impl::castToGpu(*dk_impl);
        auto contig_dk = dk_gpu.ensureContiguousSelf();
        const auto *eff_dk = contig_dk ? contig_dk.get() : &dk_gpu;

        const auto &dv_gpu = Gpu_Tensor_Impl::castToGpu(*dv_impl);
        auto contig_dv = dv_gpu.ensureContiguousSelf();
        const auto *eff_dv = contig_dv ? contig_dv.get() : &dv_gpu;

        std::shared_ptr<Gpu_Tensor_Impl> c_dq, c_dk, c_dv;
        if (!is_fp16)
        {
            if (eff_dq->getDataType() == Data_Type::FLOAT16)
            {
                c_dq = std::make_shared<Gpu_Tensor_Impl>(eff_dq->getShape(), Data_Type::FLOAT32);
                eff_dq->to(Data_Type::FLOAT32, *c_dq);
                eff_dq = c_dq.get();
            }
            if (eff_dk->getDataType() == Data_Type::FLOAT16)
            {
                c_dk = std::make_shared<Gpu_Tensor_Impl>(eff_dk->getShape(), Data_Type::FLOAT32);
                eff_dk->to(Data_Type::FLOAT32, *c_dk);
                eff_dk = c_dk.get();
            }
            if (eff_dv->getDataType() == Data_Type::FLOAT16)
            {
                c_dv = std::make_shared<Gpu_Tensor_Impl>(eff_dv->getShape(), Data_Type::FLOAT32);
                eff_dv->to(Data_Type::FLOAT32, *c_dv);
                eff_dv = c_dv.get();
            }
        }

        auto buf_dq = is_fp16 ? eff_dq->getEffectiveFp16Storage() : eff_dq->getStorageBuffer();
        auto buf_dk = is_fp16 ? eff_dk->getEffectiveFp16Storage() : eff_dk->getStorageBuffer();
        auto buf_dv = is_fp16 ? eff_dv->getEffectiveFp16Storage() : eff_dv->getStorageBuffer();

        ensureUniqueTensorTarget(dqkv_out, target, out_shape, out_dtype);
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*dqkv_out.getImplementation());

        uint32_t total_pairs = static_cast<uint32_t>(total_elems / 2);
        struct Qkv_Rope_Constants
        {
            uint32_t total_pairs;
            uint32_t seq_len;
            uint32_t head_dim;
            uint32_t num_heads;
            uint32_t num_kv_heads;
            uint32_t mode;
            uint32_t position_offset;
            float base;
        } pcs{total_pairs, seq_len, head_dim, num_heads, num_kv_heads, input_bhsd ? 1u : 0u, position_offset, rope_base};
        validatePushConstants<Qkv_Rope_Constants>();

        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((static_cast<uint64_t>(total_pairs) + 255ULL) / 256ULL);
        eff_dq->pushToGraph(is_fp16 ? Compute_Pipeline::QKV_PACK_UNROPE_FP16 : Compute_Pipeline::QKV_PACK_UNROPE,
                            {buf_dq, buf_dk, buf_dv, out_gpu.getStorageBuffer()},
                            pcs, wg_x, wg_y, 1, &engine);
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
    }
}

}

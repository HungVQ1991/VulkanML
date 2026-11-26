#include "ops/llm/attention.h"
#include "ops/llm/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::llm
{

using ops::internal::ensureDisjointGpuStorage;
using ops::internal::ensureDisjointTensorTarget;
using ops::internal::ensureUniqueTensorTarget;
using ops::internal::pushGpuFill;
using ops::internal::resolveExecutionEngine;
using ops::internal::validatePushConstants;

static void validateTargetAndType(const Tensor &t, Execution_Target expected, const char *op, const char *name)
{
    if (!t.getImplementation())
    {
        throw std::invalid_argument(std::string(op) + ": " + name + " has null implementation");
    }
    if (t.getExecutionTarget() != expected)
    {
        throw std::invalid_argument(std::string(op) + ": Execution_Target mismatch for " + name);
    }
    if (t.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op) + ": TERNARY_INT2 is not supported for " + name);
    }
}

static size_t validateAndComputeGqaGroupSize(size_t num_heads, size_t num_kv_heads, const char *op)
{
    if (num_heads == 0)
    {
        throw std::invalid_argument(std::string(op) + ": num_heads must be > 0");
    }
    size_t eff_kv = (num_kv_heads == 0) ? num_heads : num_kv_heads;
    if (eff_kv == 0 || eff_kv > num_heads || (num_heads % eff_kv) != 0)
    {
        throw std::invalid_argument(std::string(op) + ": num_kv_heads must divide num_heads and be in [1, num_heads]");
    }
    return num_heads / eff_kv;
}

struct Resolved_Mask_Layout
{
    Attention_Mask_Mode mode = Attention_Mask_Mode::NONE;
    uint32_t stride_batch = 0;
    uint32_t per_head = 0;
};

static Resolved_Mask_Layout resolveMaskLayout(const Attention_Config &config, size_t batch_size, size_t total_heads,
                                              size_t q_len, size_t kv_len, const char *op)
{
    Resolved_Mask_Layout layout{};
    if (config.mask_mode != Attention_Mask_Mode::NONE)
    {
        layout.mode = config.mask_mode;
    }
    else
    {
        layout.mode = config.is_causal ? Attention_Mask_Mode::CAUSAL : Attention_Mask_Mode::NONE;
    }

    if (layout.mode == Attention_Mask_Mode::DENSE_BOOL)
    {
        if (!config.attn_mask || !config.attn_mask->getImplementation())
        {
            throw std::invalid_argument(std::string(op) + ": attn_mask is required for DENSE_BOOL mask_mode");
        }
        if (config.attn_mask->getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument(std::string(op) + ": TERNARY_INT2 attn_mask is not supported");
        }
        const Shape &ms = config.attn_mask->getShape();
        if (ms.getRank() == 2)
        {
            if (ms[0] != q_len || ms[1] != kv_len)
            {
                throw std::invalid_argument(std::string(op) + ": 2D attn_mask shape must equal [q_seq_len, kv_seq_len]");
            }
            layout.stride_batch = 0u;
            layout.per_head = 0u;
        }
        else if (ms.getRank() == 3)
        {
            if (ms[1] != q_len || ms[2] != kv_len)
            {
                throw std::invalid_argument(std::string(op) + ": 3D attn_mask trailing dims must equal [q_seq_len, kv_seq_len]");
            }
            if (ms[0] == batch_size)
            {
                layout.per_head = 0u;
            }
            else if (ms[0] == total_heads)
            {
                layout.per_head = 1u;
            }
            else
            {
                throw std::invalid_argument(std::string(op) + ": 3D attn_mask leading dim must equal batch_size or total_heads");
            }
            layout.stride_batch = static_cast<uint32_t>(q_len * kv_len);
        }
        else
        {
            throw std::invalid_argument(std::string(op) + ": attn_mask must have rank 2 or 3");
        }
    }
    return layout;
}

static Tensor gatherKvSequenceSubset(const Tensor &kv, const Tensor &kv_index,
                                     size_t B, size_t H_kv, size_t orig_T, size_t D, bool is_bshd)
{
    const size_t sel_T = kv_index.getTotalElements();
    const Shape out_shape = is_bshd ? Shape{B, sel_T, H_kv, D} : Shape{B, H_kv, sel_T, D};
    const Execution_Target target = kv.getExecutionTarget();
    const Data_Type dtype = kv.getDataType();
    if (sel_T == 0)
    {
        return Tensor(out_shape, dtype, target);
    }

    if (target == Execution_Target::CPU)
    {
        const auto src = kv.getData();
        const auto idx = kv_index.getData();
        std::vector<float> dst(out_shape.getTotalElements(), 0.0f);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t kh = 0; kh < H_kv; ++kh)
            {
                for (size_t sj = 0; sj < sel_T; ++sj)
                {
                    const double r = std::round(static_cast<double>(idx[sj]));
                    if (r < 0.0 || r >= static_cast<double>(orig_T))
                    {
                        throw std::out_of_range("ops::llm::attention: kv_index entry out of range");
                    }
                    const size_t oj = static_cast<size_t>(r);
                    const size_t src_off = is_bshd
                        ? (((b * orig_T + oj) * H_kv + kh) * D)
                        : (((b * H_kv + kh) * orig_T + oj) * D);
                    const size_t dst_off = is_bshd
                        ? (((b * sel_T + sj) * H_kv + kh) * D)
                        : (((b * H_kv + kh) * sel_T + sj) * D);
                    std::copy_n(src.data() + src_off, D, dst.data() + dst_off);
                }
            }
        }
        return Tensor(out_shape, std::move(dst), dtype, Execution_Target::CPU);
    }

    Tensor idx_dev = (kv_index.getExecutionTarget() == target)
        ? kv_index
        : Tensor(kv_index.getShape(), kv_index.getData(), target);
    if (!is_bshd)
    {
        Tensor kv_3d = kv.view(Shape{B * H_kv, orig_T, D});
        Tensor out_3d(Shape{B * H_kv, sel_T, D}, dtype, target);
        for (size_t kh = 0; kh < B * H_kv; ++kh)
        {
            Tensor s = kv_3d.slice(0, kh, 1).contiguous().view(Shape{orig_T, D});
            Tensor g(target);
            ops::llm::embeddingForward(s, idx_dev, g, dtype == Data_Type::FLOAT16);
            if (g.getDataType() != out_3d.getDataType()) g = g.to(out_3d.getDataType());
            out_3d.updateSlice(0, kh, g.view(Shape{1, sel_T, D}));
        }
        return out_3d.view(out_shape);
    }
    const size_t flat_D = H_kv * D;
    Tensor out_3d(Shape{B, sel_T, flat_D}, dtype, target);
    for (size_t b = 0; b < B; ++b)
    {
        Tensor s = kv.slice(0, b, 1).contiguous().view(Shape{orig_T, flat_D});
        Tensor g(target);
        ops::llm::embeddingForward(s, idx_dev, g, dtype == Data_Type::FLOAT16);
        if (g.getDataType() != out_3d.getDataType()) g = g.to(out_3d.getDataType());
        out_3d.updateSlice(0, b, g.view(Shape{1, sel_T, flat_D}));
    }
    return out_3d.view(out_shape);
}

static Tensor scatterKvGradSubset(const Tensor &grad_sel, const Tensor &kv_index,
                                  size_t B, size_t H_kv, size_t orig_T, size_t sel_T, size_t D,
                                  bool is_bshd, Data_Type target_dtype)
{
    const Shape out_shape = is_bshd ? Shape{B, orig_T, H_kv, D} : Shape{B, H_kv, orig_T, D};
    const Execution_Target target = grad_sel.getExecutionTarget();
    if (target == Execution_Target::CPU)
    {
        const auto gsrc = grad_sel.getData();
        const auto idx = kv_index.getData();
        std::vector<float> dst(out_shape.getTotalElements(), 0.0f);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t kh = 0; kh < H_kv; ++kh)
            {
                for (size_t sj = 0; sj < sel_T; ++sj)
                {
                    const size_t oj = static_cast<size_t>(std::round(static_cast<double>(idx[sj])));
                    const size_t src_off = is_bshd
                        ? (((b * sel_T + sj) * H_kv + kh) * D)
                        : (((b * H_kv + kh) * sel_T + sj) * D);
                    const size_t dst_off = is_bshd
                        ? (((b * orig_T + oj) * H_kv + kh) * D)
                        : (((b * H_kv + kh) * orig_T + oj) * D);
                    for (size_t d = 0; d < D; ++d)
                    {
                        dst[dst_off + d] += gsrc[src_off + d];
                    }
                }
            }
        }
        return Tensor(out_shape, std::move(dst), target_dtype, Execution_Target::CPU);
    }

    Tensor idx_dev = (kv_index.getExecutionTarget() == target)
        ? kv_index
        : Tensor(kv_index.getShape(), kv_index.getData(), target);
    if (!is_bshd)
    {
        Tensor gs_3d = grad_sel.view(Shape{B * H_kv, sel_T, D});
        Tensor out_3d(Shape{B * H_kv, orig_T, D}, Data_Type::FLOAT32, target);
        for (size_t kh = 0; kh < B * H_kv; ++kh)
        {
            Tensor s = gs_3d.slice(0, kh, 1).contiguous().view(Shape{sel_T, D});
            Tensor dst_2d(Shape{orig_T, D}, Data_Type::FLOAT32, target);
            ops::llm::embeddingBackward(dst_2d, idx_dev, s, dst_2d, false);
            out_3d.updateSlice(0, kh, dst_2d.view(Shape{1, orig_T, D}));
        }
        Tensor res = out_3d.view(out_shape);
        return (res.getDataType() == target_dtype) ? res : res.to(target_dtype);
    }
    const size_t flat_D = H_kv * D;
    Tensor out_3d(Shape{B, orig_T, flat_D}, Data_Type::FLOAT32, target);
    for (size_t b = 0; b < B; ++b)
    {
        Tensor s = grad_sel.slice(0, b, 1).contiguous().view(Shape{sel_T, flat_D});
        Tensor dst_2d(Shape{orig_T, flat_D}, Data_Type::FLOAT32, target);
        ops::llm::embeddingBackward(dst_2d, idx_dev, s, dst_2d, false);
        out_3d.updateSlice(0, b, dst_2d.view(Shape{1, orig_T, flat_D}));
    }
    Tensor res = out_3d.view(out_shape);
    return (res.getDataType() == target_dtype) ? res : res.to(target_dtype);
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

void flashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                           const Attention_Config &config, Tensor *l_stats)
{
    auto q_impl = q.getImplementation();
    auto k_impl = k.getImplementation();
    auto v_impl = v.getImplementation();
    if (!q_impl || !k_impl || !v_impl) return;

    Execution_Target target = q.getExecutionTarget();
    validateTargetAndType(q, target, "ops::llm::flashAttentionForward", "q");
    validateTargetAndType(k, target, "ops::llm::flashAttentionForward", "k");
    validateTargetAndType(v, target, "ops::llm::flashAttentionForward", "v");
    if (l_stats && l_stats->getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::flashAttentionForward: l_stats cannot be TERNARY_INT2");
    }

    const uint32_t num_heads = config.num_heads;
    const uint32_t seq_len = config.seq_len;
    const uint32_t head_dim = config.head_dim;
    const bool is_causal = config.is_causal;
    float scale = config.scale;
    const uint32_t kv_seq_len = config.kv_seq_len;
    const uint32_t causal_offset = config.causal_offset;

    if (head_dim == 0 || num_heads == 0)
    {
        throw std::invalid_argument("ops::llm::flashAttentionForward: head_dim and num_heads must be > 0");
    }
    const size_t gqa_group_sz = validateAndComputeGqaGroupSize(num_heads, config.num_kv_heads, "ops::llm::flashAttentionForward");
    const uint32_t eff_kv_heads = static_cast<uint32_t>(num_heads / gqa_group_sz);
    const uint32_t gqa_group_size = static_cast<uint32_t>(gqa_group_sz);

    uint32_t eff_kv_len = (kv_seq_len > 0) ? kv_seq_len : seq_len;
    if (seq_len == 0 || q.getTotalElements() == 0)
    {
        const Shape q_shape = q.getShape();
        Data_Type empty_dtype = (target == Execution_Target::CPU)
            ? q.getDataType()
            : ((q.getDataType() == Data_Type::FLOAT16 && resolveExecutionEngine(config.engine).getContext().isFloat16Enabled())
                   ? Data_Type::FLOAT16
                   : Data_Type::FLOAT32);
        ensureDisjointTensorTarget(output, target, q_shape, empty_dtype, {&q, &k, &v});
        if (l_stats)
        {
            ensureDisjointTensorTarget(*l_stats, target, Shape{ num_heads, 0 }, Data_Type::FLOAT32, {&q, &k, &v, &output});
        }
        return;
    }

    size_t q_elements_per_head = static_cast<size_t>(seq_len) * head_dim;
    if (q.getTotalElements() % q_elements_per_head != 0)
    {
        throw std::invalid_argument("ops::llm::flashAttentionForward: q total elements must be divisible by seq_len * head_dim");
    }
    size_t total_heads_sz = q.getTotalElements() / q_elements_per_head;
    if (total_heads_sz % num_heads != 0)
    {
        throw std::invalid_argument("ops::llm::flashAttentionForward: inferred head count is not divisible by num_heads");
    }
    size_t batch_size_sz = total_heads_sz / num_heads;
    size_t total_kv_heads_sz = batch_size_sz * eff_kv_heads;
    if (kv_seq_len == 0 && config.kv_index != nullptr && total_kv_heads_sz * head_dim > 0 &&
        (k.getTotalElements() % (total_kv_heads_sz * head_dim)) == 0)
    {
        eff_kv_len = static_cast<uint32_t>(k.getTotalElements() / (total_kv_heads_sz * head_dim));
    }
    size_t kv_elements_per_head = static_cast<size_t>(eff_kv_len) * head_dim;
    if (k.getTotalElements() != total_kv_heads_sz * kv_elements_per_head ||
        v.getTotalElements() != total_kv_heads_sz * kv_elements_per_head)
    {
        throw std::invalid_argument("ops::llm::flashAttentionForward: k/v size mismatch (expected total_kv_heads * eff_kv_len * head_dim)");
    }

    if (scale <= 0.0f)
    {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    if (config.kv_index != nullptr)
    {
        if (!config.kv_index->getImplementation() || config.kv_index->getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::llm::flashAttentionForward: Invalid kv_index tensor");
        }
        const size_t sel_T = config.kv_index->getTotalElements();
        Tensor k_sel = gatherKvSequenceSubset(k, *config.kv_index, batch_size_sz, eff_kv_heads, eff_kv_len, head_dim, config.is_bshd);
        Tensor v_sel = gatherKvSequenceSubset(v, *config.kv_index, batch_size_sz, eff_kv_heads, eff_kv_len, head_dim, config.is_bshd);
        if (target == Execution_Target::VULKAN_GPU)
        {
            resolveExecutionEngine(config.engine).getContext().executePendingTransfers();
        }
        Attention_Config sub_cfg = config;
        sub_cfg.kv_seq_len = static_cast<uint32_t>(sel_T);
        sub_cfg.kv_index = nullptr;
        if (sel_T == 0)
        {
            Data_Type empty_dtype = (target == Execution_Target::CPU)
                ? q.getDataType()
                : ((q.getDataType() == Data_Type::FLOAT16 && resolveExecutionEngine(config.engine).getContext().isFloat16Enabled())
                       ? Data_Type::FLOAT16
                       : Data_Type::FLOAT32);
            ensureDisjointTensorTarget(output, target, q.getShape(), empty_dtype, {&q, &k, &v});
            if (target == Execution_Target::CPU) output.zero();
            else pushGpuFill(Gpu_Tensor_Impl::castToGpu(*output.getImplementation()), q.getTotalElements(), 0.0f, resolveExecutionEngine(config.engine));
            if (l_stats)
            {
                ensureDisjointTensorTarget(*l_stats, target, Shape{ total_heads_sz, seq_len }, Data_Type::FLOAT32, {&q, &k, &v, &output});
                l_stats->fill(-1e20f);
            }
            return;
        }
        flashAttentionForward(q, k_sel, v_sel, output, sub_cfg, l_stats);
        return;
    }

    const Resolved_Mask_Layout mask_layout = resolveMaskLayout(
        config, batch_size_sz, total_heads_sz, seq_len, eff_kv_len, "ops::llm::flashAttentionForward");

    if (target == Execution_Target::CPU)
    {
        const auto q_data = q.getData();
        const auto k_data = k.getData();
        const auto v_data = v.getData();
        std::vector<float> mask_data;
        if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL)
        {
            mask_data = config.attn_mask->getData();
        }

        std::vector<float> fp32_out(q.getTotalElements(), 0.0f);
        std::vector<float> l_vals;
        if (l_stats)
        {
            l_vals.assign(total_heads_sz * seq_len, -std::numeric_limits<float>::infinity());
        }

        for (size_t h = 0; h < total_heads_sz; ++h)
        {
            const size_t batch_idx = h / num_heads;
            size_t q_head_offset;
            size_t q_row_stride;
            size_t kv_head_offset;
            size_t kv_row_stride;
            if (config.is_bshd)
            {
                size_t head_idx = h % num_heads;
                size_t kv_head_idx = head_idx / gqa_group_sz;
                q_head_offset = (batch_idx * seq_len * num_heads + head_idx) * head_dim;
                q_row_stride = static_cast<size_t>(num_heads) * head_dim;
                kv_head_offset = (batch_idx * eff_kv_len * eff_kv_heads + kv_head_idx) * head_dim;
                kv_row_stride = static_cast<size_t>(eff_kv_heads) * head_dim;
            }
            else
            {
                q_head_offset = h * q_elements_per_head;
                q_row_stride = head_dim;
                kv_head_offset = (h / gqa_group_sz) * kv_elements_per_head;
                kv_row_stride = head_dim;
            }

            auto allowed_key = [&](size_t qi, size_t kj) -> bool {
                if (mask_layout.mode == Attention_Mask_Mode::CAUSAL)
                {
                    return kj <= static_cast<size_t>(causal_offset) + qi;
                }
                if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL)
                {
                    const size_t mb = mask_layout.per_head ? h : batch_idx;
                    return mask_data[mb * mask_layout.stride_batch + qi * eff_kv_len + kj] > 0.5f;
                }
                return true;
            };

            for (size_t i = 0; i < seq_len; ++i)
            {
                std::vector<float> scores(eff_kv_len, -std::numeric_limits<float>::infinity());
                float max_score = -std::numeric_limits<float>::infinity();

                size_t max_j = (mask_layout.mode == Attention_Mask_Mode::CAUSAL)
                    ? std::min<size_t>(eff_kv_len, causal_offset + i + 1)
                    : static_cast<size_t>(eff_kv_len);
                for (size_t j = 0; j < max_j; ++j)
                {
                    if (!allowed_key(i, j)) continue;
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dot += q_data[q_head_offset + i * q_row_stride + d] * k_data[kv_head_offset + j * kv_row_stride + d];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > max_score)
                    {
                        max_score = scores[j];
                    }
                }

                float sum_exp = 0.0f;
                if (std::isfinite(max_score))
                {
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        if (!allowed_key(i, j))
                        {
                            scores[j] = 0.0f;
                            continue;
                        }
                        scores[j] = std::exp(scores[j] - max_score);
                        sum_exp += scores[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        scores[j] = 0.0f;
                    }
                }
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

                if (l_stats)
                {
                    float logsumexp = (sum_exp > 0.0f) ? (max_score + std::log(sum_exp)) : -std::numeric_limits<float>::infinity();
                    l_vals[h * seq_len + i] = logsumexp;
                }

                if (inv_sum > 0.0f)
                {
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        float out_val = 0.0f;
                        for (size_t j = 0; j < max_j; ++j)
                        {
                            if (scores[j] == 0.0f) continue;
                            out_val += (scores[j] * inv_sum) * v_data[kv_head_offset + j * kv_row_stride + d];
                        }
                        fp32_out[q_head_offset + i * q_row_stride + d] = out_val;
                    }
                }
            }
        }

        writeCpuOutput(output, q.getShape(), q.getDataType(), fp32_out);
        if (l_stats)
        {
            writeCpuOutput(*l_stats, Shape{ total_heads_sz, seq_len }, Data_Type::FLOAT32, l_vals);
        }
    }
    else
    {
        Execution_Engine &engine = resolveExecutionEngine(config.engine);
        const auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q_impl);
        auto contig_q = q_gpu.ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : &q_gpu;

        const auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k_impl);
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;

        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v_impl);
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;

        if (total_heads_sz > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::llm::flashAttentionForward: total_heads exceeds uint32_t");
        }
        uint32_t total_heads = static_cast<uint32_t>(total_heads_sz);
        const uint64_t num_q_tiles_check = (static_cast<uint64_t>(seq_len) + 15ULL) / 16ULL;
        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (num_q_tiles_check > max_wg[1] || total_heads > max_wg[2])
        {
            throw std::overflow_error("ops::llm::flashAttentionForward: Workgroup grid exceeds hardware limits");
        }

        const Shape q_shape = q.getShape();
        const Data_Type q_dtype = q.getDataType();
        const Data_Type out_dtype = (q_dtype == Data_Type::FLOAT16 && engine.getContext().isFloat16Enabled())
            ? Data_Type::FLOAT16
            : Data_Type::FLOAT32;

        bool can_use_coop = engine.isCooperativeMatrixEnabled() &&
                            engine.getContext().isFloat16Enabled() &&
                            mask_layout.mode != Attention_Mask_Mode::DENSE_BOOL &&
                            head_dim > 0 && head_dim <= 64 && (head_dim % 16 == 0) &&
                            eff_kv_len == seq_len && causal_offset == 0 &&
                            engine.getNetwork().getPipeline(Compute_Pipeline::FLASH_ATTENTION_FP16) != VK_NULL_HANDLE;

        Logger::logMessage(Input_Format{"ops::llm::flashAttentionForward: path={}, seq_len={}, kv_len={}, heads={}, kv_heads={}, head_dim={}",
                                        can_use_coop ? "FLASH_ATTENTION_FP16" : "FLASH_ATTENTION_FALLBACK",
                                        seq_len, eff_kv_len, num_heads, eff_kv_heads, head_dim},
                           Log_Level::LOG_DEBUG,
                           false,
                           1,
                           Log_Feature::TRANSFORMER);

        if (can_use_coop)
        {
            std::shared_ptr<Gpu_Tensor_Impl> q_fp16, k_fp16, v_fp16;
            const Gpu_Tensor_Impl *in_q = effective_q;
            const Gpu_Tensor_Impl *in_k = effective_k;
            const Gpu_Tensor_Impl *in_v = effective_v;

            if (effective_q->getDataType() != Data_Type::FLOAT16)
            {
                q_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT16);
                effective_q->to(Data_Type::FLOAT16, *q_fp16);
                in_q = q_fp16.get();
            }
            if (effective_k->getDataType() != Data_Type::FLOAT16)
            {
                k_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT16);
                effective_k->to(Data_Type::FLOAT16, *k_fp16);
                in_k = k_fp16.get();
            }
            if (effective_v->getDataType() != Data_Type::FLOAT16)
            {
                v_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT16);
                effective_v->to(Data_Type::FLOAT16, *v_fp16);
                in_v = v_fp16.get();
            }

            ensureDisjointTensorTarget(output, target, q_shape, out_dtype, {&q, &k, &v});
            auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            ensureDisjointGpuStorage(out_gpu, q_shape.getTotalElements(), out_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get()});

            std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
            Gpu_Tensor_Impl *target_l = nullptr;
            if (l_stats)
            {
                ensureDisjointTensorTarget(*l_stats, target, Shape{ total_heads, seq_len }, Data_Type::FLOAT32, {&q, &k, &v, &output});
                auto *l_gpu = dynamic_cast<Gpu_Tensor_Impl *>(l_stats->getImplementation().get());
                if (!l_gpu)
                {
                    fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
                    target_l = fallback_l.get();
                }
                else
                {
                    size_t elem_size = sizeof(float);
                    if (!l_gpu->getStorageBuffer() || l_gpu->getStorageBuffer()->getSizeBytes() < static_cast<size_t>(total_heads) * seq_len * elem_size)
                    {
                        l_gpu->getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext(), static_cast<size_t>(total_heads) * seq_len, Data_Type::FLOAT32);
                    }
                    ensureDisjointGpuStorage(*l_gpu, static_cast<size_t>(total_heads) * seq_len, Data_Type::FLOAT32,
                                             {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(), out_gpu.getStorageBuffer().get()});
                    target_l = l_gpu;
                }
            }
            else
            {
                fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
                target_l = fallback_l.get();
            }

            std::shared_ptr<Gpu_Tensor_Impl> intermediate_out_fp16;
            Gpu_Tensor_Impl *target_out = &out_gpu;
            if (out_dtype != Data_Type::FLOAT16)
            {
                intermediate_out_fp16 = std::make_shared<Gpu_Tensor_Impl>(q_shape, Data_Type::FLOAT16);
                target_out = intermediate_out_fp16.get();
            }

            struct FlashAttention_Constants
            {
                uint32_t batch_size;
                uint32_t seq_len;
                uint32_t head_dim;
                uint32_t is_causal;
                float scale;
                uint32_t gqa_group_size;
                uint32_t num_heads;
                uint32_t is_bshd;
            } pcs{total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, gqa_group_size, num_heads, config.is_bshd ? 1u : 0u};
            validatePushConstants<FlashAttention_Constants>();

            uint32_t num_q_tiles = (seq_len + 63) / 64;
            effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_FP16,
                                     {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(), target_out->getStorageBuffer(), target_l->getStorageBuffer()},
                                     pcs, 1, num_q_tiles, total_heads, &engine);

            if (out_dtype != Data_Type::FLOAT16)
            {
                intermediate_out_fp16->to(Data_Type::FLOAT32, out_gpu);
            }
            if (target_l)
            {
                target_l->clearHostCache();
                target_l->invalidateFp16Cache();
            }
        }
        else
        {
            std::shared_ptr<Gpu_Tensor_Impl> q_fp32, k_fp32, v_fp32;
            const Gpu_Tensor_Impl *in_q = effective_q;
            const Gpu_Tensor_Impl *in_k = effective_k;
            const Gpu_Tensor_Impl *in_v = effective_v;

            if (effective_q->getDataType() != Data_Type::FLOAT32)
            {
                q_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT32);
                effective_q->to(Data_Type::FLOAT32, *q_fp32);
                in_q = q_fp32.get();
            }
            if (effective_k->getDataType() != Data_Type::FLOAT32)
            {
                k_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT32);
                effective_k->to(Data_Type::FLOAT32, *k_fp32);
                in_k = k_fp32.get();
            }
            if (effective_v->getDataType() != Data_Type::FLOAT32)
            {
                v_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT32);
                effective_v->to(Data_Type::FLOAT32, *v_fp32);
                in_v = v_fp32.get();
            }

            ensureDisjointTensorTarget(output, target, q_shape, out_dtype, {&q, &k, &v});
            auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            ensureDisjointGpuStorage(out_gpu, q_shape.getTotalElements(), out_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get()});

            std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
            Gpu_Tensor_Impl *target_l = nullptr;
            if (l_stats)
            {
                ensureDisjointTensorTarget(*l_stats, target, Shape{ total_heads, seq_len }, Data_Type::FLOAT32, {&q, &k, &v, &output});
                auto *l_gpu = dynamic_cast<Gpu_Tensor_Impl *>(l_stats->getImplementation().get());
                if (!l_gpu)
                {
                    fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
                    target_l = fallback_l.get();
                }
                else
                {
                    size_t elem_size = sizeof(float);
                    if (!l_gpu->getStorageBuffer() || l_gpu->getStorageBuffer()->getSizeBytes() < static_cast<size_t>(total_heads) * seq_len * elem_size)
                    {
                        l_gpu->getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext(), static_cast<size_t>(total_heads) * seq_len, Data_Type::FLOAT32);
                    }
                    ensureDisjointGpuStorage(*l_gpu, static_cast<size_t>(total_heads) * seq_len, Data_Type::FLOAT32,
                                             {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(), out_gpu.getStorageBuffer().get()});
                    target_l = l_gpu;
                }
            }
            else
            {
                fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
                target_l = fallback_l.get();
            }

            std::shared_ptr<Gpu_Tensor_Impl> intermediate_out_fp32;
            Gpu_Tensor_Impl *target_out = &out_gpu;
            if (out_dtype != Data_Type::FLOAT32)
            {
                intermediate_out_fp32 = std::make_shared<Gpu_Tensor_Impl>(q_shape, Data_Type::FLOAT32);
                target_out = intermediate_out_fp32.get();
            }

            uint32_t num_q_tiles = (seq_len + 15) / 16;
            if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL)
            {
                Tensor mask_dev(Execution_Target::VULKAN_GPU);
                if (config.attn_mask->getExecutionTarget() == Execution_Target::VULKAN_GPU &&
                    config.attn_mask->getDataType() == Data_Type::FLOAT32)
                {
                    mask_dev = config.attn_mask->contiguous();
                }
                else
                {
                    mask_dev = Tensor(config.attn_mask->getShape(), config.attn_mask->getData(), Execution_Target::VULKAN_GPU);
                    engine.getContext().executePendingTransfers();
                }
                const auto &mask_gpu = Gpu_Tensor_Impl::castToGpu(*mask_dev.getImplementation());

                struct FlashAttention_Masked_Constants
                {
                    uint32_t batch_size;
                    uint32_t q_seq_len;
                    uint32_t kv_seq_len;
                    uint32_t head_dim;
                    uint32_t mask_mode;
                    uint32_t causal_offset;
                    float scale;
                    uint32_t gqa_group_size;
                    uint32_t num_heads;
                    uint32_t is_bshd;
                    uint32_t mask_stride_batch;
                    uint32_t mask_per_head;
                } pcs{
                    total_heads, seq_len, eff_kv_len, head_dim,
                    static_cast<uint32_t>(mask_layout.mode), causal_offset,
                    scale, gqa_group_size, num_heads, config.is_bshd ? 1u : 0u,
                    mask_layout.stride_batch, mask_layout.per_head
                };
                validatePushConstants<FlashAttention_Masked_Constants>();

                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_MASKED,
                                         {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                                          target_out->getStorageBuffer(), target_l->getStorageBuffer(), mask_gpu.getStorageBuffer()},
                                         pcs, 1, num_q_tiles, total_heads, &engine);
            }
            else
            {
                struct FlashAttention_Fallback_Constants
                {
                    uint32_t batch_size;
                    uint32_t q_seq_len;
                    uint32_t kv_seq_len;
                    uint32_t head_dim;
                    uint32_t is_causal;
                    uint32_t causal_offset;
                    float scale;
                    uint32_t gqa_group_size;
                    uint32_t num_heads;
                    uint32_t is_bshd;
                } pcs{total_heads, seq_len, eff_kv_len, head_dim, is_causal ? 1u : 0u, causal_offset, scale, gqa_group_size, num_heads, config.is_bshd ? 1u : 0u};
                validatePushConstants<FlashAttention_Fallback_Constants>();

                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_FALLBACK,
                                         {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(), target_out->getStorageBuffer(), target_l->getStorageBuffer()},
                                         pcs, 1, num_q_tiles, total_heads, &engine);
            }

            if (out_dtype != Data_Type::FLOAT32)
            {
                intermediate_out_fp32->to(out_dtype, out_gpu);
            }
            if (target_l)
            {
                target_l->clearHostCache();
                target_l->invalidateFp16Cache();
            }
        }

        auto &out_cache_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        out_cache_gpu.clearHostCache();
        out_cache_gpu.invalidateFp16Cache();
    }
}

void flashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                           uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                           bool is_causal, float scale, Tensor *l_stats,
                           uint32_t kv_seq_len, uint32_t causal_offset,
                           uint32_t num_kv_heads, Execution_Engine *engine)
{
    Attention_Config config{
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .seq_len = seq_len,
        .head_dim = head_dim,
        .is_causal = is_causal,
        .scale = scale,
        .kv_seq_len = kv_seq_len,
        .causal_offset = causal_offset,
        .is_bshd = false,
        .engine = engine};
    flashAttentionForward(q, k, v, output, config, l_stats);
}

void flashAttentionBackward(const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &o, const Tensor &do_grad,
                            Tensor &dq, Tensor &dk, Tensor &dv,
                            const Attention_Config &config, const Tensor *l_stats)
{
    auto q_impl = q.getImplementation();
    auto k_impl = k.getImplementation();
    auto v_impl = v.getImplementation();
    auto o_impl = o.getImplementation();
    auto do_impl = do_grad.getImplementation();
    if (!q_impl || !k_impl || !v_impl || !o_impl || !do_impl) return;

    Execution_Target target = q.getExecutionTarget();
    validateTargetAndType(q, target, "ops::llm::flashAttentionBackward", "q");
    validateTargetAndType(k, target, "ops::llm::flashAttentionBackward", "k");
    validateTargetAndType(v, target, "ops::llm::flashAttentionBackward", "v");
    validateTargetAndType(o, target, "ops::llm::flashAttentionBackward", "o");
    validateTargetAndType(do_grad, target, "ops::llm::flashAttentionBackward", "do_grad");
    if (l_stats && l_stats->getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::flashAttentionBackward: l_stats cannot be TERNARY_INT2");
    }

    const uint32_t num_heads = config.num_heads;
    const uint32_t seq_len = config.seq_len;
    const uint32_t head_dim = config.head_dim;
    const bool is_causal = config.is_causal;
    float scale = config.scale;
    uint32_t eff_kv_len = (config.kv_seq_len > 0) ? config.kv_seq_len : seq_len;
    const uint32_t causal_offset = config.causal_offset;

    if (head_dim == 0 || num_heads == 0)
    {
        throw std::invalid_argument("ops::llm::flashAttentionBackward: head_dim and num_heads must be > 0");
    }
    const size_t gqa_group_sz = validateAndComputeGqaGroupSize(num_heads, config.num_kv_heads, "ops::llm::flashAttentionBackward");
    const uint32_t eff_kv_heads = static_cast<uint32_t>(num_heads / gqa_group_sz);
    const uint32_t gqa_group_size = static_cast<uint32_t>(gqa_group_sz);

    if (seq_len == 0 || q.getTotalElements() == 0)
    {
        const Shape q_shape = q.getShape();
        const Shape k_shape = k.getShape();
        const Shape v_shape = v.getShape();
        Data_Type empty_dtype = (target == Execution_Target::CPU)
            ? q.getDataType()
            : ((q.getDataType() == Data_Type::FLOAT16 && resolveExecutionEngine(config.engine).getContext().isFloat16Enabled())
                   ? Data_Type::FLOAT16
                   : Data_Type::FLOAT32);
        ensureDisjointTensorTarget(dq, target, q_shape, empty_dtype, {&q, &k, &v, &o, &do_grad, l_stats});
        ensureDisjointTensorTarget(dk, target, k_shape, empty_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq});
        ensureDisjointTensorTarget(dv, target, v_shape, empty_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq, &dk});
        return;
    }

    size_t total_elements_per_head = static_cast<size_t>(seq_len) * head_dim;
    size_t kv_elements_per_head = static_cast<size_t>(eff_kv_len) * head_dim;
    if (q.getTotalElements() % total_elements_per_head != 0)
    {
        throw std::invalid_argument("ops::llm::flashAttentionBackward: q total elements must be divisible by seq_len * head_dim");
    }
    size_t total_heads_sz = q.getTotalElements() / total_elements_per_head;
    if (total_heads_sz % num_heads != 0)
    {
        throw std::invalid_argument("ops::llm::flashAttentionBackward: inferred head count is not divisible by num_heads");
    }
    size_t batch_size_sz = total_heads_sz / num_heads;
    size_t total_kv_heads_sz = batch_size_sz * eff_kv_heads;
    if (config.kv_seq_len == 0 && config.kv_index != nullptr && total_kv_heads_sz * head_dim > 0 &&
        (k.getTotalElements() % (total_kv_heads_sz * head_dim)) == 0)
    {
        eff_kv_len = static_cast<uint32_t>(k.getTotalElements() / (total_kv_heads_sz * head_dim));
        kv_elements_per_head = static_cast<size_t>(eff_kv_len) * head_dim;
    }
    if (k.getTotalElements() != total_kv_heads_sz * kv_elements_per_head ||
        v.getTotalElements() != total_kv_heads_sz * kv_elements_per_head ||
        o.getTotalElements() != q.getTotalElements() ||
        do_grad.getTotalElements() != q.getTotalElements())
    {
        throw std::invalid_argument("ops::llm::flashAttentionBackward: tensor element count mismatch for GQA dimensions");
    }
    if (l_stats && !l_stats->isEmpty() && l_stats->getTotalElements() < total_heads_sz * seq_len)
    {
        throw std::invalid_argument("ops::llm::flashAttentionBackward: l_stats size is smaller than total_heads * seq_len");
    }

    if (scale <= 0.0f)
    {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    if (config.kv_index != nullptr)
    {
        if (!config.kv_index->getImplementation() || config.kv_index->getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::llm::flashAttentionBackward: Invalid kv_index tensor");
        }
        const size_t sel_T = config.kv_index->getTotalElements();
        Tensor k_sel = gatherKvSequenceSubset(k, *config.kv_index, batch_size_sz, eff_kv_heads, eff_kv_len, head_dim, config.is_bshd);
        Tensor v_sel = gatherKvSequenceSubset(v, *config.kv_index, batch_size_sz, eff_kv_heads, eff_kv_len, head_dim, config.is_bshd);
        if (target == Execution_Target::VULKAN_GPU)
        {
            resolveExecutionEngine(config.engine).getContext().executePendingTransfers();
        }
        Attention_Config sub_cfg = config;
        sub_cfg.kv_seq_len = static_cast<uint32_t>(sel_T);
        sub_cfg.kv_index = nullptr;
        if (sel_T == 0)
        {
            const Data_Type empty_dtype = q.getDataType();
            ensureDisjointTensorTarget(dq, target, q.getShape(), empty_dtype, {&q, &k, &v, &o, &do_grad, l_stats});
            ensureDisjointTensorTarget(dk, target, k.getShape(), empty_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq});
            ensureDisjointTensorTarget(dv, target, v.getShape(), empty_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq, &dk});
            dq.zero();
            dk.zero();
            dv.zero();
            return;
        }
        Tensor dk_sel(target), dv_sel(target);
        flashAttentionBackward(q, k_sel, v_sel, o, do_grad, dq, dk_sel, dv_sel, sub_cfg, l_stats);
        dk = scatterKvGradSubset(dk_sel, *config.kv_index, batch_size_sz, eff_kv_heads, eff_kv_len, sel_T, head_dim, config.is_bshd, q.getDataType());
        dv = scatterKvGradSubset(dv_sel, *config.kv_index, batch_size_sz, eff_kv_heads, eff_kv_len, sel_T, head_dim, config.is_bshd, q.getDataType());
        return;
    }

    const Resolved_Mask_Layout mask_layout = resolveMaskLayout(
        config, batch_size_sz, total_heads_sz, seq_len, eff_kv_len, "ops::llm::flashAttentionBackward");

    if (target == Execution_Target::CPU)
    {
        const auto q_data = q.getData();
        const auto k_data = k.getData();
        const auto v_data = v.getData();
        const auto o_data = o.getData();
        const auto do_data = do_grad.getData();
        std::vector<float> mask_data;
        if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL)
        {
            mask_data = config.attn_mask->getData();
        }

        std::vector<float> dq_vals(q.getTotalElements(), 0.0f);
        std::vector<float> dk_vals(k.getTotalElements(), 0.0f);
        std::vector<float> dv_vals(v.getTotalElements(), 0.0f);

        std::vector<float> l_data_cache;
        if (l_stats && !l_stats->isEmpty())
        {
            l_data_cache = l_stats->getData();
        }

        for (size_t h = 0; h < total_heads_sz; ++h)
        {
            const size_t batch_idx = h / num_heads;
            size_t q_head_offset;
            size_t q_row_stride;
            size_t kv_head_offset;
            size_t kv_row_stride;
            if (config.is_bshd)
            {
                size_t head_idx = h % num_heads;
                size_t kv_head_idx = head_idx / gqa_group_sz;
                q_head_offset = (batch_idx * seq_len * num_heads + head_idx) * head_dim;
                q_row_stride = static_cast<size_t>(num_heads) * head_dim;
                kv_head_offset = (batch_idx * eff_kv_len * eff_kv_heads + kv_head_idx) * head_dim;
                kv_row_stride = static_cast<size_t>(eff_kv_heads) * head_dim;
            }
            else
            {
                q_head_offset = h * total_elements_per_head;
                q_row_stride = head_dim;
                kv_head_offset = (h / gqa_group_sz) * kv_elements_per_head;
                kv_row_stride = head_dim;
            }

            auto allowed_key = [&](size_t qi, size_t kj) -> bool {
                if (mask_layout.mode == Attention_Mask_Mode::CAUSAL)
                {
                    return kj <= static_cast<size_t>(causal_offset) + qi;
                }
                if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL)
                {
                    const size_t mb = mask_layout.per_head ? h : batch_idx;
                    return mask_data[mb * mask_layout.stride_batch + qi * eff_kv_len + kj] > 0.5f;
                }
                return true;
            };

            for (size_t i = 0; i < seq_len; ++i)
            {
                std::vector<float> p(eff_kv_len, 0.0f);
                size_t max_j = (mask_layout.mode == Attention_Mask_Mode::CAUSAL)
                    ? std::min<size_t>(eff_kv_len, causal_offset + i + 1)
                    : static_cast<size_t>(eff_kv_len);
                bool has_any = true;

                if (l_stats != nullptr && !l_data_cache.empty())
                {
                    float l_i = l_data_cache[h * seq_len + i];
                    if (l_i <= -1e19f || !std::isfinite(l_i))
                    {
                        has_any = false;
                    }
                    else
                    {
                        for (size_t j = 0; j < max_j; ++j)
                        {
                            if (!allowed_key(i, j)) continue;
                            float dot = 0.0f;
                            for (size_t d = 0; d < head_dim; ++d)
                            {
                                dot += q_data[q_head_offset + i * q_row_stride + d] * k_data[kv_head_offset + j * kv_row_stride + d];
                            }
                            p[j] = std::exp(dot * scale - l_i);
                        }
                    }
                }
                else
                {
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        if (!allowed_key(i, j)) continue;
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            dot += q_data[q_head_offset + i * q_row_stride + d] * k_data[kv_head_offset + j * kv_row_stride + d];
                        }
                        p[j] = dot * scale;
                        if (p[j] > max_score)
                        {
                            max_score = p[j];
                        }
                    }

                    float sum_exp = 0.0f;
                    if (std::isfinite(max_score))
                    {
                        for (size_t j = 0; j < max_j; ++j)
                        {
                            if (!allowed_key(i, j))
                            {
                                p[j] = 0.0f;
                                continue;
                            }
                            p[j] = std::exp(p[j] - max_score);
                            sum_exp += p[j];
                        }
                    }
                    else
                    {
                        has_any = false;
                    }
                    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                    if (inv_sum == 0.0f)
                    {
                        has_any = false;
                    }
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        p[j] *= inv_sum;
                    }
                }

                if (!has_any) continue;

                float di = 0.0f;
                for (size_t d = 0; d < head_dim; ++d)
                {
                    di += do_data[q_head_offset + i * q_row_stride + d] * o_data[q_head_offset + i * q_row_stride + d];
                }

                for (size_t j = 0; j < max_j; ++j)
                {
                    if (!allowed_key(i, j) || p[j] == 0.0f) continue;
                    float dp = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dp += do_data[q_head_offset + i * q_row_stride + d] * v_data[kv_head_offset + j * kv_row_stride + d];
                    }

                    float ds = p[j] * (dp - di) * scale;

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dq_vals[q_head_offset + i * q_row_stride + d] += ds * k_data[kv_head_offset + j * kv_row_stride + d];
                    }

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dk_vals[kv_head_offset + j * kv_row_stride + d] += ds * q_data[q_head_offset + i * q_row_stride + d];
                    }

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dv_vals[kv_head_offset + j * kv_row_stride + d] += p[j] * do_data[q_head_offset + i * q_row_stride + d];
                    }
                }
            }
        }

        const Shape q_shape = q.getShape();
        const Shape k_shape = k.getShape();
        const Shape v_shape = v.getShape();
        const Data_Type q_dtype = q.getDataType();
        writeCpuOutput(dq, q_shape, q_dtype, dq_vals);
        writeCpuOutput(dk, k_shape, q_dtype, dk_vals);
        writeCpuOutput(dv, v_shape, q_dtype, dv_vals);
    }
    else
    {
        Execution_Engine &engine = resolveExecutionEngine(config.engine);
        const auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q_impl);
        auto contig_q = q_gpu.ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : &q_gpu;

        const auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k_impl);
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;

        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v_impl);
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;

        const auto &o_gpu = Gpu_Tensor_Impl::castToGpu(*o_impl);
        auto contig_o = o_gpu.ensureContiguousSelf();
        const auto *effective_o = contig_o ? contig_o.get() : &o_gpu;

        const auto &do_gpu = Gpu_Tensor_Impl::castToGpu(*do_impl);
        auto contig_do = do_gpu.ensureContiguousSelf();
        const auto *effective_do = contig_do ? contig_do.get() : &do_gpu;

        if (total_heads_sz > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::llm::flashAttentionBackward: total_heads exceeds uint32_t");
        }
        uint32_t total_heads = static_cast<uint32_t>(total_heads_sz);
        uint32_t total_kv_heads = static_cast<uint32_t>(total_kv_heads_sz);
        const uint64_t num_tiles_check = (static_cast<uint64_t>(seq_len) + 15ULL) / 16ULL;
        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (num_tiles_check > max_wg[1] || total_heads > max_wg[2])
        {
            throw std::overflow_error("ops::llm::flashAttentionBackward: Workgroup grid exceeds hardware limits");
        }

        std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
        std::shared_ptr<Gpu_Tensor_Impl> contig_l;
        const Gpu_Tensor_Impl *target_l = nullptr;
        bool need_recompute_l = false;
        if (l_stats && !l_stats->isEmpty() && l_stats->getImplementation())
        {
            const auto *l_gpu = dynamic_cast<const Gpu_Tensor_Impl *>(l_stats->getImplementation().get());
            if (l_gpu)
            {
                contig_l = l_gpu->ensureContiguousSelf();
                const Gpu_Tensor_Impl *eff_l = contig_l ? contig_l.get() : l_gpu;
                if (eff_l->getDataType() != Data_Type::FLOAT32)
                {
                    fallback_l = std::make_shared<Gpu_Tensor_Impl>(eff_l->getShape(), Data_Type::FLOAT32);
                    eff_l->to(Data_Type::FLOAT32, *fallback_l);
                    target_l = fallback_l.get();
                }
                else
                {
                    target_l = eff_l;
                }
            }
            else
            {
                std::vector<float> host_l = l_stats->getData();
                if (host_l.size() >= static_cast<size_t>(total_heads) * seq_len)
                {
                    host_l.resize(static_cast<size_t>(total_heads) * seq_len);
                    fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, host_l, Data_Type::FLOAT32);
                    engine.getContext().executePendingTransfers();
                    target_l = fallback_l.get();
                }
                else
                {
                    fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, Data_Type::FLOAT32);
                    target_l = fallback_l.get();
                    need_recompute_l = true;
                }
            }
        }
        else
        {
            fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, Data_Type::FLOAT32);
            target_l = fallback_l.get();
            need_recompute_l = true;
        }

        const Shape q_shape = q.getShape();
        const Shape k_shape = k.getShape();
        const Shape v_shape = v.getShape();
        Data_Type target_dtype = (effective_q->getDataType() == Data_Type::FLOAT16 && engine.getContext().isFloat16Enabled())
            ? Data_Type::FLOAT16
            : Data_Type::FLOAT32;

        bool can_use_coop = engine.isCooperativeMatrixEnabled() &&
                            engine.getContext().isFloat16Enabled() &&
                            mask_layout.mode != Attention_Mask_Mode::DENSE_BOOL &&
                            eff_kv_len == seq_len && causal_offset == 0 &&
                            head_dim > 0 && head_dim <= 64 && (head_dim % 16 == 0) &&
                            engine.getNetwork().getPipeline(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16) != VK_NULL_HANDLE &&
                            engine.getNetwork().getPipeline(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16) != VK_NULL_HANDLE;

        if (can_use_coop)
        {
            std::shared_ptr<Gpu_Tensor_Impl> q_fp16, k_fp16, v_fp16, o_fp16, do_fp16;
            const Gpu_Tensor_Impl *in_q = effective_q;
            const Gpu_Tensor_Impl *in_k = effective_k;
            const Gpu_Tensor_Impl *in_v = effective_v;
            const Gpu_Tensor_Impl *in_o = effective_o;
            const Gpu_Tensor_Impl *in_do = effective_do;

            if (effective_q->getDataType() != Data_Type::FLOAT16)
            {
                q_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT16);
                effective_q->to(Data_Type::FLOAT16, *q_fp16);
                in_q = q_fp16.get();
            }
            if (effective_k->getDataType() != Data_Type::FLOAT16)
            {
                k_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT16);
                effective_k->to(Data_Type::FLOAT16, *k_fp16);
                in_k = k_fp16.get();
            }
            if (effective_v->getDataType() != Data_Type::FLOAT16)
            {
                v_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT16);
                effective_v->to(Data_Type::FLOAT16, *v_fp16);
                in_v = v_fp16.get();
            }
            if (effective_o->getDataType() != Data_Type::FLOAT16)
            {
                o_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_o->getShape(), Data_Type::FLOAT16);
                effective_o->to(Data_Type::FLOAT16, *o_fp16);
                in_o = o_fp16.get();
            }
            if (effective_do->getDataType() != Data_Type::FLOAT16)
            {
                do_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_do->getShape(), Data_Type::FLOAT16);
                effective_do->to(Data_Type::FLOAT16, *do_fp16);
                in_do = do_fp16.get();
            }

            ensureDisjointTensorTarget(dq, target, q_shape, target_dtype, {&q, &k, &v, &o, &do_grad, l_stats});
            ensureDisjointTensorTarget(dk, target, k_shape, target_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq});
            ensureDisjointTensorTarget(dv, target, v_shape, target_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq, &dk});

            auto &dq_gpu = Gpu_Tensor_Impl::castToGpu(*dq.getImplementation());
            auto &dk_gpu = Gpu_Tensor_Impl::castToGpu(*dk.getImplementation());
            auto &dv_gpu = Gpu_Tensor_Impl::castToGpu(*dv.getImplementation());
            ensureDisjointGpuStorage(dq_gpu, q_shape.getTotalElements(), target_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(),
                                      in_o->getStorageBuffer().get(), in_do->getStorageBuffer().get(), target_l->getStorageBuffer().get()});
            ensureDisjointGpuStorage(dk_gpu, k_shape.getTotalElements(), target_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(),
                                      in_o->getStorageBuffer().get(), in_do->getStorageBuffer().get(), target_l->getStorageBuffer().get(),
                                      dq_gpu.getStorageBuffer().get()});
            ensureDisjointGpuStorage(dv_gpu, v_shape.getTotalElements(), target_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(),
                                      in_o->getStorageBuffer().get(), in_do->getStorageBuffer().get(), target_l->getStorageBuffer().get(),
                                      dq_gpu.getStorageBuffer().get(), dk_gpu.getStorageBuffer().get()});

            std::shared_ptr<Gpu_Tensor_Impl> dq_fp16, dk_fp16, dv_fp16;
            Gpu_Tensor_Impl *target_dq = &dq_gpu;
            Gpu_Tensor_Impl *target_dk = &dk_gpu;
            Gpu_Tensor_Impl *target_dv = &dv_gpu;

            if (target_dtype != Data_Type::FLOAT16)
            {
                dq_fp16 = std::make_shared<Gpu_Tensor_Impl>(q_shape, Data_Type::FLOAT16);
                dk_fp16 = std::make_shared<Gpu_Tensor_Impl>(k_shape, Data_Type::FLOAT16);
                dv_fp16 = std::make_shared<Gpu_Tensor_Impl>(v_shape, Data_Type::FLOAT16);
                target_dq = dq_fp16.get();
                target_dk = dk_fp16.get();
                target_dv = dv_fp16.get();
            }

            if (need_recompute_l)
            {
                struct FlashAttention_Lse_Constants
                {
                    uint32_t batch_size;
                    uint32_t seq_len;
                    uint32_t head_dim;
                    uint32_t is_causal;
                    float scale;
                    uint32_t gqa_group_size;
                    uint32_t num_heads;
                    uint32_t is_bshd;
                } pcs_lse{total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, gqa_group_size, num_heads, config.is_bshd ? 1u : 0u};
                validatePushConstants<FlashAttention_Lse_Constants>();

                uint32_t num_lse_tiles = (seq_len + 15) / 16;
                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_LSE_FP16,
                                         {in_q->getStorageBuffer(), in_k->getStorageBuffer(), target_l->getStorageBuffer()},
                                         pcs_lse, 1, num_lse_tiles, total_heads, &engine);
            }

            uint32_t num_q_tiles = (seq_len + 63) / 64;
            uint32_t num_k_tiles = (seq_len + 63) / 64;

            size_t d_elements = static_cast<size_t>(total_heads) * seq_len;
            auto d_buf = engine.acquireReusableScratchBuffer(Execution_Engine::SCRATCH_FLASH_ATTN_DI,
                                                             d_elements * sizeof(float),
                                                             Data_Type::FLOAT32,
                                                             d_elements);

            struct FlashAttention_Backward_Constants
            {
                uint32_t batch_size;
                uint32_t seq_len;
                uint32_t head_dim;
                uint32_t is_causal;
                float scale;
                uint32_t pass_type;
                uint32_t recompute_l;
                uint32_t gqa_group_size;
                uint32_t num_heads;
                uint32_t is_bshd;
            };
            validatePushConstants<FlashAttention_Backward_Constants>();

            uint32_t is_bshd_u32 = config.is_bshd ? 1u : 0u;
            const uint32_t spec_head_dim = (head_dim == 32u || head_dim == 64u) ? head_dim : 0u;
            FlashAttention_Backward_Constants pcs_pass0{
                total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 0u, 0u, gqa_group_size, num_heads, is_bshd_u32
            };
            effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16,
                        {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                         in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                         target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(),
                         target_l->getStorageBuffer(), d_buf},
                        pcs_pass0, 1, num_q_tiles, total_heads, &engine, spec_head_dim);

            FlashAttention_Backward_Constants pcs_pass1{
                total_kv_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 1u, 0u, gqa_group_size, num_heads, is_bshd_u32
            };
            effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16,
                        {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                         in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                         target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(),
                         target_l->getStorageBuffer(), d_buf},
                        pcs_pass1, 1, num_k_tiles, total_kv_heads, &engine, spec_head_dim);

            if (target_dtype != Data_Type::FLOAT16)
            {
                dq_fp16->to(Data_Type::FLOAT32, dq_gpu);
                dk_fp16->to(Data_Type::FLOAT32, dk_gpu);
                dv_fp16->to(Data_Type::FLOAT32, dv_gpu);
            }
        }
        else
        {
            std::shared_ptr<Gpu_Tensor_Impl> q_fp32, k_fp32, v_fp32, o_fp32, do_fp32;
            const Gpu_Tensor_Impl *in_q = effective_q;
            const Gpu_Tensor_Impl *in_k = effective_k;
            const Gpu_Tensor_Impl *in_v = effective_v;
            const Gpu_Tensor_Impl *in_o = effective_o;
            const Gpu_Tensor_Impl *in_do = effective_do;

            if (effective_q->getDataType() != Data_Type::FLOAT32)
            {
                q_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT32);
                effective_q->to(Data_Type::FLOAT32, *q_fp32);
                in_q = q_fp32.get();
            }
            if (effective_k->getDataType() != Data_Type::FLOAT32)
            {
                k_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT32);
                effective_k->to(Data_Type::FLOAT32, *k_fp32);
                in_k = k_fp32.get();
            }
            if (effective_v->getDataType() != Data_Type::FLOAT32)
            {
                v_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT32);
                effective_v->to(Data_Type::FLOAT32, *v_fp32);
                in_v = v_fp32.get();
            }
            if (effective_o->getDataType() != Data_Type::FLOAT32)
            {
                o_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_o->getShape(), Data_Type::FLOAT32);
                effective_o->to(Data_Type::FLOAT32, *o_fp32);
                in_o = o_fp32.get();
            }
            if (effective_do->getDataType() != Data_Type::FLOAT32)
            {
                do_fp32 = std::make_shared<Gpu_Tensor_Impl>(effective_do->getShape(), Data_Type::FLOAT32);
                effective_do->to(Data_Type::FLOAT32, *do_fp32);
                in_do = do_fp32.get();
            }

            ensureDisjointTensorTarget(dq, target, q_shape, target_dtype, {&q, &k, &v, &o, &do_grad, l_stats});
            ensureDisjointTensorTarget(dk, target, k_shape, target_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq});
            ensureDisjointTensorTarget(dv, target, v_shape, target_dtype, {&q, &k, &v, &o, &do_grad, l_stats, &dq, &dk});

            auto &dq_gpu = Gpu_Tensor_Impl::castToGpu(*dq.getImplementation());
            auto &dk_gpu = Gpu_Tensor_Impl::castToGpu(*dk.getImplementation());
            auto &dv_gpu = Gpu_Tensor_Impl::castToGpu(*dv.getImplementation());
            ensureDisjointGpuStorage(dq_gpu, q_shape.getTotalElements(), target_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(),
                                      in_o->getStorageBuffer().get(), in_do->getStorageBuffer().get(), target_l->getStorageBuffer().get()});
            ensureDisjointGpuStorage(dk_gpu, k_shape.getTotalElements(), target_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(),
                                      in_o->getStorageBuffer().get(), in_do->getStorageBuffer().get(), target_l->getStorageBuffer().get(),
                                      dq_gpu.getStorageBuffer().get()});
            ensureDisjointGpuStorage(dv_gpu, v_shape.getTotalElements(), target_dtype,
                                     {in_q->getStorageBuffer().get(), in_k->getStorageBuffer().get(), in_v->getStorageBuffer().get(),
                                      in_o->getStorageBuffer().get(), in_do->getStorageBuffer().get(), target_l->getStorageBuffer().get(),
                                      dq_gpu.getStorageBuffer().get(), dk_gpu.getStorageBuffer().get()});

            std::shared_ptr<Gpu_Tensor_Impl> dq_fp32, dk_fp32, dv_fp32;
            Gpu_Tensor_Impl *target_dq = &dq_gpu;
            Gpu_Tensor_Impl *target_dk = &dk_gpu;
            Gpu_Tensor_Impl *target_dv = &dv_gpu;

            if (target_dtype != Data_Type::FLOAT32)
            {
                dq_fp32 = std::make_shared<Gpu_Tensor_Impl>(q.getShape(), Data_Type::FLOAT32);
                dk_fp32 = std::make_shared<Gpu_Tensor_Impl>(k.getShape(), Data_Type::FLOAT32);
                dv_fp32 = std::make_shared<Gpu_Tensor_Impl>(v.getShape(), Data_Type::FLOAT32);
                target_dq = dq_fp32.get();
                target_dk = dk_fp32.get();
                target_dv = dv_fp32.get();
            }

            uint32_t num_q_tiles = (seq_len + 15) / 16;
            uint32_t num_k_tiles = (eff_kv_len + 15) / 16;
            uint32_t is_bshd_u32 = config.is_bshd ? 1u : 0u;

            if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL || eff_kv_len != seq_len || causal_offset != 0u)
            {
                Tensor mask_dev(Execution_Target::VULKAN_GPU);
                std::shared_ptr<gpu::vector> mask_buf;
                if (mask_layout.mode == Attention_Mask_Mode::DENSE_BOOL)
                {
                    if (config.attn_mask->getExecutionTarget() == Execution_Target::VULKAN_GPU &&
                        config.attn_mask->getDataType() == Data_Type::FLOAT32)
                    {
                        mask_dev = config.attn_mask->contiguous();
                    }
                    else
                    {
                        mask_dev = Tensor(config.attn_mask->getShape(), config.attn_mask->getData(), Execution_Target::VULKAN_GPU);
                        engine.getContext().executePendingTransfers();
                    }
                    mask_buf = Gpu_Tensor_Impl::castToGpu(*mask_dev.getImplementation()).getStorageBuffer();
                }
                else
                {
                    mask_buf = in_q->getStorageBuffer();
                }

                struct FlashAttention_Masked_Backward_Constants
                {
                    uint32_t batch_size;
                    uint32_t q_seq_len;
                    uint32_t kv_seq_len;
                    uint32_t head_dim;
                    uint32_t mask_mode;
                    uint32_t causal_offset;
                    float scale;
                    uint32_t pass_type;
                    uint32_t recompute_l;
                    uint32_t gqa_group_size;
                    uint32_t num_heads;
                    uint32_t is_bshd;
                    uint32_t mask_stride_batch;
                    uint32_t mask_per_head;
                };
                validatePushConstants<FlashAttention_Masked_Backward_Constants>();

                FlashAttention_Masked_Backward_Constants pcs_pass0{
                    total_heads, seq_len, eff_kv_len, head_dim,
                    static_cast<uint32_t>(mask_layout.mode), causal_offset, scale,
                    0u, need_recompute_l ? 1u : 0u, gqa_group_size, num_heads, is_bshd_u32,
                    mask_layout.stride_batch, mask_layout.per_head
                };
                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_MASKED_BACKWARD,
                            {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                             in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                             target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(),
                             target_l->getStorageBuffer(), mask_buf},
                            pcs_pass0, 1, num_q_tiles, total_heads, &engine);

                FlashAttention_Masked_Backward_Constants pcs_pass1{
                    total_kv_heads, seq_len, eff_kv_len, head_dim,
                    static_cast<uint32_t>(mask_layout.mode), causal_offset, scale,
                    1u, 0u, gqa_group_size, num_heads, is_bshd_u32,
                    mask_layout.stride_batch, mask_layout.per_head
                };
                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_MASKED_BACKWARD,
                            {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                             in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                             target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(),
                             target_l->getStorageBuffer(), mask_buf},
                            pcs_pass1, 1, num_k_tiles, total_kv_heads, &engine);
            }
            else
            {
                struct FlashAttention_Backward_Fallback_Constants
                {
                    uint32_t batch_size;
                    uint32_t seq_len;
                    uint32_t head_dim;
                    uint32_t is_causal;
                    float scale;
                    uint32_t pass_type;
                    uint32_t recompute_l;
                    uint32_t gqa_group_size;
                    uint32_t num_heads;
                    uint32_t is_bshd;
                };
                validatePushConstants<FlashAttention_Backward_Fallback_Constants>();

                FlashAttention_Backward_Fallback_Constants pcs_pass0{
                    total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 0u, need_recompute_l ? 1u : 0u, gqa_group_size, num_heads, is_bshd_u32
                };
                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_FALLBACK,
                            {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                             in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                             target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(), target_l->getStorageBuffer()},
                            pcs_pass0, 1, num_q_tiles, total_heads, &engine);

                FlashAttention_Backward_Fallback_Constants pcs_pass1{
                    total_kv_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 1u, 0u, gqa_group_size, num_heads, is_bshd_u32
                };
                effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_FALLBACK,
                            {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                             in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                             target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(), target_l->getStorageBuffer()},
                            pcs_pass1, 1, num_k_tiles, total_kv_heads, &engine);
            }

            if (target_dtype != Data_Type::FLOAT32)
            {
                dq_fp32->to(target_dtype, dq_gpu);
                dk_fp32->to(target_dtype, dk_gpu);
                dv_fp32->to(target_dtype, dv_gpu);
            }
        }

        auto &dq_cache_gpu = Gpu_Tensor_Impl::castToGpu(*dq.getImplementation());
        auto &dk_cache_gpu = Gpu_Tensor_Impl::castToGpu(*dk.getImplementation());
        auto &dv_cache_gpu = Gpu_Tensor_Impl::castToGpu(*dv.getImplementation());
        dq_cache_gpu.clearHostCache();
        dq_cache_gpu.invalidateFp16Cache();
        dk_cache_gpu.clearHostCache();
        dk_cache_gpu.invalidateFp16Cache();
        dv_cache_gpu.clearHostCache();
        dv_cache_gpu.invalidateFp16Cache();
    }
}

void flashAttentionBackward(const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &o, const Tensor &do_grad,
                            Tensor &dq, Tensor &dk, Tensor &dv,
                            uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                            bool is_causal, float scale, const Tensor *l_stats,
                            uint32_t num_kv_heads, Execution_Engine *engine)
{
    Attention_Config config{
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .seq_len = seq_len,
        .head_dim = head_dim,
        .is_causal = is_causal,
        .scale = scale,
        .kv_seq_len = 0u,
        .causal_offset = 0u,
        .is_bshd = false,
        .engine = engine};
    flashAttentionBackward(q, k, v, o, do_grad, dq, dk, dv, config, l_stats);
}

static void singleTokenAttentionForwardImpl(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                                            size_t num_heads, size_t num_kv_heads, size_t head_dim,
                                            size_t total_seq_len, float scale, Execution_Engine *engine_ptr)
{
    auto q_impl = q.getImplementation();
    auto k_impl = k.getImplementation();
    auto v_impl = v.getImplementation();
    if (!q_impl || !k_impl || !v_impl) return;

    Execution_Target target = q.getExecutionTarget();
    validateTargetAndType(q, target, "ops::llm::singleTokenAttentionForward", "q");
    validateTargetAndType(k, target, "ops::llm::singleTokenAttentionForward", "k");
    validateTargetAndType(v, target, "ops::llm::singleTokenAttentionForward", "v");

    if (head_dim == 0 || num_heads == 0)
    {
        throw std::invalid_argument("ops::llm::singleTokenAttentionForward: head_dim and num_heads must be > 0");
    }
    const size_t gqa_group_sz = validateAndComputeGqaGroupSize(num_heads, num_kv_heads, "ops::llm::singleTokenAttentionForward");
    const size_t eff_kv_heads = num_heads / gqa_group_sz;

    size_t B = 0;
    if (q.getShape().getRank() == 4)
    {
        B = q.getShape()[0];
    }
    else
    {
        if (q.getTotalElements() % (num_heads * head_dim) != 0)
        {
            throw std::invalid_argument("ops::llm::singleTokenAttentionForward: q total elements must be divisible by num_heads * head_dim");
        }
        B = q.getTotalElements() / (num_heads * head_dim);
    }

    size_t total_heads = B * num_heads;
    size_t total_kv_heads = B * eff_kv_heads;
    if (q.getTotalElements() != total_heads * head_dim)
    {
        throw std::invalid_argument("ops::llm::singleTokenAttentionForward: q size mismatch (expected B * num_heads * head_dim)");
    }

    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(engine_ptr) : nullptr;
    Shape out_shape{ B, num_heads, 1, head_dim };
    Data_Type desired_out_dtype = (target != Execution_Target::CPU && q.getDataType() == Data_Type::FLOAT16 && !resolved_engine->getContext().isFloat16Enabled())
        ? Data_Type::FLOAT32
        : q.getDataType();

    if (B == 0)
    {
        ensureDisjointTensorTarget(output, target, out_shape, desired_out_dtype, {&q, &k, &v});
        return;
    }
    if (total_seq_len == 0)
    {
        ensureDisjointTensorTarget(output, target, out_shape, desired_out_dtype, {&q, &k, &v});
        if (target == Execution_Target::CPU)
        {
            writeCpuOutput(output, out_shape, desired_out_dtype, std::vector<float>(total_heads * head_dim, 0.0f));
        }
        else
        {
            auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            pushGpuFill(out_gpu, total_heads * head_dim, 0.0f, *resolved_engine);
        }
        return;
    }

    size_t required_kv = total_kv_heads * total_seq_len * head_dim;
    if (k.getTotalElements() < required_kv || v.getTotalElements() < required_kv)
    {
        throw std::invalid_argument("ops::llm::singleTokenAttentionForward: k/v size smaller than B * num_kv_heads * total_seq_len * head_dim");
    }

    const float eff_scale = (scale > 0.0f) ? scale : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    if (target == Execution_Target::CPU)
    {
        const auto q_data = q.getData();
        const auto k_data = k.getData();
        const auto v_data = v.getData();
        ensureDisjointTensorTarget(output, target, out_shape, desired_out_dtype, {&q, &k, &v});

        std::vector<float> fp32_out(total_heads * head_dim, 0.0f);
        std::vector<float> scores(total_seq_len);

        for (size_t h = 0; h < total_heads; ++h)
        {
            size_t kv_h = h / gqa_group_sz;
            const float *q_h = q_data.data() + h * head_dim;
            const float *k_h = k_data.data() + kv_h * total_seq_len * head_dim;
            const float *v_h = v_data.data() + kv_h * total_seq_len * head_dim;
            float *out_h = fp32_out.data() + h * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t t = 0; t < total_seq_len; ++t)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; ++i)
                {
                    dot += q_h[i] * k_h[t * head_dim + i];
                }
                scores[t] = dot * eff_scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            float sum_exp = 0.0f;
            if (std::isfinite(max_score))
            {
                for (size_t t = 0; t < total_seq_len; ++t)
                {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
            }
            else
            {
                for (size_t t = 0; t < total_seq_len; ++t)
                {
                    scores[t] = 0.0f;
                }
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            for (size_t i = 0; i < head_dim; ++i)
            {
                float out_val = 0.0f;
                for (size_t t = 0; t < total_seq_len; ++t)
                {
                    out_val += (scores[t] * inv_sum) * v_h[t * head_dim + i];
                }
                out_h[i] = out_val;
            }
        }

        writeCpuOutput(output, out_shape, desired_out_dtype, fp32_out);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        const auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q_impl);
        auto contig_q = q_gpu.ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : &q_gpu;

        const auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k_impl);
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;

        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v_impl);
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;

        bool is_fp16 = engine.getContext().isFloat16Enabled() &&
                       (effective_q->getDataType() == Data_Type::FLOAT16 ||
                        effective_k->getDataType() == Data_Type::FLOAT16 ||
                        effective_v->getDataType() == Data_Type::FLOAT16);
        Data_Type pipe_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        std::shared_ptr<Gpu_Tensor_Impl> casted_q, casted_k, casted_v;
        if (!is_fp16)
        {
            if (effective_q->getDataType() == Data_Type::FLOAT16)
            {
                casted_q = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT32);
                effective_q->to(Data_Type::FLOAT32, *casted_q);
                effective_q = casted_q.get();
            }
            if (effective_k->getDataType() == Data_Type::FLOAT16)
            {
                casted_k = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT32);
                effective_k->to(Data_Type::FLOAT32, *casted_k);
                effective_k = casted_k.get();
            }
            if (effective_v->getDataType() == Data_Type::FLOAT16)
            {
                casted_v = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT32);
                effective_v->to(Data_Type::FLOAT32, *casted_v);
                effective_v = casted_v.get();
            }
        }

        auto buf_q = is_fp16 ? effective_q->getEffectiveFp16Storage() : effective_q->getStorageBuffer();
        auto buf_k = is_fp16 ? effective_k->getEffectiveFp16Storage() : effective_k->getStorageBuffer();
        auto buf_v = is_fp16 ? effective_v->getEffectiveFp16Storage() : effective_v->getStorageBuffer();

        ensureDisjointTensorTarget(output, target, out_shape, desired_out_dtype, {&q, &k, &v});
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        std::shared_ptr<Gpu_Tensor_Impl> intermediate_out;
        Gpu_Tensor_Impl *target_out = &out_gpu;
        if (desired_out_dtype != pipe_dtype)
        {
            intermediate_out = std::make_shared<Gpu_Tensor_Impl>(out_shape, pipe_dtype);
            target_out = intermediate_out.get();
        }
        else
        {
            size_t elem_size = getDataTypeSize(pipe_dtype);
            if (!out_gpu.getStorageBuffer() || out_gpu.getStorageBuffer()->getSizeBytes() < total_heads * head_dim * elem_size)
            {
                out_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext());
                out_gpu.getStorageBuffer()->allocateMemory(total_heads * head_dim, pipe_dtype);
            }
            ensureDisjointGpuStorage(out_gpu, total_heads * head_dim, pipe_dtype, {buf_q.get(), buf_k.get(), buf_v.get()});
        }

        if (total_heads > std::numeric_limits<uint32_t>::max() ||
            head_dim > std::numeric_limits<uint32_t>::max() ||
            total_seq_len > std::numeric_limits<uint32_t>::max() ||
            gqa_group_sz > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::llm::singleTokenAttentionForward: Dimensions exceed uint32_t");
        }
        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (total_heads > max_wg[0])
        {
            throw std::overflow_error("ops::llm::singleTokenAttentionForward: Workgroup count exceeds maxComputeWorkGroupCount[0]");
        }

        struct Attention_Decode_Constants
        {
            uint32_t num_heads;
            uint32_t head_dim;
            uint32_t total_seq_len;
            float scale;
            uint32_t gqa_group_size;
        } pcs{ static_cast<uint32_t>(total_heads), static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(total_seq_len), eff_scale, static_cast<uint32_t>(gqa_group_sz) };
        validatePushConstants<Attention_Decode_Constants>();

        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ATTENTION_DECODE_FP16 : Compute_Pipeline::ATTENTION_DECODE;

        effective_q->pushToGraph(pipe,
                                 { buf_q, buf_k, buf_v, target_out->getStorageBuffer() },
                                 pcs,
                                 static_cast<uint32_t>(total_heads), 1, 1, &engine);
        if (desired_out_dtype != pipe_dtype)
        {
            intermediate_out->to(desired_out_dtype, out_gpu);
        }
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
    }
}

void singleTokenAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                                 const Attention_Config &config)
{
    const size_t total_seq_len = (config.kv_seq_len > 0) ? config.kv_seq_len : config.seq_len;
    singleTokenAttentionForwardImpl(q, k, v, output,
                                    config.num_heads, config.num_kv_heads, config.head_dim,
                                    total_seq_len, config.scale, config.engine);
}

void singleTokenAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                                 size_t num_heads, size_t head_dim, size_t total_seq_len,
                                 size_t num_kv_heads, Execution_Engine *engine)
{
    singleTokenAttentionForwardImpl(q, k, v, output,
                                    num_heads, num_kv_heads, head_dim,
                                    total_seq_len, 0.0f, engine);
}

}

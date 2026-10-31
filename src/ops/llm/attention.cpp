#include "ops/llm/attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::llm
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

void flashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                           uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                           bool is_causal, float scale, Tensor *l_stats)
{
    auto q_impl = q.getImplementation();
    auto k_impl = k.getImplementation();
    auto v_impl = v.getImplementation();
    if (!q_impl || !k_impl || !v_impl) return;

    Execution_Target target = q.getExecutionTarget();
    if (scale <= 0.0f)
    {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    if (target == Execution_Target::CPU)
    {
        ensureTensorTarget(output, target, q.getShape(), q.getDataType());

        const auto &q_data = q.getData();
        const auto &k_data = k.getData();
        const auto &v_data = v.getData();

        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        if (!out_cpu.getStorageBuffer() || out_cpu.getStorageBuffer()->size() != q.getTotalElements())
        {
            out_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(q.getTotalElements(), 0.0f);
        }

        size_t total_elements_per_head = static_cast<size_t>(seq_len) * head_dim;
        size_t total_heads = (total_elements_per_head > 0) ? (q.getTotalElements() / total_elements_per_head) : num_heads;

        Cpu_Tensor_Impl *l_cpu = nullptr;
        if (l_stats)
        {
            ensureTensorTarget(*l_stats, Execution_Target::CPU, Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
            l_cpu = &Cpu_Tensor_Impl::castToCpu(*l_stats->getImplementation());
            if (!l_cpu->getStorageBuffer() || l_cpu->getStorageBuffer()->size() != total_heads * seq_len)
            {
                l_cpu->getStorageBuffer() = std::make_shared<std::vector<float>>(total_heads * seq_len, 0.0f);
            }
        }

        for (size_t h = 0; h < total_heads; ++h)
        {
            size_t head_offset = h * total_elements_per_head;

            for (size_t i = 0; i < seq_len; ++i)
            {
                std::vector<float> scores(seq_len, -1e20f);
                float max_score = -1e20f;

                size_t max_j = is_causal ? (i + 1) : seq_len;
                for (size_t j = 0; j < max_j; ++j)
                {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dot += q_data[head_offset + i * head_dim + d] * k_data[head_offset + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > max_score)
                    {
                        max_score = scores[j];
                    }
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j < max_j; ++j)
                {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

                if (l_cpu)
                {
                    float logsumexp = (sum_exp > 0.0f) ? (max_score + std::log(sum_exp)) : -1e20f;
                    (*l_cpu->getStorageBuffer())[h * seq_len + i] = logsumexp;
                }

                for (size_t d = 0; d < head_dim; ++d)
                {
                    float out_val = 0.0f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        out_val += (scores[j] * inv_sum) * v_data[head_offset + j * head_dim + d];
                    }
                    (*out_cpu.getStorageBuffer())[head_offset + i * head_dim + d] = out_val;
                }
            }
        }
    }
    else
    {
        const auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q.getImplementation());
        auto contig_q = q_gpu.ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : &q_gpu;

        const auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k.getImplementation());
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;

        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v.getImplementation());
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;

        bool needs_cast_input = (effective_q->getDataType() != Data_Type::FLOAT16);

        std::shared_ptr<Gpu_Tensor_Impl> q_fp16, k_fp16, v_fp16;
        const Gpu_Tensor_Impl *in_q = effective_q;
        const Gpu_Tensor_Impl *in_k = effective_k;
        const Gpu_Tensor_Impl *in_v = effective_v;

        if (needs_cast_input)
        {
            q_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT16);
            k_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT16);
            v_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT16);
            effective_q->to(Data_Type::FLOAT16, *q_fp16);
            effective_k->to(Data_Type::FLOAT16, *k_fp16);
            effective_v->to(Data_Type::FLOAT16, *v_fp16);
            in_q = q_fp16.get();
            in_k = k_fp16.get();
            in_v = v_fp16.get();
        }

        Data_Type out_dtype = needs_cast_input ? Data_Type::FLOAT32 : Data_Type::FLOAT16;
        ensureTensorTarget(output, target, q.getShape(), out_dtype);
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        std::shared_ptr<Gpu_Tensor_Impl> intermediate_out_fp16;
        Gpu_Tensor_Impl *target_out = &out_gpu;

        if (needs_cast_input)
        {
            intermediate_out_fp16 = std::make_shared<Gpu_Tensor_Impl>(q.getShape(), Data_Type::FLOAT16);
            target_out = intermediate_out_fp16.get();
        }
        else
        {
            out_gpu.setDataType(Data_Type::FLOAT16);
            out_gpu.reshape(q.getShape());
        }

        uint32_t total_elements_per_head = seq_len * head_dim;
        uint32_t total_heads = (total_elements_per_head > 0) ? static_cast<uint32_t>(q.getTotalElements() / total_elements_per_head) : num_heads;

        std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
        Gpu_Tensor_Impl *target_l = nullptr;
        if (l_stats)
        {
            ensureTensorTarget(*l_stats, target, Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
            auto *l_gpu = dynamic_cast<Gpu_Tensor_Impl *>(l_stats->getImplementation().get());
            if (!l_gpu)
            {
                fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
                target_l = fallback_l.get();
            }
            else
            {
                size_t elem_size = sizeof(float);
                if (!l_gpu->getStorageBuffer() || l_gpu->getStorageBuffer()->getSizeBytes() < total_heads * seq_len * elem_size)
                {
                    l_gpu->getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_heads * seq_len, Data_Type::FLOAT32);
                }
                target_l = l_gpu;
            }
        }
        else
        {
            fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
            target_l = fallback_l.get();
        }

        struct FlashAttention_Constants
        {
            uint32_t batch_size;  // B * H
            uint32_t seq_len;     // N
            uint32_t head_dim;    // D
            uint32_t is_causal;   // 1 or 0
            float scale;
        } pcs{total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale};

        uint32_t num_q_tiles = (seq_len + 15) / 16;
        effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_FP16,
                                 {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(), target_out->getStorageBuffer(), target_l->getStorageBuffer()},
                                 pcs, 1, num_q_tiles, total_heads);

        if (needs_cast_input)
        {
            out_gpu.setDataType(Data_Type::FLOAT32);
            out_gpu.reshape(q.getShape());
            intermediate_out_fp16->to(Data_Type::FLOAT32, out_gpu);
        }
    }
}

void flashAttentionBackward(const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &o, const Tensor &do_grad,
                            Tensor &dq, Tensor &dk, Tensor &dv,
                            uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                            bool is_causal, float scale, const Tensor *l_stats)
{
    auto q_impl = q.getImplementation();
    auto k_impl = k.getImplementation();
    auto v_impl = v.getImplementation();
    auto o_impl = o.getImplementation();
    auto do_impl = do_grad.getImplementation();
    if (!q_impl || !k_impl || !v_impl || !o_impl || !do_impl) return;

    Execution_Target target = q.getExecutionTarget();
    if (scale <= 0.0f)
    {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    if (target == Execution_Target::CPU)
    {
        ensureTensorTarget(dq, target, q.getShape(), Data_Type::FLOAT32);
        ensureTensorTarget(dk, target, k.getShape(), Data_Type::FLOAT32);
        ensureTensorTarget(dv, target, v.getShape(), Data_Type::FLOAT32);

        const auto &q_data = q.getData();
        const auto &k_data = k.getData();
        const auto &v_data = v.getData();
        const auto &o_data = o.getData();
        const auto &do_data = do_grad.getData();

        auto &dq_cpu = Cpu_Tensor_Impl::castToCpu(*dq.getImplementation());
        auto &dk_cpu = Cpu_Tensor_Impl::castToCpu(*dk.getImplementation());
        auto &dv_cpu = Cpu_Tensor_Impl::castToCpu(*dv.getImplementation());

        if (!dq_cpu.getStorageBuffer() || dq_cpu.getStorageBuffer()->size() != q.getTotalElements())
            dq_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(q.getTotalElements(), 0.0f);
        else
            std::fill(dq_cpu.getStorageBuffer()->begin(), dq_cpu.getStorageBuffer()->end(), 0.0f);

        if (!dk_cpu.getStorageBuffer() || dk_cpu.getStorageBuffer()->size() != k.getTotalElements())
            dk_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(k.getTotalElements(), 0.0f);
        else
            std::fill(dk_cpu.getStorageBuffer()->begin(), dk_cpu.getStorageBuffer()->end(), 0.0f);

        if (!dv_cpu.getStorageBuffer() || dv_cpu.getStorageBuffer()->size() != v.getTotalElements())
            dv_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(v.getTotalElements(), 0.0f);
        else
            std::fill(dv_cpu.getStorageBuffer()->begin(), dv_cpu.getStorageBuffer()->end(), 0.0f);

        size_t total_elements_per_head = static_cast<size_t>(seq_len) * head_dim;
        size_t total_heads = (total_elements_per_head > 0) ? (q.getTotalElements() / total_elements_per_head) : num_heads;

        std::vector<float> l_data_cache;
        if (l_stats)
        {
            l_data_cache = l_stats->getData();
        }

        for (size_t h = 0; h < total_heads; ++h)
        {
            size_t head_offset = h * total_elements_per_head;

            for (size_t i = 0; i < seq_len; ++i)
            {
                std::vector<float> p(seq_len, 0.0f);
                size_t max_j = is_causal ? (i + 1) : seq_len;

                if (l_stats != nullptr && !l_data_cache.empty())
                {
                    float l_i = l_data_cache[h * seq_len + i];
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            dot += q_data[head_offset + i * head_dim + d] * k_data[head_offset + j * head_dim + d];
                        }
                        p[j] = std::exp(dot * scale - l_i);
                    }
                }
                else
                {
                    float max_score = -1e20f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            dot += q_data[head_offset + i * head_dim + d] * k_data[head_offset + j * head_dim + d];
                        }
                        p[j] = dot * scale;
                        if (p[j] > max_score)
                        {
                            max_score = p[j];
                        }
                    }

                    float sum_exp = 0.0f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        p[j] = std::exp(p[j] - max_score);
                        sum_exp += p[j];
                    }
                    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        p[j] *= inv_sum;
                    }
                }

                float di = 0.0f;
                for (size_t d = 0; d < head_dim; ++d)
                {
                    di += do_data[head_offset + i * head_dim + d] * o_data[head_offset + i * head_dim + d];
                }

                for (size_t j = 0; j < max_j; ++j)
                {
                    float dp = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dp += do_data[head_offset + i * head_dim + d] * v_data[head_offset + j * head_dim + d];
                    }

                    float ds = p[j] * (dp - di) * scale;

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        (*dq_cpu.getStorageBuffer())[head_offset + i * head_dim + d] += ds * k_data[head_offset + j * head_dim + d];
                    }

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        (*dk_cpu.getStorageBuffer())[head_offset + j * head_dim + d] += ds * q_data[head_offset + i * head_dim + d];
                    }

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        (*dv_cpu.getStorageBuffer())[head_offset + j * head_dim + d] += p[j] * do_data[head_offset + i * head_dim + d];
                    }
                }
            }
        }
    }
    else
    {
        const auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q.getImplementation());
        auto contig_q = q_gpu.ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : &q_gpu;

        const auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k.getImplementation());
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;

        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v.getImplementation());
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;

        const auto &o_gpu = Gpu_Tensor_Impl::castToGpu(*o.getImplementation());
        auto contig_o = o_gpu.ensureContiguousSelf();
        const auto *effective_o = contig_o ? contig_o.get() : &o_gpu;

        const auto &do_gpu = Gpu_Tensor_Impl::castToGpu(*do_grad.getImplementation());
        auto contig_do = do_gpu.ensureContiguousSelf();
        const auto *effective_do = contig_do ? contig_do.get() : &do_gpu;

        bool needs_cast_input = (effective_q->getDataType() != Data_Type::FLOAT16);

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

        Data_Type target_dtype = needs_cast_input ? Data_Type::FLOAT32 : Data_Type::FLOAT16;
        ensureTensorTarget(dq, target, q.getShape(), target_dtype);
        ensureTensorTarget(dk, target, k.getShape(), target_dtype);
        ensureTensorTarget(dv, target, v.getShape(), target_dtype);

        auto &dq_gpu = Gpu_Tensor_Impl::castToGpu(*dq.getImplementation());
        auto &dk_gpu = Gpu_Tensor_Impl::castToGpu(*dk.getImplementation());
        auto &dv_gpu = Gpu_Tensor_Impl::castToGpu(*dv.getImplementation());

        std::shared_ptr<Gpu_Tensor_Impl> dq_fp16, dk_fp16, dv_fp16;
        Gpu_Tensor_Impl *target_dq = &dq_gpu;
        Gpu_Tensor_Impl *target_dk = &dk_gpu;
        Gpu_Tensor_Impl *target_dv = &dv_gpu;

        if (needs_cast_input)
        {
            dq_fp16 = std::make_shared<Gpu_Tensor_Impl>(q.getShape(), Data_Type::FLOAT16);
            dk_fp16 = std::make_shared<Gpu_Tensor_Impl>(k.getShape(), Data_Type::FLOAT16);
            dv_fp16 = std::make_shared<Gpu_Tensor_Impl>(v.getShape(), Data_Type::FLOAT16);
            target_dq = dq_fp16.get();
            target_dk = dk_fp16.get();
            target_dv = dv_fp16.get();
        }
        else
        {
            dq_gpu.setDataType(Data_Type::FLOAT16);
            dq_gpu.reshape(q.getShape());
            dk_gpu.setDataType(Data_Type::FLOAT16);
            dk_gpu.reshape(k.getShape());
            dv_gpu.setDataType(Data_Type::FLOAT16);
            dv_gpu.reshape(v.getShape());
        }

        uint32_t total_elements_per_head = seq_len * head_dim;
        uint32_t total_heads = (total_elements_per_head > 0) ? static_cast<uint32_t>(q.getTotalElements() / total_elements_per_head) : num_heads;

        std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
        const Gpu_Tensor_Impl *target_l = nullptr;
        if (l_stats)
        {
            target_l = dynamic_cast<const Gpu_Tensor_Impl *>(l_stats->getImplementation().get());
            if (!target_l)
            {
                fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, Data_Type::FLOAT32);
                target_l = fallback_l.get();
            }
        }
        else
        {
            fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, Data_Type::FLOAT32);
            target_l = fallback_l.get();
        }

        struct FlashAttention_Backward_Constants
        {
            uint32_t batch_size;  
            uint32_t seq_len;
            uint32_t head_dim;
            uint32_t is_causal;
            float scale;
            uint32_t pass_type;
        };

        uint32_t num_q_tiles = (seq_len + 15) / 16;
        uint32_t num_k_tiles = (seq_len + 15) / 16;

        FlashAttention_Backward_Constants pcs_pass0{
            total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 0u
        };
        effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16,
                    {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                     in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                     target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(), target_l->getStorageBuffer()},
                    pcs_pass0, 1, num_q_tiles, total_heads);

        FlashAttention_Backward_Constants pcs_pass1{
            total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 1u
        };
        effective_q->pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16,
                    {in_q->getStorageBuffer(), in_k->getStorageBuffer(), in_v->getStorageBuffer(),
                     in_o->getStorageBuffer(), in_do->getStorageBuffer(),
                     target_dq->getStorageBuffer(), target_dk->getStorageBuffer(), target_dv->getStorageBuffer(), target_l->getStorageBuffer()},
                    pcs_pass1, 1, num_k_tiles, total_heads);

        if (needs_cast_input)
        {
            dq_gpu.setDataType(Data_Type::FLOAT32);
            dq_gpu.reshape(q.getShape());
            dq_fp16->to(Data_Type::FLOAT32, dq_gpu);

            dk_gpu.setDataType(Data_Type::FLOAT32);
            dk_gpu.reshape(k.getShape());
            dk_fp16->to(Data_Type::FLOAT32, dk_gpu);

            dv_gpu.setDataType(Data_Type::FLOAT32);
            dv_gpu.reshape(v.getShape());
            dv_fp16->to(Data_Type::FLOAT32, dv_gpu);
        }
    }
}

void singleTokenAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                                 size_t num_heads, size_t head_dim, size_t total_seq_len)
{
    auto q_impl = q.getImplementation();
    auto k_impl = k.getImplementation();
    auto v_impl = v.getImplementation();
    if (!q_impl || !k_impl || !v_impl) return;

    Execution_Target target = q.getExecutionTarget();
    ensureTensorTarget(output, target, Shape{ 1, num_heads, 1, head_dim }, q.getDataType());

    if (target == Execution_Target::CPU)
    {
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        if (!out_cpu.getStorageBuffer() || out_cpu.getStorageBuffer()->size() != num_heads * head_dim)
        {
            out_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(num_heads * head_dim, 0.0f);
        }

        const auto &q_data = q.getData();
        const auto &k_data = k.getData();
        const auto &v_data = v.getData();
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::vector<float> scores(total_seq_len);

        for (size_t h = 0; h < num_heads; ++h)
        {
            const float *q_h = q_data.data() + h * head_dim;
            const float *k_h = k_data.data() + h * total_seq_len * head_dim;
            const float *v_h = v_data.data() + h * total_seq_len * head_dim;
            float *out_h = out_cpu.getStorageBuffer()->data() + h * head_dim;

            float max_score = -1e20f;
            for (size_t t = 0; t < total_seq_len; ++t)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; ++i)
                {
                    dot += q_h[i] * k_h[t * head_dim + i];
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_seq_len; ++t)
            {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
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
    }
    else
    {
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        const auto &q_gpu = Gpu_Tensor_Impl::castToGpu(*q.getImplementation());
        const auto &k_gpu = Gpu_Tensor_Impl::castToGpu(*k.getImplementation());
        const auto &v_gpu = Gpu_Tensor_Impl::castToGpu(*v.getImplementation());
        size_t elem_size = getDataTypeSize(q.getDataType());

        if (!out_gpu.getStorageBuffer() || out_gpu.getStorageBuffer()->getSizeBytes() < num_heads * head_dim * elem_size)
        {
            out_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            out_gpu.getStorageBuffer()->allocateMemory(num_heads * head_dim, q.getDataType());
        }

        struct Attention_Decode_Constants
        {
            uint32_t num_heads;
            uint32_t head_dim;
            uint32_t total_seq_len;
            float scale;
        } pcs{ static_cast<uint32_t>(num_heads), static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(total_seq_len), 1.0f / std::sqrt(static_cast<float>(head_dim)) };

        bool is_fp16 = (q.getDataType() == Data_Type::FLOAT16 || k_gpu.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ATTENTION_DECODE_FP16 : Compute_Pipeline::ATTENTION_DECODE;
        auto buf_q = is_fp16 ? q_gpu.getEffectiveFp16Storage() : q_gpu.getStorageBuffer();
        auto buf_k = is_fp16 ? k_gpu.getEffectiveFp16Storage() : k_gpu.getStorageBuffer();
        auto buf_v = is_fp16 ? v_gpu.getEffectiveFp16Storage() : v_gpu.getStorageBuffer();

        q_gpu.pushToGraph(pipe,
                          { buf_q, buf_k, buf_v, out_gpu.getStorageBuffer() },
                          pcs,
                          static_cast<uint32_t>(num_heads), 1, 1);
        out_gpu.clearHostCache();
    }
}

} // namespace ops::llm

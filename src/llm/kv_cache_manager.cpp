#include "llm/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"

namespace
{
    size_t validateKvDim(size_t val, const char *name, Data_Type dtype)
    {
        if (val == 0)
        {
            throw std::invalid_argument(std::string("KV_Cache_Manager: ") + name + " must be > 0");
        }
        if (dtype == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("KV_Cache_Manager: TERNARY_INT2 data_type is not supported for KV cache");
        }
        return val;
    }
}

KV_Cache_Manager::KV_Cache_Manager(size_t _batch_size,
                                   size_t _num_heads,
                                   size_t _max_seq_len,
                                   size_t _head_dim,
                                   Execution_Target _execution_target,
                                   Data_Type _data_type,
                                   KV_Cache_Mode _mode,
                                   bool _auto_grow)
    : batch_size(validateKvDim(_batch_size, "batch_size", _data_type)),
      num_heads(validateKvDim(_num_heads, "num_heads", _data_type)),
      max_seq_len(validateKvDim(_max_seq_len, "max_seq_len", _data_type)),
      head_dim(validateKvDim(_head_dim, "head_dim", _data_type)),
      execution_target(_execution_target),
      data_type(_data_type),
      mode(_mode),
      auto_grow(_auto_grow),
      current_seq_len(0),
      k_cache(Shape{batch_size, num_heads, max_seq_len, head_dim}, _data_type, _execution_target),
      v_cache(Shape{batch_size, num_heads, max_seq_len, head_dim}, _data_type, _execution_target),
      k_shift_scratch(_execution_target),
      v_shift_scratch(_execution_target),
      k_cast_scratch(_execution_target),
      v_cast_scratch(_execution_target)
{
}

void KV_Cache_Manager::copySeqRange(const Tensor &src, Tensor &dst, size_t src_stride_seq, size_t dst_stride_seq,
                                    size_t src_offset_seq, size_t dst_offset_seq, size_t copy_seq_len)
{
    if (copy_seq_len == 0) return;

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        if (is_fp16 && !engine.getContext().isFloat16Enabled())
        {
            Logger::logMessage("KV_Cache_Manager::copySeqRange: FP16 disabled on Vulkan context; falling back to tensor slice update",
                               Log_Level::LOG_WARNING,
                               true,
                               1,
                               Log_Feature::LLM_INFERENCE);
            Tensor src_view = (src.getRank() == 4)
                ? src
                : src.view(Shape{batch_size, num_heads, src_stride_seq, head_dim});
            Tensor sliced = src_view.slice(2, src_offset_seq, copy_seq_len).contiguous();
            dst.updateSlice(2, dst_offset_seq, sliced);
            return;
        }

        const auto &src_gpu = Gpu_Tensor_Impl::castToGpu(*src.getImplementation());
        auto contig_src = src_gpu.ensureContiguousSelf();
        const auto *effective_src = contig_src ? contig_src.get() : &src_gpu;
        auto &dst_gpu = Gpu_Tensor_Impl::castToGpu(*dst.getImplementation());

        uint32_t batch_heads = static_cast<uint32_t>(batch_size * num_heads);
        uint32_t total_elems = batch_heads * static_cast<uint32_t>(copy_seq_len) * static_cast<uint32_t>(head_dim);

        struct Kv_Cache_Update_Constants
        {
            uint32_t batch_heads;
            uint32_t copy_seq_len;
            uint32_t src_stride_seq;
            uint32_t dst_stride_seq;
            uint32_t head_dim;
            uint32_t src_offset_seq;
            uint32_t dst_offset_seq;
        } pcs{
            batch_heads,
            static_cast<uint32_t>(copy_seq_len),
            static_cast<uint32_t>(src_stride_seq),
            static_cast<uint32_t>(dst_stride_seq),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(src_offset_seq),
            static_cast<uint32_t>(dst_offset_seq)
        };

        Compute_Pipeline pipe = is_fp16
            ? Compute_Pipeline::KV_CACHE_UPDATE_FP16
            : Compute_Pipeline::KV_CACHE_UPDATE;

        effective_src->pushToGraph(pipe,
                                   {effective_src->getStorageBuffer(), dst_gpu.getStorageBuffer()},
                                   pcs,
                                   (total_elems + 255) / 256, 1, 1);
        dst_gpu.clearHostCache();
        dst_gpu.invalidateFp16Cache();
    }
    else
    {
        Tensor src_view = (src.getRank() == 4)
            ? src
            : src.view(Shape{batch_size, num_heads, src_stride_seq, head_dim});
        Tensor sliced = src_view.slice(2, src_offset_seq, copy_seq_len).contiguous();
        dst.updateSlice(2, dst_offset_seq, sliced);
    }
}

void KV_Cache_Manager::append(const Tensor &k_new, const Tensor &v_new)
{
    appendImpl(k_new, v_new, current_seq_len, true);
}

void KV_Cache_Manager::appendAt(const Tensor &k_new, const Tensor &v_new, size_t pos)
{
    appendImpl(k_new, v_new, pos, false);
}

void KV_Cache_Manager::appendImpl(const Tensor &k_new, const Tensor &v_new, size_t pos, bool sequential_append)
{
    if (!k_new.getImplementation() || !v_new.getImplementation())
    {
        throw std::invalid_argument("KV_Cache_Manager::appendAt: Null tensor implementation");
    }
    if (k_new.getExecutionTarget() != execution_target || v_new.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("KV_Cache_Manager::appendAt: Execution target mismatch");
    }
    if (k_new.getDataType() == Data_Type::TERNARY_INT2 || v_new.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("KV_Cache_Manager::appendAt: TERNARY_INT2 is not supported");
    }
    if (k_new.getShape() != v_new.getShape())
    {
        throw std::invalid_argument("KV_Cache_Manager::appendAt: k_new and v_new shapes must match");
    }
    if (!(mode == KV_Cache_Mode::STATIC && auto_grow) && pos > max_seq_len)
    {
        throw std::out_of_range("KV_Cache_Manager::appendAt: pos exceeds max_seq_len");
    }
    if (k_new.isEmpty() || k_new.getTotalElements() == 0)
    {
        return;
    }

    const Shape &k_shape = k_new.getShape();
    const size_t rank = k_shape.getRank();
    const size_t per_tok = batch_size * num_heads * head_dim;
    size_t new_seq_len = 0;
    bool needs_bshd_permute = false;
    if (rank == 4)
    {
        if (k_shape[0] != batch_size || k_shape[1] != num_heads || k_shape[3] != head_dim)
        {
            throw std::invalid_argument("KV_Cache_Manager::appendAt: 4D shape mismatch with KV cache configuration");
        }
        new_seq_len = k_shape[2];
    }
    else if (rank == 3)
    {
        const bool layout_bshd = (k_shape[0] == batch_size && k_shape[2] == num_heads * head_dim);
        const bool layout_bhsd = (k_shape[0] == batch_size * num_heads && k_shape[2] == head_dim);
        if (!layout_bshd && !layout_bhsd)
        {
            throw std::invalid_argument("KV_Cache_Manager::appendAt: 3D shape mismatch with KV cache configuration");
        }
        needs_bshd_permute = layout_bshd && (num_heads > 1);
        new_seq_len = k_shape[1];
    }
    else
    {
        if (per_tok == 0 || (k_new.getTotalElements() % per_tok) != 0)
        {
            throw std::invalid_argument("KV_Cache_Manager::appendAt: Tensor element count not divisible by batch_size * num_heads * head_dim");
        }
        new_seq_len = k_new.getTotalElements() / per_tok;
    }

    if (new_seq_len == 0) return;

    const Tensor *eff_k = &k_new;
    const Tensor *eff_v = &v_new;

    if (k_new.getDataType() != data_type)
    {
        k_new.to(data_type, k_cast_scratch);
        eff_k = &k_cast_scratch;
    }
    if (v_new.getDataType() != data_type)
    {
        v_new.to(data_type, v_cast_scratch);
        eff_v = &v_cast_scratch;
    }

    Tensor k_layout_tmp(execution_target);
    Tensor v_layout_tmp(execution_target);
    if (needs_bshd_permute)
    {
        eff_k->view(Shape{batch_size, new_seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous(k_layout_tmp);
        eff_v->view(Shape{batch_size, new_seq_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous(v_layout_tmp);
        eff_k = &k_layout_tmp;
        eff_v = &v_layout_tmp;
    }

    const size_t orig_pos = pos;
    if (pos + new_seq_len > max_seq_len)
    {
        if (mode == KV_Cache_Mode::STATIC && auto_grow)
        {
            growTo(std::max(max_seq_len * 2, pos + new_seq_len));
        }
        else if (mode == KV_Cache_Mode::STATIC)
        {
            throw std::runtime_error("KV_Cache_Manager: max_seq_len exceeded in STATIC mode");
        }
        else
        {
            Logger::logMessage(Input_Format{"KV_Cache_Manager: Ring-buffer eviction triggered (pos={}, new_seq_len={}, max_seq_len={})",
                                            pos, new_seq_len, max_seq_len},
                               Log_Level::LOG_DEBUG,
                               false,
                               4,
                               Log_Feature::LLM_INFERENCE);
            if (new_seq_len >= max_seq_len)
            {
                size_t src_offset = new_seq_len - max_seq_len;
                copySeqRange(*eff_k, k_cache, new_seq_len, max_seq_len, src_offset, 0, max_seq_len);
                copySeqRange(*eff_v, v_cache, new_seq_len, max_seq_len, src_offset, 0, max_seq_len);
                current_seq_len = max_seq_len;
                total_tokens_seen = sequential_append
                    ? (total_tokens_seen + new_seq_len)
                    : std::max(total_tokens_seen, orig_pos + new_seq_len);
                return;
            }

            size_t overflow = (pos + new_seq_len) - max_seq_len;
            size_t keep_len = (overflow < max_seq_len) ? (max_seq_len - overflow) : 0;
            if (keep_len > 0)
            {
                Shape cache_shape{batch_size, num_heads, max_seq_len, head_dim};
                if (k_shift_scratch.getExecutionTarget() != execution_target ||
                    k_shift_scratch.getDataType() != data_type ||
                    k_shift_scratch.getShape() != cache_shape)
                {
                    k_shift_scratch = Tensor(cache_shape, data_type, execution_target);
                }
                if (v_shift_scratch.getExecutionTarget() != execution_target ||
                    v_shift_scratch.getDataType() != data_type ||
                    v_shift_scratch.getShape() != cache_shape)
                {
                    v_shift_scratch = Tensor(cache_shape, data_type, execution_target);
                }

                copySeqRange(k_cache, k_shift_scratch, max_seq_len, max_seq_len, overflow, 0, keep_len);
                copySeqRange(k_shift_scratch, k_cache, max_seq_len, max_seq_len, 0, 0, keep_len);
                copySeqRange(v_cache, v_shift_scratch, max_seq_len, max_seq_len, overflow, 0, keep_len);
                copySeqRange(v_shift_scratch, v_cache, max_seq_len, max_seq_len, 0, 0, keep_len);
            }
            pos = max_seq_len - new_seq_len;
        }
    }

    copySeqRange(*eff_k, k_cache, new_seq_len, max_seq_len, 0, pos, new_seq_len);
    copySeqRange(*eff_v, v_cache, new_seq_len, max_seq_len, 0, pos, new_seq_len);

    current_seq_len = std::min(max_seq_len, std::max(current_seq_len, pos + new_seq_len));
    total_tokens_seen = sequential_append
        ? (total_tokens_seen + new_seq_len)
        : std::max(total_tokens_seen, orig_pos + new_seq_len);
}

Tensor KV_Cache_Manager::getK(size_t len) const
{
    size_t slice_len = (len > 0) ? std::min(len, current_seq_len) : current_seq_len;
    if (slice_len == 0)
    {
        return Tensor(Shape{batch_size, num_heads, 0, head_dim}, data_type, execution_target);
    }
    return k_cache.slice(2, 0, slice_len);
}

Tensor KV_Cache_Manager::getV(size_t len) const
{
    size_t slice_len = (len > 0) ? std::min(len, current_seq_len) : current_seq_len;
    if (slice_len == 0)
    {
        return Tensor(Shape{batch_size, num_heads, 0, head_dim}, data_type, execution_target);
    }
    return v_cache.slice(2, 0, slice_len);
}

void KV_Cache_Manager::reset() noexcept
{
    current_seq_len = 0;
    total_tokens_seen = 0;
}

void KV_Cache_Manager::truncate(size_t new_seq_len)
{
    if (new_seq_len > current_seq_len)
    {
        throw std::invalid_argument("KV_Cache_Manager::truncate: new_seq_len exceeds current_seq_len");
    }
    current_seq_len = new_seq_len;
    total_tokens_seen = new_seq_len;
}

void KV_Cache_Manager::growTo(size_t new_max_seq_len)
{
    if (new_max_seq_len <= max_seq_len)
    {
        return;
    }
    Tensor new_k(Shape{batch_size, num_heads, new_max_seq_len, head_dim}, data_type, execution_target);
    Tensor new_v(Shape{batch_size, num_heads, new_max_seq_len, head_dim}, data_type, execution_target);
    if (current_seq_len > 0)
    {
        copySeqRange(k_cache, new_k, max_seq_len, new_max_seq_len, 0, 0, current_seq_len);
        copySeqRange(v_cache, new_v, max_seq_len, new_max_seq_len, 0, 0, current_seq_len);
    }
    k_cache = std::move(new_k);
    v_cache = std::move(new_v);
    max_seq_len = new_max_seq_len;
}

std::unique_ptr<KV_Cache_Manager> KV_Cache_Manager::clone(size_t copy_len) const
{
    auto cloned = std::make_unique<KV_Cache_Manager>(batch_size, num_heads, max_seq_len, head_dim,
                                                     execution_target, data_type, mode, auto_grow);
    const size_t eff_len = (copy_len == 0 || copy_len > current_seq_len) ? current_seq_len : copy_len;
    if (eff_len > 0)
    {
        cloned->copySeqRange(k_cache, cloned->k_cache, max_seq_len, max_seq_len, 0, 0, eff_len);
        cloned->copySeqRange(v_cache, cloned->v_cache, max_seq_len, max_seq_len, 0, 0, eff_len);
    }
    cloned->current_seq_len = eff_len;
    cloned->total_tokens_seen = eff_len;
    return cloned;
}

void KV_Cache_Manager::setExecutionTarget(Execution_Target new_target)
{
    if (execution_target == new_target) return;
    execution_target = new_target;
    k_cache.setExecutionTarget(new_target);
    v_cache.setExecutionTarget(new_target);
    k_shift_scratch = Tensor(new_target);
    v_shift_scratch = Tensor(new_target);
    k_cast_scratch = Tensor(new_target);
    v_cast_scratch = Tensor(new_target);
}

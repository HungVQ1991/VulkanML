#include "llm/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "math/tensor.h"

KV_Cache_Manager::KV_Cache_Manager(size_t _batch_size,
                                   size_t _num_heads,
                                   size_t _max_seq_len,
                                   size_t _head_dim,
                                   Execution_Target _execution_target,
                                   Data_Type _data_type,
                                   KV_Cache_Mode _mode)
    : batch_size(_batch_size),
      num_heads(_num_heads),
      max_seq_len(_max_seq_len),
      head_dim(_head_dim),
      execution_target(_execution_target),
      data_type(_data_type),
      mode(_mode),
      current_seq_len(0),
      k_cache(Shape{_batch_size, _num_heads, _max_seq_len, _head_dim}, _data_type, _execution_target),
      v_cache(Shape{_batch_size, _num_heads, _max_seq_len, _head_dim}, _data_type, _execution_target)
{
}

void KV_Cache_Manager::append(const Tensor &k_new, const Tensor &v_new)
{
    appendAt(k_new, v_new, current_seq_len);
}

void KV_Cache_Manager::appendAt(const Tensor &k_new, const Tensor &v_new, size_t pos)
{
    size_t new_seq_len = 1;
    if (k_new.getShape().getRank() == 4)
    {
        new_seq_len = k_new.getShape()[2];
    }
    else if (k_new.getShape().getRank() == 3)
    {
        new_seq_len = k_new.getShape()[1];
    }
    else
    {
        size_t per_tok = batch_size * num_heads * head_dim;
        if (per_tok > 0 && k_new.getTotalElements() >= per_tok)
        {
            new_seq_len = k_new.getTotalElements() / per_tok;
        }
    }

    if (pos + new_seq_len > max_seq_len)
    {
        if (mode == KV_Cache_Mode::STATIC)
        {
            throw std::runtime_error("KV_Cache_Manager: max_seq_len exceeded in STATIC mode");
        }
        else
        {
            size_t overflow = (pos + new_seq_len) - max_seq_len;
            size_t keep_len = max_seq_len - overflow;
            if (keep_len > 0)
            {
                Tensor k_keep = k_cache.slice(2, overflow, keep_len).clone();
                Tensor v_keep = v_cache.slice(2, overflow, keep_len).clone();
                k_cache.updateSlice(2, 0, k_keep);
                v_cache.updateSlice(2, 0, v_keep);
            }
            pos = max_seq_len - new_seq_len;
        }
    }

    const Tensor *eff_k = &k_new;
    const Tensor *eff_v = &v_new;
    Tensor cast_k, cast_v;

    if (k_new.getDataType() != data_type)
    {
        k_new.to(data_type, cast_k);
        eff_k = &cast_k;
    }
    if (v_new.getDataType() != data_type)
    {
        v_new.to(data_type, cast_v);
        eff_v = &cast_v;
    }

    k_cache.updateSlice(2, pos, *eff_k);
    v_cache.updateSlice(2, pos, *eff_v);

    current_seq_len = std::min(max_seq_len, std::max(current_seq_len, pos + new_seq_len));
}

Tensor KV_Cache_Manager::getK(size_t len) const
{
    size_t slice_len = (len > 0) ? std::min(len, current_seq_len) : current_seq_len;
    if (slice_len == 0)
    {
        slice_len = 1;
    }
    return k_cache.slice(2, 0, slice_len);
}

Tensor KV_Cache_Manager::getV(size_t len) const
{
    size_t slice_len = (len > 0) ? std::min(len, current_seq_len) : current_seq_len;
    if (slice_len == 0)
    {
        slice_len = 1;
    }
    return v_cache.slice(2, 0, slice_len);
}

void KV_Cache_Manager::reset() noexcept
{
    current_seq_len = 0;
}

void KV_Cache_Manager::setExecutionTarget(Execution_Target new_target)
{
    execution_target = new_target;
    k_cache.setExecutionTarget(new_target);
    v_cache.setExecutionTarget(new_target);
}

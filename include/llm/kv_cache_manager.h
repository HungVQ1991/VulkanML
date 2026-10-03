#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include "math/tensor.h"
#include "engine/execution_engine.h"

enum class KV_Cache_Mode
{
    STATIC,
    RING_BUFFER
};

class KV_Cache_Manager
{
private:
    size_t batch_size = 1;
    size_t num_heads = 1;
    size_t max_seq_len = 512;
    size_t head_dim = 64;
    Execution_Target execution_target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT16;
    KV_Cache_Mode mode = KV_Cache_Mode::STATIC;

    size_t current_seq_len = 0;

    Tensor k_cache;
    Tensor v_cache;

public:
    KV_Cache_Manager(size_t _batch_size,
                     size_t _num_heads,
                     size_t _max_seq_len,
                     size_t _head_dim,
                     Execution_Target _execution_target = Execution_Target::CPU,
                     Data_Type _data_type = Data_Type::FLOAT16,
                     KV_Cache_Mode _mode = KV_Cache_Mode::STATIC)
        : batch_size(_batch_size),
          num_heads(_num_heads),
          max_seq_len(_max_seq_len),
          head_dim(_head_dim),
          execution_target(_execution_target),
          data_type(_data_type),
          mode(_mode),
          current_seq_len(0),
          k_cache(Shape{ _batch_size, _num_heads, _max_seq_len, _head_dim }, _data_type, _execution_target),
          v_cache(Shape{ _batch_size, _num_heads, _max_seq_len, _head_dim }, _data_type, _execution_target)
    {
    }

    void append(const Tensor &k_new, const Tensor &v_new)
    {
        appendAt(k_new, v_new, current_seq_len);
    }

    void appendAt(const Tensor &k_new, const Tensor &v_new, size_t pos)
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

    Tensor getK(size_t len = 0) const
    {
        size_t slice_len = (len > 0) ? std::min(len, current_seq_len) : current_seq_len;
        if (slice_len == 0)
        {
            slice_len = 1;
        }
        return k_cache.slice(2, 0, slice_len);
    }

    Tensor getV(size_t len = 0) const
    {
        size_t slice_len = (len > 0) ? std::min(len, current_seq_len) : current_seq_len;
        if (slice_len == 0)
        {
            slice_len = 1;
        }
        return v_cache.slice(2, 0, slice_len);
    }

    void reset() noexcept
    {
        current_seq_len = 0;
    }

    size_t getCurrentSeqLen() const noexcept { return current_seq_len; }
    size_t getMaxSeqLen() const noexcept { return max_seq_len; }
    size_t getBatchSize() const noexcept { return batch_size; }
    size_t getNumHeads() const noexcept { return num_heads; }
    size_t getHeadDim() const noexcept { return head_dim; }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    Data_Type getDataType() const noexcept { return data_type; }
    KV_Cache_Mode getMode() const noexcept { return mode; }

    void setExecutionTarget(Execution_Target new_target)
    {
        execution_target = new_target;
        k_cache.setExecutionTarget(new_target);
        v_cache.setExecutionTarget(new_target);
    }

    const Tensor &getRawKCache() const noexcept { return k_cache; }
    const Tensor &getRawVCache() const noexcept { return v_cache; }
};

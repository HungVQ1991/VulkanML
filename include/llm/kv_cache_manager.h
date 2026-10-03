#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "math/tensor.h"

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
                     KV_Cache_Mode _mode = KV_Cache_Mode::STATIC);

    void append(const Tensor &k_new, const Tensor &v_new);
    void appendAt(const Tensor &k_new, const Tensor &v_new, size_t pos);

    Tensor getK(size_t len = 0) const;
    Tensor getV(size_t len = 0) const;

    void reset() noexcept;

    size_t getCurrentSeqLen() const noexcept { return current_seq_len; }
    size_t getMaxSeqLen() const noexcept { return max_seq_len; }
    size_t getBatchSize() const noexcept { return batch_size; }
    size_t getNumHeads() const noexcept { return num_heads; }
    size_t getHeadDim() const noexcept { return head_dim; }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    Data_Type getDataType() const noexcept { return data_type; }
    KV_Cache_Mode getMode() const noexcept { return mode; }

    void setExecutionTarget(Execution_Target new_target);

    const Tensor &getRawKCache() const noexcept { return k_cache; }
    const Tensor &getRawVCache() const noexcept { return v_cache; }
};

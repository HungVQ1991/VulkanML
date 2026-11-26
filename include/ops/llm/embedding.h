#pragma once

#include <cstdint>
#include "math/tensor.h"

class Execution_Engine;

namespace ops::llm
{

struct Embedding_Config
{
    bool force_fp16 = false;
    bool accumulate = false;
    int32_t padding_idx = -1;
    Execution_Engine *engine = nullptr;
};

void embeddingForward(const Tensor &weights, const Tensor &indices, Tensor &output,
                      const Embedding_Config &config);

void embeddingForward(const Tensor &weights, const Tensor &indices, Tensor &output, bool force_fp16 = false);

void embeddingBackward(const Tensor &weights, const Tensor &indices,
                       const Tensor &output_gradient, Tensor &weight_gradient,
                       const Embedding_Config &config);

void embeddingBackward(const Tensor &weights, const Tensor &indices,
                       const Tensor &output_gradient, Tensor &weight_gradient,
                       bool accumulate = false);

}

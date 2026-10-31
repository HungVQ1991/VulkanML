#pragma once

#include <cstdint>
#include <vector>
#include "math/tensor.h"

namespace ops::llm
{

void embeddingForward(const Tensor &weights, const Tensor &indices, Tensor &output);

void embeddingBackward(const Tensor &weights, const Tensor &indices,
                       const Tensor &output_gradient, Tensor &weight_gradient);

} // namespace ops::llm

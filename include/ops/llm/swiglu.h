#pragma once

#include "math/tensor.h"

namespace ops::llm
{

void swigluForward(const Tensor &a, const Tensor &b, Tensor &output);

void swigluBackward(const Tensor &output_gradient, const Tensor &a, const Tensor &b,
                    Tensor &grad_a, Tensor &grad_b);

void fusedSwiGLUForward(const Tensor &input, Tensor &output);

void fusedSwiGLUBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient);

} // namespace ops::llm

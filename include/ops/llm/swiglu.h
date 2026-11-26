#pragma once

#include "math/tensor.h"

class Execution_Engine;

namespace ops::llm
{

struct SwiGLU_Config
{
    Execution_Engine *engine = nullptr;
};

void swigluForward(const Tensor &a, const Tensor &b, Tensor &output, const SwiGLU_Config &config);

void swigluForward(const Tensor &a, const Tensor &b, Tensor &output);

void swigluBackward(const Tensor &a, const Tensor &b, const Tensor &output_gradient,
                    Tensor &grad_a, Tensor &grad_b, const SwiGLU_Config &config = {});

void fusedSwiGLUForward(const Tensor &input, Tensor &output, const SwiGLU_Config &config);

void fusedSwiGLUForward(const Tensor &input, Tensor &output);

void fusedSwiGLUBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient,
                         const SwiGLU_Config &config);

void fusedSwiGLUBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient);

}

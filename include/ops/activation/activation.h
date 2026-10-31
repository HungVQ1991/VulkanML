#pragma once

#include <cstddef>
#include "math/tensor.h"

namespace ops::activation
{

void relu(const Tensor &input, Tensor &output);
void reluBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient);

void leakyRelu(const Tensor &input, Tensor &output, float negative_slope = 0.01f);
void leakyReluBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient, float negative_slope = 0.01f);

void gelu(const Tensor &input, Tensor &output);
void geluBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient);

void softmax(const Tensor &input, Tensor &output);
void softmaxBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient);

void sigmoid(const Tensor &input, Tensor &output);
void sigmoidBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient);

} // namespace ops::activation

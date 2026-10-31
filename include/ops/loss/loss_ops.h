#pragma once

#include <cstdint>
#include <vector>
#include "math/tensor.h"

namespace ops::loss
{

void mse(const Tensor &prediction, const Tensor &target, Tensor &output);
void mae(const Tensor &prediction, const Tensor &target, Tensor &output);
void bce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon = 1e-7f);
void cce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon = 1e-7f);
void huber(const Tensor &prediction, const Tensor &target, Tensor &output, float delta = 1.0f);
void asl(const Tensor &prediction, const Tensor &target, Tensor &output, float gamma_pos = 1.0f, float gamma_neg = 4.0f, float clip_margin = 0.05f, float epsilon = 1e-7f);
void aslGradient(const Tensor &prediction, const Tensor &target, Tensor &output, float gamma_pos = 1.0f, float gamma_neg = 4.0f, float clip_margin = 0.05f, float epsilon = 1e-7f);
void dice(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon = 1e-7f);
void diceGradient(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon = 1e-7f);

float fusedCrossEntropy(const Tensor &logits, const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens = 0);
float fusedCrossEntropy(const Tensor &logits, const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens = 0);

} // namespace ops::loss

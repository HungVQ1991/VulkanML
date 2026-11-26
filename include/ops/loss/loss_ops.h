#pragma once

#include <cstdint>
#include <vector>
#include "math/tensor.h"

class Execution_Engine;

namespace ops::loss
{

enum class Loss_Reduction : uint32_t
{
    SUM = 0,
    MEAN = 1
};

struct Cross_Entropy_Config
{
    uint32_t valid_tokens = 0;
    int32_t ignore_index = -100;
    float label_smoothing = 0.0f;
    bool compute_grad = true;
    bool readback_loss = true;
    bool defer_execution = false;
    Execution_Engine *engine = nullptr;
};

void mse(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine = nullptr);
void mseGradient(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine = nullptr);
void mae(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine = nullptr);
void maeGradient(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine = nullptr);
void bce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon = 1e-7f, Execution_Engine *engine = nullptr);
void bceGradient(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon = 1e-7f, Execution_Engine *engine = nullptr);
void cce(const Tensor &prediction, const Tensor &target, Tensor &output, float epsilon = 1e-7f, Execution_Engine *engine = nullptr);
void cceGradient(const Tensor &prediction, const Tensor &target, Tensor &output, Execution_Engine *engine = nullptr);
void huber(const Tensor &prediction, const Tensor &target, Tensor &output, float delta = 1.0f, Execution_Engine *engine = nullptr);
void huberGradient(const Tensor &prediction, const Tensor &target, Tensor &output, float delta = 1.0f, Execution_Engine *engine = nullptr);
void asl(const Tensor &prediction, const Tensor &target, Tensor &output, float gamma_pos = 1.0f, float gamma_neg = 4.0f, float clip_margin = 0.05f, float epsilon = 1e-7f, Execution_Engine *engine = nullptr);
void aslGradient(const Tensor &prediction, const Tensor &target, Tensor &output, float gamma_pos = 1.0f, float gamma_neg = 4.0f, float clip_margin = 0.05f, float epsilon = 1e-7f, Execution_Engine *engine = nullptr);
void dice(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon = 1e-7f, bool write_stats = false, Execution_Engine *engine = nullptr);
void diceGradient(const Tensor& prediction, const Tensor& target, Tensor& output, float epsilon = 1e-7f, const Tensor* stats_tensor = nullptr, Execution_Engine *engine = nullptr);

float fusedCrossEntropy(const Tensor &logits, const Tensor &targets, Tensor &d_logits, const Cross_Entropy_Config &config);
float fusedCrossEntropy(const Tensor &logits, const std::vector<int32_t> &targets, Tensor &d_logits, const Cross_Entropy_Config &config);
float fusedCrossEntropy(const Tensor &logits, const Tensor &targets, Tensor &d_logits, uint32_t valid_tokens = 0, int32_t ignore_index = -100, bool compute_grad = true, bool readback_loss = true);
float fusedCrossEntropy(const Tensor &logits, const std::vector<int32_t> &targets, Tensor &d_logits, uint32_t valid_tokens = 0, int32_t ignore_index = -100, bool compute_grad = true, bool readback_loss = true);

}

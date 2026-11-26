#pragma once

#include <cstddef>
#include "math/tensor.h"

class Execution_Engine;

namespace ops::loss
{

enum class Dice_Bce_Stats_Index : size_t
{
    LOSS = 0,
    INTERSECTION = 1,
    PRED_SUM = 2,
    TARGET_SUM = 3,
    COUNT = 4
};

struct Dice_Bce_Config
{
    float dice_weight = 1.0f;
    float bce_weight = 1.0f;
    float epsilon = 1e-7f;
    Execution_Engine *engine = nullptr;
};

void diceBce(const Tensor &prediction, const Tensor &target, Tensor &output,
             const Dice_Bce_Config &config);

void diceBce(const Tensor &prediction, const Tensor &target, Tensor &output,
             float dice_weight = 1.0f, float bce_weight = 1.0f, float epsilon = 1e-7f);

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     const Dice_Bce_Config &config, const Tensor *stats_tensor = nullptr);

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     const Dice_Bce_Config &config,
                     float intersection, float pred_sum, float target_sum);

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight = 1.0f, float bce_weight = 1.0f, float epsilon = 1e-7f,
                     const Tensor *stats_tensor = nullptr);

void diceBceGradient(const Tensor &prediction, const Tensor &target, Tensor &output,
                     float dice_weight, float bce_weight, float epsilon,
                     float intersection, float pred_sum, float target_sum);

float diceBceLossAndGradient(const Tensor &prediction, const Tensor &target, Tensor &gradient_out,
                             const Dice_Bce_Config &config);

float diceBceLossAndGradient(const Tensor &prediction, const Tensor &target, Tensor &gradient_out,
                             float dice_weight = 1.0f, float bce_weight = 1.0f, float epsilon = 1e-7f);

}

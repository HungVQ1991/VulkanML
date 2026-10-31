#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "icost_function.h"
#include "math/tensor.h"

class Dice_BCE_Cost : public ICost_Function
{
private:
    float dice_weight = 1.0f;
    float bce_weight = 1.0f;
    float epsilon = 1e-7f;
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;

public:
    explicit Dice_BCE_Cost(
        float _dice_weight = 1.0f,
        float _bce_weight = 1.0f,
        float _epsilon = 1e-7f,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~Dice_BCE_Cost() noexcept override = default;

    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    float computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const override;

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&dice_weight), sizeof(dice_weight));
        _output_file_stream.write(reinterpret_cast<const char *>(&bce_weight), sizeof(bce_weight));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&dice_weight), sizeof(dice_weight));
        _input_file_stream.read(reinterpret_cast<char *>(&bce_weight), sizeof(bce_weight));
        _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::DICE_BCE; }
    float getDiceWeight() const noexcept { return dice_weight; }
    float getBceWeight() const noexcept { return bce_weight; }
    float getEpsilon() const noexcept { return epsilon; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix) { loss_matrix = _matrix; }
    void setDiceWeight(float _dice_weight) noexcept { dice_weight = _dice_weight; }
    void setBceWeight(float _bce_weight) noexcept { bce_weight = _bce_weight; }
    void setEpsilon(float _epsilon) noexcept { epsilon = _epsilon; }
};

#pragma once

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "cost_function/icost_function.h"
#include "math/tensor.h"

class Dice_BCE_Cost : public ICost_Function
{
private:
    float dice_weight = 1.0f;
    float bce_weight = 1.0f;
    float epsilon = 1e-7f;
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;
    mutable Tensor grad_stats_matrix;

    static void validateParameters(float _dw, float _bw, float _eps)
    {
        if (!std::isfinite(_dw) || _dw < 0.0f)
        {
            throw std::invalid_argument("Dice_BCE_Cost: dice_weight must be finite and >= 0");
        }
        if (!std::isfinite(_bw) || _bw < 0.0f)
        {
            throw std::invalid_argument("Dice_BCE_Cost: bce_weight must be finite and >= 0");
        }
        if (!std::isfinite(_eps) || _eps <= 0.0f || _eps >= 0.5f)
        {
            throw std::invalid_argument("Dice_BCE_Cost: epsilon must be finite and in (0, 0.5)");
        }
    }

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

    void saveCheckpoint(std::ostream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&dice_weight), sizeof(dice_weight));
        _output_file_stream.write(reinterpret_cast<const char *>(&bce_weight), sizeof(bce_weight));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Dice_BCE_Cost::saveCheckpoint: Failed to write checkpoint");
        }
    }

    void loadCheckpoint(std::istream &_input_file_stream) override
    {
        float dw = dice_weight;
        float bw = bce_weight;
        float eps = epsilon;
        _input_file_stream.read(reinterpret_cast<char *>(&dw), sizeof(dw));
        _input_file_stream.read(reinterpret_cast<char *>(&bw), sizeof(bw));
        _input_file_stream.read(reinterpret_cast<char *>(&eps), sizeof(eps));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Dice_BCE_Cost::loadCheckpoint: Failed to read checkpoint");
        }
        validateParameters(dw, bw, eps);
        dice_weight = dw;
        bce_weight = bw;
        epsilon = eps;
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::DICE_BCE; }
    float getDiceWeight() const noexcept { return dice_weight; }
    float getBceWeight() const noexcept { return bce_weight; }
    float getEpsilon() const noexcept { return epsilon; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix)
    {
        if (_matrix.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Dice_BCE_Cost::setLossMatrix: TERNARY_INT2 is not supported");
        }
        loss_matrix = _matrix;
    }
    void setDiceWeight(float _dice_weight)
    {
        validateParameters(_dice_weight, bce_weight, epsilon);
        dice_weight = _dice_weight;
    }
    void setBceWeight(float _bce_weight)
    {
        validateParameters(dice_weight, _bce_weight, epsilon);
        bce_weight = _bce_weight;
    }
    void setEpsilon(float _epsilon)
    {
        validateParameters(dice_weight, bce_weight, _epsilon);
        epsilon = _epsilon;
    }
};

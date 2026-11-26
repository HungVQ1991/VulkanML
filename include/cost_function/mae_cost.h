#pragma once

#include <istream>
#include <ostream>
#include <stdexcept>

#include "cost_function/icost_function.h"
#include "math/tensor.h"

class Mae_Cost : public ICost_Function
{
private:
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;

public:
    explicit Mae_Cost(Execution_Target _execution_target = Execution_Target::CPU);

    ~Mae_Cost() noexcept override = default;
    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    float computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const override;

    void saveCheckpoint(std::ostream &_output_file_stream) const override
    {
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Mae_Cost::saveCheckpoint: Output stream is not ready");
        }
    }
    void loadCheckpoint(std::istream &_input_file_stream) override
    {
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Mae_Cost::loadCheckpoint: Input stream is not ready");
        }
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::MAE; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix)
    {
        if (_matrix.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Mae_Cost::setLossMatrix: TERNARY_INT2 is not supported");
        }
        loss_matrix = _matrix;
    }
};

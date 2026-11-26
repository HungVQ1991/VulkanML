#pragma once

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "cost_function/icost_function.h"
#include "math/tensor.h"

class Huber_Cost : public ICost_Function
{
private:
    float delta = 1.0f;
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;

    static void validateDelta(float _delta)
    {
        if (!std::isfinite(_delta) || _delta <= 0.0f)
        {
            throw std::invalid_argument("Huber_Cost: delta must be finite and > 0");
        }
    }

public:
    explicit Huber_Cost(float _delta = 1.0f, Execution_Target _execution_target = Execution_Target::CPU);

    ~Huber_Cost() noexcept override = default;
    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    float computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const override;

    void saveCheckpoint(std::ostream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&delta), sizeof(delta));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Huber_Cost::saveCheckpoint: Failed to write checkpoint");
        }
    }

    void loadCheckpoint(std::istream &_input_file_stream) override
    {
        float loaded_delta = delta;
        _input_file_stream.read(reinterpret_cast<char *>(&loaded_delta), sizeof(loaded_delta));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Huber_Cost::loadCheckpoint: Failed to read checkpoint");
        }
        validateDelta(loaded_delta);
        delta = loaded_delta;
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::HUBER; }
    float getDelta() const noexcept { return delta; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix)
    {
        if (_matrix.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Huber_Cost::setLossMatrix: TERNARY_INT2 is not supported");
        }
        loss_matrix = _matrix;
    }
    void setDelta(float _delta)
    {
        validateDelta(_delta);
        delta = _delta;
    }
};

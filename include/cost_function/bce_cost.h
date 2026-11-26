#pragma once

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "cost_function/icost_function.h"
#include "math/tensor.h"

class Bce_Cost : public ICost_Function
{
private:
    float epsilon = 1e-7f;
    mutable Tensor loss_matrix;
    mutable Tensor gradient_matrix;

    static void validateEpsilon(float _epsilon)
    {
        if (!std::isfinite(_epsilon) || _epsilon <= 0.0f || _epsilon >= 0.5f)
        {
            throw std::invalid_argument("Bce_Cost: epsilon must be finite and in (0, 0.5)");
        }
    }

public:
    explicit Bce_Cost(float _epsilon = 1e-7f, Execution_Target _execution_target = Execution_Target::CPU);

    ~Bce_Cost() noexcept override = default;
    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    float computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const override;

    void saveCheckpoint(std::ostream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Bce_Cost::saveCheckpoint: Failed to write checkpoint");
        }
    }

    void loadCheckpoint(std::istream &_input_file_stream) override
    {
        float loaded_epsilon = epsilon;
        _input_file_stream.read(reinterpret_cast<char *>(&loaded_epsilon), sizeof(loaded_epsilon));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Bce_Cost::loadCheckpoint: Failed to read checkpoint");
        }
        validateEpsilon(loaded_epsilon);
        epsilon = loaded_epsilon;
    }

    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::BCE; }
    float getEpsilon() const noexcept { return epsilon; }

    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix)
    {
        if (_matrix.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Bce_Cost::setLossMatrix: TERNARY_INT2 is not supported");
        }
        loss_matrix = _matrix;
    }
    void setEpsilon(float _epsilon)
    {
        validateEpsilon(_epsilon);
        epsilon = _epsilon;
    }
};

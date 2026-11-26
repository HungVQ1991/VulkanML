#pragma once

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "cost_function/icost_function.h"
#include "math/tensor.h"

class Cce_Cost : public ICost_Function
{
private:
    float epsilon = 1e-7f;
    mutable Tensor loss_matrix;
    mutable Tensor synced_target_matrix;
    mutable Tensor difference_matrix;
    mutable Tensor gradient_matrix;

    static void validateEpsilon(float _epsilon)
    {
        if (!std::isfinite(_epsilon) || _epsilon <= 0.0f || _epsilon >= 0.5f)
        {
            throw std::invalid_argument("Cce_Cost: epsilon must be finite and in (0, 0.5)");
        }
    }

public:
    explicit Cce_Cost(float _epsilon = 1e-7f, Execution_Target _execution_target = Execution_Target::CPU);

    ~Cce_Cost() noexcept override = default;
    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    float computeLossAndGradient(const Tensor &prediction_tensor, const Tensor &target_tensor, Tensor &gradient_out) const override;

    void saveCheckpoint(std::ostream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        if (!_output_file_stream.good())
        {
            throw std::runtime_error("Cce_Cost::saveCheckpoint: Failed to write checkpoint");
        }
    }

    void loadCheckpoint(std::istream &_input_file_stream) override
    {
        float loaded_epsilon = epsilon;
        _input_file_stream.read(reinterpret_cast<char *>(&loaded_epsilon), sizeof(loaded_epsilon));
        if (!_input_file_stream.good())
        {
            throw std::runtime_error("Cce_Cost::loadCheckpoint: Failed to read checkpoint");
        }
        validateEpsilon(loaded_epsilon);
        epsilon = loaded_epsilon;
    }

    const Tensor &getDifferenceMatrix() const noexcept { return difference_matrix; }
    const Tensor &getSyncedTargetMatrix() const noexcept { return synced_target_matrix; }
    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::CCE; }
    float getEpsilon() const noexcept { return epsilon; }

    void setSyncedTargetMatrix(const Tensor &_matrix) { synced_target_matrix = _matrix; }
    void setDifferenceMatrix(const Tensor &_matrix) { difference_matrix = _matrix; }
    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix)
    {
        if (_matrix.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Cce_Cost::setLossMatrix: TERNARY_INT2 is not supported");
        }
        loss_matrix = _matrix;
    }
    void setEpsilon(float _epsilon)
    {
        validateEpsilon(_epsilon);
        epsilon = _epsilon;
    }
};

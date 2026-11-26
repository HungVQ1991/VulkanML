#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "math/tensor.h"

enum class Cost_Type : uint32_t
{
    MSE = 0,
    MAE,
    BCE,
    CCE,
    HUBER,
    FUSED_CCE,
    ASL,
    DICE,
    DICE_BCE,
    COST_TYPE_END
};

class ICost_Function
{
public:
    virtual ~ICost_Function() noexcept = default;

    virtual float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const = 0;
    virtual Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const = 0;

    virtual float computeLossAndGradient(const Tensor& prediction_tensor, const Tensor& target_tensor, Tensor& gradient_out) const
    {
        gradient_out = computeGradient(prediction_tensor, target_tensor);
        return computeLoss(prediction_tensor, target_tensor);
    }

    virtual float computeLossAndGradient(const Tensor& , const std::vector<int32_t>& , Tensor& ) const
    {
        throw std::logic_error("ICost_Function::computeLossAndGradient: Integer targets are not supported by this cost function");
    }

    virtual bool isFused() const noexcept { return false; }

    virtual bool supportsIntegerTargets() const noexcept { return false; }

    virtual void saveCheckpoint(std::ostream &_output_stream) const
    {
        if (!_output_stream.good())
        {
            throw std::runtime_error("ICost_Function::saveCheckpoint: Output stream is not ready");
        }
    }
    virtual void loadCheckpoint(std::istream &_input_stream)
    {
        if (!_input_stream.good())
        {
            throw std::runtime_error("ICost_Function::loadCheckpoint: Input stream is not ready");
        }
    }

    virtual Cost_Type getType() const noexcept = 0;
};

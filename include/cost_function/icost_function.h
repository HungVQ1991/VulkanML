#pragma once

#include <cstdint>
#include <fstream>

#include "helper/logger.h"
#include "math/tensor.h"

enum class Cost_Type : uint32_t
{
    MSE = 0,
    MAE,
    BCE,
    CCE,
    HUBER,
    FUSED_CCE,
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

    virtual float computeLossAndGradient(const Tensor& prediction_tensor, const std::vector<int32_t>& target_indices, Tensor& gradient_out) const
    {   
        return 0.0f;
    }

    virtual bool isFused() const noexcept { return false; }

    virtual bool supportsIntegerTargets() const noexcept { return false; }

    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const {}
    virtual void loadCheckpoint(std::ifstream &_input_file_stream) {}

    virtual Cost_Type getType() const noexcept = 0;
};
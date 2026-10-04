#pragma once

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

class Mse_Cost : public ICost_Function
{
private:
    mutable Tensor loss_matrix;
    mutable Tensor synced_target_matrix;
    mutable Tensor difference_matrix;
    mutable Tensor gradient_matrix;

public:    explicit Mse_Cost(Execution_Target _execution_target = Execution_Target::CPU);


    ~Mse_Cost() noexcept override = default;    float computeLoss(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;
    Tensor computeGradient(const Tensor &_prediction_matrix, const Tensor &_target_matrix) const override;


    void saveCheckpoint(std::ofstream &_output_file_stream) const override {}
    void loadCheckpoint(std::ifstream &_input_file_stream) override {}

    const Tensor &getDifferenceMatrix() const noexcept { return difference_matrix; }
    const Tensor &getSyncedTargetMatrix() const noexcept { return synced_target_matrix; }
    const Tensor &getGradientMatrix() const noexcept { return gradient_matrix; }
    const Tensor &getLossMatrix() const noexcept { return loss_matrix; }
    Cost_Type getType() const noexcept override { return Cost_Type::MSE; }

    void setSyncedTargetMatrix(const Tensor &_matrix) { synced_target_matrix = _matrix; }
    void setDifferenceMatrix(const Tensor &_matrix) { difference_matrix = _matrix; }
    void setGradientMatrix(const Tensor &_matrix) { gradient_matrix = _matrix; }
    void setLossMatrix(const Tensor &_matrix) { loss_matrix = _matrix; }
};;
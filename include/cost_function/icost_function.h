#pragma once

#include <cstdint>
#include <fstream>

#include "helper/logger.h"
#include "math/matrix.h"

enum class Cost_Type : std::uint32_t
{
    MSE = 0,
    MAE,
    BCE,
    CCE,
    COST_TYPE_END
};

class ICost_Function
{
public:
    virtual ~ICost_Function() noexcept = default;

    [[nodiscard]] virtual float computeLoss(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const = 0;
    [[nodiscard]] virtual Matrix computeGradient(const Matrix &_prediction_matrix, const Matrix &_target_matrix) const = 0;
    [[nodiscard]] virtual Cost_Type getType() const noexcept = 0;

    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const {}
    virtual void loadCheckpoint(std::ifstream &_input_file_stream) {}
};
#pragma once

#include <cstdint>
#include <fstream>
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

    virtual float computeLoss(const Matrix &pred, const Matrix &target) const = 0;
    virtual Matrix computeGradient(const Matrix &pred, const Matrix &target) const = 0;
    virtual Cost_Type getType() const = 0;

    virtual void saveCheckpoint(std::ofstream &out_file) const {}
    virtual void loadCheckpoint(std::ifstream &in_file) {}
};
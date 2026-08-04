#pragma once

#include <fstream>
#include <stdexcept>
#include <vector>

#include "icost_function.h"
#include "helper/logger.h"
#include "math/matrix.h"

class MSE_Cost : public ICost_Function
{
public:
    MSE_Cost() = default;
    ~MSE_Cost() override = default;

    float computeLoss(const Matrix &pred, const Matrix &target) const override
    {
        if (pred.getRows() != target.getRows() || pred.getCols() != target.getCols())
        {
            Logger::logMessage("MSE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        Matrix diff = pred - target;
        Matrix sq_diff = diff.hadamardMul(diff);
        std::vector<float> data = sq_diff.getData();

        float sum = 0.0f;
        for (float val : data)
        {
            sum += val;
        }
        return data.empty() ? 0.0f : (sum / static_cast<float>(data.size()));
    }

    Matrix computeGradient(const Matrix &pred, const Matrix &target) const override
    {
        if (pred.getRows() != target.getRows() || pred.getCols() != target.getCols())
        {
            Logger::logMessage("MSE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        return (pred - target) * 2.0f;
    }

    Cost_Type getType() const override
    {
        return Cost_Type::MSE;
    }

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}
};
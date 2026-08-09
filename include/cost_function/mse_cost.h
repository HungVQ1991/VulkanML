#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"
#include "icost_function.h"
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
            Logger::logMessage("MSE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (pred.getTarget() != target.getTarget())
        {
            Logger::logMessage("MSE_Cost::computeLoss: Execution target mismatch between prediction and target matrix", LOG_WARNING);
        }

        if (pred.getRows() == 0 || pred.getCols() == 0)
        {
            Logger::logMessage("MSE_Cost::computeLoss: Empty input matrix encountered", LOG_WARNING);
            return 0.0f;
        }

        COST_LOG_DEBUG("MSE_Cost::computeLoss: rows=" + std::to_string(pred.getRows()) + ", cols=" + std::to_string(pred.getCols()));

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
            Logger::logMessage("MSE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (pred.getTarget() != target.getTarget())
        {
            Logger::logMessage("MSE_Cost::computeGradient: Execution target mismatch between prediction and target matrix", LOG_WARNING);
        }

        if (pred.getRows() == 0 || pred.getCols() == 0)
        {
            Logger::logMessage("MSE_Cost::computeGradient: Empty input matrix encountered", LOG_WARNING);
            return Matrix(0, 0, pred.getTarget());
        }

        COST_LOG_DEBUG("MSE_Cost::computeGradient: rows=" + std::to_string(pred.getRows()) + ", cols=" + std::to_string(pred.getCols()));

        std::size_t total_elements = pred.getRows() * pred.getCols();
        float factor = 2.0f / static_cast<float>(total_elements);
        return (pred - target) * factor;
    }

    Cost_Type getType() const override
    {
        return Cost_Type::MSE;
    }

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}
};
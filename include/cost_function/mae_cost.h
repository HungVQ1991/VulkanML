#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "icost_function.h"
#include "math/matrix.h"

class MAE_Cost : public ICost_Function
{
public:
    MAE_Cost() = default;
    ~MAE_Cost() override = default;

    float computeLoss(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("MAE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (pred_matrix.getTarget() != target_matrix.getTarget())
        {
            Logger::logMessage("MAE_Cost::computeLoss: Execution target mismatch between prediction and target matrix", LOG_WARNING);
        }

        if (pred_matrix.getRows() == 0 || pred_matrix.getCols() == 0)
        {
            Logger::logMessage("MAE_Cost::computeLoss: Empty input matrix encountered", LOG_WARNING);
            return 0.0f;
        }

        COST_LOG_DEBUG("MAE_Cost::computeLoss: rows=" + std::to_string(pred_matrix.getRows()) + ", cols=" + std::to_string(pred_matrix.getCols()));

        std::size_t total_elements = pred_matrix.getRows() * pred_matrix.getCols();
        Matrix loss_matrix = pred_matrix.maeLoss(target_matrix);
        return loss_matrix.getScalar() / static_cast<float>(total_elements);
    }

    Matrix computeGradient(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("MAE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (pred_matrix.getTarget() != target_matrix.getTarget())
        {
            Logger::logMessage("MAE_Cost::computeGradient: Execution target mismatch between prediction and target matrix", LOG_WARNING);
        }

        if (pred_matrix.getRows() == 0 || pred_matrix.getCols() == 0)
        {
            Logger::logMessage("MAE_Cost::computeGradient: Empty input matrix encountered", LOG_WARNING);
            return Matrix(0, 0, pred_matrix.getTarget());
        }

        COST_LOG_DEBUG("MAE_Cost::computeGradient: rows=" + std::to_string(pred_matrix.getRows()) + ", cols=" + std::to_string(pred_matrix.getCols()));

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();
        std::size_t total_elements = pred_data.size();
        std::vector<float> grad_data(total_elements);
        float inv_total = 1.0f / static_cast<float>(total_elements);

        for (std::size_t i = 0; i < total_elements; ++i)
        {
            float diff_val = pred_data[i] - target_data[i];
            if (diff_val > 0.0f)
            {
                grad_data[i] = inv_total;
            }
            else if (diff_val < 0.0f)
            {
                grad_data[i] = -inv_total;
            }
            else
            {
                grad_data[i] = 0.0f;
            }
        }

        return Matrix(pred_matrix.getRows(), pred_matrix.getCols(), std::move(grad_data), pred_matrix.getTarget());
    }

    Cost_Type getType() const override
    {
        return Cost_Type::MAE;
    }

    void saveCheckpoint(std::ofstream &out_file) const override {}
    void loadCheckpoint(std::ifstream &in_file) override {}
};
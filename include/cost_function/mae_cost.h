#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "icost_function.h"
#include "helper/logger.h"
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
            Logger::logMessage("MAE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();

        float sum_val = 0.0f;
        for (std::size_t i = 0; i < pred_data.size(); ++i)
        {
            sum_val += std::abs(pred_data[i] - target_data[i]);
        }

        return pred_data.empty() ? 0.0f : (sum_val / static_cast<float>(pred_data.size()));
    }

    Matrix computeGradient(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("MAE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();
        std::size_t total_elements = pred_data.size();
        std::vector<float> grad_data(total_elements);

        for (std::size_t i = 0; i < total_elements; ++i)
        {
            float diff_val = pred_data[i] - target_data[i];
            if (diff_val > 0.0f)
            {
                grad_data[i] = 1.0f;
            }
            else if (diff_val < 0.0f)
            {
                grad_data[i] = -1.0f;
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
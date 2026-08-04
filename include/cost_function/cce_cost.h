#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "icost_function.h"
#include "helper/logger.h"
#include "math/matrix.h"

class CCE_Cost : public ICost_Function
{
private:
    float epsilon_val = 1e-7f;

public:
    explicit CCE_Cost(float eps_param = 1e-7f)
        : epsilon_val(eps_param) {}

    ~CCE_Cost() override = default;

    float computeLoss(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("CCE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();

        float sum_val = 0.0f;
        for (std::size_t i = 0; i < pred_data.size(); ++i)
        {
            float p_val = std::clamp(pred_data[i], epsilon_val, 1.0f - epsilon_val);
            float y_val = target_data[i];
            sum_val += -y_val * std::log(p_val);
        }

        std::size_t batch_size = pred_matrix.getRows();
        return batch_size == 0 ? 0.0f : (sum_val / static_cast<float>(batch_size));
    }

    Matrix computeGradient(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("CCE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        return pred_matrix - target_matrix;
    }

    Cost_Type getType() const override
    {
        return Cost_Type::CCE;
    }

    void saveCheckpoint(std::ofstream &out_file) const override
    {
        out_file.write(reinterpret_cast<const char *>(&epsilon_val), sizeof(epsilon_val));
    }

    void loadCheckpoint(std::ifstream &in_file) override
    {
        in_file.read(reinterpret_cast<char *>(&epsilon_val), sizeof(epsilon_val));
    }
};
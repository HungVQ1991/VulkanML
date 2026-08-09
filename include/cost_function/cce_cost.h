#pragma once

#include <algorithm>
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
            Logger::logMessage("CCE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (pred_matrix.getTarget() != target_matrix.getTarget())
        {
            Logger::logMessage("CCE_Cost::computeLoss: Execution target mismatch between prediction and target matrix", LOG_WARNING);
        }

        std::size_t batch_size = pred_matrix.getRows();
        if (batch_size == 0 || pred_matrix.getCols() == 0)
        {
            Logger::logMessage("CCE_Cost::computeLoss: Empty input matrix encountered", LOG_WARNING);
            return 0.0f;
        }

        COST_LOG_DEBUG("CCE_Cost::computeLoss: rows=" + std::to_string(pred_matrix.getRows()) + ", cols=" + std::to_string(pred_matrix.getCols()));

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();

        float sum_val = 0.0f;
        for (std::size_t i = 0; i < pred_data.size(); ++i)
        {
            float p_val = std::clamp(pred_data[i], epsilon_val, 1.0f - epsilon_val);
            float y_val = target_data[i];
            sum_val += -y_val * std::log(p_val);
        }

        return sum_val / static_cast<float>(batch_size);
    }

    Matrix computeGradient(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("CCE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR, true);
            throw std::invalid_argument("Dimensions mismatch");
        }

        if (pred_matrix.getTarget() != target_matrix.getTarget())
        {
            Logger::logMessage("CCE_Cost::computeGradient: Execution target mismatch between prediction and target matrix", LOG_WARNING);
        }

        std::size_t batch_size = pred_matrix.getRows();
        if (batch_size == 0 || pred_matrix.getCols() == 0)
        {
            Logger::logMessage("CCE_Cost::computeGradient: Empty input matrix encountered", LOG_WARNING);
            return Matrix(0, 0, pred_matrix.getTarget());
        }

        COST_LOG_DEBUG("CCE_Cost::computeGradient: rows=" + std::to_string(pred_matrix.getRows()) + ", cols=" + std::to_string(pred_matrix.getCols()));

        float inv_batch = 1.0f / static_cast<float>(batch_size);
        return (pred_matrix - target_matrix) * inv_batch;
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
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
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

        std::size_t batch_size = pred_matrix.getRows();
        if (batch_size == 0 || pred_matrix.getCols() == 0)
        {
            Logger::logMessage("CCE_Cost::computeLoss: Empty input matrix encountered", LOG_WARNING);
            return 0.0f;
        }

        // Matrix loss_matrix = pred_matrix.cceLoss(target_matrix, epsilon_val);
        // float total_loss = loss_matrix.getScalar();

        return 0;
    }

    Matrix computeGradient(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        auto log_node_count = [](const std::string &checkpoint)
        {
            std::size_t count = Execution_Engine::getInstance().getCurrentGraph().getNodes().size();
             Logger::logMessage(std::format("[TRACE_CCE] {} -> Node count = {}", checkpoint, count), LOG_DEBUG);
        };

        // log_node_count("CCE_01_ENTRY");

        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            std::string err_msg = std::format(
                "CCE_Cost::computeGradient: Dimensions mismatch! Pred: ({}x{}), Target: ({}x{})",
                pred_matrix.getRows(), pred_matrix.getCols(),
                target_matrix.getRows(), target_matrix.getCols());
            Logger::logMessage(err_msg, LOG_ERROR, true);
            throw std::invalid_argument(err_msg);
        }

        std::size_t batch_size = pred_matrix.getRows();
        if (batch_size == 0 || pred_matrix.getCols() == 0)
        {
            Logger::logMessage("CCE_Cost::computeGradient: Empty input matrix encountered", LOG_WARNING);
            return Matrix(0, 0, pred_matrix.getTarget());
        }

        // log_node_count("CCE_02_BEFORE_COPY_TARGET");
        Matrix synced_target = target_matrix;
        // log_node_count("CCE_03_AFTER_COPY_TARGET");

        if (synced_target.getTarget() != pred_matrix.getTarget())
        {
            // log_node_count("CCE_04_BEFORE_SET_TARGET");
            synced_target.setExecutionTarget(pred_matrix.getTarget());
            // log_node_count("CCE_05_AFTER_SET_TARGET");
        }

        float inv_batch = 1.0f / static_cast<float>(batch_size);

        // log_node_count("CCE_08_BEFORE_SUB_MUL");
        Matrix grad_matrix = (pred_matrix - synced_target) * inv_batch;
        // log_node_count("CCE_09_AFTER_SUB_MUL");

        // COST_LOG_DEBUG(std::format("CCE_Cost::computeGradient: shape=({}x{})", grad_matrix.getRows(), grad_matrix.getCols()));

        return grad_matrix;
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
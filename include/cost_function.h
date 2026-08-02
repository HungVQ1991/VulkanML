#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "math/matrix.h"
#include "math/logger.h"

class ICostFunction
{
public:
    virtual ~ICostFunction() = default;

    virtual float computeLoss(const Matrix &pred, const Matrix &target) const = 0;
    virtual Matrix computeGradient(const Matrix &pred, const Matrix &target) const = 0;
};

class MSE_Cost : public ICostFunction
{
public:
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
};

class MAE_Cost : public ICostFunction
{
public:
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
};

class BCE_Cost : public ICostFunction
{
private:
    float epsilon_val = 1e-7f;

public:
    explicit BCE_Cost(float eps_param = 1e-7f)
        : epsilon_val(eps_param) {}

    float computeLoss(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("BCE_Cost::computeLoss: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();

        float sum_val = 0.0f;
        for (std::size_t i = 0; i < pred_data.size(); ++i)
        {
            float p_val = std::clamp(pred_data[i], epsilon_val, 1.0f - epsilon_val);
            float y_val = target_data[i];
            sum_val += -(y_val * std::log(p_val) + (1.0f - y_val) * std::log(1.0f - p_val));
        }

        return pred_data.empty() ? 0.0f : (sum_val / static_cast<float>(pred_data.size()));
    }

    Matrix computeGradient(const Matrix &pred_matrix, const Matrix &target_matrix) const override
    {
        if (pred_matrix.getRows() != target_matrix.getRows() || pred_matrix.getCols() != target_matrix.getCols())
        {
            Logger::logMessage("BCE_Cost::computeGradient: Dimensions mismatch", LOG_ERROR);
            throw std::invalid_argument("Dimensions mismatch");
        }

        std::vector<float> pred_data = pred_matrix.getData();
        std::vector<float> target_data = target_matrix.getData();
        std::size_t total_elements = pred_data.size();
        std::vector<float> grad_data(total_elements);

        for (std::size_t i = 0; i < total_elements; ++i)
        {
            float p_val = std::clamp(pred_data[i], epsilon_val, 1.0f - epsilon_val);
            float y_val = target_data[i];
            grad_data[i] = (p_val - y_val) / (p_val * (1.0f - p_val));
        }

        return Matrix(pred_matrix.getRows(), pred_matrix.getCols(), std::move(grad_data), pred_matrix.getTarget());
    }
};

class CCE_Cost : public ICostFunction
{
private:
    float epsilon_val = 1e-7f;

public:
    explicit CCE_Cost(float eps_param = 1e-7f)
        : epsilon_val(eps_param) {}

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
};
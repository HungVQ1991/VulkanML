#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ioptimizer.h"
#include "math/matrix.h"

class Adam_Optimizer : public IOptimizer
{
private:
    struct Parameter_State
    {
        Matrix m_matrix;
        Matrix v_matrix;

        Parameter_State(std::size_t rows, std::size_t cols, Execution_Target target)
            : m_matrix(rows, cols, std::vector<float>(rows * cols, 0.0f), target),
              v_matrix(rows, cols, std::vector<float>(rows * cols, 0.0f), target) {}

        Parameter_State(Matrix m, Matrix v)
            : m_matrix(std::move(m)), v_matrix(std::move(v)) {}
    };

    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float max_gradient;
    std::size_t timestep = 0;
    ILearning_Rate *lr_scheduler = nullptr;

    std::unordered_map<Matrix *, Parameter_State> states;
    std::vector<Matrix *> param_order;
    std::vector<Parameter_State> loaded_states;

public:
    explicit Adam_Optimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float max_grad = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), max_gradient(max_grad), lr_scheduler(nullptr)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Adam_Optimizer::Adam_Optimizer: Initial learning rate is non-positive", LOG_WARNING);
        }
        OPTIMIZER_LOG_DEBUG("Adam_Optimizer::Adam_Optimizer: lr=" + std::to_string(learning_rate) + ", beta1=" + std::to_string(beta1) + ", beta2=" + std::to_string(beta2) + ", epsilon=" + std::to_string(epsilon) + ", max_gradient=" + std::to_string(max_gradient));
    }

    explicit Adam_Optimizer(ILearning_Rate &lr, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float max_grad = 1.0f)
        : learning_rate(lr.getCurrentRate()), beta1(b1), beta2(b2), epsilon(eps), max_gradient(max_grad), lr_scheduler(&lr)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Adam_Optimizer::Adam_Optimizer: Initial learning rate is non-positive", LOG_WARNING);
        }
        OPTIMIZER_LOG_DEBUG("Adam_Optimizer::Adam_Optimizer (ILearning_Rate): lr=" + std::to_string(learning_rate) + ", beta1=" + std::to_string(beta1) + ", beta2=" + std::to_string(beta2) + ", epsilon=" + std::to_string(epsilon) + ", max_gradient=" + std::to_string(max_gradient));
    }

    ~Adam_Optimizer() override = default;

    Optimizer_Type getType() const override { return Optimizer_Type::ADAM_OPTIMIZER; }

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) override
    {
        if (lr_scheduler != nullptr)
        {
            learning_rate = lr_scheduler->getCurrentRate();
        }

        if (states.empty() && !loaded_states.empty())
        {
            for (std::size_t i = 0; i < param_grad_pairs.size() && i < loaded_states.size(); ++i)
            {
                Matrix *param = param_grad_pairs[i].first;
                if (param)
                {
                    if (param->getRows() == loaded_states[i].m_matrix.getRows() &&
                        param->getCols() == loaded_states[i].m_matrix.getCols())
                    {
                        states.emplace(param, std::move(loaded_states[i]));
                    }
                    else
                    {
                        Logger::logMessage("Adam_Optimizer::step: Parameter shape mismatch with loaded state, reinitializing state", LOG_WARNING);
                        states.emplace(param, Parameter_State(param->getRows(), param->getCols(), param->getTarget()));
                    }
                    param_order.push_back(param);
                }
            }
            loaded_states.clear();
        }

        ++timestep;

        std::size_t pair_index = 0;
        for (const auto &[param, grad] : param_grad_pairs)
        {
            if (!param || !grad)
            {
                Logger::logMessage("Adam_Optimizer::step: Null parameter or gradient pointer encountered", LOG_WARNING);
                continue;
            }

            auto it = states.find(param);
            if (it == states.end())
            {
                auto [new_it, inserted] = states.emplace(param, Parameter_State(param->getRows(), param->getCols(), param->getTarget()));
                it = new_it;
                param_order.push_back(param);
            }

            param->adamUpdate(*grad, it->second.m_matrix, it->second.v_matrix, learning_rate, beta1, beta2, epsilon, timestep, max_gradient);

            ++pair_index;
        }
    }

    void reset() override
    {
        OPTIMIZER_LOG_DEBUG("Adam_Optimizer::reset: Resetting optimizer states and timestep");
        states.clear();
        param_order.clear();
        loaded_states.clear();
        timestep = 0;
    }

    float getLearningRate() const { return learning_rate; }
    float getMaxGradient() const { return max_gradient; }
    void setLearningRate(float lr) override
    {
        if (lr <= 0.0f)
        {
            Logger::logMessage("Adam_Optimizer::setLearningRate: learning_rate is non-positive", LOG_WARNING);
        }
        OPTIMIZER_LOG_DEBUG("Adam_Optimizer::setLearningRate: old_lr=" + std::to_string(learning_rate) + ", new_lr=" + std::to_string(lr));
        learning_rate = lr;
        lr_scheduler = nullptr;
    }

    void saveCheckpoint(std::ofstream &out_file) const override
    {
        if (!out_file.is_open())
        {
            Logger::logMessage("Adam_Optimizer::saveCheckpoint: Output stream is not open", LOG_ERROR, true);
            return;
        }

        OPTIMIZER_LOG_DEBUG("Adam_Optimizer::saveCheckpoint: Saving checkpoint at timestep=" + std::to_string(timestep) + ", lr=" + std::to_string(learning_rate));

        std::uint64_t step_val = static_cast<std::uint64_t>(timestep);
        out_file.write(reinterpret_cast<const char *>(&step_val), sizeof(step_val));
        out_file.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        out_file.write(reinterpret_cast<const char *>(&beta1), sizeof(beta1));
        out_file.write(reinterpret_cast<const char *>(&beta2), sizeof(beta2));
        out_file.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        out_file.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));

        std::uint32_t num_states = static_cast<std::uint32_t>(param_order.size());
        out_file.write(reinterpret_cast<const char *>(&num_states), sizeof(num_states));

        for (Matrix *param : param_order)
        {
            auto it = states.find(param);
            if (it != states.end())
            {
                it->second.m_matrix.saveMatrix(out_file);
                it->second.v_matrix.saveMatrix(out_file);
            }
        }
    }

    void loadCheckpoint(std::ifstream &in_file, Execution_Target target = Execution_Target::CPU) override
    {
        if (!in_file.is_open())
        {
            Logger::logMessage("Adam_Optimizer::loadCheckpoint: Input stream is not open", LOG_ERROR, true);
            return;
        }

        reset();

        std::uint64_t step_val{0};
        in_file.read(reinterpret_cast<char *>(&step_val), sizeof(step_val));
        timestep = static_cast<std::size_t>(step_val);

        in_file.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        in_file.read(reinterpret_cast<char *>(&beta1), sizeof(beta1));
        in_file.read(reinterpret_cast<char *>(&beta2), sizeof(beta2));
        in_file.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
        in_file.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));

        std::uint32_t num_states{0};
        in_file.read(reinterpret_cast<char *>(&num_states), sizeof(num_states));

        OPTIMIZER_LOG_DEBUG("Adam_Optimizer::loadCheckpoint: Loaded timestep=" + std::to_string(timestep) + ", lr=" + std::to_string(learning_rate) + ", beta1=" + std::to_string(beta1) + ", beta2=" + std::to_string(beta2) + ", epsilon=" + std::to_string(epsilon) + ", max_gradient=" + std::to_string(max_gradient) + ", num_states=" + std::to_string(num_states));

        loaded_states.reserve(num_states);
        for (std::uint32_t i = 0; i < num_states; ++i)
        {
            Matrix m = Matrix::loadMatrix(in_file, target);
            Matrix v = Matrix::loadMatrix(in_file, target);
            loaded_states.emplace_back(std::move(m), std::move(v));
        }
    }
};
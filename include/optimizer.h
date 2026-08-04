#pragma once
#include "math/logger.h"
#include "math/matrix.h"
#include "learning_rate.h"

#include <utility>
#include <vector>
#include <map>

class IOptimizer
{
public:
    virtual ~IOptimizer() noexcept = default;
    virtual void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) = 0;
    virtual void reset() {}
    virtual void setLearningRate(float lr) = 0;
};

class SGD_Optimizer : public IOptimizer
{
private:
    float learning_rate = 0.0f;
    float max_gradient = 0.0f;

public:
    SGD_Optimizer() = default;
    SGD_Optimizer(float _learning_rate = 0.01f, float _max_gradient = 1.0f) : learning_rate(_learning_rate), max_gradient(_max_gradient) {}
    SGD_Optimizer(Learning_Rate _learning_rate, float _max_gradient = 1.0f) : learning_rate(_learning_rate.getCurrentRate()), max_gradient(_max_gradient) {}

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) override
    {
        for (const std::pair<Matrix *, Matrix *> param_grad_pair : param_grad_pairs) param_grad_pair.first->sgdUpdate(*param_grad_pair.second, learning_rate, max_gradient);
    }

    float getLearningRate() { return learning_rate; }
    float getMaxGradient() { return max_gradient; }
    void setLearningRate(float lr) override { learning_rate = lr; }
};

class Adam_Optimizer : public IOptimizer
{
private:
    struct Parameter_State
    {
        Matrix m_matrix;
        Matrix v_matrix;

        Parameter_State(std::size_t rows, std::size_t cols, Execution_Target target)
            : m_matrix(rows, cols, target), v_matrix(rows, cols, target) {}
    };

    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float max_gradient;
    std::size_t timestep = 0;

    std::unordered_map<Matrix *, Parameter_State> states;

public:
    explicit Adam_Optimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float max_grad = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), max_gradient(max_grad) {}

        
    explicit Adam_Optimizer(Learning_Rate lr, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float max_grad = 1.0f)
        : learning_rate(lr.getCurrentRate()), beta1(b1), beta2(b2), epsilon(eps), max_gradient(max_grad) {}

    ~Adam_Optimizer() override = default;

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) override
    {
        ++timestep;
        for (const auto &[param, grad] : param_grad_pairs)
        {
            if (!param || !grad) continue;

            auto it = states.find(param);
            if (it == states.end())
            {
                auto [new_it, inserted] = states.emplace(param, Parameter_State(param->getRows(), param->getCols(), param->getTarget()));
                it = new_it;
            }

            param->adamUpdate(*grad, it->second.m_matrix, it->second.v_matrix, learning_rate, beta1, beta2, epsilon, timestep, max_gradient);
        }
    }

    void reset() override
    {
        states.clear();
        timestep = 0;
    }

    float getLearningRate() { return learning_rate; }
    float getMaxGradient() { return max_gradient; }
    void setLearningRate(float lr) override { learning_rate = lr; }
};
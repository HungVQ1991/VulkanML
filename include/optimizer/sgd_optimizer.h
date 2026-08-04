#pragma once

#include <fstream>
#include <utility>
#include <vector>

#include "ioptimizer.h"
#include "math/matrix.h"

class SGD_Optimizer : public IOptimizer
{
private:
    float learning_rate = 0.0f;
    float max_gradient = 0.0f;

public:
    SGD_Optimizer() = default;

    explicit SGD_Optimizer(float lr = 0.01f, float max_grad = 1.0f)
        : learning_rate(lr), max_gradient(max_grad) {}

    explicit SGD_Optimizer(ILearning_Rate &lr, float max_grad = 1.0f)
        : learning_rate(lr.getCurrentRate()), max_gradient(max_grad) {}

    ~SGD_Optimizer() override = default;

    Optimizer_Type getType() const override { return Optimizer_Type::SGD_OPTIMIZER; }

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) override
    {
        for (const auto &[param, grad] : param_grad_pairs)
        {
            if (param && grad)
            {
                param->sgdUpdate(*grad, learning_rate, max_gradient);
            }
        }
    }

    void reset() override {}

    float getLearningRate() const { return learning_rate; }
    float getMaxGradient() const { return max_gradient; }
    void setLearningRate(float lr) override { learning_rate = lr; }

    void saveCheckpoint(std::ofstream &out_file) const override
    {
        out_file.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        out_file.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));
    }

    void loadCheckpoint(std::ifstream &in_file, Execution_Target target = Execution_Target::CPU) override
    {
        in_file.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        in_file.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));
    }
};
#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ioptimizer.h"
#include "math/matrix.h"

class SGD_Optimizer : public IOptimizer
{
private:
    float learning_rate = 0.0f;
    float max_gradient = 0.0f;

public:
    explicit SGD_Optimizer(float lr = 0.01f, float max_grad = 1.0f)
        : learning_rate(lr), max_gradient(max_grad)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("SGD_Optimizer::SGD_Optimizer: Initial learning rate is non-positive", LOG_WARNING);
        }
    }

    explicit SGD_Optimizer(ILearning_Rate &lr, float max_grad = 1.0f)
        : learning_rate(lr.getCurrentRate()), max_gradient(max_grad)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("SGD_Optimizer::SGD_Optimizer: Initial learning rate is non-positive", LOG_WARNING);
        }
    }

    ~SGD_Optimizer() override = default;

    Optimizer_Type getType() const override { return Optimizer_Type::SGD_OPTIMIZER; }

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) override
    {
        OPTIMIZER_LOG_DEBUG("SGD_Optimizer::step: lr=" + std::to_string(learning_rate) + ", pairs=" + std::to_string(param_grad_pairs.size()));

        for (const auto &[param, grad] : param_grad_pairs)
        {
            if (param && grad)
            {
                param->sgdUpdate(*grad, learning_rate, max_gradient);
            }
            else
            {
                Logger::logMessage("SGD_Optimizer::step: Null parameter or gradient pointer encountered", LOG_WARNING);
            }
        }
    }

    void reset() override
    {
        OPTIMIZER_LOG_DEBUG("SGD_Optimizer::reset: Resetting SGD optimizer");
    }

    float getLearningRate() const { return learning_rate; }
    float getMaxGradient() const { return max_gradient; }
    void setLearningRate(float lr) override
    {
        if (lr <= 0.0f)
        {
            Logger::logMessage("SGD_Optimizer::setLearningRate: learning_rate is non-positive", LOG_WARNING);
        }
        learning_rate = lr;
    }

    void saveCheckpoint(std::ofstream &out_file) const override
    {
        if (!out_file.is_open())
        {
            Logger::logMessage("SGD_Optimizer::saveCheckpoint: Output stream is not open", LOG_ERROR, true);
            return;
        }

        OPTIMIZER_LOG_DEBUG("SGD_Optimizer::saveCheckpoint: Saving SGD checkpoint");

        out_file.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        out_file.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));
    }

    void loadCheckpoint(std::ifstream &in_file, Execution_Target target = Execution_Target::CPU) override
    {
        if (!in_file.is_open())
        {
            Logger::logMessage("SGD_Optimizer::loadCheckpoint: Input stream is not open", LOG_ERROR, true);
            return;
        }

        in_file.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        in_file.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));

        OPTIMIZER_LOG_DEBUG("SGD_Optimizer::loadCheckpoint: Loaded lr=" + std::to_string(learning_rate) + ", max_grad=" + std::to_string(max_gradient));
    }
};
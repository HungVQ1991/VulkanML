#pragma once

#include <cstddef>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ioptimizer.h"
#include "math/matrix.h"

class Sgd_Optimizer : public IOptimizer
{
private:
    float learning_rate = 0.01f;
    float max_gradient = 1.0f;
    ILearning_Rate *learning_rate_scheduler = nullptr;

public:
    explicit Sgd_Optimizer(float _learning_rate = 0.01f, float _max_gradient = 1.0f)
        : learning_rate(_learning_rate),
          max_gradient(_max_gradient),
          learning_rate_scheduler(nullptr)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Sgd_Optimizer::Sgd_Optimizer: Initial learning rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
    }

    explicit Sgd_Optimizer(ILearning_Rate &_learning_rate_scheduler, float _max_gradient = 1.0f)
        : learning_rate(_learning_rate_scheduler.getCurrentRate()),
          max_gradient(_max_gradient),
          learning_rate_scheduler(&_learning_rate_scheduler)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Sgd_Optimizer::Sgd_Optimizer: Initial learning rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
    }

    ~Sgd_Optimizer() noexcept override = default;

    [[nodiscard]] Optimizer_Type getType() const noexcept override
    {
        return Optimizer_Type::SGD_OPTIMIZER;
    }

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &_parameter_gradient_pairs) override
    {
        if (learning_rate_scheduler != nullptr)
        {
            learning_rate = learning_rate_scheduler->getCurrentRate();
        }

        Logger::logMessage(std::format("Sgd_Optimizer::step: learning_rate={}, pairs_count={}",
                                       learning_rate,
                                       _parameter_gradient_pairs.size()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);

        for (const auto &[parameter, gradient] : _parameter_gradient_pairs)
        {
            if (parameter && gradient)
            {
                parameter->sgdUpdate(*gradient, learning_rate, max_gradient);
            }
            else
            {
                Logger::logMessage("Sgd_Optimizer::step: Null parameter or gradient pointer encountered",
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::OPTIMIZER_STEP);
            }
        }
    }

    void reset() override
    {
        Logger::logMessage("Sgd_Optimizer::reset: Resetting SGD optimizer",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    [[nodiscard]] float getLearningRate() const noexcept
    {
        return learning_rate;
    }

    [[nodiscard]] float getMaxGradient() const noexcept
    {
        return max_gradient;
    }

    void setLearningRate(float _learning_rate) override
    {
        if (_learning_rate <= 0.0f)
        {
            Logger::logMessage("Sgd_Optimizer::setLearningRate: learning_rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
        Logger::logMessage(std::format("Sgd_Optimizer::setLearningRate: old_learning_rate={}, new_learning_rate={}",
                                       learning_rate,
                                       _learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
        learning_rate = _learning_rate;
        learning_rate_scheduler = nullptr;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        if (!_output_file_stream.is_open())
        {
            Logger::logMessage("Sgd_Optimizer::saveCheckpoint: Output stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return;
        }

        Logger::logMessage("Sgd_Optimizer::saveCheckpoint: Saving SGD checkpoint",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) override
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Sgd_Optimizer::loadCheckpoint: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return;
        }

        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));

        Logger::logMessage(std::format("Sgd_Optimizer::loadCheckpoint: Loaded learning_rate={}, max_gradient={}",
                                       learning_rate,
                                       max_gradient),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
    }
};

using SGD_Optimizer = Sgd_Optimizer;
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Step_Decay : public ILearning_Rate
{
private:
    float learning_rate;
    float min_learning_rate;
    float decay_rate;
    float current_rate;
    int step_size;
    int current_epoch{0};

    void validateParameters() const
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Step_Decay::validateParameters: Initial learning rate must be greater than 0.", LOG_ERROR, true);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (min_learning_rate < 0.0f || min_learning_rate > learning_rate)
        {
            Logger::logMessage("Step_Decay::validateParameters: min_learning_rate is out of valid bounds.", LOG_ERROR, true);
            throw std::invalid_argument("min_learning_rate is out of valid bounds.");
        }
        if (step_size <= 0)
        {
            Logger::logMessage("Step_Decay::validateParameters: step_size must be greater than 0.", LOG_ERROR, true);
            throw std::invalid_argument("step_size must be greater than 0.");
        }
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
        {
            Logger::logMessage("Step_Decay::validateParameters: decay_rate should be in range (0, 1).", LOG_ERROR, true);
            throw std::invalid_argument("decay_rate should be in range (0, 1).");
        }
    }

public:
    Step_Decay(float init_lr = 0.001f, float min_lr = 1e-6f, float lr_decay_rate = 0.1f, int lr_step_size = 10)
        : learning_rate(init_lr), min_learning_rate(min_lr), decay_rate(lr_decay_rate), current_rate(init_lr), step_size(lr_step_size)
    {
        validateParameters();
    }

    ~Step_Decay() override = default;

    Decay_Mode getType() const override
    {
        return Decay_Mode::STEP_DECAY;
    }

    float updateRate() override
    {
        int steps = current_epoch / step_size;
        float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(steps));
        if (calculated_rate < min_learning_rate)
        {
            Logger::logMessage("Step_Decay::updateRate: Learning rate decayed below min_learning_rate, clamped.", LOG_WARNING);
        }
        current_rate = std::max(min_learning_rate, calculated_rate);
        LR_LOG_DEBUG("Step_Decay::updateRate: epoch=" + std::to_string(current_epoch) + ", steps_taken=" + std::to_string(steps) + ", current_rate=" + std::to_string(current_rate));
        return current_rate;
    }

    void step(float current_val = 0.0f) override
    {
        current_epoch++;
        updateRate();
    }

    float getCurrentRate() const override
    {
        return current_rate;
    }

    float getLearningRate() const override
    {
        return learning_rate;
    }

    void saveCheckpoint(std::ofstream &stream) const override
    {
        stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        stream.write(reinterpret_cast<const char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        stream.write(reinterpret_cast<const char *>(&current_rate), sizeof(current_rate));
        stream.write(reinterpret_cast<const char *>(&step_size), sizeof(step_size));
        stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

    void loadCheckpoint(std::ifstream &stream) override
    {
        stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        stream.read(reinterpret_cast<char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        stream.read(reinterpret_cast<char *>(&current_rate), sizeof(current_rate));
        stream.read(reinterpret_cast<char *>(&step_size), sizeof(step_size));
        stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }
};
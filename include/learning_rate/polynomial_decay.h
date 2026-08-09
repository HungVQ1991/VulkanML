#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Polynomial_Decay : public ILearning_Rate
{
private:
    float learning_rate;
    float min_learning_rate;
    float current_rate;
    int max_epoch;
    int current_epoch{0};

    void validateParameters() const
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Polynomial_Decay::validateParameters: Initial learning rate must be greater than 0.", LOG_ERROR, true);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (min_learning_rate < 0.0f || min_learning_rate > learning_rate)
        {
            Logger::logMessage("Polynomial_Decay::validateParameters: min_learning_rate is out of valid bounds.", LOG_ERROR, true);
            throw std::invalid_argument("min_learning_rate is out of valid bounds.");
        }
        if (max_epoch <= 0)
        {
            Logger::logMessage("Polynomial_Decay::validateParameters: max_epoch must be greater than 0.", LOG_ERROR, true);
            throw std::invalid_argument("max_epoch must be greater than 0.");
        }
    }

public:
    Polynomial_Decay(float init_lr = 0.001f, float min_lr = 1e-6f, int total_epochs = 100)
        : learning_rate(init_lr), min_learning_rate(min_lr), current_rate(init_lr), max_epoch(total_epochs)
    {
        validateParameters();
    }

    ~Polynomial_Decay() override = default;

    Decay_Mode getType() const override
    {
        return Decay_Mode::POLYNOMIAL_DECAY;
    }

    void setMaxEpoch(int total_epochs) override
    {
        if (total_epochs <= 0)
        {
            Logger::logMessage("Polynomial_Decay::setMaxEpoch: total_epochs must be greater than 0.", LOG_WARNING);
            return;
        }
        max_epoch = total_epochs;
        LR_LOG_DEBUG("Polynomial_Decay::setMaxEpoch: updated max_epoch=" + std::to_string(max_epoch));
    }

    float updateRate() override
    {
        if (current_epoch >= max_epoch)
        {
            Logger::logMessage("Polynomial_Decay::updateRate: current_epoch reached or exceeded max_epoch, rate clamped to min_learning_rate.", LOG_WARNING);
            current_rate = min_learning_rate;
        }
        else
        {
            float progress = 1.0f - (static_cast<float>(current_epoch) / static_cast<float>(max_epoch));
            current_rate = (learning_rate - min_learning_rate) * progress + min_learning_rate;
        }
        LR_LOG_DEBUG("Polynomial_Decay::updateRate: epoch=" + std::to_string(current_epoch) + "/" + std::to_string(max_epoch) + ", current_rate=" + std::to_string(current_rate));
        return current_rate;
    }

    void step(float current_val = 0.0f) override
    {
        current_epoch++;
        updateRate();
    }

    float getLearningRate() const override
    {
        return learning_rate;
    }

    float getCurrentRate() const override
    {
        return current_rate;
    }

    void saveCheckpoint(std::ofstream &stream) const override
    {
        stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        stream.write(reinterpret_cast<const char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.write(reinterpret_cast<const char *>(&current_rate), sizeof(current_rate));
        stream.write(reinterpret_cast<const char *>(&max_epoch), sizeof(max_epoch));
        stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

    void loadCheckpoint(std::ifstream &stream) override
    {
        stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        stream.read(reinterpret_cast<char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.read(reinterpret_cast<char *>(&current_rate), sizeof(current_rate));
        stream.read(reinterpret_cast<char *>(&max_epoch), sizeof(max_epoch));
        stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }
};
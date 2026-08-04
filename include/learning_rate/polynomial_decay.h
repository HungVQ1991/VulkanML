#pragma once

#include <fstream>
#include <stdexcept>

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
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        if (min_learning_rate < 0.0f || min_learning_rate > learning_rate)
            throw std::invalid_argument("min_learning_rate is out of valid bounds.");
        if (max_epoch <= 0)
            throw std::invalid_argument("max_epoch must be greater than 0.");
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
        max_epoch = total_epochs;
    }

    float updateRate() override
    {
        if (current_epoch >= max_epoch)
        {
            current_rate = min_learning_rate;
        }
        else
        {
            float progress = 1.0f - (static_cast<float>(current_epoch) / static_cast<float>(max_epoch));
            current_rate = (learning_rate - min_learning_rate) * progress + min_learning_rate;
        }
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
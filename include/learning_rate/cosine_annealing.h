#pragma once

#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>

#include "ilearning_rate.h"

class Cosine_Annealing : public ILearning_Rate
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
    Cosine_Annealing(float init_lr = 0.001f, float min_lr = 0.0f, int total_epochs = 100)
        : learning_rate(init_lr), min_learning_rate(min_lr), current_rate(init_lr), max_epoch(total_epochs)
    {
        validateParameters();
    }

    ~Cosine_Annealing() override = default;

    Decay_Mode getType() const override
    {
        return Decay_Mode::COSINE_ANNEALING;
    }

    float updateRate() override
    {
        if (current_epoch >= max_epoch)
        {
            current_rate = min_learning_rate;
        }
        else
        {
            float progress = static_cast<float>(current_epoch) / static_cast<float>(max_epoch);
            float cos_val = std::cos(progress * std::numbers::pi_v<float>);
            current_rate = min_learning_rate + 0.5f * (learning_rate - min_learning_rate) * (1.0f + cos_val);
        }
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

    void setMaxEpoch(int total_epochs) override
    {
        max_epoch = total_epochs;
        if (current_epoch >= max_epoch)
        {
            current_rate = min_learning_rate;
            return;
        }
        constexpr float pi_val = 3.14159265358979323846f;
        float cosine_decay = 0.5f * (1.0f + std::cos(pi_val * static_cast<float>(current_epoch) / static_cast<float>(max_epoch)));
        current_rate = min_learning_rate + (learning_rate - min_learning_rate) * cosine_decay;
    }

    float getLearningRate() const override
    {
        return learning_rate;
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
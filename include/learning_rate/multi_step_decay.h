#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "ilearning_rate.h"

class Multi_Step_Decay : public ILearning_Rate
{
private:
    float learning_rate;
    float min_learning_rate;
    float decay_rate;
    float current_rate;
    int current_epoch{0};
    std::vector<float> decay_epochs;

    void validateParameters()
    {
        if (learning_rate <= 0.0f)
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        if (min_learning_rate < 0.0f || min_learning_rate > learning_rate)
            throw std::invalid_argument("min_learning_rate is out of valid bounds.");
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
            throw std::invalid_argument("decay_rate should be in range (0, 1).");

        if (decay_epochs.empty())
            throw std::invalid_argument("decay_epochs cannot be empty.");

        for (float epoch : decay_epochs)
        {
            if (epoch <= 0.0f)
                throw std::invalid_argument("decay_epochs must contain positive values.");
        }
        std::sort(decay_epochs.begin(), decay_epochs.end());
    }

public:
    Multi_Step_Decay(float init_lr = 0.001f, float min_lr = 1e-6f, float lr_decay_rate = 0.1f, std::vector<float> epochs_list = {30.0f, 60.0f, 80.0f})
        : learning_rate(init_lr), min_learning_rate(min_lr), decay_rate(lr_decay_rate),
          current_rate(init_lr), decay_epochs(std::move(epochs_list))
    {
        validateParameters();
    }

    ~Multi_Step_Decay() override = default;

    Decay_Mode getType() const override
    {
        return Decay_Mode::MULTI_STEP_DECAY;
    }

    float updateRate() override
    {
        int milestone_count = 0;
        for (float epoch_milestone : decay_epochs)
        {
            if (static_cast<float>(current_epoch) >= epoch_milestone)
            {
                milestone_count++;
            }
        }
        current_rate = std::max(min_learning_rate, learning_rate * std::pow(decay_rate, static_cast<float>(milestone_count)));
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
        stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));

        std::uint32_t vector_size = static_cast<std::uint32_t>(decay_epochs.size());
        stream.write(reinterpret_cast<const char *>(&vector_size), sizeof(vector_size));
        if (vector_size > 0)
        {
            stream.write(reinterpret_cast<const char *>(decay_epochs.data()), vector_size * sizeof(float));
        }
    }

    void loadCheckpoint(std::ifstream &stream) override
    {
        stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        stream.read(reinterpret_cast<char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        stream.read(reinterpret_cast<char *>(&current_rate), sizeof(current_rate));
        stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));

        std::uint32_t vector_size{0};
        stream.read(reinterpret_cast<char *>(&vector_size), sizeof(vector_size));
        decay_epochs.resize(vector_size);
        if (vector_size > 0)
        {
            stream.read(reinterpret_cast<char *>(decay_epochs.data()), vector_size * sizeof(float));
        }
    }
};
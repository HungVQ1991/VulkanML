#pragma once

#include <fstream>
#include <stdexcept>

#include "ilearning_rate.h"

class No_Decay : public ILearning_Rate
{
private:
    float learning_rate;

public:
    explicit No_Decay(float init_lr = 0.01f)
        : learning_rate(init_lr)
    {
        if (learning_rate <= 0.0f)
        {
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
    }

    ~No_Decay() override = default;

    Decay_Mode getType() const override { return Decay_Mode::NO_DECAY; }
    float updateRate() override { return learning_rate; }
    void step(float current_val = 0.0f) override {}
    float getCurrentRate() const override { return learning_rate; }
    float getLearningRate() const override { return learning_rate; }
    void saveCheckpoint(std::ofstream &stream) const override { stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate)); }
    void loadCheckpoint(std::ifstream &stream) override { stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate)); }
};
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "ilearning_rate.h"

class Reduce_On_Plateau : public ILearning_Rate
{
private:
    float learning_rate;
    float min_learning_rate;
    float decay_rate;
    float current_rate;
    int patience;
    int current_epoch{0};
    int bad_epochs_count{0};
    int reductions_count{0};
    float best_metric{0.0f};
    bool higher_is_better;
    bool is_first_step{true};

    void validateParameters() const
    {
        if (learning_rate <= 0.0f)
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        if (min_learning_rate < 0.0f || min_learning_rate > learning_rate)
            throw std::invalid_argument("min_learning_rate is out of valid bounds.");
        if (patience <= 0)
            throw std::invalid_argument("patience must be greater than 0.");
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
            throw std::invalid_argument("decay_rate should be in range (0, 1).");
    }

public:
    Reduce_On_Plateau(float init_lr = 0.001f, float min_lr = 1e-6f, float lr_decay_rate = 0.1f, int lr_patience = 10, bool mode_higher_is_better = false)
        : learning_rate(init_lr), min_learning_rate(min_lr), decay_rate(lr_decay_rate),
          current_rate(init_lr), patience(lr_patience), higher_is_better(mode_higher_is_better)
    {
        validateParameters();
    }

    ~Reduce_On_Plateau() override = default;

    Decay_Mode getType() const override
    {
        return Decay_Mode::REDUCE_ON_PLATEAU;
    }

    float updateRate() override
    {
        current_rate = std::max(min_learning_rate, learning_rate * std::pow(decay_rate, static_cast<float>(reductions_count)));
        return current_rate;
    }

    float getLearningRate() const override
    {
        return learning_rate;
    }

    void step(float current_val = 0.0f) override
    {
        if (is_first_step)
        {
            best_metric = current_val;
            is_first_step = false;
        }
        else
        {
            bool is_improved = higher_is_better ? (current_val > best_metric) : (current_val < best_metric);
            if (is_improved)
            {
                best_metric = current_val;
                bad_epochs_count = 0;
            }
            else
            {
                bad_epochs_count++;
                if (bad_epochs_count >= patience)
                {
                    reductions_count++;
                    bad_epochs_count = 0;
                    updateRate();
                }
            }
        }
        current_epoch++;
    }

    float getCurrentRate() const override
    {
        return current_rate;
    }

    void saveCheckpoint(std::ofstream &stream) const override
    {
        stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        stream.write(reinterpret_cast<const char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        stream.write(reinterpret_cast<const char *>(&current_rate), sizeof(current_rate));
        stream.write(reinterpret_cast<const char *>(&patience), sizeof(patience));
        stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
        stream.write(reinterpret_cast<const char *>(&bad_epochs_count), sizeof(bad_epochs_count));
        stream.write(reinterpret_cast<const char *>(&reductions_count), sizeof(reductions_count));
        stream.write(reinterpret_cast<const char *>(&best_metric), sizeof(best_metric));
        stream.write(reinterpret_cast<const char *>(&higher_is_better), sizeof(higher_is_better));
        stream.write(reinterpret_cast<const char *>(&is_first_step), sizeof(is_first_step));
    }

    void loadCheckpoint(std::ifstream &stream) override
    {
        stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        stream.read(reinterpret_cast<char *>(&min_learning_rate), sizeof(min_learning_rate));
        stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        stream.read(reinterpret_cast<char *>(&current_rate), sizeof(current_rate));
        stream.read(reinterpret_cast<char *>(&patience), sizeof(patience));
        stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
        stream.read(reinterpret_cast<char *>(&bad_epochs_count), sizeof(bad_epochs_count));
        stream.read(reinterpret_cast<char *>(&reductions_count), sizeof(reductions_count));
        stream.read(reinterpret_cast<char *>(&best_metric), sizeof(best_metric));
        stream.read(reinterpret_cast<char *>(&higher_is_better), sizeof(higher_is_better));
        stream.read(reinterpret_cast<char *>(&is_first_step), sizeof(is_first_step));
    }
};
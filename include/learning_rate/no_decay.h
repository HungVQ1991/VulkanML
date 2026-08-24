#pragma once

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class No_Decay : public ILearning_Rate
{
private:
    float learning_rate = 0.01f;

public:
    explicit No_Decay(float _initial_learning_rate = 0.01f)
        : learning_rate(_initial_learning_rate)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("No_Decay::No_Decay: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
    }

    ~No_Decay() noexcept override = default;

    [[nodiscard]] Decay_Mode getType() const noexcept override
    {
        return Decay_Mode::NO_DECAY;
    }

    float updateRate() override
    {
        Logger::logMessage(std::format("No_Decay::updateRate: current_rate={}", learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return learning_rate;
    }

    void step(float _current_value = 0.0f) override
    {
    }

    [[nodiscard]] float getCurrentRate() const noexcept override
    {
        return learning_rate;
    }

    [[nodiscard]] float getLearningRate() const noexcept override
    {
        return learning_rate;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
    }
};
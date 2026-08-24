#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Multi_Step_Decay : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 1e-6f;
    float decay_rate = 0.1f;
    float current_learning_rate = 0.001f;
    int current_epoch = 0;
    std::vector<float> decay_epochs;

    void validateParameters()
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Multi_Step_Decay::validateParameters: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (minimum_learning_rate < 0.0f || minimum_learning_rate > learning_rate)
        {
            Logger::logMessage("Multi_Step_Decay::validateParameters: minimum_learning_rate is out of valid bounds.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("minimum_learning_rate is out of valid bounds.");
        }
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
        {
            Logger::logMessage("Multi_Step_Decay::validateParameters: decay_rate should be in range (0, 1).",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("decay_rate should be in range (0, 1).");
        }
        if (decay_epochs.empty())
        {
            Logger::logMessage("Multi_Step_Decay::validateParameters: decay_epochs cannot be empty.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("decay_epochs cannot be empty.");
        }

        for (float epoch_milestone : decay_epochs)
        {
            if (epoch_milestone <= 0.0f)
            {
                Logger::logMessage("Multi_Step_Decay::validateParameters: decay_epochs must contain positive values.",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::LR_SCHEDULER);
                throw std::invalid_argument("decay_epochs must contain positive values.");
            }
        }
        std::sort(decay_epochs.begin(), decay_epochs.end());
    }

public:
    Multi_Step_Decay(float _initial_learning_rate = 0.001f,
                     float _minimum_learning_rate = 1e-6f,
                     float _decay_rate = 0.1f,
                     std::vector<float> _decay_epochs = {30.0f, 60.0f, 80.0f})
        : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          decay_rate(_decay_rate),
          current_learning_rate(_initial_learning_rate),
          decay_epochs(std::move(_decay_epochs))
    {
        validateParameters();
    }

    ~Multi_Step_Decay() noexcept override = default;

    [[nodiscard]] Decay_Mode getType() const noexcept override
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
        float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(milestone_count));
        if (calculated_rate < minimum_learning_rate)
        {
            Logger::logMessage("Multi_Step_Decay::updateRate: Learning rate decayed below minimum_learning_rate, clamped.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
        }
        current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
        Logger::logMessage(std::format("Multi_Step_Decay::updateRate: epoch={}, milestones_hit={}, current_rate={}",
                                       current_epoch,
                                       milestone_count,
                                       current_learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

    void step(float _current_value = 0.0f) override
    {
        current_epoch++;
        updateRate();
    }

    [[nodiscard]] float getCurrentRate() const noexcept override
    {
        return current_learning_rate;
    }

    [[nodiscard]] float getLearningRate() const noexcept override
    {
        return learning_rate;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));

        std::uint32_t vector_size = static_cast<std::uint32_t>(decay_epochs.size());
        _output_file_stream.write(reinterpret_cast<const char *>(&vector_size), sizeof(vector_size));
        if (vector_size > 0)
        {
            _output_file_stream.write(reinterpret_cast<const char *>(decay_epochs.data()), vector_size * sizeof(float));
        }
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));

        std::uint32_t vector_size = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&vector_size), sizeof(vector_size));
        decay_epochs.resize(vector_size);
        if (vector_size > 0)
        {
            _input_file_stream.read(reinterpret_cast<char *>(decay_epochs.data()), vector_size * sizeof(float));
        }
    }
};
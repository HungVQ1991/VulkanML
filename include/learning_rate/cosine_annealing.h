#pragma once

#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Cosine_Annealing : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 0.0f;
    float current_learning_rate = 0.001f;
    int maximum_epoch = 100;
    int current_epoch = 0;

    void validateParameters() const
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Cosine_Annealing::validateParameters: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (minimum_learning_rate < 0.0f || minimum_learning_rate > learning_rate)
        {
            Logger::logMessage("Cosine_Annealing::validateParameters: minimum_learning_rate is out of valid bounds.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("minimum_learning_rate is out of valid bounds.");
        }
        if (maximum_epoch <= 0)
        {
            Logger::logMessage("Cosine_Annealing::validateParameters: maximum_epoch must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("maximum_epoch must be greater than 0.");
        }
    }

public:
    Cosine_Annealing(float _initial_learning_rate = 0.001f,
                     float _minimum_learning_rate = 0.0f,
                     int _maximum_epoch = 100)
        : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          current_learning_rate(_initial_learning_rate),
          maximum_epoch(_maximum_epoch)
    {
        validateParameters();
    }

    ~Cosine_Annealing() noexcept override = default;

    [[nodiscard]] Decay_Mode getType() const noexcept override
    {
        return Decay_Mode::COSINE_ANNEALING;
    }

    float updateRate() override
    {
        if (current_epoch >= maximum_epoch)
        {
            Logger::logMessage("Cosine_Annealing::updateRate: current_epoch reached or exceeded maximum_epoch, rate clamped to minimum_learning_rate.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
            current_learning_rate = minimum_learning_rate;
        }
        else
        {
            float progress = static_cast<float>(current_epoch) / static_cast<float>(maximum_epoch);
            float cosine_value = std::cos(progress * std::numbers::pi_v<float>);
            current_learning_rate = minimum_learning_rate + 0.5f * (learning_rate - minimum_learning_rate) * (1.0f + cosine_value);
        }
        Logger::logMessage(std::format("Cosine_Annealing::updateRate: epoch={}/{}, current_rate={}",
                                       current_epoch,
                                       maximum_epoch,
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

    void setMaxEpoch(int _maximum_epoch) override
    {
        if (_maximum_epoch <= 0)
        {
            Logger::logMessage("Cosine_Annealing::setMaxEpoch: maximum_epoch must be greater than 0.",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            return;
        }
        maximum_epoch = _maximum_epoch;
        if (current_epoch >= maximum_epoch)
        {
            Logger::logMessage("Cosine_Annealing::setMaxEpoch: current_epoch already reached or exceeded new maximum_epoch.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
            current_learning_rate = minimum_learning_rate;
            return;
        }
        float cosine_decay = 0.5f * (1.0f + std::cos(std::numbers::pi_v<float> * static_cast<float>(current_epoch) / static_cast<float>(maximum_epoch)));
        current_learning_rate = minimum_learning_rate + (learning_rate - minimum_learning_rate) * cosine_decay;
        Logger::logMessage(std::format("Cosine_Annealing::setMaxEpoch: updated maximum_epoch={}, current_rate={}",
                                       maximum_epoch,
                                       current_learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
    }

    [[nodiscard]] float getLearningRate() const noexcept override
    {
        return learning_rate;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&maximum_epoch), sizeof(maximum_epoch));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&maximum_epoch), sizeof(maximum_epoch));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }
};
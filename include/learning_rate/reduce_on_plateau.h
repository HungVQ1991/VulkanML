#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Reduce_On_Plateau : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 1e-6f;
    float decay_rate = 0.1f;
    float current_learning_rate = 0.001f;
    int patience = 10;
    int current_epoch = 0;
    int bad_epochs_count = 0;
    int reductions_count = 0;
    float best_metric = 0.0f;
    bool is_higher_better = false;
    bool is_first_step = true;

    void validateParameters() const
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Reduce_On_Plateau::validateParameters: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (minimum_learning_rate < 0.0f || minimum_learning_rate > learning_rate)
        {
            Logger::logMessage("Reduce_On_Plateau::validateParameters: minimum_learning_rate is out of valid bounds.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("minimum_learning_rate is out of valid bounds.");
        }
        if (patience <= 0)
        {
            Logger::logMessage("Reduce_On_Plateau::validateParameters: patience must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("patience must be greater than 0.");
        }
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
        {
            Logger::logMessage("Reduce_On_Plateau::validateParameters: decay_rate should be in range (0, 1).",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("decay_rate should be in range (0, 1).");
        }
    }

public:
    Reduce_On_Plateau(float _initial_learning_rate = 0.001f,
                      float _minimum_learning_rate = 1e-6f,
                      float _decay_rate = 0.1f,
                      int _patience = 10,
                      bool _is_higher_better = false)
        : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          decay_rate(_decay_rate),
          current_learning_rate(_initial_learning_rate),
          patience(_patience),
          is_higher_better(_is_higher_better)
    {
        validateParameters();
    }

    ~Reduce_On_Plateau() noexcept override = default;

    [[nodiscard]] Decay_Mode getType() const noexcept override
    {
        return Decay_Mode::REDUCE_ON_PLATEAU;
    }

    float updateRate() override
    {
        float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(reductions_count));
        if (calculated_rate < minimum_learning_rate)
        {
            Logger::logMessage("Reduce_On_Plateau::updateRate: Learning rate decayed below minimum_learning_rate, clamped.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
        }
        current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
        Logger::logMessage(std::format("Reduce_On_Plateau::updateRate: reductions_count={}, current_rate={}",
                                       reductions_count,
                                       current_learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

    [[nodiscard]] float getLearningRate() const noexcept override
    {
        return learning_rate;
    }

    void step(float _current_value = 0.0f) override
    {
        if (is_first_step)
        {
            best_metric = _current_value;
            is_first_step = false;
        }
        else
        {
            bool is_improved = is_higher_better ? (_current_value > best_metric) : (_current_value < best_metric);
            if (is_improved)
            {
                best_metric = _current_value;
                bad_epochs_count = 0;
            }
            else
            {
                bad_epochs_count++;
                Logger::logMessage(std::format("Reduce_On_Plateau::step: Plateau detected, bad_epochs_count={}/{}",
                                               bad_epochs_count,
                                               patience),
                                   Log_Level::LOG_WARNING,
                                   false,
                                   0,
                                   Log_Feature::LR_SCHEDULER);
                if (bad_epochs_count >= patience)
                {
                    reductions_count++;
                    bad_epochs_count = 0;
                    Logger::logMessage(std::format("Reduce_On_Plateau::step: Patience exhausted, triggering learning rate reduction #{}",
                                                   reductions_count),
                                       Log_Level::LOG_WARNING,
                                       true,
                                       0,
                                       Log_Feature::LR_SCHEDULER);
                    updateRate();
                }
            }
        }
        current_epoch++;
        Logger::logMessage(std::format("Reduce_On_Plateau::step: epoch={}, current_value={}, best_metric={}",
                                       current_epoch,
                                       _current_value,
                                       best_metric),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
    }

    [[nodiscard]] float getCurrentRate() const noexcept override
    {
        return current_learning_rate;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&patience), sizeof(patience));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
        _output_file_stream.write(reinterpret_cast<const char *>(&bad_epochs_count), sizeof(bad_epochs_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&reductions_count), sizeof(reductions_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&best_metric), sizeof(best_metric));
        _output_file_stream.write(reinterpret_cast<const char *>(&is_higher_better), sizeof(is_higher_better));
        _output_file_stream.write(reinterpret_cast<const char *>(&is_first_step), sizeof(is_first_step));
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&patience), sizeof(patience));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
        _input_file_stream.read(reinterpret_cast<char *>(&bad_epochs_count), sizeof(bad_epochs_count));
        _input_file_stream.read(reinterpret_cast<char *>(&reductions_count), sizeof(reductions_count));
        _input_file_stream.read(reinterpret_cast<char *>(&best_metric), sizeof(best_metric));
        _input_file_stream.read(reinterpret_cast<char *>(&is_higher_better), sizeof(is_higher_better));
        _input_file_stream.read(reinterpret_cast<char *>(&is_first_step), sizeof(is_first_step));
    }
};
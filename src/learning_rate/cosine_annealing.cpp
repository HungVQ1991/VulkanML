#include "learning_rate/cosine_annealing.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"


void Cosine_Annealing::validateParameters() const
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
        if (warmup_steps < 0 || warmup_steps >= maximum_epoch)
        {
            Logger::logMessage("Cosine_Annealing::validateParameters: warmup_steps must be >= 0 and < maximum_epoch.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("warmup_steps must be >= 0 and < maximum_epoch.");
        }
    }

Cosine_Annealing::Cosine_Annealing(float _initial_learning_rate, float _minimum_learning_rate, int _maximum_epoch, int _warmup_steps)
    : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          current_learning_rate((_warmup_steps > 0) ? _minimum_learning_rate : _initial_learning_rate),
          maximum_epoch(_maximum_epoch),
          warmup_steps(_warmup_steps)
{
        validateParameters();
    }

void Cosine_Annealing::setWarmupSteps(int _warmup_steps)
{
        warmup_steps = _warmup_steps;
        validateParameters();
        updateRate();
    }

float Cosine_Annealing::updateRate()
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
        else if (warmup_steps > 0 && current_epoch < warmup_steps)
        {
            float progress = static_cast<float>(current_epoch) / static_cast<float>(warmup_steps);
            current_learning_rate = minimum_learning_rate + progress * (learning_rate - minimum_learning_rate);
        }
        else
        {
            int decay_epoch = current_epoch - warmup_steps;
            int total_decay_epochs = maximum_epoch - warmup_steps;
            float progress = static_cast<float>(decay_epoch) / static_cast<float>(total_decay_epochs);
            float cosine_value = std::cos(progress * std::numbers::pi_v<float>);
            current_learning_rate = minimum_learning_rate + 0.5f * (learning_rate - minimum_learning_rate) * (1.0f + cosine_value);
        }
        Logger::logMessage(Input_Format{"Cosine_Annealing::updateRate: epoch={}/{}, current_rate={}",
                                        current_epoch,
                                        maximum_epoch,
                                        current_learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

void Cosine_Annealing::step(float _current_value)
{
        current_epoch++;
        updateRate();
    }

void Cosine_Annealing::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&maximum_epoch), sizeof(maximum_epoch));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

void Cosine_Annealing::loadCheckpoint(std::ifstream &_input_file_stream)
{
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&maximum_epoch), sizeof(maximum_epoch));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }

void Cosine_Annealing::setMaxEpoch(int _maximum_epoch)
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
        Logger::logMessage(Input_Format{"Cosine_Annealing::setMaxEpoch: updated maximum_epoch={}, current_rate={}",
                                        maximum_epoch,
                                        current_learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
    }

void Cosine_Annealing::setMinimumLearningRate(float _minimum_learning_rate) noexcept
{ minimum_learning_rate = _minimum_learning_rate; }

void Cosine_Annealing::setCurrentRate(float _current_rate) noexcept
{ current_learning_rate = _current_rate; }

void Cosine_Annealing::setLearningRate(float _learning_rate) noexcept
{ learning_rate = _learning_rate; }

void Cosine_Annealing::setCurrentEpoch(int _current_epoch) noexcept
{ current_epoch = _current_epoch; }

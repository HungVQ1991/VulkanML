#include "learning_rate/exponential_decay.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"


void Exponential_Decay::validateParameters() const
{
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Exponential_Decay::validateParameters: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (minimum_learning_rate < 0.0f || minimum_learning_rate > learning_rate)
        {
            Logger::logMessage("Exponential_Decay::validateParameters: minimum_learning_rate is out of valid bounds.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("minimum_learning_rate is out of valid bounds.");
        }
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
        {
            Logger::logMessage("Exponential_Decay::validateParameters: decay_rate should be in range (0, 1).",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("decay_rate should be in range (0, 1).");
        }
    }

Exponential_Decay::Exponential_Decay(float _initial_learning_rate, float _minimum_learning_rate, float _decay_rate)
    : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          decay_rate(_decay_rate),
          current_learning_rate(_initial_learning_rate)
{
        validateParameters();
    }

float Exponential_Decay::updateRate()
{
        float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(current_epoch));
        if (calculated_rate < minimum_learning_rate)
        {
            Logger::logMessage("Exponential_Decay::updateRate: Learning rate decayed below minimum_learning_rate, clamped.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
        }
        current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
        Logger::logMessage(Input_Format{"Exponential_Decay::updateRate: epoch={}, current_rate={}",
                                        current_epoch,
                                        current_learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

void Exponential_Decay::step(float _current_value)
{
        current_epoch++;
        updateRate();
    }

void Exponential_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

void Exponential_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }

float Exponential_Decay::getCurrentLearningRate() const noexcept
{ return current_learning_rate; }

void Exponential_Decay::setMinimumLearningRate(float _min_lr) noexcept
{ minimum_learning_rate = _min_lr; }

void Exponential_Decay::setCurrentLearningRate(float _rate) noexcept
{ current_learning_rate = _rate; }

void Exponential_Decay::setLearningRate(float _learning_rate) noexcept
{ learning_rate = _learning_rate; }

void Exponential_Decay::setCurrentEpoch(int _epoch) noexcept
{ current_epoch = _epoch; }

void Exponential_Decay::setDecayRate(float _decay_rate) noexcept
{ decay_rate = _decay_rate; }

#include "learning_rate/polynomial_decay.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"


void Polynomial_Decay::validateParameters() const
{
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Polynomial_Decay::validateParameters: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (minimum_learning_rate < 0.0f || minimum_learning_rate > learning_rate)
        {
            Logger::logMessage("Polynomial_Decay::validateParameters: minimum_learning_rate is out of valid bounds.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("minimum_learning_rate is out of valid bounds.");
        }
        if (maximum_epoch <= 0)
        {
            Logger::logMessage("Polynomial_Decay::validateParameters: maximum_epoch must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("maximum_epoch must be greater than 0.");
        }
    }

Polynomial_Decay::Polynomial_Decay(float _initial_learning_rate, float _minimum_learning_rate, int _maximum_epoch)
    : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          current_learning_rate(_initial_learning_rate),
          maximum_epoch(_maximum_epoch)
{
        validateParameters();
    }

float Polynomial_Decay::updateRate()
{
        if (current_epoch >= maximum_epoch)
        {
            Logger::logMessage("Polynomial_Decay::updateRate: current_epoch reached or exceeded maximum_epoch, rate clamped to minimum_learning_rate.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
            current_learning_rate = minimum_learning_rate;
        }
        else
        {
            float progress = 1.0f - (static_cast<float>(current_epoch) / static_cast<float>(maximum_epoch));
            current_learning_rate = (learning_rate - minimum_learning_rate) * progress + minimum_learning_rate;
        }
        Logger::logMessage(Input_Format{"Polynomial_Decay::updateRate: epoch={}/{}, current_rate={}",
                                        current_epoch,
                                        maximum_epoch,
                                        current_learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

void Polynomial_Decay::step(float _current_value)
{
        current_epoch++;
        updateRate();
    }

void Polynomial_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&maximum_epoch), sizeof(maximum_epoch));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

void Polynomial_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&maximum_epoch), sizeof(maximum_epoch));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }

void Polynomial_Decay::setMaxEpoch(int _maximum_epoch)
{
        if (_maximum_epoch <= 0)
        {
            Logger::logMessage("Polynomial_Decay::setMaxEpoch: maximum_epoch must be greater than 0.",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            return;
        }
        maximum_epoch = _maximum_epoch;
        Logger::logMessage(Input_Format{"Polynomial_Decay::setMaxEpoch: updated maximum_epoch={}", maximum_epoch},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
    }

void Polynomial_Decay::setMinimumLearningRate(float _minimum_learning_rate) noexcept
{ minimum_learning_rate = _minimum_learning_rate; }

void Polynomial_Decay::setCurrentRate(float _current_rate) noexcept
{ current_learning_rate = _current_rate; }

void Polynomial_Decay::setLearningRate(float _learning_rate) noexcept
{ learning_rate = _learning_rate; }

void Polynomial_Decay::setCurrentEpoch(int _current_epoch) noexcept
{ current_epoch = _current_epoch; }

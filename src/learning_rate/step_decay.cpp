#include "learning_rate/step_decay.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"


void Step_Decay::validateParameters() const
{
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Step_Decay::validateParameters: Initial learning rate must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("Initial learning rate must be greater than 0.");
        }
        if (minimum_learning_rate < 0.0f || minimum_learning_rate > learning_rate)
        {
            Logger::logMessage("Step_Decay::validateParameters: minimum_learning_rate is out of valid bounds.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("minimum_learning_rate is out of valid bounds.");
        }
        if (step_size <= 0)
        {
            Logger::logMessage("Step_Decay::validateParameters: step_size must be greater than 0.",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("step_size must be greater than 0.");
        }
        if (decay_rate <= 0.0f || decay_rate >= 1.0f)
        {
            Logger::logMessage("Step_Decay::validateParameters: decay_rate should be in range (0, 1).",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LR_SCHEDULER);
            throw std::invalid_argument("decay_rate should be in range (0, 1).");
        }
    }

Step_Decay::Step_Decay(float _initial_learning_rate, float _minimum_learning_rate, float _decay_rate, int _step_size)
    : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          decay_rate(_decay_rate),
          current_learning_rate(_initial_learning_rate),
          step_size(_step_size)
{
        validateParameters();
    }

float Step_Decay::updateRate()
{
        int steps_taken = current_epoch / step_size;
        float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(steps_taken));
        if (calculated_rate < minimum_learning_rate)
        {
            Logger::logMessage("Step_Decay::updateRate: Learning rate decayed below minimum_learning_rate, clamped.",
                               Log_Level::LOG_WARNING,
                               false,
                               0,
                               Log_Feature::LR_SCHEDULER);
        }
        current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
        Logger::logMessage(Input_Format{"Step_Decay::updateRate: epoch={}, steps_taken={}, current_rate={}",
                                        current_epoch,
                                        steps_taken,
                                        current_learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

void Step_Decay::step(float _current_value)
{
        current_epoch++;
        updateRate();
    }

void Step_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&step_size), sizeof(step_size));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));
    }

void Step_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&step_size), sizeof(step_size));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));
    }

float Step_Decay::getCurrentLearningRate() const noexcept
{ return current_learning_rate; }

void Step_Decay::setMinimumLearningRate(float _min_lr) noexcept
{ minimum_learning_rate = _min_lr; }

void Step_Decay::setCurrentLearningRate(float _rate) noexcept
{ current_learning_rate = _rate; }

void Step_Decay::setLearningRate(float _learning_rate) noexcept
{ learning_rate = _learning_rate; }

void Step_Decay::setCurrentEpoch(int _epoch) noexcept
{ current_epoch = _epoch; }

void Step_Decay::setDecayRate(float _decay_rate) noexcept
{ decay_rate = _decay_rate; }

void Step_Decay::setStepSize(int _step_size) noexcept
{ step_size = _step_size; }

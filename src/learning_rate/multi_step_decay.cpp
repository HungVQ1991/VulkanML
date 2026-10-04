#include "learning_rate/multi_step_decay.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"


void Multi_Step_Decay::validateParameters()
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

Multi_Step_Decay::Multi_Step_Decay(float _initial_learning_rate, float _minimum_learning_rate, float _decay_rate, std::vector<float> _decay_epochs)
    : learning_rate(_initial_learning_rate),
          minimum_learning_rate(_minimum_learning_rate),
          decay_rate(_decay_rate),
          current_learning_rate(_initial_learning_rate),
          decay_epochs(std::move(_decay_epochs))
{
        validateParameters();
    }

float Multi_Step_Decay::updateRate()
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
        Logger::logMessage(Input_Format{"Multi_Step_Decay::updateRate: epoch={}, milestones_hit={}, current_rate={}",
                                        current_epoch,
                                        milestone_count,
                                        current_learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return current_learning_rate;
    }

void Multi_Step_Decay::step(float _current_value)
{
        current_epoch++;
        updateRate();
    }

void Multi_Step_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&decay_rate), sizeof(decay_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_learning_rate), sizeof(current_learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&current_epoch), sizeof(current_epoch));

        uint32_t vector_size = static_cast<uint32_t>(decay_epochs.size());
        _output_file_stream.write(reinterpret_cast<const char *>(&vector_size), sizeof(vector_size));
        if (vector_size > 0)
        {
            _output_file_stream.write(reinterpret_cast<const char *>(decay_epochs.data()), vector_size * sizeof(float));
        }
    }

void Multi_Step_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&minimum_learning_rate), sizeof(minimum_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&decay_rate), sizeof(decay_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_learning_rate), sizeof(current_learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&current_epoch), sizeof(current_epoch));

        uint32_t vector_size = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&vector_size), sizeof(vector_size));
        decay_epochs.resize(vector_size);
        if (vector_size > 0)
        {
            _input_file_stream.read(reinterpret_cast<char *>(decay_epochs.data()), vector_size * sizeof(float));
        }
    }

const std::vector<float> & Multi_Step_Decay::getDecayEpochs() const noexcept
{ return decay_epochs; }

float Multi_Step_Decay::getCurrentLearningRate() const noexcept
{ return current_learning_rate; }

void Multi_Step_Decay::setDecayEpochs(std::vector<float> _decay_epochs)
{
        decay_epochs = std::move(_decay_epochs);
        validateParameters();
    }

void Multi_Step_Decay::setMinimumLearningRate(float _min_lr) noexcept
{ minimum_learning_rate = _min_lr; }

void Multi_Step_Decay::setCurrentLearningRate(float _rate) noexcept
{ current_learning_rate = _rate; }

void Multi_Step_Decay::setLearningRate(float _learning_rate) noexcept
{ learning_rate = _learning_rate; }

void Multi_Step_Decay::setCurrentEpoch(int _epoch) noexcept
{ current_epoch = _epoch; }

void Multi_Step_Decay::setDecayRate(float _decay_rate) noexcept
{ decay_rate = _decay_rate; }

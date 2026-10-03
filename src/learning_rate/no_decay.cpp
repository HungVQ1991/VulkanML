#include "learning_rate/no_decay.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "helper/logger.h"


No_Decay::No_Decay(float _initial_learning_rate)
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

float No_Decay::updateRate()
{
        Logger::logMessage(Input_Format{"No_Decay::updateRate: current_rate={}", learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LR_SCHEDULER);
        return learning_rate;
    }

void No_Decay::step(float _current_value)
{
    }

void No_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
    }

void No_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
    }

void No_Decay::setLearningRate(float _learning_rate) noexcept
{ learning_rate = _learning_rate; }

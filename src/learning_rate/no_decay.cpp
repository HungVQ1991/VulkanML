#include "learning_rate/no_decay.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include "helper/logger.h"
#include "lr_internal_utils.h"

No_Decay::No_Decay(float _initial_learning_rate)
    : learning_rate(_initial_learning_rate)
{
    lr_internal::requirePositiveFinite(learning_rate,
                                       "Initial learning rate must be finite and greater than 0.",
                                       "No_Decay::No_Decay");
}

float No_Decay::updateRate()
{
    Logger::logMessage(Input_Format{"No_Decay::updateRate: current_rate={}", learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return learning_rate;
}

void No_Decay::step(float )
{
}

void No_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    lr_internal::writePods(_output_file_stream, "No_Decay::saveCheckpoint", learning_rate);
}

void No_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "No_Decay::loadCheckpoint");
    float lr = 0.0f;
    lr_internal::readPodsOrThrow(_input_file_stream, "No_Decay::loadCheckpoint: Failed to read checkpoint", lr);
    lr_internal::requirePositiveFinite(lr, "No_Decay::loadCheckpoint: Invalid learning_rate in checkpoint");
    learning_rate = lr;
}

void No_Decay::setLearningRate(float _learning_rate)
{
    lr_internal::requirePositiveFinite(_learning_rate,
                                       "No_Decay::setLearningRate: learning_rate must be finite and greater than 0.");
    learning_rate = _learning_rate;
}

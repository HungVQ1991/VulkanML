#include "learning_rate/exponential_decay.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "helper/logger.h"
#include "lr_internal_utils.h"

void Exponential_Decay::validateParameters() const
{
    constexpr const char *ctx = "Exponential_Decay::validateParameters";
    lr_internal::requirePositiveFinite(learning_rate, "Initial learning rate must be finite and greater than 0.", ctx);
    lr_internal::requireMinLrInBounds(minimum_learning_rate, learning_rate, "minimum_learning_rate is out of valid bounds.", ctx);
    lr_internal::requireUnitOpenInterval(decay_rate, "decay_rate should be in range (0, 1).", ctx);
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
    const int eff_epoch = std::max(current_epoch, 0);
    const float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(eff_epoch));
    if (calculated_rate < minimum_learning_rate)
    {
        lr_internal::logClampedBelowMin("Exponential_Decay::updateRate");
    }
    current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
    Logger::logMessage(Input_Format{"Exponential_Decay::updateRate: epoch={}, current_rate={}",
                                    current_epoch, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return current_learning_rate;
}

void Exponential_Decay::step(float )
{
    current_epoch++;
    updateRate();
}

void Exponential_Decay::reset()
{
    current_epoch = 0;
    current_learning_rate = learning_rate;
}

void Exponential_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    lr_internal::writePods(_output_file_stream, "Exponential_Decay::saveCheckpoint",
                           learning_rate, minimum_learning_rate, decay_rate,
                           current_learning_rate, current_epoch);
}

void Exponential_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "Exponential_Decay::loadCheckpoint");
    float lr = 0.0f, min_lr = 0.0f, dr = 0.0f, cur_lr = 0.0f;
    int cur_ep = 0;
    lr_internal::readPodsOrThrow(_input_file_stream, "Exponential_Decay::loadCheckpoint: Failed to read checkpoint",
                                 lr, min_lr, dr, cur_lr, cur_ep);
    if (!lr_internal::isValidBaseParams(lr, min_lr, cur_lr, cur_ep) || !std::isfinite(dr) || dr <= 0.0f || dr >= 1.0f)
    {
        throw std::invalid_argument("Exponential_Decay::loadCheckpoint: Invalid scheduler parameters in checkpoint");
    }
    learning_rate = lr;
    minimum_learning_rate = min_lr;
    decay_rate = dr;
    current_learning_rate = cur_lr;
    current_epoch = cur_ep;
}

float Exponential_Decay::getCurrentLearningRate() const noexcept
{
    return current_learning_rate;
}

void Exponential_Decay::setMinimumLearningRate(float _minimum_learning_rate)
{
    lr_internal::requireMinLrInBounds(_minimum_learning_rate, learning_rate,
                                      "Exponential_Decay::setMinimumLearningRate: minimum_learning_rate out of valid bounds.");
    minimum_learning_rate = _minimum_learning_rate;
    updateRate();
}

void Exponential_Decay::setCurrentLearningRate(float _rate)
{
    lr_internal::requireNonNegativeFinite(_rate,
                                          "Exponential_Decay::setCurrentLearningRate: rate must be finite and >= 0.");
    current_learning_rate = _rate;
}

void Exponential_Decay::setLearningRate(float _learning_rate)
{
    lr_internal::requireLearningRateSetterBounds(_learning_rate, minimum_learning_rate,
                                                 "Exponential_Decay::setLearningRate: learning_rate out of valid bounds.");
    learning_rate = _learning_rate;
    updateRate();
}

void Exponential_Decay::setCurrentEpoch(int _current_epoch)
{
    lr_internal::requireNonNegativeInt(_current_epoch,
                                       "Exponential_Decay::setCurrentEpoch: current_epoch must be >= 0.");
    current_epoch = _current_epoch;
    updateRate();
}

void Exponential_Decay::setDecayRate(float _decay_rate)
{
    lr_internal::requireUnitOpenInterval(_decay_rate,
                                         "Exponential_Decay::setDecayRate: decay_rate must be in (0, 1).");
    decay_rate = _decay_rate;
    updateRate();
}

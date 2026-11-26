#include "learning_rate/polynomial_decay.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "helper/logger.h"
#include "lr_internal_utils.h"

void Polynomial_Decay::validateParameters() const
{
    constexpr const char *ctx = "Polynomial_Decay::validateParameters";
    lr_internal::requirePositiveFinite(learning_rate, "Initial learning rate must be finite and greater than 0.", ctx);
    lr_internal::requireMinLrInBounds(minimum_learning_rate, learning_rate, "minimum_learning_rate is out of valid bounds.", ctx);
    lr_internal::requirePositiveInt(maximum_epoch, "maximum_epoch must be greater than 0.", ctx);
    lr_internal::requirePositiveFinite(power, "power must be finite and greater than 0.", ctx);
}

Polynomial_Decay::Polynomial_Decay(float _initial_learning_rate, float _minimum_learning_rate, int _maximum_epoch, float _power)
    : learning_rate(_initial_learning_rate),
      minimum_learning_rate(_minimum_learning_rate),
      current_learning_rate(_initial_learning_rate),
      power(_power),
      maximum_epoch(_maximum_epoch)
{
    validateParameters();
}

float Polynomial_Decay::updateRate()
{
    const int eff_epoch = std::max(current_epoch, 0);
    const int eff_max = std::max(maximum_epoch, 1);
    if (eff_epoch >= eff_max)
    {
        Logger::logMessage("Polynomial_Decay::updateRate: current_epoch reached or exceeded maximum_epoch, rate clamped to minimum_learning_rate.",
                           Log_Level::LOG_DEBUG, false, 0, Log_Feature::LR_SCHEDULER);
        current_learning_rate = minimum_learning_rate;
    }
    else
    {
        const float progress = std::clamp(1.0f - (static_cast<float>(eff_epoch) / static_cast<float>(eff_max)), 0.0f, 1.0f);
        const float decayed = (power == 1.0f) ? progress : std::pow(progress, power);
        current_learning_rate = (learning_rate - minimum_learning_rate) * decayed + minimum_learning_rate;
    }
    Logger::logMessage(Input_Format{"Polynomial_Decay::updateRate: epoch={}/{}, current_rate={}",
                                    current_epoch, maximum_epoch, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return current_learning_rate;
}

void Polynomial_Decay::step(float )
{
    current_epoch++;
    updateRate();
}

void Polynomial_Decay::reset()
{
    current_epoch = 0;
    current_learning_rate = learning_rate;
}

void Polynomial_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    lr_internal::writePods(_output_file_stream, "Polynomial_Decay::saveCheckpoint",
                           POLY_CHECKPOINT_MAGIC_V2, learning_rate, minimum_learning_rate,
                           current_learning_rate, maximum_epoch, current_epoch, power);
}

void Polynomial_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "Polynomial_Decay::loadCheckpoint");
    uint32_t first_word = 0;
    lr_internal::readPodsOrThrow(_input_file_stream, "Polynomial_Decay::loadCheckpoint: Failed to read checkpoint header", first_word);

    float lr = 0.0f, min_lr = 0.0f, cur_lr = 0.0f, pwr = 1.0f;
    int max_ep = 0, cur_ep = 0;
    if (first_word == POLY_CHECKPOINT_MAGIC_V2)
    {
        lr_internal::readPodsOrThrow(_input_file_stream, "Polynomial_Decay::loadCheckpoint: Failed to read checkpoint",
                                     lr, min_lr, cur_lr, max_ep, cur_ep, pwr);
    }
    else
    {
        std::memcpy(&lr, &first_word, sizeof(lr));
        lr_internal::readPodsOrThrow(_input_file_stream, "Polynomial_Decay::loadCheckpoint: Failed to read checkpoint",
                                     min_lr, cur_lr, max_ep, cur_ep);
    }

    if (!lr_internal::isValidBaseParams(lr, min_lr, cur_lr, cur_ep) || !std::isfinite(pwr) || pwr <= 0.0f || max_ep <= 0)
    {
        throw std::invalid_argument("Polynomial_Decay::loadCheckpoint: Invalid scheduler parameters in checkpoint");
    }
    learning_rate = lr;
    minimum_learning_rate = min_lr;
    current_learning_rate = cur_lr;
    maximum_epoch = max_ep;
    current_epoch = cur_ep;
    power = pwr;
}

void Polynomial_Decay::setMaxEpoch(int _maximum_epoch)
{
    lr_internal::requirePositiveInt(_maximum_epoch, "Polynomial_Decay::setMaxEpoch: maximum_epoch must be > 0.",
                                    "Polynomial_Decay::setMaxEpoch");
    maximum_epoch = _maximum_epoch;
    updateRate();
    Logger::logMessage(Input_Format{"Polynomial_Decay::setMaxEpoch: updated maximum_epoch={}, current_rate={}",
                                    maximum_epoch, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
}

void Polynomial_Decay::setMinimumLearningRate(float _minimum_learning_rate)
{
    lr_internal::requireMinLrInBounds(_minimum_learning_rate, learning_rate,
                                      "Polynomial_Decay::setMinimumLearningRate: minimum_learning_rate out of valid bounds.");
    minimum_learning_rate = _minimum_learning_rate;
    updateRate();
}

void Polynomial_Decay::setCurrentRate(float _current_rate)
{
    lr_internal::requireNonNegativeFinite(_current_rate,
                                          "Polynomial_Decay::setCurrentRate: current_rate must be finite and >= 0.");
    current_learning_rate = _current_rate;
}

void Polynomial_Decay::setLearningRate(float _learning_rate)
{
    lr_internal::requireLearningRateSetterBounds(_learning_rate, minimum_learning_rate,
                                                 "Polynomial_Decay::setLearningRate: learning_rate out of valid bounds.");
    learning_rate = _learning_rate;
    updateRate();
}

void Polynomial_Decay::setPower(float _power)
{
    lr_internal::requirePositiveFinite(_power, "Polynomial_Decay::setPower: power must be finite and > 0.");
    power = _power;
    updateRate();
}

void Polynomial_Decay::setCurrentEpoch(int _current_epoch)
{
    lr_internal::requireNonNegativeInt(_current_epoch,
                                       "Polynomial_Decay::setCurrentEpoch: current_epoch must be >= 0.");
    current_epoch = _current_epoch;
    updateRate();
}

#include "learning_rate/cosine_annealing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>

#include "helper/logger.h"
#include "lr_internal_utils.h"

void Cosine_Annealing::validateParameters() const
{
    constexpr const char *ctx = "Cosine_Annealing::validateParameters";
    lr_internal::requirePositiveFinite(learning_rate, "Initial learning rate must be finite and greater than 0.", ctx);
    lr_internal::requireMinLrInBounds(minimum_learning_rate, learning_rate, "minimum_learning_rate is out of valid bounds.", ctx);
    lr_internal::requirePositiveInt(maximum_epoch, "maximum_epoch must be greater than 0.", ctx);
    if (warmup_steps < 0 || warmup_steps >= maximum_epoch)
    {
        lr_internal::throwInvalid("warmup_steps must be >= 0 and < maximum_epoch.", ctx);
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
    if (_warmup_steps < 0 || _warmup_steps >= maximum_epoch)
    {
        lr_internal::throwInvalid("warmup_steps must be >= 0 and < maximum_epoch.", "Cosine_Annealing::setWarmupSteps");
    }
    warmup_steps = _warmup_steps;
    updateRate();
}

float Cosine_Annealing::updateRate()
{
    const int eff_epoch = std::max(current_epoch, 0);
    if (eff_epoch >= maximum_epoch)
    {
        Logger::logMessage("Cosine_Annealing::updateRate: current_epoch reached or exceeded maximum_epoch, rate clamped to minimum_learning_rate.",
                           Log_Level::LOG_DEBUG, false, 0, Log_Feature::LR_SCHEDULER);
        current_learning_rate = minimum_learning_rate;
    }
    else if (warmup_steps > 0 && eff_epoch < warmup_steps)
    {
        const float progress = static_cast<float>(eff_epoch) / static_cast<float>(warmup_steps);
        current_learning_rate = minimum_learning_rate + progress * (learning_rate - minimum_learning_rate);
    }
    else
    {
        const int decay_epoch = eff_epoch - warmup_steps;
        const int total_decay_epochs = std::max(maximum_epoch - warmup_steps, 1);
        const float progress = static_cast<float>(decay_epoch) / static_cast<float>(total_decay_epochs);
        const float cosine_value = std::cos(progress * std::numbers::pi_v<float>);
        current_learning_rate = minimum_learning_rate + 0.5f * (learning_rate - minimum_learning_rate) * (1.0f + cosine_value);
    }
    Logger::logMessage(Input_Format{"Cosine_Annealing::updateRate: epoch={}/{}, current_rate={}",
                                    current_epoch, maximum_epoch, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return current_learning_rate;
}

void Cosine_Annealing::step(float )
{
    current_epoch++;
    updateRate();
}

void Cosine_Annealing::reset()
{
    current_epoch = 0;
    current_learning_rate = (warmup_steps > 0) ? minimum_learning_rate : learning_rate;
}

void Cosine_Annealing::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    static constexpr uint32_t COSINE_CHECKPOINT_MAGIC_V2 = 0xFFC04132u;
    lr_internal::writePods(_output_file_stream, "Cosine_Annealing::saveCheckpoint",
                           COSINE_CHECKPOINT_MAGIC_V2, learning_rate, minimum_learning_rate,
                           current_learning_rate, maximum_epoch, current_epoch, warmup_steps);
}

void Cosine_Annealing::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "Cosine_Annealing::loadCheckpoint");
    static constexpr uint32_t COSINE_CHECKPOINT_MAGIC_V2 = 0xFFC04132u;
    uint32_t first_word = 0;
    lr_internal::readPodsOrThrow(_input_file_stream, "Cosine_Annealing::loadCheckpoint: Failed to read checkpoint header", first_word);

    float lr = learning_rate, min_lr = minimum_learning_rate, cur_lr = current_learning_rate;
    int max_ep = maximum_epoch, cur_ep = current_epoch, warm = 0;
    if (first_word == COSINE_CHECKPOINT_MAGIC_V2)
    {
        lr_internal::readPodsOrThrow(_input_file_stream, "Cosine_Annealing::loadCheckpoint: Failed to read checkpoint body",
                                     lr, min_lr, cur_lr, max_ep, cur_ep, warm);
    }
    else
    {
        std::memcpy(&lr, &first_word, sizeof(lr));
        lr_internal::readPodsOrThrow(_input_file_stream, "Cosine_Annealing::loadCheckpoint: Failed to read checkpoint body",
                                     min_lr, cur_lr, max_ep, cur_ep);
    }

    if (!lr_internal::isValidBaseParams(lr, min_lr, cur_lr, cur_ep) || max_ep <= 0 || warm < 0 || warm >= max_ep)
    {
        throw std::invalid_argument("Cosine_Annealing::loadCheckpoint: Invalid scheduler parameters in checkpoint");
    }
    learning_rate = lr;
    minimum_learning_rate = min_lr;
    current_learning_rate = cur_lr;
    maximum_epoch = max_ep;
    current_epoch = cur_ep;
    warmup_steps = warm;
}

void Cosine_Annealing::setMaxEpoch(int _maximum_epoch)
{
    if (_maximum_epoch <= 0 || warmup_steps >= _maximum_epoch)
    {
        Logger::logMessage("Cosine_Annealing::setMaxEpoch: maximum_epoch must be greater than 0 and greater than warmup_steps.",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::LR_SCHEDULER);
        throw std::invalid_argument("Cosine_Annealing::setMaxEpoch: maximum_epoch must be > 0 and > warmup_steps.");
    }
    maximum_epoch = _maximum_epoch;
    updateRate();
    Logger::logMessage(Input_Format{"Cosine_Annealing::setMaxEpoch: updated maximum_epoch={}, current_rate={}",
                                    maximum_epoch, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
}

void Cosine_Annealing::setMinimumLearningRate(float _minimum_learning_rate)
{
    lr_internal::requireMinLrInBounds(_minimum_learning_rate, learning_rate,
                                      "Cosine_Annealing::setMinimumLearningRate: minimum_learning_rate out of valid bounds.");
    minimum_learning_rate = _minimum_learning_rate;
    updateRate();
}

void Cosine_Annealing::setCurrentRate(float _current_rate)
{
    lr_internal::requireNonNegativeFinite(_current_rate,
                                          "Cosine_Annealing::setCurrentRate: current_rate must be finite and >= 0.");
    current_learning_rate = _current_rate;
}

void Cosine_Annealing::setLearningRate(float _learning_rate)
{
    lr_internal::requireLearningRateSetterBounds(_learning_rate, minimum_learning_rate,
                                                 "Cosine_Annealing::setLearningRate: learning_rate out of valid bounds.");
    learning_rate = _learning_rate;
    updateRate();
}

void Cosine_Annealing::setCurrentEpoch(int _current_epoch)
{
    lr_internal::requireNonNegativeInt(_current_epoch,
                                       "Cosine_Annealing::setCurrentEpoch: current_epoch must be >= 0.");
    current_epoch = _current_epoch;
    updateRate();
}

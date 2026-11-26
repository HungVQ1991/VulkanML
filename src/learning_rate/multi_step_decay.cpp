#include "learning_rate/multi_step_decay.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "helper/logger.h"
#include "lr_internal_utils.h"

void Multi_Step_Decay::validateParameters()
{
    constexpr const char *ctx = "Multi_Step_Decay::validateParameters";
    lr_internal::requirePositiveFinite(learning_rate, "Initial learning rate must be finite and greater than 0.", ctx);
    lr_internal::requireMinLrInBounds(minimum_learning_rate, learning_rate, "minimum_learning_rate is out of valid bounds.", ctx);
    lr_internal::requireUnitOpenInterval(decay_rate, "decay_rate should be in range (0, 1).", ctx);
    if (decay_epochs.empty())
    {
        lr_internal::throwInvalid("decay_epochs cannot be empty.", ctx);
    }
    for (float epoch_milestone : decay_epochs)
    {
        lr_internal::requirePositiveFinite(epoch_milestone, "decay_epochs must contain finite positive values.", ctx);
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
    const int eff_epoch = std::max(current_epoch, 0);
    int milestone_count = 0;
    for (float epoch_milestone : decay_epochs)
    {
        if (static_cast<float>(eff_epoch) >= epoch_milestone)
        {
            milestone_count++;
        }
    }
    const float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(milestone_count));
    if (calculated_rate < minimum_learning_rate)
    {
        lr_internal::logClampedBelowMin("Multi_Step_Decay::updateRate");
    }
    current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
    Logger::logMessage(Input_Format{"Multi_Step_Decay::updateRate: epoch={}, milestones_hit={}, current_rate={}",
                                    current_epoch, milestone_count, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return current_learning_rate;
}

void Multi_Step_Decay::step(float )
{
    current_epoch++;
    updateRate();
}

void Multi_Step_Decay::reset()
{
    current_epoch = 0;
    current_learning_rate = learning_rate;
}

void Multi_Step_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    const uint32_t vector_size = static_cast<uint32_t>(decay_epochs.size());
    lr_internal::requireOpenOut(_output_file_stream, "Multi_Step_Decay::saveCheckpoint");
    lr_internal::writePods(_output_file_stream, "Multi_Step_Decay::saveCheckpoint",
                           learning_rate, minimum_learning_rate, decay_rate,
                           current_learning_rate, current_epoch, vector_size);
    if (vector_size > 0)
    {
        _output_file_stream.write(reinterpret_cast<const char *>(decay_epochs.data()), vector_size * sizeof(float));
        lr_internal::requireGoodStream(_output_file_stream, "Multi_Step_Decay::saveCheckpoint: Failed to write checkpoint");
    }
}

void Multi_Step_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "Multi_Step_Decay::loadCheckpoint");
    float lr = 0.0f, min_lr = 0.0f, dr = 0.0f, cur_lr = 0.0f;
    int cur_ep = 0;
    uint32_t vector_size = 0;
    lr_internal::readPodsOrThrow(_input_file_stream, "Multi_Step_Decay::loadCheckpoint: Failed to read checkpoint header",
                                 lr, min_lr, dr, cur_lr, cur_ep, vector_size);
    if (vector_size == 0 || vector_size > 1000000u)
    {
        throw std::runtime_error("Multi_Step_Decay::loadCheckpoint: Invalid decay_epochs size in checkpoint");
    }

    std::vector<float> tmp_epochs(vector_size);
    _input_file_stream.read(reinterpret_cast<char *>(tmp_epochs.data()), static_cast<std::streamsize>(vector_size * sizeof(float)));
    lr_internal::requireGoodStream(_input_file_stream, "Multi_Step_Decay::loadCheckpoint: Failed to read decay_epochs array");

    if (!lr_internal::isValidBaseParams(lr, min_lr, cur_lr, cur_ep) || !std::isfinite(dr) || dr <= 0.0f || dr >= 1.0f)
    {
        throw std::invalid_argument("Multi_Step_Decay::loadCheckpoint: Invalid scheduler parameters in checkpoint");
    }
    for (float m : tmp_epochs)
    {
        lr_internal::requirePositiveFinite(m, "Multi_Step_Decay::loadCheckpoint: Invalid milestone in checkpoint");
    }
    std::sort(tmp_epochs.begin(), tmp_epochs.end());

    learning_rate = lr;
    minimum_learning_rate = min_lr;
    decay_rate = dr;
    current_learning_rate = cur_lr;
    current_epoch = cur_ep;
    decay_epochs = std::move(tmp_epochs);
}

const std::vector<float> &Multi_Step_Decay::getDecayEpochs() const noexcept
{
    return decay_epochs;
}

float Multi_Step_Decay::getCurrentLearningRate() const noexcept
{
    return current_learning_rate;
}

void Multi_Step_Decay::setDecayEpochs(std::vector<float> _decay_epochs)
{
    if (_decay_epochs.empty())
    {
        throw std::invalid_argument("Multi_Step_Decay::setDecayEpochs: decay_epochs cannot be empty.");
    }
    for (float epoch_milestone : _decay_epochs)
    {
        lr_internal::requirePositiveFinite(epoch_milestone,
                                           "Multi_Step_Decay::setDecayEpochs: decay_epochs must contain finite positive values.");
    }
    std::sort(_decay_epochs.begin(), _decay_epochs.end());
    decay_epochs = std::move(_decay_epochs);
    updateRate();
}

void Multi_Step_Decay::setMinimumLearningRate(float _min_lr)
{
    lr_internal::requireMinLrInBounds(_min_lr, learning_rate,
                                      "Multi_Step_Decay::setMinimumLearningRate: minimum_learning_rate out of valid bounds.");
    minimum_learning_rate = _min_lr;
    updateRate();
}

void Multi_Step_Decay::setCurrentLearningRate(float _rate)
{
    lr_internal::requireNonNegativeFinite(_rate,
                                          "Multi_Step_Decay::setCurrentLearningRate: rate must be finite and >= 0.");
    current_learning_rate = _rate;
}

void Multi_Step_Decay::setLearningRate(float _learning_rate)
{
    lr_internal::requireLearningRateSetterBounds(_learning_rate, minimum_learning_rate,
                                                 "Multi_Step_Decay::setLearningRate: learning_rate out of valid bounds.");
    learning_rate = _learning_rate;
    updateRate();
}

void Multi_Step_Decay::setCurrentEpoch(int _epoch)
{
    lr_internal::requireNonNegativeInt(_epoch, "Multi_Step_Decay::setCurrentEpoch: epoch must be >= 0.");
    current_epoch = _epoch;
    updateRate();
}

void Multi_Step_Decay::setDecayRate(float _decay_rate)
{
    lr_internal::requireUnitOpenInterval(_decay_rate,
                                         "Multi_Step_Decay::setDecayRate: decay_rate must be in (0, 1).");
    decay_rate = _decay_rate;
    updateRate();
}

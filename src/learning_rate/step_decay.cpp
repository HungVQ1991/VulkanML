#include "learning_rate/step_decay.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "helper/logger.h"
#include "lr_internal_utils.h"

void Step_Decay::validateParameters() const
{
    constexpr const char *ctx = "Step_Decay::validateParameters";
    lr_internal::requirePositiveFinite(learning_rate, "Initial learning rate must be finite and greater than 0.", ctx);
    lr_internal::requireMinLrInBounds(minimum_learning_rate, learning_rate, "minimum_learning_rate is out of valid bounds.", ctx);
    lr_internal::requirePositiveInt(step_size, "step_size must be greater than 0.", ctx);
    lr_internal::requireUnitOpenInterval(decay_rate, "decay_rate should be in range (0, 1).", ctx);
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
    const int eff_step = std::max(step_size, 1);
    const int eff_epoch = std::max(current_epoch, 0);
    const int steps_taken = eff_epoch / eff_step;
    const float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(steps_taken));
    if (calculated_rate < minimum_learning_rate)
    {
        lr_internal::logClampedBelowMin("Step_Decay::updateRate");
    }
    current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
    Logger::logMessage(Input_Format{"Step_Decay::updateRate: epoch={}, steps_taken={}, current_rate={}",
                                    current_epoch, steps_taken, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return current_learning_rate;
}

void Step_Decay::step(float )
{
    current_epoch++;
    updateRate();
}

void Step_Decay::reset()
{
    current_epoch = 0;
    current_learning_rate = learning_rate;
}

void Step_Decay::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    lr_internal::writePods(_output_file_stream, "Step_Decay::saveCheckpoint",
                           learning_rate, minimum_learning_rate, decay_rate,
                           current_learning_rate, step_size, current_epoch);
}

void Step_Decay::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "Step_Decay::loadCheckpoint");
    float lr = 0.0f, min_lr = 0.0f, dr = 0.0f, cur_lr = 0.0f;
    int ss = 0, cur_ep = 0;
    lr_internal::readPodsOrThrow(_input_file_stream, "Step_Decay::loadCheckpoint: Failed to read checkpoint",
                                 lr, min_lr, dr, cur_lr, ss, cur_ep);
    if (!lr_internal::isValidBaseParams(lr, min_lr, cur_lr, cur_ep) ||
        !std::isfinite(dr) || dr <= 0.0f || dr >= 1.0f || ss <= 0)
    {
        throw std::invalid_argument("Step_Decay::loadCheckpoint: Invalid scheduler parameters in checkpoint");
    }
    learning_rate = lr;
    minimum_learning_rate = min_lr;
    decay_rate = dr;
    current_learning_rate = cur_lr;
    step_size = ss;
    current_epoch = cur_ep;
}

float Step_Decay::getCurrentLearningRate() const noexcept
{
    return current_learning_rate;
}

void Step_Decay::setMinimumLearningRate(float _min_lr)
{
    lr_internal::requireMinLrInBounds(_min_lr, learning_rate,
                                      "Step_Decay::setMinimumLearningRate: minimum_learning_rate out of valid bounds.");
    minimum_learning_rate = _min_lr;
    updateRate();
}

void Step_Decay::setCurrentLearningRate(float _rate)
{
    lr_internal::requireNonNegativeFinite(_rate,
                                          "Step_Decay::setCurrentLearningRate: rate must be finite and >= 0.");
    current_learning_rate = _rate;
}

void Step_Decay::setLearningRate(float _learning_rate)
{
    lr_internal::requireLearningRateSetterBounds(_learning_rate, minimum_learning_rate,
                                                 "Step_Decay::setLearningRate: learning_rate out of valid bounds.");
    learning_rate = _learning_rate;
    updateRate();
}

void Step_Decay::setCurrentEpoch(int _epoch)
{
    lr_internal::requireNonNegativeInt(_epoch, "Step_Decay::setCurrentEpoch: epoch must be >= 0.");
    current_epoch = _epoch;
    updateRate();
}

void Step_Decay::setDecayRate(float _decay_rate)
{
    lr_internal::requireUnitOpenInterval(_decay_rate,
                                         "Step_Decay::setDecayRate: decay_rate must be in (0, 1).");
    decay_rate = _decay_rate;
    updateRate();
}

void Step_Decay::setStepSize(int _step_size)
{
    lr_internal::requirePositiveInt(_step_size, "Step_Decay::setStepSize: step_size must be > 0.");
    step_size = _step_size;
    updateRate();
}

#include "learning_rate/reduce_on_plateau.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "helper/logger.h"
#include "lr_internal_utils.h"

void Reduce_On_Plateau::validateParameters() const
{
    constexpr const char *ctx = "Reduce_On_Plateau::validateParameters";
    lr_internal::requirePositiveFinite(learning_rate, "Initial learning rate must be finite and greater than 0.", ctx);
    lr_internal::requireMinLrInBounds(minimum_learning_rate, learning_rate, "minimum_learning_rate is out of valid bounds.", ctx);
    lr_internal::requirePositiveInt(patience, "patience must be greater than 0.", ctx);
    lr_internal::requireUnitOpenInterval(decay_rate, "decay_rate should be in range (0, 1).", ctx);
    lr_internal::requireNonNegativeFinite(threshold, "threshold must be finite and >= 0.", ctx);
    lr_internal::requireNonNegativeInt(cooldown, "cooldown must be >= 0.", ctx);
}

Reduce_On_Plateau::Reduce_On_Plateau(float _initial_learning_rate, float _minimum_learning_rate, float _decay_rate, int _patience, bool _is_higher_better, float _threshold, int _cooldown)
    : learning_rate(_initial_learning_rate),
      minimum_learning_rate(_minimum_learning_rate),
      decay_rate(_decay_rate),
      current_learning_rate(_initial_learning_rate),
      threshold(_threshold),
      patience(_patience),
      cooldown(_cooldown),
      is_higher_better(_is_higher_better)
{
    validateParameters();
}

float Reduce_On_Plateau::updateRate()
{
    const int eff_reductions = std::max(reductions_count, 0);
    const float calculated_rate = learning_rate * std::pow(decay_rate, static_cast<float>(eff_reductions));
    if (calculated_rate < minimum_learning_rate)
    {
        lr_internal::logClampedBelowMin("Reduce_On_Plateau::updateRate");
    }
    current_learning_rate = std::max(minimum_learning_rate, calculated_rate);
    Logger::logMessage(Input_Format{"Reduce_On_Plateau::updateRate: reductions_count={}, current_rate={}",
                                    reductions_count, current_learning_rate},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
    return current_learning_rate;
}

void Reduce_On_Plateau::step(float _current_value)
{
    auto apply_step_reduction = [this]()
    {
        bad_epochs_count = 0;
        if (current_learning_rate > minimum_learning_rate)
        {
            reductions_count++;
            current_learning_rate = std::max(minimum_learning_rate, current_learning_rate * decay_rate);
            cooldown_counter = cooldown;
            Logger::logMessage(Input_Format{"Reduce_On_Plateau::step: Patience exhausted, triggering learning rate reduction #{} (rate={})",
                                            reductions_count, current_learning_rate},
                               Log_Level::LOG_WARNING, true, 0, Log_Feature::LR_SCHEDULER);
        }
    };

    if (!std::isfinite(_current_value))
    {
        Logger::logMessage("Reduce_On_Plateau::step: Non-finite metric encountered, ignoring best_metric update.",
                           Log_Level::LOG_WARNING, false, 0, Log_Feature::LR_SCHEDULER);
        if (!is_first_step)
        {
            if (cooldown_counter > 0)
            {
                cooldown_counter--;
            }
            else if (++bad_epochs_count >= patience)
            {
                apply_step_reduction();
            }
        }
        current_epoch++;
        return;
    }

    if (is_first_step)
    {
        best_metric = _current_value;
        is_first_step = false;
    }
    else
    {
        const bool is_improved = is_higher_better ? (_current_value > best_metric + threshold)
                                                  : (_current_value < best_metric - threshold);
        if (is_improved)
        {
            best_metric = _current_value;
            bad_epochs_count = 0;
        }
        if (cooldown_counter > 0)
        {
            cooldown_counter--;
        }
        else if (!is_improved)
        {
            bad_epochs_count++;
            Logger::logMessage(Input_Format{"Reduce_On_Plateau::step: Plateau detected, bad_epochs_count={}/{}",
                                            bad_epochs_count, patience},
                               Log_Level::LOG_WARNING, false, 0, Log_Feature::LR_SCHEDULER);
            if (bad_epochs_count >= patience)
            {
                apply_step_reduction();
            }
        }
    }
    current_epoch++;
    Logger::logMessage(Input_Format{"Reduce_On_Plateau::step: epoch={}, current_value={}, best_metric={}",
                                    current_epoch, _current_value, best_metric},
                       Log_Level::LOG_DEBUG, true, 0, Log_Feature::LR_SCHEDULER);
}

void Reduce_On_Plateau::reset()
{
    current_epoch = 0;
    bad_epochs_count = 0;
    reductions_count = 0;
    cooldown_counter = 0;
    best_metric = 0.0f;
    is_first_step = true;
    current_learning_rate = learning_rate;
}

void Reduce_On_Plateau::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    const uint8_t higher_u8 = is_higher_better ? 1u : 0u;
    const uint8_t first_u8 = is_first_step ? 1u : 0u;
    lr_internal::writePods(_output_file_stream, "Reduce_On_Plateau::saveCheckpoint",
                           ROP_CHECKPOINT_MAGIC_V2, learning_rate, minimum_learning_rate, decay_rate,
                           current_learning_rate, patience, current_epoch, bad_epochs_count, reductions_count,
                           best_metric, higher_u8, first_u8, threshold, cooldown, cooldown_counter);
}

void Reduce_On_Plateau::loadCheckpoint(std::ifstream &_input_file_stream)
{
    lr_internal::requireOpenIn(_input_file_stream, "Reduce_On_Plateau::loadCheckpoint");
    uint32_t first_word = 0;
    lr_internal::readPodsOrThrow(_input_file_stream, "Reduce_On_Plateau::loadCheckpoint: Failed to read checkpoint header", first_word);

    float lr = 0.0f, min_lr = 0.0f, dr = 0.0f, cur_lr = 0.0f, bm = 0.0f, thr = 0.0f;
    int pat = 0, cur_ep = 0, bad_ep = 0, red_cnt = 0, cd = 0, cd_cnt = 0;
    bool higher = false, first = true;

    if (first_word == ROP_CHECKPOINT_MAGIC_V2)
    {
        uint8_t higher_u8 = 0, first_u8 = 1;
        lr_internal::readPodsOrThrow(_input_file_stream, "Reduce_On_Plateau::loadCheckpoint: Failed to read checkpoint",
                                     lr, min_lr, dr, cur_lr, pat, cur_ep, bad_ep, red_cnt, bm,
                                     higher_u8, first_u8, thr, cd, cd_cnt);
        higher = (higher_u8 != 0);
        first = (first_u8 != 0);
    }
    else
    {
        std::memcpy(&lr, &first_word, sizeof(lr));
        lr_internal::readPodsOrThrow(_input_file_stream, "Reduce_On_Plateau::loadCheckpoint: Failed to read checkpoint",
                                     min_lr, dr, cur_lr, pat, cur_ep, bad_ep, red_cnt, bm, higher, first);
    }

    if (!lr_internal::isValidBaseParams(lr, min_lr, cur_lr, cur_ep) ||
        !std::isfinite(dr) || dr <= 0.0f || dr >= 1.0f ||
        !std::isfinite(thr) || thr < 0.0f ||
        pat <= 0 || bad_ep < 0 || red_cnt < 0 || cd < 0 || cd_cnt < 0 ||
        (!first && !std::isfinite(bm)))
    {
        throw std::invalid_argument("Reduce_On_Plateau::loadCheckpoint: Invalid scheduler parameters in checkpoint");
    }

    learning_rate = lr;
    minimum_learning_rate = min_lr;
    decay_rate = dr;
    current_learning_rate = cur_lr;
    patience = pat;
    current_epoch = cur_ep;
    bad_epochs_count = bad_ep;
    reductions_count = red_cnt;
    best_metric = bm;
    is_higher_better = higher;
    is_first_step = first;
    threshold = thr;
    cooldown = cd;
    cooldown_counter = cd_cnt;
}

bool Reduce_On_Plateau::isFirstStep() const noexcept
{
    return is_first_step;
}

void Reduce_On_Plateau::setMinimumLearningRate(float _minimum_learning_rate)
{
    lr_internal::requireMinLrInBounds(_minimum_learning_rate, learning_rate,
                                      "Reduce_On_Plateau::setMinimumLearningRate: minimum_learning_rate out of valid bounds.");
    minimum_learning_rate = _minimum_learning_rate;
    current_learning_rate = std::max(current_learning_rate, minimum_learning_rate);
}

void Reduce_On_Plateau::setCurrentRate(float _current_rate)
{
    lr_internal::requireNonNegativeFinite(_current_rate,
                                          "Reduce_On_Plateau::setCurrentRate: current_rate must be finite and >= 0.");
    current_learning_rate = _current_rate;
}

void Reduce_On_Plateau::setLearningRate(float _learning_rate)
{
    lr_internal::requireLearningRateSetterBounds(_learning_rate, minimum_learning_rate,
                                                 "Reduce_On_Plateau::setLearningRate: learning_rate out of valid bounds.");
    learning_rate = _learning_rate;
    updateRate();
}

void Reduce_On_Plateau::setDecayRate(float _decay_rate)
{
    lr_internal::requireUnitOpenInterval(_decay_rate,
                                         "Reduce_On_Plateau::setDecayRate: decay_rate must be in (0, 1).");
    decay_rate = _decay_rate;
    updateRate();
}

void Reduce_On_Plateau::setBestMetric(float _best_metric)
{
    lr_internal::requireFinite(_best_metric, "Reduce_On_Plateau::setBestMetric: best_metric must be finite.");
    best_metric = _best_metric;
}

void Reduce_On_Plateau::setBadEpochsCount(int _bad_epochs_count)
{
    lr_internal::requireNonNegativeInt(_bad_epochs_count,
                                       "Reduce_On_Plateau::setBadEpochsCount: bad_epochs_count must be >= 0.");
    bad_epochs_count = _bad_epochs_count;
}

void Reduce_On_Plateau::setReductionsCount(int _reductions_count)
{
    lr_internal::requireNonNegativeInt(_reductions_count,
                                       "Reduce_On_Plateau::setReductionsCount: reductions_count must be >= 0.");
    reductions_count = _reductions_count;
    updateRate();
}

void Reduce_On_Plateau::setCurrentEpoch(int _current_epoch)
{
    lr_internal::requireNonNegativeInt(_current_epoch,
                                       "Reduce_On_Plateau::setCurrentEpoch: current_epoch must be >= 0.");
    current_epoch = _current_epoch;
}

void Reduce_On_Plateau::setPatience(int _patience)
{
    lr_internal::requirePositiveInt(_patience, "Reduce_On_Plateau::setPatience: patience must be > 0.");
    patience = _patience;
}

void Reduce_On_Plateau::setThreshold(float _threshold)
{
    lr_internal::requireNonNegativeFinite(_threshold,
                                          "Reduce_On_Plateau::setThreshold: threshold must be finite and >= 0.");
    threshold = _threshold;
}

void Reduce_On_Plateau::setCooldown(int _cooldown)
{
    lr_internal::requireNonNegativeInt(_cooldown, "Reduce_On_Plateau::setCooldown: cooldown must be >= 0.");
    cooldown = _cooldown;
}

void Reduce_On_Plateau::setCooldownCounter(int _cooldown_counter)
{
    lr_internal::requireNonNegativeInt(_cooldown_counter,
                                       "Reduce_On_Plateau::setCooldownCounter: cooldown_counter must be >= 0.");
    cooldown_counter = _cooldown_counter;
}

void Reduce_On_Plateau::setIsHigherBetter(bool _is_higher_better) noexcept
{
    is_higher_better = _is_higher_better;
}

void Reduce_On_Plateau::setIsFirstStep(bool _is_first_step) noexcept
{
    is_first_step = _is_first_step;
}

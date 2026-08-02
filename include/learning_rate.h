#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <format>
#include <stdexcept>
#include "math/logger.h"

enum class Decay_Mode
{
    NO_DECAY,
    STEP_DECAY,
    MULTI_STEP_DECAY,
    EXPONENTIAL_DECAY,
    COSINE_ANNEALING,
    POLYNOMIAL_DECAY,
    REDUCE_ON_PLATEAU,
    DECAY_MODE_END
};

class Learning_Rate
{
private:
    float learning_rate = 0.0f;
    float min_learning_rate = 0.0f;
    float decay_rate = 0.0f;
    float current_rate = 0.0f;
    int step_size = 0;
    int current_epoch = 0;
    int max_epoch = 0;
    std::vector<float> decay_epochs = {};

    int patience = 0;
    int bad_epochs_count = 0;
    int reductions_count = 0;
    float best_metric = 0.0f;
    bool higher_is_better = false;
    bool is_first_step = true;

    Decay_Mode mode = Decay_Mode::NO_DECAY;

    bool isEpochsValid() const
    {
        if (decay_epochs.empty())
            return false;
        for (float epoch : decay_epochs)
            if (epoch <= 0.0f)
                return false;
        return true;
    }

    void validateParameters()
    {
        if (learning_rate <= 0.0f)
            Logger::logMessage("Learning_Rate: Initial learning rate must be greater than 0.", LOG_WARNING, true);

        if (min_learning_rate < 0.0f || min_learning_rate > learning_rate)
            Logger::logMessage("Learning_Rate: min_learning_rate is out of valid bounds.", LOG_WARNING, true);

        switch (mode)
        {
        case Decay_Mode::STEP_DECAY:
            if (step_size <= 0)
                Logger::logMessage("Learning_Rate: step_size must be greater than 0 for STEP_DECAY.", LOG_ERROR, true);
            if (decay_rate <= 0.0f || decay_rate >= 1.0f)
                Logger::logMessage("Learning_Rate: decay_rate should be in range (0, 1).", LOG_WARNING, true);
            break;

        case Decay_Mode::MULTI_STEP_DECAY:
            if (!isEpochsValid())
                Logger::logMessage("Learning_Rate: decay_epochs contains invalid values or is empty for MULTI_STEP_DECAY.", LOG_ERROR, true);
            else
                std::sort(decay_epochs.begin(), decay_epochs.end());
            break;

        case Decay_Mode::EXPONENTIAL_DECAY:
            if (decay_rate <= 0.0f || decay_rate >= 1.0f)
                Logger::logMessage("Learning_Rate: decay_rate should be in range (0, 1) for EXPONENTIAL_DECAY.", LOG_WARNING, true);
            break;

        case Decay_Mode::COSINE_ANNEALING:
        case Decay_Mode::POLYNOMIAL_DECAY:
            if (max_epoch <= 0)
                Logger::logMessage("Learning_Rate: max_epoch must be greater than 0.", LOG_ERROR, true);
            break;

        case Decay_Mode::REDUCE_ON_PLATEAU:
            if (patience <= 0)
                Logger::logMessage("Learning_Rate: patience must be greater than 0 for REDUCE_ON_PLATEAU.", LOG_ERROR, true);
            if (decay_rate <= 0.0f || decay_rate >= 1.0f)
                Logger::logMessage("Learning_Rate: decay_rate should be in range (0, 1) for REDUCE_ON_PLATEAU.", LOG_WARNING, true);
            break;

        default:
            break;
        }
    }

public:
    Learning_Rate() = default;
    ~Learning_Rate() = default;

    Learning_Rate(float init_lr,
                  Decay_Mode decay_mode = Decay_Mode::NO_DECAY,
                  float lr_decay_rate = 0.1f,
                  int lr_step_size = 10,
                  int total_epochs = 30,
                  float min_lr = 0.0f)
        : learning_rate(init_lr),
          min_learning_rate(min_lr),
          decay_rate(lr_decay_rate),
          current_rate(init_lr),
          step_size(lr_step_size),
          max_epoch(total_epochs),
          mode(decay_mode)
    {
        validateParameters();
        updateRate();
    }

    Learning_Rate(float init_lr,
                  Decay_Mode decay_mode,
                  float lr_decay_rate,
                  std::vector<float> epochs_list)
        : learning_rate(init_lr),
          decay_rate(lr_decay_rate),
          current_rate(init_lr),
          decay_epochs(std::move(epochs_list)),
          mode(decay_mode)
    {
        validateParameters();
        updateRate();
    }

    Learning_Rate(float init_lr,
                  Decay_Mode decay_mode,
                  float lr_decay_rate,
                  int lr_patience,
                  float min_lr,
                  bool mode_higher_is_better)
        : learning_rate(init_lr),
          min_learning_rate(min_lr),
          decay_rate(lr_decay_rate),
          current_rate(init_lr),
          patience(lr_patience),
          higher_is_better(mode_higher_is_better),
          mode(decay_mode)
    {
        validateParameters();
        updateRate();
    }

    float updateRate()
    {
        if (learning_rate <= 0.0f)
        {
            std::string err = (learning_rate < 0.0f)
                                  ? "Learning rate must not be smaller than 0"
                                  : "Learning rate must be greater than 0";
            Logger::logMessage(err, LOG_ERROR, true);
            throw std::logic_error(err);
        }

        switch (mode)
        {
        case Decay_Mode::NO_DECAY:
            current_rate = learning_rate;
            break;

        case Decay_Mode::STEP_DECAY:
        {
            if (step_size <= 0)
            {
                current_rate = learning_rate;
            }
            else
            {
                int steps = current_epoch / step_size;
                current_rate = std::max(min_learning_rate, learning_rate * std::pow(decay_rate, static_cast<float>(steps)));
            }
            break;
        }

        case Decay_Mode::MULTI_STEP_DECAY:
        {
            if (decay_epochs.empty() || !isEpochsValid())
            {
                current_rate = learning_rate;
            }
            else
            {
                int milestone_count = 0;
                for (float epoch_milestone : decay_epochs)
                {
                    if (static_cast<float>(current_epoch) >= epoch_milestone)
                    {
                        milestone_count++;
                    }
                }
                current_rate = std::max(min_learning_rate, learning_rate * std::pow(decay_rate, static_cast<float>(milestone_count)));
            }
            break;
        }

        case Decay_Mode::EXPONENTIAL_DECAY:
            current_rate = std::max(min_learning_rate, learning_rate * std::pow(decay_rate, static_cast<float>(current_epoch)));
            break;

        case Decay_Mode::COSINE_ANNEALING:
        {
            if (max_epoch <= 0)
            {
                current_rate = learning_rate;
            }
            else if (current_epoch >= max_epoch)
            {
                current_rate = min_learning_rate;
            }
            else
            {
                float progress = static_cast<float>(current_epoch) / static_cast<float>(max_epoch);
                float cos_val = std::cos(progress * std::numbers::pi_v<float>);
                current_rate = min_learning_rate + 0.5f * (learning_rate - min_learning_rate) * (1.0f + cos_val);
            }
            break;
        }

        case Decay_Mode::POLYNOMIAL_DECAY:
        {
            if (max_epoch <= 0)
            {
                current_rate = learning_rate;
            }
            else if (current_epoch >= max_epoch)
            {
                current_rate = min_learning_rate;
            }
            else
            {
                float progress = 1.0f - (static_cast<float>(current_epoch) / static_cast<float>(max_epoch));
                current_rate = (learning_rate - min_learning_rate) * progress + min_learning_rate;
            }
            break;
        }

        case Decay_Mode::REDUCE_ON_PLATEAU:
            current_rate = std::max(min_learning_rate, learning_rate * std::pow(decay_rate, static_cast<float>(reductions_count)));
            break;

        default:
            current_rate = learning_rate;
            break;
        }

        return current_rate;
    }

    void step(float current_val = 0.0f)
    {
        if (mode == Decay_Mode::REDUCE_ON_PLATEAU)
        {
            if (is_first_step)
            {
                best_metric = current_val;
                is_first_step = false;
            }
            else
            {
                bool is_improved = higher_is_better ? (current_val > best_metric) : (current_val < best_metric);
                if (is_improved)
                {
                    best_metric = current_val;
                    bad_epochs_count = 0;
                }
                else
                {
                    bad_epochs_count++;
                    if (bad_epochs_count >= patience)
                    {
                        reductions_count++;
                        bad_epochs_count = 0;
                        updateRate();
                        Logger::logMessage(std::format("Learning_Rate: Plateau detected. Reduced learning rate to {:.6f}", current_rate), LOG_INFO, true);
                    }
                }
            }
        }
        current_epoch++;
        updateRate();
    }

    float getCurrentRate() { return current_rate; }
};
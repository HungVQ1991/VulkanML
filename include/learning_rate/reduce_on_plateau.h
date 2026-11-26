#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Reduce_On_Plateau : public ILearning_Rate
{
private:
    static constexpr uint32_t ROP_CHECKPOINT_MAGIC_V2 = 0xFF524F50u;
    float learning_rate = 0.001f;
    float minimum_learning_rate = 1e-6f;
    float decay_rate = 0.1f;
    float current_learning_rate = 0.001f;
    float threshold = 0.0f;
    int patience = 10;
    int cooldown = 0;
    int cooldown_counter = 0;
    int current_epoch = 0;
    int bad_epochs_count = 0;
    int reductions_count = 0;
    float best_metric = 0.0f;
    bool is_higher_better = false;
    bool is_first_step = true;
    void validateParameters() const;

public:
    Reduce_On_Plateau(float _initial_learning_rate = 0.001f,
                      float _minimum_learning_rate = 1e-6f,
                      float _decay_rate = 0.1f,
                      int _patience = 10,
                      bool _is_higher_better = false,
                      float _threshold = 0.0f,
                      int _cooldown = 0);

    ~Reduce_On_Plateau() noexcept override = default;
    float updateRate() override;
    void step(float _current_value = 0.0f) override;
    bool requiresMetric() const noexcept override { return true; }
    void reset() override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    float getMinimumLearningRate() const noexcept { return minimum_learning_rate; }
    float getCurrentRate() const noexcept override { return current_learning_rate; }
    float getLearningRate() const noexcept override { return learning_rate; }
    float getDecayRate() const noexcept { return decay_rate; }
    float getThreshold() const noexcept { return threshold; }
    float getBestMetric() const noexcept { return best_metric; }
    Decay_Mode getType() const noexcept override { return Decay_Mode::REDUCE_ON_PLATEAU; }
    int getBadEpochsCount() const noexcept { return bad_epochs_count; }
    int getReductionsCount() const noexcept { return reductions_count; }
    int getCurrentEpoch() const noexcept { return current_epoch; }
    int getPatience() const noexcept { return patience; }
    int getCooldown() const noexcept { return cooldown; }
    int getCooldownCounter() const noexcept { return cooldown_counter; }
    bool isHigherBetter() const noexcept { return is_higher_better; }
    bool isFirstStep() const noexcept;
    void setMinimumLearningRate(float _minimum_learning_rate);
    void setCurrentRate(float _current_rate);
    void setLearningRate(float _learning_rate);
    void setDecayRate(float _decay_rate);
    void setThreshold(float _threshold);
    void setBestMetric(float _best_metric);
    void setBadEpochsCount(int _bad_epochs_count);
    void setReductionsCount(int _reductions_count);
    void setCurrentEpoch(int _current_epoch);
    void setPatience(int _patience);
    void setCooldown(int _cooldown);
    void setCooldownCounter(int _cooldown_counter);
    void setIsHigherBetter(bool _is_higher_better) noexcept;
    void setIsFirstStep(bool _is_first_step) noexcept;

};

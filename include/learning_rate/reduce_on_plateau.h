#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Reduce_On_Plateau : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 1e-6f;
    float decay_rate = 0.1f;
    float current_learning_rate = 0.001f;
    int patience = 10;
    int current_epoch = 0;
    int bad_epochs_count = 0;
    int reductions_count = 0;
    float best_metric = 0.0f;
    bool is_higher_better = false;
    bool is_first_step = true;    void validateParameters() const;


public:    Reduce_On_Plateau(float _initial_learning_rate = 0.001f,
                      float _minimum_learning_rate = 1e-6f,
                      float _decay_rate = 0.1f,
                      int _patience = 10,
                      bool _is_higher_better = false);


    ~Reduce_On_Plateau() noexcept override = default;    float updateRate() override;
    void step(float _current_value = 0.0f) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;


    float getMinimumLearningRate() const noexcept { return minimum_learning_rate; }
    float getCurrentRate() const noexcept override { return current_learning_rate; }
    float getLearningRate() const noexcept override { return learning_rate; }
    float getDecayRate() const noexcept { return decay_rate; }
    float getBestMetric() const noexcept { return best_metric; }
    Decay_Mode getType() const noexcept override { return Decay_Mode::REDUCE_ON_PLATEAU; }
    int getBadEpochsCount() const noexcept { return bad_epochs_count; }
    int getReductionsCount() const noexcept { return reductions_count; }
    int getCurrentEpoch() const noexcept { return current_epoch; }
    int getPatience() const noexcept { return patience; }
    bool isHigherBetter() const noexcept { return is_higher_better; }    bool isFirstStep() const noexcept;
    void setMinimumLearningRate(float _minimum_learning_rate) noexcept;
    void setCurrentRate(float _current_rate) noexcept;
    void setLearningRate(float _learning_rate) noexcept;
    void setDecayRate(float _decay_rate) noexcept;
    void setBestMetric(float _best_metric) noexcept;
    void setBadEpochsCount(int _bad_epochs_count) noexcept;
    void setReductionsCount(int _reductions_count) noexcept;
    void setCurrentEpoch(int _current_epoch) noexcept;
    void setPatience(int _patience) noexcept;
    void setIsHigherBetter(bool _is_higher_better) noexcept;
    void setIsFirstStep(bool _is_first_step) noexcept;

};;
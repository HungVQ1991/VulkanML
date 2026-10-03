#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Step_Decay : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 1e-6f;
    float decay_rate = 0.1f;
    float current_learning_rate = 0.001f;
    int step_size = 10;
    int current_epoch = 0;    void validateParameters() const;


public:    Step_Decay(float _initial_learning_rate = 0.001f,
               float _minimum_learning_rate = 1e-6f,
               float _decay_rate = 0.1f,
               int _step_size = 10);


    ~Step_Decay() noexcept override = default;    float updateRate() override;
    void step(float _current_value = 0.0f) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    float getCurrentLearningRate() const noexcept;

    float getMinimumLearningRate() const noexcept { return minimum_learning_rate; }
    float getCurrentRate() const noexcept override { return current_learning_rate; }
    float getLearningRate() const noexcept override { return learning_rate; }
    Decay_Mode getType() const noexcept override { return Decay_Mode::STEP_DECAY; }
    float getDecayRate() const noexcept { return decay_rate; }
    int getCurrentEpoch() const noexcept { return current_epoch; }
    int getStepSize() const noexcept { return step_size; }    void setMinimumLearningRate(float _min_lr) noexcept;
    void setCurrentLearningRate(float _rate) noexcept;
    void setLearningRate(float _learning_rate) noexcept;
    void setCurrentEpoch(int _epoch) noexcept;
    void setDecayRate(float _decay_rate) noexcept;
    void setStepSize(int _step_size) noexcept;

};;
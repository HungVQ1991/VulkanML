#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Multi_Step_Decay : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 1e-6f;
    float decay_rate = 0.1f;
    float current_learning_rate = 0.001f;
    int current_epoch = 0;
    std::vector<float> decay_epochs;    void validateParameters();


public:    Multi_Step_Decay(float _initial_learning_rate = 0.001f,
                     float _minimum_learning_rate = 1e-6f,
                     float _decay_rate = 0.1f,
                     std::vector<float> _decay_epochs = {30.0f, 60.0f, 80.0f});


    ~Multi_Step_Decay() noexcept override = default;    float updateRate() override;
    void step(float _current_value = 0.0f) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    const std::vector<float> &getDecayEpochs() const noexcept;
    float getCurrentLearningRate() const noexcept;

    float getMinimumLearningRate() const noexcept { return minimum_learning_rate; }
    float getCurrentRate() const noexcept override { return current_learning_rate; }
    float getLearningRate() const noexcept override { return learning_rate; }
    Decay_Mode getType() const noexcept override { return Decay_Mode::MULTI_STEP_DECAY; }
    float getDecayRate() const noexcept { return decay_rate; }
    int getCurrentEpoch() const noexcept { return current_epoch; }    void setDecayEpochs(std::vector<float> _decay_epochs);
    void setMinimumLearningRate(float _min_lr) noexcept;
    void setCurrentLearningRate(float _rate) noexcept;
    void setLearningRate(float _learning_rate) noexcept;
    void setCurrentEpoch(int _epoch) noexcept;
    void setDecayRate(float _decay_rate) noexcept;

};;
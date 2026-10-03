#pragma once

#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class Cosine_Annealing : public ILearning_Rate
{
private:
    float learning_rate = 0.001f;
    float minimum_learning_rate = 0.0f;
    float current_learning_rate = 0.001f;
    int maximum_epoch = 100;
    int current_epoch = 0;
    int warmup_steps = 0;    void validateParameters() const;


public:    Cosine_Annealing(float _initial_learning_rate = 0.001f,
                     float _minimum_learning_rate = 0.0f,
                     int _maximum_epoch = 100,
                     int _warmup_steps = 0);


    ~Cosine_Annealing() noexcept override = default;

    int getWarmupSteps() const noexcept { return warmup_steps; }    void setWarmupSteps(int _warmup_steps);
    float updateRate() override;
    void step(float _current_value = 0.0f) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;


    float getMinimumLearningRate() const noexcept { return minimum_learning_rate; }
    float getCurrentRate() const noexcept override { return current_learning_rate; }
    float getLearningRate() const noexcept override { return learning_rate; }
    Decay_Mode getType() const noexcept override { return Decay_Mode::COSINE_ANNEALING; }
    int getMaximumEpoch() const noexcept { return maximum_epoch; }
    int getCurrentEpoch() const noexcept { return current_epoch; }    void setMaxEpoch(int _maximum_epoch) override;
    void setMinimumLearningRate(float _minimum_learning_rate) noexcept;
    void setCurrentRate(float _current_rate) noexcept;
    void setLearningRate(float _learning_rate) noexcept;
    void setCurrentEpoch(int _current_epoch) noexcept;

};;
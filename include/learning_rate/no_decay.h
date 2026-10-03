#pragma once

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "helper/logger.h"
#include "ilearning_rate.h"

class No_Decay : public ILearning_Rate
{
private:
    float learning_rate = 0.01f;

public:    explicit No_Decay(float _initial_learning_rate = 0.01f);


    ~No_Decay() noexcept override = default;    float updateRate() override;
    void step(float _current_value = 0.0f) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;


    float getCurrentRate() const noexcept override { return learning_rate; }
    float getLearningRate() const noexcept override { return learning_rate; }
    Decay_Mode getType() const noexcept override { return Decay_Mode::NO_DECAY; }    void setLearningRate(float _learning_rate) noexcept;

};;
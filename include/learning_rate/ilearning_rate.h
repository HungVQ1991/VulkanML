#pragma once

#include <cstdint>
#include <fstream>

#include "helper/logger.h"

enum class Decay_Mode : std::uint32_t
{
    NO_DECAY = 0,
    STEP_DECAY,
    MULTI_STEP_DECAY,
    EXPONENTIAL_DECAY,
    COSINE_ANNEALING,
    POLYNOMIAL_DECAY,
    REDUCE_ON_PLATEAU,
    DECAY_MODE_END
};

class ILearning_Rate
{
public:
    virtual ~ILearning_Rate() noexcept = default;

    [[nodiscard]] virtual Decay_Mode getType() const noexcept = 0;
    virtual float updateRate() = 0;
    virtual void step(float _current_value = 0.0f) = 0;
    [[nodiscard]] virtual float getCurrentRate() const noexcept = 0;
    [[nodiscard]] virtual float getLearningRate() const noexcept = 0;

    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &_input_file_stream) = 0;

    virtual void setMaxEpoch(int _maximum_epoch) {}
};
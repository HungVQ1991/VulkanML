#pragma once

#include <cstdint>
#include <fstream>
#include <memory>

#include "helper/logger.h"

enum class Decay_Mode : uint32_t
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
private:
    std::shared_ptr<bool> lifetime_token = std::make_shared<bool>(true);

public:
    ILearning_Rate() = default;
    ILearning_Rate(const ILearning_Rate &) : lifetime_token(std::make_shared<bool>(true)) {}
    ILearning_Rate &operator=(const ILearning_Rate &) { return *this; }
    ILearning_Rate(ILearning_Rate &&) : lifetime_token(std::make_shared<bool>(true)) {}
    ILearning_Rate &operator=(ILearning_Rate &&) noexcept { return *this; }
    virtual ~ILearning_Rate() noexcept = default;

    std::weak_ptr<bool> getLifetimeToken() const noexcept { return lifetime_token; }

    virtual float updateRate() = 0;
    virtual void step(float _current_value = 0.0f) = 0;
    virtual bool requiresMetric() const noexcept { return false; }
    virtual void reset() {}

    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &_input_file_stream) = 0;

    virtual float getCurrentRate() const noexcept = 0;
    virtual float getLearningRate() const noexcept = 0;
    virtual Decay_Mode getType() const noexcept = 0;

    virtual void setMaxEpoch(int) {}
};

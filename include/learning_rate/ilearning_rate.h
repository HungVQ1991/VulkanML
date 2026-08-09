#pragma once

#include <cstdint>
#include <fstream>
#include "helper/logger.h"

#ifndef ENABLE_LR_DEBUG_LOGS
#define ENABLE_LR_DEBUG_LOGS 0
#endif

#if ENABLE_LR_DEBUG_LOGS
#define LR_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define LR_LOG_DEBUG(msg) ((void)0)
#endif

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

    virtual Decay_Mode getType() const = 0;
    virtual float updateRate() = 0;
    virtual void step(float current_val = 0.0f) = 0;
    virtual float getCurrentRate() const = 0;
    virtual float getLearningRate() const = 0;

    virtual void saveCheckpoint(std::ofstream &stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &stream) = 0;

    virtual void setMaxEpoch(int total_epochs) {}
};
#pragma once

#include <fstream>
#include <map>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "learning_rate/ilearning_rate.h"
#include "math/matrix.h"

#ifndef ENABLE_OPTIMIZER_DEBUG_LOGS
#define ENABLE_OPTIMIZER_DEBUG_LOGS 0
#endif

#if ENABLE_OPTIMIZER_DEBUG_LOGS
#define OPTIMIZER_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define OPTIMIZER_LOG_DEBUG(msg) ((void)0)
#endif

enum class Optimizer_Type
{
    SGD_OPTIMIZER,
    ADAM_OPTIMIZER,
    OPTIMIZER_TYPE_END
};

class IOptimizer
{
public:
    virtual Optimizer_Type getType() const = 0;
    virtual ~IOptimizer() noexcept = default;
    virtual void step(const std::vector<std::pair<Matrix *, Matrix *>> &param_grad_pairs) = 0;
    virtual void reset() {}
    virtual void setLearningRate(float lr) = 0;
    virtual void saveCheckpoint(std::ofstream &out_file) const = 0;
    virtual void loadCheckpoint(std::ifstream &in_file, Execution_Target target = Execution_Target::CPU) = 0;
};
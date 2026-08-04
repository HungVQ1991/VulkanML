#pragma once
#include "helper/logger.h"
#include "math/matrix.h"
#include "learning_rate/ilearning_rate.h"
#include "engine/execution_engine.h"

#include <fstream>
#include <utility>
#include <vector>
#include <map>

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




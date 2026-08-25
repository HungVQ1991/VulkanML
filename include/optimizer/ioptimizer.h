#pragma once

#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "learning_rate/ilearning_rate.h"
#include "math/matrix.h"

enum class Optimizer_Type : std::uint32_t
{
    SGD_OPTIMIZER,
    ADAM_OPTIMIZER,
    OPTIMIZER_TYPE_END
};

class IOptimizer
{
public:
    virtual ~IOptimizer() noexcept = default;

     virtual Optimizer_Type getType() const noexcept = 0;
    virtual void step(const std::vector<std::pair<Matrix *, Matrix *>> &_parameter_gradient_pairs) = 0;
    virtual void reset() {}
    virtual void setLearningRate(float _learning_rate) = 0;
    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) = 0;
};
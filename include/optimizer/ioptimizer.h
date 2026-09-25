#pragma once

#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "learning_rate/ilearning_rate.h"
#include "math/tensor.h"

enum class Optimizer_Type : uint32_t
{
    SGD_OPTIMIZER,
    ADAM_OPTIMIZER,
    OPTIMIZER_TYPE_END
};

class IOptimizer
{
public:
    virtual ~IOptimizer() noexcept = default;

    virtual void step(const std::vector<std::pair<Matrix *, Matrix *>> &_parameter_gradient_pairs) = 0;
    virtual void step(const std::vector<std::pair<Matrix *, Matrix *>> &_parameter_gradient_pairs, float _grad_scale)
    {
        (void)_grad_scale;
        step(_parameter_gradient_pairs);
    }
    virtual void stepDynamicParams(float _grad_scale = 1.0f)
    {
        (void)_grad_scale;
    }
    virtual void reset() {}
    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) = 0;

    virtual float getLearningRate() const noexcept = 0;
    virtual Optimizer_Type getType() const noexcept = 0;

    virtual void setLearningRate(float _learning_rate) = 0;
};
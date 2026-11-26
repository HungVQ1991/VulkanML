#pragma once

#include <cstdint>
#include <fstream>
#include <unordered_set>
#include <utility>
#include <vector>

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

    virtual void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs) = 0;
    virtual void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale) = 0;
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
    virtual void setLearningRateScheduler(ILearning_Rate *_scheduler) noexcept { (void)_scheduler; }
    virtual void detachLearningRateScheduler() noexcept {}
    virtual ILearning_Rate *getLearningRateScheduler() const noexcept { return nullptr; }
    virtual void setExecutionTarget(Execution_Target _target) { (void)_target; }

    virtual void setMaxGradient(float _max_gradient) { (void)_max_gradient; }
    virtual float getMaxGradient() const noexcept { return 0.0f; }

    virtual void setWeightDecay(float ) {}
    virtual float getWeightDecay() const noexcept { return 0.0f; }
    virtual void setNoDecayParameters(std::unordered_set<const Tensor *> ) {}
    virtual uint64_t getConfigGeneration() const noexcept { return 0; }
    virtual bool consumeGpuOverflowDetected() noexcept { return false; }
};

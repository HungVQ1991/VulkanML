#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ioptimizer.h"
#include "math/tensor.h"

class Sgd_Optimizer : public IOptimizer
{
private:
    static constexpr uint32_t SGD_CHECKPOINT_MAGIC_V2 = 0xFF534744u;

    float learning_rate = 0.01f;
    float max_gradient = 1.0f;
    float weight_decay = 0.0f;
    float momentum = 0.0f;
    ILearning_Rate *learning_rate_scheduler = nullptr;
    std::weak_ptr<bool> scheduler_lifetime_token;
    struct Backend_State;
    std::unique_ptr<Backend_State> backend_state;
    bool pending_gpu_overflow = false;
    uint64_t config_generation = 0;

    std::unordered_map<Tensor *, Tensor> velocity_states;
    std::unordered_map<Tensor *, std::weak_ptr<bool>> velocity_tokens;
    std::vector<Tensor *> velocity_order;
    std::vector<Tensor> loaded_velocities;
    size_t loaded_velocity_cursor = 0;
    std::vector<uint8_t> loaded_no_decay_flags;
    std::unordered_set<const Tensor *> no_decay_parameters;
    std::unordered_map<const Tensor *, std::weak_ptr<bool>> no_decay_tokens;

    static void validateHyperparameters(float _lr, float _max_grad, float _wd, float _momentum);
    static Tensor createZeroVelocityTensor(const Shape &_shape, Execution_Target _target);
    void latchCompletedGpuOverflows() noexcept;

public:
    explicit Sgd_Optimizer(float _learning_rate = 0.01f,
                           float _max_gradient = 1.0f,
                           float _weight_decay = 0.0f,
                           float _momentum = 0.0f);
    explicit Sgd_Optimizer(ILearning_Rate &_learning_rate_scheduler,
                           float _max_gradient = 1.0f,
                           float _weight_decay = 0.0f,
                           float _momentum = 0.0f);

    Sgd_Optimizer(const Sgd_Optimizer &) = delete;
    Sgd_Optimizer &operator=(const Sgd_Optimizer &) = delete;
    Sgd_Optimizer(Sgd_Optimizer &&other) noexcept;
    Sgd_Optimizer &operator=(Sgd_Optimizer &&other) noexcept;
    ~Sgd_Optimizer() noexcept override;
    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs) override;
    void stepDynamicParams(float _grad_scale = 1.0f) override;
    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale) override;
    void reset() override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) override;

    ILearning_Rate *getLearningRateScheduler() const noexcept override
    {
        return (learning_rate_scheduler && !scheduler_lifetime_token.expired()) ? learning_rate_scheduler : nullptr;
    }
    Optimizer_Type getType() const noexcept override { return Optimizer_Type::SGD_OPTIMIZER; }
    float getLearningRate() const noexcept override { return learning_rate; }
    float getMaxGradient() const noexcept override { return max_gradient; }
    float getWeightDecay() const noexcept override { return weight_decay; }
    float getMomentum() const noexcept { return momentum; }
    uint64_t getConfigGeneration() const noexcept override { return config_generation; }
    bool consumeGpuOverflowDetected() noexcept override;

    void setExecutionTarget(Execution_Target _target) override;
    void setMaxGradient(float _max_gradient) override
    {
        validateHyperparameters(learning_rate, _max_gradient, weight_decay, momentum);
        if (max_gradient != _max_gradient)
        {
            max_gradient = _max_gradient;
            ++config_generation;
        }
    }
    void setWeightDecay(float _weight_decay) override
    {
        validateHyperparameters(learning_rate, max_gradient, _weight_decay, momentum);
        if (weight_decay != _weight_decay)
        {
            weight_decay = _weight_decay;
            ++config_generation;
        }
    }
    void setMomentum(float _momentum)
    {
        validateHyperparameters(learning_rate, max_gradient, weight_decay, _momentum);
        if (momentum != _momentum)
        {
            momentum = _momentum;
            ++config_generation;
        }
    }
    void setNoDecayParameters(std::unordered_set<const Tensor *> _no_decay) override
    {
        no_decay_parameters = std::move(_no_decay);
        no_decay_tokens.clear();
        for (const Tensor *p : no_decay_parameters)
        {
            if (p)
            {
                no_decay_tokens[p] = p->getLifetimeToken();
            }
        }
        ++config_generation;
    }
    void addNoDecayParameter(const Tensor *_param)
    {
        if (_param)
        {
            no_decay_parameters.insert(_param);
            no_decay_tokens[_param] = _param->getLifetimeToken();
            ++config_generation;
        }
    }
    const std::unordered_set<const Tensor *> &getNoDecayParameters() const noexcept { return no_decay_parameters; }
    const std::unordered_map<Tensor *, Tensor> &getVelocityStates() const noexcept { return velocity_states; }
    const std::vector<Tensor *> &getVelocityOrder() const noexcept { return velocity_order; }
    const std::vector<Tensor> &getLoadedVelocities() const noexcept { return loaded_velocities; }
    void setVelocityStates(const std::unordered_map<Tensor *, Tensor> &_states);
    void setVelocityOrder(const std::vector<Tensor *> &_order);
    void setLoadedVelocities(const std::vector<Tensor> &_loaded);

    void setLearningRateScheduler(ILearning_Rate *_learning_rate_scheduler) noexcept override
    {
        learning_rate_scheduler = _learning_rate_scheduler;
        scheduler_lifetime_token = _learning_rate_scheduler ? _learning_rate_scheduler->getLifetimeToken() : std::weak_ptr<bool>{};
    }
    void detachLearningRateScheduler() noexcept override
    {
        learning_rate_scheduler = nullptr;
        scheduler_lifetime_token.reset();
    }
    void setLearningRate(float _learning_rate) override
    {
        validateHyperparameters(_learning_rate, max_gradient, weight_decay, momentum);
        if (_learning_rate <= 0.0f)
        {
            Logger::logMessage("Sgd_Optimizer::setLearningRate: learning_rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
        Logger::logMessage(Input_Format{"Sgd_Optimizer::setLearningRate: old_learning_rate={}, new_learning_rate={}",
                                        learning_rate,
                                        _learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
        learning_rate = _learning_rate;
    }
};

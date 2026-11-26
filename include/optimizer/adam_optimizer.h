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

class Adam_Optimizer : public IOptimizer
{
public:
    struct Parameter_State
    {
        Tensor first_moment_matrix;
        Tensor second_moment_matrix;

        Parameter_State(size_t _rows, size_t _columns, Execution_Target _execution_target)
            : Parameter_State(Shape{_rows, _columns}, _execution_target)
        {
        }

        Parameter_State(const Shape &_shape, Execution_Target _execution_target);

        Parameter_State(Tensor _first_moment, Tensor _second_moment)
            : first_moment_matrix(std::move(_first_moment)),
              second_moment_matrix(std::move(_second_moment))
        {
        }

        Parameter_State clone() const
        {
            return Parameter_State(first_moment_matrix.clone(), second_moment_matrix.clone());
        }

        const Tensor &getSecondMomentMatrix() const noexcept { return second_moment_matrix; }
        const Tensor &getFirstMomentMatrix() const noexcept { return first_moment_matrix; }

        void setSecondMomentMatrix(const Tensor &_matrix) { second_moment_matrix = _matrix; }
        void setFirstMomentMatrix(const Tensor &_matrix) { first_moment_matrix = _matrix; }
    };

private:
    static constexpr uint64_t ADAM_CHECKPOINT_MAGIC_V2 = 0x4144414D56320001ULL;

    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float max_gradient = 1.0f;
    float weight_decay = 0.0f;
    size_t timestep = 0;
    ILearning_Rate *learning_rate_scheduler = nullptr;
    std::weak_ptr<bool> scheduler_lifetime_token;

    std::unordered_map<Tensor *, Parameter_State> parameter_states;
    std::unordered_map<Tensor *, std::weak_ptr<bool>> parameter_tokens;
    std::vector<Tensor *> parameter_order;
    std::vector<Parameter_State> loaded_states;
    size_t loaded_state_cursor = 0;
    std::vector<uint8_t> loaded_no_decay_flags;
    std::unordered_set<const Tensor *> no_decay_parameters;
    std::unordered_map<const Tensor *, std::weak_ptr<bool>> no_decay_tokens;
    struct Backend_State;
    std::unique_ptr<Backend_State> backend_state;
    bool pending_gpu_overflow = false;
    uint64_t config_generation = 0;

    static void validateHyperparameters(float _lr, float _b1, float _b2, float _eps, float _max_grad, float _wd);
    void latchCompletedGpuOverflows() noexcept;

public:
    explicit Adam_Optimizer(float _learning_rate = 0.001f,
                            float _beta1 = 0.9f,
                            float _beta2 = 0.999f,
                            float _epsilon = 1e-8f,
                            float _max_gradient = 1.0f,
                            float _weight_decay = 0.0f);
    explicit Adam_Optimizer(ILearning_Rate &_learning_rate_scheduler,
                            float _beta1 = 0.9f,
                            float _beta2 = 0.999f,
                            float _epsilon = 1e-8f,
                            float _max_gradient = 1.0f,
                            float _weight_decay = 0.0f);

    Adam_Optimizer(const Adam_Optimizer &) = delete;
    Adam_Optimizer &operator=(const Adam_Optimizer &) = delete;
    Adam_Optimizer(Adam_Optimizer &&other) noexcept;
    Adam_Optimizer &operator=(Adam_Optimizer &&other) noexcept;
    ~Adam_Optimizer() noexcept override;
    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs) override;
    void stepDynamicParams(float _grad_scale = 1.0f) override;
    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale) override;
    void reset() override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) override;

    const std::unordered_map<Tensor *, Parameter_State> &getParameterStates() const noexcept { return parameter_states; }
    const std::vector<Parameter_State> &getLoadedStates() const noexcept { return loaded_states; }
    const std::vector<Tensor *> &getParameterOrder() const noexcept { return parameter_order; }
    ILearning_Rate *getLearningRateScheduler() const noexcept override
    {
        return (learning_rate_scheduler && !scheduler_lifetime_token.expired()) ? learning_rate_scheduler : nullptr;
    }
    size_t getTimestep() const noexcept { return timestep; }
    Optimizer_Type getType() const noexcept override { return Optimizer_Type::ADAM_OPTIMIZER; }
    float getLearningRate() const noexcept override { return learning_rate; }
    float getMaxGradient() const noexcept override { return max_gradient; }
    float getEpsilon() const noexcept { return epsilon; }
    float getBeta1() const noexcept { return beta1; }
    float getBeta2() const noexcept { return beta2; }
    float getWeightDecay() const noexcept override { return weight_decay; }
    uint64_t getConfigGeneration() const noexcept override { return config_generation; }
    bool consumeGpuOverflowDetected() noexcept override;

    void setWeightDecay(float _weight_decay) override
    {
        validateHyperparameters(learning_rate, beta1, beta2, epsilon, max_gradient, _weight_decay);
        if (weight_decay != _weight_decay)
        {
            weight_decay = _weight_decay;
            ++config_generation;
        }
    }
    void setParameterStates(const std::unordered_map<Tensor *, Parameter_State> &_parameter_states)
    {
        parameter_states = _parameter_states;
        parameter_tokens.clear();
        parameter_order.clear();
        parameter_order.reserve(parameter_states.size());
        for (const auto &[p, st] : parameter_states)
        {
            (void)st;
            if (p)
            {
                parameter_tokens[p] = p->getLifetimeToken();
                parameter_order.push_back(p);
            }
        }
    }
    void setLoadedStates(const std::vector<Parameter_State> &_loaded_states);
    void setParameterOrder(const std::vector<Tensor *> &_parameter_order);

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
    void setExecutionTarget(Execution_Target _target) override;
    void setTimestep(size_t _timestep) noexcept { timestep = _timestep; }
    void setLearningRate(float _learning_rate) override
    {
        validateHyperparameters(_learning_rate, beta1, beta2, epsilon, max_gradient, weight_decay);
        if (_learning_rate <= 0.0f)
        {
            Logger::logMessage("Adam_Optimizer::setLearningRate: learning_rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
        Logger::logMessage(Input_Format{"Adam_Optimizer::setLearningRate: old_learning_rate={}, new_learning_rate={}",
                                        learning_rate,
                                        _learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
        learning_rate = _learning_rate;
    }
    void setMaxGradient(float _max_gradient) override
    {
        validateHyperparameters(learning_rate, beta1, beta2, epsilon, _max_gradient, weight_decay);
        if (max_gradient != _max_gradient)
        {
            max_gradient = _max_gradient;
            ++config_generation;
        }
    }
    void setEpsilon(float _epsilon)
    {
        validateHyperparameters(learning_rate, beta1, beta2, _epsilon, max_gradient, weight_decay);
        if (epsilon != _epsilon)
        {
            epsilon = _epsilon;
            ++config_generation;
        }
    }
    void setBeta1(float _beta1)
    {
        validateHyperparameters(learning_rate, _beta1, beta2, epsilon, max_gradient, weight_decay);
        if (beta1 != _beta1)
        {
            beta1 = _beta1;
            ++config_generation;
        }
    }
    void setBeta2(float _beta2)
    {
        validateHyperparameters(learning_rate, beta1, _beta2, epsilon, max_gradient, weight_decay);
        if (beta2 != _beta2)
        {
            beta2 = _beta2;
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
};

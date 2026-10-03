#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>
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
            : first_moment_matrix(_rows, _columns, std::vector<float>(_rows * _columns, 0.0f), _execution_target),
              second_moment_matrix(_rows, _columns, std::vector<float>(_rows * _columns, 0.0f), _execution_target)
        {
        }

        Parameter_State(Tensor _first_moment, Tensor _second_moment)
            : first_moment_matrix(std::move(_first_moment)),
              second_moment_matrix(std::move(_second_moment))
        {
        }

        const Tensor &getSecondMomentMatrix() const noexcept { return second_moment_matrix; }
        const Tensor &getFirstMomentMatrix() const noexcept { return first_moment_matrix; }

        void setSecondMomentMatrix(const Tensor &_matrix) { second_moment_matrix = _matrix; }
        void setFirstMomentMatrix(const Tensor &_matrix) { first_moment_matrix = _matrix; }
    };

private:

    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float max_gradient = 1.0f;
    float weight_decay = 0.0f;
    size_t timestep = 0;
    ILearning_Rate *learning_rate_scheduler = nullptr;

    std::unordered_map<Tensor *, Parameter_State> parameter_states;
    std::vector<Tensor *> parameter_order;
    std::vector<Parameter_State> loaded_states;

public:
    explicit Adam_Optimizer(float _learning_rate = 0.001f,
                            float _beta1 = 0.9f,
                            float _beta2 = 0.999f,
                            float _epsilon = 1e-8f,
                            float _max_gradient = 1.0f,
                            float _weight_decay = 0.0f)
        : learning_rate(_learning_rate),
          beta1(_beta1),
          beta2(_beta2),
          epsilon(_epsilon),
          max_gradient(_max_gradient),
          weight_decay(_weight_decay),
          learning_rate_scheduler(nullptr)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Adam_Optimizer::Adam_Optimizer: Initial learning rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
        Logger::logMessage(Input_Format{"Adam_Optimizer::Adam_Optimizer: learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, weight_decay={}",
                                        learning_rate,
                                        beta1,
                                        beta2,
                                        epsilon,
                                        max_gradient,
                                        weight_decay},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    explicit Adam_Optimizer(ILearning_Rate &_learning_rate_scheduler,
                            float _beta1 = 0.9f,
                            float _beta2 = 0.999f,
                            float _epsilon = 1e-8f,
                            float _max_gradient = 1.0f,
                            float _weight_decay = 0.0f)
        : learning_rate(_learning_rate_scheduler.getCurrentRate()),
          beta1(_beta1),
          beta2(_beta2),
          epsilon(_epsilon),
          max_gradient(_max_gradient),
          weight_decay(_weight_decay),
          learning_rate_scheduler(&_learning_rate_scheduler)
    {
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Adam_Optimizer::Adam_Optimizer: Initial learning rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
        Logger::logMessage(Input_Format{"Adam_Optimizer::Adam_Optimizer (ILearning_Rate): learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, weight_decay={}",
                                        learning_rate,
                                        beta1,
                                        beta2,
                                        epsilon,
                                        max_gradient,
                                        weight_decay},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    ~Adam_Optimizer() noexcept override = default;

    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs) override
    {
        step(_parameter_gradient_pairs, 1.0f);
    }

    void stepDynamicParams(float _grad_scale = 1.0f) override
    {
        if (learning_rate_scheduler != nullptr)
        {
            learning_rate = learning_rate_scheduler->getCurrentRate();
        }

        ++timestep;
        float inv_scale = (_grad_scale > 0.0f) ? (1.0f / _grad_scale) : 1.0f;
        size_t effective_t = std::max<size_t>(timestep, 1);
        float bc1 = std::max(1.0F - std::pow(beta1, static_cast<float>(effective_t)), 1e-8F);
        float bc2 = std::max(1.0F - std::pow(beta2, static_cast<float>(effective_t)), 1e-8F);

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0F / bc1, 1.0F / std::sqrt(bc2), inv_scale, current_frame);
    }

    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale) override
    {
        if (parameter_states.empty() && !loaded_states.empty())
        {
            for (size_t i = 0; i < _parameter_gradient_pairs.size() && i < loaded_states.size(); ++i)
            {
                Tensor *parameter = _parameter_gradient_pairs[i].first;
                if (parameter)
                {
                    if (parameter->getRows() == loaded_states[i].first_moment_matrix.getRows() &&
                        parameter->getColumns() == loaded_states[i].first_moment_matrix.getColumns())
                    {
                        parameter_states.emplace(parameter, std::move(loaded_states[i]));
                    }
                    else
                    {
                        Logger::logMessage("Adam_Optimizer::step: Parameter shape mismatch with loaded state, reinitializing state",
                                           Log_Level::LOG_WARNING,
                                           true,
                                           0,
                                           Log_Feature::OPTIMIZER_STEP);
                        parameter_states.emplace(parameter, Parameter_State(parameter->getRows(), parameter->getColumns(), parameter->getExecutionTarget()));
                    }
                    parameter_order.push_back(parameter);
                }
            }
            loaded_states.clear();
        }

        stepDynamicParams(_grad_scale);
        float inv_scale = (_grad_scale > 0.0f) ? (1.0f / _grad_scale) : 1.0f;

        for (const auto &[parameter, gradient] : _parameter_gradient_pairs)
        {
            if (!parameter || !gradient)
            {
                Logger::logMessage("Adam_Optimizer::step: Null parameter or gradient pointer encountered",
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::OPTIMIZER_STEP);
                continue;
            }

            auto iterator = parameter_states.find(parameter);
            if (iterator == parameter_states.end())
            {
                auto [new_iterator, is_inserted] = parameter_states.emplace(parameter, Parameter_State(parameter->getRows(), parameter->getColumns(), parameter->getExecutionTarget()));
                iterator = new_iterator;
                parameter_order.push_back(parameter);
            }

            parameter->adamUpdate(*gradient,
                                  iterator->second.first_moment_matrix,
                                  iterator->second.second_moment_matrix,
                                  learning_rate,
                                  beta1,
                                  beta2,
                                  epsilon,
                                  timestep,
                                  max_gradient,
                                  inv_scale,
                                  weight_decay);
        }
    }

    void reset() override
    {
        Logger::logMessage("Adam_Optimizer::reset: Resetting optimizer states and timestep",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
        parameter_states.clear();
        parameter_order.clear();
        loaded_states.clear();
        timestep = 0;
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        if (!_output_file_stream.is_open())
        {
            Logger::logMessage("Adam_Optimizer::saveCheckpoint: Output stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return;
        }

        Logger::logMessage(Input_Format{"Adam_Optimizer::saveCheckpoint: Saving checkpoint at timestep={}, learning_rate={}",
                                        timestep,
                                        learning_rate},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        uint64_t timestep_value = static_cast<uint64_t>(timestep);
        _output_file_stream.write(reinterpret_cast<const char *>(&timestep_value), sizeof(timestep_value));
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&beta1), sizeof(beta1));
        _output_file_stream.write(reinterpret_cast<const char *>(&beta2), sizeof(beta2));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));

        uint32_t state_count = static_cast<uint32_t>(parameter_order.size());
        _output_file_stream.write(reinterpret_cast<const char *>(&state_count), sizeof(state_count));

        for (Tensor *parameter : parameter_order)
        {
            auto iterator = parameter_states.find(parameter);
            if (iterator != parameter_states.end())
            {
                iterator->second.first_moment_matrix.saveMatrix(_output_file_stream);
                iterator->second.second_moment_matrix.saveMatrix(_output_file_stream);
            }
        }
    }

    void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) override
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Adam_Optimizer::loadCheckpoint: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return;
        }

        reset();

        uint64_t timestep_value = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&timestep_value), sizeof(timestep_value));
        timestep = static_cast<size_t>(timestep_value);

        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&beta1), sizeof(beta1));
        _input_file_stream.read(reinterpret_cast<char *>(&beta2), sizeof(beta2));
        _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
        _input_file_stream.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));

        uint32_t state_count = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&state_count), sizeof(state_count));

        Logger::logMessage(Input_Format{"Adam_Optimizer::loadCheckpoint: Loaded timestep={}, learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, state_count={}",
                                        timestep,
                                        learning_rate,
                                        beta1,
                                        beta2,
                                        epsilon,
                                        max_gradient,
                                        state_count},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        loaded_states.reserve(state_count);
        for (uint32_t i = 0; i < state_count; ++i)
        {
            Tensor first_moment = Tensor::loadMatrix(_input_file_stream, _execution_target);
            Tensor second_moment = Tensor::loadMatrix(_input_file_stream, _execution_target);
            loaded_states.emplace_back(std::move(first_moment), std::move(second_moment));
        }
    }

    const std::unordered_map<Tensor *, Parameter_State> &getParameterStates() const noexcept { return parameter_states; }
    const std::vector<Parameter_State> &getLoadedStates() const noexcept { return loaded_states; }
    const std::vector<Tensor *> &getParameterOrder() const noexcept { return parameter_order; }
    ILearning_Rate *getLearningRateScheduler() const noexcept { return learning_rate_scheduler; }
    size_t getTimestep() const noexcept { return timestep; }
    Optimizer_Type getType() const noexcept override { return Optimizer_Type::ADAM_OPTIMIZER; }
    float getLearningRate() const noexcept override { return learning_rate; }
    float getMaxGradient() const noexcept override { return max_gradient; }
    float getEpsilon() const noexcept { return epsilon; }
    float getBeta1() const noexcept { return beta1; }
    float getBeta2() const noexcept { return beta2; }
    float getWeightDecay() const noexcept { return weight_decay; }

    void setWeightDecay(float _weight_decay) noexcept { weight_decay = _weight_decay; }
    void setParameterStates(const std::unordered_map<Tensor *, Parameter_State> &_parameter_states) { parameter_states = _parameter_states; }
    void setLoadedStates(const std::vector<Parameter_State> &_loaded_states) { loaded_states = _loaded_states; }
    void setParameterOrder(const std::vector<Tensor *> &_parameter_order) { parameter_order = _parameter_order; }
    void setLearningRateScheduler(ILearning_Rate *_learning_rate_scheduler) noexcept { learning_rate_scheduler = _learning_rate_scheduler; }
    void setTimestep(size_t _timestep) noexcept { timestep = _timestep; }
    void setLearningRate(float _learning_rate) override
    {
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
        learning_rate_scheduler = nullptr;
    }
    void setMaxGradient(float _max_gradient) noexcept override { max_gradient = _max_gradient; }
    void setEpsilon(float _epsilon) noexcept { epsilon = _epsilon; }
    void setBeta1(float _beta1) noexcept { beta1 = _beta1; }
    void setBeta2(float _beta2) noexcept { beta2 = _beta2; }
};
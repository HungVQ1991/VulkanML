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
#include "math/matrix.h"

class Adam_Optimizer : public IOptimizer
{
private:
    struct Parameter_State
    {
        Matrix first_moment_matrix;
        Matrix second_moment_matrix;

        Parameter_State(std::size_t _rows, std::size_t _columns, Execution_Target _execution_target)
            : first_moment_matrix(_rows, _columns, std::vector<float>(_rows * _columns, 0.0f), _execution_target),
              second_moment_matrix(_rows, _columns, std::vector<float>(_rows * _columns, 0.0f), _execution_target)
        {
        }

        Parameter_State(Matrix _first_moment, Matrix _second_moment)
            : first_moment_matrix(std::move(_first_moment)),
              second_moment_matrix(std::move(_second_moment))
        {
        }
    };

    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float max_gradient = 1.0f;
    std::size_t timestep = 0;
    ILearning_Rate *learning_rate_scheduler = nullptr;

    std::unordered_map<Matrix *, Parameter_State> parameter_states;
    std::vector<Matrix *> parameter_order;
    std::vector<Parameter_State> loaded_states;

public:
    explicit Adam_Optimizer(float _learning_rate = 0.001f,
                            float _beta1 = 0.9f,
                            float _beta2 = 0.999f,
                            float _epsilon = 1e-8f,
                            float _max_gradient = 1.0f)
        : learning_rate(_learning_rate),
          beta1(_beta1),
          beta2(_beta2),
          epsilon(_epsilon),
          max_gradient(_max_gradient),
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
        Logger::logMessage(std::format("Adam_Optimizer::Adam_Optimizer: learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}",
                                       learning_rate,
                                       beta1,
                                       beta2,
                                       epsilon,
                                       max_gradient),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    explicit Adam_Optimizer(ILearning_Rate &_learning_rate_scheduler,
                            float _beta1 = 0.9f,
                            float _beta2 = 0.999f,
                            float _epsilon = 1e-8f,
                            float _max_gradient = 1.0f)
        : learning_rate(_learning_rate_scheduler.getCurrentRate()),
          beta1(_beta1),
          beta2(_beta2),
          epsilon(_epsilon),
          max_gradient(_max_gradient),
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
        Logger::logMessage(std::format("Adam_Optimizer::Adam_Optimizer (ILearning_Rate): learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}",
                                       learning_rate,
                                       beta1,
                                       beta2,
                                       epsilon,
                                       max_gradient),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    ~Adam_Optimizer() noexcept override = default;

    [[nodiscard]] Optimizer_Type getType() const noexcept override
    {
        return Optimizer_Type::ADAM_OPTIMIZER;
    }

    void step(const std::vector<std::pair<Matrix *, Matrix *>> &_parameter_gradient_pairs) override
    {
        if (learning_rate_scheduler != nullptr)
        {
            learning_rate = learning_rate_scheduler->getCurrentRate();
        }

        if (parameter_states.empty() && !loaded_states.empty())
        {
            for (std::size_t i = 0; i < _parameter_gradient_pairs.size() && i < loaded_states.size(); ++i)
            {
                Matrix *parameter = _parameter_gradient_pairs[i].first;
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

        ++timestep;

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
                                  max_gradient);
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

    [[nodiscard]] float getLearningRate() const noexcept
    {
        return learning_rate;
    }

    [[nodiscard]] float getMaxGradient() const noexcept
    {
        return max_gradient;
    }

    [[nodiscard]] float getBeta1() const noexcept
    {
        return beta1;
    }

    [[nodiscard]] float getBeta2() const noexcept
    {
        return beta2;
    }

    [[nodiscard]] float getEpsilon() const noexcept
    {
        return epsilon;
    }

    [[nodiscard]] std::size_t getTimestep() const noexcept
    {
        return timestep;
    }

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
        Logger::logMessage(std::format("Adam_Optimizer::setLearningRate: old_learning_rate={}, new_learning_rate={}",
                                       learning_rate,
                                       _learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
        learning_rate = _learning_rate;
        learning_rate_scheduler = nullptr;
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

        Logger::logMessage(std::format("Adam_Optimizer::saveCheckpoint: Saving checkpoint at timestep={}, learning_rate={}",
                                       timestep,
                                       learning_rate),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        std::uint64_t timestep_value = static_cast<std::uint64_t>(timestep);
        _output_file_stream.write(reinterpret_cast<const char *>(&timestep_value), sizeof(timestep_value));
        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&beta1), sizeof(beta1));
        _output_file_stream.write(reinterpret_cast<const char *>(&beta2), sizeof(beta2));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
        _output_file_stream.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));

        std::uint32_t state_count = static_cast<std::uint32_t>(parameter_order.size());
        _output_file_stream.write(reinterpret_cast<const char *>(&state_count), sizeof(state_count));

        for (Matrix *parameter : parameter_order)
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

        std::uint64_t timestep_value = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&timestep_value), sizeof(timestep_value));
        timestep = static_cast<std::size_t>(timestep_value);

        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&beta1), sizeof(beta1));
        _input_file_stream.read(reinterpret_cast<char *>(&beta2), sizeof(beta2));
        _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
        _input_file_stream.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));

        std::uint32_t state_count = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&state_count), sizeof(state_count));

        Logger::logMessage(std::format("Adam_Optimizer::loadCheckpoint: Loaded timestep={}, learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, state_count={}",
                                       timestep,
                                       learning_rate,
                                       beta1,
                                       beta2,
                                       epsilon,
                                       max_gradient,
                                       state_count),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        loaded_states.reserve(state_count);
        for (std::uint32_t i = 0; i < state_count; ++i)
        {
            Matrix first_moment = Matrix::loadMatrix(_input_file_stream, _execution_target);
            Matrix second_moment = Matrix::loadMatrix(_input_file_stream, _execution_target);
            loaded_states.emplace_back(std::move(first_moment), std::move(second_moment));
        }
    }
};
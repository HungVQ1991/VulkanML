#include "optimizer/adam_optimizer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"


Adam_Optimizer::Adam_Optimizer(float _learning_rate, float _beta1, float _beta2, float _epsilon, float _max_gradient, float _weight_decay)
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

Adam_Optimizer::Adam_Optimizer(ILearning_Rate &_learning_rate_scheduler, float _beta1, float _beta2, float _epsilon, float _max_gradient, float _weight_decay)
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

void Adam_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs)
{
        step(_parameter_gradient_pairs, 1.0f);
    }

void Adam_Optimizer::stepDynamicParams(float _grad_scale)
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

void Adam_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale)
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

void Adam_Optimizer::reset()
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

void Adam_Optimizer::saveCheckpoint(std::ofstream &_output_file_stream) const
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

void Adam_Optimizer::loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target)
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

void Adam_Optimizer::setLoadedStates(const std::vector<Parameter_State> &_loaded_states)
{ loaded_states = _loaded_states; }

void Adam_Optimizer::setParameterOrder(const std::vector<Tensor *> &_parameter_order)
{ parameter_order = _parameter_order; }

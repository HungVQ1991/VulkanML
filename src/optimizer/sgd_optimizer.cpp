#include "optimizer/sgd_optimizer.h"

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


Sgd_Optimizer::Sgd_Optimizer(float _learning_rate, float _max_gradient)
    : learning_rate(_learning_rate),
          max_gradient(_max_gradient),
          learning_rate_scheduler(nullptr)
{
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Sgd_Optimizer::Sgd_Optimizer: Initial learning rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
    }

Sgd_Optimizer::Sgd_Optimizer(ILearning_Rate &_learning_rate_scheduler, float _max_gradient)
    : learning_rate(_learning_rate_scheduler.getCurrentRate()),
          max_gradient(_max_gradient),
          learning_rate_scheduler(&_learning_rate_scheduler)
{
        if (learning_rate <= 0.0f)
        {
            Logger::logMessage("Sgd_Optimizer::Sgd_Optimizer: Initial learning rate is non-positive",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::OPTIMIZER_STEP);
        }
    }

void Sgd_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs)
{
        step(_parameter_gradient_pairs, 1.0f);
    }

void Sgd_Optimizer::stepDynamicParams(float _grad_scale)
{
        if (learning_rate_scheduler != nullptr)
        {
            learning_rate = learning_rate_scheduler->getCurrentRate();
        }
        float inv_scale = (_grad_scale > 0.0f) ? (1.0f / _grad_scale) : 1.0f;
        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0f, 1.0f, inv_scale, current_frame);
    }

void Sgd_Optimizer::step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale)
{
        stepDynamicParams(_grad_scale);

        Logger::logMessage(Input_Format{"Sgd_Optimizer::step: learning_rate={}, pairs_count={}",
                                        learning_rate,
                                        _parameter_gradient_pairs.size()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);

        float inv_scale = (_grad_scale > 0.0f) ? (1.0f / _grad_scale) : 1.0f;

        for (const auto &[parameter, gradient] : _parameter_gradient_pairs)
        {
            if (parameter && gradient)
            {
                parameter->sgdUpdate(*gradient, learning_rate, max_gradient, inv_scale);
            }
            else
            {
                Logger::logMessage("Sgd_Optimizer::step: Null parameter or gradient pointer encountered",
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::OPTIMIZER_STEP);
            }
        }
    }

void Sgd_Optimizer::reset()
{
        Logger::logMessage("Sgd_Optimizer::reset: Resetting SGD optimizer",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

void Sgd_Optimizer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        if (!_output_file_stream.is_open())
        {
            Logger::logMessage("Sgd_Optimizer::saveCheckpoint: Output stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return;
        }

        Logger::logMessage("Sgd_Optimizer::saveCheckpoint: Saving SGD checkpoint",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        _output_file_stream.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
        _output_file_stream.write(reinterpret_cast<const char *>(&max_gradient), sizeof(max_gradient));
    }

void Sgd_Optimizer::loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target)
{
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Sgd_Optimizer::loadCheckpoint: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return;
        }

        _input_file_stream.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
        _input_file_stream.read(reinterpret_cast<char *>(&max_gradient), sizeof(max_gradient));

        Logger::logMessage(Input_Format{"Sgd_Optimizer::loadCheckpoint: Loaded learning_rate={}, max_gradient={}",
                                        learning_rate,
                                        max_gradient},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
    }

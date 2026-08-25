#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/cost_function.h"
#include "helper/layer.h"
#include "helper/learning_rate.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/optimizer.h"

class Training_Context
{
private:
    std::size_t current_epoch = 0;
    std::unique_ptr<ICost_Function> cost_function;
    std::unique_ptr<ILearning_Rate> learning_rate_scheduler;
    std::unique_ptr<IOptimizer> optimizer;

    void createCostFunction(Cost_Type _cost_type, std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU)
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Training_Context::createCostFunction: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Input stream is not open");
        }

        Logger::logMessage(std::format("Training_Context::createCostFunction: Cost type = {}",
                                       magic_enum::enum_name(_cost_type)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        switch (_cost_type)
        {
        case Cost_Type::MSE:
            cost_function = std::make_unique<Mse_Cost>(_execution_target);
            break;
        case Cost_Type::MAE:
            cost_function = std::make_unique<Mae_Cost>(_execution_target);
            break;
        case Cost_Type::BCE:
            cost_function = std::make_unique<Bce_Cost>(1e-7f, _execution_target);
            break;
        case Cost_Type::CCE:
            cost_function = std::make_unique<Cce_Cost>(1e-7f, _execution_target);
            break;
        default:
            Logger::logMessage("Training_Context::createCostFunction: Unknown cost type",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Unknown cost type");
        }
        cost_function->loadCheckpoint(_input_file_stream);
    }

    void createLearningRateScheduler(Decay_Mode _decay_mode, std::ifstream &_input_file_stream)
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Training_Context::createLearningRateScheduler: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Input stream is not open");
        }

        Logger::logMessage(std::format("Training_Context::createLearningRateScheduler: Decay mode = {}",
                                       magic_enum::enum_name(_decay_mode)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        switch (_decay_mode)
        {
        case Decay_Mode::NO_DECAY:
            learning_rate_scheduler = std::make_unique<No_Decay>();
            break;
        case Decay_Mode::STEP_DECAY:
            learning_rate_scheduler = std::make_unique<Step_Decay>();
            break;
        case Decay_Mode::MULTI_STEP_DECAY:
            learning_rate_scheduler = std::make_unique<Multi_Step_Decay>();
            break;
        case Decay_Mode::EXPONENTIAL_DECAY:
            learning_rate_scheduler = std::make_unique<Exponential_Decay>();
            break;
        case Decay_Mode::COSINE_ANNEALING:
            learning_rate_scheduler = std::make_unique<Cosine_Annealing>(0.001f, 0.0f, 1);
            break;
        case Decay_Mode::POLYNOMIAL_DECAY:
            learning_rate_scheduler = std::make_unique<Polynomial_Decay>();
            break;
        case Decay_Mode::REDUCE_ON_PLATEAU:
            learning_rate_scheduler = std::make_unique<Reduce_On_Plateau>();
            break;
        default:
            Logger::logMessage("Training_Context::createLearningRateScheduler: Unknown decay mode",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Unknown decay mode");
        }
        learning_rate_scheduler->loadCheckpoint(_input_file_stream);
        Logger::logMessage(std::format("Training_Context::createLearningRateScheduler: Learning rate: {}",
                                       learning_rate_scheduler->getCurrentRate()),
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
    }

    void createOptimizer(Optimizer_Type _optimizer_type, std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU)
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Training_Context::createOptimizer: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Input stream is not open");
        }

        if (!learning_rate_scheduler)
        {
            Logger::logMessage("Training_Context::createOptimizer: Learning rate scheduler must be created before optimizer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Learning rate scheduler must be created before optimizer");
        }

        Logger::logMessage(std::format("Training_Context::createOptimizer: Optimizer type = {}",
                                       magic_enum::enum_name(_optimizer_type)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        switch (_optimizer_type)
        {
        case Optimizer_Type::SGD_OPTIMIZER:
            optimizer = std::make_unique<Sgd_Optimizer>(*learning_rate_scheduler);
            break;
        case Optimizer_Type::ADAM_OPTIMIZER:
            optimizer = std::make_unique<Adam_Optimizer>(*learning_rate_scheduler, 0.9f, 0.999f, 1e-8f, 1.0f);
            break;
        default:
            Logger::logMessage("Training_Context::createOptimizer: Unknown optimizer type",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Unknown optimizer type");
        }

        optimizer->loadCheckpoint(_input_file_stream, _execution_target);
    }

public:
    Training_Context() = default;

     std::size_t getCurrentEpoch() const noexcept
    {
        return current_epoch;
    }

    void setCurrentEpoch(std::size_t _epoch) noexcept
    {
        current_epoch = _epoch;
    }

     ICost_Function &getCostFunction()
    {
        return *cost_function;
    }

     const ICost_Function &getCostFunction() const
    {
        return *cost_function;
    }

     ILearning_Rate &getLearningRate()
    {
        return *learning_rate_scheduler;
    }

     const ILearning_Rate &getLearningRate() const
    {
        return *learning_rate_scheduler;
    }

     ILearning_Rate &getLearningRateScheduler()
    {
        return *learning_rate_scheduler;
    }

     const ILearning_Rate &getLearningRateScheduler() const
    {
        return *learning_rate_scheduler;
    }

     IOptimizer &getOptimizer()
    {
        return *optimizer;
    }

     const IOptimizer &getOptimizer() const
    {
        return *optimizer;
    }

    void setLearningRate(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler)
    {
        if (!_learning_rate_scheduler)
        {
            Logger::logMessage("Training_Context::setLearningRate: Attempted to set null learning rate scheduler",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::TRAINING);
        }
        learning_rate_scheduler = std::move(_learning_rate_scheduler);
    }

    void setLearningRateScheduler(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler)
    {
        setLearningRate(std::move(_learning_rate_scheduler));
    }

    void setOptimizer(std::unique_ptr<IOptimizer> _optimizer)
    {
        if (!_optimizer)
        {
            Logger::logMessage("Training_Context::setOptimizer: Attempted to set null optimizer",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::TRAINING);
        }
        optimizer = std::move(_optimizer);
    }

    void setCostFunction(std::unique_ptr<ICost_Function> _cost_function)
    {
        if (!_cost_function)
        {
            Logger::logMessage("Training_Context::setCostFunction: Attempted to set null cost function",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::TRAINING);
        }
        cost_function = std::move(_cost_function);
    }

    static std::unique_ptr<ILayer> constructLayerFromConfig(std::ifstream &_input_file_stream, Layer_Type _layer_type, Execution_Target _execution_target)
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Training_Context::constructLayerFromConfig: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Input stream is not open");
        }

        Logger::logMessage(std::format("Training_Context::constructLayerFromConfig: Constructing layer type = {}",
                                       magic_enum::enum_name(_layer_type)),
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        switch (_layer_type)
        {
        case Layer_Type::LINEAR:
        {
            std::uint32_t input_dimension = 0;
            std::uint32_t output_dimension = 0;
            _input_file_stream.read(reinterpret_cast<char *>(&input_dimension), sizeof(input_dimension));
            _input_file_stream.read(reinterpret_cast<char *>(&output_dimension), sizeof(output_dimension));
            if (input_dimension == 0 || output_dimension == 0)
            {
                Logger::logMessage("Training_Context::constructLayerFromConfig: Linear layer dimension is zero",
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::MODEL_SERIALIZATION);
            }
            return std::make_unique<Linear_Layer>(input_dimension, output_dimension, _execution_target);
        }
        case Layer_Type::CONV2D:
        {
            std::uint32_t input_height = 0;
            std::uint32_t input_width = 0;
            std::uint32_t input_channels = 0;
            std::uint32_t output_channels = 0;
            std::uint32_t kernel_size = 0;
            std::uint32_t stride = 0;
            std::uint32_t padding = 0;
            _input_file_stream.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
            _input_file_stream.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
            _input_file_stream.read(reinterpret_cast<char *>(&input_channels), sizeof(input_channels));
            _input_file_stream.read(reinterpret_cast<char *>(&output_channels), sizeof(output_channels));
            _input_file_stream.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
            _input_file_stream.read(reinterpret_cast<char *>(&stride), sizeof(stride));
            _input_file_stream.read(reinterpret_cast<char *>(&padding), sizeof(padding));
            return std::make_unique<Conv2d_Layer>(input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, _execution_target);
        }
        case Layer_Type::BATCH_NORM:
        {
            std::uint32_t feature_count = 0;
            float epsilon = 0.0f;
            float momentum = 0.0f;
            _input_file_stream.read(reinterpret_cast<char *>(&feature_count), sizeof(feature_count));
            _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
            _input_file_stream.read(reinterpret_cast<char *>(&momentum), sizeof(momentum));
            return std::make_unique<Batch_Norm_Layer>(feature_count, epsilon, momentum, _execution_target);
        }
        case Layer_Type::BATCH_NORM_2D:
        {
            std::uint32_t input_height = 0;
            std::uint32_t input_width = 0;
            std::uint32_t channels = 0;
            float epsilon = 0.0f;
            float momentum = 0.0f;
            _input_file_stream.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
            _input_file_stream.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
            _input_file_stream.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            _input_file_stream.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
            _input_file_stream.read(reinterpret_cast<char *>(&momentum), sizeof(momentum));
            return std::make_unique<Batch_Norm_2d_Layer>(input_height, input_width, channels, epsilon, momentum, _execution_target);
        }
        case Layer_Type::GLOBAL_AVG_POOL_2D:
        {
            std::uint32_t input_height = 0;
            std::uint32_t input_width = 0;
            std::uint32_t channels = 0;
            _input_file_stream.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
            _input_file_stream.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
            _input_file_stream.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            return std::make_unique<Global_Avg_Pool_2d_Layer>(input_height, input_width, channels, _execution_target);
        }
        case Layer_Type::MAX_POOL_2D:
        {
            std::uint32_t input_height = 0;
            std::uint32_t input_width = 0;
            std::uint32_t channels = 0;
            std::uint32_t kernel_size = 0;
            std::uint32_t stride = 0;
            std::uint32_t padding = 0;
            _input_file_stream.read(reinterpret_cast<char *>(&input_height), sizeof(input_height));
            _input_file_stream.read(reinterpret_cast<char *>(&input_width), sizeof(input_width));
            _input_file_stream.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            _input_file_stream.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
            _input_file_stream.read(reinterpret_cast<char *>(&stride), sizeof(stride));
            _input_file_stream.read(reinterpret_cast<char *>(&padding), sizeof(padding));
            return std::make_unique<Max_Pool_2d_Layer>(input_height, input_width, channels, kernel_size, stride, padding, _execution_target);
        }
        case Layer_Type::SOFTMAX:
        {
            std::uint8_t is_fused_value = 0;
            _input_file_stream.read(reinterpret_cast<char *>(&is_fused_value), sizeof(is_fused_value));
            bool is_fused = (is_fused_value != 0);
            return std::make_unique<Softmax_Layer>(is_fused, _execution_target);
        }
        case Layer_Type::RELU:
        {
            return std::make_unique<Relu_Layer>(_execution_target);
        }
        case Layer_Type::GELU:
        {
            return std::make_unique<Gelu_Layer>(_execution_target);
        }
        default:
            Logger::logMessage("Training_Context::constructLayerFromConfig: Unsupported layer type",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Unsupported or unknown layer type");
        }
    }

    bool loadHeader(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU)
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Training_Context::loadHeader: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return false;
        }

        char magic_header[4];
        _input_file_stream.read(magic_header, 4);
        if (magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'C' || magic_header[3] != 'K')
        {
            Logger::logMessage("Training_Context::loadHeader: Invalid magic header",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            return false;
        }

        std::uint32_t epoch_value = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&epoch_value), sizeof(epoch_value));
        current_epoch = static_cast<std::size_t>(epoch_value);
        Logger::logMessage(std::format("Training_Context::loadHeader: Current epoch = {}", current_epoch),
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        Cost_Type cost_type;
        _input_file_stream.read(reinterpret_cast<char *>(&cost_type), sizeof(cost_type));
        Logger::logMessage(std::format("Training_Context::loadHeader: Cost function type = {}",
                                       magic_enum::enum_name(cost_type)),
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        createCostFunction(cost_type, _input_file_stream, _execution_target);

        Decay_Mode decay_mode;
        _input_file_stream.read(reinterpret_cast<char *>(&decay_mode), sizeof(decay_mode));
        Logger::logMessage(std::format("Training_Context::loadHeader: Decay mode = {}",
                                       magic_enum::enum_name(decay_mode)),
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        createLearningRateScheduler(decay_mode, _input_file_stream);

        Optimizer_Type optimizer_type;
        _input_file_stream.read(reinterpret_cast<char *>(&optimizer_type), sizeof(optimizer_type));
        Logger::logMessage(std::format("Training_Context::loadHeader: Optimizer type = {}",
                                       magic_enum::enum_name(optimizer_type)),
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        createOptimizer(optimizer_type, _input_file_stream, _execution_target);

        return true;
    }
};
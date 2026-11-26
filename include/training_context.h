#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cost_function/icost_function.h"
#include "layer/ilayer.h"
#include "learning_rate/ilearning_rate.h"
#include "optimizer/ioptimizer.h"

class Training_Context
{
private:
    size_t current_epoch = 0;
    std::unique_ptr<ICost_Function> cost_function;
    std::unique_ptr<ILearning_Rate> learning_rate_scheduler;
    std::unique_ptr<IOptimizer> optimizer;

    void createCostFunction(Cost_Type _cost_type, std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU);
    void createLearningRateScheduler(Decay_Mode _decay_mode, std::ifstream &_input_file_stream);
    void createOptimizer(Optimizer_Type _optimizer_type, std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU);

public:
    Training_Context() = default;
    ~Training_Context() = default;

    Training_Context(const Training_Context &) = delete;
    Training_Context &operator=(const Training_Context &) = delete;
    Training_Context(Training_Context &&) noexcept = default;
    Training_Context &operator=(Training_Context &&) noexcept = default;

    static std::unique_ptr<ILayer> constructLayerFromConfig(std::ifstream &_input_file_stream, Layer_Type _layer_type, Execution_Target _execution_target);
    bool loadHeader(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU);

    const ILearning_Rate &getLearningRateScheduler() const
    {
        if (!learning_rate_scheduler)
        {
            throw std::logic_error("Training_Context::getLearningRateScheduler: Learning rate scheduler is not set");
        }
        return *learning_rate_scheduler;
    }
    ILearning_Rate &getLearningRateScheduler()
    {
        if (!learning_rate_scheduler)
        {
            throw std::logic_error("Training_Context::getLearningRateScheduler: Learning rate scheduler is not set");
        }
        return *learning_rate_scheduler;
    }
    const ICost_Function &getCostFunction() const
    {
        if (!cost_function)
        {
            throw std::logic_error("Training_Context::getCostFunction: Cost function is not set");
        }
        return *cost_function;
    }
    ICost_Function &getCostFunction()
    {
        if (!cost_function)
        {
            throw std::logic_error("Training_Context::getCostFunction: Cost function is not set");
        }
        return *cost_function;
    }
    const IOptimizer &getOptimizer() const
    {
        if (!optimizer)
        {
            throw std::logic_error("Training_Context::getOptimizer: Optimizer is not set");
        }
        return *optimizer;
    }
    IOptimizer &getOptimizer()
    {
        if (!optimizer)
        {
            throw std::logic_error("Training_Context::getOptimizer: Optimizer is not set");
        }
        return *optimizer;
    }
    size_t getCurrentEpoch() const noexcept { return current_epoch; }
    bool hasCostFunction() const noexcept { return cost_function != nullptr; }
    bool hasOptimizer() const noexcept { return optimizer != nullptr; }
    bool hasLearningRateScheduler() const noexcept { return learning_rate_scheduler != nullptr; }

    void setLearningRateScheduler(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler);
    void setOptimizer(std::unique_ptr<IOptimizer> _optimizer);
    void setCostFunction(std::unique_ptr<ICost_Function> _cost_function);
    void setCurrentEpoch(size_t _epoch) noexcept { current_epoch = _epoch; }
};

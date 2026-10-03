#pragma once

#include <cstddef>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "ioptimizer.h"
#include "math/tensor.h"

class Sgd_Optimizer : public IOptimizer
{
private:
    float learning_rate = 0.01f;
    float max_gradient = 1.0f;
    ILearning_Rate *learning_rate_scheduler = nullptr;

public:    explicit Sgd_Optimizer(float _learning_rate = 0.01f, float _max_gradient = 1.0f);
    explicit Sgd_Optimizer(ILearning_Rate &_learning_rate_scheduler, float _max_gradient = 1.0f);


    ~Sgd_Optimizer() noexcept override = default;    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs) override;
    void stepDynamicParams(float _grad_scale = 1.0f) override;
    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale) override;
    void reset() override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) override;


    ILearning_Rate *getLearningRateScheduler() const noexcept { return learning_rate_scheduler; }
    Optimizer_Type getType() const noexcept override { return Optimizer_Type::SGD_OPTIMIZER; }
    float getLearningRate() const noexcept override { return learning_rate; }
    float getMaxGradient() const noexcept override { return max_gradient; }

    void setLearningRateScheduler(ILearning_Rate *_learning_rate_scheduler) noexcept { learning_rate_scheduler = _learning_rate_scheduler; }
    void setLearningRate(float _learning_rate) override
    {
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
        learning_rate_scheduler = nullptr;
    }
};;
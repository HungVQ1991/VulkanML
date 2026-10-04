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

public:    explicit Adam_Optimizer(float _learning_rate = 0.001f,
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


    ~Adam_Optimizer() noexcept override = default;    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs) override;
    void stepDynamicParams(float _grad_scale = 1.0f) override;
    void step(const std::vector<std::pair<Tensor *, Tensor *>> &_parameter_gradient_pairs, float _grad_scale) override;
    void reset() override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU) override;


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
    void setParameterStates(const std::unordered_map<Tensor *, Parameter_State> &_parameter_states) { parameter_states = _parameter_states; }    void setLoadedStates(const std::vector<Parameter_State> &_loaded_states);
    void setParameterOrder(const std::vector<Tensor *> &_parameter_order);

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
};;
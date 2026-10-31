#pragma once

#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class Dropout_Layer : public ILayer
{
private:
    float dropout_rate = 0.5f;
    bool is_training = true;
    uint64_t seed = 0;
    Tensor mask_tensor;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    explicit Dropout_Layer(float _dropout_rate = 0.5f,
                           Execution_Target _execution_target = Execution_Target::CPU,
                           uint64_t _seed = 0);

    ~Dropout_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    std::unique_ptr<ILayer> clone() const override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::function<float(std::mt19937 &)> getPopulationParameterInitializer(size_t /*param_index*/) const override
    {
        return [](std::mt19937 &) { return 0.0f; };
    }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }

    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getMask() const noexcept { return mask_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::DROPOUT; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }
    bool isTraining() const noexcept { return is_training; }
    float getDropoutRate() const noexcept { return dropout_rate; }

    void setPopulationParameter(size_t /*param_index*/, std::vector<float> /*flat_data*/) override {}
    void setTrainingMode(bool _is_training) override { is_training = _is_training; }
    void setDropoutRate(float _rate);
    void setSeed(uint64_t _seed) noexcept { seed = _seed; }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;
};

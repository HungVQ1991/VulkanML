#pragma once

#include <cstddef>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class Gelu_Layer : public ILayer
{
private:
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;    explicit Gelu_Layer(Execution_Target _execution_target = Execution_Target::CPU);


    ~Gelu_Layer() noexcept override = default;    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    std::unique_ptr<ILayer> clone() const override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;


    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override { return [](std::mt19937&) { return 0.0f; }; }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::GELU; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }    void setExecutionTarget(Execution_Target _new_execution_target) override;

};;
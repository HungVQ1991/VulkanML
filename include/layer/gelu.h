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
    using ILayer::forward;
    explicit Gelu_Layer(Execution_Target _execution_target = Execution_Target::CPU)
        : input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
    }

    ~Gelu_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override
    {
        Logger::logMessage(Input_Format{"Gelu_Layer::forward: rows={}, columns={}",
                                        _input_tensor.getRows(),
                                        _input_tensor.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_tensor = _input_tensor;
        input_tensor.gelu(output_tensor);
        is_forward_completed = true;
        return output_tensor;
    }

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Gelu_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Gelu_Layer expects 0 parameters");
        }

        Tensor output_tensor_result(_batched_input.getExecutionTarget());
        _batched_input.gelu(output_tensor_result);
        return output_tensor_result;
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Gelu_Layer::backward: GeLU backward called before forward"},
                               Log_Level::LOG_ERROR,
                                true,
                                0,
                                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("GeLU backward called before forward");
        }

        if (_output_gradient.getRows() != output_tensor.getRows() || _output_gradient.getColumns() != output_tensor.getColumns())
        {
            Logger::logMessage(Input_Format{"Gelu_Layer::backward: GeLU gradient dimensions must match output dimensions"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("GeLU gradient dimensions must match output dimensions");
        }

        Logger::logMessage(Input_Format{"Gelu_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        logBufferAddress(&input_tensor, "input_tensor (Backward)");

        input_tensor.geluBackward(_output_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Gelu_Layer>(execution_target);
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override {}
    void saveInference(std::ofstream &_output_file_stream) const override {}
    void loadInference(std::ifstream &_input_file_stream) override {}
    void saveCheckpoint(std::ofstream &_output_file_stream) const override {}
    void loadCheckpoint(std::ifstream &_input_file_stream) override {}

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override { return [](std::mt19937&) { return 0.0f; }; }
    std::vector<Shape> getPopulationParameterDims() const override { return {}; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {}; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::GELU; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return false; }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        throw std::out_of_range("Gelu_Layer::setPopulationParameter: Layer has no parameters");
    }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
};

using GeLU = Gelu_Layer;
#pragma once

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class Relu_Layer : public ILayer
{
private:
    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;
    bool is_forward_completed = false;
    bool is_accumulated = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    explicit Relu_Layer(Execution_Target _execution_target = Execution_Target::CPU)
        : input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_forward_completed(false),
          is_accumulated(false),
          execution_target(_execution_target)
    {
    }

    ~Relu_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(Input_Format{"Relu_Layer::forward: rows={}, columns={}",
                                        _input_matrix.getRows(),
                                        _input_matrix.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.relu(output_matrix);
        is_forward_completed = true;
        return output_matrix;
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

        Tensor output_tensor(_batched_input.getExecutionTarget());
        _batched_input.gelu(output_tensor);
        return output_tensor;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Relu_Layer::backward: Relu backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Relu backward called before forward");
        }

        if (_output_gradient.getRows() != output_matrix.getRows() || _output_gradient.getColumns() != output_matrix.getColumns())
        {
            Logger::logMessage(Input_Format{"Relu_Layer::backward: Relu gradient dimensions must match output dimensions"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::invalid_argument("Relu gradient dimensions must match output dimensions");
        }

        Logger::logMessage(Input_Format{"Relu_Layer::backward: output_gradient rows={}, columns={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        output_matrix.reluBackward(_output_gradient, input_gradient);
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    bool hasParameters() const noexcept override
    {
        return false;
    }

    Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::RELU;
    }

    const Matrix &getInput() const override
    {
        return input_matrix;
    }

    const Matrix &getOutput() const override
    {
        return output_matrix;
    }

    std::vector<Shape> getPopulationParameterDims() const override
    {
        return {};
    }

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        return {};
    }

    bool supportsPopulationBatch() const noexcept override
    {
        return true;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Relu_Layer>(execution_target);
    }

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override
    {
        return [](std::mt19937&)
            {
                return 0.0f;
            };
    }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        throw std::out_of_range("Relu_Layer::setPopulationParameter: Layer has no parameters");
    }

    bool isAccumulated() const noexcept
    {
        return is_accumulated;
    }

    void setAccumulated(bool _is_accumulated) noexcept
    {
        is_accumulated = _is_accumulated;
    }

    Execution_Target getExecutionTarget() const override { return execution_target; }

    void saveConfiguration(std::ofstream &_output_file_stream) const override {}
    void saveInference(std::ofstream &_output_file_stream) const override {}
    void loadInference(std::ifstream &_input_file_stream) override {}
    void saveCheckpoint(std::ofstream &_output_file_stream) const override {}
    void loadCheckpoint(std::ifstream &_input_file_stream) override {}

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);

        execution_target = _new_execution_target;
        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};

using ReLU = Relu_Layer;
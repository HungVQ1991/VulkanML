#pragma once

#include <cstdint>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class Softmax_Layer : public ILayer
{
private:
    Matrix input_matrix;
    Matrix cached_output_matrix;
    Matrix input_gradient;
    bool is_fused_with_loss = false;
    bool is_accumulated = false;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    explicit Softmax_Layer(bool _is_fused_with_loss = false, Execution_Target _execution_target = Execution_Target::CPU)
        : input_matrix(0, 0, _execution_target),
          cached_output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
          is_fused_with_loss(_is_fused_with_loss),
          execution_target(_execution_target),
          is_accumulated(false),
          is_forward_completed(false)
    {
    }

    ~Softmax_Layer() noexcept override = default;

    Matrix forward(const Matrix &_input_matrix) override
    {
        Logger::logMessage(Input_Format{"Softmax_Layer::forward: rows={}, columns={}",
                                        _input_matrix.getRows(),
                                        _input_matrix.getColumns()},
                           Log_Level::LOG_DEBUG,
                           true,
                           1,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        input_matrix = _input_matrix;
        input_matrix.softmax(cached_output_matrix);
        return cached_output_matrix;
    }

    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override
    {
        if (!_batched_params.empty())
        {
            Logger::logMessage(Input_Format{ "Softmax_Layer::forward: expected 0 batched parameter tensors, got {}",
                                            _batched_params.size() },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
            throw std::invalid_argument("Invalid input params size: Softmax_Layer expects 0 parameters");
        }

        Tensor output_tensor(_batched_input.getExecutionTarget());
        _batched_input.softmax(output_tensor);
        return output_tensor;
    }

    std::vector<Shape> getPopulationParameterDims() const override
    {
        return {};
    }

    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        return {};
    }

    bool supportsPopulationBatch() const override
    {
        return true;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Softmax_Layer>(is_fused_with_loss, execution_target);
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
        throw std::out_of_range("Softmax_Layer::setPopulationParameter: Layer has no parameters");
    }

    bool isAccumulated() const noexcept
    {
        return is_accumulated;
    }

    void setAccumulated(bool _is_accumulated) noexcept
    {
        is_accumulated = _is_accumulated;
    }


    Matrix backward(const Matrix& _output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{ "Softmax_Layer::backward: Backward called before forward" },
                Log_Level::LOG_ERROR,
                true,
                0,
                Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Logger::logMessage(Input_Format{ "Softmax_Layer::backward: output_gradient rows={}, columns={}, is_fused_with_loss={}",
                                        _output_gradient.getRows(),
                                        _output_gradient.getColumns(),
                                        is_fused_with_loss },
            Log_Level::LOG_DEBUG,
            true,
            1,
            Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        logBufferAddress(&input_matrix, "input_matrix (Backward)");
        if (is_fused_with_loss)
        {
            return _output_gradient;
        }

        cached_output_matrix.softmaxBackward(_output_gradient, input_gradient);
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
        return Layer_Type::SOFTMAX;
    }

    const Matrix &getInput() const override
    {
        return input_matrix;
    }

    const Matrix &getOutput() const override
    {
        return cached_output_matrix;
    }

    Execution_Target getExecutionTarget() const override { return execution_target; }

    bool isFusedWithLoss() const noexcept
    {
        return is_fused_with_loss;
    }

    void setFusedWithLoss(bool _is_fused_with_loss) noexcept
    {
        is_fused_with_loss = _is_fused_with_loss;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        std::uint8_t fused_value = is_fused_with_loss ? 1 : 0;
        _output_file_stream.write(reinterpret_cast<const char *>(&fused_value), sizeof(fused_value));
    }

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
        cached_output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
    }
};

using Softmax = Softmax_Layer;
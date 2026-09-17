#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "batch_norm2d_layer.h"
#include "conv2d_layer.h"
#include "gelu.h"
#include "globalavgpool2d_layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "linear_layer.h"
#include "math/tensor.h"
#include "res_net_block_2d_layer.h"

class Res_Net_20_Layer : public ILayer
{
private:
    std::uint32_t input_height = 32;
    std::uint32_t input_width = 32;
    std::uint32_t input_channels = 3;
    std::uint32_t num_classes = 100;

    Execution_Target execution_target = Execution_Target::CPU;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    bool is_training = true;
    bool is_accumulated = false;

    std::vector<std::unique_ptr<ILayer>> layers;

    void buildNetwork()
    {
        layers.clear();
        layers.reserve(14);

        std::uint32_t current_h = input_height;
        std::uint32_t current_w = input_width;

        layers.push_back(std::make_unique<Conv2d_Layer>(current_h, current_w, input_channels, 16, 3, 1, 1, execution_target));
        layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(current_h, current_w, 16, 1e-5f, 0.1f, execution_target));
        layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

        auto add_stage = [this, &current_h, &current_w](std::uint32_t _in_channels, std::uint32_t _out_channels, std::uint32_t _stride, std::size_t _count)
        {
            for (std::size_t i = 0; i < _count; ++i)
            {
                std::uint32_t block_stride = (i == 0) ? _stride : 1;
                std::uint32_t block_in_channels = (i == 0) ? _in_channels : _out_channels;
                layers.push_back(std::make_unique<Res_Net_Block_2d_Layer>(
                    current_h, current_w, block_in_channels, _out_channels, block_stride, execution_target));
                current_h = (current_h + block_stride - 1) / block_stride;
                current_w = (current_w + block_stride - 1) / block_stride;
            }
        };

        add_stage(16, 16, 1, 3);
        add_stage(16, 32, 2, 3);
        add_stage(32, 64, 2, 3);

        layers.push_back(std::make_unique<Global_Avg_Pool_2d_Layer>(current_h, current_w, 64, execution_target));
        layers.push_back(std::make_unique<Linear_Layer>(64, num_classes, execution_target));
    }

public:
    using ILayer::forward;
    Res_Net_20_Layer(
        std::uint32_t _height = 32,
        std::uint32_t _width = 32,
        std::uint32_t _input_channels = 3,
        std::uint32_t _num_classes = 100,
        Execution_Target _execution_target = Execution_Target::CPU)
        : input_height(_height),
          input_width(_width),
          input_channels(_input_channels),
          num_classes(_num_classes),
          execution_target(_execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false)
    {
        buildNetwork();
    }

    ~Res_Net_20_Layer() noexcept override = default;

    std::unique_ptr<ILayer> clone() const override
    {
        return std::make_unique<Res_Net_20_Layer>(
            input_height, input_width, input_channels, num_classes, execution_target);
    }

    Tensor forward(const Tensor &_input_tensor) override
    {
        input_tensor = _input_tensor;

        Tensor current_tensor = input_tensor;
        for (auto &layer : layers)
        {
            current_tensor = layer->forward(current_tensor);
        }
        output_tensor = current_tensor;

        is_forward_completed = true;
        return output_tensor;
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Res_Net_20_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Tensor current_gradient = _output_gradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            current_gradient = (*it)->backward(current_gradient);
        }
        input_gradient_tensor = current_gradient;

        return input_gradient_tensor;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
        for (auto &layer : layers)
        {
            layer->resetGradient();
        }
    }

    void resetGradients() override
    {
        resetGradient();
        for (auto &layer : layers)
        {
            layer->resetGradients();
        }
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_channels), sizeof(input_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&num_classes), sizeof(num_classes));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        for (const auto &layer : layers)
        {
            layer->saveInference(_output_file_stream);
        }
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        for (auto &layer : layers)
        {
            layer->loadInference(_input_file_stream);
        }
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        for (const auto &layer : layers)
        {
            layer->saveCheckpoint(_output_file_stream);
        }
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        for (auto &layer : layers)
        {
            layer->loadCheckpoint(_input_file_stream);
        }
    }

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override
    {
        std::size_t current_offset = 0;
        for (const auto& layer : layers)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_20_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override
    {
        std::vector<std::pair<Tensor *, Tensor *>> all_parameters;
        for (auto &layer : layers)
        {
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        return all_parameters;
    }
    std::vector<float> getPopulationParameter(std::size_t param_index) const override
    {
        std::size_t current_offset = 0;
        for (const auto &layer : layers)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_20_Layer::getPopulationParameter: Parameter index out of range");
    }
    std::vector<Shape> getPopulationParameterDims() const override
    {
        std::vector<Shape> total_dims;
        for (const auto& layer : layers)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        return total_dims;
    }
    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        std::vector<bool> total_evolvable;
        for (const auto& layer : layers)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        return total_evolvable;
    }
    const std::vector<std::unique_ptr<ILayer>> &getLayers() const noexcept { return layers; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    std::uint32_t getInputChannels() const noexcept { return input_channels; }
    std::uint32_t getInputHeight() const noexcept { return input_height; }
    std::uint32_t getNumClasses() const noexcept { return num_classes; }
    std::uint32_t getInputWidth() const noexcept { return input_width; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::RES_NET_20; }
    bool supportsPopulationBatch() const noexcept override
    {
        for (const auto& layer : layers)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        return true;
    }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool isAccumulated() const noexcept { return is_accumulated; }
    bool hasParameters() const noexcept override { return true; }
    bool isTraining() const noexcept { return is_training; }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        std::size_t current_offset = 0;
        for (auto& layer : layers)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_20_Layer::setPopulationParameter: Parameter index out of range");
    }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (_new_execution_target == execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);
        execution_target = _new_execution_target;

        for (auto &layer : layers)
        {
            layer->setExecutionTarget(_new_execution_target);
        }

        input_tensor.setExecutionTarget(_new_execution_target);
        output_tensor.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    void setInputChannels(std::uint32_t _channels) noexcept { input_channels = _channels; }
    void setInputHeight(std::uint32_t _height) noexcept { input_height = _height; }
    void setNumClasses(std::uint32_t _classes) noexcept { num_classes = _classes; }
    void setInputWidth(std::uint32_t _width) noexcept { input_width = _width; }
    void setAccumulated(bool _is_accumulated) noexcept override
    {
        is_accumulated = _is_accumulated;
        for (auto &layer : layers)
        {
            layer->setAccumulated(_is_accumulated);
        }
    }
    void setTrainingMode(bool _is_training) override
    {
        is_training = _is_training;
        for (auto &layer : layers)
        {
            layer->setTrainingMode(_is_training);
        }
    }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
    void setIsTraining(bool _is_training) noexcept { is_training = _is_training; }
};

using Res_Net_20 = Res_Net_20_Layer;
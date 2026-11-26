#include "layer/res_net_20_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"

void Res_Net_20_Layer::buildNetwork()
{
        layers.clear();
        layers.reserve(14);

        uint32_t current_h = input_height;
        uint32_t current_w = input_width;

        layers.push_back(std::make_unique<Conv2d_Layer>(current_h, current_w, input_channels, 16, 3, 1, 1, execution_target, data_type));
        layers.push_back(std::make_unique<Batch_Norm_2d_Layer>(current_h, current_w, 16, 1e-5f, 0.1f, execution_target));
        layers.push_back(std::make_unique<Gelu_Layer>(execution_target));

        auto add_stage = [this, &current_h, &current_w](uint32_t _in_channels, uint32_t _out_channels, uint32_t _stride, size_t _count)
        {
            for (size_t i = 0; i < _count; ++i)
            {
                uint32_t block_stride = (i == 0) ? _stride : 1;
                uint32_t block_in_channels = (i == 0) ? _in_channels : _out_channels;
                layers.push_back(std::make_unique<Res_Net_Block_2d_Layer>(
                    current_h, current_w, block_in_channels, _out_channels, block_stride, execution_target, true, data_type));
                current_h = (current_h + block_stride - 1) / block_stride;
                current_w = (current_w + block_stride - 1) / block_stride;
            }
        };

        add_stage(16, 16, 1, 3);
        add_stage(16, 32, 2, 3);
        add_stage(32, 64, 2, 3);

        layers.push_back(std::make_unique<Global_Avg_Pool_2d_Layer>(current_h, current_w, 64, execution_target));
        layers.push_back(std::make_unique<Linear_Layer>(64, num_classes, execution_target, 2.0f, data_type));
    }

Res_Net_20_Layer::Res_Net_20_Layer(uint32_t _height, uint32_t _width, uint32_t _input_channels, uint32_t _num_classes, Execution_Target _execution_target, Data_Type _data_type)
    : input_height(_height),
          input_width(_width),
          input_channels(_input_channels),
          num_classes(_num_classes),
          execution_target(_execution_target),
          data_type(_data_type),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false)
{
    if (input_height == 0 || input_width == 0 || input_channels == 0 || num_classes == 0)
    {
        throw std::invalid_argument("Res_Net_20_Layer: Dimensions and num_classes must be > 0");
    }
    if (static_cast<uint64_t>(input_height) * input_width * input_channels > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Res_Net_20_Layer: H * W * C exceeds uint32_t limit");
    }
    buildNetwork();
}

void Res_Net_20_Layer::prepareInference()
{
    for (auto &layer : layers) layer->prepareInference();
}

void Res_Net_20_Layer::quantizeToTernary()
{
    data_type = Data_Type::TERNARY_INT2;
    for (auto &layer : layers)
    {
        if (layer) layer->quantizeToTernary();
    }
}

void Res_Net_20_Layer::dequantizeFromTernary()
{
    data_type = is_mixed_precision_enabled ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    for (auto &layer : layers)
    {
        if (layer) layer->dequantizeFromTernary();
    }
}

void Res_Net_20_Layer::snapshotNonTrainableState(std::vector<std::vector<float>> &_out) const
{
    for (const auto &layer : layers)
    {
        if (layer) layer->snapshotNonTrainableState(_out);
    }
}

void Res_Net_20_Layer::restoreNonTrainableState(const std::vector<std::vector<float>> &_in, size_t &_cursor)
{
    for (auto &layer : layers)
    {
        if (layer) layer->restoreNonTrainableState(_in, _cursor);
    }
}

std::unique_ptr<ILayer> Res_Net_20_Layer::clone() const
{
    auto cloned = std::make_unique<Res_Net_20_Layer>(
        input_height, input_width, input_channels, num_classes, execution_target, data_type);
    cloned->layers.clear();
    cloned->layers.reserve(layers.size());
    for (const auto &layer : layers)
    {
        cloned->layers.push_back(layer ? layer->clone() : nullptr);
    }
    cloned->setTrainingMode(is_training);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setAccumulated(is_accumulated);
    return cloned;
}

Tensor Res_Net_20_Layer::forward(const Tensor &_input_tensor)
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

Tensor Res_Net_20_Layer::backward(const Tensor &_output_gradient)
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

void Res_Net_20_Layer::resetGradient()
{
        is_forward_completed = false;
        for (auto &layer : layers)
        {
            layer->resetGradient();
        }
    }

void Res_Net_20_Layer::resetGradients()
{
    is_forward_completed = false;
    for (auto &layer : layers)
    {
        layer->resetGradients();
    }
}

void Res_Net_20_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_channels), sizeof(input_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&num_classes), sizeof(num_classes));
    }

void Res_Net_20_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        for (const auto &layer : layers)
        {
            layer->saveInference(_output_file_stream);
        }
    }

void Res_Net_20_Layer::loadInference(std::ifstream &_input_file_stream)
{
    for (auto &layer : layers)
    {
        layer->loadInference(_input_file_stream);
    }
    bool any_ternary = false;
    for (const auto &layer : layers)
    {
        if (layer && layer->isTernary())
        {
            any_ternary = true;
            break;
        }
    }
    data_type = any_ternary ? Data_Type::TERNARY_INT2 : Data_Type::FLOAT32;
}

void Res_Net_20_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        for (const auto &layer : layers)
        {
            layer->saveCheckpoint(_output_file_stream);
        }
    }

void Res_Net_20_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    for (auto &layer : layers)
    {
        layer->loadCheckpoint(_input_file_stream);
    }
    bool any_ternary = false;
    for (const auto &layer : layers)
    {
        if (layer && layer->isTernary())
        {
            any_ternary = true;
            break;
        }
    }
    data_type = any_ternary ? Data_Type::TERNARY_INT2 : Data_Type::FLOAT32;
}

std::function<float(std::mt19937&)> Res_Net_20_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto& layer : layers)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_20_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }

std::vector<std::pair<Tensor *, Tensor *>> Res_Net_20_Layer::getParametersAndGradients()
{
        std::vector<std::pair<Tensor *, Tensor *>> all_parameters;
        for (auto &layer : layers)
        {
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        return all_parameters;
    }

std::vector<const Tensor *> Res_Net_20_Layer::getNoDecayParameters() const
{
    std::vector<const Tensor *> no_decay;
    for (const auto &layer : layers)
    {
        if (layer)
        {
            auto params = layer->getNoDecayParameters();
            no_decay.insert(no_decay.end(), params.begin(), params.end());
        }
    }
    return no_decay;
}

std::vector<float> Res_Net_20_Layer::getPopulationParameter(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto &layer : layers)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_20_Layer::getPopulationParameter: Parameter index out of range");
    }

const std::vector<std::unique_ptr<ILayer>> & Res_Net_20_Layer::getLayers() const noexcept
{ return layers; }

uint32_t Res_Net_20_Layer::getNumClasses() const noexcept
{ return num_classes; }

bool Res_Net_20_Layer::isTraining() const noexcept
{ return is_training; }

void Res_Net_20_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        size_t current_offset = 0;
        for (auto& layer : layers)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_20_Layer::setPopulationParameter: Parameter index out of range");
    }

void Res_Net_20_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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

void Res_Net_20_Layer::setAccumulated(bool _is_accumulated) noexcept
{
    ILayer::setAccumulated(_is_accumulated);
    for (auto &layer : layers)
    {
        layer->setAccumulated(_is_accumulated);
    }
}

void Res_Net_20_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    for (auto &layer : layers)
    {
        layer->setMixedPrecision(_enable);
    }
}

void Res_Net_20_Layer::invalidateWeightCache() noexcept
{
    for (auto &layer : layers)
    {
        layer->invalidateWeightCache();
    }
}

void Res_Net_20_Layer::setTrainingMode(bool _is_training)
{
        is_training = _is_training;
        for (auto &layer : layers)
        {
            layer->setTrainingMode(_is_training);
        }
    }

void Res_Net_20_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

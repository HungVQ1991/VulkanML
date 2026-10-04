#include "layer/res_net_block_2d_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
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


Res_Net_Block_2d_Layer::Res_Net_Block_2d_Layer(Execution_Target _execution_target)
    : input_tensor(0, 0, _execution_target),
          main_branch_output(0, 0, _execution_target),
          shortcut_branch_output(0, 0, _execution_target),
          sum_tensor(0, 0, _execution_target),
          final_output(0, 0, _execution_target),
          main_gradient(0, 0, _execution_target),
          shortcut_gradient(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
{}

Res_Net_Block_2d_Layer::Res_Net_Block_2d_Layer(uint32_t _height, uint32_t _width, uint32_t _in_channels, uint32_t _out_channels, uint32_t _stride, Execution_Target _execution_target)
    : Res_Net_Block_2d_Layer(_execution_target)
{
        uint32_t out_h = (_height + _stride - 1) / _stride;
        uint32_t out_w = (_width + _stride - 1) / _stride;

        addMainLayer<Conv2d_Layer>(_height, _width, _in_channels, _out_channels, 3, _stride, 1, _execution_target);
        addMainLayer<Batch_Norm_2d_Layer>(out_h, out_w, _out_channels, 1e-5f, 0.1f, _execution_target);
        addMainLayer<Gelu_Layer>(_execution_target);
        addMainLayer<Conv2d_Layer>(out_h, out_w, _out_channels, _out_channels, 3, 1, 1, _execution_target);
        addMainLayer<Batch_Norm_2d_Layer>(out_h, out_w, _out_channels, 1e-5f, 0.1f, _execution_target);

        if (_stride != 1 || _in_channels != _out_channels)
        {
            addShortcutLayer<Conv2d_Layer>(_height, _width, _in_channels, _out_channels, 1, _stride, 0, _execution_target);
            addShortcutLayer<Batch_Norm_2d_Layer>(out_h, out_w, _out_channels, 1e-5f, 0.1f, _execution_target);
        }

        setPostActivation<Gelu_Layer>(_execution_target);
    }

void Res_Net_Block_2d_Layer::addMainLayer(std::unique_ptr<ILayer> _layer)
{
        main_branch.push_back(std::move(_layer));
    }

void Res_Net_Block_2d_Layer::addShortcutLayer(std::unique_ptr<ILayer> _layer)
{
        shortcut_branch.push_back(std::move(_layer));
    }

void Res_Net_Block_2d_Layer::setPostActivation(std::unique_ptr<ILayer> _layer)
{
        post_activation = std::move(_layer);
    }

std::unique_ptr<ILayer> Res_Net_Block_2d_Layer::clone() const
{
        auto cloned_layer = std::make_unique<Res_Net_Block_2d_Layer>(execution_target);
        for (const auto& layer : main_branch)
        {
            cloned_layer->addMainLayer(layer->clone());
        }
        for (const auto& layer : shortcut_branch)
        {
            cloned_layer->addShortcutLayer(layer->clone());
        }
        if (post_activation)
        {
            cloned_layer->setPostActivation(post_activation->clone());
        }
        return cloned_layer;
    }

Tensor Res_Net_Block_2d_Layer::forward(const Tensor &_input_tensor)
{
        input_tensor = _input_tensor;

        Tensor current_main = input_tensor;
        for (auto &layer : main_branch)
        {
            current_main = layer->forward(current_main);
        }
        main_branch_output = current_main;

        if (shortcut_branch.empty())
        {
            shortcut_branch_output = input_tensor;
        }
        else
        {
            Tensor current_shortcut = input_tensor;
            for (auto &layer : shortcut_branch)
            {
                current_shortcut = layer->forward(current_shortcut);
            }
            shortcut_branch_output = current_shortcut;
        }

        main_branch_output.add(shortcut_branch_output, sum_tensor);

        if (post_activation)
        {
            final_output = post_activation->forward(sum_tensor);
        }
        else
        {
            final_output = sum_tensor;
        }

        is_forward_completed = true;
        return final_output;
    }

Tensor Res_Net_Block_2d_Layer::backward(const Tensor &_output_gradient)
{
        if (!is_forward_completed)
        {
            Logger::logMessage(Input_Format{"Res_Net_Block_2d_Layer::backward: Backward called before forward"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Backward called before forward");
        }

        Tensor grad_sum = (post_activation != nullptr) ? post_activation->backward(_output_gradient) : _output_gradient;

        Tensor current_main_grad = grad_sum;
        for (auto it = main_branch.rbegin(); it != main_branch.rend(); ++it)
        {
            current_main_grad = (*it)->backward(current_main_grad);
        }
        main_gradient = current_main_grad;

        if (shortcut_branch.empty())
        {
            shortcut_gradient = grad_sum;
        }
        else
        {
            Tensor current_shortcut_grad = grad_sum;
            for (auto it = shortcut_branch.rbegin(); it != shortcut_branch.rend(); ++it)
            {
                current_shortcut_grad = (*it)->backward(current_shortcut_grad);
            }
            shortcut_gradient = current_shortcut_grad;
        }

        main_gradient.add(shortcut_gradient, input_gradient_tensor);
        return input_gradient_tensor;
    }

void Res_Net_Block_2d_Layer::resetGradient()
{
        is_forward_completed = false;
        for (auto &layer : main_branch)
        {
            layer->resetGradient();
        }
        for (auto &layer : shortcut_branch)
        {
            layer->resetGradient();
        }
        if (post_activation)
        {
            post_activation->resetGradient();
        }
    }

void Res_Net_Block_2d_Layer::resetGradients()
{
        resetGradient();
        for (auto &layer : main_branch)
        {
            layer->resetGradients();
        }
        for (auto &layer : shortcut_branch)
        {
            layer->resetGradients();
        }
        if (post_activation)
        {
            post_activation->resetGradients();
        }
    }

void Res_Net_Block_2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
        uint64_t main_count = static_cast<uint64_t>(main_branch.size());
        uint64_t shortcut_count = static_cast<uint64_t>(shortcut_branch.size());
        std::uint8_t has_post_act = (post_activation != nullptr) ? 1 : 0;

        _output_file_stream.write(reinterpret_cast<const char *>(&main_count), sizeof(main_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&shortcut_count), sizeof(shortcut_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&has_post_act), sizeof(has_post_act));

        for (const auto &layer : main_branch)
        {
            Layer_Type type = layer->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
            layer->saveConfiguration(_output_file_stream);
        }
        for (const auto &layer : shortcut_branch)
        {
            Layer_Type type = layer->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
            layer->saveConfiguration(_output_file_stream);
        }
        if (post_activation)
        {
            Layer_Type type = post_activation->getLayerType();
            _output_file_stream.write(reinterpret_cast<const char *>(&type), sizeof(type));
            post_activation->saveConfiguration(_output_file_stream);
        }
    }

void Res_Net_Block_2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
        for (const auto &layer : main_branch)
        {
            layer->saveInference(_output_file_stream);
        }
        for (const auto &layer : shortcut_branch)
        {
            layer->saveInference(_output_file_stream);
        }
        if (post_activation)
        {
            post_activation->saveInference(_output_file_stream);
        }
    }

void Res_Net_Block_2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
        for (auto &layer : main_branch)
        {
            layer->loadInference(_input_file_stream);
        }
        for (auto &layer : shortcut_branch)
        {
            layer->loadInference(_input_file_stream);
        }
        if (post_activation)
        {
            post_activation->loadInference(_input_file_stream);
        }
    }

void Res_Net_Block_2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
        for (const auto &layer : main_branch)
        {
            layer->saveCheckpoint(_output_file_stream);
        }
        for (const auto &layer : shortcut_branch)
        {
            layer->saveCheckpoint(_output_file_stream);
        }
        if (post_activation)
        {
            post_activation->saveCheckpoint(_output_file_stream);
        }
    }

void Res_Net_Block_2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
        for (auto &layer : main_branch)
        {
            layer->loadCheckpoint(_input_file_stream);
        }
        for (auto &layer : shortcut_branch)
        {
            layer->loadCheckpoint(_input_file_stream);
        }
        if (post_activation)
        {
            post_activation->loadCheckpoint(_input_file_stream);
        }
    }

std::function<float(std::mt19937&)> Res_Net_Block_2d_Layer::getPopulationParameterInitializer(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto& layer : main_branch)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto& layer : shortcut_branch)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        if (post_activation)
        {
            size_t count = post_activation->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return post_activation->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_Block_2d_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }

std::vector<std::pair<Tensor *, Tensor *>> Res_Net_Block_2d_Layer::getParametersAndGradients()
{
        std::vector<std::pair<Tensor *, Tensor *>> total_params;
        for (auto &layer : main_branch)
        {
            auto params = layer->getParametersAndGradients();
            total_params.insert(total_params.end(), params.begin(), params.end());
        }
        for (auto &layer : shortcut_branch)
        {
            auto params = layer->getParametersAndGradients();
            total_params.insert(total_params.end(), params.begin(), params.end());
        }
        if (post_activation && post_activation->hasParameters())
        {
            auto params = post_activation->getParametersAndGradients();
            total_params.insert(total_params.end(), params.begin(), params.end());
        }
        return total_params;
    }

std::vector<float> Res_Net_Block_2d_Layer::getPopulationParameter(size_t param_index) const
{
        size_t current_offset = 0;
        for (const auto &layer : main_branch)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto &layer : shortcut_branch)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        if (post_activation)
        {
            size_t count = post_activation->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return post_activation->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_Block_2d_Layer::getPopulationParameter: Parameter index out of range");
    }

const std::vector<std::unique_ptr<ILayer>> & Res_Net_Block_2d_Layer::getShortcutBranch() const noexcept
{ return shortcut_branch; }

const std::vector<std::unique_ptr<ILayer>> & Res_Net_Block_2d_Layer::getMainBranch() const noexcept
{ return main_branch; }

const Tensor & Res_Net_Block_2d_Layer::getShortcutOutput() const noexcept
{ return shortcut_branch_output; }

const Tensor & Res_Net_Block_2d_Layer::getShortcutGradient() const noexcept
{ return shortcut_gradient; }

const Tensor & Res_Net_Block_2d_Layer::getMainOutput() const noexcept
{ return main_branch_output; }

const Tensor & Res_Net_Block_2d_Layer::getMainGradient() const noexcept
{ return main_gradient; }

const Tensor & Res_Net_Block_2d_Layer::getFinalOutput() const noexcept
{ return final_output; }

const Tensor & Res_Net_Block_2d_Layer::getSumTensor() const noexcept
{ return sum_tensor; }

const ILayer * Res_Net_Block_2d_Layer::getPostActivation() const noexcept
{ return post_activation.get(); }

void Res_Net_Block_2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
        size_t current_offset = 0;
        for (auto& layer : main_branch)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        for (auto& layer : shortcut_branch)
        {
            size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        if (post_activation)
        {
            size_t count = post_activation->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                post_activation->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_Block_2d_Layer::setPopulationParameter: Parameter index out of range");
    }

void Res_Net_Block_2d_Layer::setShortcutOutput(const Tensor &_tensor)
{ shortcut_branch_output = _tensor; }

void Res_Net_Block_2d_Layer::setShortcutGradient(const Tensor &_tensor)
{ shortcut_gradient = _tensor; }

void Res_Net_Block_2d_Layer::setMainOutput(const Tensor &_tensor)
{ main_branch_output = _tensor; }

void Res_Net_Block_2d_Layer::setMainGradient(const Tensor &_tensor)
{ main_gradient = _tensor; }

void Res_Net_Block_2d_Layer::setFinalOutput(const Tensor &_tensor)
{ final_output = _tensor; }

void Res_Net_Block_2d_Layer::setSumTensor(const Tensor &_tensor)
{ sum_tensor = _tensor; }

void Res_Net_Block_2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
        if (_new_execution_target == execution_target)
        {
            return;
        }

        logChangeExecutionTarget(_new_execution_target);
        execution_target = _new_execution_target;

        for (auto &layer : main_branch)
        {
            layer->setExecutionTarget(_new_execution_target);
        }
        for (auto &layer : shortcut_branch)
        {
            layer->setExecutionTarget(_new_execution_target);
        }
        if (post_activation)
        {
            post_activation->setExecutionTarget(_new_execution_target);
        }

        input_tensor.setExecutionTarget(_new_execution_target);
        main_branch_output.setExecutionTarget(_new_execution_target);
        shortcut_branch_output.setExecutionTarget(_new_execution_target);
        sum_tensor.setExecutionTarget(_new_execution_target);
        final_output.setExecutionTarget(_new_execution_target);
        main_gradient.setExecutionTarget(_new_execution_target);
        shortcut_gradient.setExecutionTarget(_new_execution_target);
        input_gradient_tensor.setExecutionTarget(_new_execution_target);
    }

void Res_Net_Block_2d_Layer::setAccumulated(bool _is_accumulated) noexcept
{
        is_accumulated = _is_accumulated;
        for (auto &layer : main_branch)
        {
            layer->setAccumulated(_is_accumulated);
        }
        for (auto &layer : shortcut_branch)
        {
            layer->setAccumulated(_is_accumulated);
        }
        if (post_activation)
        {
            post_activation->setAccumulated(_is_accumulated);
        }
    }

void Res_Net_Block_2d_Layer::setTrainingMode(bool _is_training)
{
        for (auto &layer : main_branch)
        {
            layer->setTrainingMode(_is_training);
        }
        for (auto &layer : shortcut_branch)
        {
            layer->setTrainingMode(_is_training);
        }
        if (post_activation)
        {
            post_activation->setTrainingMode(_is_training);
        }
    }

void Res_Net_Block_2d_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

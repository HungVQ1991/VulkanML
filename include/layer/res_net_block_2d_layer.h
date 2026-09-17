#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class Res_Net_Block_2d_Layer : public ILayer
{
private:
    std::vector<std::unique_ptr<ILayer>> main_branch;
    std::vector<std::unique_ptr<ILayer>> shortcut_branch;
    std::unique_ptr<ILayer> post_activation;

    Tensor input_tensor;
    Tensor main_branch_output;
    Tensor shortcut_branch_output;
    Tensor sum_tensor;
    Tensor final_output;

    Tensor main_gradient;
    Tensor shortcut_gradient;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    using ILayer::forward;
    explicit Res_Net_Block_2d_Layer(Execution_Target _execution_target = Execution_Target::CPU)
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

    Res_Net_Block_2d_Layer(
        std::uint32_t _height,
        std::uint32_t _width,
        std::uint32_t _in_channels,
        std::uint32_t _out_channels,
        std::uint32_t _stride = 1,
        Execution_Target _execution_target = Execution_Target::CPU)
        : Res_Net_Block_2d_Layer(_execution_target)
    {
        std::uint32_t out_h = (_height + _stride - 1) / _stride;
        std::uint32_t out_w = (_width + _stride - 1) / _stride;

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

    ~Res_Net_Block_2d_Layer() noexcept override = default;

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addMainLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        main_branch.push_back(std::move(new_layer));
        return layer_reference;
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addShortcutLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        shortcut_branch.push_back(std::move(new_layer));
        return layer_reference;
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &setPostActivation(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        post_activation = std::move(new_layer);
        return layer_reference;
    }

    void addMainLayer(std::unique_ptr<ILayer> _layer)
    {
        main_branch.push_back(std::move(_layer));
    }

    void addShortcutLayer(std::unique_ptr<ILayer> _layer)
    {
        shortcut_branch.push_back(std::move(_layer));
    }

    void setPostActivation(std::unique_ptr<ILayer> _layer)
    {
        post_activation = std::move(_layer);
    }

    std::unique_ptr<ILayer> clone() const override
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

    Tensor forward(const Tensor &_input_tensor) override
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

    Tensor backward(const Tensor &_output_gradient) override
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

    void resetGradient() override
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

    void resetGradients() override
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

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        std::uint64_t main_count = static_cast<std::uint64_t>(main_branch.size());
        std::uint64_t shortcut_count = static_cast<std::uint64_t>(shortcut_branch.size());
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

    void saveInference(std::ofstream &_output_file_stream) const override
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

    void loadInference(std::ifstream &_input_file_stream) override
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

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
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

    void loadCheckpoint(std::ifstream &_input_file_stream) override
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

    std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const override
    {
        std::size_t current_offset = 0;
        for (const auto& layer : main_branch)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto& layer : shortcut_branch)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        if (post_activation)
        {
            std::size_t count = post_activation->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return post_activation->getPopulationParameterInitializer(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_Block_2d_Layer::getPopulationParameterInitializer: Parameter index out of range");
    }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override
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
    std::vector<float> getPopulationParameter(std::size_t param_index) const override
    {
        std::size_t current_offset = 0;
        for (const auto &layer : main_branch)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        for (const auto &layer : shortcut_branch)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return layer->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        if (post_activation)
        {
            std::size_t count = post_activation->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                return post_activation->getPopulationParameter(param_index - current_offset);
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_Block_2d_Layer::getPopulationParameter: Parameter index out of range");
    }
    std::vector<Shape> getPopulationParameterDims() const override
    {
        std::vector<Shape> total_dims;
        for (const auto& layer : main_branch)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        for (const auto& layer : shortcut_branch)
        {
            auto dims = layer->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        if (post_activation)
        {
            auto dims = post_activation->getPopulationParameterDims();
            total_dims.insert(total_dims.end(), dims.begin(), dims.end());
        }
        return total_dims;
    }
    std::vector<bool> getPopulationParameterIsEvolvable() const override
    {
        std::vector<bool> total_evolvable;
        for (const auto& layer : main_branch)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        for (const auto& layer : shortcut_branch)
        {
            auto evolvable = layer->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        if (post_activation)
        {
            auto evolvable = post_activation->getPopulationParameterIsEvolvable();
            total_evolvable.insert(total_evolvable.end(), evolvable.begin(), evolvable.end());
        }
        return total_evolvable;
    }
    const std::vector<std::unique_ptr<ILayer>> &getShortcutBranch() const noexcept { return shortcut_branch; }
    const std::vector<std::unique_ptr<ILayer>> &getMainBranch() const noexcept { return main_branch; }
    const Tensor &getShortcutOutput() const noexcept { return shortcut_branch_output; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getShortcutGradient() const noexcept { return shortcut_gradient; }
    const Tensor &getMainOutput() const noexcept { return main_branch_output; }
    const Tensor &getMainGradient() const noexcept { return main_gradient; }
    const Tensor &getFinalOutput() const noexcept { return final_output; }
    const Tensor &getSumTensor() const noexcept { return sum_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return final_output; }
    const ILayer *getPostActivation() const noexcept { return post_activation.get(); }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::RES_NET_BLOCK_2D; }
    bool supportsPopulationBatch() const noexcept override
    {
        for (const auto& layer : main_branch)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        for (const auto& layer : shortcut_branch)
        {
            if (!layer->supportsPopulationBatch())
            {
                return false;
            }
        }
        if (post_activation && !post_activation->supportsPopulationBatch())
        {
            return false;
        }
        return true;
    }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }

    void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) override
    {
        std::size_t current_offset = 0;
        for (auto& layer : main_branch)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        for (auto& layer : shortcut_branch)
        {
            std::size_t count = layer->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                layer->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        if (post_activation)
        {
            std::size_t count = post_activation->getPopulationParameterDims().size();
            if (param_index < current_offset + count)
            {
                post_activation->setPopulationParameter(param_index - current_offset, std::move(flat_data));
                return;
            }
            current_offset += count;
        }
        throw std::out_of_range("Res_Net_Block_2d_Layer::setPopulationParameter: Parameter index out of range");
    }
    void setShortcutOutput(const Tensor &_tensor) { shortcut_branch_output = _tensor; }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setShortcutGradient(const Tensor &_tensor) { shortcut_gradient = _tensor; }
    void setMainOutput(const Tensor &_tensor) { main_branch_output = _tensor; }
    void setMainGradient(const Tensor &_tensor) { main_gradient = _tensor; }
    void setFinalOutput(const Tensor &_tensor) { final_output = _tensor; }
    void setSumTensor(const Tensor &_tensor) { sum_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override
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
    void setAccumulated(bool _is_accumulated) noexcept override
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
    void setTrainingMode(bool _is_training) override
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
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
};
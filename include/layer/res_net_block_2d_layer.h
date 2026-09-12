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
#include "math/matrix.h"

class Res_Net_Block_2d_Layer : public ILayer
{
private:
    std::vector<std::unique_ptr<ILayer>> main_branch;
    std::vector<std::unique_ptr<ILayer>> shortcut_branch;
    std::unique_ptr<ILayer> post_activation;

    Matrix input_matrix;
    Matrix main_branch_output;
    Matrix shortcut_branch_output;
    Matrix sum_matrix;
    Matrix final_output;

    Matrix main_gradient;
    Matrix shortcut_gradient;
    Matrix input_gradient;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

public:
    explicit Res_Net_Block_2d_Layer(Execution_Target _execution_target = Execution_Target::CPU)
        : input_matrix(0, 0, _execution_target),
          main_branch_output(0, 0, _execution_target),
          shortcut_branch_output(0, 0, _execution_target),
          sum_matrix(0, 0, _execution_target),
          final_output(0, 0, _execution_target),
          main_gradient(0, 0, _execution_target),
          shortcut_gradient(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target),
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

        input_matrix.setExecutionTarget(_new_execution_target);
        main_branch_output.setExecutionTarget(_new_execution_target);
        shortcut_branch_output.setExecutionTarget(_new_execution_target);
        sum_matrix.setExecutionTarget(_new_execution_target);
        final_output.setExecutionTarget(_new_execution_target);
        main_gradient.setExecutionTarget(_new_execution_target);
        shortcut_gradient.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
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

    Matrix forward(const Matrix &_input_matrix) override
    {
        input_matrix = _input_matrix;

        Matrix current_main = input_matrix;
        for (auto &layer : main_branch)
        {
            current_main = layer->forward(current_main);
        }
        main_branch_output = current_main;

        if (shortcut_branch.empty())
        {
            shortcut_branch_output = input_matrix;
        }
        else
        {
            Matrix current_shortcut = input_matrix;
            for (auto &layer : shortcut_branch)
            {
                current_shortcut = layer->forward(current_shortcut);
            }
            shortcut_branch_output = current_shortcut;
        }

        main_branch_output.add(shortcut_branch_output, sum_matrix);

        if (post_activation)
        {
            final_output = post_activation->forward(sum_matrix);
        }
        else
        {
            final_output = sum_matrix;
        }

        is_forward_completed = true;
        return final_output;
    }

    Matrix backward(const Matrix &_output_gradient) override
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

        Matrix grad_sum = (post_activation != nullptr) ? post_activation->backward(_output_gradient) : _output_gradient;

        Matrix current_main_grad = grad_sum;
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
            Matrix current_shortcut_grad = grad_sum;
            for (auto it = shortcut_branch.rbegin(); it != shortcut_branch.rend(); ++it)
            {
                current_shortcut_grad = (*it)->backward(current_shortcut_grad);
            }
            shortcut_gradient = current_shortcut_grad;
        }

        main_gradient.add(shortcut_gradient, input_gradient);
        return input_gradient;
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

    bool hasParameters() const noexcept override
    {
        return true;
    }

    Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::RES_NET_BLOCK_2D;
    }

    Matrix getInput() override
    {
        return input_matrix;
    }

    Matrix getOutput() override
    {
        return final_output;
    }

    Matrix getMainOutput() const
    {
        return main_branch_output;
    }

    Matrix getShortcutOutput() const
    {
        return shortcut_branch_output;
    }

    Execution_Target getExecutionTarget() const override
    {
        return execution_target;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        std::vector<std::pair<Matrix *, Matrix *>> total_params;
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
            layer->saveConfiguration(_output_file_stream);
        }
        for (const auto &layer : shortcut_branch)
        {
            layer->saveConfiguration(_output_file_stream);
        }
        if (post_activation)
        {
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
};
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "batch_norm2d_layer.h"
#include "conv2d_layer.h"
#include "gelu.h"
#include "ilayer.h"
#include "math/matrix.h"

class Res_Net_Block_2d_Layer : public ILayer
{
private:
    std::uint32_t input_height = 0;
    std::uint32_t input_width = 0;
    std::uint32_t input_channels = 0;
    std::uint32_t output_height = 0;
    std::uint32_t output_width = 0;
    std::uint32_t output_channels = 0;
    std::uint32_t stride = 1;
    bool has_projection = false;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;

    Conv2d_Layer conv1;
    Batch_Norm_2d_Layer bn1;
    Gelu_Layer act1;
    Conv2d_Layer conv2;
    Batch_Norm_2d_Layer bn2;
    Gelu_Layer act2;

    std::unique_ptr<Conv2d_Layer> conv_proj;
    std::unique_ptr<Batch_Norm_2d_Layer> bn_proj;

    Matrix input_matrix;
    Matrix main_branch_output;
    Matrix shortcut_branch_output;
    Matrix sum_matrix;
    Matrix final_output;

    Matrix main_gradient;
    Matrix shortcut_gradient;
    Matrix input_gradient;

    std::vector<std::pair<Matrix *, Matrix *>> cached_parameters_and_gradients;

    void initializeParameterCache()
    {
        cached_parameters_and_gradients.clear();

        auto append_layer_params = [this](auto &layer)
        {
            auto params = layer.getParametersAndGradients();
            cached_parameters_and_gradients.insert(cached_parameters_and_gradients.end(), params.begin(), params.end());
        };

        append_layer_params(conv1);
        append_layer_params(bn1);
        append_layer_params(conv2);
        append_layer_params(bn2);

        if (has_projection && conv_proj && bn_proj)
        {
            append_layer_params(*conv_proj);
            append_layer_params(*bn_proj);
        }
    }

public:
    Res_Net_Block_2d_Layer(
        std::uint32_t _height,
        std::uint32_t _width,
        std::uint32_t _in_channels,
        std::uint32_t _out_channels,
        std::uint32_t _stride = 1,
        Execution_Target _execution_target = Execution_Target::CPU)
        : input_height(_height),
          input_width(_width),
          input_channels(_in_channels),
          output_channels(_out_channels),
          stride(_stride),
          has_projection(_stride != 1 || _in_channels != _out_channels),
          execution_target(_execution_target),
          conv1(_height, _width, _in_channels, _out_channels, 3, _stride, 1, _execution_target),
          bn1((_height + _stride - 1) / _stride, (_width + _stride - 1) / _stride, _out_channels, 1e-5f, 0.1f, _execution_target),
          act1(_execution_target),
          conv2((_height + _stride - 1) / _stride, (_width + _stride - 1) / _stride, _out_channels, _out_channels, 3, 1, 1, _execution_target),
          bn2((_height + _stride - 1) / _stride, (_width + _stride - 1) / _stride, _out_channels, 1e-5f, 0.1f, _execution_target),
          act2(_execution_target),
          input_matrix(0, 0, _execution_target),
          main_branch_output(0, 0, _execution_target),
          shortcut_branch_output(0, 0, _execution_target),
          sum_matrix(0, 0, _execution_target),
          final_output(0, 0, _execution_target),
          main_gradient(0, 0, _execution_target),
          shortcut_gradient(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target)
    {
        output_height = (_height + _stride - 1) / _stride;
        output_width = (_width + _stride - 1) / _stride;

        if (has_projection)
        {
            conv_proj = std::make_unique<Conv2d_Layer>(_height, _width, _in_channels, _out_channels, 1, _stride, 0, _execution_target);
            bn_proj = std::make_unique<Batch_Norm_2d_Layer>(output_height, output_width, _out_channels, 1e-5f, 0.1f, _execution_target);
        }

        initializeParameterCache();
    }

    ~Res_Net_Block_2d_Layer() noexcept override = default;

    void setTrainingMode(bool _is_training) override
    {
        conv1.setTrainingMode(_is_training);
        bn1.setTrainingMode(_is_training);
        conv2.setTrainingMode(_is_training);
        bn2.setTrainingMode(_is_training);

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->setTrainingMode(_is_training);
            bn_proj->setTrainingMode(_is_training);
        }
    }

    Matrix forward(const Matrix &_input_matrix) override
    {
        input_matrix = _input_matrix;

        Matrix x1 = conv1.forward(input_matrix);
        Matrix x2 = bn1.forward(x1);
        Matrix x3 = act1.forward(x2);
        Matrix x4 = conv2.forward(x3);
        main_branch_output = bn2.forward(x4);

        if (has_projection && conv_proj && bn_proj)
        {
            Matrix proj1 = conv_proj->forward(input_matrix);
            shortcut_branch_output = bn_proj->forward(proj1);
        }
        else
        {
            shortcut_branch_output = input_matrix;
        }

        main_branch_output.add(shortcut_branch_output, sum_matrix);
        final_output = act2.forward(sum_matrix);

        is_forward_completed = true;
        return final_output;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            throw std::logic_error("Backward called before forward");
        }

        Matrix grad_sum = act2.backward(_output_gradient);

        Matrix grad_bn2 = bn2.backward(grad_sum);
        Matrix grad_conv2 = conv2.backward(grad_bn2);
        Matrix grad_act1 = act1.backward(grad_conv2);
        Matrix grad_bn1 = bn1.backward(grad_act1);
        main_gradient = conv1.backward(grad_bn1);

        if (has_projection && conv_proj && bn_proj)
        {
            Matrix grad_bn_proj = bn_proj->backward(grad_sum);
            shortcut_gradient = conv_proj->backward(grad_bn_proj);
        }
        else
        {
            shortcut_gradient = grad_sum;
        }

        main_gradient.add(shortcut_gradient, input_gradient);
        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
        conv1.resetGradient();
        bn1.resetGradient();
        act1.resetGradient();
        conv2.resetGradient();
        bn2.resetGradient();
        act2.resetGradient();

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->resetGradient();
            bn_proj->resetGradient();
        }
    }

    [[nodiscard]] bool hasParameters() const noexcept override
    {
        return true;
    }

    [[nodiscard]] Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::RES_NET_BLOCK_2D;
    }

    [[nodiscard]] Matrix getInput() override
    {
        return input_matrix;
    }

    [[nodiscard]] Matrix getOutput() override
    {
        return final_output;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        return cached_parameters_and_gradients;
    }

    void setExecutionTarget(Execution_Target _new_execution_target) override
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }

        execution_target = _new_execution_target;

        conv1.setExecutionTarget(_new_execution_target);
        bn1.setExecutionTarget(_new_execution_target);
        act1.setExecutionTarget(_new_execution_target);
        conv2.setExecutionTarget(_new_execution_target);
        bn2.setExecutionTarget(_new_execution_target);
        act2.setExecutionTarget(_new_execution_target);

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->setExecutionTarget(_new_execution_target);
            bn_proj->setExecutionTarget(_new_execution_target);
        }

        input_matrix.setExecutionTarget(_new_execution_target);
        main_branch_output.setExecutionTarget(_new_execution_target);
        shortcut_branch_output.setExecutionTarget(_new_execution_target);
        sum_matrix.setExecutionTarget(_new_execution_target);
        final_output.setExecutionTarget(_new_execution_target);

        main_gradient.setExecutionTarget(_new_execution_target);
        shortcut_gradient.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);

        initializeParameterCache();
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
        _output_file_stream.write(reinterpret_cast<const char *>(&input_channels), sizeof(input_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&output_channels), sizeof(output_channels));
        _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        conv1.saveInference(_output_file_stream);
        bn1.saveInference(_output_file_stream);
        conv2.saveInference(_output_file_stream);
        bn2.saveInference(_output_file_stream);

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->saveInference(_output_file_stream);
            bn_proj->saveInference(_output_file_stream);
        }
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        conv1.loadInference(_input_file_stream);
        bn1.loadInference(_input_file_stream);
        conv2.loadInference(_input_file_stream);
        bn2.loadInference(_input_file_stream);

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->loadInference(_input_file_stream);
            bn_proj->loadInference(_input_file_stream);
        }
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        conv1.saveCheckpoint(_output_file_stream);
        bn1.saveCheckpoint(_output_file_stream);
        conv2.saveCheckpoint(_output_file_stream);
        bn2.saveCheckpoint(_output_file_stream);

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->saveCheckpoint(_output_file_stream);
            bn_proj->saveCheckpoint(_output_file_stream);
        }
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        conv1.loadCheckpoint(_input_file_stream);
        bn1.loadCheckpoint(_input_file_stream);
        conv2.loadCheckpoint(_input_file_stream);
        bn2.loadCheckpoint(_input_file_stream);

        if (has_projection && conv_proj && bn_proj)
        {
            conv_proj->loadCheckpoint(_input_file_stream);
            bn_proj->loadCheckpoint(_input_file_stream);
        }
    }
};
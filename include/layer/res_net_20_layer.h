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
#include "ilayer.h"
#include "linear_layer.h"
#include "math/matrix.h"
#include "res_net_block_2d_layer.h"

class Res_Net_20_Layer : public ILayer
{
private:
    std::uint32_t input_height = 32;
    std::uint32_t input_width = 32;
    std::uint32_t input_channels = 3;
    std::uint32_t num_classes = 100;

    bool is_forward_completed = false;
    bool is_training = true;
    Execution_Target execution_target = Execution_Target::CPU;

    Conv2d_Layer stem_conv;
    Batch_Norm_2d_Layer stem_bn;
    Gelu_Layer stem_act;

    Res_Net_Block_2d_Layer stage1_block1;
    Res_Net_Block_2d_Layer stage1_block2;
    Res_Net_Block_2d_Layer stage1_block3;

    Res_Net_Block_2d_Layer stage2_block1;
    Res_Net_Block_2d_Layer stage2_block2;
    Res_Net_Block_2d_Layer stage2_block3;

    Res_Net_Block_2d_Layer stage3_block1;
    Res_Net_Block_2d_Layer stage3_block2;
    Res_Net_Block_2d_Layer stage3_block3;

    Global_Avg_Pool_2d_Layer global_pool;
    Linear_Layer fc_out;

    Matrix input_matrix;
    Matrix output_matrix;
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

        append_layer_params(stem_conv);
        append_layer_params(stem_bn);

        append_layer_params(stage1_block1);
        append_layer_params(stage1_block2);
        append_layer_params(stage1_block3);

        append_layer_params(stage2_block1);
        append_layer_params(stage2_block2);
        append_layer_params(stage2_block3);

        append_layer_params(stage3_block1);
        append_layer_params(stage3_block2);
        append_layer_params(stage3_block3);

        append_layer_params(fc_out);
    }

public:
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
          stem_conv(_height, _width, _input_channels, 16, 3, 1, 1, _execution_target),
          stem_bn(_height, _width, 16, 1e-5f, 0.1f, _execution_target),
          stem_act(_execution_target),
          stage1_block1(_height, _width, 16, 16, 1, _execution_target),
          stage1_block2(_height, _width, 16, 16, 1, _execution_target),
          stage1_block3(_height, _width, 16, 16, 1, _execution_target),
          stage2_block1(_height, _width, 16, 32, 2, _execution_target),
          stage2_block2(_height / 2, _width / 2, 32, 32, 1, _execution_target),
          stage2_block3(_height / 2, _width / 2, 32, 32, 1, _execution_target),
          stage3_block1(_height / 2, _width / 2, 32, 64, 2, _execution_target),
          stage3_block2(_height / 4, _width / 4, 64, 64, 1, _execution_target),
          stage3_block3(_height / 4, _width / 4, 64, 64, 1, _execution_target),
          global_pool(_height / 4, _width / 4, 64, _execution_target),
          fc_out(64, _num_classes, _execution_target),
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target)
    {
        initializeParameterCache();
    }

    ~Res_Net_20_Layer() noexcept override = default;

    void setTrainingMode(bool _is_training) override
    {
        is_training = _is_training;
        stem_bn.setTrainingMode(_is_training);
        stage1_block1.setTrainingMode(_is_training);
        stage1_block2.setTrainingMode(_is_training);
        stage1_block3.setTrainingMode(_is_training);
        stage2_block1.setTrainingMode(_is_training);
        stage2_block2.setTrainingMode(_is_training);
        stage2_block3.setTrainingMode(_is_training);
        stage3_block1.setTrainingMode(_is_training);
        stage3_block2.setTrainingMode(_is_training);
        stage3_block3.setTrainingMode(_is_training);
    }

    Matrix forward(const Matrix &_input_matrix) override
    {
        input_matrix = _input_matrix;

        Matrix x = stem_conv.forward(input_matrix);
        x = stem_bn.forward(x);
        x = stem_act.forward(x);

        x = stage1_block1.forward(x);
        x = stage1_block2.forward(x);
        x = stage1_block3.forward(x);

        x = stage2_block1.forward(x);
        x = stage2_block2.forward(x);
        x = stage2_block3.forward(x);

        x = stage3_block1.forward(x);
        x = stage3_block2.forward(x);
        x = stage3_block3.forward(x);

        x = global_pool.forward(x);
        output_matrix = fc_out.forward(x);

        is_forward_completed = true;
        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            throw std::logic_error("Backward called before forward");
        }

        Matrix grad = fc_out.backward(_output_gradient);
        grad = global_pool.backward(grad);

        grad = stage3_block3.backward(grad);
        grad = stage3_block2.backward(grad);
        grad = stage3_block1.backward(grad);

        grad = stage2_block3.backward(grad);
        grad = stage2_block2.backward(grad);
        grad = stage2_block1.backward(grad);

        grad = stage1_block3.backward(grad);
        grad = stage1_block2.backward(grad);
        grad = stage1_block1.backward(grad);

        grad = stem_act.backward(grad);
        grad = stem_bn.backward(grad);
        input_gradient = stem_conv.backward(grad);

        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
        stem_conv.resetGradient();
        stem_bn.resetGradient();
        stem_act.resetGradient();

        stage1_block1.resetGradient();
        stage1_block2.resetGradient();
        stage1_block3.resetGradient();

        stage2_block1.resetGradient();
        stage2_block2.resetGradient();
        stage2_block3.resetGradient();

        stage3_block1.resetGradient();
        stage3_block2.resetGradient();
        stage3_block3.resetGradient();

        global_pool.resetGradient();
        fc_out.resetGradient();
    }

     bool hasParameters() const noexcept override
    {
        return true;
    }

     Layer_Type getLayerType() const noexcept override
    {
        return Layer_Type::RES_NET_20;
    }

     Matrix getInput() override
    {
        return input_matrix;
    }

     Matrix getOutput() override
    {
        return output_matrix;
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

        stem_conv.setExecutionTarget(_new_execution_target);
        stem_bn.setExecutionTarget(_new_execution_target);
        stem_act.setExecutionTarget(_new_execution_target);

        stage1_block1.setExecutionTarget(_new_execution_target);
        stage1_block2.setExecutionTarget(_new_execution_target);
        stage1_block3.setExecutionTarget(_new_execution_target);

        stage2_block1.setExecutionTarget(_new_execution_target);
        stage2_block2.setExecutionTarget(_new_execution_target);
        stage2_block3.setExecutionTarget(_new_execution_target);

        stage3_block1.setExecutionTarget(_new_execution_target);
        stage3_block2.setExecutionTarget(_new_execution_target);
        stage3_block3.setExecutionTarget(_new_execution_target);

        global_pool.setExecutionTarget(_new_execution_target);
        fc_out.setExecutionTarget(_new_execution_target);

        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);

        initializeParameterCache();
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
        stem_conv.saveInference(_output_file_stream);
        stem_bn.saveInference(_output_file_stream);

        stage1_block1.saveInference(_output_file_stream);
        stage1_block2.saveInference(_output_file_stream);
        stage1_block3.saveInference(_output_file_stream);

        stage2_block1.saveInference(_output_file_stream);
        stage2_block2.saveInference(_output_file_stream);
        stage2_block3.saveInference(_output_file_stream);

        stage3_block1.saveInference(_output_file_stream);
        stage3_block2.saveInference(_output_file_stream);
        stage3_block3.saveInference(_output_file_stream);

        fc_out.saveInference(_output_file_stream);
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        stem_conv.loadInference(_input_file_stream);
        stem_bn.loadInference(_input_file_stream);

        stage1_block1.loadInference(_input_file_stream);
        stage1_block2.loadInference(_input_file_stream);
        stage1_block3.loadInference(_input_file_stream);

        stage2_block1.loadInference(_input_file_stream);
        stage2_block2.loadInference(_input_file_stream);
        stage2_block3.loadInference(_input_file_stream);

        stage3_block1.loadInference(_input_file_stream);
        stage3_block2.loadInference(_input_file_stream);
        stage3_block3.loadInference(_input_file_stream);

        fc_out.loadInference(_input_file_stream);
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        stem_conv.saveCheckpoint(_output_file_stream);
        stem_bn.saveCheckpoint(_output_file_stream);

        stage1_block1.saveCheckpoint(_output_file_stream);
        stage1_block2.saveCheckpoint(_output_file_stream);
        stage1_block3.saveCheckpoint(_output_file_stream);

        stage2_block1.saveCheckpoint(_output_file_stream);
        stage2_block2.saveCheckpoint(_output_file_stream);
        stage2_block3.saveCheckpoint(_output_file_stream);

        stage3_block1.saveCheckpoint(_output_file_stream);
        stage3_block2.saveCheckpoint(_output_file_stream);
        stage3_block3.saveCheckpoint(_output_file_stream);

        fc_out.saveCheckpoint(_output_file_stream);
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        stem_conv.loadCheckpoint(_input_file_stream);
        stem_bn.loadCheckpoint(_input_file_stream);

        stage1_block1.loadCheckpoint(_input_file_stream);
        stage1_block2.loadCheckpoint(_input_file_stream);
        stage1_block3.loadCheckpoint(_input_file_stream);

        stage2_block1.loadCheckpoint(_input_file_stream);
        stage2_block2.loadCheckpoint(_input_file_stream);
        stage2_block3.loadCheckpoint(_input_file_stream);

        stage3_block1.loadCheckpoint(_input_file_stream);
        stage3_block2.loadCheckpoint(_input_file_stream);
        stage3_block3.loadCheckpoint(_input_file_stream);

        fc_out.loadCheckpoint(_input_file_stream);
    }
};

using Res_Net_20 = Res_Net_20_Layer;
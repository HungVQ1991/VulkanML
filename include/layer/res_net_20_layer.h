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

    std::vector<std::unique_ptr<ILayer>> layers;

    Matrix input_matrix;
    Matrix output_matrix;
    Matrix input_gradient;

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
          input_matrix(0, 0, _execution_target),
          output_matrix(0, 0, _execution_target),
          input_gradient(0, 0, _execution_target)
    {
        buildNetwork();
    }

    ~Res_Net_20_Layer() noexcept override = default;

    void setTrainingMode(bool _is_training) override
    {
        is_training = _is_training;
        for (auto &layer : layers)
        {
            layer->setTrainingMode(_is_training);
        }
    }

    Matrix forward(const Matrix &_input_matrix) override
    {
        input_matrix = _input_matrix;

        Matrix current_tensor = input_matrix;
        for (auto &layer : layers)
        {
            current_tensor = layer->forward(current_tensor);
        }
        output_matrix = current_tensor;

        is_forward_completed = true;
        return output_matrix;
    }

    Matrix backward(const Matrix &_output_gradient) override
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

        Matrix current_gradient = _output_gradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            current_gradient = (*it)->backward(current_gradient);
        }
        input_gradient = current_gradient;

        return input_gradient;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
        for (auto &layer : layers)
        {
            layer->resetGradient();
        }
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

    Execution_Target getExecutionTarget() const override
    {
        return execution_target;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() override
    {
        std::vector<std::pair<Matrix *, Matrix *>> all_parameters;
        for (auto &layer : layers)
        {
            auto params = layer->getParametersAndGradients();
            all_parameters.insert(all_parameters.end(), params.begin(), params.end());
        }
        return all_parameters;
    }

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

        input_matrix.setExecutionTarget(_new_execution_target);
        output_matrix.setExecutionTarget(_new_execution_target);
        input_gradient.setExecutionTarget(_new_execution_target);
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
};

using Res_Net_20 = Res_Net_20_Layer;
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <utility>
#include <windows.h>

#include "batch_norm_layer.h"
#include "conv2d_layer.h"
#include "layer.h"
#include "maxpool2d_layer.h"
#include "relu.h"
#include "gelu.h"
#include "softmax.h"
#include "globalavgpool2d_layer.h"
#include "learning_rate.h"
#include "ilayer.h"
#include "cost_function.h"
#include "math/execution_engine.h"
#include "math/matrix.h"
#include "math/logger.h"


class Neural_Network
{
private:
    std::vector<std::unique_ptr<ILayer>> layers;
    Matrix last_prediction;
    Execution_Target target = Execution_Target::CPU;

    bool is_target_synced = false;
    bool is_graph_built = false;
    Matrix persistent_input;
    Matrix persistent_target;

    void saveMatrix(std::ofstream &out_file, const Matrix &mat) const
    {
        std::uint64_t rows = static_cast<std::uint64_t>(mat.getRows());
        std::uint64_t cols = static_cast<std::uint64_t>(mat.getCols());
        out_file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        out_file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

        std::vector<float> host_data = mat.getData();
        out_file.write(reinterpret_cast<const char *>(host_data.data()), host_data.size() * sizeof(float));
    }

public:
    explicit Neural_Network(Execution_Target exec_target = Execution_Target::CPU)
        : last_prediction(0, 0, exec_target),
          target(exec_target),
          persistent_input(0, 0, exec_target),
          persistent_target(0, 0, exec_target) {}

    ~Neural_Network()
    {
        if (target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().waitIdle();
        }
    }

    void addLayer(std::unique_ptr<ILayer> layer)
    {
        layers.push_back(std::move(layer));
    }

    Matrix forward(const Matrix &input_matrix)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::forward: Neural network has no layers to execute forward pass", LOG_ERROR);
            throw std::logic_error("Neural network has no layers to execute forward pass");
        }

        Matrix current_output = input_matrix;
        for (const auto &layer : layers)
        {
            current_output = layer->forward(current_output);
        }
        last_prediction = current_output;
        return current_output;
    }

    Matrix backward(const Matrix &target_matrix, const ICostFunction &cost_fn)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::backward: Neural network has no layers to execute backward pass", LOG_ERROR);
            throw std::logic_error("Neural network has no layers to execute backward pass");
        }

        Matrix gradient_matrix = cost_fn.computeGradient(last_prediction, target_matrix);
        for (std::size_t i = layers.size(); i > 0; --i)
        {
            gradient_matrix = layers[i - 1]->backward(gradient_matrix);
        }
        return gradient_matrix;
    }

    std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads()
    {
        std::vector<std::pair<Matrix *, Matrix *>> param_grad_pairs;
        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                auto pairs = layer->getParamsAndGrads();
                param_grad_pairs.insert(param_grad_pairs.end(), pairs.begin(), pairs.end());
            }
        }
        return param_grad_pairs;
    }

    void reset()
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::reset: Neural network has no layers to reset gradients", LOG_ERROR);
            throw std::logic_error("Neural network has no layers to reset gradients");
        }

        for (const auto &layer : layers)
        {
            layer->resetGradient();
        }
        last_prediction = Matrix(0, 0, target);
    }

    void trainStep(const Matrix &input_matrix, const Matrix &target_matrix, const ICostFunction &cost_fn, float learning_rate, IOptimizer &optimizer)
    {
        if (!is_target_synced)
        {
            for (std::unique_ptr<ILayer> &layer : layers) layer->setTarget(target);
            is_target_synced = true;
        }
        optimizer.setLearningRate(learning_rate);
        Matrix pred = forward(input_matrix);
        backward(target_matrix, cost_fn);

        optimizer.step(getParamsAndGrads());
        reset();
        if (target == Execution_Target::VULKAN_GPU)
            Execution_Engine::getInstance().executeGraph();
    }

    float evaluate(const Matrix &input_matrix, const Matrix &target_matrix, const ICostFunction &cost_fn)
    {
        Matrix pred = forward(input_matrix);
        if (target == Execution_Target::VULKAN_GPU)
            Execution_Engine::getInstance().executeGraph();
        return cost_fn.computeLoss(pred, target_matrix);
    }

    void saveModel(const std::string &file_path) const
    {
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            Logger::logMessage("Neural_Network::saveModel: Failed to open file for saving model: " + file_path, LOG_ERROR);
            throw std::runtime_error("Failed to open file for saving model");
        }

        const char magic[4] = {'N', 'N', 'M', '1'};
        out_file.write(magic, 4);

        std::uint32_t total_layer_count = static_cast<std::uint32_t>(layers.size());
        out_file.write(reinterpret_cast<const char *>(&total_layer_count), sizeof(total_layer_count));

        for (const auto &layer : layers)
        {
            Layer_Type type_val = layer->getLayerType();
            out_file.write(reinterpret_cast<const char *>(&type_val), sizeof(type_val));
            layer->saveConfig(out_file);
        }

        for (const auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->saveState(out_file);
            }
        }
    }

    void loadModel(const std::string &file_path, Execution_Target exec_target = Execution_Target::CPU)
    {
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            Logger::logMessage("Neural_Network::loadModel: Failed to open file for loading model: " + file_path, LOG_ERROR);
            throw std::runtime_error("Failed to open file for loading model");
        }

        char magic[4];
        in_file.read(magic, 4);
        if (magic[0] != 'N' || magic[1] != 'N' || magic[2] != 'M' || magic[3] != '1')
        {
            Logger::logMessage("Neural_Network::loadModel: Invalid file format or magic header", LOG_ERROR);
            throw std::runtime_error("Invalid file format or magic header");
        }

        std::uint32_t total_layer_count = 0;
        in_file.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type type_val;
            in_file.read(reinterpret_cast<char *>(&type_val), sizeof(type_val));

            switch (type_val)
            {
            case Layer_Type::LINEAR:
            {
                std::uint32_t in_dim = 0;
                std::uint32_t out_dim = 0;
                in_file.read(reinterpret_cast<char *>(&in_dim), sizeof(in_dim));
                in_file.read(reinterpret_cast<char *>(&out_dim), sizeof(out_dim));
                layers.push_back(std::make_unique<Layer>(in_dim, out_dim, exec_target));
                break;
            }
            case Layer_Type::CONV2D:
            {
                std::uint32_t in_h = 0;
                std::uint32_t in_w = 0;
                std::uint32_t in_c = 0;
                std::uint32_t out_c = 0;
                std::uint32_t kernel_size = 0;
                std::uint32_t stride = 0;
                std::uint32_t padding = 0;
                in_file.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
                in_file.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
                in_file.read(reinterpret_cast<char *>(&in_c), sizeof(in_c));
                in_file.read(reinterpret_cast<char *>(&out_c), sizeof(out_c));
                in_file.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
                in_file.read(reinterpret_cast<char *>(&stride), sizeof(stride));
                in_file.read(reinterpret_cast<char *>(&padding), sizeof(padding));
                layers.push_back(std::make_unique<Conv2d_Layer>(in_h, in_w, in_c, out_c, kernel_size, stride, padding, exec_target));
                break;
            }
            case Layer_Type::BATCH_NORM:
            {
                std::uint32_t num_features = 0;
                float epsilon = 0.0f;
                float momentum = 0.0f;
                in_file.read(reinterpret_cast<char *>(&num_features), sizeof(num_features));
                in_file.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
                in_file.read(reinterpret_cast<char *>(&momentum), sizeof(momentum));
                layers.push_back(std::make_unique<Batch_Norm_Layer>(num_features, epsilon, momentum, exec_target));
                break;
            }
            case Layer_Type::GLOBAL_AVG_POOL_2D:
            {
                std::uint32_t in_h = 0;
                std::uint32_t in_w = 0;
                std::uint32_t channels = 0;
                in_file.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
                in_file.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
                in_file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
                layers.push_back(std::make_unique<GlobalAvgPool2d_Layer>(in_h, in_w, channels, exec_target));
                break;
            }
            case Layer_Type::MAX_POOL_2D:
            {
                std::uint32_t in_h = 0;
                std::uint32_t in_w = 0;
                std::uint32_t channels = 0;
                std::uint32_t kernel_size = 0;
                std::uint32_t stride = 0;
                std::uint32_t padding = 0;
                in_file.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
                in_file.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
                in_file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
                in_file.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
                in_file.read(reinterpret_cast<char *>(&stride), sizeof(stride));
                in_file.read(reinterpret_cast<char *>(&padding), sizeof(padding));
                layers.push_back(std::make_unique<MaxPool2d_Layer>(in_h, in_w, channels, kernel_size, stride, padding, exec_target));
                break;
            }
            case Layer_Type::SOFTMAX:
            {
                std::uint8_t fused_val = 0;
                in_file.read(reinterpret_cast<char *>(&fused_val), sizeof(fused_val));
                bool fused = (fused_val != 0);
                layers.push_back(std::make_unique<Softmax>(fused, exec_target));
                break;
            }
            default:
                Logger::logMessage("Neural_Network::loadModel: Unsupported or unknown layer type", LOG_ERROR);
                throw std::runtime_error("Unsupported or unknown layer type");
            }
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadState(in_file);
            }
        }
    }

    const Matrix &getLastPrediction() const
    {
        return last_prediction;
    }

    const ILayer &getLayer(size_t index) const
    {
        return *layers[index];
    }
};
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <utility>
#include <windows.h>

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

    void update(float learning_rate, float max_gradient)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::update: Neural network has no layers to update parameters", LOG_ERROR);
            throw std::logic_error("Neural network has no layers to update parameters");
        }

        for (const auto &layer : layers)
        {
            layer->update(learning_rate, max_gradient);
        }
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

    float trainStep(const Matrix &input_matrix, const Matrix &target_matrix, const ICostFunction &cost_fn, float learning_rate, float max_gradient = 1.0f, bool return_loss = false)
    {
        Matrix pred = forward(input_matrix);
        backward(target_matrix, cost_fn);
        update(learning_rate, max_gradient);

        reset();
        if (target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }
        float loss = 0.0f;
        if (return_loss)
            loss = cost_fn.computeLoss(pred, target_matrix);

        return loss;
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

        std::uint32_t parameter_layer_count = 0;
        for (const auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                ++parameter_layer_count;
            }
        }
        out_file.write(reinterpret_cast<const char *>(&parameter_layer_count), sizeof(parameter_layer_count));

        for (const auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                saveMatrix(out_file, layer->getWeights());
                saveMatrix(out_file, layer->getBiases());
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
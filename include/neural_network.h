#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cost_function/icost_function.h"
#include "engine/execution_engine.h"
#include "helper/layer.h"
#include "helper/logger.h"
#include "math/matrix.h"
#include "training_context.h"

class Neural_Network
{
private:
    std::vector<std::unique_ptr<ILayer>> layers;
    Matrix last_prediction;
    Execution_Target target = Execution_Target::CPU;
    Training_Context context;

    bool is_target_synced = false;
    bool is_graph_built = false;

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
          target(exec_target) {}

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
            Logger::logMessage("Neural_Network::forward: Network has no layers", LOG_ERROR);
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

    Matrix backward(const Matrix &target_matrix, const ICost_Function &cost_fn)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::backward: Network has no layers", LOG_ERROR);
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
            Logger::logMessage("Neural_Network::reset: Network has no layers", LOG_ERROR);
            throw std::logic_error("Neural network has no layers to reset gradients");
        }

        for (const auto &layer : layers)
        {
            layer->resetGradient();
        }
        last_prediction = Matrix(0, 0, target);
    }

    void trainStep(const Matrix &input_matrix, const Matrix &target_matrix, const ICost_Function &cost_fn, float learning_rate, IOptimizer &optimizer)
    {
        if (!is_target_synced)
        {
            for (std::unique_ptr<ILayer> &layer : layers)
                layer->setTarget(target);
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

    float evaluate(const Matrix &input_matrix, const Matrix &target_matrix, const ICost_Function &cost_fn)
    {
        Matrix pred = forward(input_matrix);
        if (target == Execution_Target::VULKAN_GPU)
            Execution_Engine::getInstance().executeGraph();
        return cost_fn.computeLoss(pred, target_matrix);
    }

    void saveInference(const std::string &file_path) const
    {
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            Logger::logMessage("Neural_Network::saveInference: Failed to open file: " + file_path, LOG_ERROR);
            throw std::runtime_error("Failed to open file for saving inference model");
        }

        const char magic[4] = {'N', 'N', 'I', '1'};
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
                layer->saveInference(out_file);
            }
        }
    }

    void loadInference(const std::string &file_path, Execution_Target exec_target = Execution_Target::CPU)
    {
        if (!context.loadInference(file_path, exec_target))
        {
            Logger::logMessage("Neural_Network::loadInference: Failed to load inference model context", LOG_ERROR);
            throw std::runtime_error("Failed to load inference model context");
        }

        layers = context.extractLayers();
    }

    void saveTrainingCheckpoint(const std::string &file_path,
                                std::size_t current_epoch,
                                const ICost_Function &cost_fn,
                                const ILearning_Rate &lr_scheduler,
                                const IOptimizer &optimizer) const
    {
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            Logger::logMessage("Neural_Network::saveTrainingCheckpoint: Failed to open file: " + file_path, LOG_ERROR);
            throw std::runtime_error("Failed to open file for saving training checkpoint");
        }

        const char magic[4] = {'N', 'N', 'C', 'K'};
        out_file.write(magic, 4);

        std::uint32_t epoch_val = static_cast<std::uint32_t>(current_epoch);
        out_file.write(reinterpret_cast<const char *>(&epoch_val), sizeof(epoch_val));

        Cost_Type cost_type = cost_fn.getType();
        out_file.write(reinterpret_cast<const char *>(&cost_type), sizeof(cost_type));
        cost_fn.saveCheckpoint(out_file);

        Decay_Mode decay_type = lr_scheduler.getType();
        out_file.write(reinterpret_cast<const char *>(&decay_type), sizeof(decay_type));
        lr_scheduler.saveCheckpoint(out_file);
        std::cout << lr_scheduler.getCurrentRate();

        Optimizer_Type opt_type = optimizer.getType();
        out_file.write(reinterpret_cast<const char *>(&opt_type), sizeof(opt_type));
        optimizer.saveCheckpoint(out_file);

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
                layer->saveCheckpoint(out_file);
            }
        }
    }

    void loadTrainingCheckpoint(const std::string &file_path, std::size_t total_epochs, Execution_Target exec_target = Execution_Target::CPU)
    {
        if (!context.loadCheckpoint(file_path, exec_target))
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: Failed to load checkpoint context", LOG_ERROR);
            throw std::runtime_error("Failed to load checkpoint context");
        }

        layers = context.extractLayers();
        if (total_epochs < context.getCurrentEpoch())
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: The total epoch is currently smaller than epochs that the network trained", LOG_WARNING, true);
        }
        context.getLearningRate().setMaxEpoch(static_cast<int>(total_epochs));
    }

    const Matrix &getLastPrediction() const
    {
        return last_prediction;
    }

    const ILayer &getLayer(std::size_t index) const
    {
        return *layers[index];
    }

    Training_Context &getContext() { return context; }
    const Training_Context &getContext() const { return context; }

    std::size_t getCurrentEpoch() const { return context.getCurrentEpoch(); }

    IOptimizer &getOptimizer() { return context.getOptimizer(); }
    const IOptimizer &getOptimizer() const { return context.getOptimizer(); }

    ICost_Function &getCostFunction() { return context.getCostFunction(); }
    const ICost_Function &getCostFunction() const { return context.getCostFunction(); }

    ILearning_Rate &getLearningRate() { return context.getLearningRate(); }
    const ILearning_Rate &getLearningRate() const { return context.getLearningRate(); }
};
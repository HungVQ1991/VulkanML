#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cost_function/icost_function.h"
#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "helper/layer.h"
#include "helper/logger.h"
#include "math/matrix.h"
#include "training_context.h"
#include "engine/graph_optimizer.h"

#ifndef ENABLE_NEURAL_NETWORK_DEBUG_LOGS
#define ENABLE_NEURAL_NETWORK_DEBUG_LOGS 0
#endif

#if ENABLE_NEURAL_NETWORK_DEBUG_LOGS
#define NEURAL_NETWORK_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define NEURAL_NETWORK_LOG_DEBUG(msg) ((void)0)
#endif

class Neural_Network;

void debugCheckGraphNodes(const std::string &checkpoint_name)
{
    std::size_t count = Execution_Engine::getInstance().getCurrentGraph().getNodes().size();
    Logger::logMessage(std::format("[CHECKPOINT {}] Node count = {}", checkpoint_name, count), LOG_DEBUG);
}

class Neural_Network
{
private:
    std::vector<std::unique_ptr<ILayer>> layers;
    Matrix last_prediction;
    Execution_Target target = Execution_Target::CPU;
    Training_Context context;

    bool is_target_synced = false;

public:
    explicit Neural_Network(Execution_Target exec_target = Execution_Target::CPU)
        : last_prediction(0, 0, exec_target),
          target(exec_target) {}

    ~Neural_Network()
    {
        if (target == Execution_Target::VULKAN_GPU)
            Execution_Engine::getInstance().waitIdle();
    }

    void addLayer(std::unique_ptr<ILayer> layer)
    {
        if (!layer)
        {
            Logger::logMessage("Neural_Network::addLayer: Attempted to add a null layer pointer", LOG_ERROR, true);
            throw std::invalid_argument("Cannot add null layer pointer");
        }
        layer->setTarget(target);
        NEURAL_NETWORK_LOG_DEBUG("Neural_Network::addLayer: Added layer type " + static_cast<std::string>(magic_enum::enum_name<Layer_Type>(layer->getLayerType())));
        layers.push_back(std::move(layer));
    }

    void setTarget(Execution_Target new_target)
    {
        if (target != new_target)
        {
            Logger::logMessage("Neural_Network::setTarget: Changing network execution target from " + std::to_string(static_cast<int>(target)) + " to " + std::to_string(static_cast<int>(new_target)), LOG_WARNING);
        }
        target = new_target;
        last_prediction.setExecutionTarget(new_target);
        for (auto &layer : layers)
        {
            layer->setTarget(new_target);
        }
        is_target_synced = true;
    }

    void setTrainingMode(bool is_training_mode)
    {
        NEURAL_NETWORK_LOG_DEBUG("Neural_Network::setTrainingMode: Setting training mode to " + std::string(is_training_mode ? "true" : "false"));
        for (auto &layer : layers)
        {
            layer->setTrainingMode(is_training_mode);
        }
    }

    Matrix forward(const Matrix &input_matrix)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::forward: Network has no layers", LOG_ERROR, true);
            throw std::logic_error("Neural network has no layers to execxute forward pass");
        }

        Matrix current_output = input_matrix;
        for (const auto &layer : layers)
        {
            current_output = layer->forward(current_output);
        }
        last_prediction = current_output;
        return current_output;
    }

    Matrix backward(const Matrix &target_matrix)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::backward: Network has no layers", LOG_ERROR, true);
            throw std::runtime_error("Network has no layers");
        }
        const ICost_Function &cost_fn = context.getCostFunction();

        Matrix last_prediction_output = layers.back()->getOutput();

        Matrix gradient_matrix = cost_fn.computeGradient(last_prediction_output, target_matrix);
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
            Logger::logMessage("Neural_Network::reset: Network has no layers", LOG_ERROR, true);
            throw std::logic_error("Neural network has no layers to reset gradients");
        }

        for (const auto &layer : layers)
        {
            layer->resetGradient();
        }
        last_prediction = Matrix(0, 0, target);
    }

    void compileAndWarmup(std::size_t batch_size, std::size_t input_dim, std::size_t output_dim)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::compileAndWarmup: Network has no layers", LOG_WARNING);
            return;
        }

        Matrix dummy_input(batch_size, input_dim, target);
        Matrix dummy_target(batch_size, output_dim, target);

        forward(dummy_input);
        backward(dummy_target);

        IOptimizer &optimizer = context.getOptimizer();
        optimizer.step(getParamsAndGrads());

        reset();

        if (target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine &engine = Execution_Engine::getInstance();
            engine.compileGraph(engine.getCurrentGraph(),
                                dummy_input.getGVector(),
                                dummy_target.getGVector());
            engine.getCurrentGraph().clear();
            engine.waitIdle();
            engine.enableGraphCaching(true);
        }
    }

    void printL2Norms()
    {
        auto param_grad_pairs = getParamsAndGrads();
        std::size_t index = 0;

        auto compute_l2_norm = [](const Matrix &matrix) -> float
        {
            const auto &data = matrix.getData();
            float sum_sq = 0.0f;
            for (float val : data)
            {
                sum_sq += val * val;
            }
            return std::sqrt(sum_sq);
        };

        std::string result = "\n--- Gradient & Parameter L2 Norm Inspection ---\n";
        for (const auto &[param, grad] : param_grad_pairs)
        {
            if (!param || !grad)
            {
                continue;
            }

            float param_norm = compute_l2_norm(*param);
            float grad_norm = compute_l2_norm(*grad);
            float ratio = (param_norm > 1e-8f) ? (grad_norm / param_norm) : 0.0f;

            result += std::format("Param #{:<2} | Shape: {:>4}x{:<4} | ||W||: {:>10.4e} | ||dW||: {:>10.4e} | Ratio: {:>10.4e}\n",
                                     index++, param->getRows(), param->getCols(), param_norm, grad_norm, ratio);
        }
        result += "-----------------------------------------------\n\n";
        Logger::logMessage(result, LOG_INFO, true);
    }

    void trainStep(Matrix &input_matrix, Matrix &target_matrix)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();

        if (input_matrix.getTarget() != target)
        {
            input_matrix.setExecutionTarget(target);
        }

        if (target_matrix.getTarget() != target)
        {
            target_matrix.setExecutionTarget(target);
        }

        forward(input_matrix);
        backward(target_matrix);
        printL2Norms();
        IOptimizer &optimizer = context.getOptimizer();
        optimizer.step(getParamsAndGrads());

        if (target == Execution_Target::VULKAN_GPU)
        {
            engine.executeGraph();
        }
    }

    void fit(Async_Data_Pipeline &data_pipeline, std::size_t total_epochs, std::size_t steps_per_epoch)
    {
        setTrainingMode(true);

        Execution_Engine &engine = Execution_Engine::getInstance();
        VkQueue compute_queue = engine.getContext().getComputeQueue();

        data_pipeline.setDevice(engine.getContext().getDevice());
        data_pipeline.start();

        for (std::size_t epoch = context.getCurrentEpoch(); epoch < total_epochs; ++epoch)
        {
            context.setCurrentEpoch(epoch);

            for (std::size_t step = 0; step < steps_per_epoch; ++step)
            {
                Batch_Data batch = data_pipeline.nextBatch(data_pipeline.getBatchSize(), 784, 10);

                if (batch.inputs && batch.targets)
                {
                    if (batch.inputs->getTarget() != target)
                    {
                        batch.inputs->setExecutionTarget(target);
                        batch.targets->setExecutionTarget(target);
                    }

                    trainStep(*batch.inputs, *batch.targets);
                    vkQueueWaitIdle(compute_queue);
                }
            }

            context.getLearningRate().step();
        }

        context.setCurrentEpoch(total_epochs);
        data_pipeline.stop();
    }

    void saveInference(const std::string &file_path) const
    {
        Execution_Engine::getInstance().waitIdle();
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            Logger::logMessage("Neural_Network::saveInference: Failed to open file: " + file_path, LOG_ERROR, true);
            throw std::runtime_error("Failed to open file for saving inference model");
        }

        NEURAL_NETWORK_LOG_DEBUG("Neural_Network::saveInference: Saving inference model to " + file_path);

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
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            Logger::logMessage("Neural_Network::loadInference: Failed to open file: " + file_path, LOG_ERROR, true);
            throw std::runtime_error("Failed to open file for loading inference model");
        }

        char magic[4];
        in_file.read(magic, 4);
        if (magic[0] != 'N' || magic[1] != 'N' || magic[2] != 'I' || magic[3] != '1')
        {
            Logger::logMessage("Neural_Network::loadInference: Invalid magic header", LOG_ERROR, true);
            throw std::runtime_error("Invalid magic header for inference model");
        }

        std::uint32_t total_layer_count = 0;
        in_file.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

        NEURAL_NETWORK_LOG_DEBUG("Neural_Network::loadInference: Loading inference model from " + file_path + ", total_layers=" + std::to_string(total_layer_count));

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type type_val;
            in_file.read(reinterpret_cast<char *>(&type_val), sizeof(type_val));
            Logger::logMessage("Neural_Network::loadInference: Layer " + std::to_string(i) + " type = " + static_cast<std::string>(magic_enum::enum_name(type_val)), LOG_INFO, true);
            layers.push_back(Training_Context::constructLayerFromConfig(in_file, type_val, exec_target));
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadInference(in_file);
            }
        }

        setTarget(exec_target);
    }

    void saveTrainingCheckpoint(const std::string &file_path, std::size_t current_epoch) const
    {
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            Logger::logMessage("Neural_Network::saveTrainingCheckpoint: Failed to open file: " + file_path, LOG_ERROR, true);
            throw std::runtime_error("Failed to open file for saving training checkpoint");
        }

        NEURAL_NETWORK_LOG_DEBUG("Neural_Network::saveTrainingCheckpoint: Saving checkpoint to " + file_path + ", epoch=" + std::to_string(current_epoch));

        const char magic[4] = {'N', 'N', 'C', 'K'};
        out_file.write(magic, 4);

        std::uint32_t epoch_val = static_cast<std::uint32_t>(current_epoch);
        out_file.write(reinterpret_cast<const char *>(&epoch_val), sizeof(epoch_val));

        const ICost_Function &cost_fn = context.getCostFunction();
        Cost_Type cost_type = cost_fn.getType();
        out_file.write(reinterpret_cast<const char *>(&cost_type), sizeof(cost_type));
        cost_fn.saveCheckpoint(out_file);

        const ILearning_Rate &lr_scheduler = context.getLearningRate();
        Decay_Mode decay_type = lr_scheduler.getType();
        out_file.write(reinterpret_cast<const char *>(&decay_type), sizeof(decay_type));
        lr_scheduler.saveCheckpoint(out_file);

        const IOptimizer &optimizer = context.getOptimizer();
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
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: Failed to open file: " + file_path, LOG_ERROR, true);
            throw std::runtime_error("Failed to open checkpoint file");
        }

        if (!context.loadHeader(in_file, exec_target))
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: Failed to load context header", LOG_ERROR, true);
            throw std::runtime_error("Failed to load context header");
        }

        std::uint32_t total_layer_count = 0;
        in_file.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

        NEURAL_NETWORK_LOG_DEBUG("Neural_Network::loadTrainingCheckpoint: Loading checkpoint from " + file_path + ", total_layers=" + std::to_string(total_layer_count));

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type type_val;
            in_file.read(reinterpret_cast<char *>(&type_val), sizeof(type_val));
            layers.push_back(Training_Context::constructLayerFromConfig(in_file, type_val, exec_target));
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadCheckpoint(in_file);
            }
        }

        if (total_epochs < context.getCurrentEpoch())
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: The total epoch is currently smaller than epochs that the network trained", LOG_WARNING, true);
        }
        context.getLearningRate().setMaxEpoch(static_cast<int>(total_epochs));

        setTarget(exec_target);
    }

    const Matrix &getLastPrediction() const { return last_prediction; }

    const ILayer &getLayer(std::size_t index) const { return *layers[index]; }

    Training_Context &getContext() { return context; }
    const Training_Context &getContext() const { return context; }

    std::size_t getCurrentEpoch() const { return context.getCurrentEpoch(); }

    IOptimizer &getOptimizer() { return context.getOptimizer(); }

    ICost_Function &getCostFunction() { return context.getCostFunction(); }

    ILearning_Rate &getLearningRate() { return context.getLearningRate(); }

    void setCostFunction(std::unique_ptr<ICost_Function> cost_fn)
    {
        if (!cost_fn)
        {
            Logger::logMessage("Neural_Network::setCostFunction: Attempted to set null cost function", LOG_ERROR, true);
            throw std::invalid_argument("Cannot set null cost function");
        }
        context.setCostFunction(std::move(cost_fn));
    }

    void setLearningRate(std::unique_ptr<ILearning_Rate> learning_rate)
    {
        if (!learning_rate)
        {
            Logger::logMessage("Neural_Network::setLearningRate: Attempted to set null learning rate scheduler", LOG_ERROR, true);
            throw std::invalid_argument("Cannot set null learning rate scheduler");
        }
        context.setLearningRate(std::move(learning_rate));
    }

    void setOptimizer(std::unique_ptr<IOptimizer> optimizer)
    {
        if (!optimizer)
        {
            Logger::logMessage("Neural_Network::setOptimizer: Attempted to set null optimizer", LOG_ERROR, true);
            throw std::invalid_argument("Cannot set null optimizer");
        }
        context.setOptimizer(std::move(optimizer));
    }
};
#pragma once

#include <chrono>
#include <cmath>
#include <concepts>
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
#include "engine/graph_optimizer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/matrix.h"
#include "training_context.h"

class Neural_Network
{
private:
    std::vector<std::unique_ptr<ILayer>> layers;
    Matrix last_prediction;
    Execution_Target execution_target = Execution_Target::CPU;
    Training_Context training_context;
    bool is_target_synchronized = false;

public:
    explicit Neural_Network(Execution_Target _execution_target = Execution_Target::CPU)
        : last_prediction(0, 0, _execution_target),
          execution_target(_execution_target)
    {
    }

    ~Neural_Network()
    {
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().waitIdle();
        }
    }

    void addLayer(std::unique_ptr<ILayer> _layer)
    {
        if (!_layer)
        {
            Logger::logMessage("Neural_Network::addLayer: Attempted to add a null layer pointer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LAYER_INSPECTION);
            throw std::invalid_argument("Cannot add null layer pointer");
        }
        _layer->setExecutionTarget(execution_target);
        Logger::logMessage(std::format("Neural_Network::addLayer: Added layer type {}",
                                       magic_enum::enum_name<Layer_Type>(_layer->getLayerType())),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LAYER_INSPECTION);
        layers.push_back(std::move(_layer));
    }

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        addLayer(std::move(new_layer));
        return layer_reference;
    }

    void setExecutionTarget(Execution_Target _new_execution_target)
    {
        if (execution_target != _new_execution_target)
        {
            Logger::logMessage(std::format("Neural_Network::setExecutionTarget: Changing network execution target from {} to {}",
                                           magic_enum::enum_name(execution_target),
                                           magic_enum::enum_name(_new_execution_target)),
                               Log_Level::LOG_WARNING,
                               true,
                               1,
                               Log_Feature::DEVICE_MANAGEMENT);
        }
        execution_target = _new_execution_target;
        last_prediction.setExecutionTarget(_new_execution_target);
        for (auto &layer : layers)
        {
            layer->setExecutionTarget(_new_execution_target);
        }
        is_target_synchronized = true;
    }

    void setTarget(Execution_Target _new_execution_target)
    {
        setExecutionTarget(_new_execution_target);
    }

    void setTrainingMode(bool _is_training_mode)
    {
        Logger::logMessage(std::format("Neural_Network::setTrainingMode: Setting training mode to {}",
                                       _is_training_mode ? "true" : "false"),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::TRAINING);
        for (auto &layer : layers)
        {
            layer->setTrainingMode(_is_training_mode);
        }
    }

    Matrix forward(const Matrix &_input_matrix)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::forward: Network has no layers",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::FORWARD_EVALUATION);
            throw std::logic_error("Neural network has no layers to execute forward pass");
        }

        Matrix current_output = _input_matrix;
        for (const auto &layer : layers)
        {
            current_output = layer->forward(current_output);
        }
        last_prediction = current_output;
        return current_output;
    }

    Matrix backward(const Matrix &_target_matrix)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::backward: Network has no layers",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::BACKWARD_PROPAGATION);
            throw std::runtime_error("Network has no layers");
        }
        const ICost_Function &cost_function = training_context.getCostFunction();

        Matrix last_prediction_output = layers.back()->getOutput();
        Matrix gradient_matrix = cost_function.computeGradient(last_prediction_output, _target_matrix);
        for (std::size_t i = layers.size(); i > 0; --i)
        {
            gradient_matrix = layers[i - 1]->backward(gradient_matrix);
        }

        return gradient_matrix;
    }

     std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients()
    {
        std::vector<std::pair<Matrix *, Matrix *>> parameter_gradient_pairs;
        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                auto pairs = layer->getParametersAndGradients();
                parameter_gradient_pairs.insert(parameter_gradient_pairs.end(), pairs.begin(), pairs.end());
            }
        }
        return parameter_gradient_pairs;
    }

     std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads()
    {
        return getParametersAndGradients();
    }

    void reset()
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::reset: Network has no layers",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TRAINING);
            throw std::logic_error("Neural network has no layers to reset gradients");
        }

        for (const auto &layer : layers)
        {
            layer->resetGradient();
        }
    }

    void resetGradients()
    {
        reset();
    }

    void compileAndWarmup(std::size_t _batch_size, std::size_t _input_dimension, std::size_t _output_dimension)
    {
        if (layers.empty())
        {
            Logger::logMessage("Neural_Network::compileAndWarmup: Network has no layers",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::TRAINING);
            return;
        }

        Matrix dummy_input(_batch_size, _input_dimension, execution_target);
        Matrix dummy_target(_batch_size, _output_dimension, execution_target);

        forward(dummy_input);
        backward(dummy_target);

        IOptimizer &optimizer = training_context.getOptimizer();
        optimizer.step(getParametersAndGradients());

        reset();
        optimizer.reset();

        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine &engine = Execution_Engine::getInstance();
            engine.enableGraphCaching(true);
            engine.warmCache(engine.getCurrentGraph());
            Gpu_Matrix_Impl::distinct_operations_count = engine.getCurrentGraph().getNodeCount();
            engine.getCurrentGraph().clear();
            engine.waitIdle();
        }
    }

    void printL2Norms()
    {
        auto parameter_gradient_pairs = getParametersAndGradients();
        std::size_t parameter_index = 0;

        auto compute_l2_norm = [](const Matrix &_matrix) -> float
        {
            const auto &data = _matrix.getData();
            float sum_of_squares = 0.0f;
            for (float value : data)
            {
                sum_of_squares += value * value;
            }
            return std::sqrt(sum_of_squares);
        };

        std::string inspection_result = "\n--- Gradient & Parameter L2 Norm Inspection ---\n";
        for (const auto &[parameter, gradient] : parameter_gradient_pairs)
        {
            if (!parameter || !gradient)
            {
                continue;
            }

            float parameter_norm = compute_l2_norm(*parameter);
            float gradient_norm = compute_l2_norm(*gradient);
            float norm_ratio = (parameter_norm > 1e-8f) ? (gradient_norm / parameter_norm) : 0.0f;

            inspection_result += std::format("Param #{:<2} | Shape: {:>4}x{:<4} | ||W||: {:>10.4e} | ||dW||: {:>10.4e} | Ratio: {:>10.4e}\n",
                                             parameter_index++,
                                             parameter->getRows(),
                                             parameter->getColumns(),
                                             parameter_norm,
                                             gradient_norm,
                                             norm_ratio);
        }
        inspection_result += "-----------------------------------------------\n\n";
        Logger::logMessage(inspection_result, Log_Level::LOG_DEBUG, true, 0, Log_Feature::LAYER_INSPECTION);
    }

    void trainStep(Matrix &_input_matrix, Matrix &_target_matrix, VkFence _fence = VK_NULL_HANDLE)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();

        if (_input_matrix.getExecutionTarget() != execution_target)
        {
            _input_matrix.setExecutionTarget(execution_target);
        }

        if (_target_matrix.getExecutionTarget() != execution_target)
        {
            _target_matrix.setExecutionTarget(execution_target);
        }

        forward(_input_matrix);
        backward(_target_matrix);

        // printL2Norms();

        IOptimizer &optimizer = training_context.getOptimizer();
        optimizer.step(getParametersAndGradients());

        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            engine.executeGraph(_fence);
        }
    }

    void fit(Async_Data_Pipeline &_data_pipeline,
             std::size_t _total_epochs,
             std::size_t _steps_per_epoch,
             std::size_t _batch_size,
             std::size_t _input_dimension,
             std::size_t _output_dimension)
    {
        setTrainingMode(true);

        Execution_Engine &engine = Execution_Engine::getInstance();

        _data_pipeline.setDevice(engine.getContext().getDevice());
        _data_pipeline.start();

        for (std::size_t epoch = training_context.getCurrentEpoch(); epoch < _total_epochs; ++epoch)
        {
            training_context.setCurrentEpoch(epoch);

            for (std::size_t step_index = 0; step_index < _steps_per_epoch; ++step_index)
            {
                Batch_Data batch_data = _data_pipeline.nextBatch(_batch_size, _input_dimension, _output_dimension);

                if (batch_data.input_matrix && batch_data.target_matrix)
                {
                    if (batch_data.input_matrix->getExecutionTarget() != execution_target)
                    {
                        batch_data.input_matrix->setExecutionTarget(execution_target);
                        batch_data.target_matrix->setExecutionTarget(execution_target);
                    }

                    trainStep(*batch_data.input_matrix, *batch_data.target_matrix, batch_data.fence);
                }
            }

            training_context.getLearningRate().step();
        }

        engine.waitIdle();
        training_context.setCurrentEpoch(_total_epochs);
        _data_pipeline.stop();
    }

    void saveInference(const std::string &_file_path) const
    {
        Execution_Engine::getInstance().waitIdle();
        std::ofstream output_file_stream(_file_path, std::ios::binary);
        if (!output_file_stream.is_open())
        {
            Logger::logMessage(std::format("Neural_Network::saveInference: Failed to open file: {}", _file_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Failed to open file for saving inference model");
        }

        Logger::logMessage(std::format("Neural_Network::saveInference: Saving inference model to {}", _file_path),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        const char magic_header[4] = {'N', 'N', 'I', '1'};
        output_file_stream.write(magic_header, 4);

        std::uint32_t total_layer_count = static_cast<std::uint32_t>(layers.size());
        output_file_stream.write(reinterpret_cast<const char *>(&total_layer_count), sizeof(total_layer_count));

        for (const auto &layer : layers)
        {
            Layer_Type layer_type = layer->getLayerType();
            output_file_stream.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));
            layer->saveConfiguration(output_file_stream);
        }

        for (const auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->saveInference(output_file_stream);
            }
        }
        Logger::logMessage("Inference saved to " + _file_path, Log_Level::LOG_INFO, 1, true);
    }

    void loadInference(const std::string &_file_path, Execution_Target _execution_target = Execution_Target::CPU)
    {
        std::ifstream input_file_stream(_file_path, std::ios::binary);
        if (!input_file_stream.is_open())
        {
            Logger::logMessage(std::format("Neural_Network::loadInference: Failed to open file: {}", _file_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Failed to open file for loading inference model");
        }

        char magic_header[4];
        input_file_stream.read(magic_header, 4);
        if (magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'I' || magic_header[3] != '1')
        {
            Logger::logMessage("Neural_Network::loadInference: Invalid magic header",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Invalid magic header for inference model");
        }

        std::uint32_t total_layer_count = 0;
        input_file_stream.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

        Logger::logMessage(std::format("Neural_Network::loadInference: Loading inference model from {}, total_layers={}",
                                       _file_path,
                                       total_layer_count),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type layer_type;
            input_file_stream.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type));
            Logger::logMessage(std::format("Neural_Network::loadInference: Layer {} type = {}",
                                           i,
                                           magic_enum::enum_name(layer_type)),
                               Log_Level::LOG_INFO,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            layers.push_back(Training_Context::constructLayerFromConfig(input_file_stream, layer_type, _execution_target));
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadInference(input_file_stream);
            }
        }

        setExecutionTarget(_execution_target);
        Logger::logMessage("Inference loaded from {}" + _file_path, Log_Level::LOG_INFO, true, 1);
    }

    void saveTrainingCheckpoint(const std::string &_file_path, std::size_t _current_epoch) const
    {
        std::ofstream output_file_stream(_file_path, std::ios::binary);
        if (!output_file_stream.is_open())
        {
            Logger::logMessage(std::format("Neural_Network::saveTrainingCheckpoint: Failed to open file: {}", _file_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Failed to open file for saving training checkpoint");
        }

        Logger::logMessage(std::format("Neural_Network::saveTrainingCheckpoint: Saving checkpoint to {}, epoch={}",
                                       _file_path,
                                       _current_epoch),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        const char magic_header[4] = {'N', 'N', 'C', 'K'};
        output_file_stream.write(magic_header, 4);

        std::uint32_t epoch_value = static_cast<std::uint32_t>(_current_epoch);
        output_file_stream.write(reinterpret_cast<const char *>(&epoch_value), sizeof(epoch_value));

        const ICost_Function &cost_function = training_context.getCostFunction();
        Cost_Type cost_type = cost_function.getType();
        output_file_stream.write(reinterpret_cast<const char *>(&cost_type), sizeof(cost_type));
        cost_function.saveCheckpoint(output_file_stream);

        const ILearning_Rate &learning_rate_scheduler = training_context.getLearningRate();
        Decay_Mode decay_mode = learning_rate_scheduler.getType();
        output_file_stream.write(reinterpret_cast<const char *>(&decay_mode), sizeof(decay_mode));
        learning_rate_scheduler.saveCheckpoint(output_file_stream);

        const IOptimizer &optimizer = training_context.getOptimizer();
        Optimizer_Type optimizer_type = optimizer.getType();
        output_file_stream.write(reinterpret_cast<const char *>(&optimizer_type), sizeof(optimizer_type));
        optimizer.saveCheckpoint(output_file_stream);

        std::uint32_t total_layer_count = static_cast<std::uint32_t>(layers.size());
        output_file_stream.write(reinterpret_cast<const char *>(&total_layer_count), sizeof(total_layer_count));

        for (const auto &layer : layers)
        {
            Layer_Type layer_type = layer->getLayerType();
            output_file_stream.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));
            layer->saveConfiguration(output_file_stream);
        }

        for (const auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->saveCheckpoint(output_file_stream);
            }
        }
    }

    void loadTrainingCheckpoint(const std::string &_file_path, std::size_t _total_epochs, Execution_Target _execution_target = Execution_Target::CPU)
    {
        std::ifstream input_file_stream(_file_path, std::ios::binary);
        if (!input_file_stream.is_open())
        {
            Logger::logMessage(std::format("Neural_Network::loadTrainingCheckpoint: Failed to open file: {}", _file_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Failed to open checkpoint file");
        }

        if (!training_context.loadHeader(input_file_stream, _execution_target))
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: Failed to load context header",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Failed to load context header");
        }

        std::uint32_t total_layer_count = 0;
        input_file_stream.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

        Logger::logMessage(std::format("Neural_Network::loadTrainingCheckpoint: Loading checkpoint from {}, total_layers={}",
                                       _file_path,
                                       total_layer_count),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type layer_type;
            input_file_stream.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type));
            layers.push_back(Training_Context::constructLayerFromConfig(input_file_stream, layer_type, _execution_target));
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadCheckpoint(input_file_stream);
            }
        }

        if (_total_epochs < training_context.getCurrentEpoch())
        {
            Logger::logMessage("Neural_Network::loadTrainingCheckpoint: The total epoch is currently smaller than epochs that the network trained",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
        }
        training_context.getLearningRate().setMaxEpoch(static_cast<int>(_total_epochs));

        setExecutionTarget(_execution_target);
    }

     const Matrix &getLastPrediction() const noexcept
    {
        return last_prediction;
    }

     const ILayer &getLayer(std::size_t _index) const
    {
        return *layers.at(_index);
    }

     ILayer &getLayer(std::size_t _index)
    {
        return *layers.at(_index);
    }

     std::size_t getLayerCount() const noexcept
    {
        return layers.size();
    }

     Training_Context &getContext() noexcept
    {
        return training_context;
    }

     const Training_Context &getContext() const noexcept
    {
        return training_context;
    }

     Training_Context &getTrainingContext() noexcept
    {
        return training_context;
    }

     const Training_Context &getTrainingContext() const noexcept
    {
        return training_context;
    }

     std::size_t getCurrentEpoch() const noexcept
    {
        return training_context.getCurrentEpoch();
    }

     IOptimizer &getOptimizer()
    {
        return training_context.getOptimizer();
    }

     const IOptimizer &getOptimizer() const
    {
        return training_context.getOptimizer();
    }

     ICost_Function &getCostFunction()
    {
        return training_context.getCostFunction();
    }

     const ICost_Function &getCostFunction() const
    {
        return training_context.getCostFunction();
    }

     ILearning_Rate &getLearningRate()
    {
        return training_context.getLearningRate();
    }

     const ILearning_Rate &getLearningRate() const
    {
        return training_context.getLearningRate();
    }

     Execution_Target getExecutionTarget() const noexcept
    {
        return execution_target;
    }

    void setCostFunction(std::unique_ptr<ICost_Function> _cost_function)
    {
        if (!_cost_function)
        {
            Logger::logMessage("Neural_Network::setCostFunction: Attempted to set null cost function",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TRAINING);
            throw std::invalid_argument("Cannot set null cost function");
        }
        training_context.setCostFunction(std::move(_cost_function));
    }

    template <std::derived_from<ICost_Function> Cost_Type_T, typename... Args>
    Cost_Type_T &setCostFunction(Args &&...args)
    {
        auto cost = std::make_unique<Cost_Type_T>(std::forward<Args>(args)...);
        Cost_Type_T &cost_reference = *cost;
        setCostFunction(std::move(cost));
        return cost_reference;
    }

    void setLearningRate(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler)
    {
        if (!_learning_rate_scheduler)
        {
            Logger::logMessage("Neural_Network::setLearningRate: Attempted to set null learning rate scheduler",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TRAINING);
            throw std::invalid_argument("Cannot set null learning rate scheduler");
        }
        training_context.setLearningRate(std::move(_learning_rate_scheduler));
    }

    template <std::derived_from<ILearning_Rate> Scheduler_Type_T, typename... Args>
    Scheduler_Type_T &setLearningRate(Args &&...args)
    {
        auto scheduler = std::make_unique<Scheduler_Type_T>(std::forward<Args>(args)...);
        Scheduler_Type_T &scheduler_reference = *scheduler;
        setLearningRate(std::move(scheduler));
        return scheduler_reference;
    }

    void setOptimizer(std::unique_ptr<IOptimizer> _optimizer)
    {
        if (!_optimizer)
        {
            Logger::logMessage("Neural_Network::setOptimizer: Attempted to set null optimizer",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TRAINING);
            throw std::invalid_argument("Cannot set null optimizer");
        }
        training_context.setOptimizer(std::move(_optimizer));
    }

    template <std::derived_from<IOptimizer> Optimizer_Type_T, typename... Args>
    Optimizer_Type_T &setOptimizer(Args &&...args)
    {
        auto opt = std::make_unique<Optimizer_Type_T>(std::forward<Args>(args)...);
        Optimizer_Type_T &opt_reference = *opt;
        setOptimizer(std::move(opt));
        return opt_reference;
    }
};
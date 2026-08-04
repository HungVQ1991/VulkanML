#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/cost_function.h"
#include "helper/layer.h"
#include "helper/learning_rate.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/optimizer.h"

class Training_Context
{
private:
    std::size_t current_epoch = 0;
    std::unique_ptr<ICost_Function> cost_fn;
    std::unique_ptr<ILearning_Rate> learning_rate;
    std::unique_ptr<IOptimizer> optimizer;
    std::vector<std::unique_ptr<ILayer>> layers;

public:
    Training_Context() = default;

    std::size_t getCurrentEpoch() const { return current_epoch; }
    void setCurrentEpoch(std::size_t epoch) { current_epoch = epoch; }

    ICost_Function &getCostFunction() { return *cost_fn; }
    const ICost_Function &getCostFunction() const { return *cost_fn; }

    ILearning_Rate &getLearningRate() { return *learning_rate; }
    const ILearning_Rate &getLearningRate() const { return *learning_rate; }

    IOptimizer &getOptimizer() { return *optimizer; }
    const IOptimizer &getOptimizer() const { return *optimizer; }

    std::vector<std::unique_ptr<ILayer>> &getLayers() { return layers; }
    const std::vector<std::unique_ptr<ILayer>> &getLayers() const { return layers; }

    std::vector<std::unique_ptr<ILayer>> extractLayers() { return std::move(layers); }

    static std::unique_ptr<ILayer> constructLayerFromConfig(std::ifstream &in_file, Layer_Type type_val, Execution_Target exec_target)
    {
        switch (type_val)
        {
        case Layer_Type::LINEAR:
        {
            std::uint32_t in_dim = 0;
            std::uint32_t out_dim = 0;
            in_file.read(reinterpret_cast<char *>(&in_dim), sizeof(in_dim));
            in_file.read(reinterpret_cast<char *>(&out_dim), sizeof(out_dim));
            return std::make_unique<Linear_Layer>(in_dim, out_dim, exec_target);
        }
        case Layer_Type::CONV2D:
        {
            std::uint32_t in_h = 0, in_w = 0, in_c = 0, out_c = 0;
            std::uint32_t kernel_size = 0, stride = 0, padding = 0;
            in_file.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
            in_file.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
            in_file.read(reinterpret_cast<char *>(&in_c), sizeof(in_c));
            in_file.read(reinterpret_cast<char *>(&out_c), sizeof(out_c));
            in_file.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
            in_file.read(reinterpret_cast<char *>(&stride), sizeof(stride));
            in_file.read(reinterpret_cast<char *>(&padding), sizeof(padding));
            return std::make_unique<Conv2d_Layer>(in_h, in_w, in_c, out_c, kernel_size, stride, padding, exec_target);
        }
        case Layer_Type::BATCH_NORM:
        {
            std::uint32_t num_features = 0;
            float epsilon = 0.0f;
            float momentum = 0.0f;
            in_file.read(reinterpret_cast<char *>(&num_features), sizeof(num_features));
            in_file.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
            in_file.read(reinterpret_cast<char *>(&momentum), sizeof(momentum));
            return std::make_unique<Batch_Norm_Layer>(num_features, epsilon, momentum, exec_target);
        }
        case Layer_Type::GLOBAL_AVG_POOL_2D:
        {
            std::uint32_t in_h = 0, in_w = 0, channels = 0;
            in_file.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
            in_file.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
            in_file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            return std::make_unique<GlobalAvgPool2d_Layer>(in_h, in_w, channels, exec_target);
        }
        case Layer_Type::MAX_POOL_2D:
        {
            std::uint32_t in_h = 0, in_w = 0, channels = 0;
            std::uint32_t kernel_size = 0, stride = 0, padding = 0;
            in_file.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
            in_file.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
            in_file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            in_file.read(reinterpret_cast<char *>(&kernel_size), sizeof(kernel_size));
            in_file.read(reinterpret_cast<char *>(&stride), sizeof(stride));
            in_file.read(reinterpret_cast<char *>(&padding), sizeof(padding));
            return std::make_unique<MaxPool2d_Layer>(in_h, in_w, channels, kernel_size, stride, padding, exec_target);
        }
        case Layer_Type::SOFTMAX:
        {
            std::uint8_t fused_val = 0;
            in_file.read(reinterpret_cast<char *>(&fused_val), sizeof(fused_val));
            bool fused = (fused_val != 0);
            return std::make_unique<Softmax>(fused, exec_target);
        }
        case Layer_Type::RELU:
        {
            return std::make_unique<ReLU>(exec_target);
        }
        case Layer_Type::GELU:
        {
            return std::make_unique<GeLU>(exec_target);
        }
        default:
            Logger::logMessage("Training_Context::constructLayerFromConfig: Unsupported layer type", LOG_ERROR, true);
            throw std::runtime_error("Unsupported or unknown layer type");
        }
    }

    void createCostFunction(Cost_Type type_val, std::ifstream &in_file)
    {
        switch (type_val)
        {
        case Cost_Type::MSE:
            cost_fn = std::make_unique<MSE_Cost>();
            break;
        case Cost_Type::MAE:
            cost_fn = std::make_unique<MAE_Cost>();
            break;
        case Cost_Type::BCE:
            cost_fn = std::make_unique<BCE_Cost>();
            break;
        case Cost_Type::CCE:
            cost_fn = std::make_unique<CCE_Cost>();
            break;
        default:
            Logger::logMessage("createCostFunction: Unknown cost type", LOG_ERROR, true);
            throw std::runtime_error("Unknown cost type");
        }
        cost_fn->loadCheckpoint(in_file);
    }

    void createLRScheduler(Decay_Mode type_val, std::ifstream &in_file)
    {
        switch (type_val)
        {
        case Decay_Mode::STEP_DECAY:
            learning_rate = std::make_unique<Step_Decay>();
            break;
        case Decay_Mode::EXPONENTIAL_DECAY:
            learning_rate = std::make_unique<Exponential_Decay>();
            break;
        case Decay_Mode::COSINE_ANNEALING:
            learning_rate = std::make_unique<Cosine_Annealing>(0.001f, 0.0f, 1);
            break;
        default:
            Logger::logMessage("createLRScheduler: Unknown decay mode", LOG_ERROR, true);
            throw std::runtime_error("Unknown decay mode");
        }
        learning_rate->loadCheckpoint(in_file);
        Logger::logMessage("Training_Context::createLRScheduler: Learning rate: " + std::to_string(learning_rate->getCurrentRate()), LOG_INFO, true);
    }

    void createOptimizer(Optimizer_Type type_val, std::ifstream &in_file, Execution_Target exec_target = Execution_Target::CPU)
    {
        if (!learning_rate)
        {
            Logger::logMessage("createOptimizer: Learning rate scheduler must be created before optimizer", LOG_ERROR, true);
            throw std::runtime_error("Learning rate scheduler must be created before optimizer");
        }

        switch (type_val)
        {
        case Optimizer_Type::SGD_OPTIMIZER:
            optimizer = std::make_unique<SGD_Optimizer>(*learning_rate);
            break;
        case Optimizer_Type::ADAM_OPTIMIZER:
            optimizer = std::make_unique<Adam_Optimizer>(*learning_rate, 0.9f, 0.999f, 1e-8f, 1.0f);
            break;
        default:
            Logger::logMessage("createOptimizer: Unknown optimizer type", LOG_ERROR, true);
            throw std::runtime_error("Unknown optimizer type");
        }

        optimizer->loadCheckpoint(in_file, exec_target);
    }

    bool loadCheckpoint(const std::string &file_path, Execution_Target exec_target = Execution_Target::CPU)
    {
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            Logger::logMessage("Training_Context::loadCheckpoint: Failed to open file: " + file_path, LOG_ERROR, true);
            return false;
        }

        char magic[4];
        in_file.read(magic, 4);
        if (magic[0] != 'N' || magic[1] != 'N' || magic[2] != 'C' || magic[3] != 'K')
        {
            Logger::logMessage("Training_Context::loadCheckpoint: Invalid magic header", LOG_ERROR, true);
            return false;
        }

        Logger::logMessage("Training_Context::loadCheckpoint: Reading checkpoint file " + file_path, LOG_INFO, true);

        std::uint32_t epoch_val = 0;
        in_file.read(reinterpret_cast<char *>(&epoch_val), sizeof(epoch_val));
        current_epoch = static_cast<std::size_t>(epoch_val);
        Logger::logMessage("Training_Context::loadCheckpoint: Current epoch = " + std::to_string(current_epoch), LOG_INFO, true);

        Cost_Type cost_type;
        in_file.read(reinterpret_cast<char *>(&cost_type), sizeof(cost_type));
        Logger::logMessage("Training_Context::loadCheckpoint: Cost function type = " + static_cast<std::string>(magic_enum::enum_name(cost_type)), LOG_INFO, true);
        createCostFunction(cost_type, in_file);

        Decay_Mode decay_type;
        in_file.read(reinterpret_cast<char *>(&decay_type), sizeof(decay_type));
        Logger::logMessage("Training_Context::loadCheckpoint: Decay mode = " + static_cast<std::string>(magic_enum::enum_name(decay_type)), LOG_INFO, true);
        createLRScheduler(decay_type, in_file);

        Optimizer_Type opt_type;
        in_file.read(reinterpret_cast<char *>(&opt_type), sizeof(opt_type));
        Logger::logMessage("Training_Context::loadCheckpoint: Optimizer type = " + static_cast<std::string>(magic_enum::enum_name(opt_type)), LOG_INFO, true);
        createOptimizer(opt_type, in_file, exec_target);

        std::uint32_t total_layer_count = 0;
        in_file.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));
        Logger::logMessage("Training_Context::loadCheckpoint: Total layer count = " + std::to_string(total_layer_count), LOG_INFO, true);

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type type_val;
            in_file.read(reinterpret_cast<char *>(&type_val), sizeof(type_val));
            Logger::logMessage("Training_Context::loadCheckpoint: Layer " + std::to_string(i) + " type = " + static_cast<std::string>(magic_enum::enum_name(type_val)), LOG_INFO, true);
            layers.push_back(constructLayerFromConfig(in_file, type_val, exec_target));
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadCheckpoint(in_file);
            }
        }

        Logger::logMessage("Training_Context::loadCheckpoint: Successfully loaded checkpoint", LOG_INFO, true);
        return true;
    }

    bool loadInference(const std::string &file_path, Execution_Target exec_target = Execution_Target::CPU)
    {
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            Logger::logMessage("Training_Context::loadInference: Failed to open file: " + file_path, LOG_ERROR, true);
            return false;
        }

        char magic[4];
        in_file.read(magic, 4);
        if (magic[0] != 'N' || magic[1] != 'N' || magic[2] != 'I' || magic[3] != '1')
        {
            Logger::logMessage("Training_Context::loadInference: Invalid magic header", LOG_ERROR, true);
            return false;
        }

        std::uint32_t total_layer_count = 0;
        in_file.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

        layers.clear();
        layers.reserve(total_layer_count);

        for (std::uint32_t i = 0; i < total_layer_count; ++i)
        {
            Layer_Type type_val;
            in_file.read(reinterpret_cast<char *>(&type_val), sizeof(type_val));
            layers.push_back(constructLayerFromConfig(in_file, type_val, exec_target));
        }

        for (auto &layer : layers)
        {
            if (layer->hasParameters())
            {
                layer->loadInference(in_file);
            }
        }

        return true;
    }
};
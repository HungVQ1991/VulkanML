#include "neural_network.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "engine/loss_scaler.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/gpu_tensor_impl.h"
#include "optimizer/adam_optimizer.h"
#include "optimizer/sgd_optimizer.h"
#include "training_context.h"

VkBuffer Neural_Network::extractBufferHandle(const Tensor &_tensor) noexcept
{
    auto storage_handle = _tensor.getStorage();
    if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
    {
        const auto &gpu_vec = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
        if (gpu_vec)
        {
            return gpu_vec->getBuffer();
        }
    }
    return VK_NULL_HANDLE;
}

void Neural_Network::invalidateStaticBufferBindings() noexcept
{
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        static_baked_input_buffers[i] = VK_NULL_HANDLE;
        static_baked_target_buffers[i] = VK_NULL_HANDLE;
    }
}

Neural_Network::Neural_Network(Execution_Target _execution_target)
    : last_prediction(0, 0, _execution_target),
      execution_target(_execution_target)
{
}

Neural_Network::~Neural_Network()
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().waitIdle();
    }
}

void Neural_Network::addLayer(std::unique_ptr<ILayer> _layer)
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
    _layer->setAccumulated(is_gradient_accumulation_enabled);
    _layer->setMixedPrecision(is_mixed_precision_enabled);
    Logger::logMessage(Input_Format{"Neural_Network::addLayer: Added layer type {}",
                                    magic_enum::enum_name<Layer_Type>(_layer->getLayerType())},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::LAYER_INSPECTION);
    layers.push_back(std::move(_layer));
    invalidateStaticBufferBindings();
}

void Neural_Network::zeroGradients()
{
    for (auto &layer : layers)
    {
        layer->resetGradients();
    }
}

Tensor Neural_Network::forward(const Tensor &_input_tensor)
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

    Tensor current_output = _input_tensor;
    for (const auto &layer : layers)
    {
        current_output = layer->forward(current_output);
    }
    last_prediction = current_output;
    return current_output;
}

Tensor Neural_Network::backward(const Tensor &_target_tensor)
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
    Tensor gradient_tensor;
    if (training_context.hasCostFunction())
    {
        const ICost_Function &cost_function = training_context.getCostFunction();
        Tensor last_prediction_output = layers.back()->getOutput();
        gradient_tensor = cost_function.computeGradient(last_prediction_output, _target_tensor);
    }
    else
    {
        gradient_tensor = _target_tensor;
    }

    if (is_mixed_precision_enabled && loss_scaler.getScaleFactor() != 1.0f)
    {
        loss_scaler.scaleGradient(gradient_tensor);
    }

    for (size_t i = layers.size(); i > 0; --i)
    {
        gradient_tensor = layers[i - 1]->backward(gradient_tensor);
    }

    return gradient_tensor;
}

std::vector<std::pair<Tensor *, Tensor *>> Neural_Network::getParametersAndGradients()
{
    std::vector<std::pair<Tensor *, Tensor *>> parameter_gradient_pairs;
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

std::vector<std::pair<Tensor *, Tensor *>> Neural_Network::getParamsAndGrads()
{
    return getParametersAndGradients();
}

void Neural_Network::reset()
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

void Neural_Network::compileAndWarmup(size_t _batch_size, size_t _input_dimension, size_t _output_dimension)
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

    Tensor dummy_input(_batch_size, _input_dimension, execution_target);
    Tensor dummy_target(_batch_size, _output_dimension, execution_target);

    IOptimizer &optimizer = training_context.getOptimizer();
    bool is_adam = (optimizer.getType() == Optimizer_Type::ADAM_OPTIMIZER);
    std::vector<Adam_Optimizer::Parameter_State> saved_loaded_states;
    size_t saved_timestep = 0;
    if (is_adam)
    {
        auto &adam = static_cast<Adam_Optimizer &>(optimizer);
        saved_loaded_states = adam.getLoadedStates();
        saved_timestep = adam.getTimestep();
    }

    auto params = getParametersAndGradients();
    std::vector<std::vector<float>> saved_parameters;
    saved_parameters.reserve(params.size());
    for (auto &[param, grad] : params)
    {
        if (param)
        {
            saved_parameters.push_back(param->getData());
        }
        else
        {
            saved_parameters.push_back({});
        }
    }

    forward(dummy_input);
    backward(dummy_target);

    optimizer.step(getParametersAndGradients());

    reset();
    optimizer.reset();

    for (size_t i = 0; i < params.size(); ++i)
    {
        if (params[i].first && !saved_parameters[i].empty())
        {
            params[i].first->uploadData(saved_parameters[i]);
        }
    }
    for (auto &layer : layers)
    {
        layer->invalidateWeightCache();
    }

    if (is_adam && (!saved_loaded_states.empty() || saved_timestep > 0))
    {
        auto &adam = static_cast<Adam_Optimizer &>(optimizer);
        adam.setLoadedStates(saved_loaded_states);
        adam.setTimestep(saved_timestep);
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();
        engine.enableGraphCaching(true);
        engine.warmCache(engine.getCurrentGraph());
        Gpu_Tensor_Impl::distinct_operations_count = engine.getCurrentGraph().getNodeCount();
        engine.getCurrentGraph().clear();
        engine.waitIdle();
    }
}

void Neural_Network::printL2Norms()
{
    auto parameter_gradient_pairs = getParametersAndGradients();
    size_t parameter_index = 0;

    auto compute_l2_norm = [](const Tensor &_tensor) -> float
    {
        const auto &data = _tensor.getData();
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

void Neural_Network::trainStep(Tensor &_input_tensor, Tensor &_target_tensor, VkFence _fence)
{
    Execution_Engine &engine = Execution_Engine::getInstance();

    if (_input_tensor.getExecutionTarget() != execution_target)
    {
        _input_tensor.setExecutionTarget(execution_target);
    }

    if (_target_tensor.getExecutionTarget() != execution_target)
    {
        _target_tensor.setExecutionTarget(execution_target);
    }

    if (_target_tensor.getDataType() != _input_tensor.getDataType())
    {
        _target_tensor.setDataType(_input_tensor.getDataType());
    }

    uint32_t current_frame = (execution_target == Execution_Target::VULKAN_GPU) ? engine.getContext().getCurrentFrame() : 0;
    VkBuffer current_in_buf = (execution_target == Execution_Target::VULKAN_GPU) ? extractBufferHandle(_input_tensor) : VK_NULL_HANDLE;
    VkBuffer current_tgt_buf = (execution_target == Execution_Target::VULKAN_GPU) ? extractBufferHandle(_target_tensor) : VK_NULL_HANDLE;

    bool can_fast_replay = is_training_mode &&
                           engine.isStaticGraphEnabled() &&
                           execution_target == Execution_Target::VULKAN_GPU &&
                           engine.getGraphExecutor().isStaticBaked(current_frame) &&
                           current_in_buf != VK_NULL_HANDLE &&
                           current_tgt_buf != VK_NULL_HANDLE &&
                           static_baked_input_buffers[current_frame] == current_in_buf &&
                           static_baked_target_buffers[current_frame] == current_tgt_buf;

    if (can_fast_replay)
    {
        IOptimizer &optimizer = training_context.getOptimizer();
        if (is_mixed_precision_enabled)
        {
            optimizer.stepDynamicParams(loss_scaler.getScaleFactor());
            loss_scaler.step(false);
        }
        else
        {
            optimizer.stepDynamicParams(1.0f);
        }

        invalidateLayerWeightCaches();

        engine.executeStaticReplay(_fence);
        return;
    }

    forward(_input_tensor);
    backward(_target_tensor);

    // printL2Norms();

    IOptimizer &optimizer = training_context.getOptimizer();
    if (is_mixed_precision_enabled)
    {
        auto param_grad_pairs = getParametersAndGradients();
        if (execution_target == Execution_Target::CPU)
        {
            bool overflow = loss_scaler.hasOverflow(param_grad_pairs);
            bool step_accepted = loss_scaler.step(overflow);
            if (step_accepted)
            {
                loss_scaler.unscaleGradients(param_grad_pairs);
                optimizer.step(param_grad_pairs);
            }
            else
            {
                zeroGradients();
            }
        }
        else
        {
            optimizer.step(param_grad_pairs, loss_scaler.getScaleFactor());
            loss_scaler.step(false);
        }
    }
    else
    {
        optimizer.step(getParametersAndGradients());
    }

    invalidateLayerWeightCaches();

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        engine.executeGraph(_fence);
        if (engine.isStaticGraphEnabled() && engine.getGraphExecutor().isStaticBaked(current_frame))
        {
            static_baked_input_buffers[current_frame] = current_in_buf;
            static_baked_target_buffers[current_frame] = current_tgt_buf;
        }
    }
}

int Neural_Network::findSubString(std::string_view string, std::string_view sub_string)
{
    if (sub_string.empty())
        return 0;
    auto pos = string.find(sub_string);
    return (pos != std::string_view::npos) ? static_cast<int>(pos) : -1;
}

std::string Neural_Network::modifyFilepath(const std::string &input, size_t epoch)
{
    if (input.empty())
        return "";
    std::string output = "";
    if (findSubString(input, ".nnck") != static_cast<int>(input.size()) - 5)
    {
        if (!input.contains("{}"))
        {
            if (input.back() == '_')
                output = input + std::format("epoch_{}.nnck", epoch);
            else
                output = input + std::format("_epoch_{}.nnck", epoch);
        }
        else
            output = std::vformat(input + ".nnck", std::make_format_args(epoch));
    }
    else
    {
        if (input.contains("{}"))
            output = std::vformat(input, std::make_format_args(epoch));
        else
            output = input;
    }
    return output;
}

void Neural_Network::fit(Async_Data_Pipeline &_data_pipeline,
                         size_t _total_epochs,
                         size_t _steps_per_epoch,
                         size_t _batch_size,
                         size_t _input_dimension,
                         size_t _output_dimension,
                         std::string checkpoint_file_path)
{
    setTrainingMode(true);

    Execution_Engine &engine = Execution_Engine::getInstance();

    _data_pipeline.setDevice(engine.getContext().getDevice());
    _data_pipeline.start();

    compileAndWarmup(_batch_size, _input_dimension, _output_dimension);

    for (size_t epoch = training_context.getCurrentEpoch(); epoch < _total_epochs; ++epoch)
    {
        Logger::logMessage("Start of epoch " + std::to_string(epoch), Log_Level::LOG_INFO, true);
        training_context.setCurrentEpoch(epoch);
        for (size_t step_index = 0; step_index < _steps_per_epoch; ++step_index)
        {
            // Logger::logMessage("Start of step " + std::to_string(step_index), Log_Level::LOG_INFO, true);
            Batch_Data batch_data = _data_pipeline.nextBatch(_batch_size, _input_dimension, _output_dimension);

            if (batch_data.input_matrix && batch_data.target_matrix)
            {
                if (batch_data.input_matrix->getExecutionTarget() != execution_target)
                {
                    batch_data.input_matrix->setExecutionTarget(execution_target);
                    batch_data.target_matrix->setExecutionTarget(execution_target);
                }

                trainStep(*batch_data.input_matrix, *batch_data.target_matrix, batch_data.fence);

                if (is_step_lr_per_batch)
                {
                    training_context.getLearningRate().step();
                }
            }
        }
        if (checkpoint_file_path != "")
            saveTrainingCheckpoint(modifyFilepath(checkpoint_file_path, epoch), epoch);
        if (!is_step_lr_per_batch)
        {
            training_context.getLearningRate().step();
        }
        Logger::resetLogCounters();
    }

    training_context.setCurrentEpoch(_total_epochs);
    _data_pipeline.stop();
}

void Neural_Network::saveInference(const std::string &_file_path) const
{
    Execution_Engine::getInstance().waitIdle();
    std::ofstream output_file_stream(_file_path, std::ios::binary);
    if (!output_file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Neural_Network::saveInference: Failed to open file: {}", _file_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Failed to open file for saving inference model");
    }

    Logger::logMessage(Input_Format{"Neural_Network::saveInference: Saving inference model to {}", _file_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    const char magic_header[4] = {'N', 'N', 'I', '1'};
    output_file_stream.write(magic_header, 4);

    uint32_t total_layer_count = static_cast<uint32_t>(layers.size());
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
    Logger::logMessage(Input_Format{"Neural_Network::saveInference: Inference saved to {}", _file_path}, Log_Level::LOG_INFO, true, 1);
}

void Neural_Network::loadInference(const std::string &_file_path, Execution_Target _execution_target)
{
    std::ifstream input_file_stream(_file_path, std::ios::binary);
    if (!input_file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Neural_Network::loadInference: Failed to open file: {}", _file_path},
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

    uint32_t total_layer_count = 0;
    input_file_stream.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

    Logger::logMessage(Input_Format{"Neural_Network::loadInference: Loading inference model from {}, total_layers={}",
                                    _file_path,
                                    total_layer_count},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    layers.clear();
    layers.reserve(total_layer_count);

    for (uint32_t i = 0; i < total_layer_count; ++i)
    {
        Layer_Type layer_type;
        input_file_stream.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type));
        Logger::logMessage(Input_Format{"Neural_Network::loadInference: Layer {} type = {}",
                                        i,
                                        magic_enum::enum_name(layer_type)},
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
        layer->setExecutionTarget(_execution_target);
        layer->setAccumulated(is_gradient_accumulation_enabled);
        layer->setMixedPrecision(is_mixed_precision_enabled);
        layer->invalidateWeightCache();
    }

    setExecutionTarget(_execution_target);
    if (_execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
    Logger::logMessage(Input_Format{"Neural_Network::loadInference: Inference loaded from {}", _file_path}, Log_Level::LOG_INFO, true, 1);
}

void Neural_Network::saveTrainingCheckpoint(const std::string &_file_path, size_t _current_epoch) const
{
    std::ofstream output_file_stream(_file_path, std::ios::binary);
    if (!output_file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Neural_Network::saveTrainingCheckpoint: Failed to open file: {}", _file_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Failed to open file for saving training checkpoint");
    }

    Logger::logMessage(Input_Format{"Neural_Network::saveTrainingCheckpoint: Saving checkpoint to {}, epoch={}",
                                    _file_path,
                                    _current_epoch},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    const char magic_header[4] = {'N', 'N', 'C', 'K'};
    output_file_stream.write(magic_header, 4);

    uint32_t epoch_value = static_cast<uint32_t>(_current_epoch);
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

    uint32_t total_layer_count = static_cast<uint32_t>(layers.size());
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

void Neural_Network::loadTrainingCheckpoint(const std::string &_file_path, size_t _total_epochs)
{
    Execution_Target _execution_target = getExecutionTarget();
    std::ifstream input_file_stream(_file_path, std::ios::binary);
    if (!input_file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Neural_Network::loadTrainingCheckpoint: Failed to open file: {}", _file_path},
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

    uint32_t total_layer_count = 0;
    input_file_stream.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count));

    Logger::logMessage(Input_Format{"Neural_Network::loadTrainingCheckpoint: Loading checkpoint from {}, total_layers={}",
                                    _file_path,
                                    total_layer_count},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::MODEL_SERIALIZATION);

    layers.clear();
    layers.reserve(total_layer_count);

    for (uint32_t i = 0; i < total_layer_count; ++i)
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
        layer->setExecutionTarget(_execution_target);
        layer->setAccumulated(is_gradient_accumulation_enabled);
        layer->setMixedPrecision(is_mixed_precision_enabled);
        layer->invalidateWeightCache();
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
    if (_execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
}

void Neural_Network::setCostFunction(std::unique_ptr<ICost_Function> _cost_function)
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

void Neural_Network::setLearningRate(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler)
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

void Neural_Network::setOptimizer(std::unique_ptr<IOptimizer> _optimizer)
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

void Neural_Network::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target != _new_execution_target)
    {
        Logger::logMessage(Input_Format{"Neural_Network::setExecutionTarget: Changing network execution target from {} to {}",
                                        magic_enum::enum_name(execution_target),
                                        magic_enum::enum_name(_new_execution_target)},
                           Log_Level::LOG_WARNING,
                           true,
                           1,
                           Log_Feature::DEVICE_MANAGEMENT);
    }
    execution_target = _new_execution_target;
    if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU)
    {
        loss_scaler.setMaxScale(1.0f);
        loss_scaler.setScaleFactor(1.0f);
    }
    last_prediction.setExecutionTarget(_new_execution_target);
    for (auto &layer : layers)
    {
        layer->setExecutionTarget(_new_execution_target);
        layer->setAccumulated(is_gradient_accumulation_enabled);
        layer->setMixedPrecision(is_mixed_precision_enabled);
        layer->invalidateWeightCache();
    }
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
    is_target_synchronized = true;
}

void Neural_Network::setTrainingMode(bool _is_training_mode)
{
    is_training_mode = _is_training_mode;
    Logger::logMessage(Input_Format{"Neural_Network::setTrainingMode: Setting training mode to {}",
                                    _is_training_mode ? "true" : "false"},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::TRAINING);
    for (auto &layer : layers)
    {
        layer->setTrainingMode(_is_training_mode);
        layer->invalidateWeightCache();
    }
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
}

void Neural_Network::setGradientAccumulation(bool _enable) noexcept
{
    is_gradient_accumulation_enabled = _enable;
    for (auto &layer : layers)
    {
        layer->setAccumulated(_enable);
    }
}

void Neural_Network::setMixedPrecision(bool _enable) noexcept
{
    is_mixed_precision_enabled = _enable;
    loss_scaler.setEnabled(_enable);
    if (_enable && execution_target == Execution_Target::VULKAN_GPU)
    {
        loss_scaler.setMaxScale(1.0f);
        loss_scaler.setScaleFactor(1.0f);
    }
    for (auto &layer : layers)
    {
        layer->setMixedPrecision(_enable);
        layer->invalidateWeightCache();
    }
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
    Logger::logMessage(Input_Format{"Neural_Network::setMixedPrecision: Mixed precision {} across {} layers. Target = {}, LossScaler: scale={:.1f}, max_scale={:.1f}",
                                    _enable ? "ENABLED" : "DISABLED", layers.size(),
                                    execution_target == Execution_Target::VULKAN_GPU ? "VULKAN_GPU" : "CPU",
                                    loss_scaler.getScaleFactor(), loss_scaler.getMaxScale()},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::FP16_METRICS | Log_Feature::LAYER_INSPECTION);
}

void Neural_Network::invalidateLayerWeightCaches() noexcept
{
    for (auto &layer : layers)
    {
        layer->invalidateWeightCache();
    }
}

void Neural_Network::invalidateStaticGraph() noexcept
{
    invalidateStaticBufferBindings();
    Execution_Engine::getInstance().invalidateStaticGraph();
}

void Neural_Network::enableCooperationMatrix(bool _enable) { Execution_Engine::getInstance().setCooperativeMatrixEnabled(_enable); }
void Neural_Network::enableStaticGraph(bool _enable) noexcept { Execution_Engine::getInstance().setStaticGraphEnabled(_enable); }
void Neural_Network::setStaticGraphEnabled(bool _enable) noexcept { Execution_Engine::getInstance().setStaticGraphEnabled(_enable); }
bool Neural_Network::isStaticGraphEnabled() const noexcept { return Execution_Engine::getInstance().isStaticGraphEnabled(); }
void Neural_Network::enableFusedGemmAdam(bool _enable) noexcept { Execution_Engine::getInstance().setFusedGemmAdamEnabled(_enable); }
void Neural_Network::setFusedGemmAdamEnabled(bool _enable) noexcept { Execution_Engine::getInstance().setFusedGemmAdamEnabled(_enable); }
bool Neural_Network::isFusedGemmAdamEnabled() const noexcept { return Execution_Engine::getInstance().isFusedGemmAdamEnabled(); }

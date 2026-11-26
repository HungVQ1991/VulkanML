#include "neural_network.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engine/async_data_pipeline.h"
#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "engine/loss_scaler.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "helper/training_profiler.h"
#include "learning_rate/cosine_annealing.h"
#include "math/gpu_tensor_impl.h"
#include "optimizer/adam_optimizer.h"
#include "optimizer/sgd_optimizer.h"
#include "training_context.h"

namespace
{
    std::pair<VkBuffer, size_t> extractBufferIdentity(const Tensor &_tensor) noexcept
    {
        if (_tensor.isEmpty() || _tensor.getExecutionTarget() != Execution_Target::VULKAN_GPU)
        {
            return {VK_NULL_HANDLE, 0};
        }
        const auto *gpu_impl = dynamic_cast<const Gpu_Tensor_Impl *>(_tensor.getImplementation().get());
        if (!gpu_impl)
        {
            return {VK_NULL_HANDLE, 0};
        }
        const auto &gpu_vec = gpu_impl->getStorageBuffer();
        return gpu_vec
            ? std::pair<VkBuffer, size_t>{gpu_vec->getBuffer(), gpu_vec->getId()}
            : std::pair<VkBuffer, size_t>{VK_NULL_HANDLE, 0};
    }

    VkFence toVkFence(Sync_Handle _handle) noexcept
    {
        return _handle.raw != 0 ? reinterpret_cast<VkFence>(static_cast<uintptr_t>(_handle.raw)) : VK_NULL_HANDLE;
    }

    void ensureParentDirectoryExists(const std::string &_file_path)
    {
        std::filesystem::path parent = std::filesystem::path(_file_path).parent_path();
        if (!parent.empty())
        {
            std::filesystem::create_directories(parent);
        }
    }
}

struct Neural_Network::Runtime_State
{
    struct Static_Bake_Key
    {
        uint64_t bake_generation = 0;
        VkBuffer in_handle = VK_NULL_HANDLE;
        size_t in_vector_id = 0;
        Shape in_shape;
        Data_Type in_dtype = Data_Type::FLOAT32;
        VkBuffer tgt_handle = VK_NULL_HANDLE;
        size_t tgt_vector_id = 0;
        Shape tgt_shape;
        Data_Type tgt_dtype = Data_Type::FLOAT32;
        float loss_scale = 1.0f;
        bool is_mixed_precision = false;
        bool is_grad_accum = false;
        const IOptimizer *optimizer_ptr = nullptr;
        uint64_t optimizer_config_generation = 0;

        bool operator==(const Static_Bake_Key &) const = default;
    };

    std::array<Static_Bake_Key, MAX_FRAMES_IN_FLIGHT> static_bake_keys{};
    std::array<size_t, MAX_FRAMES_IN_FLIGHT> static_fwd_nodes{};
    std::array<size_t, MAX_FRAMES_IN_FLIGHT> static_bwd_nodes{};
    std::array<size_t, MAX_FRAMES_IN_FLIGHT> static_opt_nodes{};
};

void Neural_Network::invalidateStaticBufferBindings() noexcept
{
    if (!runtime_state)
    {
        return;
    }
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        runtime_state->static_bake_keys[i] = Runtime_State::Static_Bake_Key{};
        runtime_state->static_fwd_nodes[i] = 0;
        runtime_state->static_bwd_nodes[i] = 0;
        runtime_state->static_opt_nodes[i] = 0;
    }
}

Neural_Network::Neural_Network(Execution_Target _execution_target)
    : last_prediction(0, 0, _execution_target),
      train_step_target_cache(0, 0, _execution_target),
      execution_target(_execution_target),
      runtime_state(std::make_unique<Runtime_State>())
{
}

Neural_Network::Neural_Network(Neural_Network &&other) noexcept
    : layers(std::move(other.layers)),
      last_prediction(std::move(other.last_prediction)),
      train_step_target_cache(std::move(other.train_step_target_cache)),
      execution_target(other.execution_target),
      training_context(std::move(other.training_context)),
      loss_scaler(other.loss_scaler),
      has_custom_loss_scaler(other.has_custom_loss_scaler),
      is_target_synchronized(other.is_target_synchronized),
      is_gradient_accumulation_enabled(other.is_gradient_accumulation_enabled),
      is_mixed_precision_enabled(other.is_mixed_precision_enabled),
      is_ternary_enabled(other.is_ternary_enabled),
      is_step_lr_per_batch(other.is_step_lr_per_batch),
      is_training_mode(other.is_training_mode),
      runtime_state(std::make_unique<Runtime_State>())
{
    invalidateStaticBufferBindings();
    other.invalidateStaticBufferBindings();
    other.execution_target = Execution_Target::CPU;
}

Neural_Network &Neural_Network::operator=(Neural_Network &&other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    if (execution_target == Execution_Target::VULKAN_GPU && (!layers.empty() || !last_prediction.isEmpty()))
    {
        try
        {
            Execution_Engine::getInstance().waitIdle();
        }
        catch (...)
        {
        }
    }
    layers = std::move(other.layers);
    last_prediction = std::move(other.last_prediction);
    train_step_target_cache = std::move(other.train_step_target_cache);
    execution_target = other.execution_target;
    training_context = std::move(other.training_context);
    loss_scaler = other.loss_scaler;
    has_custom_loss_scaler = other.has_custom_loss_scaler;
    is_target_synchronized = other.is_target_synchronized;
    is_gradient_accumulation_enabled = other.is_gradient_accumulation_enabled;
    is_mixed_precision_enabled = other.is_mixed_precision_enabled;
    is_ternary_enabled = other.is_ternary_enabled;
    is_step_lr_per_batch = other.is_step_lr_per_batch;
    is_training_mode = other.is_training_mode;
    if (!runtime_state)
    {
        runtime_state = std::make_unique<Runtime_State>();
    }
    invalidateStaticBufferBindings();
    other.invalidateStaticBufferBindings();
    other.execution_target = Execution_Target::CPU;
    return *this;
}

Neural_Network::~Neural_Network()
{
    if (execution_target == Execution_Target::VULKAN_GPU && (!layers.empty() || !last_prediction.isEmpty()))
    {
        try
        {
            Execution_Engine::getInstance().waitIdle();
        }
        catch (...)
        {
        }
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
    _layer->setTrainingMode(is_training_mode);
    if (is_ternary_enabled)
    {
        _layer->quantizeToTernary();
        _layer->setTrainingMode(is_training_mode);
    }
    Logger::logMessage(Input_Format{"Neural_Network::addLayer: Added layer type {}",
                                    magic_enum::enum_name<Layer_Type>(_layer->getLayerType())},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::LAYER_INSPECTION);
    layers.push_back(std::move(_layer));
    invalidateStaticBufferBindings();
}

void Neural_Network::resetGradients()
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
        if (layer->hasForwardHooks())
        {
            current_output = layer->applyForwardHooks(std::move(current_output));
        }
    }
    last_prediction = current_output;
    return current_output;
}

Tensor Neural_Network::backward(const Tensor &_target_tensor, bool _is_loss_gradient)
{
    return backwardInternal(_target_tensor, _is_loss_gradient, true);
}

Tensor Neural_Network::backwardInternal(const Tensor &_target_tensor, bool _is_loss_gradient, bool _compute_first_layer_input_grad)
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
    if (!_is_loss_gradient && training_context.hasCostFunction())
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
        ILayer *layer = layers[i - 1].get();
        if (layer->hasPreBackwardHooks())
        {
            gradient_tensor = layer->applyPreBackwardHooks(std::move(gradient_tensor));
        }
        const bool need_input_grad = (i > 1) || _compute_first_layer_input_grad || layer->hasPostBackwardHooks();
        gradient_tensor = need_input_grad ? layer->backward(gradient_tensor) : layer->backward(gradient_tensor, false);
        if (layer->hasPostBackwardHooks())
        {
            gradient_tensor = layer->applyPostBackwardHooks(std::move(gradient_tensor));
        }
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

std::vector<const Tensor *> Neural_Network::getNoDecayParameters() const
{
    std::vector<const Tensor *> no_decay_parameters;
    for (const auto &layer : layers)
    {
        if (layer && layer->hasParameters())
        {
            auto nd = layer->getNoDecayParameters();
            no_decay_parameters.insert(no_decay_parameters.end(), nd.begin(), nd.end());
        }
    }
    return no_decay_parameters;
}

float Neural_Network::clipGradients(float max_norm)
{
    if (std::isnan(max_norm) || max_norm <= 0.0f)
    {
        return 0.0f;
    }

    auto pairs = getParametersAndGradients();
    if (pairs.empty())
    {
        return 0.0f;
    }

    if (execution_target == Execution_Target::VULKAN_GPU &&
        !Execution_Engine::getInstance().getCurrentGraph().getNodes().empty())
    {
        Execution_Engine::getInstance().executeGraph();
    }

    double sum_squares = 0.0;
    bool has_nan_inf = false;
    size_t param_idx = 0;

    for (const auto &[param, grad] : pairs)
    {
        if (grad && !grad->isEmpty() && grad->getTotalElements() > 0)
        {
            const auto &data = grad->getData();
            size_t nan_count = 0;
            double p_sum_sq = 0.0;
            for (float val : data)
            {
                if (std::isnan(val) || std::isinf(val))
                {
                    has_nan_inf = true;
                    nan_count++;
                }
                else
                {
                    p_sum_sq += static_cast<double>(val) * static_cast<double>(val);
                }
            }
            if (nan_count > 0)
            {
                Logger::logMessage(Input_Format{"Neural_Network::clipGradients: Param #{} (shape={}) has {} NaN/Inf values! First={:.6e}",
                                                param_idx, grad->getShape().toString(), nan_count, data.empty() ? 0.0f : data[0]},
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::GRADIENT_HEALTH | Log_Feature::TRAINING);
            }
            sum_squares += p_sum_sq;
        }
        param_idx++;
    }

    if (has_nan_inf)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const double current_scale = (is_mixed_precision_enabled && loss_scaler.isEnabled() && loss_scaler.getScaleFactor() > 0.0f)
        ? static_cast<double>(loss_scaler.getScaleFactor())
        : 1.0;
    double effective_norm = std::sqrt(sum_squares) / current_scale;

    if (effective_norm > static_cast<double>(max_norm) && effective_norm > 1e-6)
    {
        float scale = static_cast<float>(static_cast<double>(max_norm) / (effective_norm + 1e-6));
        for (auto &[param, grad] : pairs)
        {
            if (grad && !grad->isEmpty() && grad->getTotalElements() > 0)
            {
                grad->mulScalar(scale, *grad);
            }
        }

        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }
        invalidateStaticBufferBindings();
    }

    return static_cast<float>(effective_norm);
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
    if (!training_context.hasOptimizer() || !training_context.hasCostFunction())
    {
        throw std::logic_error("Neural_Network::compileAndWarmup: Optimizer and CostFunction must be set before warmup");
    }

    Tensor dummy_input(_batch_size, _input_dimension, execution_target);
    Tensor dummy_target(_batch_size, _output_dimension, execution_target);

    IOptimizer &optimizer = training_context.getOptimizer();
    bool is_adam = (optimizer.getType() == Optimizer_Type::ADAM_OPTIMIZER);
    bool is_sgd = (optimizer.getType() == Optimizer_Type::SGD_OPTIMIZER);
    std::vector<Adam_Optimizer::Parameter_State> saved_loaded_states;
    std::unordered_map<Tensor *, Adam_Optimizer::Parameter_State> saved_param_states;
    std::vector<Tensor *> saved_param_order;
    std::unordered_set<const Tensor *> saved_no_decay;
    size_t saved_timestep = 0;

    std::vector<Tensor> saved_sgd_loaded;
    std::unordered_map<Tensor *, Tensor> saved_sgd_vels;
    std::vector<Tensor *> saved_sgd_order;

    if (is_adam)
    {
        auto &adam = static_cast<Adam_Optimizer &>(optimizer);
        for (const auto &st : adam.getLoadedStates())
        {
            saved_loaded_states.push_back(st.clone());
        }
        for (const auto &[p, st] : adam.getParameterStates())
        {
            saved_param_states.emplace(p, st.clone());
        }
        saved_param_order = adam.getParameterOrder();
        saved_no_decay = adam.getNoDecayParameters();
        saved_timestep = adam.getTimestep();
    }
    else if (is_sgd)
    {
        auto &sgd = static_cast<Sgd_Optimizer &>(optimizer);
        for (const auto &v : sgd.getLoadedVelocities())
        {
            saved_sgd_loaded.push_back(v.clone());
        }
        for (const auto &[p, v] : sgd.getVelocityStates())
        {
            saved_sgd_vels.emplace(p, v.clone());
        }
        saved_sgd_order = sgd.getVelocityOrder();
        saved_no_decay = sgd.getNoDecayParameters();
    }

    std::vector<std::vector<float>> saved_non_trainable;
    for (const auto &layer : layers)
    {
        if (layer)
        {
            layer->snapshotNonTrainableState(saved_non_trainable);
        }
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
    backwardInternal(dummy_target, false, false);

    float warmup_scale = is_mixed_precision_enabled ? loss_scaler.getScaleFactor() : 1.0f;
    optimizer.step(getParametersAndGradients(), warmup_scale);

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();
        engine.enableGraphCaching(true);
        Gpu_Tensor_Impl::distinct_operations_count = engine.getCurrentGraph().getNodeCount();
        engine.warmCache(engine.getCurrentGraph());
        engine.executeGraph();
        engine.waitIdle();
    }

    resetGradients();
    reset();
    optimizer.reset();

    for (size_t i = 0; i < params.size(); ++i)
    {
        if (params[i].first && !saved_parameters[i].empty())
        {
            params[i].first->uploadData(saved_parameters[i]);
        }
    }
    size_t non_trainable_cursor = 0;
    for (auto &layer : layers)
    {
        if (layer)
        {
            layer->restoreNonTrainableState(saved_non_trainable, non_trainable_cursor);
            layer->invalidateWeightCache();
        }
    }

    if (!saved_no_decay.empty())
    {
        optimizer.setNoDecayParameters(saved_no_decay);
    }
    if (is_adam)
    {
        auto &adam = static_cast<Adam_Optimizer &>(optimizer);
        if (!saved_param_states.empty())
        {
            adam.setParameterStates(saved_param_states);
            adam.setParameterOrder(saved_param_order);
        }
        if (!saved_loaded_states.empty())
        {
            adam.setLoadedStates(saved_loaded_states);
        }
        if (saved_timestep > 0)
        {
            adam.setTimestep(saved_timestep);
        }
    }
    else if (is_sgd)
    {
        auto &sgd = static_cast<Sgd_Optimizer &>(optimizer);
        if (!saved_sgd_vels.empty())
        {
            sgd.setVelocityStates(saved_sgd_vels);
            sgd.setVelocityOrder(saved_sgd_order);
        }
        if (!saved_sgd_loaded.empty())
        {
            sgd.setLoadedVelocities(saved_sgd_loaded);
        }
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();
        engine.getContext().executePendingTransfers();
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

        inspection_result += std::format("Param #{:<2} | Shape: {:<12} | ||W||: {:>10.4e} | ||dW||: {:>10.4e} | Ratio: {:>10.4e}\n",
                                         parameter_index++,
                                         parameter->getShape().toString(),
                                         parameter_norm,
                                         gradient_norm,
                                         norm_ratio);
    }
    inspection_result += "-----------------------------------------------\n\n";
    Logger::logMessage(inspection_result, Log_Level::LOG_DEBUG, true, 0, Log_Feature::LAYER_INSPECTION);
}

void Neural_Network::runOptimizerStepAndExecute(Sync_Handle _sync_handle)
{
    Execution_Engine *engine_ptr = (execution_target == Execution_Target::VULKAN_GPU) ? &Execution_Engine::getInstance() : nullptr;
    IOptimizer &optimizer = training_context.getOptimizer();
    const auto opt_record_start = std::chrono::high_resolution_clock::now();
    const size_t opt_nodes_before = engine_ptr ? engine_ptr->getCurrentGraph().getNodeCount() : 0;
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
                resetGradients();
            }
        }
        else
        {
            optimizer.step(param_grad_pairs, loss_scaler.getScaleFactor());
            const bool gpu_overflow = optimizer.consumeGpuOverflowDetected();
            loss_scaler.step(gpu_overflow);
        }
    }
    else
    {
        optimizer.step(getParametersAndGradients());
    }

    invalidateLayerWeightCaches();
    const auto opt_record_end = std::chrono::high_resolution_clock::now();
    Step_Timings::getInstance().recordDuration(
        Timing_Stage::OPT_RECORD,
        std::chrono::duration<double, std::milli>(opt_record_end - opt_record_start).count());

    if (engine_ptr)
    {
        const size_t opt_nodes_added = engine_ptr->getCurrentGraph().getNodeCount() - opt_nodes_before;
        Step_Timings::getInstance().setNodeCount(Node_Stage::OPT_GPU_NODES, opt_nodes_added);
        const uint32_t frame = engine_ptr->getContext().getCurrentFrame();
        if (runtime_state && frame < MAX_FRAMES_IN_FLIGHT)
        {
            runtime_state->static_opt_nodes[frame] = opt_nodes_added;
        }
        engine_ptr->executeGraph(toVkFence(_sync_handle), Execution_Stage::BACKWARD_OPTIMIZER);
    }
}

void Neural_Network::backwardAndStep(Tensor &_target_tensor, Sync_Handle _sync_handle)
{
    if (!training_context.hasOptimizer())
    {
        throw std::logic_error("Neural_Network::backwardAndStep: Optimizer must be set before backwardAndStep");
    }
    if (_target_tensor.getExecutionTarget() != execution_target)
    {
        _target_tensor.setExecutionTarget(execution_target);
    }

    const Data_Type pred_dtype = last_prediction.getDataType();
    Tensor *effective_target = &_target_tensor;
    if (_target_tensor.getDataType() != pred_dtype)
    {
        _target_tensor.to(pred_dtype, train_step_target_cache);
        effective_target = &train_step_target_cache;
    }

    backwardInternal(*effective_target, false, false);
    runOptimizerStepAndExecute(_sync_handle);
}

void Neural_Network::trainStep(const Batch_Data &_batch)
{
    if (!_batch.input_matrix || !_batch.target_matrix)
    {
        throw std::invalid_argument("Neural_Network::trainStep: Batch_Data matrices must not be null");
    }
    trainStep(*_batch.input_matrix, *_batch.target_matrix, _batch.sync_handle);
}

void Neural_Network::trainStep(Tensor &_input_tensor, Tensor &_target_tensor, Sync_Handle _sync_handle)
{
    if (!training_context.hasOptimizer())
    {
        throw std::logic_error("Neural_Network::trainStep: Optimizer must be set before trainStep");
    }
    if (!runtime_state)
    {
        runtime_state = std::make_unique<Runtime_State>();
    }
    if (_input_tensor.getExecutionTarget() != execution_target)
    {
        _input_tensor.setExecutionTarget(execution_target);
    }

    if (_target_tensor.getExecutionTarget() != execution_target)
    {
        _target_tensor.setExecutionTarget(execution_target);
    }

    Execution_Engine *engine_ptr = (execution_target == Execution_Target::VULKAN_GPU) ? &Execution_Engine::getInstance() : nullptr;
    uint32_t current_frame = 0;
    Runtime_State::Static_Bake_Key current_key{};
    bool network_supports_static_graph = true;
    for (const auto &layer : layers)
    {
        if (layer && !layer->supportsStaticGraph())
        {
            network_supports_static_graph = false;
            break;
        }
    }

    if (engine_ptr)
    {
        current_frame = engine_ptr->getContext().getCurrentFrame();
        current_key.bake_generation = engine_ptr->getGraphExecutor().getStaticBakeGeneration(current_frame);
        const auto [in_handle, in_vector_id] = extractBufferIdentity(_input_tensor);
        current_key.in_handle = in_handle;
        current_key.in_vector_id = in_vector_id;
        current_key.in_shape = _input_tensor.getShape();
        current_key.in_dtype = _input_tensor.getDataType();
        const auto [tgt_handle, tgt_vector_id] = extractBufferIdentity(_target_tensor);
        current_key.tgt_handle = tgt_handle;
        current_key.tgt_vector_id = tgt_vector_id;
        current_key.tgt_shape = _target_tensor.getShape();
        current_key.tgt_dtype = _target_tensor.getDataType();
        current_key.loss_scale = is_mixed_precision_enabled ? loss_scaler.getScaleFactor() : 1.0f;
        current_key.is_mixed_precision = is_mixed_precision_enabled;
        current_key.is_grad_accum = is_gradient_accumulation_enabled;
        current_key.optimizer_ptr = &training_context.getOptimizer();
        current_key.optimizer_config_generation = training_context.getOptimizer().getConfigGeneration();

        bool can_fast_replay = network_supports_static_graph &&
                               !hasHooks() &&
                               is_training_mode &&
                               engine_ptr->isStaticGraphEnabled() &&
                               engine_ptr->getGraphExecutor().isStaticBaked(current_frame) &&
                               engine_ptr->getGraphExecutor().areStaticBakedBuffersValid(current_frame) &&
                               current_key.bake_generation != 0 &&
                               current_key.in_handle != VK_NULL_HANDLE &&
                               current_key.tgt_handle != VK_NULL_HANDLE &&
                               runtime_state->static_bake_keys[current_frame] == current_key;

        if (can_fast_replay)
        {
            const auto opt_record_start = std::chrono::high_resolution_clock::now();
            IOptimizer &optimizer = training_context.getOptimizer();
            if (is_mixed_precision_enabled)
            {
                optimizer.stepDynamicParams(loss_scaler.getScaleFactor());
                const bool gpu_overflow = optimizer.consumeGpuOverflowDetected();
                loss_scaler.step(gpu_overflow);
            }
            else
            {
                optimizer.stepDynamicParams(1.0f);
                (void)optimizer.consumeGpuOverflowDetected();
            }

            if (!layers.empty())
            {
                last_prediction = layers.back()->getOutput();
            }

            invalidateLayerWeightCaches();
            const auto opt_record_end = std::chrono::high_resolution_clock::now();
            auto &timings = Step_Timings::getInstance();
            timings.recordDuration(
                Timing_Stage::OPT_RECORD,
                std::chrono::duration<double, std::milli>(opt_record_end - opt_record_start).count());
            if (current_frame < MAX_FRAMES_IN_FLIGHT)
            {
                timings.setNodeCount(Node_Stage::FWD_LOSS_GPU_NODES, runtime_state->static_fwd_nodes[current_frame]);
                timings.setNodeCount(Node_Stage::BWD_GPU_NODES, runtime_state->static_bwd_nodes[current_frame]);
                timings.setNodeCount(Node_Stage::OPT_GPU_NODES, runtime_state->static_opt_nodes[current_frame]);
            }

            engine_ptr->executeStaticReplay(toVkFence(_sync_handle), Execution_Stage::BACKWARD_OPTIMIZER);
            return;
        }

        if (engine_ptr->isStaticGraphEnabled() && is_mixed_precision_enabled)
        {
            invalidateLayerWeightCaches();
        }
    }

    Tensor *effective_target = &_target_tensor;
    if (_target_tensor.getDataType() != _input_tensor.getDataType())
    {
        _target_tensor.to(_input_tensor.getDataType(), train_step_target_cache);
        effective_target = &train_step_target_cache;
    }

    const auto fwd_record_start = std::chrono::high_resolution_clock::now();
    const size_t fwd_nodes_before = engine_ptr ? engine_ptr->getCurrentGraph().getNodeCount() : 0;
    forward(_input_tensor);
    const auto fwd_record_end = std::chrono::high_resolution_clock::now();
    const size_t fwd_nodes_added = engine_ptr ? (engine_ptr->getCurrentGraph().getNodeCount() - fwd_nodes_before) : 0;

    const auto bwd_record_start = std::chrono::high_resolution_clock::now();
    const size_t bwd_nodes_before = engine_ptr ? engine_ptr->getCurrentGraph().getNodeCount() : 0;
    backwardInternal(*effective_target, false, false);
    const auto bwd_record_end = std::chrono::high_resolution_clock::now();
    const size_t bwd_nodes_added = engine_ptr ? (engine_ptr->getCurrentGraph().getNodeCount() - bwd_nodes_before) : 0;

    {
        auto &timings = Step_Timings::getInstance();
        timings.recordDuration(
            Timing_Stage::FWD_RECORD,
            std::chrono::duration<double, std::milli>(fwd_record_end - fwd_record_start).count());
        timings.recordDuration(
            Timing_Stage::BWD_RECORD,
            std::chrono::duration<double, std::milli>(bwd_record_end - bwd_record_start).count());
        if (engine_ptr)
        {
            timings.setNodeCount(Node_Stage::FWD_LOSS_GPU_NODES, fwd_nodes_added);
            timings.setNodeCount(Node_Stage::BWD_GPU_NODES, bwd_nodes_added);
        }
    }

    uint32_t submitted_frame = engine_ptr ? engine_ptr->getContext().getCurrentFrame() : current_frame;
    runOptimizerStepAndExecute(_sync_handle);

    if (engine_ptr)
    {
        if (submitted_frame < MAX_FRAMES_IN_FLIGHT)
        {
            runtime_state->static_fwd_nodes[submitted_frame] = fwd_nodes_added;
            runtime_state->static_bwd_nodes[submitted_frame] = bwd_nodes_added;
        }
        if (network_supports_static_graph &&
            engine_ptr->isStaticGraphEnabled() &&
            engine_ptr->getGraphExecutor().isStaticBaked(submitted_frame))
        {
            current_key.bake_generation = engine_ptr->getGraphExecutor().getStaticBakeGeneration(submitted_frame);
            runtime_state->static_bake_keys[submitted_frame] = current_key;
        }
        else if (!network_supports_static_graph)
        {
            runtime_state->static_bake_keys[submitted_frame] = Runtime_State::Static_Bake_Key{};
        }
    }
}

Tensor Neural_Network::predict(const Tensor &_input_tensor)
{
    struct Training_Mode_Guard
    {
        Neural_Network &net;
        bool saved_mode;
        ~Training_Mode_Guard()
        {
            if (net.is_training_mode != saved_mode)
            {
                try
                {
                    net.setTrainingMode(saved_mode);
                }
                catch (...)
                {
                }
            }
        }
    };

    const bool prev_mode = is_training_mode;
    if (prev_mode)
    {
        setTrainingMode(false);
    }
    Training_Mode_Guard guard{*this, prev_mode};

    Tensor prediction = forward(_input_tensor);
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        engine.waitIdle();
    }
    return prediction;
}

std::vector<float> Neural_Network::predictData(const Tensor &_input_tensor)
{
    return predict(_input_tensor).getData();
}

int Neural_Network::findSubString(std::string_view string, std::string_view sub_string)
{
    if (sub_string.empty())
        return 0;
    auto pos = string.find(sub_string);
    if (pos == std::string_view::npos)
        return -1;
    if (pos > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("Neural_Network::findSubString: Position exceeds int range");
    }
    return static_cast<int>(pos);
}

std::string Neural_Network::modifyFilepath(const std::string &input, size_t epoch)
{
    if (input.empty())
        return "";
    std::string output = "";
    if (input.size() < 5 || !input.ends_with(".nnck"))
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
    Fit_Config config;
    config.epochs = _total_epochs;
    config.steps_per_epoch = _steps_per_epoch;
    config.batch_size = _batch_size;
    config.input_dim = _input_dimension;
    config.output_dim = _output_dimension;
    config.checkpoint_path = std::move(checkpoint_file_path);
    config.checkpoint_every = config.checkpoint_path.empty() ? 0 : 1;
    fit(_data_pipeline, config, Fit_Callbacks{});
}

void Neural_Network::fit(Async_Data_Pipeline &_data_pipeline,
                         const Fit_Config &_config,
                         const Fit_Callbacks &_callbacks)
{
    if (!training_context.hasOptimizer() || !training_context.hasLearningRateScheduler() || !training_context.hasCostFunction())
    {
        throw std::logic_error("Neural_Network::fit: Optimizer, LearningRate, and CostFunction must be set before fit");
    }
    if (_config.batch_size == 0 || _config.input_dim == 0 || _config.output_dim == 0)
    {
        throw std::invalid_argument("Neural_Network::fit: batch_size, input_dim, and output_dim must be > 0");
    }

    setTrainingMode(true);

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        _data_pipeline.bindActiveDevice();
    }
    else
    {
        _data_pipeline.releaseDevice();
    }
    _data_pipeline.start();

    struct Pipeline_Stop_Guard
    {
        Async_Data_Pipeline &pipeline;
        ~Pipeline_Stop_Guard()
        {
            pipeline.stop();
        }
    } pipeline_guard{_data_pipeline};

    compileAndWarmup(_config.batch_size, _config.input_dim, _config.output_dim);

    const bool lr_needs_metric = training_context.getLearningRateScheduler().requiresMetric();
    const bool track_loss = _config.compute_step_loss || lr_needs_metric || _config.log_every_steps > 0;
    const auto fit_start = std::chrono::steady_clock::now();
    bool stop_requested = false;

    for (size_t epoch = training_context.getCurrentEpoch(); epoch < _config.epochs && !stop_requested; ++epoch)
    {
        Logger::logMessage(Input_Format{"Start of epoch {}", epoch}, Log_Level::LOG_INFO, true, 0, Log_Feature::TRAINING);
        training_context.setCurrentEpoch(epoch);
        double epoch_loss_sum = 0.0;
        size_t epoch_loss_count = 0;

        for (size_t step_index = 0; step_index < _config.steps_per_epoch; ++step_index)
        {
            const auto batch_step_start = std::chrono::high_resolution_clock::now();
            auto &timings = Step_Timings::getInstance();
            const double dp_before = timings.getDuration(Timing_Stage::DATA_PREP);
            const auto dp_start = std::chrono::high_resolution_clock::now();
            Batch_Data batch_data = _data_pipeline.nextBatch(_config.batch_size, _config.input_dim, _config.output_dim);
            const auto dp_end = std::chrono::high_resolution_clock::now();
            const double dp_after = timings.getDuration(Timing_Stage::DATA_PREP);
            const double dp_ms = std::chrono::duration<double, std::milli>(dp_end - dp_start).count();
            timings.recordDuration(Timing_Stage::DATA_PREP, std::max(0.0, dp_ms - std::max(0.0, dp_after - dp_before)));

            float batch_loss = 0.0f;

            if (batch_data.input_matrix && batch_data.target_matrix)
            {
                if (batch_data.input_matrix->getExecutionTarget() != execution_target)
                {
                    batch_data.input_matrix->setExecutionTarget(execution_target);
                    batch_data.target_matrix->setExecutionTarget(execution_target);
                }

                trainStep(batch_data);

                if (track_loss && !layers.empty() && getLastPrediction().getShape() == batch_data.target_matrix->getShape())
                {
                    const auto loss_start = std::chrono::high_resolution_clock::now();
                    batch_loss = training_context.getCostFunction().computeLoss(getLastPrediction(), *batch_data.target_matrix);
                    const auto loss_end = std::chrono::high_resolution_clock::now();
                    timings.recordDuration(
                        Timing_Stage::LOSS_READBACK,
                        std::chrono::duration<double, std::milli>(loss_end - loss_start).count());
                    epoch_loss_sum += static_cast<double>(batch_loss);
                    ++epoch_loss_count;
                    if (lr_needs_metric && is_step_lr_per_batch)
                    {
                        const auto sched_start = std::chrono::high_resolution_clock::now();
                        training_context.getLearningRateScheduler().step(batch_loss);
                        const auto sched_end = std::chrono::high_resolution_clock::now();
                        timings.recordDuration(
                            Timing_Stage::SCHED_STEP,
                            std::chrono::duration<double, std::milli>(sched_end - sched_start).count());
                    }
                }
                if (!lr_needs_metric && is_step_lr_per_batch)
                {
                    const auto sched_start = std::chrono::high_resolution_clock::now();
                    training_context.getLearningRateScheduler().step();
                    const auto sched_end = std::chrono::high_resolution_clock::now();
                    timings.recordDuration(
                        Timing_Stage::SCHED_STEP,
                        std::chrono::duration<double, std::milli>(sched_end - sched_start).count());
                }
            }

            const auto batch_step_end = std::chrono::high_resolution_clock::now();
            timings.recordDuration(
                Timing_Stage::TOTAL_BATCH,
                std::chrono::duration<double, std::milli>(batch_step_end - batch_step_start).count());

            const auto now = std::chrono::steady_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(now - fit_start).count();
            const float avg_loss = epoch_loss_count > 0
                ? static_cast<float>(epoch_loss_sum / static_cast<double>(epoch_loss_count))
                : 0.0f;
            const Fit_Progress progress{
                epoch,
                step_index + 1,
                batch_loss,
                avg_loss,
                elapsed_ms,
                training_context.getLearningRateScheduler().getCurrentRate()
            };

            if (_config.log_every_steps > 0 &&
                (((step_index + 1) % _config.log_every_steps == 0) || (step_index + 1 == _config.steps_per_epoch)))
            {
                Logger::logMessage(
                    Input_Format{"Epoch {}/{} Step {}/{} | loss={:.6f} avg={:.6f} lr={:.6e} elapsed={:.1f}ms",
                                 epoch + 1, _config.epochs, step_index + 1, _config.steps_per_epoch,
                                 batch_loss, avg_loss, progress.learning_rate, elapsed_ms},
                    Log_Level::LOG_INFO, true, 0, Log_Feature::TRAINING);
            }

            if (_callbacks.on_step)
            {
                _callbacks.on_step(progress);
            }
            if (_callbacks.should_stop && _callbacks.should_stop(progress))
            {
                stop_requested = true;
                break;
            }
        }

        if (!_config.checkpoint_path.empty() && _config.checkpoint_every > 0 &&
            (((epoch + 1) % _config.checkpoint_every == 0) || (epoch + 1 == _config.epochs)))
        {
            saveTrainingCheckpoint(modifyFilepath(_config.checkpoint_path, epoch), epoch);
        }

        const float epoch_loss = epoch_loss_count > 0
            ? static_cast<float>(epoch_loss_sum / static_cast<double>(epoch_loss_count))
            : 0.0F;
        if (!is_step_lr_per_batch)
        {
            if (lr_needs_metric)
            {
                training_context.getLearningRateScheduler().step(epoch_loss);
            }
            else
            {
                training_context.getLearningRateScheduler().step();
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const Fit_Progress epoch_progress{
            epoch,
            _config.steps_per_epoch,
            epoch_loss,
            epoch_loss,
            std::chrono::duration<double, std::milli>(now - fit_start).count(),
            training_context.getLearningRateScheduler().getCurrentRate()
        };
        if (_callbacks.on_epoch_end)
        {
            _callbacks.on_epoch_end(epoch_progress);
        }
        if (_callbacks.should_stop && _callbacks.should_stop(epoch_progress))
        {
            stop_requested = true;
        }
        Logger::resetLogCounters();
        if (stop_requested)
        {
            training_context.setCurrentEpoch(epoch + 1);
            break;
        }
    }

    if (!stop_requested)
    {
        training_context.setCurrentEpoch(_config.epochs);
    }
    _data_pipeline.stop();
}

void Neural_Network::saveInference(const std::string &_file_path) const
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        engine.waitIdle();
    }
    ensureParentDirectoryExists(_file_path);
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
    output_file_stream.flush();
    if (!output_file_stream.good())
    {
        throw std::runtime_error("Neural_Network::saveInference: Failed while writing inference model");
    }
    Logger::logMessage(Input_Format{"Neural_Network::saveInference: Inference saved to {}", _file_path},
                       Log_Level::LOG_INFO, true, 1, Log_Feature::MODEL_SERIALIZATION);
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

    char magic_header[4] = {};
    if (!input_file_stream.read(magic_header, 4) ||
        magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'I' || magic_header[3] != '1')
    {
        Logger::logMessage("Neural_Network::loadInference: Invalid magic header",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        throw std::runtime_error("Invalid magic header for inference model");
    }

    uint32_t total_layer_count = 0;
    if (!input_file_stream.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count)) ||
        total_layer_count > 100000u)
    {
        throw std::runtime_error("Neural_Network::loadInference: Invalid or truncated layer count");
    }

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
        Layer_Type layer_type{};
        if (!input_file_stream.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type)))
        {
            throw std::runtime_error("Neural_Network::loadInference: Truncated layer configuration header");
        }
        Logger::logMessage(Input_Format{"Neural_Network::loadInference: Layer {} type = {}",
                                        i,
                                        magic_enum::enum_name(layer_type)},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        layers.push_back(Training_Context::constructLayerFromConfig(input_file_stream, layer_type, _execution_target));
    }

    bool any_ternary = false;
    for (auto &layer : layers)
    {
        if (layer->hasParameters())
        {
            layer->loadInference(input_file_stream);
        }
        if (layer->isTernary())
        {
            any_ternary = true;
        }
        layer->setExecutionTarget(_execution_target);
        layer->setAccumulated(is_gradient_accumulation_enabled);
        layer->setMixedPrecision(is_mixed_precision_enabled);
        layer->invalidateWeightCache();
    }
    is_ternary_enabled = any_ternary;

    setExecutionTarget(_execution_target);
    setTrainingMode(false);
    if (_execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
    Logger::logMessage(Input_Format{"Neural_Network::loadInference: Inference loaded from {}", _file_path},
                       Log_Level::LOG_INFO, true, 1, Log_Feature::MODEL_SERIALIZATION);
}

void Neural_Network::saveTrainingCheckpoint(const std::string &_file_path, size_t _current_epoch) const
{
    if (!training_context.hasCostFunction() || !training_context.hasLearningRateScheduler() || !training_context.hasOptimizer())
    {
        throw std::logic_error("Neural_Network::saveTrainingCheckpoint: CostFunction, LearningRate, and Optimizer must be set");
    }
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        engine.waitIdle();
    }

    ensureParentDirectoryExists(_file_path);
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

    const ILearning_Rate &learning_rate_scheduler = training_context.getLearningRateScheduler();
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

    const char ext_magic[4] = {'N', 'N', 'X', '2'};
    uint8_t flags = 0;
    if (is_mixed_precision_enabled) flags |= 0x01u;
    if (is_ternary_enabled) flags |= 0x02u;
    if (is_gradient_accumulation_enabled) flags |= 0x04u;
    if (is_step_lr_per_batch) flags |= 0x08u;
    if (loss_scaler.isEnabled()) flags |= 0x10u;

    const float ls_scale = loss_scaler.getScaleFactor();
    const float ls_growth = loss_scaler.getGrowthFactor();
    const float ls_backoff = loss_scaler.getBackoffFactor();
    const float ls_min = loss_scaler.getMinScale();
    const float ls_max = loss_scaler.getMaxScale();
    const uint32_t ls_interval = loss_scaler.getGrowthInterval();
    const uint32_t ls_good = loss_scaler.getGoodSteps();

    output_file_stream.write(ext_magic, 4);
    output_file_stream.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_scale), sizeof(ls_scale));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_growth), sizeof(ls_growth));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_backoff), sizeof(ls_backoff));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_min), sizeof(ls_min));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_max), sizeof(ls_max));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_interval), sizeof(ls_interval));
    output_file_stream.write(reinterpret_cast<const char *>(&ls_good), sizeof(ls_good));
    output_file_stream.flush();
    if (!output_file_stream.good())
    {
        throw std::runtime_error("Neural_Network::saveTrainingCheckpoint: Failed while writing checkpoint");
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
    if (!input_file_stream.read(reinterpret_cast<char *>(&total_layer_count), sizeof(total_layer_count)) ||
        total_layer_count > 100000u)
    {
        throw std::runtime_error("Neural_Network::loadTrainingCheckpoint: Invalid or truncated layer count");
    }

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
        Layer_Type layer_type{};
        if (!input_file_stream.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type)))
        {
            throw std::runtime_error("Neural_Network::loadTrainingCheckpoint: Truncated layer configuration header");
        }
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

    bool has_v2_trailer = false;
    uint8_t loaded_flags = 0;
    float ls_scale = 1.0f;
    float ls_growth = 2.0f;
    float ls_backoff = 0.5f;
    float ls_min = 1.0f;
    float ls_max = 1024.0f;
    uint32_t ls_interval = 2000;
    uint32_t ls_good = 0;
    if (input_file_stream.peek() != std::ifstream::traits_type::eof())
    {
        char ext_magic[4] = {};
        if (input_file_stream.read(ext_magic, 4) &&
            ext_magic[0] == 'N' && ext_magic[1] == 'N' && ext_magic[2] == 'X' && ext_magic[3] == '2')
        {
            input_file_stream.read(reinterpret_cast<char *>(&loaded_flags), sizeof(loaded_flags));
            input_file_stream.read(reinterpret_cast<char *>(&ls_scale), sizeof(ls_scale));
            input_file_stream.read(reinterpret_cast<char *>(&ls_growth), sizeof(ls_growth));
            input_file_stream.read(reinterpret_cast<char *>(&ls_backoff), sizeof(ls_backoff));
            input_file_stream.read(reinterpret_cast<char *>(&ls_min), sizeof(ls_min));
            input_file_stream.read(reinterpret_cast<char *>(&ls_max), sizeof(ls_max));
            input_file_stream.read(reinterpret_cast<char *>(&ls_interval), sizeof(ls_interval));
            input_file_stream.read(reinterpret_cast<char *>(&ls_good), sizeof(ls_good));
            if (!input_file_stream)
            {
                throw std::runtime_error("Neural_Network::loadTrainingCheckpoint: Truncated NNX2 trailer");
            }
            has_v2_trailer = true;
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
    if (_total_epochs > 0)
    {
        const int clamped_total_epochs = (_total_epochs > static_cast<size_t>(std::numeric_limits<int>::max()))
            ? std::numeric_limits<int>::max()
            : static_cast<int>(_total_epochs);
        if (auto *cos_sched = dynamic_cast<Cosine_Annealing *>(&training_context.getLearningRateScheduler()))
        {
            if (cos_sched->getWarmupSteps() >= clamped_total_epochs)
            {
                cos_sched->setWarmupSteps(std::max(0, clamped_total_epochs - 1));
            }
        }
        training_context.getLearningRateScheduler().setMaxEpoch(clamped_total_epochs);
    }

    setExecutionTarget(_execution_target);

    if (has_v2_trailer)
    {
        is_mixed_precision_enabled = (loaded_flags & 0x01u) != 0;
        is_ternary_enabled = (loaded_flags & 0x02u) != 0;
        is_gradient_accumulation_enabled = (loaded_flags & 0x04u) != 0;
        is_step_lr_per_batch = (loaded_flags & 0x08u) != 0;
        for (auto &layer : layers)
        {
            if (layer)
            {
                layer->setAccumulated(is_gradient_accumulation_enabled);
                layer->setMixedPrecision(is_mixed_precision_enabled);
                layer->invalidateWeightCache();
            }
        }
        loss_scaler.setEnabled((loaded_flags & 0x10u) != 0);
        if (std::isfinite(ls_min) && ls_min >= 1.0f &&
            std::isfinite(ls_max) && ls_max >= ls_min &&
            std::isfinite(ls_scale) && ls_scale >= 1.0f &&
            std::isfinite(ls_growth) && ls_growth >= 1.0f &&
            std::isfinite(ls_backoff) && ls_backoff > 0.0f && ls_backoff < 1.0f)
        {
            loss_scaler.setMinScale(ls_min);
            loss_scaler.setMaxScale(ls_max);
            loss_scaler.setScaleFactor(ls_scale);
            loss_scaler.setGrowthFactor(ls_growth);
            loss_scaler.setBackoffFactor(ls_backoff);
            loss_scaler.setGrowthInterval(ls_interval);
            loss_scaler.setGoodSteps(ls_good);
            has_custom_loss_scaler = true;
        }
    }
    else
    {
        bool any_ternary = false;
        for (const auto &l : layers)
        {
            if (l && l->isTernary())
            {
                any_ternary = true;
                break;
            }
        }
        is_ternary_enabled = any_ternary;
    }

    auto nd = getNoDecayParameters();
    if (!nd.empty() && training_context.hasOptimizer())
    {
        training_context.getOptimizer().setNoDecayParameters(
            std::unordered_set<const Tensor *>(nd.begin(), nd.end()));
    }

    if (_execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
}

void Neural_Network::setTrainingContext(Training_Context _training_context) noexcept
{
    training_context = std::move(_training_context);
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
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
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
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
    training_context.setLearningRateScheduler(std::move(_learning_rate_scheduler));
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
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
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
    if (!has_custom_loss_scaler && is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU)
    {
        loss_scaler.setMaxScale(1.0f);
        loss_scaler.setScaleFactor(1.0f);
    }
    last_prediction.setExecutionTarget(_new_execution_target);
    train_step_target_cache.setExecutionTarget(_new_execution_target);
    if (training_context.hasOptimizer())
    {
        training_context.getOptimizer().setExecutionTarget(_new_execution_target);
    }
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

void Neural_Network::enableGradientAccumulation(bool _enable)
{
    is_gradient_accumulation_enabled = _enable;
    for (auto &layer : layers)
    {
        layer->setAccumulated(_enable);
    }
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
}

void Neural_Network::enableMixedPrecision(bool _enable)
{
    is_mixed_precision_enabled = _enable;
    loss_scaler.setEnabled(_enable);
    if (_enable && execution_target == Execution_Target::VULKAN_GPU && !has_custom_loss_scaler)
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
    Logger::logMessage(Input_Format{"Neural_Network::enableMixedPrecision: Mixed precision {} across {} layers. Target = {}, LossScaler: scale={:.1f}, max_scale={:.1f}",
                                    _enable ? "ENABLED" : "DISABLED", layers.size(),
                                    execution_target == Execution_Target::VULKAN_GPU ? "VULKAN_GPU" : "CPU",
                                    loss_scaler.getScaleFactor(), loss_scaler.getMaxScale()},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::FP16_METRICS | Log_Feature::LAYER_INSPECTION);
}

void Neural_Network::enableTernary(bool _enable)
{
    is_ternary_enabled = _enable;
    for (auto &layer : layers)
    {
        if (layer)
        {
            if (_enable)
            {
                layer->quantizeToTernary();
            }
            else
            {
                layer->dequantizeFromTernary();
            }
            layer->setTrainingMode(is_training_mode);
        }
    }
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
            engine.waitIdle();
        }
        engine.invalidateStaticGraph();
    }
    Logger::logMessage(Input_Format{"Neural_Network::enableTernary: Ternary mode {} across {} layers",
                                    _enable ? "ENABLED" : "DISABLED", layers.size()},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::TERNARY_METRICS | Log_Feature::LAYER_INSPECTION);
}

void Neural_Network::prepareInference()
{
    for (auto &layer : layers)
    {
        if (layer)
        {
            layer->prepareInference();
        }
    }
    invalidateStaticBufferBindings();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
            engine.waitIdle();
        }
        engine.invalidateStaticGraph();
    }
    Logger::logMessage(Input_Format{"Neural_Network::prepareInference: Prepared inference across {} layers", layers.size()},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::TERNARY_METRICS | Log_Feature::LAYER_INSPECTION);
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

void Neural_Network::enableCooperativeMatrix(bool _enable) { Execution_Engine::getInstance().setCooperativeMatrixEnabled(_enable); }
void Neural_Network::enableStaticGraph(bool _enable) noexcept { Execution_Engine::getInstance().enableStaticGraph(_enable); }
bool Neural_Network::isStaticGraphEnabled() const noexcept { return Execution_Engine::getInstance().isStaticGraphEnabled(); }
void Neural_Network::enableFusedGemmAdam(bool _enable) noexcept { Execution_Engine::getInstance().enableFusedGemmAdam(_enable); }
bool Neural_Network::isFusedGemmAdamEnabled() const noexcept { return Execution_Engine::getInstance().isFusedGemmAdamEnabled(); }

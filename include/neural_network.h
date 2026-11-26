#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cost_function/icost_function.h"
#include "engine/async_data_pipeline.h"
#include "engine/loss_scaler.h"
#include "layer/ilayer.h"
#include "math/tensor.h"
#include "training_context.h"

struct Fit_Config
{
    size_t epochs = 1;
    size_t steps_per_epoch = 0;
    size_t batch_size = 1;
    size_t input_dim = 0;
    size_t output_dim = 0;
    std::string checkpoint_path;
    size_t checkpoint_every = 1;
    size_t log_every_steps = 0;
    bool compute_step_loss = false;
};

struct Fit_Progress
{
    size_t epoch = 0;
    size_t step = 0;
    float batch_loss = 0.0f;
    float epoch_avg_loss = 0.0f;
    double elapsed_ms = 0.0;
    float learning_rate = 0.0f;
};

struct Fit_Callbacks
{
    std::function<void(const Fit_Progress &)> on_step;
    std::function<void(const Fit_Progress &)> on_epoch_end;
    std::function<bool(const Fit_Progress &)> should_stop;
};

class Neural_Network
{
private:
    std::vector<std::unique_ptr<ILayer>> layers;
    Tensor last_prediction;
    Tensor train_step_target_cache;
    Execution_Target execution_target = Execution_Target::CPU;
    Training_Context training_context;
    Loss_Scaler loss_scaler;
    bool has_custom_loss_scaler = false;
    bool is_target_synchronized = false;
    bool is_gradient_accumulation_enabled = false;
    bool is_mixed_precision_enabled = false;
    bool is_ternary_enabled = false;
    bool is_step_lr_per_batch = false;
    bool is_training_mode = true;
    struct Runtime_State;
    std::unique_ptr<Runtime_State> runtime_state;

    void invalidateStaticBufferBindings() noexcept;
    void runOptimizerStepAndExecute(Sync_Handle _sync_handle);
    Tensor backwardInternal(const Tensor &_target_tensor, bool _is_loss_gradient, bool _compute_first_layer_input_grad);

public:
    explicit Neural_Network(Execution_Target _execution_target = Execution_Target::CPU);
    ~Neural_Network();

    Neural_Network(const Neural_Network &) = delete;
    Neural_Network &operator=(const Neural_Network &) = delete;
    Neural_Network(Neural_Network &&other) noexcept;
    Neural_Network &operator=(Neural_Network &&other) noexcept;

    void addLayer(std::unique_ptr<ILayer> _layer);

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        addLayer(std::move(new_layer));
        return layer_reference;
    }

    void resetGradients();
    Tensor forward(const Tensor &_input_tensor);
    Tensor backward(const Tensor &_target_tensor, bool _is_loss_gradient = true);

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients();
    std::vector<const Tensor *> getNoDecayParameters() const;
    float clipGradients(float max_norm = 1.0f);

    void reset();
    void zeroGradients() { resetGradients(); }
    void prepareInference();
    Tensor predict(const Tensor &_input_tensor);
    std::vector<float> predictData(const Tensor &_input_tensor);
    void compileAndWarmup(size_t _batch_size, size_t _input_dimension, size_t _output_dimension);
    void printL2Norms();
    void trainStep(Tensor &_input_tensor, Tensor &_target_tensor, Sync_Handle _sync_handle = {});
    void trainStep(const Batch_Data &_batch);
    void backwardAndStep(Tensor &_target_tensor, Sync_Handle _sync_handle = {});

    static int findSubString(std::string_view string, std::string_view sub_string);
    static std::string modifyFilepath(const std::string &input, size_t epoch);

    void fit(Async_Data_Pipeline &_data_pipeline,
             size_t _total_epochs,
             size_t _steps_per_epoch,
             size_t _batch_size,
             size_t _input_dimension,
             size_t _output_dimension,
             std::string checkpoint_file_path);
    void fit(Async_Data_Pipeline &_data_pipeline,
             const Fit_Config &_config,
             const Fit_Callbacks &_callbacks = {});

    void saveInference(const std::string &_file_path) const;
    void loadInference(const std::string &_file_path, Execution_Target _execution_target = Execution_Target::CPU);

    void saveTrainingCheckpoint(const std::string &_file_path, size_t _current_epoch) const;
    void loadTrainingCheckpoint(const std::string &_file_path, size_t _total_epochs);

    const ILearning_Rate &getLearningRate() const { return training_context.getLearningRateScheduler(); }
    ILearning_Rate &getLearningRate() { return training_context.getLearningRateScheduler(); }
    const ICost_Function &getCostFunction() const { return training_context.getCostFunction(); }
    ICost_Function &getCostFunction() { return training_context.getCostFunction(); }
    size_t getCurrentEpoch() const noexcept { return training_context.getCurrentEpoch(); }
    const IOptimizer &getOptimizer() const { return training_context.getOptimizer(); }
    IOptimizer &getOptimizer() { return training_context.getOptimizer(); }
    const ILayer &getLayer(size_t _index) const { return *layers.at(_index); }
    ILayer &getLayer(size_t _index) { return *layers.at(_index); }
    const Training_Context &getTrainingContext() const noexcept { return training_context; }
    Training_Context &getTrainingContext() noexcept { return training_context; }
    const Tensor &getLastPrediction() const noexcept { return last_prediction; }
    Tensor &getLastPrediction() noexcept { return last_prediction; }
    size_t getLayerCount() const noexcept { return layers.size(); }
    const std::vector<std::unique_ptr<ILayer>> &getLayers() const noexcept { return layers; }
    std::vector<std::unique_ptr<ILayer>> &getLayers() noexcept { return layers; }
    const Loss_Scaler &getLossScaler() const noexcept { return loss_scaler; }
    Loss_Scaler &getLossScaler() noexcept { return loss_scaler; }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    bool isGradientAccumulationEnabled() const noexcept { return is_gradient_accumulation_enabled; }
    bool isMixedPrecisionEnabled() const noexcept { return is_mixed_precision_enabled; }
    bool isTargetSynchronized() const noexcept { return is_target_synchronized; }

    void setTrainingContext(Training_Context _training_context) noexcept;
    void setCostFunction(std::unique_ptr<ICost_Function> _cost_function);

    template <std::derived_from<ICost_Function> Cost_Type_T, typename... Args>
    Cost_Type_T &setCostFunction(Args &&...args)
    {
        auto cost = std::make_unique<Cost_Type_T>(std::forward<Args>(args)...);
        Cost_Type_T &cost_reference = *cost;
        setCostFunction(std::move(cost));
        return cost_reference;
    }

    void setLearningRate(std::unique_ptr<ILearning_Rate> _learning_rate_scheduler);

    template <std::derived_from<ILearning_Rate> Scheduler_Type_T, typename... Args>
    Scheduler_Type_T &setLearningRate(Args &&...args)
    {
        auto scheduler = std::make_unique<Scheduler_Type_T>(std::forward<Args>(args)...);
        Scheduler_Type_T &scheduler_reference = *scheduler;
        setLearningRate(std::move(scheduler));
        return scheduler_reference;
    }

    void setOptimizer(std::unique_ptr<IOptimizer> _optimizer);

    template <std::derived_from<IOptimizer> Optimizer_Type_T, typename... Args>
    Optimizer_Type_T &setOptimizer(Args &&...args)
    {
        auto opt = std::make_unique<Optimizer_Type_T>(std::forward<Args>(args)...);
        Optimizer_Type_T &opt_reference = *opt;
        setOptimizer(std::move(opt));
        return opt_reference;
    }

    void setCurrentEpoch(size_t _epoch) noexcept { training_context.setCurrentEpoch(_epoch); }
    void setLastPrediction(const Tensor &_prediction) { last_prediction = _prediction; }

    void setExecutionTarget(Execution_Target _new_execution_target);
    void setTrainingMode(bool _is_training_mode);
    bool isTrainingMode() const noexcept { return is_training_mode; }
    void enableGradientAccumulation(bool _enable = true);
    void setLossScaler(Loss_Scaler _loss_scaler) noexcept { loss_scaler = _loss_scaler; has_custom_loss_scaler = true; }
    void enableMixedPrecision(bool _enable = true);
    void invalidateLayerWeightCaches() noexcept;

    void enableTernary(bool _enable = true);
    bool isTernaryEnabled() const noexcept { return is_ternary_enabled; }
    void setStepLearningRatePerBatch(bool _enable) noexcept { is_step_lr_per_batch = _enable; }
    bool isStepLearningRatePerBatch() const noexcept { return is_step_lr_per_batch; }
    void setTargetSynchronized(bool _synced) noexcept { is_target_synchronized = _synced; }
    void enableCooperativeMatrix(bool _enable = true);
    void enableStaticGraph(bool _enable = true) noexcept;
    bool isStaticGraphEnabled() const noexcept;
    void enableFusedGemmAdam(bool _enable = true) noexcept;
    bool isFusedGemmAdamEnabled() const noexcept;
    void invalidateStaticGraph() noexcept;

    Neural_Network &withTarget(Execution_Target _target) { setExecutionTarget(_target); return *this; }
    Neural_Network &withMixedPrecision(bool _enable = true) { enableMixedPrecision(_enable); return *this; }
    Neural_Network &withTernary(bool _enable = true) { enableTernary(_enable); return *this; }
    Neural_Network &withStaticGraph(bool _enable = true) noexcept { enableStaticGraph(_enable); return *this; }
    Neural_Network &withCooperativeMatrix(bool _enable = true) { enableCooperativeMatrix(_enable); return *this; }
    Neural_Network &withFusedGemmAdam(bool _enable = true) noexcept { enableFusedGemmAdam(_enable); return *this; }
    Neural_Network &withGradientAccumulation(bool _enable = true) { enableGradientAccumulation(_enable); return *this; }
    Neural_Network &withStepLrPerBatch(bool _enable = true) noexcept { setStepLearningRatePerBatch(_enable); return *this; }

    using Hook_Type = ::Hook_Type;
    using Hook_Fn = ILayer::Hook_Fn;
    using Void_Hook_Fn = ILayer::Void_Hook_Fn;

    bool hasHooks() const noexcept
    {
        for (const auto &layer : layers)
        {
            if (layer && (layer->hasForwardHooks() || layer->hasPreBackwardHooks() || layer->hasPostBackwardHooks()))
            {
                return true;
            }
        }
        return false;
    }

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &>
    Scoped_Hook registerForwardHook(size_t layer_index, Callback_T &&callback)
    {
        return layers.at(layer_index)->registerForwardHook(std::forward<Callback_T>(callback));
    }

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &>
    Scoped_Hook registerBackwardHook(size_t layer_index, Callback_T &&callback, Hook_Type type = Hook_Type::PRE_BACKWARD)
    {
        return layers.at(layer_index)->registerBackwardHook(std::forward<Callback_T>(callback), type);
    }
};

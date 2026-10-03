#pragma once

#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cost_function/icost_function.h"
#include "engine/async_data_pipeline.h"
#include "engine/loss_scaler.h"
#include "engine/vulkan_context.h"
#include "layer/ilayer.h"
#include "math/tensor.h"
#include "training_context.h"

class Neural_Network
{
private:
    std::vector<std::unique_ptr<ILayer>> layers;
    Tensor last_prediction;
    Execution_Target execution_target = Execution_Target::CPU;
    Training_Context training_context;
    Loss_Scaler loss_scaler;
    bool is_target_synchronized = false;
    bool is_gradient_accumulation_enabled = false;
    bool is_mixed_precision_enabled = false;
    bool is_step_lr_per_batch = false;
    bool is_training_mode = true;
    VkBuffer static_baked_input_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkBuffer static_baked_target_buffers[MAX_FRAMES_IN_FLIGHT]{VK_NULL_HANDLE, VK_NULL_HANDLE};

    static VkBuffer extractBufferHandle(const Tensor &_tensor) noexcept;
    void invalidateStaticBufferBindings() noexcept;

public:
    explicit Neural_Network(Execution_Target _execution_target = Execution_Target::CPU);
    ~Neural_Network();

    Neural_Network(const Neural_Network &) = delete;
    Neural_Network &operator=(const Neural_Network &) = delete;
    Neural_Network(Neural_Network &&) noexcept = default;
    Neural_Network &operator=(Neural_Network &&) noexcept = default;

    void addLayer(std::unique_ptr<ILayer> _layer);

    template <std::derived_from<ILayer> Layer_Type_T, typename... Args>
    Layer_Type_T &addLayer(Args &&...args)
    {
        auto new_layer = std::make_unique<Layer_Type_T>(std::forward<Args>(args)...);
        Layer_Type_T &layer_reference = *new_layer;
        addLayer(std::move(new_layer));
        return layer_reference;
    }

    void zeroGradients();
    Tensor forward(const Tensor &_input_tensor);
    Tensor backward(const Tensor &_target_tensor);

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients();
    std::vector<std::pair<Tensor *, Tensor *>> getParamsAndGrads();

    void reset();
    void resetGradients() { zeroGradients(); }
    void compileAndWarmup(size_t _batch_size, size_t _input_dimension, size_t _output_dimension);
    void printL2Norms();
    void trainStep(Tensor &_input_tensor, Tensor &_target_tensor, VkFence _fence = VK_NULL_HANDLE);

    static int findSubString(std::string_view string, std::string_view sub_string);
    static std::string modifyFilepath(const std::string &input, size_t epoch);

    void fit(Async_Data_Pipeline &_data_pipeline,
             size_t _total_epochs,
             size_t _steps_per_epoch,
             size_t _batch_size,
             size_t _input_dimension,
             size_t _output_dimension,
             std::string checkpoint_file_path);

    void saveInference(const std::string &_file_path) const;
    void loadInference(const std::string &_file_path, Execution_Target _execution_target = Execution_Target::CPU);

    void saveTrainingCheckpoint(const std::string &_file_path, size_t _current_epoch) const;
    void loadTrainingCheckpoint(const std::string &_file_path, size_t _total_epochs);

    const ILearning_Rate &getLearningRate() const { return training_context.getLearningRate(); }
    ILearning_Rate &getLearningRate() { return training_context.getLearningRate(); }
    const ICost_Function &getCostFunction() const { return training_context.getCostFunction(); }
    ICost_Function &getCostFunction() { return training_context.getCostFunction(); }
    size_t getCurrentEpoch() const noexcept { return training_context.getCurrentEpoch(); }
    const IOptimizer &getOptimizer() const { return training_context.getOptimizer(); }
    IOptimizer &getOptimizer() { return training_context.getOptimizer(); }
    const ILayer &getLayer(size_t _index) const { return *layers.at(_index); }
    ILayer &getLayer(size_t _index) { return *layers.at(_index); }
    const Training_Context &getTrainingContext() const noexcept { return training_context; }
    Training_Context &getTrainingContext() noexcept { return training_context; }
    const Training_Context &getContext() const noexcept { return training_context; }
    Training_Context &getContext() noexcept { return training_context; }
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

    void setTrainingContext(Training_Context _training_context) noexcept { training_context = std::move(_training_context); }
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
    void setTarget(Execution_Target _new_execution_target) { setExecutionTarget(_new_execution_target); }
    void setTrainingMode(bool _is_training_mode);
    void setGradientAccumulation(bool _enable) noexcept;
    void setLossScaler(Loss_Scaler _loss_scaler) noexcept { loss_scaler = _loss_scaler; }
    void setMixedPrecision(bool _enable) noexcept;
    void invalidateLayerWeightCaches() noexcept;

    void setMixedPrecisionEnabled(bool _enable) noexcept { setMixedPrecision(_enable); }
    void enableMixedPrecision(bool _enable = true) noexcept { setMixedPrecision(_enable); }
    void setStepLearningRatePerBatch(bool _enable) noexcept { is_step_lr_per_batch = _enable; }
    bool isStepLearningRatePerBatch() const noexcept { return is_step_lr_per_batch; }
    void setGradientAccumulationEnabled(bool _enable) noexcept { setGradientAccumulation(_enable); }
    void setTargetSynchronized(bool _synced) noexcept { is_target_synchronized = _synced; }
    void enableCooperationMatrix(bool _enable = true);
    void enableStaticGraph(bool _enable = true) noexcept;
    void setStaticGraphEnabled(bool _enable) noexcept;
    bool isStaticGraphEnabled() const noexcept;
    void enableFusedGemmAdam(bool _enable = true) noexcept;
    void setFusedGemmAdamEnabled(bool _enable) noexcept;
    bool isFusedGemmAdamEnabled() const noexcept;
    void invalidateStaticGraph() noexcept;
};
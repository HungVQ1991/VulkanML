#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <functional>
#include <random>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"

template<typename T>
std::string getEnumString(T str)
{
    return static_cast<std::string>(magic_enum::enum_name<T>(str));
}

enum class Layer_Type
{
    LINEAR,
    CONV2D,
    BATCH_NORM,
    BATCH_NORM_2D,
    MAX_POOL_2D,
    GLOBAL_AVG_POOL_2D,
    RELU,
    GELU,
    SOFTMAX,
    RES_NET_BLOCK_2D,
    RES_NET_20,
    PPO_ACTOR_CRITIC,
    RMSNORM,
    SWIGLU,
    EMBEDDING,
    TRANSFORMER_BLOCK, SIGMOID, DROPOUT,
    UP_SAMPLE_2D,
    U_NET,
    INSTANCE_NORM_2D,
    LEAKY_RELU,
    NN_UNET,
    MULTI_SCALE_RESNET_18,
    AVG_POOL_2D,
    TERNARY_LINEAR,
    LAYER_TYPE_END
};

enum class Hook_Type
{
    FORWARD,
    PRE_BACKWARD,
    POST_BACKWARD
};

class ILayer
{
private:
    static const Tensor &emptyTensor()
    {
        static const Tensor empty_tensor(0, 0);
        return empty_tensor;
    }

protected:
    bool is_accumulated = false;
    bool is_mixed_precision_enabled = false;

public:
    void logBufferAddress(Tensor *_target_tensor, const std::string &_tensor_name) const
    {
        return;
        std::string layer_name = std::string(magic_enum::enum_name<Layer_Type>(getLayerType()));
        if (!_target_tensor)
        {
            Logger::logMessage(Input_Format{"{}::logBufferAddress: Target tensor is null", layer_name},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LAYER_INSPECTION);
            return;
        }

        std::shared_ptr<gpu::vector> gpu_vector = nullptr;
        auto storage_handle = _target_tensor->getStorage();
        if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
        {
            gpu_vector = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
        }

        if (!gpu_vector || gpu_vector->isEmpty())
        {
            Logger::logMessage(Input_Format{"{}::logBufferAddress: GPU storage of target is empty or invalid", layer_name},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LAYER_INSPECTION);
            return;
        }

        Logger::logMessage(Input_Format{"{}::logBufferAddress: Buffer info at {}: Address: {:p}, Size: {}",
                                       layer_name,
                                       _tensor_name,
                                       static_cast<const void *>(gpu_vector->getBuffer()),
                                       gpu_vector->getSize()},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LAYER_INSPECTION);

        const auto &host_data = _target_tensor->getData();
        if (host_data.empty())
        {
            Logger::logMessage(Input_Format{"{}::logBufferAddress: {}: Data of target is empty", layer_name, _tensor_name},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::LAYER_INSPECTION);
            return;
        }

        double sum_of_squares = 0.0;
        for (float value : host_data)
        {
            sum_of_squares += static_cast<double>(value) * value;
        }
        float norm_value = static_cast<float>(std::sqrt(sum_of_squares));

        size_t sample_size = std::min<size_t>(4, host_data.size());
        std::string sample_string;
        for (size_t i = 0; i < sample_size; ++i)
        {
            sample_string += std::format("{:.4e} ", host_data[i]);
        }

        Logger::logMessage(Input_Format{"{}::inspectGradient: {:<18}| Shape: {:>4}x{:<5} | ||G||: {:.6e} | Top: [{}]",
                                       layer_name,
                                       _tensor_name,
                                       _target_tensor->getRows(),
                                       _target_tensor->getColumns(),
                                       norm_value,
                                       sample_string},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LAYER_INSPECTION);
    }

    void logChangeExecutionTarget(Execution_Target new_target)
    {
        Logger::logMessage(Input_Format{"ILayer::setExecutionTarget: Change target at layer {} from {} to {}", 
            getEnumString<Layer_Type>(getLayerType()),
            getEnumString<Execution_Target>(getExecutionTarget()), 
            getEnumString<Execution_Target>(new_target)},
            Log_Level::LOG_DEBUG, true, 0, Log_Feature::DEVICE_MANAGEMENT);
    }

    virtual ~ILayer() noexcept = default;

    virtual Tensor forward(const Tensor &_input_tensor) { return Tensor{}; }
    virtual Tensor backward(const Tensor &_output_gradient) = 0;

    virtual bool hasParameters() const { return false; }
    virtual bool supportsPopulationBatch() const { return false; }
    virtual void resetGradient() {}
    virtual void resetGradients() { resetGradient(); }

    virtual Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
    {
        throw std::logic_error(std::format("{} does not support population-batched forward",
            getEnumString<Layer_Type>(getLayerType())));
    }

    virtual std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const
    {
        return [](std::mt19937&) { return 0.0f; }; 
    }

    virtual std::unique_ptr<ILayer> clone() const = 0; 

    virtual void saveConfiguration(std::ofstream &_output_file_stream) const = 0;
    virtual void saveConfig(std::ofstream &_output_file_stream) const { saveConfiguration(_output_file_stream); }

    virtual void saveInference(std::ofstream &_output_file_stream) const = 0;
    virtual void loadInference(std::ifstream &_input_file_stream) = 0;
    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &_input_file_stream) = 0;

    virtual const Tensor &getWeights() const { return emptyTensor(); }
    virtual const Tensor &getBiases() const { return emptyTensor(); }
    virtual const Tensor &getWeightsGradient() const { return emptyTensor(); }
    virtual const Tensor &getInput() const { return emptyTensor(); }
    virtual const Tensor &getOutput() const { return emptyTensor(); }
    virtual std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() { return {}; }
    virtual std::vector<std::pair<Tensor *, Tensor *>> getParamsAndGrads() { return getParametersAndGradients(); }
    virtual std::vector<const Tensor *> getNoDecayParameters() const { return {}; }
    virtual std::vector<Shape> getPopulationParameterDims() const { return {}; }
    virtual std::vector<bool> getPopulationParameterIsEvolvable() const { return std::vector<bool>(getPopulationParameterDims().size(), true); }
    virtual std::vector<float> getPopulationParameter(size_t param_index) const { return {}; }
    virtual Execution_Target getExecutionTarget() const = 0;
    virtual Layer_Type getLayerType() const = 0;
    virtual bool isAccumulated() const noexcept { return is_accumulated; }

    virtual void setPopulationParameter(size_t param_index, std::vector<float> flat_data) {}
    virtual void setExecutionTarget(Execution_Target _execution_target) = 0;
    virtual void setTarget(Execution_Target _execution_target) { setExecutionTarget(_execution_target); }
    virtual void setAccumulated(bool _is_accumulated) noexcept { is_accumulated = _is_accumulated; }
    virtual void setTrainingMode(bool _is_training) {}
    virtual void setMixedPrecision(bool _enable) noexcept { is_mixed_precision_enabled = _enable; }
    virtual bool isMixedPrecisionEnabled() const noexcept { return is_mixed_precision_enabled; }
    virtual void invalidateWeightCache() noexcept {}

public:

    using Hook_Fn = std::function<Tensor(const Tensor &)>;
    using Void_Hook_Fn = std::function<void(const Tensor &)>;

private:
    struct Hook_Entry
    {
        uint64_t id = 0;
        Hook_Fn callback;
    };

    std::vector<Hook_Entry> forward_hooks;
    std::vector<Hook_Entry> pre_backward_hooks;
    std::vector<Hook_Entry> post_backward_hooks;
    inline static std::atomic<uint64_t> global_next_hook_id{1};

public:
    bool hasForwardHooks() const noexcept { return !forward_hooks.empty(); }
    bool hasPreBackwardHooks() const noexcept { return !pre_backward_hooks.empty(); }
    bool hasPostBackwardHooks() const noexcept { return !post_backward_hooks.empty(); }

    Tensor applyForwardHooks(Tensor tensor) const
    {
        for (const auto &entry : forward_hooks)
        {
            Tensor result = entry.callback(tensor);
            if (!result.isEmpty())
            {
                tensor = std::move(result);
            }
        }
        return tensor;
    }

    Tensor applyPreBackwardHooks(Tensor tensor) const
    {
        for (const auto &entry : pre_backward_hooks)
        {
            Tensor result = entry.callback(tensor);
            if (!result.isEmpty())
            {
                tensor = std::move(result);
            }
        }
        return tensor;
    }

    Tensor applyPostBackwardHooks(Tensor tensor) const
    {
        for (const auto &entry : post_backward_hooks)
        {
            Tensor result = entry.callback(tensor);
            if (!result.isEmpty())
            {
                tensor = std::move(result);
            }
        }
        return tensor;
    }

    void removeHook(uint64_t id, Hook_Type type)
    {
        auto remove_from = [id](std::vector<Hook_Entry> &hooks) {
            std::erase_if(hooks, [id](const Hook_Entry &entry) {
                return entry.id == id;
            });
        };

        switch (type)
        {
        case Hook_Type::FORWARD:
            remove_from(forward_hooks);
            break;
        case Hook_Type::PRE_BACKWARD:
            remove_from(pre_backward_hooks);
            break;
        case Hook_Type::POST_BACKWARD:
            remove_from(post_backward_hooks);
            break;
        }
    }

    void clearHooks() noexcept
    {
        forward_hooks.clear();
        pre_backward_hooks.clear();
        post_backward_hooks.clear();
    }

    class Scoped_Hook
    {
    private:
        ILayer *layer = nullptr;
        uint64_t hook_id = 0;
        Hook_Type hook_type = Hook_Type::FORWARD;

    public:
        Scoped_Hook() noexcept = default;

        Scoped_Hook(ILayer *target_layer, uint64_t id, Hook_Type type) noexcept
            : layer(target_layer), hook_id(id), hook_type(type)
        {
        }

        ~Scoped_Hook()
        {
            release();
        }

        Scoped_Hook(const Scoped_Hook &) = delete;
        Scoped_Hook &operator=(const Scoped_Hook &) = delete;

        Scoped_Hook(Scoped_Hook &&other) noexcept
            : layer(other.layer), hook_id(other.hook_id), hook_type(other.hook_type)
        {
            other.layer = nullptr;
            other.hook_id = 0;
        }

        Scoped_Hook &operator=(Scoped_Hook &&other) noexcept
        {
            if (this != &other)
            {
                release();
                layer = other.layer;
                hook_id = other.hook_id;
                hook_type = other.hook_type;
                other.layer = nullptr;
                other.hook_id = 0;
            }
            return *this;
        }

        void release()
        {
            if (layer && hook_id != 0)
            {
                layer->removeHook(hook_id, hook_type);
                layer = nullptr;
                hook_id = 0;
            }
        }

        void unregister()
        {
            release();
        }

        bool isActive() const noexcept
        {
            return layer != nullptr && hook_id != 0;
        }

        uint64_t getId() const noexcept
        {
            return hook_id;
        }

        Hook_Type getType() const noexcept
        {
            return hook_type;
        }

        ILayer *getLayer() const noexcept
        {
            return layer;
        }
    };

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &>
    Scoped_Hook registerForwardHook(Callback_T &&callback)
    {
        using Ret_T = std::invoke_result_t<Callback_T, const Tensor &>;
        uint64_t id = global_next_hook_id.fetch_add(1);
        if constexpr (std::is_void_v<Ret_T>)
        {
            forward_hooks.push_back(Hook_Entry{id, [fn = std::forward<Callback_T>(callback)](const Tensor &t) -> Tensor {
                fn(t);
                return Tensor{};
            }});
        }
        else
        {
            forward_hooks.push_back(Hook_Entry{id, Hook_Fn(std::forward<Callback_T>(callback))});
        }
        return Scoped_Hook(this, id, Hook_Type::FORWARD);
    }

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &>
    Scoped_Hook registerBackwardHook(Callback_T &&callback, Hook_Type type = Hook_Type::PRE_BACKWARD)
    {
        using Ret_T = std::invoke_result_t<Callback_T, const Tensor &>;
        uint64_t id = global_next_hook_id.fetch_add(1);
        Hook_Fn fn_target;
        if constexpr (std::is_void_v<Ret_T>)
        {
            fn_target = [fn = std::forward<Callback_T>(callback)](const Tensor &t) -> Tensor {
                fn(t);
                return Tensor{};
            };
        }
        else
        {
            fn_target = Hook_Fn(std::forward<Callback_T>(callback));
        }

        if (type == Hook_Type::PRE_BACKWARD)
        {
            pre_backward_hooks.push_back(Hook_Entry{id, std::move(fn_target)});
        }
        else
        {
            post_backward_hooks.push_back(Hook_Entry{id, std::move(fn_target)});
        }
        return Scoped_Hook(this, id, type);
    }

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &>
    Scoped_Hook registerPreBackwardHook(Callback_T &&callback)
    {
        return registerBackwardHook(std::forward<Callback_T>(callback), Hook_Type::PRE_BACKWARD);
    }

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &>
    Scoped_Hook registerPostBackwardHook(Callback_T &&callback)
    {
        return registerBackwardHook(std::forward<Callback_T>(callback), Hook_Type::POST_BACKWARD);
    }
};

using Hook_Fn = ILayer::Hook_Fn;
using Void_Hook_Fn = ILayer::Void_Hook_Fn;
using Scoped_Hook = ILayer::Scoped_Hook;
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <functional>
#include <random>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"

enum class Layer_Type : uint32_t
{
    LINEAR = 0,
    CONV2D = 1,
    BATCH_NORM = 2,
    BATCH_NORM_2D = 3,
    MAX_POOL_2D = 4,
    GLOBAL_AVG_POOL_2D = 5,
    RELU = 6,
    GELU = 7,
    SOFTMAX = 8,
    RES_NET_BLOCK_2D = 9,
    RES_NET_20 = 10,
    PPO_ACTOR_CRITIC = 11,
    RMSNORM = 12,
    SWIGLU = 13,
    EMBEDDING = 14,
    TRANSFORMER_BLOCK = 15,
    SIGMOID = 16,
    DROPOUT = 17,
    UP_SAMPLE_2D = 18,
    U_NET = 19,
    INSTANCE_NORM_2D = 20,
    LEAKY_RELU = 21,
    NN_UNET = 22,
    MULTI_SCALE_RESNET_18 = 23,
    AVG_POOL_2D = 24,
    TERNARY_LINEAR = 25,
    LORA_LINEAR = 26,
    LAYER_TYPE_END = 27
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
    static const Tensor &emptyTensor();
    std::shared_ptr<bool> hook_lifetime_token = std::make_shared<bool>(true);

protected:
    bool is_accumulated = false;
    bool is_mixed_precision_enabled = false;
    bool is_trainable = true;

    template<typename T>
    static std::string getEnumString(T str)
    {
        return static_cast<std::string>(magic_enum::enum_name<T>(str));
    }

    void copyBaseStateTo(ILayer &_target) const noexcept
    {
        _target.setAccumulated(is_accumulated);
        _target.setMixedPrecision(is_mixed_precision_enabled);
        _target.setTrainable(is_trainable);
    }

public:
    void logBufferAddress(const Tensor * , const std::string & ) const noexcept {}

    void logChangeExecutionTarget(Execution_Target new_target)
    {
        Logger::logMessage(Input_Format{"ILayer::setExecutionTarget: Change target at layer {} from {} to {}",
            getEnumString<Layer_Type>(getLayerType()),
            getEnumString<Execution_Target>(getExecutionTarget()),
            getEnumString<Execution_Target>(new_target)},
            Log_Level::LOG_DEBUG, true, 0, Log_Feature::DEVICE_MANAGEMENT);
    }

        ILayer() = default;
    ILayer(const ILayer &other)
        : hook_lifetime_token(std::make_shared<bool>(true)),
          is_accumulated(other.is_accumulated),
          is_mixed_precision_enabled(other.is_mixed_precision_enabled),
          is_trainable(other.is_trainable)
    {
    }
    ILayer &operator=(const ILayer &other)
    {
        if (this != &other)
        {
            is_accumulated = other.is_accumulated;
            is_mixed_precision_enabled = other.is_mixed_precision_enabled;
            is_trainable = other.is_trainable;
        }
        return *this;
    }
    ILayer(ILayer &&other)
        : hook_lifetime_token(std::make_shared<bool>(true)),
          is_accumulated(other.is_accumulated),
          is_mixed_precision_enabled(other.is_mixed_precision_enabled),
          is_trainable(other.is_trainable)
    {
    }
    ILayer &operator=(ILayer &&other) noexcept
    {
        if (this != &other)
        {
            is_accumulated = other.is_accumulated;
            is_mixed_precision_enabled = other.is_mixed_precision_enabled;
            is_trainable = other.is_trainable;
        }
        return *this;
    }

    virtual ~ILayer() noexcept = default;

    virtual Tensor forward(const Tensor &_input_tensor) = 0;
    virtual Tensor backward(const Tensor &_output_gradient) = 0;
    virtual Tensor backward(const Tensor &_output_gradient, bool _compute_input_gradient)
    {
        (void)_compute_input_gradient;
        return backward(_output_gradient);
    }

    virtual bool hasParameters() const { return false; }
    virtual bool supportsPopulationBatch() const { return false; }
    virtual bool supportsStaticGraph() const noexcept { return true; }
    virtual void resetGradient() {}
    virtual void resetForwardState() { resetGradient(); }
    virtual void resetGradients() { resetGradient(); }

    virtual Tensor forward(const Tensor &, const std::vector<Tensor> &) const
    {
        throw std::logic_error(std::format("{} does not support population-batched forward",
            getEnumString<Layer_Type>(getLayerType())));
    }

    virtual std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t) const
    {
        return [](std::mt19937&) { return 0.0f; };
    }

    virtual std::unique_ptr<ILayer> clone() const = 0;

    virtual void saveConfiguration(std::ofstream &_output_file_stream) const = 0;

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
    virtual std::vector<const Tensor *> getNoDecayParameters() const { return {}; }
    virtual std::vector<Shape> getPopulationParameterDims() const { return {}; }
    virtual std::vector<bool> getPopulationParameterIsEvolvable() const { return std::vector<bool>(getPopulationParameterDims().size(), true); }
    virtual std::vector<float> getPopulationParameter(size_t) const { return {}; }
    virtual Execution_Target getExecutionTarget() const = 0;
    virtual Layer_Type getLayerType() const = 0;
    virtual bool isAccumulated() const noexcept { return is_accumulated; }

    virtual void setPopulationParameter(size_t, std::vector<float>) {}
    virtual void setExecutionTarget(Execution_Target _execution_target) = 0;
    virtual void setAccumulated(bool _is_accumulated) noexcept { is_accumulated = _is_accumulated; }
    virtual void setTrainingMode(bool) {}
    virtual void setMixedPrecision(bool _enable) noexcept { is_mixed_precision_enabled = _enable; }
    virtual bool isMixedPrecisionEnabled() const noexcept { return is_mixed_precision_enabled; }
    virtual void setTrainable(bool _trainable) noexcept { is_trainable = _trainable; }
    virtual bool isTrainable() const noexcept { return is_trainable; }
    virtual void invalidateWeightCache() noexcept {}
    virtual void prepareInference() {}
    virtual void quantizeToTernary() {}
    virtual void dequantizeFromTernary() {}
    virtual bool isTernary() const noexcept { return false; }
    virtual void snapshotNonTrainableState(std::vector<std::vector<float>> &) const {}
    virtual void restoreNonTrainableState(const std::vector<std::vector<float>> &, size_t &) {}

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
    static uint64_t nextHookId() noexcept;

public:
    bool hasForwardHooks() const noexcept { return !forward_hooks.empty(); }
    bool hasPreBackwardHooks() const noexcept { return !pre_backward_hooks.empty(); }
    bool hasPostBackwardHooks() const noexcept { return !post_backward_hooks.empty(); }

    Tensor applyForwardHooks(Tensor tensor) const
    {
        const auto hooks_snapshot = forward_hooks;
        for (const auto &entry : hooks_snapshot)
        {
            if (!entry.callback)
            {
                continue;
            }
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
        const auto hooks_snapshot = pre_backward_hooks;
        for (const auto &entry : hooks_snapshot)
        {
            if (!entry.callback)
            {
                continue;
            }
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
        const auto hooks_snapshot = post_backward_hooks;
        for (const auto &entry : hooks_snapshot)
        {
            if (!entry.callback)
            {
                continue;
            }
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
        std::weak_ptr<bool> lifetime_token;
        bool has_lifetime_token = false;

    public:
        Scoped_Hook() noexcept = default;

        Scoped_Hook(ILayer *target_layer, uint64_t id, Hook_Type type) noexcept
            : layer(target_layer), hook_id(id), hook_type(type)
        {
        }

        Scoped_Hook(ILayer *target_layer, uint64_t id, Hook_Type type, std::weak_ptr<bool> token) noexcept
            : layer(target_layer), hook_id(id), hook_type(type), lifetime_token(std::move(token)), has_lifetime_token(true)
        {
        }

        ~Scoped_Hook()
        {
            release();
        }

        Scoped_Hook(const Scoped_Hook &) = delete;
        Scoped_Hook &operator=(const Scoped_Hook &) = delete;

        Scoped_Hook(Scoped_Hook &&other) noexcept
            : layer(other.layer),
              hook_id(other.hook_id),
              hook_type(other.hook_type),
              lifetime_token(std::move(other.lifetime_token)),
              has_lifetime_token(other.has_lifetime_token)
        {
            other.layer = nullptr;
            other.hook_id = 0;
            other.lifetime_token.reset();
            other.has_lifetime_token = false;
        }

        Scoped_Hook &operator=(Scoped_Hook &&other) noexcept
        {
            if (this != &other)
            {
                release();
                layer = other.layer;
                hook_id = other.hook_id;
                hook_type = other.hook_type;
                lifetime_token = std::move(other.lifetime_token);
                has_lifetime_token = other.has_lifetime_token;
                other.layer = nullptr;
                other.hook_id = 0;
                other.lifetime_token.reset();
                other.has_lifetime_token = false;
            }
            return *this;
        }

        void release()
        {
            if (layer && hook_id != 0)
            {
                if (!has_lifetime_token || !lifetime_token.expired())
                {
                    layer->removeHook(hook_id, hook_type);
                }
                layer = nullptr;
                hook_id = 0;
                lifetime_token.reset();
                has_lifetime_token = false;
            }
        }

        bool isActive() const noexcept
        {
            return layer != nullptr && hook_id != 0 && (!has_lifetime_token || !lifetime_token.expired());
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
        requires std::is_invocable_v<Callback_T, const Tensor &> &&
                 (std::is_void_v<std::invoke_result_t<Callback_T, const Tensor &>> ||
                  std::convertible_to<std::invoke_result_t<Callback_T, const Tensor &>, Tensor>)
    Scoped_Hook registerForwardHook(Callback_T &&callback)
    {
        using Ret_T = std::invoke_result_t<Callback_T, const Tensor &>;
        if constexpr (std::is_constructible_v<bool, const std::remove_reference_t<Callback_T> &>)
        {
            if (!callback)
            {
                throw std::invalid_argument("ILayer::registerForwardHook: callback must not be empty");
            }
        }
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
            if (!fn_target)
            {
                throw std::invalid_argument("ILayer::registerForwardHook: callback must not be empty");
            }
        }
        const uint64_t id = nextHookId();
        forward_hooks.push_back(Hook_Entry{id, std::move(fn_target)});
        return Scoped_Hook(this, id, Hook_Type::FORWARD, hook_lifetime_token);
    }

    template <typename Callback_T>
        requires std::is_invocable_v<Callback_T, const Tensor &> &&
                 (std::is_void_v<std::invoke_result_t<Callback_T, const Tensor &>> ||
                  std::convertible_to<std::invoke_result_t<Callback_T, const Tensor &>, Tensor>)
    Scoped_Hook registerBackwardHook(Callback_T &&callback, Hook_Type type = Hook_Type::PRE_BACKWARD)
    {
        if (type != Hook_Type::PRE_BACKWARD && type != Hook_Type::POST_BACKWARD)
        {
            throw std::invalid_argument("ILayer::registerBackwardHook: type must be PRE_BACKWARD or POST_BACKWARD");
        }
        using Ret_T = std::invoke_result_t<Callback_T, const Tensor &>;
        if constexpr (std::is_constructible_v<bool, const std::remove_reference_t<Callback_T> &>)
        {
            if (!callback)
            {
                throw std::invalid_argument("ILayer::registerBackwardHook: callback must not be empty");
            }
        }
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
            if (!fn_target)
            {
                throw std::invalid_argument("ILayer::registerBackwardHook: callback must not be empty");
            }
        }
        const uint64_t id = nextHookId();
        if (type == Hook_Type::PRE_BACKWARD)
        {
            pre_backward_hooks.push_back(Hook_Entry{id, std::move(fn_target)});
        }
        else
        {
            post_backward_hooks.push_back(Hook_Entry{id, std::move(fn_target)});
        }
        return Scoped_Hook(this, id, type, hook_lifetime_token);
    }
};

using Hook_Fn = ILayer::Hook_Fn;
using Void_Hook_Fn = ILayer::Void_Hook_Fn;
using Scoped_Hook = ILayer::Scoped_Hook;

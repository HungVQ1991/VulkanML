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
    LAYER_TYPE_END
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

        std::size_t sample_size = std::min<std::size_t>(4, host_data.size());
        std::string sample_string;
        for (std::size_t i = 0; i < sample_size; ++i)
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

    virtual std::function<float(std::mt19937&)> getPopulationParameterInitializer(std::size_t param_index) const
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
    virtual std::vector<Shape> getPopulationParameterDims() const { return {}; }
    virtual std::vector<bool> getPopulationParameterIsEvolvable() const { return std::vector<bool>(getPopulationParameterDims().size(), true); }
    virtual std::vector<float> getPopulationParameter(std::size_t param_index) const { return {}; }
    virtual Execution_Target getExecutionTarget() const = 0;
    virtual Layer_Type getLayerType() const = 0;
    virtual bool isAccumulated() const noexcept { return is_accumulated; }

    virtual void setPopulationParameter(std::size_t param_index, std::vector<float> flat_data) {}
    virtual void setExecutionTarget(Execution_Target _execution_target) = 0;
    virtual void setTarget(Execution_Target _execution_target) { setExecutionTarget(_execution_target); }
    virtual void setAccumulated(bool _is_accumulated) noexcept { is_accumulated = _is_accumulated; }
    virtual void setTrainingMode(bool _is_training) {}
};
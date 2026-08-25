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

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/matrix.h"

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
    LAYER_TYPE_END
};

class ILayer
{
public:
    void logBufferAddress(Matrix *_target_matrix, const std::string &_matrix_name) const
    {
        return;
        std::string layer_name = std::string(magic_enum::enum_name<Layer_Type>(getLayerType()));
        if (!_target_matrix)
        {
            Logger::logMessage(std::format("{}::logBufferAddress: Target matrix is null", layer_name),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LAYER_INSPECTION);
            return;
        }

        std::shared_ptr<gpu::vector> gpu_vector = nullptr;
        auto storage_handle = _target_matrix->getStorage();
        if (std::holds_alternative<std::shared_ptr<gpu::vector>>(storage_handle))
        {
            gpu_vector = std::get<std::shared_ptr<gpu::vector>>(storage_handle);
        }

        if (!gpu_vector || gpu_vector->isEmpty())
        {
            Logger::logMessage(std::format("{}::logBufferAddress: GPU storage of target is empty or invalid", layer_name),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::LAYER_INSPECTION);
            return;
        }

        Logger::logMessage(std::format("{}::logBufferAddress: Buffer info at {}: Address: {:p}, Size: {}",
                                       layer_name,
                                       _matrix_name,
                                       static_cast<const void *>(gpu_vector->getBuffer()),
                                       gpu_vector->getSize()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LAYER_INSPECTION);

        const auto &host_data = _target_matrix->getData();
        if (host_data.empty())
        {
            Logger::logMessage(std::format("{}::logBufferAddress: {}: Data of target is empty", layer_name, _matrix_name),
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

        Logger::logMessage(std::format("{}::inspectGradient: {:<18}| Shape: {:>4}x{:<5} | ||G||: {:.6e} | Top: [{}]",
                                       layer_name,
                                       _matrix_name,
                                       _target_matrix->getRows(),
                                       _target_matrix->getColumns(),
                                       norm_value,
                                       sample_string),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LAYER_INSPECTION);
    }

    virtual ~ILayer() noexcept = default;

    virtual Matrix forward(const Matrix &_input_matrix) = 0;
    virtual Matrix backward(const Matrix &_output_gradient) = 0;

    [[nodiscard]] virtual Matrix getWeights() const { return Matrix(0, 0); }
    [[nodiscard]] virtual Matrix getBiases() const { return Matrix(0, 0); }
    [[nodiscard]] virtual Matrix getWeightsGradient() { return Matrix(0, 0); }
    [[nodiscard]] virtual Matrix getInput() { return Matrix(0, 0); }
    [[nodiscard]] virtual Matrix getOutput() { return Matrix(0, 0); }

    [[nodiscard]] virtual bool hasParameters() const { return false; }
    virtual void resetGradient() {}
    virtual void resetGradients() { resetGradient(); }
    virtual void setTrainingMode(bool _is_training) {}

    virtual std::vector<std::pair<Matrix *, Matrix *>> getParametersAndGradients() { return {}; }
    virtual std::vector<std::pair<Matrix *, Matrix *>> getParamsAndGrads() { return getParametersAndGradients(); }

    [[nodiscard]] virtual Layer_Type getLayerType() const = 0;

    virtual void saveConfiguration(std::ofstream &_output_file_stream) const = 0;
    virtual void saveConfig(std::ofstream &_output_file_stream) const { saveConfiguration(_output_file_stream); }

    virtual void saveInference(std::ofstream &_output_file_stream) const = 0;
    virtual void loadInference(std::ifstream &_input_file_stream) = 0;
    virtual void saveCheckpoint(std::ofstream &_output_file_stream) const = 0;
    virtual void loadCheckpoint(std::ifstream &_input_file_stream) = 0;

    virtual void setExecutionTarget(Execution_Target _execution_target) = 0;
    virtual void setTarget(Execution_Target _execution_target) { setExecutionTarget(_execution_target); }
};
#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "engine/gpu_vector.h"
#include "helper/logger.h"

using Storage_Handle = std::variant<std::reference_wrapper<const std::vector<float>>, std::shared_ptr<gpu::vector>>;
using Mutable_Storage_Handle = std::variant<std::reference_wrapper<std::vector<float>>, std::shared_ptr<gpu::vector>>;

class Impl
{
public:
    virtual ~Impl() noexcept = default;

     virtual Storage_Handle getStorage() const = 0;
     virtual Mutable_Storage_Handle getStorage() = 0;
     virtual std::size_t getRows() const noexcept = 0;
     virtual std::size_t getColumns() const noexcept = 0;
     virtual std::size_t getCols() const noexcept { return getColumns(); }
     virtual const std::vector<float> &getData() const noexcept = 0;

    virtual void reshape(std::size_t _rows, std::size_t _columns) = 0;

    void validateSameDimensions(const Impl &_other_implementation) const
    {
        if (getRows() != _other_implementation.getRows() || getColumns() != _other_implementation.getColumns())
        {
            std::string error_message = std::format("Impl::validateSameDimensions: Matrix dimension mismatch: ({}x{}) vs ({}x{})",
                                                    getRows(), getColumns(), _other_implementation.getRows(), _other_implementation.getColumns());
            Logger::logMessage(error_message, Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument(error_message);
        }
    }

    void validateMatmulDimensions(const Impl &_other_implementation) const
    {
        if (getColumns() != _other_implementation.getRows())
        {
            std::string error_message = std::format("Impl::validateMatmulDimensions: Matrix matmul dimension mismatch: columns_a ({}) != rows_b ({})",
                                                    getColumns(), _other_implementation.getRows());
            Logger::logMessage(error_message, Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument(error_message);
        }
    }

    void validateSquare() const
    {
        if (getRows() != getColumns())
        {
            std::string error_message = std::format("Impl::validateSquare: Matrix is not square: ({}x{})",
                                                    getRows(), getColumns());
            Logger::logMessage(error_message, Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument(error_message);
        }
    }

     virtual std::shared_ptr<gpu::vector> getVector() { return {}; }

    virtual void matmul(const Impl &_other_implementation, Impl &_output_result) const = 0;
    virtual void matdiv(const Impl &_other_implementation, Impl &_output_result) const = 0;
    virtual void add(const Impl &_other_implementation, Impl &_output_result) const = 0;
    virtual void sub(const Impl &_other_implementation, Impl &_output_result) const = 0;
    virtual void mulScalar(float _scalar, Impl &_output_result) const = 0;
    virtual void divScalar(float _scalar, Impl &_output_result) const = 0;
    virtual void hadamardMul(const Impl &_other_implementation, Impl &_output_result) const = 0;
    virtual void hadamardDiv(const Impl &_other_implementation, Impl &_output_result) const = 0;
    virtual void transpose(Impl &_output_result) const = 0;
    virtual void inverse(Impl &_output_result) const = 0;
    virtual void normalize(Impl &_output_result) const = 0;
    virtual void relu(Impl &_output_result) const = 0;
    virtual void reluBackward(const Impl &_output_gradient, Impl &_input_gradient) const = 0;
    virtual void gelu(Impl &_output_result) const = 0;
    virtual void geluBackward(const Impl &_output_gradient, Impl &_input_gradient) const = 0;
    virtual void softmax(Impl &_output_result) const = 0;
    virtual void softmaxBackward(const Impl &_output_gradient, Impl &_input_gradient) const = 0;

    virtual void sgdUpdate(const Impl &_gradient_implementation, float _learning_rate, float _max_gradient = 0.0f) = 0;
    virtual void adamUpdate(
        const Impl &_gradient_implementation,
        const Impl &_first_moment_implementation,
        const Impl &_second_moment_implementation,
        float _learning_rate,
        float _beta1,
        float _beta2,
        float _epsilon,
        std::size_t _timestep,
        float _max_gradient = 1.0f) = 0;

    virtual void matmulAdd(const Impl &_other_implementation, const Impl &_bias_implementation, Impl &_output_result) const = 0;

    virtual void uploadData(const std::vector<float> &_host_data) = 0;

    virtual void conv2d(
        const Impl &_weights, const Impl &_biases, Impl &_output_result,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_channels, std::uint32_t _kernel_size,
        std::uint32_t _stride, std::uint32_t _padding) const = 0;

    virtual void conv2dBackwardInput(
        const Impl &_weights, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width, std::uint32_t _output_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const = 0;

    virtual void conv2dBackwardWeight(
        const Impl &_output_gradient, Impl &_weight_gradient, Impl &_bias_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width, std::uint32_t _output_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const = 0;

    virtual void maxpool2d(
        Impl &_output_result, Impl &_output_mask,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const = 0;

    virtual void maxpool2dBackward(
        const Impl &_mask, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const = 0;

    virtual void globalAvgPool2d(Impl &_output_result, std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels) const = 0;
    virtual void globalAvgPool2dBackward(Impl &_input_gradient, std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels) const = 0;

    virtual void batchNormForward(
        const Impl &_gamma, const Impl &_beta, Impl &_running_mean, Impl &_running_variance,
        Impl &_batch_mean, Impl &_batch_variance, Impl &_normalized_input, Impl &_output_result,
        float _epsilon, float _momentum, bool _is_training) const = 0;

    virtual void batchNormBackward(
        const Impl &_output_gradient, const Impl &_gamma, const Impl &_batch_variance, const Impl &_normalized_input,
        Impl &_gamma_gradient, Impl &_beta_gradient, Impl &_input_gradient, float _epsilon) const = 0;

    virtual void linearForward(const Impl &_weights, const Impl &_biases, Impl &_output_result) const = 0;
    virtual void linearBackwardInput(const Impl &_weights, Impl &_input_gradient) const = 0;
    virtual void linearBackwardWeightBias(const Impl &_output_gradient, Impl &_weight_gradient, Impl &_bias_gradient) const = 0;

    virtual void batchNorm2dForward(
        const Impl &_gamma, const Impl &_beta,
        Impl &_running_mean, Impl &_running_variance,
        Impl &_batch_mean, Impl &_batch_variance,
        Impl &_normalized_input, Impl &_output_result,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        float _epsilon, float _momentum, bool _is_training) const = 0;

    virtual void batchNorm2dBackward(
        const Impl &_gamma, const Impl &_batch_variance, const Impl &_normalized_input,
        Impl &_gamma_gradient, Impl &_beta_gradient, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels, float _epsilon) const = 0;

    virtual void cceLoss(const Impl &_target_implementation, Impl &_output_result, float _epsilon) const = 0;
    virtual void mseLoss(const Impl &_target_implementation, Impl &_output_result) const = 0;
    virtual void maeLoss(const Impl &_target_implementation, Impl &_output_result) const = 0;
    virtual void bceLoss(const Impl &_target_implementation, Impl &_output_result, float _epsilon) const = 0;

     virtual bool isEmpty() const noexcept = 0;
};
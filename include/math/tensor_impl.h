#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "shape.h"

using Storage_Handle = std::variant<std::reference_wrapper<const std::vector<float>>, std::shared_ptr<gpu::vector>>;
using Mutable_Storage_Handle = std::variant<std::reference_wrapper<std::vector<float>>, std::shared_ptr<gpu::vector>>;

class Tensor_Impl
{
protected:
    Shape shape;
    Stride strides;
    std::size_t byte_offset = 0;
    std::size_t total_elements = 0;

    void updateShapeAndStrides(Shape target_shape)
    {
        shape = target_shape;
        strides = shape.computeContiguousStrides();
        total_elements = shape.getTotalElements();
    }

public:
    virtual ~Tensor_Impl() noexcept = default;

    virtual Storage_Handle getStorage() const = 0;
    virtual Mutable_Storage_Handle getStorage() = 0;
    virtual const std::vector<float> &getData() const noexcept = 0;
    virtual void uploadData(const std::vector<float> &host_data) = 0;
    virtual bool isEmpty() const noexcept = 0;
    virtual std::shared_ptr<gpu::vector> getVector() { return {}; }

    bool isContiguous() const noexcept
    {
        std::size_t acc_stride = 1;
        for (std::size_t i = shape.getRank(); i > 0; --i)
        {
            if (shape[i - 1] == 1)
                continue;
            if (strides[i - 1] != acc_stride)
                return false;
            acc_stride *= shape[i - 1];
        }
        return true;
    }

    void validateSameDimensions(const Tensor_Impl &other) const
    {
        if (shape != other.shape)
        {
            Logger::logMessage(Input_Format{"Tensor_Impl::validateSameDimensions: Shape mismatch: {} vs {}",
                                            shape.toString(), other.shape.toString()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Tensor shape mismatch");
        }
    }

    void validateMatmulDimensions(const Tensor_Impl &other) const
    {
        if (getColumns() != other.getRows())
        {
            Logger::logMessage(Input_Format{"Tensor_Impl::validateMatmulDimensions: Mismatch: cols_a ({}) != rows_b ({})",
                                            getColumns(), other.getRows()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Matrix matmul dimension mismatch");
        }
    }

    void validateSquare() const
    {
        if (getRows() != getColumns())
        {
            Logger::logMessage(Input_Format{"Tensor_Impl::validateSquare: Non-square dimensions: ({}x{})",
                                            getRows(), getColumns()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Matrix is not square");
        }
    }

    virtual void reshape(std::size_t rows, std::size_t columns) = 0;
    virtual void reshape(Shape new_shape) = 0;
    virtual void permute(const std::vector<std::size_t> &axes_permutation, Tensor_Impl &output) const = 0;
    virtual void slice(std::size_t axis, std::size_t start, std::size_t length, Tensor_Impl &output) const = 0;
    virtual void contiguous(Tensor_Impl &output) const = 0;

    virtual void matmul(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void add(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void sub(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void mulScalar(float scalar, Tensor_Impl &output) const = 0;
    virtual void divScalar(float scalar, Tensor_Impl &output) const = 0;
    virtual void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void transpose(Tensor_Impl &output) const = 0;
    virtual void inverse(Tensor_Impl &output) const = 0;
    virtual void normalize(Tensor_Impl &output) const = 0;
    virtual void relu(Tensor_Impl &output) const = 0;
    virtual void reluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const = 0;
    virtual void gelu(Tensor_Impl &output) const = 0;
    virtual void geluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const = 0;
    virtual void softmax(Tensor_Impl &output) const = 0;
    virtual void softmaxBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const = 0;

    virtual void sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient = 0.0f) = 0;
    virtual void adamUpdate(const Tensor_Impl &gradient,
                            const Tensor_Impl &first_moment,
                            const Tensor_Impl &second_moment,
                            float learning_rate,
                            float beta1,
                            float beta2,
                            float epsilon,
                            std::size_t timestep,
                            float max_gradient = 1.0f) = 0;

    virtual void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const = 0;

    virtual void conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output,
                        std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                        std::uint32_t output_channels, std::uint32_t kernel_size,
                        std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient,
                                     std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                                     std::uint32_t output_height, std::uint32_t output_width, std::uint32_t output_channels,
                                     std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient,
                                      std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                                      std::uint32_t output_height, std::uint32_t output_width, std::uint32_t output_channels,
                                      std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask,
                           std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels,
                           std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient,
                                   std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels,
                                   std::uint32_t output_height, std::uint32_t output_width,
                                   std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const = 0;

    virtual void globalAvgPool2d(Tensor_Impl &output, std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels) const = 0;
    virtual void globalAvgPool2dBackward(Tensor_Impl &input_gradient, std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels) const = 0;

    virtual void batchNormForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                                  Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                                  Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                                  Tensor_Impl &normalized_input, Tensor_Impl &output,
                                  float epsilon, float momentum, bool is_training) const = 0;

    virtual void batchNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                                   Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, float epsilon) const = 0;

    virtual void linearForward(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const = 0;
    virtual void linearBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient) const = 0;
    virtual void linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient) const = 0;

    virtual void batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                                    Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                                    Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                                    Tensor_Impl &normalized_input, Tensor_Impl &output,
                                    std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                                    float epsilon, float momentum, bool is_training) const = 0;

    virtual void batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                                     Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient,
                                     std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels, float epsilon) const = 0;

    virtual void cceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const = 0;
    virtual void mseLoss(const Tensor_Impl &target, Tensor_Impl &output) const = 0;
    virtual void maeLoss(const Tensor_Impl &target, Tensor_Impl &output) const = 0;
    virtual void bceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const = 0;
    virtual void huberLoss(const Tensor_Impl &target, Tensor_Impl &output, float delta) const = 0;

    virtual void concatenateCollumns(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const = 0;
    virtual void splitCollumns(std::size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const = 0;
    virtual void splitRows(std::size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const = 0;
    
    const Shape &getShape() const noexcept { return shape; }
    const Stride &getStrides() const noexcept { return strides; }
    std::size_t getColumns() const noexcept
    {
        if (shape.getRank() == 0)
            return 0;
        if (shape.getRank() == 1)
            return shape[0];
        std::size_t cols = 1;
        for (std::size_t i = 1; i < shape.getRank(); ++i)
        {
            cols *= shape[i];
        }
        return cols;
    }
    std::size_t getRows() const noexcept
    {
        if (shape.getRank() == 0)
            return 0;
        if (shape.getRank() == 1)
            return 1;
        return shape[0];
    }
    std::size_t getTotalElements() const noexcept { return total_elements; }
    std::size_t getByteOffset() const noexcept { return byte_offset; }
    std::size_t getRank() const noexcept { return shape.getRank(); }
    std::size_t getCols() const noexcept { return getColumns(); }

    void setShape(const Shape &_shape) { updateShapeAndStrides(_shape); }
    void setStrides(const Stride &_strides) noexcept { strides = _strides; }
    void setTotalElements(std::size_t _total_elements) noexcept { total_elements = _total_elements; }
    void setByteOffset(std::size_t _byte_offset) noexcept { byte_offset = _byte_offset; }
};

using Impl = Tensor_Impl;
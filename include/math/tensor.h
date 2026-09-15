#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cpu_tensor_impl.h"
#include "gpu_tensor_impl.h"
#include "helper/logger.h"
#include "shape.h"
#include "tensor_impl.h"

enum class Execution_Target
{
    CPU,
    VULKAN_GPU
};

class Tensor
{
private:
    std::shared_ptr<Tensor_Impl> implementation;
    Execution_Target execution_target;

    static constexpr std::uint32_t TENSOR_MAGIC_HEADER = 0x7FFFFFFF;

public:
    explicit Tensor(Execution_Target target = Execution_Target::CPU)
        : execution_target(target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(0, 0);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(0, 0);
        }
    }

    Tensor(std::size_t rows, std::size_t columns, Execution_Target target = Execution_Target::CPU)
        : execution_target(target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(rows, columns);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(rows, columns);
        }
    }

    Tensor(std::size_t rows, std::size_t columns, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU)
        : execution_target(target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(rows, columns, host_data);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(rows, columns, host_data);
        }
    }

    Tensor(std::size_t rows, std::size_t columns, std::vector<float> &&host_data, Execution_Target target = Execution_Target::CPU)
        : execution_target(target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(rows, columns, std::move(host_data));
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(rows, columns, std::move(host_data));
        }
    }

    explicit Tensor(Shape shape, Execution_Target target = Execution_Target::CPU)
        : execution_target(target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(shape);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(shape);
        }
    }

    Tensor(Shape shape, const std::vector<float> &host_data, Execution_Target target = Execution_Target::CPU)
        : execution_target(target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Tensor_Impl>(shape, host_data);
        }
        else
        {
            implementation = std::make_shared<Gpu_Tensor_Impl>(shape, host_data);
        }
    }

    Tensor(std::initializer_list<std::size_t> shape_list, Execution_Target target = Execution_Target::CPU)
        : Tensor(Shape(shape_list), target)
    {
    }

    explicit Tensor(std::shared_ptr<Tensor_Impl> impl, Execution_Target target = Execution_Target::CPU)
        : implementation(std::move(impl)), execution_target(target)
    {
    }

    ~Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    void initializeShape(std::size_t rows, std::size_t columns)
    {
        if (implementation->getRows() == rows && implementation->getColumns() == columns)
        {
            return;
        }
        implementation->reshape(rows, columns);
    }

    void initShape(std::size_t rows, std::size_t columns)
    {
        initializeShape(rows, columns);
    }

    void reshape(Shape new_shape)
    {
        implementation->reshape(new_shape);
    }

    const Shape &getShape() const noexcept { return implementation->getShape(); }
    const Stride &getStrides() const noexcept { return implementation->getStrides(); }
    std::size_t getRank() const noexcept { return implementation->getRank(); }
    std::size_t getTotalElements() const noexcept { return implementation->getTotalElements(); }

    std::size_t getRows() const noexcept { return implementation->getRows(); }
    std::size_t getColumns() const noexcept { return implementation->getColumns(); }
    std::size_t getCols() const noexcept { return implementation->getColumns(); }

    Execution_Target getExecutionTarget() const noexcept { return execution_target; }
    Execution_Target getTarget() const noexcept { return execution_target; }
    std::shared_ptr<Tensor_Impl> getImplementation() const noexcept { return implementation; }

    void setExecutionTarget(Execution_Target new_target)
    {
        if (execution_target == new_target)
        {
            return;
        }
        Shape current_shape = getShape();
        std::vector<float> current_data = getData();
        *this = Tensor(current_shape, current_data, new_target);
        if (new_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
    }

    std::vector<float> getData() const { return implementation->getData(); }
    Storage_Handle getStorage() const
    {
        const Tensor_Impl &const_implementation = *implementation;
        return const_implementation.getStorage();
    }
    Mutable_Storage_Handle getStorage() { return implementation->getStorage(); }
    void uploadData(const std::vector<float> &host_data) { implementation->uploadData(host_data); }
    bool isEmpty() const noexcept { return implementation->isEmpty(); }

    Tensor permute(const std::vector<std::size_t> &axes_permutation) const
    {
        Tensor result(execution_target);
        implementation->permute(axes_permutation, *result.implementation);
        return result;
    }

    Tensor slice(std::size_t axis, std::size_t start, std::size_t length) const
    {
        Tensor result(execution_target);
        implementation->slice(axis, start, length, *result.implementation);
        return result;
    }

    Tensor contiguous() const
    {
        if (implementation->isContiguous() && implementation->getByteOffset() == 0)
        {
            return *this;
        }
        Tensor result(execution_target);
        implementation->contiguous(*result.implementation);
        return result;
    }

    void matmul(const Tensor &other, Tensor &output) const { implementation->matmul(*other.implementation, *output.implementation); }
    void matdiv(const Tensor &other, Tensor &output) const { implementation->matdiv(*other.implementation, *output.implementation); }
    void add(const Tensor &other, Tensor &output) const { implementation->add(*other.implementation, *output.implementation); }
    void sub(const Tensor &other, Tensor &output) const { implementation->sub(*other.implementation, *output.implementation); }
    void mulScalar(float scalar, Tensor &output) const { implementation->mulScalar(scalar, *output.implementation); }
    void divScalar(float scalar, Tensor &output) const { implementation->divScalar(scalar, *output.implementation); }
    void hadamardMul(const Tensor &other, Tensor &output) const { implementation->hadamardMul(*other.implementation, *output.implementation); }
    void hadamardDiv(const Tensor &other, Tensor &output) const { implementation->hadamardDiv(*other.implementation, *output.implementation); }
    void transpose(Tensor &output) const { implementation->transpose(*output.implementation); }
    void inverse(Tensor &output) const { implementation->inverse(*output.implementation); }
    void normalize(Tensor &output) const { implementation->normalize(*output.implementation); }
    void relu(Tensor &output) const { implementation->relu(*output.implementation); }
    void reluBackward(const Tensor &output_gradient, Tensor &input_gradient) const { implementation->reluBackward(*output_gradient.implementation, *input_gradient.implementation); }
    void gelu(Tensor &output) const { implementation->gelu(*output.implementation); }
    void geluBackward(const Tensor &output_gradient, Tensor &input_gradient) const { implementation->geluBackward(*output_gradient.implementation, *input_gradient.implementation); }
    void softmax(Tensor &output) const { implementation->softmax(*output.implementation); }
    void softmaxBackward(const Tensor &output_gradient, Tensor &input_gradient) const { implementation->softmaxBackward(*output_gradient.implementation, *input_gradient.implementation); }
    void matmulAdd(const Tensor &other, const Tensor &biases, Tensor &output) const { implementation->matmulAdd(*other.implementation, *biases.implementation, *output.implementation); }

    void sgdUpdate(const Tensor &gradient, float learning_rate, float max_gradient = 0.0F)
    {
        implementation->sgdUpdate(*gradient.implementation, learning_rate, max_gradient);
    }

    void adamUpdate(const Tensor &gradient,
                    const Tensor &first_moment,
                    const Tensor &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    std::size_t timestep,
                    float max_gradient = 1.0F)
    {
        implementation->adamUpdate(*gradient.implementation,
                                   *first_moment.implementation,
                                   *second_moment.implementation,
                                   learning_rate,
                                   beta1,
                                   beta2,
                                   epsilon,
                                   timestep,
                                   max_gradient);
    }

    void conv2d(const Tensor &weights, const Tensor &biases, Tensor &output,
                std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                std::uint32_t output_channels, std::uint32_t kernel_size,
                std::uint32_t stride, std::uint32_t padding) const
    {
        implementation->conv2d(*weights.implementation, *biases.implementation, *output.implementation,
                               input_height, input_width, input_channels, output_channels, kernel_size, stride, padding);
    }

    void conv2dBackwardInput(const Tensor &weights, Tensor &input_gradient,
                             std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                             std::uint32_t output_height, std::uint32_t output_width, std::uint32_t output_channels,
                             std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        implementation->conv2dBackwardInput(*weights.implementation, *input_gradient.implementation,
                                            input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
    }

    void conv2dBackwardWeight(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient,
                              std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                              std::uint32_t output_height, std::uint32_t output_width, std::uint32_t output_channels,
                              std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        implementation->conv2dBackwardWeight(*output_gradient.implementation, *weight_gradient.implementation, *bias_gradient.implementation,
                                             input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
    }

    void maxpool2d(Tensor &output, Tensor &output_mask,
                   std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels,
                   std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        implementation->maxpool2d(*output.implementation, *output_mask.implementation,
                                  input_height, input_width, channels, kernel_size, stride, padding);
    }

    void maxpool2dBackward(const Tensor &mask, Tensor &input_gradient,
                           std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels,
                           std::uint32_t output_height, std::uint32_t output_width,
                           std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const
    {
        implementation->maxpool2dBackward(*mask.implementation, *input_gradient.implementation,
                                          input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding);
    }

    void globalAvgPool2d(Tensor &output, std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels) const
    {
        implementation->globalAvgPool2d(*output.implementation, input_height, input_width, channels);
    }

    void globalAvgPool2dBackward(Tensor &input_gradient, std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels) const
    {
        implementation->globalAvgPool2dBackward(*input_gradient.implementation, input_height, input_width, channels);
    }

    void batchNormForward(const Tensor &gamma, const Tensor &beta,
                          Tensor &running_mean, Tensor &running_variance,
                          Tensor &batch_mean, Tensor &batch_variance,
                          Tensor &normalized_input, Tensor &output,
                          float epsilon, float momentum, bool is_training) const
    {
        implementation->batchNormForward(*gamma.implementation, *beta.implementation,
                                         *running_mean.implementation, *running_variance.implementation,
                                         *batch_mean.implementation, *batch_variance.implementation,
                                         *normalized_input.implementation, *output.implementation,
                                         epsilon, momentum, is_training);
    }

    void batchNormBackward(const Tensor &output_gradient, const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input,
                           Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient, float epsilon) const
    {
        implementation->batchNormBackward(*output_gradient.implementation, *gamma.implementation,
                                          *batch_variance.implementation, *normalized_input.implementation,
                                          *gamma_gradient.implementation, *beta_gradient.implementation,
                                          *input_gradient.implementation, epsilon);
    }

    void linearForward(const Tensor &weights, const Tensor &biases, Tensor &output) const
    {
        implementation->linearForward(*weights.implementation, *biases.implementation, *output.implementation);
    }

    void linearBackwardInput(const Tensor &weights, Tensor &input_gradient) const
    {
        implementation->linearBackwardInput(*weights.implementation, *input_gradient.implementation);
    }

    void linearBackwardWeightBias(const Tensor &output_gradient, Tensor &weight_gradient, Tensor &bias_gradient) const
    {
        implementation->linearBackwardWeightBias(*output_gradient.implementation, *weight_gradient.implementation, *bias_gradient.implementation);
    }

    void batchNorm2dForward(const Tensor &gamma, const Tensor &beta,
                            Tensor &running_mean, Tensor &running_variance,
                            Tensor &batch_mean, Tensor &batch_variance,
                            Tensor &normalized_input, Tensor &output,
                            std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const
    {
        implementation->batchNorm2dForward(*gamma.implementation, *beta.implementation,
                                           *running_mean.implementation, *running_variance.implementation,
                                           *batch_mean.implementation, *batch_variance.implementation,
                                           *normalized_input.implementation, *output.implementation,
                                           input_height, input_width, input_channels, epsilon, momentum, is_training);
    }

    void batchNorm2dBackward(const Tensor &gamma, const Tensor &batch_variance, const Tensor &normalized_input,
                             Tensor &gamma_gradient, Tensor &beta_gradient, Tensor &input_gradient,
                             std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels, float epsilon) const
    {
        implementation->batchNorm2dBackward(*gamma.implementation, *batch_variance.implementation,
                                            *normalized_input.implementation, *gamma_gradient.implementation,
                                            *beta_gradient.implementation, *input_gradient.implementation,
                                            input_height, input_width, input_channels, epsilon);
    }

    void cceLoss(const Tensor &target, Tensor &output, float epsilon = 1e-7F) const
    {
        implementation->cceLoss(*target.implementation, *output.implementation, epsilon);
    }

    void mseLoss(const Tensor &target, Tensor &output) const
    {
        implementation->mseLoss(*target.implementation, *output.implementation);
    }

    void maeLoss(const Tensor &target, Tensor &output) const
    {
        implementation->maeLoss(*target.implementation, *output.implementation);
    }

    void bceLoss(const Tensor &target, Tensor &output, float epsilon = 1e-7F) const
    {
        implementation->bceLoss(*target.implementation, *output.implementation, epsilon);
    }

    void huberLoss(const Tensor &target, Tensor &output, float delta = 1.0F) const
    {
        implementation->huberLoss(*target.implementation, *output.implementation, delta);
    }

    Tensor operator*(const Tensor &other) const
    {
        Tensor result(execution_target);
        matmul(other, result);
        return result;
    }

    Tensor operator/(const Tensor &other) const
    {
        Tensor result(execution_target);
        matdiv(other, result);
        return result;
    }

    Tensor operator+(const Tensor &other) const
    {
        Tensor result(execution_target);
        add(other, result);
        return result;
    }

    Tensor operator-(const Tensor &other) const
    {
        Tensor result(execution_target);
        sub(other, result);
        return result;
    }

    Tensor operator*(float scalar) const
    {
        Tensor result(execution_target);
        mulScalar(scalar, result);
        return result;
    }

    Tensor operator/(float scalar) const
    {
        Tensor result(execution_target);
        divScalar(scalar, result);
        return result;
    }

    Tensor hadamardMul(const Tensor &other) const
    {
        Tensor result(execution_target);
        hadamardMul(other, result);
        return result;
    }

    Tensor hadamardDiv(const Tensor &other) const
    {
        Tensor result(execution_target);
        hadamardDiv(other, result);
        return result;
    }

    Tensor transpose() const
    {
        Tensor result(execution_target);
        transpose(result);
        return result;
    }

    Tensor inverse() const
    {
        Tensor result(execution_target);
        inverse(result);
        return result;
    }

    Tensor normalize() const
    {
        Tensor result(execution_target);
        normalize(result);
        return result;
    }

    Tensor relu() const
    {
        Tensor result(execution_target);
        relu(result);
        return result;
    }

    Tensor reluBackward(const Tensor &output_gradient) const
    {
        Tensor result(execution_target);
        reluBackward(output_gradient, result);
        return result;
    }

    Tensor gelu() const
    {
        Tensor result(execution_target);
        gelu(result);
        return result;
    }

    Tensor geluBackward(const Tensor &output_gradient) const
    {
        Tensor result(execution_target);
        geluBackward(output_gradient, result);
        return result;
    }

    Tensor softmax() const
    {
        Tensor result(execution_target);
        softmax(result);
        return result;
    }

    Tensor softmaxBackward(const Tensor &output_gradient) const
    {
        Tensor result(execution_target);
        softmaxBackward(output_gradient, result);
        return result;
    }

    Tensor matmulAdd(const Tensor &other, const Tensor &biases) const
    {
        Tensor result(execution_target);
        matmulAdd(other, biases, result);
        return result;
    }

    void concatenateCollumns(const Tensor &other, Tensor &output) const
    {
        implementation->concatenateCollumns(*other.implementation, *output.implementation);
    }

    Tensor concatenateCollumns(const Tensor &other) const
    {
        Tensor result(execution_target);
        concatenateCollumns(other, result);
        return result;
    }

    void concatenateRows(const Tensor &other, Tensor &output) const
    {
        implementation->concatenateRows(*other.implementation, *output.implementation);
    }

    Tensor concatenateRows(const Tensor &other) const
    {
        Tensor result(execution_target);
        concatenateRows(other, result);
        return result;
    }

    void splitCollumns(std::size_t split_index, Tensor &result_left, Tensor &result_right) const
    {
        implementation->splitCollumns(split_index, *result_left.implementation, *result_right.implementation);
    }

    std::pair<Tensor, Tensor> splitCollumns(std::size_t split_index) const
    {
        Tensor result_left(execution_target);
        Tensor result_right(execution_target);
        splitCollumns(split_index, result_left, result_right);
        return {std::move(result_left), std::move(result_right)};
    }

    void splitRows(std::size_t split_index, Tensor &result_up, Tensor &result_down) const
    {
        implementation->splitRows(split_index, *result_up.implementation, *result_down.implementation);
    }

    std::pair<Tensor, Tensor> splitRows(std::size_t split_index) const
    {
        Tensor result_up(execution_target);
        Tensor result_down(execution_target);
        splitRows(split_index, result_up, result_down);
        return {std::move(result_up), std::move(result_down)};
    }

    float getScalar() const
    {
        const auto host_data = getData();
        float sum = 0.0F;
        for (float val : host_data)
        {
            sum += val;
        }
        return sum;
    }

    void print(std::size_t max_display_rows = 10, std::size_t max_display_columns = 10) const
    {
        const auto data_vector = getData();
        std::size_t print_rows = std::min(getRows(), max_display_rows);
        std::size_t print_cols = std::min(getColumns(), max_display_columns);
        std::cout << "Tensor " << getShape().toString() << ":\n";
        for (std::size_t r = 0; r < print_rows; ++r)
        {
            std::cout << "  [ ";
            for (std::size_t c = 0; c < print_cols; ++c)
            {
                std::cout << std::format("{:8.4f} ", data_vector[r * getColumns() + c]);
            }
            if (getColumns() > print_cols)
            {
                std::cout << "... ";
            }
            std::cout << "]\n";
        }
        if (getRows() > print_rows)
        {
            std::cout << "  ...\n";
        }
    }

    void saveMatrix(std::ofstream &output_file_stream) const
    {
        if (!output_file_stream.is_open())
        {
            throw std::runtime_error("Tensor::saveMatrix: Output stream is not open");
        }

        output_file_stream.write(reinterpret_cast<const char *>(&TENSOR_MAGIC_HEADER), sizeof(TENSOR_MAGIC_HEADER));

        std::uint32_t rank = static_cast<std::uint32_t>(getRank());
        output_file_stream.write(reinterpret_cast<const char *>(&rank), sizeof(rank));

        const auto &dims = getShape();
        for (std::size_t i = 0; i < rank; ++i)
        {
            std::uint32_t dim_val = static_cast<std::uint32_t>(dims[i]);
            output_file_stream.write(reinterpret_cast<const char *>(&dim_val), sizeof(dim_val));
        }

        std::vector<float> host_data = getData();
        output_file_stream.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
    }

    static Tensor loadMatrix(std::ifstream &input_file_stream, Execution_Target target = Execution_Target::CPU)
    {
        if (!input_file_stream.is_open())
        {
            throw std::runtime_error("Tensor::loadMatrix: Input stream is not open");
        }

        std::uint32_t first_header_field = 0;
        input_file_stream.read(reinterpret_cast<char *>(&first_header_field), sizeof(first_header_field));

        if (first_header_field == TENSOR_MAGIC_HEADER || first_header_field == 0xFFFFFFFF)
        {
            std::uint32_t rank = 0;
            input_file_stream.read(reinterpret_cast<char *>(&rank), sizeof(rank));

            std::vector<std::size_t> dims(rank);
            std::size_t total = 1;
            for (std::size_t i = 0; i < rank; ++i)
            {
                std::uint32_t dim_val = 0;
                input_file_stream.read(reinterpret_cast<char *>(&dim_val), sizeof(dim_val));
                dims[i] = static_cast<std::size_t>(dim_val);
                total *= dims[i];
            }

            std::vector<float> host_data(total);
            input_file_stream.read(reinterpret_cast<char *>(host_data.data()), static_cast<std::streamsize>(total * sizeof(float)));
            return Tensor(Shape(dims), std::move(host_data), target);
        }

        std::uint32_t rows_count = first_header_field;
        std::uint32_t columns_count = 0;
        input_file_stream.read(reinterpret_cast<char *>(&columns_count), sizeof(columns_count));
        std::vector<float> host_data(rows_count * columns_count);
        input_file_stream.read(reinterpret_cast<char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
        return Tensor(rows_count, columns_count, std::move(host_data), target);
    }

    Tensor clone() const
    {
        return Tensor(getShape(), getData(), execution_target);
    }

    void fill(float value)
    {
        std::vector<float> buffer(getTotalElements(), value);
        uploadData(buffer);
    }

    void zero()
    {
        fill(0.0f);
    }

    void saveTensor(std::ofstream &output_file_stream) const
    {
        saveMatrix(output_file_stream);
    }

    static Tensor loadTensor(std::ifstream &input_file_stream, Execution_Target target = Execution_Target::CPU)
    {
        return loadMatrix(input_file_stream, target);
    }

};

using Matrix = Tensor;
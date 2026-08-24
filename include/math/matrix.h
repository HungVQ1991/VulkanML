#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "cpu_matrix_impl.h"
#include "gpu_matrix_impl.h"
#include "helper/logger.h"
#include "impl.h"

enum class Execution_Target
{
    CPU,
    VULKAN_GPU
};

class Matrix
{
private:
    std::shared_ptr<Impl> implementation;
    Execution_Target execution_target;

public:
    explicit Matrix(Execution_Target _execution_target = Execution_Target::CPU)
        : execution_target(_execution_target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Matrix_Impl>(0, 0);
        }
        else
        {
            implementation = std::make_shared<Gpu_Matrix_Impl>(0, 0);
        }
    }

    Matrix(std::size_t _rows, std::size_t _columns, Execution_Target _execution_target = Execution_Target::CPU)
        : execution_target(_execution_target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Matrix_Impl>(_rows, _columns);
        }
        else
        {
            implementation = std::make_shared<Gpu_Matrix_Impl>(_rows, _columns);
        }
    }

    Matrix(std::size_t _rows, std::size_t _columns, const std::vector<float> &_host_data, Execution_Target _execution_target = Execution_Target::CPU)
        : execution_target(_execution_target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Matrix_Impl>(_rows, _columns, _host_data);
        }
        else
        {
            implementation = std::make_shared<Gpu_Matrix_Impl>(_rows, _columns, _host_data);
        }
    }

    Matrix(std::size_t _rows, std::size_t _columns, std::vector<float> &&_host_data, Execution_Target _execution_target = Execution_Target::CPU)
        : execution_target(_execution_target)
    {
        if (execution_target == Execution_Target::CPU)
        {
            implementation = std::make_shared<Cpu_Matrix_Impl>(_rows, _columns, std::move(_host_data));
        }
        else
        {
            implementation = std::make_shared<Gpu_Matrix_Impl>(_rows, _columns, std::move(_host_data));
        }
    }

    explicit Matrix(std::shared_ptr<Impl> _implementation, Execution_Target _execution_target = Execution_Target::CPU)
        : implementation(std::move(_implementation)), execution_target(_execution_target)
    {
    }

    ~Matrix() = default;
    Matrix(const Matrix &) = default;
    Matrix &operator=(const Matrix &) = default;
    Matrix(Matrix &&) noexcept = default;
    Matrix &operator=(Matrix &&) noexcept = default;

    void initializeShape(std::size_t _rows, std::size_t _columns)
    {
        if (implementation->getRows() == _rows && implementation->getColumns() == _columns)
        {
            return;
        }
        implementation->reshape(_rows, _columns);
    }

    void initShape(std::size_t _rows, std::size_t _columns)
    {
        initializeShape(_rows, _columns);
    }

    [[nodiscard]] std::size_t getRows() const noexcept
    {
        return implementation->getRows();
    }

    [[nodiscard]] std::size_t getColumns() const noexcept
    {
        return implementation->getColumns();
    }

    [[nodiscard]] std::size_t getCols() const noexcept
    {
        return implementation->getColumns();
    }

    [[nodiscard]] Execution_Target getExecutionTarget() const noexcept
    {
        return execution_target;
    }

    [[nodiscard]] Execution_Target getTarget() const noexcept
    {
        return execution_target;
    }

    [[nodiscard]] std::shared_ptr<Impl> getImplementation() const noexcept
    {
        return implementation;
    }

    void setExecutionTarget(Execution_Target _new_execution_target)
    {
        if (execution_target == _new_execution_target)
        {
            return;
        }
        std::size_t row_count = getRows();
        std::size_t column_count = getColumns();
        std::vector<float> current_data = getData();
        *this = Matrix(row_count, column_count, current_data, _new_execution_target);
        if (_new_execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
    }

    [[nodiscard]] std::vector<float> getData() const
    {
        return implementation->getData();
    }

    [[nodiscard]] Storage_Handle getStorage() const
    {
        const Impl &const_implementation = *implementation;
        return const_implementation.getStorage();
    }

    [[nodiscard]] Mutable_Storage_Handle getStorage()
    {
        return implementation->getStorage();
    }

    void uploadData(const std::vector<float> &_host_data)
    {
        implementation->uploadData(_host_data);
    }

    [[nodiscard]] bool isEmpty() const noexcept
    {
        return implementation->isEmpty();
    }

    void matmul(const Matrix &_other, Matrix &_output_result) const
    {
        implementation->matmul(*_other.implementation, *_output_result.implementation);
    }

    void matdiv(const Matrix &_other, Matrix &_output_result) const
    {
        implementation->matdiv(*_other.implementation, *_output_result.implementation);
    }

    void add(const Matrix &_other, Matrix &_output_result) const
    {
        implementation->add(*_other.implementation, *_output_result.implementation);
    }

    void sub(const Matrix &_other, Matrix &_output_result) const
    {
        implementation->sub(*_other.implementation, *_output_result.implementation);
    }

    void mulScalar(float _scalar, Matrix &_output_result) const
    {
        implementation->mulScalar(_scalar, *_output_result.implementation);
    }

    void divScalar(float _scalar, Matrix &_output_result) const
    {
        implementation->divScalar(_scalar, *_output_result.implementation);
    }

    void hadamardMul(const Matrix &_other, Matrix &_output_result) const
    {
        implementation->hadamardMul(*_other.implementation, *_output_result.implementation);
    }

    void hadamardDiv(const Matrix &_other, Matrix &_output_result) const
    {
        implementation->hadamardDiv(*_other.implementation, *_output_result.implementation);
    }

    void transpose(Matrix &_output_result) const
    {
        implementation->transpose(*_output_result.implementation);
    }

    void inverse(Matrix &_output_result) const
    {
        implementation->inverse(*_output_result.implementation);
    }

    void normalize(Matrix &_output_result) const
    {
        implementation->normalize(*_output_result.implementation);
    }

    void relu(Matrix &_output_result) const
    {
        implementation->relu(*_output_result.implementation);
    }

    void reluBackward(const Matrix &_output_gradient, Matrix &_input_gradient) const
    {
        implementation->reluBackward(*_output_gradient.implementation, *_input_gradient.implementation);
    }

    void gelu(Matrix &_output_result) const
    {
        implementation->gelu(*_output_result.implementation);
    }

    void geluBackward(const Matrix &_output_gradient, Matrix &_input_gradient) const
    {
        implementation->geluBackward(*_output_gradient.implementation, *_input_gradient.implementation);
    }

    void softmax(Matrix &_output_result) const
    {
        implementation->softmax(*_output_result.implementation);
    }

    void softmaxBackward(const Matrix &_output_gradient, Matrix &_input_gradient) const
    {
        implementation->softmaxBackward(*_output_gradient.implementation, *_input_gradient.implementation);
    }

    void matmulAdd(const Matrix &_other, const Matrix &_biases, Matrix &_output_result) const
    {
        implementation->matmulAdd(*_other.implementation, *_biases.implementation, *_output_result.implementation);
    }

    void sgdUpdate(const Matrix &_gradient, float _learning_rate, float _max_gradient = 0.0f)
    {
        implementation->sgdUpdate(*_gradient.implementation, _learning_rate, _max_gradient);
    }

    void adamUpdate(const Matrix &_gradient,
                    const Matrix &_first_moment_matrix,
                    const Matrix &_second_moment_matrix,
                    float _learning_rate,
                    float _beta1,
                    float _beta2,
                    float _epsilon,
                    std::size_t _timestep,
                    float _max_gradient = 1.0f)
    {
        implementation->adamUpdate(*_gradient.implementation,
                                   *_first_moment_matrix.implementation,
                                   *_second_moment_matrix.implementation,
                                   _learning_rate,
                                   _beta1,
                                   _beta2,
                                   _epsilon,
                                   _timestep,
                                   _max_gradient);
    }

    void conv2d(const Matrix &_weights,
                const Matrix &_biases,
                Matrix &_output_result,
                std::uint32_t _input_height,
                std::uint32_t _input_width,
                std::uint32_t _input_channels,
                std::uint32_t _output_channels,
                std::uint32_t _kernel_size,
                std::uint32_t _stride,
                std::uint32_t _padding) const
    {
        implementation->conv2d(*_weights.implementation,
                               *_biases.implementation,
                               *_output_result.implementation,
                               _input_height,
                               _input_width,
                               _input_channels,
                               _output_channels,
                               _kernel_size,
                               _stride,
                               _padding);
    }

    void conv2dBackwardInput(const Matrix &_weights,
                             Matrix &_input_gradient,
                             std::uint32_t _input_height,
                             std::uint32_t _input_width,
                             std::uint32_t _input_channels,
                             std::uint32_t _output_height,
                             std::uint32_t _output_width,
                             std::uint32_t _output_channels,
                             std::uint32_t _kernel_size,
                             std::uint32_t _stride,
                             std::uint32_t _padding) const
    {
        implementation->conv2dBackwardInput(*_weights.implementation,
                                            *_input_gradient.implementation,
                                            _input_height,
                                            _input_width,
                                            _input_channels,
                                            _output_height,
                                            _output_width,
                                            _output_channels,
                                            _kernel_size,
                                            _stride,
                                            _padding);
    }

    void conv2dBackwardWeight(const Matrix &_output_gradient,
                              Matrix &_weight_gradient,
                              Matrix &_bias_gradient,
                              std::uint32_t _input_height,
                              std::uint32_t _input_width,
                              std::uint32_t _input_channels,
                              std::uint32_t _output_height,
                              std::uint32_t _output_width,
                              std::uint32_t _output_channels,
                              std::uint32_t _kernel_size,
                              std::uint32_t _stride,
                              std::uint32_t _padding) const
    {
        implementation->conv2dBackwardWeight(*_output_gradient.implementation,
                                             *_weight_gradient.implementation,
                                             *_bias_gradient.implementation,
                                             _input_height,
                                             _input_width,
                                             _input_channels,
                                             _output_height,
                                             _output_width,
                                             _output_channels,
                                             _kernel_size,
                                             _stride,
                                             _padding);
    }

    void maxpool2d(Matrix &_output_result,
                   Matrix &_output_mask,
                   std::uint32_t _input_height,
                   std::uint32_t _input_width,
                   std::uint32_t _channels,
                   std::uint32_t _kernel_size,
                   std::uint32_t _stride,
                   std::uint32_t _padding) const
    {
        implementation->maxpool2d(*_output_result.implementation,
                                  *_output_mask.implementation,
                                  _input_height,
                                  _input_width,
                                  _channels,
                                  _kernel_size,
                                  _stride,
                                  _padding);
    }

    void maxpool2dBackward(const Matrix &_mask,
                           Matrix &_input_gradient,
                           std::uint32_t _input_height,
                           std::uint32_t _input_width,
                           std::uint32_t _channels,
                           std::uint32_t _output_height,
                           std::uint32_t _output_width,
                           std::uint32_t _kernel_size,
                           std::uint32_t _stride,
                           std::uint32_t _padding) const
    {
        implementation->maxpool2dBackward(*_mask.implementation,
                                          *_input_gradient.implementation,
                                          _input_height,
                                          _input_width,
                                          _channels,
                                          _output_height,
                                          _output_width,
                                          _kernel_size,
                                          _stride,
                                          _padding);
    }

    void globalAvgPool2d(Matrix &_output_result,
                         std::uint32_t _input_height,
                         std::uint32_t _input_width,
                         std::uint32_t _channels) const
    {
        implementation->globalAvgPool2d(*_output_result.implementation, _input_height, _input_width, _channels);
    }

    void globalAvgPool2dBackward(Matrix &_input_gradient,
                                 std::uint32_t _input_height,
                                 std::uint32_t _input_width,
                                 std::uint32_t _channels) const
    {
        implementation->globalAvgPool2dBackward(*_input_gradient.implementation, _input_height, _input_width, _channels);
    }

    void batchNormForward(const Matrix &_gamma,
                          const Matrix &_beta,
                          Matrix &_running_mean,
                          Matrix &_running_variance,
                          Matrix &_batch_mean,
                          Matrix &_batch_variance,
                          Matrix &_normalized_input,
                          Matrix &_output_result,
                          float _epsilon,
                          float _momentum,
                          bool _is_training) const
    {
        implementation->batchNormForward(*_gamma.implementation,
                                         *_beta.implementation,
                                         *_running_mean.implementation,
                                         *_running_variance.implementation,
                                         *_batch_mean.implementation,
                                         *_batch_variance.implementation,
                                         *_normalized_input.implementation,
                                         *_output_result.implementation,
                                         _epsilon,
                                         _momentum,
                                         _is_training);
    }

    void batchNormBackward(const Matrix &_output_gradient,
                           const Matrix &_gamma,
                           const Matrix &_batch_variance,
                           const Matrix &_normalized_input,
                           Matrix &_gamma_gradient,
                           Matrix &_beta_gradient,
                           Matrix &_input_gradient,
                           float _epsilon) const
    {
        implementation->batchNormBackward(*_output_gradient.implementation,
                                          *_gamma.implementation,
                                          *_batch_variance.implementation,
                                          *_normalized_input.implementation,
                                          *_gamma_gradient.implementation,
                                          *_beta_gradient.implementation,
                                          *_input_gradient.implementation,
                                          _epsilon);
    }

    void linearForward(const Matrix &_weights, const Matrix &_biases, Matrix &_output_result) const
    {
        implementation->linearForward(*_weights.implementation, *_biases.implementation, *_output_result.implementation);
    }

    void linearBackwardInput(const Matrix &_weights, Matrix &_input_gradient) const
    {
        implementation->linearBackwardInput(*_weights.implementation, *_input_gradient.implementation);
    }

    void linearBackwardWeightBias(const Matrix &_output_gradient, Matrix &_weight_gradient, Matrix &_bias_gradient) const
    {
        implementation->linearBackwardWeightBias(*_output_gradient.implementation, *_weight_gradient.implementation, *_bias_gradient.implementation);
    }

    void batchNorm2dForward(const Matrix &_gamma,
                            const Matrix &_beta,
                            Matrix &_running_mean,
                            Matrix &_running_variance,
                            Matrix &_batch_mean,
                            Matrix &_batch_variance,
                            Matrix &_normalized_input,
                            Matrix &_output_result,
                            std::uint32_t _input_height,
                            std::uint32_t _input_width,
                            std::uint32_t _input_channels,
                            float _epsilon,
                            float _momentum,
                            bool _is_training) const
    {
        implementation->batchNorm2dForward(*_gamma.implementation,
                                           *_beta.implementation,
                                           *_running_mean.implementation,
                                           *_running_variance.implementation,
                                           *_batch_mean.implementation,
                                           *_batch_variance.implementation,
                                           *_normalized_input.implementation,
                                           *_output_result.implementation,
                                           _input_height,
                                           _input_width,
                                           _input_channels,
                                           _epsilon,
                                           _momentum,
                                           _is_training);
    }

    void batchNorm2dBackward(const Matrix &_gamma,
                             const Matrix &_batch_variance,
                             const Matrix &_normalized_input,
                             Matrix &_gamma_gradient,
                             Matrix &_beta_gradient,
                             Matrix &_input_gradient,
                             std::uint32_t _input_height,
                             std::uint32_t _input_width,
                             std::uint32_t _input_channels,
                             float _epsilon) const
    {
        implementation->batchNorm2dBackward(*_gamma.implementation,
                                            *_batch_variance.implementation,
                                            *_normalized_input.implementation,
                                            *_gamma_gradient.implementation,
                                            *_beta_gradient.implementation,
                                            *_input_gradient.implementation,
                                            _input_height,
                                            _input_width,
                                            _input_channels,
                                            _epsilon);
    }

    void cceLoss(const Matrix &_target_matrix, Matrix &_output_result, float _epsilon = 1e-7f) const
    {
        implementation->cceLoss(*_target_matrix.implementation, *_output_result.implementation, _epsilon);
    }

    void mseLoss(const Matrix &_target_matrix, Matrix &_output_result) const
    {
        implementation->mseLoss(*_target_matrix.implementation, *_output_result.implementation);
    }

    void maeLoss(const Matrix &_target_matrix, Matrix &_output_result) const
    {
        implementation->maeLoss(*_target_matrix.implementation, *_output_result.implementation);
    }

    void bceLoss(const Matrix &_target_matrix, Matrix &_output_result, float _epsilon = 1e-7f) const
    {
        implementation->bceLoss(*_target_matrix.implementation, *_output_result.implementation, _epsilon);
    }

    [[nodiscard]] Matrix operator*(const Matrix &_other) const
    {
        Matrix result(execution_target);
        matmul(_other, result);
        return result;
    }

    [[nodiscard]] Matrix operator/(const Matrix &_other) const
    {
        Matrix result(execution_target);
        matdiv(_other, result);
        return result;
    }

    [[nodiscard]] Matrix operator+(const Matrix &_other) const
    {
        Matrix result(execution_target);
        add(_other, result);
        return result;
    }

    [[nodiscard]] Matrix operator-(const Matrix &_other) const
    {
        Matrix result(execution_target);
        sub(_other, result);
        return result;
    }

    [[nodiscard]] Matrix operator*(float _scalar) const
    {
        Matrix result(execution_target);
        mulScalar(_scalar, result);
        return result;
    }

    [[nodiscard]] Matrix operator/(float _scalar) const
    {
        Matrix result(execution_target);
        divScalar(_scalar, result);
        return result;
    }

    [[nodiscard]] Matrix hadamardMul(const Matrix &_other) const
    {
        Matrix result(execution_target);
        hadamardMul(_other, result);
        return result;
    }

    [[nodiscard]] Matrix hadamardDiv(const Matrix &_other) const
    {
        Matrix result(execution_target);
        hadamardDiv(_other, result);
        return result;
    }

    [[nodiscard]] Matrix transpose() const
    {
        Matrix result(execution_target);
        transpose(result);
        return result;
    }

    [[nodiscard]] Matrix inverse() const
    {
        Matrix result(execution_target);
        inverse(result);
        return result;
    }

    [[nodiscard]] Matrix normalize() const
    {
        Matrix result(execution_target);
        normalize(result);
        return result;
    }

    [[nodiscard]] Matrix relu() const
    {
        Matrix result(execution_target);
        relu(result);
        return result;
    }

    [[nodiscard]] Matrix reluBackward(const Matrix &_output_gradient) const
    {
        Matrix result(execution_target);
        reluBackward(_output_gradient, result);
        return result;
    }

    [[nodiscard]] Matrix gelu() const
    {
        Matrix result(execution_target);
        gelu(result);
        return result;
    }

    [[nodiscard]] Matrix geluBackward(const Matrix &_output_gradient) const
    {
        Matrix result(execution_target);
        geluBackward(_output_gradient, result);
        return result;
    }

    [[nodiscard]] Matrix softmax() const
    {
        Matrix result(execution_target);
        softmax(result);
        return result;
    }

    [[nodiscard]] Matrix softmaxBackward(const Matrix &_output_gradient) const
    {
        Matrix result(execution_target);
        softmaxBackward(_output_gradient, result);
        return result;
    }

    [[nodiscard]] Matrix matmulAdd(const Matrix &_other, const Matrix &_biases) const
    {
        Matrix result(execution_target);
        matmulAdd(_other, _biases, result);
        return result;
    }

    [[nodiscard]] float getScalar() const
    {
        const auto host_data = getData();
        float sum_value = 0.0f;
        for (float value : host_data)
        {
            sum_value += value;
        }
        return sum_value;
    }

    void print(std::size_t _max_display_rows = 10, std::size_t _max_display_columns = 10) const
    {
        const auto data_vector = getData();
        std::size_t print_rows = std::min(getRows(), _max_display_rows);
        std::size_t print_columns = std::min(getColumns(), _max_display_columns);
        std::cout << "Matrix (" << getRows() << "x" << getColumns() << "):\n";
        for (std::size_t r = 0; r < print_rows; ++r)
        {
            std::cout << "  [ ";
            for (std::size_t c = 0; c < print_columns; ++c)
            {
                std::cout << std::format("{:8.4f} ", data_vector[r * getColumns() + c]);
            }
            if (getColumns() > print_columns)
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

    void saveMatrix(std::ofstream &_output_file_stream) const   
    {
        if (!_output_file_stream.is_open())
        {
            Logger::logMessage("Matrix::saveMatrix: Output stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Output stream is not open");
        }
        std::uint32_t rows_count = static_cast<std::uint32_t>(implementation->getRows());
        std::uint32_t columns_count = static_cast<std::uint32_t>(implementation->getColumns());
        _output_file_stream.write(reinterpret_cast<const char *>(&rows_count), sizeof(rows_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&columns_count), sizeof(columns_count));
        std::vector<float> host_data = implementation->getData();
        _output_file_stream.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
    }

    [[nodiscard]] static Matrix loadMatrix(std::ifstream &_input_file_stream, Execution_Target _execution_target = Execution_Target::CPU)
    {
        if (!_input_file_stream.is_open())
        {
            Logger::logMessage("Matrix::loadMatrix: Input stream is not open",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::MODEL_SERIALIZATION);
            throw std::runtime_error("Input stream is not open");
        }
        std::uint32_t rows_count = 0;
        std::uint32_t columns_count = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&rows_count), sizeof(rows_count));
        _input_file_stream.read(reinterpret_cast<char *>(&columns_count), sizeof(columns_count));
        std::vector<float> host_data(rows_count * columns_count);
        _input_file_stream.read(reinterpret_cast<char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(float)));
        return Matrix(rows_count, columns_count, std::move(host_data), _execution_target);
    }
};
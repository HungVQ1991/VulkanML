#pragma once

#include <algorithm>
#include <cmath>
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

#include "helper/logger.h"
#include "impl.h"

class Cpu_Matrix_Impl : public Impl
{
private:
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<float> storage;

    static std::string formatDataSample(const std::vector<float> &_data, std::size_t _sample_limit = 5)
    {
        if (_data.empty())
        {
            return "[]";
        }
        std::string formatted_sample = "[";
        std::size_t count = std::min(_data.size(), _sample_limit);
        for (std::size_t i = 0; i < count; ++i)
        {
            formatted_sample += std::format("{:.4e}{}", _data[i], (i + 1 < count) ? ", " : "");
        }
        if (_data.size() > _sample_limit)
        {
            formatted_sample += std::format(", ... (total {})", _data.size());
        }
        formatted_sample += "]";
        return formatted_sample;
    }

public:
    Cpu_Matrix_Impl(std::size_t _rows, std::size_t _columns)
        : rows(_rows), columns(_columns), storage(_rows * _columns, 0.0f)
    {
    }

    Cpu_Matrix_Impl(std::size_t _rows, std::size_t _columns, const std::vector<float> &_host_data)
        : rows(_rows), columns(_columns), storage(_host_data)
    {
        if (storage.size() != rows * columns)
        {
            Logger::logMessage("Cpu_Matrix_Impl::Cpu_Matrix_Impl: Host data size mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
    }

    Cpu_Matrix_Impl(std::size_t _rows, std::size_t _columns, std::vector<float> &&_host_data)
        : rows(_rows), columns(_columns), storage(std::move(_host_data))
    {
        if (storage.size() != rows * columns)
        {
            Logger::logMessage("Cpu_Matrix_Impl::Cpu_Matrix_Impl: Host data size mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
    }

    ~Cpu_Matrix_Impl() noexcept override = default;

    [[nodiscard]] std::size_t getRows() const noexcept override
    {
        return rows;
    }

    [[nodiscard]] std::size_t getColumns() const noexcept override
    {
        return columns;
    }

    [[nodiscard]] std::size_t getCols() const noexcept override
    {
        return columns;
    }

    [[nodiscard]] const std::vector<float> &getData() const noexcept override
    {
        return storage;
    }

    [[nodiscard]] std::vector<float> &getData() noexcept
    {
        return storage;
    }

    [[nodiscard]] Storage_Handle getStorage() const override
    {
        return std::cref(storage);
    }

    [[nodiscard]] Mutable_Storage_Handle getStorage() override
    {
        return std::ref(storage);
    }

    [[nodiscard]] bool isEmpty() const noexcept override
    {
        return storage.empty();
    }

    void reshape(std::size_t _rows, std::size_t _columns) override
    {
        rows = _rows;
        columns = _columns;
        if (storage.size() != _rows * _columns)
        {
            storage.resize(_rows * _columns, 0.0f);
        }
    }

    void matmul(const Impl &_other_implementation, Impl &_output_result) const override
    {
        validateMatmulDimensions(_other_implementation);
        const auto &other_cpu = static_cast<const Cpu_Matrix_Impl &>(_other_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);

        std::size_t other_columns = other_cpu.getColumns();
        output_cpu.reshape(rows, other_columns);
        std::fill(output_cpu.storage.begin(), output_cpu.storage.end(), 0.0f);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t k = 0; k < columns; ++k)
            {
                float a_value = storage[i * columns + k];
                for (std::size_t j = 0; j < other_columns; ++j)
                {
                    output_cpu.storage[i * other_columns + j] += a_value * other_cpu.storage[k * other_columns + j];
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::matmul: output shape=({}x{}), result={}",
                                       rows, other_columns, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void matdiv(const Impl &_other_implementation, Impl &_output_result) const override
    {
        Cpu_Matrix_Impl temporary_inverse(0, 0);
        _other_implementation.inverse(temporary_inverse);
        matmul(temporary_inverse, _output_result);
    }

    void add(const Impl &_other_implementation, Impl &_output_result) const override
    {
        const auto &other_cpu = static_cast<const Cpu_Matrix_Impl &>(_other_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        std::size_t other_rows = other_cpu.getRows();
        std::size_t other_columns = other_cpu.getColumns();
        bool is_broadcast_mode = (other_rows == 1);

        if (rows == other_rows && columns == other_columns)
        {
            for (std::size_t i = 0; i < storage.size(); ++i)
            {
                output_cpu.storage[i] = storage[i] + other_cpu.storage[i];
            }
        }
        else if (is_broadcast_mode && columns == other_columns)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < columns; ++j)
                {
                    output_cpu.storage[i * columns + j] = storage[i * columns + j] + other_cpu.storage[j];
                }
            }
        }
        else
        {
            validateSameDimensions(_other_implementation);
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::add: output shape=({}x{}), broadcast={}, result={}",
                                       rows, columns, is_broadcast_mode, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void sub(const Impl &_other_implementation, Impl &_output_result) const override
    {
        const auto &other_cpu = static_cast<const Cpu_Matrix_Impl &>(_other_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        std::size_t other_rows = other_cpu.getRows();
        std::size_t other_columns = other_cpu.getColumns();
        bool is_broadcast_mode = (other_rows == 1);

        if (rows == other_rows && columns == other_columns)
        {
            for (std::size_t i = 0; i < storage.size(); ++i)
            {
                output_cpu.storage[i] = storage[i] - other_cpu.storage[i];
            }
        }
        else if (is_broadcast_mode && columns == other_columns)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < columns; ++j)
                {
                    output_cpu.storage[i * columns + j] = storage[i * columns + j] - other_cpu.storage[j];
                }
            }
        }
        else
        {
            validateSameDimensions(_other_implementation);
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::sub: output shape=({}x{}), broadcast={}, result={}",
                                       rows, columns, is_broadcast_mode, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void mulScalar(float _scalar, Impl &_output_result) const override
    {
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            output_cpu.storage[i] = storage[i] * _scalar;
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::mulScalar: scalar={}, elements={}, result={}",
                                       _scalar, storage.size(), formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void divScalar(float _scalar, Impl &_output_result) const override
    {
        if (std::abs(_scalar) < 1e-8f)
        {
            Logger::logMessage("Cpu_Matrix_Impl::divScalar: Division by zero",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);
        float inverse_scalar = 1.0f / _scalar;

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            output_cpu.storage[i] = storage[i] * inverse_scalar;
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::divScalar: scalar={}, elements={}, result={}",
                                       _scalar, storage.size(), formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void hadamardMul(const Impl &_other_implementation, Impl &_output_result) const override
    {
        validateSameDimensions(_other_implementation);
        const auto &other_cpu = static_cast<const Cpu_Matrix_Impl &>(_other_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            output_cpu.storage[i] = storage[i] * other_cpu.storage[i];
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::hadamardMul: elements={}, result={}",
                                       storage.size(), formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void hadamardDiv(const Impl &_other_implementation, Impl &_output_result) const override
    {
        validateSameDimensions(_other_implementation);
        const auto &other_cpu = static_cast<const Cpu_Matrix_Impl &>(_other_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            if (std::abs(other_cpu.storage[i]) < 1e-8f)
            {
                Logger::logMessage("Cpu_Matrix_Impl::hadamardDiv: Division by zero",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DENSE_COMPUTE);
                throw std::runtime_error("Division by zero in hadamardDiv");
            }
            output_cpu.storage[i] = storage[i] / other_cpu.storage[i];
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::hadamardDiv: elements={}, result={}",
                                       storage.size(), formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void transpose(Impl &_output_result) const override
    {
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(columns, rows);

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < columns; ++j)
            {
                output_cpu.storage[j * rows + i] = storage[i * columns + j];
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::transpose: ({}x{}) -> ({}x{}), result={}",
                                       rows, columns, columns, rows, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void inverse(Impl &_output_result) const override
    {
        validateSquare();
        std::size_t dimension_size = rows;
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(dimension_size, dimension_size);

        std::vector<float> augmented_matrix(dimension_size * 2 * dimension_size, 0.0f);
        for (std::size_t i = 0; i < dimension_size; ++i)
        {
            for (std::size_t j = 0; j < dimension_size; ++j)
            {
                augmented_matrix[i * (2 * dimension_size) + j] = storage[i * dimension_size + j];
            }
            augmented_matrix[i * (2 * dimension_size) + (dimension_size + i)] = 1.0f;
        }

        for (std::size_t i = 0; i < dimension_size; ++i)
        {
            std::size_t pivot_row = i;
            float max_pivot_value = std::abs(augmented_matrix[i * (2 * dimension_size) + i]);

            for (std::size_t k = i + 1; k < dimension_size; ++k)
            {
                float current_value = std::abs(augmented_matrix[k * (2 * dimension_size) + i]);
                if (current_value > max_pivot_value)
                {
                    max_pivot_value = current_value;
                    pivot_row = k;
                }
            }

            if (max_pivot_value < 1e-7f)
            {
                Logger::logMessage("Cpu_Matrix_Impl::inverse: Matrix is singular and cannot be inverted",
                                   Log_Level::LOG_ERROR,
                                   true,
                                   0,
                                   Log_Feature::DENSE_COMPUTE);
                throw std::runtime_error("Matrix is singular");
            }

            if (pivot_row != i)
            {
                for (std::size_t j = 0; j < 2 * dimension_size; ++j)
                {
                    std::swap(augmented_matrix[i * (2 * dimension_size) + j], augmented_matrix[pivot_row * (2 * dimension_size) + j]);
                }
            }

            float pivot = augmented_matrix[i * (2 * dimension_size) + i];
            for (std::size_t j = 0; j < 2 * dimension_size; ++j)
            {
                augmented_matrix[i * (2 * dimension_size) + j] /= pivot;
            }

            for (std::size_t k = 0; k < dimension_size; ++k)
            {
                if (k != i)
                {
                    float factor = augmented_matrix[k * (2 * dimension_size) + i];
                    for (std::size_t j = 0; j < 2 * dimension_size; ++j)
                    {
                        augmented_matrix[k * (2 * dimension_size) + j] -= factor * augmented_matrix[i * (2 * dimension_size) + j];
                    }
                }
            }
        }

        for (std::size_t i = 0; i < dimension_size; ++i)
        {
            for (std::size_t j = 0; j < dimension_size; ++j)
            {
                output_cpu.storage[i * dimension_size + j] = augmented_matrix[i * (2 * dimension_size) + (dimension_size + j)];
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::inverse: dimension={}, result={}",
                                       dimension_size, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void normalize(Impl &_output_result) const override
    {
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        float sum_of_squares = 0.0f;
        for (float value : storage)
        {
            sum_of_squares += value * value;
        }
        float norm_value = std::sqrt(sum_of_squares);

        if (norm_value < 1e-8f)
        {
            Logger::logMessage(std::format("Cpu_Matrix_Impl::normalize: Matrix norm near zero ({:.4e}), skipping normalization", norm_value),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::NORMALIZATION_COMPUTE);
            output_cpu.storage = storage;
            return;
        }

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            output_cpu.storage[i] = storage[i] / norm_value;
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::normalize: norm={:.4e}, result={}",
                                       norm_value, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE);
    }

    void relu(Impl &_output_result) const override
    {
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            output_cpu.storage[i] = std::max(0.0f, storage[i]);
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::relu: elements={}, result={}",
                                       storage.size(), formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE);
    }

    void reluBackward(const Impl &_output_gradient, Impl &_input_gradient) const override
    {
        validateSameDimensions(_output_gradient);
        const auto &grad_cpu = static_cast<const Cpu_Matrix_Impl &>(_output_gradient);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);
        input_gradient_cpu.reshape(rows, columns);

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            input_gradient_cpu.storage[i] = (storage[i] > 0.0f) ? grad_cpu.storage[i] : 0.0f;
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::reluBackward: elements={}, gradient_result={}",
                                       storage.size(), formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void gelu(Impl &_output_result) const override
    {
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);
        constexpr float APPROXIMATION_ALPHA = 0.7978845608028654f;
        constexpr float APPROXIMATION_BETA = 0.044715f;

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            float x_value = storage[i];
            float x_cubed = x_value * x_value * x_value;
            float tanh_inner = std::tanh(APPROXIMATION_ALPHA * (x_value + APPROXIMATION_BETA * x_cubed));
            output_cpu.storage[i] = 0.5f * x_value * (1.0f + tanh_inner);
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::gelu: elements={}, result={}",
                                       storage.size(), formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE);
    }

    void geluBackward(const Impl &_output_gradient, Impl &_input_gradient) const override
    {
        validateSameDimensions(_output_gradient);
        const auto &grad_cpu = static_cast<const Cpu_Matrix_Impl &>(_output_gradient);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);
        input_gradient_cpu.reshape(rows, columns);

        constexpr float APPROXIMATION_ALPHA = 0.7978845608028654f;
        constexpr float APPROXIMATION_BETA = 0.044715f;

        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            float x_value = storage[i];
            float x_squared = x_value * x_value;
            float x_cubed = x_squared * x_value;
            float tanh_inner = std::tanh(APPROXIMATION_ALPHA * (x_value + APPROXIMATION_BETA * x_cubed));
            float sech_squared = 1.0f - tanh_inner * tanh_inner;
            float derivative = 0.5f * (1.0f + tanh_inner) + 0.5f * x_value * sech_squared * APPROXIMATION_ALPHA * (1.0f + 3.0f * APPROXIMATION_BETA * x_squared);
            input_gradient_cpu.storage[i] = grad_cpu.storage[i] * derivative;
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::geluBackward: elements={}, gradient_result={}",
                                       storage.size(), formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void softmax(Impl &_output_result) const override
    {
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(rows, columns);

        for (std::size_t r = 0; r < rows; ++r)
        {
            float max_value = storage[r * columns];
            for (std::size_t c = 1; c < columns; ++c)
            {
                max_value = std::max(max_value, storage[r * columns + c]);
            }
            float exponential_sum = 0.0f;
            for (std::size_t c = 0; c < columns; ++c)
            {
                float exp_val = std::exp(storage[r * columns + c] - max_value);
                output_cpu.storage[r * columns + c] = exp_val;
                exponential_sum += exp_val;
            }
            for (std::size_t c = 0; c < columns; ++c)
            {
                output_cpu.storage[r * columns + c] /= exponential_sum;
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::softmax: shape=({}x{}), result={}",
                                       rows, columns, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE);
    }

    void softmaxBackward(const Impl &_output_gradient, Impl &_input_gradient) const override
    {
        const auto &grad_cpu = static_cast<const Cpu_Matrix_Impl &>(_output_gradient);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);
        input_gradient_cpu.reshape(rows, columns);

        for (std::size_t r = 0; r < rows; ++r)
        {
            for (std::size_t i = 0; i < columns; ++i)
            {
                float accumulated_value = 0.0f;
                for (std::size_t j = 0; j < columns; ++j)
                {
                    float delta = (i == j) ? 1.0f : 0.0f;
                    float jacobian_element = storage[r * columns + i] * (delta - storage[r * columns + j]);
                    accumulated_value += jacobian_element * grad_cpu.storage[r * columns + j];
                }
                input_gradient_cpu.storage[r * columns + i] = accumulated_value;
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::softmaxBackward: shape=({}x{}), gradient_result={}",
                                       rows, columns, formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void sgdUpdate(const Impl &_gradient_implementation, float _learning_rate, float _max_gradient = 0.0f) override
    {
        validateSameDimensions(_gradient_implementation);
        const auto &grad_cpu = static_cast<const Cpu_Matrix_Impl &>(_gradient_implementation);
        std::size_t total_elements = storage.size();

        if (_max_gradient > 0.0f)
        {
            for (std::size_t i = 0; i < total_elements; ++i)
            {
                float gradient_clipped = std::clamp(grad_cpu.storage[i], -_max_gradient, _max_gradient);
                storage[i] -= _learning_rate * gradient_clipped;
            }
        }
        else
        {
            for (std::size_t i = 0; i < total_elements; ++i)
            {
                storage[i] -= _learning_rate * grad_cpu.storage[i];
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::sgdUpdate: elements={}, lr={}, max_grad={}, updated_weights={}",
                                       total_elements, _learning_rate, _max_gradient, formatDataSample(storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    void adamUpdate(
        const Impl &_gradient_implementation,
        const Impl &_first_moment_implementation,
        const Impl &_second_moment_implementation,
        float _learning_rate,
        float _beta1,
        float _beta2,
        float _epsilon,
        std::size_t _timestep,
        float _max_gradient = 1.0f) override
    {
        validateSameDimensions(_gradient_implementation);
        validateSameDimensions(_first_moment_implementation);
        validateSameDimensions(_second_moment_implementation);

        const auto &grad_cpu = static_cast<const Cpu_Matrix_Impl &>(_gradient_implementation);
        auto &first_moment_cpu = static_cast<Cpu_Matrix_Impl &>(const_cast<Impl &>(_first_moment_implementation));
        auto &second_moment_cpu = static_cast<Cpu_Matrix_Impl &>(const_cast<Impl &>(_second_moment_implementation));

        std::size_t total_elements = storage.size();

        float bias_correction_first = 1.0f - std::pow(_beta1, static_cast<float>(_timestep));
        float bias_correction_second = 1.0f - std::pow(_beta2, static_cast<float>(_timestep));

        for (std::size_t i = 0; i < total_elements; ++i)
        {
            float gradient_value = (_max_gradient > 0.0f) ? std::clamp(grad_cpu.storage[i], -_max_gradient, _max_gradient) : grad_cpu.storage[i];
            first_moment_cpu.storage[i] = _beta1 * first_moment_cpu.storage[i] + (1.0f - _beta1) * gradient_value;
            second_moment_cpu.storage[i] = _beta2 * second_moment_cpu.storage[i] + (1.0f - _beta2) * (gradient_value * gradient_value);

            float corrected_first_moment = first_moment_cpu.storage[i] / bias_correction_first;
            float corrected_second_moment = second_moment_cpu.storage[i] / bias_correction_second;
            storage[i] -= _learning_rate * (corrected_first_moment / (std::sqrt(corrected_second_moment) + _epsilon));
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::adamUpdate: step={}, lr={}, updated_weights={}, first_moment={}, second_moment={}",
                                       _timestep, _learning_rate, formatDataSample(storage),
                                       formatDataSample(first_moment_cpu.storage), formatDataSample(second_moment_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);
    }

    void matmulAdd(const Impl &_other_implementation, const Impl &_bias_implementation, Impl &_output_result) const override
    {
        validateMatmulDimensions(_other_implementation);
        const auto &weights_cpu = static_cast<const Cpu_Matrix_Impl &>(_other_implementation);
        const auto &bias_cpu = static_cast<const Cpu_Matrix_Impl &>(_bias_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);

        std::size_t output_rows = rows;
        std::size_t output_columns = weights_cpu.columns;
        output_cpu.reshape(output_rows, output_columns);

        for (std::size_t i = 0; i < output_rows; ++i)
        {
            for (std::size_t j = 0; j < output_columns; ++j)
            {
                output_cpu.storage[i * output_columns + j] = (bias_cpu.rows == 1) ? bias_cpu.storage[j] : bias_cpu.storage[i * output_columns + j];
            }
            for (std::size_t k = 0; k < columns; ++k)
            {
                float input_value = storage[i * columns + k];
                for (std::size_t j = 0; j < output_columns; ++j)
                {
                    output_cpu.storage[i * output_columns + j] += input_value * weights_cpu.storage[k * output_columns + j];
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::matmulAdd: output shape=({}x{}), result={}",
                                       output_rows, output_columns, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);
    }

    void conv2d(
        const Impl &_weights, const Impl &_biases, Impl &_output_result,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_channels, std::uint32_t _kernel_size,
        std::uint32_t _stride, std::uint32_t _padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t output_height = (_input_height + 2 * _padding - _kernel_size) / _stride + 1;
        std::uint32_t output_width = (_input_width + 2 * _padding - _kernel_size) / _stride + 1;

        const auto &weights_data = _weights.getData();
        const auto &biases_data = _biases.getData();
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(batch_size, output_height * output_width * _output_channels);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t oh = 0; oh < output_height; ++oh)
            {
                for (std::uint32_t ow = 0; ow < output_width; ++ow)
                {
                    for (std::uint32_t oc = 0; oc < _output_channels; ++oc)
                    {
                        float sum = biases_data[oc];
                        for (std::uint32_t ky = 0; ky < _kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < _kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * _stride + ky) - static_cast<int>(_padding);
                                int iw = static_cast<int>(ow * _stride + kx) - static_cast<int>(_padding);
                                if (ih >= 0 && ih < static_cast<int>(_input_height) && iw >= 0 && iw < static_cast<int>(_input_width))
                                {
                                    for (std::uint32_t ic = 0; ic < _input_channels; ++ic)
                                    {
                                        std::size_t input_index = n * _input_height * _input_width * _input_channels + ih * _input_width * _input_channels + iw * _input_channels + ic;
                                        std::size_t weight_index = ky * _kernel_size * _input_channels * _output_channels + kx * _input_channels * _output_channels + ic * _output_channels + oc;
                                        sum += storage[input_index] * weights_data[weight_index];
                                    }
                                }
                            }
                        }
                        output_cpu.storage[n * output_height * output_width * _output_channels + oh * output_width * _output_channels + ow * _output_channels + oc] = sum;
                    }
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::conv2d: output=({}x{}x{}x{}), result={}",
                                       batch_size, output_height, output_width, _output_channels, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void conv2dBackwardInput(
        const Impl &_weights, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width, std::uint32_t _output_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        const auto &weights_data = _weights.getData();
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);
        input_gradient_cpu.reshape(batch_size, _input_height * _input_width * _input_channels);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t ih = 0; ih < _input_height; ++ih)
            {
                for (std::uint32_t iw = 0; iw < _input_width; ++iw)
                {
                    for (std::uint32_t ic = 0; ic < _input_channels; ++ic)
                    {
                        float sum = 0.0f;
                        for (std::uint32_t ky = 0; ky < _kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < _kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + _padding - ky);
                                int ow_calc = static_cast<int>(iw + _padding - kx);
                                if (oh_calc >= 0 && oh_calc % static_cast<int>(_stride) == 0 && ow_calc >= 0 && ow_calc % static_cast<int>(_stride) == 0)
                                {
                                    std::uint32_t oh = static_cast<std::uint32_t>(oh_calc) / _stride;
                                    std::uint32_t ow = static_cast<std::uint32_t>(ow_calc) / _stride;
                                    if (oh < _output_height && ow < _output_width)
                                    {
                                        for (std::uint32_t oc = 0; oc < _output_channels; ++oc)
                                        {
                                            std::size_t output_gradient_index = n * _output_height * _output_width * _output_channels + oh * _output_width * _output_channels + ow * _output_channels + oc;
                                            std::size_t weight_index = ky * _kernel_size * _input_channels * _output_channels + kx * _input_channels * _output_channels + ic * _output_channels + oc;
                                            sum += storage[output_gradient_index] * weights_data[weight_index];
                                        }
                                    }
                                }
                            }
                        }
                        input_gradient_cpu.storage[n * _input_height * _input_width * _input_channels + ih * _input_width * _input_channels + iw * _input_channels + ic] = sum;
                    }
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::conv2dBackwardInput: input_gradient=({}x{}x{}x{}), result={}",
                                       batch_size, _input_height, _input_width, _input_channels, formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void conv2dBackwardWeight(
        const Impl &_output_gradient, Impl &_weight_gradient, Impl &_bias_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width, std::uint32_t _output_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        const auto &output_gradient_data = _output_gradient.getData();
        auto &weight_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_weight_gradient);
        auto &bias_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_bias_gradient);

        weight_gradient_cpu.reshape(1, _kernel_size * _kernel_size * _input_channels * _output_channels);
        bias_gradient_cpu.reshape(1, _output_channels);
        std::fill(weight_gradient_cpu.storage.begin(), weight_gradient_cpu.storage.end(), 0.0f);
        std::fill(bias_gradient_cpu.storage.begin(), bias_gradient_cpu.storage.end(), 0.0f);

        for (std::uint32_t oc = 0; oc < _output_channels; ++oc)
        {
            for (std::uint32_t ic = 0; ic < _input_channels; ++ic)
            {
                for (std::uint32_t ky = 0; ky < _kernel_size; ++ky)
                {
                    for (std::uint32_t kx = 0; kx < _kernel_size; ++kx)
                    {
                        float weight_sum = 0.0f;
                        for (std::uint32_t n = 0; n < batch_size; ++n)
                        {
                            for (std::uint32_t oh = 0; oh < _output_height; ++oh)
                            {
                                for (std::uint32_t ow = 0; ow < _output_width; ++ow)
                                {
                                    std::size_t output_gradient_index = n * _output_height * _output_width * _output_channels + oh * _output_width * _output_channels + ow * _output_channels + oc;
                                    float grad_val = output_gradient_data[output_gradient_index];
                                    if (ic == 0 && ky == 0 && kx == 0)
                                    {
                                        bias_gradient_cpu.storage[oc] += grad_val;
                                    }
                                    int ih = static_cast<int>(oh * _stride + ky) - static_cast<int>(_padding);
                                    int iw = static_cast<int>(ow * _stride + kx) - static_cast<int>(_padding);
                                    if (ih >= 0 && ih < static_cast<int>(_input_height) && iw >= 0 && iw < static_cast<int>(_input_width))
                                    {
                                        std::size_t input_index = n * _input_height * _input_width * _input_channels + ih * _input_width * _input_channels + iw * _input_channels + ic;
                                        weight_sum += storage[input_index] * grad_val;
                                    }
                                }
                            }
                        }
                        std::size_t weight_index = ky * _kernel_size * _input_channels * _output_channels + kx * _input_channels * _output_channels + ic * _output_channels + oc;
                        weight_gradient_cpu.storage[weight_index] = weight_sum;
                    }
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::conv2dBackwardWeight: weight_grad={}, bias_grad={}",
                                       formatDataSample(weight_gradient_cpu.storage), formatDataSample(bias_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void maxpool2d(
        Impl &_output_result, Impl &_output_mask,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        auto &result_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        auto &mask_cpu = static_cast<Cpu_Matrix_Impl &>(_output_mask);

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t output_height = (_input_height + 2 * _padding - _kernel_size) / _stride + 1;
        std::uint32_t output_width = (_input_width + 2 * _padding - _kernel_size) / _stride + 1;

        result_cpu.reshape(batch_size, output_height * output_width * _channels);
        mask_cpu.reshape(batch_size, output_height * output_width * _channels);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t oh = 0; oh < output_height; ++oh)
            {
                for (std::uint32_t ow = 0; ow < output_width; ++ow)
                {
                    for (std::uint32_t c = 0; c < _channels; ++c)
                    {
                        float max_value = -3.402823466e+38f;
                        float max_index = 0.0f;
                        for (std::uint32_t ky = 0; ky < _kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < _kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * _stride + ky) - static_cast<int>(_padding);
                                int iw = static_cast<int>(ow * _stride + kx) - static_cast<int>(_padding);
                                if (ih >= 0 && ih < static_cast<int>(_input_height) && iw >= 0 && iw < static_cast<int>(_input_width))
                                {
                                    std::size_t input_index = n * _input_height * _input_width * _channels + ih * _input_width * _channels + iw * _channels + c;
                                    float val = storage[input_index];
                                    if (val > max_value)
                                    {
                                        max_value = val;
                                        max_index = static_cast<float>(input_index);
                                    }
                                }
                            }
                        }
                        std::size_t output_index = n * output_height * output_width * _channels + oh * output_width * _channels + ow * _channels + c;
                        result_cpu.storage[output_index] = max_value;
                        mask_cpu.storage[output_index] = max_index;
                    }
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::maxpool2d: output=({}x{}x{}x{}), result={}, mask={}",
                                       batch_size, output_height, output_width, _channels,
                                       formatDataSample(result_cpu.storage), formatDataSample(mask_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void maxpool2dBackward(
        const Impl &_mask, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels,
        std::uint32_t _output_height, std::uint32_t _output_width,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        const auto &mask_cpu = static_cast<const Cpu_Matrix_Impl &>(_mask);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        input_gradient_cpu.reshape(batch_size, _input_height * _input_width * _channels);
        std::fill(input_gradient_cpu.storage.begin(), input_gradient_cpu.storage.end(), 0.0f);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t ih = 0; ih < _input_height; ++ih)
            {
                for (std::uint32_t iw = 0; iw < _input_width; ++iw)
                {
                    for (std::uint32_t c = 0; c < _channels; ++c)
                    {
                        std::size_t input_index = n * _input_height * _input_width * _channels + ih * _input_width * _channels + iw * _channels + c;
                        float accumulated_gradient = 0.0f;
                        for (std::uint32_t ky = 0; ky < _kernel_size; ++ky)
                        {
                            for (std::uint32_t kx = 0; kx < _kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + _padding - ky);
                                int ow_calc = static_cast<int>(iw + _padding - kx);
                                if (oh_calc >= 0 && oh_calc % static_cast<int>(_stride) == 0 && ow_calc >= 0 && ow_calc % static_cast<int>(_stride) == 0)
                                {
                                    std::uint32_t oh = static_cast<std::uint32_t>(oh_calc) / _stride;
                                    std::uint32_t ow = static_cast<std::uint32_t>(ow_calc) / _stride;
                                    if (oh < _output_height && ow < _output_width)
                                    {
                                        std::size_t output_index = n * _output_height * _output_width * _channels + oh * _output_width * _channels + ow * _channels + c;
                                        if (static_cast<std::size_t>(mask_cpu.storage[output_index]) == input_index)
                                        {
                                            accumulated_gradient += storage[output_index];
                                        }
                                    }
                                }
                            }
                        }
                        input_gradient_cpu.storage[input_index] = accumulated_gradient;
                    }
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::maxpool2dBackward: gradient_result={}",
                                       formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void globalAvgPool2d(Impl &_output_result, std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(batch_size, _channels);
        float pooling_area = static_cast<float>(_input_height * _input_width);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t c = 0; c < _channels; ++c)
            {
                float sum = 0.0f;
                for (std::uint32_t h = 0; h < _input_height; ++h)
                {
                    for (std::uint32_t w = 0; w < _input_width; ++w)
                    {
                        std::size_t input_index = n * _input_height * _input_width * _channels + h * _input_width * _channels + w * _channels + c;
                        sum += storage[input_index];
                    }
                }
                output_cpu.storage[n * _channels + c] = sum / pooling_area;
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::globalAvgPool2d: result={}",
                                       formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void globalAvgPool2dBackward(Impl &_input_gradient, std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);
        input_gradient_cpu.reshape(batch_size, _input_height * _input_width * _channels);
        float pooling_area = static_cast<float>(_input_height * _input_width);

        for (std::uint32_t n = 0; n < batch_size; ++n)
        {
            for (std::uint32_t c = 0; c < _channels; ++c)
            {
                float scaled_gradient = storage[n * _channels + c] / pooling_area;
                for (std::uint32_t h = 0; h < _input_height; ++h)
                {
                    for (std::uint32_t w = 0; w < _input_width; ++w)
                    {
                        std::size_t input_index = n * _input_height * _input_width * _channels + h * _input_width * _channels + w * _channels + c;
                        input_gradient_cpu.storage[input_index] = scaled_gradient;
                    }
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::globalAvgPool2dBackward: gradient_result={}",
                                       formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void batchNormForward(
        const Impl &_gamma, const Impl &_beta,
        Impl &_running_mean, Impl &_running_variance,
        Impl &_batch_mean, Impl &_batch_variance,
        Impl &_normalized_input, Impl &_output_result,
        float _epsilon, float _momentum, bool _is_training) const override
    {
        std::size_t batch_count = rows;
        std::size_t feature_dimension = columns;

        const auto &gamma_cpu = static_cast<const Cpu_Matrix_Impl &>(_gamma);
        const auto &beta_cpu = static_cast<const Cpu_Matrix_Impl &>(_beta);
        auto &running_mean_cpu = static_cast<Cpu_Matrix_Impl &>(_running_mean);
        auto &running_variance_cpu = static_cast<Cpu_Matrix_Impl &>(_running_variance);
        auto &normalized_input_cpu = static_cast<Cpu_Matrix_Impl &>(_normalized_input);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);

        output_cpu.reshape(batch_count, feature_dimension);
        normalized_input_cpu.reshape(batch_count, feature_dimension);

        if (_is_training)
        {
            auto &batch_mean_cpu = static_cast<Cpu_Matrix_Impl &>(_batch_mean);
            auto &batch_variance_cpu = static_cast<Cpu_Matrix_Impl &>(_batch_variance);
            batch_mean_cpu.reshape(1, feature_dimension);
            batch_variance_cpu.reshape(1, feature_dimension);
            std::fill(batch_mean_cpu.storage.begin(), batch_mean_cpu.storage.end(), 0.0f);
            std::fill(batch_variance_cpu.storage.begin(), batch_variance_cpu.storage.end(), 0.0f);

            for (std::size_t i = 0; i < batch_count; ++i)
            {
                for (std::size_t j = 0; j < feature_dimension; ++j)
                {
                    batch_mean_cpu.storage[j] += storage[i * feature_dimension + j];
                }
            }
            float inverse_batch_count = 1.0f / static_cast<float>(batch_count);
            for (std::size_t j = 0; j < feature_dimension; ++j)
            {
                batch_mean_cpu.storage[j] *= inverse_batch_count;
            }

            for (std::size_t i = 0; i < batch_count; ++i)
            {
                for (std::size_t j = 0; j < feature_dimension; ++j)
                {
                    float difference = storage[i * feature_dimension + j] - batch_mean_cpu.storage[j];
                    batch_variance_cpu.storage[j] += difference * difference;
                }
            }
            for (std::size_t j = 0; j < feature_dimension; ++j)
            {
                batch_variance_cpu.storage[j] *= inverse_batch_count;
                running_mean_cpu.storage[j] = (1.0f - _momentum) * running_mean_cpu.storage[j] + _momentum * batch_mean_cpu.storage[j];
                running_variance_cpu.storage[j] = (1.0f - _momentum) * running_variance_cpu.storage[j] + _momentum * batch_variance_cpu.storage[j];
            }

            for (std::size_t i = 0; i < batch_count; ++i)
            {
                for (std::size_t j = 0; j < feature_dimension; ++j)
                {
                    float normalized_value = (storage[i * feature_dimension + j] - batch_mean_cpu.storage[j]) / std::sqrt(batch_variance_cpu.storage[j] + _epsilon);
                    normalized_input_cpu.storage[i * feature_dimension + j] = normalized_value;
                    output_cpu.storage[i * feature_dimension + j] = gamma_cpu.storage[j] * normalized_value + beta_cpu.storage[j];
                }
            }
        }
        else
        {
            for (std::size_t i = 0; i < batch_count; ++i)
            {
                for (std::size_t j = 0; j < feature_dimension; ++j)
                {
                    float normalized_value = (storage[i * feature_dimension + j] - running_mean_cpu.storage[j]) / std::sqrt(running_variance_cpu.storage[j] + _epsilon);
                    normalized_input_cpu.storage[i * feature_dimension + j] = normalized_value;
                    output_cpu.storage[i * feature_dimension + j] = gamma_cpu.storage[j] * normalized_value + beta_cpu.storage[j];
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::batchNormForward: is_training={}, output={}, running_mean={}, running_var={}",
                                       _is_training, formatDataSample(output_cpu.storage),
                                       formatDataSample(running_mean_cpu.storage), formatDataSample(running_variance_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void batchNormBackward(
        const Impl &_output_gradient, const Impl &_gamma, const Impl &_batch_variance, const Impl &_normalized_input,
        Impl &_gamma_gradient, Impl &_beta_gradient, Impl &_input_gradient, float _epsilon) const override
    {
        std::size_t batch_count = rows;
        std::size_t feature_dimension = columns;

        const auto &output_gradient_cpu = static_cast<const Cpu_Matrix_Impl &>(_output_gradient);
        const auto &gamma_cpu = static_cast<const Cpu_Matrix_Impl &>(_gamma);
        const auto &batch_variance_cpu = static_cast<const Cpu_Matrix_Impl &>(_batch_variance);
        const auto &normalized_input_cpu = static_cast<const Cpu_Matrix_Impl &>(_normalized_input);

        auto &gamma_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_gamma_gradient);
        auto &beta_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_beta_gradient);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);

        gamma_gradient_cpu.reshape(1, feature_dimension);
        beta_gradient_cpu.reshape(1, feature_dimension);
        input_gradient_cpu.reshape(batch_count, feature_dimension);

        std::fill(gamma_gradient_cpu.storage.begin(), gamma_gradient_cpu.storage.end(), 0.0f);
        std::fill(beta_gradient_cpu.storage.begin(), beta_gradient_cpu.storage.end(), 0.0f);

        for (std::size_t i = 0; i < batch_count; ++i)
        {
            for (std::size_t j = 0; j < feature_dimension; ++j)
            {
                std::size_t index = i * feature_dimension + j;
                float gradient_out = output_gradient_cpu.storage[index];
                gamma_gradient_cpu.storage[j] += gradient_out * normalized_input_cpu.storage[index];
                beta_gradient_cpu.storage[j] += gradient_out;
            }
        }

        float inverse_batch_count = 1.0f / static_cast<float>(batch_count);
        for (std::size_t j = 0; j < feature_dimension; ++j)
        {
            float inverse_standard_deviation = 1.0f / std::sqrt(batch_variance_cpu.storage[j] + _epsilon);
            float gamma_value = gamma_cpu.storage[j];
            float delta_gamma = gamma_gradient_cpu.storage[j];
            float delta_beta = beta_gradient_cpu.storage[j];
            float scaling_coefficient = gamma_value * inverse_standard_deviation * inverse_batch_count;

            for (std::size_t i = 0; i < batch_count; ++i)
            {
                std::size_t index = i * feature_dimension + j;
                input_gradient_cpu.storage[index] = scaling_coefficient * (static_cast<float>(batch_count) * output_gradient_cpu.storage[index] - delta_beta - normalized_input_cpu.storage[index] * delta_gamma);
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::batchNormBackward: input_grad={}, gamma_grad={}, beta_grad={}",
                                       formatDataSample(input_gradient_cpu.storage),
                                       formatDataSample(gamma_gradient_cpu.storage),
                                       formatDataSample(beta_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void linearForward(
        const Impl &_weights, const Impl &_biases, Impl &_output_result) const override
    {
        const auto &weights_cpu = static_cast<const Cpu_Matrix_Impl &>(_weights);
        const auto &biases_cpu = static_cast<const Cpu_Matrix_Impl &>(_biases);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);

        std::size_t batch_size = rows;
        std::size_t input_dimension = columns;
        std::size_t output_dimension = weights_cpu.columns;
        output_cpu.reshape(batch_size, output_dimension);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < output_dimension; ++j)
            {
                output_cpu.storage[i * output_dimension + j] = (biases_cpu.rows == 1) ? biases_cpu.storage[j] : biases_cpu.storage[i * output_dimension + j];
            }
            for (std::size_t k = 0; k < input_dimension; ++k)
            {
                float input_value = storage[i * input_dimension + k];
                for (std::size_t j = 0; j < output_dimension; ++j)
                {
                    output_cpu.storage[i * output_dimension + j] += input_value * weights_cpu.storage[k * output_dimension + j];
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::linearForward: output shape=({}x{}), result={}",
                                       batch_size, output_dimension, formatDataSample(output_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void linearBackwardInput(const Impl &_weights, Impl &_input_gradient) const override
    {
        const auto &weights_cpu = static_cast<const Cpu_Matrix_Impl &>(_weights);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);

        std::size_t batch_size = rows;
        std::size_t output_dimension = columns;
        std::size_t input_dimension = weights_cpu.rows;
        input_gradient_cpu.reshape(batch_size, input_dimension);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < input_dimension; ++j)
            {
                float sum = 0.0f;
                for (std::size_t k = 0; k < output_dimension; ++k)
                {
                    sum += storage[i * output_dimension + k] * weights_cpu.storage[j * output_dimension + k];
                }
                input_gradient_cpu.storage[i * input_dimension + j] = sum;
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::linearBackwardInput: gradient_result={}",
                                       formatDataSample(input_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void linearBackwardWeightBias(
        const Impl &_output_gradient, Impl &_weight_gradient, Impl &_bias_gradient) const override
    {
        const auto &output_gradient_cpu = static_cast<const Cpu_Matrix_Impl &>(_output_gradient);
        auto &weight_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_weight_gradient);
        auto &bias_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_bias_gradient);

        std::size_t batch_size = rows;
        std::size_t input_dimension = columns;
        std::size_t output_dimension = output_gradient_cpu.columns;

        weight_gradient_cpu.reshape(input_dimension, output_dimension);
        bias_gradient_cpu.reshape(1, output_dimension);
        std::fill(weight_gradient_cpu.storage.begin(), weight_gradient_cpu.storage.end(), 0.0f);
        std::fill(bias_gradient_cpu.storage.begin(), bias_gradient_cpu.storage.end(), 0.0f);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < output_dimension; ++j)
            {
                float gradient_output_value = output_gradient_cpu.storage[i * output_dimension + j];
                bias_gradient_cpu.storage[j] += gradient_output_value;
                for (std::size_t k = 0; k < input_dimension; ++k)
                {
                    weight_gradient_cpu.storage[k * output_dimension + j] += storage[i * input_dimension + k] * gradient_output_value;
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::linearBackwardWeightBias: weight_grad={}, bias_grad={}",
                                       formatDataSample(weight_gradient_cpu.storage),
                                       formatDataSample(bias_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void batchNorm2dForward(
        const Impl &_gamma, const Impl &_beta,
        Impl &_running_mean, Impl &_running_variance,
        Impl &_batch_mean, Impl &_batch_variance,
        Impl &_normalized_input, Impl &_output_result,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        float _epsilon, float _momentum, bool _is_training) const override
    {
        const auto &gamma_cpu = static_cast<const Cpu_Matrix_Impl &>(_gamma);
        const auto &beta_cpu = static_cast<const Cpu_Matrix_Impl &>(_beta);
        auto &running_mean_cpu = static_cast<Cpu_Matrix_Impl &>(_running_mean);
        auto &running_variance_cpu = static_cast<Cpu_Matrix_Impl &>(_running_variance);
        auto &batch_mean_cpu = static_cast<Cpu_Matrix_Impl &>(_batch_mean);
        auto &batch_variance_cpu = static_cast<Cpu_Matrix_Impl &>(_batch_variance);
        auto &normalized_input_cpu = static_cast<Cpu_Matrix_Impl &>(_normalized_input);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);

        std::size_t batch_size = rows;
        std::size_t total_features = _input_height * _input_width * _input_channels;
        std::uint32_t spatial_count = static_cast<std::uint32_t>(batch_size * _input_height * _input_width);

        output_cpu.reshape(batch_size, total_features);
        normalized_input_cpu.reshape(batch_size, total_features);

        if (_is_training)
        {
            batch_mean_cpu.reshape(1, _input_channels);
            batch_variance_cpu.reshape(1, _input_channels);
            std::fill(batch_mean_cpu.storage.begin(), batch_mean_cpu.storage.end(), 0.0f);
            std::fill(batch_variance_cpu.storage.begin(), batch_variance_cpu.storage.end(), 0.0f);

            for (std::size_t i = 0; i < spatial_count; ++i)
            {
                for (std::uint32_t c = 0; c < _input_channels; ++c)
                {
                    batch_mean_cpu.storage[c] += storage[i * _input_channels + c];
                }
            }
            float inverse_spatial = 1.0f / static_cast<float>(spatial_count);
            for (std::uint32_t c = 0; c < _input_channels; ++c)
            {
                batch_mean_cpu.storage[c] *= inverse_spatial;
            }

            for (std::size_t i = 0; i < spatial_count; ++i)
            {
                for (std::uint32_t c = 0; c < _input_channels; ++c)
                {
                    float diff = storage[i * _input_channels + c] - batch_mean_cpu.storage[c];
                    batch_variance_cpu.storage[c] += diff * diff;
                }
            }
            for (std::uint32_t c = 0; c < _input_channels; ++c)
            {
                batch_variance_cpu.storage[c] *= inverse_spatial;
                running_mean_cpu.storage[c] = (1.0f - _momentum) * running_mean_cpu.storage[c] + _momentum * batch_mean_cpu.storage[c];
                running_variance_cpu.storage[c] = (1.0f - _momentum) * running_variance_cpu.storage[c] + _momentum * batch_variance_cpu.storage[c];
            }
            for (std::size_t i = 0; i < spatial_count; ++i)
            {
                for (std::uint32_t c = 0; c < _input_channels; ++c)
                {
                    std::size_t index = i * _input_channels + c;
                    float normalized_value = (storage[index] - batch_mean_cpu.storage[c]) / std::sqrt(batch_variance_cpu.storage[c] + _epsilon);
                    normalized_input_cpu.storage[index] = normalized_value;
                    output_cpu.storage[index] = gamma_cpu.storage[c] * normalized_value + beta_cpu.storage[c];
                }
            }
        }
        else
        {
            for (std::size_t i = 0; i < spatial_count; ++i)
            {
                for (std::uint32_t c = 0; c < _input_channels; ++c)
                {
                    std::size_t index = i * _input_channels + c;
                    float normalized_value = (storage[index] - running_mean_cpu.storage[c]) / std::sqrt(running_variance_cpu.storage[c] + _epsilon);
                    normalized_input_cpu.storage[index] = normalized_value;
                    output_cpu.storage[index] = gamma_cpu.storage[c] * normalized_value + beta_cpu.storage[c];
                }
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::batchNorm2dForward: is_training={}, output={}, running_mean={}, running_var={}",
                                       _is_training, formatDataSample(output_cpu.storage),
                                       formatDataSample(running_mean_cpu.storage), formatDataSample(running_variance_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void batchNorm2dBackward(
        const Impl &_gamma, const Impl &_batch_variance, const Impl &_normalized_input,
        Impl &_gamma_gradient, Impl &_beta_gradient, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels, float _epsilon) const override
    {
        const auto &gamma_cpu = static_cast<const Cpu_Matrix_Impl &>(_gamma);
        const auto &batch_variance_cpu = static_cast<const Cpu_Matrix_Impl &>(_batch_variance);
        const auto &normalized_input_cpu = static_cast<const Cpu_Matrix_Impl &>(_normalized_input);
        auto &gamma_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_gamma_gradient);
        auto &beta_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_beta_gradient);
        auto &input_gradient_cpu = static_cast<Cpu_Matrix_Impl &>(_input_gradient);

        std::size_t batch_size = rows;
        std::size_t total_features = _input_height * _input_width * _input_channels;
        std::uint32_t spatial_count = static_cast<std::uint32_t>(batch_size * _input_height * _input_width);

        input_gradient_cpu.reshape(batch_size, total_features);
        gamma_gradient_cpu.reshape(1, _input_channels);
        beta_gradient_cpu.reshape(1, _input_channels);
        std::fill(gamma_gradient_cpu.storage.begin(), gamma_gradient_cpu.storage.end(), 0.0f);
        std::fill(beta_gradient_cpu.storage.begin(), beta_gradient_cpu.storage.end(), 0.0f);

        for (std::size_t i = 0; i < spatial_count; ++i)
        {
            for (std::uint32_t c = 0; c < _input_channels; ++c)
            {
                std::size_t index = i * _input_channels + c;
                float gradient_output_value = storage[index];
                gamma_gradient_cpu.storage[c] += gradient_output_value * normalized_input_cpu.storage[index];
                beta_gradient_cpu.storage[c] += gradient_output_value;
            }
        }

        float inverse_spatial = 1.0f / static_cast<float>(spatial_count);
        for (std::uint32_t c = 0; c < _input_channels; ++c)
        {
            float inverse_standard_deviation = 1.0f / std::sqrt(batch_variance_cpu.storage[c] + _epsilon);
            float gamma_value = gamma_cpu.storage[c];
            float delta_gamma = gamma_gradient_cpu.storage[c];
            float delta_beta = beta_gradient_cpu.storage[c];
            float scaling_coefficient = gamma_value * inverse_standard_deviation * inverse_spatial;

            for (std::size_t i = 0; i < spatial_count; ++i)
            {
                std::size_t index = i * _input_channels + c;
                input_gradient_cpu.storage[index] = scaling_coefficient * (static_cast<float>(spatial_count) * storage[index] - delta_beta - normalized_input_cpu.storage[index] * delta_gamma);
            }
        }

        Logger::logMessage(std::format("Cpu_Matrix_Impl::batchNorm2dBackward: input_grad={}, gamma_grad={}, beta_grad={}",
                                       formatDataSample(input_gradient_cpu.storage),
                                       formatDataSample(gamma_gradient_cpu.storage),
                                       formatDataSample(beta_gradient_cpu.storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void cceLoss(const Impl &_target_implementation, Impl &_output_result, float _epsilon) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_cpu = static_cast<const Cpu_Matrix_Impl &>(_target_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(1, 1);

        float loss_sum = 0.0f;
        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            float probability_clamped = std::clamp(storage[i], _epsilon, 1.0f - _epsilon);
            loss_sum += -target_cpu.storage[i] * std::log(probability_clamped);
        }
        output_cpu.storage[0] = loss_sum;

        Logger::logMessage(std::format("Cpu_Matrix_Impl::cceLoss: elements={}, epsilon={}, calculated_loss={:.6f}",
                                       storage.size(), _epsilon, loss_sum),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
    }

    void mseLoss(const Impl &_target_implementation, Impl &_output_result) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_cpu = static_cast<const Cpu_Matrix_Impl &>(_target_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(1, 1);

        float loss_sum = 0.0f;
        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            float difference = storage[i] - target_cpu.storage[i];
            loss_sum += difference * difference;
        }
        output_cpu.storage[0] = loss_sum;

        Logger::logMessage(std::format("Cpu_Matrix_Impl::mseLoss: elements={}, calculated_loss={:.6f}",
                                       storage.size(), loss_sum),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
    }

    void maeLoss(const Impl &_target_implementation, Impl &_output_result) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_cpu = static_cast<const Cpu_Matrix_Impl &>(_target_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(1, 1);

        float loss_sum = 0.0f;
        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            loss_sum += std::abs(storage[i] - target_cpu.storage[i]);
        }
        output_cpu.storage[0] = loss_sum;

        Logger::logMessage(std::format("Cpu_Matrix_Impl::maeLoss: elements={}, calculated_loss={:.6f}",
                                       storage.size(), loss_sum),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
    }

    void bceLoss(const Impl &_target_implementation, Impl &_output_result, float _epsilon) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_cpu = static_cast<const Cpu_Matrix_Impl &>(_target_implementation);
        auto &output_cpu = static_cast<Cpu_Matrix_Impl &>(_output_result);
        output_cpu.reshape(1, 1);

        float loss_sum = 0.0f;
        for (std::size_t i = 0; i < storage.size(); ++i)
        {
            float probability_clamped = std::clamp(storage[i], _epsilon, 1.0f - _epsilon);
            float target_value = target_cpu.storage[i];
            loss_sum += -(target_value * std::log(probability_clamped) + (1.0f - target_value) * std::log(1.0f - probability_clamped));
        }
        output_cpu.storage[0] = loss_sum;

        Logger::logMessage(std::format("Cpu_Matrix_Impl::bceLoss: elements={}, epsilon={}, calculated_loss={:.6f}",
                                       storage.size(), _epsilon, loss_sum),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);
    }

    void uploadData(const std::vector<float> &_host_data) override
    {
        if (_host_data.size() != rows * columns)
        {
            Logger::logMessage("Cpu_Matrix_Impl::uploadData: Host data size mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        storage = _host_data;

        Logger::logMessage(std::format("Cpu_Matrix_Impl::uploadData: uploaded {} elements, sample={}",
                                       storage.size(), formatDataSample(storage)),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::TENSOR_INSPECTION);
    }
};
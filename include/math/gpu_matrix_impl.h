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

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "impl.h"

struct Matrix_Dimensions
{
    std::uint32_t rows_a = 0;
    std::uint32_t columns_a = 0;
    std::uint32_t columns_b = 0;
};

struct Elementwise_Dimensions
{
    std::uint32_t total_elements = 0;
    std::uint32_t columns = 0;
    std::uint32_t is_broadcast = 0;
};

struct Transpose_Dimensions
{
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
};

class Gpu_Matrix_Impl : public Impl
{
private:
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::shared_ptr<gpu::vector> storage;
    mutable std::vector<float> host_cache;

    static inline bool is_graph_logging_enabled = true;

public:
    static inline std::size_t distinct_operations_count;

private:
    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum _pipeline_id,
                     const std::vector<std::shared_ptr<gpu::vector>> &_buffers,
                     const Push_Constants_Type &_push_constants,
                     std::uint32_t _workgroup_count_x,
                     std::uint32_t _workgroup_count_y = 1,
                     std::uint32_t _workgroup_count_z = 1) const
    {
        if (_buffers.size() > 16)
        {
            Logger::logMessage("Gpu_Matrix_Impl::pushToGraph: Exceeded maximum supported buffer count",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::GRAPH_RECORDING);
            throw std::runtime_error("Exceeded maximum supported buffer count");
        }

        Compute_Node node;
        node.pipeline_id = _pipeline_id;
        node.buffers = _buffers;

        const auto *byte_pointer = reinterpret_cast<const std::uint8_t *>(&_push_constants);
        node.push_constants_data.assign(byte_pointer, byte_pointer + sizeof(Push_Constants_Type));

        node.workgroup_count_x = _workgroup_count_x;
        node.workgroup_count_y = _workgroup_count_y;
        node.workgroup_count_z = _workgroup_count_z;

        Execution_Engine::getInstance().getCurrentGraph().addNode(node);

        if (is_graph_logging_enabled)
        {
            std::string buffer_trace;
            for (std::size_t index = 0; index < _buffers.size(); ++index)
            {
                if (_buffers[index])
                {
                    buffer_trace += std::format(" [Arg{}: ID={}, Buffer={:p}, Offset={}]",
                                                index,
                                                _buffers[index]->getId(),
                                                static_cast<void *>(_buffers[index]->getBuffer()),
                                                _buffers[index]->getAllocation().offset);
                }
            }

            is_graph_logging_enabled = Logger::logMessage(
                std::format("Gpu_Matrix_Impl::pushToGraph: Operation: {} | {}",
                            magic_enum::enum_name(_pipeline_id),
                            buffer_trace),
                Log_Level::LOG_DEBUG,
                true,
                distinct_operations_count,
                Log_Feature::GRAPH_RECORDING);
        }
    }

    const Gpu_Matrix_Impl &castToGpuMatrix(const Impl &_other_implementation, const std::string &_error_message) const noexcept
    {
        return static_cast<const Gpu_Matrix_Impl &>(_other_implementation);
    }

    Gpu_Matrix_Impl &castToGpuMatrix(Impl &_other_implementation, const std::string &_error_message) const noexcept
    {
        return static_cast<Gpu_Matrix_Impl &>(_other_implementation);
    }

    template <typename Pipeline_Enum>
    void executeElementwise(const Impl &_other_implementation,
                            Impl &_output_result,
                            Pipeline_Enum _pipeline_id,
                            const std::string &_error_message,
                            bool _is_broadcast_allowed) const
    {
        const auto &other_gpu = castToGpuMatrix(_other_implementation, _error_message);
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::executeElementwise: Invalid output matrix");

        bool is_broadcast_applied = false;
        if (_is_broadcast_allowed)
        {
            bool has_same_shape = (rows == other_gpu.getRows() && columns == other_gpu.getColumns());
            is_broadcast_applied = (other_gpu.getRows() == 1 && columns == other_gpu.getColumns());
            if (!has_same_shape && !is_broadcast_applied)
            {
                validateSameDimensions(_other_implementation);
            }
        }
        else
        {
            validateSameDimensions(_other_implementation);
        }

        output_gpu.reshape(rows, columns);

        Elementwise_Dimensions elementwise_dimensions{
            .total_elements = static_cast<std::uint32_t>(rows * columns),
            .columns = static_cast<std::uint32_t>(columns),
            .is_broadcast = static_cast<std::uint32_t>(is_broadcast_applied ? 1 : 0)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::executeElementwise: total_elements={}, columns={}, is_broadcast={}",
                                       elementwise_dimensions.total_elements,
                                       elementwise_dimensions.columns,
                                       elementwise_dimensions.is_broadcast),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(_pipeline_id,
                    {storage, other_gpu.storage, output_gpu.storage},
                    elementwise_dimensions,
                    (elementwise_dimensions.total_elements + 255) / 256);
    }

public:
    Gpu_Matrix_Impl(std::size_t _rows, std::size_t _columns)
        : rows(_rows), columns(_columns)
    {
        std::size_t total_elements = _rows * _columns;
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements);
        }
    }

    Gpu_Matrix_Impl(std::size_t _rows, std::size_t _columns, const std::vector<float> &_host_data)
        : rows(_rows), columns(_columns)
    {
        if (_host_data.size() != _rows * _columns)
        {
            Logger::logMessage("Gpu_Matrix_Impl::Gpu_Matrix_Impl: Host data size mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        if (_rows * _columns > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), _host_data);
        }
    }

    Gpu_Matrix_Impl(std::size_t _rows, std::size_t _columns, std::shared_ptr<gpu::vector> _storage)
        : rows(_rows), columns(_columns), storage(std::move(_storage))
    {
    }

    ~Gpu_Matrix_Impl() noexcept override = default;

     std::size_t getRows() const noexcept override
    {
        return rows;
    }

     std::size_t getColumns() const noexcept override
    {
        return columns;
    }

     std::size_t getCols() const noexcept override
    {
        return columns;
    }

     const std::vector<float> &getData() const noexcept override
    {
        Logger::logMessage("Gpu_Matrix_Impl::getData: Reading data from GPU, risk of synchronization stalls",
                           Log_Level::LOG_WARNING,
                           true,
                           1,
                           Log_Feature::MEMORY_TRANSFER);
        if (rows == 0 || columns == 0)
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::getData: Zero dimension encountered (rows={}, columns={})", rows, columns),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            host_cache.clear();
            return host_cache;
        }
        Execution_Engine::getInstance().getContext().flush();
        host_cache.resize(rows * columns);
        if (storage)
        {
            storage->downloadData(host_cache);
        }
        return host_cache;
    }

     Storage_Handle getStorage() const override
    {
        return storage;
    }

     Mutable_Storage_Handle getStorage() override
    {
        return storage;
    }

     std::shared_ptr<gpu::vector> getVector() override
    {
        return storage;
    }

     VkBuffer getBuffer() const noexcept
    {
        return storage ? storage->getBuffer() : VK_NULL_HANDLE;
    }

     bool isEmpty() const noexcept override
    {
        return !storage || storage->isEmpty();
    }

    void reshape(std::size_t _rows, std::size_t _columns) override
    {
        std::size_t total_elements = _rows * _columns;
        if (rows == _rows && columns == _columns && storage && storage->getSize() == total_elements)
        {
            return;
        }
        rows = _rows;
        columns = _columns;
        if (total_elements == 0)
        {
            return;
        }
        if (!storage || storage->getSize() != total_elements)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements);
        }
    }

    void matmul(const Impl &_other_implementation, Impl &_output_result) const override
    {
        validateMatmulDimensions(_other_implementation);
        const auto &other_gpu = castToGpuMatrix(_other_implementation, "Gpu_Matrix_Impl::matmul: Target matrix is not a Gpu_Matrix_Impl");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::matmul: Output matrix is not a Gpu_Matrix_Impl");

        output_gpu.reshape(rows, other_gpu.getColumns());

        Matrix_Dimensions matrix_dimensions{
            .rows_a = static_cast<std::uint32_t>(rows),
            .columns_a = static_cast<std::uint32_t>(columns),
            .columns_b = static_cast<std::uint32_t>(output_gpu.getColumns())};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::matmul: rows_a={}, columns_a={}, columns_b={}",
                                       matrix_dimensions.rows_a,
                                       matrix_dimensions.columns_a,
                                       matrix_dimensions.columns_b),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MATMUL,
                    {storage, other_gpu.storage, output_gpu.storage},
                    matrix_dimensions,
                    (matrix_dimensions.columns_b + 15) / 16,
                    (matrix_dimensions.rows_a + 15) / 16);
    }

    void add(const Impl &_other_implementation, Impl &_output_result) const override
    {
        executeElementwise(_other_implementation, _output_result, Compute_Pipeline::ADD, "Gpu_Matrix_Impl::add: Invalid matrix", true);
    }

    void sub(const Impl &_other_implementation, Impl &_output_result) const override
    {
        executeElementwise(_other_implementation, _output_result, Compute_Pipeline::SUB, "Gpu_Matrix_Impl::sub: Invalid matrix", true);
    }

    void mulScalar(float _scalar, Impl &_output_result) const override
    {
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::mulScalar: Invalid output matrix");
        output_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        struct Scalar_Push_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{total_elements, _scalar};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::mulScalar: total_elements={}, scalar={}", constants.total_elements, constants.scalar),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MUL_SCALAR, {storage, output_gpu.storage}, constants, (total_elements + 255) / 256);
    }

    void divScalar(float _scalar, Impl &_output_result) const override
    {
        if (std::abs(_scalar) < 1e-8f)
        {
            Logger::logMessage("Gpu_Matrix_Impl::divScalar: Division by zero",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::divScalar: Invalid output matrix");
        output_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        struct Scalar_Push_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{total_elements, 1.0f / _scalar};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::divScalar: total_elements={}, scalar={}", constants.total_elements, constants.scalar),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MUL_SCALAR, {storage, output_gpu.storage}, constants, (total_elements + 255) / 256);
    }

    void hadamardMul(const Impl &_other_implementation, Impl &_output_result) const override
    {
        executeElementwise(_other_implementation, _output_result, Compute_Pipeline::HADAMARD_MUL, "Gpu_Matrix_Impl::hadamardMul: Invalid matrix", false);
    }

    void hadamardDiv(const Impl &_other_implementation, Impl &_output_result) const override
    {
        executeElementwise(_other_implementation, _output_result, Compute_Pipeline::HADAMARD_DIV, "Gpu_Matrix_Impl::hadamardDiv: Invalid matrix", false);
    }

    void transpose(Impl &_output_result) const override
    {
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::transpose: Invalid output matrix");
        output_gpu.reshape(columns, rows);

        Transpose_Dimensions transpose_dimensions{
            .rows = static_cast<std::uint32_t>(rows),
            .columns = static_cast<std::uint32_t>(columns)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::transpose: rows={}, columns={}", transpose_dimensions.rows, transpose_dimensions.columns),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::TRANSPOSE,
                    {storage, output_gpu.storage},
                    transpose_dimensions,
                    (transpose_dimensions.columns + 15) / 16,
                    (transpose_dimensions.rows + 15) / 16);
    }

    void relu(Impl &_output_result) const override
    {
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::relu: Invalid output matrix");
        output_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::relu: total_elements={}", total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::RELU, {storage, output_gpu.storage}, total_elements, (total_elements + 255) / 256);
    }

    void reluBackward(const Impl &_output_gradient, Impl &_input_gradient) const override
    {
        validateSameDimensions(_output_gradient);
        const auto &output_gradient_gpu = castToGpuMatrix(_output_gradient, "Gpu_Matrix_Impl::reluBackward: Invalid gradient matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::reluBackward: Invalid output matrix");
        input_gradient_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::reluBackward: total_elements={}", total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::RELU_BACKWARD, {storage, output_gradient_gpu.storage, input_gradient_gpu.storage}, total_elements, (total_elements + 255) / 256);
    }

    void gelu(Impl &_output_result) const override
    {
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::gelu: Invalid output matrix");
        output_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::gelu: total_elements={}", total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::GELU, {storage, output_gpu.storage}, total_elements, (total_elements + 255) / 256);
    }

    void geluBackward(const Impl &_output_gradient, Impl &_input_gradient) const override
    {
        validateSameDimensions(_output_gradient);
        const auto &output_gradient_gpu = castToGpuMatrix(_output_gradient, "Gpu_Matrix_Impl::geluBackward: Invalid gradient matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::geluBackward: Invalid output matrix");
        input_gradient_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::geluBackward: total_elements={}", total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::GELU_BACKWARD, {storage, output_gradient_gpu.storage, input_gradient_gpu.storage}, total_elements, (total_elements + 255) / 256);
    }

    void inverse(Impl &_output_result) const override
    {
        validateSquare();
        std::uint32_t dimension_size = static_cast<std::uint32_t>(rows);
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::inverse: Invalid output matrix");
        output_gpu.reshape(dimension_size, dimension_size);

        struct Inverse_Push_Constants
        {
            std::uint32_t dimension_size;
        } constants{dimension_size};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::inverse: dimension_size={}", constants.dimension_size),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MATRIX_INVERSE, {storage, output_gpu.storage}, constants, 1, 1, 1);
    }

    void normalize(Impl &_output_result) const override
    {
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::normalize: Invalid output matrix");
        output_gpu.reshape(rows, columns);
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        struct Normalize_Push_Constants
        {
            std::uint32_t total_elements;
        } constants{total_elements};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::normalize: total_elements={}", constants.total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE);

        pushToGraph(Compute_Pipeline::NORMALIZE, {storage, output_gpu.storage}, constants, 1, 1, 1);
    }

    void softmax(Impl &_output_result) const override
    {
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::softmax: Invalid output matrix");
        output_gpu.reshape(rows, columns);

        struct Softmax_Dimensions
        {
            std::uint32_t rows;
            std::uint32_t columns;
        } constants{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::softmax: rows={}, columns={}", constants.rows, constants.columns),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::SOFTMAX, {storage, output_gpu.storage}, constants, constants.rows, 1, 1);
    }

    void softmaxBackward(const Impl &_output_gradient, Impl &_input_gradient) const override
    {
        validateSameDimensions(_output_gradient);
        const auto &output_gradient_gpu = castToGpuMatrix(_output_gradient, "Gpu_Matrix_Impl::softmaxBackward: Invalid gradient matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::softmaxBackward: Invalid output matrix");
        input_gradient_gpu.reshape(rows, columns);

        struct Softmax_Dimensions
        {
            std::uint32_t rows;
            std::uint32_t columns;
        } constants{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::softmaxBackward: rows={}, columns={}", constants.rows, constants.columns),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::SOFTMAX_BACKWARD, {storage, output_gradient_gpu.storage, input_gradient_gpu.storage}, constants, constants.rows, 1, 1);
    }

    void sgdUpdate(const Impl &_gradient_implementation, float _learning_rate, float _max_gradient = 0.0f) override
    {
        validateSameDimensions(_gradient_implementation);
        const auto &gradient_gpu = castToGpuMatrix(_gradient_implementation, "Gpu_Matrix_Impl::sgdUpdate: Invalid gradient matrix");
        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);

        struct Sgd_Push_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float max_gradient;
        } constants{total_elements, _learning_rate, _max_gradient};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::sgdUpdate: total_elements={}, learning_rate={}, max_gradient={}",
                                       constants.total_elements,
                                       constants.learning_rate,
                                       constants.max_gradient),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);

        pushToGraph(Compute_Pipeline::SGD_UPDATE, {storage, gradient_gpu.storage}, constants, (total_elements + 255) / 256);
    }

    void adamUpdate(const Impl &_gradient_implementation,
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

        const auto &gradient_gpu = castToGpuMatrix(_gradient_implementation, "Gpu_Matrix_Impl::adamUpdate: Invalid gradient matrix");
        const auto &first_moment_gpu = castToGpuMatrix(_first_moment_implementation, "Gpu_Matrix_Impl::adamUpdate: Invalid first moment matrix");
        const auto &second_moment_gpu = castToGpuMatrix(_second_moment_implementation, "Gpu_Matrix_Impl::adamUpdate: Invalid second moment matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);
        std::size_t effective_timestep = std::max<std::size_t>(_timestep, 1);
        float bias_correction_first = std::max(1.0f - std::pow(_beta1, static_cast<float>(effective_timestep)), 1e-8f);
        float bias_correction_second = std::max(1.0f - std::pow(_beta2, static_cast<float>(effective_timestep)), 1e-8f);

        struct Adam_Push_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float inverse_bias_correction_first;
            float inverse_sqrt_bias_correction_second;
        } constants{total_elements,
                    _learning_rate,
                    _beta1,
                    _beta2,
                    _epsilon,
                    _max_gradient,
                    1.0f / bias_correction_first,
                    1.0f / std::sqrt(bias_correction_second)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::adamUpdate: total_elements={}, learning_rate={}, beta1={}, beta2={}, epsilon={}, max_gradient={}, inv_first={}, inv_sqrt_second={}",
                                       constants.total_elements,
                                       constants.learning_rate,
                                       constants.beta1,
                                       constants.beta2,
                                       constants.epsilon,
                                       constants.max_gradient,
                                       constants.inverse_bias_correction_first,
                                       constants.inverse_sqrt_bias_correction_second),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::OPTIMIZER_STEP);

        pushToGraph(Compute_Pipeline::ADAM_UPDATE,
                    {storage, gradient_gpu.storage, first_moment_gpu.storage, second_moment_gpu.storage},
                    constants,
                    (total_elements + 255) / 256);
    }

    void matdiv(const Impl &_other_implementation, Impl &_output_result) const override
    {
        Gpu_Matrix_Impl temporary_inverse(0, 0);
        _other_implementation.inverse(temporary_inverse);
        matmul(temporary_inverse, _output_result);
    }

    void matmulAdd(const Impl &_other_implementation, const Impl &_bias_implementation, Impl &_output_result) const override
    {
        validateMatmulDimensions(_other_implementation);
        const auto &weights_gpu = castToGpuMatrix(_other_implementation, "Gpu_Matrix_Impl::matmulAdd: Invalid weights matrix");
        const auto &biases_gpu = castToGpuMatrix(_bias_implementation, "Gpu_Matrix_Impl::matmulAdd: Invalid biases matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::matmulAdd: Invalid output matrix");

        if (biases_gpu.getRows() != 1 && biases_gpu.getRows() != rows)
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::matmulAdd: Bias rows mismatch (expected 1 or {}, got {})", rows, biases_gpu.getRows()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Bias rows mismatch in matmulAdd");
        }
        if (biases_gpu.getColumns() != weights_gpu.getColumns())
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::matmulAdd: Bias columns mismatch (expected {}, got {})", weights_gpu.getColumns(), biases_gpu.getColumns()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Bias columns mismatch in matmulAdd");
        }

        output_gpu.reshape(rows, weights_gpu.columns);

        struct Matmul_Add_Push_Constants
        {
            std::uint32_t rows_x;
            std::uint32_t columns_x;
            std::uint32_t columns_weights;
            std::uint32_t padding;
        } constants{static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(columns),
                    static_cast<std::uint32_t>(weights_gpu.columns),
                    0};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::matmulAdd: rows_x={}, columns_x={}, columns_w={}, padding={}",
                                       constants.rows_x,
                                       constants.columns_x,
                                       constants.columns_weights,
                                       constants.padding),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MATMUL_ADD,
                    {storage, weights_gpu.storage, biases_gpu.storage, output_gpu.storage},
                    constants,
                    (constants.columns_weights + 15) / 16,
                    (constants.rows_x + 15) / 16,
                    1);
    }

    void conv2d(
        const Impl &_weights, const Impl &_biases, Impl &_output_result,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_channels, std::uint32_t _kernel_size,
        std::uint32_t _stride, std::uint32_t _padding) const override
    {
        const auto &weights_gpu = castToGpuMatrix(_weights, "Gpu_Matrix_Impl::conv2d: Invalid weights matrix");
        const auto &biases_gpu = castToGpuMatrix(_biases, "Gpu_Matrix_Impl::conv2d: Invalid biases matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::conv2d: Invalid output matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t output_height = (_input_height + 2 * _padding - _kernel_size) / _stride + 1;
        std::uint32_t output_width = (_input_width + 2 * _padding - _kernel_size) / _stride + 1;
        output_gpu.reshape(batch_size, output_height * output_width * _output_channels);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t input_channels;
            std::uint32_t output_height;
            std::uint32_t output_width;
            std::uint32_t output_channels;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, _input_height, _input_width, _input_channels, output_height, output_width, _output_channels, _kernel_size, _stride, _padding};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::conv2d: batch_size={}, input_height={}, input_width={}, input_channels={}, output_height={}, output_width={}, output_channels={}, kernel_size={}, stride={}, padding={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.input_channels,
                                       constants.output_height,
                                       constants.output_width,
                                       constants.output_channels,
                                       constants.kernel_size,
                                       constants.stride,
                                       constants.padding),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::CONV2D_FORWARD_PASS,
                    {storage, weights_gpu.storage, biases_gpu.storage, output_gpu.storage},
                    constants,
                    (_output_channels + 15) / 16,
                    (output_width + 15) / 16,
                    batch_size * output_height);
    }

    void conv2dBackwardInput(
        const Impl &_weights, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width, std::uint32_t _output_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        const auto &weights_gpu = castToGpuMatrix(_weights, "Gpu_Matrix_Impl::conv2dBackwardInput: Invalid weights matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::conv2dBackwardInput: Invalid gradient input matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        input_gradient_gpu.reshape(batch_size, _input_height * _input_width * _input_channels);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t input_channels;
            std::uint32_t output_height;
            std::uint32_t output_width;
            std::uint32_t output_channels;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, _input_height, _input_width, _input_channels, _output_height, _output_width, _output_channels, _kernel_size, _stride, _padding};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::conv2dBackwardInput: batch_size={}, input_height={}, input_width={}, input_channels={}, output_height={}, output_width={}, output_channels={}, kernel_size={}, stride={}, padding={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.input_channels,
                                       constants.output_height,
                                       constants.output_width,
                                       constants.output_channels,
                                       constants.kernel_size,
                                       constants.stride,
                                       constants.padding),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT,
                    {storage, weights_gpu.storage, input_gradient_gpu.storage},
                    constants,
                    (_input_channels + 15) / 16,
                    (_input_width + 15) / 16,
                    batch_size * _input_height);
    }

    void conv2dBackwardWeight(
        const Impl &_output_gradient, Impl &_weight_gradient, Impl &_bias_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        std::uint32_t _output_height, std::uint32_t _output_width, std::uint32_t _output_channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        const auto &output_gradient_gpu = castToGpuMatrix(_output_gradient, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid output gradient matrix");
        auto &weight_gradient_gpu = castToGpuMatrix(_weight_gradient, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid weight gradient matrix");
        auto &bias_gradient_gpu = castToGpuMatrix(_bias_gradient, "Gpu_Matrix_Impl::conv2dBackwardWeight: Invalid bias gradient matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        weight_gradient_gpu.reshape(1, _kernel_size * _kernel_size * _input_channels * _output_channels);
        bias_gradient_gpu.reshape(1, _output_channels);

        struct Conv2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t input_channels;
            std::uint32_t output_height;
            std::uint32_t output_width;
            std::uint32_t output_channels;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, _input_height, _input_width, _input_channels, _output_height, _output_width, _output_channels, _kernel_size, _stride, _padding};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::conv2dBackwardWeight: batch_size={}, input_height={}, input_width={}, input_channels={}, output_height={}, output_width={}, output_channels={}, kernel_size={}, stride={}, padding={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.input_channels,
                                       constants.output_height,
                                       constants.output_width,
                                       constants.output_channels,
                                       constants.kernel_size,
                                       constants.stride,
                                       constants.padding),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT,
                    {storage, output_gradient_gpu.storage, weight_gradient_gpu.storage, bias_gradient_gpu.storage},
                    constants,
                    (_output_channels + 15) / 16,
                    (_input_channels + 15) / 16,
                    _kernel_size * _kernel_size);
    }

    void maxpool2d(
        Impl &_output_result, Impl &_output_mask,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        auto &result_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::maxpool2d: Invalid output result matrix");
        auto &mask_gpu = castToGpuMatrix(_output_mask, "Gpu_Matrix_Impl::maxpool2d: Invalid output mask matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t output_height = (_input_height + 2 * _padding - _kernel_size) / _stride + 1;
        std::uint32_t output_width = (_input_width + 2 * _padding - _kernel_size) / _stride + 1;

        result_gpu.reshape(batch_size, output_height * output_width * _channels);
        mask_gpu.reshape(batch_size, output_height * output_width * _channels);

        struct Max_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t channels;
            std::uint32_t output_height;
            std::uint32_t output_width;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, _input_height, _input_width, _channels, output_height, output_width, _kernel_size, _stride, _padding};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::maxpool2d: batch_size={}, input_height={}, input_width={}, channels={}, output_height={}, output_width={}, kernel_size={}, stride={}, padding={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.channels,
                                       constants.output_height,
                                       constants.output_width,
                                       constants.kernel_size,
                                       constants.stride,
                                       constants.padding),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::MAXPOOL2D_FORWARD,
                    {storage, result_gpu.storage, mask_gpu.storage},
                    constants,
                    (_channels + 15) / 16,
                    (output_width + 15) / 16,
                    batch_size * output_height);
    }

    void maxpool2dBackward(
        const Impl &_mask, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels,
        std::uint32_t _output_height, std::uint32_t _output_width,
        std::uint32_t _kernel_size, std::uint32_t _stride, std::uint32_t _padding) const override
    {
        const auto &mask_gpu = castToGpuMatrix(_mask, "Gpu_Matrix_Impl::maxpool2dBackward: Invalid mask matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::maxpool2dBackward: Invalid input gradient matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        input_gradient_gpu.reshape(batch_size, _input_height * _input_width * _channels);

        struct Max_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t channels;
            std::uint32_t output_height;
            std::uint32_t output_width;
            std::uint32_t kernel_size;
            std::uint32_t stride;
            std::uint32_t padding;
        } constants{batch_size, _input_height, _input_width, _channels, _output_height, _output_width, _kernel_size, _stride, _padding};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::maxpool2dBackward: batch_size={}, input_height={}, input_width={}, channels={}, output_height={}, output_width={}, kernel_size={}, stride={}, padding={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.channels,
                                       constants.output_height,
                                       constants.output_width,
                                       constants.kernel_size,
                                       constants.stride,
                                       constants.padding),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::MAXPOOL2D_BACKWARD,
                    {mask_gpu.storage, storage, input_gradient_gpu.storage},
                    constants,
                    (_channels + 15) / 16,
                    (_input_width + 15) / 16,
                    batch_size * _input_height);
    }

    void globalAvgPool2d(Impl &_output_result, std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::globalAvgPool2d: Invalid output matrix");
        output_gpu.reshape(batch_size, _channels);

        struct Global_Avg_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t channels;
        } constants{batch_size, _input_height, _input_width, _channels};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::globalAvgPool2d: batch_size={}, input_height={}, input_width={}, channels={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.channels),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD, {storage, output_gpu.storage}, constants, (_channels + 255) / 256, batch_size, 1);
    }

    void globalAvgPool2dBackward(Impl &_input_gradient, std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _channels) const override
    {
        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::globalAvgPool2dBackward: Invalid gradient matrix");
        input_gradient_gpu.reshape(batch_size, _input_height * _input_width * _channels);

        struct Global_Avg_Pool_2d_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t channels;
        } constants{batch_size, _input_height, _input_width, _channels};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::globalAvgPool2dBackward: batch_size={}, input_height={}, input_width={}, channels={}",
                                       constants.batch_size,
                                       constants.input_height,
                                       constants.input_width,
                                       constants.channels),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::POOLING_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD, {storage, input_gradient_gpu.storage}, constants, (_channels + 15) / 16, (_input_width + 15) / 16, batch_size * _input_height);
    }

    void batchNormForward(
        const Impl &_gamma, const Impl &_beta, Impl &_running_mean, Impl &_running_variance,
        Impl &_batch_mean, Impl &_batch_variance, Impl &_normalized_input, Impl &_output_result,
        float _epsilon, float _momentum, bool _is_training) const override
    {
        const auto &gamma_gpu = castToGpuMatrix(_gamma, "Gpu_Matrix_Impl::batchNormForward: Invalid gamma matrix");
        const auto &beta_gpu = castToGpuMatrix(_beta, "Gpu_Matrix_Impl::batchNormForward: Invalid beta matrix");
        auto &running_mean_gpu = castToGpuMatrix(_running_mean, "Gpu_Matrix_Impl::batchNormForward: Invalid running mean matrix");
        auto &running_variance_gpu = castToGpuMatrix(_running_variance, "Gpu_Matrix_Impl::batchNormForward: Invalid running variance matrix");
        auto &normalized_input_gpu = castToGpuMatrix(_normalized_input, "Gpu_Matrix_Impl::batchNormForward: Invalid normalized input matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::batchNormForward: Invalid output matrix");

        std::uint32_t batch_count = static_cast<std::uint32_t>(rows);
        std::uint32_t feature_dimension = static_cast<std::uint32_t>(columns);
        output_gpu.reshape(batch_count, feature_dimension);
        normalized_input_gpu.reshape(batch_count, feature_dimension);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::batchNormForward: batch_count={}, feature_dimension={}, momentum={}, epsilon={}, is_training={}",
                                       batch_count, feature_dimension, _momentum, _epsilon, _is_training),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        if (_is_training)
        {
            auto &batch_mean_gpu = castToGpuMatrix(_batch_mean, "Gpu_Matrix_Impl::batchNormForward: Invalid batch mean matrix");
            auto &batch_variance_gpu = castToGpuMatrix(_batch_variance, "Gpu_Matrix_Impl::batchNormForward: Invalid batch variance matrix");
            batch_mean_gpu.reshape(1, feature_dimension);
            batch_variance_gpu.reshape(1, feature_dimension);

            struct BatchNorm_Stats_Push_Constants
            {
                std::uint32_t batch_size;
                std::uint32_t feature_dimension;
                float momentum;
            } stats_constants{batch_count, feature_dimension, _momentum};

            pushToGraph(Compute_Pipeline::BATCH_NORM_STATS_FORWARD,
                        {storage, batch_mean_gpu.storage, batch_variance_gpu.storage, running_mean_gpu.storage, running_variance_gpu.storage},
                        stats_constants,
                        (feature_dimension + 255) / 256,
                        1,
                        1);

            struct BatchNorm_Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t feature_dimension;
                float epsilon;
            } transform_constants{batch_count * feature_dimension, feature_dimension, _epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                        {storage, batch_mean_gpu.storage, batch_variance_gpu.storage, gamma_gpu.storage, beta_gpu.storage, output_gpu.storage, normalized_input_gpu.storage},
                        transform_constants,
                        (batch_count * feature_dimension + 255) / 256,
                        1,
                        1);
        }
        else
        {
            struct BatchNorm_Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t feature_dimension;
                float epsilon;
            } transform_constants{batch_count * feature_dimension, feature_dimension, _epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                        {storage, running_mean_gpu.storage, running_variance_gpu.storage, gamma_gpu.storage, beta_gpu.storage, output_gpu.storage, normalized_input_gpu.storage},
                        transform_constants,
                        (batch_count * feature_dimension + 255) / 256,
                        1,
                        1);
        }
    }

    void batchNormBackward(
        const Impl &_output_gradient, const Impl &_gamma, const Impl &_batch_variance, const Impl &_normalized_input,
        Impl &_gamma_gradient, Impl &_beta_gradient, Impl &_input_gradient, float _epsilon) const override
    {
        const auto &output_gradient_gpu = castToGpuMatrix(_output_gradient, "Gpu_Matrix_Impl::batchNormBackward: Invalid output gradient matrix");
        const auto &gamma_gpu = castToGpuMatrix(_gamma, "Gpu_Matrix_Impl::batchNormBackward: Invalid gamma matrix");
        const auto &batch_variance_gpu = castToGpuMatrix(_batch_variance, "Gpu_Matrix_Impl::batchNormBackward: Invalid batch variance matrix");
        const auto &normalized_input_gpu = castToGpuMatrix(_normalized_input, "Gpu_Matrix_Impl::batchNormBackward: Invalid normalized input matrix");

        auto &gamma_gradient_gpu = castToGpuMatrix(_gamma_gradient, "Gpu_Matrix_Impl::batchNormBackward: Invalid gamma gradient matrix");
        auto &beta_gradient_gpu = castToGpuMatrix(_beta_gradient, "Gpu_Matrix_Impl::batchNormBackward: Invalid beta gradient matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::batchNormBackward: Invalid input gradient matrix");

        std::uint32_t batch_count = static_cast<std::uint32_t>(rows);
        std::uint32_t feature_dimension = static_cast<std::uint32_t>(columns);

        gamma_gradient_gpu.reshape(1, feature_dimension);
        beta_gradient_gpu.reshape(1, feature_dimension);
        input_gradient_gpu.reshape(batch_count, feature_dimension);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::batchNormBackward: batch_count={}, feature_dimension={}, epsilon={}",
                                       batch_count, feature_dimension, _epsilon),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        struct BatchNorm_Backward_Stats_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t feature_dimension;
        } stats_constants{batch_count, feature_dimension};

        pushToGraph(Compute_Pipeline::BATCH_NORM_STATS_BACKWARD,
                    {output_gradient_gpu.storage, normalized_input_gpu.storage, gamma_gradient_gpu.storage, beta_gradient_gpu.storage},
                    stats_constants,
                    (feature_dimension + 255) / 256,
                    1,
                    1);

        struct BatchNorm_Backward_Transform_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t batch_size;
            std::uint32_t feature_dimension;
            float epsilon;
        } transform_constants{batch_count * feature_dimension, batch_count, feature_dimension, _epsilon};

        pushToGraph(Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
                    {output_gradient_gpu.storage, normalized_input_gpu.storage, gamma_gpu.storage, gamma_gradient_gpu.storage, beta_gradient_gpu.storage, batch_variance_gpu.storage, input_gradient_gpu.storage},
                    transform_constants,
                    (batch_count * feature_dimension + 255) / 256,
                    1,
                    1);
    }

    void linearForward(const Impl &_weights, const Impl &_biases, Impl &_output_result) const override
    {
        validateMatmulDimensions(_weights);
        const auto &weights_gpu = castToGpuMatrix(_weights, "Gpu_Matrix_Impl::linearForward: Invalid weights matrix");
        const auto &biases_gpu = castToGpuMatrix(_biases, "Gpu_Matrix_Impl::linearForward: Invalid biases matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::linearForward: Invalid output matrix");

        if (biases_gpu.getRows() != 1 && biases_gpu.getRows() != rows)
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::linearForward: Bias rows mismatch (expected 1 or {}, got {})", rows, biases_gpu.getRows()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Bias rows mismatch in linearForward");
        }
        if (biases_gpu.getColumns() != weights_gpu.getColumns())
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::linearForward: Bias columns mismatch (expected {}, got {})", weights_gpu.getColumns(), biases_gpu.getColumns()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Bias columns mismatch in linearForward");
        }

        output_gpu.reshape(rows, weights_gpu.columns);

        struct Linear_Forward_Push_Constants
        {
            std::uint32_t rows_x;
            std::uint32_t columns_x;
            std::uint32_t columns_weights;
            std::uint32_t padding;
        } constants{static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(columns),
                    static_cast<std::uint32_t>(weights_gpu.columns),
                    0};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::linearForward: batch_size={}, input_dimension={}, output_dimension={}",
                                       constants.rows_x,
                                       constants.columns_x,
                                       constants.columns_weights),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        pushToGraph(Compute_Pipeline::MATMUL_ADD,
                    {storage, weights_gpu.storage, biases_gpu.storage, output_gpu.storage},
                    constants,
                    (constants.columns_weights + 15) / 16,
                    (constants.rows_x + 15) / 16,
                    1);
    }

    void linearBackwardInput(const Impl &_weights, Impl &_input_gradient) const override
    {
        const auto &weights_gpu = castToGpuMatrix(_weights, "Gpu_Matrix_Impl::linearBackwardInput: Invalid weights matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::linearBackwardInput: Invalid input gradient matrix");

        if (columns != weights_gpu.getColumns())
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::linearBackwardInput: Output gradient columns ({}) mismatch with weights columns ({})", columns, weights_gpu.getColumns()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Output gradient columns mismatch with weights columns in linearBackwardInput");
        }

        input_gradient_gpu.reshape(rows, weights_gpu.rows);

        struct Linear_Backward_Input_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_dimension;
            std::uint32_t output_dimension;
        } constants{static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(weights_gpu.rows),
                    static_cast<std::uint32_t>(columns)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::linearBackwardInput: batch_size={}, input_dimension={}, output_dimension={}",
                                       constants.batch_size,
                                       constants.input_dimension,
                                       constants.output_dimension),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_INPUT,
                    {storage, weights_gpu.storage, input_gradient_gpu.storage},
                    constants,
                    (constants.input_dimension + 15) / 16,
                    (constants.batch_size + 15) / 16,
                    1);
    }

    void linearBackwardWeightBias(const Impl &_output_gradient, Impl &_weight_gradient, Impl &_bias_gradient) const override
    {
        const auto &output_gradient_gpu = castToGpuMatrix(_output_gradient, "Gpu_Matrix_Impl::linearBackwardWeightBias: Invalid output gradient matrix");
        auto &weight_gradient_gpu = castToGpuMatrix(_weight_gradient, "Gpu_Matrix_Impl::linearBackwardWeightBias: Invalid weight gradient matrix");
        auto &bias_gradient_gpu = castToGpuMatrix(_bias_gradient, "Gpu_Matrix_Impl::linearBackwardWeightBias: Invalid bias gradient matrix");

        if (rows != output_gradient_gpu.getRows())
        {
            Logger::logMessage(std::format("Gpu_Matrix_Impl::linearBackwardWeightBias: Batch size mismatch between input ({}) and output gradient ({})", rows, output_gradient_gpu.getRows()),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE | Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Batch size mismatch in linearBackwardWeightBias");
        }

        weight_gradient_gpu.reshape(columns, output_gradient_gpu.columns);
        bias_gradient_gpu.reshape(1, output_gradient_gpu.columns);

        struct Linear_Backward_Weight_Bias_Push_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_dimension;
            std::uint32_t output_dimension;
        } constants{static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(columns),
                    static_cast<std::uint32_t>(output_gradient_gpu.columns)};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::linearBackwardWeightBias: batch_size={}, input_dimension={}, output_dimension={}",
                                       constants.batch_size,
                                       constants.input_dimension,
                                       constants.output_dimension),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS,
                    {storage, output_gradient_gpu.storage, weight_gradient_gpu.storage, bias_gradient_gpu.storage},
                    constants,
                    (constants.output_dimension + 15) / 16,
                    (constants.input_dimension + 15) / 16,
                    1);
    }

    void batchNorm2dForward(
        const Impl &_gamma, const Impl &_beta,
        Impl &_running_mean, Impl &_running_variance,
        Impl &_batch_mean, Impl &_batch_variance,
        Impl &_normalized_input, Impl &_output_result,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels,
        float _epsilon, float _momentum, bool _is_training) const override
    {
        const auto &gamma_gpu = castToGpuMatrix(_gamma, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid gamma matrix");
        const auto &beta_gpu = castToGpuMatrix(_beta, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid beta matrix");
        auto &running_mean_gpu = castToGpuMatrix(_running_mean, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid running mean matrix");
        auto &running_variance_gpu = castToGpuMatrix(_running_variance, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid running variance matrix");
        auto &normalized_input_gpu = castToGpuMatrix(_normalized_input, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid normalized input matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid output matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t total_features = _input_height * _input_width * _input_channels;
        std::uint32_t spatial_count = batch_size * _input_height * _input_width;
        output_gpu.reshape(batch_size, total_features);
        normalized_input_gpu.reshape(batch_size, total_features);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::batchNorm2dForward: batch_size={}, input_channels={}, input_height={}, input_width={}, momentum={}, epsilon={}, is_training={}",
                                       batch_size, _input_channels, _input_height, _input_width, _momentum, _epsilon, _is_training),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::FORWARD_EVALUATION);

        if (_is_training)
        {
            auto &batch_mean_gpu = castToGpuMatrix(_batch_mean, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid batch mean matrix");
            auto &batch_variance_gpu = castToGpuMatrix(_batch_variance, "Gpu_Matrix_Impl::batchNorm2dForward: Invalid batch variance matrix");
            batch_mean_gpu.reshape(1, _input_channels);
            batch_variance_gpu.reshape(1, _input_channels);

            struct BatchNorm2d_Stats_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                std::uint32_t spatial_count;
                float momentum;
            } stats_constants{batch_size * total_features, _input_channels, spatial_count, _momentum};

            pushToGraph(Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD,
                        {storage, batch_mean_gpu.storage, batch_variance_gpu.storage, running_mean_gpu.storage, running_variance_gpu.storage},
                        stats_constants,
                        (_input_channels + 255) / 256,
                        1,
                        1);

            struct BatchNorm2d_Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                float epsilon;
            } transform_constants{batch_size * total_features, _input_channels, _epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                        {storage, batch_mean_gpu.storage, batch_variance_gpu.storage, gamma_gpu.storage, beta_gpu.storage, output_gpu.storage, normalized_input_gpu.storage},
                        transform_constants,
                        (batch_size * total_features + 255) / 256,
                        1,
                        1);
        }
        else
        {
            struct BatchNorm2d_Transform_Push_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                float epsilon;
            } transform_constants{batch_size * total_features, _input_channels, _epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                        {storage, running_mean_gpu.storage, running_variance_gpu.storage, gamma_gpu.storage, beta_gpu.storage, output_gpu.storage, normalized_input_gpu.storage},
                        transform_constants,
                        (batch_size * total_features + 255) / 256,
                        1,
                        1);
        }
    }

    void batchNorm2dBackward(
        const Impl &_gamma, const Impl &_batch_variance, const Impl &_normalized_input,
        Impl &_gamma_gradient, Impl &_beta_gradient, Impl &_input_gradient,
        std::uint32_t _input_height, std::uint32_t _input_width, std::uint32_t _input_channels, float _epsilon) const override
    {
        const auto &gamma_gpu = castToGpuMatrix(_gamma, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid gamma matrix");
        const auto &batch_variance_gpu = castToGpuMatrix(_batch_variance, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid batch variance matrix");
        const auto &normalized_input_gpu = castToGpuMatrix(_normalized_input, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid normalized input matrix");
        auto &gamma_gradient_gpu = castToGpuMatrix(_gamma_gradient, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid gamma gradient matrix");
        auto &beta_gradient_gpu = castToGpuMatrix(_beta_gradient, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid beta gradient matrix");
        auto &input_gradient_gpu = castToGpuMatrix(_input_gradient, "Gpu_Matrix_Impl::batchNorm2dBackward: Invalid input gradient matrix");

        std::uint32_t batch_size = static_cast<std::uint32_t>(rows);
        std::uint32_t total_features = _input_height * _input_width * _input_channels;
        std::uint32_t spatial_count = batch_size * _input_height * _input_width;
        input_gradient_gpu.reshape(batch_size, total_features);
        gamma_gradient_gpu.reshape(1, _input_channels);
        beta_gradient_gpu.reshape(1, _input_channels);

        Logger::logMessage(std::format("Gpu_Matrix_Impl::batchNorm2dBackward: batch_size={}, input_channels={}, input_height={}, input_width={}, epsilon={}",
                                       batch_size, _input_channels, _input_height, _input_width, _epsilon),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::NORMALIZATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);

        struct BatchNorm2d_Backward_Stats_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t channels;
            std::uint32_t spatial_count;
        } stats_constants{batch_size * total_features, _input_channels, spatial_count};

        pushToGraph(Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD,
                    {storage, normalized_input_gpu.storage, gamma_gradient_gpu.storage, beta_gradient_gpu.storage},
                    stats_constants,
                    (_input_channels + 255) / 256,
                    1,
                    1);

        struct BatchNorm2d_Backward_Transform_Push_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t channels;
            std::uint32_t spatial_count;
            float epsilon;
        } transform_constants{batch_size * total_features, _input_channels, spatial_count, _epsilon};

        pushToGraph(Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
                    {storage, normalized_input_gpu.storage, gamma_gpu.storage, gamma_gradient_gpu.storage, beta_gradient_gpu.storage, batch_variance_gpu.storage, input_gradient_gpu.storage},
                    transform_constants,
                    (batch_size * total_features + 255) / 256,
                    1,
                    1);
    }

    void cceLoss(const Impl &_target_implementation, Impl &_output_result, float _epsilon) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_gpu = castToGpuMatrix(_target_implementation, "Gpu_Matrix_Impl::cceLoss: Invalid target matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::cceLoss: Invalid output matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);
        std::uint32_t workgroup_count_x = (total_elements + 255) / 256;
        output_gpu.reshape(1, workgroup_count_x);

        struct Cce_Loss_Push_Constants
        {
            std::uint32_t total_elements;
            float epsilon;
        } constants{total_elements, _epsilon};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::cceLoss: total_elements={}, epsilon={}", constants.total_elements, constants.epsilon),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        pushToGraph(Compute_Pipeline::CCE_LOSS, {storage, target_gpu.storage, output_gpu.storage}, constants, workgroup_count_x, 1, 1);
    }

    void mseLoss(const Impl &_target_implementation, Impl &_output_result) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_gpu = castToGpuMatrix(_target_implementation, "Gpu_Matrix_Impl::mseLoss: Invalid target matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::mseLoss: Invalid output matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);
        std::uint32_t workgroup_count_x = (total_elements + 255) / 256;
        output_gpu.reshape(1, workgroup_count_x);

        struct Mse_Loss_Push_Constants
        {
            std::uint32_t total_elements;
        } constants{total_elements};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::mseLoss: total_elements={}", constants.total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        pushToGraph(Compute_Pipeline::MSE_LOSS, {storage, target_gpu.storage, output_gpu.storage}, constants, workgroup_count_x, 1, 1);
    }

    void maeLoss(const Impl &_target_implementation, Impl &_output_result) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_gpu = castToGpuMatrix(_target_implementation, "Gpu_Matrix_Impl::maeLoss: Invalid target matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::maeLoss: Invalid output matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);
        std::uint32_t workgroup_count_x = (total_elements + 255) / 256;
        output_gpu.reshape(1, workgroup_count_x);

        struct Mae_Loss_Push_Constants
        {
            std::uint32_t total_elements;
        } constants{total_elements};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::maeLoss: total_elements={}", constants.total_elements),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        pushToGraph(Compute_Pipeline::MAE_LOSS, {storage, target_gpu.storage, output_gpu.storage}, constants, workgroup_count_x, 1, 1);
    }

    void bceLoss(const Impl &_target_implementation, Impl &_output_result, float _epsilon) const override
    {
        validateSameDimensions(_target_implementation);
        const auto &target_gpu = castToGpuMatrix(_target_implementation, "Gpu_Matrix_Impl::bceLoss: Invalid target matrix");
        auto &output_gpu = castToGpuMatrix(_output_result, "Gpu_Matrix_Impl::bceLoss: Invalid output matrix");

        std::uint32_t total_elements = static_cast<std::uint32_t>(rows * columns);
        std::uint32_t workgroup_count_x = (total_elements + 255) / 256;
        output_gpu.reshape(1, workgroup_count_x);

        struct Bce_Loss_Push_Constants
        {
            std::uint32_t total_elements;
            float epsilon;
        } constants{total_elements, _epsilon};

        Logger::logMessage(std::format("Gpu_Matrix_Impl::bceLoss: total_elements={}, epsilon={}", constants.total_elements, constants.epsilon),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::LOSS_COMPUTE);

        pushToGraph(Compute_Pipeline::BCE_LOSS, {storage, target_gpu.storage, output_gpu.storage}, constants, workgroup_count_x, 1, 1);
    }

    void uploadData(const std::vector<float> &_host_data) override
    {
        if (_host_data.size() != rows * columns)
        {
            Logger::logMessage("Gpu_Matrix_Impl::uploadData: Host data size mismatch",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        if (storage)
        {
            storage->uploadData(_host_data);
        }
    }
};
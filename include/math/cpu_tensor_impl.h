#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "tensor_impl.h"

class Cpu_Tensor_Impl : public Tensor_Impl
{
private:
    std::shared_ptr<std::vector<float>> storage_buffer;
    std::shared_ptr<std::vector<float16_t>> storage_buffer_fp16;
    mutable std::vector<float> materialized_cache;
    mutable std::mutex cache_mutex;

    static std::string formatDataSample(const std::vector<float> &data, size_t sample_limit = 5)
    {
        if (data.empty())
            return "[]";
        std::string formatted = "[";
        size_t count = std::min(data.size(), sample_limit);
        for (size_t i = 0; i < count; ++i)
        {
            formatted += std::format("{:.4e}{}", data[i], (i + 1 < count) ? ", " : "");
        }
        if (data.size() > sample_limit)
        {
            formatted += std::format(", ... (total {})", data.size());
        }
        formatted += "]";
        return formatted;
    }

    size_t resolveFlatIndex(std::span<const size_t> indices) const noexcept
    {
        size_t elem_size = getDataTypeSize(data_type);
        size_t flat_idx = byte_offset / elem_size;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            flat_idx += indices[i] * strides[i];
        }
        return flat_idx;
    }

    template <typename Op>
    void iterateCoordinates(Op &&op) const
    {
        size_t rank = shape.getRank();
        if (rank == 0 || total_elements == 0)
            return;
        std::array<size_t, MAX_TENSOR_RANK> coords{};
        for (size_t i = 0; i < total_elements; ++i)
        {
            op(i, resolveFlatIndex(std::span{coords.data(), rank}));
            for (size_t d = rank; d > 0; --d)
            {
                if (++coords[d - 1] < shape[d - 1])
                    break;
                coords[d - 1] = 0;
            }
        }
    }

public:
    Cpu_Tensor_Impl(size_t rows, size_t columns)
    {
        updateShapeAndStrides(Shape{rows, columns});
        storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
    }

    Cpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data)
    {
        updateShapeAndStrides(Shape{rows, columns});
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        storage_buffer = std::make_shared<std::vector<float>>(host_data);
    }

    Cpu_Tensor_Impl(size_t rows, size_t columns, std::vector<float> &&host_data)
    {
        updateShapeAndStrides(Shape{rows, columns});
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        storage_buffer = std::make_shared<std::vector<float>>(std::move(host_data));
    }

    explicit Cpu_Tensor_Impl(Shape tensor_shape)
    {
        updateShapeAndStrides(tensor_shape);
        storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
    }

    Cpu_Tensor_Impl(Shape tensor_shape, Data_Type type)
    {
        data_type = type;
        updateShapeAndStrides(tensor_shape);
        if (data_type == Data_Type::FLOAT16)
        {
            storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(total_elements, static_cast<float16_t>(0.0f));
        }
        else
        {
            storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
        }
    }

    Cpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data)
    {
        updateShapeAndStrides(tensor_shape);
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl: Host data size mismatch (expected {}, got {})",
                                             total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        storage_buffer = std::make_shared<std::vector<float>>(host_data);
    }

    Cpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<std::vector<float>> buffer, size_t offset_elements)
    {
        shape = tensor_shape;
        strides = tensor_strides;
        storage_buffer = std::move(buffer);
        byte_offset = offset_elements * getDataTypeSize(data_type);
        total_elements = shape.getTotalElements();
    }

    ~Cpu_Tensor_Impl() noexcept override = default;

    void reshape(size_t rows, size_t columns) override
    {
        reshape(Shape{rows, columns});
    }

    void reshape(Shape new_shape) override
    {
        size_t new_total = new_shape.getTotalElements();
        if (!isContiguous() || byte_offset != 0)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::reshape: Cannot reshape non-contiguous view directly"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::runtime_error("Cannot reshape non-contiguous tensor view");
        }

        if (data_type == Data_Type::FLOAT16)
        {
            if (storage_buffer_fp16 && storage_buffer_fp16.use_count() > 1 && total_elements != new_total)
            {
                storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(new_total, static_cast<float16_t>(0.0f));
            }
            else if (!storage_buffer_fp16 || total_elements != new_total)
            {
                if (!storage_buffer_fp16)
                {
                    storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(new_total, static_cast<float16_t>(0.0f));
                }
                else
                {
                    storage_buffer_fp16->resize(new_total, static_cast<float16_t>(0.0f));
                }
            }
        }
        else
        {
            if (storage_buffer.use_count() > 1 && total_elements != new_total)
            {
                storage_buffer = std::make_shared<std::vector<float>>(new_total, 0.0F);
            }
            else if (total_elements != new_total)
            {
                storage_buffer->resize(new_total, 0.0F);
            }
        }
        updateShapeAndStrides(new_shape);
    }

    void permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const override
    {
        Shape new_shape = shape;
        Stride new_strides = strides;
        for (size_t i = 0; i < shape.getRank(); ++i)
        {
            new_shape[i] = shape[axes_permutation[i]];
            new_strides[i] = strides[axes_permutation[i]];
        }
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.data_type = data_type;
        output_cpu.shape = new_shape;
        output_cpu.strides = new_strides;
        output_cpu.storage_buffer = storage_buffer;
        output_cpu.storage_buffer_fp16 = storage_buffer_fp16;
        output_cpu.byte_offset = byte_offset;
        output_cpu.total_elements = total_elements;

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::permute: shape {} -> {}",
                                        shape.toString(), new_shape.toString()},
                            Log_Level::LOG_DEBUG, true, 0, Log_Feature::TENSOR_INSPECTION);
    }

    void slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const override
    {
        Shape new_shape = shape;
        new_shape[axis] = length;
        size_t additional_offset = start * strides[axis];

        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.data_type = data_type;
        output_cpu.shape = new_shape;
        output_cpu.strides = strides;
        output_cpu.storage_buffer = storage_buffer;
        output_cpu.storage_buffer_fp16 = storage_buffer_fp16;
        output_cpu.byte_offset = byte_offset + additional_offset * getDataTypeSize(data_type);
        output_cpu.total_elements = new_shape.getTotalElements();

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::slice: axis={}, start={}, length={}, shape={}",
                                        axis, start, length, new_shape.toString()},
                            Log_Level::LOG_DEBUG, true, 0, Log_Feature::TENSOR_INSPECTION);
    }

    void contiguous(Tensor_Impl &output) const override
    {
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.data_type = data_type;
        output_cpu.shape = shape;
        output_cpu.strides = shape.computeContiguousStrides();
        output_cpu.byte_offset = 0;
        output_cpu.total_elements = total_elements;

        if (data_type == Data_Type::FLOAT16 && storage_buffer_fp16)
        {
            output_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(total_elements, static_cast<float16_t>(0.0f));
            output_cpu.storage_buffer.reset();
            iterateCoordinates([this, &output_cpu](size_t out_idx, size_t src_idx)
                               { (*output_cpu.storage_buffer_fp16)[out_idx] = (*storage_buffer_fp16)[src_idx]; });
        }
        else
        {
            output_cpu.storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
            output_cpu.storage_buffer_fp16.reset();
            if (storage_buffer)
            {
                iterateCoordinates([this, &output_cpu](size_t out_idx, size_t src_idx)
                                   { (*output_cpu.storage_buffer)[out_idx] = (*storage_buffer)[src_idx]; });
            }
        }
    }

    void to(Data_Type target_type, Tensor_Impl &output) const override
    {
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.setDataType(target_type);
        output_cpu.updateShapeAndStrides(shape);
        output_cpu.byte_offset = 0;

        if (target_type == data_type)
        {
            if (data_type == Data_Type::FLOAT16 && storage_buffer_fp16)
            {
                output_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(*storage_buffer_fp16);
                output_cpu.storage_buffer.reset();
            }
            else if (storage_buffer)
            {
                output_cpu.storage_buffer = std::make_shared<std::vector<float>>(*storage_buffer);
                output_cpu.storage_buffer_fp16.reset();
            }
            return;
        }

        if (data_type == Data_Type::FLOAT32 && target_type == Data_Type::FLOAT16)
        {
            const auto &src = getData();
            output_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(total_elements);
            convertFp32ToFp16(src.data(), output_cpu.storage_buffer_fp16->data(), total_elements);
            output_cpu.storage_buffer.reset();
        }
        else if (data_type == Data_Type::FLOAT16 && target_type == Data_Type::FLOAT32)
        {
            const auto &src = getData();
            output_cpu.storage_buffer = std::make_shared<std::vector<float>>(src);
            output_cpu.storage_buffer_fp16.reset();
        }
    }

    void matmul(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        size_t rank_a = shape.getRank();
        size_t m_dim = (rank_a >= 2) ? shape[rank_a - 2] : getRows();
        size_t k_dim = (rank_a >= 2) ? shape[rank_a - 1] : getColumns();
        size_t b_dim = (rank_a >= 3) ? (total_elements / (m_dim * k_dim)) : 1;

        size_t rank_b = other.getShape().getRank();
        size_t k_other = (rank_b >= 2) ? other.getShape()[rank_b - 2] : other.getRows();
        size_t n_dim = (rank_b >= 2) ? other.getShape()[rank_b - 1] : other.getColumns();
        size_t b_other = (rank_b >= 3) ? (other.getTotalElements() / (k_other * n_dim)) : 1;

        if (k_dim != k_other)
        {
            throw std::invalid_argument("Matrix inner dimensions must match for multiplication");
        }

        bool broadcast_b = (b_other == 1 && b_dim > 1);
        if (!broadcast_b && b_dim != b_other)
        {
            throw std::invalid_argument("Batch dimensions must match or be broadcastable");
        }

        Shape out_shape;
        if (rank_a <= 2 && rank_b <= 2)
        {
            out_shape = Shape{m_dim, n_dim};
        }
        else if (rank_a == 3)
        {
            out_shape = Shape{b_dim, m_dim, n_dim};
        }
        else if (rank_a >= 4)
        {
            std::vector<size_t> dims(shape.getDimensions().begin(), shape.getDimensions().end());
            dims[rank_a - 2] = m_dim;
            dims[rank_a - 1] = n_dim;
            out_shape = Shape(dims);
        }
        else
        {
            out_shape = Shape{b_dim, m_dim, n_dim};
        }

        output_cpu.reshape(out_shape);
        std::fill(output_cpu.storage_buffer->begin(), output_cpu.storage_buffer->end(), 0.0F);

        for (size_t b = 0; b < b_dim; ++b)
        {
            size_t a_batch_offset = b * m_dim * k_dim;
            size_t b_batch_offset = broadcast_b ? 0 : (b * k_dim * n_dim);
            size_t c_batch_offset = b * m_dim * n_dim;

            for (size_t i = 0; i < m_dim; ++i)
            {
                for (size_t k = 0; k < k_dim; ++k)
                {
                    float a_val = a_data[a_batch_offset + i * k_dim + k];
                    for (size_t j = 0; j < n_dim; ++j)
                    {
                        (*output_cpu.storage_buffer)[c_batch_offset + i * n_dim + j] += a_val * b_data[b_batch_offset + k * n_dim + j];
                    }
                }
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::matmul: batch={}, ({}x{}) x ({}x{}) -> ({}x{}), sample={}",
                                        b_dim, m_dim, k_dim, k_dim, n_dim, m_dim, n_dim, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        Cpu_Tensor_Impl temp_inverse(0, 0);
        other.inverse(temp_inverse);
        matmul(temp_inverse, output);
    }

    void add(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        bool is_broadcast = (other.getRows() == 1 && getColumns() == other.getColumns());

        if (shape == other.getShape())
        {
            for (size_t i = 0; i < total_elements; ++i)
            {
                (*output_cpu.storage_buffer)[i] = a_data[i] + b_data[i];
            }
        }
        else if (is_broadcast)
        {
            size_t cols = getColumns();
            for (size_t r = 0; r < getRows(); ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    (*output_cpu.storage_buffer)[r * cols + c] = a_data[r * cols + c] + b_data[c];
                }
            }
        }
        else
        {
            validateSameDimensions(other);
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::add: shape={}, broadcast={}, sample={}",
                                        shape.toString(), is_broadcast, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void sub(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        bool is_broadcast = (other.getRows() == 1 && getColumns() == other.getColumns());

        if (shape == other.getShape())
        {
            for (size_t i = 0; i < total_elements; ++i)
            {
                (*output_cpu.storage_buffer)[i] = a_data[i] - b_data[i];
            }
        }
        else if (is_broadcast)
        {
            size_t cols = getColumns();
            for (size_t r = 0; r < getRows(); ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    (*output_cpu.storage_buffer)[r * cols + c] = a_data[r * cols + c] - b_data[c];
                }
            }
        }
        else
        {
            validateSameDimensions(other);
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::sub: shape={}, broadcast={}, sample={}",
                                        shape.toString(), is_broadcast, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void mulScalar(float scalar, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        for (size_t i = 0; i < total_elements; ++i)
        {
            (*output_cpu.storage_buffer)[i] = a_data[i] * scalar;
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::mulScalar: scalar={}, elements={}, sample={}",
                                        scalar, total_elements, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void divScalar(float scalar, Tensor_Impl &output) const override
    {
        if (std::abs(scalar) < 1e-8F)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::divScalar: Division by zero encountered"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        mulScalar(1.0F / scalar, output);
    }

    void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        bool same_shape = (shape == other.getShape());
        bool is_broadcast = (other.getRows() == 1 && getColumns() == other.getColumns());
        if (!same_shape && !is_broadcast)
        {
            validateSameDimensions(other);
        }

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        if (same_shape)
        {
            for (size_t i = 0; i < total_elements; ++i)
            {
                (*output_cpu.storage_buffer)[i] = a_data[i] * b_data[i];
            }
        }
        else if (is_broadcast)
        {
            size_t cols = getColumns();
            for (size_t r = 0; r < getRows(); ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    (*output_cpu.storage_buffer)[r * cols + c] = a_data[r * cols + c] * b_data[c];
                }
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardMul: elements={}, sample={}",
                                        total_elements, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        bool same_shape = (shape == other.getShape());
        bool is_broadcast = (other.getRows() == 1 && getColumns() == other.getColumns());
        if (!same_shape && !is_broadcast)
        {
            validateSameDimensions(other);
        }

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        if (same_shape)
        {
            for (size_t i = 0; i < total_elements; ++i)
            {
                if (std::abs(b_data[i]) < 1e-8F)
                {
                    Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardDiv: Division by zero at index {}", i},
                                       Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
                    throw std::runtime_error("Division by zero in hadamardDiv");
                }
                (*output_cpu.storage_buffer)[i] = a_data[i] / b_data[i];
            }
        }
        else if (is_broadcast)
        {
            size_t cols = getColumns();
            for (size_t r = 0; r < getRows(); ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    if (std::abs(b_data[c]) < 1e-8F)
                    {
                        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardDiv: Division by zero at column {}", c},
                                           Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
                        throw std::runtime_error("Division by zero in hadamardDiv");
                    }
                    (*output_cpu.storage_buffer)[r * cols + c] = a_data[r * cols + c] / b_data[c];
                }
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardDiv: elements={}, sample={}",
                                        total_elements, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void transpose(Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        size_t r_count = getRows();
        size_t c_count = getColumns();
        output_cpu.reshape(c_count, r_count);

        for (size_t i = 0; i < r_count; ++i)
        {
            for (size_t j = 0; j < c_count; ++j)
            {
                (*output_cpu.storage_buffer)[j * r_count + i] = a_data[i * c_count + j];
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::transpose: ({}x{}) -> ({}x{}), sample={}",
                                        r_count, c_count, c_count, r_count, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void inverse(Tensor_Impl &output) const override
    {
        validateSquare();
        const auto &a_data = getData();
        size_t n = getRows();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(n, n);

        std::vector<float> aug(n * 2 * n, 0.0F);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                aug[i * (2 * n) + j] = a_data[i * n + j];
            }
            aug[i * (2 * n) + (n + i)] = 1.0F;
        }

        for (size_t i = 0; i < n; ++i)
        {
            size_t pivot_row = i;
            float max_val = std::abs(aug[i * (2 * n) + i]);
            for (size_t k = i + 1; k < n; ++k)
            {
                float val = std::abs(aug[k * (2 * n) + i]);
                if (val > max_val)
                {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if (max_val < 1e-7F)
            {
                Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::inverse: Singular matrix detected"},
                                   Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
                throw std::runtime_error("Matrix is singular");
            }

            if (pivot_row != i)
            {
                for (size_t j = 0; j < 2 * n; ++j)
                {
                    std::swap(aug[i * (2 * n) + j], aug[pivot_row * (2 * n) + j]);
                }
            }

            float pivot = aug[i * (2 * n) + i];
            for (size_t j = 0; j < 2 * n; ++j)
            {
                aug[i * (2 * n) + j] /= pivot;
            }

            for (size_t k = 0; k < n; ++k)
            {
                if (k != i)
                {
                    float factor = aug[k * (2 * n) + i];
                    for (size_t j = 0; j < 2 * n; ++j)
                    {
                        aug[k * (2 * n) + j] -= factor * aug[i * (2 * n) + j];
                    }
                }
            }
        }

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                (*output_cpu.storage_buffer)[i * n + j] = aug[i * (2 * n) + (n + j)];
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::inverse: dimension={}, sample={}",
                                        n, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void normalize(Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        float sum_sq = 0.0F;
        for (float val : a_data)
            sum_sq += val * val;
        float norm = std::sqrt(sum_sq);

        if (norm < 1e-8F)
        {
            *output_cpu.storage_buffer = a_data;
            return;
        }

        for (size_t i = 0; i < total_elements; ++i)
        {
            (*output_cpu.storage_buffer)[i] = a_data[i] / norm;
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::normalize: norm={:.4e}, sample={}",
                                        norm, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
    }

    void relu(Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);

        for (size_t i = 0; i < total_elements; ++i)
        {
            (*output_cpu.storage_buffer)[i] = std::max(0.0F, a_data[i]);
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::relu: elements={}, sample={}",
                                        total_elements, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }

    void reluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        validateSameDimensions(output_gradient);
        const auto &a_data = getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);
        in_grad_cpu.reshape(shape);

        for (size_t i = 0; i < total_elements; ++i)
        {
            (*in_grad_cpu.storage_buffer)[i] = (a_data[i] > 0.0F) ? grad_data[i] : 0.0F;
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::reluBackward: elements={}, sample={}",
                                        total_elements, formatDataSample(*in_grad_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void gelu(Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(shape);
        constexpr float ALPHA = 0.7978845608F;
        constexpr float BETA = 0.044715F;

        for (size_t i = 0; i < total_elements; ++i)
        {
            float x = a_data[i];
            float tanh_in = std::tanh(ALPHA * (x + BETA * x * x * x));
            (*output_cpu.storage_buffer)[i] = 0.5F * x * (1.0F + tanh_in);
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::gelu: elements={}, sample={}",
                                        total_elements, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }

    void geluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        validateSameDimensions(output_gradient);
        const auto &a_data = getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);
        in_grad_cpu.reshape(shape);

        constexpr float ALPHA = 0.7978845608F;
        constexpr float BETA = 0.044715F;

        for (size_t i = 0; i < total_elements; ++i)
        {
            float x = a_data[i];
            float x2 = x * x;
            float tanh_in = std::tanh(ALPHA * (x + BETA * x2 * x));
            float sech2 = 1.0F - tanh_in * tanh_in;
            float deriv = 0.5F * (1.0F + tanh_in) + 0.5F * x * sech2 * ALPHA * (1.0F + 3.0F * BETA * x2);
            (*in_grad_cpu.storage_buffer)[i] = grad_data[i] * deriv;
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::geluBackward: elements={}, sample={}",
                                        total_elements, formatDataSample(*in_grad_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void softmax(Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        size_t rows_count = getRows();
        size_t cols_count = getColumns();
        output_cpu.reshape(rows_count, cols_count);

        for (size_t r = 0; r < rows_count; ++r)
        {
            float max_val = a_data[r * cols_count];
            for (size_t c = 1; c < cols_count; ++c)
            {
                max_val = std::max(max_val, a_data[r * cols_count + c]);
            }
            float sum_exp = 0.0F;
            for (size_t c = 0; c < cols_count; ++c)
            {
                float exp_val = std::exp(a_data[r * cols_count + c] - max_val);
                (*output_cpu.storage_buffer)[r * cols_count + c] = exp_val;
                sum_exp += exp_val;
            }
            for (size_t c = 0; c < cols_count; ++c)
            {
                (*output_cpu.storage_buffer)[r * cols_count + c] /= sum_exp;
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::softmax: shape=({}x{}), sample={}",
                                        rows_count, cols_count, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }

    void softmaxBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        const auto &a_data = getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);
        size_t rows_count = getRows();
        size_t cols_count = getColumns();
        in_grad_cpu.reshape(rows_count, cols_count);

        for (size_t r = 0; r < rows_count; ++r)
        {
            for (size_t i = 0; i < cols_count; ++i)
            {
                float acc = 0.0F;
                for (size_t j = 0; j < cols_count; ++j)
                {
                    float delta = (i == j) ? 1.0F : 0.0F;
                    float jac = a_data[r * cols_count + i] * (delta - a_data[r * cols_count + j]);
                    acc += jac * grad_data[r * cols_count + j];
                }
                (*in_grad_cpu.storage_buffer)[r * cols_count + i] = acc;
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::softmaxBackward: shape=({}x{}), sample={}",
                                        rows_count, cols_count, formatDataSample(*in_grad_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }

    void sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient = 0.0F, float inv_scale = 1.0F) override
    {
        validateSameDimensions(gradient);
        const auto &grad_data = gradient.getData();

        for (size_t i = 0; i < total_elements; ++i)
        {
            float g = grad_data[i] * inv_scale;
            if (std::isnan(g) || std::isinf(g)) continue;
            if (max_gradient > 0.0F)
            {
                g = std::clamp(g, -max_gradient, max_gradient);
            }
            (*storage_buffer)[i] -= learning_rate * g;
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::sgdUpdate: elements={}, lr={}, max_grad={}, sample={}",
                                        total_elements, learning_rate, max_gradient, formatDataSample(*storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }

    void adamUpdate(const Tensor_Impl &gradient,
                    const Tensor_Impl &first_moment,
                    const Tensor_Impl &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    size_t timestep,
                    float max_gradient = 1.0F,
                    float inv_scale = 1.0F) override
    {
        validateSameDimensions(gradient);
        validateSameDimensions(first_moment);
        validateSameDimensions(second_moment);

        const auto &grad_data = gradient.getData();
        auto &m_cpu = static_cast<Cpu_Tensor_Impl &>(const_cast<Tensor_Impl &>(first_moment));
        auto &v_cpu = static_cast<Cpu_Tensor_Impl &>(const_cast<Tensor_Impl &>(second_moment));

        float bc1 = 1.0F - std::pow(beta1, static_cast<float>(timestep));
        float bc2 = 1.0F - std::pow(beta2, static_cast<float>(timestep));

        for (size_t i = 0; i < total_elements; ++i)
        {
            float raw_g = grad_data[i] * inv_scale;
            if (std::isnan(raw_g) || std::isinf(raw_g)) continue;
            float g = (max_gradient > 0.0F) ? std::clamp(raw_g, -max_gradient, max_gradient) : raw_g;
            (*m_cpu.storage_buffer)[i] = beta1 * (*m_cpu.storage_buffer)[i] + (1.0F - beta1) * g;
            (*v_cpu.storage_buffer)[i] = beta2 * (*v_cpu.storage_buffer)[i] + (1.0F - beta2) * (g * g);

            float m_hat = (*m_cpu.storage_buffer)[i] / bc1;
            float v_hat = (*v_cpu.storage_buffer)[i] / bc2;
            (*storage_buffer)[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::adamUpdate: step={}, lr={}, sample={}",
                                        timestep, learning_rate, formatDataSample(*storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }

    void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        const auto &w_data = weights.getData();
        const auto &b_data = biases.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        size_t rank_a = shape.getRank();
        size_t m_dim = (rank_a >= 2) ? shape[rank_a - 2] : getRows();
        size_t k_dim = (rank_a >= 2) ? shape[rank_a - 1] : getColumns();
        size_t b_dim = (rank_a >= 3) ? (total_elements / (m_dim * k_dim)) : 1;

        size_t rank_w = weights.getShape().getRank();
        size_t k_w = (rank_w >= 2) ? weights.getShape()[rank_w - 2] : weights.getRows();
        size_t n_dim = (rank_w >= 2) ? weights.getShape()[rank_w - 1] : weights.getColumns();
        size_t b_w = (rank_w >= 3) ? (weights.getTotalElements() / (k_w * n_dim)) : 1;

        if (k_dim != k_w)
        {
            throw std::invalid_argument("Matrix inner dimensions must match for multiplication");
        }

        bool broadcast_w = (b_w == 1 && b_dim > 1);
        if (!broadcast_w && b_dim != b_w)
        {
            throw std::invalid_argument("Batch dimensions must match or be broadcastable");
        }

        Shape out_shape;
        if (rank_a <= 2 && rank_w <= 2)
        {
            out_shape = Shape{m_dim, n_dim};
        }
        else if (rank_a == 3)
        {
            out_shape = Shape{b_dim, m_dim, n_dim};
        }
        else if (rank_a >= 4)
        {
            std::vector<size_t> dims(shape.getDimensions().begin(), shape.getDimensions().end());
            dims[rank_a - 2] = m_dim;
            dims[rank_a - 1] = n_dim;
            out_shape = Shape(dims);
        }
        else
        {
            out_shape = Shape{b_dim, m_dim, n_dim};
        }

        output_cpu.reshape(out_shape);

        size_t b_total_elems = biases.getTotalElements();

        for (size_t b = 0; b < b_dim; ++b)
        {
            size_t a_batch_offset = b * m_dim * k_dim;
            size_t w_batch_offset = broadcast_w ? 0 : (b * k_dim * n_dim);
            size_t c_batch_offset = b * m_dim * n_dim;
            size_t b_batch_offset = (b_total_elems >= b_dim * m_dim * n_dim) ? (b * m_dim * n_dim) : 0;

            for (size_t i = 0; i < m_dim; ++i)
            {
                for (size_t j = 0; j < n_dim; ++j)
                {
                    float bias_val = 0.0F;
                    if (b_total_elems == n_dim || biases.getRows() == 1)
                    {
                        bias_val = b_data[j];
                    }
                    else if (b_total_elems == m_dim * n_dim)
                    {
                        bias_val = b_data[i * n_dim + j];
                    }
                    else
                    {
                        bias_val = b_data[b_batch_offset + i * n_dim + j];
                    }
                    (*output_cpu.storage_buffer)[c_batch_offset + i * n_dim + j] = bias_val;
                }

                for (size_t k = 0; k < k_dim; ++k)
                {
                    float in_val = a_data[a_batch_offset + i * k_dim + k];
                    for (size_t j = 0; j < n_dim; ++j)
                    {
                        (*output_cpu.storage_buffer)[c_batch_offset + i * n_dim + j] += in_val * w_data[w_batch_offset + k * n_dim + j];
                    }
                }
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::matmulAdd: batch={}, ({}x{}) x ({}x{}) -> ({}x{}), sample={}",
                                        b_dim, m_dim, k_dim, k_dim, n_dim, m_dim, n_dim, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
    }

    void uploadData(const std::vector<float> &host_data) override
    {
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::uploadData: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }

        if (data_type == Data_Type::FLOAT16)
        {
            if (!storage_buffer_fp16 || storage_buffer_fp16->size() != total_elements)
            {
                storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(total_elements);
            }
            if (isContiguous() && byte_offset == 0)
            {
                convertFp32ToFp16(host_data.data(), storage_buffer_fp16->data(), total_elements);
                return;
            }
            std::vector<float16_t> fp16_data(total_elements);
            convertFp32ToFp16(host_data.data(), fp16_data.data(), total_elements);
            iterateCoordinates([this, &fp16_data](size_t in_idx, size_t dst_idx)
                               { (*storage_buffer_fp16)[dst_idx] = fp16_data[in_idx]; });
            return;
        }

        if (!storage_buffer || storage_buffer->size() != total_elements)
        {
            storage_buffer = std::make_shared<std::vector<float>>(total_elements);
        }

        if (isContiguous() && byte_offset == 0)
        {
            *storage_buffer = host_data;
            return;
        }

        iterateCoordinates([this, &host_data](size_t in_idx, size_t dst_idx)
                           { (*storage_buffer)[dst_idx] = host_data[in_idx]; });
    }

    void conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output,
                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                uint32_t output_channels, uint32_t kernel_size,
                uint32_t stride, uint32_t padding,
                Tensor_Impl *scratch = nullptr) const override
    {
        uint32_t batch_size = static_cast<uint32_t>(getRows());
        uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
        uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

        const auto &in_data = getData();
        const auto &w_data = weights.getData();
        const auto &b_data = biases.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(batch_size, out_h * out_w * output_channels);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t oc = 0; oc < output_channels; ++oc)
                    {
                        float sum = b_data[oc];
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    for (uint32_t ic = 0; ic < input_channels; ++ic)
                                    {
                                        size_t in_idx = n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic;
                                        size_t w_idx = ky * kernel_size * input_channels * output_channels + kx * input_channels * output_channels + ic * output_channels + oc;
                                        sum += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        output_cpu.storage_buffer->at(n * out_h * out_w * output_channels + oh * out_w * output_channels + ow * output_channels + oc) = sum;
                    }
                }
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::conv2d: ({}x{}x{}x{}), sample={}",
                                        batch_size, out_h, out_w, output_channels, formatDataSample(*output_cpu.storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::CONV2D_COMPUTE | Log_Feature::FORWARD_EVALUATION);
    }

    void conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                             uint32_t kernel_size, uint32_t stride, uint32_t padding) const override
    {
        uint32_t batch_size = static_cast<uint32_t>(getRows());
        const auto &out_grad_data = getData();
        const auto &w_data = weights.getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);
        in_grad_cpu.reshape(batch_size, input_height * input_width * input_channels);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t ic = 0; ic < input_channels; ++ic)
                    {
                        float sum = 0.0F;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + padding - ky);
                                int ow_calc = static_cast<int>(iw + padding - kx);
                                if (oh_calc >= 0 && oh_calc % static_cast<int>(stride) == 0 && ow_calc >= 0 && ow_calc % static_cast<int>(stride) == 0)
                                {
                                    uint32_t oh = static_cast<uint32_t>(oh_calc) / stride;
                                    uint32_t ow = static_cast<uint32_t>(ow_calc) / stride;
                                    if (oh < output_height && ow < output_width)
                                    {
                                        for (uint32_t oc = 0; oc < output_channels; ++oc)
                                        {
                                            size_t out_grad_idx = n * output_height * output_width * output_channels + oh * output_width * output_channels + ow * output_channels + oc;
                                            size_t w_idx = ky * kernel_size * input_channels * output_channels + kx * input_channels * output_channels + ic * output_channels + oc;
                                            sum += out_grad_data[out_grad_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        (*in_grad_cpu.storage_buffer)[n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic] = sum;
                    }
                }
            }
        }
    }

    void conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient,
                              uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                              uint32_t kernel_size, uint32_t stride, uint32_t padding,
                              Tensor_Impl * /*im2col_scratch*/ = nullptr) const override
    {
        uint32_t batch_size = static_cast<uint32_t>(getRows());
        const auto &in_data = getData();
        const auto &out_grad_data = output_gradient.getData();
        auto &w_grad_cpu = static_cast<Cpu_Tensor_Impl &>(weight_gradient);
        auto &b_grad_cpu = static_cast<Cpu_Tensor_Impl &>(bias_gradient);

        w_grad_cpu.reshape(1, kernel_size * kernel_size * input_channels * output_channels);
        b_grad_cpu.reshape(1, output_channels);
        std::fill(w_grad_cpu.storage_buffer->begin(), w_grad_cpu.storage_buffer->end(), 0.0F);
        std::fill(b_grad_cpu.storage_buffer->begin(), b_grad_cpu.storage_buffer->end(), 0.0F);

        for (uint32_t oc = 0; oc < output_channels; ++oc)
        {
            for (uint32_t ic = 0; ic < input_channels; ++ic)
            {
                for (uint32_t ky = 0; ky < kernel_size; ++ky)
                {
                    for (uint32_t kx = 0; kx < kernel_size; ++kx)
                    {
                        float w_sum = 0.0F;
                        for (uint32_t n = 0; n < batch_size; ++n)
                        {
                            for (uint32_t oh = 0; oh < output_height; ++oh)
                            {
                                for (uint32_t ow = 0; ow < output_width; ++ow)
                                {
                                    size_t out_grad_idx = n * output_height * output_width * output_channels + oh * output_width * output_channels + ow * output_channels + oc;
                                    float grad_val = out_grad_data[out_grad_idx];
                                    if (ic == 0 && ky == 0 && kx == 0)
                                    {
                                        (*b_grad_cpu.storage_buffer)[oc] += grad_val;
                                    }
                                    int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                    int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                    if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                    {
                                        size_t in_idx = n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic;
                                        w_sum += in_data[in_idx] * grad_val;
                                    }
                                }
                            }
                        }
                        size_t w_idx = ky * kernel_size * input_channels * output_channels + kx * input_channels * output_channels + ic * output_channels + oc;
                        (*w_grad_cpu.storage_buffer)[w_idx] = w_sum;
                    }
                }
            }
        }
    }

    void maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask,
                   uint32_t input_height, uint32_t input_width, uint32_t channels,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding) const override
    {
        auto &res_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        auto &mask_cpu = static_cast<Cpu_Tensor_Impl &>(output_mask);

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
        uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

        const auto &in_data = getData();
        res_cpu.reshape(batch_size, out_h * out_w * channels);
        mask_cpu.reshape(batch_size, out_h * out_w * channels);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float max_val = -3.402823466e+38F;
                        float max_idx = 0.0F;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                                    float val = in_data[in_idx];
                                    if (val > max_val)
                                    {
                                        max_val = val;
                                        max_idx = static_cast<float>(in_idx);
                                    }
                                }
                            }
                        }
                        size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                        (*res_cpu.storage_buffer)[out_idx] = max_val;
                        (*mask_cpu.storage_buffer)[out_idx] = max_idx;
                    }
                }
            }
        }
    }

    void maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient,
                           uint32_t input_height, uint32_t input_width, uint32_t channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) const override
    {
        const auto &mask_data = mask.getData();
        const auto &out_grad_data = getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        in_grad_cpu.reshape(batch_size, input_height * input_width * channels);
        std::fill(in_grad_cpu.storage_buffer->begin(), in_grad_cpu.storage_buffer->end(), 0.0F);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                        float acc_grad = 0.0F;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + padding - ky);
                                int ow_calc = static_cast<int>(iw + padding - kx);
                                if (oh_calc >= 0 && oh_calc % static_cast<int>(stride) == 0 && ow_calc >= 0 && ow_calc % static_cast<int>(stride) == 0)
                                {
                                    uint32_t oh = static_cast<uint32_t>(oh_calc) / stride;
                                    uint32_t ow = static_cast<uint32_t>(ow_calc) / stride;
                                    if (oh < output_height && ow < output_width)
                                    {
                                        size_t out_idx = n * output_height * output_width * channels + oh * output_width * channels + ow * channels + c;
                                        if (static_cast<size_t>(mask_data[out_idx]) == in_idx)
                                        {
                                            acc_grad += out_grad_data[out_idx];
                                        }
                                    }
                                }
                            }
                        }
                        (*in_grad_cpu.storage_buffer)[in_idx] = acc_grad;
                    }
                }
            }
        }
    }

    void globalAvgPool2d(Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const override
    {
        uint32_t batch_size = static_cast<uint32_t>(getRows());
        const auto &in_data = getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(batch_size, channels);
        float area = static_cast<float>(input_height * input_width);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                float sum = 0.0F;
                for (uint32_t h = 0; h < input_height; ++h)
                {
                    for (uint32_t w = 0; w < input_width; ++w)
                    {
                        sum += in_data[n * input_height * input_width * channels + h * input_width * channels + w * channels + c];
                    }
                }
                (*output_cpu.storage_buffer)[n * channels + c] = sum / area;
            }
        }
    }

    void globalAvgPool2dBackward(Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const override
    {
        uint32_t batch_size = static_cast<uint32_t>(getRows());
        const auto &out_grad_data = getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);
        in_grad_cpu.reshape(batch_size, input_height * input_width * channels);
        float area = static_cast<float>(input_height * input_width);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                float scaled = out_grad_data[n * channels + c] / area;
                for (uint32_t h = 0; h < input_height; ++h)
                {
                    for (uint32_t w = 0; w < input_width; ++w)
                    {
                        (*in_grad_cpu.storage_buffer)[n * input_height * input_width * channels + h * input_width * channels + w * channels + c] = scaled;
                    }
                }
            }
        }
    }

    void batchNormForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                          Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                          Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                          Tensor_Impl &normalized_input, Tensor_Impl &output,
                          float epsilon, float momentum, bool is_training) const override
    {
        size_t b_count = getRows();
        size_t f_dim = getColumns();
        const auto &in_data = getData();
        const auto &gamma_data = gamma.getData();
        const auto &beta_data = beta.getData();

        auto &rm_cpu = static_cast<Cpu_Tensor_Impl &>(running_mean);
        auto &rv_cpu = static_cast<Cpu_Tensor_Impl &>(running_variance);
        auto &norm_in_cpu = static_cast<Cpu_Tensor_Impl &>(normalized_input);
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        out_cpu.reshape(b_count, f_dim);
        norm_in_cpu.reshape(b_count, f_dim);

        if (is_training)
        {
            auto &bm_cpu = static_cast<Cpu_Tensor_Impl &>(batch_mean);
            auto &bv_cpu = static_cast<Cpu_Tensor_Impl &>(batch_variance);
            bm_cpu.reshape(1, f_dim);
            bv_cpu.reshape(1, f_dim);
            std::fill(bm_cpu.storage_buffer->begin(), bm_cpu.storage_buffer->end(), 0.0F);
            std::fill(bv_cpu.storage_buffer->begin(), bv_cpu.storage_buffer->end(), 0.0F);

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    (*bm_cpu.storage_buffer)[j] += in_data[i * f_dim + j];
                }
            }
            float inv_b = 1.0F / static_cast<float>(b_count);
            for (size_t j = 0; j < f_dim; ++j)
            {
                (*bm_cpu.storage_buffer)[j] *= inv_b;
            }

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float diff = in_data[i * f_dim + j] - (*bm_cpu.storage_buffer)[j];
                    (*bv_cpu.storage_buffer)[j] += diff * diff;
                }
            }
            for (size_t j = 0; j < f_dim; ++j)
            {
                (*bv_cpu.storage_buffer)[j] *= inv_b;
                (*rm_cpu.storage_buffer)[j] = (1.0F - momentum) * (*rm_cpu.storage_buffer)[j] + momentum * (*bm_cpu.storage_buffer)[j];
                (*rv_cpu.storage_buffer)[j] = (1.0F - momentum) * (*rv_cpu.storage_buffer)[j] + momentum * (*bv_cpu.storage_buffer)[j];
            }

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float norm_val = (in_data[i * f_dim + j] - (*bm_cpu.storage_buffer)[j]) / std::sqrt((*bv_cpu.storage_buffer)[j] + epsilon);
                    (*norm_in_cpu.storage_buffer)[i * f_dim + j] = norm_val;
                    (*out_cpu.storage_buffer)[i * f_dim + j] = gamma_data[j] * norm_val + beta_data[j];
                }
            }
        }
        else
        {
            const auto &rm_data = running_mean.getData();
            const auto &rv_data = running_variance.getData();

            for (size_t i = 0; i < b_count; ++i)
            {
                for (size_t j = 0; j < f_dim; ++j)
                {
                    float norm_val = (in_data[i * f_dim + j] - rm_data[j]) / std::sqrt(rv_data[j] + epsilon);
                    (*norm_in_cpu.storage_buffer)[i * f_dim + j] = norm_val;
                    (*out_cpu.storage_buffer)[i * f_dim + j] = gamma_data[j] * norm_val + beta_data[j];
                }
            }
        }
    }

    void batchNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                           Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, float epsilon) const override
    {
        size_t b_count = getRows();
        size_t f_dim = getColumns();

        const auto &out_grad_data = output_gradient.getData();
        const auto &gamma_data = gamma.getData();
        const auto &bv_data = batch_variance.getData();
        const auto &norm_in_data = normalized_input.getData();

        auto &g_grad_cpu = static_cast<Cpu_Tensor_Impl &>(gamma_gradient);
        auto &b_grad_cpu = static_cast<Cpu_Tensor_Impl &>(beta_gradient);
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);

        g_grad_cpu.reshape(1, f_dim);
        b_grad_cpu.reshape(1, f_dim);
        in_grad_cpu.reshape(b_count, f_dim);

        std::fill(g_grad_cpu.storage_buffer->begin(), g_grad_cpu.storage_buffer->end(), 0.0F);
        std::fill(b_grad_cpu.storage_buffer->begin(), b_grad_cpu.storage_buffer->end(), 0.0F);

        for (size_t i = 0; i < b_count; ++i)
        {
            for (size_t j = 0; j < f_dim; ++j)
            {
                size_t idx = i * f_dim + j;
                float go = out_grad_data[idx];
                (*g_grad_cpu.storage_buffer)[j] += go * norm_in_data[idx];
                (*b_grad_cpu.storage_buffer)[j] += go;
            }
        }

        float inv_b = 1.0F / static_cast<float>(b_count);
        for (size_t j = 0; j < f_dim; ++j)
        {
            float inv_std = 1.0F / std::sqrt(bv_data[j] + epsilon);
            float coeff = gamma_data[j] * inv_std * inv_b;

            for (size_t i = 0; i < b_count; ++i)
            {
                size_t idx = i * f_dim + j;
                (*in_grad_cpu.storage_buffer)[idx] = coeff * (static_cast<float>(b_count) * out_grad_data[idx] - (*b_grad_cpu.storage_buffer)[j] - norm_in_data[idx] * (*g_grad_cpu.storage_buffer)[j]);
            }
        }
    }

    void linearForward(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override
    {
        matmulAdd(weights, biases, output);
    }

    void linearBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient) const override
    {
        const auto &w_data = weights.getData();
        const auto &out_grad_data = getData();
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);

        size_t b_size = getRows();
        size_t out_dim = getColumns();
        size_t in_dim = weights.getRows();
        in_grad_cpu.reshape(b_size, in_dim);

        for (size_t i = 0; i < b_size; ++i)
        {
            for (size_t j = 0; j < in_dim; ++j)
            {
                float sum = 0.0F;
                for (size_t k = 0; k < out_dim; ++k)
                {
                    sum += out_grad_data[i * out_dim + k] * w_data[j * out_dim + k];
                }
                (*in_grad_cpu.storage_buffer)[i * in_dim + j] = sum;
            }
        }
    }

    void linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient) const override
    {
        const auto &in_data = getData();
        const auto &out_grad_data = output_gradient.getData();
        auto &w_grad_cpu = static_cast<Cpu_Tensor_Impl &>(weight_gradient);
        auto &b_grad_cpu = static_cast<Cpu_Tensor_Impl &>(bias_gradient);

        size_t b_size = getRows();
        size_t in_dim = getColumns();
        size_t out_dim = output_gradient.getColumns();

        w_grad_cpu.reshape(in_dim, out_dim);
        b_grad_cpu.reshape(1, out_dim);
        std::fill(w_grad_cpu.storage_buffer->begin(), w_grad_cpu.storage_buffer->end(), 0.0F);
        std::fill(b_grad_cpu.storage_buffer->begin(), b_grad_cpu.storage_buffer->end(), 0.0F);

        for (size_t i = 0; i < b_size; ++i)
        {
            for (size_t j = 0; j < out_dim; ++j)
            {
                float grad_val = out_grad_data[i * out_dim + j];
                (*b_grad_cpu.storage_buffer)[j] += grad_val;
                for (size_t k = 0; k < in_dim; ++k)
                {
                    (*w_grad_cpu.storage_buffer)[k * out_dim + j] += in_data[i * in_dim + k] * grad_val;
                }
            }
        }
    }

    void batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                            Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                            Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                            Tensor_Impl &normalized_input, Tensor_Impl &output,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const override
    {
        const auto &in_data = getData();
        const auto &gamma_data = gamma.getData();
        const auto &beta_data = beta.getData();
        auto &rm_cpu = static_cast<Cpu_Tensor_Impl &>(running_mean);
        auto &rv_cpu = static_cast<Cpu_Tensor_Impl &>(running_variance);
        auto &bm_cpu = static_cast<Cpu_Tensor_Impl &>(batch_mean);
        auto &bv_cpu = static_cast<Cpu_Tensor_Impl &>(batch_variance);
        auto &norm_in_cpu = static_cast<Cpu_Tensor_Impl &>(normalized_input);
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        size_t b_size = getRows();
        size_t total_feat = input_height * input_width * input_channels;
        uint32_t sp_count = static_cast<uint32_t>(b_size * input_height * input_width);

        out_cpu.reshape(b_size, total_feat);
        norm_in_cpu.reshape(b_size, total_feat);

        if (is_training)
        {
            bm_cpu.reshape(1, input_channels);
            bv_cpu.reshape(1, input_channels);
            std::fill(bm_cpu.storage_buffer->begin(), bm_cpu.storage_buffer->end(), 0.0F);
            std::fill(bv_cpu.storage_buffer->begin(), bv_cpu.storage_buffer->end(), 0.0F);

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    (*bm_cpu.storage_buffer)[c] += in_data[i * input_channels + c];
                }
            }
            float inv_sp = 1.0F / static_cast<float>(sp_count);
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                (*bm_cpu.storage_buffer)[c] *= inv_sp;
            }

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    float diff = in_data[i * input_channels + c] - (*bm_cpu.storage_buffer)[c];
                    (*bv_cpu.storage_buffer)[c] += diff * diff;
                }
            }
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                (*bv_cpu.storage_buffer)[c] *= inv_sp;
                (*rm_cpu.storage_buffer)[c] = (1.0F - momentum) * (*rm_cpu.storage_buffer)[c] + momentum * (*bm_cpu.storage_buffer)[c];
                (*rv_cpu.storage_buffer)[c] = (1.0F - momentum) * (*rv_cpu.storage_buffer)[c] + momentum * (*bv_cpu.storage_buffer)[c];
            }
            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float norm_val = (in_data[idx] - (*bm_cpu.storage_buffer)[c]) / std::sqrt((*bv_cpu.storage_buffer)[c] + epsilon);
                    (*norm_in_cpu.storage_buffer)[idx] = norm_val;
                    (*out_cpu.storage_buffer)[idx] = gamma_data[c] * norm_val + beta_data[c];
                }
            }
        }
        else
        {
            const auto &rm_data = running_mean.getData();
            const auto &rv_data = running_variance.getData();

            for (size_t i = 0; i < sp_count; ++i)
            {
                for (uint32_t c = 0; c < input_channels; ++c)
                {
                    size_t idx = i * input_channels + c;
                    float norm_val = (in_data[idx] - rm_data[c]) / std::sqrt(rv_data[c] + epsilon);
                    (*norm_in_cpu.storage_buffer)[idx] = norm_val;
                    (*out_cpu.storage_buffer)[idx] = gamma_data[c] * norm_val + beta_data[c];
                }
            }
        }
    }

    void batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                             Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const override
    {
        const auto &out_grad_data = getData();
        const auto &gamma_data = gamma.getData();
        const auto &bv_data = batch_variance.getData();
        const auto &norm_in_data = normalized_input.getData();
        auto &g_grad_cpu = static_cast<Cpu_Tensor_Impl &>(gamma_gradient);
        auto &b_grad_cpu = static_cast<Cpu_Tensor_Impl &>(beta_gradient);
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);

        size_t b_size = getRows();
        size_t total_feat = input_height * input_width * input_channels;
        uint32_t sp_count = static_cast<uint32_t>(b_size * input_height * input_width);

        in_grad_cpu.reshape(b_size, total_feat);
        g_grad_cpu.reshape(1, input_channels);
        b_grad_cpu.reshape(1, input_channels);
        std::fill(g_grad_cpu.storage_buffer->begin(), g_grad_cpu.storage_buffer->end(), 0.0F);
        std::fill(b_grad_cpu.storage_buffer->begin(), b_grad_cpu.storage_buffer->end(), 0.0F);

        for (size_t i = 0; i < sp_count; ++i)
        {
            for (uint32_t c = 0; c < input_channels; ++c)
            {
                size_t idx = i * input_channels + c;
                float go = out_grad_data[idx];
                (*g_grad_cpu.storage_buffer)[c] += go * norm_in_data[idx];
                (*b_grad_cpu.storage_buffer)[c] += go;
            }
        }

        float inv_sp = 1.0F / static_cast<float>(sp_count);
        for (uint32_t c = 0; c < input_channels; ++c)
        {
            float inv_std = 1.0F / std::sqrt(bv_data[c] + epsilon);
            float coeff = gamma_data[c] * inv_std * inv_sp;

            for (size_t i = 0; i < sp_count; ++i)
            {
                size_t idx = i * input_channels + c;
                (*in_grad_cpu.storage_buffer)[idx] = coeff * (static_cast<float>(sp_count) * out_grad_data[idx] - (*b_grad_cpu.storage_buffer)[c] - norm_in_data[idx] * (*g_grad_cpu.storage_buffer)[c]);
            }
        }
    }

    void cceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override
    {
        validateSameDimensions(target);
        const auto &a_data = getData();
        const auto &t_data = target.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(1, 1);

        float loss = 0.0F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0F - epsilon);
            loss += -t_data[i] * std::log(p);
        }
        (*output_cpu.storage_buffer)[0] = loss;
    }

    void mseLoss(const Tensor_Impl &target, Tensor_Impl &output) const override
    {
        validateSameDimensions(target);
        const auto &a_data = getData();
        const auto &t_data = target.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(1, 1);

        float loss = 0.0F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            float diff = a_data[i] - t_data[i];
            loss += diff * diff;
        }
        (*output_cpu.storage_buffer)[0] = loss;
    }

    void maeLoss(const Tensor_Impl &target, Tensor_Impl &output) const override
    {
        validateSameDimensions(target);
        const auto &a_data = getData();
        const auto &t_data = target.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(1, 1);

        float loss = 0.0F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            loss += std::abs(a_data[i] - t_data[i]);
        }
        (*output_cpu.storage_buffer)[0] = loss;
    }

    void bceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override
    {
        validateSameDimensions(target);
        const auto &a_data = getData();
        const auto &t_data = target.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(1, 1);

        float loss = 0.0F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            float p = std::clamp(a_data[i], epsilon, 1.0F - epsilon);
            float y = t_data[i];
            loss += -(y * std::log(p) + (1.0F - y) * std::log(1.0F - p));
        }
        (*output_cpu.storage_buffer)[0] = loss;
    }

    void huberLoss(const Tensor_Impl &target, Tensor_Impl &output, float delta) const override
    {
        validateSameDimensions(target);
        const auto &a_data = getData();
        const auto &t_data = target.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        output_cpu.reshape(1, 1);

        float loss = 0.0F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            float diff = std::abs(a_data[i] - t_data[i]);
            loss += (diff <= delta) ? 0.5F * diff * diff : delta * (diff - 0.5F * delta);
        }
        (*output_cpu.storage_buffer)[0] = loss;
    }

    void concatenateCollumns(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        if (getRows() != other.getRows())
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::concatenateCollumns: Row mismatch: {} vs {}",
                                            getRows(), other.getRows()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Row count mismatch in concatenateCollumns");
        }

        size_t cols_a = getColumns();
        size_t cols_b = other.getColumns();
        size_t tot_cols = cols_a + cols_b;
        output_cpu.reshape(getRows(), tot_cols);

        for (size_t r = 0; r < getRows(); ++r)
        {
            std::copy_n(a_data.data() + r * cols_a, cols_a, output_cpu.storage_buffer->data() + r * tot_cols);
            std::copy_n(b_data.data() + r * cols_b, cols_b, output_cpu.storage_buffer->data() + r * tot_cols + cols_a);
        }
    }

    void concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        if (getColumns() != other.getColumns())
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::concatenateRows: Column mismatch: {} vs {}",
                                            getColumns(), other.getColumns()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Column count mismatch in concatenateRows");
        }

        size_t tot_rows = getRows() + other.getRows();
        output_cpu.reshape(tot_rows, getColumns());

        std::copy(a_data.begin(), a_data.end(), output_cpu.storage_buffer->begin());
        std::copy(b_data.begin(), b_data.end(), output_cpu.storage_buffer->begin() + a_data.size());
    }

    void splitCollumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override
    {
        if (split_index == 0 || split_index >= getColumns())
        {
            throw std::out_of_range("Split index out of range in splitCollumns");
        }

        const auto &a_data = getData();
        auto &left_cpu = static_cast<Cpu_Tensor_Impl &>(result_left);
        auto &right_cpu = static_cast<Cpu_Tensor_Impl &>(result_right);
        size_t cols_left = split_index;
        size_t cols_right = getColumns() - split_index;

        left_cpu.reshape(getRows(), cols_left);
        right_cpu.reshape(getRows(), cols_right);

        for (size_t r = 0; r < getRows(); ++r)
        {
            std::copy_n(a_data.data() + r * getColumns(), cols_left, left_cpu.storage_buffer->data() + r * cols_left);
            std::copy_n(a_data.data() + r * getColumns() + cols_left, cols_right, right_cpu.storage_buffer->data() + r * cols_right);
        }
    }

    void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override
    {
        if (split_index == 0 || split_index >= getRows())
        {
            throw std::out_of_range("Split index out of range in splitRows");
        }

        const auto &a_data = getData();
        auto &up_cpu = static_cast<Cpu_Tensor_Impl &>(result_up);
        auto &down_cpu = static_cast<Cpu_Tensor_Impl &>(result_down);
        size_t rows_up = split_index;
        size_t rows_down = getRows() - split_index;

        up_cpu.reshape(rows_up, getColumns());
        down_cpu.reshape(rows_down, getColumns());

        size_t up_elems = rows_up * getColumns();
        std::copy_n(a_data.data(), up_elems, up_cpu.storage_buffer->data());
        std::copy_n(a_data.data() + up_elems, rows_down * getColumns(), down_cpu.storage_buffer->data());
    }

    const std::vector<float> &getData() const noexcept override
    {
        if (data_type == Data_Type::FLOAT16)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            materialized_cache.resize(total_elements);
            if (storage_buffer_fp16)
            {
                if (isContiguous() && byte_offset == 0)
                {
                    convertFp16ToFp32(storage_buffer_fp16->data(), materialized_cache.data(), total_elements);
                }
                else
                {
                    iterateCoordinates([this](size_t out_idx, size_t src_idx)
                                       { materialized_cache[out_idx] = static_cast<float>((*storage_buffer_fp16)[src_idx]); });
                }
            }
            return materialized_cache;
        }

        if (isContiguous() && byte_offset == 0)
        {
            if (storage_buffer)
            {
                return *storage_buffer;
            }
        }

        std::lock_guard<std::mutex> lock(cache_mutex);
        materialized_cache.resize(total_elements);
        if (storage_buffer)
        {
            iterateCoordinates([this](size_t out_idx, size_t src_idx)
                               { materialized_cache[out_idx] = (*storage_buffer)[src_idx]; });
        }
        return materialized_cache;
    }

    Mutable_Storage_Handle getStorage() override
    {
        if (!isContiguous() || byte_offset != 0)
        {
            auto owned_data = getData();
            storage_buffer = std::make_shared<std::vector<float>>(std::move(owned_data));
            byte_offset = 0;
            strides = shape.computeContiguousStrides();
        }
        return std::ref(*storage_buffer);
    }

    Storage_Handle getStorage() const override { return std::cref(getData()); }
    const std::shared_ptr<std::vector<float16_t>> &getStorageBufferFp16() const noexcept { return storage_buffer_fp16; }
    std::shared_ptr<std::vector<float16_t>> &getStorageBufferFp16() noexcept { return storage_buffer_fp16; }
    const std::shared_ptr<std::vector<float>> &getStorageBuffer() const noexcept { return storage_buffer; }
    std::shared_ptr<std::vector<float>> &getStorageBuffer() noexcept { return storage_buffer; }
    bool isEmpty() const noexcept override { return (data_type == Data_Type::FLOAT16) ? (!storage_buffer_fp16 || storage_buffer_fp16->empty()) : (!storage_buffer || storage_buffer->empty()); }

    void setStorageBufferFp16(std::shared_ptr<std::vector<float16_t>> _buf) noexcept { storage_buffer_fp16 = std::move(_buf); }
    void setStorageBuffer(std::shared_ptr<std::vector<float>> _storage_buffer) noexcept { storage_buffer = std::move(_storage_buffer); }
};

using Cpu_Matrix_Impl = Cpu_Tensor_Impl;
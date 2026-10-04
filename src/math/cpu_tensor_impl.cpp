#include "math/cpu_tensor_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

#include "helper/logger.h"


std::string Cpu_Tensor_Impl::formatDataSample(const std::vector<float> &data, size_t sample_limit)
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

size_t Cpu_Tensor_Impl::resolveFlatIndex(std::span<const size_t> indices) const noexcept
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
    void Cpu_Tensor_Impl::iterateCoordinates(Op &&op) const
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

Cpu_Tensor_Impl::Cpu_Tensor_Impl(size_t rows, size_t columns)
{
        updateShapeAndStrides(Shape{rows, columns});
        storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
    }

Cpu_Tensor_Impl::Cpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data)
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

Cpu_Tensor_Impl::Cpu_Tensor_Impl(size_t rows, size_t columns, std::vector<float> &&host_data)
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

Cpu_Tensor_Impl::Cpu_Tensor_Impl(Shape tensor_shape)
{
        updateShapeAndStrides(tensor_shape);
        storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
    }

Cpu_Tensor_Impl::Cpu_Tensor_Impl(Shape tensor_shape, Data_Type type)
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

Cpu_Tensor_Impl::Cpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data)
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

Cpu_Tensor_Impl::Cpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<std::vector<float>> buffer, size_t offset_elements)
{
        shape = tensor_shape;
        strides = tensor_strides;
        storage_buffer = std::move(buffer);
        byte_offset = offset_elements * getDataTypeSize(data_type);
        total_elements = shape.getTotalElements();
    }

void Cpu_Tensor_Impl::reshape(size_t rows, size_t columns)
{
        reshape(Shape{rows, columns});
    }

void Cpu_Tensor_Impl::reshape(Shape new_shape)
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

void Cpu_Tensor_Impl::permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::updateSlice(size_t axis, size_t start, const Tensor_Impl &source)
{
        const auto &source_cpu = static_cast<const Cpu_Tensor_Impl &>(source);
        if (axis >= shape.getRank())
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::updateSlice: axis out of range");
        }
        size_t src_axis_len = source_cpu.shape[axis];
        if (start + src_axis_len > shape[axis])
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::updateSlice: slice range exceeds destination dimension");
        }

        size_t elem_size = getDataTypeSize(data_type);
        size_t outer_count = 1;
        for (size_t i = 0; i < axis; ++i) outer_count *= shape[i];
        size_t inner_count = 1;
        for (size_t i = axis + 1; i < shape.getRank(); ++i) inner_count *= shape[i];
        size_t this_axis_len = shape[axis];
        size_t copy_bytes = src_axis_len * inner_count * elem_size;

        if (data_type == Data_Type::FLOAT16 && storage_buffer_fp16 && source_cpu.storage_buffer_fp16)
        {
            uint8_t *dst_base = reinterpret_cast<uint8_t *>(storage_buffer_fp16->data()) + byte_offset;
            const uint8_t *src_base = reinterpret_cast<const uint8_t *>(source_cpu.storage_buffer_fp16->data()) + source_cpu.byte_offset;
            for (size_t outer = 0; outer < outer_count; ++outer)
            {
                size_t dst_off = (outer * this_axis_len + start) * inner_count * elem_size;
                size_t src_off = (outer * src_axis_len) * inner_count * elem_size;
                std::memcpy(dst_base + dst_off, src_base + src_off, copy_bytes);
            }
        }
        else if (storage_buffer && source_cpu.storage_buffer)
        {
            uint8_t *dst_base = reinterpret_cast<uint8_t *>(storage_buffer->data()) + byte_offset;
            const uint8_t *src_base = reinterpret_cast<const uint8_t *>(source_cpu.storage_buffer->data()) + source_cpu.byte_offset;
            for (size_t outer = 0; outer < outer_count; ++outer)
            {
                size_t dst_off = (outer * this_axis_len + start) * inner_count * elem_size;
                size_t src_off = (outer * src_axis_len) * inner_count * elem_size;
                std::memcpy(dst_base + dst_off, src_base + src_off, copy_bytes);
            }
        }
    }

void Cpu_Tensor_Impl::gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const
{
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        size_t D = getColumns();
        size_t elem_size = getDataTypeSize(data_type);
        size_t S = indices.size();
        out_cpu.setDataType(data_type);
        out_cpu.reshape(Shape{ S, D });

        if (data_type == Data_Type::FLOAT16)
        {
            if (!out_cpu.storage_buffer_fp16 || out_cpu.storage_buffer_fp16->size() != S * D)
            {
                out_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(S * D);
            }
            out_cpu.storage_buffer.reset();
        }
        else
        {
            if (!out_cpu.storage_buffer || out_cpu.storage_buffer->size() != S * D)
            {
                out_cpu.storage_buffer = std::make_shared<std::vector<float>>(S * D);
            }
            out_cpu.storage_buffer_fp16.reset();
        }

        for (size_t s = 0; s < S; ++s)
        {
            int32_t row = indices[s];
            if (row < 0 || static_cast<size_t>(row) >= shape[0]) row = 0;
            size_t src_off = byte_offset + row * D * elem_size;
            size_t dst_off = out_cpu.byte_offset + s * D * elem_size;
            if (data_type == Data_Type::FLOAT16 && storage_buffer_fp16 && out_cpu.storage_buffer_fp16)
            {
                uint8_t *dst = reinterpret_cast<uint8_t *>(out_cpu.storage_buffer_fp16->data()) + dst_off;
                const uint8_t *src = reinterpret_cast<const uint8_t *>(storage_buffer_fp16->data()) + src_off;
                std::memcpy(dst, src, D * elem_size);
            }
            else if (storage_buffer && out_cpu.storage_buffer)
            {
                uint8_t *dst = reinterpret_cast<uint8_t *>(out_cpu.storage_buffer->data()) + dst_off;
                const uint8_t *src = reinterpret_cast<const uint8_t *>(storage_buffer->data()) + src_off;
                std::memcpy(dst, src, D * elem_size);
            }
        }
    }

void Cpu_Tensor_Impl::contiguous(Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::to(Data_Type target_type, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::matmul(const Tensor_Impl &other, Tensor_Impl &output) const
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
            throw std::invalid_argument("Tensor inner dimensions must match for multiplication");
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

void Cpu_Tensor_Impl::matdiv(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Cpu_Tensor_Impl temp_inverse(0, 0);
        other.inverse(temp_inverse);
        matmul(temp_inverse, output);
    }

void Cpu_Tensor_Impl::add(const Tensor_Impl &other, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::sub(const Tensor_Impl &other, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::mulScalar(float scalar, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::divScalar(float scalar, Tensor_Impl &output) const
{
        if (std::abs(scalar) < 1e-8F)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::divScalar: Division by zero encountered"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        mulScalar(1.0F / scalar, output);
    }

void Cpu_Tensor_Impl::hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::transpose(Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::inverse(Tensor_Impl &output) const
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
                throw std::runtime_error("Tensor is singular");
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

void Cpu_Tensor_Impl::normalize(Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::relu(Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::reluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const
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

void Cpu_Tensor_Impl::gelu(Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::geluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const
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

void Cpu_Tensor_Impl::softmax(Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::softmaxBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const
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

void Cpu_Tensor_Impl::sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient, float inv_scale)
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

void Cpu_Tensor_Impl::adamUpdate(const Tensor_Impl &gradient, const Tensor_Impl &first_moment, const Tensor_Impl &second_moment, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay)
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
            float step_val = m_hat / (std::sqrt(v_hat) + epsilon);
            if (weight_decay > 0.0F)
            {
                (*storage_buffer)[i] -= learning_rate * (step_val + weight_decay * (*storage_buffer)[i]);
            }
            else
            {
                (*storage_buffer)[i] -= learning_rate * step_val;
            }
        }

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::adamUpdate: step={}, lr={}, sample={}",
                                        timestep, learning_rate, formatDataSample(*storage_buffer)},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::OPTIMIZER_STEP);
    }

void Cpu_Tensor_Impl::matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const
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
            throw std::invalid_argument("Tensor inner dimensions must match for multiplication");
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

void Cpu_Tensor_Impl::uploadData(const std::vector<float> &host_data)
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

void Cpu_Tensor_Impl::zero()
{
        if (data_type == Data_Type::FLOAT16)
        {
            if (storage_buffer_fp16)
            {
                std::fill(storage_buffer_fp16->begin(), storage_buffer_fp16->end(), 0.0f);
            }
        }
        else
        {
            if (storage_buffer)
            {
                std::fill(storage_buffer->begin(), storage_buffer->end(), 0.0f);
            }
        }
    }

void Cpu_Tensor_Impl::fill(float value)
{
        if (data_type == Data_Type::FLOAT16)
        {
            if (storage_buffer_fp16)
            {
                std::fill(storage_buffer_fp16->begin(), storage_buffer_fp16->end(), value);
            }
        }
        else
        {
            if (storage_buffer)
            {
                std::fill(storage_buffer->begin(), storage_buffer->end(), value);
            }
        }
    }

void Cpu_Tensor_Impl::conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor_Impl *scratch) const
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

void Cpu_Tensor_Impl::conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
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

void Cpu_Tensor_Impl::conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, uint32_t output_height, uint32_t output_width, uint32_t output_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, Tensor_Impl * /*im2col_scratch*/) const
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

void Cpu_Tensor_Impl::maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
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

void Cpu_Tensor_Impl::maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels, uint32_t output_height, uint32_t output_width, uint32_t kernel_size, uint32_t stride, uint32_t padding) const
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

void Cpu_Tensor_Impl::globalAvgPool2d(Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const
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

void Cpu_Tensor_Impl::globalAvgPool2dBackward(Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const
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

void Cpu_Tensor_Impl::batchNormForward(const Tensor_Impl &gamma, const Tensor_Impl &beta, Tensor_Impl &running_mean, Tensor_Impl &running_variance, Tensor_Impl &batch_mean, Tensor_Impl &batch_variance, Tensor_Impl &normalized_input, Tensor_Impl &output, float epsilon, float momentum, bool is_training) const
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

void Cpu_Tensor_Impl::batchNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input, Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, float epsilon) const
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

void Cpu_Tensor_Impl::linearForward(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const
{
        matmulAdd(weights, biases, output);
    }

void Cpu_Tensor_Impl::linearBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient) const
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

void Cpu_Tensor_Impl::linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient, bool accumulate) const
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
        if (!accumulate)
        {
            std::fill(w_grad_cpu.storage_buffer->begin(), w_grad_cpu.storage_buffer->end(), 0.0F);
            std::fill(b_grad_cpu.storage_buffer->begin(), b_grad_cpu.storage_buffer->end(), 0.0F);
        }

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

void Cpu_Tensor_Impl::linearBackwardWeightAdam(const Tensor_Impl &output_gradient, Tensor_Impl &weights, Tensor_Impl &first_moment, Tensor_Impl &second_moment, Tensor_Impl &bias_gradient, float learning_rate, float beta1, float beta2, float epsilon, size_t timestep, float max_gradient, float inv_scale, float weight_decay)
{
        Cpu_Tensor_Impl temp_w_grad(weights.getRows(), weights.getColumns());
        linearBackwardWeightBias(output_gradient, temp_w_grad, bias_gradient, false);
        weights.adamUpdate(temp_w_grad, first_moment, second_moment, learning_rate, beta1, beta2, epsilon, timestep, max_gradient, inv_scale, weight_decay);
    }

void Cpu_Tensor_Impl::batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta, Tensor_Impl &running_mean, Tensor_Impl &running_variance, Tensor_Impl &batch_mean, Tensor_Impl &batch_variance, Tensor_Impl &normalized_input, Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon, float momentum, bool is_training) const
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

void Cpu_Tensor_Impl::batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input, Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const
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

void Cpu_Tensor_Impl::cceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const
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

void Cpu_Tensor_Impl::mseLoss(const Tensor_Impl &target, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::maeLoss(const Tensor_Impl &target, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::bceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const
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

void Cpu_Tensor_Impl::huberLoss(const Tensor_Impl &target, Tensor_Impl &output, float delta) const
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

void Cpu_Tensor_Impl::concatenateColumns(const Tensor_Impl &other, Tensor_Impl &output) const
{
        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        if (getRows() != other.getRows())
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::concatenateColumns: Row mismatch: {} vs {}",
                                            getRows(), other.getRows()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Row count mismatch in concatenateColumns");
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

void Cpu_Tensor_Impl::concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const
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

void Cpu_Tensor_Impl::splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const
{
        if (split_index == 0 || split_index >= getColumns())
        {
            throw std::out_of_range("Split index out of range in splitColumns");
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

void Cpu_Tensor_Impl::splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const
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

const std::vector<float> & Cpu_Tensor_Impl::getData() const noexcept
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

Mutable_Storage_Handle Cpu_Tensor_Impl::getStorage()
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

void Cpu_Tensor_Impl::rmsNormForward(const Tensor_Impl &gamma, Tensor_Impl &inv_rms, Tensor_Impl &output, float epsilon) const
{
        size_t b_count = getRows();
        size_t f_dim = getColumns();
        const auto &in_data = getData();
        const auto &gamma_data = gamma.getData();

        auto &inv_rms_cpu = static_cast<Cpu_Tensor_Impl &>(inv_rms);
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);

        inv_rms_cpu.reshape(b_count, 1);
        out_cpu.reshape(b_count, f_dim);

        for (size_t i = 0; i < b_count; ++i)
        {
            float sum_sq = 0.0f;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float val = in_data[i * f_dim + j];
                sum_sq += val * val;
            }
            float mean_sq = sum_sq / static_cast<float>(f_dim);
            float inv_val = 1.0f / std::sqrt(mean_sq + epsilon);
            (*inv_rms_cpu.storage_buffer)[i] = inv_val;

            for (size_t j = 0; j < f_dim; ++j)
            {
                (*out_cpu.storage_buffer)[i * f_dim + j] = in_data[i * f_dim + j] * inv_val * gamma_data[j];
            }
        }
    }

void Cpu_Tensor_Impl::rmsNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &inv_rms, Tensor_Impl &gamma_gradient, Tensor_Impl &input_gradient, bool accumulate_gamma) const
{
        size_t b_count = getRows();
        size_t f_dim = getColumns();
        const auto &in_data = getData();
        const auto &out_grad = output_gradient.getData();
        const auto &gamma_data = gamma.getData();
        const auto &inv_rms_data = inv_rms.getData();

        auto &g_grad_cpu = static_cast<Cpu_Tensor_Impl &>(gamma_gradient);
        auto &in_grad_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);

        g_grad_cpu.reshape(1, f_dim);
        in_grad_cpu.reshape(b_count, f_dim);

        if (!accumulate_gamma)
        {
            std::fill(g_grad_cpu.storage_buffer->begin(), g_grad_cpu.storage_buffer->end(), 0.0f);
        }

        for (size_t j = 0; j < f_dim; ++j)
        {
            float sum_dg = 0.0f;
            for (size_t i = 0; i < b_count; ++i)
            {
                sum_dg += out_grad[i * f_dim + j] * in_data[i * f_dim + j] * inv_rms_data[i];
            }
            (*g_grad_cpu.storage_buffer)[j] += sum_dg;
        }

        float inv_dim = 1.0f / static_cast<float>(f_dim);
        for (size_t i = 0; i < b_count; ++i)
        {
            float inv_r = inv_rms_data[i];
            float s = 0.0f;
            for (size_t j = 0; j < f_dim; ++j)
            {
                float x_hat = in_data[i * f_dim + j] * inv_r;
                s += out_grad[i * f_dim + j] * gamma_data[j] * x_hat;
            }
            for (size_t j = 0; j < f_dim; ++j)
            {
                float x_hat = in_data[i * f_dim + j] * inv_r;
                float dx = inv_r * (out_grad[i * f_dim + j] * gamma_data[j] - x_hat * (s * inv_dim));
                (*in_grad_cpu.storage_buffer)[i * f_dim + j] = dx;
            }
        }
    }

void Cpu_Tensor_Impl::applyRoPE(Tensor_Impl &output, uint32_t seq_len, uint32_t head_dim, int direction, float base, uint32_t num_heads, uint32_t mode) const
{
        const auto &in_data = getData();
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        if (mode == 1)
        {
            out_cpu.reshape(Shape{shape[0], num_heads, seq_len, head_dim});
        }
        else if (mode == 2)
        {
            out_cpu.reshape(Shape{shape[0], seq_len, num_heads, head_dim});
        }
        else
        {
            out_cpu.reshape(shape);
        }

        size_t total = total_elements;
        size_t total_pairs = total / 2;
        size_t D = head_dim;
        size_t S = seq_len;
        size_t H = num_heads;

        for (size_t pair_id = 0; pair_id < total_pairs; ++pair_id)
        {
            size_t base_idx = pair_id * 2;
            size_t c = base_idx % D;
            size_t pair_idx = c / 2;
            size_t token_pos = 0;
            size_t in_idx = base_idx;
            size_t out_idx = base_idx;

            if (mode == 1)
            {
                token_pos = (base_idx / D) % S;
                size_t h = (base_idx / (D * S)) % H;
                size_t b = base_idx / (D * S * H);
                in_idx = b * (S * H * D) + token_pos * (H * D) + h * D + c;
                out_idx = base_idx;
            }
            else if (mode == 2)
            {
                token_pos = (base_idx / D) % S;
                size_t h = (base_idx / (D * S)) % H;
                size_t b = base_idx / (D * S * H);
                in_idx = base_idx;
                out_idx = b * (S * H * D) + token_pos * (H * D) + h * D + c;
            }
            else
            {
                token_pos = (base_idx / D) % S;
                in_idx = base_idx;
                out_idx = base_idx;
            }

            float theta = std::pow(base, -2.0f * static_cast<float>(pair_idx) / static_cast<float>(D));
            float alpha = static_cast<float>(direction) * static_cast<float>(token_pos) * theta;
            float cos_a = std::cos(alpha);
            float sin_a = std::sin(alpha);

            float x0 = in_data[in_idx];
            float x1 = in_data[in_idx + 1];

            (*out_cpu.storage_buffer)[out_idx]     = x0 * cos_a - x1 * sin_a;
            (*out_cpu.storage_buffer)[out_idx + 1] = x0 * sin_a + x1 * cos_a;
        }
    }

void Cpu_Tensor_Impl::swigluForward(const Tensor_Impl &b, Tensor_Impl &output) const
{
        const auto &a_data = getData();
        const auto &b_data = b.getData();
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        out_cpu.reshape(shape);

        size_t total = total_elements;
        for (size_t idx = 0; idx < total; ++idx)
        {
            float a_val = a_data[idx];
            float b_val = b_data[idx];
            float sig_a = 1.0f / (1.0f + std::exp(-a_val));
            (*out_cpu.storage_buffer)[idx] = (a_val * sig_a) * b_val;
        }
    }

void Cpu_Tensor_Impl::swigluBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &b, Tensor_Impl &grad_a, Tensor_Impl &grad_b) const
{
        const auto &dy_data = output_gradient.getData();
        const auto &a_data = getData();
        const auto &b_data = b.getData();

        auto &da_cpu = static_cast<Cpu_Tensor_Impl &>(grad_a);
        auto &db_cpu = static_cast<Cpu_Tensor_Impl &>(grad_b);
        da_cpu.reshape(shape);
        db_cpu.reshape(shape);

        size_t total = total_elements;
        for (size_t idx = 0; idx < total; ++idx)
        {
            float dy = dy_data[idx];
            float a_val = a_data[idx];
            float b_val = b_data[idx];

            float sig_a = 1.0f / (1.0f + std::exp(-a_val));
            float silu_a = a_val * sig_a;
            float d_silu_a = sig_a * (1.0f + a_val * (1.0f - sig_a));

            (*db_cpu.storage_buffer)[idx] = dy * silu_a;
            (*da_cpu.storage_buffer)[idx] = dy * b_val * d_silu_a;
        }
    }

void Cpu_Tensor_Impl::fusedSwiGLUForward(Tensor_Impl &output) const
{
        size_t rows = shape[0];
        size_t total_cols = shape[1];
        size_t half_dim = total_cols / 2;
        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        out_cpu.reshape(Shape{rows, half_dim});

        const auto &in_data = getData();
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < half_dim; ++c)
            {
                float a = in_data[r * total_cols + c];
                float b = in_data[r * total_cols + half_dim + c];
                float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a, -85.0f, 85.0f)));
                float silu_a = a * sig_a;
                (*out_cpu.storage_buffer)[r * half_dim + c] = silu_a * b;
            }
        }
    }

void Cpu_Tensor_Impl::fusedSwiGLUBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const
{
        size_t rows = shape[0];
        size_t total_cols = shape[1];
        size_t half_dim = total_cols / 2;
        auto &dx_cpu = static_cast<Cpu_Tensor_Impl &>(input_gradient);
        dx_cpu.reshape(Shape{rows, total_cols});

        const auto &in_data = getData();
        const auto &dy_data = output_gradient.getData();
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < half_dim; ++c)
            {
                float dy = dy_data[r * half_dim + c];
                float a = in_data[r * total_cols + c];
                float b = in_data[r * total_cols + half_dim + c];
                float sig_a = 1.0f / (1.0f + std::exp(-std::clamp(a, -85.0f, 85.0f)));
                float silu_a = a * sig_a;
                float d_silu_a = sig_a * (1.0f + a * (1.0f - sig_a));

                (*dx_cpu.storage_buffer)[r * total_cols + c] = dy * b * d_silu_a;
                (*dx_cpu.storage_buffer)[r * total_cols + half_dim + c] = dy * silu_a;
            }
        }
    }

void Cpu_Tensor_Impl::flashAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, Tensor_Impl *l_stats) const
{
        const auto &q_data = getData();
        const auto &k_data = k.getData();
        const auto &v_data = v.getData();

        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        out_cpu.reshape(shape);

        if (scale <= 0.0f)
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        size_t total_elements_per_head = static_cast<size_t>(seq_len) * head_dim;
        size_t total_heads = (total_elements_per_head > 0) ? (total_elements / total_elements_per_head) : num_heads;

        Cpu_Tensor_Impl *l_cpu = nullptr;
        if (l_stats)
        {
            l_stats->setDataType(Data_Type::FLOAT32);
            l_stats->reshape(Shape{ total_heads, seq_len });
            l_cpu = static_cast<Cpu_Tensor_Impl *>(l_stats);
            if (!l_cpu->storage_buffer || l_cpu->storage_buffer->size() != total_heads * seq_len)
            {
                l_cpu->storage_buffer = std::make_shared<std::vector<float>>(total_heads * seq_len, 0.0f);
            }
        }

        for (size_t h = 0; h < total_heads; ++h)
        {
            size_t head_offset = h * total_elements_per_head;

            for (size_t i = 0; i < seq_len; ++i)
            {
                std::vector<float> scores(seq_len, -1e20f);
                float max_score = -1e20f;

                size_t max_j = is_causal ? (i + 1) : seq_len;
                for (size_t j = 0; j < max_j; ++j)
                {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dot += q_data[head_offset + i * head_dim + d] * k_data[head_offset + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > max_score)
                    {
                        max_score = scores[j];
                    }
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j < max_j; ++j)
                {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

                if (l_cpu)
                {
                    float logsumexp = (sum_exp > 0.0f) ? (max_score + std::log(sum_exp)) : -1e20f;
                    (*l_cpu->storage_buffer)[h * seq_len + i] = logsumexp;
                }

                for (size_t d = 0; d < head_dim; ++d)
                {
                    float out_val = 0.0f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        out_val += (scores[j] * inv_sum) * v_data[head_offset + j * head_dim + d];
                    }
                    (*out_cpu.storage_buffer)[head_offset + i * head_dim + d] = out_val;
                }
            }
        }
    }

void Cpu_Tensor_Impl::flashAttentionBackward(const Tensor_Impl &k, const Tensor_Impl &v, const Tensor_Impl &o, const Tensor_Impl &do_grad, Tensor_Impl &dq, Tensor_Impl &dk, Tensor_Impl &dv, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim, bool is_causal, float scale, const Tensor_Impl *l_stats) const
{
        const auto &q_data = getData();
        const auto &k_data = k.getData();
        const auto &v_data = v.getData();
        const auto &o_data = o.getData();
        const auto &do_data = do_grad.getData();

        auto &dq_cpu = static_cast<Cpu_Tensor_Impl &>(dq);
        auto &dk_cpu = static_cast<Cpu_Tensor_Impl &>(dk);
        auto &dv_cpu = static_cast<Cpu_Tensor_Impl &>(dv);

        dq_cpu.reshape(shape);
        dk_cpu.reshape(k.getShape());
        dv_cpu.reshape(v.getShape());

        std::fill(dq_cpu.storage_buffer->begin(), dq_cpu.storage_buffer->end(), 0.0f);
        std::fill(dk_cpu.storage_buffer->begin(), dk_cpu.storage_buffer->end(), 0.0f);
        std::fill(dv_cpu.storage_buffer->begin(), dv_cpu.storage_buffer->end(), 0.0f);

        if (scale <= 0.0f)
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        size_t total_elements_per_head = static_cast<size_t>(seq_len) * head_dim;
        size_t total_heads = (total_elements_per_head > 0) ? (total_elements / total_elements_per_head) : num_heads;

        std::vector<float> l_data_cache;
        if (l_stats)
        {
            l_data_cache = l_stats->getData();
        }

        for (size_t h = 0; h < total_heads; ++h)
        {
            size_t head_offset = h * total_elements_per_head;

            for (size_t i = 0; i < seq_len; ++i)
            {
                std::vector<float> p(seq_len, 0.0f);
                size_t max_j = is_causal ? (i + 1) : seq_len;

                if (l_stats != nullptr && !l_data_cache.empty())
                {
                    float l_i = l_data_cache[h * seq_len + i];
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            dot += q_data[head_offset + i * head_dim + d] * k_data[head_offset + j * head_dim + d];
                        }
                        p[j] = std::exp(dot * scale - l_i);
                    }
                }
                else
                {
                    float max_score = -1e20f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d)
                        {
                            dot += q_data[head_offset + i * head_dim + d] * k_data[head_offset + j * head_dim + d];
                        }
                        p[j] = dot * scale;
                        if (p[j] > max_score)
                        {
                            max_score = p[j];
                        }
                    }

                    float sum_exp = 0.0f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        p[j] = std::exp(p[j] - max_score);
                        sum_exp += p[j];
                    }
                    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                    for (size_t j = 0; j < max_j; ++j)
                    {
                        p[j] *= inv_sum;
                    }
                }

                float di = 0.0f;
                for (size_t d = 0; d < head_dim; ++d)
                {
                    di += do_data[head_offset + i * head_dim + d] * o_data[head_offset + i * head_dim + d];
                }

                for (size_t j = 0; j < max_j; ++j)
                {
                    float dp = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        dp += do_data[head_offset + i * head_dim + d] * v_data[head_offset + j * head_dim + d];
                    }

                    float ds = p[j] * (dp - di) * scale;

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        (*dq_cpu.storage_buffer)[head_offset + i * head_dim + d] += ds * k_data[head_offset + j * head_dim + d];
                    }

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        (*dk_cpu.storage_buffer)[head_offset + j * head_dim + d] += ds * q_data[head_offset + i * head_dim + d];
                    }

                    for (size_t d = 0; d < head_dim; ++d)
                    {
                        (*dv_cpu.storage_buffer)[head_offset + j * head_dim + d] += p[j] * do_data[head_offset + i * head_dim + d];
                    }
                }
            }
        }
    }

void Cpu_Tensor_Impl::embeddingForward(const Tensor_Impl &indices, Tensor_Impl &output) const
{
        size_t S = indices.getTotalElements();
        size_t D = getColumns();
        size_t V = shape[0];
        output.setDataType(data_type);
        if (indices.getShape().getRank() == 2)
        {
            output.reshape(Shape{ indices.getShape()[0], indices.getShape()[1], D });
        }
        else
        {
            output.reshape(Shape{ S, D });
        }

        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        if (data_type == Data_Type::FLOAT16)
        {
            if (!out_cpu.storage_buffer_fp16 || out_cpu.storage_buffer_fp16->size() != S * D)
            {
                out_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(S * D, float16_t(0.0f));
            }
            out_cpu.storage_buffer.reset();
        }
        else
        {
            if (!out_cpu.storage_buffer || out_cpu.storage_buffer->size() != S * D)
            {
                out_cpu.storage_buffer = std::make_shared<std::vector<float>>(S * D, 0.0f);
            }
            out_cpu.storage_buffer_fp16.reset();
        }

        const auto &indices_data = indices.getData();
        const auto &weight_data = getData();

        for (size_t s = 0; s < S; ++s)
        {
            int row = static_cast<int>(std::round(indices_data[s]));
            if (row >= 0 && static_cast<size_t>(row) < V)
            {
                if (data_type == Data_Type::FLOAT16)
                {
                    std::memcpy(out_cpu.storage_buffer_fp16->data() + s * D,
                                storage_buffer_fp16->data() + row * D,
                                D * sizeof(float16_t));
                }
                else
                {
                    std::memcpy(out_cpu.storage_buffer->data() + s * D,
                                weight_data.data() + row * D,
                                D * sizeof(float));
                }
            }
            else
            {
                if (data_type == Data_Type::FLOAT16)
                {
                    std::memset(out_cpu.storage_buffer_fp16->data() + s * D, 0, D * sizeof(float16_t));
                }
                else
                {
                    std::memset(out_cpu.storage_buffer->data() + s * D, 0, D * sizeof(float));
                }
            }
        }
    }

void Cpu_Tensor_Impl::embeddingBackward(const Tensor_Impl &indices, const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient) const
{
        size_t S = indices.getTotalElements();
        size_t D = getColumns();
        size_t V = shape[0];

        auto &w_grad_cpu = static_cast<Cpu_Tensor_Impl &>(weight_gradient);
        if (!w_grad_cpu.storage_buffer || w_grad_cpu.storage_buffer->size() != V * D)
        {
            w_grad_cpu.storage_buffer = std::make_shared<std::vector<float>>(V * D, 0.0f);
        }

        const auto &indices_data = indices.getData();
        const auto &grad_out_data = output_gradient.getData();

        for (size_t s = 0; s < S; ++s)
        {
            int row = static_cast<int>(std::round(indices_data[s]));
            if (row >= 0 && static_cast<size_t>(row) < V)
            {
                for (size_t d = 0; d < D; ++d)
                {
                    (*w_grad_cpu.storage_buffer)[row * D + d] += grad_out_data[s * D + d];
                }
            }
        }
    }

void Cpu_Tensor_Impl::singleTokenAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output, size_t num_heads, size_t head_dim, size_t total_seq_len) const
{
        output.setDataType(data_type);
        output.reshape(Shape{ 1, num_heads, 1, head_dim });

        auto &out_cpu = static_cast<Cpu_Tensor_Impl &>(output);
        if (!out_cpu.storage_buffer || out_cpu.storage_buffer->size() != num_heads * head_dim)
        {
            out_cpu.storage_buffer = std::make_shared<std::vector<float>>(num_heads * head_dim, 0.0f);
        }

        const auto &q_data = getData();
        const auto &k_data = k.getData();
        const auto &v_data = v.getData();
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::vector<float> scores(total_seq_len);

        for (size_t h = 0; h < num_heads; ++h)
        {
            const float *q_h = q_data.data() + h * head_dim;
            const float *k_h = k_data.data() + h * total_seq_len * head_dim;
            const float *v_h = v_data.data() + h * total_seq_len * head_dim;
            float *out_h = out_cpu.storage_buffer->data() + h * head_dim;

            float max_score = -1e20f;
            for (size_t t = 0; t < total_seq_len; ++t)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; ++i)
                {
                    dot += q_h[i] * k_h[t * head_dim + i];
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_seq_len; ++t)
            {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (size_t t = 0; t < total_seq_len; ++t)
            {
                scores[t] *= inv_sum;
            }

            for (size_t i = 0; i < head_dim; ++i)
            {
                float val = 0.0f;
                for (size_t t = 0; t < total_seq_len; ++t)
                {
                    val += scores[t] * v_h[t * head_dim + i];
                }
                out_h[i] = val;
            }
        }
    }

float Cpu_Tensor_Impl::fusedCrossEntropyLoss(const Tensor_Impl &targets, Tensor_Impl &d_logits, uint32_t valid_tokens) const
{
        size_t total_tokens = (shape.getRank() == 3) ? (shape[0] * shape[1]) : getRows();
        size_t V = (shape.getRank() == 3) ? shape[2] : getColumns();

        d_logits.setDataType(data_type);
        d_logits.reshape(shape);

        if (total_tokens == 0 || V == 0)
        {
            return 0.0f;
        }

        const auto &logits_data = getData();
        const auto &targets_data = targets.getData();
        auto &grad_data = static_cast<Cpu_Tensor_Impl &>(d_logits).storage_buffer;
        if (!grad_data || grad_data->size() < total_tokens * V)
        {
            grad_data = std::make_shared<std::vector<float>>(total_tokens * V, 0.0f);
        }
        else
        {
            std::fill(grad_data->begin(), grad_data->begin() + total_tokens * V, 0.0f);
        }

        if (valid_tokens == 0)
        {
            for (size_t s = 0; s < total_tokens; ++s)
            {
                if (s < targets_data.size())
                {
                    int32_t target = static_cast<int32_t>(std::round(targets_data[s]));
                    if (target >= 0 && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
            }
        }

        if (valid_tokens == 0)
        {
            return 0.0f;
        }

        float scale = 1.0f / static_cast<float>(valid_tokens);
        float total_loss = 0.0f;

        for (size_t s = 0; s < total_tokens; ++s)
        {
            int32_t target = (s < targets_data.size()) ? static_cast<int32_t>(std::round(targets_data[s])) : -100;
            if (target < 0 || static_cast<size_t>(target) >= V)
            {
                continue;
            }

            const float *row_logits = logits_data.data() + s * V;
            float *row_grad = grad_data->data() + s * V;

            float max_val = -1e20f;
            for (size_t v = 0; v < V; ++v)
            {
                float val = row_logits[v];
                if (!std::isnan(val) && !std::isinf(val))
                {
                    max_val = std::max(max_val, val);
                }
            }

            float sum_exp = 0.0f;
            for (size_t v = 0; v < V; ++v)
            {
                float val = row_logits[v];
                if (!std::isnan(val) && !std::isinf(val))
                {
                    sum_exp += std::exp(val - max_val);
                }
            }

            float lse = max_val + std::log(std::max(sum_exp, 1e-12f));
            float target_logit = row_logits[target];
            float token_loss = lse - target_logit;
            total_loss += token_loss;

            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (size_t v = 0; v < V; ++v)
            {
                float logit_val = row_logits[v];
                float prob = 0.0f;
                if (!std::isnan(logit_val) && !std::isinf(logit_val))
                {
                    prob = std::exp(logit_val - max_val) * inv_sum;
                }
                float delta = (v == static_cast<size_t>(target)) ? 1.0f : 0.0f;
                row_grad[v] = (prob - delta) * scale;
            }
        }

        if (data_type == Data_Type::FLOAT16)
        {
            auto &d_logits_cpu = static_cast<Cpu_Tensor_Impl &>(d_logits);
            if (!d_logits_cpu.storage_buffer_fp16 || d_logits_cpu.storage_buffer_fp16->size() < total_tokens * V)
            {
                d_logits_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(total_tokens * V);
            }
            convertFp32ToFp16(grad_data->data(), d_logits_cpu.storage_buffer_fp16->data(), total_tokens * V);
            d_logits_cpu.storage_buffer.reset();
        }

        return total_loss * scale;
    }

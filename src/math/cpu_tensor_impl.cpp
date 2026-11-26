#include "math/cpu_tensor_impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>

#include "helper/logger.h"

namespace
{
    size_t computeMaxFlatIndex(const Shape &shape, const Stride &strides, size_t byte_offset, Data_Type dt) noexcept
    {
        size_t elem_size = getDataTypeSize(dt);
        size_t idx = byte_offset / elem_size;
        for (size_t d = 0; d < shape.getRank(); ++d)
        {
            if (shape[d] > 1)
            {
                idx += (shape[d] - 1) * strides[d];
            }
        }
        return idx;
    }

    Shape computeBroadcastShapeChecked(std::span<const size_t> a_dims, std::span<const size_t> b_dims, const char *context_name)
    {
        size_t out_rank = std::max(a_dims.size(), b_dims.size());
        if (out_rank > MAX_TENSOR_RANK)
        {
            throw std::invalid_argument(std::format("{}: Broadcast rank exceeds MAX_TENSOR_RANK", context_name));
        }
        std::array<size_t, MAX_TENSOR_RANK> out_dims{};
        for (size_t i = 0; i < out_rank; ++i)
        {
            size_t da = (i + a_dims.size() >= out_rank) ? a_dims[i + a_dims.size() - out_rank] : 1;
            size_t db = (i + b_dims.size() >= out_rank) ? b_dims[i + b_dims.size() - out_rank] : 1;
            if (da != db && da != 1 && db != 1)
            {
                throw std::invalid_argument(std::format("{}: Shape mismatch at dim {} ({} vs {})", context_name, i, da, db));
            }
            out_dims[i] = (da == db) ? da : ((da == 1) ? db : da);
        }
        return Shape(std::span<const size_t>(out_dims.data(), out_rank));
    }

    std::array<size_t, MAX_TENSOR_RANK> computeBroadcastStrides(std::span<const size_t> src_dims, size_t out_rank) noexcept
    {
        std::array<size_t, MAX_TENSOR_RANK> b_strides{};
        size_t stride = 1;
        for (size_t k = 0; k < src_dims.size(); ++k)
        {
            size_t src_idx = src_dims.size() - 1 - k;
            size_t out_idx = out_rank - 1 - k;
            size_t d = src_dims[src_idx];
            b_strides[out_idx] = (d <= 1) ? 0 : stride;
            stride *= d;
        }
        return b_strides;
    }

    inline void advanceBroadcastBatchCoords(
        std::array<size_t, MAX_TENSOR_RANK> &coords,
        const Shape &broadcast_shape,
        const std::array<size_t, MAX_TENSOR_RANK> &strides_a,
        const std::array<size_t, MAX_TENSOR_RANK> &strides_b,
        size_t &idx_a,
        size_t &idx_b) noexcept
    {
        for (size_t d = broadcast_shape.getRank(); d-- > 0;)
        {
            ++coords[d];
            if (coords[d] < broadcast_shape[d])
            {
                idx_a += strides_a[d];
                idx_b += strides_b[d];
                return;
            }
            idx_a -= (broadcast_shape[d] - 1) * strides_a[d];
            idx_b -= (broadcast_shape[d] - 1) * strides_b[d];
            coords[d] = 0;
        }
    }

    template <typename Fn>
    std::vector<float> applyBroadcastElementwise(
        const Shape &sa, const std::vector<float> &a,
        const Shape &sb, const std::vector<float> &b,
        Shape &out_shape, const char *op_name, Fn &&fn)
    {
        if (sa == sb)
        {
            out_shape = sa;
            size_t total = out_shape.getTotalElements();
            std::vector<float> res(total);
            for (size_t i = 0; i < total; ++i)
            {
                res[i] = fn(a[i], b[i]);
            }
            return res;
        }
        if (sa.getRank() == 0 || sb.getRank() == 0)
        {
            throw std::invalid_argument(std::format("{}: Incompatible shapes {} vs {}", op_name, sa.toString(), sb.toString()));
        }
        out_shape = computeBroadcastShapeChecked(sa.getDimensions(), sb.getDimensions(), op_name);
        size_t out_total = out_shape.getTotalElements();
        if (out_total == 0)
        {
            return {};
        }
        size_t out_rank = out_shape.getRank();
        auto strides_a = computeBroadcastStrides(sa.getDimensions(), out_rank);
        auto strides_b = computeBroadcastStrides(sb.getDimensions(), out_rank);
        std::array<size_t, MAX_TENSOR_RANK> coords{};
        std::vector<float> res(out_total);
        size_t idx_a = 0;
        size_t idx_b = 0;
        for (size_t flat = 0; flat < out_total; ++flat)
        {
            res[flat] = fn(a[idx_a], b[idx_b]);
            advanceBroadcastBatchCoords(coords, out_shape, strides_a, strides_b, idx_a, idx_b);
        }
        return res;
    }

    struct Matmul_Broadcast_Plan
    {
        size_t m_dim = 0;
        size_t k_dim = 0;
        size_t n_dim = 0;
        size_t batch_total = 1;
        Shape batch_shape;
        Shape out_shape;
        std::array<size_t, MAX_TENSOR_RANK> strides_a{};
        std::array<size_t, MAX_TENSOR_RANK> strides_b{};
    };

    Matmul_Broadcast_Plan planMatmulBroadcast(const Tensor_Impl &lhs, const Tensor_Impl &rhs, const char *op_name)
    {
        const Shape &sa = lhs.getShape();
        const Shape &sb = rhs.getShape();
        size_t rank_a = sa.getRank();
        size_t rank_b = sb.getRank();

        Matmul_Broadcast_Plan plan;
        plan.m_dim = (rank_a >= 2) ? sa[rank_a - 2] : lhs.getRows();
        plan.k_dim = (rank_a >= 2) ? sa[rank_a - 1] : lhs.getColumns();
        size_t k_other = (rank_b >= 2) ? sb[rank_b - 2] : rhs.getRows();
        plan.n_dim = (rank_b >= 2) ? sb[rank_b - 1] : rhs.getColumns();

        if (plan.k_dim != k_other)
        {
            throw std::invalid_argument("Tensor inner dimensions must match for multiplication");
        }

        if (rank_a <= 2 && rank_b <= 2)
        {
            plan.batch_total = 1;
            plan.out_shape = Shape{plan.m_dim, plan.n_dim};
            return plan;
        }

        auto a_dims = sa.getDimensions();
        auto b_dims = sb.getDimensions();
        std::span<const size_t> a_batch = (rank_a >= 3) ? a_dims.subspan(0, rank_a - 2) : std::span<const size_t>{};
        std::span<const size_t> b_batch = (rank_b >= 3) ? b_dims.subspan(0, rank_b - 2) : std::span<const size_t>{};

        plan.batch_shape = computeBroadcastShapeChecked(a_batch, b_batch, op_name);
        size_t batch_rank = plan.batch_shape.getRank();
        plan.batch_total = (batch_rank == 0) ? 1 : plan.batch_shape.getTotalElements();
        plan.strides_a = computeBroadcastStrides(a_batch, batch_rank);
        plan.strides_b = computeBroadcastStrides(b_batch, batch_rank);

        std::array<size_t, MAX_TENSOR_RANK> out_dims{};
        for (size_t i = 0; i < batch_rank; ++i)
        {
            out_dims[i] = plan.batch_shape[i];
        }
        out_dims[batch_rank] = plan.m_dim;
        out_dims[batch_rank + 1] = plan.n_dim;
        plan.out_shape = Shape(std::span<const size_t>(out_dims.data(), batch_rank + 2));
        return plan;
    }

    void writeCpuResult(Cpu_Tensor_Impl &out, const Shape &out_shape, std::vector<float> vals)
    {
        out.setShape(out_shape);
        out.setByteOffset(0);
        size_t total = out.getTotalElements();
        if (out.getDataType() == Data_Type::FLOAT16)
        {
            auto &fp16_buf = out.getStorageBufferFp16();
            if (!fp16_buf || fp16_buf.use_count() > 1 || fp16_buf->size() != total)
            {
                fp16_buf = std::make_shared<std::vector<float16_t>>(total);
            }
            if (total > 0)
            {
                convertFp32ToFp16(vals.data(), fp16_buf->data(), total);
            }
            out.getStorageBuffer().reset();
            out.getStorageBufferU32().reset();
        }
        else
        {
            out.setDataType(Data_Type::FLOAT32);
            auto &fp32_buf = out.getStorageBuffer();
            if (!fp32_buf || fp32_buf.use_count() > 1 || fp32_buf->size() != total)
            {
                fp32_buf = std::make_shared<std::vector<float>>(std::move(vals));
            }
            else
            {
                *fp32_buf = std::move(vals);
            }
            out.getStorageBufferFp16().reset();
            out.getStorageBufferU32().reset();
        }
    }
}

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
        else if (data_type == Data_Type::TERNARY_INT2)
        {
            size_t bytes = getTensorStorageBytes(data_type, shape);
            size_t words = bytes / sizeof(uint32_t);
            storage_buffer_u32 = std::make_shared<std::vector<uint32_t>>(words, 0u);
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

Cpu_Tensor_Impl::Cpu_Tensor_Impl(Shape tensor_shape, std::vector<float> &&host_data)
{
        updateShapeAndStrides(tensor_shape);
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl: Host data size mismatch (expected {}, got {})",
                                             total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        storage_buffer = std::make_shared<std::vector<float>>(std::move(host_data));
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
            if (!storage_buffer_fp16 || (storage_buffer_fp16.use_count() > 1 && storage_buffer_fp16->size() != new_total))
            {
                storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(new_total, static_cast<float16_t>(0.0f));
            }
            else if (storage_buffer_fp16->size() != new_total)
            {
                storage_buffer_fp16->resize(new_total, static_cast<float16_t>(0.0f));
            }
            storage_buffer.reset();
            storage_buffer_u32.reset();
        }
        else if (data_type == Data_Type::TERNARY_INT2)
        {
            size_t new_words = getTensorStorageBytes(Data_Type::TERNARY_INT2, new_shape) / sizeof(uint32_t);
            if (!storage_buffer_u32 || (storage_buffer_u32.use_count() > 1 && storage_buffer_u32->size() != new_words))
            {
                storage_buffer_u32 = std::make_shared<std::vector<uint32_t>>(new_words, 0u);
            }
            else if (storage_buffer_u32->size() != new_words)
            {
                storage_buffer_u32->resize(new_words, 0u);
            }
            storage_buffer.reset();
            storage_buffer_fp16.reset();
        }
        else
        {
            if (!storage_buffer || (storage_buffer.use_count() > 1 && storage_buffer->size() != new_total))
            {
                storage_buffer = std::make_shared<std::vector<float>>(new_total, 0.0F);
            }
            else if (storage_buffer->size() != new_total)
            {
                storage_buffer->resize(new_total, 0.0F);
            }
            storage_buffer_fp16.reset();
            storage_buffer_u32.reset();
        }
        updateShapeAndStrides(new_shape);
    }

void Cpu_Tensor_Impl::setDataType(Data_Type _type)
{
        invalidateFp16Cache();
        if (_type == Data_Type::FLOAT16)
        {
            if (storage_buffer_fp16 && storage_buffer_fp16->size() >= total_elements)
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                materialized_cache.clear();
                storage_buffer.reset();
                storage_buffer_u32.reset();
                data_type = _type;
                return;
            }
            std::shared_ptr<std::vector<float16_t>> next_buf;
            if (total_elements > 0 && data_type != Data_Type::FLOAT16 && !isEmpty())
            {
                std::vector<float> vals = getData();
                next_buf = std::make_shared<std::vector<float16_t>>(total_elements);
                convertFp32ToFp16(vals.data(), next_buf->data(), total_elements);
            }
            else
            {
                next_buf = std::make_shared<std::vector<float16_t>>(total_elements, static_cast<float16_t>(0.0f));
            }
            std::lock_guard<std::mutex> lock(cache_mutex);
            materialized_cache.clear();
            storage_buffer_fp16 = std::move(next_buf);
            storage_buffer.reset();
            storage_buffer_u32.reset();
            strides = shape.computeContiguousStrides();
            byte_offset = 0;
            data_type = _type;
            return;
        }
        if (_type == Data_Type::FLOAT32)
        {
            if (storage_buffer && storage_buffer->size() >= total_elements)
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                materialized_cache.clear();
                storage_buffer_fp16.reset();
                storage_buffer_u32.reset();
                data_type = _type;
                return;
            }
            std::shared_ptr<std::vector<float>> next_buf;
            if (total_elements > 0 && data_type != Data_Type::FLOAT32 && !isEmpty())
            {
                std::vector<float> vals = getData();
                next_buf = std::make_shared<std::vector<float>>(std::move(vals));
            }
            else
            {
                next_buf = std::make_shared<std::vector<float>>(total_elements, 0.0F);
            }
            std::lock_guard<std::mutex> lock(cache_mutex);
            materialized_cache.clear();
            storage_buffer = std::move(next_buf);
            storage_buffer_fp16.reset();
            storage_buffer_u32.reset();
            strides = shape.computeContiguousStrides();
            byte_offset = 0;
            data_type = _type;
            return;
        }
        if (_type == Data_Type::TERNARY_INT2)
        {
            size_t words = getTensorStorageBytes(Data_Type::TERNARY_INT2, shape) / sizeof(uint32_t);
            if (storage_buffer_u32 && storage_buffer_u32->size() >= words)
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                materialized_cache.clear();
                storage_buffer.reset();
                storage_buffer_fp16.reset();
                data_type = _type;
                return;
            }
            auto next_buf = std::make_shared<std::vector<uint32_t>>(words, 0u);
            std::lock_guard<std::mutex> lock(cache_mutex);
            materialized_cache.clear();
            storage_buffer_u32 = std::move(next_buf);
            storage_buffer.reset();
            storage_buffer_fp16.reset();
            strides = shape.computeContiguousStrides();
            byte_offset = 0;
            data_type = _type;
            return;
        }
        std::lock_guard<std::mutex> lock(cache_mutex);
        materialized_cache.clear();
        data_type = _type;
    }

void Cpu_Tensor_Impl::permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::permute: TERNARY_INT2 is not supported");
        }
        size_t rank = shape.getRank();
        if (axes_permutation.size() != rank)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::permute: Permutation size must match tensor rank");
        }
        std::array<bool, MAX_TENSOR_RANK> seen{};
        for (size_t ax : axes_permutation)
        {
            if (ax >= rank || seen[ax])
            {
                throw std::invalid_argument("Cpu_Tensor_Impl::permute: Invalid or duplicate axis");
            }
            seen[ax] = true;
        }

        Shape new_shape = shape;
        Stride new_strides = strides;
        for (size_t i = 0; i < rank; ++i)
        {
            new_shape[i] = shape[axes_permutation[i]];
            new_strides[i] = strides[axes_permutation[i]];
        }
        std::string src_shape_str = shape.toString();
        Data_Type src_type = data_type;
        size_t src_offset = byte_offset;
        size_t src_total = total_elements;
        auto src_fp32 = storage_buffer;
        auto src_fp16 = storage_buffer_fp16;

        auto &output_cpu = castToCpu(output);
        output_cpu.data_type = src_type;
        output_cpu.shape = new_shape;
        output_cpu.strides = new_strides;
        output_cpu.storage_buffer = std::move(src_fp32);
        output_cpu.storage_buffer_fp16 = std::move(src_fp16);
        output_cpu.storage_buffer_u32.reset();
        output_cpu.byte_offset = src_offset;
        output_cpu.total_elements = src_total;

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::permute: shape {} -> {}",
                                        src_shape_str, new_shape.toString()},
                            Log_Level::LOG_DEBUG, true, 0, Log_Feature::TENSOR_INSPECTION);
    }

void Cpu_Tensor_Impl::slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::slice: TERNARY_INT2 is not supported");
        }
        size_t rank = shape.getRank();
        if (axis >= rank)
        {
            throw std::out_of_range("Cpu_Tensor_Impl::slice: axis out of range");
        }
        size_t dim = shape[axis];
        if (start > dim || length > dim - start)
        {
            throw std::out_of_range("Cpu_Tensor_Impl::slice: slice range exceeds dimension bounds");
        }

        Shape new_shape = shape;
        new_shape[axis] = length;
        size_t additional_offset = start * strides[axis];
        Data_Type src_type = data_type;
        Stride src_strides = strides;
        size_t new_byte_offset = byte_offset + additional_offset * getDataTypeSize(data_type);
        auto src_fp32 = storage_buffer;
        auto src_fp16 = storage_buffer_fp16;

        auto &output_cpu = castToCpu(output);
        output_cpu.data_type = src_type;
        output_cpu.shape = new_shape;
        output_cpu.strides = src_strides;
        output_cpu.storage_buffer = std::move(src_fp32);
        output_cpu.storage_buffer_fp16 = std::move(src_fp16);
        output_cpu.storage_buffer_u32.reset();
        output_cpu.byte_offset = new_byte_offset;
        output_cpu.total_elements = new_shape.getTotalElements();

        Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::slice: axis={}, start={}, length={}, shape={}",
                                        axis, start, length, new_shape.toString()},
                            Log_Level::LOG_DEBUG, true, 0, Log_Feature::TENSOR_INSPECTION);
    }

void Cpu_Tensor_Impl::updateSlice(size_t axis, size_t start, const Tensor_Impl &source)
{
        const auto &source_cpu = castToCpu(source);
        if (data_type == Data_Type::TERNARY_INT2 || source_cpu.data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::updateSlice: TERNARY_INT2 is not supported");
        }
        if (data_type != source_cpu.data_type)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::updateSlice: Data_Type mismatch");
        }
        size_t rank = shape.getRank();
        if (axis >= rank || rank != source_cpu.shape.getRank())
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::updateSlice: Rank or axis mismatch");
        }
        for (size_t d = 0; d < rank; ++d)
        {
            if (d != axis && shape[d] != source_cpu.shape[d])
            {
                throw std::invalid_argument("Cpu_Tensor_Impl::updateSlice: Non-axis dimensions must match");
            }
        }
        size_t src_axis_len = source_cpu.shape[axis];
        if (start > shape[axis] || src_axis_len > shape[axis] - start)
        {
            throw std::out_of_range("Cpu_Tensor_Impl::updateSlice: slice range exceeds destination dimension");
        }
        if (source_cpu.total_elements == 0)
        {
            return;
        }

        size_t elem_size = getDataTypeSize(data_type);
        size_t outer_count = 1;
        for (size_t i = 0; i < axis; ++i) outer_count *= shape[i];
        size_t inner_count = 1;
        for (size_t i = axis + 1; i < rank; ++i) inner_count *= shape[i];
        size_t this_axis_len = shape[axis];
        size_t copy_bytes = src_axis_len * inner_count * elem_size;

        if (data_type == Data_Type::FLOAT16 && storage_buffer_fp16 && source_cpu.storage_buffer_fp16)
        {
            if (isContiguous() && source_cpu.isContiguous() && storage_buffer_fp16.get() != source_cpu.storage_buffer_fp16.get())
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
            else
            {
                std::vector<float16_t> src_flat(source_cpu.total_elements);
                source_cpu.iterateCoordinates([&](size_t out_idx, size_t src_idx)
                                              { src_flat[out_idx] = (*source_cpu.storage_buffer_fp16)[src_idx]; });
                std::array<size_t, MAX_TENSOR_RANK> coords{};
                for (size_t i = 0; i < source_cpu.total_elements; ++i)
                {
                    std::array<size_t, MAX_TENSOR_RANK> dst_coords = coords;
                    dst_coords[axis] += start;
                    size_t dst_idx = resolveFlatIndex(std::span{dst_coords.data(), rank});
                    (*storage_buffer_fp16)[dst_idx] = src_flat[i];
                    for (size_t d = rank; d > 0; --d)
                    {
                        if (++coords[d - 1] < source_cpu.shape[d - 1]) break;
                        coords[d - 1] = 0;
                    }
                }
            }
        }
        else if (storage_buffer && source_cpu.storage_buffer)
        {
            if (isContiguous() && source_cpu.isContiguous() && storage_buffer.get() != source_cpu.storage_buffer.get())
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
            else
            {
                std::vector<float> src_flat(source_cpu.total_elements);
                source_cpu.iterateCoordinates([&](size_t out_idx, size_t src_idx)
                                              { src_flat[out_idx] = (*source_cpu.storage_buffer)[src_idx]; });
                std::array<size_t, MAX_TENSOR_RANK> coords{};
                for (size_t i = 0; i < source_cpu.total_elements; ++i)
                {
                    std::array<size_t, MAX_TENSOR_RANK> dst_coords = coords;
                    dst_coords[axis] += start;
                    size_t dst_idx = resolveFlatIndex(std::span{dst_coords.data(), rank});
                    (*storage_buffer)[dst_idx] = src_flat[i];
                    for (size_t d = rank; d > 0; --d)
                    {
                        if (++coords[d - 1] < source_cpu.shape[d - 1]) break;
                        coords[d - 1] = 0;
                    }
                }
            }
        }
    }

void Cpu_Tensor_Impl::gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::gatherRows: TERNARY_INT2 is not supported");
        }
        size_t rank = shape.getRank();
        if (rank < 1)
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::gatherRows: Tensor rank must be at least 1");
        }
        size_t V = shape[0];
        size_t D = (rank == 1) ? 1 : getColumns();
        size_t S = indices.size();
        Data_Type src_type = data_type;

        auto &out_cpu = castToCpu(output);
        if (src_type == Data_Type::FLOAT16)
        {
            std::vector<float16_t> gathered(S * D, static_cast<float16_t>(0.0f));
            if (V > 0 && D > 0 && storage_buffer_fp16)
            {
                if (isContiguous() && byte_offset == 0)
                {
                    for (size_t s = 0; s < S; ++s)
                    {
                        int32_t row = indices[s];
                        if (row >= 0 && static_cast<size_t>(row) < V)
                        {
                            std::copy_n(storage_buffer_fp16->data() + static_cast<size_t>(row) * D, D, gathered.data() + s * D);
                        }
                    }
                }
                else
                {
                    for (size_t s = 0; s < S; ++s)
                    {
                        int32_t row = indices[s];
                        if (row >= 0 && static_cast<size_t>(row) < V)
                        {
                            std::array<size_t, MAX_TENSOR_RANK> coords{};
                            coords[0] = static_cast<size_t>(row);
                            for (size_t d = 0; d < D; ++d)
                            {
                                if (rank > 1)
                                {
                                    size_t rem = d;
                                    for (size_t a = rank; a > 1; --a)
                                    {
                                        size_t dim = shape[a - 1];
                                        coords[a - 1] = (dim > 0) ? (rem % dim) : 0;
                                        rem = (dim > 0) ? (rem / dim) : 0;
                                    }
                                }
                                gathered[s * D + d] = (*storage_buffer_fp16)[resolveFlatIndex(std::span{coords.data(), rank})];
                            }
                        }
                    }
                }
            }
            out_cpu.setDataType(src_type);
            out_cpu.setShape(Shape{S, D});
            out_cpu.setByteOffset(0);
            out_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(std::move(gathered));
            out_cpu.storage_buffer.reset();
            out_cpu.storage_buffer_u32.reset();
        }
        else
        {
            std::vector<float> gathered(S * D, 0.0F);
            if (V > 0 && D > 0 && storage_buffer)
            {
                if (isContiguous() && byte_offset == 0)
                {
                    for (size_t s = 0; s < S; ++s)
                    {
                        int32_t row = indices[s];
                        if (row >= 0 && static_cast<size_t>(row) < V)
                        {
                            std::copy_n(storage_buffer->data() + static_cast<size_t>(row) * D, D, gathered.data() + s * D);
                        }
                    }
                }
                else
                {
                    for (size_t s = 0; s < S; ++s)
                    {
                        int32_t row = indices[s];
                        if (row >= 0 && static_cast<size_t>(row) < V)
                        {
                            std::array<size_t, MAX_TENSOR_RANK> coords{};
                            coords[0] = static_cast<size_t>(row);
                            for (size_t d = 0; d < D; ++d)
                            {
                                if (rank > 1)
                                {
                                    size_t rem = d;
                                    for (size_t a = rank; a > 1; --a)
                                    {
                                        size_t dim = shape[a - 1];
                                        coords[a - 1] = (dim > 0) ? (rem % dim) : 0;
                                        rem = (dim > 0) ? (rem / dim) : 0;
                                    }
                                }
                                gathered[s * D + d] = (*storage_buffer)[resolveFlatIndex(std::span{coords.data(), rank})];
                            }
                        }
                    }
                }
            }
            out_cpu.setDataType(src_type);
            out_cpu.setShape(Shape{S, D});
            out_cpu.setByteOffset(0);
            out_cpu.storage_buffer = std::make_shared<std::vector<float>>(std::move(gathered));
            out_cpu.storage_buffer_fp16.reset();
            out_cpu.storage_buffer_u32.reset();
        }
    }

void Cpu_Tensor_Impl::contiguous(Tensor_Impl &output) const
{
        Data_Type src_type = data_type;
        Shape src_shape = shape;
        size_t src_total = total_elements;
        auto &output_cpu = castToCpu(output);

        if (src_type == Data_Type::TERNARY_INT2)
        {
            if (!isContiguous() || byte_offset != 0)
            {
                throw std::invalid_argument("Cpu_Tensor_Impl::contiguous: Non-contiguous TERNARY_INT2 view is not supported");
            }
            std::vector<uint32_t> words = storage_buffer_u32 ? *storage_buffer_u32 : std::vector<uint32_t>{};
            output_cpu.data_type = src_type;
            output_cpu.updateShapeAndStrides(src_shape);
            output_cpu.byte_offset = 0;
            output_cpu.storage_buffer_u32 = std::make_shared<std::vector<uint32_t>>(std::move(words));
            output_cpu.storage_buffer.reset();
            output_cpu.storage_buffer_fp16.reset();
            return;
        }

        if (src_type == Data_Type::FLOAT16)
        {
            std::vector<float16_t> dst_fp16(src_total, static_cast<float16_t>(0.0f));
            if (storage_buffer_fp16)
            {
                iterateCoordinates([this, &dst_fp16](size_t out_idx, size_t src_idx)
                                   { dst_fp16[out_idx] = (*storage_buffer_fp16)[src_idx]; });
            }
            output_cpu.data_type = src_type;
            output_cpu.updateShapeAndStrides(src_shape);
            output_cpu.byte_offset = 0;
            output_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(std::move(dst_fp16));
            output_cpu.storage_buffer.reset();
            output_cpu.storage_buffer_u32.reset();
        }
        else
        {
            std::vector<float> dst_fp32(src_total, 0.0F);
            if (storage_buffer)
            {
                iterateCoordinates([this, &dst_fp32](size_t out_idx, size_t src_idx)
                                   { dst_fp32[out_idx] = (*storage_buffer)[src_idx]; });
            }
            output_cpu.data_type = src_type;
            output_cpu.updateShapeAndStrides(src_shape);
            output_cpu.byte_offset = 0;
            output_cpu.storage_buffer = std::make_shared<std::vector<float>>(std::move(dst_fp32));
            output_cpu.storage_buffer_fp16.reset();
            output_cpu.storage_buffer_u32.reset();
        }
    }

void Cpu_Tensor_Impl::to(Data_Type target_type, Tensor_Impl &output) const
{
        Data_Type src_type = data_type;
        Shape src_shape = shape;
        size_t src_total = total_elements;
        auto &output_cpu = castToCpu(output);

        if (target_type == src_type)
        {
            contiguous(output);
            return;
        }

        std::vector<float> src_copy = getData();
        output_cpu.setDataType(target_type);
        output_cpu.updateShapeAndStrides(src_shape);
        output_cpu.byte_offset = 0;

        if (target_type == Data_Type::FLOAT16)
        {
            output_cpu.storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(src_total);
            convertFp32ToFp16(src_copy.data(), output_cpu.storage_buffer_fp16->data(), src_total);
            output_cpu.storage_buffer.reset();
            output_cpu.storage_buffer_u32.reset();
        }
        else if (target_type == Data_Type::TERNARY_INT2)
        {
            output_cpu.uploadData(src_copy);
        }
        else
        {
            output_cpu.storage_buffer = std::make_shared<std::vector<float>>(std::move(src_copy));
            output_cpu.storage_buffer_fp16.reset();
            output_cpu.storage_buffer_u32.reset();
        }
    }

void Cpu_Tensor_Impl::matmul(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16)
                               ? Data_Type::FLOAT16
                               : Data_Type::FLOAT32;
        Matmul_Broadcast_Plan plan = planMatmulBroadcast(*this, other, "Cpu_Tensor_Impl::matmul");

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);
        std::vector<float> res(plan.out_shape.getTotalElements(), 0.0F);

        if (plan.batch_total > 0 && plan.m_dim > 0 && plan.k_dim > 0 && plan.n_dim > 0)
        {
            std::array<size_t, MAX_TENSOR_RANK> batch_coords{};
            size_t batch_idx_a = 0;
            size_t batch_idx_b = 0;
            for (size_t b = 0; b < plan.batch_total; ++b)
            {
                size_t a_batch_offset = batch_idx_a * plan.m_dim * plan.k_dim;
                size_t b_batch_offset = batch_idx_b * plan.k_dim * plan.n_dim;
                size_t c_batch_offset = b * plan.m_dim * plan.n_dim;

                for (size_t i = 0; i < plan.m_dim; ++i)
                {
                    for (size_t k = 0; k < plan.k_dim; ++k)
                    {
                        float a_val = a_data[a_batch_offset + i * plan.k_dim + k];
                        for (size_t j = 0; j < plan.n_dim; ++j)
                        {
                            res[c_batch_offset + i * plan.n_dim + j] += a_val * b_data[b_batch_offset + k * plan.n_dim + j];
                        }
                    }
                }
                advanceBroadcastBatchCoords(batch_coords, plan.batch_shape, plan.strides_a, plan.strides_b, batch_idx_a, batch_idx_b);
            }
        }

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::matmul: batch={}, ({}x{}) x ({}x{}) -> ({}x{}), sample={}",
                                            plan.batch_total, plan.m_dim, plan.k_dim, plan.k_dim, plan.n_dim, plan.m_dim, plan.n_dim, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, plan.out_shape, std::move(res));
    }

void Cpu_Tensor_Impl::matdiv(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Cpu_Tensor_Impl temp_inverse(0, 0);
        other.inverse(temp_inverse);
        matmul(temp_inverse, output);
    }

void Cpu_Tensor_Impl::add(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16)
                               ? Data_Type::FLOAT16
                               : Data_Type::FLOAT32;
        Shape src_shape = shape;
        Shape other_shape = other.getShape();
        bool is_broadcast = (src_shape != other_shape);

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);
        Shape out_shape;
        std::vector<float> res = applyBroadcastElementwise(
            src_shape, a_data, other_shape, b_data, out_shape, "Cpu_Tensor_Impl::add",
            [](float x, float y) { return x + y; });

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::add: shape={}, broadcast={}, sample={}",
                                            out_shape.toString(), is_broadcast, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, out_shape, std::move(res));
    }

void Cpu_Tensor_Impl::sub(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16)
                               ? Data_Type::FLOAT16
                               : Data_Type::FLOAT32;
        Shape src_shape = shape;
        Shape other_shape = other.getShape();
        bool is_broadcast = (src_shape != other_shape);

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);
        Shape out_shape;
        std::vector<float> res = applyBroadcastElementwise(
            src_shape, a_data, other_shape, b_data, out_shape, "Cpu_Tensor_Impl::sub",
            [](float x, float y) { return x - y; });

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::sub: shape={}, broadcast={}, sample={}",
                                            out_shape.toString(), is_broadcast, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, out_shape, std::move(res));
    }

void Cpu_Tensor_Impl::mulScalar(float scalar, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        Shape out_shape = shape;
        const auto &a_data = getData();
        auto &output_cpu = castToCpu(output);
        std::vector<float> res(total_elements, 0.0F);

        for (size_t i = 0; i < total_elements; ++i)
        {
            res[i] = a_data[i] * scalar;
        }

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::mulScalar: scalar={}, elements={}, sample={}",
                                            scalar, total_elements, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, out_shape, std::move(res));
    }

void Cpu_Tensor_Impl::divScalar(float scalar, Tensor_Impl &output) const
{
        if (scalar == 0.0F || !std::isfinite(scalar) || !std::isfinite(1.0F / scalar))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::divScalar: Division by zero encountered"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        mulScalar(1.0F / scalar, output);
    }

void Cpu_Tensor_Impl::hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16)
                               ? Data_Type::FLOAT16
                               : Data_Type::FLOAT32;
        Shape src_shape = shape;
        Shape other_shape = other.getShape();

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);
        Shape out_shape;
        std::vector<float> res = applyBroadcastElementwise(
            src_shape, a_data, other_shape, b_data, out_shape, "Cpu_Tensor_Impl::hadamardMul",
            [](float x, float y) { return x * y; });

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardMul: elements={}, sample={}",
                                            out_shape.getTotalElements(), formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, out_shape, std::move(res));
    }

void Cpu_Tensor_Impl::hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16)
                               ? Data_Type::FLOAT16
                               : Data_Type::FLOAT32;
        Shape src_shape = shape;
        Shape other_shape = other.getShape();

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);
        Shape out_shape;
        std::vector<float> res = applyBroadcastElementwise(
            src_shape, a_data, other_shape, b_data, out_shape, "Cpu_Tensor_Impl::hadamardDiv",
            [](float x, float y)
            {
                if (!std::isfinite(y) || y == 0.0F)
                {
                    Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardDiv: Invalid or zero divisor"},
                                       Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
                    throw std::runtime_error("Division by zero in hadamardDiv");
                }
                return x / y;
            });

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::hadamardDiv: elements={}, sample={}",
                                            out_shape.getTotalElements(), formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, out_shape, std::move(res));
    }

void Cpu_Tensor_Impl::transpose(Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        const auto &a_data = getData();
        auto &output_cpu = castToCpu(output);
        size_t r_count = getRows();
        size_t c_count = getColumns();
        std::vector<float> res(r_count * c_count, 0.0F);

        for (size_t i = 0; i < r_count; ++i)
        {
            for (size_t j = 0; j < c_count; ++j)
            {
                res[j * r_count + i] = a_data[i * c_count + j];
            }
        }

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::transpose: ({}x{}) -> ({}x{}), sample={}",
                                            r_count, c_count, c_count, r_count, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, Shape{c_count, r_count}, std::move(res));
    }

void Cpu_Tensor_Impl::inverse(Tensor_Impl &output) const
{
        validateSquare();
        const auto &a_data = getData();
        size_t n = getRows();
        auto &output_cpu = castToCpu(output);
        output_cpu.setDataType(Data_Type::FLOAT32);
        if (n == 0)
        {
            writeCpuResult(output_cpu, Shape{0, 0}, {});
            return;
        }
        for (size_t idx = 0; idx < n * n; ++idx)
        {
            if (!std::isfinite(a_data[idx]))
            {
                Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::inverse: Non-finite element in matrix"},
                                   Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
                throw std::runtime_error("Tensor contains NaN or Inf in inverse");
            }
        }

        std::vector<double> aug(n * 2 * n, 0.0);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                aug[i * (2 * n) + j] = static_cast<double>(a_data[i * n + j]);
            }
            aug[i * (2 * n) + (n + i)] = 1.0;
        }

        for (size_t i = 0; i < n; ++i)
        {
            size_t pivot_row = i;
            double max_val = std::abs(aug[i * (2 * n) + i]);
            for (size_t k = i + 1; k < n; ++k)
            {
                double val = std::abs(aug[k * (2 * n) + i]);
                if (std::isnan(val) || val > max_val)
                {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if (!std::isfinite(max_val) || max_val < 1e-7)
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

            double pivot = aug[i * (2 * n) + i];
            for (size_t j = 0; j < 2 * n; ++j)
            {
                aug[i * (2 * n) + j] /= pivot;
            }

            for (size_t k = 0; k < n; ++k)
            {
                if (k != i)
                {
                    double factor = aug[k * (2 * n) + i];
                    for (size_t j = 0; j < 2 * n; ++j)
                    {
                        aug[k * (2 * n) + j] -= factor * aug[i * (2 * n) + j];
                    }
                }
            }
        }

        std::vector<float> res(n * n, 0.0F);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                res[i * n + j] = static_cast<float>(aug[i * (2 * n) + (n + j)]);
            }
        }

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::inverse: dimension={}, sample={}",
                                            n, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        writeCpuResult(output_cpu, Shape{n, n}, std::move(res));
    }

void Cpu_Tensor_Impl::normalize(Tensor_Impl &output) const
{
        const auto &a_data = getData();
        auto &output_cpu = castToCpu(output);
        output_cpu.setDataType(Data_Type::FLOAT32);
        if (total_elements == 0)
        {
            writeCpuResult(output_cpu, shape, {});
            return;
        }

        double max_abs = 0.0;
        for (float val : a_data)
        {
            if (!std::isfinite(val))
            {
                Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::normalize: Non-finite element in tensor"},
                                   Log_Level::LOG_ERROR, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
                throw std::runtime_error("Non-finite value in normalize");
            }
            max_abs = std::max(max_abs, std::abs(static_cast<double>(val)));
        }

        if (max_abs == 0.0)
        {
            std::vector<float> res = a_data;
            writeCpuResult(output_cpu, shape, std::move(res));
            return;
        }

        double scaled_sum_sq = 0.0;
        for (float val : a_data)
        {
            double scaled = static_cast<double>(val) / max_abs;
            scaled_sum_sq += scaled * scaled;
        }
        double norm = max_abs * std::sqrt(scaled_sum_sq);

        if (!std::isfinite(norm) || norm < 1e-8)
        {
            std::vector<float> res = a_data;
            writeCpuResult(output_cpu, shape, std::move(res));
            return;
        }

        std::vector<float> res(total_elements, 0.0F);
        for (size_t i = 0; i < total_elements; ++i)
        {
            res[i] = static_cast<float>(static_cast<double>(a_data[i]) / norm);
        }

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::NORMALIZATION_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::normalize: norm={:.4e}, sample={}",
                                            norm, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::NORMALIZATION_COMPUTE);
        }
        writeCpuResult(output_cpu, shape, std::move(res));
    }

void Cpu_Tensor_Impl::matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const
{
        Data_Type out_dt = (data_type == Data_Type::FLOAT16 || weights.getDataType() == Data_Type::FLOAT16)
                               ? Data_Type::FLOAT16
                               : Data_Type::FLOAT32;
        Matmul_Broadcast_Plan plan = planMatmulBroadcast(*this, weights, "Cpu_Tensor_Impl::matmulAdd");

        size_t b_total_elems = biases.getTotalElements();
        enum class Bias_Mode
        {
            PER_COL,
            PER_MATRIX,
            FULL_BATCH
        };
        Bias_Mode bias_mode = Bias_Mode::PER_COL;

        if (plan.batch_total == 0 || plan.m_dim == 0 || plan.n_dim == 0)
        {
            if (b_total_elems != plan.n_dim && b_total_elems != 0 &&
                b_total_elems != plan.m_dim * plan.n_dim &&
                b_total_elems != plan.batch_total * plan.m_dim * plan.n_dim)
            {
                throw std::invalid_argument("Cpu_Tensor_Impl::matmulAdd: Bias size does not match output dimensions");
            }
            auto &output_cpu = castToCpu(output);
            output_cpu.setDataType(out_dt);
            writeCpuResult(output_cpu, plan.out_shape, {});
            return;
        }

        if (b_total_elems == plan.n_dim)
        {
            bias_mode = Bias_Mode::PER_COL;
        }
        else if (b_total_elems == plan.m_dim * plan.n_dim)
        {
            bias_mode = Bias_Mode::PER_MATRIX;
        }
        else if (b_total_elems == plan.batch_total * plan.m_dim * plan.n_dim)
        {
            bias_mode = Bias_Mode::FULL_BATCH;
        }
        else
        {
            throw std::invalid_argument("Cpu_Tensor_Impl::matmulAdd: Bias size does not match output dimensions");
        }

        const auto &a_data = getData();
        const auto &w_data = weights.getData();
        const auto &b_data = biases.getData();
        auto &output_cpu = castToCpu(output);
        std::vector<float> res(plan.out_shape.getTotalElements(), 0.0F);

        std::array<size_t, MAX_TENSOR_RANK> batch_coords{};
        size_t batch_idx_a = 0;
        size_t batch_idx_w = 0;
        for (size_t b = 0; b < plan.batch_total; ++b)
        {
            size_t a_batch_offset = batch_idx_a * plan.m_dim * plan.k_dim;
            size_t w_batch_offset = batch_idx_w * plan.k_dim * plan.n_dim;
            size_t c_batch_offset = b * plan.m_dim * plan.n_dim;
            size_t b_batch_offset = (bias_mode == Bias_Mode::FULL_BATCH) ? (b * plan.m_dim * plan.n_dim) : 0;

            for (size_t i = 0; i < plan.m_dim; ++i)
            {
                for (size_t j = 0; j < plan.n_dim; ++j)
                {
                    float bias_val = 0.0F;
                    if (bias_mode == Bias_Mode::PER_COL)
                    {
                        bias_val = b_data[j];
                    }
                    else if (bias_mode == Bias_Mode::PER_MATRIX)
                    {
                        bias_val = b_data[i * plan.n_dim + j];
                    }
                    else
                    {
                        bias_val = b_data[b_batch_offset + i * plan.n_dim + j];
                    }
                    res[c_batch_offset + i * plan.n_dim + j] = bias_val;
                }

                for (size_t k = 0; k < plan.k_dim; ++k)
                {
                    float in_val = a_data[a_batch_offset + i * plan.k_dim + k];
                    for (size_t j = 0; j < plan.n_dim; ++j)
                    {
                        res[c_batch_offset + i * plan.n_dim + j] += in_val * w_data[w_batch_offset + k * plan.n_dim + j];
                    }
                }
            }
            advanceBroadcastBatchCoords(batch_coords, plan.batch_shape, plan.strides_a, plan.strides_b, batch_idx_a, batch_idx_w);
        }

        if (Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::DENSE_COMPUTE))
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::matmulAdd: batch={}, ({}x{}) x ({}x{}) -> ({}x{}), sample={}",
                                            plan.batch_total, plan.m_dim, plan.k_dim, plan.k_dim, plan.n_dim, plan.m_dim, plan.n_dim, formatDataSample(res)},
                               Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);
        }
        output_cpu.setDataType(out_dt);
        writeCpuResult(output_cpu, plan.out_shape, std::move(res));
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

        if (data_type == Data_Type::TERNARY_INT2)
        {
            if (!isContiguous() || byte_offset != 0)
            {
                throw std::invalid_argument("Cpu_Tensor_Impl::uploadData: Non-contiguous TERNARY_INT2 view is not supported");
            }
            size_t words = getTensorStorageBytes(data_type, shape) / sizeof(uint32_t);
            storage_buffer_u32 = std::make_shared<std::vector<uint32_t>>(words, 0u);
            storage_buffer.reset();
            storage_buffer_fp16.reset();
            if (total_elements > 0)
            {
                size_t rank = shape.getRank();
                size_t cols = (rank == 0) ? 1 : shape[rank - 1];
                size_t rows = (rank <= 1 || cols == 0) ? 1 : (total_elements / cols);
                packTernary(host_data.data(), storage_buffer_u32->data(), rows, cols);
            }
            return;
        }

        if (data_type == Data_Type::FLOAT16)
        {
            if (total_elements == 0)
            {
                if (!storage_buffer_fp16)
                {
                    storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>();
                }
                return;
            }
            size_t needed = computeMaxFlatIndex(shape, strides, byte_offset, data_type) + 1;
            if (!storage_buffer_fp16 || storage_buffer_fp16->size() < needed)
            {
                storage_buffer_fp16 = std::make_shared<std::vector<float16_t>>(total_elements);
                byte_offset = 0;
                strides = shape.computeContiguousStrides();
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

        if (total_elements == 0)
        {
            if (!storage_buffer)
            {
                storage_buffer = std::make_shared<std::vector<float>>();
            }
            return;
        }
        size_t needed = computeMaxFlatIndex(shape, strides, byte_offset, data_type) + 1;
        if (!storage_buffer || storage_buffer->size() < needed)
        {
            storage_buffer = std::make_shared<std::vector<float>>(total_elements);
            byte_offset = 0;
            strides = shape.computeContiguousStrides();
        }

        if (isContiguous() && byte_offset == 0)
        {
            std::copy_n(host_data.data(), total_elements, storage_buffer->data());
            return;
        }

        iterateCoordinates([this, &host_data](size_t in_idx, size_t dst_idx)
                           { (*storage_buffer)[dst_idx] = host_data[in_idx]; });
    }

void Cpu_Tensor_Impl::zero()
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            if (storage_buffer_u32)
            {
                std::fill(storage_buffer_u32->begin(), storage_buffer_u32->end(), 0u);
            }
            return;
        }
        if (data_type == Data_Type::FLOAT16)
        {
            if (storage_buffer_fp16)
            {
                float16_t zero_val = static_cast<float16_t>(0.0f);
                if (isContiguous() && byte_offset == 0 && storage_buffer_fp16->size() == total_elements)
                {
                    std::fill(storage_buffer_fp16->begin(), storage_buffer_fp16->end(), zero_val);
                }
                else
                {
                    iterateCoordinates([this, zero_val](size_t, size_t dst_idx)
                                       { (*storage_buffer_fp16)[dst_idx] = zero_val; });
                }
            }
        }
        else
        {
            if (storage_buffer)
            {
                if (isContiguous() && byte_offset == 0 && storage_buffer->size() == total_elements)
                {
                    std::fill(storage_buffer->begin(), storage_buffer->end(), 0.0f);
                }
                else
                {
                    iterateCoordinates([this](size_t, size_t dst_idx)
                                       { (*storage_buffer)[dst_idx] = 0.0f; });
                }
            }
        }
    }

void Cpu_Tensor_Impl::fill(float value)
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            uploadData(std::vector<float>(total_elements, value));
            return;
        }
        if (data_type == Data_Type::FLOAT16)
        {
            if (storage_buffer_fp16)
            {
                float16_t hval = static_cast<float16_t>(value);
                if (isContiguous() && byte_offset == 0 && storage_buffer_fp16->size() == total_elements)
                {
                    std::fill(storage_buffer_fp16->begin(), storage_buffer_fp16->end(), hval);
                }
                else
                {
                    iterateCoordinates([this, hval](size_t, size_t dst_idx)
                                       { (*storage_buffer_fp16)[dst_idx] = hval; });
                }
            }
        }
        else
        {
            if (storage_buffer)
            {
                if (isContiguous() && byte_offset == 0 && storage_buffer->size() == total_elements)
                {
                    std::fill(storage_buffer->begin(), storage_buffer->end(), value);
                }
                else
                {
                    iterateCoordinates([this, value](size_t, size_t dst_idx)
                                       { (*storage_buffer)[dst_idx] = value; });
                }
            }
        }
    }

void Cpu_Tensor_Impl::concatenateColumns(const Tensor_Impl &other, Tensor_Impl &output) const
{
        if (getRows() != other.getRows())
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::concatenateColumns: Row mismatch: {} vs {}",
                                            getRows(), other.getRows()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Row count mismatch in concatenateColumns");
        }
        if (data_type != other.getDataType())
        {
            throw std::invalid_argument("concatenateColumns: Data type mismatch between input tensors");
        }

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);

        size_t rows = getRows();
        size_t cols_a = getColumns();
        size_t cols_b = other.getColumns();
        size_t tot_cols = cols_a + cols_b;
        std::vector<float> res(rows * tot_cols, 0.0F);

        for (size_t r = 0; r < rows; ++r)
        {
            std::copy_n(a_data.data() + r * cols_a, cols_a, res.data() + r * tot_cols);
            std::copy_n(b_data.data() + r * cols_b, cols_b, res.data() + r * tot_cols + cols_a);
        }
        output_cpu.setDataType(data_type);
        writeCpuResult(output_cpu, Shape(rows, tot_cols), std::move(res));
    }

void Cpu_Tensor_Impl::concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const
{
        if (getColumns() != other.getColumns())
        {
            Logger::logMessage(Input_Format{"Cpu_Tensor_Impl::concatenateRows: Column mismatch: {} vs {}",
                                            getColumns(), other.getColumns()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Column count mismatch in concatenateRows");
        }
        if (data_type != other.getDataType())
        {
            throw std::invalid_argument("concatenateRows: Data type mismatch between input tensors");
        }

        const auto &a_data = getData();
        const auto &b_data = other.getData();
        auto &output_cpu = castToCpu(output);

        size_t tot_rows = getRows() + other.getRows();
        size_t cols = getColumns();
        std::vector<float> res(tot_rows * cols, 0.0F);

        std::copy(a_data.begin(), a_data.end(), res.begin());
        std::copy(b_data.begin(), b_data.end(), res.begin() + a_data.size());
        output_cpu.setDataType(data_type);
        writeCpuResult(output_cpu, Shape(tot_rows, cols), std::move(res));
    }

void Cpu_Tensor_Impl::splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const
{
        if (split_index == 0 || split_index >= getColumns())
        {
            throw std::out_of_range("Split index out of range in splitColumns");
        }

        const auto &a_data = getData();
        auto &left_cpu = castToCpu(result_left);
        auto &right_cpu = castToCpu(result_right);
        size_t rows = getRows();
        size_t total_cols = getColumns();
        size_t cols_left = split_index;
        size_t cols_right = total_cols - split_index;

        std::vector<float> left_vals(rows * cols_left, 0.0F);
        std::vector<float> right_vals(rows * cols_right, 0.0F);

        for (size_t r = 0; r < rows; ++r)
        {
            std::copy_n(a_data.data() + r * total_cols, cols_left, left_vals.data() + r * cols_left);
            std::copy_n(a_data.data() + r * total_cols + cols_left, cols_right, right_vals.data() + r * cols_right);
        }
        Data_Type src_type = data_type;
        left_cpu.setDataType(src_type);
        right_cpu.setDataType(src_type);
        writeCpuResult(left_cpu, Shape(rows, cols_left), std::move(left_vals));
        writeCpuResult(right_cpu, Shape(rows, cols_right), std::move(right_vals));
    }

void Cpu_Tensor_Impl::splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const
{
        if (split_index == 0 || split_index >= getRows())
        {
            throw std::out_of_range("Split index out of range in splitRows");
        }

        const auto &a_data = getData();
        auto &up_cpu = castToCpu(result_up);
        auto &down_cpu = castToCpu(result_down);
        size_t rows_up = split_index;
        size_t rows_down = getRows() - split_index;
        size_t cols = getColumns();

        size_t up_elems = rows_up * cols;
        size_t down_elems = rows_down * cols;
        std::vector<float> up_vals(a_data.begin(), a_data.begin() + up_elems);
        std::vector<float> down_vals(a_data.begin() + up_elems, a_data.begin() + up_elems + down_elems);

        Data_Type src_type = data_type;
        up_cpu.setDataType(src_type);
        down_cpu.setDataType(src_type);
        writeCpuResult(up_cpu, Shape(rows_up, cols), std::move(up_vals));
        writeCpuResult(down_cpu, Shape(rows_down, cols), std::move(down_vals));
    }

const std::vector<float> & Cpu_Tensor_Impl::getData() const
{
        if (data_type == Data_Type::FLOAT16)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (!storage_buffer_fp16)
            {
                materialized_cache.assign(total_elements, 0.0f);
                return materialized_cache;
            }
            materialized_cache.resize(total_elements);
            if (isContiguous() && byte_offset == 0)
            {
                convertFp16ToFp32(storage_buffer_fp16->data(), materialized_cache.data(), total_elements);
            }
            else
            {
                iterateCoordinates([this](size_t out_idx, size_t src_idx)
                                   { materialized_cache[out_idx] = static_cast<float>((*storage_buffer_fp16)[src_idx]); });
            }
            return materialized_cache;
        }

        if (data_type == Data_Type::TERNARY_INT2)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (!storage_buffer_u32 || total_elements == 0)
            {
                materialized_cache.assign(total_elements, 0.0f);
                return materialized_cache;
            }
            materialized_cache.resize(total_elements);
            size_t rank = shape.getRank();
            size_t cols = (rank == 0) ? 1 : shape[rank - 1];
            size_t rows = (rank <= 1 || cols == 0) ? 1 : (total_elements / cols);
            unpackTernary(storage_buffer_u32->data(), materialized_cache.data(), rows, cols, 1.0f);
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
        if (!storage_buffer)
        {
            materialized_cache.assign(total_elements, 0.0f);
            return materialized_cache;
        }
        materialized_cache.resize(total_elements);
        iterateCoordinates([this](size_t out_idx, size_t src_idx)
                           { materialized_cache[out_idx] = (*storage_buffer)[src_idx]; });
        return materialized_cache;
    }

Mutable_Storage_Handle Cpu_Tensor_Impl::getStorage()
{
        if (data_type == Data_Type::FLOAT16 || data_type == Data_Type::TERNARY_INT2)
        {
            throw std::logic_error("Cpu_Tensor_Impl::getStorage: Mutable FP32 storage is not available for FLOAT16 or TERNARY_INT2");
        }
        if (!isContiguous() || byte_offset != 0)
        {
            auto owned_data = getData();
            storage_buffer = std::make_shared<std::vector<float>>(std::move(owned_data));
            byte_offset = 0;
            strides = shape.computeContiguousStrides();
        }
        if (!storage_buffer)
        {
            storage_buffer = std::make_shared<std::vector<float>>(total_elements, 0.0F);
        }
        return std::ref(*storage_buffer);
    }

const std::vector<uint32_t> &Cpu_Tensor_Impl::getPackedData() const
{
    if (data_type != Data_Type::TERNARY_INT2 || !storage_buffer_u32)
    {
        static const std::vector<uint32_t> empty_vec;
        return empty_vec;
    }
    return *storage_buffer_u32;
}

void Cpu_Tensor_Impl::uploadPackedData(const std::vector<uint32_t> &packed_data)
{
    if (data_type != Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("uploadPackedData called on non-TERNARY_INT2 tensor");
    }
    size_t expected_words = getTensorStorageBytes(data_type, shape) / sizeof(uint32_t);
    if (packed_data.size() != expected_words)
    {
        throw std::invalid_argument("uploadPackedData size mismatch");
    }
    if (!storage_buffer_u32)
    {
        storage_buffer_u32 = std::make_shared<std::vector<uint32_t>>(packed_data);
    }
    else
    {
        *storage_buffer_u32 = packed_data;
    }
}

void Cpu_Tensor_Impl::packTernary(const float *src, uint32_t *dst, size_t rows, size_t cols, float inv_gamma)
{
    if (!src || !dst) return;
    const size_t words_per_row = (cols + 15) / 16;
    for (size_t r = 0; r < rows; ++r)
    {
        const float *src_row = src + r * cols;
        uint32_t *dst_row = dst + r * words_per_row;
        for (size_t w = 0; w < words_per_row; ++w)
        {
            uint32_t packed_val = 0;
            for (size_t bit_idx = 0; bit_idx < 16; ++bit_idx)
            {
                size_t col_idx = w * 16 + bit_idx;
                if (col_idx < cols)
                {
                    float val = src_row[col_idx];
                    int q = 0;
                    if (std::isfinite(val))
                    {
                        float scaled = std::abs(val) * inv_gamma;
                        if (std::floor(scaled + 0.5f) >= 1.0f)
                        {
                            q = (val < 0.0f) ? -1 : 1;
                        }
                    }
                    uint32_t code = static_cast<uint32_t>(q) & 3u;
                    packed_val |= (code << (bit_idx * 2));
                }
            }
            dst_row[w] = packed_val;
        }
    }
}

void Cpu_Tensor_Impl::unpackTernary(const uint32_t *src, float *dst, size_t rows, size_t cols, float gamma)
{
    if (!src || !dst) return;
    static constexpr int TERNARY_LUT[4] = {0, 1, 0, -1};
    const size_t words_per_row = (cols + 15) / 16;
    for (size_t r = 0; r < rows; ++r)
    {
        const uint32_t *src_row = src + r * words_per_row;
        float *dst_row = dst + r * cols;
        for (size_t w = 0; w < words_per_row; ++w)
        {
            uint32_t packed_val = src_row[w];
            for (size_t bit_idx = 0; bit_idx < 16; ++bit_idx)
            {
                size_t col_idx = w * 16 + bit_idx;
                if (col_idx < cols)
                {
                    uint32_t code = (packed_val >> (bit_idx * 2)) & 3u;
                    int val = TERNARY_LUT[code];
                    dst_row[col_idx] = static_cast<float>(val) * gamma;
                }
            }
        }
    }
}

void Cpu_Tensor_Impl::ternaryMatmul(const float *A, const uint32_t *B_packed, float *C,
                                    size_t M, size_t K, size_t N, float gamma, const float *biases)
{
    if (!A || !B_packed || !C) return;
    static constexpr int TERNARY_LUT[4] = {0, 1, 0, -1};
    const size_t words_per_row = (K + 15) / 16;

    for (size_t m = 0; m < M; ++m)
    {
        const float *a_row = A + m * K;
        float *c_row = C + m * N;

        for (size_t n = 0; n < N; ++n)
        {
            float acc = (biases != nullptr) ? biases[n] : 0.0f;
            const uint32_t *b_row = B_packed + n * words_per_row;

            for (size_t w = 0; w < words_per_row; ++w)
            {
                uint32_t packed_val = b_row[w];
                size_t k_base = w * 16;
                size_t k_limit = std::min<size_t>(16, K - k_base);

                for (size_t bit_idx = 0; bit_idx < k_limit; ++bit_idx)
                {
                    uint32_t code = (packed_val >> (bit_idx * 2)) & 3u;
                    int val = TERNARY_LUT[code];
                    acc += a_row[k_base + bit_idx] * (static_cast<float>(val) * gamma);
                }
            }
            c_row[n] = acc;
        }
    }
}

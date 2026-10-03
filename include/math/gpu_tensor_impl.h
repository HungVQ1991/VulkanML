#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "tensor_impl.h"

struct Matrix_Dimensions
{
    uint32_t batch_count = 1;
    uint32_t rows_a = 0;
    uint32_t columns_a = 0;
    uint32_t columns_b = 0;
    uint32_t broadcast_b = 0;
};

struct Elementwise_Dimensions
{
    uint32_t total_elements = 0;
    uint32_t columns = 0;
    uint32_t is_broadcast = 0;
};

struct Transpose_Dimensions
{
    uint32_t rows = 0;
    uint32_t columns = 0;
};

struct Contiguous_Push_Constants
{
    uint32_t total_elements = 0;
    uint32_t rank = 0;
    uint32_t offset_elements = 0;
    uint32_t shape_0 = 1;
    uint32_t shape_1 = 1;
    uint32_t shape_2 = 1;
    uint32_t shape_3 = 1;
    uint32_t shape_4 = 1;
    uint32_t shape_5 = 1;
    uint32_t stride_0 = 1;
    uint32_t stride_1 = 1;
    uint32_t stride_2 = 1;
    uint32_t stride_3 = 1;
    uint32_t stride_4 = 1;
    uint32_t stride_5 = 1;
};

class Gpu_Tensor_Impl : public Tensor_Impl
{
private:
    std::shared_ptr<gpu::vector> storage;
    mutable std::vector<float> host_cache;
    mutable std::shared_ptr<gpu::vector> cached_fp16_storage;
    mutable bool is_fp16_cache_dirty = true;

    static inline bool is_graph_logging_enabled = true;

public:
    static inline size_t distinct_operations_count;

    void invalidateFp16Cache() noexcept override
    {
        is_fp16_cache_dirty = true;
    }

    void prewarmFp16Cache() override
    {
        getEffectiveFp16Storage();
    }

    std::shared_ptr<gpu::vector> getEffectiveFp16Storage() const
    {
        if (data_type == Data_Type::FLOAT16)
        {
            return storage;
        }

        if (total_elements == 0)
        {
            return storage;
        }

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        if (!cached_fp16_storage || cached_fp16_storage->getSize() < total_elements)
        {
            cached_fp16_storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, Data_Type::FLOAT16);
            is_fp16_cache_dirty = true;
        }

        if (is_fp16_cache_dirty)
        {
            struct Cast_Push_Constants
            {
                uint32_t total_elements = 0;
            } pc{static_cast<uint32_t>(total_elements)};

            pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16,
                        {effective_self->storage, cached_fp16_storage},
                        pc,
                        (pc.total_elements + 255) / 256);
            is_fp16_cache_dirty = false;
        }

        return cached_fp16_storage;
    }

private:
    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum pipeline_id,
                     const std::vector<std::shared_ptr<gpu::vector>> &buffers,
                     const Push_Constants_Type &push_constants,
                     uint32_t workgroup_count_x,
                     uint32_t workgroup_count_y = 1,
                     uint32_t workgroup_count_z = 1) const
    {
        if (buffers.size() > 16)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::pushToGraph: Exceeded max buffer count"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::GRAPH_RECORDING);
            throw std::runtime_error("Exceeded maximum supported buffer count");
        }

        Compute_Node node;
        node.pipeline_id = pipeline_id;
        node.buffers = buffers;

        const auto *byte_ptr = reinterpret_cast<const std::uint8_t *>(&push_constants);
        node.push_constants_data.assign(byte_ptr, byte_ptr + sizeof(Push_Constants_Type));

        node.workgroup_count_x = workgroup_count_x;
        node.workgroup_count_y = workgroup_count_y;
        node.workgroup_count_z = workgroup_count_z;

        Execution_Engine::getInstance().getCurrentGraph().addNode(std::move(node));

        if (is_graph_logging_enabled)
        {
            is_graph_logging_enabled = Logger::logMessage(
                Input_Format{"Gpu_Tensor_Impl::pushToGraph: Op={}, Dispatch=({}, {}, {}), Buffers={}",
                             magic_enum::enum_name(pipeline_id),
                             workgroup_count_x, workgroup_count_y, workgroup_count_z,
                             buffers.size()},
                Log_Level::LOG_DEBUG, true, distinct_operations_count, Log_Feature::GRAPH_RECORDING);
        }
    }

    const Gpu_Tensor_Impl &castToGpu(const Tensor_Impl &other) const noexcept
    {
        return static_cast<const Gpu_Tensor_Impl &>(other);
    }

    Gpu_Tensor_Impl &castToGpu(Tensor_Impl &other) const noexcept
    {
        return static_cast<Gpu_Tensor_Impl &>(other);
    }

    std::shared_ptr<Gpu_Tensor_Impl> ensureContiguousSelf() const
    {
        if (isContiguous() && byte_offset == 0)
        {
            return nullptr;
        }
        auto contiguous_tensor = std::make_shared<Gpu_Tensor_Impl>(shape, data_type);
        contiguous(*contiguous_tensor);
        return contiguous_tensor;
    }

    template <typename Pipeline_Enum>
    void executeElementwise(const Tensor_Impl &other, Tensor_Impl &output, Pipeline_Enum pipeline_id, bool is_broadcast_allowed) const
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        auto &output_gpu = castToGpu(output);

        bool is_broadcast = false;
        if (is_broadcast_allowed)
        {
            bool same_shape = (shape == effective_other->shape);
            is_broadcast = (effective_other->getRows() == 1 && getColumns() == effective_other->getColumns());
            if (!same_shape && !is_broadcast)
            {
                validateSameDimensions(*effective_other);
            }
        }
        else
        {
            validateSameDimensions(*effective_other);
        }

        bool is_fp16_pipe = (pipeline_id == Compute_Pipeline::ADD_FP16 ||
                             pipeline_id == Compute_Pipeline::SUB_FP16 ||
                             pipeline_id == Compute_Pipeline::HADAMARD_MUL_FP16 ||
                             pipeline_id == Compute_Pipeline::HADAMARD_DIV_FP16);
        if (is_fp16_pipe)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            output_gpu.setDataType(data_type);
        }
        output_gpu.reshape(shape);

        Elementwise_Dimensions dims{
            .total_elements = static_cast<uint32_t>(total_elements),
            .columns = static_cast<uint32_t>(getColumns()),
            .is_broadcast = static_cast<uint32_t>(is_broadcast ? 1 : 0)};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::executeElementwise: total={}, cols={}, broadcast={}",
                                        dims.total_elements, dims.columns, dims.is_broadcast},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);

        std::shared_ptr<gpu::vector> buf_self;
        std::shared_ptr<gpu::vector> buf_other;
        if (is_fp16_pipe)
        {
            buf_self = (effective_self->getDataType() == Data_Type::FLOAT16)
                           ? effective_self->storage
                           : effective_self->getEffectiveFp16Storage();
            buf_other = (effective_other->getDataType() == Data_Type::FLOAT16)
                            ? effective_other->storage
                            : effective_other->getEffectiveFp16Storage();
        }
        else
        {
            buf_self = effective_self->storage;
            buf_other = effective_other->storage;
        }

        pushToGraph(pipeline_id, {buf_self, buf_other, output_gpu.storage}, dims, (dims.total_elements + 255) / 256);
    }

public:
    Gpu_Tensor_Impl(size_t rows, size_t columns)
    {
        updateShapeAndStrides(Shape{rows, columns});
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements);
        }
    }

    Gpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data)
    {
        updateShapeAndStrides(Shape{rows, columns});
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), host_data);
        }
    }

    explicit Gpu_Tensor_Impl(Shape tensor_shape, Data_Type type = Data_Type::FLOAT32)
    {
        data_type = type;
        updateShapeAndStrides(tensor_shape);
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
        }
    }

    Gpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data, Data_Type type = Data_Type::FLOAT32)
    {
        data_type = type;
        updateShapeAndStrides(tensor_shape);
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), host_data);
            if (data_type != Data_Type::FLOAT32)
            {
                storage->setDataType(data_type);
                storage->uploadData(host_data);
            }
        }
    }

    Gpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<gpu::vector> existing_storage, size_t offset_bytes, Data_Type type = Data_Type::FLOAT32)
    {
        data_type = type;
        shape = tensor_shape;
        strides = tensor_strides;
        storage = std::move(existing_storage);
        byte_offset = offset_bytes;
        total_elements = shape.getTotalElements();
    }

    ~Gpu_Tensor_Impl() noexcept override = default;

    void reshape(size_t rows, size_t columns) override
    {
        reshape(Shape{rows, columns});
    }

    void reshape(Shape new_shape) override
    {
        size_t new_total = new_shape.getTotalElements();
        if (!isContiguous() || byte_offset != 0)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::reshape: Cannot reshape non-contiguous view directly"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::runtime_error("Cannot reshape non-contiguous GPU tensor view");
        }

        if (shape == new_shape && storage && storage->getSize() == new_total && storage->getDataType() == data_type)
        {
            return;
        }

        if (storage.use_count() > 1 && (total_elements != new_total || storage->getDataType() != data_type))
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), new_total, data_type);
            is_fp16_cache_dirty = true;
        }
        else if (!storage || storage->getSize() != new_total || storage->getDataType() != data_type)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), new_total, data_type);
            is_fp16_cache_dirty = true;
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
        auto &output_gpu = static_cast<Gpu_Tensor_Impl &>(output);
        output_gpu.data_type = data_type;
        output_gpu.shape = new_shape;
        output_gpu.strides = new_strides;
        output_gpu.storage = storage;
        output_gpu.byte_offset = byte_offset;
        output_gpu.total_elements = total_elements;
    }

    void slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const override
    {
        Shape new_shape = shape;
        new_shape[axis] = length;
        size_t add_bytes = start * strides[axis] * getDataTypeSize(data_type);

        auto &output_gpu = static_cast<Gpu_Tensor_Impl &>(output);
        output_gpu.data_type = data_type;
        output_gpu.shape = new_shape;
        output_gpu.strides = strides;
        output_gpu.storage = storage;
        output_gpu.byte_offset = byte_offset + add_bytes;
        output_gpu.total_elements = new_shape.getTotalElements();
    }

    void updateSlice(size_t axis, size_t start, const Tensor_Impl &source) override
    {
        const auto &source_gpu = static_cast<const Gpu_Tensor_Impl &>(source);
        if (axis >= shape.getRank())
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: axis out of range");
        }
        size_t src_axis_len = source_gpu.shape[axis];
        if (start + src_axis_len > shape[axis])
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: slice range exceeds destination dimension");
        }

        size_t elem_size = getDataTypeSize(data_type);
        size_t outer_count = 1;
        for (size_t i = 0; i < axis; ++i) outer_count *= shape[i];
        size_t inner_count = 1;
        for (size_t i = axis + 1; i < shape.getRank(); ++i) inner_count *= shape[i];
        size_t this_axis_len = shape[axis];
        size_t copy_bytes = src_axis_len * inner_count * elem_size;

        auto &context = Execution_Engine::getInstance().getContext();
        for (size_t outer = 0; outer < outer_count; ++outer)
        {
            size_t dst_off = byte_offset + (outer * this_axis_len + start) * inner_count * elem_size;
            size_t src_off = source_gpu.byte_offset + (outer * src_axis_len) * inner_count * elem_size;
            context.copyBuffer(source_gpu.storage->getBuffer(), storage->getBuffer(), copy_bytes, src_off, dst_off);
        }
        context.executePendingTransfers();
        host_cache.clear();
        is_fp16_cache_dirty = true;
    }

    void gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const override
    {
        std::vector<float> idx_float(indices.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            idx_float[i] = static_cast<float>(indices[i]);
        }
        Gpu_Tensor_Impl indices_gpu(Shape{ indices.size() }, idx_float);
        embeddingForward(indices_gpu, output);
    }

    void contiguous(Tensor_Impl &output) const override
    {
        auto &output_gpu = static_cast<Gpu_Tensor_Impl &>(output);
        output_gpu.setDataType(data_type);
        output_gpu.reshape(shape);

        if (isContiguous())
        {
            Execution_Engine::getInstance().getContext().copyBuffer(
                storage->getBuffer(),
                output_gpu.storage->getBuffer(),
                total_elements * getDataTypeSize(data_type),
                byte_offset,
                0);
            return;
        }

        Contiguous_Push_Constants constants{
            .total_elements = static_cast<uint32_t>(total_elements),
            .rank = static_cast<uint32_t>(shape.getRank()),
            .offset_elements = static_cast<uint32_t>(byte_offset / getDataTypeSize(data_type))};

        for (size_t i = 0; i < shape.getRank() && i < 6; ++i)
        {
            if (i == 0)
            {
                constants.shape_0 = static_cast<uint32_t>(shape[0]);
                constants.stride_0 = static_cast<uint32_t>(strides[0]);
            }
            if (i == 1)
            {
                constants.shape_1 = static_cast<uint32_t>(shape[1]);
                constants.stride_1 = static_cast<uint32_t>(strides[1]);
            }
            if (i == 2)
            {
                constants.shape_2 = static_cast<uint32_t>(shape[2]);
                constants.stride_2 = static_cast<uint32_t>(strides[2]);
            }
            if (i == 3)
            {
                constants.shape_3 = static_cast<uint32_t>(shape[3]);
                constants.stride_3 = static_cast<uint32_t>(strides[3]);
            }
            if (i == 4)
            {
                constants.shape_4 = static_cast<uint32_t>(shape[4]);
                constants.stride_4 = static_cast<uint32_t>(strides[4]);
            }
            if (i == 5)
            {
                constants.shape_5 = static_cast<uint32_t>(shape[5]);
                constants.stride_5 = static_cast<uint32_t>(strides[5]);
            }
        }

        Compute_Pipeline pipe = (data_type == Data_Type::FLOAT16) ? Compute_Pipeline::CONTIGUOUS_FP16 : Compute_Pipeline::CONTIGUOUS;
        pushToGraph(pipe, {storage, output_gpu.storage}, constants, (constants.total_elements + 255) / 256);
    }

    void to(Data_Type target_type, Tensor_Impl &output) const override
    {
        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(target_type);
        output_gpu.reshape(shape);

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        if (target_type == data_type)
        {
            Execution_Engine::getInstance().getContext().copyBuffer(
                effective_self->storage->getBuffer(),
                output_gpu.storage->getBuffer(),
                total_elements * getDataTypeSize(data_type));
            return;
        }

        struct Cast_Push_Constants
        {
            uint32_t total_elements = 0;
        } pc{static_cast<uint32_t>(total_elements)};

        if (data_type == Data_Type::FLOAT32 && target_type == Data_Type::FLOAT16)
        {
            pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16,
                        {effective_self->storage, output_gpu.storage},
                        pc,
                        (pc.total_elements + 255) / 256);
        }
        else if (data_type == Data_Type::FLOAT16 && target_type == Data_Type::FLOAT32)
        {
            pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32,
                        {effective_self->storage, output_gpu.storage},
                        pc,
                        (pc.total_elements + 255) / 256);
        }
    }

    void matmul(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

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

        auto &output_gpu = castToGpu(output);
        if (data_type == Data_Type::FLOAT16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        output_gpu.reshape(out_shape);

        Matrix_Dimensions dims{
            .batch_count = static_cast<uint32_t>(b_dim),
            .rows_a = static_cast<uint32_t>(m_dim),
            .columns_a = static_cast<uint32_t>(k_dim),
            .columns_b = static_cast<uint32_t>(n_dim),
            .broadcast_b = broadcast_b ? 1u : 0u};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::matmul: batch={}, rows_a={}, cols_a={}, cols_b={}",
                                        dims.batch_count, dims.rows_a, dims.columns_a, dims.columns_b},
                           Log_Level::LOG_DEBUG, true, 1, Log_Feature::DENSE_COMPUTE);

        bool use_fp16 = (data_type == Data_Type::FLOAT16 ||
                         other_gpu.getDataType() == Data_Type::FLOAT16 ||
                         Execution_Engine::getInstance().isCooperativeMatrixEnabled());

        if (use_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
            output_gpu.reshape(out_shape);

            auto buf_a = effective_self->getEffectiveFp16Storage();
            auto buf_b = effective_other->getEffectiveFp16Storage();

            Compute_Pipeline pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::MATMUL_COOPMAT_FP16
                : Compute_Pipeline::MATMUL_FP16;

            pushToGraph(pipeline,
                        {buf_a, buf_b, output_gpu.storage},
                        dims, (dims.columns_b + 15) / 16, (dims.rows_a + 15) / 16, dims.batch_count);
        }
        else
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
            output_gpu.reshape(out_shape);

            pushToGraph(Compute_Pipeline::MATMUL,
                        {effective_self->storage, effective_other->storage, output_gpu.storage},
                        dims, (dims.columns_b + 15) / 16, (dims.rows_a + 15) / 16, dims.batch_count);
        }
    }

    void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        Gpu_Tensor_Impl temp_inv(0, 0);
        other.inverse(temp_inv);
        matmul(temp_inv, output);
    }

    void add(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ADD_FP16 : Compute_Pipeline::ADD;
        executeElementwise(other, output, pipe, true);
    }

    void sub(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SUB_FP16 : Compute_Pipeline::SUB;
        executeElementwise(other, output, pipe, true);
    }

    void mulScalar(float scalar, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(data_type);
        output_gpu.reshape(shape);

        struct Scalar_Constants
        {
            uint32_t total_elements;
            float scalar;
        } constants{static_cast<uint32_t>(total_elements), scalar};

        Compute_Pipeline pipe = (data_type == Data_Type::FLOAT16) ? Compute_Pipeline::MUL_SCALAR_FP16 : Compute_Pipeline::MUL_SCALAR;
        pushToGraph(pipe, {effective_self->storage, output_gpu.storage}, constants, (total_elements + 255) / 256);
    }

    void divScalar(float scalar, Tensor_Impl &output) const override
    {
        if (std::abs(scalar) < 1e-8F)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::divScalar: Division by zero"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        mulScalar(1.0F / scalar, output);
    }

    void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::HADAMARD_MUL_FP16 : Compute_Pipeline::HADAMARD_MUL;
        executeElementwise(other, output, pipe, true);
    }

    void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::HADAMARD_DIV_FP16 : Compute_Pipeline::HADAMARD_DIV;
        executeElementwise(other, output, pipe, true);
    }

    void transpose(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(data_type);
        output_gpu.reshape(getColumns(), getRows());

        Transpose_Dimensions dims{
            .rows = static_cast<uint32_t>(getRows()),
            .columns = static_cast<uint32_t>(getColumns())};

        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::TRANSPOSE_FP16 : Compute_Pipeline::TRANSPOSE;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(pipe, {buf_self, output_gpu.storage}, dims,
                    (dims.columns + 15) / 16, (dims.rows + 15) / 16);
    }

    void inverse(Tensor_Impl &output) const override
    {
        validateSquare();
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        uint32_t dim_size = static_cast<uint32_t>(getRows());
        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(dim_size, dim_size);

        struct Inverse_Constants
        {
            uint32_t dimension_size;
        } constants{dim_size};

        pushToGraph(Compute_Pipeline::MATRIX_INVERSE, {effective_self->storage, output_gpu.storage}, constants, 1, 1, 1);
    }

    void normalize(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(shape);

        struct Norm_Constants
        {
            uint32_t total_elements;
        } constants{static_cast<uint32_t>(total_elements)};

        pushToGraph(Compute_Pipeline::NORMALIZE, {effective_self->storage, output_gpu.storage}, constants, 1, 1, 1);
    }

    void relu(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        if (data_type == Data_Type::FLOAT16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        output_gpu.reshape(shape);
        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::RELU_FP16 : Compute_Pipeline::RELU,
                    {effective_self->storage, output_gpu.storage},
                    static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }

    void reluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        validateSameDimensions(output_gradient);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        if (data_type == Data_Type::FLOAT16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        in_grad_gpu.reshape(shape);

        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::RELU_BACKWARD_FP16 : Compute_Pipeline::RELU_BACKWARD,
                    {effective_self->storage, effective_grad->storage, in_grad_gpu.storage},
                    static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }

    void gelu(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        if (data_type == Data_Type::FLOAT16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        output_gpu.reshape(shape);
        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::GELU_FP16 : Compute_Pipeline::GELU,
                    {effective_self->storage, output_gpu.storage},
                    static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }

    void geluBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        validateSameDimensions(output_gradient);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        if (data_type == Data_Type::FLOAT16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        in_grad_gpu.reshape(shape);

        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::GELU_BACKWARD_FP16 : Compute_Pipeline::GELU_BACKWARD,
                    {effective_self->storage, effective_grad->storage, in_grad_gpu.storage},
                    static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }

    void softmax(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(data_type);
        output_gpu.reshape(getRows(), getColumns());

        struct Softmax_Constants
        {
            uint32_t rows;
            uint32_t columns;
        } constants{static_cast<uint32_t>(getRows()), static_cast<uint32_t>(getColumns())};

        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SOFTMAX_FP16 : Compute_Pipeline::SOFTMAX;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(pipe, {buf_self, output_gpu.storage}, constants, constants.rows, 1, 1);
    }

    void softmaxBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        validateSameDimensions(output_gradient);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        in_grad_gpu.setDataType(data_type);
        in_grad_gpu.reshape(getRows(), getColumns());

        struct Softmax_Constants
        {
            uint32_t rows;
            uint32_t columns;
        } constants{static_cast<uint32_t>(getRows()), static_cast<uint32_t>(getColumns())};

        bool is_fp16 = (data_type == Data_Type::FLOAT16 || effective_grad->getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SOFTMAX_BACKWARD_FP16 : Compute_Pipeline::SOFTMAX_BACKWARD;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;
        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->storage;

        pushToGraph(pipe, {buf_self, buf_grad, in_grad_gpu.storage}, constants, constants.rows, 1, 1);
    }

    void sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient = 0.0F, float inv_scale = 1.0F) override
    {
        validateSameDimensions(gradient);
        const auto &grad_gpu = castToGpu(gradient);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0f, 1.0f, inv_scale, current_frame);

        struct Sgd_Constants
        {
            uint32_t total_elements;
            float max_gradient;
        } constants{static_cast<uint32_t>(total_elements), max_gradient};

        auto dyn_buf = engine.getDynamicOptimizerBuffer(current_frame);
        pushToGraph(Compute_Pipeline::SGD_UPDATE, {storage, effective_grad->storage, dyn_buf}, constants, (total_elements + 255) / 256);
        is_fp16_cache_dirty = true;
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
                    float inv_scale = 1.0F,
                    float weight_decay = 0.0F) override
    {
        validateSameDimensions(gradient);
        validateSameDimensions(first_moment);
        validateSameDimensions(second_moment);

        const auto &grad_gpu = castToGpu(gradient);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        const auto &m_gpu = castToGpu(first_moment);
        const auto &v_gpu = castToGpu(second_moment);

        size_t effective_t = std::max<size_t>(timestep, 1);
        float bc1 = std::max(1.0F - std::pow(beta1, static_cast<float>(effective_t)), 1e-8F);
        float bc2 = std::max(1.0F - std::pow(beta2, static_cast<float>(effective_t)), 1e-8F);

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0F / bc1, 1.0F / std::sqrt(bc2), inv_scale, current_frame);

        struct Adam_Constants
        {
            uint32_t total_elements;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float weight_decay;
        } constants{static_cast<uint32_t>(total_elements), beta1, beta2, epsilon, max_gradient, weight_decay};

        auto dyn_buf = engine.getDynamicOptimizerBuffer(current_frame);
        bool use_fp16 = (data_type == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled() || cached_fp16_storage != nullptr);
        if (use_fp16)
        {
            if (!cached_fp16_storage || cached_fp16_storage->getSize() < total_elements)
            {
                cached_fp16_storage = std::make_shared<gpu::vector>(engine.getContext(), total_elements, Data_Type::FLOAT16);
            }
            pushToGraph(Compute_Pipeline::ADAM_UPDATE_FP16,
                        {storage, effective_grad->storage, m_gpu.storage, v_gpu.storage, dyn_buf, cached_fp16_storage},
                        constants, (total_elements + 255) / 256);
            is_fp16_cache_dirty = false;
        }
        else
        {
            pushToGraph(Compute_Pipeline::ADAM_UPDATE,
                        {storage, effective_grad->storage, m_gpu.storage, v_gpu.storage, dyn_buf},
                        constants, (total_elements + 255) / 256);
            is_fp16_cache_dirty = true;
        }
    }

    void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &w_gpu = castToGpu(weights);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        const auto &b_gpu = castToGpu(biases);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

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

        auto &output_gpu = castToGpu(output);

        size_t b_total_elems = biases.getTotalElements();
        uint32_t broadcast_b_flag = 0;
        if (b_total_elems == n_dim || biases.getRows() == 1)
        {
            broadcast_b_flag = 1;
        }
        else if (b_total_elems == m_dim * n_dim)
        {
            broadcast_b_flag = (b_dim > 1) ? 2 : 0;
        }
        else
        {
            broadcast_b_flag = 0;
        }

        struct Matmul_Add_Constants
        {
            uint32_t batch_count;
            uint32_t rows_x;
            uint32_t columns_x;
            uint32_t columns_weights;
            uint32_t broadcast_w;
            uint32_t broadcast_b;
        } constants{
            .batch_count = static_cast<uint32_t>(b_dim),
            .rows_x = static_cast<uint32_t>(m_dim),
            .columns_x = static_cast<uint32_t>(k_dim),
            .columns_weights = static_cast<uint32_t>(n_dim),
            .broadcast_w = broadcast_w ? 1u : 0u,
            .broadcast_b = broadcast_b_flag};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::matmulAdd: batch={}, rows_x={}, cols_x={}, cols_w={}",
                                        constants.batch_count, constants.rows_x, constants.columns_x, constants.columns_weights},
                           Log_Level::LOG_DEBUG, true, 1, Log_Feature::DENSE_COMPUTE);

        bool use_fp16 = (data_type == Data_Type::FLOAT16 ||
                         w_gpu.getDataType() == Data_Type::FLOAT16 ||
                         Execution_Engine::getInstance().isCooperativeMatrixEnabled());

        if (use_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
            output_gpu.reshape(out_shape);

            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_w = effective_w->getEffectiveFp16Storage();
            auto buf_b = effective_b->getEffectiveFp16Storage();

            Compute_Pipeline pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16
                : Compute_Pipeline::MATMUL_ADD_FP16;

            pushToGraph(pipeline,
                        {buf_self, buf_w, buf_b, output_gpu.storage}, constants,
                        (constants.columns_weights + 15) / 16, (constants.rows_x + 15) / 16, constants.batch_count);
        }
        else
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
            output_gpu.reshape(out_shape);

            pushToGraph(Compute_Pipeline::MATMUL_ADD,
                        {effective_self->storage, effective_w->storage, effective_b->storage, output_gpu.storage}, constants,
                        (constants.columns_weights + 15) / 16, (constants.rows_x + 15) / 16, constants.batch_count);
        }
    }

    void uploadData(const std::vector<float> &host_data) override
    {
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::uploadData: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }

        if (!isContiguous() || byte_offset != 0)
        {
            Gpu_Tensor_Impl temp_upload(shape, host_data, data_type);
            Execution_Engine::getInstance().getContext().copyBuffer(
                temp_upload.storage->getBuffer(),
                storage->getBuffer(),
                total_elements * getDataTypeSize(data_type),
                0,
                byte_offset);
            Execution_Engine::getInstance().getContext().executePendingTransfers();
            is_fp16_cache_dirty = true;
            return;
        }

        if (!storage || storage->getSize() != total_elements || storage->getDataType() != data_type)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
        }
        storage->uploadData(host_data);
        is_fp16_cache_dirty = true;
    }

    void zero() override
    {
        host_cache.clear();
        is_fp16_cache_dirty = true;
        if (total_elements == 0)
        {
            return;
        }

        if (!storage || storage->getSize() != total_elements || storage->getDataType() != data_type)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
        }

        if (storage->getBuffer() == VK_NULL_HANDLE)
        {
            return;
        }

        size_t byte_size = total_elements * getDataTypeSize(data_type);
        byte_size = ((byte_size + 3) / 4) * 4;
        Execution_Engine::getInstance().getContext().fillBuffer(storage->getBuffer(), byte_size, byte_offset, 0);
    }

    void fill(float value) override
    {
        if (value == 0.0f)
        {
            zero();
            return;
        }
        std::vector<float> buffer(total_elements, value);
        uploadData(buffer);
    }

    void conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output,
                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                uint32_t output_channels, uint32_t kernel_size,
                uint32_t stride, uint32_t padding,
                Tensor_Impl *scratch = nullptr) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &w_gpu = castToGpu(weights);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        const auto &b_gpu = castToGpu(biases);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        auto &output_gpu = castToGpu(output);

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
        uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 && effective_w->getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }

        uint32_t M = batch_size * out_h * out_w;
        uint32_t K = kernel_size * kernel_size * input_channels;
        uint32_t N = output_channels;

        struct Conv2d_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t input_channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t output_channels;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, input_channels, out_h, out_w, output_channels, kernel_size, stride, padding};

        if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled())
        {
            Gpu_Tensor_Impl *effective_scratch = nullptr;
            std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
            if (scratch != nullptr)
            {
                effective_scratch = &castToGpu(*scratch);
                effective_scratch->setDataType(Data_Type::FLOAT16);
                effective_scratch->reshape(M, K);
            }
            else
            {
                local_scratch = std::make_shared<Gpu_Tensor_Impl>(Shape{M, K}, Data_Type::FLOAT16);
                effective_scratch = local_scratch.get();
            }

            // 1. Unroll input patches directly into [M, K]
            pushToGraph(Compute_Pipeline::CONV2D_IM2COL_FP16,
                        {effective_self->storage, effective_scratch->storage},
                        constants,
                        (M + 15) / 16, (K + 15) / 16, 1);

            // 2. GEMM with Cooperative Tensor: [M, K] * [K, N] + [1, N] -> [M, N]
            output_gpu.reshape(M, N);
            struct Matmul_Add_Constants
            {
                uint32_t batch_count;
                uint32_t rows_x;
                uint32_t columns_x;
                uint32_t columns_weights;
                uint32_t broadcast_w;
                uint32_t broadcast_b;
            } matmul_constants{
                .batch_count = 1,
                .rows_x = M,
                .columns_x = K,
                .columns_weights = N,
                .broadcast_w = 0,
                .broadcast_b = 1};

            pushToGraph(Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16,
                        {effective_scratch->storage, effective_w->storage, effective_b->storage, output_gpu.storage},
                        matmul_constants,
                        (N + 15) / 16, (M + 15) / 16, 1);

            output_gpu.reshape(batch_size, out_h * out_w * output_channels);
            return;
        }

        output_gpu.reshape(batch_size, out_h * out_w * output_channels);

        Compute_Pipeline pipeline = is_fp16 ? Compute_Pipeline::CONV2D_FORWARD_PASS_FP16 : Compute_Pipeline::CONV2D_FORWARD_PASS;
        pushToGraph(pipeline, {effective_self->storage, effective_w->storage, effective_b->storage, output_gpu.storage}, constants,
                    (output_channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }

    void conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                             uint32_t kernel_size, uint32_t stride, uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &w_gpu = castToGpu(weights);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        uint32_t batch_size = static_cast<uint32_t>(getRows());

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 && effective_w->getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        in_grad_gpu.reshape(batch_size, input_height * input_width * input_channels);

        struct Conv2d_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t input_channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t output_channels;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding};

        Compute_Pipeline pipeline = is_fp16 ? Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT_FP16 : Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT;
        pushToGraph(pipeline, {effective_self->storage, effective_w->storage, in_grad_gpu.storage}, constants,
                    (input_channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }

    void conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient,
                              uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                              uint32_t kernel_size, uint32_t stride, uint32_t padding,
                              Tensor_Impl *im2col_scratch = nullptr) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &w_grad_gpu = castToGpu(weight_gradient);
        auto &b_grad_gpu = castToGpu(bias_gradient);

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        uint32_t K = kernel_size * kernel_size * input_channels;
        uint32_t M = batch_size * output_height * output_width;

        w_grad_gpu.reshape(1, K * output_channels);
        b_grad_gpu.reshape(1, output_channels);

        struct Conv2d_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t input_channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t output_channels;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding};

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 && effective_grad->getDataType() == Data_Type::FLOAT16);

        std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
        Gpu_Tensor_Impl *effective_scratch = nullptr;

        auto *gpu_scratch_candidate = dynamic_cast<Gpu_Tensor_Impl *>(im2col_scratch);
        if (gpu_scratch_candidate != nullptr)
        {
            effective_scratch = gpu_scratch_candidate;
            if (is_fp16)
            {
                effective_scratch->setDataType(Data_Type::FLOAT16);
            }
            else
            {
                effective_scratch->setDataType(Data_Type::FLOAT32);
            }
            effective_scratch->reshape(K, M);
        }
        else
        {
            local_scratch = std::make_shared<Gpu_Tensor_Impl>(Shape{K, M}, is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            effective_scratch = local_scratch.get();
        }

        // 1. im2col_transposed: unrolls input patches directly into [K, M]
        Compute_Pipeline im2col_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_IM2COL_TRANSPOSED_FP16 : Compute_Pipeline::CONV2D_IM2COL_TRANSPOSED;
        pushToGraph(im2col_pipeline, {effective_self->storage, effective_scratch->storage}, constants,
                    (M + 15) / 16, (K + 15) / 16, 1);

        // 2. Weight gradient: dW = col_transposed [K, M] * out_grad [M, output_channels] -> [K, output_channels]
        Matrix_Dimensions matmul_dims{
            .batch_count = 1,
            .rows_a = K,
            .columns_a = M,
            .columns_b = output_channels,
            .broadcast_b = 0};

        if (is_fp16)
        {
            Compute_Pipeline dw_pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::CONV2D_WEIGHT_GRADIENT_COOPMAT_FP16
                : Compute_Pipeline::CONV2D_WEIGHT_GRADIENT_FP16;
            pushToGraph(dw_pipeline, {effective_scratch->storage, effective_grad->storage, w_grad_gpu.storage}, matmul_dims,
                        (output_channels + 15) / 16, (K + 15) / 16, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::MATMUL, {effective_scratch->storage, effective_grad->storage, w_grad_gpu.storage}, matmul_dims,
                        (output_channels + 15) / 16, (K + 15) / 16, 1);
        }

        // 3. Bias gradient: dB = sum over M of out_grad [M, output_channels] -> [output_channels]
        Compute_Pipeline bias_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_BIAS_GRADIENT_FP16 : Compute_Pipeline::CONV2D_BIAS_GRADIENT;
        pushToGraph(bias_pipeline, {effective_grad->storage, b_grad_gpu.storage}, constants,
                    output_channels, 1, 1);
    }

    void maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask,
                   uint32_t input_height, uint32_t input_width, uint32_t channels,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &res_gpu = castToGpu(output);
        auto &mask_gpu = castToGpu(output_mask);

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
        uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

        if (data_type == Data_Type::FLOAT16)
        {
            res_gpu.setDataType(Data_Type::FLOAT16);
            mask_gpu.setDataType(Data_Type::FLOAT32);
        }
        res_gpu.reshape(batch_size, out_h * out_w * channels);
        mask_gpu.reshape(batch_size, out_h * out_w * channels);

        struct Pool_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, channels, out_h, out_w, kernel_size, stride, padding};

        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::MAXPOOL2D_FORWARD_FP16 : Compute_Pipeline::MAXPOOL2D_FORWARD,
                    {effective_self->storage, res_gpu.storage, mask_gpu.storage}, constants,
                    (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }

    void maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient,
                           uint32_t input_height, uint32_t input_width, uint32_t channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &mask_gpu = castToGpu(mask);
        auto contig_mask = mask_gpu.ensureContiguousSelf();
        const auto *effective_mask = contig_mask ? contig_mask.get() : &mask_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        uint32_t batch_size = static_cast<uint32_t>(getRows());
        if (data_type == Data_Type::FLOAT16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        in_grad_gpu.reshape(batch_size, input_height * input_width * channels);

        struct Pool_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding};

        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::MAXPOOL2D_BACKWARD_FP16 : Compute_Pipeline::MAXPOOL2D_BACKWARD,
                    {effective_mask->storage, effective_self->storage, in_grad_gpu.storage}, constants,
                    (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }

    void globalAvgPool2d(Tensor_Impl &output, uint32_t input_height, uint32_t input_width, uint32_t channels) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(data_type);
        output_gpu.reshape(batch_size, channels);

        struct Avg_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
        } constants{batch_size, input_height, input_width, channels};

        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD_FP16 : Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(pipe, {buf_self, output_gpu.storage}, constants,
                    (channels + 255) / 256, batch_size, 1);
    }

    void globalAvgPool2dBackward(Tensor_Impl &input_gradient, uint32_t input_height, uint32_t input_width, uint32_t channels) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        auto &in_grad_gpu = castToGpu(input_gradient);
        in_grad_gpu.setDataType(data_type);
        in_grad_gpu.reshape(batch_size, input_height * input_width * channels);

        struct Avg_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
        } constants{batch_size, input_height, input_width, channels};

        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD_FP16 : Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(pipe, {buf_self, in_grad_gpu.storage}, constants,
                    (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }

    void batchNormForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                          Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                          Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                          Tensor_Impl &normalized_input, Tensor_Impl &output,
                          float epsilon, float momentum, bool is_training) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &gamma_gpu = castToGpu(gamma);
        const auto &beta_gpu = castToGpu(beta);
        auto &rm_gpu = castToGpu(running_mean);
        auto &rv_gpu = castToGpu(running_variance);
        auto &norm_in_gpu = castToGpu(normalized_input);
        auto &out_gpu = castToGpu(output);

        uint32_t b_count = static_cast<uint32_t>(getRows());
        uint32_t f_dim = static_cast<uint32_t>(getColumns());
        if (data_type == Data_Type::FLOAT16)
        {
            out_gpu.setDataType(Data_Type::FLOAT16);
            norm_in_gpu.setDataType(Data_Type::FLOAT32);
        }
        out_gpu.reshape(b_count, f_dim);
        norm_in_gpu.reshape(b_count, f_dim);

        if (is_training)
        {
            auto &bm_gpu = castToGpu(batch_mean);
            auto &bv_gpu = castToGpu(batch_variance);
            bm_gpu.reshape(1, f_dim);
            bv_gpu.reshape(1, f_dim);

            struct Stats_Constants
            {
                uint32_t batch_size;
                uint32_t feature_dimension;
                float momentum;
            } stats{b_count, f_dim, momentum};

            pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM_STATS_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_STATS_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, rm_gpu.storage, rv_gpu.storage},
                        stats, f_dim, 1, 1);

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, f_dim, epsilon};

            pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_count * f_dim + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, f_dim, epsilon};

            pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                        {effective_self->storage, rm_gpu.storage, rv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_count * f_dim + 255) / 256, 1, 1);
        }
    }

    void batchNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                           Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient, float epsilon) const override
    {
        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = castToGpu(gamma);
        const auto &bv_gpu = castToGpu(batch_variance);
        const auto &norm_in_gpu = castToGpu(normalized_input);
        auto &g_grad_gpu = castToGpu(gamma_gradient);
        auto &b_grad_gpu = castToGpu(beta_gradient);
        auto &in_grad_gpu = castToGpu(input_gradient);

        uint32_t b_count = static_cast<uint32_t>(getRows());
        uint32_t f_dim = static_cast<uint32_t>(getColumns());
        bool is_fp16 = (effective_grad->getDataType() == Data_Type::FLOAT16 || data_type == Data_Type::FLOAT16);
        if (is_fp16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
            g_grad_gpu.setDataType(Data_Type::FLOAT32);
            b_grad_gpu.setDataType(Data_Type::FLOAT32);
        }
        g_grad_gpu.reshape(1, f_dim);
        b_grad_gpu.reshape(1, f_dim);
        in_grad_gpu.reshape(b_count, f_dim);

        struct Stats_Constants
        {
            uint32_t batch_size;
            uint32_t feature_dimension;
        } stats{b_count, f_dim};

        pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_STATS_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_STATS_BACKWARD,
                    {effective_grad->storage, norm_in_gpu.storage, g_grad_gpu.storage, b_grad_gpu.storage},
                    stats, f_dim, 1, 1);

        struct Transform_Constants
        {
            uint32_t total_elements;
            uint32_t batch_size;
            uint32_t feature_dimension;
            float epsilon;
        } tf{b_count * f_dim, b_count, f_dim, epsilon};

        pushToGraph(is_fp16 ? Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
                    {effective_grad->storage, norm_in_gpu.storage, gamma_gpu.storage, g_grad_gpu.storage, b_grad_gpu.storage, bv_gpu.storage, in_grad_gpu.storage},
                    tf, (b_count * f_dim + 255) / 256, 1, 1);
    }

    void linearForward(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override
    {
        matmulAdd(weights, biases, output);
    }

    void linearBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &w_gpu = castToGpu(weights);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        bool use_fp16 = (data_type == Data_Type::FLOAT16 || w_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        if (use_fp16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT32);
        }
        in_grad_gpu.reshape(getRows(), effective_w->getRows());

        struct Constants
        {
            uint32_t batch_size;
            uint32_t input_dimension;
            uint32_t output_dimension;
        } c{static_cast<uint32_t>(getRows()), static_cast<uint32_t>(effective_w->getRows()), static_cast<uint32_t>(getColumns())};

        if (use_fp16)
        {
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_w = effective_w->getEffectiveFp16Storage();
            Compute_Pipeline pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16
                : Compute_Pipeline::LINEAR_BACKWARD_INPUT_FP16;
            pushToGraph(pipeline,
                        {buf_self, buf_w, in_grad_gpu.storage}, c,
                        (c.input_dimension + 15) / 16, (c.batch_size + 15) / 16, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_INPUT,
                        {effective_self->storage, effective_w->storage, in_grad_gpu.storage}, c,
                        (c.input_dimension + 15) / 16, (c.batch_size + 15) / 16, 1);
        }
    }

    void linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient, bool accumulate = false) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &w_grad_gpu = castToGpu(weight_gradient);
        auto &b_grad_gpu = castToGpu(bias_gradient);
        w_grad_gpu.setDataType(Data_Type::FLOAT32);
        b_grad_gpu.setDataType(Data_Type::FLOAT32);
        w_grad_gpu.reshape(getColumns(), effective_grad->getColumns());
        b_grad_gpu.reshape(1, effective_grad->getColumns());

        struct Constants
        {
            uint32_t batch_size;
            uint32_t input_dimension;
            uint32_t output_dimension;
            uint32_t accumulate;
        } c{static_cast<uint32_t>(getRows()), static_cast<uint32_t>(getColumns()), static_cast<uint32_t>(effective_grad->getColumns()), accumulate ? 1u : 0u};

        bool use_fp16 = (data_type == Data_Type::FLOAT16 || out_grad_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        if (use_fp16)
        {
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_grad = effective_grad->getEffectiveFp16Storage();
            Compute_Pipeline pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16
                : Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_FP16;
            pushToGraph(pipeline,
                        {buf_self, buf_grad, w_grad_gpu.storage, b_grad_gpu.storage}, c,
                        (c.output_dimension + 15) / 16, (c.input_dimension + 15) / 16, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS,
                        {effective_self->storage, effective_grad->storage, w_grad_gpu.storage, b_grad_gpu.storage}, c,
                        (c.output_dimension + 15) / 16, (c.input_dimension + 15) / 16, 1);
        }
    }

    void linearBackwardWeightAdam(
        const Tensor_Impl &output_gradient,
        Tensor_Impl &weights,
        Tensor_Impl &first_moment,
        Tensor_Impl &second_moment,
        Tensor_Impl &bias_gradient,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        size_t timestep,
        float max_gradient = 1.0F,
        float inv_scale = 1.0F,
        float weight_decay = 0.0F) override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &w_gpu = castToGpu(weights);
        auto &m_gpu = castToGpu(first_moment);
        auto &v_gpu = castToGpu(second_moment);
        auto &b_grad_gpu = castToGpu(bias_gradient);
        b_grad_gpu.setDataType(Data_Type::FLOAT32);
        b_grad_gpu.reshape(1, effective_grad->getColumns());

        uint32_t batch_size = static_cast<uint32_t>(getRows());
        uint32_t in_dim = static_cast<uint32_t>(getColumns());
        uint32_t out_dim = static_cast<uint32_t>(effective_grad->getColumns());
        size_t total_elements = static_cast<size_t>(in_dim) * out_dim;

        size_t effective_t = std::max<size_t>(timestep, 1);
        float bc1 = std::max(1.0F - std::pow(beta1, static_cast<float>(effective_t)), 1e-8F);
        float bc2 = std::max(1.0F - std::pow(beta2, static_cast<float>(effective_t)), 1e-8F);

        Execution_Engine &engine = Execution_Engine::getInstance();
        uint32_t current_frame = engine.getContext().getCurrentFrame();
        engine.updateDynamicOptimizerParams(learning_rate, 1.0F / bc1, 1.0F / std::sqrt(bc2), inv_scale, current_frame);

        struct Fused_Adam_Constants
        {
            uint32_t batch_size;
            uint32_t in_dim;
            uint32_t out_dim;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float weight_decay;
        } c{batch_size, in_dim, out_dim, beta1, beta2, epsilon, max_gradient, weight_decay};

        auto dyn_buf = engine.getDynamicOptimizerBuffer(current_frame);

        if (!w_gpu.cached_fp16_storage || w_gpu.cached_fp16_storage->getSize() < total_elements)
        {
            w_gpu.cached_fp16_storage = std::make_shared<gpu::vector>(engine.getContext(), total_elements, Data_Type::FLOAT16);
        }

        auto buf_self = effective_self->getEffectiveFp16Storage();
        auto buf_grad = effective_grad->getEffectiveFp16Storage();

        Compute_Pipeline pipeline = engine.isCooperativeMatrixEnabled()
            ? Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16
            : Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_FP16;

        pushToGraph(pipeline,
                    {buf_self, buf_grad, w_gpu.storage, m_gpu.storage, v_gpu.storage, dyn_buf, w_gpu.cached_fp16_storage, b_grad_gpu.storage},
                    c,
                    (out_dim + 15) / 16, (in_dim + 15) / 16, 1);

        w_gpu.is_fp16_cache_dirty = false;
    }

    void batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                            Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                            Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                            Tensor_Impl &normalized_input, Tensor_Impl &output,
                            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                            float epsilon, float momentum, bool is_training) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &gamma_gpu = castToGpu(gamma);
        const auto &beta_gpu = castToGpu(beta);
        auto &rm_gpu = castToGpu(running_mean);
        auto &rv_gpu = castToGpu(running_variance);
        auto &norm_in_gpu = castToGpu(normalized_input);
        auto &out_gpu = castToGpu(output);

        uint32_t b_size = static_cast<uint32_t>(getRows());
        uint32_t tot_feat = input_height * input_width * input_channels;
        uint32_t sp_count = b_size * input_height * input_width;
        if (data_type == Data_Type::FLOAT16)
        {
            out_gpu.setDataType(Data_Type::FLOAT16);
            norm_in_gpu.setDataType(Data_Type::FLOAT32);
        }
        out_gpu.reshape(b_size, tot_feat);
        norm_in_gpu.reshape(b_size, tot_feat);

        if (is_training)
        {
            auto &bm_gpu = castToGpu(batch_mean);
            auto &bv_gpu = castToGpu(batch_variance);
            bm_gpu.reshape(1, input_channels);
            bv_gpu.reshape(1, input_channels);

            struct Stats_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                uint32_t spatial_count;
                float momentum;
            } stats{b_size * tot_feat, input_channels, sp_count, momentum};

            pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, rm_gpu.storage, rv_gpu.storage},
                        stats, 1, 1, 1);

            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, epsilon};

            pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Constants
            {
                uint32_t total_elements;
                uint32_t channels;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, epsilon};

            pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                        {effective_self->storage, rm_gpu.storage, rv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
    }

    void batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                             Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t input_channels, float epsilon) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &gamma_gpu = castToGpu(gamma);
        const auto &bv_gpu = castToGpu(batch_variance);
        const auto &norm_in_gpu = castToGpu(normalized_input);
        auto &g_grad_gpu = castToGpu(gamma_gradient);
        auto &b_grad_gpu = castToGpu(beta_gradient);
        auto &in_grad_gpu = castToGpu(input_gradient);

        uint32_t b_size = static_cast<uint32_t>(getRows());
        uint32_t tot_feat = input_height * input_width * input_channels;
        uint32_t sp_count = b_size * input_height * input_width;
        if (data_type == Data_Type::FLOAT16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
            g_grad_gpu.setDataType(Data_Type::FLOAT32);
            b_grad_gpu.setDataType(Data_Type::FLOAT32);
        }
        in_grad_gpu.reshape(b_size, tot_feat);
        g_grad_gpu.reshape(1, input_channels);
        b_grad_gpu.reshape(1, input_channels);

        struct Stats_Constants
        {
            uint32_t total_elements;
            uint32_t channels;
            uint32_t spatial_count;
        } stats{b_size * tot_feat, input_channels, sp_count};

        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD,
                    {effective_self->storage, norm_in_gpu.storage, g_grad_gpu.storage, b_grad_gpu.storage},
                    stats, 1, 1, 1);

        struct Transform_Constants
        {
            uint32_t total_elements;
            uint32_t channels;
            uint32_t spatial_count;
            float epsilon;
        } tf{b_size * tot_feat, input_channels, sp_count, epsilon};

        pushToGraph(data_type == Data_Type::FLOAT16 ? Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD_FP16 : Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
                    {effective_self->storage, norm_in_gpu.storage, gamma_gpu.storage, g_grad_gpu.storage, b_grad_gpu.storage, bv_gpu.storage, in_grad_gpu.storage},
                    tf, (b_size * tot_feat + 255) / 256, 1, 1);
    }

    void cceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override
    {
        validateSameDimensions(target);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &target_gpu = castToGpu(target);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto &output_gpu = castToGpu(output);

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float epsilon;
        } c{total, epsilon};

        pushToGraph(Compute_Pipeline::CCE_LOSS, {effective_self->storage, effective_target->storage, output_gpu.storage}, c, wg_x, 1, 1);
    }

    void mseLoss(const Tensor_Impl &target, Tensor_Impl &output) const override
    {
        validateSameDimensions(target);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &target_gpu = castToGpu(target);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto &output_gpu = castToGpu(output);

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
        } c{total};

        pushToGraph(Compute_Pipeline::MSE_LOSS, {effective_self->storage, effective_target->storage, output_gpu.storage}, c, wg_x, 1, 1);
    }

    void maeLoss(const Tensor_Impl &target, Tensor_Impl &output) const override
    {
        validateSameDimensions(target);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &target_gpu = castToGpu(target);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto &output_gpu = castToGpu(output);

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
        } c{total};

        pushToGraph(Compute_Pipeline::MAE_LOSS, {effective_self->storage, effective_target->storage, output_gpu.storage}, c, wg_x, 1, 1);
    }

    void bceLoss(const Tensor_Impl &target, Tensor_Impl &output, float epsilon) const override
    {
        validateSameDimensions(target);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &target_gpu = castToGpu(target);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto &output_gpu = castToGpu(output);

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float epsilon;
        } c{total, epsilon};

        pushToGraph(Compute_Pipeline::BCE_LOSS, {effective_self->storage, effective_target->storage, output_gpu.storage}, c, wg_x, 1, 1);
    }

    void huberLoss(const Tensor_Impl &target, Tensor_Impl &output, float delta) const override
    {
        validateSameDimensions(target);
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &target_gpu = castToGpu(target);
        auto contig_target = target_gpu.ensureContiguousSelf();
        const auto *effective_target = contig_target ? contig_target.get() : &target_gpu;

        auto &output_gpu = castToGpu(output);

        uint32_t total = static_cast<uint32_t>(total_elements);
        uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            uint32_t total_elements;
            float delta;
        } c{total, delta};

        pushToGraph(Compute_Pipeline::HUBER_LOSS, {effective_self->storage, effective_target->storage, output_gpu.storage}, c, wg_x, 1, 1);
    }

    void concatenateColumns(const Tensor_Impl& other, Tensor_Impl& output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto* effective_self = contig_self ? contig_self.get() : this;

        const auto& other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto* effective_other = contig_other ? contig_other.get() : &other_gpu;

        if (effective_self->getRows() != effective_other->getRows())
        {
            throw std::invalid_argument("concatenateColumns: Row count mismatch");
        }

        if (effective_self->getDataType() != effective_other->getDataType())
        {
            throw std::invalid_argument("concatenateColumns: Data type mismatch between input tensors");
        }

        auto& output_gpu = castToGpu(output);
        if (output_gpu.getDataType() != effective_self->getDataType())
        {
            output_gpu.setDataType(effective_self->getDataType());
        }

        uint32_t cols_a = static_cast<uint32_t>(effective_self->getColumns());
        uint32_t cols_b = static_cast<uint32_t>(effective_other->getColumns());
        uint32_t tot_cols = cols_a + cols_b;
        output_gpu.reshape(effective_self->getRows(), tot_cols);

        struct Constants
        {
            uint32_t rows;
            uint32_t columns_a;
            uint32_t columns_b;
        } c{ static_cast<uint32_t>(effective_self->getRows()), cols_a, cols_b };

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline target_pipeline = is_fp16 ? Compute_Pipeline::CONCATENATE_COLUMNS_FP16 : Compute_Pipeline::CONCATENATE_COLUMNS;

        pushToGraph(target_pipeline, { effective_self->storage, effective_other->storage, output_gpu.storage }, c,
            (tot_cols + 15) / 16, (c.rows + 15) / 16);
    }

    void concatenateRows(const Tensor_Impl& other, Tensor_Impl& output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto* effective_self = contig_self ? contig_self.get() : this;

        const auto& other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto* effective_other = contig_other ? contig_other.get() : &other_gpu;

        if (effective_self->getColumns() != effective_other->getColumns())
        {
            throw std::invalid_argument("concatenateRows: Column count mismatch");
        }

        if (effective_self->getDataType() != effective_other->getDataType())
        {
            throw std::invalid_argument("concatenateRows: Data type mismatch between input tensors");
        }

        auto& output_gpu = castToGpu(output);
        if (output_gpu.getDataType() != effective_self->getDataType())
        {
            output_gpu.setDataType(effective_self->getDataType());
        }

        uint32_t rows_a = static_cast<uint32_t>(effective_self->getRows());
        uint32_t rows_b = static_cast<uint32_t>(effective_other->getRows());
        uint32_t tot_rows = rows_a + rows_b;
        uint32_t cols = static_cast<uint32_t>(effective_self->getColumns());

        output_gpu.reshape(tot_rows, cols);

        struct Constants
        {
            uint32_t rows_a;
            uint32_t rows_b;
            uint32_t columns;
        } c{ rows_a, rows_b, cols };

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline target_pipeline = is_fp16
            ? Compute_Pipeline::CONCATENATE_ROWS_FP16
            : Compute_Pipeline::CONCATENATE_ROWS;

            pushToGraph(target_pipeline, { effective_self->storage, effective_other->storage, output_gpu.storage }, c,
                (c.columns + 15) / 16, (tot_rows + 15) / 16);
    }

    void splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &left_gpu = castToGpu(result_left);
        auto &right_gpu = castToGpu(result_right);
        left_gpu.setDataType(data_type);
        right_gpu.setDataType(data_type);
        uint32_t cols_left = static_cast<uint32_t>(split_index);
        uint32_t cols_right = static_cast<uint32_t>(getColumns() - split_index);

        left_gpu.reshape(getRows(), cols_left);
        right_gpu.reshape(getRows(), cols_right);

        struct Constants
        {
            uint32_t rows;
            uint32_t columns_left;
            uint32_t columns_right;
        } c{static_cast<uint32_t>(getRows()), cols_left, cols_right};

        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SPLIT_COLUMNS_FP16 : Compute_Pipeline::SPLIT_COLUMNS;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(pipe, {buf_self, left_gpu.storage, right_gpu.storage}, c,
                    (static_cast<uint32_t>(getColumns()) + 15) / 16, (c.rows + 15) / 16);
    }

    void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &up_gpu = castToGpu(result_up);
        auto &down_gpu = castToGpu(result_down);
        up_gpu.setDataType(data_type);
        down_gpu.setDataType(data_type);
        uint32_t rows_up = static_cast<uint32_t>(split_index);
        uint32_t rows_down = static_cast<uint32_t>(getRows() - split_index);

        up_gpu.reshape(rows_up, getColumns());
        down_gpu.reshape(rows_down, getColumns());

        struct Constants
        {
            uint32_t rows_up;
            uint32_t rows_down;
            uint32_t columns;
        } c{rows_up, rows_down, static_cast<uint32_t>(getColumns())};

        bool is_fp16 = (data_type == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SPLIT_ROWS_FP16 : Compute_Pipeline::SPLIT_ROWS;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(pipe, {buf_self, up_gpu.storage, down_gpu.storage}, c,
                    (c.columns + 15) / 16, (static_cast<uint32_t>(getRows()) + 15) / 16);
    }

    void rmsNormForward(const Tensor_Impl &gamma, Tensor_Impl &inv_rms, Tensor_Impl &output, float epsilon) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &gamma_gpu = castToGpu(gamma);
        auto &inv_rms_gpu = castToGpu(inv_rms);
        auto &out_gpu = castToGpu(output);

        uint32_t b_count = static_cast<uint32_t>(getRows());
        uint32_t f_dim = static_cast<uint32_t>(getColumns());

        bool is_fp16 = (data_type == Data_Type::FLOAT16 || gamma_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        if (is_fp16)
        {
            out_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            out_gpu.setDataType(Data_Type::FLOAT32);
        }
        inv_rms_gpu.setDataType(Data_Type::FLOAT32);

        out_gpu.reshape(b_count, f_dim);
        inv_rms_gpu.reshape(b_count, 1);

        struct RmsNorm_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            float epsilon;
        } pcs{b_count, f_dim, epsilon};

        if (is_fp16)
        {
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_gamma = gamma_gpu.getEffectiveFp16Storage();
            pushToGraph(Compute_Pipeline::RMSNORM_FP16,
                        {buf_self, buf_gamma, out_gpu.storage, inv_rms_gpu.storage},
                        pcs, b_count, 1, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::RMSNORM,
                        {effective_self->storage, gamma_gpu.storage, out_gpu.storage, inv_rms_gpu.storage},
                        pcs, b_count, 1, 1);
        }
    }

    void rmsNormBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &gamma, const Tensor_Impl &inv_rms,
                         Tensor_Impl &gamma_gradient, Tensor_Impl &input_gradient, bool accumulate_gamma = false) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &gamma_gpu = castToGpu(gamma);
        const auto &inv_rms_gpu = castToGpu(inv_rms);
        auto &g_grad_gpu = castToGpu(gamma_gradient);
        auto &in_grad_gpu = castToGpu(input_gradient);

        uint32_t b_count = static_cast<uint32_t>(getRows());
        uint32_t f_dim = static_cast<uint32_t>(getColumns());

        bool is_fp16 = (effective_self->data_type == Data_Type::FLOAT16 || effective_grad->getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        if (is_fp16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT32);
        }
        g_grad_gpu.setDataType(Data_Type::FLOAT32);

        in_grad_gpu.reshape(b_count, f_dim);
        g_grad_gpu.reshape(1, f_dim);

        struct RMSNorm_Backward_Constants
        {
            uint32_t batch_size;
            uint32_t dim;
            uint32_t accumulate;
        } pcs{b_count, f_dim, accumulate_gamma ? 1u : 0u};

        if (is_fp16)
        {
            auto buf_grad = effective_grad->getEffectiveFp16Storage();
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_gamma = gamma_gpu.getEffectiveFp16Storage();

            pushToGraph(Compute_Pipeline::RMSNORM_BACKWARD_FP16,
                        {buf_grad, buf_self, buf_gamma, inv_rms_gpu.storage, in_grad_gpu.storage, g_grad_gpu.storage},
                        pcs, b_count, 1, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::RMSNORM_BACKWARD,
                        {effective_grad->storage, effective_self->storage, gamma_gpu.storage, inv_rms_gpu.storage, in_grad_gpu.storage, g_grad_gpu.storage},
                        pcs, b_count, 1, 1);
        }
    }

    void applyRoPE(Tensor_Impl &output, uint32_t seq_len, uint32_t head_dim, int direction = 1, float base = 10000.0f, uint32_t num_heads = 1, uint32_t mode = 0) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;
        auto &out_gpu = castToGpu(output);

        bool is_fp16 = (effective_self->data_type == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        out_gpu.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);

        if (mode == 1)
        {
            out_gpu.reshape(Shape{effective_self->shape[0], num_heads, seq_len, head_dim});
        }
        else if (mode == 2)
        {
            out_gpu.reshape(Shape{effective_self->shape[0], seq_len, num_heads, head_dim});
        }
        else
        {
            out_gpu.reshape(effective_self->shape);
        }

        uint32_t total = static_cast<uint32_t>(effective_self->total_elements);
        uint32_t total_pairs = total / 2;

        struct RoPE_Constants
        {
            uint32_t total_pairs;
            uint32_t seq_len;
            uint32_t head_dim;
            int32_t direction;
            float base;
            uint32_t num_heads;
            uint32_t mode;
        } pcs{total_pairs, seq_len, head_dim, direction, base, num_heads, mode};

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(is_fp16 ? Compute_Pipeline::ROPE_FP16 : Compute_Pipeline::ROPE,
                    {buf_self, out_gpu.storage},
                    pcs, (total_pairs + 255) / 256, 1, 1);
    }

    void swigluForward(const Tensor_Impl &b, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;
        const auto &b_gpu = castToGpu(b);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;
        auto &out_gpu = castToGpu(output);

        bool is_fp16 = (effective_self->data_type == Data_Type::FLOAT16 || effective_b->getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        out_gpu.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
        out_gpu.reshape(shape);

        uint32_t total = static_cast<uint32_t>(total_elements);
        struct Swiglu_Constants
        {
            uint32_t total_elements;
        } pcs{total};

        if (is_fp16)
        {
            auto buf_a = effective_self->getEffectiveFp16Storage();
            auto buf_b = effective_b->getEffectiveFp16Storage();
            pushToGraph(Compute_Pipeline::SWIGLU_FP16,
                        {buf_a, buf_b, out_gpu.storage},
                        pcs, (total + 255) / 256, 1, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::SWIGLU,
                        {effective_self->storage, effective_b->storage, out_gpu.storage},
                        pcs, (total + 255) / 256, 1, 1);
        }
    }

    void swigluBackward(const Tensor_Impl &output_gradient, const Tensor_Impl &b, Tensor_Impl &grad_a, Tensor_Impl &grad_b) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;
        const auto &dy_gpu = castToGpu(output_gradient);
        auto contig_dy = dy_gpu.ensureContiguousSelf();
        const auto *effective_dy = contig_dy ? contig_dy.get() : &dy_gpu;
        const auto &b_gpu = castToGpu(b);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        auto &da_gpu = castToGpu(grad_a);
        auto &db_gpu = castToGpu(grad_b);

        bool is_fp16 = (effective_dy->getDataType() == Data_Type::FLOAT16 || effective_self->data_type == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        da_gpu.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
        db_gpu.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
        da_gpu.reshape(shape);
        db_gpu.reshape(shape);

        uint32_t total = static_cast<uint32_t>(total_elements);
        struct Swiglu_Constants
        {
            uint32_t total_elements;
        } pcs{total};

        if (is_fp16)
        {
            auto buf_dy = effective_dy->getEffectiveFp16Storage();
            auto buf_a = effective_self->getEffectiveFp16Storage();
            auto buf_b = effective_b->getEffectiveFp16Storage();
            pushToGraph(Compute_Pipeline::SWIGLU_BACKWARD_FP16,
                        {buf_dy, buf_a, buf_b, da_gpu.storage, db_gpu.storage},
                        pcs, (total + 255) / 256, 1, 1);
        }
        else
        {
            pushToGraph(Compute_Pipeline::SWIGLU_BACKWARD,
                        {effective_dy->storage, effective_self->storage, effective_b->storage, da_gpu.storage, db_gpu.storage},
                        pcs, (total + 255) / 256, 1, 1);
        }
    }

    void fusedSwiGLUForward(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;
        auto &out_gpu = castToGpu(output);

        bool is_fp16 = (effective_self->data_type == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        out_gpu.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);

        size_t rows = effective_self->shape[0];
        size_t total_cols = effective_self->shape[1];
        size_t half_dim = total_cols / 2;
        out_gpu.reshape(Shape{rows, half_dim});

        uint32_t total = static_cast<uint32_t>(rows * half_dim);
        struct Fused_Swiglu_Constants
        {
            uint32_t total_elements;
            uint32_t half_dim;
        } pcs{total, static_cast<uint32_t>(half_dim)};

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(is_fp16 ? Compute_Pipeline::FUSED_SWIGLU_FORWARD_FP16 : Compute_Pipeline::FUSED_SWIGLU_FORWARD,
                    {buf_self, out_gpu.storage},
                    pcs, (total + 255) / 256, 1, 1);
        out_gpu.host_cache.clear();
        out_gpu.is_fp16_cache_dirty = true;
    }

    void fusedSwiGLUBackward(const Tensor_Impl &output_gradient, Tensor_Impl &input_gradient) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;
        const auto &dy_gpu = castToGpu(output_gradient);
        auto contig_dy = dy_gpu.ensureContiguousSelf();
        const auto *effective_dy = contig_dy ? contig_dy.get() : &dy_gpu;
        auto &dx_gpu = castToGpu(input_gradient);

        bool is_fp16 = (effective_self->data_type == Data_Type::FLOAT16 || effective_dy->getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        dx_gpu.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);

        size_t rows = effective_self->shape[0];
        size_t total_cols = effective_self->shape[1];
        size_t half_dim = total_cols / 2;
        dx_gpu.reshape(Shape{rows, total_cols});

        uint32_t total = static_cast<uint32_t>(rows * half_dim);
        struct Fused_Swiglu_Constants
        {
            uint32_t total_elements;
            uint32_t half_dim;
        } pcs{total, static_cast<uint32_t>(half_dim)};

        auto buf_dy = is_fp16 ? effective_dy->getEffectiveFp16Storage() : effective_dy->storage;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage;

        pushToGraph(is_fp16 ? Compute_Pipeline::FUSED_SWIGLU_BACKWARD_FP16 : Compute_Pipeline::FUSED_SWIGLU_BACKWARD,
                    {buf_dy, buf_self, dx_gpu.storage},
                    pcs, (total + 255) / 256, 1, 1);
        dx_gpu.host_cache.clear();
        dx_gpu.is_fp16_cache_dirty = true;
    }

    void flashAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output,
                               uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                               bool is_causal = false, float scale = 0.0f,
                               Tensor_Impl *l_stats = nullptr) const override
    {
        auto contig_q = ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : this;
        const auto &k_gpu = castToGpu(k);
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;
        const auto &v_gpu = castToGpu(v);
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;

        bool needs_cast_input = (effective_q->getDataType() != Data_Type::FLOAT16);

        std::shared_ptr<Gpu_Tensor_Impl> q_fp16, k_fp16, v_fp16;
        const Gpu_Tensor_Impl *in_q = effective_q;
        const Gpu_Tensor_Impl *in_k = effective_k;
        const Gpu_Tensor_Impl *in_v = effective_v;

        if (needs_cast_input)
        {
            q_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT16);
            k_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT16);
            v_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT16);
            effective_q->to(Data_Type::FLOAT16, *q_fp16);
            effective_k->to(Data_Type::FLOAT16, *k_fp16);
            effective_v->to(Data_Type::FLOAT16, *v_fp16);
            in_q = q_fp16.get();
            in_k = k_fp16.get();
            in_v = v_fp16.get();
        }

        auto &out_gpu = castToGpu(output);

        std::shared_ptr<Gpu_Tensor_Impl> intermediate_out_fp16;
        Gpu_Tensor_Impl *target_out = &out_gpu;

        if (needs_cast_input)
        {
            intermediate_out_fp16 = std::make_shared<Gpu_Tensor_Impl>(shape, Data_Type::FLOAT16);
            target_out = intermediate_out_fp16.get();
        }
        else
        {
            out_gpu.setDataType(Data_Type::FLOAT16);
            out_gpu.reshape(shape);
        }

        if (scale <= 0.0f)
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        uint32_t total_elements_per_head = seq_len * head_dim;
        uint32_t total_heads = (total_elements_per_head > 0) ? static_cast<uint32_t>(total_elements / total_elements_per_head) : num_heads;

        std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
        Gpu_Tensor_Impl *target_l = nullptr;
        if (l_stats)
        {
            auto *l_gpu = dynamic_cast<Gpu_Tensor_Impl *>(l_stats);
            if (!l_gpu)
            {
                fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
                target_l = fallback_l.get();
            }
            else
            {
                l_gpu->setDataType(Data_Type::FLOAT32);
                l_gpu->reshape(Shape{ total_heads, seq_len });
                size_t elem_size = sizeof(float);
                if (!l_gpu->storage || l_gpu->storage->getSizeBytes() < total_heads * seq_len * elem_size)
                {
                    l_gpu->storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_heads * seq_len, Data_Type::FLOAT32);
                }
                target_l = l_gpu;
            }
        }
        else
        {
            fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{ total_heads, seq_len }, Data_Type::FLOAT32);
            target_l = fallback_l.get();
        }

        struct FlashAttention_Constants
        {
            uint32_t batch_size;  // B * H
            uint32_t seq_len;     // N
            uint32_t head_dim;    // D
            uint32_t is_causal;   // 1 or 0
            float scale;
        } pcs{total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale};

        uint32_t num_q_tiles = (seq_len + 15) / 16;
        pushToGraph(Compute_Pipeline::FLASH_ATTENTION_FP16,
                    {in_q->storage, in_k->storage, in_v->storage, target_out->storage, target_l->storage},
                    pcs, 1, num_q_tiles, total_heads);

        if (needs_cast_input)
        {
            out_gpu.setDataType(Data_Type::FLOAT32);
            out_gpu.reshape(shape);
            intermediate_out_fp16->to(Data_Type::FLOAT32, out_gpu);
        }
    }

    void flashAttentionBackward(const Tensor_Impl &k, const Tensor_Impl &v,
                                const Tensor_Impl &o, const Tensor_Impl &do_grad,
                                Tensor_Impl &dq, Tensor_Impl &dk, Tensor_Impl &dv,
                                uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                                bool is_causal = false, float scale = 0.0f,
                                const Tensor_Impl *l_stats = nullptr) const override
    {
        auto contig_q = ensureContiguousSelf();
        const auto *effective_q = contig_q ? contig_q.get() : this;
        const auto &k_gpu = castToGpu(k);
        auto contig_k = k_gpu.ensureContiguousSelf();
        const auto *effective_k = contig_k ? contig_k.get() : &k_gpu;
        const auto &v_gpu = castToGpu(v);
        auto contig_v = v_gpu.ensureContiguousSelf();
        const auto *effective_v = contig_v ? contig_v.get() : &v_gpu;
        const auto &o_gpu = castToGpu(o);
        auto contig_o = o_gpu.ensureContiguousSelf();
        const auto *effective_o = contig_o ? contig_o.get() : &o_gpu;
        const auto &do_gpu = castToGpu(do_grad);
        auto contig_do = do_gpu.ensureContiguousSelf();
        const auto *effective_do = contig_do ? contig_do.get() : &do_gpu;

        bool needs_cast_input = (effective_q->getDataType() != Data_Type::FLOAT16);

        std::shared_ptr<Gpu_Tensor_Impl> q_fp16, k_fp16, v_fp16, o_fp16, do_fp16;
        const Gpu_Tensor_Impl *in_q = effective_q;
        const Gpu_Tensor_Impl *in_k = effective_k;
        const Gpu_Tensor_Impl *in_v = effective_v;
        const Gpu_Tensor_Impl *in_o = effective_o;
        const Gpu_Tensor_Impl *in_do = effective_do;

        if (effective_q->getDataType() != Data_Type::FLOAT16)
        {
            q_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_q->getShape(), Data_Type::FLOAT16);
            effective_q->to(Data_Type::FLOAT16, *q_fp16);
            in_q = q_fp16.get();
        }
        if (effective_k->getDataType() != Data_Type::FLOAT16)
        {
            k_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_k->getShape(), Data_Type::FLOAT16);
            effective_k->to(Data_Type::FLOAT16, *k_fp16);
            in_k = k_fp16.get();
        }
        if (effective_v->getDataType() != Data_Type::FLOAT16)
        {
            v_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_v->getShape(), Data_Type::FLOAT16);
            effective_v->to(Data_Type::FLOAT16, *v_fp16);
            in_v = v_fp16.get();
        }
        if (effective_o->getDataType() != Data_Type::FLOAT16)
        {
            o_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_o->getShape(), Data_Type::FLOAT16);
            effective_o->to(Data_Type::FLOAT16, *o_fp16);
            in_o = o_fp16.get();
        }
        if (effective_do->getDataType() != Data_Type::FLOAT16)
        {
            do_fp16 = std::make_shared<Gpu_Tensor_Impl>(effective_do->getShape(), Data_Type::FLOAT16);
            effective_do->to(Data_Type::FLOAT16, *do_fp16);
            in_do = do_fp16.get();
        }

        auto &dq_gpu = castToGpu(dq);
        auto &dk_gpu = castToGpu(dk);
        auto &dv_gpu = castToGpu(dv);

        std::shared_ptr<Gpu_Tensor_Impl> dq_fp16, dk_fp16, dv_fp16;
        Gpu_Tensor_Impl *target_dq = &dq_gpu;
        Gpu_Tensor_Impl *target_dk = &dk_gpu;
        Gpu_Tensor_Impl *target_dv = &dv_gpu;

        if (needs_cast_input)
        {
            dq_fp16 = std::make_shared<Gpu_Tensor_Impl>(shape, Data_Type::FLOAT16);
            dk_fp16 = std::make_shared<Gpu_Tensor_Impl>(k.getShape(), Data_Type::FLOAT16);
            dv_fp16 = std::make_shared<Gpu_Tensor_Impl>(v.getShape(), Data_Type::FLOAT16);
            target_dq = dq_fp16.get();
            target_dk = dk_fp16.get();
            target_dv = dv_fp16.get();
        }
        else
        {
            dq_gpu.setDataType(Data_Type::FLOAT16);
            dq_gpu.reshape(shape);
            dk_gpu.setDataType(Data_Type::FLOAT16);
            dk_gpu.reshape(k.getShape());
            dv_gpu.setDataType(Data_Type::FLOAT16);
            dv_gpu.reshape(v.getShape());
        }

        if (scale <= 0.0f)
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        uint32_t total_elements_per_head = seq_len * head_dim;
        uint32_t total_heads = (total_elements_per_head > 0) ? static_cast<uint32_t>(total_elements / total_elements_per_head) : num_heads;

        std::shared_ptr<Gpu_Tensor_Impl> fallback_l;
        const Gpu_Tensor_Impl *target_l = nullptr;
        if (l_stats)
        {
            target_l = dynamic_cast<const Gpu_Tensor_Impl *>(l_stats);
            if (!target_l)
            {
                fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, Data_Type::FLOAT32);
                target_l = fallback_l.get();
            }
        }
        else
        {
            fallback_l = std::make_shared<Gpu_Tensor_Impl>(Shape{total_heads, seq_len}, Data_Type::FLOAT32);
            target_l = fallback_l.get();
        }

        struct FlashAttention_Backward_Constants
        {
            uint32_t batch_size;  // B * H
            uint32_t seq_len;     // S
            uint32_t head_dim;    // D
            uint32_t is_causal;   // 1 or 0
            float scale;
            uint32_t pass_type;   // 0 = dQ pass, 1 = dK & dV pass
        };

        uint32_t num_q_tiles = (seq_len + 15) / 16;
        uint32_t num_k_tiles = (seq_len + 15) / 16;

        FlashAttention_Backward_Constants pcs_pass0{
            total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 0u
        };
        pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_FP16,
                    {in_q->storage, in_k->storage, in_v->storage,
                     in_o->storage, in_do->storage,
                     target_dq->storage, target_dk->storage, target_dv->storage, target_l->storage},
                    pcs_pass0, 1, num_q_tiles, total_heads);

        FlashAttention_Backward_Constants pcs_pass1{
            total_heads, seq_len, head_dim, is_causal ? 1u : 0u, scale, 1u
        };
        pushToGraph(Compute_Pipeline::FLASH_ATTENTION_BACKWARD_FP16,
                    {in_q->storage, in_k->storage, in_v->storage,
                     in_o->storage, in_do->storage,
                     target_dq->storage, target_dk->storage, target_dv->storage, target_l->storage},
                    pcs_pass1, 1, num_k_tiles, total_heads);

        if (needs_cast_input)
        {
            dq_gpu.setDataType(Data_Type::FLOAT32);
            dq_gpu.reshape(shape);
            dq_fp16->to(Data_Type::FLOAT32, dq_gpu);

            dk_gpu.setDataType(Data_Type::FLOAT32);
            dk_gpu.reshape(k.getShape());
            dk_fp16->to(Data_Type::FLOAT32, dk_gpu);

            dv_gpu.setDataType(Data_Type::FLOAT32);
            dv_gpu.reshape(v.getShape());
            dv_fp16->to(Data_Type::FLOAT32, dv_gpu);
        }
    }

    void embeddingForward(const Tensor_Impl &indices, Tensor_Impl &output) const override
    {
        size_t S = indices.getTotalElements();
        size_t D = getColumns();
        size_t V = shape[0];

        bool is_fp16 = (output.getDataType() == Data_Type::FLOAT16 || data_type == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        output.setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
        if (indices.getShape().getRank() == 2)
        {
            output.reshape(Shape{ indices.getShape()[0], indices.getShape()[1], D });
        }
        else
        {
            output.reshape(Shape{ S, D });
        }

        auto &out_gpu = castToGpu(output);
        auto &indices_gpu = castToGpu(indices);
        size_t elem_size = getDataTypeSize(out_gpu.getDataType());

        if (!out_gpu.storage || out_gpu.storage->getSizeBytes() < S * D * elem_size)
        {
            out_gpu.storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            out_gpu.storage->allocateMemory(S * D, out_gpu.getDataType());
        }

        struct Embedding_Constants
        {
            uint32_t num_tokens;
            uint32_t embedding_dim;
            uint32_t vocab_size;
        } pcs{ static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V) };

        auto table_storage = is_fp16 ? getEffectiveFp16Storage() : storage;
        pushToGraph(is_fp16 ? Compute_Pipeline::EMBEDDING_FORWARD_FP16 : Compute_Pipeline::EMBEDDING_FORWARD,
                    { indices_gpu.storage, table_storage, out_gpu.storage },
                    pcs,
                    static_cast<uint32_t>((S * D + 255) / 256), 1, 1);
        out_gpu.host_cache.clear();
    }

    void embeddingBackward(const Tensor_Impl &indices, const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient) const override
    {
        size_t S = indices.getTotalElements();
        size_t D = getColumns();
        size_t V = shape[0];

        auto &indices_gpu = castToGpu(indices);
        auto &grad_out_gpu = castToGpu(output_gradient);
        auto &w_grad_gpu = castToGpu(weight_gradient);
        w_grad_gpu.setDataType(Data_Type::FLOAT32);

        if (!w_grad_gpu.storage || w_grad_gpu.storage->getSizeBytes() < V * D * sizeof(float))
        {
            w_grad_gpu.storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            w_grad_gpu.storage->allocateMemory(V * D, Data_Type::FLOAT32);
        }

        struct Embedding_Constants
        {
            uint32_t num_tokens;
            uint32_t embedding_dim;
            uint32_t vocab_size;
        } pcs{ static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V) };

        bool is_fp16 = (grad_out_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        auto grad_storage = is_fp16 ? grad_out_gpu.getEffectiveFp16Storage() : grad_out_gpu.storage;

        Compute_Pipeline pipe = is_fp16
            ? Compute_Pipeline::EMBEDDING_BACKWARD_FP16
            : Compute_Pipeline::EMBEDDING_BACKWARD;

        pushToGraph(pipe,
                    { indices_gpu.storage, grad_storage, w_grad_gpu.storage },
                    pcs,
                    static_cast<uint32_t>((S * D + 255) / 256), 1, 1);
        w_grad_gpu.host_cache.clear();
    }

    void singleTokenAttentionForward(const Tensor_Impl &k, const Tensor_Impl &v, Tensor_Impl &output,
                                     size_t num_heads, size_t head_dim, size_t total_seq_len) const override
    {
        output.setDataType(data_type);
        output.reshape(Shape{ 1, num_heads, 1, head_dim });

        auto &out_gpu = castToGpu(output);
        auto &k_gpu = castToGpu(k);
        auto &v_gpu = castToGpu(v);
        size_t elem_size = getDataTypeSize(data_type);

        if (!out_gpu.storage || out_gpu.storage->getSizeBytes() < num_heads * head_dim * elem_size)
        {
            out_gpu.storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            out_gpu.storage->allocateMemory(num_heads * head_dim, data_type);
        }

        struct Attention_Decode_Constants
        {
            uint32_t num_heads;
            uint32_t head_dim;
            uint32_t total_seq_len;
            float scale;
        } pcs{ static_cast<uint32_t>(num_heads), static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(total_seq_len), 1.0f / std::sqrt(static_cast<float>(head_dim)) };

        bool is_fp16 = (data_type == Data_Type::FLOAT16 || k_gpu.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ATTENTION_DECODE_FP16 : Compute_Pipeline::ATTENTION_DECODE;
        auto buf_q = is_fp16 ? getEffectiveFp16Storage() : storage;
        auto buf_k = is_fp16 ? k_gpu.getEffectiveFp16Storage() : k_gpu.storage;
        auto buf_v = is_fp16 ? v_gpu.getEffectiveFp16Storage() : v_gpu.storage;

        pushToGraph(pipe,
                    { buf_q, buf_k, buf_v, out_gpu.storage },
                    pcs,
                    static_cast<uint32_t>(num_heads), 1, 1);
        out_gpu.host_cache.clear();
    }

    float fusedCrossEntropyLoss(const Tensor_Impl &targets, Tensor_Impl &d_logits, uint32_t valid_tokens = 0) const override
    {
        size_t total_tokens = (shape.getRank() == 3) ? (shape[0] * shape[1]) : getRows();
        size_t V = (shape.getRank() == 3) ? shape[2] : getColumns();

        d_logits.setDataType(data_type);
        d_logits.reshape(shape);

        if (total_tokens == 0 || V == 0)
        {
            return 0.0f;
        }

        auto &d_logits_gpu = castToGpu(d_logits);
        size_t elem_size = getDataTypeSize(data_type);
        if (!d_logits_gpu.storage || d_logits_gpu.storage->getSizeBytes() < total_tokens * V * elem_size)
        {
            d_logits_gpu.storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            d_logits_gpu.storage->allocateMemory(total_tokens * V, data_type);
        }

        std::shared_ptr<gpu::vector> targets_buf;
        const auto *targets_gpu_ptr = dynamic_cast<const Gpu_Tensor_Impl *>(&targets);
        if (targets_gpu_ptr && targets_gpu_ptr->storage && targets_gpu_ptr->storage->getSizeBytes() >= total_tokens * sizeof(float))
        {
            targets_buf = targets_gpu_ptr->storage;
            if (valid_tokens == 0)
            {
                if (targets_gpu_ptr->host_cache.size() >= total_tokens)
                {
                    for (size_t s = 0; s < total_tokens; ++s)
                    {
                        int32_t target = static_cast<int32_t>(std::round(targets_gpu_ptr->host_cache[s]));
                        if (target >= 0 && static_cast<size_t>(target) < V)
                        {
                            ++valid_tokens;
                        }
                    }
                }
                else
                {
                    valid_tokens = static_cast<uint32_t>(total_tokens);
                }
            }
        }
        else
        {
            const auto &targets_data = targets.getData();
            if (valid_tokens == 0)
            {
                for (size_t s = 0; s < total_tokens && s < targets_data.size(); ++s)
                {
                    int32_t target = static_cast<int32_t>(std::round(targets_data[s]));
                    if (target >= 0 && static_cast<size_t>(target) < V)
                    {
                        ++valid_tokens;
                    }
                }
            }
            std::vector<float> tgt_float(total_tokens, -100.0f);
            for (size_t s = 0; s < std::min(total_tokens, targets_data.size()); ++s)
            {
                tgt_float[s] = targets_data[s];
            }
            targets_buf = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), tgt_float);
        }

        uint32_t cur_frame = Execution_Engine::getInstance().getContext().getCurrentFrame();
        auto& slot = Execution_Engine::getInstance().getLossSlot(cur_frame);
        size_t required_bytes = total_tokens * sizeof(float);
        if (!slot.buffer || slot.buffer->getSizeBytes() < required_bytes)
        {
            slot.buffer = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            slot.buffer->allocateHostVisible(required_bytes);
        }

        struct Fused_CE_Constants
        {
            uint32_t seq_len;
            uint32_t vocab_size;
            uint32_t valid_tokens;
        } pcs{ static_cast<uint32_t>(total_tokens), static_cast<uint32_t>(V), valid_tokens };

        Compute_Pipeline pipe = (data_type == Data_Type::FLOAT16)
            ? Compute_Pipeline::FUSED_CROSS_ENTROPY_FP16
            : Compute_Pipeline::FUSED_CROSS_ENTROPY;

        pushToGraph(pipe,
                    { storage, targets_buf, d_logits_gpu.storage, slot.buffer },
                    pcs,
                    static_cast<uint32_t>(total_tokens), 1, 1);
        d_logits_gpu.host_cache.clear();

        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::FORWARD);

        slot.valid_tokens = valid_tokens;
        slot.total_tokens = total_tokens;
        slot.has_pending_read = true;

        if (valid_tokens == 0)
        {
            return 0.0f;
        }

        if (Execution_Engine::getInstance().isAsyncLossEnabled())
        {
            return Execution_Engine::getInstance().getLatestLoss();
        }
        else
        {
            return Execution_Engine::getInstance().readPendingLoss(cur_frame);
        }
    }

    const std::vector<float> &getData() const noexcept override
    {
        Logger::logMessage("Gpu_Tensor_Impl::getData: Reading data from GPU, risk of sync stall",
                           Log_Level::LOG_WARNING, true, 1, Log_Feature::MEMORY_TRANSFER);
        if (total_elements == 0)
        {
            host_cache.clear();
            return host_cache;
        }

        if (!isContiguous() || byte_offset != 0)
        {
            Gpu_Tensor_Impl contig_gpu(shape);
            contiguous(contig_gpu);
            Execution_Engine::getInstance().executeGraph();
            host_cache = contig_gpu.getData();
            return host_cache;
        }

        Execution_Engine::getInstance().getContext().executePendingTransfers();
        Execution_Engine::getInstance().getContext().flush();
        host_cache.resize(total_elements);
        if (storage)
        {
            storage->downloadData(host_cache);
        }
        return host_cache;
    }

    Mutable_Storage_Handle getStorage() override
    {
        if (!isContiguous() || byte_offset != 0)
        {
            auto contiguous_tensor = std::make_shared<Gpu_Tensor_Impl>(shape);
            contiguous(*contiguous_tensor);
            storage = contiguous_tensor->storage;
            byte_offset = 0;
            strides = shape.computeContiguousStrides();
        }
        return storage;
    }

    Storage_Handle getStorage() const override { return storage; }
    std::shared_ptr<gpu::vector> getVector() override { return storage; }
    bool isEmpty() const noexcept override { return !storage || storage->isEmpty(); }

};
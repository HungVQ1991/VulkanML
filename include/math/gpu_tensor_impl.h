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
    std::uint32_t batch_count = 1;
    std::uint32_t rows_a = 0;
    std::uint32_t columns_a = 0;
    std::uint32_t columns_b = 0;
    std::uint32_t broadcast_b = 0;
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

struct Contiguous_Push_Constants
{
    std::uint32_t total_elements = 0;
    std::uint32_t rank = 0;
    std::uint32_t offset_elements = 0;
    std::uint32_t shape_0 = 1;
    std::uint32_t shape_1 = 1;
    std::uint32_t shape_2 = 1;
    std::uint32_t shape_3 = 1;
    std::uint32_t shape_4 = 1;
    std::uint32_t shape_5 = 1;
    std::uint32_t stride_0 = 1;
    std::uint32_t stride_1 = 1;
    std::uint32_t stride_2 = 1;
    std::uint32_t stride_3 = 1;
    std::uint32_t stride_4 = 1;
    std::uint32_t stride_5 = 1;
};

class Gpu_Tensor_Impl : public Tensor_Impl
{
private:
    std::shared_ptr<gpu::vector> storage;
    mutable std::vector<float> host_cache;

    static inline bool is_graph_logging_enabled = true;

public:
    static inline std::size_t distinct_operations_count;

private:
    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum pipeline_id,
                     const std::vector<std::shared_ptr<gpu::vector>> &buffers,
                     const Push_Constants_Type &push_constants,
                     std::uint32_t workgroup_count_x,
                     std::uint32_t workgroup_count_y = 1,
                     std::uint32_t workgroup_count_z = 1) const
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
        auto contiguous_tensor = std::make_shared<Gpu_Tensor_Impl>(shape);
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

        output_gpu.reshape(shape);

        Elementwise_Dimensions dims{
            .total_elements = static_cast<std::uint32_t>(total_elements),
            .columns = static_cast<std::uint32_t>(getColumns()),
            .is_broadcast = static_cast<std::uint32_t>(is_broadcast ? 1 : 0)};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::executeElementwise: total={}, cols={}, broadcast={}",
                                        dims.total_elements, dims.columns, dims.is_broadcast},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);

        pushToGraph(pipeline_id, {effective_self->storage, effective_other->storage, output_gpu.storage}, dims, (dims.total_elements + 255) / 256);
    }

public:
    Gpu_Tensor_Impl(std::size_t rows, std::size_t columns)
    {
        updateShapeAndStrides(Shape{rows, columns});
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements);
        }
    }

    Gpu_Tensor_Impl(std::size_t rows, std::size_t columns, const std::vector<float> &host_data)
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

    explicit Gpu_Tensor_Impl(Shape tensor_shape)
    {
        updateShapeAndStrides(tensor_shape);
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements);
        }
    }

    Gpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data)
    {
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
        }
    }

    Gpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<gpu::vector> existing_storage, std::size_t offset_bytes)
    {
        shape = tensor_shape;
        strides = tensor_strides;
        storage = std::move(existing_storage);
        byte_offset = offset_bytes;
        total_elements = shape.getTotalElements();
    }

    ~Gpu_Tensor_Impl() noexcept override = default;

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
            return contig_gpu.getData();
        }

        Execution_Engine::getInstance().getContext().flush();
        host_cache.resize(total_elements);
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

    std::shared_ptr<gpu::vector> getVector() override
    {
        return storage;
    }

    bool isEmpty() const noexcept override
    {
        return !storage || storage->isEmpty();
    }

    void reshape(std::size_t rows, std::size_t columns) override
    {
        reshape(Shape{rows, columns});
    }

    void reshape(Shape new_shape) override
    {
        std::size_t new_total = new_shape.getTotalElements();
        if (!isContiguous() || byte_offset != 0)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::reshape: Cannot reshape non-contiguous view directly"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::runtime_error("Cannot reshape non-contiguous GPU tensor view");
        }

        if (shape == new_shape && storage && storage->getSize() == new_total)
        {
            return;
        }

        if (storage.use_count() > 1 && total_elements != new_total)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), new_total);
        }
        else if (!storage || storage->getSize() != new_total)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), new_total);
        }
        updateShapeAndStrides(new_shape);
    }

    void permute(const std::vector<std::size_t> &axes_permutation, Tensor_Impl &output) const override
    {
        Shape new_shape = shape;
        Stride new_strides = strides;
        for (std::size_t i = 0; i < shape.getRank(); ++i)
        {
            new_shape[i] = shape[axes_permutation[i]];
            new_strides[i] = strides[axes_permutation[i]];
        }
        auto &output_gpu = static_cast<Gpu_Tensor_Impl &>(output);
        output_gpu.shape = new_shape;
        output_gpu.strides = new_strides;
        output_gpu.storage = storage;
        output_gpu.byte_offset = byte_offset;
        output_gpu.total_elements = total_elements;
    }

    void slice(std::size_t axis, std::size_t start, std::size_t length, Tensor_Impl &output) const override
    {
        Shape new_shape = shape;
        new_shape[axis] = length;
        std::size_t add_bytes = start * strides[axis] * sizeof(float);

        auto &output_gpu = static_cast<Gpu_Tensor_Impl &>(output);
        output_gpu.shape = new_shape;
        output_gpu.strides = strides;
        output_gpu.storage = storage;
        output_gpu.byte_offset = byte_offset + add_bytes;
        output_gpu.total_elements = new_shape.getTotalElements();
    }

    void contiguous(Tensor_Impl &output) const override
    {
        auto &output_gpu = static_cast<Gpu_Tensor_Impl &>(output);
        output_gpu.reshape(shape);

        if (isContiguous())
        {
            Execution_Engine::getInstance().getContext().copyBuffer(
                storage->getBuffer(),
                output_gpu.storage->getBuffer(),
                total_elements * sizeof(float),
                byte_offset,
                0);
            return;
        }

        Contiguous_Push_Constants constants{
            .total_elements = static_cast<std::uint32_t>(total_elements),
            .rank = static_cast<std::uint32_t>(shape.getRank()),
            .offset_elements = static_cast<std::uint32_t>(byte_offset / sizeof(float))};

        for (std::size_t i = 0; i < shape.getRank() && i < 6; ++i)
        {
            if (i == 0)
            {
                constants.shape_0 = static_cast<std::uint32_t>(shape[0]);
                constants.stride_0 = static_cast<std::uint32_t>(strides[0]);
            }
            if (i == 1)
            {
                constants.shape_1 = static_cast<std::uint32_t>(shape[1]);
                constants.stride_1 = static_cast<std::uint32_t>(strides[1]);
            }
            if (i == 2)
            {
                constants.shape_2 = static_cast<std::uint32_t>(shape[2]);
                constants.stride_2 = static_cast<std::uint32_t>(strides[2]);
            }
            if (i == 3)
            {
                constants.shape_3 = static_cast<std::uint32_t>(shape[3]);
                constants.stride_3 = static_cast<std::uint32_t>(strides[3]);
            }
            if (i == 4)
            {
                constants.shape_4 = static_cast<std::uint32_t>(shape[4]);
                constants.stride_4 = static_cast<std::uint32_t>(strides[4]);
            }
            if (i == 5)
            {
                constants.shape_5 = static_cast<std::uint32_t>(shape[5]);
                constants.stride_5 = static_cast<std::uint32_t>(strides[5]);
            }
        }

        pushToGraph(Compute_Pipeline::CONTIGUOUS, {storage, output_gpu.storage}, constants, (constants.total_elements + 255) / 256);
    }

    void matmul(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        std::size_t rank_a = shape.getRank();
        std::size_t m_dim = (rank_a >= 2) ? shape[rank_a - 2] : getRows();
        std::size_t k_dim = (rank_a >= 2) ? shape[rank_a - 1] : getColumns();
        std::size_t b_dim = (rank_a >= 3) ? (total_elements / (m_dim * k_dim)) : 1;

        std::size_t rank_b = other.getShape().getRank();
        std::size_t k_other = (rank_b >= 2) ? other.getShape()[rank_b - 2] : other.getRows();
        std::size_t n_dim = (rank_b >= 2) ? other.getShape()[rank_b - 1] : other.getColumns();
        std::size_t b_other = (rank_b >= 3) ? (other.getTotalElements() / (k_other * n_dim)) : 1;

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
            std::vector<std::size_t> dims(shape.getDimensions().begin(), shape.getDimensions().end());
            dims[rank_a - 2] = m_dim;
            dims[rank_a - 1] = n_dim;
            out_shape = Shape(dims);
        }
        else
        {
            out_shape = Shape{b_dim, m_dim, n_dim};
        }

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(out_shape);

        Matrix_Dimensions dims{
            .batch_count = static_cast<std::uint32_t>(b_dim),
            .rows_a = static_cast<std::uint32_t>(m_dim),
            .columns_a = static_cast<std::uint32_t>(k_dim),
            .columns_b = static_cast<std::uint32_t>(n_dim),
            .broadcast_b = broadcast_b ? 1u : 0u};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::matmul: batch={}, rows_a={}, cols_a={}, cols_b={}",
                                        dims.batch_count, dims.rows_a, dims.columns_a, dims.columns_b},
                           Log_Level::LOG_DEBUG, true, 1, Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MATMUL, {effective_self->storage, effective_other->storage, output_gpu.storage},
                    dims, (dims.columns_b + 15) / 16, (dims.rows_a + 15) / 16, dims.batch_count);
    }

    void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        Gpu_Tensor_Impl temp_inv(0, 0);
        other.inverse(temp_inv);
        matmul(temp_inv, output);
    }

    void add(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        executeElementwise(other, output, Compute_Pipeline::ADD, true);
    }

    void sub(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        executeElementwise(other, output, Compute_Pipeline::SUB, true);
    }

    void mulScalar(float scalar, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(shape);

        struct Scalar_Constants
        {
            std::uint32_t total_elements;
            float scalar;
        } constants{static_cast<std::uint32_t>(total_elements), scalar};

        pushToGraph(Compute_Pipeline::MUL_SCALAR, {effective_self->storage, output_gpu.storage}, constants, (total_elements + 255) / 256);
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
        executeElementwise(other, output, Compute_Pipeline::HADAMARD_MUL, false);
    }

    void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        executeElementwise(other, output, Compute_Pipeline::HADAMARD_DIV, false);
    }

    void transpose(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(getColumns(), getRows());

        Transpose_Dimensions dims{
            .rows = static_cast<std::uint32_t>(getRows()),
            .columns = static_cast<std::uint32_t>(getColumns())};

        pushToGraph(Compute_Pipeline::TRANSPOSE, {effective_self->storage, output_gpu.storage}, dims,
                    (dims.columns + 15) / 16, (dims.rows + 15) / 16);
    }

    void inverse(Tensor_Impl &output) const override
    {
        validateSquare();
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        std::uint32_t dim_size = static_cast<std::uint32_t>(getRows());
        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(dim_size, dim_size);

        struct Inverse_Constants
        {
            std::uint32_t dimension_size;
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
            std::uint32_t total_elements;
        } constants{static_cast<std::uint32_t>(total_elements)};

        pushToGraph(Compute_Pipeline::NORMALIZE, {effective_self->storage, output_gpu.storage}, constants, 1, 1, 1);
    }

    void relu(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(shape);
        pushToGraph(Compute_Pipeline::RELU, {effective_self->storage, output_gpu.storage},
                    static_cast<std::uint32_t>(total_elements), (static_cast<std::uint32_t>(total_elements) + 255) / 256);
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
        in_grad_gpu.reshape(shape);

        pushToGraph(Compute_Pipeline::RELU_BACKWARD, {effective_self->storage, effective_grad->storage, in_grad_gpu.storage},
                    static_cast<std::uint32_t>(total_elements), (static_cast<std::uint32_t>(total_elements) + 255) / 256);
    }

    void gelu(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(shape);
        pushToGraph(Compute_Pipeline::GELU, {effective_self->storage, output_gpu.storage},
                    static_cast<std::uint32_t>(total_elements), (static_cast<std::uint32_t>(total_elements) + 255) / 256);
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
        in_grad_gpu.reshape(shape);

        pushToGraph(Compute_Pipeline::GELU_BACKWARD, {effective_self->storage, effective_grad->storage, in_grad_gpu.storage},
                    static_cast<std::uint32_t>(total_elements), (static_cast<std::uint32_t>(total_elements) + 255) / 256);
    }

    void softmax(Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(getRows(), getColumns());

        struct Softmax_Constants
        {
            std::uint32_t rows;
            std::uint32_t columns;
        } constants{static_cast<std::uint32_t>(getRows()), static_cast<std::uint32_t>(getColumns())};

        pushToGraph(Compute_Pipeline::SOFTMAX, {effective_self->storage, output_gpu.storage}, constants, constants.rows, 1, 1);
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
        in_grad_gpu.reshape(getRows(), getColumns());

        struct Softmax_Constants
        {
            std::uint32_t rows;
            std::uint32_t columns;
        } constants{static_cast<std::uint32_t>(getRows()), static_cast<std::uint32_t>(getColumns())};

        pushToGraph(Compute_Pipeline::SOFTMAX_BACKWARD, {effective_self->storage, effective_grad->storage, in_grad_gpu.storage}, constants, constants.rows, 1, 1);
    }

    void sgdUpdate(const Tensor_Impl &gradient, float learning_rate, float max_gradient = 0.0F) override
    {
        validateSameDimensions(gradient);
        const auto &grad_gpu = castToGpu(gradient);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        struct Sgd_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float max_gradient;
        } constants{static_cast<std::uint32_t>(total_elements), learning_rate, max_gradient};

        pushToGraph(Compute_Pipeline::SGD_UPDATE, {storage, effective_grad->storage}, constants, (total_elements + 255) / 256);
    }

    void adamUpdate(const Tensor_Impl &gradient,
                    const Tensor_Impl &first_moment,
                    const Tensor_Impl &second_moment,
                    float learning_rate,
                    float beta1,
                    float beta2,
                    float epsilon,
                    std::size_t timestep,
                    float max_gradient = 1.0F) override
    {
        validateSameDimensions(gradient);
        validateSameDimensions(first_moment);
        validateSameDimensions(second_moment);

        const auto &grad_gpu = castToGpu(gradient);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        const auto &m_gpu = castToGpu(first_moment);
        const auto &v_gpu = castToGpu(second_moment);

        std::size_t effective_t = std::max<std::size_t>(timestep, 1);
        float bc1 = std::max(1.0F - std::pow(beta1, static_cast<float>(effective_t)), 1e-8F);
        float bc2 = std::max(1.0F - std::pow(beta2, static_cast<float>(effective_t)), 1e-8F);

        struct Adam_Constants
        {
            std::uint32_t total_elements;
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float max_gradient;
            float inv_bc1;
            float inv_sqrt_bc2;
        } constants{static_cast<std::uint32_t>(total_elements), learning_rate, beta1, beta2, epsilon, max_gradient,
                    1.0F / bc1, 1.0F / std::sqrt(bc2)};

        pushToGraph(Compute_Pipeline::ADAM_UPDATE, {storage, effective_grad->storage, m_gpu.storage, v_gpu.storage}, constants, (total_elements + 255) / 256);
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

        std::size_t rank_a = shape.getRank();
        std::size_t m_dim = (rank_a >= 2) ? shape[rank_a - 2] : getRows();
        std::size_t k_dim = (rank_a >= 2) ? shape[rank_a - 1] : getColumns();
        std::size_t b_dim = (rank_a >= 3) ? (total_elements / (m_dim * k_dim)) : 1;

        std::size_t rank_w = weights.getShape().getRank();
        std::size_t k_w = (rank_w >= 2) ? weights.getShape()[rank_w - 2] : weights.getRows();
        std::size_t n_dim = (rank_w >= 2) ? weights.getShape()[rank_w - 1] : weights.getColumns();
        std::size_t b_w = (rank_w >= 3) ? (weights.getTotalElements() / (k_w * n_dim)) : 1;

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
            std::vector<std::size_t> dims(shape.getDimensions().begin(), shape.getDimensions().end());
            dims[rank_a - 2] = m_dim;
            dims[rank_a - 1] = n_dim;
            out_shape = Shape(dims);
        }
        else
        {
            out_shape = Shape{b_dim, m_dim, n_dim};
        }

        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(out_shape);

        std::size_t b_total_elems = biases.getTotalElements();
        std::uint32_t broadcast_b_flag = 0;
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
            std::uint32_t batch_count;
            std::uint32_t rows_x;
            std::uint32_t columns_x;
            std::uint32_t columns_weights;
            std::uint32_t broadcast_w;
            std::uint32_t broadcast_b;
        } constants{
            .batch_count = static_cast<std::uint32_t>(b_dim),
            .rows_x = static_cast<std::uint32_t>(m_dim),
            .columns_x = static_cast<std::uint32_t>(k_dim),
            .columns_weights = static_cast<std::uint32_t>(n_dim),
            .broadcast_w = broadcast_w ? 1u : 0u,
            .broadcast_b = broadcast_b_flag};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::matmulAdd: batch={}, rows_x={}, cols_x={}, cols_w={}",
                                        constants.batch_count, constants.rows_x, constants.columns_x, constants.columns_weights},
                           Log_Level::LOG_DEBUG, true, 1, Log_Feature::DENSE_COMPUTE);

        pushToGraph(Compute_Pipeline::MATMUL_ADD, {effective_self->storage, effective_w->storage, effective_b->storage, output_gpu.storage}, constants,
                    (constants.columns_weights + 15) / 16, (constants.rows_x + 15) / 16, constants.batch_count);
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
            Gpu_Tensor_Impl temp_upload(shape, host_data);
            Execution_Engine::getInstance().getContext().copyBuffer(
                temp_upload.storage->getBuffer(),
                storage->getBuffer(),
                total_elements * sizeof(float),
                0,
                byte_offset);
            return;
        }

        if (storage)
        {
            storage->uploadData(host_data);
        }
    }

    void conv2d(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output,
                std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                std::uint32_t output_channels, std::uint32_t kernel_size,
                std::uint32_t stride, std::uint32_t padding) const override
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

        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        std::uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;
        output_gpu.reshape(batch_size, out_h * out_w * output_channels);

        struct Conv2d_Constants
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
        } constants{batch_size, input_height, input_width, input_channels, out_h, out_w, output_channels, kernel_size, stride, padding};

        pushToGraph(Compute_Pipeline::CONV2D_FORWARD_PASS, {effective_self->storage, effective_w->storage, effective_b->storage, output_gpu.storage}, constants,
                    (output_channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }

    void conv2dBackwardInput(const Tensor_Impl &weights, Tensor_Impl &input_gradient,
                             std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                             std::uint32_t output_height, std::uint32_t output_width, std::uint32_t output_channels,
                             std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &w_gpu = castToGpu(weights);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        in_grad_gpu.reshape(batch_size, input_height * input_width * input_channels);

        struct Conv2d_Constants
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
        } constants{batch_size, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding};

        pushToGraph(Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT, {effective_self->storage, effective_w->storage, in_grad_gpu.storage}, constants,
                    (input_channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }

    void conv2dBackwardWeight(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient,
                              std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
                              std::uint32_t output_height, std::uint32_t output_width, std::uint32_t output_channels,
                              std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &w_grad_gpu = castToGpu(weight_gradient);
        auto &b_grad_gpu = castToGpu(bias_gradient);

        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        w_grad_gpu.reshape(1, kernel_size * kernel_size * input_channels * output_channels);
        b_grad_gpu.reshape(1, output_channels);

        struct Conv2d_Constants
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
        } constants{batch_size, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding};

        pushToGraph(Compute_Pipeline::CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT, {effective_self->storage, effective_grad->storage, w_grad_gpu.storage, b_grad_gpu.storage}, constants,
                    (output_channels + 15) / 16, (input_channels + 15) / 16, kernel_size * kernel_size);
    }

    void maxpool2d(Tensor_Impl &output, Tensor_Impl &output_mask,
                   std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels,
                   std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &res_gpu = castToGpu(output);
        auto &mask_gpu = castToGpu(output_mask);

        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        std::uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
        std::uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

        res_gpu.reshape(batch_size, out_h * out_w * channels);
        mask_gpu.reshape(batch_size, out_h * out_w * channels);

        struct Pool_Constants
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
        } constants{batch_size, input_height, input_width, channels, out_h, out_w, kernel_size, stride, padding};

        pushToGraph(Compute_Pipeline::MAXPOOL2D_FORWARD, {effective_self->storage, res_gpu.storage, mask_gpu.storage}, constants,
                    (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }

    void maxpool2dBackward(const Tensor_Impl &mask, Tensor_Impl &input_gradient,
                           std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels,
                           std::uint32_t output_height, std::uint32_t output_width,
                           std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t padding) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &mask_gpu = castToGpu(mask);
        auto contig_mask = mask_gpu.ensureContiguousSelf();
        const auto *effective_mask = contig_mask ? contig_mask.get() : &mask_gpu;

        auto &in_grad_gpu = castToGpu(input_gradient);
        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        in_grad_gpu.reshape(batch_size, input_height * input_width * channels);

        struct Pool_Constants
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
        } constants{batch_size, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding};

        pushToGraph(Compute_Pipeline::MAXPOOL2D_BACKWARD, {effective_mask->storage, effective_self->storage, in_grad_gpu.storage}, constants,
                    (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }

    void globalAvgPool2d(Tensor_Impl &output, std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        auto &output_gpu = castToGpu(output);
        output_gpu.reshape(batch_size, channels);

        struct Avg_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t channels;
        } constants{batch_size, input_height, input_width, channels};

        pushToGraph(Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD, {effective_self->storage, output_gpu.storage}, constants,
                    (channels + 255) / 256, batch_size, 1);
    }

    void globalAvgPool2dBackward(Tensor_Impl &input_gradient, std::uint32_t input_height, std::uint32_t input_width, std::uint32_t channels) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        std::uint32_t batch_size = static_cast<std::uint32_t>(getRows());
        auto &in_grad_gpu = castToGpu(input_gradient);
        in_grad_gpu.reshape(batch_size, input_height * input_width * channels);

        struct Avg_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_height;
            std::uint32_t input_width;
            std::uint32_t channels;
        } constants{batch_size, input_height, input_width, channels};

        pushToGraph(Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD, {effective_self->storage, in_grad_gpu.storage}, constants,
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

        std::uint32_t b_count = static_cast<std::uint32_t>(getRows());
        std::uint32_t f_dim = static_cast<std::uint32_t>(getColumns());
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
                std::uint32_t batch_size;
                std::uint32_t feature_dimension;
                float momentum;
            } stats{b_count, f_dim, momentum};

            pushToGraph(Compute_Pipeline::BATCH_NORM_STATS_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, rm_gpu.storage, rv_gpu.storage},
                        stats, (f_dim + 255) / 256, 1, 1);

            struct Transform_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, f_dim, epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_count * f_dim + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t feature_dimension;
                float epsilon;
            } tf{b_count * f_dim, f_dim, epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM_TRANSFORM_FORWARD,
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

        std::uint32_t b_count = static_cast<std::uint32_t>(getRows());
        std::uint32_t f_dim = static_cast<std::uint32_t>(getColumns());
        g_grad_gpu.reshape(1, f_dim);
        b_grad_gpu.reshape(1, f_dim);
        in_grad_gpu.reshape(b_count, f_dim);

        struct Stats_Constants
        {
            std::uint32_t batch_size;
            std::uint32_t feature_dimension;
        } stats{b_count, f_dim};

        pushToGraph(Compute_Pipeline::BATCH_NORM_STATS_BACKWARD,
                    {effective_grad->storage, norm_in_gpu.storage, g_grad_gpu.storage, b_grad_gpu.storage},
                    stats, (f_dim + 255) / 256, 1, 1);

        struct Transform_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t batch_size;
            std::uint32_t feature_dimension;
            float epsilon;
        } tf{b_count * f_dim, b_count, f_dim, epsilon};

        pushToGraph(Compute_Pipeline::BATCH_NORM_TRANSFORM_BACKWARD,
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
        in_grad_gpu.reshape(getRows(), effective_w->getRows());

        struct Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_dimension;
            std::uint32_t output_dimension;
        } c{static_cast<std::uint32_t>(getRows()), static_cast<std::uint32_t>(effective_w->getRows()), static_cast<std::uint32_t>(getColumns())};

        pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_INPUT, {effective_self->storage, effective_w->storage, in_grad_gpu.storage}, c,
                    (c.input_dimension + 15) / 16, (c.batch_size + 15) / 16, 1);
    }

    void linearBackwardWeightBias(const Tensor_Impl &output_gradient, Tensor_Impl &weight_gradient, Tensor_Impl &bias_gradient) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &out_grad_gpu = castToGpu(output_gradient);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        auto &w_grad_gpu = castToGpu(weight_gradient);
        auto &b_grad_gpu = castToGpu(bias_gradient);

        w_grad_gpu.reshape(getColumns(), effective_grad->getColumns());
        b_grad_gpu.reshape(1, effective_grad->getColumns());

        struct Constants
        {
            std::uint32_t batch_size;
            std::uint32_t input_dimension;
            std::uint32_t output_dimension;
        } c{static_cast<std::uint32_t>(getRows()), static_cast<std::uint32_t>(getColumns()), static_cast<std::uint32_t>(effective_grad->getColumns())};

        pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS, {effective_self->storage, effective_grad->storage, w_grad_gpu.storage, b_grad_gpu.storage}, c,
                    (c.output_dimension + 15) / 16, (c.input_dimension + 15) / 16, 1);
    }

    void batchNorm2dForward(const Tensor_Impl &gamma, const Tensor_Impl &beta,
                            Tensor_Impl &running_mean, Tensor_Impl &running_variance,
                            Tensor_Impl &batch_mean, Tensor_Impl &batch_variance,
                            Tensor_Impl &normalized_input, Tensor_Impl &output,
                            std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels,
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

        std::uint32_t b_size = static_cast<std::uint32_t>(getRows());
        std::uint32_t tot_feat = input_height * input_width * input_channels;
        std::uint32_t sp_count = b_size * input_height * input_width;
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
                std::uint32_t total_elements;
                std::uint32_t channels;
                std::uint32_t spatial_count;
                float momentum;
            } stats{b_size * tot_feat, input_channels, sp_count, momentum};

            pushToGraph(Compute_Pipeline::BATCH_NORM2D_STATS_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, rm_gpu.storage, rv_gpu.storage},
                        stats, (input_channels + 255) / 256, 1, 1);

            struct Transform_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                        {effective_self->storage, bm_gpu.storage, bv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
        else
        {
            struct Transform_Constants
            {
                std::uint32_t total_elements;
                std::uint32_t channels;
                float epsilon;
            } tf{b_size * tot_feat, input_channels, epsilon};

            pushToGraph(Compute_Pipeline::BATCH_NORM2D_TRANSFORM_FORWARD,
                        {effective_self->storage, rm_gpu.storage, rv_gpu.storage, gamma_gpu.storage, beta_gpu.storage, out_gpu.storage, norm_in_gpu.storage},
                        tf, (b_size * tot_feat + 255) / 256, 1, 1);
        }
    }

    void batchNorm2dBackward(const Tensor_Impl &gamma, const Tensor_Impl &batch_variance, const Tensor_Impl &normalized_input,
                             Tensor_Impl &gamma_gradient, Tensor_Impl &beta_gradient, Tensor_Impl &input_gradient,
                             std::uint32_t input_height, std::uint32_t input_width, std::uint32_t input_channels, float epsilon) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &gamma_gpu = castToGpu(gamma);
        const auto &bv_gpu = castToGpu(batch_variance);
        const auto &norm_in_gpu = castToGpu(normalized_input);
        auto &g_grad_gpu = castToGpu(gamma_gradient);
        auto &b_grad_gpu = castToGpu(beta_gradient);
        auto &in_grad_gpu = castToGpu(input_gradient);

        std::uint32_t b_size = static_cast<std::uint32_t>(getRows());
        std::uint32_t tot_feat = input_height * input_width * input_channels;
        std::uint32_t sp_count = b_size * input_height * input_width;
        in_grad_gpu.reshape(b_size, tot_feat);
        g_grad_gpu.reshape(1, input_channels);
        b_grad_gpu.reshape(1, input_channels);

        struct Stats_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t channels;
            std::uint32_t spatial_count;
        } stats{b_size * tot_feat, input_channels, sp_count};

        pushToGraph(Compute_Pipeline::BATCH_NORM2D_STATS_BACKWARD,
                    {effective_self->storage, norm_in_gpu.storage, g_grad_gpu.storage, b_grad_gpu.storage},
                    stats, (input_channels + 255) / 256, 1, 1);

        struct Transform_Constants
        {
            std::uint32_t total_elements;
            std::uint32_t channels;
            std::uint32_t spatial_count;
            float epsilon;
        } tf{b_size * tot_feat, input_channels, sp_count, epsilon};

        pushToGraph(Compute_Pipeline::BATCH_NORM2D_TRANSFORM_BACKWARD,
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

        std::uint32_t total = static_cast<std::uint32_t>(total_elements);
        std::uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            std::uint32_t total_elements;
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

        std::uint32_t total = static_cast<std::uint32_t>(total_elements);
        std::uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            std::uint32_t total_elements;
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

        std::uint32_t total = static_cast<std::uint32_t>(total_elements);
        std::uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            std::uint32_t total_elements;
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

        std::uint32_t total = static_cast<std::uint32_t>(total_elements);
        std::uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            std::uint32_t total_elements;
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

        std::uint32_t total = static_cast<std::uint32_t>(total_elements);
        std::uint32_t wg_x = (total + 255) / 256;
        output_gpu.reshape(1, wg_x);

        struct Constants
        {
            std::uint32_t total_elements;
            float delta;
        } c{total, delta};

        pushToGraph(Compute_Pipeline::HUBER_LOSS, {effective_self->storage, effective_target->storage, output_gpu.storage}, c, wg_x, 1, 1);
    }

    void concatenateCollumns(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        auto &output_gpu = castToGpu(output);

        std::uint32_t cols_a = static_cast<std::uint32_t>(getColumns());
        std::uint32_t cols_b = static_cast<std::uint32_t>(effective_other->getColumns());
        std::uint32_t tot_cols = cols_a + cols_b;
        output_gpu.reshape(getRows(), tot_cols);

        struct Constants
        {
            std::uint32_t rows;
            std::uint32_t columns_a;
            std::uint32_t columns_b;
        } c{static_cast<std::uint32_t>(getRows()), cols_a, cols_b};

        pushToGraph(Compute_Pipeline::CONCATENATE_COLUMNS, {effective_self->storage, effective_other->storage, output_gpu.storage}, c,
                    (tot_cols + 15) / 16, (c.rows + 15) / 16);
    }

    void concatenateRows(const Tensor_Impl &other, Tensor_Impl &output) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        auto &output_gpu = castToGpu(output);

        std::uint32_t rows_a = static_cast<std::uint32_t>(getRows());
        std::uint32_t rows_b = static_cast<std::uint32_t>(effective_other->getRows());
        std::uint32_t tot_rows = rows_a + rows_b;
        output_gpu.reshape(tot_rows, getColumns());

        struct Constants
        {
            std::uint32_t rows_a;
            std::uint32_t rows_b;
            std::uint32_t columns;
        } c{rows_a, rows_b, static_cast<std::uint32_t>(getColumns())};

        pushToGraph(Compute_Pipeline::CONCATENATE_ROWS, {effective_self->storage, effective_other->storage, output_gpu.storage}, c,
                    (c.columns + 15) / 16, (tot_rows + 15) / 16);
    }

    void splitCollumns(std::size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &left_gpu = castToGpu(result_left);
        auto &right_gpu = castToGpu(result_right);
        std::uint32_t cols_left = static_cast<std::uint32_t>(split_index);
        std::uint32_t cols_right = static_cast<std::uint32_t>(getColumns() - split_index);

        left_gpu.reshape(getRows(), cols_left);
        right_gpu.reshape(getRows(), cols_right);

        struct Constants
        {
            std::uint32_t rows;
            std::uint32_t columns_left;
            std::uint32_t columns_right;
        } c{static_cast<std::uint32_t>(getRows()), cols_left, cols_right};

        pushToGraph(Compute_Pipeline::SPLIT_COLUMNS, {effective_self->storage, left_gpu.storage, right_gpu.storage}, c,
                    (static_cast<std::uint32_t>(getColumns()) + 15) / 16, (c.rows + 15) / 16);
    }

    void splitRows(std::size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override
    {
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &up_gpu = castToGpu(result_up);
        auto &down_gpu = castToGpu(result_down);
        std::uint32_t rows_up = static_cast<std::uint32_t>(split_index);
        std::uint32_t rows_down = static_cast<std::uint32_t>(getRows() - split_index);

        up_gpu.reshape(rows_up, getColumns());
        down_gpu.reshape(rows_down, getColumns());

        struct Constants
        {
            std::uint32_t rows_up;
            std::uint32_t rows_down;
            std::uint32_t columns;
        } c{rows_up, rows_down, static_cast<std::uint32_t>(getColumns())};

        pushToGraph(Compute_Pipeline::SPLIT_ROWS, {effective_self->storage, up_gpu.storage, down_gpu.storage}, c,
                    (c.columns + 15) / 16, (static_cast<std::uint32_t>(getRows()) + 15) / 16);
    }
};

using Gpu_Matrix_Impl = Gpu_Tensor_Impl;
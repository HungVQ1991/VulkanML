#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
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
    uint32_t batch_rank = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t batch_dims[4] = {1, 1, 1, 1};
    uint32_t a_batch_strides[4] = {0, 0, 0, 0};
    uint32_t b_batch_strides[4] = {0, 0, 0, 0};
};
static_assert(offsetof(Matrix_Dimensions, batch_dims) == 32);
static_assert(sizeof(Matrix_Dimensions) == 80);

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
    mutable std::vector<uint32_t> host_cache_u32;
    mutable std::mutex cache_mutex;

    static inline std::atomic<bool> is_graph_logging_enabled{true};

public:
    static inline size_t distinct_operations_count = 0;

    void invalidateFp16Cache() noexcept override
    {
        if (storage)
        {
            storage->invalidateFp16ShadowCache();
        }
    }

    void prewarmFp16Cache() override
    {
        getEffectiveFp16Storage();
    }
    std::shared_ptr<gpu::vector> getEffectiveFp16Storage() const;

public:
    template <typename Pipeline_Enum, typename Push_Constants_Type>
    void pushToGraph(Pipeline_Enum pipeline_id,
                     const std::vector<std::shared_ptr<gpu::vector>> &buffers,
                     const Push_Constants_Type &push_constants,
                     uint32_t workgroup_count_x,
                     uint32_t workgroup_count_y = 1,
                     uint32_t workgroup_count_z = 1,
                     Execution_Engine *engine = nullptr,
                     uint32_t spec_constant_u32 = 0) const
    {
        static_assert(std::is_trivially_copyable_v<Push_Constants_Type>,
                      "Gpu_Tensor_Impl::pushToGraph: Push constants must be trivially copyable");
        static_assert(sizeof(Push_Constants_Type) <= 128,
                      "Gpu_Tensor_Impl::pushToGraph: Push constants exceed Vulkan 128-byte limit");

        if (buffers.size() > DESCRIPTOR_BINDINGS_COUNT)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::pushToGraph: Exceeded max buffer count ({} > {})",
                                            buffers.size(), DESCRIPTOR_BINDINGS_COUNT},
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
        node.spec_constant_u32 = spec_constant_u32;

        (engine ? *engine : Execution_Engine::getInstance()).getCurrentGraph().addNode(std::move(node));

        if (is_graph_logging_enabled.load(std::memory_order_relaxed))
        {
            is_graph_logging_enabled.store(Logger::logMessage(
                Input_Format{"Gpu_Tensor_Impl::pushToGraph: Op={}, Dispatch=({}, {}, {}), Buffers={}",
                             magic_enum::enum_name(pipeline_id),
                             workgroup_count_x, workgroup_count_y, workgroup_count_z,
                             buffers.size()},
                Log_Level::LOG_DEBUG, true, distinct_operations_count, Log_Feature::GRAPH_RECORDING),
                std::memory_order_relaxed);
        }
    }

    static const Gpu_Tensor_Impl &castToGpu(const Tensor_Impl &other)
    {
        const auto *ptr = dynamic_cast<const Gpu_Tensor_Impl *>(&other);
        if (!ptr)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::castToGpu: Expected Gpu_Tensor_Impl (Execution_Target mismatch)");
        }
        return *ptr;
    }

    static Gpu_Tensor_Impl &castToGpu(Tensor_Impl &other)
    {
        auto *ptr = dynamic_cast<Gpu_Tensor_Impl *>(&other);
        if (!ptr)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::castToGpu: Expected Gpu_Tensor_Impl (Execution_Target mismatch)");
        }
        return *ptr;
    }

    std::shared_ptr<Gpu_Tensor_Impl> ensureContiguousSelf() const;

    const std::shared_ptr<gpu::vector> &getStorageBuffer() const noexcept { return storage; }
    std::shared_ptr<gpu::vector> &getStorageBuffer() noexcept { return storage; }
    std::shared_ptr<gpu::vector> getCachedFp16Storage() const;
    void setCachedFp16Storage(std::shared_ptr<gpu::vector> s) const;
    std::shared_ptr<gpu::vector> ensureCachedFp16Storage(size_t elements) const;
    bool isFp16CacheDirty() const;
    void setFp16CacheDirty(bool dirty) const;
    const std::vector<float> &getHostCache() const noexcept { return host_cache; }
    std::vector<float> &getHostCache() noexcept { return host_cache; }
    void clearHostCache() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        host_cache.clear();
        host_cache_u32.clear();
    }
    void markGpuMutated()
    {
        invalidateFp16Cache();
        clearHostCache();
    }
    std::shared_ptr<gpu::vector> getOrUploadIndicesBuffer(const std::vector<int32_t>& indices) const;

    const std::vector<uint32_t> &getPackedData() const override;
    void uploadPackedData(const std::vector<uint32_t> &packed_data) override;

private:
    template <typename Pipeline_Enum>
    void executeElementwise(const Tensor_Impl &other, Tensor_Impl &output, Pipeline_Enum pipeline_id, bool is_broadcast_allowed) const
    {
        if (data_type == Data_Type::TERNARY_INT2 || other.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::executeElementwise: TERNARY_INT2 is not supported");
        }

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        auto &output_gpu = castToGpu(output);

        bool same_shape = (shape == effective_other->shape);
        bool is_broadcast = false;
        if (is_broadcast_allowed && !same_shape)
        {
            size_t other_rank = effective_other->shape.getRank();
            size_t self_rank = shape.getRank();
            if (effective_other->getTotalElements() > 0 && other_rank > 0 && self_rank >= other_rank)
            {
                size_t first_non_one = 0;
                while (first_non_one < other_rank && effective_other->shape[first_non_one] == 1)
                {
                    ++first_non_one;
                }
                size_t suffix_len = other_rank - first_non_one;
                bool suffix_match = (suffix_len <= self_rank);
                for (size_t i = 0; suffix_match && i < suffix_len; ++i)
                {
                    if (effective_other->shape[first_non_one + i] != shape[self_rank - suffix_len + i])
                    {
                        suffix_match = false;
                    }
                }
                is_broadcast = suffix_match;
            }
            if (!is_broadcast)
            {
                validateSameDimensions(*effective_other);
            }
        }
        else if (!same_shape)
        {
            validateSameDimensions(*effective_other);
        }

        bool is_fp16_pipe = (pipeline_id == Compute_Pipeline::ADD_FP16 ||
                             pipeline_id == Compute_Pipeline::SUB_FP16 ||
                             pipeline_id == Compute_Pipeline::HADAMARD_MUL_FP16 ||
                             pipeline_id == Compute_Pipeline::HADAMARD_DIV_FP16);

        const Shape src_shape = shape;
        const Data_Type out_dtype = is_fp16_pipe ? Data_Type::FLOAT16 : data_type;
        const size_t src_total = total_elements;
        const uint32_t eff_cols = is_broadcast
                                      ? static_cast<uint32_t>(effective_other->getTotalElements())
                                      : static_cast<uint32_t>(getColumns());

        std::shared_ptr<gpu::vector> buf_self;
        std::shared_ptr<gpu::vector> buf_other;
        if (src_total > 0)
        {
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
        }

        output_gpu.setDataType(out_dtype);
        output_gpu.reshape(src_shape);

        if (src_total == 0)
        {
            output_gpu.markGpuMutated();
            return;
        }

        if ((is_broadcast && output_gpu.storage.get() == buf_other.get()) ||
            (&output != this && &output != &other &&
             (output_gpu.storage.get() == buf_self.get() || output_gpu.storage.get() == buf_other.get())))
        {
            output_gpu.storage = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), src_total, out_dtype);
        }

        Elementwise_Dimensions dims{
            .total_elements = static_cast<uint32_t>(src_total),
            .columns = eff_cols,
            .is_broadcast = static_cast<uint32_t>(is_broadcast ? 1 : 0)};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::executeElementwise: total={}, cols={}, broadcast={}",
                                        dims.total_elements, dims.columns, dims.is_broadcast},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::DENSE_COMPUTE);

        pushToGraph(pipeline_id, {buf_self, buf_other, output_gpu.storage}, dims, (dims.total_elements + 255) / 256);
        output_gpu.markGpuMutated();
    }

public:
    Gpu_Tensor_Impl(size_t rows, size_t columns);
    Gpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data);
    explicit Gpu_Tensor_Impl(Shape tensor_shape, Data_Type type = Data_Type::FLOAT32);
    Gpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data, Data_Type type = Data_Type::FLOAT32);
    Gpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<gpu::vector> existing_storage, size_t offset_bytes, Data_Type type = Data_Type::FLOAT32);

    ~Gpu_Tensor_Impl() noexcept override = default;
    void reshape(size_t rows, size_t columns) override;
    void reshape(Shape new_shape) override;
    void permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const override;
    void slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const override;
    void updateSlice(size_t axis, size_t start, const Tensor_Impl &source) override;
    void gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const override;
    void contiguous(Tensor_Impl &output) const override;
    void to(Data_Type target_type, Tensor_Impl &output) const override;
    void matmul(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void matdiv(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void add(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void sub(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void mulScalar(float scalar, Tensor_Impl &output) const override;
    void divScalar(float scalar, Tensor_Impl &output) const override;
    void hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const override;
    void transpose(Tensor_Impl &output) const override;
    void inverse(Tensor_Impl &output) const override;
    void normalize(Tensor_Impl &output) const override;
    void matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const override;
    void uploadData(const std::vector<float> &host_data) override;
    void zero() override;
    void fill(float value) override;
    void concatenateColumns(const Tensor_Impl& other, Tensor_Impl& output) const override;
    void concatenateRows(const Tensor_Impl& other, Tensor_Impl& output) const override;
    void splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const override;
    void splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const override;
    const std::vector<float> &getData() const override;
    Mutable_Storage_Handle getStorage() override;

    Storage_Handle getStorage() const override { return storage; }
    bool isEmpty() const noexcept override { return !storage || storage->isEmpty(); }

};

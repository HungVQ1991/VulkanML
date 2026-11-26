#include "math/gpu_tensor_impl.h"

#include <algorithm>
#include <array>
#include <bit>
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
#include "math/cpu_tensor_impl.h"

std::shared_ptr<gpu::vector> Gpu_Tensor_Impl::getCachedFp16Storage() const
{
    if (!storage)
    {
        return {};
    }
    auto cache = storage->getFp16ShadowCache();
    if (!cache)
    {
        return {};
    }
    std::lock_guard<std::mutex> lock(cache->mutex);
    return cache->fp16_storage;
}

void Gpu_Tensor_Impl::setCachedFp16Storage(std::shared_ptr<gpu::vector> s) const
{
    if (!storage)
    {
        return;
    }
    auto cache = storage->ensureFp16ShadowCache();
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->fp16_storage = std::move(s);
    cache->cached_version = 0;
    cache->valid_elements = 0;
}

std::shared_ptr<gpu::vector> Gpu_Tensor_Impl::ensureCachedFp16Storage(size_t elements) const
{
    if (!storage)
    {
        return {};
    }
    auto cache = storage->ensureFp16ShadowCache();
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (!cache->fp16_storage || cache->fp16_storage->getSize() < elements)
    {
        cache->fp16_storage = std::make_shared<gpu::vector>(storage->getContext(), elements, Data_Type::FLOAT16);
        cache->cached_version = 0;
        cache->valid_elements = 0;
    }
    return cache->fp16_storage;
}

bool Gpu_Tensor_Impl::isFp16CacheDirty() const
{
    if (!storage)
    {
        return true;
    }
    auto cache = storage->getFp16ShadowCache();
    if (!cache)
    {
        return true;
    }
    std::lock_guard<std::mutex> lock(cache->mutex);
    const uint64_t cur_ver = cache->storage_version.load(std::memory_order_acquire);
    return !cache->fp16_storage || cache->cached_version != cur_ver || cache->valid_elements < total_elements;
}

void Gpu_Tensor_Impl::setFp16CacheDirty(bool dirty) const
{
    if (!storage)
    {
        return;
    }
    if (dirty)
    {
        storage->invalidateFp16ShadowCache();
        return;
    }
    auto cache = storage->ensureFp16ShadowCache();
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->cached_version = cache->storage_version.load(std::memory_order_acquire);
    cache->valid_elements = total_elements;
}

std::shared_ptr<gpu::vector> Gpu_Tensor_Impl::getEffectiveFp16Storage() const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::getEffectiveFp16Storage: TERNARY_INT2 is not supported");
        }

        if (total_elements == 0)
        {
            return storage;
        }

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        if (data_type == Data_Type::FLOAT16)
        {
            return effective_self->storage;
        }

        if (contig_self != nullptr)
        {
            auto view_fp16 = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, Data_Type::FLOAT16);
            struct Cast_Push_Constants
            {
                uint32_t total_elements = 0;
            } pc{static_cast<uint32_t>(total_elements)};

            pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16,
                        {effective_self->storage, view_fp16},
                        pc,
                        (pc.total_elements + 255) / 256);
            return view_fp16;
        }

        auto cache = effective_self->storage->ensureFp16ShadowCache();
        std::shared_ptr<gpu::vector> out_fp16;
        bool needs_cast = false;
        {
            std::lock_guard<std::mutex> lock(cache->mutex);
            const uint64_t cur_ver = cache->storage_version.load(std::memory_order_acquire);
            if (!cache->fp16_storage || cache->fp16_storage->getSize() < total_elements)
            {
                cache->fp16_storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, Data_Type::FLOAT16);
                cache->cached_version = 0;
                cache->valid_elements = 0;
            }
            if (cache->cached_version != cur_ver || cache->valid_elements < total_elements)
            {
                needs_cast = true;
                cache->cached_version = cur_ver;
                cache->valid_elements = total_elements;
            }
            out_fp16 = cache->fp16_storage;
        }

        if (needs_cast)
        {
            struct Cast_Push_Constants
            {
                uint32_t total_elements = 0;
            } pc{static_cast<uint32_t>(total_elements)};

            pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16,
                        {effective_self->storage, out_fp16},
                        pc,
                        (pc.total_elements + 255) / 256);
        }

        return out_fp16;
    }

std::shared_ptr<Gpu_Tensor_Impl> Gpu_Tensor_Impl::ensureContiguousSelf() const
{
        if (isContiguous() && byte_offset == 0)
        {
            return nullptr;
        }
        auto contiguous_tensor = std::make_shared<Gpu_Tensor_Impl>(shape, data_type);
        contiguous(*contiguous_tensor);
        return contiguous_tensor;
    }

Gpu_Tensor_Impl::Gpu_Tensor_Impl(size_t rows, size_t columns)
{
        updateShapeAndStrides(Shape{rows, columns});
        if (total_elements > 0)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements);
        }
    }

Gpu_Tensor_Impl::Gpu_Tensor_Impl(size_t rows, size_t columns, const std::vector<float> &host_data)
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

Gpu_Tensor_Impl::Gpu_Tensor_Impl(Shape tensor_shape, Data_Type type)
{
        data_type = type;
        updateShapeAndStrides(tensor_shape);
        if (total_elements > 0)
        {
            if (data_type == Data_Type::TERNARY_INT2)
            {
                size_t bytes = getTensorStorageBytes(data_type, shape);
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
                storage->allocateMemoryByBytes(bytes, data_type, total_elements);
            }
            else
            {
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
            }
        }
    }

Gpu_Tensor_Impl::Gpu_Tensor_Impl(Shape tensor_shape, const std::vector<float> &host_data, Data_Type type)
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
            if (data_type == Data_Type::FLOAT32)
            {
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), host_data);
            }
            else if (data_type == Data_Type::TERNARY_INT2)
            {
                size_t req_bytes = getTensorStorageBytes(data_type, shape);
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
                storage->allocateMemoryByBytes(req_bytes, data_type, total_elements);
                size_t rank = shape.getRank();
                size_t cols = (rank == 0) ? 1 : shape[rank - 1];
                size_t rows = (rank <= 1 || cols == 0) ? 1 : (total_elements / cols);
                std::vector<uint32_t> packed(req_bytes / sizeof(uint32_t), 0u);
                Cpu_Tensor_Impl::packTernary(host_data.data(), packed.data(), rows, cols);
                storage->uploadData(packed);
            }
            else
            {
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
                storage->uploadData(host_data);
            }
        }
    }

Gpu_Tensor_Impl::Gpu_Tensor_Impl(Shape tensor_shape, Stride tensor_strides, std::shared_ptr<gpu::vector> existing_storage, size_t offset_bytes, Data_Type type)
{
        data_type = type;
        shape = tensor_shape;
        strides = tensor_strides;
        storage = std::move(existing_storage);
        byte_offset = offset_bytes;
        total_elements = shape.getTotalElements();
    }

void Gpu_Tensor_Impl::reshape(size_t rows, size_t columns)
{
        reshape(Shape{rows, columns});
    }

void Gpu_Tensor_Impl::reshape(Shape new_shape)
{
        size_t new_total = new_shape.getTotalElements();
        if (!isContiguous() || byte_offset != 0)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::reshape: Cannot reshape non-contiguous view directly"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::runtime_error("Cannot reshape non-contiguous GPU tensor view");
        }

        if (data_type == Data_Type::TERNARY_INT2)
        {
            size_t req_bytes = getTensorStorageBytes(data_type, new_shape);
            if (shape == new_shape && storage && storage->getBufferSizeInBytes() == req_bytes && storage->getDataType() == data_type)
            {
                return;
            }
            if (new_total == 0)
            {
                storage.reset();
                markGpuMutated();
                updateShapeAndStrides(new_shape);
                return;
            }
            if (!storage || storage.use_count() > 1 || storage->getBufferSizeInBytes() != req_bytes || storage->getDataType() != data_type)
            {
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
                storage->allocateMemoryByBytes(req_bytes, data_type, new_total);
                markGpuMutated();
            }
            updateShapeAndStrides(new_shape);
            return;
        }

        if (shape == new_shape && storage && storage->getSize() == new_total && storage->getDataType() == data_type)
        {
            return;
        }

        if (new_total == 0)
        {
            storage.reset();
            markGpuMutated();
            updateShapeAndStrides(new_shape);
            return;
        }

        if (storage.use_count() > 1 && (total_elements != new_total || storage->getDataType() != data_type))
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), new_total, data_type);
            markGpuMutated();
        }
        else if (!storage || storage->getSize() != new_total || storage->getDataType() != data_type)
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), new_total, data_type);
            markGpuMutated();
        }
        updateShapeAndStrides(new_shape);
    }

void Gpu_Tensor_Impl::permute(const std::vector<size_t> &axes_permutation, Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::permute: TERNARY_INT2 is not supported");
        }
        size_t rank = shape.getRank();
        if (axes_permutation.size() != rank)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::permute: Permutation size must match tensor rank");
        }
        std::array<bool, MAX_TENSOR_RANK> seen{};
        for (size_t ax : axes_permutation)
        {
            if (ax >= rank || seen[ax])
            {
                throw std::invalid_argument("Gpu_Tensor_Impl::permute: Invalid or duplicate axis");
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
        Data_Type src_type = data_type;
        auto src_storage = storage;
        size_t src_offset = byte_offset;
        size_t src_total = total_elements;

        auto &output_gpu = castToGpu(output);
        output_gpu.data_type = src_type;
        output_gpu.shape = new_shape;
        output_gpu.strides = new_strides;
        output_gpu.storage = std::move(src_storage);
        output_gpu.byte_offset = src_offset;
        output_gpu.total_elements = src_total;
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::slice(size_t axis, size_t start, size_t length, Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::slice: TERNARY_INT2 is not supported");
        }
        size_t rank = shape.getRank();
        if (axis >= rank)
        {
            throw std::out_of_range("Gpu_Tensor_Impl::slice: axis out of range");
        }
        size_t dim = shape[axis];
        if (start > dim || length > dim - start)
        {
            throw std::out_of_range("Gpu_Tensor_Impl::slice: slice range exceeds dimension bounds");
        }

        Shape new_shape = shape;
        new_shape[axis] = length;
        size_t add_bytes = start * strides[axis] * getDataTypeSize(data_type);
        Data_Type src_type = data_type;
        Stride src_strides = strides;
        auto src_storage = storage;
        size_t new_offset = byte_offset + add_bytes;

        auto &output_gpu = castToGpu(output);
        output_gpu.data_type = src_type;
        output_gpu.shape = new_shape;
        output_gpu.strides = src_strides;
        output_gpu.storage = std::move(src_storage);
        output_gpu.byte_offset = new_offset;
        output_gpu.total_elements = new_shape.getTotalElements();
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::updateSlice(size_t axis, size_t start, const Tensor_Impl &source)
{
        const auto &source_gpu = castToGpu(source);
        if (data_type == Data_Type::TERNARY_INT2 || source_gpu.data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: TERNARY_INT2 is not supported");
        }
        if (data_type != source_gpu.data_type)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: Data_Type mismatch");
        }
        size_t rank = shape.getRank();
        if (axis >= rank || rank != source_gpu.shape.getRank())
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: Rank or axis mismatch");
        }
        for (size_t d = 0; d < rank; ++d)
        {
            if (d != axis && shape[d] != source_gpu.shape[d])
            {
                throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: Non-axis dimensions must match");
            }
        }
        size_t src_axis_len = source_gpu.shape[axis];
        if (start > shape[axis] || src_axis_len > shape[axis] - start)
        {
            throw std::out_of_range("Gpu_Tensor_Impl::updateSlice: slice range exceeds destination dimension");
        }
        if (!isContiguous())
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::updateSlice: Destination tensor must be contiguous");
        }
        if (total_elements == 0 || source_gpu.total_elements == 0 || !storage || !source_gpu.storage)
        {
            markGpuMutated();
            return;
        }

        auto contig_src = source_gpu.ensureContiguousSelf();
        const auto *eff_src = contig_src ? contig_src.get() : &source_gpu;
        if (!Execution_Engine::getInstance().getCurrentGraph().getNodes().empty())
        {
            Execution_Engine::getInstance().executeGraph();
            Execution_Engine::getInstance().waitIdle();
        }

        size_t elem_size = getDataTypeSize(data_type);
        size_t outer_count = 1;
        for (size_t i = 0; i < axis; ++i) outer_count *= shape[i];
        size_t inner_count = 1;
        for (size_t i = axis + 1; i < rank; ++i) inner_count *= shape[i];
        size_t this_axis_len = shape[axis];
        size_t copy_bytes = src_axis_len * inner_count * elem_size;

        auto &context = Execution_Engine::getInstance().getContext();
        uint32_t current_frame = context.getCurrentFrame();
        eff_src->storage->markAsUsedInFrame(current_frame);
        storage->markAsUsedInFrame(current_frame);
        if (eff_src->storage->getBuffer() == storage->getBuffer())
        {
            gpu::vector temp_stage(context);
            temp_stage.allocateMemoryByBytes(outer_count * copy_bytes, data_type, outer_count * src_axis_len * inner_count);
            temp_stage.markAsUsedInFrame(current_frame);
            for (size_t outer = 0; outer < outer_count; ++outer)
            {
                size_t src_off = eff_src->byte_offset + (outer * src_axis_len) * inner_count * elem_size;
                context.copyBuffer(eff_src->storage->getBuffer(), temp_stage.getBuffer(), copy_bytes, src_off, outer * copy_bytes);
            }
            context.executePendingTransfers();
            for (size_t outer = 0; outer < outer_count; ++outer)
            {
                size_t dst_off = byte_offset + (outer * this_axis_len + start) * inner_count * elem_size;
                context.copyBuffer(temp_stage.getBuffer(), storage->getBuffer(), copy_bytes, outer * copy_bytes, dst_off);
            }
        }
        else
        {
            for (size_t outer = 0; outer < outer_count; ++outer)
            {
                size_t dst_off = byte_offset + (outer * this_axis_len + start) * inner_count * elem_size;
                size_t src_off = eff_src->byte_offset + (outer * src_axis_len) * inner_count * elem_size;
                context.copyBuffer(eff_src->storage->getBuffer(), storage->getBuffer(), copy_bytes, src_off, dst_off);
            }
        }
        context.executePendingTransfers();
        markGpuMutated();
    }

std::shared_ptr<gpu::vector> Gpu_Tensor_Impl::getOrUploadIndicesBuffer(const std::vector<int32_t>& indices) const
{
    size_t count = indices.size();
    if (count == 0)
    {
        return std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
    }
    std::vector<float> idx_float(count);
    for (size_t i = 0; i < count; ++i)
    {
        idx_float[i] = static_cast<float>(indices[i]);
    }

    auto buf = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), count, Data_Type::FLOAT32);
    buf->uploadData(idx_float);
    return buf;
}

void Gpu_Tensor_Impl::gatherRows(const std::vector<int32_t> &indices, Tensor_Impl &output) const
{
    if (data_type == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Gpu_Tensor_Impl::gatherRows: TERNARY_INT2 is not supported");
    }
    if (shape.getRank() < 1)
    {
        throw std::invalid_argument("Gpu_Tensor_Impl::gatherRows: Tensor rank must be at least 1");
    }
    auto contig_self = ensureContiguousSelf();
    const auto *effective_self = contig_self ? contig_self.get() : this;

    const size_t S = indices.size();
    const size_t V = effective_self->shape[0];
    const size_t D = (effective_self->shape.getRank() == 1) ? 1 : effective_self->getColumns();
    Execution_Engine &engine = Execution_Engine::getInstance();
    const bool is_fp16 = engine.getContext().isFloat16Enabled() &&
                         (effective_self->data_type == Data_Type::FLOAT16 || engine.isCooperativeMatrixEnabled());
    const Data_Type out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    auto buf_idx = (S > 0 && D > 0 && V > 0) ? getOrUploadIndicesBuffer(indices) : std::shared_ptr<gpu::vector>{};
    auto table_storage = (S > 0 && D > 0 && V > 0)
                             ? (is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage)
                             : std::shared_ptr<gpu::vector>{};

    auto &out_gpu = castToGpu(output);
    out_gpu.setDataType(out_dtype);
    out_gpu.reshape(Shape{S, D});
    if (S == 0 || D == 0 || V == 0)
    {
        if (S * D > 0)
        {
            out_gpu.fill(0.0f);
        }
        return;
    }

    if (out_gpu.storage.get() == table_storage.get() || out_gpu.storage.get() == buf_idx.get())
    {
        out_gpu.storage = std::make_shared<gpu::vector>(engine.getContext(), S * D, out_dtype);
    }
    struct Embedding_Constants
    {
        uint32_t num_tokens;
        uint32_t embedding_dim;
        uint32_t vocab_size;
        int32_t padding_idx;
    } pcs{static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V), -1};
    const auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((static_cast<uint64_t>(S) * D + 255ULL) / 256ULL);
    pushToGraph(is_fp16 ? Compute_Pipeline::EMBEDDING_FORWARD_FP16 : Compute_Pipeline::EMBEDDING_FORWARD,
                {buf_idx, table_storage, out_gpu.storage}, pcs, wg_x, wg_y, 1, &engine);
    out_gpu.clearHostCache();
    out_gpu.invalidateFp16Cache();
}

void Gpu_Tensor_Impl::contiguous(Tensor_Impl &output) const
{
        auto src_storage = storage;
        Shape src_shape = shape;
        Stride src_strides = strides;
        size_t src_offset = byte_offset;
        size_t src_total = total_elements;
        Data_Type src_type = data_type;
        bool src_contig = isContiguous();

        auto &output_gpu = castToGpu(output);
        if (src_type == Data_Type::TERNARY_INT2)
        {
            if (!src_contig || src_offset != 0)
            {
                throw std::invalid_argument("Gpu_Tensor_Impl::contiguous: Non-contiguous TERNARY_INT2 view is not supported");
            }
            if (&output == this)
            {
                output_gpu.markGpuMutated();
                return;
            }
        }
        if (&output == this || (src_storage && output_gpu.storage.get() == src_storage.get()) ||
            !output_gpu.isContiguous() || output_gpu.byte_offset != 0)
        {
            output_gpu.byte_offset = 0;
            output_gpu.strides = output_gpu.shape.computeContiguousStrides();
            output_gpu.storage.reset();
        }
        output_gpu.setDataType(src_type);
        output_gpu.reshape(src_shape);

        if (src_type == Data_Type::TERNARY_INT2)
        {
            if (src_total == 0 || !src_storage || !output_gpu.storage)
            {
                output_gpu.markGpuMutated();
                return;
            }
            if (!Execution_Engine::getInstance().getCurrentGraph().getNodes().empty())
            {
                Execution_Engine::getInstance().executeGraph();
                Execution_Engine::getInstance().waitIdle();
            }
            auto &context = Execution_Engine::getInstance().getContext();
            uint32_t current_frame = context.getCurrentFrame();
            src_storage->markAsUsedInFrame(current_frame);
            output_gpu.storage->markAsUsedInFrame(current_frame);
            context.copyBuffer(
                src_storage->getBuffer(),
                output_gpu.storage->getBuffer(),
                getTensorStorageBytes(src_type, src_shape),
                0,
                0);
            context.executePendingTransfers();
            output_gpu.markGpuMutated();
            return;
        }

        if (src_total == 0 || !src_storage || !output_gpu.storage)
        {
            output_gpu.markGpuMutated();
            return;
        }

        Contiguous_Push_Constants constants{
            .total_elements = static_cast<uint32_t>(src_total),
            .rank = static_cast<uint32_t>(src_shape.getRank()),
            .offset_elements = static_cast<uint32_t>(src_offset / getDataTypeSize(src_type))};

        for (size_t i = 0; i < src_shape.getRank() && i < 6; ++i)
        {
            if (i == 0)
            {
                constants.shape_0 = static_cast<uint32_t>(src_shape[0]);
                constants.stride_0 = static_cast<uint32_t>(src_strides[0]);
            }
            if (i == 1)
            {
                constants.shape_1 = static_cast<uint32_t>(src_shape[1]);
                constants.stride_1 = static_cast<uint32_t>(src_strides[1]);
            }
            if (i == 2)
            {
                constants.shape_2 = static_cast<uint32_t>(src_shape[2]);
                constants.stride_2 = static_cast<uint32_t>(src_strides[2]);
            }
            if (i == 3)
            {
                constants.shape_3 = static_cast<uint32_t>(src_shape[3]);
                constants.stride_3 = static_cast<uint32_t>(src_strides[3]);
            }
            if (i == 4)
            {
                constants.shape_4 = static_cast<uint32_t>(src_shape[4]);
                constants.stride_4 = static_cast<uint32_t>(src_strides[4]);
            }
            if (i == 5)
            {
                constants.shape_5 = static_cast<uint32_t>(src_shape[5]);
                constants.stride_5 = static_cast<uint32_t>(src_strides[5]);
            }
        }

        Compute_Pipeline pipe = (src_type == Data_Type::FLOAT16) ? Compute_Pipeline::CONTIGUOUS_FP16 : Compute_Pipeline::CONTIGUOUS;
        pushToGraph(pipe, {src_storage, output_gpu.storage}, constants, (constants.total_elements + 255) / 256);
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::to(Data_Type target_type, Tensor_Impl &output) const
{
        Data_Type src_type = data_type;
        Shape src_shape = shape;
        size_t src_total = total_elements;
        auto &output_gpu = castToGpu(output);

        if (target_type == Data_Type::TERNARY_INT2 || src_type == Data_Type::TERNARY_INT2)
        {
            if (target_type == src_type)
            {
                contiguous(output);
                return;
            }
            std::vector<float> vals = getData();
            output_gpu.byte_offset = 0;
            output_gpu.strides = src_shape.computeContiguousStrides();
            output_gpu.setDataType(target_type);
            output_gpu.reshape(src_shape);
            output_gpu.uploadData(vals);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
            return;
        }

        if (src_total == 0 || !storage)
        {
            output_gpu.byte_offset = 0;
            output_gpu.strides = src_shape.computeContiguousStrides();
            output_gpu.setDataType(target_type);
            output_gpu.reshape(src_shape);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
            return;
        }

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;
        auto src_buf = effective_self->storage;

        if (target_type == src_type)
        {
            if (&output == this && isContiguous() && byte_offset == 0)
            {
                return;
            }
            effective_self->contiguous(output);
            return;
        }

        if (&output == this || (src_buf && output_gpu.storage.get() == src_buf.get()) ||
            !output_gpu.isContiguous() || output_gpu.byte_offset != 0)
        {
            output_gpu.byte_offset = 0;
            output_gpu.strides = output_gpu.shape.computeContiguousStrides();
            output_gpu.storage.reset();
        }
        output_gpu.setDataType(target_type);
        output_gpu.reshape(src_shape);

        struct Cast_Push_Constants
        {
            uint32_t total_elements = 0;
        } pc{static_cast<uint32_t>(src_total)};

        if (src_type == Data_Type::FLOAT32 && target_type == Data_Type::FLOAT16)
        {
            pushToGraph(Compute_Pipeline::CAST_FP32_TO_FP16,
                        {src_buf, output_gpu.storage},
                        pc,
                        (pc.total_elements + 255) / 256);
        }
        else if (src_type == Data_Type::FLOAT16 && target_type == Data_Type::FLOAT32)
        {
            pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32,
                        {src_buf, output_gpu.storage},
                        pc,
                        (pc.total_elements + 255) / 256);
        }
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

namespace
{
struct Gpu_Matmul_Broadcast_Plan
{
    size_t m_dim = 0;
    size_t k_dim = 0;
    size_t n_dim = 0;
    size_t batch_total = 1;
    size_t batch_rank = 0;
    bool broadcast_b_flag = false;
    Shape out_shape;
    std::array<uint32_t, 4> batch_dims{1, 1, 1, 1};
    std::array<uint32_t, 4> a_batch_strides{0, 0, 0, 0};
    std::array<uint32_t, 4> b_batch_strides{0, 0, 0, 0};
};

Gpu_Matmul_Broadcast_Plan planGpuMatmulBroadcast(const Shape &sa, size_t rows_a, size_t cols_a,
                                                 const Shape &sb, size_t rows_b, size_t cols_b)
{
    size_t rank_a = sa.getRank();
    size_t rank_b = sb.getRank();

    Gpu_Matmul_Broadcast_Plan plan;
    plan.m_dim = (rank_a >= 2) ? sa[rank_a - 2] : rows_a;
    plan.k_dim = (rank_a >= 2) ? sa[rank_a - 1] : cols_a;
    size_t k_other = (rank_b >= 2) ? sb[rank_b - 2] : rows_b;
    plan.n_dim = (rank_b >= 2) ? sb[rank_b - 1] : cols_b;

    if (plan.k_dim != k_other)
    {
        throw std::invalid_argument("Tensor inner dimensions must match for multiplication");
    }

    if (rank_a <= 2 && rank_b <= 2)
    {
        plan.batch_total = 1;
        plan.batch_rank = 0;
        plan.out_shape = Shape{plan.m_dim, plan.n_dim};
        return plan;
    }

    size_t b_rank_a = (rank_a >= 3) ? (rank_a - 2) : 0;
    size_t b_rank_b = (rank_b >= 3) ? (rank_b - 2) : 0;
    size_t batch_rank = std::max(b_rank_a, b_rank_b);
    if (batch_rank > 4)
    {
        throw std::invalid_argument("Gpu_Tensor_Impl: Matmul batch rank exceeds MAX_TENSOR_RANK - 2");
    }

    std::array<size_t, MAX_TENSOR_RANK> out_dims{};
    std::array<size_t, 4> a_padded{1, 1, 1, 1};
    std::array<size_t, 4> b_padded{1, 1, 1, 1};
    size_t batch_total = 1;
    bool any_empty_batch = false;

    for (size_t i = 0; i < batch_rank; ++i)
    {
        size_t a_dim = (i >= batch_rank - b_rank_a) ? sa[i - (batch_rank - b_rank_a)] : 1;
        size_t b_dim = (i >= batch_rank - b_rank_b) ? sb[i - (batch_rank - b_rank_b)] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
        {
            throw std::invalid_argument("Batch dimensions must match or be broadcastable");
        }
        size_t out_d = (a_dim == 0 || b_dim == 0) ? 0 : std::max(a_dim, b_dim);
        a_padded[i] = a_dim;
        b_padded[i] = b_dim;
        out_dims[i] = out_d;
        plan.batch_dims[i] = static_cast<uint32_t>(out_d);
        if (out_d == 0)
        {
            any_empty_batch = true;
        }
        else
        {
            batch_total *= out_d;
        }
    }
    if (any_empty_batch)
    {
        batch_total = 0;
    }

    out_dims[batch_rank] = plan.m_dim;
    out_dims[batch_rank + 1] = plan.n_dim;
    plan.out_shape = Shape(std::span<const size_t>(out_dims.data(), batch_rank + 2));
    plan.batch_total = batch_total;
    plan.batch_rank = batch_rank;

    size_t stride_a = 1;
    size_t stride_b = 1;
    bool b_is_broadcast = false;
    bool a_is_broadcast = false;
    for (size_t k = batch_rank; k > 0; --k)
    {
        size_t d = k - 1;
        if (a_padded[d] == 1 && plan.batch_dims[d] != 1)
        {
            plan.a_batch_strides[d] = 0;
            a_is_broadcast = true;
        }
        else
        {
            plan.a_batch_strides[d] = static_cast<uint32_t>(stride_a);
        }
        stride_a *= a_padded[d];

        if (b_padded[d] == 1 && plan.batch_dims[d] != 1)
        {
            plan.b_batch_strides[d] = 0;
            b_is_broadcast = true;
        }
        else
        {
            plan.b_batch_strides[d] = static_cast<uint32_t>(stride_b);
        }
        stride_b *= b_padded[d];
    }

    plan.broadcast_b_flag = b_is_broadcast && !a_is_broadcast && (stride_b == 1);
    return plan;
}
}

void Gpu_Tensor_Impl::matmul(const Tensor_Impl &other, Tensor_Impl &output) const
{
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &other_gpu = castToGpu(other);
        auto contig_other = other_gpu.ensureContiguousSelf();
        const auto *effective_other = contig_other ? contig_other.get() : &other_gpu;

        const Gpu_Matmul_Broadcast_Plan plan = planGpuMatmulBroadcast(
            shape, getRows(), getColumns(), other.getShape(), other.getRows(), other.getColumns());

        auto &output_gpu = castToGpu(output);
        bool use_fp16 = (data_type == Data_Type::FLOAT16 ||
                         other_gpu.getDataType() == Data_Type::FLOAT16 ||
                         Execution_Engine::getInstance().isCooperativeMatrixEnabled());

        if (plan.batch_total == 0 || plan.m_dim == 0 || plan.n_dim == 0)
        {
            output_gpu.setDataType(use_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            output_gpu.reshape(plan.out_shape);
            output_gpu.markGpuMutated();
            return;
        }
        if (plan.k_dim == 0)
        {
            output_gpu.setDataType(use_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            output_gpu.reshape(plan.out_shape);
            output_gpu.zero();
            output_gpu.markGpuMutated();
            return;
        }

        Matrix_Dimensions dims{
            .batch_count = static_cast<uint32_t>(plan.batch_total),
            .rows_a = static_cast<uint32_t>(plan.m_dim),
            .columns_a = static_cast<uint32_t>(plan.k_dim),
            .columns_b = static_cast<uint32_t>(plan.n_dim),
            .broadcast_b = plan.broadcast_b_flag ? 1u : 0u,
            .batch_rank = (plan.batch_total <= 1) ? 0u : static_cast<uint32_t>(plan.batch_rank),
            .pad0 = 0,
            .pad1 = 0,
            .batch_dims = {plan.batch_dims[0], plan.batch_dims[1], plan.batch_dims[2], plan.batch_dims[3]},
            .a_batch_strides = {plan.a_batch_strides[0], plan.a_batch_strides[1], plan.a_batch_strides[2], plan.a_batch_strides[3]},
            .b_batch_strides = {plan.b_batch_strides[0], plan.b_batch_strides[1], plan.b_batch_strides[2], plan.b_batch_strides[3]}};

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::matmul: batch={}, rows_a={}, cols_a={}, cols_b={}",
                                        dims.batch_count, dims.rows_a, dims.columns_a, dims.columns_b},
                           Log_Level::LOG_DEBUG, true, 1, Log_Feature::DENSE_COMPUTE);

        if (use_fp16)
        {
            auto buf_a = effective_self->getEffectiveFp16Storage();
            auto buf_b = effective_other->getEffectiveFp16Storage();

            output_gpu.setDataType(Data_Type::FLOAT16);
            output_gpu.reshape(plan.out_shape);
            if (output_gpu.storage.get() == buf_a.get() || output_gpu.storage.get() == buf_b.get())
            {
                output_gpu.storage = std::make_shared<gpu::vector>(
                    Execution_Engine::getInstance().getContext(), plan.out_shape.getTotalElements(), Data_Type::FLOAT16);
            }

            bool can_use_coop = Execution_Engine::getInstance().isCooperativeMatrixEnabled() &&
                                (plan.k_dim % 8 == 0) && (plan.n_dim % 8 == 0);
            Compute_Pipeline pipeline = can_use_coop
                ? Compute_Pipeline::MATMUL_COOPMAT_FP16
                : Compute_Pipeline::MATMUL_FP16;
            uint32_t tile = can_use_coop ? 64u : 16u;

            pushToGraph(pipeline,
                        {buf_a, buf_b, output_gpu.storage},
                        dims, (dims.columns_b + tile - 1u) / tile, (dims.rows_a + tile - 1u) / tile, dims.batch_count);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
        }
        else
        {
            auto buf_a = effective_self->storage;
            auto buf_b = effective_other->storage;

            output_gpu.setDataType(Data_Type::FLOAT32);
            output_gpu.reshape(plan.out_shape);
            if (output_gpu.storage.get() == buf_a.get() || output_gpu.storage.get() == buf_b.get())
            {
                output_gpu.storage = std::make_shared<gpu::vector>(
                    Execution_Engine::getInstance().getContext(), plan.out_shape.getTotalElements(), Data_Type::FLOAT32);
            }

            pushToGraph(Compute_Pipeline::MATMUL,
                        {buf_a, buf_b, output_gpu.storage},
                        dims, (dims.columns_b + 15) / 16, (dims.rows_a + 15) / 16, dims.batch_count);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
        }
    }

void Gpu_Tensor_Impl::matdiv(const Tensor_Impl &other, Tensor_Impl &output) const
{
        Gpu_Tensor_Impl temp_inv(0, 0);
        other.inverse(temp_inv);
        matmul(temp_inv, output);
    }

void Gpu_Tensor_Impl::add(const Tensor_Impl &other, Tensor_Impl &output) const
{
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::ADD_FP16 : Compute_Pipeline::ADD;
        executeElementwise(other, output, pipe, true);
    }

void Gpu_Tensor_Impl::sub(const Tensor_Impl &other, Tensor_Impl &output) const
{
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SUB_FP16 : Compute_Pipeline::SUB;
        executeElementwise(other, output, pipe, true);
    }

void Gpu_Tensor_Impl::mulScalar(float scalar, Tensor_Impl &output) const
{
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(data_type);
        output_gpu.reshape(shape);

        if (data_type == Data_Type::FLOAT16)
        {
            struct Scalar_Fp16_Constants
            {
                uint32_t total_elements;
                float scalar;
                uint32_t preserve_nonfinite;
            } fp16_constants{static_cast<uint32_t>(total_elements), scalar, 0u};
            pushToGraph(Compute_Pipeline::MUL_SCALAR_FP16, {effective_self->storage, output_gpu.storage}, fp16_constants, (total_elements + 255) / 256);
        }
        else
        {
            struct Scalar_Constants
            {
                uint32_t total_elements;
                float scalar;
            } constants{static_cast<uint32_t>(total_elements), scalar};
            pushToGraph(Compute_Pipeline::MUL_SCALAR, {effective_self->storage, output_gpu.storage}, constants, (total_elements + 255) / 256);
        }
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::divScalar(float scalar, Tensor_Impl &output) const
{
        if (scalar == 0.0F || !std::isfinite(scalar) || !std::isfinite(1.0F / scalar))
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::divScalar: Division by zero"},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Division by zero in divScalar");
        }
        mulScalar(1.0F / scalar, output);
    }

void Gpu_Tensor_Impl::hadamardMul(const Tensor_Impl &other, Tensor_Impl &output) const
{
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::HADAMARD_MUL_FP16 : Compute_Pipeline::HADAMARD_MUL;
        executeElementwise(other, output, pipe, true);
    }

void Gpu_Tensor_Impl::hadamardDiv(const Tensor_Impl &other, Tensor_Impl &output) const
{
        bool is_fp16 = (data_type == Data_Type::FLOAT16 || other.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::HADAMARD_DIV_FP16 : Compute_Pipeline::HADAMARD_DIV;
        executeElementwise(other, output, pipe, true);
    }

void Gpu_Tensor_Impl::transpose(Tensor_Impl &output) const
{
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const size_t orig_rows = getRows();
        const size_t orig_cols = getColumns();
        const Data_Type src_dtype = data_type;
        const bool is_fp16 = (src_dtype == Data_Type::FLOAT16);
        auto buf_self = (orig_rows > 0 && orig_cols > 0)
                            ? (is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage)
                            : std::shared_ptr<gpu::vector>{};

        auto &output_gpu = castToGpu(output);
        output_gpu.setDataType(src_dtype);
        output_gpu.reshape(orig_cols, orig_rows);

        if (orig_rows == 0 || orig_cols == 0)
        {
            output_gpu.markGpuMutated();
            return;
        }

        if (output_gpu.storage.get() == buf_self.get())
        {
            output_gpu.storage = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), orig_rows * orig_cols, src_dtype);
        }

        Transpose_Dimensions dims{
            .rows = static_cast<uint32_t>(orig_rows),
            .columns = static_cast<uint32_t>(orig_cols)};

        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::TRANSPOSE_FP16 : Compute_Pipeline::TRANSPOSE;

        pushToGraph(pipe, {buf_self, output_gpu.storage}, dims,
                    (dims.columns + 15) / 16, (dims.rows + 15) / 16);
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::inverse(Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::inverse: TERNARY_INT2 is not supported");
        }
        validateSquare();
        uint32_t dim_size = static_cast<uint32_t>(getRows());
        if (dim_size > 128)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::inverse: Matrix dimension exceeds shader limit of 128");
        }
        auto &output_gpu = castToGpu(output);
        if (dim_size == 0)
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
            output_gpu.reshape(0, 0);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
            return;
        }
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        std::shared_ptr<gpu::vector> in_buf = effective_self->storage;
        if (effective_self->data_type == Data_Type::FLOAT16)
        {
            auto &context = Execution_Engine::getInstance().getContext();
            auto fp32_in = std::make_shared<gpu::vector>(context, total_elements, Data_Type::FLOAT32);
            struct Cast_Push_Constants
            {
                uint32_t total_elements = 0;
            } cast_pc{static_cast<uint32_t>(total_elements)};
            auto [wg_x, wg_y] = context.compute2DWorkgroupGrid((total_elements + 255) / 256);
            pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {in_buf, fp32_in}, cast_pc, wg_x, wg_y);
            in_buf = std::move(fp32_in);
        }

        output_gpu.setDataType(Data_Type::FLOAT32);
        output_gpu.reshape(dim_size, dim_size);

        struct Inverse_Constants
        {
            uint32_t dimension_size;
        } constants{dim_size};

        pushToGraph(Compute_Pipeline::MATRIX_INVERSE, {in_buf, output_gpu.storage}, constants, 1, 1, 1);
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::normalize(Tensor_Impl &output) const
{
        if (data_type == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::normalize: TERNARY_INT2 is not supported");
        }
        auto &output_gpu = castToGpu(output);
        if (total_elements == 0)
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
            output_gpu.reshape(shape);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
            return;
        }
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        std::shared_ptr<gpu::vector> in_buf = effective_self->storage;
        if (effective_self->data_type == Data_Type::FLOAT16)
        {
            auto &context = Execution_Engine::getInstance().getContext();
            auto fp32_in = std::make_shared<gpu::vector>(context, total_elements, Data_Type::FLOAT32);
            struct Cast_Push_Constants
            {
                uint32_t total_elements = 0;
            } cast_pc{static_cast<uint32_t>(total_elements)};
            auto [wg_x, wg_y] = context.compute2DWorkgroupGrid((total_elements + 255) / 256);
            pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {in_buf, fp32_in}, cast_pc, wg_x, wg_y);
            in_buf = std::move(fp32_in);
        }

        output_gpu.setDataType(Data_Type::FLOAT32);
        output_gpu.reshape(shape);

        struct Norm_Constants
        {
            uint32_t total_elements;
        } constants{static_cast<uint32_t>(total_elements)};

        pushToGraph(Compute_Pipeline::NORMALIZE, {in_buf, output_gpu.storage}, constants, 1, 1, 1);
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::matmulAdd(const Tensor_Impl &weights, const Tensor_Impl &biases, Tensor_Impl &output) const
{
        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const auto &w_gpu = castToGpu(weights);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        const auto &b_gpu = castToGpu(biases);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        const Gpu_Matmul_Broadcast_Plan plan = planGpuMatmulBroadcast(
            shape, getRows(), getColumns(), weights.getShape(), weights.getRows(), weights.getColumns());

        auto &output_gpu = castToGpu(output);

        bool use_fp16 = (data_type == Data_Type::FLOAT16 ||
                         w_gpu.getDataType() == Data_Type::FLOAT16 ||
                         Execution_Engine::getInstance().isCooperativeMatrixEnabled());

        const size_t b_dim = plan.batch_total;
        const size_t m_dim = plan.m_dim;
        const size_t k_dim = plan.k_dim;
        const size_t n_dim = plan.n_dim;
        size_t b_total_elems = biases.getTotalElements();
        uint32_t broadcast_b_flag = 0;
        if (b_dim == 0 || m_dim == 0 || n_dim == 0)
        {
            if (b_total_elems != n_dim && b_total_elems != 0 && b_total_elems != m_dim * n_dim && b_total_elems != b_dim * m_dim * n_dim)
            {
                throw std::invalid_argument("Gpu_Tensor_Impl::matmulAdd: Bias size does not match output dimensions");
            }
            output_gpu.setDataType(use_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            output_gpu.reshape(plan.out_shape);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
            return;
        }
        if (b_total_elems == n_dim)
        {
            broadcast_b_flag = 1;
        }
        else if (b_total_elems == m_dim * n_dim)
        {
            broadcast_b_flag = (b_dim > 1) ? 2 : 0;
        }
        else if (b_total_elems == b_dim * m_dim * n_dim)
        {
            broadcast_b_flag = 0;
        }
        else
        {
            throw std::invalid_argument("Gpu_Tensor_Impl::matmulAdd: Bias size does not match output dimensions");
        }

        struct Matmul_Add_Constants
        {
            uint32_t batch_count;
            uint32_t rows_x;
            uint32_t columns_x;
            uint32_t columns_weights;
            uint32_t broadcast_w;
            uint32_t broadcast_b;
            uint32_t batch_rank;
            uint32_t pad0 = 0;
            uint32_t batch_dims[4];
            uint32_t a_batch_strides[4];
            uint32_t b_batch_strides[4];
        } constants{
            .batch_count = static_cast<uint32_t>(b_dim),
            .rows_x = static_cast<uint32_t>(m_dim),
            .columns_x = static_cast<uint32_t>(k_dim),
            .columns_weights = static_cast<uint32_t>(n_dim),
            .broadcast_w = plan.broadcast_b_flag ? 1u : 0u,
            .broadcast_b = broadcast_b_flag,
            .batch_rank = (plan.batch_total <= 1) ? 0u : static_cast<uint32_t>(plan.batch_rank),
            .pad0 = 0,
            .batch_dims = {plan.batch_dims[0], plan.batch_dims[1], plan.batch_dims[2], plan.batch_dims[3]},
            .a_batch_strides = {plan.a_batch_strides[0], plan.a_batch_strides[1], plan.a_batch_strides[2], plan.a_batch_strides[3]},
            .b_batch_strides = {plan.b_batch_strides[0], plan.b_batch_strides[1], plan.b_batch_strides[2], plan.b_batch_strides[3]}};
        static_assert(offsetof(Matmul_Add_Constants, batch_dims) == 32);
        static_assert(sizeof(Matmul_Add_Constants) == 80);

        Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::matmulAdd: batch={}, rows_x={}, cols_x={}, cols_w={}",
                                        constants.batch_count, constants.rows_x, constants.columns_x, constants.columns_weights},
                           Log_Level::LOG_DEBUG, true, 1, Log_Feature::DENSE_COMPUTE);

        if (use_fp16)
        {
            auto buf_self = effective_self->getEffectiveFp16Storage();
            auto buf_w = effective_w->getEffectiveFp16Storage();
            auto buf_b = effective_b->getEffectiveFp16Storage();

            output_gpu.setDataType(Data_Type::FLOAT16);
            output_gpu.reshape(plan.out_shape);
            if (output_gpu.storage.get() == buf_self.get() ||
                output_gpu.storage.get() == buf_w.get() ||
                output_gpu.storage.get() == buf_b.get())
            {
                output_gpu.storage = std::make_shared<gpu::vector>(
                    Execution_Engine::getInstance().getContext(), plan.out_shape.getTotalElements(), Data_Type::FLOAT16);
            }

            bool can_use_coop = Execution_Engine::getInstance().isCooperativeMatrixEnabled();
            Compute_Pipeline pipeline = can_use_coop
                ? Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16
                : Compute_Pipeline::MATMUL_ADD_FP16;
            uint32_t tile = can_use_coop ? 64u : 16u;

            pushToGraph(pipeline,
                        {buf_self, buf_w, buf_b, output_gpu.storage}, constants,
                        (constants.columns_weights + tile - 1u) / tile, (constants.rows_x + tile - 1u) / tile, constants.batch_count);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
        }
        else
        {
            auto buf_self = effective_self->storage;
            auto buf_w = effective_w->storage;
            auto buf_b = effective_b->storage;

            output_gpu.setDataType(Data_Type::FLOAT32);
            output_gpu.reshape(plan.out_shape);
            if (output_gpu.storage.get() == buf_self.get() ||
                output_gpu.storage.get() == buf_w.get() ||
                output_gpu.storage.get() == buf_b.get())
            {
                output_gpu.storage = std::make_shared<gpu::vector>(
                    Execution_Engine::getInstance().getContext(), plan.out_shape.getTotalElements(), Data_Type::FLOAT32);
            }

            pushToGraph(Compute_Pipeline::MATMUL_ADD,
                        {buf_self, buf_w, buf_b, output_gpu.storage}, constants,
                        (constants.columns_weights + 15) / 16, (constants.rows_x + 15) / 16, constants.batch_count);
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
        }
    }

void Gpu_Tensor_Impl::uploadData(const std::vector<float> &host_data)
{
        if (host_data.size() != total_elements)
        {
            Logger::logMessage(Input_Format{"Gpu_Tensor_Impl::uploadData: Host data size mismatch (expected {}, got {})",
                                            total_elements, host_data.size()},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::invalid_argument("Host data size mismatch");
        }

        if (total_elements == 0)
        {
            markGpuMutated();
            return;
        }

        if (!isContiguous())
        {
            Logger::logMessage("Gpu_Tensor_Impl::uploadData: Cannot upload directly into non-contiguous GPU tensor view",
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::TENSOR_INSPECTION);
            throw std::runtime_error("Gpu_Tensor_Impl::uploadData: Non-contiguous GPU tensor view is not supported");
        }

        if (data_type == Data_Type::TERNARY_INT2)
        {
            if (byte_offset != 0)
            {
                throw std::invalid_argument("Gpu_Tensor_Impl::uploadData: Non-zero byte_offset TERNARY_INT2 view is not supported");
            }
            size_t req_bytes = getTensorStorageBytes(data_type, shape);
            size_t rank = shape.getRank();
            size_t cols = (rank == 0) ? 1 : shape[rank - 1];
            size_t rows = (rank <= 1 || cols == 0) ? 1 : (total_elements / cols);
            std::vector<uint32_t> packed(req_bytes / sizeof(uint32_t), 0u);
            Cpu_Tensor_Impl::packTernary(host_data.data(), packed.data(), rows, cols);
            uploadPackedData(packed);
            return;
        }

        Execution_Engine &engine = Execution_Engine::getInstance();
        if (byte_offset != 0)
        {
            if (!storage || storage->getBuffer() == VK_NULL_HANDLE)
            {
                throw std::runtime_error("Gpu_Tensor_Impl::uploadData: Null backing storage for offset GPU tensor view");
            }
            if (!engine.getCurrentGraph().getNodes().empty())
            {
                engine.executeGraph();
            }
            Gpu_Tensor_Impl temp_upload(shape, host_data, data_type);
            auto &context = engine.getContext();
            context.executePendingTransfers();
            uint32_t current_frame = context.getCurrentFrame();
            temp_upload.storage->markAsUsedInFrame(current_frame);
            storage->markAsUsedInFrame(current_frame);
            context.copyBuffer(
                temp_upload.storage->getBuffer(),
                storage->getBuffer(),
                total_elements * getDataTypeSize(data_type),
                0,
                byte_offset);
            context.executePendingTransfers();
            markGpuMutated();
            return;
        }

        if (!storage || storage->getSize() != total_elements || storage->getDataType() != data_type)
        {
            storage = std::make_shared<gpu::vector>(engine.getContext(), total_elements, data_type);
        }
        else if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        storage->uploadData(host_data);
        markGpuMutated();
    }

void Gpu_Tensor_Impl::zero()
{
        markGpuMutated();
        if (total_elements == 0)
        {
            return;
        }

        Execution_Engine &engine = Execution_Engine::getInstance();
        const bool graph_has_nodes = !engine.getCurrentGraph().getNodes().empty();

        if (graph_has_nodes &&
            isContiguous() && byte_offset == 0 &&
            (data_type == Data_Type::FLOAT32 || data_type == Data_Type::FLOAT16) &&
            total_elements <= std::numeric_limits<uint32_t>::max())
        {
            if (!storage || storage->getSize() != total_elements || storage->getDataType() != data_type)
            {
                storage = std::make_shared<gpu::vector>(engine.getContext(), total_elements, data_type);
            }
            if (!storage || storage->getBuffer() == VK_NULL_HANDLE)
            {
                return;
            }
            struct Fill_Constants
            {
                uint32_t total_elements;
                float fill_value;
            } pc{static_cast<uint32_t>(total_elements), 0.0f};
            auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total_elements + 255ull) / 256ull);
            Compute_Pipeline pipe = (data_type == Data_Type::FLOAT16) ? Compute_Pipeline::FILL_FP16 : Compute_Pipeline::FILL_FLOAT;
            pushToGraph(pipe, {storage}, pc, wg_x, wg_y, 1, &engine);
            return;
        }

        if (graph_has_nodes)
        {
            engine.executeGraph();
            engine.waitIdle();
        }

        if (data_type == Data_Type::TERNARY_INT2)
        {
            size_t req_bytes = getTensorStorageBytes(data_type, shape);
            if (!storage || storage->getBufferSizeInBytes() != req_bytes || storage->getDataType() != data_type)
            {
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
                storage->allocateMemoryByBytes(req_bytes, data_type, total_elements);
            }
            if (storage->getBuffer() == VK_NULL_HANDLE)
            {
                return;
            }
            auto &context = Execution_Engine::getInstance().getContext();
            storage->markAsUsedInFrame(context.getCurrentFrame());
            context.fillBuffer(storage->getBuffer(), req_bytes, 0, 0);
            return;
        }

        if (!storage || (isContiguous() && byte_offset == 0 && (storage->getSize() != total_elements || storage->getDataType() != data_type)))
        {
            storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
        }

        if (!storage || storage->getBuffer() == VK_NULL_HANDLE)
        {
            return;
        }

        size_t byte_size = total_elements * getDataTypeSize(data_type);
        if (isContiguous() && (byte_offset % 4 == 0) && (byte_size % 4 == 0))
        {
            auto &context = Execution_Engine::getInstance().getContext();
            storage->markAsUsedInFrame(context.getCurrentFrame());
            context.fillBuffer(storage->getBuffer(), byte_size, byte_offset, 0);
            return;
        }
        if (isContiguous())
        {
            if (byte_offset == 0 && storage->getBufferSizeInBytes() == byte_size)
            {
                std::vector<float16_t> zeros(total_elements, static_cast<float16_t>(0.0f));
                storage->uploadData(zeros);
            }
            else
            {
                auto &context = Execution_Engine::getInstance().getContext();
                gpu::vector temp_zeros(context, std::vector<float16_t>(total_elements, static_cast<float16_t>(0.0f)));
                context.executePendingTransfers();
                uint32_t current_frame = context.getCurrentFrame();
                temp_zeros.markAsUsedInFrame(current_frame);
                storage->markAsUsedInFrame(current_frame);
                context.copyBuffer(
                    temp_zeros.getBuffer(),
                    storage->getBuffer(),
                    byte_size,
                    0,
                    byte_offset);
                context.executePendingTransfers();
            }
            return;
        }
        throw std::invalid_argument("Gpu_Tensor_Impl::zero: Non-contiguous GPU view zeroing is not supported");
    }

void Gpu_Tensor_Impl::fill(float value)
{
        if (total_elements == 0)
        {
            markGpuMutated();
            return;
        }
        if (value == 0.0f)
        {
            zero();
            return;
        }
        Execution_Engine &engine = Execution_Engine::getInstance();
        const bool graph_has_nodes = !engine.getCurrentGraph().getNodes().empty();
        if (graph_has_nodes &&
            isContiguous() && byte_offset == 0 &&
            total_elements <= std::numeric_limits<uint32_t>::max() &&
            (data_type == Data_Type::FLOAT32 ||
             (data_type == Data_Type::FLOAT16 && std::isfinite(value))))
        {
            markGpuMutated();
            if (!storage || storage->getSize() != total_elements || storage->getDataType() != data_type)
            {
                storage = std::make_shared<gpu::vector>(engine.getContext(), total_elements, data_type);
            }
            if (!storage || storage->getBuffer() == VK_NULL_HANDLE)
            {
                return;
            }
            struct Fill_Constants
            {
                uint32_t total_elements;
                float fill_value;
            } pc{static_cast<uint32_t>(total_elements), value};
            auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total_elements + 255ull) / 256ull);
            Compute_Pipeline pipe = (data_type == Data_Type::FLOAT16) ? Compute_Pipeline::FILL_FP16 : Compute_Pipeline::FILL_FLOAT;
            pushToGraph(pipe, {storage}, pc, wg_x, wg_y, 1, &engine);
            return;
        }
        if (isContiguous() && (byte_offset % 4 == 0) &&
            (data_type == Data_Type::FLOAT32 ||
             (data_type == Data_Type::FLOAT16 && (total_elements % 2) == 0)))
        {
            markGpuMutated();
            if (graph_has_nodes)
            {
                engine.executeGraph();
                engine.waitIdle();
            }
            if (!storage || (byte_offset == 0 && (storage->getSize() != total_elements || storage->getDataType() != data_type)))
            {
                storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext(), total_elements, data_type);
            }
            if (!storage || storage->getBuffer() == VK_NULL_HANDLE)
            {
                return;
            }
            uint32_t pattern = 0;
            if (data_type == Data_Type::FLOAT32)
            {
                pattern = std::bit_cast<uint32_t>(value);
            }
            else
            {
                const uint16_t half_bits = floatToHalfBits(value);
                pattern = (static_cast<uint32_t>(half_bits) << 16) | static_cast<uint32_t>(half_bits);
            }
            const size_t byte_size = total_elements * getDataTypeSize(data_type);
            auto &context = Execution_Engine::getInstance().getContext();
            storage->markAsUsedInFrame(context.getCurrentFrame());
            context.fillBuffer(storage->getBuffer(), byte_size, byte_offset, pattern);
            return;
        }
        std::vector<float> buffer(total_elements, value);
        uploadData(buffer);
    }

void Gpu_Tensor_Impl::concatenateColumns(const Tensor_Impl& other, Tensor_Impl& output) const
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

        uint64_t rows64 = effective_self->getRows();
        uint64_t cols_a64 = effective_self->getColumns();
        uint64_t cols_b64 = effective_other->getColumns();
        uint64_t tot_cols64 = cols_a64 + cols_b64;
        if (rows64 > std::numeric_limits<uint32_t>::max() ||
            cols_a64 > std::numeric_limits<uint32_t>::max() ||
            cols_b64 > std::numeric_limits<uint32_t>::max() ||
            tot_cols64 > std::numeric_limits<uint32_t>::max() ||
            (tot_cols64 > 0 && rows64 > std::numeric_limits<uint32_t>::max() / tot_cols64))
        {
            throw std::overflow_error("Gpu_Tensor_Impl::concatenateColumns: Tensor dimensions exceed uint32_t dispatch limit");
        }
        uint64_t total_elems64 = rows64 * tot_cols64;
        const Data_Type src_dtype = effective_self->getDataType();
        auto buf_self = effective_self->storage;
        auto buf_other = effective_other->storage;

        auto& output_gpu = castToGpu(output);
        if (output_gpu.getDataType() != src_dtype)
        {
            output_gpu.setDataType(src_dtype);
        }

        uint32_t cols_a = static_cast<uint32_t>(cols_a64);
        uint32_t cols_b = static_cast<uint32_t>(cols_b64);
        uint32_t tot_cols = static_cast<uint32_t>(tot_cols64);
        output_gpu.reshape(static_cast<size_t>(rows64), static_cast<size_t>(tot_cols));

        if (total_elems64 == 0)
        {
            output_gpu.invalidateFp16Cache();
            output_gpu.clearHostCache();
            return;
        }

        if (output_gpu.storage.get() == buf_self.get() || output_gpu.storage.get() == buf_other.get())
        {
            output_gpu.storage = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), static_cast<size_t>(total_elems64), src_dtype);
        }

        struct Constants
        {
            uint32_t rows;
            uint32_t columns_a;
            uint32_t columns_b;
        } c{ static_cast<uint32_t>(rows64), cols_a, cols_b };

        bool is_fp16 = (src_dtype == Data_Type::FLOAT16);
        Compute_Pipeline target_pipeline = is_fp16 ? Compute_Pipeline::CONCATENATE_COLUMNS_FP16 : Compute_Pipeline::CONCATENATE_COLUMNS;

        auto [wg_x, wg_y] = Execution_Engine::getInstance().getContext().compute2DWorkgroupGrid((total_elems64 + 255ULL) / 256ULL);
        pushToGraph(target_pipeline, { buf_self, buf_other, output_gpu.storage }, c,
            wg_x, wg_y, 1);
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::concatenateRows(const Tensor_Impl& other, Tensor_Impl& output) const
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

        const Data_Type src_dtype = effective_self->getDataType();
        uint32_t rows_a = static_cast<uint32_t>(effective_self->getRows());
        uint32_t rows_b = static_cast<uint32_t>(effective_other->getRows());
        uint32_t tot_rows = rows_a + rows_b;
        uint32_t cols = static_cast<uint32_t>(effective_self->getColumns());
        auto buf_self = effective_self->storage;
        auto buf_other = effective_other->storage;

        auto& output_gpu = castToGpu(output);
        if (output_gpu.getDataType() != src_dtype)
        {
            output_gpu.setDataType(src_dtype);
        }

        output_gpu.reshape(tot_rows, cols);

        const size_t total_elems = static_cast<size_t>(tot_rows) * static_cast<size_t>(cols);
        if (total_elems > 0 &&
            (output_gpu.storage.get() == buf_self.get() || output_gpu.storage.get() == buf_other.get()))
        {
            output_gpu.storage = std::make_shared<gpu::vector>(
                Execution_Engine::getInstance().getContext(), total_elems, src_dtype);
        }

        struct Constants
        {
            uint32_t rows_a;
            uint32_t rows_b;
            uint32_t columns;
        } c{ rows_a, rows_b, cols };

        bool is_fp16 = (src_dtype == Data_Type::FLOAT16);
        Compute_Pipeline target_pipeline = is_fp16
            ? Compute_Pipeline::CONCATENATE_ROWS_FP16
            : Compute_Pipeline::CONCATENATE_ROWS;

        pushToGraph(target_pipeline, { buf_self, buf_other, output_gpu.storage }, c,
            (c.columns + 15) / 16, (tot_rows + 15) / 16);
        output_gpu.invalidateFp16Cache();
        output_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::splitColumns(size_t split_index, Tensor_Impl &result_left, Tensor_Impl &result_right) const
{
        if (split_index == 0 || split_index >= getColumns())
        {
            throw std::out_of_range("Split index out of range in splitColumns");
        }

        uint64_t rows64 = getRows();
        uint64_t tot_cols64 = getColumns();
        uint64_t cols_left64 = split_index;
        uint64_t cols_right64 = tot_cols64 - cols_left64;
        if (rows64 > std::numeric_limits<uint32_t>::max() ||
            tot_cols64 > std::numeric_limits<uint32_t>::max() ||
            (tot_cols64 > 0 && rows64 > std::numeric_limits<uint32_t>::max() / tot_cols64))
        {
            throw std::overflow_error("Gpu_Tensor_Impl::splitColumns: Tensor dimensions exceed uint32_t dispatch limit");
        }
        uint64_t total_elems64 = rows64 * tot_cols64;

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const Data_Type src_dtype = data_type;
        const bool is_fp16 = (src_dtype == Data_Type::FLOAT16);
        auto buf_self = (rows64 > 0)
                            ? (is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage)
                            : std::shared_ptr<gpu::vector>{};

        auto &left_gpu = castToGpu(result_left);
        auto &right_gpu = castToGpu(result_right);
        left_gpu.setDataType(src_dtype);
        right_gpu.setDataType(src_dtype);
        uint32_t cols_left = static_cast<uint32_t>(cols_left64);
        uint32_t cols_right = static_cast<uint32_t>(cols_right64);

        left_gpu.reshape(static_cast<size_t>(rows64), cols_left);
        right_gpu.reshape(static_cast<size_t>(rows64), cols_right);

        if (rows64 == 0)
        {
            left_gpu.invalidateFp16Cache();
            left_gpu.clearHostCache();
            right_gpu.invalidateFp16Cache();
            right_gpu.clearHostCache();
            return;
        }

        auto &context = Execution_Engine::getInstance().getContext();
        if (left_gpu.storage.get() == buf_self.get())
        {
            left_gpu.storage = std::make_shared<gpu::vector>(context, static_cast<size_t>(rows64) * cols_left, src_dtype);
        }
        if (right_gpu.storage.get() == buf_self.get() || right_gpu.storage.get() == left_gpu.storage.get())
        {
            right_gpu.storage = std::make_shared<gpu::vector>(context, static_cast<size_t>(rows64) * cols_right, src_dtype);
        }

        struct Constants
        {
            uint32_t rows;
            uint32_t columns_left;
            uint32_t columns_right;
        } c{static_cast<uint32_t>(rows64), cols_left, cols_right};

        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SPLIT_COLUMNS_FP16 : Compute_Pipeline::SPLIT_COLUMNS;

        auto [wg_x, wg_y] = context.compute2DWorkgroupGrid((total_elems64 + 255ULL) / 256ULL);
        pushToGraph(pipe, {buf_self, left_gpu.storage, right_gpu.storage}, c,
                    wg_x, wg_y, 1);
        left_gpu.invalidateFp16Cache();
        left_gpu.clearHostCache();
        right_gpu.invalidateFp16Cache();
        right_gpu.clearHostCache();
    }

void Gpu_Tensor_Impl::splitRows(size_t split_index, Tensor_Impl &result_up, Tensor_Impl &result_down) const
{
        if (split_index == 0 || split_index >= getRows())
        {
            throw std::out_of_range("Split index out of range in splitRows");
        }

        auto contig_self = ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : this;

        const size_t orig_rows = getRows();
        const size_t orig_cols = getColumns();
        const Data_Type src_dtype = data_type;
        const bool is_fp16 = (src_dtype == Data_Type::FLOAT16);
        auto buf_self = (orig_cols > 0)
                            ? (is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->storage)
                            : std::shared_ptr<gpu::vector>{};

        auto &up_gpu = castToGpu(result_up);
        auto &down_gpu = castToGpu(result_down);
        up_gpu.setDataType(src_dtype);
        down_gpu.setDataType(src_dtype);
        uint32_t rows_up = static_cast<uint32_t>(split_index);
        uint32_t rows_down = static_cast<uint32_t>(orig_rows - split_index);

        up_gpu.reshape(rows_up, orig_cols);
        down_gpu.reshape(rows_down, orig_cols);

        if (orig_cols == 0)
        {
            up_gpu.invalidateFp16Cache();
            up_gpu.clearHostCache();
            down_gpu.invalidateFp16Cache();
            down_gpu.clearHostCache();
            return;
        }

        auto &context = Execution_Engine::getInstance().getContext();
        if (up_gpu.storage.get() == buf_self.get())
        {
            up_gpu.storage = std::make_shared<gpu::vector>(context, static_cast<size_t>(rows_up) * orig_cols, src_dtype);
        }
        if (down_gpu.storage.get() == buf_self.get() || down_gpu.storage.get() == up_gpu.storage.get())
        {
            down_gpu.storage = std::make_shared<gpu::vector>(context, static_cast<size_t>(rows_down) * orig_cols, src_dtype);
        }

        struct Constants
        {
            uint32_t rows_up;
            uint32_t rows_down;
            uint32_t columns;
        } c{rows_up, rows_down, static_cast<uint32_t>(orig_cols)};

        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SPLIT_ROWS_FP16 : Compute_Pipeline::SPLIT_ROWS;

        pushToGraph(pipe, {buf_self, up_gpu.storage, down_gpu.storage}, c,
                    (c.columns + 15) / 16, (static_cast<uint32_t>(orig_rows) + 15) / 16);
        up_gpu.invalidateFp16Cache();
        up_gpu.clearHostCache();
        down_gpu.invalidateFp16Cache();
        down_gpu.clearHostCache();
    }

const std::vector<float> & Gpu_Tensor_Impl::getData() const
{
        Logger::logMessage("Gpu_Tensor_Impl::getData: Reading data from GPU, risk of sync stall",
                           Log_Level::LOG_WARNING, true, 1, Log_Feature::MEMORY_TRANSFER);
        if (total_elements == 0)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            host_cache.clear();
            return host_cache;
        }

        if (!isContiguous() || byte_offset != 0)
        {
            Gpu_Tensor_Impl contig_gpu(shape, data_type);
            contiguous(contig_gpu);
            Execution_Engine::getInstance().executeGraph();
            Execution_Engine::getInstance().waitIdle();
            std::vector<float> tmp = contig_gpu.getData();
            std::lock_guard<std::mutex> lock(cache_mutex);
            host_cache = std::move(tmp);
            return host_cache;
        }

        if (data_type == Data_Type::TERNARY_INT2)
        {
            std::vector<uint32_t> packed = getPackedData();
            std::lock_guard<std::mutex> lock(cache_mutex);
            host_cache.assign(total_elements, 0.0f);
            if (!packed.empty())
            {
                size_t rank = shape.getRank();
                size_t cols = (rank == 0) ? 1 : shape[rank - 1];
                size_t rows = (rank <= 1 || cols == 0) ? 1 : (total_elements / cols);
                Cpu_Tensor_Impl::unpackTernary(packed.data(), host_cache.data(), rows, cols, 1.0f);
            }
            return host_cache;
        }

        Execution_Engine::getInstance().getContext().flush();
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (storage)
        {
            host_cache.resize(total_elements);
            storage->downloadData(host_cache);
        }
        else
        {
            host_cache.assign(total_elements, 0.0f);
        }
        return host_cache;
    }

Mutable_Storage_Handle Gpu_Tensor_Impl::getStorage()
{
        if (!isContiguous() || byte_offset != 0)
        {
            auto contiguous_tensor = std::make_shared<Gpu_Tensor_Impl>(shape, data_type);
            contiguous(*contiguous_tensor);
            storage = contiguous_tensor->storage;
            byte_offset = 0;
            strides = shape.computeContiguousStrides();
        }
        invalidateFp16Cache();
        clearHostCache();
        return storage;
    }

const std::vector<uint32_t> &Gpu_Tensor_Impl::getPackedData() const
{
    if (data_type != Data_Type::TERNARY_INT2 || total_elements == 0 || !storage)
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        host_cache_u32.clear();
        return host_cache_u32;
    }
    Execution_Engine::getInstance().getContext().flush();
    Execution_Engine::getInstance().getContext().executePendingTransfers();
    size_t words = storage->getBufferSizeInBytes() / sizeof(uint32_t);
    std::lock_guard<std::mutex> lock(cache_mutex);
    host_cache_u32.resize(words);
    storage->downloadData(host_cache_u32);
    return host_cache_u32;
}

void Gpu_Tensor_Impl::uploadPackedData(const std::vector<uint32_t> &packed_data)
{
    if (data_type != Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("uploadPackedData called on non-TERNARY_INT2 tensor");
    }
    size_t required_bytes = getTensorStorageBytes(data_type, shape);
    if (packed_data.size() * sizeof(uint32_t) != required_bytes)
    {
        throw std::invalid_argument("uploadPackedData size mismatch");
    }
    if (!storage || storage->getBufferSizeInBytes() != required_bytes)
    {
        storage = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
        storage->allocateMemoryByBytes(required_bytes, data_type, total_elements);
    }
    storage->uploadData(packed_data);
    markGpuMutated();
}

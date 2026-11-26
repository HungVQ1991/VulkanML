#include "ops/llm/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::llm
{

using ops::internal::ensureUniqueTensorTarget;
using ops::internal::pushGpuFill;
using ops::internal::resolveExecutionEngine;
using ops::internal::validatePushConstants;

static void writeCpuOutput(Tensor &out, const Shape &shape, Data_Type dtype, const std::vector<float> &vals)
{
    auto impl = out.getImplementation();
    if (!impl || impl.use_count() > 2 || impl->getByteOffset() != 0 || !impl->isContiguous() ||
        out.getExecutionTarget() != Execution_Target::CPU || out.getDataType() != dtype || out.getShape() != shape)
    {
        out = Tensor(shape, dtype, Execution_Target::CPU);
    }
    else
    {
        out.reshape(shape);
    }

    auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*out.getImplementation());
    if (dtype == Data_Type::FLOAT16)
    {
        std::vector<float> clamped(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
        {
            clamped[i] = std::clamp(vals[i], -65504.0f, 65504.0f);
        }
        out_cpu.uploadData(clamped);
    }
    else
    {
        out_cpu.setStorageBufferFp16(nullptr);
        out_cpu.setDataType(Data_Type::FLOAT32);
        out_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(vals);
    }
}

static std::shared_ptr<gpu::vector> getGpuFloat32ReadBuffer(
    const Gpu_Tensor_Impl &src,
    std::vector<std::shared_ptr<Gpu_Tensor_Impl>> &keep_impls,
    std::vector<std::shared_ptr<gpu::vector>> &keep_buffers,
    Execution_Engine &engine)
{
    auto contig = src.ensureContiguousSelf();
    const Gpu_Tensor_Impl *effective = contig ? contig.get() : &src;
    if (contig)
    {
        keep_impls.push_back(contig);
    }
    if (effective->getDataType() == Data_Type::FLOAT32)
    {
        return effective->getStorageBuffer();
    }
    if (effective->getDataType() == Data_Type::FLOAT16)
    {
        size_t total = effective->getTotalElements();
        if (total > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("ops::llm::embedding: indices element count exceeds uint32_t limit");
        }
        auto fp32_buf = std::make_shared<gpu::vector>(engine.getContext());
        fp32_buf->allocateMemory(total, Data_Type::FLOAT32);
        struct Cast_PC
        {
            uint32_t total_elements;
        } pc{ static_cast<uint32_t>(total) };
        validatePushConstants<Cast_PC>();
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid((total + 255) / 256);
        src.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32,
                        { effective->getStorageBuffer(), fp32_buf },
                        pc, wg_x, wg_y, 1, &engine);
        keep_buffers.push_back(fp32_buf);
        return fp32_buf;
    }
    throw std::invalid_argument("ops::llm::embedding: Unsupported Data_Type for GPU indices buffer");
}

static bool tryFloatToTokenIndex(float raw_idx, size_t vocab_size, int32_t padding_idx, size_t &out_idx) noexcept
{
    if (!std::isfinite(raw_idx))
    {
        return false;
    }
    double rounded = std::round(static_cast<double>(raw_idx));
    if (rounded < 0.0 || rounded >= static_cast<double>(vocab_size) ||
        rounded > static_cast<double>(std::numeric_limits<size_t>::max()))
    {
        return false;
    }
    size_t idx = static_cast<size_t>(rounded);
    if (padding_idx >= 0 && idx == static_cast<size_t>(padding_idx))
    {
        return false;
    }
    out_idx = idx;
    return true;
}

void embeddingForward(const Tensor &weights, const Tensor &indices, Tensor &output,
                      const Embedding_Config &config)
{
    auto w_impl = weights.getImplementation();
    auto idx_impl = indices.getImplementation();
    if (!w_impl || !idx_impl) return;

    if (weights.getDataType() == Data_Type::TERNARY_INT2 || indices.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::embeddingForward: TERNARY_INT2 is not supported");
    }
    if (weights.getShape().getRank() < 2)
    {
        throw std::invalid_argument("ops::llm::embeddingForward: weights must have rank >= 2");
    }

    size_t S = indices.getTotalElements();
    size_t D = weights.getColumns();
    size_t V = weights.getShape()[0];
    const bool force_fp16 = config.force_fp16;
    const int32_t padding_idx = config.padding_idx;

    Shape out_shape;
    if (indices.getShape().getRank() >= 2)
    {
        auto dim_span = indices.getShape().getDimensions();
        if (dim_span.size() >= MAX_TENSOR_RANK)
        {
            throw std::invalid_argument("ops::llm::embeddingForward: Output rank exceeds MAX_TENSOR_RANK");
        }
        std::vector<size_t> dims(dim_span.begin(), dim_span.end());
        dims.push_back(D);
        out_shape = Shape(dims);
    }
    else
    {
        out_shape = Shape{ S, D };
    }

    Execution_Target target = weights.getExecutionTarget();
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;
    bool is_fp16 = false;
    if (target != Execution_Target::CPU)
    {
        is_fp16 = resolved_engine->getContext().isFloat16Enabled() &&
                  (force_fp16 || weights.getDataType() == Data_Type::FLOAT16 || resolved_engine->isCooperativeMatrixEnabled());
    }
    Data_Type out_dtype = (target != Execution_Target::CPU)
        ? (is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
        : (force_fp16 ? Data_Type::FLOAT16 : weights.getDataType());

    ensureUniqueTensorTarget(output, target, out_shape, out_dtype);

    if (S == 0 || D == 0 || V == 0)
    {
        if (S * D > 0)
        {
            if (target == Execution_Target::CPU)
            {
                writeCpuOutput(output, out_shape, out_dtype, std::vector<float>(S * D, 0.0f));
            }
            else
            {
                auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
                pushGpuFill(out_gpu, S * D, 0.0f, *resolved_engine);
            }
        }
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto indices_data = indices.getData();
        const auto &w_cpu = Cpu_Tensor_Impl::castToCpu(*w_impl);
        const auto &w_fp16_buf = w_cpu.getStorageBufferFp16();
        const bool can_fast_fp16 = (weights.getDataType() == Data_Type::FLOAT16 &&
                                    w_cpu.isContiguous() && w_cpu.getByteOffset() == 0 &&
                                    w_fp16_buf && w_fp16_buf->size() >= V * D);

        if (can_fast_fp16 && out_dtype == Data_Type::FLOAT16)
        {
            auto out_impl = output.getImplementation();
            if (!out_impl || out_impl.use_count() > 2 || out_impl->getByteOffset() != 0 || !out_impl->isContiguous() ||
                output.getExecutionTarget() != Execution_Target::CPU || output.getDataType() != Data_Type::FLOAT16 ||
                output.getShape() != out_shape)
            {
                output = Tensor(out_shape, Data_Type::FLOAT16, Execution_Target::CPU);
            }
            else
            {
                output.reshape(out_shape);
            }
            auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
            auto fp16_out = std::make_shared<std::vector<float16_t>>(S * D, static_cast<float16_t>(0.0f));
            const float16_t *w_ptr = w_fp16_buf->data();
            for (size_t s = 0; s < S; ++s)
            {
                size_t row = 0;
                if (tryFloatToTokenIndex(indices_data[s], V, padding_idx, row))
                {
                    std::memcpy(fp16_out->data() + s * D, w_ptr + row * D, D * sizeof(float16_t));
                }
            }
            out_cpu.setStorageBufferFp16(std::move(fp16_out));
            out_cpu.setDataType(Data_Type::FLOAT16);
            out_cpu.getStorageBuffer().reset();
        }
        else if (can_fast_fp16)
        {
            std::vector<float> out_vals(S * D, 0.0f);
            const float16_t *w_ptr = w_fp16_buf->data();
            for (size_t s = 0; s < S; ++s)
            {
                size_t row = 0;
                if (tryFloatToTokenIndex(indices_data[s], V, padding_idx, row))
                {
                    convertFp16ToFp32(w_ptr + row * D, out_vals.data() + s * D, D);
                }
            }
            writeCpuOutput(output, out_shape, out_dtype, out_vals);
        }
        else
        {
            const auto weight_data = weights.getData();
            std::vector<float> out_vals(S * D, 0.0f);
            for (size_t s = 0; s < S; ++s)
            {
                size_t row = 0;
                if (tryFloatToTokenIndex(indices_data[s], V, padding_idx, row))
                {
                    std::memcpy(out_vals.data() + s * D,
                                weight_data.data() + row * D,
                                D * sizeof(float));
                }
            }
            writeCpuOutput(output, out_shape, out_dtype, out_vals);
        }
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (S > std::numeric_limits<uint32_t>::max() ||
            D > std::numeric_limits<uint32_t>::max() ||
            V > std::numeric_limits<uint32_t>::max() ||
            (D > 0 && S > std::numeric_limits<uint32_t>::max() / D))
        {
            throw std::invalid_argument("ops::llm::embeddingForward: Dimensions exceed uint32_t limit");
        }

        Tensor ind_dev = indices;
        if (indices.getExecutionTarget() != target)
        {
            ind_dev = Tensor(indices.getShape(), indices.getData(), target);
        }

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*weights.getImplementation());
        const auto &idx_gpu = Gpu_Tensor_Impl::castToGpu(*ind_dev.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        auto contig_w = w_gpu.ensureContiguousSelf();
        const Gpu_Tensor_Impl *effective_w = contig_w ? contig_w.get() : &w_gpu;

        std::vector<std::shared_ptr<Gpu_Tensor_Impl>> temp_impls;
        std::vector<std::shared_ptr<gpu::vector>> temp_buffers;
        auto buf_idx = getGpuFloat32ReadBuffer(idx_gpu, temp_impls, temp_buffers, engine);

        size_t elem_size = getDataTypeSize(out_gpu.getDataType());
        if (!out_gpu.getStorageBuffer() || out_gpu.getStorageBuffer()->getSizeBytes() < S * D * elem_size)
        {
            out_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext());
            out_gpu.getStorageBuffer()->allocateMemory(S * D, out_gpu.getDataType());
        }

        struct Embedding_Constants
        {
            uint32_t num_tokens;
            uint32_t embedding_dim;
            uint32_t vocab_size;
            int32_t padding_idx;
        } pcs{ static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V), padding_idx };
        validatePushConstants<Embedding_Constants>();

        auto table_storage = is_fp16 ? effective_w->getEffectiveFp16Storage() : effective_w->getStorageBuffer();
        const uint64_t wg_total = (static_cast<uint64_t>(S) * D + 255ULL) / 256ULL;
        auto [wg_x, wg_y] = engine.getContext().compute2DWorkgroupGrid(wg_total);
        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        if (wg_x > max_wg[0] || wg_y > max_wg[1])
        {
            throw std::overflow_error("ops::llm::embeddingForward: Workgroup grid exceeds hardware limits");
        }
        w_gpu.pushToGraph(is_fp16 ? Compute_Pipeline::EMBEDDING_FORWARD_FP16 : Compute_Pipeline::EMBEDDING_FORWARD,
                          { buf_idx, table_storage, out_gpu.getStorageBuffer() },
                          pcs, wg_x, wg_y, 1, &engine);
        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
    }
}

void embeddingForward(const Tensor &weights, const Tensor &indices, Tensor &output, bool force_fp16)
{
    Embedding_Config config{};
    config.force_fp16 = force_fp16;
    embeddingForward(weights, indices, output, config);
}

void embeddingBackward(const Tensor &weights, const Tensor &indices,
                       const Tensor &output_gradient, Tensor &weight_gradient,
                       const Embedding_Config &config)
{
    auto w_impl = weights.getImplementation();
    auto idx_impl = indices.getImplementation();
    auto og_impl = output_gradient.getImplementation();
    if (!w_impl || !idx_impl || !og_impl) return;

    if (weights.getDataType() == Data_Type::TERNARY_INT2 ||
        indices.getDataType() == Data_Type::TERNARY_INT2 ||
        output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::llm::embeddingBackward: TERNARY_INT2 is not supported");
    }
    if (weights.getShape().getRank() < 2)
    {
        throw std::invalid_argument("ops::llm::embeddingBackward: weights must have rank >= 2");
    }

    size_t S = indices.getTotalElements();
    size_t D = weights.getColumns();
    size_t V = weights.getShape()[0];
    const bool accumulate = config.accumulate;
    const int32_t padding_idx = config.padding_idx;

    if (output_gradient.getTotalElements() != S * D)
    {
        throw std::invalid_argument("ops::llm::embeddingBackward: output_gradient size mismatch (expected S * D)");
    }

    Execution_Target target = weights.getExecutionTarget();
    Execution_Engine *resolved_engine = (target != Execution_Target::CPU) ? &resolveExecutionEngine(config.engine) : nullptr;
    Shape w_shape = weights.getShape();

    auto wg_impl = weight_gradient.getImplementation();
    bool newly_allocated = (!wg_impl || wg_impl.use_count() > 2 || wg_impl->getByteOffset() != 0 ||
                            !wg_impl->isContiguous() ||
                            weight_gradient.getExecutionTarget() != target ||
                            weight_gradient.getDataType() != Data_Type::FLOAT32 ||
                            weight_gradient.getShape() != w_shape ||
                            weight_gradient.isEmpty());
    if (newly_allocated)
    {
        if (accumulate && wg_impl && weight_gradient.getShape() == w_shape && V * D > 0)
        {
            Tensor preserved(w_shape, weight_gradient.getData(), target);
            weight_gradient = std::move(preserved);
            newly_allocated = false;
        }
        else
        {
            weight_gradient = Tensor(w_shape, Data_Type::FLOAT32, target);
        }
    }
    else
    {
        weight_gradient.reshape(w_shape);
    }

    if (target == Execution_Target::CPU)
    {
        std::vector<float> grad_buf;
        if (accumulate && !newly_allocated)
        {
            grad_buf = weight_gradient.getData();
            grad_buf.resize(V * D, 0.0f);
        }
        else
        {
            grad_buf.assign(V * D, 0.0f);
        }

        if (S > 0 && D > 0 && V > 0)
        {
            const auto indices_data = indices.getData();
            const auto grad_out_data = output_gradient.getData();
            for (size_t s = 0; s < S; ++s)
            {
                size_t row = 0;
                if (tryFloatToTokenIndex(indices_data[s], V, padding_idx, row))
                {
                    for (size_t d = 0; d < D; ++d)
                    {
                        grad_buf[row * D + d] += grad_out_data[s * D + d];
                    }
                }
            }
        }
        writeCpuOutput(weight_gradient, w_shape, Data_Type::FLOAT32, grad_buf);
    }
    else
    {
        Execution_Engine &engine = *resolved_engine;
        if (S > std::numeric_limits<uint32_t>::max() ||
            D > std::numeric_limits<uint32_t>::max() ||
            V > std::numeric_limits<uint32_t>::max() ||
            (D > 0 && (S > std::numeric_limits<uint32_t>::max() / D ||
                       V > std::numeric_limits<uint32_t>::max() / D)))
        {
            throw std::invalid_argument("ops::llm::embeddingBackward: tensor size exceeds uint32_t limit");
        }

        auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
        bool buffer_reallocated = (V * D > 0) &&
            (!w_grad_gpu.getStorageBuffer() || w_grad_gpu.getStorageBuffer()->getSizeBytes() < V * D * sizeof(float));
        if (buffer_reallocated)
        {
            w_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext());
            w_grad_gpu.getStorageBuffer()->allocateMemory(V * D, Data_Type::FLOAT32);
        }
        if ((!accumulate || newly_allocated || buffer_reallocated) && V * D > 0)
        {
            pushGpuFill(w_grad_gpu, V * D, 0.0f, engine);
        }

        if (S == 0 || D == 0 || V == 0)
        {
            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            return;
        }

        Tensor ind_dev = indices;
        if (indices.getExecutionTarget() != target)
        {
            ind_dev = Tensor(indices.getShape(), indices.getData(), target);
        }

        Tensor og_dev = output_gradient;
        if (output_gradient.getExecutionTarget() != target)
        {
            og_dev = Tensor(output_gradient.getShape(), output_gradient.getData(), target);
        }

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*weights.getImplementation());
        const auto &idx_gpu = Gpu_Tensor_Impl::castToGpu(*ind_dev.getImplementation());
        const auto &grad_out_gpu = Gpu_Tensor_Impl::castToGpu(*og_dev.getImplementation());

        std::vector<std::shared_ptr<Gpu_Tensor_Impl>> temp_impls;
        std::vector<std::shared_ptr<gpu::vector>> temp_buffers;
        auto buf_idx = getGpuFloat32ReadBuffer(idx_gpu, temp_impls, temp_buffers, engine);

        auto contig_grad = grad_out_gpu.ensureContiguousSelf();
        const Gpu_Tensor_Impl *effective_grad = contig_grad ? contig_grad.get() : &grad_out_gpu;

        const auto &max_wg = engine.getContext().getMaxComputeWorkGroupCount();
        uint64_t wg_x_u64 = (static_cast<uint64_t>(D) + 255ULL) / 256ULL;
        uint32_t max_wg_y = std::min<uint32_t>(65535u, max_wg[1]);
        uint32_t chunk_size = 256u;
        size_t num_chunks = (S + chunk_size - 1) / chunk_size;
        if (num_chunks > max_wg_y && max_wg_y > 0)
        {
            uint64_t cs_u64 = (S + max_wg_y - 1) / max_wg_y;
            if (cs_u64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("ops::llm::embeddingBackward: chunk_size exceeds uint32_t");
            }
            chunk_size = static_cast<uint32_t>(cs_u64);
        }
        uint64_t wg_y_u64 = (S + chunk_size - 1) / chunk_size;
        if (wg_x_u64 > max_wg[0] || wg_y_u64 > max_wg[1])
        {
            throw std::overflow_error("ops::llm::embeddingBackward: Workgroup grid exceeds hardware limits");
        }
        uint32_t wg_x = static_cast<uint32_t>(wg_x_u64);
        uint32_t wg_y = static_cast<uint32_t>(wg_y_u64);

        struct Embedding_Constants
        {
            uint32_t num_tokens;
            uint32_t embedding_dim;
            uint32_t vocab_size;
            uint32_t chunk_size;
            int32_t padding_idx;
        } pcs{ static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V), chunk_size, padding_idx };
        validatePushConstants<Embedding_Constants>();

        bool is_fp16 = engine.getContext().isFloat16Enabled() &&
                       (effective_grad->getDataType() == Data_Type::FLOAT16 || engine.isCooperativeMatrixEnabled());
        auto grad_storage = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

        Compute_Pipeline pipe = is_fp16
            ? Compute_Pipeline::EMBEDDING_BACKWARD_FP16
            : Compute_Pipeline::EMBEDDING_BACKWARD;

        w_gpu.pushToGraph(pipe,
                          { buf_idx, grad_storage, w_grad_gpu.getStorageBuffer() },
                          pcs,
                          wg_x, wg_y, 1, &engine);
        w_grad_gpu.clearHostCache();
        w_grad_gpu.invalidateFp16Cache();
    }
}

void embeddingBackward(const Tensor &weights, const Tensor &indices,
                       const Tensor &output_gradient, Tensor &weight_gradient,
                       bool accumulate)
{
    Embedding_Config config{};
    config.accumulate = accumulate;
    embeddingBackward(weights, indices, output_gradient, weight_gradient, config);
}

}

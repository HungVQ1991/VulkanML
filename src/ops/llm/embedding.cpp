#include "ops/llm/embedding.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::llm
{

static void ensureTensorTarget(Tensor &t, Execution_Target target, const Shape &shape, Data_Type dtype)
{
    if (t.getExecutionTarget() != target || t.getDataType() != dtype || t.getShape() != shape)
    {
        t = Tensor(shape, dtype, target);
    }
    else
    {
        t.reshape(shape);
    }
}

void embeddingForward(const Tensor &weights, const Tensor &indices, Tensor &output)
{
    auto w_impl = weights.getImplementation();
    auto idx_impl = indices.getImplementation();
    if (!w_impl || !idx_impl) return;

    size_t S = indices.getTotalElements();
    size_t D = weights.getColumns();
    size_t V = weights.getShape()[0];

    Shape out_shape;
    if (indices.getShape().getRank() == 2)
    {
        out_shape = Shape{ indices.getShape()[0], indices.getShape()[1], D };
    }
    else
    {
        out_shape = Shape{ S, D };
    }

    Execution_Target target = weights.getExecutionTarget();
    bool is_fp16 = (weights.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
    Data_Type out_dtype = (target != Execution_Target::CPU && is_fp16) ? Data_Type::FLOAT16 : weights.getDataType();

    ensureTensorTarget(output, target, out_shape, out_dtype);

    if (target == Execution_Target::CPU)
    {
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        const auto &w_cpu = Cpu_Tensor_Impl::castToCpu(*weights.getImplementation());

        if (weights.getDataType() == Data_Type::FLOAT16)
        {
            if (!out_cpu.getStorageBufferFp16() || out_cpu.getStorageBufferFp16()->size() != S * D)
            {
                out_cpu.getStorageBufferFp16() = std::make_shared<std::vector<float16_t>>(S * D, float16_t(0.0f));
            }
            out_cpu.getStorageBuffer().reset();
        }
        else
        {
            if (!out_cpu.getStorageBuffer() || out_cpu.getStorageBuffer()->size() != S * D)
            {
                out_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(S * D, 0.0f);
            }
            out_cpu.getStorageBufferFp16().reset();
        }

        const auto &indices_data = indices.getData();
        const auto &weight_data = weights.getData();

        for (size_t s = 0; s < S; ++s)
        {
            int row = static_cast<int>(std::round(indices_data[s]));
            if (row >= 0 && static_cast<size_t>(row) < V)
            {
                if (weights.getDataType() == Data_Type::FLOAT16)
                {
                    std::memcpy(out_cpu.getStorageBufferFp16()->data() + s * D,
                                w_cpu.getStorageBufferFp16()->data() + row * D,
                                D * sizeof(float16_t));
                }
                else
                {
                    std::memcpy(out_cpu.getStorageBuffer()->data() + s * D,
                                weight_data.data() + row * D,
                                D * sizeof(float));
                }
            }
            else
            {
                if (weights.getDataType() == Data_Type::FLOAT16)
                {
                    std::memset(out_cpu.getStorageBufferFp16()->data() + s * D, 0, D * sizeof(float16_t));
                }
                else
                {
                    std::memset(out_cpu.getStorageBuffer()->data() + s * D, 0, D * sizeof(float));
                }
            }
        }
    }
    else
    {
        Tensor ind_dev = indices;
        if (indices.getExecutionTarget() != target)
        {
            ind_dev = Tensor(indices.getShape(), indices.getData(), target);
        }

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*weights.getImplementation());
        const auto &idx_gpu = Gpu_Tensor_Impl::castToGpu(*ind_dev.getImplementation());
        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        size_t elem_size = getDataTypeSize(out_gpu.getDataType());
        if (!out_gpu.getStorageBuffer() || out_gpu.getStorageBuffer()->getSizeBytes() < S * D * elem_size)
        {
            out_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            out_gpu.getStorageBuffer()->allocateMemory(S * D, out_gpu.getDataType());
        }

        struct Embedding_Constants
        {
            uint32_t num_tokens;
            uint32_t embedding_dim;
            uint32_t vocab_size;
        } pcs{ static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V) };

        auto table_storage = is_fp16 ? w_gpu.getEffectiveFp16Storage() : w_gpu.getStorageBuffer();
        w_gpu.pushToGraph(is_fp16 ? Compute_Pipeline::EMBEDDING_FORWARD_FP16 : Compute_Pipeline::EMBEDDING_FORWARD,
                          { idx_gpu.getStorageBuffer(), table_storage, out_gpu.getStorageBuffer() },
                          pcs,
                          static_cast<uint32_t>((S * D + 255) / 256), 1, 1);
        out_gpu.clearHostCache();
    }
}

void embeddingBackward(const Tensor &weights, const Tensor &indices,
                       const Tensor &output_gradient, Tensor &weight_gradient)
{
    auto w_impl = weights.getImplementation();
    auto idx_impl = indices.getImplementation();
    auto og_impl = output_gradient.getImplementation();
    if (!w_impl || !idx_impl || !og_impl) return;

    size_t S = indices.getTotalElements();
    size_t D = weights.getColumns();
    size_t V = weights.getShape()[0];

    Execution_Target target = weights.getExecutionTarget();
    Shape w_shape = weights.getShape();

    ensureTensorTarget(weight_gradient, target, w_shape, Data_Type::FLOAT32);

    if (target == Execution_Target::CPU)
    {
        auto &w_grad_cpu = Cpu_Tensor_Impl::castToCpu(*weight_gradient.getImplementation());
        if (!w_grad_cpu.getStorageBuffer() || w_grad_cpu.getStorageBuffer()->size() != V * D)
        {
            w_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(V * D, 0.0f);
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
                    (*w_grad_cpu.getStorageBuffer())[row * D + d] += grad_out_data[s * D + d];
                }
            }
        }
    }
    else
    {
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
        auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());

        if (!w_grad_gpu.getStorageBuffer() || w_grad_gpu.getStorageBuffer()->getSizeBytes() < V * D * sizeof(float))
        {
            w_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            w_grad_gpu.getStorageBuffer()->allocateMemory(V * D, Data_Type::FLOAT32);
        }

        struct Embedding_Constants
        {
            uint32_t num_tokens;
            uint32_t embedding_dim;
            uint32_t vocab_size;
        } pcs{ static_cast<uint32_t>(S), static_cast<uint32_t>(D), static_cast<uint32_t>(V) };

        bool is_fp16 = (grad_out_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        auto grad_storage = is_fp16 ? grad_out_gpu.getEffectiveFp16Storage() : grad_out_gpu.getStorageBuffer();

        Compute_Pipeline pipe = is_fp16
            ? Compute_Pipeline::EMBEDDING_BACKWARD_FP16
            : Compute_Pipeline::EMBEDDING_BACKWARD;

        w_gpu.pushToGraph(pipe,
                          { idx_gpu.getStorageBuffer(), grad_storage, w_grad_gpu.getStorageBuffer() },
                          pcs,
                          static_cast<uint32_t>((S * D + 255) / 256), 1, 1);
        w_grad_gpu.clearHostCache();
    }
}

} // namespace ops::llm

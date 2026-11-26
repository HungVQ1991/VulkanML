#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "math/shape.h"
#include "math/tensor.h"

namespace ops::internal
{

inline Execution_Engine &resolveExecutionEngine(Execution_Engine *engine)
{
    return engine ? *engine : Execution_Engine::getInstance();
}

inline uint32_t validateU32Count(size_t count, const char *op)
{
    if (count > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error(std::string(op) + ": Element count exceeds UINT32_MAX");
    }
    return static_cast<uint32_t>(count);
}

inline std::pair<uint32_t, uint32_t> dispatch2D(uint64_t total_wgs, Execution_Engine *engine = nullptr)
{
    return resolveExecutionEngine(engine).getContext().compute2DWorkgroupGrid(total_wgs);
}

inline void ensureTensorTarget(Tensor &t, Execution_Target target, const Shape &shape, Data_Type dtype)
{
    auto impl = t.getImplementation();
    if (!impl || t.getExecutionTarget() != target || t.getDataType() != dtype ||
        t.getTotalElements() != shape.getTotalElements() ||
        !impl->isContiguous() || impl->getByteOffset() != 0)
    {
        t = Tensor(shape, dtype, target);
    }
    else
    {
        t.reshape(shape);
    }
}

inline void ensureUniqueTensorTarget(Tensor &t, Execution_Target target, const Shape &shape, Data_Type dtype)
{
    auto impl = t.getImplementation();
    if (!impl || impl.use_count() > 2 ||
        t.getExecutionTarget() != target || t.getDataType() != dtype ||
        t.getShape() != shape || !impl->isContiguous() || impl->getByteOffset() != 0)
    {
        t = Tensor(shape, dtype, target);
    }
}

inline bool sharesUnderlyingStorage(const Tensor &a, const Tensor &b)
{
    if (&a == &b)
    {
        return true;
    }
    auto impl_a = a.getImplementation();
    auto impl_b = b.getImplementation();
    if (!impl_a || !impl_b)
    {
        return false;
    }
    if (impl_a.get() == impl_b.get())
    {
        return true;
    }
    if (a.getExecutionTarget() != b.getExecutionTarget())
    {
        return false;
    }
    if (a.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &ca = Cpu_Tensor_Impl::castToCpu(*impl_a);
        const auto &cb = Cpu_Tensor_Impl::castToCpu(*impl_b);
        return (ca.getStorageBuffer() && ca.getStorageBuffer().get() == cb.getStorageBuffer().get()) ||
               (ca.getStorageBufferFp16() && ca.getStorageBufferFp16().get() == cb.getStorageBufferFp16().get()) ||
               (ca.getStorageBufferU32() && ca.getStorageBufferU32().get() == cb.getStorageBufferU32().get());
    }
    const auto &ga = Gpu_Tensor_Impl::castToGpu(*impl_a);
    const auto &gb = Gpu_Tensor_Impl::castToGpu(*impl_b);
    return ga.getStorageBuffer() && ga.getStorageBuffer().get() == gb.getStorageBuffer().get();
}

inline bool ensureDisjointTensorTarget(
    Tensor &t,
    Execution_Target target,
    const Shape &shape,
    Data_Type dtype,
    std::initializer_list<const Tensor *> inputs)
{
    bool aliased = false;
    for (const Tensor *in : inputs)
    {
        if (in != nullptr && sharesUnderlyingStorage(t, *in))
        {
            aliased = true;
            break;
        }
    }
    auto impl = t.getImplementation();
    if (aliased || !impl || impl.use_count() > 2 ||
        t.getExecutionTarget() != target || t.getDataType() != dtype ||
        t.getShape() != shape || !impl->isContiguous() || impl->getByteOffset() != 0)
    {
        t = Tensor(shape, dtype, target);
        return true;
    }
    return false;
}

inline void ensureDisjointGpuStorage(
    Gpu_Tensor_Impl &out_gpu,
    size_t required_elements,
    Data_Type dtype,
    std::initializer_list<const gpu::vector *> forbidden_buffers)
{
    const gpu::vector *cur = out_gpu.getStorageBuffer().get();
    bool aliased = (cur == nullptr);
    if (!aliased)
    {
        for (const gpu::vector *f : forbidden_buffers)
        {
            if (f != nullptr && f == cur)
            {
                aliased = true;
                break;
            }
        }
    }
    if (aliased && required_elements > 0)
    {
        out_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(
            Execution_Engine::getInstance().getContext(), required_elements, dtype);
        out_gpu.markGpuMutated();
    }
}

inline void writeCpuOutput(
    Cpu_Tensor_Impl &out_cpu,
    const Shape &out_shape,
    Data_Type dtype,
    std::vector<float> values,
    bool clamp_fp16 = false)
{
    out_cpu.setShape(out_shape);
    out_cpu.setByteOffset(0);
    out_cpu.setDataType(dtype);
    const size_t total = values.size();
    if (dtype == Data_Type::FLOAT16)
    {
        auto &fp16_buf = out_cpu.getStorageBufferFp16();
        if (!fp16_buf || fp16_buf.use_count() > 1 || fp16_buf->size() != total)
        {
            fp16_buf = std::make_shared<std::vector<float16_t>>(total);
        }
        if (total > 0)
        {
            if (clamp_fp16)
            {
                for (size_t i = 0; i < total; ++i)
                {
                    if (!std::isnan(values[i]))
                    {
                        values[i] = std::clamp(values[i], -65504.0f, 65504.0f);
                    }
                }
            }
            convertFp32ToFp16(values.data(), fp16_buf->data(), total);
        }
        out_cpu.getStorageBuffer().reset();
        out_cpu.getStorageBufferU32().reset();
    }
    else
    {
        auto &fp32_buf = out_cpu.getStorageBuffer();
        if (!fp32_buf || fp32_buf.use_count() > 1 || fp32_buf->size() != total)
        {
            fp32_buf = std::make_shared<std::vector<float>>(std::move(values));
        }
        else
        {
            *fp32_buf = std::move(values);
        }
        out_cpu.getStorageBufferFp16().reset();
        out_cpu.getStorageBufferU32().reset();
    }
}

template <typename T>
constexpr void validatePushConstants() noexcept
{
    static_assert(std::is_trivially_copyable_v<T>, "Push constants struct must be trivially copyable");
    static_assert(sizeof(T) <= 128, "Push constants struct exceeds Vulkan 128-byte limit");
}

inline void pushGpuFill(const Gpu_Tensor_Impl &anchor,
                        const std::shared_ptr<gpu::vector> &dst_buf,
                        size_t total_elements,
                        float fill_value,
                        Data_Type dtype = Data_Type::FLOAT32,
                        Execution_Engine *engine_ptr = nullptr)
{
    if (total_elements == 0)
    {
        return;
    }
    if (!dst_buf)
    {
        throw std::runtime_error("ops::internal::pushGpuFill: Missing GPU destination buffer");
    }
    const uint32_t count_u32 = validateU32Count(total_elements, "ops::internal::pushGpuFill");
    struct Fill_Constants
    {
        uint32_t total_elements;
        float fill_value;
    } pc{count_u32, fill_value};
    validatePushConstants<Fill_Constants>();
    auto [wg_x, wg_y] = dispatch2D((static_cast<uint64_t>(count_u32) + 255ULL) / 256ULL, engine_ptr);
    const Compute_Pipeline pipe = (dtype == Data_Type::FLOAT16) ? Compute_Pipeline::FILL_FP16 : Compute_Pipeline::FILL_FLOAT;
    anchor.pushToGraph(pipe, {dst_buf}, pc, wg_x, wg_y, 1, engine_ptr);
}

inline void pushGpuFill(Gpu_Tensor_Impl &dst,
                        size_t count,
                        float value,
                        Execution_Engine &engine)
{
    if (count == 0)
    {
        dst.clearHostCache();
        dst.invalidateFp16Cache();
        return;
    }
    const uint32_t count_u32 = validateU32Count(count, "ops::internal::pushGpuFill");
    const bool is_fp16 = (dst.getDataType() == Data_Type::FLOAT16) && engine.getContext().isFloat16Enabled();
    const Data_Type buf_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    const size_t elem_size = getDataTypeSize(buf_dtype);
    if (!dst.getStorageBuffer() || dst.getStorageBuffer()->getSizeBytes() < static_cast<size_t>(count_u32) * elem_size)
    {
        dst.getStorageBuffer() = std::make_shared<gpu::vector>(engine.getContext());
        dst.getStorageBuffer()->allocateMemory(count, buf_dtype);
    }
    pushGpuFill(dst, dst.getStorageBuffer(), count, value, buf_dtype, &engine);
    dst.clearHostCache();
    dst.invalidateFp16Cache();
}

}

#include "ops/dropout/dropout.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::dropout
{

static void writeCpuDropoutOutput(Cpu_Tensor_Impl &out_cpu, const Shape &out_shape, Data_Type dtype, std::vector<float> values)
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

void dropout(const Tensor &input, Tensor &output, Tensor &mask, float dropout_rate, bool is_training, uint64_t seed)
{
    if (!input.getImplementation())
    {
        throw std::invalid_argument("ops::dropout: Input tensor has null implementation");
    }
    const Data_Type dtype = input.getDataType();
    if (dtype == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::dropout: TERNARY_INT2 input is not supported");
    }

    if (!std::isfinite(dropout_rate) || dropout_rate < 0.0f || dropout_rate >= 1.0f)
    {
        throw std::invalid_argument("ops::dropout: dropout_rate must be finite and in [0.0f, 1.0f)");
    }

    if (!is_training || dropout_rate <= 0.0f)
    {
        output = input;
        mask = Tensor(0, 0, input.getExecutionTarget());
        return;
    }

    const float scale = 1.0f / (1.0f - dropout_rate);
    if (dtype == Data_Type::FLOAT16 && scale > 65504.0f)
    {
        throw std::invalid_argument("ops::dropout: dropout_rate scale exceeds FLOAT16 range");
    }

    const size_t total_elements = input.getShape().getTotalElements();
    if (total_elements == 0)
    {
        output = input;
        mask = Tensor(input.getShape(), dtype, input.getExecutionTarget());
        return;
    }

    uint64_t effective_seed = seed;
    if (effective_seed == 0)
    {
        static std::atomic<uint64_t> fallback_counter{1};
        static const uint64_t base_entropy = []() -> uint64_t {
            std::random_device rd;
            return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        }();
        uint64_t z = base_entropy + 0x9E3779B97F4A7C15ULL * fallback_counter.fetch_add(1, std::memory_order_relaxed);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        effective_seed = (z == 0ULL) ? 1ULL : z;
    }

    auto mask_impl = mask.getImplementation();
    if (!mask_impl ||
        mask_impl == input.getImplementation() ||
        mask.getExecutionTarget() != input.getExecutionTarget() ||
        mask.getDataType() != dtype ||
        mask.getShape() != input.getShape() ||
        !mask_impl->isContiguous() ||
        mask_impl->getByteOffset() != 0 ||
        mask.isEmpty())
    {
        mask = Tensor(input.getShape(), dtype, input.getExecutionTarget());
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        std::vector<float> mask_data(total_elements);
        std::mt19937_64 rng(effective_seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < total_elements; ++i)
        {
            mask_data[i] = (dist(rng) >= dropout_rate) ? scale : 0.0f;
        }
        const auto &input_data = Cpu_Tensor_Impl::castToCpu(*input.getImplementation()).getData();
        std::vector<float> output_data(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            output_data[i] = input_data[i] * mask_data[i];
        }
        auto &mask_cpu = Cpu_Tensor_Impl::castToCpu(*mask.getImplementation());
        writeCpuDropoutOutput(mask_cpu, input.getShape(), dtype, std::move(mask_data));

        auto out_impl = output.getImplementation();
        if (!out_impl ||
            out_impl == input.getImplementation() ||
            out_impl == mask.getImplementation() ||
            output.getExecutionTarget() != Execution_Target::CPU ||
            output.getDataType() != dtype ||
            output.getShape() != input.getShape() ||
            !out_impl->isContiguous() ||
            out_impl->getByteOffset() != 0)
        {
            output = Tensor(input.getShape(), dtype, Execution_Target::CPU);
        }
        auto &out_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        writeCpuDropoutOutput(out_cpu, input.getShape(), dtype, std::move(output_data));
    }
    else
    {
        if (total_elements > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::dropout: Element count exceeds UINT32_MAX");
        }
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_in = in_gpu.ensureContiguousSelf();
        const auto *eff_in = contig_in ? contig_in.get() : &in_gpu;

        auto out_impl = output.getImplementation();
        if (!out_impl ||
            out_impl == input.getImplementation() ||
            out_impl == mask.getImplementation() ||
            output.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output.getDataType() != dtype ||
            output.getShape() != input.getShape() ||
            !out_impl->isContiguous() ||
            out_impl->getByteOffset() != 0 ||
            output.isEmpty())
        {
            output = Tensor(input.getShape(), dtype, Execution_Target::VULKAN_GPU);
        }

        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        auto &mask_gpu = Gpu_Tensor_Impl::castToGpu(*mask.getImplementation());

        struct Dropout_Constants
        {
            uint32_t total_elements;
            float dropout_rate;
            float scale;
            uint32_t seed_lo;
            uint32_t seed_hi;
        } pc{
            static_cast<uint32_t>(total_elements),
            dropout_rate,
            scale,
            static_cast<uint32_t>(effective_seed & 0xFFFFFFFFULL),
            static_cast<uint32_t>((effective_seed >> 32) & 0xFFFFFFFFULL)};

        auto [wg_x, wg_y] = Execution_Engine::getInstance().getContext().compute2DWorkgroupGrid((total_elements + 255ull) / 256ull);
        eff_in->pushToGraph(
            (dtype == Data_Type::FLOAT16) ? Compute_Pipeline::DROPOUT_FORWARD_FP16 : Compute_Pipeline::DROPOUT_FORWARD,
            {eff_in->getStorageBuffer(), out_gpu.getStorageBuffer(), mask_gpu.getStorageBuffer()},
            pc, wg_x, wg_y, 1);

        out_gpu.clearHostCache();
        out_gpu.invalidateFp16Cache();
        mask_gpu.clearHostCache();
        mask_gpu.invalidateFp16Cache();
    }
}

void dropoutBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient, float dropout_rate)
{
    if (!output_gradient.getImplementation() || !mask.getImplementation())
    {
        throw std::invalid_argument("ops::dropoutBackward: Null tensor implementation");
    }
    if (output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        mask.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::dropoutBackward: TERNARY_INT2 is not supported");
    }
    if (!std::isfinite(dropout_rate) || dropout_rate < 0.0f || dropout_rate >= 1.0f)
    {
        throw std::invalid_argument("ops::dropoutBackward: dropout_rate must be finite and in [0.0f, 1.0f)");
    }

    if (dropout_rate <= 0.0f || mask.getShape().getTotalElements() == 0)
    {
        input_gradient = output_gradient;
        return;
    }

    if (output_gradient.getExecutionTarget() != mask.getExecutionTarget())
    {
        throw std::invalid_argument("ops::dropoutBackward: output_gradient and mask must share the same Execution_Target");
    }

    if (output_gradient.getShape() != mask.getShape())
    {
        throw std::invalid_argument("ops::dropoutBackward: output_gradient and mask shapes must match");
    }

    if (output_gradient.getExecutionTarget() == Execution_Target::CPU)
    {
        const Data_Type out_dtype = output_gradient.getDataType();
        const auto &grad_data = Cpu_Tensor_Impl::castToCpu(*output_gradient.getImplementation()).getData();
        const auto &mask_data = Cpu_Tensor_Impl::castToCpu(*mask.getImplementation()).getData();
        const size_t total_elements = grad_data.size();
        std::vector<float> in_grad_data(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            in_grad_data[i] = grad_data[i] * mask_data[i];
        }
        auto in_grad_impl = input_gradient.getImplementation();
        if (!in_grad_impl ||
            in_grad_impl == output_gradient.getImplementation() ||
            in_grad_impl == mask.getImplementation() ||
            input_gradient.getExecutionTarget() != Execution_Target::CPU ||
            input_gradient.getDataType() != out_dtype ||
            input_gradient.getShape() != output_gradient.getShape() ||
            !in_grad_impl->isContiguous() ||
            in_grad_impl->getByteOffset() != 0)
        {
            input_gradient = Tensor(output_gradient.getShape(), out_dtype, Execution_Target::CPU);
        }
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
        writeCpuDropoutOutput(in_grad_cpu, output_gradient.getShape(), out_dtype, std::move(in_grad_data));
    }
    else
    {
        if (mask.getDataType() != output_gradient.getDataType())
        {
            Tensor mask_matched = mask.to(output_gradient.getDataType());
            output_gradient.hadamardMul(mask_matched, input_gradient);
        }
        else
        {
            output_gradient.hadamardMul(mask, input_gradient);
        }
    }
}

}

#include "ops/linear/linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::linear
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

void linearForward(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output)
{
    input.matmulAdd(weights, biases, output);
}

void linearBackwardInput(const Tensor &output_gradient, const Tensor &weights, Tensor &input_gradient)
{
    auto og_impl = output_gradient.getImplementation();
    auto w_impl = weights.getImplementation();
    if (!og_impl || !w_impl) return;

    Execution_Target target = output_gradient.getExecutionTarget();

    if (target == Execution_Target::CPU)
    {
        size_t b_size = output_gradient.getRows();
        size_t out_dim = output_gradient.getColumns();
        size_t in_dim = weights.getRows();

        ensureTensorTarget(input_gradient, target, Shape{ b_size, in_dim }, output_gradient.getDataType());

        const auto &w_data = weights.getData();
        const auto &out_grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        if (!in_grad_cpu.getStorageBuffer() || in_grad_cpu.getStorageBuffer()->size() != b_size * in_dim)
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(b_size * in_dim, 0.0f);
        }

        for (size_t i = 0; i < b_size; ++i)
        {
            for (size_t j = 0; j < in_dim; ++j)
            {
                float sum = 0.0F;
                for (size_t k = 0; k < out_dim; ++k)
                {
                    sum += out_grad_data[i * out_dim + k] * w_data[j * out_dim + k];
                }
                (*in_grad_cpu.getStorageBuffer())[i * in_dim + j] = sum;
            }
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*weights.getImplementation());
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        bool use_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16 || w_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        Data_Type out_dtype = use_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        ensureTensorTarget(input_gradient, target, Shape{ output_gradient.getRows(), effective_w->getRows() }, out_dtype);
        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Constants
        {
            uint32_t batch_size;
            uint32_t input_dimension;
            uint32_t output_dimension;
        } c{static_cast<uint32_t>(output_gradient.getRows()), static_cast<uint32_t>(effective_w->getRows()), static_cast<uint32_t>(output_gradient.getColumns())};

        if (use_fp16)
        {
            auto buf_self = effective_grad->getEffectiveFp16Storage();
            auto buf_w = effective_w->getEffectiveFp16Storage();
            Compute_Pipeline pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16
                : Compute_Pipeline::LINEAR_BACKWARD_INPUT_FP16;
            effective_grad->pushToGraph(pipeline,
                        {buf_self, buf_w, in_grad_gpu.getStorageBuffer()}, c,
                        (c.input_dimension + 15) / 16, (c.batch_size + 15) / 16, 1);
        }
        else
        {
            effective_grad->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_INPUT,
                        {effective_grad->getStorageBuffer(), effective_w->getStorageBuffer(), in_grad_gpu.getStorageBuffer()}, c,
                        (c.input_dimension + 15) / 16, (c.batch_size + 15) / 16, 1);
        }
        in_grad_gpu.clearHostCache();
    }
}

void linearBackwardWeightBias(const Tensor &input, const Tensor &output_gradient,
                             Tensor &weight_gradient, Tensor &bias_gradient, bool accumulate)
{
    auto in_impl = input.getImplementation();
    auto og_impl = output_gradient.getImplementation();
    if (!in_impl || !og_impl) return;

    Execution_Target target = input.getExecutionTarget();

    if (target == Execution_Target::CPU)
    {
        size_t b_size = input.getRows();
        size_t in_dim = input.getColumns();
        size_t out_dim = output_gradient.getColumns();

        ensureTensorTarget(weight_gradient, target, Shape{ in_dim, out_dim }, Data_Type::FLOAT32);
        ensureTensorTarget(bias_gradient, target, Shape{ 1, out_dim }, Data_Type::FLOAT32);

        auto &w_grad_cpu = Cpu_Tensor_Impl::castToCpu(*weight_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*bias_gradient.getImplementation());

        if (!w_grad_cpu.getStorageBuffer() || w_grad_cpu.getStorageBuffer()->size() != in_dim * out_dim)
        {
            w_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(in_dim * out_dim, 0.0f);
        }
        if (!b_grad_cpu.getStorageBuffer() || b_grad_cpu.getStorageBuffer()->size() != out_dim)
        {
            b_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(out_dim, 0.0f);
        }

        if (!accumulate)
        {
            std::fill(w_grad_cpu.getStorageBuffer()->begin(), w_grad_cpu.getStorageBuffer()->end(), 0.0F);
            std::fill(b_grad_cpu.getStorageBuffer()->begin(), b_grad_cpu.getStorageBuffer()->end(), 0.0F);
        }

        const auto &in_data = input.getData();
        const auto &out_grad_data = output_gradient.getData();

        for (size_t i = 0; i < b_size; ++i)
        {
            for (size_t j = 0; j < out_dim; ++j)
            {
                float grad_val = out_grad_data[i * out_dim + j];
                (*b_grad_cpu.getStorageBuffer())[j] += grad_val;
                for (size_t k = 0; k < in_dim; ++k)
                {
                    (*w_grad_cpu.getStorageBuffer())[k * out_dim + j] += in_data[i * in_dim + k] * grad_val;
                }
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input.getImplementation());
        auto contig_in = in_gpu.ensureContiguousSelf();
        const auto *effective_in = contig_in ? contig_in.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation());
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        ensureTensorTarget(weight_gradient, target, Shape{ input.getColumns(), effective_grad->getColumns() }, Data_Type::FLOAT32);
        ensureTensorTarget(bias_gradient, target, Shape{ 1, effective_grad->getColumns() }, Data_Type::FLOAT32);

        auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*bias_gradient.getImplementation());

        size_t elem_size = sizeof(float);
        if (!w_grad_gpu.getStorageBuffer() || w_grad_gpu.getStorageBuffer()->getSizeBytes() < input.getColumns() * effective_grad->getColumns() * elem_size)
        {
            w_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            w_grad_gpu.getStorageBuffer()->allocateMemory(input.getColumns() * effective_grad->getColumns(), Data_Type::FLOAT32);
        }
        if (!b_grad_gpu.getStorageBuffer() || b_grad_gpu.getStorageBuffer()->getSizeBytes() < effective_grad->getColumns() * elem_size)
        {
            b_grad_gpu.getStorageBuffer() = std::make_shared<gpu::vector>(Execution_Engine::getInstance().getContext());
            b_grad_gpu.getStorageBuffer()->allocateMemory(effective_grad->getColumns(), Data_Type::FLOAT32);
        }

        struct Constants
        {
            uint32_t batch_size;
            uint32_t input_dimension;
            uint32_t output_dimension;
            uint32_t accumulate;
        } c{static_cast<uint32_t>(input.getRows()), static_cast<uint32_t>(input.getColumns()), static_cast<uint32_t>(effective_grad->getColumns()), accumulate ? 1u : 0u};

        bool use_fp16 = (input.getDataType() == Data_Type::FLOAT16 || out_grad_gpu.getDataType() == Data_Type::FLOAT16 || Execution_Engine::getInstance().isCooperativeMatrixEnabled());
        if (use_fp16)
        {
            auto buf_in = effective_in->getEffectiveFp16Storage();
            auto buf_grad = effective_grad->getEffectiveFp16Storage();
            Compute_Pipeline pipeline = Execution_Engine::getInstance().isCooperativeMatrixEnabled()
                ? Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16
                : Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_FP16;
            effective_in->pushToGraph(pipeline,
                        {buf_in, buf_grad, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()}, c,
                        (c.output_dimension + 15) / 16, (c.input_dimension + 15) / 16, 1);
        }
        else
        {
            effective_in->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS,
                        {effective_in->getStorageBuffer(), effective_grad->getStorageBuffer(), w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()}, c,
                        (c.output_dimension + 15) / 16, (c.input_dimension + 15) / 16, 1);
        }
        w_grad_gpu.clearHostCache();
        b_grad_gpu.clearHostCache();
    }
}

} // namespace ops::linear

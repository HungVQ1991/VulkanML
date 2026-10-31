#include "ops/activation/activation.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::activation
{

void relu(const Tensor &input, Tensor &output)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }
    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        output.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*out_impl);
        output_cpu.reshape(input.getShape());
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() < total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &out_buf = *output_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            out_buf[i] = std::max(0.0F, a_data[i]);
        }
        Logger::logMessage(Input_Format{"ops::activation::relu (CPU): elements={}", total_elements},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        if (in_gpu.getDataType() == Data_Type::FLOAT16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        output_gpu.reshape(in_gpu.getShape());
        in_gpu.pushToGraph(in_gpu.getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::RELU_FP16 : Compute_Pipeline::RELU,
                          {effective_self->getStorageBuffer(), output_gpu.getStorageBuffer()},
                          static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }
}

void reluBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient)
{
    auto out_impl = output.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!out_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (input_gradient.getExecutionTarget() != output.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(output.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    out_impl->validateSameDimensions(*grad_impl);
    size_t total_elements = output.getTotalElements();
    if (total_elements == 0)
    {
        input_gradient.reshape(output.getShape());
        return;
    }

    if (output.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = output.getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*in_grad_impl);
        in_grad_cpu.reshape(output.getShape());
        if (!in_grad_cpu.getStorageBuffer())
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (in_grad_cpu.getStorageBuffer()->size() < total_elements)
        {
            in_grad_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &in_grad_buf = *in_grad_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            in_grad_buf[i] = (a_data[i] > 0.0F) ? grad_data[i] : 0.0F;
        }
        Logger::logMessage(Input_Format{"ops::activation::reluBackward (CPU): elements={}", total_elements},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    else
    {
        const auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        auto contig_self = out_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_gpu;

        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
        if (out_gpu.getDataType() == Data_Type::FLOAT16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        in_grad_gpu.reshape(out_gpu.getShape());

        out_gpu.pushToGraph(out_gpu.getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::RELU_BACKWARD_FP16 : Compute_Pipeline::RELU_BACKWARD,
                            {effective_self->getStorageBuffer(), effective_grad->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                            static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }
}

void leakyRelu(const Tensor &input, Tensor &output, float negative_slope)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }
    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        output.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*out_impl);
        output_cpu.reshape(input.getShape());
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() < total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &out_buf = *output_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float val = a_data[i];
            out_buf[i] = (val >= 0.0f) ? val : (val * negative_slope);
        }

        if (input.getDataType() == Data_Type::FLOAT16)
        {
            if (!output_cpu.getStorageBufferFp16() || output_cpu.getStorageBufferFp16()->size() < total_elements)
            {
                output_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(total_elements));
            }
            convertFp32ToFp16(out_buf.data(), output_cpu.getStorageBufferFp16()->data(), total_elements);
            output_cpu.getStorageBuffer().reset();
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
        }
        output_gpu.reshape(in_gpu.getShape());

        bool use_coopmat = (is_fp16 && Execution_Engine::getInstance().getContext().isCooperativeMatrixEnabled());
        Compute_Pipeline pipe = Compute_Pipeline::LEAKY_RELU_FORWARD;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::LEAKY_RELU_FORWARD_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::LEAKY_RELU_FORWARD_FP16;
        }

        struct PushConstants
        {
            uint32_t total_elements;
            float negative_slope;
        } pc{static_cast<uint32_t>(total_elements), negative_slope};

        uint32_t wg_x = (static_cast<uint32_t>(total_elements) + 255) / 256;
        in_gpu.pushToGraph(pipe,
                           {effective_self->getStorageBuffer(), output_gpu.getStorageBuffer()},
                           pc, wg_x, 1, 1);
        output_gpu.clearHostCache();
        if (is_fp16)
        {
            output_gpu.invalidateFp16Cache();
        }
    }
}

void leakyReluBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient, float negative_slope)
{
    auto in_impl = input.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!in_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (input_gradient.getExecutionTarget() != input.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(input.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    in_impl->validateSameDimensions(*grad_impl);
    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        input_gradient.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*in_grad_impl);
        in_grad_cpu.reshape(input.getShape());
        if (!in_grad_cpu.getStorageBuffer())
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (in_grad_cpu.getStorageBuffer()->size() < total_elements)
        {
            in_grad_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &in_grad_buf = *in_grad_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float val = a_data[i];
            in_grad_buf[i] = (val >= 0.0f) ? grad_data[i] : (grad_data[i] * negative_slope);
        }

        if (input.getDataType() == Data_Type::FLOAT16)
        {
            if (!in_grad_cpu.getStorageBufferFp16() || in_grad_cpu.getStorageBufferFp16()->size() < total_elements)
            {
                in_grad_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(total_elements));
            }
            convertFp32ToFp16(in_grad_buf.data(), in_grad_cpu.getStorageBufferFp16()->data(), total_elements);
            in_grad_cpu.getStorageBuffer().reset();
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT32);
        }
        in_grad_gpu.reshape(in_gpu.getShape());

        bool use_coopmat = (is_fp16 && Execution_Engine::getInstance().getContext().isCooperativeMatrixEnabled());
        Compute_Pipeline pipe = Compute_Pipeline::LEAKY_RELU_BACKWARD;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::LEAKY_RELU_BACKWARD_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::LEAKY_RELU_BACKWARD_FP16;
        }

        struct PushConstants
        {
            uint32_t total_elements;
            float negative_slope;
        } pc{static_cast<uint32_t>(total_elements), negative_slope};

        uint32_t wg_x = (static_cast<uint32_t>(total_elements) + 255) / 256;
        in_gpu.pushToGraph(pipe,
                           {effective_self->getStorageBuffer(), effective_grad->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                           pc, wg_x, 1, 1);
        in_grad_gpu.clearHostCache();
        if (is_fp16)
        {
            in_grad_gpu.invalidateFp16Cache();
        }
    }
}

void gelu(const Tensor &input, Tensor &output)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }
    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        output.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*out_impl);
        output_cpu.reshape(input.getShape());
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() < total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &out_buf = *output_cpu.getStorageBuffer();
        constexpr float sqrt_2_over_pi = 0.7978845608F;
        constexpr float coeff = 0.044715F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            float x = a_data[i];
            float cube = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * cube);
            out_buf[i] = 0.5F * x * (1.0F + std::tanh(inner));
        }
        Logger::logMessage(Input_Format{"ops::activation::gelu (CPU): elements={}", total_elements},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        if (in_gpu.getDataType() == Data_Type::FLOAT16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        output_gpu.reshape(in_gpu.getShape());
        in_gpu.pushToGraph(in_gpu.getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::GELU_FP16 : Compute_Pipeline::GELU,
                          {effective_self->getStorageBuffer(), output_gpu.getStorageBuffer()},
                          static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }
}

void geluBackward(const Tensor &input, const Tensor &output_gradient, Tensor &input_gradient)
{
    auto in_impl = input.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!in_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (input_gradient.getExecutionTarget() != input.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(input.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    in_impl->validateSameDimensions(*grad_impl);
    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        input_gradient.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*in_grad_impl);
        in_grad_cpu.reshape(input.getShape());
        if (!in_grad_cpu.getStorageBuffer())
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (in_grad_cpu.getStorageBuffer()->size() < total_elements)
        {
            in_grad_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &in_grad_buf = *in_grad_cpu.getStorageBuffer();
        constexpr float sqrt_2_over_pi = 0.7978845608F;
        constexpr float coeff = 0.044715F;
        for (size_t i = 0; i < total_elements; ++i)
        {
            float x = a_data[i];
            float x2 = x * x;
            float cube = x2 * x;
            float inner = sqrt_2_over_pi * (x + coeff * cube);
            float tanh_val = std::tanh(inner);
            float sech2_val = 1.0F - tanh_val * tanh_val;
            float d_inner = sqrt_2_over_pi * (1.0F + 3.0F * coeff * x2);
            float d_gelu = 0.5F * (1.0F + tanh_val) + 0.5F * x * sech2_val * d_inner;
            in_grad_buf[i] = grad_data[i] * d_gelu;
        }
        Logger::logMessage(Input_Format{"ops::activation::geluBackward (CPU): elements={}", total_elements},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
        if (in_gpu.getDataType() == Data_Type::FLOAT16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        in_grad_gpu.reshape(in_gpu.getShape());

        in_gpu.pushToGraph(in_gpu.getDataType() == Data_Type::FLOAT16 ? Compute_Pipeline::GELU_BACKWARD_FP16 : Compute_Pipeline::GELU_BACKWARD,
                           {effective_self->getStorageBuffer(), effective_grad->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                           static_cast<uint32_t>(total_elements), (static_cast<uint32_t>(total_elements) + 255) / 256);
    }
}

void softmax(const Tensor &input, Tensor &output)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }
    size_t rows_count = input.getRows();
    size_t cols_count = input.getColumns();
    if (rows_count == 0 || cols_count == 0)
    {
        output.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = input.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*out_impl);
        output_cpu.reshape(rows_count, cols_count);
        size_t total_elements = rows_count * cols_count;
        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() < total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &out_buf = *output_cpu.getStorageBuffer();
        for (size_t r = 0; r < rows_count; ++r)
        {
            float max_val = a_data[r * cols_count];
            for (size_t c = 1; c < cols_count; ++c)
            {
                max_val = std::max(max_val, a_data[r * cols_count + c]);
            }
            float sum_exp = 0.0F;
            for (size_t c = 0; c < cols_count; ++c)
            {
                float exp_val = std::exp(a_data[r * cols_count + c] - max_val);
                out_buf[r * cols_count + c] = exp_val;
                sum_exp += exp_val;
            }
            for (size_t c = 0; c < cols_count; ++c)
            {
                out_buf[r * cols_count + c] /= sum_exp;
            }
        }
        Logger::logMessage(Input_Format{"ops::activation::softmax (CPU): shape=({}x{})",
                                        rows_count, cols_count},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        output_gpu.setDataType(in_gpu.getDataType());
        output_gpu.reshape(rows_count, cols_count);

        struct Softmax_Constants
        {
            uint32_t rows;
            uint32_t columns;
        } constants{static_cast<uint32_t>(rows_count), static_cast<uint32_t>(cols_count)};

        bool is_fp16 = (in_gpu.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SOFTMAX_FP16 : Compute_Pipeline::SOFTMAX;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        in_gpu.pushToGraph(pipe, {buf_self, output_gpu.getStorageBuffer()}, constants, constants.rows, 1, 1);
    }
}

void softmaxBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient)
{
    auto out_impl = output.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!out_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (input_gradient.getExecutionTarget() != output.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(output.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    out_impl->validateSameDimensions(*grad_impl);
    size_t rows_count = output.getRows();
    size_t cols_count = output.getColumns();
    if (rows_count == 0 || cols_count == 0)
    {
        input_gradient.reshape(output.getShape());
        return;
    }

    if (output.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &a_data = output.getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*in_grad_impl);
        in_grad_cpu.reshape(rows_count, cols_count);
        size_t total_elements = rows_count * cols_count;
        if (!in_grad_cpu.getStorageBuffer())
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (in_grad_cpu.getStorageBuffer()->size() < total_elements)
        {
            in_grad_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &in_grad_buf = *in_grad_cpu.getStorageBuffer();
        for (size_t r = 0; r < rows_count; ++r)
        {
            for (size_t i = 0; i < cols_count; ++i)
            {
                float acc = 0.0F;
                for (size_t j = 0; j < cols_count; ++j)
                {
                    float delta = (i == j) ? 1.0F : 0.0F;
                    float jac = a_data[r * cols_count + i] * (delta - a_data[r * cols_count + j]);
                    acc += jac * grad_data[r * cols_count + j];
                }
                in_grad_buf[r * cols_count + i] = acc;
            }
        }
        Logger::logMessage(Input_Format{"ops::activation::softmaxBackward (CPU): shape=({}x{})",
                                        rows_count, cols_count},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    else
    {
        const auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        auto contig_self = out_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_gpu;

        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
        in_grad_gpu.setDataType(out_gpu.getDataType());
        in_grad_gpu.reshape(rows_count, cols_count);

        struct Softmax_Constants
        {
            uint32_t rows;
            uint32_t columns;
        } constants{static_cast<uint32_t>(rows_count), static_cast<uint32_t>(cols_count)};

        bool is_fp16 = (out_gpu.getDataType() == Data_Type::FLOAT16 || effective_grad->getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::SOFTMAX_BACKWARD_FP16 : Compute_Pipeline::SOFTMAX_BACKWARD;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

        out_gpu.pushToGraph(pipe, {buf_self, buf_grad, in_grad_gpu.getStorageBuffer()}, constants, constants.rows, 1, 1);
    }
}

void sigmoid(const Tensor &input, Tensor &output)
{
    auto in_impl = input.getImplementation();
    auto out_impl = output.getImplementation();
    if (!in_impl || !out_impl)
    {
        return;
    }
    if (output.getExecutionTarget() != input.getExecutionTarget())
    {
        output.setExecutionTarget(input.getExecutionTarget());
        out_impl = output.getImplementation();
    }

    size_t total_elements = input.getTotalElements();
    if (total_elements == 0)
    {
        output.reshape(input.getShape());
        return;
    }

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &in_data = input.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*out_impl);
        output_cpu.reshape(input.getShape());

        if (!output_cpu.getStorageBuffer())
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (output_cpu.getStorageBuffer()->size() < total_elements)
        {
            output_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &out_buf = *output_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            out_buf[i] = 1.0f / (1.0f + std::exp(-in_data[i]));
        }

        if (input.getDataType() == Data_Type::FLOAT16)
        {
            if (!output_cpu.getStorageBufferFp16() || output_cpu.getStorageBufferFp16()->size() < total_elements)
            {
                output_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(total_elements));
            }
            convertFp32ToFp16(out_buf.data(), output_cpu.getStorageBufferFp16()->data(), total_elements);
            output_cpu.getStorageBuffer().reset();
        }

        Logger::logMessage(Input_Format{"ops::activation::sigmoid (CPU): elements={}", total_elements},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE);
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            output_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            output_gpu.setDataType(Data_Type::FLOAT32);
        }
        output_gpu.reshape(in_gpu.getShape());

        bool use_coopmat = (is_fp16 && Execution_Engine::getInstance().getContext().isCooperativeMatrixEnabled());
        Compute_Pipeline pipe = Compute_Pipeline::SIGMOID_FORWARD;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::SIGMOID_FORWARD_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::SIGMOID_FORWARD_FP16;
        }

        struct PushConstants
        {
            uint32_t total_elements;
        } pc{static_cast<uint32_t>(total_elements)};

        uint32_t wg_x = (static_cast<uint32_t>(total_elements) + 255) / 256;
        in_gpu.pushToGraph(pipe,
                           {effective_self->getStorageBuffer(), output_gpu.getStorageBuffer()},
                           pc, wg_x, 1, 1);
        output_gpu.clearHostCache();
        if (is_fp16)
        {
            output_gpu.invalidateFp16Cache();
        }
    }
}

void sigmoidBackward(const Tensor &output, const Tensor &output_gradient, Tensor &input_gradient)
{
    auto out_impl = output.getImplementation();
    auto grad_impl = output_gradient.getImplementation();
    auto in_grad_impl = input_gradient.getImplementation();
    if (!out_impl || !grad_impl || !in_grad_impl)
    {
        return;
    }
    if (input_gradient.getExecutionTarget() != output.getExecutionTarget())
    {
        input_gradient.setExecutionTarget(output.getExecutionTarget());
        in_grad_impl = input_gradient.getImplementation();
    }
    if (output.getTotalElements() != output_gradient.getTotalElements())
    {
        out_impl->validateSameDimensions(*grad_impl);
        return;
    }

    size_t total_elements = output.getTotalElements();
    if (total_elements == 0)
    {
        input_gradient.reshape(output_gradient.getShape());
        return;
    }

    if (output.getExecutionTarget() == Execution_Target::CPU)
    {
        const auto &out_data = output.getData();
        const auto &grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*in_grad_impl);
        in_grad_cpu.reshape(output_gradient.getShape());

        if (!in_grad_cpu.getStorageBuffer())
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        }
        else if (in_grad_cpu.getStorageBuffer()->size() < total_elements)
        {
            in_grad_cpu.getStorageBuffer()->resize(total_elements, 0.0f);
        }

        auto &in_grad_buf = *in_grad_cpu.getStorageBuffer();
        for (size_t i = 0; i < total_elements; ++i)
        {
            float s = out_data[i];
            in_grad_buf[i] = grad_data[i] * s * (1.0f - s);
        }

        if (output.getDataType() == Data_Type::FLOAT16)
        {
            if (!in_grad_cpu.getStorageBufferFp16() || in_grad_cpu.getStorageBufferFp16()->size() < total_elements)
            {
                in_grad_cpu.setStorageBufferFp16(std::make_shared<std::vector<float16_t>>(total_elements));
            }
            convertFp32ToFp16(in_grad_buf.data(), in_grad_cpu.getStorageBufferFp16()->data(), total_elements);
            in_grad_cpu.getStorageBuffer().reset();
        }

        Logger::logMessage(Input_Format{"ops::activation::sigmoidBackward (CPU): elements={}", total_elements},
                           Log_Level::LOG_DEBUG, true, 0, Log_Feature::ACTIVATION_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
    }
    else
    {
        const auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*out_impl);
        auto contig_self = out_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_gpu;

        const auto &grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_grad = grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &grad_gpu;

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*in_grad_impl);
        bool is_fp16 = (output.getDataType() == Data_Type::FLOAT16);
        if (is_fp16)
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT16);
        }
        else
        {
            in_grad_gpu.setDataType(Data_Type::FLOAT32);
        }
        in_grad_gpu.reshape(grad_gpu.getShape());

        bool use_coopmat = (is_fp16 && Execution_Engine::getInstance().getContext().isCooperativeMatrixEnabled());
        Compute_Pipeline pipe = Compute_Pipeline::SIGMOID_BACKWARD;
        if (use_coopmat)
        {
            pipe = Compute_Pipeline::SIGMOID_BACKWARD_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::SIGMOID_BACKWARD_FP16;
        }

        struct PushConstants
        {
            uint32_t total_elements;
        } pc{static_cast<uint32_t>(total_elements)};

        uint32_t wg_x = (static_cast<uint32_t>(total_elements) + 255) / 256;
        out_gpu.pushToGraph(pipe,
                            {effective_self->getStorageBuffer(), effective_grad->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                            pc, wg_x, 1, 1);
        in_grad_gpu.clearHostCache();
        if (is_fp16)
        {
            in_grad_gpu.invalidateFp16Cache();
        }
    }
}

} // namespace ops::activation

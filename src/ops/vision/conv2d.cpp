#include "ops/vision/conv2d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::vision
{

void conv2d(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output,
            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
            uint32_t output_channels, uint32_t kernel_size,
            uint32_t stride, uint32_t padding,
            Tensor *scratch)
{
    auto in_impl = input.getImplementation();
    auto w_impl = weights.getImplementation();
    auto b_impl = biases.getImplementation();
    if (!in_impl || !w_impl || !b_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
    uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;
    Shape out_shape{batch_size, out_h * out_w * output_channels};

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        if (output.getExecutionTarget() != Execution_Target::CPU ||
            output.getDataType() != input.getDataType() ||
            output.getShape() != out_shape)
        {
            output = Tensor(out_shape, input.getDataType(), Execution_Target::CPU);
        }
        else
        {
            output.reshape(out_shape);
        }

        const auto &in_data = input.getData();
        const auto &w_data = weights.getData();
        const auto &b_data = biases.getData();
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        if (!output_cpu.getStorageBuffer() || output_cpu.getStorageBuffer()->size() < batch_size * out_h * out_w * output_channels)
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * out_h * out_w * output_channels, 0.0f);
        }

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t oc = 0; oc < output_channels; ++oc)
                    {
                        float sum = b_data[oc];
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    for (uint32_t ic = 0; ic < input_channels; ++ic)
                                    {
                                        size_t in_idx = n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic;
                                        size_t w_idx = ky * kernel_size * input_channels * output_channels + kx * input_channels * output_channels + ic * output_channels + oc;
                                        sum += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        (*output_cpu.getStorageBuffer())[n * out_h * out_w * output_channels + oh * out_w * output_channels + ow * output_channels + oc] = sum;
                    }
                }
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*w_impl);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*b_impl);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 && effective_w->getDataType() == Data_Type::FLOAT16);
        Data_Type target_dtype = is_fp16 ? Data_Type::FLOAT16 : input.getDataType();

        if (output.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output.getDataType() != target_dtype ||
            output.getShape() != out_shape)
        {
            output = Tensor(out_shape, target_dtype, Execution_Target::VULKAN_GPU);
        }
        else
        {
            output.reshape(out_shape);
        }

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

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

        constexpr uint64_t MAX_IM2COL_SCRATCH_BYTES = 128 * 1024 * 1024;
        uint64_t scratch_bytes = static_cast<uint64_t>(M) * K * sizeof(uint16_t);


        if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled() && scratch_bytes <= MAX_IM2COL_SCRATCH_BYTES)
        {
            Gpu_Tensor_Impl *effective_scratch = nullptr;
            std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
            if (scratch != nullptr)
            {
                effective_scratch = &Gpu_Tensor_Impl::castToGpu(*scratch->getImplementation());
                effective_scratch->setDataType(Data_Type::FLOAT16);
                effective_scratch->reshape(M, K);
            }
            else
            {
                local_scratch = std::make_shared<Gpu_Tensor_Impl>(Shape{M, K}, Data_Type::FLOAT16);
                effective_scratch = local_scratch.get();
            }

            // 1. Unroll input patches directly into [M, K]
            effective_self->pushToGraph(Compute_Pipeline::CONV2D_IM2COL_FP16,
                                        {effective_self->getStorageBuffer(), effective_scratch->getStorageBuffer()},
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

            effective_self->pushToGraph(Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16,
                                        {effective_scratch->getStorageBuffer(), effective_w->getStorageBuffer(), effective_b->getStorageBuffer(), output_gpu.getStorageBuffer()},
                                        matmul_constants,
                                        (N + 15) / 16, (M + 15) / 16, 1);

            output_gpu.reshape(batch_size, out_h * out_w * output_channels);
            return;
        }

        output_gpu.reshape(batch_size, out_h * out_w * output_channels);

        Compute_Pipeline pipeline = is_fp16 ? Compute_Pipeline::CONV2D_FORWARD_PASS_FP16 : Compute_Pipeline::CONV2D_FORWARD_PASS;
        effective_self->pushToGraph(pipeline,
                                    {effective_self->getStorageBuffer(), effective_w->getStorageBuffer(), effective_b->getStorageBuffer(), output_gpu.getStorageBuffer()},
                                    constants,
                                    (output_channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }
}

void conv2dBackwardInput(const Tensor &output_gradient, const Tensor &weights, Tensor &input_gradient,
                         uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                         uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                         uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    auto out_grad_impl = output_gradient.getImplementation();
    auto w_impl = weights.getImplementation();
    if (!out_grad_impl || !w_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(output_gradient.getRows());
    Shape in_grad_shape{batch_size, input_height * input_width * input_channels};

    if (output_gradient.getExecutionTarget() == Execution_Target::CPU)
    {
        if (input_gradient.getExecutionTarget() != Execution_Target::CPU ||
            input_gradient.getDataType() != output_gradient.getDataType() ||
            input_gradient.getShape() != in_grad_shape)
        {
            input_gradient = Tensor(in_grad_shape, output_gradient.getDataType(), Execution_Target::CPU);
        }
        else
        {
            input_gradient.reshape(in_grad_shape);
        }

        const auto &out_grad_data = output_gradient.getData();
        const auto &w_data = weights.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        if (!in_grad_cpu.getStorageBuffer() || in_grad_cpu.getStorageBuffer()->size() < batch_size * input_height * input_width * input_channels)
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * input_height * input_width * input_channels, 0.0f);
        }

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t ic = 0; ic < input_channels; ++ic)
                    {
                        float sum = 0.0f;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int oh_calc = static_cast<int>(ih + padding - ky);
                                int ow_calc = static_cast<int>(iw + padding - kx);
                                if (oh_calc >= 0 && oh_calc % static_cast<int>(stride) == 0 && ow_calc >= 0 && ow_calc % static_cast<int>(stride) == 0)
                                {
                                    uint32_t oh = static_cast<uint32_t>(oh_calc) / stride;
                                    uint32_t ow = static_cast<uint32_t>(ow_calc) / stride;
                                    if (oh < output_height && ow < output_width)
                                    {
                                        for (uint32_t oc = 0; oc < output_channels; ++oc)
                                        {
                                            size_t out_grad_idx = n * output_height * output_width * output_channels + oh * output_width * output_channels + ow * output_channels + oc;
                                            size_t w_idx = ky * kernel_size * input_channels * output_channels + kx * input_channels * output_channels + ic * output_channels + oc;
                                            sum += out_grad_data[out_grad_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        (*in_grad_cpu.getStorageBuffer())[n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic] = sum;
                    }
                }
            }
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
        auto contig_self = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_grad_gpu;

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*w_impl);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 && effective_w->getDataType() == Data_Type::FLOAT16);
        Data_Type target_dtype = is_fp16 ? Data_Type::FLOAT16 : output_gradient.getDataType();

        if (input_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            input_gradient.getDataType() != target_dtype ||
            input_gradient.getShape() != in_grad_shape)
        {
            input_gradient = Tensor(in_grad_shape, target_dtype, Execution_Target::VULKAN_GPU);
        }
        else
        {
            input_gradient.reshape(in_grad_shape);
        }

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

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
        effective_self->pushToGraph(pipeline,
                                    {effective_self->getStorageBuffer(), effective_w->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                    constants,
                                    (input_channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }
}

void conv2dBackwardWeight(const Tensor &input, const Tensor &output_gradient,
                          Tensor &weight_gradient, Tensor &bias_gradient,
                          uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_size, uint32_t stride, uint32_t padding,
                          Tensor *im2col_scratch)
{
    auto in_impl = input.getImplementation();
    auto out_grad_impl = output_gradient.getImplementation();
    if (!in_impl || !out_grad_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint32_t K = kernel_size * kernel_size * input_channels;
    Shape w_shape{1, K * output_channels};
    Shape b_shape{1, output_channels};

    if (input.getExecutionTarget() == Execution_Target::CPU)
    {
        if (weight_gradient.getExecutionTarget() != Execution_Target::CPU ||
            weight_gradient.getDataType() != input.getDataType() ||
            weight_gradient.getShape() != w_shape)
        {
            weight_gradient = Tensor(w_shape, input.getDataType(), Execution_Target::CPU);
        }
        else
        {
            weight_gradient.reshape(w_shape);
        }

        if (bias_gradient.getExecutionTarget() != Execution_Target::CPU ||
            bias_gradient.getDataType() != input.getDataType() ||
            bias_gradient.getShape() != b_shape)
        {
            bias_gradient = Tensor(b_shape, input.getDataType(), Execution_Target::CPU);
        }
        else
        {
            bias_gradient.reshape(b_shape);
        }

        const auto &in_data = input.getData();
        const auto &out_grad_data = output_gradient.getData();
        auto &w_grad_cpu = Cpu_Tensor_Impl::castToCpu(*weight_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*bias_gradient.getImplementation());

        if (!w_grad_cpu.getStorageBuffer() || w_grad_cpu.getStorageBuffer()->size() < K * output_channels)
        {
            w_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(K * output_channels, 0.0f);
        }
        if (!b_grad_cpu.getStorageBuffer() || b_grad_cpu.getStorageBuffer()->size() < output_channels)
        {
            b_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(output_channels, 0.0f);
        }

        std::fill(w_grad_cpu.getStorageBuffer()->begin(), w_grad_cpu.getStorageBuffer()->end(), 0.0f);
        std::fill(b_grad_cpu.getStorageBuffer()->begin(), b_grad_cpu.getStorageBuffer()->end(), 0.0f);

        for (uint32_t oc = 0; oc < output_channels; ++oc)
        {
            for (uint32_t ic = 0; ic < input_channels; ++ic)
            {
                for (uint32_t ky = 0; ky < kernel_size; ++ky)
                {
                    for (uint32_t kx = 0; kx < kernel_size; ++kx)
                    {
                        float w_sum = 0.0f;
                        for (uint32_t n = 0; n < batch_size; ++n)
                        {
                            for (uint32_t oh = 0; oh < output_height; ++oh)
                            {
                                for (uint32_t ow = 0; ow < output_width; ++ow)
                                {
                                    size_t out_grad_idx = n * output_height * output_width * output_channels + oh * output_width * output_channels + ow * output_channels + oc;
                                    float grad_val = out_grad_data[out_grad_idx];
                                    if (ic == 0 && ky == 0 && kx == 0)
                                    {
                                        (*b_grad_cpu.getStorageBuffer())[oc] += grad_val;
                                    }
                                    int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                    int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                    if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                    {
                                        size_t in_idx = n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic;
                                        w_sum += in_data[in_idx] * grad_val;
                                    }
                                }
                            }
                        }
                        size_t w_idx = ky * kernel_size * input_channels * output_channels + kx * input_channels * output_channels + ic * output_channels + oc;
                        (*w_grad_cpu.getStorageBuffer())[w_idx] = w_sum;
                    }
                }
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 && effective_grad->getDataType() == Data_Type::FLOAT16);
        Data_Type target_dtype = Data_Type::FLOAT32;

        if (weight_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            weight_gradient.getDataType() != target_dtype ||
            weight_gradient.getShape() != w_shape)
        {
            weight_gradient = Tensor(w_shape, target_dtype, Execution_Target::VULKAN_GPU);
        }
        else
        {
            weight_gradient.reshape(w_shape);
        }

        if (bias_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            bias_gradient.getDataType() != target_dtype ||
            bias_gradient.getShape() != b_shape)
        {
            bias_gradient = Tensor(b_shape, target_dtype, Execution_Target::VULKAN_GPU);
        }
        else
        {
            bias_gradient.reshape(b_shape);
        }

        auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*bias_gradient.getImplementation());

        uint32_t M = batch_size * output_height * output_width;

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

        constexpr uint64_t MAX_IM2COL_SCRATCH_BYTES = 128 * 1024 * 1024;
        uint64_t scratch_bytes = static_cast<uint64_t>(K) * M * (is_fp16 ? sizeof(uint16_t) : sizeof(float));
        if (scratch_bytes > MAX_IM2COL_SCRATCH_BYTES)
        {
            Compute_Pipeline direct_pipeline = is_fp16
                ? Compute_Pipeline::CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT_FP16
                : Compute_Pipeline::CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT;
            effective_self->pushToGraph(direct_pipeline,
                                        {effective_self->getStorageBuffer(), effective_grad->getStorageBuffer(), w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                        constants,
                                        (output_channels + 15) / 16, (input_channels + 15) / 16, kernel_size * kernel_size);
            w_grad_gpu.clearHostCache();
            b_grad_gpu.clearHostCache();
            return;
        }

        std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
        Gpu_Tensor_Impl *effective_scratch = nullptr;

        if (im2col_scratch != nullptr)
        {
            effective_scratch = &Gpu_Tensor_Impl::castToGpu(*im2col_scratch->getImplementation());
            effective_scratch->setDataType(is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            effective_scratch->reshape(K, M);
        }
        else
        {
            local_scratch = std::make_shared<Gpu_Tensor_Impl>(Shape{K, M}, is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            effective_scratch = local_scratch.get();
        }

        // 1. im2col_transposed: unrolls input patches directly into [K, M]
        Compute_Pipeline im2col_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_IM2COL_TRANSPOSED_FP16 : Compute_Pipeline::CONV2D_IM2COL_TRANSPOSED;
        effective_self->pushToGraph(im2col_pipeline,
                                    {effective_self->getStorageBuffer(), effective_scratch->getStorageBuffer()},
                                    constants,
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
            effective_self->pushToGraph(dw_pipeline,
                                        {effective_scratch->getStorageBuffer(), effective_grad->getStorageBuffer(), w_grad_gpu.getStorageBuffer()},
                                        matmul_dims,
                                        (output_channels + 15) / 16, (K + 15) / 16, 1);
        }
        else
        {
            effective_self->pushToGraph(Compute_Pipeline::MATMUL,
                                        {effective_scratch->getStorageBuffer(), effective_grad->getStorageBuffer(), w_grad_gpu.getStorageBuffer()},
                                        matmul_dims,
                                        (output_channels + 15) / 16, (K + 15) / 16, 1);
        }

        // 3. Bias gradient: dB = sum over M of out_grad [M, output_channels] -> [output_channels]
        Compute_Pipeline bias_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_BIAS_GRADIENT_FP16 : Compute_Pipeline::CONV2D_BIAS_GRADIENT;
        effective_self->pushToGraph(bias_pipeline,
                                    {effective_grad->getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                    constants,
                                    output_channels, 1, 1);
        w_grad_gpu.clearHostCache();
        b_grad_gpu.clearHostCache();
    }
}

} // namespace ops::vision

#include "ops/vision/pooling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"

namespace ops::vision
{

void maxpool2d(const Tensor &input, Tensor &output, Tensor &output_mask,
               uint32_t input_height, uint32_t input_width, uint32_t channels,
               uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
    uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;
    Shape out_shape{batch_size, out_h * out_w * channels};

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

        if (output_mask.getExecutionTarget() != Execution_Target::CPU ||
            output_mask.getDataType() != Data_Type::FLOAT32 ||
            output_mask.getShape() != out_shape)
        {
            output_mask = Tensor(out_shape, Data_Type::FLOAT32, Execution_Target::CPU);
        }
        else
        {
            output_mask.reshape(out_shape);
        }

        auto &res_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        auto &mask_cpu = Cpu_Tensor_Impl::castToCpu(*output_mask.getImplementation());

        if (!res_cpu.getStorageBuffer() || res_cpu.getStorageBuffer()->size() < batch_size * out_h * out_w * channels)
        {
            res_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * out_h * out_w * channels, 0.0f);
        }
        if (!mask_cpu.getStorageBuffer() || mask_cpu.getStorageBuffer()->size() < batch_size * out_h * out_w * channels)
        {
            mask_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * out_h * out_w * channels, 0.0f);
        }

        const auto &in_data = input.getData();

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float max_val = -3.402823466e+38F;
                        float max_idx = 0.0F;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                                    float val = in_data[in_idx];
                                    if (val > max_val)
                                    {
                                        max_val = val;
                                        max_idx = static_cast<float>(in_idx);
                                    }
                                }
                            }
                        }
                        size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                        (*res_cpu.getStorageBuffer())[out_idx] = max_val;
                        (*mask_cpu.getStorageBuffer())[out_idx] = max_idx;
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

        if (output.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output.getDataType() != input.getDataType() ||
            output.getShape() != out_shape)
        {
            output = Tensor(out_shape, input.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            output.reshape(out_shape);
        }

        if (output_mask.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output_mask.getDataType() != Data_Type::FLOAT32 ||
            output_mask.getShape() != out_shape)
        {
            output_mask = Tensor(out_shape, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
        }
        else
        {
            output_mask.reshape(out_shape);
        }

        auto &res_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
        auto &mask_gpu = Gpu_Tensor_Impl::castToGpu(*output_mask.getImplementation());

        struct Pool_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, channels, out_h, out_w, kernel_size, stride, padding};

        Compute_Pipeline pipe = (input.getDataType() == Data_Type::FLOAT16)
            ? Compute_Pipeline::MAXPOOL2D_FORWARD_FP16
            : Compute_Pipeline::MAXPOOL2D_FORWARD;

        effective_self->pushToGraph(pipe,
                                    {effective_self->getStorageBuffer(), res_gpu.getStorageBuffer(), mask_gpu.getStorageBuffer()},
                                    constants,
                                    (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }
}

void maxpool2dBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient,
                       uint32_t input_height, uint32_t input_width, uint32_t channels,
                       uint32_t output_height, uint32_t output_width,
                       uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    auto out_grad_impl = output_gradient.getImplementation();
    auto mask_impl = mask.getImplementation();
    if (!out_grad_impl || !mask_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(output_gradient.getRows());
    Shape in_grad_shape{batch_size, input_height * input_width * channels};

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

        const auto &mask_data = mask.getData();
        const auto &out_grad_data = output_gradient.getData();
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        if (!in_grad_cpu.getStorageBuffer() || in_grad_cpu.getStorageBuffer()->size() < batch_size * input_height * input_width * channels)
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * input_height * input_width * channels, 0.0f);
        }

        std::fill(in_grad_cpu.getStorageBuffer()->begin(), in_grad_cpu.getStorageBuffer()->end(), 0.0f);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                        float acc_grad = 0.0f;
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
                                        size_t out_idx = n * output_height * output_width * channels + oh * output_width * channels + ow * channels + c;
                                        if (static_cast<size_t>(mask_data[out_idx]) == in_idx)
                                        {
                                            acc_grad += out_grad_data[out_idx];
                                        }
                                    }
                                }
                            }
                        }
                        (*in_grad_cpu.getStorageBuffer())[in_idx] = acc_grad;
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

        const auto &mask_gpu = Gpu_Tensor_Impl::castToGpu(*mask_impl);
        auto contig_mask = mask_gpu.ensureContiguousSelf();
        const auto *effective_mask = contig_mask ? contig_mask.get() : &mask_gpu;

        if (input_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            input_gradient.getDataType() != output_gradient.getDataType() ||
            input_gradient.getShape() != in_grad_shape)
        {
            input_gradient = Tensor(in_grad_shape, output_gradient.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            input_gradient.reshape(in_grad_shape);
        }

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Pool_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding};

        Compute_Pipeline pipe = (output_gradient.getDataType() == Data_Type::FLOAT16)
            ? Compute_Pipeline::MAXPOOL2D_BACKWARD_FP16
            : Compute_Pipeline::MAXPOOL2D_BACKWARD;

        effective_self->pushToGraph(pipe,
                                    {effective_mask->getStorageBuffer(), effective_self->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                    constants,
                                    (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }
}

void avgpool2d(const Tensor &input, Tensor &output,
               uint32_t input_height, uint32_t input_width, uint32_t channels,
               uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint32_t out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
    uint32_t out_w = (input_width + 2 * padding - kernel_size) / stride + 1;
    Shape out_shape{batch_size, out_h * out_w * channels};

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

        auto &res_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        if (!res_cpu.getStorageBuffer() || res_cpu.getStorageBuffer()->size() < batch_size * out_h * out_w * channels)
        {
            res_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * out_h * out_w * channels, 0.0f);
        }

        const auto &in_data = input.getData();
        float area = static_cast<float>(kernel_size * kernel_size);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float sum_val = 0.0f;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                                    sum_val += in_data[in_idx];
                                }
                            }
                        }
                        size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                        (*res_cpu.getStorageBuffer())[out_idx] = sum_val / area;
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

        if (output.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output.getDataType() != input.getDataType() ||
            output.getShape() != out_shape)
        {
            output = Tensor(out_shape, input.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            output.reshape(out_shape);
        }

        auto &res_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        struct Pool_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, channels, out_h, out_w, kernel_size, stride, padding};

        Compute_Pipeline pipe = (input.getDataType() == Data_Type::FLOAT16)
            ? Compute_Pipeline::AVGPOOL2D_FORWARD_FP16
            : Compute_Pipeline::AVGPOOL2D_FORWARD;

        effective_self->pushToGraph(pipe,
                                    {effective_self->getStorageBuffer(), res_gpu.getStorageBuffer()},
                                    constants,
                                    (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    }
}

void avgpool2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                       uint32_t input_height, uint32_t input_width, uint32_t channels,
                       uint32_t output_height, uint32_t output_width,
                       uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    auto grad_impl = output_gradient.getImplementation();
    if (!grad_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(output_gradient.getRows());
    Shape in_shape{batch_size, input_height * input_width * channels};

    if (output_gradient.getExecutionTarget() == Execution_Target::CPU)
    {
        if (input_gradient.getExecutionTarget() != Execution_Target::CPU ||
            input_gradient.getDataType() != output_gradient.getDataType() ||
            input_gradient.getShape() != in_shape)
        {
            input_gradient = Tensor(in_shape, output_gradient.getDataType(), Execution_Target::CPU);
        }
        else
        {
            input_gradient.reshape(in_shape);
        }

        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
        if (!in_grad_cpu.getStorageBuffer() || in_grad_cpu.getStorageBuffer()->size() < batch_size * input_height * input_width * channels)
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * input_height * input_width * channels, 0.0f);
        }
        else
        {
            std::fill(in_grad_cpu.getStorageBuffer()->begin(), in_grad_cpu.getStorageBuffer()->end(), 0.0f);
        }

        const auto &out_grad_data = output_gradient.getData();
        float area = static_cast<float>(kernel_size * kernel_size);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float grad_val = 0.0f;
                        int pad = static_cast<int>(padding);
                        int k_sz = static_cast<int>(kernel_size);
                        int str = static_cast<int>(stride);

                        int ky_start = (static_cast<int>(ih) + pad) % str;
                        int kx_start = (static_cast<int>(iw) + pad) % str;

                        for (int ky = ky_start; ky < k_sz; ky += str)
                        {
                            int out_y_calc = static_cast<int>(ih) + pad - ky;
                            if (out_y_calc < 0) continue;
                            uint32_t oh = static_cast<uint32_t>(out_y_calc / str);
                            if (oh < output_height)
                            {
                                for (int kx = kx_start; kx < k_sz; kx += str)
                                {
                                    int out_x_calc = static_cast<int>(iw) + pad - kx;
                                    if (out_x_calc < 0) continue;
                                    uint32_t ow = static_cast<uint32_t>(out_x_calc / str);
                                    if (ow < output_width)
                                    {
                                        size_t out_idx = n * output_height * output_width * channels + oh * output_width * channels + ow * channels + c;
                                        grad_val += out_grad_data[out_idx] / area;
                                    }
                                }
                            }
                        }

                        size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                        (*in_grad_cpu.getStorageBuffer())[in_idx] = grad_val;
                    }
                }
            }
        }
    }
    else
    {
        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*grad_impl);
        auto contig_self = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_grad_gpu;

        if (input_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            input_gradient.getDataType() != output_gradient.getDataType() ||
            input_gradient.getShape() != in_shape)
        {
            input_gradient = Tensor(in_shape, output_gradient.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            input_gradient.reshape(in_shape);
        }

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Pool_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel_size;
            uint32_t stride;
            uint32_t padding;
        } constants{batch_size, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding};

        Compute_Pipeline pipe = (output_gradient.getDataType() == Data_Type::FLOAT16)
            ? Compute_Pipeline::AVGPOOL2D_BACKWARD_FP16
            : Compute_Pipeline::AVGPOOL2D_BACKWARD;

        effective_self->pushToGraph(pipe,
                                    {effective_self->getStorageBuffer(), in_grad_gpu.getStorageBuffer()},
                                    constants,
                                    (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }
}

void globalAvgPool2d(const Tensor &input, Tensor &output,
                     uint32_t input_height, uint32_t input_width, uint32_t channels)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    Shape out_shape{batch_size, channels};

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
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());

        if (!output_cpu.getStorageBuffer() || output_cpu.getStorageBuffer()->size() < batch_size * channels)
        {
            output_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * channels, 0.0f);
        }

        float area = static_cast<float>(input_height * input_width);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                float sum = 0.0f;
                for (uint32_t h = 0; h < input_height; ++h)
                {
                    for (uint32_t w = 0; w < input_width; ++w)
                    {
                        sum += in_data[n * input_height * input_width * channels + h * input_width * channels + w * channels + c];
                    }
                }
                (*output_cpu.getStorageBuffer())[n * channels + c] = sum / area;
            }
        }
    }
    else
    {
        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        if (output.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output.getDataType() != input.getDataType() ||
            output.getShape() != out_shape)
        {
            output = Tensor(out_shape, input.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            output.reshape(out_shape);
        }

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        struct Avg_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
        } constants{batch_size, input_height, input_width, channels};

        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD_FP16 : Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        effective_self->pushToGraph(pipe,
                                    {buf_self, output_gpu.getStorageBuffer()},
                                    constants,
                                    (channels + 255) / 256, batch_size, 1);
    }
}

void globalAvgPool2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t channels)
{
    auto out_grad_impl = output_gradient.getImplementation();
    if (!out_grad_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(output_gradient.getRows());
    Shape in_grad_shape{batch_size, input_height * input_width * channels};

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
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());

        if (!in_grad_cpu.getStorageBuffer() || in_grad_cpu.getStorageBuffer()->size() < batch_size * input_height * input_width * channels)
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(batch_size * input_height * input_width * channels, 0.0f);
        }

        float area = static_cast<float>(input_height * input_width);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                float scaled = out_grad_data[n * channels + c] / area;
                for (uint32_t h = 0; h < input_height; ++h)
                {
                    for (uint32_t w = 0; w < input_width; ++w)
                    {
                        (*in_grad_cpu.getStorageBuffer())[n * input_height * input_width * channels + h * input_width * channels + w * channels + c] = scaled;
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

        if (input_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            input_gradient.getDataType() != output_gradient.getDataType() ||
            input_gradient.getShape() != in_grad_shape)
        {
            input_gradient = Tensor(in_grad_shape, output_gradient.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            input_gradient.reshape(in_grad_shape);
        }

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Avg_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
        } constants{batch_size, input_height, input_width, channels};

        bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16);
        Compute_Pipeline pipe = is_fp16 ? Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD_FP16 : Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD;
        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        effective_self->pushToGraph(pipe,
                                    {buf_self, in_grad_gpu.getStorageBuffer()},
                                    constants,
                                    (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    }
}

void upsample2d(const Tensor &input, Tensor &output,
                uint32_t input_height, uint32_t input_width, uint32_t channels,
                uint32_t scale_factor)
{
    auto in_impl = input.getImplementation();
    if (!in_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint32_t out_h = input_height * scale_factor;
    uint32_t out_w = input_width * scale_factor;
    Shape out_shape{batch_size, out_h * out_w * channels};

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

        auto &res_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        size_t total_out_elements = batch_size * out_h * out_w * channels;
        if (!res_cpu.getStorageBuffer() || res_cpu.getStorageBuffer()->size() < total_out_elements)
        {
            res_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_out_elements, 0.0f);
        }

        const auto &in_data = input.getData();
        auto &out_data = *res_cpu.getStorageBuffer();

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                        float val = in_data[in_idx];

                        for (uint32_t sy = 0; sy < scale_factor; ++sy)
                        {
                            uint32_t oh = ih * scale_factor + sy;
                            for (uint32_t sx = 0; sx < scale_factor; ++sx)
                            {
                                uint32_t ow = iw * scale_factor + sx;
                                size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                                out_data[out_idx] = val;
                            }
                        }
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

        if (output.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            output.getDataType() != input.getDataType() ||
            output.getShape() != out_shape)
        {
            output = Tensor(out_shape, input.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            output.reshape(out_shape);
        }

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        struct Upsample_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t scale_factor;
        } constants{batch_size, input_height, input_width, channels, out_h, out_w, scale_factor};

        bool is_fp16 = (input.getDataType() == Data_Type::FLOAT16);
        bool use_coop = false; // Test standard FP16 pipeline

        Compute_Pipeline pipe;
        if (use_coop)
        {
            pipe = Compute_Pipeline::UPSAMPLE2D_FORWARD_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::UPSAMPLE2D_FORWARD_FP16;
        }
        else
        {
            pipe = Compute_Pipeline::UPSAMPLE2D_FORWARD;
        }

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        if (use_coop)
        {
            effective_self->pushToGraph(pipe,
                                        {buf_self, output_gpu.getStorageBuffer()},
                                        constants,
                                        (channels + 15) / 16, (out_h * out_w + 15) / 16, batch_size);
        }
        else
        {
            effective_self->pushToGraph(pipe,
                                        {buf_self, output_gpu.getStorageBuffer()},
                                        constants,
                                        (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
        }
    }
}

void upsample2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                        uint32_t input_height, uint32_t input_width, uint32_t channels,
                        uint32_t scale_factor)
{
    auto out_grad_impl = output_gradient.getImplementation();
    if (!out_grad_impl) return;

    uint32_t batch_size = static_cast<uint32_t>(output_gradient.getRows());
    uint32_t out_h = input_height * scale_factor;
    uint32_t out_w = input_width * scale_factor;
    Shape in_grad_shape{batch_size, input_height * input_width * channels};

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

        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
        size_t total_in_elements = batch_size * input_height * input_width * channels;
        if (!in_grad_cpu.getStorageBuffer() || in_grad_cpu.getStorageBuffer()->size() < total_in_elements)
        {
            in_grad_cpu.getStorageBuffer() = std::make_shared<std::vector<float>>(total_in_elements, 0.0f);
        }

        const auto &out_grad_data = output_gradient.getData();
        auto &in_grad_data = *in_grad_cpu.getStorageBuffer();

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float sum_grad = 0.0f;
                        for (uint32_t sy = 0; sy < scale_factor; ++sy)
                        {
                            uint32_t oh = ih * scale_factor + sy;
                            for (uint32_t sx = 0; sx < scale_factor; ++sx)
                            {
                                uint32_t ow = iw * scale_factor + sx;
                                size_t out_idx = n * out_h * out_w * channels + oh * out_w * channels + ow * channels + c;
                                sum_grad += out_grad_data[out_idx];
                            }
                        }

                        size_t in_idx = n * input_height * input_width * channels + ih * input_width * channels + iw * channels + c;
                        in_grad_data[in_idx] = sum_grad;
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

        if (input_gradient.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
            input_gradient.getDataType() != output_gradient.getDataType() ||
            input_gradient.getShape() != in_grad_shape)
        {
            input_gradient = Tensor(in_grad_shape, output_gradient.getDataType(), Execution_Target::VULKAN_GPU);
        }
        else
        {
            input_gradient.reshape(in_grad_shape);
        }

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        struct Upsample_Constants
        {
            uint32_t batch_size;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t channels;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t scale_factor;
        } constants{batch_size, input_height, input_width, channels, out_h, out_w, scale_factor};

        bool is_fp16 = (output_gradient.getDataType() == Data_Type::FLOAT16);
        bool use_coop = false; // Test standard FP16 pipeline

        Compute_Pipeline pipe;
        if (use_coop)
        {
            pipe = Compute_Pipeline::UPSAMPLE2D_BACKWARD_COOPMAT_FP16;
        }
        else if (is_fp16)
        {
            pipe = Compute_Pipeline::UPSAMPLE2D_BACKWARD_FP16;
        }
        else
        {
            pipe = Compute_Pipeline::UPSAMPLE2D_BACKWARD;
        }

        auto buf_self = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();

        if (use_coop)
        {
            effective_self->pushToGraph(pipe,
                                        {buf_self, in_grad_gpu.getStorageBuffer()},
                                        constants,
                                        (channels + 15) / 16, (input_height * input_width + 15) / 16, batch_size);
        }
        else
        {
            effective_self->pushToGraph(pipe,
                                        {buf_self, in_grad_gpu.getStorageBuffer()},
                                        constants,
                                        (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
        }
    }
}

} // namespace ops::vision

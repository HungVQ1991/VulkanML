#include "ops/vision/conv2d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "../ops_internal_utils.h"

namespace ops::vision
{

using ops::internal::ensureTensorTarget;
using ops::internal::pushGpuFill;
using ops::internal::validatePushConstants;

static void writeCpuOutput(Cpu_Tensor_Impl &out_cpu, const Shape &out_shape, Data_Type dtype, std::vector<float> values)
{
    out_cpu.setShape(out_shape);
    out_cpu.setByteOffset(0);
    out_cpu.setDataType(dtype);
    const size_t total = values.size();
    if (dtype == Data_Type::FLOAT16)
    {
        std::vector<float> clamped(total);
        for (size_t i = 0; i < total; ++i)
        {
            clamped[i] = std::clamp(values[i], -65504.0f, 65504.0f);
        }
        auto &fp16_buf = out_cpu.getStorageBufferFp16();
        if (!fp16_buf || fp16_buf.use_count() > 1 || fp16_buf->size() != total)
        {
            fp16_buf = std::make_shared<std::vector<float16_t>>(total);
        }
        if (total > 0)
        {
            convertFp32ToFp16(clamped.data(), fp16_buf->data(), total);
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

void conv2d(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output,
            uint32_t input_height, uint32_t input_width, uint32_t input_channels,
            uint32_t output_channels, uint32_t kernel_size,
            uint32_t stride, uint32_t padding,
            Tensor *scratch, bool *scratch_populated)
{
    if (scratch_populated != nullptr)
    {
        *scratch_populated = false;
    }
    auto in_impl = input.getImplementation();
    auto w_impl = weights.getImplementation();
    auto b_impl = biases.getImplementation();
    if (!in_impl || !w_impl || !b_impl)
    {
        Logger::logMessage("ops::vision::conv2d: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CONV2D_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (weights.getExecutionTarget() != target || biases.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::vision::conv2d: Execution_Target mismatch");
    }
    if (stride == 0 || kernel_size == 0)
    {
        throw std::invalid_argument("ops::vision::conv2d: stride and kernel_size must be non-zero");
    }
    if (static_cast<uint64_t>(input_height) + 2ULL * padding < kernel_size ||
        static_cast<uint64_t>(input_width) + 2ULL * padding < kernel_size)
    {
        throw std::invalid_argument("ops::vision::conv2d: Invalid convolution dimensions");
    }
    if (input.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2d: Input batch size exceeds uint32_t");
    }

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint64_t out_h_u64 = (static_cast<uint64_t>(input_height) + 2ULL * padding - kernel_size) / stride + 1ULL;
    uint64_t out_w_u64 = (static_cast<uint64_t>(input_width) + 2ULL * padding - kernel_size) / stride + 1ULL;
    if (out_h_u64 > std::numeric_limits<uint32_t>::max() ||
        out_w_u64 > std::numeric_limits<uint32_t>::max() ||
        out_h_u64 * out_w_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2d: Output spatial size exceeds uint32_t");
    }
    uint32_t out_h = static_cast<uint32_t>(out_h_u64);
    uint32_t out_w = static_cast<uint32_t>(out_w_u64);
    uint64_t out_feat_u64 = out_h_u64 * out_w_u64 * output_channels;
    if (out_feat_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2d: Output feature dimension exceeds uint32_t");
    }
    if (output_channels > 0 && biases.getTotalElements() < output_channels)
    {
        throw std::invalid_argument("ops::vision::conv2d: Bias size smaller than output_channels");
    }
    if (batch_size > 0 && input_height > 0 && input_width > 0 && input_channels > 0 &&
        out_h > 0 && out_w > 0 && output_channels > 0)
    {
        uint64_t req_in = static_cast<uint64_t>(batch_size) * input_height * input_width * input_channels;
        uint64_t req_w = static_cast<uint64_t>(kernel_size) * kernel_size * input_channels * output_channels;
        if (input.getTotalElements() < req_in || weights.getTotalElements() < req_w)
        {
            throw std::invalid_argument("ops::vision::conv2d: Input or weight tensor smaller than required dimensions");
        }
    }
    Shape out_shape{batch_size, static_cast<size_t>(out_feat_u64)};

    if (target == Execution_Target::CPU)
    {
        Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (batch_size == 0 || out_h == 0 || out_w == 0 || output_channels == 0)
        {
            ensureTensorTarget(output, Execution_Target::CPU, out_shape, out_dtype);
            return;
        }
        if (input_height == 0 || input_width == 0 || input_channels == 0)
        {
            ensureTensorTarget(output, Execution_Target::CPU, out_shape, out_dtype);
            writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*output.getImplementation()), out_shape, out_dtype,
                           std::vector<float>(out_shape.getTotalElements(), 0.0f));
            return;
        }

        std::vector<float> in_data = input.getData();
        std::vector<float> w_data = weights.getData();
        std::vector<float> b_data = biases.getData();

        ensureTensorTarget(output, Execution_Target::CPU, out_shape, out_dtype);

        std::vector<float> out_vals(static_cast<size_t>(batch_size) * out_h * out_w * output_channels, 0.0f);
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
                        out_vals[n * out_h * out_w * output_channels + oh * out_w * output_channels + ow * output_channels + oc] = sum;
                    }
                }
            }
        }
        auto &output_cpu = Cpu_Tensor_Impl::castToCpu(*output.getImplementation());
        writeCpuOutput(output_cpu, out_shape, out_dtype, std::move(out_vals));
    }
    else
    {
        if (input.getDataType() == Data_Type::TERNARY_INT2 ||
            weights.getDataType() == Data_Type::TERNARY_INT2 ||
            biases.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::vision::conv2d: TERNARY_INT2 is not supported on GPU");
        }

        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*w_impl);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*b_impl);
        auto contig_b = b_gpu.ensureContiguousSelf();
        const auto *effective_b = contig_b ? contig_b.get() : &b_gpu;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 ||
                        effective_w->getDataType() == Data_Type::FLOAT16 ||
                        effective_b->getDataType() == Data_Type::FLOAT16);
        Data_Type target_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (batch_size == 0 || out_h == 0 || out_w == 0 || output_channels == 0)
        {
            ensureTensorTarget(output, Execution_Target::VULKAN_GPU, out_shape, target_dtype);
            auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            output_gpu.clearHostCache();
            output_gpu.invalidateFp16Cache();
            return;
        }
        if (input_height == 0 || input_width == 0 || input_channels == 0)
        {
            ensureTensorTarget(output, Execution_Target::VULKAN_GPU, out_shape, target_dtype);
            auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());
            pushGpuFill(*effective_self, output_gpu.getStorageBuffer(), out_shape.getTotalElements(), 0.0f, target_dtype);
            output_gpu.clearHostCache();
            output_gpu.invalidateFp16Cache();
            return;
        }

        auto buf_in = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_w = is_fp16 ? effective_w->getEffectiveFp16Storage() : effective_w->getStorageBuffer();
        auto buf_b = is_fp16 ? effective_b->getEffectiveFp16Storage() : effective_b->getStorageBuffer();

        ensureTensorTarget(output, Execution_Target::VULKAN_GPU, out_shape, target_dtype);

        auto &output_gpu = Gpu_Tensor_Impl::castToGpu(*output.getImplementation());

        uint64_t M_u64 = static_cast<uint64_t>(batch_size) * out_h * out_w;
        uint64_t K_u64 = static_cast<uint64_t>(kernel_size) * kernel_size * input_channels;
        if (M_u64 > std::numeric_limits<uint32_t>::max() || K_u64 > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::vision::conv2d: Matrix dimensions M or K exceed uint32_t");
        }
        uint32_t M = static_cast<uint32_t>(M_u64);
        uint32_t K = static_cast<uint32_t>(K_u64);
        uint32_t N = output_channels;
        const auto &max_wg = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount();

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
        validatePushConstants<Conv2d_Constants>();

        constexpr uint64_t MAX_IM2COL_SCRATCH_BYTES = 128 * 1024 * 1024;
        uint64_t scratch_bytes = M_u64 * K_u64 * sizeof(uint16_t);

        if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled() && scratch_bytes <= MAX_IM2COL_SCRATCH_BYTES)
        {
            const uint64_t total_im2col_items = ((input_channels & 7u) == 0u) ? ((M_u64 * K_u64) >> 3ULL) : (M_u64 * K_u64);
            auto [im2col_wg_x, im2col_wg_y] = Execution_Engine::getInstance().getContext().compute2DWorkgroupGrid((total_im2col_items + 255ULL) / 256ULL);
            uint64_t gemm_wg_x = (static_cast<uint64_t>(N) + 63ULL) / 64ULL;
            uint64_t gemm_wg_y = (M_u64 + 63ULL) / 64ULL;
            if (gemm_wg_x > max_wg[0] || gemm_wg_y > max_wg[1])
            {
                throw std::overflow_error("ops::vision::conv2d: Workgroup grid exceeds hardware limits");
            }

            Gpu_Tensor_Impl *effective_scratch = nullptr;
            std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
            Shape scratch_shape{M, K};
            if (scratch != nullptr)
            {
                ensureTensorTarget(*scratch, Execution_Target::VULKAN_GPU, scratch_shape, Data_Type::FLOAT16);
                effective_scratch = &Gpu_Tensor_Impl::castToGpu(*scratch->getImplementation());
            }
            else
            {
                auto scratch_buf = Execution_Engine::getInstance().acquireReusableScratchBuffer(
                    Execution_Engine::SCRATCH_CONV2D_IM2COL, scratch_bytes, Data_Type::FLOAT16, static_cast<size_t>(M_u64 * K_u64));
                local_scratch = std::make_shared<Gpu_Tensor_Impl>(
                    scratch_shape, scratch_shape.computeContiguousStrides(), std::move(scratch_buf), 0, Data_Type::FLOAT16);
                effective_scratch = local_scratch.get();
            }

            effective_self->pushToGraph(Compute_Pipeline::CONV2D_IM2COL_FP16,
                                        {buf_in, effective_scratch->getStorageBuffer()},
                                        constants,
                                        im2col_wg_x, im2col_wg_y, 1);

            output_gpu.reshape(M, N);
            struct Matmul_Add_Constants
            {
                uint32_t batch_count;
                uint32_t rows_x;
                uint32_t columns_x;
                uint32_t columns_weights;
                uint32_t broadcast_w;
                uint32_t broadcast_b;
                uint32_t batch_rank;
                uint32_t pad0;
                uint32_t batch_dims[4];
                uint32_t a_batch_strides[4];
                uint32_t b_batch_strides[4];
            } matmul_constants{
                .batch_count = 1,
                .rows_x = M,
                .columns_x = K,
                .columns_weights = N,
                .broadcast_w = 0,
                .broadcast_b = 1,
                .batch_rank = 0,
                .pad0 = 0,
                .batch_dims = {1, 1, 1, 1},
                .a_batch_strides = {0, 0, 0, 0},
                .b_batch_strides = {0, 0, 0, 0}};
            static_assert(offsetof(Matmul_Add_Constants, batch_dims) == 32);
            static_assert(sizeof(Matmul_Add_Constants) == 80);
            validatePushConstants<Matmul_Add_Constants>();

            effective_self->pushToGraph(Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16,
                                        {effective_scratch->getStorageBuffer(), buf_w, buf_b, output_gpu.getStorageBuffer()},
                                        matmul_constants,
                                        static_cast<uint32_t>(gemm_wg_x), static_cast<uint32_t>(gemm_wg_y), 1);

            output_gpu.reshape(batch_size, out_h * out_w * output_channels);
            output_gpu.clearHostCache();
            output_gpu.invalidateFp16Cache();
            if (scratch != nullptr)
            {
                effective_scratch->clearHostCache();
                effective_scratch->invalidateFp16Cache();
                if (scratch_populated != nullptr)
                {
                    *scratch_populated = true;
                }
            }
            return;
        }

        uint64_t wg_oc = (static_cast<uint64_t>(output_channels) + 15ULL) / 16ULL;
        uint64_t wg_ow = (static_cast<uint64_t>(out_w) + 15ULL) / 16ULL;
        uint64_t wg_bh = static_cast<uint64_t>(batch_size) * out_h;
        if (wg_oc > max_wg[0] || wg_ow > max_wg[1] || wg_bh > max_wg[2])
        {
            throw std::overflow_error("ops::vision::conv2d: Workgroup grid exceeds hardware limits");
        }

        output_gpu.reshape(batch_size, out_h * out_w * output_channels);

        Compute_Pipeline pipeline = is_fp16 ? Compute_Pipeline::CONV2D_FORWARD_PASS_FP16 : Compute_Pipeline::CONV2D_FORWARD_PASS;
        effective_self->pushToGraph(pipeline,
                                    {buf_in, buf_w, buf_b, output_gpu.getStorageBuffer()},
                                    constants,
                                    static_cast<uint32_t>(wg_oc), static_cast<uint32_t>(wg_ow), static_cast<uint32_t>(wg_bh));
        output_gpu.clearHostCache();
        output_gpu.invalidateFp16Cache();
    }
}

void conv2dBackwardInput(const Tensor &output_gradient, const Tensor &weights, Tensor &input_gradient,
                         uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                         uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                         uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    auto out_grad_impl = output_gradient.getImplementation();
    auto w_impl = weights.getImplementation();
    if (!out_grad_impl || !w_impl)
    {
        Logger::logMessage("ops::vision::conv2dBackwardInput: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CONV2D_COMPUTE);
        return;
    }

    Execution_Target target = output_gradient.getExecutionTarget();
    if (weights.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::vision::conv2dBackwardInput: Execution_Target mismatch");
    }
    if (stride == 0 || kernel_size == 0)
    {
        throw std::invalid_argument("ops::vision::conv2dBackwardInput: stride and kernel_size must be non-zero");
    }
    if (output_gradient.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2dBackwardInput: Batch size exceeds uint32_t");
    }

    uint32_t batch_size = static_cast<uint32_t>(output_gradient.getRows());
    uint64_t in_feat_u64 = static_cast<uint64_t>(input_height) * input_width * input_channels;
    if (in_feat_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2dBackwardInput: Input feature dimension exceeds uint32_t");
    }
    if (batch_size > 0 && input_height > 0 && input_width > 0 && input_channels > 0 &&
        output_height > 0 && output_width > 0 && output_channels > 0)
    {
        uint64_t req_grad = static_cast<uint64_t>(batch_size) * output_height * output_width * output_channels;
        uint64_t req_w = static_cast<uint64_t>(kernel_size) * kernel_size * input_channels * output_channels;
        if (output_gradient.getTotalElements() < req_grad || weights.getTotalElements() < req_w)
        {
            throw std::invalid_argument("ops::vision::conv2dBackwardInput: output_gradient or weights smaller than required dimensions");
        }
    }
    Shape in_grad_shape{batch_size, static_cast<size_t>(in_feat_u64)};

    if (target == Execution_Target::CPU)
    {
        Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (batch_size == 0 || input_height == 0 || input_width == 0 || input_channels == 0)
        {
            ensureTensorTarget(input_gradient, Execution_Target::CPU, in_grad_shape, out_dtype);
            return;
        }

        std::vector<float> out_grad_data = output_gradient.getData();
        std::vector<float> w_data = weights.getData();

        ensureTensorTarget(input_gradient, Execution_Target::CPU, in_grad_shape, out_dtype);

        std::vector<float> dx_vals(static_cast<size_t>(batch_size) * input_height * input_width * input_channels, 0.0f);
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
                        dx_vals[n * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic] = sum;
                    }
                }
            }
        }
        auto &in_grad_cpu = Cpu_Tensor_Impl::castToCpu(*input_gradient.getImplementation());
        writeCpuOutput(in_grad_cpu, in_grad_shape, out_dtype, std::move(dx_vals));
    }
    else
    {
        if (output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
            weights.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::vision::conv2dBackwardInput: TERNARY_INT2 is not supported on GPU");
        }

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
        auto contig_self = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &out_grad_gpu;

        const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*w_impl);
        auto contig_w = w_gpu.ensureContiguousSelf();
        const auto *effective_w = contig_w ? contig_w.get() : &w_gpu;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 || effective_w->getDataType() == Data_Type::FLOAT16);
        Data_Type target_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

        if (batch_size == 0 || input_height == 0 || input_width == 0 || input_channels == 0)
        {
            ensureTensorTarget(input_gradient, Execution_Target::VULKAN_GPU, in_grad_shape, target_dtype);
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
            in_grad_gpu.clearHostCache();
            in_grad_gpu.invalidateFp16Cache();
            return;
        }
        if (output_height == 0 || output_width == 0 || output_channels == 0)
        {
            ensureTensorTarget(input_gradient, Execution_Target::VULKAN_GPU, in_grad_shape, target_dtype);
            auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());
            pushGpuFill(*effective_self, in_grad_gpu.getStorageBuffer(), in_grad_shape.getTotalElements(), 0.0f, target_dtype);
            in_grad_gpu.clearHostCache();
            in_grad_gpu.invalidateFp16Cache();
            return;
        }

        auto buf_grad = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_w = is_fp16 ? effective_w->getEffectiveFp16Storage() : effective_w->getStorageBuffer();

        ensureTensorTarget(input_gradient, Execution_Target::VULKAN_GPU, in_grad_shape, target_dtype);

        auto &in_grad_gpu = Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation());

        const auto &max_wg = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount();

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
        validatePushConstants<Conv2d_Constants>();

        if (is_fp16)
        {
            const uint64_t total_in_u64 = static_cast<uint64_t>(batch_size) * input_height * input_width * input_channels;
            if (total_in_u64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("ops::vision::conv2dBackwardInput: Input element count exceeds uint32_t");
            }
            auto [wg_x, wg_y] = Execution_Engine::getInstance().getContext().compute2DWorkgroupGrid((total_in_u64 + 255ULL) / 256ULL);
            effective_self->pushToGraph(Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT_FP16,
                                        {buf_grad, buf_w, in_grad_gpu.getStorageBuffer()},
                                        constants,
                                        wg_x, wg_y, 1);
        }
        else
        {
            uint64_t wg_ic = (static_cast<uint64_t>(input_channels) + 15ULL) / 16ULL;
            uint64_t wg_iw = (static_cast<uint64_t>(input_width) + 15ULL) / 16ULL;
            uint64_t wg_bh = static_cast<uint64_t>(batch_size) * input_height;
            if (wg_ic > max_wg[0] || wg_iw > max_wg[1] || wg_bh > max_wg[2])
            {
                throw std::overflow_error("ops::vision::conv2dBackwardInput: Workgroup grid exceeds hardware limits");
            }
            effective_self->pushToGraph(Compute_Pipeline::CONV2D_BACKWARD_PASS_INPUT_GRADIENT,
                                        {buf_grad, buf_w, in_grad_gpu.getStorageBuffer()},
                                        constants,
                                        static_cast<uint32_t>(wg_ic), static_cast<uint32_t>(wg_iw), static_cast<uint32_t>(wg_bh));
        }
        in_grad_gpu.clearHostCache();
        in_grad_gpu.invalidateFp16Cache();
    }
}

void conv2dBackwardWeight(const Tensor &input, const Tensor &output_gradient,
                          Tensor &weight_gradient, Tensor &bias_gradient,
                          uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_size, uint32_t stride, uint32_t padding,
                          Tensor *im2col_scratch, bool im2col_scratch_ready)
{
    auto in_impl = input.getImplementation();
    auto out_grad_impl = output_gradient.getImplementation();
    if (!in_impl || !out_grad_impl)
    {
        Logger::logMessage("ops::vision::conv2dBackwardWeight: Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CONV2D_COMPUTE);
        return;
    }

    Execution_Target target = input.getExecutionTarget();
    if (output_gradient.getExecutionTarget() != target)
    {
        throw std::invalid_argument("ops::vision::conv2dBackwardWeight: Execution_Target mismatch");
    }
    if (stride == 0 || kernel_size == 0)
    {
        throw std::invalid_argument("ops::vision::conv2dBackwardWeight: stride and kernel_size must be non-zero");
    }
    if (input.getRows() > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2dBackwardWeight: Batch size exceeds uint32_t");
    }
    if (output_gradient.getRows() != input.getRows())
    {
        throw std::invalid_argument("ops::vision::conv2dBackwardWeight: Batch size mismatch between input and output_gradient");
    }

    uint32_t batch_size = static_cast<uint32_t>(input.getRows());
    uint64_t K_u64 = static_cast<uint64_t>(kernel_size) * kernel_size * input_channels;
    if (K_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2dBackwardWeight: Kernel dimension K exceeds uint32_t");
    }
    uint32_t K = static_cast<uint32_t>(K_u64);
    uint64_t w_total_u64 = K_u64 * output_channels;
    if (w_total_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::conv2dBackwardWeight: Weight gradient size exceeds uint32_t");
    }
    if (batch_size > 0 && output_height > 0 && output_width > 0 && output_channels > 0)
    {
        uint64_t req_grad = static_cast<uint64_t>(batch_size) * output_height * output_width * output_channels;
        if (output_gradient.getTotalElements() < req_grad)
        {
            throw std::invalid_argument("ops::vision::conv2dBackwardWeight: output_gradient smaller than required dimensions");
        }
        if (input_height > 0 && input_width > 0 && input_channels > 0)
        {
            uint64_t req_in = static_cast<uint64_t>(batch_size) * input_height * input_width * input_channels;
            if (input.getTotalElements() < req_in)
            {
                throw std::invalid_argument("ops::vision::conv2dBackwardWeight: input smaller than required dimensions");
            }
        }
    }
    Shape w_shape{1, static_cast<size_t>(w_total_u64)};
    Shape b_shape{1, output_channels};

    if (target == Execution_Target::CPU)
    {
        std::vector<float> in_data = input.getData();
        std::vector<float> out_grad_data = output_gradient.getData();

        ensureTensorTarget(weight_gradient, Execution_Target::CPU, w_shape, Data_Type::FLOAT32);
        ensureTensorTarget(bias_gradient, Execution_Target::CPU, b_shape, Data_Type::FLOAT32);

        auto &w_grad_cpu = Cpu_Tensor_Impl::castToCpu(*weight_gradient.getImplementation());
        auto &b_grad_cpu = Cpu_Tensor_Impl::castToCpu(*bias_gradient.getImplementation());

        std::vector<float> dw_vals(static_cast<size_t>(K) * output_channels, 0.0f);
        std::vector<float> db_vals(output_channels, 0.0f);

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
                                        db_vals[oc] += grad_val;
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
                        dw_vals[w_idx] = w_sum;
                    }
                }
            }
        }
        writeCpuOutput(w_grad_cpu, w_shape, Data_Type::FLOAT32, std::move(dw_vals));
        writeCpuOutput(b_grad_cpu, b_shape, Data_Type::FLOAT32, std::move(db_vals));
    }
    else
    {
        if (input.getDataType() == Data_Type::TERNARY_INT2 ||
            output_gradient.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::invalid_argument("ops::vision::conv2dBackwardWeight: TERNARY_INT2 is not supported on GPU");
        }

        const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*in_impl);
        auto contig_self = in_gpu.ensureContiguousSelf();
        const auto *effective_self = contig_self ? contig_self.get() : &in_gpu;

        const auto &out_grad_gpu = Gpu_Tensor_Impl::castToGpu(*out_grad_impl);
        auto contig_grad = out_grad_gpu.ensureContiguousSelf();
        const auto *effective_grad = contig_grad ? contig_grad.get() : &out_grad_gpu;

        bool is_fp16 = (effective_self->getDataType() == Data_Type::FLOAT16 || effective_grad->getDataType() == Data_Type::FLOAT16);
        Data_Type target_dtype = Data_Type::FLOAT32;

        uint64_t M_u64 = static_cast<uint64_t>(batch_size) * output_height * output_width;
        if (M_u64 > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("ops::vision::conv2dBackwardWeight: Spatial matrix dimension M exceeds uint32_t");
        }
        uint32_t M = static_cast<uint32_t>(M_u64);
        if (M == 0 || input_channels == 0 || output_channels == 0)
        {
            ensureTensorTarget(weight_gradient, Execution_Target::VULKAN_GPU, w_shape, target_dtype);
            ensureTensorTarget(bias_gradient, Execution_Target::VULKAN_GPU, b_shape, target_dtype);
            auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*bias_gradient.getImplementation());
            pushGpuFill(*effective_self, w_grad_gpu.getStorageBuffer(), w_shape.getTotalElements(), 0.0f, target_dtype);
            pushGpuFill(*effective_self, b_grad_gpu.getStorageBuffer(), b_shape.getTotalElements(), 0.0f, target_dtype);
            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            return;
        }

        const auto &max_wg = Execution_Engine::getInstance().getContext().getMaxComputeWorkGroupCount();
        if (output_channels > max_wg[0])
        {
            throw std::overflow_error("ops::vision::conv2dBackwardWeight: Bias gradient workgroup count exceeds hardware limits");
        }

        if (input_height == 0 || input_width == 0)
        {
            auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

            ensureTensorTarget(weight_gradient, Execution_Target::VULKAN_GPU, w_shape, target_dtype);
            ensureTensorTarget(bias_gradient, Execution_Target::VULKAN_GPU, b_shape, target_dtype);

            auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
            auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*bias_gradient.getImplementation());

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
            validatePushConstants<Conv2d_Constants>();

            pushGpuFill(*effective_self, w_grad_gpu.getStorageBuffer(), w_shape.getTotalElements(), 0.0f, target_dtype);
            Compute_Pipeline bias_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_BIAS_GRADIENT_FP16 : Compute_Pipeline::CONV2D_BIAS_GRADIENT;
            effective_self->pushToGraph(bias_pipeline,
                                        {buf_grad, b_grad_gpu.getStorageBuffer()},
                                        constants,
                                        output_channels, 1, 1);
            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            return;
        }

        auto buf_in = is_fp16 ? effective_self->getEffectiveFp16Storage() : effective_self->getStorageBuffer();
        auto buf_grad = is_fp16 ? effective_grad->getEffectiveFp16Storage() : effective_grad->getStorageBuffer();

        ensureTensorTarget(weight_gradient, Execution_Target::VULKAN_GPU, w_shape, target_dtype);
        ensureTensorTarget(bias_gradient, Execution_Target::VULKAN_GPU, b_shape, target_dtype);

        auto &w_grad_gpu = Gpu_Tensor_Impl::castToGpu(*weight_gradient.getImplementation());
        auto &b_grad_gpu = Gpu_Tensor_Impl::castToGpu(*bias_gradient.getImplementation());

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
        validatePushConstants<Conv2d_Constants>();

        constexpr uint64_t MAX_IM2COL_SCRATCH_BYTES = 128 * 1024 * 1024;
        uint64_t scratch_bytes = K_u64 * M_u64 * (is_fp16 ? sizeof(uint16_t) : sizeof(float));
        if (scratch_bytes > MAX_IM2COL_SCRATCH_BYTES)
        {
            uint64_t wg_oc = (static_cast<uint64_t>(output_channels) + 15ULL) / 16ULL;
            uint64_t wg_ic = (static_cast<uint64_t>(input_channels) + 15ULL) / 16ULL;
            uint64_t wg_k2 = static_cast<uint64_t>(kernel_size) * kernel_size;
            if (wg_oc > max_wg[0] || wg_ic > max_wg[1] || wg_k2 > max_wg[2])
            {
                throw std::overflow_error("ops::vision::conv2dBackwardWeight: Direct gradient workgroup grid exceeds hardware limits");
            }
            Compute_Pipeline direct_pipeline = is_fp16
                ? Compute_Pipeline::CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT_FP16
                : Compute_Pipeline::CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT;
            effective_self->pushToGraph(direct_pipeline,
                                        {buf_in, buf_grad, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                        constants,
                                        static_cast<uint32_t>(wg_oc), static_cast<uint32_t>(wg_ic), static_cast<uint32_t>(wg_k2));
            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            return;
        }

        if (is_fp16 && Execution_Engine::getInstance().isCooperativeMatrixEnabled())
        {
            const uint32_t wg_x = (output_channels + 63u) / 64u;
            const uint32_t wg_y = (K + 63u) / 64u;
            if (wg_x > max_wg[0] || wg_y > max_wg[1])
            {
                throw std::overflow_error("ops::vision::conv2dBackwardWeight: CoopMat workgroup grid exceeds hardware limits");
            }

            std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
            Gpu_Tensor_Impl *effective_scratch = nullptr;
            Shape scratch_shape{M, K};
            const bool can_reuse_scratch = im2col_scratch_ready &&
                                           im2col_scratch != nullptr &&
                                           im2col_scratch->getExecutionTarget() == Execution_Target::VULKAN_GPU &&
                                           im2col_scratch->getDataType() == Data_Type::FLOAT16 &&
                                           im2col_scratch->getShape() == scratch_shape;
            if (can_reuse_scratch)
            {
                effective_scratch = &Gpu_Tensor_Impl::castToGpu(*im2col_scratch->getImplementation());
            }
            else
            {
                if (im2col_scratch != nullptr)
                {
                    ensureTensorTarget(*im2col_scratch, Execution_Target::VULKAN_GPU, scratch_shape, Data_Type::FLOAT16);
                    effective_scratch = &Gpu_Tensor_Impl::castToGpu(*im2col_scratch->getImplementation());
                }
                else
                {
                    auto scratch_buf = Execution_Engine::getInstance().acquireReusableScratchBuffer(
                        Execution_Engine::SCRATCH_CONV2D_BWD_IM2COL, scratch_bytes, Data_Type::FLOAT16, static_cast<size_t>(K_u64 * M_u64));
                    local_scratch = std::make_shared<Gpu_Tensor_Impl>(
                        scratch_shape, scratch_shape.computeContiguousStrides(), std::move(scratch_buf), 0, Data_Type::FLOAT16);
                    effective_scratch = local_scratch.get();
                }

                const uint64_t total_im2col_items = ((input_channels & 7u) == 0u) ? ((M_u64 * K_u64) >> 3ULL) : (M_u64 * K_u64);
                auto [im2col_wg_x, im2col_wg_y] = Execution_Engine::getInstance().getContext().compute2DWorkgroupGrid((total_im2col_items + 255ULL) / 256ULL);
                effective_self->pushToGraph(Compute_Pipeline::CONV2D_IM2COL_FP16,
                                            {buf_in, effective_scratch->getStorageBuffer()},
                                            constants,
                                            im2col_wg_x, im2col_wg_y, 1);
            }

            const uint32_t total_tiles = wg_x * wg_y;
            const uint32_t total_k_blocks = (M + 31u) / 32u;
            constexpr uint32_t TARGET_TOTAL_WGS = 64u;
            constexpr uint32_t MAX_SPLIT_K = 64u;
            constexpr uint32_t MIN_K_BLOCKS_PER_SPLIT = 8u;
            constexpr uint64_t MAX_SPLITK_PARTIAL_BYTES = 32ULL * 1024ULL * 1024ULL;
            const uint64_t total_reduce_elems = w_total_u64 + static_cast<uint64_t>(output_channels);
            const uint64_t red_wg_x_u64 = (total_reduce_elems + 127ULL) / 128ULL;

            uint32_t split_k = 1u;
            if (!Execution_Engine::getInstance().isFusedGemmAdamEnabled() &&
                total_tiles > 0u &&
                total_tiles < TARGET_TOTAL_WGS &&
                total_k_blocks >= 2u * MIN_K_BLOCKS_PER_SPLIT &&
                red_wg_x_u64 <= max_wg[0])
            {
                const uint32_t desired_sk = (TARGET_TOTAL_WGS + total_tiles - 1u) / total_tiles;
                const uint32_t max_sk_work = total_k_blocks / MIN_K_BLOCKS_PER_SPLIT;
                const uint64_t bytes_per_split = total_reduce_elems * sizeof(float);
                const uint32_t max_sk_mem = (bytes_per_split > 0ULL && bytes_per_split <= MAX_SPLITK_PARTIAL_BYTES)
                    ? static_cast<uint32_t>(std::min<uint64_t>(MAX_SPLIT_K, MAX_SPLITK_PARTIAL_BYTES / bytes_per_split))
                    : 1u;
                split_k = std::min({desired_sk, max_sk_work, max_sk_mem, MAX_SPLIT_K, max_wg[2]});
                if (split_k < 2u)
                {
                    split_k = 1u;
                }
            }

            struct Coopmat_Linear_Bwd_Constants
            {
                uint32_t batch_size;
                uint32_t input_dimension;
                uint32_t output_dimension;
                uint32_t accumulate;
                uint32_t split_k;
                uint32_t pass_mode;
            };
            validatePushConstants<Coopmat_Linear_Bwd_Constants>();

            if (split_k > 1u)
            {
                const size_t total_reduce_sz = static_cast<size_t>(total_reduce_elems);
                const size_t partials_elems = static_cast<size_t>(split_k) * total_reduce_sz;
                auto partials_buf = Execution_Engine::getInstance().acquireReusableScratchBuffer(
                    Execution_Engine::SCRATCH_GEMM_SPLITK,
                    partials_elems * sizeof(float),
                    Data_Type::FLOAT32,
                    partials_elems);

                Coopmat_Linear_Bwd_Constants pass1_c{M, K, output_channels, 0u, split_k, 1u};
                effective_self->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
                                            {effective_scratch->getStorageBuffer(), buf_grad, partials_buf, partials_buf},
                                            pass1_c,
                                            wg_x, wg_y, split_k);

                Coopmat_Linear_Bwd_Constants pass2_c{M, K, output_channels, 0u, split_k, 2u};
                effective_self->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
                                            {partials_buf, partials_buf, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                            pass2_c,
                                            static_cast<uint32_t>(red_wg_x_u64), 1, 1);
            }
            else
            {
                Coopmat_Linear_Bwd_Constants single_c{M, K, output_channels, 0u, 1u, 0u};
                effective_self->pushToGraph(Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
                                            {effective_scratch->getStorageBuffer(), buf_grad, w_grad_gpu.getStorageBuffer(), b_grad_gpu.getStorageBuffer()},
                                            single_c,
                                            wg_x, wg_y, 1);
            }

            w_grad_gpu.clearHostCache();
            w_grad_gpu.invalidateFp16Cache();
            b_grad_gpu.clearHostCache();
            b_grad_gpu.invalidateFp16Cache();
            if (!can_reuse_scratch && im2col_scratch != nullptr)
            {
                effective_scratch->clearHostCache();
                effective_scratch->invalidateFp16Cache();
            }
            return;
        }

        uint64_t wg_m = (M_u64 + 15ULL) / 16ULL;
        uint64_t wg_k = (K_u64 + 15ULL) / 16ULL;
        uint64_t wg_oc = (static_cast<uint64_t>(output_channels) + 15ULL) / 16ULL;
        if (wg_m > max_wg[0] || wg_k > max_wg[1] || wg_oc > max_wg[0])
        {
            throw std::overflow_error("ops::vision::conv2dBackwardWeight: Workgroup grid exceeds hardware limits");
        }

        std::shared_ptr<Gpu_Tensor_Impl> local_scratch;
        Gpu_Tensor_Impl *effective_scratch = nullptr;
        Data_Type scratch_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        Shape scratch_shape{K, M};

        if (im2col_scratch != nullptr)
        {
            ensureTensorTarget(*im2col_scratch, Execution_Target::VULKAN_GPU, scratch_shape, scratch_dtype);
            effective_scratch = &Gpu_Tensor_Impl::castToGpu(*im2col_scratch->getImplementation());
        }
        else
        {
            auto scratch_buf = Execution_Engine::getInstance().acquireReusableScratchBuffer(
                Execution_Engine::SCRATCH_CONV2D_BWD_IM2COL, scratch_bytes, scratch_dtype, static_cast<size_t>(K_u64 * M_u64));
            local_scratch = std::make_shared<Gpu_Tensor_Impl>(
                scratch_shape, scratch_shape.computeContiguousStrides(), std::move(scratch_buf), 0, scratch_dtype);
            effective_scratch = local_scratch.get();
        }

        Compute_Pipeline im2col_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_IM2COL_TRANSPOSED_FP16 : Compute_Pipeline::CONV2D_IM2COL_TRANSPOSED;
        effective_self->pushToGraph(im2col_pipeline,
                                    {buf_in, effective_scratch->getStorageBuffer()},
                                    constants,
                                    static_cast<uint32_t>(wg_m), static_cast<uint32_t>(wg_k), 1);

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
                                        {effective_scratch->getStorageBuffer(), buf_grad, w_grad_gpu.getStorageBuffer()},
                                        matmul_dims,
                                        static_cast<uint32_t>(wg_oc), static_cast<uint32_t>(wg_k), 1);
        }
        else
        {
            effective_self->pushToGraph(Compute_Pipeline::MATMUL,
                                        {effective_scratch->getStorageBuffer(), buf_grad, w_grad_gpu.getStorageBuffer()},
                                        matmul_dims,
                                        static_cast<uint32_t>(wg_oc), static_cast<uint32_t>(wg_k), 1);
        }

        Compute_Pipeline bias_pipeline = is_fp16 ? Compute_Pipeline::CONV2D_BIAS_GRADIENT_FP16 : Compute_Pipeline::CONV2D_BIAS_GRADIENT;
        effective_self->pushToGraph(bias_pipeline,
                                    {buf_grad, b_grad_gpu.getStorageBuffer()},
                                    constants,
                                    output_channels, 1, 1);
        w_grad_gpu.clearHostCache();
        w_grad_gpu.invalidateFp16Cache();
        b_grad_gpu.clearHostCache();
        b_grad_gpu.invalidateFp16Cache();
        if (im2col_scratch != nullptr)
        {
            effective_scratch->clearHostCache();
            effective_scratch->invalidateFp16Cache();
        }
    }
}

}

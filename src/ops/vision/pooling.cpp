#include "ops/vision/pooling.h"

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

static size_t checkedMulSize(size_t a, size_t b, const char *ctx)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        throw std::overflow_error(std::string(ctx) + ": Size calculation overflows size_t");
    }
    return a * b;
}

static uint32_t checkedBatchSize(const Shape &shape, const char *ctx)
{
    if (shape.getRank() < 2)
    {
        throw std::invalid_argument(std::string(ctx) + ": Expected tensor rank >= 2");
    }
    const size_t batch_sz = shape[0];
    if (batch_sz > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error(std::string(ctx) + ": Batch size exceeds uint32_t range");
    }
    return static_cast<uint32_t>(batch_sz);
}

static Shape makeNhwcOutputShape(const Shape &ref_shape, uint32_t batch_size,
                                 uint32_t out_h, uint32_t out_w, uint32_t channels,
                                 const char *ctx)
{
    const size_t per_sample = checkedMulSize(checkedMulSize(out_h, out_w, ctx), channels, ctx);
    checkedMulSize(batch_size, per_sample, ctx);
    if (ref_shape.getRank() == 4)
    {
        return Shape{batch_size, out_h, out_w, channels};
    }
    return Shape{batch_size, per_sample};
}

static void validateNhwcTensorShape(const Tensor &tensor, uint32_t batch_size,
                                    uint32_t height, uint32_t width, uint32_t channels,
                                    bool check_flat_elements, const char *ctx)
{
    const Shape &sh = tensor.getShape();
    if (sh.getRank() == 4)
    {
        if (sh[0] != batch_size || sh[1] != height || sh[2] != width || sh[3] != channels)
        {
            throw std::invalid_argument(std::string(ctx) + ": Rank-4 NHWC shape mismatch");
        }
    }
    else if (check_flat_elements && batch_size > 0 && height > 0 && width > 0 && channels > 0)
    {
        const size_t expected = checkedMulSize(
            batch_size,
            checkedMulSize(checkedMulSize(height, width, ctx), channels, ctx),
            ctx);
        if (tensor.getTotalElements() != expected)
        {
            throw std::invalid_argument(std::string(ctx) + ": Tensor element count does not match N*H*W*C");
        }
    }
}

static void clearGpuCaches(Tensor &tensor)
{
    auto &gpu = Gpu_Tensor_Impl::castToGpu(*tensor.getImplementation());
    gpu.clearHostCache();
    gpu.invalidateFp16Cache();
}

static void initEmptyOutput(Execution_Target target, Tensor &dst, const Shape &shape, Data_Type dtype)
{
    ensureTensorTarget(dst, target, shape, dtype);
    if (target == Execution_Target::VULKAN_GPU)
    {
        clearGpuCaches(dst);
    }
}

static void writeCpu(Tensor &dst, const Shape &shape, Data_Type dtype, std::vector<float> vals)
{
    ops::internal::writeCpuOutput(Cpu_Tensor_Impl::castToCpu(*dst.getImplementation()), shape, dtype, std::move(vals), true);
}

static std::shared_ptr<gpu::vector> getGpuFloat32ReadBuffer(const Gpu_Tensor_Impl &src_gpu, const Gpu_Tensor_Impl &anchor)
{
    if (src_gpu.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("ops::vision: TERNARY_INT2 is not supported");
    }
    auto contig = src_gpu.ensureContiguousSelf();
    const auto *effective = contig ? contig.get() : &src_gpu;
    const size_t total = effective->getTotalElements();
    if (total == 0)
    {
        return effective->getStorageBuffer();
    }
    if (!effective->getStorageBuffer())
    {
        throw std::runtime_error("ops::vision: Missing GPU storage buffer");
    }
    if (total > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision: Tensor element count exceeds uint32_t dispatch limit");
    }
    if (effective->getDataType() == Data_Type::FLOAT16)
    {
        auto &ctx = Execution_Engine::getInstance().getContext();
        auto tmp = std::make_shared<gpu::vector>(ctx, total, Data_Type::FLOAT32);
        const uint32_t total_u32 = static_cast<uint32_t>(total);
        auto [wg_x, wg_y] = ctx.compute2DWorkgroupGrid((static_cast<uint64_t>(total_u32) + 255ULL) / 256ULL);
        anchor.pushToGraph(Compute_Pipeline::CAST_FP16_TO_FP32, {effective->getStorageBuffer(), tmp}, total_u32, wg_x, wg_y, 1);
        return tmp;
    }
    return effective->getStorageBuffer();
}

static void fillConstantOutput(const std::shared_ptr<Tensor_Impl> &anchor_impl, Execution_Target target,
                               Tensor &dst, const Shape &shape, Data_Type dtype, float fill_value)
{
    ensureTensorTarget(dst, target, shape, dtype);
    const size_t elems = shape.getTotalElements();
    if (target == Execution_Target::CPU)
    {
        writeCpu(dst, shape, dtype, std::vector<float>(elems, fill_value));
    }
    else
    {
        const auto &anchor_gpu = Gpu_Tensor_Impl::castToGpu(*anchor_impl);
        auto &dst_gpu = Gpu_Tensor_Impl::castToGpu(*dst.getImplementation());
        pushGpuFill(anchor_gpu, dst_gpu.getStorageBuffer(), elems, fill_value, dtype);
        clearGpuCaches(dst);
    }
}

static void prepareGpuPoolingInput(
    const Gpu_Tensor_Impl &src_gpu,
    std::shared_ptr<Gpu_Tensor_Impl> &contig_keep,
    const Gpu_Tensor_Impl *&effective_out,
    Data_Type &gpu_out_dtype,
    std::shared_ptr<gpu::vector> &buf_out)
{
    contig_keep = src_gpu.ensureContiguousSelf();
    effective_out = contig_keep ? contig_keep.get() : &src_gpu;
    const bool is_fp16 = (effective_out->getDataType() == Data_Type::FLOAT16);
    gpu_out_dtype = is_fp16 ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    buf_out = is_fp16 ? effective_out->getEffectiveFp16Storage() : effective_out->getStorageBuffer();
}

static void validatePool2dParams(
    uint32_t input_height, uint32_t input_width,
    uint32_t kernel_size, uint32_t stride, uint32_t padding,
    const char *op_name, uint32_t &out_h, uint32_t &out_w)
{
    if (stride == 0 || kernel_size == 0)
    {
        throw std::invalid_argument(std::string(op_name) + ": stride and kernel_size must be non-zero");
    }
    if (static_cast<uint64_t>(input_height) + 2ULL * padding < kernel_size ||
        static_cast<uint64_t>(input_width) + 2ULL * padding < kernel_size)
    {
        throw std::invalid_argument(std::string(op_name) + ": Invalid pooling dimensions");
    }
    const uint64_t out_h_u64 = (static_cast<uint64_t>(input_height) + 2ULL * padding - kernel_size) / stride + 1ULL;
    const uint64_t out_w_u64 = (static_cast<uint64_t>(input_width) + 2ULL * padding - kernel_size) / stride + 1ULL;
    if (out_h_u64 > std::numeric_limits<uint32_t>::max() || out_w_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error(std::string(op_name) + ": Output dimensions exceed uint32_t range");
    }
    out_h = static_cast<uint32_t>(out_h_u64);
    out_w = static_cast<uint32_t>(out_w_u64);
}

static bool validateBackwardPair(
    const Tensor &output_gradient, const Tensor &mask, const char *op_name,
    Execution_Target &target_out, uint32_t &batch_out)
{
    if (!output_gradient.getImplementation() || !mask.getImplementation())
    {
        Logger::logMessage(std::string(op_name) + ": Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE);
        return false;
    }
    target_out = output_gradient.getExecutionTarget();
    if (mask.getExecutionTarget() != target_out)
    {
        throw std::invalid_argument(std::string(op_name) + ": Execution_Target mismatch");
    }
    if (output_gradient.getDataType() == Data_Type::TERNARY_INT2 ||
        mask.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op_name) + ": TERNARY_INT2 is not supported");
    }
    batch_out = checkedBatchSize(output_gradient.getShape(), op_name);
    return true;
}

static bool validateSingleInput(const Tensor &input, const char *op_name, Execution_Target &target_out, uint32_t &batch_out)
{
    if (!input.getImplementation())
    {
        Logger::logMessage(std::string(op_name) + ": Null tensor implementation encountered",
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::POOLING_COMPUTE);
        return false;
    }
    if (input.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument(std::string(op_name) + ": TERNARY_INT2 is not supported");
    }
    target_out = input.getExecutionTarget();
    batch_out = checkedBatchSize(input.getShape(), op_name);
    return true;
}

struct Pool2d_Constants
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
};

struct Global_Pool_Constants
{
    uint32_t batch_size;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t channels;
};

struct Upsample_Constants
{
    uint32_t batch_size;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t scale_factor;
};

void maxpool2d(const Tensor &input, Tensor &output, Tensor &output_mask,
               uint32_t input_height, uint32_t input_width, uint32_t channels,
               uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(input, "ops::vision::maxpool2d", target, batch_size)) return;

    uint32_t out_h = 0, out_w = 0;
    validatePool2dParams(input_height, input_width, kernel_size, stride, padding, "ops::vision::maxpool2d", out_h, out_w);
    validateNhwcTensorShape(input, batch_size, input_height, input_width, channels, true, "ops::vision::maxpool2d");
    const Shape out_shape = makeNhwcOutputShape(input.getShape(), batch_size, out_h, out_w, channels, "ops::vision::maxpool2d");
    const Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || channels == 0 || out_h == 0 || out_w == 0)
    {
        initEmptyOutput(target, output, out_shape, out_dtype);
        initEmptyOutput(target, output_mask, out_shape, Data_Type::FLOAT32);
        return;
    }

    auto in_impl = input.getImplementation();
    if (input_height == 0 || input_width == 0)
    {
        fillConstantOutput(in_impl, target, output, out_shape, out_dtype, 0.0f);
        fillConstantOutput(in_impl, target, output_mask, out_shape, Data_Type::FLOAT32, -1.0f);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        ensureTensorTarget(output, target, out_shape, out_dtype);
        ensureTensorTarget(output_mask, target, out_shape, Data_Type::FLOAT32);

        const size_t out_elems = out_shape.getTotalElements();
        std::vector<float> out_vals(out_elems, 0.0f);
        std::vector<float> mask_vals(out_elems, -1.0f);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        bool has_valid = false;
                        float best_val = 0.0f;
                        int32_t max_k_idx = -1;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                const int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                const int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    const size_t in_idx = ((static_cast<size_t>(n) * input_height + ih) * input_width + iw) * channels + c;
                                    const float val = in_data[in_idx];
                                    if (!std::isnan(val) && (!has_valid || val > best_val))
                                    {
                                        best_val = val;
                                        max_k_idx = static_cast<int32_t>(ky * kernel_size + kx);
                                        has_valid = true;
                                    }
                                }
                            }
                        }
                        const size_t out_idx = ((static_cast<size_t>(n) * out_h + oh) * out_w + ow) * channels + c;
                        out_vals[out_idx] = best_val;
                        mask_vals[out_idx] = static_cast<float>(max_k_idx);
                    }
                }
            }
        }
        writeCpu(output, out_shape, out_dtype, std::move(out_vals));
        writeCpu(output_mask, out_shape, Data_Type::FLOAT32, std::move(mask_vals));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*in_impl), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(output, target, out_shape, gpu_out_dtype);
    ensureTensorTarget(output_mask, target, out_shape, Data_Type::FLOAT32);

    const Pool2d_Constants constants{batch_size, input_height, input_width, channels, out_h, out_w, kernel_size, stride, padding};
    validatePushConstants<Pool2d_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::MAXPOOL2D_FORWARD_FP16 : Compute_Pipeline::MAXPOOL2D_FORWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self,
                                 Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).getStorageBuffer(),
                                 Gpu_Tensor_Impl::castToGpu(*output_mask.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    clearGpuCaches(output);
    clearGpuCaches(output_mask);
}

void maxpool2dBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient,
                       uint32_t input_height, uint32_t input_width, uint32_t channels,
                       uint32_t output_height, uint32_t output_width,
                       uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateBackwardPair(output_gradient, mask, "ops::vision::maxpool2dBackward", target, batch_size)) return;
    if (stride == 0 || kernel_size == 0)
    {
        throw std::invalid_argument("ops::vision::maxpool2dBackward: stride and kernel_size must be non-zero");
    }

    validateNhwcTensorShape(output_gradient, batch_size, output_height, output_width, channels, true, "ops::vision::maxpool2dBackward");
    if (batch_size > 0 && output_height > 0 && output_width > 0 && channels > 0 &&
        mask.getTotalElements() != output_gradient.getTotalElements())
    {
        throw std::invalid_argument("ops::vision::maxpool2dBackward: Mask element count does not match output gradient");
    }
    const Shape in_grad_shape = makeNhwcOutputShape(output_gradient.getShape(), batch_size, input_height, input_width, channels, "ops::vision::maxpool2dBackward");
    const Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || input_height == 0 || input_width == 0 || channels == 0)
    {
        initEmptyOutput(target, input_gradient, in_grad_shape, out_dtype);
        return;
    }

    auto out_grad_impl = output_gradient.getImplementation();
    if (output_height == 0 || output_width == 0)
    {
        fillConstantOutput(out_grad_impl, target, input_gradient, in_grad_shape, out_dtype, 0.0f);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto mask_data = mask.getData();
        const auto out_grad_data = output_gradient.getData();
        ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);
        std::vector<float> in_grad_vals(in_grad_shape.getTotalElements(), 0.0f);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float acc_grad = 0.0f;
                        for (uint32_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel_size; ++kx)
                            {
                                const int oh_calc = static_cast<int>(ih) + static_cast<int>(padding) - static_cast<int>(ky);
                                const int ow_calc = static_cast<int>(iw) + static_cast<int>(padding) - static_cast<int>(kx);
                                if (oh_calc >= 0 && oh_calc % static_cast<int>(stride) == 0 &&
                                    ow_calc >= 0 && ow_calc % static_cast<int>(stride) == 0)
                                {
                                    const uint32_t oh = static_cast<uint32_t>(oh_calc) / stride;
                                    const uint32_t ow = static_cast<uint32_t>(ow_calc) / stride;
                                    if (oh < output_height && ow < output_width)
                                    {
                                        const size_t out_idx = ((static_cast<size_t>(n) * output_height + oh) * output_width + ow) * channels + c;
                                        const float m = mask_data[out_idx];
                                        if (m >= 0.0f && static_cast<uint32_t>(std::round(m)) == ky * kernel_size + kx)
                                        {
                                            acc_grad += out_grad_data[out_idx];
                                        }
                                    }
                                }
                            }
                        }
                        in_grad_vals[((static_cast<size_t>(n) * input_height + ih) * input_width + iw) * channels + c] = acc_grad;
                    }
                }
            }
        }
        writeCpu(input_gradient, in_grad_shape, out_dtype, std::move(in_grad_vals));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*out_grad_impl), contig_self, effective_self, gpu_out_dtype, buf_self);
    auto buf_mask = getGpuFloat32ReadBuffer(Gpu_Tensor_Impl::castToGpu(*mask.getImplementation()), *effective_self);

    ensureTensorTarget(input_gradient, target, in_grad_shape, gpu_out_dtype);
    const Pool2d_Constants constants{batch_size, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding};
    validatePushConstants<Pool2d_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::MAXPOOL2D_BACKWARD_FP16 : Compute_Pipeline::MAXPOOL2D_BACKWARD;
    effective_self->pushToGraph(pipe,
                                {buf_mask, buf_self, Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    clearGpuCaches(input_gradient);
}

void avgpool2d(const Tensor &input, Tensor &output,
               uint32_t input_height, uint32_t input_width, uint32_t channels,
               uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(input, "ops::vision::avgpool2d", target, batch_size)) return;

    uint32_t out_h = 0, out_w = 0;
    validatePool2dParams(input_height, input_width, kernel_size, stride, padding, "ops::vision::avgpool2d", out_h, out_w);
    validateNhwcTensorShape(input, batch_size, input_height, input_width, channels, true, "ops::vision::avgpool2d");
    const Shape out_shape = makeNhwcOutputShape(input.getShape(), batch_size, out_h, out_w, channels, "ops::vision::avgpool2d");
    const Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || channels == 0 || out_h == 0 || out_w == 0)
    {
        initEmptyOutput(target, output, out_shape, out_dtype);
        return;
    }

    auto in_impl = input.getImplementation();
    if (input_height == 0 || input_width == 0)
    {
        fillConstantOutput(in_impl, target, output, out_shape, out_dtype, 0.0f);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        ensureTensorTarget(output, target, out_shape, out_dtype);
        std::vector<float> out_vals(out_shape.getTotalElements(), 0.0f);
        const float area = static_cast<float>(static_cast<uint64_t>(kernel_size) * kernel_size);

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
                                const int ih = static_cast<int>(oh * stride + ky) - static_cast<int>(padding);
                                const int iw = static_cast<int>(ow * stride + kx) - static_cast<int>(padding);
                                if (ih >= 0 && ih < static_cast<int>(input_height) && iw >= 0 && iw < static_cast<int>(input_width))
                                {
                                    sum_val += in_data[((static_cast<size_t>(n) * input_height + ih) * input_width + iw) * channels + c];
                                }
                            }
                        }
                        out_vals[((static_cast<size_t>(n) * out_h + oh) * out_w + ow) * channels + c] = sum_val / area;
                    }
                }
            }
        }
        writeCpu(output, out_shape, out_dtype, std::move(out_vals));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*in_impl), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(output, target, out_shape, gpu_out_dtype);
    const Pool2d_Constants constants{batch_size, input_height, input_width, channels, out_h, out_w, kernel_size, stride, padding};
    validatePushConstants<Pool2d_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::AVGPOOL2D_FORWARD_FP16 : Compute_Pipeline::AVGPOOL2D_FORWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    clearGpuCaches(output);
}

void avgpool2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                       uint32_t input_height, uint32_t input_width, uint32_t channels,
                       uint32_t output_height, uint32_t output_width,
                       uint32_t kernel_size, uint32_t stride, uint32_t padding)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(output_gradient, "ops::vision::avgpool2dBackward", target, batch_size)) return;
    if (stride == 0 || kernel_size == 0)
    {
        throw std::invalid_argument("ops::vision::avgpool2dBackward: stride and kernel_size must be non-zero");
    }

    validateNhwcTensorShape(output_gradient, batch_size, output_height, output_width, channels, true, "ops::vision::avgpool2dBackward");
    const Shape in_shape = makeNhwcOutputShape(output_gradient.getShape(), batch_size, input_height, input_width, channels, "ops::vision::avgpool2dBackward");
    const Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || input_height == 0 || input_width == 0 || channels == 0)
    {
        initEmptyOutput(target, input_gradient, in_shape, out_dtype);
        return;
    }

    auto grad_impl = output_gradient.getImplementation();
    if (output_height == 0 || output_width == 0)
    {
        fillConstantOutput(grad_impl, target, input_gradient, in_shape, out_dtype, 0.0f);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto out_grad_data = output_gradient.getData();
        ensureTensorTarget(input_gradient, target, in_shape, out_dtype);
        std::vector<float> in_grad_vals(in_shape.getTotalElements(), 0.0f);
        const float area = static_cast<float>(static_cast<uint64_t>(kernel_size) * kernel_size);
        const int pad = static_cast<int>(padding);
        const int k_sz = static_cast<int>(kernel_size);
        const int str = static_cast<int>(stride);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        float grad_val = 0.0f;
                        const int ky_start = (static_cast<int>(ih) + pad) % str;
                        const int kx_start = (static_cast<int>(iw) + pad) % str;
                        for (int ky = ky_start; ky < k_sz; ky += str)
                        {
                            const int out_y_calc = static_cast<int>(ih) + pad - ky;
                            if (out_y_calc < 0) continue;
                            const uint32_t oh = static_cast<uint32_t>(out_y_calc / str);
                            if (oh < output_height)
                            {
                                for (int kx = kx_start; kx < k_sz; kx += str)
                                {
                                    const int out_x_calc = static_cast<int>(iw) + pad - kx;
                                    if (out_x_calc < 0) continue;
                                    const uint32_t ow = static_cast<uint32_t>(out_x_calc / str);
                                    if (ow < output_width)
                                    {
                                        grad_val += out_grad_data[((static_cast<size_t>(n) * output_height + oh) * output_width + ow) * channels + c] / area;
                                    }
                                }
                            }
                        }
                        in_grad_vals[((static_cast<size_t>(n) * input_height + ih) * input_width + iw) * channels + c] = grad_val;
                    }
                }
            }
        }
        writeCpu(input_gradient, in_shape, out_dtype, std::move(in_grad_vals));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*grad_impl), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(input_gradient, target, in_shape, gpu_out_dtype);
    const Pool2d_Constants constants{batch_size, input_height, input_width, channels, output_height, output_width, kernel_size, stride, padding};
    validatePushConstants<Pool2d_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::AVGPOOL2D_BACKWARD_FP16 : Compute_Pipeline::AVGPOOL2D_BACKWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    clearGpuCaches(input_gradient);
}

void globalAvgPool2d(const Tensor &input, Tensor &output,
                     uint32_t input_height, uint32_t input_width, uint32_t channels)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(input, "ops::vision::globalAvgPool2d", target, batch_size)) return;

    validateNhwcTensorShape(input, batch_size, input_height, input_width, channels, true, "ops::vision::globalAvgPool2d");
    checkedMulSize(batch_size, channels, "ops::vision::globalAvgPool2d");
    const Shape out_shape{batch_size, channels};
    const Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || channels == 0)
    {
        initEmptyOutput(target, output, out_shape, out_dtype);
        return;
    }

    auto in_impl = input.getImplementation();
    if (input_height == 0 || input_width == 0)
    {
        fillConstantOutput(in_impl, target, output, out_shape, out_dtype, 0.0f);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        ensureTensorTarget(output, target, out_shape, out_dtype);
        std::vector<float> out_vals(out_shape.getTotalElements(), 0.0f);
        const float area = static_cast<float>(static_cast<uint64_t>(input_height) * input_width);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                float sum = 0.0f;
                for (uint32_t h = 0; h < input_height; ++h)
                {
                    for (uint32_t w = 0; w < input_width; ++w)
                    {
                        sum += in_data[((static_cast<size_t>(n) * input_height + h) * input_width + w) * channels + c];
                    }
                }
                out_vals[static_cast<size_t>(n) * channels + c] = sum / area;
            }
        }
        writeCpu(output, out_shape, out_dtype, std::move(out_vals));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*in_impl), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(output, target, out_shape, gpu_out_dtype);
    const Global_Pool_Constants constants{batch_size, input_height, input_width, channels};
    validatePushConstants<Global_Pool_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD_FP16 : Compute_Pipeline::GLOBAL_AVGPOOL_FORWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).getStorageBuffer()},
                                constants, (channels + 255) / 256, batch_size, 1);
    clearGpuCaches(output);
}

void globalAvgPool2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t channels)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(output_gradient, "ops::vision::globalAvgPool2dBackward", target, batch_size)) return;

    validateNhwcTensorShape(output_gradient, batch_size, 1, 1, channels, true, "ops::vision::globalAvgPool2dBackward");
    const Shape in_grad_shape = makeNhwcOutputShape(output_gradient.getShape(), batch_size, input_height, input_width, channels, "ops::vision::globalAvgPool2dBackward");
    const Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || input_height == 0 || input_width == 0 || channels == 0)
    {
        initEmptyOutput(target, input_gradient, in_grad_shape, out_dtype);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto out_grad_data = output_gradient.getData();
        ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);
        std::vector<float> in_grad_vals(in_grad_shape.getTotalElements(), 0.0f);
        const float area = static_cast<float>(static_cast<uint64_t>(input_height) * input_width);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                const float scaled = out_grad_data[static_cast<size_t>(n) * channels + c] / area;
                for (uint32_t h = 0; h < input_height; ++h)
                {
                    for (uint32_t w = 0; w < input_width; ++w)
                    {
                        in_grad_vals[((static_cast<size_t>(n) * input_height + h) * input_width + w) * channels + c] = scaled;
                    }
                }
            }
        }
        writeCpu(input_gradient, in_grad_shape, out_dtype, std::move(in_grad_vals));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation()), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(input_gradient, target, in_grad_shape, gpu_out_dtype);
    const Global_Pool_Constants constants{batch_size, input_height, input_width, channels};
    validatePushConstants<Global_Pool_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD_FP16 : Compute_Pipeline::GLOBAL_AVGPOOL_BACKWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    clearGpuCaches(input_gradient);
}

void globalMaxPool2d(const Tensor &input, Tensor &output, Tensor &output_mask,
                     uint32_t input_height, uint32_t input_width, uint32_t channels)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(input, "ops::vision::globalMaxPool2d", target, batch_size)) return;

    validateNhwcTensorShape(input, batch_size, input_height, input_width, channels, true, "ops::vision::globalMaxPool2d");
    checkedMulSize(batch_size, channels, "ops::vision::globalMaxPool2d");
    const Shape out_shape{batch_size, channels};
    const Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || channels == 0)
    {
        initEmptyOutput(target, output, out_shape, out_dtype);
        initEmptyOutput(target, output_mask, out_shape, Data_Type::FLOAT32);
        return;
    }

    auto in_impl = input.getImplementation();
    if (input_height == 0 || input_width == 0)
    {
        fillConstantOutput(in_impl, target, output, out_shape, out_dtype, 0.0f);
        fillConstantOutput(in_impl, target, output_mask, out_shape, Data_Type::FLOAT32, -1.0f);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        ensureTensorTarget(output, target, out_shape, out_dtype);
        ensureTensorTarget(output_mask, target, out_shape, Data_Type::FLOAT32);

        const size_t out_elems = out_shape.getTotalElements();
        std::vector<float> out_vec(out_elems, 0.0f);
        std::vector<float> mask_vec(out_elems, -1.0f);
        const size_t spatial_total = static_cast<size_t>(input_height) * input_width;

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                bool has_valid = false;
                float best_val = 0.0f;
                float best_spatial = -1.0f;
                for (size_t s = 0; s < spatial_total; ++s)
                {
                    const float val = in_data[(static_cast<size_t>(n) * spatial_total + s) * channels + c];
                    if (!std::isnan(val) && (!has_valid || val > best_val))
                    {
                        best_val = val;
                        best_spatial = static_cast<float>(s);
                        has_valid = true;
                    }
                }
                const size_t out_idx = static_cast<size_t>(n) * channels + c;
                out_vec[out_idx] = best_val;
                mask_vec[out_idx] = best_spatial;
            }
        }
        writeCpu(output, out_shape, out_dtype, std::move(out_vec));
        writeCpu(output_mask, out_shape, Data_Type::FLOAT32, std::move(mask_vec));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*in_impl), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(output, target, out_shape, gpu_out_dtype);
    ensureTensorTarget(output_mask, target, out_shape, Data_Type::FLOAT32);

    const Global_Pool_Constants constants{batch_size, input_height, input_width, channels};
    validatePushConstants<Global_Pool_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::GLOBAL_MAXPOOL_FORWARD_FP16 : Compute_Pipeline::GLOBAL_MAXPOOL_FORWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self,
                                 Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).getStorageBuffer(),
                                 Gpu_Tensor_Impl::castToGpu(*output_mask.getImplementation()).getStorageBuffer()},
                                constants, (channels + 255) / 256, batch_size, 1);
    clearGpuCaches(output);
    clearGpuCaches(output_mask);
}

void globalMaxPool2dBackward(const Tensor &output_gradient, const Tensor &mask, Tensor &input_gradient,
                             uint32_t input_height, uint32_t input_width, uint32_t channels)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateBackwardPair(output_gradient, mask, "ops::vision::globalMaxPool2dBackward", target, batch_size)) return;

    validateNhwcTensorShape(output_gradient, batch_size, 1, 1, channels, true, "ops::vision::globalMaxPool2dBackward");
    if (batch_size > 0 && channels > 0 && mask.getTotalElements() != output_gradient.getTotalElements())
    {
        throw std::invalid_argument("ops::vision::globalMaxPool2dBackward: Mask element count does not match output gradient");
    }
    const Shape in_grad_shape = makeNhwcOutputShape(output_gradient.getShape(), batch_size, input_height, input_width, channels, "ops::vision::globalMaxPool2dBackward");
    const Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || input_height == 0 || input_width == 0 || channels == 0)
    {
        initEmptyOutput(target, input_gradient, in_grad_shape, out_dtype);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto out_grad_data = output_gradient.getData();
        const auto mask_data = mask.getData();
        ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);

        const size_t spatial_total = static_cast<size_t>(input_height) * input_width;
        std::vector<float> in_grad_vec(in_grad_shape.getTotalElements(), 0.0f);
        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                const size_t out_idx = static_cast<size_t>(n) * channels + c;
                const float m = mask_data[out_idx];
                if (m >= 0.0f)
                {
                    const size_t max_spatial = static_cast<size_t>(std::round(m));
                    if (max_spatial < spatial_total)
                    {
                        in_grad_vec[(static_cast<size_t>(n) * spatial_total + max_spatial) * channels + c] = out_grad_data[out_idx];
                    }
                }
            }
        }
        writeCpu(input_gradient, in_grad_shape, out_dtype, std::move(in_grad_vec));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation()), contig_self, effective_self, gpu_out_dtype, buf_self);
    auto buf_mask = getGpuFloat32ReadBuffer(Gpu_Tensor_Impl::castToGpu(*mask.getImplementation()), *effective_self);

    ensureTensorTarget(input_gradient, target, in_grad_shape, gpu_out_dtype);
    const Global_Pool_Constants constants{batch_size, input_height, input_width, channels};
    validatePushConstants<Global_Pool_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::GLOBAL_MAXPOOL_BACKWARD_FP16 : Compute_Pipeline::GLOBAL_MAXPOOL_BACKWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, buf_mask, Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    clearGpuCaches(input_gradient);
}

void upsample2d(const Tensor &input, Tensor &output,
                uint32_t input_height, uint32_t input_width, uint32_t channels,
                uint32_t scale_factor)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(input, "ops::vision::upsample2d", target, batch_size)) return;
    if (scale_factor == 0)
    {
        throw std::invalid_argument("ops::vision::upsample2d: scale_factor must be non-zero");
    }

    const uint64_t out_h_u64 = static_cast<uint64_t>(input_height) * static_cast<uint64_t>(scale_factor);
    const uint64_t out_w_u64 = static_cast<uint64_t>(input_width) * static_cast<uint64_t>(scale_factor);
    if (out_h_u64 > std::numeric_limits<uint32_t>::max() || out_w_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::upsample2d: Output dimensions exceed uint32_t range");
    }
    const uint32_t out_h = static_cast<uint32_t>(out_h_u64);
    const uint32_t out_w = static_cast<uint32_t>(out_w_u64);
    validateNhwcTensorShape(input, batch_size, input_height, input_width, channels, true, "ops::vision::upsample2d");
    const Shape out_shape = makeNhwcOutputShape(input.getShape(), batch_size, out_h, out_w, channels, "ops::vision::upsample2d");
    const Data_Type out_dtype = (input.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || input_height == 0 || input_width == 0 || channels == 0)
    {
        initEmptyOutput(target, output, out_shape, out_dtype);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto in_data = input.getData();
        ensureTensorTarget(output, target, out_shape, out_dtype);
        std::vector<float> out_data(out_shape.getTotalElements(), 0.0f);

        for (uint32_t n = 0; n < batch_size; ++n)
        {
            for (uint32_t ih = 0; ih < input_height; ++ih)
            {
                for (uint32_t iw = 0; iw < input_width; ++iw)
                {
                    for (uint32_t c = 0; c < channels; ++c)
                    {
                        const float val = in_data[((static_cast<size_t>(n) * input_height + ih) * input_width + iw) * channels + c];
                        for (uint32_t sy = 0; sy < scale_factor; ++sy)
                        {
                            const uint32_t oh = ih * scale_factor + sy;
                            for (uint32_t sx = 0; sx < scale_factor; ++sx)
                            {
                                out_data[((static_cast<size_t>(n) * out_h + oh) * out_w + (iw * scale_factor + sx)) * channels + c] = val;
                            }
                        }
                    }
                }
            }
        }
        writeCpu(output, out_shape, out_dtype, std::move(out_data));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*input.getImplementation()), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(output, target, out_shape, gpu_out_dtype);
    const Upsample_Constants constants{batch_size, input_height, input_width, channels, out_h, out_w, scale_factor};
    validatePushConstants<Upsample_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::UPSAMPLE2D_FORWARD_FP16 : Compute_Pipeline::UPSAMPLE2D_FORWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, Gpu_Tensor_Impl::castToGpu(*output.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (out_w + 15) / 16, batch_size * out_h);
    clearGpuCaches(output);
}

void upsample2dBackward(const Tensor &output_gradient, Tensor &input_gradient,
                        uint32_t input_height, uint32_t input_width, uint32_t channels,
                        uint32_t scale_factor)
{
    Execution_Target target = Execution_Target::CPU;
    uint32_t batch_size = 0;
    if (!validateSingleInput(output_gradient, "ops::vision::upsample2dBackward", target, batch_size)) return;
    if (scale_factor == 0)
    {
        throw std::invalid_argument("ops::vision::upsample2dBackward: scale_factor must be non-zero");
    }

    const uint64_t out_h_u64 = static_cast<uint64_t>(input_height) * static_cast<uint64_t>(scale_factor);
    const uint64_t out_w_u64 = static_cast<uint64_t>(input_width) * static_cast<uint64_t>(scale_factor);
    if (out_h_u64 > std::numeric_limits<uint32_t>::max() || out_w_u64 > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("ops::vision::upsample2dBackward: Output dimensions exceed uint32_t range");
    }
    const uint32_t out_h = static_cast<uint32_t>(out_h_u64);
    const uint32_t out_w = static_cast<uint32_t>(out_w_u64);
    validateNhwcTensorShape(output_gradient, batch_size, out_h, out_w, channels, true, "ops::vision::upsample2dBackward");
    const Shape in_grad_shape = makeNhwcOutputShape(output_gradient.getShape(), batch_size, input_height, input_width, channels, "ops::vision::upsample2dBackward");
    const Data_Type out_dtype = (output_gradient.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;

    if (batch_size == 0 || input_height == 0 || input_width == 0 || channels == 0)
    {
        initEmptyOutput(target, input_gradient, in_grad_shape, out_dtype);
        return;
    }

    if (target == Execution_Target::CPU)
    {
        const auto out_grad_data = output_gradient.getData();
        ensureTensorTarget(input_gradient, target, in_grad_shape, out_dtype);
        std::vector<float> in_grad_data(in_grad_shape.getTotalElements(), 0.0f);

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
                            const uint32_t oh = ih * scale_factor + sy;
                            for (uint32_t sx = 0; sx < scale_factor; ++sx)
                            {
                                sum_grad += out_grad_data[((static_cast<size_t>(n) * out_h + oh) * out_w + (iw * scale_factor + sx)) * channels + c];
                            }
                        }
                        in_grad_data[((static_cast<size_t>(n) * input_height + ih) * input_width + iw) * channels + c] = sum_grad;
                    }
                }
            }
        }
        writeCpu(input_gradient, in_grad_shape, out_dtype, std::move(in_grad_data));
        return;
    }

    std::shared_ptr<Gpu_Tensor_Impl> contig_self;
    const Gpu_Tensor_Impl *effective_self = nullptr;
    Data_Type gpu_out_dtype = Data_Type::FLOAT32;
    std::shared_ptr<gpu::vector> buf_self;
    prepareGpuPoolingInput(Gpu_Tensor_Impl::castToGpu(*output_gradient.getImplementation()), contig_self, effective_self, gpu_out_dtype, buf_self);

    ensureTensorTarget(input_gradient, target, in_grad_shape, gpu_out_dtype);
    const Upsample_Constants constants{batch_size, input_height, input_width, channels, out_h, out_w, scale_factor};
    validatePushConstants<Upsample_Constants>();
    const Compute_Pipeline pipe = (gpu_out_dtype == Data_Type::FLOAT16) ? Compute_Pipeline::UPSAMPLE2D_BACKWARD_FP16 : Compute_Pipeline::UPSAMPLE2D_BACKWARD;
    effective_self->pushToGraph(pipe,
                                {buf_self, Gpu_Tensor_Impl::castToGpu(*input_gradient.getImplementation()).getStorageBuffer()},
                                constants, (channels + 15) / 16, (input_width + 15) / 16, batch_size * input_height);
    clearGpuCaches(input_gradient);
}

}

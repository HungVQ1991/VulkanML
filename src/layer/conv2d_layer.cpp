#include "layer/conv2d_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/gpu_vector.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"
#include "ops/vision/conv2d.h"

Conv2d_Layer::~Conv2d_Layer() noexcept = default;

namespace
{
static uint32_t checkedU32(size_t value, const char *context)
{
    if (value > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error(std::format("{}: Value {} exceeds UINT32_MAX", context, value));
    }
    return static_cast<uint32_t>(value);
}

static uint32_t ceilDivU32(uint32_t n, uint32_t d) noexcept
{
    return (n == 0u) ? 0u : (1u + (n - 1u) / d);
}
}

std::pair<uint32_t, uint32_t> Conv2d_Layer::validateAndComputeOutput(
    uint32_t _height, uint32_t _width,
    uint32_t _input_channels, uint32_t _output_channels,
    uint32_t _kernel_size, uint32_t _stride, uint32_t _padding)
{
    if (_height == 0 || _width == 0 || _input_channels == 0 || _output_channels == 0 ||
        _kernel_size == 0 || _stride == 0)
    {
        Logger::logMessage(Input_Format{"Conv2d_Layer: Invalid non-positive parameter (h={}, w={}, cin={}, cout={}, k={}, s={})",
                                        _height, _width, _input_channels, _output_channels, _kernel_size, _stride},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CONV2D_COMPUTE);
        throw std::invalid_argument("Conv2d_Layer: dimensions, channels, kernel_size, and stride must be > 0");
    }
    const uint64_t padded_h = static_cast<uint64_t>(_height) + 2ULL * static_cast<uint64_t>(_padding);
    const uint64_t padded_w = static_cast<uint64_t>(_width) + 2ULL * static_cast<uint64_t>(_padding);
    if (padded_h < static_cast<uint64_t>(_kernel_size) || padded_w < static_cast<uint64_t>(_kernel_size))
    {
        throw std::invalid_argument("Conv2d_Layer: Padded input dimensions must be >= kernel_size");
    }
    const uint64_t out_h = (padded_h - static_cast<uint64_t>(_kernel_size)) / static_cast<uint64_t>(_stride) + 1ULL;
    const uint64_t out_w = (padded_w - static_cast<uint64_t>(_kernel_size)) / static_cast<uint64_t>(_stride) + 1ULL;
    if (out_h > std::numeric_limits<uint32_t>::max() || out_w > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Conv2d_Layer: Output dimensions exceed uint32_t range");
    }
    const uint64_t k_sq = static_cast<uint64_t>(_kernel_size) * _kernel_size;
    if (k_sq > std::numeric_limits<size_t>::max() / _input_channels)
    {
        throw std::overflow_error("Conv2d_Layer: Kernel patch size K overflows size_t");
    }
    const size_t k_dim = static_cast<size_t>(k_sq) * _input_channels;
    if (k_dim > std::numeric_limits<size_t>::max() / _output_channels)
    {
        throw std::overflow_error("Conv2d_Layer: Weight count overflows size_t");
    }
    const uint64_t in_spatial = static_cast<uint64_t>(_height) * _width;
    const uint64_t out_spatial = out_h * out_w;
    if (in_spatial > std::numeric_limits<size_t>::max() / _input_channels ||
        out_spatial > std::numeric_limits<size_t>::max() / _output_channels)
    {
        throw std::overflow_error("Conv2d_Layer: Feature map size per sample overflows size_t");
    }
    return {static_cast<uint32_t>(out_h), static_cast<uint32_t>(out_w)};
}

void Conv2d_Layer::ensureTernaryGpuBuffers(size_t weight_count)
{
    if (execution_target != Execution_Target::VULKAN_GPU)
    {
        return;
    }
    auto &context = Execution_Engine::getInstance().getContext();
    if (!gpu_gamma_buffer)
    {
        gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
        gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
    }
    const uint32_t total_elements = checkedU32(weight_count, "Conv2d_Layer::ensureTernaryGpuBuffers");
    if (total_elements > 0)
    {
        const uint32_t num_workgroups = std::min(ceilDivU32(total_elements, 256u), 1024u);
        if (!gpu_partial_sums || gpu_partial_sums->getSize() < num_workgroups)
        {
            gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
        }
    }
}

void Conv2d_Layer::initializeWeights()
{
    const size_t k_per_out = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
    const size_t weight_count = k_per_out * output_channels;
    std::vector<float> host_weights(weight_count);
    std::vector<float> host_biases(output_channels, 0.0f);

    float fan_in = static_cast<float>(k_per_out);
    float standard_deviation = (fan_in > 0.0f) ? std::sqrt(2.0f / fan_in) : 0.02f;

    std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

    for (size_t i = 0; i < weight_count; ++i)
    {
        host_weights[i] = normal_distribution(generator);
    }

    if (isTernary())
    {
        master_weights = Tensor(1, weight_count, host_weights, execution_target);
        master_weights_gradient_tensor = Tensor(1, weight_count, execution_target);
        biases = Tensor(1, output_channels, host_biases, execution_target);
        biases_gradient_tensor = Tensor(1, output_channels, execution_target);
        weights = Tensor(0, 0, execution_target);
        weights_gradient_tensor = Tensor(0, 0, execution_target);
        master_weights_gradient_tensor.zero();
        biases_gradient_tensor.zero();
        has_master_weights = true;
        has_packed_inference = false;
        ensureTernaryGpuBuffers(weight_count);
    }
    else
    {
        weights = Tensor(1, weight_count, host_weights, execution_target);
        biases = Tensor(1, output_channels, host_biases, execution_target);
        weights_gradient_tensor = Tensor(1, weight_count, execution_target);
        biases_gradient_tensor = Tensor(1, output_channels, execution_target);
        weights_gradient_tensor.zero();
        biases_gradient_tensor.zero();
        has_master_weights = false;
        has_packed_inference = false;
    }
}

Conv2d_Layer::Conv2d_Layer(uint32_t _height,
                           uint32_t _width,
                           uint32_t _input_channels,
                           uint32_t _output_channels,
                           uint32_t _kernel_size,
                           uint32_t _stride,
                           uint32_t _padding,
                           Execution_Target _execution_target,
                           Data_Type _data_type)
    : input_height(_height),
      input_width(_width),
      input_channels(_input_channels),
      output_channels(_output_channels),
      kernel_size(_kernel_size),
      stride(_stride),
      padding(_padding),
      execution_target(_execution_target),
      data_type(_data_type),
      is_forward_completed(false),
      weights(0, 0, _execution_target),
      biases(0, 0, _execution_target),
      weights_gradient_tensor(0, 0, _execution_target),
      biases_gradient_tensor(0, 0, _execution_target),
      step_weights_grad_scratch(0, 0, _execution_target),
      step_biases_grad_scratch(0, 0, _execution_target),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      input_tensor_fp16(0, 0, _execution_target),
      output_tensor_fp16(0, 0, _execution_target),
      input_gradient_tensor_fp16(0, 0, _execution_target),
      output_gradient_tensor_fp16(0, 0, _execution_target),
      im2col_scratch(0, 0, _execution_target),
      im2col_scratch_fp16(0, 0, _execution_target),
      master_weights(0, 0, _execution_target),
      master_weights_gradient_tensor(0, 0, _execution_target),
      effective_weights(0, 0, _execution_target),
      effective_weights_fp16(0, 0, _execution_target),
      packed_weights(0, 0, _execution_target),
      gamma_tensor(0, 0, _execution_target)
{
    if (_data_type == Data_Type::FLOAT16)
    {
        is_mixed_precision_enabled = true;
    }
    auto [out_h, out_w] = validateAndComputeOutput(
        input_height, input_width, input_channels, output_channels, kernel_size, stride, padding);
    output_height = out_h;
    output_width = out_w;
    initializeWeights();
}

void Conv2d_Layer::packWeightsCpu()
{
    size_t K = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
    size_t N = output_channels;
    size_t total = K * N;
    if (total == 0) return;

    const auto &w_data = master_weights.getData();
    double sum_abs = 0.0;
    for (size_t i = 0; i < total; ++i)
    {
        sum_abs += std::abs(static_cast<double>(w_data[i]));
    }
    float gamma = std::max(1e-5f, static_cast<float>(sum_abs / static_cast<double>(total)));
    float inv_gamma = 1.0f / gamma;

    gamma_tensor = Tensor(Shape{ 2 }, std::vector<float>{ gamma, inv_gamma }, Execution_Target::CPU);

    const uint32_t k_u32 = checkedU32(K, "Conv2d_Layer::packWeightsCpu(K)");
    const uint32_t n_u32 = checkedU32(N, "Conv2d_Layer::packWeightsCpu(N)");
    const uint32_t words_per_row = ceilDivU32(k_u32, 16u);
    const size_t total_words = static_cast<size_t>(n_u32) * words_per_row;
    std::vector<uint32_t> packed_data(total_words, 0u);

    for (uint32_t n = 0; n < n_u32; ++n)
    {
        for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
        {
            uint32_t word = 0u;
            for (uint32_t b = 0; b < 16; ++b)
            {
                uint32_t k = w_idx * 16 + b;
                if (k < K)
                {
                    float w = w_data[k * N + n];
                    float q = std::clamp(std::floor(std::abs(w) * inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
                    uint32_t code = static_cast<uint32_t>(static_cast<int>(q)) & 3u;
                    word |= (code << (b * 2u));
                }
            }
            packed_data[n * words_per_row + w_idx] = word;
        }
    }
    packed_weights = Tensor(Shape{ N, K }, Data_Type::TERNARY_INT2, Execution_Target::CPU);
    packed_weights.uploadPackedData(packed_data);
}

void Conv2d_Layer::packWeightsGpu()
{
    size_t K_sz = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
    size_t weight_count = K_sz * output_channels;
    ensureTernaryGpuBuffers(weight_count);
    auto &engine = Execution_Engine::getInstance();
    const uint32_t K = checkedU32(K_sz, "Conv2d_Layer::packWeightsGpu(K)");
    const uint32_t N = checkedU32(output_channels, "Conv2d_Layer::packWeightsGpu(N)");
    const uint32_t total_elements = checkedU32(weight_count, "Conv2d_Layer::packWeightsGpu(total)");
    if (total_elements == 0) return;

    const uint32_t num_workgroups = std::min(ceilDivU32(total_elements, 256u), 1024u);

    const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*master_weights.getImplementation());
    auto contig_w = w_gpu.ensureContiguousSelf();
    const auto *eff_w = contig_w ? contig_w.get() : &w_gpu;
    auto w_buf = eff_w->getStorageBuffer();

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS1;
        node.buffers = { w_buf, gpu_partial_sums };
        struct PushConsts { uint32_t total; } pcs{ total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = num_workgroups;
        node.is_barrier_required_after = true;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS2;
        node.buffers = { gpu_partial_sums, gpu_gamma_buffer };
        struct PushConsts { uint32_t num_partials; uint32_t total; } pcs{ num_workgroups, total_elements };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = 1;
        node.is_barrier_required_after = true;
        engine.getCurrentGraph().addNode(std::move(node));
    }

    const uint32_t words_per_row = ceilDivU32(K, 16u);
    if (packed_weights.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
        packed_weights.getDataType() != Data_Type::TERNARY_INT2 ||
        packed_weights.getShape() != Shape{ N, K })
    {
        packed_weights = Tensor(Shape{ N, K }, Data_Type::TERNARY_INT2, Execution_Target::VULKAN_GPU);
    }
    const auto &packed_gpu = Gpu_Tensor_Impl::castToGpu(*packed_weights.getImplementation());
    auto packed_buf = packed_gpu.getStorageBuffer();

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::QUANTIZE_PACK_TERNARY;
        node.buffers = { w_buf, gpu_gamma_buffer, packed_buf };
        struct PushConsts { uint32_t K; uint32_t N; uint32_t words_per_row; } pcs{ K, N, words_per_row };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = ceilDivU32(words_per_row, 16u);
        node.workgroup_count_y = ceilDivU32(N, 16u);
        node.is_barrier_required_after = true;
        engine.getCurrentGraph().addNode(std::move(node));
    }
}

void Conv2d_Layer::prepareInference()
{
    has_valid_forward_im2col_fp16 = false;
    if (isTernary())
    {
        if (has_packed_inference && !has_master_weights)
        {
            return;
        }
        if (!has_master_weights)
        {
            throw std::logic_error("Conv2d_Layer::prepareInference: Missing master weights");
        }
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            packWeightsGpu();
        }
        else
        {
            packWeightsCpu();
        }
        effective_weights = Tensor(0, 0, execution_target);
        effective_weights_fp16 = Tensor(0, 0, execution_target);
        has_packed_inference = true;
        Logger::logMessage(Input_Format{"Conv2d_Layer::prepareInference: Packed weights to TERNARY_INT2 (Cin={}, Cout={}, K={})",
                                        input_channels, output_channels, kernel_size},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::TERNARY_METRICS | Log_Feature::LAYER_INSPECTION);
    }
}

void Conv2d_Layer::quantizeToTernary()
{
    has_valid_forward_im2col_fp16 = false;
    if (isTernary())
    {
        return;
    }

    size_t weight_count = static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels;
    data_type = Data_Type::TERNARY_INT2;
    master_weights = weights;
    weights = Tensor(0, 0, execution_target);
    master_weights_gradient_tensor = Tensor(1, weight_count, execution_target);
    master_weights_gradient_tensor.zero();
    weights_gradient_tensor = Tensor(0, 0, execution_target);
    has_master_weights = true;
    has_packed_inference = false;

    ensureTernaryGpuBuffers(weight_count);
    prepareInference();
    invalidateWeightCache();
    Logger::logMessage(Input_Format{"Conv2d_Layer::quantizeToTernary: Quantized layer to TERNARY_INT2 (Cin={}, Cout={}, K={})",
                                    input_channels, output_channels, kernel_size},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::TERNARY_METRICS | Log_Feature::LAYER_INSPECTION);
}

void Conv2d_Layer::dequantizeFromTernary()
{
    has_valid_forward_im2col_fp16 = false;
    if (!isTernary())
    {
        return;
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &engine = Execution_Engine::getInstance();
        if (!engine.getCurrentGraph().getNodes().empty())
        {
            engine.executeGraph();
        }
        engine.waitIdle();
    }

    const size_t K = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
    const size_t N = output_channels;
    const size_t weight_count = K * N;

    if (has_master_weights && !master_weights.isEmpty())
    {
        weights = master_weights.clone();
        if (weights.getExecutionTarget() != execution_target)
        {
            weights.setExecutionTarget(execution_target);
        }
        if (weights.getDataType() != Data_Type::FLOAT32)
        {
            weights.setDataType(Data_Type::FLOAT32);
        }
        if (weights.getShape() != Shape{ 1, weight_count })
        {
            weights.reshape(Shape{ 1, weight_count });
        }
    }
    else if (has_packed_inference && !packed_weights.isEmpty() && K > 0 && N > 0)
    {
        float gamma_val = 1.0f;
        if (execution_target == Execution_Target::VULKAN_GPU && gpu_gamma_buffer &&
            gpu_gamma_buffer->getBufferSizeInBytes() >= sizeof(float))
        {
            std::vector<float> gd(2, 1.0f);
            gpu_gamma_buffer->downloadData(gd);
            gamma_val = gd[0];
        }
        else if (!gamma_tensor.isEmpty())
        {
            const auto &gd = gamma_tensor.getData();
            if (!gd.empty())
            {
                gamma_val = gd[0];
            }
        }

        const auto &packed = packed_weights.getPackedData();
        std::vector<float> tmp_nk(N * K, 0.0f);
        if (!packed.empty())
        {
            Cpu_Tensor_Impl::unpackTernary(packed.data(), tmp_nk.data(), N, K, gamma_val);
        }
        std::vector<float> w_kn(weight_count, 0.0f);
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t k = 0; k < K; ++k)
            {
                w_kn[k * N + n] = tmp_nk[n * K + k];
            }
        }
        weights = Tensor(Shape{ 1, weight_count }, std::move(w_kn), execution_target);
    }
    else
    {
        weights = Tensor(1, weight_count, execution_target);
    }

    weights_gradient_tensor = Tensor(1, weight_count, execution_target);
    weights_gradient_tensor.zero();
    data_type = is_mixed_precision_enabled ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    has_master_weights = false;
    has_packed_inference = false;
    master_weights = Tensor(0, 0, execution_target);
    master_weights_gradient_tensor = Tensor(0, 0, execution_target);
    packed_weights = Tensor(0, 0, execution_target);
    effective_weights = Tensor(0, 0, execution_target);
    effective_weights_fp16 = Tensor(0, 0, execution_target);
    gamma_tensor = Tensor(0, 0, execution_target);
    gpu_gamma_buffer.reset();
    gpu_partial_sums.reset();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    invalidateWeightCache();
}

Tensor Conv2d_Layer::forward(const Tensor &_input_tensor)
{
    has_valid_forward_im2col_fp16 = false;
    if (!_input_tensor.getImplementation())
    {
        throw std::invalid_argument("Conv2d_Layer::forward: Input tensor is uninitialized");
    }
    if (_input_tensor.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Conv2d_Layer::forward: Execution_Target mismatch");
    }
    if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Conv2d_Layer::forward: TERNARY_INT2 input tensor is not supported");
    }

    const Shape &in_shape = _input_tensor.getShape();
    const size_t expected_in_cols = static_cast<size_t>(input_height) * input_width * input_channels;
    const bool is_rank4_input = (in_shape.getRank() == 4);
    if (in_shape.getRank() == 2)
    {
        if (in_shape[1] != expected_in_cols)
        {
            throw std::invalid_argument("Conv2d_Layer::forward: Rank-2 input column count does not match H*W*Cin");
        }
    }
    else if (is_rank4_input)
    {
        if (in_shape[1] != input_height || in_shape[2] != input_width || in_shape[3] != input_channels)
        {
            throw std::invalid_argument("Conv2d_Layer::forward: Rank-4 NHWC input dimensions do not match [N, H, W, Cin]");
        }
    }
    else
    {
        throw std::invalid_argument("Conv2d_Layer::forward: Expected rank-2 [N, H*W*Cin] or rank-4 [N, H, W, Cin] input");
    }

    const size_t batch_size_sz = in_shape[0];
    if (batch_size_sz > std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error("Conv2d_Layer::forward: Batch size exceeds uint32_t limit");
    }

    input_tensor = _input_tensor;
    const size_t K = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
    const size_t N = output_channels;
    const size_t weight_count = K * N;
    const uint32_t batch_size = static_cast<uint32_t>(batch_size_sz);
    const uint32_t out_h = output_height;
    const uint32_t out_w = output_width;
    const size_t out_per_sample = static_cast<size_t>(out_h) * out_w * output_channels;
    const Shape out_shape = is_rank4_input
        ? Shape{ batch_size, out_h, out_w, output_channels }
        : Shape{ batch_size, out_per_sample };

    auto finalize_output = [&](const Tensor &src) -> Tensor {
        output_tensor = src;
        if (output_tensor.getShape() != out_shape)
        {
            output_tensor.reshape(out_shape);
        }
        is_forward_completed = true;
        return output_tensor;
    };

    if (isTernary())
    {
        if (has_packed_inference)
        {
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                auto &engine = Execution_Engine::getInstance();
                auto &context = engine.getContext();
                bool fp16_hw = context.isFloat16Enabled();

                if (batch_size == 0)
                {
                    Data_Type empty_dtype = fp16_hw ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
                    if (!output_tensor.getImplementation() ||
                        output_tensor.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
                        output_tensor.getDataType() != empty_dtype ||
                        output_tensor.getShape() != out_shape)
                    {
                        output_tensor = Tensor(out_shape, empty_dtype, Execution_Target::VULKAN_GPU);
                    }
                    auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output_tensor.getImplementation());
                    out_gpu.clearHostCache();
                    out_gpu.invalidateFp16Cache();
                    is_forward_completed = true;
                    return output_tensor;
                }

                if (fp16_hw)
                {
                    if (input_tensor.getDataType() != Data_Type::FLOAT16)
                    {
                        input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                    }
                    else
                    {
                        input_tensor_fp16 = input_tensor;
                    }
                }

                uint64_t m_u64 = static_cast<uint64_t>(batch_size) * out_h * out_w;
                if (m_u64 > std::numeric_limits<uint32_t>::max())
                {
                    throw std::overflow_error("Conv2d_Layer::forward: M dimension exceeds uint32_t limit");
                }
                uint32_t M = static_cast<uint32_t>(m_u64);
                constexpr uint64_t MAX_IM2COL_SCRATCH_BYTES = 128 * 1024 * 1024;
                uint64_t scratch_bytes = m_u64 * K * sizeof(uint16_t);

                if (fp16_hw && scratch_bytes <= MAX_IM2COL_SCRATCH_BYTES)
                {
                    if (!gpu_gamma_buffer)
                    {
                        throw std::logic_error("Conv2d_Layer::forward: Missing GPU gamma buffer in packed ternary inference");
                    }
                    if (im2col_scratch_fp16.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
                        im2col_scratch_fp16.getDataType() != Data_Type::FLOAT16 ||
                        im2col_scratch_fp16.getShape() != Shape{ M, K })
                    {
                        im2col_scratch_fp16 = Tensor(Shape{ M, K }, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
                    }

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
                    } constants{ batch_size, input_height, input_width, input_channels, out_h, out_w, output_channels, kernel_size, stride, padding };

                    const auto &in_gpu = Gpu_Tensor_Impl::castToGpu(*input_tensor_fp16.getImplementation());
                    auto contig_in = in_gpu.ensureContiguousSelf();
                    const auto *eff_in = contig_in ? contig_in.get() : &in_gpu;
                    auto &scratch_gpu = Gpu_Tensor_Impl::castToGpu(*im2col_scratch_fp16.getImplementation());
                    const uint32_t k_u32 = checkedU32(K, "Conv2d_Layer::forward(K)");
                    const uint32_t n_u32 = checkedU32(N, "Conv2d_Layer::forward(N)");
                    const uint64_t total_im2col_items = ((input_channels & 7u) == 0u) ? ((m_u64 * K) >> 3ULL) : (m_u64 * K);
                    auto [im2col_wg_x, im2col_wg_y] = context.compute2DWorkgroupGrid((total_im2col_items + 255ULL) / 256ULL);
                    eff_in->pushToGraph(Compute_Pipeline::CONV2D_IM2COL_FP16,
                                        { eff_in->getStorageBuffer(), scratch_gpu.getStorageBuffer() },
                                        constants,
                                        im2col_wg_x, im2col_wg_y, 1);

                    if (output_tensor.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
                        output_tensor.getDataType() != Data_Type::FLOAT16 ||
                        output_tensor.getTotalElements() != out_shape.getTotalElements())
                    {
                        output_tensor = Tensor(out_shape, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
                    }
                    else
                    {
                        output_tensor.reshape(out_shape);
                    }
                    auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output_tensor.getImplementation());

                    const uint32_t words_per_row = ceilDivU32(k_u32, 16u);
                    const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*packed_weights.getImplementation());
                    const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*biases.getImplementation());
                    auto contig_b = b_gpu.ensureContiguousSelf();
                    const auto *eff_b = contig_b ? contig_b.get() : &b_gpu;
                    auto buf_b = eff_b->getEffectiveFp16Storage();

                    Compute_Node node;
                    node.pipeline_id = Compute_Pipeline::MATMUL_TERNARY_FP16;
                    node.buffers = { scratch_gpu.getStorageBuffer(), w_gpu.getStorageBuffer(), buf_b, gpu_gamma_buffer, out_gpu.getStorageBuffer() };
                    struct PushConsts {
                        uint32_t M; uint32_t K; uint32_t N; uint32_t words_per_row; uint32_t has_bias;
                    } pcs{ M, k_u32, n_u32, words_per_row, 1u };
                    const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
                    node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
                    node.workgroup_count_x = ceilDivU32(n_u32, 16u);
                    node.workgroup_count_y = ceilDivU32(M, 16u);
                    engine.getCurrentGraph().addNode(std::move(node));
                    out_gpu.clearHostCache();
                    out_gpu.invalidateFp16Cache();

                    is_forward_completed = true;
                    return output_tensor;
                }
                else
                {
                    engine.executeGraph();
                    engine.waitIdle();

                    float gamma = 1.0f;
                    if (gpu_gamma_buffer && gpu_gamma_buffer->getBufferSizeInBytes() >= sizeof(float))
                    {
                        std::vector<float> g(2, 1.0f);
                        gpu_gamma_buffer->downloadData(g);
                        gamma = g[0];
                    }
                    else if (!gamma_tensor.isEmpty())
                    {
                        gamma = gamma_tensor.getData()[0];
                    }

                    const uint32_t k_u32 = checkedU32(K, "Conv2d_Layer::forward(gpu fallback K)");
                    const uint32_t n_u32 = checkedU32(N, "Conv2d_Layer::forward(gpu fallback N)");
                    const uint32_t words_per_row = ceilDivU32(k_u32, 16u);
                    const auto &packed_data = packed_weights.getPackedData();
                    std::vector<float> unpacked(K * N, 0.0f);
                    for (uint32_t n = 0; n < n_u32; ++n)
                    {
                        for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
                        {
                            uint32_t word = (n * words_per_row + w_idx < packed_data.size()) ? packed_data[n * words_per_row + w_idx] : 0u;
                            for (uint32_t b = 0; b < 16; ++b)
                            {
                                uint32_t k = w_idx * 16 + b;
                                if (k < k_u32)
                                {
                                    uint32_t code = (word >> (b * 2u)) & 3u;
                                    float val = (code == 1u) ? 1.0f : ((code == 3u) ? -1.0f : 0.0f);
                                    unpacked[k * N + n] = val * gamma;
                                }
                            }
                        }
                    }

                    if (fp16_hw)
                    {
                        effective_weights_fp16 = Tensor(Shape{ 1, K * N }, unpacked, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
                        context.executePendingTransfers();
                        ops::vision::conv2d(input_tensor_fp16, effective_weights_fp16, biases, output_tensor_fp16,
                                            input_height, input_width, input_channels, output_channels,
                                            kernel_size, stride, padding, &im2col_scratch_fp16);
                        return finalize_output(output_tensor_fp16);
                    }
                    else
                    {
                        effective_weights = Tensor(Shape{ 1, K * N }, unpacked, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
                        context.executePendingTransfers();
                        ops::vision::conv2d(input_tensor, effective_weights, biases, output_tensor,
                                            input_height, input_width, input_channels, output_channels,
                                            kernel_size, stride, padding, &im2col_scratch);
                        return finalize_output(output_tensor);
                    }
                }
            }
            else
            {
                float gamma = gamma_tensor.isEmpty() ? 1.0f : gamma_tensor.getData()[0];
                const uint32_t k_u32 = checkedU32(K, "Conv2d_Layer::forward(cpu fallback K)");
                const uint32_t n_u32 = checkedU32(N, "Conv2d_Layer::forward(cpu fallback N)");
                const uint32_t words_per_row = ceilDivU32(k_u32, 16u);
                const auto &cpu_w = Cpu_Tensor_Impl::castToCpu(*packed_weights.getImplementation());
                const auto &packed_data = cpu_w.getPackedData();
                std::vector<float> unpacked(K * N, 0.0f);
                for (uint32_t n = 0; n < n_u32; ++n)
                {
                    for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
                    {
                        uint32_t word = packed_data[n * words_per_row + w_idx];
                        for (uint32_t b = 0; b < 16; ++b)
                        {
                            uint32_t k = w_idx * 16 + b;
                            if (k < k_u32)
                            {
                                uint32_t code = (word >> (b * 2u)) & 3u;
                                float val = (code == 1u) ? 1.0f : ((code == 3u) ? -1.0f : 0.0f);
                                unpacked[k * N + n] = val * gamma;
                            }
                        }
                    }
                }
                Tensor eff_w(1, K * N, std::move(unpacked), Execution_Target::CPU);
                ops::vision::conv2d(input_tensor, eff_w, biases, output_tensor,
                                    input_height, input_width, input_channels, output_channels,
                                    kernel_size, stride, padding);
                return finalize_output(output_tensor);
            }
        }
        else
        {
            if (!has_master_weights)
            {
                throw std::logic_error("Conv2d_Layer::forward: Missing master weights in ternary training mode");
            }
            if (execution_target == Execution_Target::VULKAN_GPU && Execution_Engine::getInstance().getContext().isFloat16Enabled())
            {
                ensureTernaryGpuBuffers(weight_count);
                auto &engine = Execution_Engine::getInstance();
                const uint32_t total_elements = checkedU32(weight_count, "Conv2d_Layer::forward(fakequant)");
                const uint32_t num_workgroups = std::min(ceilDivU32(total_elements, 256u), 1024u);

                const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*master_weights.getImplementation());
                auto contig_w = w_gpu.ensureContiguousSelf();
                const auto *eff_w = contig_w ? contig_w.get() : &w_gpu;
                auto w_buf = eff_w->getStorageBuffer();

                if (total_elements > 0)
                {
                    {
                        Compute_Node node;
                        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS1;
                        node.buffers = { w_buf, gpu_partial_sums };
                        struct PushConsts { uint32_t total; } pcs{ total_elements };
                        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
                        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
                        node.workgroup_count_x = num_workgroups;
                        node.is_barrier_required_after = true;
                        engine.getCurrentGraph().addNode(std::move(node));
                    }

                    {
                        Compute_Node node;
                        node.pipeline_id = Compute_Pipeline::REDUCE_ABSMEAN_PASS2;
                        node.buffers = { gpu_partial_sums, gpu_gamma_buffer };
                        struct PushConsts { uint32_t num_partials; uint32_t total; } pcs{ num_workgroups, total_elements };
                        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
                        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
                        node.workgroup_count_x = 1;
                        node.is_barrier_required_after = true;
                        engine.getCurrentGraph().addNode(std::move(node));
                    }

                    if (effective_weights_fp16.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
                        effective_weights_fp16.getDataType() != Data_Type::FLOAT16 ||
                        effective_weights_fp16.getShape() != Shape{ 1, weight_count })
                    {
                        effective_weights_fp16 = Tensor(Shape{ 1, weight_count }, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
                    }
                    const auto &weff_gpu = Gpu_Tensor_Impl::castToGpu(*effective_weights_fp16.getImplementation());
                    auto weff_buf = weff_gpu.getStorageBuffer();

                    {
                        Compute_Node node;
                        node.pipeline_id = Compute_Pipeline::FAKEQUANT_TERNARY;
                        node.buffers = { w_buf, gpu_gamma_buffer, weff_buf };
                        struct PushConsts { uint32_t total; } pcs{ total_elements };
                        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
                        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
                        node.workgroup_count_x = ceilDivU32(total_elements, 256u);
                        node.is_barrier_required_after = true;
                        engine.getCurrentGraph().addNode(std::move(node));
                    }
                }

                if (input_tensor.getDataType() != Data_Type::FLOAT16)
                {
                    input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                }
                else
                {
                    input_tensor_fp16 = input_tensor;
                }

                ops::vision::conv2d(input_tensor_fp16, effective_weights_fp16, biases, output_tensor_fp16,
                                    input_height, input_width, input_channels, output_channels,
                                    kernel_size, stride, padding, &im2col_scratch_fp16, &has_valid_forward_im2col_fp16);
                return finalize_output(output_tensor_fp16);
            }
            else
            {
                float gamma = 1e-5f;
                if (weight_count > 0)
                {
                    const auto &w_data = master_weights.getData();
                    double sum_abs = 0.0;
                    for (size_t i = 0; i < weight_count; ++i)
                    {
                        sum_abs += std::abs(static_cast<double>(w_data[i]));
                    }
                    gamma = std::max(1e-5f, static_cast<float>(sum_abs / static_cast<double>(weight_count)));
                    float inv_gamma = 1.0f / gamma;

                    std::vector<float> eff_data(weight_count);
                    for (size_t i = 0; i < weight_count; ++i)
                    {
                        float w = w_data[i];
                        float q = std::clamp(std::floor(std::abs(w) * inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
                        eff_data[i] = q * gamma;
                    }
                    effective_weights = Tensor(1, weight_count, std::move(eff_data), execution_target);
                    if (execution_target == Execution_Target::VULKAN_GPU)
                    {
                        Execution_Engine::getInstance().getContext().executePendingTransfers();
                    }
                }

                ops::vision::conv2d(input_tensor, effective_weights, biases, output_tensor,
                                    input_height, input_width, input_channels, output_channels,
                                    kernel_size, stride, padding, &im2col_scratch);
                return finalize_output(output_tensor);
            }
        }
    }

    if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU && Execution_Engine::getInstance().getContext().isFloat16Enabled())
    {
        if (input_tensor.getDataType() != Data_Type::FLOAT16)
        {
            input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
        }
        else
        {
            input_tensor_fp16 = input_tensor;
        }

        ops::vision::conv2d(input_tensor_fp16, weights, biases, output_tensor_fp16, input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16, &has_valid_forward_im2col_fp16);
        return finalize_output(output_tensor_fp16);
    }
    else
    {
        ops::vision::conv2d(input_tensor, weights, biases, output_tensor, input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, &im2col_scratch);
        return finalize_output(output_tensor);
    }
}

Tensor Conv2d_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
    (void)_batched_input;
    (void)_batched_params;
    throw std::logic_error("Conv2d_Layer does not support population-batched forward");
}

Tensor Conv2d_Layer::backward(const Tensor &_output_gradient)
{
    return backward(_output_gradient, true);
}

Tensor Conv2d_Layer::backward(const Tensor &_output_gradient, bool _compute_input_gradient)
{
    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Conv2d_Layer::backward: Backward called before forward"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("Backward called before forward");
    }
    if (!_output_gradient.getImplementation() || !input_tensor.getImplementation())
    {
        throw std::invalid_argument("Conv2d_Layer::backward: Uninitialized tensor state");
    }
    if (_output_gradient.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Conv2d_Layer::backward: Execution_Target mismatch");
    }
    if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Conv2d_Layer::backward: TERNARY_INT2 gradient tensor is not supported");
    }

    const size_t expected_batch = input_tensor.getShape()[0];
    const size_t expected_out_cols = static_cast<size_t>(output_height) * output_width * output_channels;
    const Shape &grad_shape = _output_gradient.getShape();
    if (grad_shape.getRank() == 2)
    {
        if (grad_shape[0] != expected_batch || grad_shape[1] != expected_out_cols)
        {
            throw std::invalid_argument("Conv2d_Layer::backward: Rank-2 gradient shape does not match output dimensions");
        }
    }
    else if (grad_shape.getRank() == 4)
    {
        if (grad_shape[0] != expected_batch || grad_shape[1] != output_height ||
            grad_shape[2] != output_width || grad_shape[3] != output_channels)
        {
            throw std::invalid_argument("Conv2d_Layer::backward: Rank-4 NHWC gradient shape does not match output dimensions");
        }
    }
    else
    {
        throw std::invalid_argument("Conv2d_Layer::backward: Expected rank-2 or rank-4 output gradient");
    }

    auto finalize_grad = [&]() -> Tensor {
        if (input_gradient_tensor.getShape() != input_tensor.getShape())
        {
            input_gradient_tensor.reshape(input_tensor.getShape());
        }
        return input_gradient_tensor;
    };

    auto accumulate_into = [&](Tensor &acc, const Tensor &step) {
        if (!acc.getImplementation() ||
            acc.getExecutionTarget() != execution_target ||
            acc.getDataType() != Data_Type::FLOAT32 ||
            acc.getShape() != step.getShape())
        {
            acc = Tensor(step.getShape(), Data_Type::FLOAT32, execution_target);
            acc.zero();
        }
        acc.add(step, acc);
    };

    if (isTernary())
    {
        if (!has_master_weights)
        {
            Logger::logMessage(Input_Format{"Conv2d_Layer::backward: Layer loaded in inference-only mode without master weights"},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Conv2d_Layer::backward: Layer loaded in inference-only mode without master weights");
        }
        bool eff_empty = (execution_target == Execution_Target::VULKAN_GPU && Execution_Engine::getInstance().getContext().isFloat16Enabled())
            ? effective_weights_fp16.isEmpty()
            : effective_weights.isEmpty();
        if (has_packed_inference || eff_empty)
        {
            Logger::logMessage("Conv2d_Layer::backward: Layer is in packed inference mode or effective weights are empty; call setTrainingMode(true) and forward() before backward()",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::CONV2D_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
            throw std::logic_error("Conv2d_Layer::backward: Layer is in packed inference mode; call setTrainingMode(true) and forward() before backward()");
        }

        if (execution_target == Execution_Target::VULKAN_GPU && Execution_Engine::getInstance().getContext().isFloat16Enabled())
        {
            if (_output_gradient.getDataType() != Data_Type::FLOAT16)
            {
                _output_gradient.to(Data_Type::FLOAT16, output_gradient_tensor_fp16);
            }
            else
            {
                output_gradient_tensor_fp16 = _output_gradient;
            }

            if (!is_accumulated)
            {
                ops::vision::conv2dBackwardWeight(input_tensor_fp16, output_gradient_tensor_fp16, master_weights_gradient_tensor, biases_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16, has_valid_forward_im2col_fp16);
            }
            else
            {
                ops::vision::conv2dBackwardWeight(input_tensor_fp16, output_gradient_tensor_fp16, step_weights_grad_scratch, step_biases_grad_scratch, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16, has_valid_forward_im2col_fp16);
                accumulate_into(master_weights_gradient_tensor, step_weights_grad_scratch);
                accumulate_into(biases_gradient_tensor, step_biases_grad_scratch);
            }

            if (_compute_input_gradient)
            {
                ops::vision::conv2dBackwardInput(output_gradient_tensor_fp16, effective_weights_fp16, input_gradient_tensor_fp16, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);

                if (input_tensor.getDataType() == Data_Type::FLOAT16)
                {
                    input_gradient_tensor = input_gradient_tensor_fp16;
                }
                else
                {
                    input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
                }
                return finalize_grad();
            }
            return input_gradient_tensor;
        }
        else
        {
            if (!is_accumulated)
            {
                ops::vision::conv2dBackwardWeight(input_tensor, _output_gradient, master_weights_gradient_tensor, biases_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch);
            }
            else
            {
                ops::vision::conv2dBackwardWeight(input_tensor, _output_gradient, step_weights_grad_scratch, step_biases_grad_scratch, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch);
                accumulate_into(master_weights_gradient_tensor, step_weights_grad_scratch);
                accumulate_into(biases_gradient_tensor, step_biases_grad_scratch);
            }

            if (_compute_input_gradient)
            {
                ops::vision::conv2dBackwardInput(_output_gradient, effective_weights, input_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
                return finalize_grad();
            }
            return input_gradient_tensor;
        }
    }

    if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU && Execution_Engine::getInstance().getContext().isFloat16Enabled())
    {
        if (_output_gradient.getDataType() != Data_Type::FLOAT16)
        {
            _output_gradient.to(Data_Type::FLOAT16, output_gradient_tensor_fp16);
        }
        else
        {
            output_gradient_tensor_fp16 = _output_gradient;
        }

        if (!is_accumulated)
        {
            ops::vision::conv2dBackwardWeight(input_tensor_fp16, output_gradient_tensor_fp16, weights_gradient_tensor, biases_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16, has_valid_forward_im2col_fp16);
        }
        else
        {
            ops::vision::conv2dBackwardWeight(input_tensor_fp16, output_gradient_tensor_fp16, step_weights_grad_scratch, step_biases_grad_scratch, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch_fp16, has_valid_forward_im2col_fp16);
            accumulate_into(weights_gradient_tensor, step_weights_grad_scratch);
            accumulate_into(biases_gradient_tensor, step_biases_grad_scratch);
        }
        if (_compute_input_gradient)
        {
            ops::vision::conv2dBackwardInput(output_gradient_tensor_fp16, weights, input_gradient_tensor_fp16, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);

            if (input_tensor.getDataType() == Data_Type::FLOAT16)
            {
                input_gradient_tensor = input_gradient_tensor_fp16;
            }
            else
            {
                input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
            }
            return finalize_grad();
        }
        return input_gradient_tensor;
    }
    else
    {
        if (!is_accumulated)
        {
            ops::vision::conv2dBackwardWeight(input_tensor, _output_gradient, weights_gradient_tensor, biases_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch);
        }
        else
        {
            ops::vision::conv2dBackwardWeight(input_tensor, _output_gradient, step_weights_grad_scratch, step_biases_grad_scratch, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding, &im2col_scratch);
            accumulate_into(weights_gradient_tensor, step_weights_grad_scratch);
            accumulate_into(biases_gradient_tensor, step_biases_grad_scratch);
        }
        if (_compute_input_gradient)
        {
            ops::vision::conv2dBackwardInput(_output_gradient, weights, input_gradient_tensor, input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size, stride, padding);
            return finalize_grad();
        }
        return input_gradient_tensor;
    }
}

void Conv2d_Layer::resetGradient()
{
    is_forward_completed = false;
}

void Conv2d_Layer::resetGradients()
{
    resetGradient();
    if (isTernary())
    {
        master_weights_gradient_tensor.zero();
    }
    else
    {
        weights_gradient_tensor.zero();
    }
    biases_gradient_tensor.zero();
}

std::unique_ptr<ILayer> Conv2d_Layer::clone() const
{
    auto cloned = std::make_unique<Conv2d_Layer>(
        input_height, input_width, input_channels, output_channels, kernel_size, stride, padding, execution_target, data_type);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    cloned->setAccumulated(is_accumulated);
    cloned->biases = biases.clone();
    if (isTernary())
    {
        cloned->has_packed_inference = has_packed_inference;
        cloned->has_master_weights = has_master_weights;
        if (has_master_weights)
        {
            cloned->master_weights = master_weights.clone();
        }
        else
        {
            cloned->master_weights = Tensor(0, 0, execution_target);
        }

        if (has_packed_inference)
        {
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().executeGraph();
                Execution_Engine::getInstance().waitIdle();
            }
            cloned->packed_weights = packed_weights.clone();
            cloned->gamma_tensor = gamma_tensor.clone();
            cloned->effective_weights = Tensor(0, 0, execution_target);
            cloned->effective_weights_fp16 = Tensor(0, 0, execution_target);
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                std::vector<float> g(2, 1.0f);
                if (gpu_gamma_buffer && gpu_gamma_buffer->getBufferSizeInBytes() >= sizeof(float) * 2)
                {
                    gpu_gamma_buffer->downloadData(g);
                }
                else if (!gamma_tensor.isEmpty())
                {
                    const auto &gd = gamma_tensor.getData();
                    g[0] = gd[0];
                    g[1] = gd.size() > 1 ? gd[1] : (1.0f / gd[0]);
                }
                auto &context = Execution_Engine::getInstance().getContext();
                if (!cloned->gpu_gamma_buffer)
                {
                    cloned->gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
                    cloned->gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
                }
                cloned->gpu_gamma_buffer->uploadData(g);
                context.executePendingTransfers();
            }
        }
        else
        {
            cloned->effective_weights = effective_weights.clone();
            cloned->effective_weights_fp16 = effective_weights_fp16.clone();
            cloned->gamma_tensor = gamma_tensor.clone();
        }
    }
    else
    {
        cloned->weights = weights.clone();
    }
    cloned->invalidateWeightCache();
    return cloned;
}

void Conv2d_Layer::invalidateWeightCache() noexcept
{
    weights.invalidateFp16Cache();
    biases.invalidateFp16Cache();
    if (isTernary())
    {
        if (has_master_weights)
        {
            master_weights.invalidateFp16Cache();
        }
        effective_weights.invalidateFp16Cache();
        effective_weights_fp16.invalidateFp16Cache();
    }
}

void Conv2d_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    has_valid_forward_im2col_fp16 = false;
    invalidateWeightCache();
}

void Conv2d_Layer::setTrainingMode(bool _is_training)
{
    has_valid_forward_im2col_fp16 = false;
    if (isTernary() && _is_training)
    {
        if (!has_master_weights)
        {
            Logger::logMessage("Conv2d_Layer::setTrainingMode: Layer loaded in inference-only mode without master weights",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::CONV2D_COMPUTE);
            throw std::runtime_error("Conv2d_Layer: Layer loaded in inference-only mode without master weights");
        }
        has_packed_inference = false;
        invalidateWeightCache();
    }
}

void Conv2d_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Conv2d_Layer::saveConfiguration: Output stream is not open or writable");
    }
    _output_file_stream.write(reinterpret_cast<const char *>(&input_height), sizeof(input_height));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_width), sizeof(input_width));
    _output_file_stream.write(reinterpret_cast<const char *>(&input_channels), sizeof(input_channels));
    _output_file_stream.write(reinterpret_cast<const char *>(&output_channels), sizeof(output_channels));
    _output_file_stream.write(reinterpret_cast<const char *>(&kernel_size), sizeof(kernel_size));
    _output_file_stream.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
    _output_file_stream.write(reinterpret_cast<const char *>(&padding), sizeof(padding));
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Conv2d_Layer::saveConfiguration: Failed writing configuration");
    }
}

void Conv2d_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Conv2d_Layer::saveInference: Output stream is not open or writable");
    }
    if (isTernary())
    {
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
            Execution_Engine::getInstance().waitIdle();
        }

        _output_file_stream.write(reinterpret_cast<const char*>(&TERNARY_INFERENCE_MAGIC), sizeof(TERNARY_INFERENCE_MAGIC));

        float gamma_val = 1.0f;
        float inv_gamma_val = 1.0f;
        std::vector<uint32_t> local_packed;
        const std::vector<uint32_t> *packed_ptr = &local_packed;

        if (has_packed_inference)
        {
            if (execution_target == Execution_Target::VULKAN_GPU && gpu_gamma_buffer)
            {
                std::vector<float> downloaded_gamma(2, 0.0f);
                gpu_gamma_buffer->downloadData(downloaded_gamma);
                gamma_val = downloaded_gamma[0];
                inv_gamma_val = downloaded_gamma[1];
            }
            else if (!gamma_tensor.isEmpty())
            {
                const auto &gdata = gamma_tensor.getData();
                gamma_val = gdata[0];
                inv_gamma_val = gdata.size() > 1 ? gdata[1] : (1.0f / gamma_val);
            }
            packed_ptr = &packed_weights.getPackedData();
        }
        else
        {
            if (!has_master_weights)
            {
                throw std::logic_error("Conv2d_Layer::saveInference: Missing both packed weights and master weights");
            }
            const size_t K = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
            const size_t N = output_channels;
            const size_t total = K * N;
            const uint32_t k_u32 = checkedU32(K, "Conv2d_Layer::saveInference(K)");
            const uint32_t n_u32 = checkedU32(N, "Conv2d_Layer::saveInference(N)");
            const uint32_t words_per_row = ceilDivU32(k_u32, 16u);
            local_packed.assign(static_cast<size_t>(N) * words_per_row, 0u);
            if (total > 0)
            {
                const auto &w_data = master_weights.getData();
                double sum_abs = 0.0;
                for (size_t i = 0; i < total; ++i)
                {
                    sum_abs += std::abs(static_cast<double>(w_data[i]));
                }
                gamma_val = std::max(1e-5f, static_cast<float>(sum_abs / static_cast<double>(total)));
                inv_gamma_val = 1.0f / gamma_val;
                for (uint32_t n = 0; n < n_u32; ++n)
                {
                    for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx)
                    {
                        uint32_t word = 0u;
                        for (uint32_t b = 0; b < 16; ++b)
                        {
                            uint32_t k = w_idx * 16 + b;
                            if (k < k_u32)
                            {
                                float w = w_data[k * N + n];
                                float q = std::clamp(std::floor(std::abs(w) * inv_gamma_val + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
                                uint32_t code = static_cast<uint32_t>(static_cast<int>(q)) & 3u;
                                word |= (code << (b * 2u));
                            }
                        }
                        local_packed[n * words_per_row + w_idx] = word;
                    }
                }
            }
        }

        _output_file_stream.write(reinterpret_cast<const char*>(&gamma_val), sizeof(gamma_val));
        _output_file_stream.write(reinterpret_cast<const char*>(&inv_gamma_val), sizeof(inv_gamma_val));

        const uint32_t words_count = checkedU32(packed_ptr->size(), "Conv2d_Layer::saveInference(words_count)");
        _output_file_stream.write(reinterpret_cast<const char*>(&words_count), sizeof(words_count));
        if (words_count > 0)
        {
            _output_file_stream.write(reinterpret_cast<const char*>(packed_ptr->data()), static_cast<std::streamsize>(static_cast<size_t>(words_count) * sizeof(uint32_t)));
        }

        biases.saveTensor(_output_file_stream);
    }
    else
    {
        weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
    }
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Conv2d_Layer::saveInference: Failed writing inference state");
    }
}

void Conv2d_Layer::loadInference(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open())
    {
        throw std::runtime_error("Conv2d_Layer::loadInference: Input stream is not open");
    }
    auto pos = _input_file_stream.tellg();
    uint32_t magic = 0;
    _input_file_stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!_input_file_stream)
    {
        throw std::runtime_error("Conv2d_Layer::loadInference: Failed reading inference header");
    }

    if (magic == TERNARY_INFERENCE_MAGIC)
    {
        float gamma_val = 0.0f;
        float inv_gamma_val = 0.0f;
        _input_file_stream.read(reinterpret_cast<char*>(&gamma_val), sizeof(gamma_val));
        _input_file_stream.read(reinterpret_cast<char*>(&inv_gamma_val), sizeof(inv_gamma_val));

        uint32_t words_count = 0;
        _input_file_stream.read(reinterpret_cast<char*>(&words_count), sizeof(words_count));
        if (!_input_file_stream)
        {
            throw std::runtime_error("Conv2d_Layer::loadInference: Failed reading ternary header metadata");
        }
        if (!std::isfinite(gamma_val) || gamma_val <= 0.0f || !std::isfinite(inv_gamma_val) || inv_gamma_val <= 0.0f)
        {
            throw std::runtime_error("Conv2d_Layer::loadInference: Invalid gamma values in ternary inference file");
        }

        const size_t K = static_cast<size_t>(kernel_size) * kernel_size * input_channels;
        const uint64_t words_per_row = (static_cast<uint64_t>(K) + 15ULL) / 16ULL;
        if ((words_per_row > 0 && (static_cast<uint64_t>(words_count) % words_per_row) != 0ULL) ||
            static_cast<uint64_t>(words_count) > (1ULL << 29))
        {
            throw std::runtime_error("Conv2d_Layer::loadInference: Invalid ternary packed word count");
        }

        std::vector<uint32_t> packed_data(words_count);
        if (words_count > 0)
        {
            _input_file_stream.read(reinterpret_cast<char*>(packed_data.data()), static_cast<std::streamsize>(static_cast<size_t>(words_count) * sizeof(uint32_t)));
            if (!_input_file_stream)
            {
                throw std::runtime_error("Conv2d_Layer::loadInference: Failed reading ternary packed payload");
            }
        }

        Tensor loaded_biases = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_biases.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::runtime_error("Conv2d_Layer::loadInference: Invalid bias dtype");
        }
        const size_t N = loaded_biases.getColumns();
        if (loaded_biases.getTotalElements() != N ||
            static_cast<uint64_t>(words_count) != static_cast<uint64_t>(N) * words_per_row)
        {
            throw std::runtime_error("Conv2d_Layer::loadInference: Bias or packed word count mismatch");
        }

        data_type = Data_Type::TERNARY_INT2;
        has_packed_inference = true;
        has_master_weights = false;
        is_forward_completed = false;
        output_channels = static_cast<uint32_t>(N);
        biases = std::move(loaded_biases);

        packed_weights = Tensor(Shape{ N, K }, Data_Type::TERNARY_INT2, execution_target);
        if (words_count > 0)
        {
            packed_weights.uploadPackedData(packed_data);
        }

        std::vector<float> gv = { gamma_val, inv_gamma_val };
        gamma_tensor = Tensor(Shape{ 2 }, gv, execution_target);

        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            auto &context = Execution_Engine::getInstance().getContext();
            gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
            gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
            gpu_gamma_buffer->uploadData(std::vector<float>{ gamma_val, inv_gamma_val });
            context.executePendingTransfers();
        }
        else
        {
            gpu_gamma_buffer.reset();
        }
        gpu_partial_sums.reset();
        master_weights = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        effective_weights_fp16 = Tensor(0, 0, execution_target);
        weights = Tensor(0, 0, execution_target);
        invalidateWeightCache();
    }
    else
    {
        _input_file_stream.clear();
        _input_file_stream.seekg(pos);
        Tensor loaded_w = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        const size_t expected_weight_count = static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels;
        if (loaded_w.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_b.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_w.getTotalElements() != expected_weight_count ||
            loaded_b.getTotalElements() != output_channels)
        {
            throw std::runtime_error("Conv2d_Layer::loadInference: Loaded tensor dimensions or dtype mismatch");
        }
        weights = std::move(loaded_w);
        biases = std::move(loaded_b);
        data_type = (weights.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        if (data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        has_packed_inference = false;
        has_master_weights = false;
        is_forward_completed = false;
        master_weights = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        effective_weights_fp16 = Tensor(0, 0, execution_target);
        packed_weights = Tensor(0, 0, execution_target);
        gamma_tensor = Tensor(0, 0, execution_target);
        gpu_gamma_buffer.reset();
        gpu_partial_sums.reset();
        invalidateWeightCache();
    }
}

void Conv2d_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Conv2d_Layer::saveCheckpoint: Output stream is not open or writable");
    }
    if (isTernary())
    {
        if (!has_master_weights)
        {
            throw std::logic_error("Conv2d_Layer::saveCheckpoint: Cannot save checkpoint from inference-only ternary layer without master weights");
        }
        const uint32_t magic = TERNARY_CHECKPOINT_MAGIC;
        _output_file_stream.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        master_weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
        master_weights_gradient_tensor.saveTensor(_output_file_stream);
        biases_gradient_tensor.saveTensor(_output_file_stream);
    }
    else
    {
        weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
        weights_gradient_tensor.saveTensor(_output_file_stream);
        biases_gradient_tensor.saveTensor(_output_file_stream);
    }
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Conv2d_Layer::saveCheckpoint: Failed writing checkpoint state");
    }
}

void Conv2d_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open())
    {
        throw std::runtime_error("Conv2d_Layer::loadCheckpoint: Input stream is not open");
    }
    auto pos = _input_file_stream.tellg();
    uint32_t magic = 0;
    _input_file_stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!_input_file_stream)
    {
        throw std::runtime_error("Conv2d_Layer::loadCheckpoint: Failed reading checkpoint header");
    }

    const size_t expected_weight_count = static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels;
    if (magic == TERNARY_CHECKPOINT_MAGIC)
    {
        Tensor loaded_mw = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_mwg = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_bg = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_mw.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_mw.getTotalElements() != expected_weight_count ||
            loaded_b.getTotalElements() != output_channels ||
            loaded_mwg.getTotalElements() != expected_weight_count ||
            loaded_bg.getTotalElements() != output_channels)
        {
            throw std::runtime_error("Conv2d_Layer::loadCheckpoint: Corrupted ternary checkpoint tensor dimensions");
        }
        data_type = Data_Type::TERNARY_INT2;
        has_master_weights = true;
        has_packed_inference = false;
        is_forward_completed = false;
        master_weights = std::move(loaded_mw);
        biases = std::move(loaded_b);
        master_weights_gradient_tensor = std::move(loaded_mwg);
        biases_gradient_tensor = std::move(loaded_bg);
        weights = Tensor(0, 0, execution_target);
        weights_gradient_tensor = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        effective_weights_fp16 = Tensor(0, 0, execution_target);
        packed_weights = Tensor(0, 0, execution_target);
        gamma_tensor = Tensor(0, 0, execution_target);
        ensureTernaryGpuBuffers(expected_weight_count);
        invalidateWeightCache();
    }
    else
    {
        _input_file_stream.clear();
        _input_file_stream.seekg(pos);
        Tensor loaded_w = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_wg = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_bg = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_w.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_w.getTotalElements() != expected_weight_count ||
            loaded_b.getTotalElements() != output_channels ||
            loaded_wg.getTotalElements() != expected_weight_count ||
            loaded_bg.getTotalElements() != output_channels)
        {
            throw std::runtime_error("Conv2d_Layer::loadCheckpoint: Corrupted checkpoint tensor dimensions");
        }
        weights = std::move(loaded_w);
        biases = std::move(loaded_b);
        weights_gradient_tensor = std::move(loaded_wg);
        biases_gradient_tensor = std::move(loaded_bg);
        data_type = (weights.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        if (data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        has_master_weights = false;
        has_packed_inference = false;
        is_forward_completed = false;
        master_weights = Tensor(0, 0, execution_target);
        master_weights_gradient_tensor = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        effective_weights_fp16 = Tensor(0, 0, execution_target);
        packed_weights = Tensor(0, 0, execution_target);
        gamma_tensor = Tensor(0, 0, execution_target);
        gpu_gamma_buffer.reset();
        gpu_partial_sums.reset();
        invalidateWeightCache();
    }
}

std::function<float(std::mt19937&)> Conv2d_Layer::getPopulationParameterInitializer(size_t param_index) const
{
    if (param_index == 0)
    {
        float fan_in = static_cast<float>(static_cast<size_t>(kernel_size) * kernel_size * input_channels);
        float standard_deviation = (fan_in > 0.0f) ? std::sqrt(2.0f / fan_in) : 0.0f;
        return [standard_deviation](std::mt19937& generator)
            {
                std::normal_distribution<float> distribution(0.0f, standard_deviation);
                return distribution(generator);
            };
    }
    else if (param_index == 1)
    {
        return [](std::mt19937&)
            {
                return 0.0f;
            };
    }

    return [](std::mt19937&)
        {
            return 0.0f;
        };
}

std::vector<float> Conv2d_Layer::getPopulationParameter(size_t param_index) const
{
    if (param_index == 0)
    {
        if (isTernary())
        {
            if (!has_master_weights)
            {
                throw std::logic_error("Conv2d_Layer::getPopulationParameter: Master weights unavailable in inference-only mode");
            }
            return master_weights.getData();
        }
        return weights.getData();
    }
    if (param_index == 1)
    {
        return biases.getData();
    }
    throw std::out_of_range("Conv2d_Layer::getPopulationParameter: Parameter index out of range");
}

std::vector<std::pair<Tensor *, Tensor *>> Conv2d_Layer::getParametersAndGradients()
{
    if (isTernary())
    {
        if (!has_master_weights)
        {
            return {};
        }
        return { { &master_weights, &master_weights_gradient_tensor }, { &biases, &biases_gradient_tensor } };
    }
    return { { &weights, &weights_gradient_tensor }, { &biases, &biases_gradient_tensor } };
}

void Conv2d_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
    size_t weight_count = static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels;
    if (param_index == 0)
    {
        if (flat_data.size() != weight_count)
        {
            throw std::invalid_argument("Conv2d_Layer::setPopulationParameter: Weight size mismatch");
        }
        if (isTernary())
        {
            master_weights = Tensor(1, weight_count, std::move(flat_data), execution_target);
            has_master_weights = true;
            has_packed_inference = false;
            ensureTernaryGpuBuffers(weight_count);
        }
        else
        {
            weights = Tensor(1, weight_count, std::move(flat_data), execution_target);
        }
        invalidateWeightCache();
    }
    else if (param_index == 1)
    {
        if (flat_data.size() != output_channels)
        {
            throw std::invalid_argument("Conv2d_Layer::setPopulationParameter: Bias size mismatch");
        }
        biases = Tensor(1, output_channels, std::move(flat_data), execution_target);
        invalidateWeightCache();
    }
    else
    {
        throw std::out_of_range("Conv2d_Layer::setPopulationParameter: Parameter index out of range");
    }
}

void Conv2d_Layer::setWeightsGradient(const Tensor &_tensor)
{
    if (isTernary())
    {
        master_weights_gradient_tensor = _tensor;
    }
    else
    {
        weights_gradient_tensor = _tensor;
    }
}

void Conv2d_Layer::setBiasesGradient(const Tensor &_tensor)
{
    biases_gradient_tensor = _tensor;
}

void Conv2d_Layer::setWeights(const Tensor &_new_weights)
{
    if (!_new_weights.getImplementation())
    {
        throw std::invalid_argument("Conv2d_Layer::setWeights: Tensor is uninitialized");
    }
    if (_new_weights.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Conv2d_Layer::setWeights: Execution_Target mismatch");
    }
    if (_new_weights.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Conv2d_Layer::setWeights: TERNARY_INT2 weights cannot be assigned directly");
    }
    const size_t expected_weight_count = static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels;
    if (_new_weights.getTotalElements() != expected_weight_count)
    {
        throw std::invalid_argument("Conv2d_Layer::setWeights: Weight element count mismatch");
    }
    if (isTernary())
    {
        master_weights = _new_weights;
        has_master_weights = true;
        has_packed_inference = false;
        ensureTernaryGpuBuffers(expected_weight_count);
    }
    else
    {
        weights = _new_weights;
    }
    invalidateWeightCache();
}

void Conv2d_Layer::setBiases(const Tensor &_new_biases)
{
    if (!_new_biases.getImplementation())
    {
        throw std::invalid_argument("Conv2d_Layer::setBiases: Tensor is uninitialized");
    }
    if (_new_biases.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Conv2d_Layer::setBiases: Execution_Target mismatch");
    }
    if (_new_biases.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Conv2d_Layer::setBiases: TERNARY_INT2 biases are not supported");
    }
    if (_new_biases.getTotalElements() != output_channels)
    {
        throw std::invalid_argument("Conv2d_Layer::setBiases: Bias element count mismatch");
    }
    biases = _new_biases;
    invalidateWeightCache();
}

void Conv2d_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target)
    {
        return;
    }

    logChangeExecutionTarget(_new_execution_target);

    if (execution_target == Execution_Target::VULKAN_GPU && isTernary())
    {
        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();
        if (gpu_gamma_buffer && gpu_gamma_buffer->getBufferSizeInBytes() >= sizeof(float) * 2)
        {
            std::vector<float> g(2, 1.0f);
            gpu_gamma_buffer->downloadData(g);
            gamma_tensor = Tensor(Shape{ 2 }, g, Execution_Target::CPU);
        }
    }

    std::vector<float> saved_gamma = { 1.0f, 1.0f };
    if (isTernary() && !gamma_tensor.isEmpty())
    {
        const auto &gd = gamma_tensor.getData();
        if (!gd.empty())
        {
            saved_gamma[0] = gd[0];
            saved_gamma[1] = (gd.size() > 1) ? gd[1] : (1.0f / gd[0]);
        }
    }

    execution_target = _new_execution_target;
    weights.setExecutionTarget(_new_execution_target);
    biases.setExecutionTarget(_new_execution_target);
    weights_gradient_tensor.setExecutionTarget(_new_execution_target);
    biases_gradient_tensor.setExecutionTarget(_new_execution_target);
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);
    input_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    output_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    input_gradient_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    output_gradient_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    im2col_scratch = Tensor(0, 0, _new_execution_target);
    im2col_scratch_fp16 = Tensor(0, 0, _new_execution_target);
    step_weights_grad_scratch = Tensor(0, 0, _new_execution_target);
    step_biases_grad_scratch = Tensor(0, 0, _new_execution_target);

    if (has_master_weights)
    {
        master_weights.setExecutionTarget(_new_execution_target);
    }
    master_weights_gradient_tensor.setExecutionTarget(_new_execution_target);
    effective_weights.setExecutionTarget(_new_execution_target);
    effective_weights_fp16 = Tensor(0, 0, _new_execution_target);
    packed_weights.setExecutionTarget(_new_execution_target);
    gamma_tensor.setExecutionTarget(_new_execution_target);

    if (isTernary())
    {
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            size_t weight_count = static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels;
            ensureTernaryGpuBuffers(weight_count);
            gpu_gamma_buffer->uploadData(saved_gamma);
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
        else
        {
            gpu_gamma_buffer = nullptr;
            gpu_partial_sums = nullptr;
        }
    }
    invalidateWeightCache();
}

void Conv2d_Layer::setInputHeight(uint32_t _height)
{
    auto [out_h, out_w] = validateAndComputeOutput(_height, input_width, input_channels, output_channels, kernel_size, stride, padding);
    input_height = _height;
    output_height = out_h;
    output_width = out_w;
    is_forward_completed = false;
}

void Conv2d_Layer::setInputWidth(uint32_t _width)
{
    auto [out_h, out_w] = validateAndComputeOutput(input_height, _width, input_channels, output_channels, kernel_size, stride, padding);
    input_width = _width;
    output_height = out_h;
    output_width = out_w;
    is_forward_completed = false;
}

void Conv2d_Layer::setPadding(uint32_t _padding)
{
    auto [out_h, out_w] = validateAndComputeOutput(input_height, input_width, input_channels, output_channels, kernel_size, stride, _padding);
    padding = _padding;
    output_height = out_h;
    output_width = out_w;
    is_forward_completed = false;
}

void Conv2d_Layer::setStride(uint32_t _stride)
{
    auto [out_h, out_w] = validateAndComputeOutput(input_height, input_width, input_channels, output_channels, kernel_size, _stride, padding);
    stride = _stride;
    output_height = out_h;
    output_width = out_w;
    is_forward_completed = false;
}

void Conv2d_Layer::setIsForwardCompleted(bool _is_completed) noexcept
{ is_forward_completed = _is_completed; }

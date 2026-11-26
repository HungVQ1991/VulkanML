#include "layer/linear_layer.h"

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
#include "engine/vulkan_context.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"
#include "ops/linear/linear.h"
#include "ops/llm/embedding.h"

Linear_Layer::~Linear_Layer() noexcept = default;

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
}

Linear_Layer::Linear_Layer()
    : weights(0, 0),
      biases(0, 0),
      input_tensor(0, 0),
      output_tensor(0, 0),
      input_gradient_tensor(0, 0),
      weights_gradient_tensor(0, 0),
      biases_gradient_tensor(0, 0),
      input_tensor_fp16(0, 0),
      output_gradient_fp16(0, 0),
      input_gradient_tensor_fp16(0, 0),
      data_type(Data_Type::FLOAT32),
      master_weights(0, 0),
      effective_weights(0, 0),
      packed_weights(Shape{ 0, 0 }, Data_Type::TERNARY_INT2),
      gamma_tensor(Shape{ 0 }, Data_Type::FLOAT32),
      has_packed_inference(false),
      has_master_weights(true),
      use_bias(true),
      tied_weight_t_scratch(0, 0),
      tied_grad_fp32_scratch(0, 0),
      unused_bias_grad_scratch(0, 0),
      indexed_hidden_scratch(0, 0),
      indexed_grad_scratch(0, 0),
      is_forward_completed(false),
      execution_target(Execution_Target::CPU)
{
}

Linear_Layer::Linear_Layer(size_t _input_dimension,
                           size_t _output_dimension,
                           Execution_Target _execution_target,
                           float _initialization_gain,
                           Data_Type _data_type,
                           bool _use_bias)
    : weights(0, 0, _execution_target),
      biases(0, 0, _execution_target),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      weights_gradient_tensor(_input_dimension, _output_dimension, _execution_target),
      biases_gradient_tensor(1, _output_dimension, _execution_target),
      input_tensor_fp16(0, 0, _execution_target),
      output_gradient_fp16(0, 0, _execution_target),
      input_gradient_tensor_fp16(0, 0, _execution_target),
      data_type(_data_type),
      master_weights(0, 0, _execution_target),
      effective_weights(0, 0, _execution_target),
      packed_weights(Shape{ 0, 0 }, Data_Type::TERNARY_INT2, _execution_target),
      gamma_tensor(Shape{ 0 }, Data_Type::FLOAT32, _execution_target),
      has_packed_inference(false),
      has_master_weights(true),
      use_bias(_use_bias),
      tied_weight_t_scratch(0, 0, _execution_target),
      tied_grad_fp32_scratch(0, 0, _execution_target),
      unused_bias_grad_scratch(0, 0, _execution_target),
      indexed_hidden_scratch(0, 0, _execution_target),
      indexed_grad_scratch(0, 0, _execution_target),
      input_dimension(_input_dimension),
      output_dimension(_output_dimension),
      initialization_gain(_initialization_gain),
      is_forward_completed(false),
      execution_target(_execution_target)
{
    if (_data_type == Data_Type::FLOAT16)
    {
        is_mixed_precision_enabled = true;
    }

    std::vector<float> weight_data(_input_dimension * _output_dimension);
    std::vector<float> bias_data(_output_dimension, 0.0f);

    float standard_deviation = (_input_dimension > 0) ? std::sqrt(_initialization_gain / static_cast<float>(_input_dimension)) : 1.0f;
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

    float weight_sum = 0.0f;
    for (float &weight_value : weight_data)
    {
        weight_value = normal_distribution(generator);
        weight_sum += std::abs(weight_value);
    }

    float mean_absolute_weight = weight_data.empty() ? 0.0f : weight_sum / static_cast<float>(weight_data.size());
    Logger::logMessage(Input_Format{"Linear_Layer::Linear_Layer: Layer weights mean_abs = {:.8f}", mean_absolute_weight},
                       Log_Level::LOG_DEBUG,
                       true,
                       1,
                       Log_Feature::DENSE_COMPUTE);

    biases = Tensor(1, _output_dimension, std::move(bias_data), execution_target);

    if (_data_type == Data_Type::TERNARY_INT2)
    {
        master_weights = Tensor(_input_dimension, _output_dimension, std::move(weight_data), execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        packed_weights = Tensor(Shape{ 0, 0 }, Data_Type::TERNARY_INT2, execution_target);
        gamma_tensor = Tensor(Shape{ 2 }, Data_Type::FLOAT32, execution_target);
        ensureTernaryGpuBuffers();
    }
    else
    {
        weights = Tensor(_input_dimension, _output_dimension, std::move(weight_data), execution_target);
    }

    weights_gradient_tensor.zero();
    biases_gradient_tensor.zero();
}

void Linear_Layer::ensureTernaryGpuBuffers()
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
    if (input_dimension != 0 && output_dimension > std::numeric_limits<size_t>::max() / input_dimension)
    {
        throw std::overflow_error("Linear_Layer: Weight element count overflows size_t");
    }
    const size_t total_sz = input_dimension * output_dimension;
    const uint32_t total_elements = checkedU32(total_sz, "Linear_Layer::ensureTernaryGpuBuffers");
    if (total_elements > 0)
    {
        const uint32_t num_workgroups = std::min(ceilDivU32(total_elements, 256u), 1024u);
        if (!gpu_partial_sums || gpu_partial_sums->getSize() < num_workgroups)
        {
            gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
        }
    }
}

void Linear_Layer::quantizeEffectiveWeightsGpu()
{
    ensureTernaryGpuBuffers();
    auto &engine = Execution_Engine::getInstance();
    auto &context = engine.getContext();
    const uint32_t total_elements = checkedU32(input_dimension * output_dimension, "Linear_Layer::quantizeEffectiveWeightsGpu");
    if (total_elements == 0)
    {
        return;
    }

    if (!context.isFloat16Enabled())
    {
        const auto &w_data = master_weights.getData();
        double sum_abs = 0.0;
        for (size_t i = 0; i < total_elements; ++i)
        {
            sum_abs += std::abs(static_cast<double>(w_data[i]));
        }
        float gamma = std::max(1e-5f, static_cast<float>(sum_abs / static_cast<double>(total_elements)));
        float inv_gamma = 1.0f / gamma;

        std::vector<float> eff_w(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            float w = w_data[i];
            float q = std::clamp(std::floor(std::abs(w) * inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
            eff_w[i] = q * gamma;
        }
        effective_weights = Tensor(Shape{ input_dimension, output_dimension }, eff_w, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
        context.executePendingTransfers();
        return;
    }

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

    if (effective_weights.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
        effective_weights.getDataType() != Data_Type::FLOAT16 ||
        effective_weights.getShape() != Shape{ input_dimension, output_dimension })
    {
        effective_weights = Tensor(Shape{ input_dimension, output_dimension }, Data_Type::FLOAT16, Execution_Target::VULKAN_GPU);
    }
    const auto &weff_gpu = Gpu_Tensor_Impl::castToGpu(*effective_weights.getImplementation());
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

void Linear_Layer::quantizeEffectiveWeightsCpu()
{
    size_t total = input_dimension * output_dimension;
    if (total == 0)
    {
        return;
    }

    const auto &w_data = master_weights.getData();
    double sum_abs = 0.0;
    for (size_t i = 0; i < total; ++i)
    {
        sum_abs += std::abs(static_cast<double>(w_data[i]));
    }
    float gamma = std::max(1e-5f, static_cast<float>(sum_abs / static_cast<double>(total)));
    float inv_gamma = 1.0f / gamma;

    std::vector<float> gamma_vec = { gamma, inv_gamma };
    gamma_tensor = Tensor(Shape{ 2 }, std::move(gamma_vec), Execution_Target::CPU);

    std::vector<float> eff_w(total);
    for (size_t i = 0; i < total; ++i)
    {
        float w = w_data[i];
        float q = std::clamp(std::floor(std::abs(w) * inv_gamma + 0.5f) * (w >= 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
        eff_w[i] = q * gamma;
    }
    effective_weights = Tensor(input_dimension, output_dimension, std::move(eff_w), Execution_Target::CPU);
}

void Linear_Layer::packWeightsGpu()
{
    ensureTernaryGpuBuffers();
    auto &engine = Execution_Engine::getInstance();
    const uint32_t K = checkedU32(input_dimension, "Linear_Layer::packWeightsGpu(K)");
    const uint32_t N = checkedU32(output_dimension, "Linear_Layer::packWeightsGpu(N)");
    const uint32_t total_elements = checkedU32(input_dimension * output_dimension, "Linear_Layer::packWeightsGpu(total)");
    if (total_elements == 0)
    {
        return;
    }

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
    const Shape packed_shape{ N, K };
    if (!packed_weights.getImplementation() ||
        packed_weights.getExecutionTarget() != Execution_Target::VULKAN_GPU ||
        packed_weights.getDataType() != Data_Type::TERNARY_INT2 ||
        packed_weights.getShape() != packed_shape)
    {
        packed_weights = Tensor(packed_shape, Data_Type::TERNARY_INT2, Execution_Target::VULKAN_GPU);
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

void Linear_Layer::packWeightsCpu()
{
    size_t K = input_dimension;
    size_t N = output_dimension;
    size_t total = K * N;
    if (total == 0)
    {
        return;
    }

    const auto &w_data = master_weights.getData();
    double sum_abs = 0.0;
    for (size_t i = 0; i < total; ++i)
    {
        sum_abs += std::abs(static_cast<double>(w_data[i]));
    }
    float gamma = std::max(1e-5f, static_cast<float>(sum_abs / static_cast<double>(total)));
    float inv_gamma = 1.0f / gamma;

    std::vector<float> gamma_vec = { gamma, inv_gamma };
    gamma_tensor = Tensor(Shape{ 2 }, std::move(gamma_vec), Execution_Target::CPU);

    const uint32_t k_u32 = checkedU32(K, "Linear_Layer::packWeightsCpu(K)");
    const uint32_t n_u32 = checkedU32(N, "Linear_Layer::packWeightsCpu(N)");
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

void Linear_Layer::prepareInference()
{
    if (isTernary())
    {
        if (has_packed_inference && !has_master_weights)
        {
            return;
        }
        if (!has_master_weights)
        {
            throw std::logic_error("Linear_Layer::prepareInference: Missing master weights");
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
        has_packed_inference = true;
        Logger::logMessage(Input_Format{"Linear_Layer::prepareInference: Packed weights to TERNARY_INT2 (K={}, N={})",
                                        input_dimension, output_dimension},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::TERNARY_METRICS | Log_Feature::LAYER_INSPECTION);
    }
}

void Linear_Layer::quantizeToTernary()
{
    if (isTernary())
    {
        return;
    }

    data_type = Data_Type::TERNARY_INT2;
    master_weights = weights;
    weights = Tensor(0, 0, execution_target);
    has_master_weights = true;
    has_packed_inference = false;
    ensureTernaryGpuBuffers();
    prepareInference();
    invalidateWeightCache();
    Logger::logMessage(Input_Format{"Linear_Layer::quantizeToTernary: Quantized layer to TERNARY_INT2 (K={}, N={})",
                                    input_dimension, output_dimension},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::TERNARY_METRICS | Log_Feature::LAYER_INSPECTION);
}

void Linear_Layer::dequantizeFromTernary()
{
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

    const size_t K = input_dimension;
    const size_t N = output_dimension;

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
        if (weights.getShape() != Shape{ K, N })
        {
            weights.reshape(Shape{ K, N });
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
        std::vector<float> w_kn(K * N, 0.0f);
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t k = 0; k < K; ++k)
            {
                w_kn[k * N + n] = tmp_nk[n * K + k];
            }
        }
        weights = Tensor(Shape{ K, N }, std::move(w_kn), execution_target);
    }
    else
    {
        weights = Tensor(K, N, execution_target);
    }

    if (weights_gradient_tensor.getShape() != Shape{ K, N } ||
        weights_gradient_tensor.getExecutionTarget() != execution_target)
    {
        weights_gradient_tensor = Tensor(K, N, execution_target);
        weights_gradient_tensor.zero();
    }

    data_type = is_mixed_precision_enabled ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    has_master_weights = false;
    has_packed_inference = false;
    master_weights = Tensor(0, 0, execution_target);
    packed_weights = Tensor(Shape{ 0, 0 }, Data_Type::TERNARY_INT2, execution_target);
    effective_weights = Tensor(0, 0, execution_target);
    gamma_tensor = Tensor(Shape{ 0 }, Data_Type::FLOAT32, execution_target);
    gpu_gamma_buffer.reset();
    gpu_partial_sums.reset();
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    invalidateWeightCache();
}

Tensor Linear_Layer::forward(const Tensor &_input_tensor)
{
    if (!_input_tensor.getImplementation())
    {
        throw std::invalid_argument("Linear_Layer::forward: Input tensor is uninitialized");
    }
    if (_input_tensor.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Linear_Layer::forward: Execution_Target mismatch");
    }
    if (_input_tensor.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Linear_Layer::forward: TERNARY_INT2 input tensor is not supported");
    }

    size_t in_features = (_input_tensor.getShape().getRank() >= 2)
                         ? _input_tensor.getShape()[_input_tensor.getShape().getRank() - 1]
                         : _input_tensor.getColumns();
    if (in_features != input_dimension)
    {
        Logger::logMessage(Input_Format{"Linear_Layer::forward: Input dimension mismatch (expected {}, got {})",
                                        input_dimension, in_features},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument(std::format("Linear_Layer input dimension mismatch (expected {}, got {})", input_dimension, in_features));
    }

    cache_orig_shape = _input_tensor.getShape();
    cache_is_nd = (cache_orig_shape.getRank() >= 3);

    Tensor eff_input = _input_tensor;
    if (cache_is_nd)
    {
        size_t outer_rows = 1;
        for (size_t d = 0; d + 1 < cache_orig_shape.getRank(); ++d)
        {
            outer_rows *= cache_orig_shape[d];
        }
        eff_input = _input_tensor.view(Shape{ outer_rows, input_dimension });
    }

    input_tensor = eff_input;
    size_t M = input_tensor.getRows();
    size_t K = input_dimension;
    size_t N = output_dimension;

    if (isTernary())
    {
        if (has_packed_inference)
        {
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                auto &engine = Execution_Engine::getInstance();
                auto &context = engine.getContext();
                if (M == 0 || N == 0 || K == 0)
                {
                    Data_Type empty_dtype = context.isFloat16Enabled() ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
                    if (M > 0 && N > 0 && K == 0)
                    {
                        Tensor zeros(Shape{ M, N }, std::vector<float>(M * N, 0.0f), execution_target);
                        output_tensor = zeros;
                    }
                    else
                    {
                        ensureTensorTarget(output_tensor, execution_target, Shape{ M, N }, empty_dtype);
                        auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output_tensor.getImplementation());
                        out_gpu.clearHostCache();
                        out_gpu.invalidateFp16Cache();
                    }
                }
                else if (context.isFloat16Enabled())
                {
                    if (!gpu_gamma_buffer)
                    {
                        throw std::logic_error("Linear_Layer::forward: Missing GPU gamma buffer in packed ternary inference");
                    }
                    const auto &x_gpu = Gpu_Tensor_Impl::castToGpu(*input_tensor.getImplementation());
                    auto contig_x = x_gpu.ensureContiguousSelf();
                    const auto *eff_x = contig_x ? contig_x.get() : &x_gpu;
                    auto buf_x = eff_x->getEffectiveFp16Storage();

                    const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*packed_weights.getImplementation());
                    auto buf_w = w_gpu.getStorageBuffer();

                    const auto &b_gpu = Gpu_Tensor_Impl::castToGpu(*biases.getImplementation());
                    auto contig_b = b_gpu.ensureContiguousSelf();
                    const auto *eff_b = contig_b ? contig_b.get() : &b_gpu;
                    auto buf_b = eff_b->getEffectiveFp16Storage();

                    ensureTensorTarget(output_tensor, execution_target, Shape{ M, N }, Data_Type::FLOAT16);
                    auto &out_gpu = Gpu_Tensor_Impl::castToGpu(*output_tensor.getImplementation());
                    auto buf_y = out_gpu.getStorageBuffer();

                    const uint32_t m_u32 = checkedU32(M, "Linear_Layer::forward(M)");
                    const uint32_t k_u32 = checkedU32(K, "Linear_Layer::forward(K)");
                    const uint32_t n_u32 = checkedU32(N, "Linear_Layer::forward(N)");
                    const uint32_t words_per_row = ceilDivU32(k_u32, 16u);
                    Compute_Node node;
                    node.pipeline_id = Compute_Pipeline::MATMUL_TERNARY_FP16;
                    node.buffers = { buf_x, buf_w, buf_b, gpu_gamma_buffer, buf_y };
                    struct PushConsts {
                        uint32_t M; uint32_t K; uint32_t N; uint32_t words_per_row; uint32_t has_bias;
                    } pcs{ m_u32, k_u32, n_u32, words_per_row, use_bias ? 1u : 0u };
                    const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
                    node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
                    node.workgroup_count_x = ceilDivU32(n_u32, 16u);
                    node.workgroup_count_y = ceilDivU32(m_u32, 16u);
                    engine.getCurrentGraph().addNode(std::move(node));
                    output_tensor.invalidateFp16Cache();
                }
                else
                {
                    engine.executeGraph();
                    engine.waitIdle();

                    float gamma_val = 1.0f;
                    if (gpu_gamma_buffer && gpu_gamma_buffer->getBufferSizeInBytes() >= sizeof(float))
                    {
                        std::vector<float> g(2, 1.0f);
                        gpu_gamma_buffer->downloadData(g);
                        gamma_val = g[0];
                    }
                    else if (!gamma_tensor.isEmpty())
                    {
                        gamma_val = gamma_tensor.getData()[0];
                    }

                    const uint32_t k_u32 = checkedU32(K, "Linear_Layer::forward(fallback K)");
                    const uint32_t n_u32 = checkedU32(N, "Linear_Layer::forward(fallback N)");
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
                                if (k < K)
                                {
                                    uint32_t code = (word >> (b * 2u)) & 3u;
                                    float val = (code == 1u) ? 1.0f : ((code == 3u) ? -1.0f : 0.0f);
                                    unpacked[k * N + n] = val * gamma_val;
                                }
                            }
                        }
                    }
                    Tensor temp_w(Shape{ K, N }, unpacked, Data_Type::FLOAT32, Execution_Target::VULKAN_GPU);
                    context.executePendingTransfers();
                    if (use_bias)
                    {
                        ops::linear::linearForward(input_tensor, temp_w, biases, output_tensor);
                    }
                    else
                    {
                        input_tensor.matmul(temp_w, output_tensor);
                    }
                }
            }
            else
            {
                ensureTensorTarget(output_tensor, execution_target, Shape{ M, N }, Data_Type::FLOAT32);
                const auto &cpu_x = Cpu_Tensor_Impl::castToCpu(*input_tensor.getImplementation());
                const auto &cpu_w = Cpu_Tensor_Impl::castToCpu(*packed_weights.getImplementation());
                const auto &cpu_b = Cpu_Tensor_Impl::castToCpu(*biases.getImplementation());
                auto &cpu_y = Cpu_Tensor_Impl::castToCpu(*output_tensor.getImplementation());

                if (!cpu_y.getStorageBuffer() || cpu_y.getStorageBuffer()->size() != M * N)
                {
                    cpu_y.getStorageBuffer() = std::make_shared<std::vector<float>>(M * N, 0.0f);
                }

                float gamma_val = gamma_tensor.isEmpty() ? 1.0f : gamma_tensor.getData()[0];
                const float *a_ptr = cpu_x.getData().data();
                const uint32_t *w_ptr = cpu_w.getPackedData().data();
                const float *b_ptr = cpu_b.getData().data();
                float *y_ptr = cpu_y.getStorageBuffer()->data();

                Cpu_Tensor_Impl::ternaryMatmul(a_ptr, w_ptr, y_ptr, M, K, N, gamma_val, b_ptr);
            }
        }
        else
        {
            if (!has_master_weights)
            {
                throw std::logic_error("Linear_Layer::forward: Missing master weights in ternary training mode");
            }
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                quantizeEffectiveWeightsGpu();
            }
            else
            {
                quantizeEffectiveWeightsCpu();
            }
            if (use_bias)
            {
                ops::linear::linearForward(input_tensor, effective_weights, biases, output_tensor);
            }
            else
            {
                input_tensor.matmul(effective_weights, output_tensor);
            }
        }
    }
    else
    {
        const Tensor *w_eff = &weights;
        if (tied_weight != nullptr)
        {
            if (tied_weight_transposed)
            {
                tied_weight->transpose().contiguous(tied_weight_t_scratch);
                w_eff = &tied_weight_t_scratch;
            }
            else
            {
                w_eff = tied_weight;
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
            if (use_bias)
            {
                ops::linear::linearForward(input_tensor_fp16, *w_eff, biases, output_tensor);
            }
            else
            {
                input_tensor_fp16.matmul(*w_eff, output_tensor);
            }
        }
        else
        {
            if (use_bias)
            {
                ops::linear::linearForward(input_tensor, *w_eff, biases, output_tensor);
            }
            else
            {
                input_tensor.matmul(*w_eff, output_tensor);
            }
        }
    }

    if (cache_is_nd)
    {
        auto orig_dims = cache_orig_shape.getDimensions();
        std::vector<size_t> out_dims(orig_dims.begin(), orig_dims.end());
        out_dims.back() = output_dimension;
        output_tensor.reshape(Shape(std::span<const size_t>{ out_dims }));
    }

    is_forward_completed = true;
    return output_tensor;
}

Tensor Linear_Layer::forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const
{
    if (isTernary())
    {
        throw std::logic_error("Linear_Layer in TERNARY_INT2 mode does not support population-batched forward");
    }
    if (_batched_params.size() != getPopulationParameterDims().size())
    {
        Logger::logMessage(Input_Format{ "Linear_Layer::forward: expected {} batched parameter tensors, got {}",
                                        getPopulationParameterDims().size(), _batched_params.size() },
            Log_Level::LOG_ERROR,
            true,
            0,
            Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument("Invalid input params size");
    }
    return _batched_input.matmulAdd(_batched_params[0], _batched_params[1]);
}

Tensor Linear_Layer::backward(const Tensor &_output_gradient)
{
    if (isTernary() && !has_master_weights)
    {
        Logger::logMessage("Linear_Layer::backward: Layer loaded in inference-only mode without master weights",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::runtime_error("Linear_Layer: Layer loaded in inference-only mode without master weights");
    }

    if (isTernary() && (has_packed_inference || effective_weights.isEmpty()))
    {
        Logger::logMessage("Linear_Layer::backward: Layer is in packed inference mode or effective weights are empty; call setTrainingMode(true) and forward() before backward()",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("Linear_Layer::backward: Layer is in packed inference mode; call setTrainingMode(true) and forward() before backward()");
    }

    if (!is_forward_completed)
    {
        Logger::logMessage(Input_Format{"Linear_Layer::backward: Backward called before forward"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("Backward called before forward");
    }

    if (!_output_gradient.getImplementation())
    {
        throw std::invalid_argument("Linear_Layer::backward: Output gradient tensor is uninitialized");
    }
    if (_output_gradient.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Linear_Layer::backward: Execution_Target mismatch");
    }
    if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Linear_Layer::backward: TERNARY_INT2 gradient tensor is not supported");
    }

    size_t out_features = (_output_gradient.getShape().getRank() >= 2)
                          ? _output_gradient.getShape()[_output_gradient.getShape().getRank() - 1]
                          : _output_gradient.getColumns();
    if (out_features != output_dimension)
    {
        Logger::logMessage(Input_Format{"Linear_Layer::backward: Gradient output dimension mismatch"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::invalid_argument(std::format("Linear_Layer gradient output dimension mismatch (expected {}, got {})", output_dimension, out_features));
    }

    Tensor eff_grad = _output_gradient;
    if (cache_is_nd)
    {
        auto orig_dims = cache_orig_shape.getDimensions();
        std::vector<size_t> expected_out_dims(orig_dims.begin(), orig_dims.end());
        expected_out_dims.back() = output_dimension;
        Shape expected_out_shape(std::span<const size_t>{ expected_out_dims });
        if (eff_grad.getShape() != expected_out_shape)
        {
            throw std::invalid_argument(std::format(
                "Linear_Layer::backward: N-D gradient shape {} mismatch (expected {})",
                eff_grad.getShape().toString(), expected_out_shape.toString()));
        }
        size_t expected_rows = 1;
        for (size_t d = 0; d + 1 < cache_orig_shape.getRank(); ++d)
        {
            expected_rows *= cache_orig_shape[d];
        }
        eff_grad = _output_gradient.view(Shape{ expected_rows, output_dimension });
    }

    if (isTernary())
    {
        if (execution_target == Execution_Target::VULKAN_GPU && Execution_Engine::getInstance().getContext().isFloat16Enabled() && input_tensor.getDataType() != Data_Type::FLOAT16)
        {
            ops::linear::linearBackwardInput(eff_grad, effective_weights, input_gradient_tensor_fp16);
            if (input_gradient_tensor_fp16.getDataType() == Data_Type::FLOAT16)
            {
                input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
            }
            else
            {
                input_gradient_tensor = input_gradient_tensor_fp16;
            }
        }
        else
        {
            ops::linear::linearBackwardInput(eff_grad, effective_weights, input_gradient_tensor);
        }
        if (is_trainable)
        {
            Tensor &eff_bgrad = use_bias ? biases_gradient_tensor : unused_bias_grad_scratch;
            ops::linear::linearBackwardWeightBias(input_tensor, eff_grad, weights_gradient_tensor, eff_bgrad, is_accumulated);
        }
    }
    else if (tied_weight != nullptr)
    {
        const bool use_amp = is_mixed_precision_enabled &&
                             execution_target == Execution_Target::VULKAN_GPU &&
                             Execution_Engine::getInstance().getContext().isFloat16Enabled();
        const Tensor *in_ptr = &input_tensor;
        const Tensor *grad_ptr = &eff_grad;
        if (use_amp)
        {
            if (input_tensor.getDataType() != Data_Type::FLOAT16)
            {
                input_tensor.to(Data_Type::FLOAT16, input_tensor_fp16);
                in_ptr = &input_tensor_fp16;
            }
            if (eff_grad.getDataType() != Data_Type::FLOAT16)
            {
                eff_grad.to(Data_Type::FLOAT16, output_gradient_fp16);
                grad_ptr = &output_gradient_fp16;
            }
        }

        if (tied_weight_transposed)
        {
            Tensor &target_in_grad = use_amp ? input_gradient_tensor_fp16 : input_gradient_tensor;
            grad_ptr->matmul(*tied_weight, target_in_grad);
            if (use_amp)
            {
                if (input_tensor.getDataType() == Data_Type::FLOAT16)
                {
                    input_gradient_tensor = input_gradient_tensor_fp16;
                }
                else
                {
                    input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
                }
            }
        }
        else
        {
            Tensor &target_in_grad = use_amp ? input_gradient_tensor_fp16 : input_gradient_tensor;
            ops::linear::linearBackwardInput(*grad_ptr, *tied_weight, target_in_grad);
            if (use_amp)
            {
                if (input_tensor.getDataType() == Data_Type::FLOAT16)
                {
                    input_gradient_tensor = input_gradient_tensor_fp16;
                }
                else
                {
                    input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
                }
            }
        }

        if (is_trainable && tied_weight_grad != nullptr)
        {
            if (tied_weight_grad->isEmpty() || tied_weight_grad->getShape() != tied_weight->getShape() ||
                tied_weight_grad->getExecutionTarget() != execution_target)
            {
                *tied_weight_grad = Tensor(tied_weight->getShape(), Data_Type::FLOAT32, execution_target);
                tied_weight_grad->zero();
            }

            Tensor dw(execution_target);
            if (tied_weight_transposed)
            {
                Tensor grad_t = grad_ptr->transpose().contiguous();
                grad_t.matmul(*in_ptr, dw);
            }
            else
            {
                Tensor in_t = in_ptr->transpose().contiguous();
                in_t.matmul(*grad_ptr, dw);
            }

            const Tensor *dw_fp32 = &dw;
            if (dw.getDataType() != tied_weight_grad->getDataType())
            {
                dw.to(tied_weight_grad->getDataType(), tied_grad_fp32_scratch);
                dw_fp32 = &tied_grad_fp32_scratch;
            }
            if (is_accumulated)
            {
                tied_weight_grad->add(*dw_fp32, *tied_weight_grad);
            }
            else
            {
                dw_fp32->mulScalar(1.0f, *tied_weight_grad);
            }

            if (use_bias)
            {
                const size_t M = in_ptr->getRows();
                Tensor ones(Shape{ 1, M }, std::vector<float>(M, 1.0f), execution_target);
                Tensor db_tmp(execution_target);
                ones.matmul(*grad_ptr, db_tmp);
                const Tensor *db_fp32 = &db_tmp;
                Tensor db_cast(execution_target);
                if (db_tmp.getDataType() != biases_gradient_tensor.getDataType())
                {
                    db_tmp.to(biases_gradient_tensor.getDataType(), db_cast);
                    db_fp32 = &db_cast;
                }
                if (is_accumulated)
                {
                    biases_gradient_tensor.add(*db_fp32, biases_gradient_tensor);
                }
                else
                {
                    db_fp32->mulScalar(1.0f, biases_gradient_tensor);
                }
            }
        }
    }
    else
    {
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
            if (eff_grad.getDataType() != Data_Type::FLOAT16)
            {
                eff_grad.to(Data_Type::FLOAT16, output_gradient_fp16);
            }
            else
            {
                output_gradient_fp16 = eff_grad;
            }
            ops::linear::linearBackwardInput(output_gradient_fp16, weights, input_gradient_tensor_fp16);
            if (is_trainable)
            {
                Tensor &eff_bgrad = use_bias ? biases_gradient_tensor : unused_bias_grad_scratch;
                ops::linear::linearBackwardWeightBias(input_tensor_fp16, output_gradient_fp16, weights_gradient_tensor, eff_bgrad, is_accumulated);
            }
            if (input_tensor.getDataType() == Data_Type::FLOAT16)
            {
                input_gradient_tensor = input_gradient_tensor_fp16;
            }
            else
            {
                input_gradient_tensor_fp16.to(Data_Type::FLOAT32, input_gradient_tensor);
            }
        }
        else
        {
            ops::linear::linearBackwardInput(eff_grad, weights, input_gradient_tensor);
            if (is_trainable)
            {
                Tensor &eff_bgrad = use_bias ? biases_gradient_tensor : unused_bias_grad_scratch;
                ops::linear::linearBackwardWeightBias(input_tensor, eff_grad, weights_gradient_tensor, eff_bgrad, is_accumulated);
            }
        }
    }

    if (cache_is_nd)
    {
        input_gradient_tensor.reshape(cache_orig_shape);
    }

    return input_gradient_tensor;
}

void Linear_Layer::copyStateTo(Linear_Layer &dst) const
{
    copyBaseStateTo(dst);
    dst.use_bias = use_bias;
    dst.biases = biases.clone();
    if (isTernary())
    {
        dst.has_master_weights = has_master_weights;
        dst.has_packed_inference = has_packed_inference;
        if (has_master_weights)
        {
            dst.master_weights = master_weights.clone();
        }
        else
        {
            dst.master_weights = Tensor(0, 0, execution_target);
        }

        if (has_packed_inference)
        {
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().executeGraph();
                Execution_Engine::getInstance().waitIdle();
            }
            dst.packed_weights = packed_weights.clone();
            dst.gamma_tensor = gamma_tensor.clone();
            dst.effective_weights = Tensor(0, 0, execution_target);
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
                if (!dst.gpu_gamma_buffer)
                {
                    dst.gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
                    dst.gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
                }
                dst.gpu_gamma_buffer->uploadData(g);
                context.executePendingTransfers();
            }
        }
        else
        {
            dst.effective_weights = effective_weights.clone();
            dst.gamma_tensor = gamma_tensor.clone();
        }
    }
    else
    {
        dst.weights = weights.clone();
    }
    dst.invalidateWeightCache();
}

std::unique_ptr<ILayer> Linear_Layer::clone() const
{
    auto cloned = std::make_unique<Linear_Layer>(input_dimension, output_dimension, execution_target, initialization_gain, data_type, use_bias);
    copyStateTo(*cloned);
    if (tied_weight != nullptr)
    {
        cloned->tieWeights(tied_weight, tied_weight_grad, tied_weight_transposed);
    }
    return cloned;
}

void Linear_Layer::invalidateWeightCache() noexcept
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
    }
}

void Linear_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    invalidateWeightCache();
}

void Linear_Layer::setTrainingMode(bool _is_training)
{
    if (isTernary() && _is_training)
    {
        if (!has_master_weights)
        {
            Logger::logMessage("Linear_Layer::setTrainingMode: Layer loaded in inference-only mode without master weights",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DENSE_COMPUTE);
            throw std::runtime_error("Linear_Layer: Layer loaded in inference-only mode without master weights");
        }
        has_packed_inference = false;
    }
}

void Linear_Layer::resetGradient()
{
    is_forward_completed = false;
}

void Linear_Layer::resetGradients()
{
    resetGradient();
    weights_gradient_tensor.zero();
    biases_gradient_tensor.zero();
}

void Linear_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Linear_Layer::saveConfiguration: Output stream is not open or writable");
    }
    const uint32_t input_dimension_value = checkedU32(input_dimension, "Linear_Layer::saveConfiguration(input_dimension)");
    const uint32_t output_dimension_value = checkedU32(output_dimension, "Linear_Layer::saveConfiguration(output_dimension)");
    _output_file_stream.write(reinterpret_cast<const char *>(&input_dimension_value), sizeof(input_dimension_value));
    _output_file_stream.write(reinterpret_cast<const char *>(&output_dimension_value), sizeof(output_dimension_value));
    if (!_output_file_stream.good())
    {
        throw std::runtime_error("Linear_Layer::saveConfiguration: Failed writing layer dimensions");
    }
}

void Linear_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Linear_Layer::saveInference: Output stream is not open or writable");
    }
    if (isTernary())
    {
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
            Execution_Engine::getInstance().waitIdle();
        }

        _output_file_stream.write(reinterpret_cast<const char *>(&TERNARY_INFERENCE_MAGIC_V2), sizeof(TERNARY_INFERENCE_MAGIC_V2));
        uint64_t saved_k = static_cast<uint64_t>(input_dimension);
        uint64_t saved_n = static_cast<uint64_t>(output_dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&saved_k), sizeof(saved_k));
        _output_file_stream.write(reinterpret_cast<const char *>(&saved_n), sizeof(saved_n));

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
                throw std::logic_error("Linear_Layer::saveInference: Missing both packed weights and master weights");
            }
            const size_t K = input_dimension;
            const size_t N = output_dimension;
            const size_t total = K * N;
            const uint32_t words_per_row = ceilDivU32(checkedU32(K, "Linear_Layer::saveInference(K)"), 16u);
            const uint32_t n_u32 = checkedU32(N, "Linear_Layer::saveInference(N)");
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
                            if (k < K)
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

        _output_file_stream.write(reinterpret_cast<const char *>(&gamma_val), sizeof(gamma_val));
        _output_file_stream.write(reinterpret_cast<const char *>(&inv_gamma_val), sizeof(inv_gamma_val));

        const uint32_t words_count = checkedU32(packed_ptr->size(), "Linear_Layer::saveInference(words_count)");
        _output_file_stream.write(reinterpret_cast<const char *>(&words_count), sizeof(words_count));
        if (words_count > 0)
        {
            _output_file_stream.write(reinterpret_cast<const char *>(packed_ptr->data()), static_cast<std::streamsize>(static_cast<size_t>(words_count) * sizeof(uint32_t)));
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
        throw std::runtime_error("Linear_Layer::saveInference: Failed writing inference state");
    }
}

void Linear_Layer::loadInference(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open())
    {
        throw std::runtime_error("Linear_Layer::loadInference: Input stream is not open");
    }
    auto current_pos = _input_file_stream.tellg();
    uint32_t magic = 0;
    _input_file_stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (!_input_file_stream)
    {
        throw std::runtime_error("Linear_Layer::loadInference: Failed reading inference header");
    }

    if (magic == TERNARY_INFERENCE_MAGIC_V2 || magic == TERNARY_INFERENCE_MAGIC)
    {
        const bool is_v2 = (magic == TERNARY_INFERENCE_MAGIC_V2);
        size_t loaded_k = input_dimension;
        size_t loaded_n = output_dimension;
        if (is_v2)
        {
            uint64_t k_u64 = 0, n_u64 = 0;
            _input_file_stream.read(reinterpret_cast<char *>(&k_u64), sizeof(k_u64));
            _input_file_stream.read(reinterpret_cast<char *>(&n_u64), sizeof(n_u64));
            if (!_input_file_stream || k_u64 > std::numeric_limits<size_t>::max() || n_u64 > std::numeric_limits<size_t>::max())
            {
                throw std::runtime_error("Linear_Layer::loadInference: Failed reading or overflowing ternary V2 dimensions");
            }
            loaded_k = static_cast<size_t>(k_u64);
            loaded_n = static_cast<size_t>(n_u64);
        }

        float gamma_val = 0.0f;
        float inv_gamma_val = 0.0f;
        _input_file_stream.read(reinterpret_cast<char *>(&gamma_val), sizeof(gamma_val));
        _input_file_stream.read(reinterpret_cast<char *>(&inv_gamma_val), sizeof(inv_gamma_val));

        uint32_t words_count = 0;
        _input_file_stream.read(reinterpret_cast<char *>(&words_count), sizeof(words_count));
        if (!_input_file_stream)
        {
            throw std::runtime_error("Linear_Layer::loadInference: Failed reading ternary header metadata");
        }
        if (!std::isfinite(gamma_val) || gamma_val <= 0.0f || !std::isfinite(inv_gamma_val) || inv_gamma_val <= 0.0f)
        {
            throw std::runtime_error("Linear_Layer::loadInference: Invalid gamma values in ternary inference file");
        }

        const uint64_t words_per_row = (static_cast<uint64_t>(loaded_k) + 15ULL) / 16ULL;
        if (is_v2)
        {
            if (words_per_row > 0 && static_cast<uint64_t>(loaded_n) > std::numeric_limits<uint64_t>::max() / words_per_row)
            {
                throw std::runtime_error("Linear_Layer::loadInference: Packed word count overflow");
            }
            const uint64_t expected_words = static_cast<uint64_t>(loaded_n) * words_per_row;
            if (words_count != expected_words)
            {
                throw std::runtime_error("Linear_Layer::loadInference: Packed word count mismatch");
            }
        }
        else
        {
            if (loaded_k == 0 && words_count != 0)
            {
                throw std::runtime_error("Linear_Layer::loadInference: Legacy V1 ternary inference requires preconfigured input_dimension");
            }
            if ((words_per_row > 0 && (static_cast<uint64_t>(words_count) % words_per_row) != 0ULL) ||
                static_cast<uint64_t>(words_count) > (1ULL << 29))
            {
                throw std::runtime_error("Linear_Layer::loadInference: Invalid legacy V1 packed word count");
            }
        }

        std::vector<uint32_t> packed_data(words_count);
        if (words_count > 0)
        {
            _input_file_stream.read(reinterpret_cast<char *>(packed_data.data()), static_cast<std::streamsize>(static_cast<size_t>(words_count) * sizeof(uint32_t)));
            if (!_input_file_stream)
            {
                throw std::runtime_error("Linear_Layer::loadInference: Failed reading ternary packed payload");
            }
        }

        Tensor loaded_biases = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_biases.getDataType() == Data_Type::TERNARY_INT2)
        {
            throw std::runtime_error("Linear_Layer::loadInference: Invalid bias dtype");
        }
        if (is_v2)
        {
            if (loaded_biases.getTotalElements() != loaded_n)
            {
                throw std::runtime_error("Linear_Layer::loadInference: Bias size does not match ternary output_dimension");
            }
        }
        else
        {
            loaded_n = loaded_biases.getColumns();
            if (loaded_biases.getTotalElements() != loaded_n ||
                static_cast<uint64_t>(words_count) != static_cast<uint64_t>(loaded_n) * words_per_row)
            {
                throw std::runtime_error("Linear_Layer::loadInference: Legacy V1 bias or packed size mismatch");
            }
        }

        data_type = Data_Type::TERNARY_INT2;
        input_dimension = loaded_k;
        output_dimension = loaded_n;
        biases = std::move(loaded_biases);

        packed_weights = Tensor(Shape{ output_dimension, input_dimension }, Data_Type::TERNARY_INT2, execution_target);
        if (words_count > 0)
        {
            packed_weights.uploadPackedData(packed_data);
        }

        std::vector<float> gv = { gamma_val, inv_gamma_val };
        gamma_tensor = Tensor(Shape{ 2 }, std::move(gv), execution_target);

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
        has_master_weights = false;
        has_packed_inference = true;
        is_forward_completed = false;
        master_weights = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        weights = Tensor(0, 0, execution_target);
        invalidateWeightCache();
    }
    else
    {
        _input_file_stream.clear();
        _input_file_stream.seekg(current_pos);
        Tensor loaded_w = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_w.getShape().getRank() != 2 || loaded_w.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_b.getDataType() == Data_Type::TERNARY_INT2 || loaded_b.getTotalElements() != loaded_w.getColumns())
        {
            throw std::runtime_error("Linear_Layer::loadInference: Corrupted weight or bias tensor shape/dtype");
        }
        weights = std::move(loaded_w);
        biases = std::move(loaded_b);
        input_dimension = weights.getRows();
        output_dimension = weights.getColumns();
        data_type = (weights.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        if (data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        master_weights = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        packed_weights = Tensor(Shape{ 0, 0 }, Data_Type::TERNARY_INT2, execution_target);
        gamma_tensor = Tensor(Shape{ 0 }, Data_Type::FLOAT32, execution_target);
        gpu_gamma_buffer.reset();
        gpu_partial_sums.reset();
        has_packed_inference = false;
        has_master_weights = true;
        is_forward_completed = false;
        invalidateWeightCache();
    }
}

void Linear_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    if (!_output_file_stream.is_open() || !_output_file_stream.good())
    {
        throw std::runtime_error("Linear_Layer::saveCheckpoint: Output stream is not open or writable");
    }
    if (isTernary())
    {
        if (!has_master_weights)
        {
            throw std::logic_error("Linear_Layer::saveCheckpoint: Cannot save checkpoint from inference-only ternary layer without master weights");
        }
        _output_file_stream.write(reinterpret_cast<const char *>(&TERNARY_CHECKPOINT_MAGIC), sizeof(TERNARY_CHECKPOINT_MAGIC));
        master_weights.saveTensor(_output_file_stream);
        biases.saveTensor(_output_file_stream);
        weights_gradient_tensor.saveTensor(_output_file_stream);
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
        throw std::runtime_error("Linear_Layer::saveCheckpoint: Failed writing checkpoint state");
    }
}

void Linear_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    if (!_input_file_stream.is_open())
    {
        throw std::runtime_error("Linear_Layer::loadCheckpoint: Input stream is not open");
    }
    auto current_pos = _input_file_stream.tellg();
    uint32_t magic = 0;
    _input_file_stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (!_input_file_stream)
    {
        throw std::runtime_error("Linear_Layer::loadCheckpoint: Failed reading checkpoint header");
    }

    if (magic == TERNARY_CHECKPOINT_MAGIC)
    {
        Tensor loaded_mw = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_wg = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_bg = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_mw.getShape().getRank() != 2 || loaded_mw.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_b.getTotalElements() != loaded_mw.getColumns() ||
            loaded_wg.getTotalElements() != loaded_mw.getTotalElements() ||
            loaded_bg.getTotalElements() != loaded_mw.getColumns())
        {
            throw std::runtime_error("Linear_Layer::loadCheckpoint: Corrupted ternary checkpoint tensor shapes");
        }
        data_type = Data_Type::TERNARY_INT2;
        master_weights = std::move(loaded_mw);
        biases = std::move(loaded_b);
        weights_gradient_tensor = std::move(loaded_wg);
        biases_gradient_tensor = std::move(loaded_bg);
        input_dimension = master_weights.getRows();
        output_dimension = master_weights.getColumns();
        has_master_weights = true;
        has_packed_inference = false;
        is_forward_completed = false;
        effective_weights = Tensor(0, 0, execution_target);
        packed_weights = Tensor(Shape{ 0, 0 }, Data_Type::TERNARY_INT2, execution_target);
        gamma_tensor = Tensor(Shape{ 2 }, Data_Type::FLOAT32, execution_target);
        weights = Tensor(0, 0, execution_target);
        ensureTernaryGpuBuffers();
        invalidateWeightCache();
    }
    else
    {
        _input_file_stream.clear();
        _input_file_stream.seekg(current_pos);
        Tensor loaded_w = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_b = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_wg = Tensor::loadTensor(_input_file_stream, execution_target);
        Tensor loaded_bg = Tensor::loadTensor(_input_file_stream, execution_target);
        if (loaded_w.getShape().getRank() != 2 || loaded_w.getDataType() == Data_Type::TERNARY_INT2 ||
            loaded_b.getTotalElements() != loaded_w.getColumns() ||
            loaded_wg.getTotalElements() != loaded_w.getTotalElements() ||
            loaded_bg.getTotalElements() != loaded_w.getColumns())
        {
            throw std::runtime_error("Linear_Layer::loadCheckpoint: Corrupted checkpoint tensor shapes");
        }
        weights = std::move(loaded_w);
        biases = std::move(loaded_b);
        weights_gradient_tensor = std::move(loaded_wg);
        biases_gradient_tensor = std::move(loaded_bg);
        input_dimension = weights.getRows();
        output_dimension = weights.getColumns();
        data_type = (weights.getDataType() == Data_Type::FLOAT16) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        if (data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        master_weights = Tensor(0, 0, execution_target);
        effective_weights = Tensor(0, 0, execution_target);
        packed_weights = Tensor(Shape{ 0, 0 }, Data_Type::TERNARY_INT2, execution_target);
        gamma_tensor = Tensor(Shape{ 0 }, Data_Type::FLOAT32, execution_target);
        gpu_gamma_buffer.reset();
        gpu_partial_sums.reset();
        has_master_weights = true;
        has_packed_inference = false;
        is_forward_completed = false;
        invalidateWeightCache();
    }
}

std::function<float(std::mt19937&)> Linear_Layer::getPopulationParameterInitializer(size_t param_index) const
{
    if (param_index == 0)
    {
        float standard_deviation = (input_dimension > 0) ? std::sqrt(initialization_gain / static_cast<float>(input_dimension)) : 0.0f;
        return [standard_deviation](std::mt19937& generator)
        {
            std::normal_distribution<float> distribution(0.0f, standard_deviation);
            return distribution(generator);
        };
    }
    return [](std::mt19937&)
    {
        return 0.0f;
    };
}

std::vector<float> Linear_Layer::getPopulationParameter(size_t param_index) const
{
    if (param_index == 0)
    {
        return isTernary() ? master_weights.getData() : weights.getData();
    }
    if (param_index == 1)
    {
        return biases.getData();
    }
    throw std::out_of_range("Linear_Layer::getPopulationParameter: Parameter index out of range");
}

void Linear_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
    if (param_index == 0)
    {
        if (flat_data.size() != input_dimension * output_dimension)
        {
            throw std::invalid_argument("Linear_Layer::setPopulationParameter: Weight size mismatch");
        }
        if (isTernary())
        {
            master_weights = Tensor(input_dimension, output_dimension, std::move(flat_data), execution_target);
            has_master_weights = true;
            has_packed_inference = false;
            ensureTernaryGpuBuffers();
        }
        else
        {
            weights = Tensor(input_dimension, output_dimension, std::move(flat_data), execution_target);
        }
        invalidateWeightCache();
    }
    else if (param_index == 1)
    {
        if (flat_data.size() != output_dimension)
        {
            throw std::invalid_argument("Linear_Layer::setPopulationParameter: Bias size mismatch");
        }
        biases = Tensor(1, output_dimension, std::move(flat_data), execution_target);
        invalidateWeightCache();
    }
    else
    {
        throw std::out_of_range("Linear_Layer::setPopulationParameter: Parameter index out of range");
    }
}

std::vector<std::pair<Tensor *, Tensor *>> Linear_Layer::getParametersAndGradients()
{
    if (!is_trainable)
    {
        return {};
    }
    std::vector<std::pair<Tensor *, Tensor *>> params;
    if (tied_weight == nullptr)
    {
        if (isTernary())
        {
            if (!has_master_weights)
            {
                return {};
            }
            params.push_back({ &master_weights, &weights_gradient_tensor });
        }
        else
        {
            params.push_back({ &weights, &weights_gradient_tensor });
        }
    }
    if (use_bias)
    {
        params.push_back({ &biases, &biases_gradient_tensor });
    }
    return params;
}

const Tensor &Linear_Layer::getWeights() const
{
    if (tied_weight != nullptr)
    {
        return *tied_weight;
    }
    if (isTernary())
    {
        return has_master_weights ? master_weights : packed_weights;
    }
    return weights;
}

void Linear_Layer::tieWeights(Tensor *shared_weight, Tensor *shared_grad, bool transposed)
{
    if (shared_weight != nullptr)
    {
        const Shape &ws = shared_weight->getShape();
        if (ws.getRank() != 2)
        {
            throw std::invalid_argument("Linear_Layer::tieWeights: Shared weight must have rank 2");
        }
        const size_t expected_r = transposed ? output_dimension : input_dimension;
        const size_t expected_c = transposed ? input_dimension : output_dimension;
        if (ws[0] != expected_r || ws[1] != expected_c)
        {
            throw std::invalid_argument("Linear_Layer::tieWeights: Shared weight dimensions mismatch");
        }
        if (shared_weight->getExecutionTarget() != execution_target)
        {
            throw std::invalid_argument("Linear_Layer::tieWeights: Execution_Target mismatch");
        }
    }
    tied_weight = shared_weight;
    tied_weight_grad = shared_grad;
    tied_weight_transposed = transposed;
    invalidateWeightCache();
}

void Linear_Layer::untieWeights() noexcept
{
    tied_weight = nullptr;
    tied_weight_grad = nullptr;
    tied_weight_transposed = false;
    tied_weight_t_scratch = Tensor(0, 0, execution_target);
}

Tensor Linear_Layer::forwardIndexed(const Tensor &hidden, const Tensor &target_indices)
{
    if (!hidden.getImplementation() || !target_indices.getImplementation())
    {
        throw std::invalid_argument("Linear_Layer::forwardIndexed: Uninitialized tensor");
    }
    if (hidden.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Linear_Layer::forwardIndexed: Execution_Target mismatch");
    }
    const Shape &h_shape = hidden.getShape();
    const size_t rank = h_shape.getRank();
    if (rank < 2 || h_shape[rank - 1] != input_dimension)
    {
        throw std::invalid_argument("Linear_Layer::forwardIndexed: Hidden dimension mismatch");
    }
    const size_t n_tgt = target_indices.getTotalElements();
    if (n_tgt == 0)
    {
        is_forward_completed = false;
        return Tensor(Shape{ 0, output_dimension }, is_mixed_precision_enabled ? Data_Type::FLOAT16 : data_type, execution_target);
    }
    const size_t outer_rows = hidden.getTotalElements() / input_dimension;
    Tensor hidden_2d = (rank == 2) ? hidden : hidden.view(Shape{ outer_rows, input_dimension });
    Tensor target_dev = (target_indices.getExecutionTarget() == execution_target)
        ? target_indices
        : Tensor(target_indices.getShape(), target_indices.getData(), execution_target);
    ops::llm::embeddingForward(hidden_2d, target_dev, indexed_hidden_scratch, is_mixed_precision_enabled);
    return forward(indexed_hidden_scratch);
}

Tensor Linear_Layer::backwardIndexed(const Tensor &d_logits, const Tensor &target_indices, const Shape &full_hidden_shape)
{
    const size_t rank = full_hidden_shape.getRank();
    if (rank < 2 || full_hidden_shape[rank - 1] != input_dimension)
    {
        throw std::invalid_argument("Linear_Layer::backwardIndexed: full_hidden_shape dimension mismatch");
    }
    const size_t n_tgt = target_indices.getTotalElements();
    if (n_tgt == 0)
    {
        Tensor z(full_hidden_shape, Data_Type::FLOAT32, execution_target);
        z.zero();
        return z;
    }
    Tensor d_gathered = backward(d_logits);
    const size_t outer_rows = full_hidden_shape.getTotalElements() / input_dimension;
    const Shape flat_shape{ outer_rows, input_dimension };
    if (!indexed_grad_scratch.getImplementation() ||
        indexed_grad_scratch.getExecutionTarget() != execution_target ||
        indexed_grad_scratch.getDataType() != Data_Type::FLOAT32 ||
        indexed_grad_scratch.getShape() != flat_shape)
    {
        indexed_grad_scratch = Tensor(flat_shape, Data_Type::FLOAT32, execution_target);
    }
    Tensor target_dev = (target_indices.getExecutionTarget() == execution_target)
        ? target_indices
        : Tensor(target_indices.getShape(), target_indices.getData(), execution_target);
    ops::llm::embeddingBackward(indexed_grad_scratch, target_dev, d_gathered, indexed_grad_scratch, false);
    Tensor d_full = indexed_grad_scratch;
    d_full.reshape(full_hidden_shape);
    return d_full;
}

void Linear_Layer::setWeights(const Tensor &_new_weights)
{
    if (isTernary())
    {
        setMasterWeights(_new_weights);
        return;
    }
    if (!_new_weights.getImplementation())
    {
        throw std::invalid_argument("Linear_Layer::setWeights: Tensor is uninitialized");
    }
    if (_new_weights.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Linear_Layer::setWeights: Execution_Target mismatch");
    }
    if (_new_weights.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Linear_Layer::setWeights: TERNARY_INT2 weights cannot be assigned in dense mode");
    }
    if (_new_weights.getRows() != input_dimension || _new_weights.getColumns() != output_dimension)
    {
        Logger::logMessage(Input_Format{"Linear_Layer::setWeights: Dimension size of weight must match"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::TENSOR_INSPECTION);
        throw std::invalid_argument("Dimension size of weight must match");
    }
    weights = _new_weights;
    invalidateWeightCache();
}

void Linear_Layer::setMasterWeights(const Tensor &_new_weights)
{
    if (!_new_weights.getImplementation())
    {
        throw std::invalid_argument("Linear_Layer::setMasterWeights: Tensor is uninitialized");
    }
    if (_new_weights.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Linear_Layer::setMasterWeights: Execution_Target mismatch");
    }
    if (_new_weights.getDataType() != Data_Type::FLOAT32)
    {
        throw std::invalid_argument("Linear_Layer::setMasterWeights: Master weights must have FLOAT32 Data_Type");
    }
    if (_new_weights.getRows() != input_dimension || _new_weights.getColumns() != output_dimension)
    {
        throw std::invalid_argument("Linear_Layer::setMasterWeights: Dimension size of weight must match");
    }
    const auto impl = _new_weights.getImplementation();
    master_weights = (impl->isContiguous() && impl->getByteOffset() == 0) ? _new_weights : _new_weights.contiguous();
    has_master_weights = true;
    has_packed_inference = false;
    ensureTernaryGpuBuffers();
    invalidateWeightCache();
}

void Linear_Layer::setBiases(const Tensor &_new_biases)
{
    if (!_new_biases.getImplementation())
    {
        throw std::invalid_argument("Linear_Layer::setBiases: Tensor is uninitialized");
    }
    if (_new_biases.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Linear_Layer::setBiases: Execution_Target mismatch");
    }
    if (_new_biases.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Linear_Layer::setBiases: TERNARY_INT2 biases are not supported");
    }
    if (_new_biases.getRows() != 1 || _new_biases.getColumns() != output_dimension)
    {
        Logger::logMessage(Input_Format{"Linear_Layer::setBiases: Dimension size of bias must match"},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::TENSOR_INSPECTION);
        throw std::invalid_argument("Dimension size of bias must match");
    }
    biases = _new_biases;
    invalidateWeightCache();
}

void Linear_Layer::setExecutionTarget(Execution_Target _new_execution_target)
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
    output_gradient_fp16 = Tensor(0, 0, _new_execution_target);
    input_gradient_tensor_fp16 = Tensor(0, 0, _new_execution_target);
    tied_weight_t_scratch = Tensor(0, 0, _new_execution_target);
    tied_grad_fp32_scratch = Tensor(0, 0, _new_execution_target);
    unused_bias_grad_scratch = Tensor(0, 0, _new_execution_target);
    indexed_hidden_scratch = Tensor(0, 0, _new_execution_target);
    indexed_grad_scratch = Tensor(0, 0, _new_execution_target);

    if (isTernary())
    {
        if (has_master_weights)
        {
            master_weights.setExecutionTarget(_new_execution_target);
        }
        effective_weights.setExecutionTarget(_new_execution_target);
        packed_weights.setExecutionTarget(_new_execution_target);
        gamma_tensor.setExecutionTarget(_new_execution_target);

        if (_new_execution_target == Execution_Target::VULKAN_GPU)
        {
            auto &context = Execution_Engine::getInstance().getContext();
            gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
            gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
            gpu_gamma_buffer->uploadData(saved_gamma);

            const uint32_t total_elements = checkedU32(input_dimension * output_dimension, "Linear_Layer::setExecutionTarget");
            if (total_elements > 0)
            {
                const uint32_t num_workgroups = std::min(ceilDivU32(total_elements, 256u), 1024u);
                gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
            }
            context.executePendingTransfers();
        }
        else
        {
            gpu_gamma_buffer = nullptr;
            gpu_partial_sums = nullptr;
        }
    }
    invalidateWeightCache();
}

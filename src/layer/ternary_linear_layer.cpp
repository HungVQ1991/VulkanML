#include "layer/ternary_linear_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "engine/vulkan_context.h"
#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/cpu_tensor_impl.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"
#include "ops/linear/linear.h"

namespace
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
} // namespace

Ternary_Linear_Layer::Ternary_Linear_Layer()
    : master_weights(0, 0),
      biases(0, 0),
      effective_weights(0, 0),
      packed_weights(Shape{ 0, 0 }, Data_Type::TERNARY_INT2),
      gamma_tensor(Shape{ 2 }, Data_Type::FLOAT32),
      input_tensor(0, 0),
      output_tensor(0, 0),
      input_gradient_tensor(0, 0),
      master_weights_gradient_tensor(0, 0),
      biases_gradient_tensor(0, 0),
      is_forward_completed(false),
      has_packed_inference(false),
      has_master_weights(true),
      execution_target(Execution_Target::CPU)
{
}

Ternary_Linear_Layer::Ternary_Linear_Layer(size_t _input_dimension,
                                           size_t _output_dimension,
                                           Execution_Target _execution_target,
                                           float _initialization_gain)
    : master_weights(_input_dimension, _output_dimension, _execution_target),
      biases(1, _output_dimension, _execution_target),
      effective_weights(_input_dimension, _output_dimension, _execution_target),
      packed_weights(Shape{ _output_dimension, _input_dimension }, Data_Type::TERNARY_INT2, _execution_target),
      gamma_tensor(Shape{ 2 }, Data_Type::FLOAT32, _execution_target),
      input_tensor(0, 0, _execution_target),
      output_tensor(0, 0, _execution_target),
      input_gradient_tensor(0, 0, _execution_target),
      master_weights_gradient_tensor(_input_dimension, _output_dimension, _execution_target),
      biases_gradient_tensor(1, _output_dimension, _execution_target),
      input_dimension(_input_dimension),
      output_dimension(_output_dimension),
      initialization_gain(_initialization_gain),
      is_forward_completed(false),
      has_packed_inference(false),
      has_master_weights(true),
      execution_target(_execution_target)
{
    std::vector<float> weight_data(_input_dimension * _output_dimension);
    std::vector<float> bias_data(_output_dimension, 0.0f);

    float standard_deviation = (_input_dimension > 0) ? std::sqrt(_initialization_gain / static_cast<float>(_input_dimension)) : 1.0f;
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::normal_distribution<float> normal_distribution(0.0f, standard_deviation);

    for (float &weight_value : weight_data)
    {
        weight_value = normal_distribution(generator);
    }

    master_weights = Tensor(_input_dimension, _output_dimension, std::move(weight_data), execution_target);
    biases = Tensor(1, _output_dimension, std::move(bias_data), execution_target);

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &context = Execution_Engine::getInstance().getContext();
        gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
        gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);

        uint32_t total_elements = static_cast<uint32_t>(_input_dimension * _output_dimension);
        uint32_t num_workgroups = std::min((total_elements + 255u) / 256u, 1024u);
        gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
    }
}

void Ternary_Linear_Layer::setExecutionTarget(Execution_Target _new_execution_target)
{
    if (execution_target == _new_execution_target)
    {
        return;
    }

    logChangeExecutionTarget(_new_execution_target);
    execution_target = _new_execution_target;

    if (has_master_weights)
    {
        master_weights.setExecutionTarget(_new_execution_target);
        master_weights_gradient_tensor.setExecutionTarget(_new_execution_target);
    }
    biases.setExecutionTarget(_new_execution_target);
    biases_gradient_tensor.setExecutionTarget(_new_execution_target);
    effective_weights.setExecutionTarget(_new_execution_target);
    packed_weights.setExecutionTarget(_new_execution_target);
    gamma_tensor.setExecutionTarget(_new_execution_target);
    input_tensor.setExecutionTarget(_new_execution_target);
    output_tensor.setExecutionTarget(_new_execution_target);
    input_gradient_tensor.setExecutionTarget(_new_execution_target);

    if (_new_execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &context = Execution_Engine::getInstance().getContext();
        gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
        gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);

        uint32_t total_elements = static_cast<uint32_t>(input_dimension * output_dimension);
        uint32_t num_workgroups = std::min((total_elements + 255u) / 256u, 1024u);
        gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
    }
    else
    {
        gpu_gamma_buffer = nullptr;
        gpu_partial_sums = nullptr;
    }
}

void Ternary_Linear_Layer::quantizeEffectiveWeightsGpu()
{
    auto &engine = Execution_Engine::getInstance();
    auto &context = engine.getContext();
    uint32_t total_elements = static_cast<uint32_t>(input_dimension * output_dimension);
    if (total_elements == 0)
    {
        return;
    }

    if (!gpu_gamma_buffer)
    {
        gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
        gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
    }
    uint32_t num_workgroups = std::min((total_elements + 255u) / 256u, 1024u);
    if (!gpu_partial_sums || gpu_partial_sums->getSize() < num_workgroups)
    {
        gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
    }

    const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*master_weights.getImplementation());
    auto w_buf = w_gpu.getStorageBuffer();

    // Pass 1: Absmean reduction
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

    // Pass 2: Absmean reduction
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

    // FakeQuant: FP32 [K, N] -> FP16 [K, N]
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
        node.workgroup_count_x = (total_elements + 255u) / 256u;
        node.is_barrier_required_after = true;
        engine.getCurrentGraph().addNode(std::move(node));
    }
}

void Ternary_Linear_Layer::quantizeEffectiveWeightsCpu()
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
    float gamma = std::max(static_cast<float>(sum_abs / static_cast<double>(total)), 1e-5f);
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

void Ternary_Linear_Layer::packWeightsGpu()
{
    auto &engine = Execution_Engine::getInstance();
    auto &context = engine.getContext();
    uint32_t K = static_cast<uint32_t>(input_dimension);
    uint32_t N = static_cast<uint32_t>(output_dimension);
    uint32_t total_elements = K * N;
    if (total_elements == 0)
    {
        return;
    }

    if (!gpu_gamma_buffer)
    {
        gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
        gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
    }
    uint32_t num_workgroups = std::min((total_elements + 255u) / 256u, 1024u);
    if (!gpu_partial_sums || gpu_partial_sums->getSize() < num_workgroups)
    {
        gpu_partial_sums = std::make_shared<gpu::vector>(context, num_workgroups, Data_Type::FLOAT32);
    }

    const auto &w_gpu = Gpu_Tensor_Impl::castToGpu(*master_weights.getImplementation());
    auto w_buf = w_gpu.getStorageBuffer();

    // Pass 1: Absmean reduction
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

    // Pass 2: Absmean reduction
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

    // Allocate packed_weights: Logic shape [N, K], TERNARY_INT2
    uint32_t words_per_row = (K + 15u) / 16u;
    packed_weights = Tensor(Shape{ N, K }, Data_Type::TERNARY_INT2, Execution_Target::VULKAN_GPU);
    const auto &packed_gpu = Gpu_Tensor_Impl::castToGpu(*packed_weights.getImplementation());
    auto packed_buf = packed_gpu.getStorageBuffer();

    {
        Compute_Node node;
        node.pipeline_id = Compute_Pipeline::QUANTIZE_PACK_TERNARY;
        node.buffers = { w_buf, gpu_gamma_buffer, packed_buf };
        struct PushConsts { uint32_t K; uint32_t N; uint32_t words_per_row; } pcs{ K, N, words_per_row };
        const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
        node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
        node.workgroup_count_x = (words_per_row + 15u) / 16u;
        node.workgroup_count_y = (N + 15u) / 16u;
        node.is_barrier_required_after = true;
        engine.getCurrentGraph().addNode(std::move(node));
    }
}

void Ternary_Linear_Layer::packWeightsCpu()
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
    float gamma = std::max(static_cast<float>(sum_abs / static_cast<double>(total)), 1e-5f);
    float inv_gamma = 1.0f / gamma;

    std::vector<float> gamma_vec = { gamma, inv_gamma };
    gamma_tensor = Tensor(Shape{ 2 }, std::move(gamma_vec), Execution_Target::CPU);

    uint32_t words_per_row = static_cast<uint32_t>((K + 15) / 16);
    uint32_t total_words = static_cast<uint32_t>(N * words_per_row);
    std::vector<uint32_t> packed_data(total_words, 0u);

    for (uint32_t n = 0; n < N; ++n)
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

void Ternary_Linear_Layer::prepareInference()
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        packWeightsGpu();
    }
    else
    {
        packWeightsCpu();
    }
    // Release temporary effective_weights to free memory
    effective_weights = Tensor(0, 0, execution_target);
    has_packed_inference = true;
}

Tensor Ternary_Linear_Layer::forward(const Tensor &_input_tensor)
{
    size_t in_features = (_input_tensor.getShape().getRank() >= 2)
                         ? _input_tensor.getShape()[_input_tensor.getShape().getRank() - 1]
                         : _input_tensor.getColumns();
    if (in_features != input_dimension)
    {
        Logger::logMessage(Input_Format{"Ternary_Linear_Layer::forward: Input dimension mismatch (expected {}, got {})",
                                        input_dimension, in_features},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::FORWARD_EVALUATION);
        throw std::invalid_argument(std::format("Ternary_Linear_Layer input dimension mismatch (expected {}, got {})", input_dimension, in_features));
    }

    cache_orig_shape = _input_tensor.getShape();
    cache_is_3d = (cache_orig_shape.getRank() == 3);

    Tensor eff_input = _input_tensor;
    if (cache_is_3d)
    {
        eff_input.reshape(Shape{ cache_orig_shape[0] * cache_orig_shape[1], input_dimension });
    }

    input_tensor = eff_input;
    size_t M = input_tensor.getRows();
    size_t K = input_dimension;
    size_t N = output_dimension;

    if (has_packed_inference)
    {
        // Pure inference branch: T-GEMM
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            auto &engine = Execution_Engine::getInstance();
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

            uint32_t words_per_row = (static_cast<uint32_t>(K) + 15u) / 16u;
            Compute_Node node;
            node.pipeline_id = Compute_Pipeline::MATMUL_TERNARY_FP16;
            node.buffers = { buf_x, buf_w, buf_b, gpu_gamma_buffer, buf_y };
            struct PushConsts {
                uint32_t M; uint32_t K; uint32_t N; uint32_t words_per_row; uint32_t has_bias;
            } pcs{ static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), words_per_row, 1u };
            const auto *ptr = reinterpret_cast<const uint8_t *>(&pcs);
            node.push_constants_data.assign(ptr, ptr + sizeof(pcs));
            node.workgroup_count_x = (static_cast<uint32_t>(N) + 15u) / 16u;
            node.workgroup_count_y = (static_cast<uint32_t>(M) + 15u) / 16u;
            engine.getCurrentGraph().addNode(std::move(node));
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
        // Training / Fake-Quant branch
        if (execution_target == Execution_Target::VULKAN_GPU)
        {
            quantizeEffectiveWeightsGpu();
        }
        else
        {
            quantizeEffectiveWeightsCpu();
        }
        ops::linear::linearForward(input_tensor, effective_weights, biases, output_tensor);
    }

    if (cache_is_3d)
    {
        output_tensor.reshape(Shape{ cache_orig_shape[0], cache_orig_shape[1], output_dimension });
    }

    is_forward_completed = true;
    return output_tensor;
}

Tensor Ternary_Linear_Layer::forward(const Tensor &/*_batched_input*/, const std::vector<Tensor> &/*_batched_params*/) const
{
    throw std::logic_error("Ternary_Linear_Layer does not support population-batched forward");
}

Tensor Ternary_Linear_Layer::backward(const Tensor &_output_gradient)
{
    if (!has_master_weights)
    {
        Logger::logMessage("Ternary_Linear_Layer::backward: Layer loaded in inference-only mode without master weights",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::runtime_error("Ternary_Linear_Layer: Layer loaded in inference-only mode without master weights");
    }

    if (!is_forward_completed)
    {
        Logger::logMessage("Ternary_Linear_Layer::backward: Backward called before forward",
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::logic_error("Backward called before forward");
    }

    size_t out_features = (_output_gradient.getShape().getRank() >= 2)
                          ? _output_gradient.getShape()[_output_gradient.getShape().getRank() - 1]
                          : _output_gradient.getColumns();
    if (out_features != output_dimension)
    {
        Logger::logMessage(Input_Format{"Ternary_Linear_Layer::backward: Gradient output dimension mismatch (expected {}, got {})",
                                        output_dimension, out_features},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::DENSE_COMPUTE | Log_Feature::BACKWARD_PROPAGATION);
        throw std::invalid_argument(std::format("Ternary_Linear_Layer gradient output dimension mismatch (expected {}, got {})", output_dimension, out_features));
    }

    Tensor eff_grad = _output_gradient;
    if (cache_is_3d && eff_grad.getShape().getRank() == 3)
    {
        eff_grad.reshape(Shape{ cache_orig_shape[0] * cache_orig_shape[1], output_dimension });
    }

    // Straight-Through Estimator: Backpropagate through effective_weights, accumulate gradients directly into master_weights
    ops::linear::linearBackwardInput(eff_grad, effective_weights, input_gradient_tensor);
    ops::linear::linearBackwardWeightBias(input_tensor, eff_grad, master_weights_gradient_tensor, biases_gradient_tensor, is_accumulated);

    if (cache_is_3d)
    {
        input_gradient_tensor.reshape(cache_orig_shape);
    }

    return input_gradient_tensor;
}

std::unique_ptr<ILayer> Ternary_Linear_Layer::clone() const
{
    auto cloned = std::make_unique<Ternary_Linear_Layer>(input_dimension, output_dimension, execution_target, initialization_gain);
    cloned->setMixedPrecision(is_mixed_precision_enabled);
    if (has_master_weights)
    {
        cloned->setMasterWeights(master_weights);
    }
    cloned->setBiases(biases);
    if (has_packed_inference)
    {
        cloned->prepareInference();
    }
    return cloned;
}

void Ternary_Linear_Layer::invalidateWeightCache() noexcept
{
    if (has_master_weights)
    {
        master_weights.invalidateFp16Cache();
    }
    biases.invalidateFp16Cache();
    effective_weights.invalidateFp16Cache();
}

void Ternary_Linear_Layer::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    invalidateWeightCache();
}

void Ternary_Linear_Layer::setTrainingMode(bool _is_training)
{
    if (_is_training)
    {
        if (!has_master_weights)
        {
            throw std::runtime_error("Ternary_Linear_Layer: Layer loaded in inference-only mode without master weights");
        }
        has_packed_inference = false;
    }
}

void Ternary_Linear_Layer::resetGradient()
{
    is_forward_completed = false;
}

void Ternary_Linear_Layer::resetGradients()
{
    resetGradient();
    if (has_master_weights)
    {
        master_weights_gradient_tensor.zero();
    }
    biases_gradient_tensor.zero();
}

void Ternary_Linear_Layer::saveConfiguration(std::ofstream &_output_file_stream) const
{
    uint32_t input_dimension_value = static_cast<uint32_t>(input_dimension);
    uint32_t output_dimension_value = static_cast<uint32_t>(output_dimension);
    _output_file_stream.write(reinterpret_cast<const char *>(&input_dimension_value), sizeof(input_dimension_value));
    _output_file_stream.write(reinterpret_cast<const char *>(&output_dimension_value), sizeof(output_dimension_value));
}

void Ternary_Linear_Layer::saveInference(std::ofstream &_output_file_stream) const
{
    if (!has_packed_inference)
    {
        const_cast<Ternary_Linear_Layer *>(this)->prepareInference();
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
        Execution_Engine::getInstance().waitIdle();
    }

    float gamma_val = 1.0f;
    float inv_gamma_val = 1.0f;
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

    _output_file_stream.write(reinterpret_cast<const char *>(&gamma_val), sizeof(gamma_val));
    _output_file_stream.write(reinterpret_cast<const char *>(&inv_gamma_val), sizeof(inv_gamma_val));

    const auto &packed_data = packed_weights.getPackedData();
    uint32_t words_count = static_cast<uint32_t>(packed_data.size());
    _output_file_stream.write(reinterpret_cast<const char *>(&words_count), sizeof(words_count));
    _output_file_stream.write(reinterpret_cast<const char *>(packed_data.data()), static_cast<std::streamsize>(words_count * sizeof(uint32_t)));

    biases.saveTensor(_output_file_stream);
}

void Ternary_Linear_Layer::loadInference(std::ifstream &_input_file_stream)
{
    float gamma_val = 0.0f;
    float inv_gamma_val = 0.0f;
    _input_file_stream.read(reinterpret_cast<char *>(&gamma_val), sizeof(gamma_val));
    _input_file_stream.read(reinterpret_cast<char *>(&inv_gamma_val), sizeof(inv_gamma_val));

    uint32_t words_count = 0;
    _input_file_stream.read(reinterpret_cast<char *>(&words_count), sizeof(words_count));
    std::vector<uint32_t> packed_data(words_count);
    _input_file_stream.read(reinterpret_cast<char *>(packed_data.data()), static_cast<std::streamsize>(words_count * sizeof(uint32_t)));

    biases = Tensor::loadTensor(_input_file_stream, execution_target);
    output_dimension = biases.getColumns();

    packed_weights = Tensor(Shape{ output_dimension, input_dimension }, Data_Type::TERNARY_INT2, execution_target);
    packed_weights.uploadPackedData(packed_data);

    std::vector<float> gv = { gamma_val, inv_gamma_val };
    gamma_tensor = Tensor(Shape{ 2 }, std::move(gv), execution_target);

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        auto &context = Execution_Engine::getInstance().getContext();
        gpu_gamma_buffer = std::make_shared<gpu::vector>(context);
        gpu_gamma_buffer->allocateMemoryByBytes(sizeof(float) * 2, Data_Type::FLOAT32, 2);
        gpu_gamma_buffer->uploadData(std::vector<float>{ gamma_val, inv_gamma_val });
    }

    has_master_weights = false;
    has_packed_inference = true;
    master_weights = Tensor(0, 0, execution_target);
    effective_weights = Tensor(0, 0, execution_target);
    invalidateWeightCache();
}

void Ternary_Linear_Layer::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    master_weights.saveTensor(_output_file_stream);
    biases.saveTensor(_output_file_stream);
    master_weights_gradient_tensor.saveTensor(_output_file_stream);
    biases_gradient_tensor.saveTensor(_output_file_stream);
}

void Ternary_Linear_Layer::loadCheckpoint(std::ifstream &_input_file_stream)
{
    master_weights = Tensor::loadTensor(_input_file_stream, execution_target);
    biases = Tensor::loadTensor(_input_file_stream, execution_target);
    master_weights_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
    biases_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
    input_dimension = master_weights.getRows();
    output_dimension = master_weights.getColumns();
    has_master_weights = true;
    has_packed_inference = false;
    invalidateWeightCache();
}

std::function<float(std::mt19937 &)> Ternary_Linear_Layer::getPopulationParameterInitializer(size_t param_index) const
{
    if (param_index == 0)
    {
        float standard_deviation = (input_dimension > 0) ? std::sqrt(initialization_gain / static_cast<float>(input_dimension)) : 0.0f;
        return [standard_deviation](std::mt19937 &generator)
        {
            std::normal_distribution<float> distribution(0.0f, standard_deviation);
            return distribution(generator);
        };
    }
    return [](std::mt19937 &) { return 0.0f; };
}

std::vector<float> Ternary_Linear_Layer::getPopulationParameter(size_t param_index) const
{
    if (param_index == 0)
    {
        return master_weights.getData();
    }
    if (param_index == 1)
    {
        return biases.getData();
    }
    throw std::out_of_range("Ternary_Linear_Layer::getPopulationParameter: Parameter index out of range");
}

void Ternary_Linear_Layer::setPopulationParameter(size_t param_index, std::vector<float> flat_data)
{
    if (param_index == 0)
    {
        if (flat_data.size() != input_dimension * output_dimension)
        {
            throw std::invalid_argument("Ternary_Linear_Layer::setPopulationParameter: Weight size mismatch");
        }
        master_weights = Tensor(input_dimension, output_dimension, std::move(flat_data), execution_target);
        has_master_weights = true;
        has_packed_inference = false;
        invalidateWeightCache();
    }
    else if (param_index == 1)
    {
        if (flat_data.size() != output_dimension)
        {
            throw std::invalid_argument("Ternary_Linear_Layer::setPopulationParameter: Bias size mismatch");
        }
        biases = Tensor(1, output_dimension, std::move(flat_data), execution_target);
        invalidateWeightCache();
    }
    else
    {
        throw std::out_of_range("Ternary_Linear_Layer::setPopulationParameter: Parameter index out of range");
    }
}

std::vector<std::pair<Tensor *, Tensor *>> Ternary_Linear_Layer::getParametersAndGradients()
{
    if (!has_master_weights)
    {
        return {};
    }
    return { { &master_weights, &master_weights_gradient_tensor }, { &biases, &biases_gradient_tensor } };
}

void Ternary_Linear_Layer::setMasterWeights(const Tensor &_new_weights)
{
    if (_new_weights.getRows() != input_dimension || _new_weights.getColumns() != output_dimension)
    {
        throw std::invalid_argument("Ternary_Linear_Layer::setMasterWeights: Dimension size of weight must match");
    }
    master_weights = _new_weights;
    has_master_weights = true;
    has_packed_inference = false;
    invalidateWeightCache();
}

void Ternary_Linear_Layer::setBiases(const Tensor &_new_biases)
{
    if (_new_biases.getRows() != 1 || _new_biases.getColumns() != output_dimension)
    {
        throw std::invalid_argument("Ternary_Linear_Layer::setBiases: Dimension size of bias must match");
    }
    biases = _new_biases;
    invalidateWeightCache();
}

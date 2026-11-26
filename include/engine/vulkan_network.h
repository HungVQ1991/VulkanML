#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <mutex>
#include <vulkan/vulkan.h>
#include "helper/magic_enum.hpp"

class Vulkan_Context;

constexpr uint32_t DESCRIPTOR_BINDINGS_COUNT = 32;

enum Compute_Pipeline
{
    ADD,
    SUB,
    MATMUL,
    MATMUL_ADD,
    MUL_SCALAR,
    HADAMARD_MUL,
    HADAMARD_DIV,
    TRANSPOSE,
    RELU,
    RELU_BACKWARD,
    LEAKY_RELU_FORWARD,
    LEAKY_RELU_BACKWARD,
    SIGMOID_FORWARD,
    SIGMOID_BACKWARD,
    GELU,
    GELU_BACKWARD,
    SGD_UPDATE,
    ADAM_UPDATE,
    SOFTMAX,
    SOFTMAX_BACKWARD,
    LINEAR_FORWARD,
    LINEAR_BACKWARD_INPUT,
    LINEAR_BACKWARD_WEIGHT_BIAS,
    CONV2D_FORWARD_PASS,
    CONV2D_BACKWARD_PASS_INPUT_GRADIENT,
    CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT,
    MAXPOOL2D_FORWARD,
    MAXPOOL2D_BACKWARD,
    AVGPOOL2D_FORWARD,
    AVGPOOL2D_BACKWARD,
    GLOBAL_AVGPOOL_FORWARD,
    GLOBAL_AVGPOOL_BACKWARD,
    UPSAMPLE2D_FORWARD,
    UPSAMPLE2D_BACKWARD,
    BATCH_NORM_STATS_FORWARD,
    BATCH_NORM_TRANSFORM_FORWARD,
    BATCH_NORM_STATS_BACKWARD,
    BATCH_NORM_TRANSFORM_BACKWARD,
    BATCH_NORM2D_STATS_FORWARD,
    BATCH_NORM2D_TRANSFORM_FORWARD,
    BATCH_NORM2D_STATS_BACKWARD,
    BATCH_NORM2D_TRANSFORM_BACKWARD,
    INSTANCE_NORM2D_FORWARD,
    INSTANCE_NORM2D_BACKWARD,
    CCE_LOSS,
    MSE_LOSS,
    MAE_LOSS,
    BCE_LOSS,
    HUBER_LOSS,
    ASL_LOSS,
    ASL_GRADIENT,
    DICE_LOSS,
    DICE_GRADIENT,
    DICE_BCE_LOSS,
    DICE_BCE_GRADIENT,
    MATRIX_INVERSE,
    NORMALIZE,
    CONCATENATE_COLUMNS,
    CONCATENATE_ROWS,
    SPLIT_COLUMNS,
    SPLIT_ROWS,
    CONTIGUOUS,
    CONV2D_IM2COL_TRANSPOSED,
    CONV2D_BIAS_GRADIENT,
    RMSNORM,
    RMSNORM_BACKWARD,
    RMSNORM_GAMMA_GRAD,
    ROPE,
    SWIGLU,
    SWIGLU_BACKWARD,
    FUSED_SWIGLU_FORWARD,
    FUSED_SWIGLU_BACKWARD,
    EMBEDDING_FORWARD,
    EMBEDDING_BACKWARD,
    ATTENTION_DECODE,
    FUSED_CROSS_ENTROPY,
    GLOBAL_MAXPOOL_FORWARD,
    GLOBAL_MAXPOOL_BACKWARD,
    FLASH_ATTENTION_FALLBACK,
    FLASH_ATTENTION_BACKWARD_FALLBACK,
    COUNT_VALID_TARGETS,
    KV_CACHE_UPDATE,
    CAST_FP32_TO_FP16,
    CAST_FP16_TO_FP32,
    CONV2D_FORWARD_PASS_FP16,
    CONV2D_BACKWARD_PASS_INPUT_GRADIENT_FP16,
    CONV2D_BACKWARD_PASS_WEIGHT_BIAS_GRADIENT_FP16,
    MATMUL_FP16,
    MATMUL_ADD_FP16,
    LINEAR_BACKWARD_INPUT_FP16,
    LINEAR_BACKWARD_WEIGHT_BIAS_FP16,
    GELU_FP16,
    GELU_BACKWARD_FP16,
    RELU_FP16,
    RELU_BACKWARD_FP16,
    BATCH_NORM2D_STATS_FORWARD_FP16,
    BATCH_NORM2D_TRANSFORM_FORWARD_FP16,
    BATCH_NORM2D_STATS_BACKWARD_FP16,
    BATCH_NORM2D_TRANSFORM_BACKWARD_FP16,
    MAXPOOL2D_FORWARD_FP16,
    MAXPOOL2D_BACKWARD_FP16,
    AVGPOOL2D_FORWARD_FP16,
    AVGPOOL2D_BACKWARD_FP16,
    BATCH_NORM_STATS_FORWARD_FP16,
    BATCH_NORM_TRANSFORM_FORWARD_FP16,
    BATCH_NORM_STATS_BACKWARD_FP16,
    BATCH_NORM_TRANSFORM_BACKWARD_FP16,
    CONV2D_IM2COL_TRANSPOSED_FP16,
    CONV2D_BIAS_GRADIENT_FP16,
    CONV2D_WEIGHT_GRADIENT_COOPMAT_FP16,
    CONV2D_WEIGHT_GRADIENT_FP16,
    MATMUL_COOPMAT_FP16,
    MATMUL_ADD_COOPMAT_FP16,
    LINEAR_BACKWARD_INPUT_COOPMAT_FP16,
    LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16,
    CONV2D_IM2COL_FP16,
    RMSNORM_FP16,
    RMSNORM_BACKWARD_FP16,
    RMSNORM_GAMMA_GRAD_FP16,
    ROPE_FP16,
    SWIGLU_FP16,
    SWIGLU_BACKWARD_FP16,
    FUSED_SWIGLU_FORWARD_FP16,
    FUSED_SWIGLU_BACKWARD_FP16,
    FLASH_ATTENTION_FP16,
    FLASH_ATTENTION_BACKWARD_FP16,
    FLASH_ATTENTION_BACKWARD_DQ_FP16,
    FLASH_ATTENTION_BACKWARD_DKV_FP16,
    FLASH_ATTENTION_LSE_FP16,
    EMBEDDING_FORWARD_FP16,
    ADD_FP16,
    SUB_FP16,
    MUL_SCALAR_FP16,
    EMBEDDING_BACKWARD_FP16,
    FUSED_CROSS_ENTROPY_FP16,
    CONCATENATE_COLUMNS_FP16,
    CONCATENATE_ROWS_FP16,
    TRANSPOSE_FP16,
    CONTIGUOUS_FP16,
    SPLIT_COLUMNS_FP16,
    SPLIT_ROWS_FP16,
    HADAMARD_MUL_FP16,
    HADAMARD_DIV_FP16,
    SOFTMAX_FP16,
    SOFTMAX_BACKWARD_FP16,
    ATTENTION_DECODE_FP16,
    KV_CACHE_UPDATE_FP16,
    GLOBAL_AVGPOOL_FORWARD_FP16,
    GLOBAL_AVGPOOL_BACKWARD_FP16,
    GLOBAL_MAXPOOL_FORWARD_FP16,
    GLOBAL_MAXPOOL_BACKWARD_FP16,
    ADAM_UPDATE_FP16,
    LINEAR_BACKWARD_WEIGHT_ADAM_FP16,
    LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16,
    ASL_LOSS_FP16,
    ASL_GRADIENT_FP16,
    DICE_LOSS_FP16,
    DICE_GRADIENT_FP16,
    UPSAMPLE2D_FORWARD_FP16,
    UPSAMPLE2D_BACKWARD_FP16,
    UPSAMPLE2D_FORWARD_COOPMAT_FP16,
    UPSAMPLE2D_BACKWARD_COOPMAT_FP16,
    INSTANCE_NORM2D_FORWARD_FP16,
    INSTANCE_NORM2D_BACKWARD_FP16,
    INSTANCE_NORM2D_FORWARD_COOPMAT_FP16,
    INSTANCE_NORM2D_BACKWARD_COOPMAT_FP16,
    DICE_BCE_LOSS_FP16,
    DICE_BCE_GRADIENT_FP16,
    DICE_BCE_LOSS_COOPMAT_FP16,
    DICE_BCE_GRADIENT_COOPMAT_FP16,
    LEAKY_RELU_FORWARD_FP16,
    LEAKY_RELU_BACKWARD_FP16,
    LEAKY_RELU_FORWARD_COOPMAT_FP16,
    LEAKY_RELU_BACKWARD_COOPMAT_FP16,
    SIGMOID_FORWARD_FP16,
    SIGMOID_BACKWARD_FP16,
    SIGMOID_FORWARD_COOPMAT_FP16,
    SIGMOID_BACKWARD_COOPMAT_FP16,
    REDUCE_ABSMEAN_PASS1,
    REDUCE_ABSMEAN_PASS2,
    FAKEQUANT_TERNARY,
    QUANTIZE_PACK_TERNARY,
    MATMUL_TERNARY_FP16,
    DROPOUT_FORWARD,
    DROPOUT_FORWARD_FP16,
    FILL_FLOAT,
    FILL_FP16,
    MAE_GRADIENT,
    MAE_GRADIENT_FP16,
    BCE_GRADIENT,
    BCE_GRADIENT_FP16,
    HUBER_GRADIENT,
    HUBER_GRADIENT_FP16,
    SGD_UPDATE_FP16,
    SGD_MOMENTUM_UPDATE,
    SGD_MOMENTUM_UPDATE_FP16,
    SUM_REDUCE_FINAL,
    DICE_REDUCE_FINAL,
    DICE_BCE_REDUCE_FINAL,
    ADAM_UPDATE_MULTI4_FP16,
    QKV_SPLIT_ROPE,
    QKV_SPLIT_ROPE_FP16,
    QKV_PACK_UNROPE,
    QKV_PACK_UNROPE_FP16,
    RESIDUAL_SCALE_RMSNORM,
    RESIDUAL_SCALE_RMSNORM_FP16,
    RMSNORM_BACKWARD_RESIDUAL_FP16,
    ROPE_INDEXED,
    ROPE_INDEXED_FP16,
    FLASH_ATTENTION_MASKED,
    FLASH_ATTENTION_MASKED_BACKWARD,
    COMPUTE_PIPELINE_END
};

template <>
struct magic_enum::customize::enum_range<Compute_Pipeline> {
    static constexpr int min = 0;
    static constexpr int max = 256;
};

static_assert(static_cast<int>(Compute_Pipeline::COMPUTE_PIPELINE_END) <= magic_enum::customize::enum_range<Compute_Pipeline>::max,
              "Compute_Pipeline exceeds magic_enum::customize::enum_range::max");

class Pipeline_Cache_Manager;
class Vulkan_Network
{
private:
    const Vulkan_Context *context = nullptr;
    const Pipeline_Cache_Manager *shared_pipeline_cache_manager = nullptr;
    std::string pipeline_folder = "compute_shader/spv";

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    bool is_push_descriptor_active = false;

    mutable std::mutex pipeline_mutex;
    mutable std::array<std::atomic<VkPipeline>, Compute_Pipeline::COMPUTE_PIPELINE_END> pipelines{};
    mutable std::array<std::atomic<bool>, Compute_Pipeline::COMPUTE_PIPELINE_END> pipeline_compile_failed{};
    mutable std::array<std::atomic<uint32_t>, Compute_Pipeline::COMPUTE_PIPELINE_END> pipeline_push_constant_sizes{};
    mutable std::unordered_map<uint64_t, VkPipeline> specialized_pipelines;
    mutable std::unordered_set<uint64_t> specialized_compile_failed;

    std::vector<uint32_t> readSpirvFile(const std::string &_file_path) const;
    VkShaderModule createShaderModule(const std::vector<uint32_t> &_code_buffer) const;
    VkPipeline createComputePipeline(const std::string &_shader_path, bool is_coop = false, const uint32_t *_spec_constant_u32 = nullptr, uint32_t *_out_push_constant_bytes = nullptr) const;
    VkPipeline createPipelineForEnum(Compute_Pipeline _pipeline, const uint32_t *_spec_constant_u32 = nullptr) const;
    void cleanUp() noexcept;

public:
    Vulkan_Network(const Vulkan_Network &) = delete;
    Vulkan_Network &operator=(const Vulkan_Network &) = delete;

    Vulkan_Network(Vulkan_Network &&other) noexcept;
    Vulkan_Network &operator=(Vulkan_Network &&other) noexcept;

    explicit Vulkan_Network(const Vulkan_Context &_context, const std::string &_pipeline_folder);
    ~Vulkan_Network();

    static bool requiresFloat16(Compute_Pipeline _pipeline) noexcept;
    static bool requiresCooperativeMatrix(Compute_Pipeline _pipeline) noexcept;
    static uint32_t reflectSpirvPushConstantBytes(const std::vector<uint32_t> &_code_buffer) noexcept;
    uint32_t getPipelinePushConstantSize(Compute_Pipeline _pipeline) const noexcept;

    std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> getPipelines() const noexcept
    {
        std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> snapshot{};
        for (size_t i = 0; i < pipelines.size(); ++i)
        {
            snapshot[i] = pipelines[i].load(std::memory_order_acquire);
        }
        return snapshot;
    }
    const std::string &getPipelineFolder() const noexcept { return pipeline_folder; }
    VkPipeline getPipeline(Compute_Pipeline _pipeline) const;
    VkPipeline getSpecializedPipeline(Compute_Pipeline _pipeline, uint32_t _spec_constant_u32) const;
    VkPipelineLayout getPipelineLayout() const noexcept { return pipeline_layout; }
    VkDescriptorSetLayout getDescriptorSetLayout() const noexcept { return descriptor_set_layout; }
    VkDevice getDevice() const noexcept { return device; }
    bool isPushDescriptorActive() const noexcept { return is_push_descriptor_active; }

    bool hasPipeline(Compute_Pipeline _pipeline) const noexcept
    {
        size_t pipeline_index = static_cast<size_t>(_pipeline);
        return pipeline_index < pipelines.size() && pipelines[pipeline_index].load(std::memory_order_acquire) != VK_NULL_HANDLE;
    }

    void setPipelines(const std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> &_pipelines);
    void setPipelineFolder(const std::string &_pipeline_folder);
    void setPipelineCacheManager(const Pipeline_Cache_Manager *_manager) noexcept;
    void setPipelineLayout(VkPipelineLayout _pipeline_layout);
    void setDescriptorSetLayout(VkDescriptorSetLayout _layout);
    void setDevice(VkDevice _device);
};

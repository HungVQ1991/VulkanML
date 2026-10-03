#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

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
    GLOBAL_AVGPOOL_FORWARD,
    GLOBAL_AVGPOOL_BACKWARD,
    BATCH_NORM_STATS_FORWARD,
    BATCH_NORM_TRANSFORM_FORWARD,
    BATCH_NORM_STATS_BACKWARD,
    BATCH_NORM_TRANSFORM_BACKWARD,
    BATCH_NORM2D_STATS_FORWARD,
    BATCH_NORM2D_TRANSFORM_FORWARD,
    BATCH_NORM2D_STATS_BACKWARD,
    BATCH_NORM2D_TRANSFORM_BACKWARD,
    CCE_LOSS,
    MSE_LOSS,
    MAE_LOSS,
    BCE_LOSS,
    HUBER_LOSS,
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
    GLOBAL_AVGPOOL_FORWARD_FP16,
    GLOBAL_AVGPOOL_BACKWARD_FP16,
    ADAM_UPDATE_FP16,
    LINEAR_BACKWARD_WEIGHT_ADAM_FP16,
    LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16,
    COMPUTE_PIPELINE_END
};

class Vulkan_Network
{
private:
    const Vulkan_Context *context = nullptr;
    std::string pipeline_folder = "compute_shader/spv";

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> pipelines{};

    std::vector<char> readSpirvFile(const std::string &_file_path) const;
    VkShaderModule createShaderModule(const std::vector<char> &_code_buffer) const;
    VkPipeline createComputePipeline(const std::string &_shader_path, bool is_coop = false) const;
    void createAllPipelines(const std::string &_folder_path);
    void cleanUp() noexcept;

public:
    Vulkan_Network(const Vulkan_Network &) = delete;
    Vulkan_Network &operator=(const Vulkan_Network &) = delete;

    Vulkan_Network(Vulkan_Network &&other) noexcept = default;
    Vulkan_Network &operator=(Vulkan_Network &&other) noexcept = default;

    explicit Vulkan_Network(const Vulkan_Context &_context, const std::string &_pipeline_folder);
    ~Vulkan_Network();

    const std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> &getPipelines() const noexcept { return pipelines; }
    const std::string &getPipelineFolder() const noexcept { return pipeline_folder; }
    VkPipeline getPipeline(Compute_Pipeline _pipeline) const;
    VkPipelineLayout getPipelineLayout() const noexcept { return pipeline_layout; }
    VkDescriptorSetLayout getDescriptorSetLayout() const noexcept { return descriptor_set_layout; }
    VkDevice getDevice() const noexcept { return device; }

    bool hasPipeline(Compute_Pipeline _pipeline) const noexcept
    {
        size_t pipeline_index = static_cast<size_t>(_pipeline);
        return pipeline_index < pipelines.size() && pipelines[pipeline_index] != VK_NULL_HANDLE;
    }

    void setPipelines(const std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> &_pipelines) noexcept { pipelines = _pipelines; }
    void setPipelineFolder(const std::string &_pipeline_folder);
    void setPipelineLayout(VkPipelineLayout _pipeline_layout) noexcept { pipeline_layout = _pipeline_layout; }
    void setDescriptorSetLayout(VkDescriptorSetLayout _layout) noexcept { descriptor_set_layout = _layout; }
    void setDevice(VkDevice _device) noexcept { device = _device; }
};
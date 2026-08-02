#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cctype>

#include "vulkan_context.h"
#include "magic_enum.hpp"
#include "logger.h"

const int BINDING_SIZE = 8;

enum Compute_Pipeline
{
    ADD,
    SUB,
    MATMUL,
    MATMUL_ADD,
    MATMUL_TRANS_A,
    MATMUL_TRANS_B,
    MUL_SCALAR,
    HADAMARD_MUL,
    HADAMARD_DIV,
    TRANSPOSE,
    RELU,
    RELU_BACKWARD,
    GELU,
    GELU_BACKWARD,
    SGD_UPDATE,
    SOFTMAX,
    SOFTMAX_BACKWARD,
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
    COMPUTE_PIPELINE_END
};

class Vulkan_Network
{
private:

    std::string pipeline_folder = "compute_shader";

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    // VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> pipelines{};

    std::vector<char> readSpvFile(const std::string &file_path) const
    {
        std::ifstream file_stream(file_path, std::ios::ate | std::ios::binary);
        if (!file_stream.is_open())
        {
            Logger::logMessage("Vulkan_Network::readSpvFile: Failed to open SPIR-V file: " + file_path, LOG_ERROR);
            throw std::runtime_error("Failed to open SPIR-V file: " + file_path);
        }

        std::size_t file_size = static_cast<std::size_t>(file_stream.tellg());
        std::vector<char> buffer_data(file_size);
        file_stream.seekg(0);
        file_stream.read(buffer_data.data(), file_size);
        file_stream.close();

        return buffer_data;
    }

    VkShaderModule createShaderModule(const std::vector<char> &code_buffer) const
    {
        VkShaderModuleCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        create_info.codeSize = code_buffer.size();
        create_info.pCode = reinterpret_cast<const std::uint32_t *>(code_buffer.data());

        VkShaderModule shader_module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Network::createShaderModule: Failed to create shader module", LOG_ERROR);
            throw std::runtime_error("Failed to create shader module");
        }

        return shader_module;
    }

    VkPipeline createComputePipeline(const std::string &shader_path) const
    {
        auto shader_code = readSpvFile(shader_path);
        VkShaderModule shader_module = createShaderModule(shader_code);

        VkPipelineShaderStageCreateInfo shader_stage_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device, shader_module, nullptr);
            Logger::logMessage("Vulkan_Network::createComputePipeline: Failed to create compute pipeline for shader: " + shader_path, LOG_ERROR);
            throw std::runtime_error("Failed to create compute pipeline");
        }

        vkDestroyShaderModule(device, shader_module, nullptr);
        return pipeline;
    }

    void createAllPipelines(const std::string &folder_path)
    {
        for (size_t i = 0; i < pipelines.size(); ++i)
        {
            std::string shader_path = folder_path + "/" + std::string(magic_enum::enum_name(magic_enum::enum_cast<Compute_Pipeline>(i).value())) + ".spv";
            std::ranges::transform(shader_path, shader_path.begin(), [](unsigned char c)
                                   { return static_cast<char>(std::tolower(c)); });

            pipelines[i] = createComputePipeline(shader_path);
        }
    }

    void cleanup() noexcept
    {
        if (device == VK_NULL_HANDLE)
            return;
        for (VkPipeline pipeline : pipelines)
            if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);

        std::fill(pipelines.begin(), pipelines.end(), VK_NULL_HANDLE);

        // if (descriptor_pool != VK_NULL_HANDLE)
        // {
        //     vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        //     descriptor_pool = VK_NULL_HANDLE;
        // }

        if (pipeline_layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            pipeline_layout = VK_NULL_HANDLE;
        }

        if (descriptor_set_layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            descriptor_set_layout = VK_NULL_HANDLE;
        }
    }

public:
    Vulkan_Network(const Vulkan_Network &) = delete;
    Vulkan_Network &operator=(const Vulkan_Network &) = delete;

    Vulkan_Network(Vulkan_Network &&other) noexcept = default;
    Vulkan_Network &operator=(Vulkan_Network &&other) noexcept = default;

    explicit Vulkan_Network(const Vulkan_Context &context, const std::string &_pipeline_folder)
        : device(context.getDevice()), pipeline_folder(_pipeline_folder)
    {

        struct CleanupGuard
        {
            Vulkan_Network *owner;
            bool active = true;
            ~CleanupGuard()
            {
                if (active)
                    owner->cleanup();
            }
        } guard{this};

        VkDescriptorSetLayoutBinding bindings[BINDING_SIZE]{};
        for (std::uint32_t i = 0; i < BINDING_SIZE; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = BINDING_SIZE;
        layout_info.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create descriptor set layout", LOG_ERROR);
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = 128;

        VkPipelineLayoutCreateInfo pipeline_layout_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create pipeline layout", LOG_ERROR);
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 4000;

        VkDescriptorPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;

        // if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
        // {
        //     Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create descriptor pool", LOG_ERROR);
        //     throw std::runtime_error("Failed to create descriptor pool");
        // }

        createAllPipelines(pipeline_folder);

        guard.active = false;
    }

    ~Vulkan_Network() { cleanup(); }

    VkPipelineLayout getPipelineLayout() const { return pipeline_layout; }
    VkPipeline getPipeline(Compute_Pipeline pipeline) const { return pipelines[static_cast<std::size_t>(pipeline)]; }
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptor_set_layout; }
    // VkDescriptorPool getDescriptorPool() const { return descriptor_pool; }
    void changeComputeFolderDirectory(const std::string &new_folder) { pipeline_folder = new_folder; }
};
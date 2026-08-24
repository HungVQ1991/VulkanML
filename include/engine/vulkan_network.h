#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "vulkan_context.h"

constexpr std::uint32_t DESCRIPTOR_BINDINGS_COUNT = 32;

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
    MATRIX_INVERSE,
    NORMALIZE,
    COMPUTE_PIPELINE_END
};

class Vulkan_Network
{
private:
    std::string pipeline_folder = "compute_shader";

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> pipelines{};

    std::vector<char> readSpirvFile(const std::string &_file_path) const
    {
        Logger::logMessage(std::format("Vulkan_Network::readSpirvFile: Reading SPIR-V file: {}", _file_path),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);

        std::ifstream file_stream(_file_path, std::ios::ate | std::ios::binary);
        if (!file_stream.is_open())
        {
            Logger::logMessage(std::format("Vulkan_Network::readSpirvFile: Failed to open SPIR-V file: {}", _file_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("Failed to open SPIR-V file: " + _file_path);
        }

        std::size_t file_size = static_cast<std::size_t>(file_stream.tellg());
        std::vector<char> buffer_data(file_size);
        file_stream.seekg(0);
        file_stream.read(buffer_data.data(), static_cast<std::streamsize>(file_size));
        file_stream.close();

        return buffer_data;
    }

    VkShaderModule createShaderModule(const std::vector<char> &_code_buffer) const
    {
        Logger::logMessage(std::format("Vulkan_Network::createShaderModule: Creating shader module of size {} bytes", _code_buffer.size()),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);

        VkShaderModuleCreateInfo create_information{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = _code_buffer.size(),
            .pCode = reinterpret_cast<const std::uint32_t *>(_code_buffer.data())};

        VkShaderModule shader_module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &create_information, nullptr, &shader_module) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Network::createShaderModule: Failed to create shader module",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("Failed to create shader module");
        }

        return shader_module;
    }

    VkPipeline createComputePipeline(const std::string &_shader_path) const
    {
        Logger::logMessage(std::format("Vulkan_Network::createComputePipeline: Creating compute pipeline for shader: {}", _shader_path),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);

        std::vector<char> shader_code = readSpirvFile(_shader_path);
        VkShaderModule shader_module = createShaderModule(shader_code);

        VkPipelineShaderStageCreateInfo shader_stage_create_information{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main",
            .pSpecializationInfo = nullptr};

        VkComputePipelineCreateInfo pipeline_create_information{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = shader_stage_create_information,
            .layout = pipeline_layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1};

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_information, nullptr, &pipeline) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device, shader_module, nullptr);
            Logger::logMessage(std::format("Vulkan_Network::createComputePipeline: Failed to create compute pipeline for shader: {}", _shader_path),
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("Failed to create compute pipeline");
        }

        vkDestroyShaderModule(device, shader_module, nullptr);
        return pipeline;
    }

    void createAllPipelines(const std::string &_folder_path)
    {
        if (_folder_path.empty())
        {
            Logger::logMessage("Vulkan_Network::createAllPipelines: Shader folder path is empty",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
        }

        Logger::logMessage(std::format("Vulkan_Network::createAllPipelines: Creating all compute pipelines from folder: {}", _folder_path),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);

        for (std::size_t i = 0; i < pipelines.size(); ++i)
        {
            auto pipeline_enum_value = magic_enum::enum_cast<Compute_Pipeline>(i);
            if (!pipeline_enum_value.has_value())
            {
                Logger::logMessage(std::format("Vulkan_Network::createAllPipelines: Failed to cast enum at index {}", i),
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::SHADER_GENERATION);
                continue;
            }

            std::string shader_path = _folder_path + "/" + std::string(magic_enum::enum_name(pipeline_enum_value.value())) + ".spv";
            std::ranges::transform(shader_path, shader_path.begin(), [](unsigned char character)
                                   { return static_cast<char>(std::tolower(character)); });

            pipelines[i] = createComputePipeline(shader_path);
        }
    }

    void cleanUp() noexcept
    {
        if (device == VK_NULL_HANDLE)
        {
            return;
        }

        Logger::logMessage("Vulkan_Network::cleanUp: Cleaning up Vulkan network pipelines and layouts",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);

        for (VkPipeline &pipeline : pipelines)
        {
            if (pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
        }

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

    explicit Vulkan_Network(const Vulkan_Context &_context, const std::string &_pipeline_folder)
        : pipeline_folder(_pipeline_folder), device(_context.getDevice())
    {
        Logger::logMessage(std::format("Vulkan_Network::Vulkan_Network: Initializing Vulkan Network with shader folder: {}", _pipeline_folder),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::SHADER_GENERATION);

        struct Cleanup_Guard
        {
            Vulkan_Network *network_instance;
            bool is_active = true;

            explicit Cleanup_Guard(Vulkan_Network *_network_instance)
                : network_instance(_network_instance) {}

            ~Cleanup_Guard()
            {
                if (is_active)
                {
                    network_instance->cleanUp();
                }
            }
        } guard{this};

        VkDescriptorSetLayoutBinding layout_bindings[DESCRIPTOR_BINDINGS_COUNT]{};
        for (std::uint32_t i = 0; i < DESCRIPTOR_BINDINGS_COUNT; ++i)
        {
            layout_bindings[i] = VkDescriptorSetLayoutBinding{
                .binding = i,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr};
        }

        VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_information{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = DESCRIPTOR_BINDINGS_COUNT,
            .pBindings = layout_bindings};

        if (vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_information, nullptr, &descriptor_set_layout) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create descriptor set layout",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 128};

        VkPipelineLayoutCreateInfo pipeline_layout_create_information{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range};

        if (vkCreatePipelineLayout(device, &pipeline_layout_create_information, nullptr, &pipeline_layout) != VK_SUCCESS)
        {
            Logger::logMessage("Vulkan_Network::Vulkan_Network: Failed to create pipeline layout",
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
            throw std::runtime_error("Failed to create pipeline layout");
        }

        createAllPipelines(pipeline_folder);

        guard.is_active = false;
    }

    ~Vulkan_Network()
    {
        cleanUp();
    }

    [[nodiscard]] VkPipelineLayout getPipelineLayout() const noexcept
    {
        return pipeline_layout;
    }

    [[nodiscard]] VkDescriptorSetLayout getDescriptorSetLayout() const noexcept
    {
        return descriptor_set_layout;
    }

    [[nodiscard]] const std::string &getPipelineFolder() const noexcept
    {
        return pipeline_folder;
    }

    [[nodiscard]] VkDevice getDevice() const noexcept
    {
        return device;
    }

    [[nodiscard]] const std::array<VkPipeline, Compute_Pipeline::COMPUTE_PIPELINE_END> &getPipelines() const noexcept
    {
        return pipelines;
    }

    [[nodiscard]] bool hasPipeline(Compute_Pipeline _pipeline) const noexcept
    {
        std::size_t pipeline_index = static_cast<std::size_t>(_pipeline);
        return pipeline_index < pipelines.size() && pipelines[pipeline_index] != VK_NULL_HANDLE;
    }

    [[nodiscard]] VkPipeline getPipeline(Compute_Pipeline _pipeline) const
    {
        std::size_t pipeline_index = static_cast<std::size_t>(_pipeline);
        if (pipeline_index >= pipelines.size())
        {
            Logger::logMessage(std::format("Vulkan_Network::getPipeline: Pipeline index out of bounds ({})", pipeline_index),
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::DISPATCH_EXECUTION);
            return VK_NULL_HANDLE;
        }
        return pipelines[pipeline_index];
    }

    void setPipelineFolder(const std::string &_pipeline_folder)
    {
        if (_pipeline_folder.empty())
        {
            Logger::logMessage("Vulkan_Network::setPipelineFolder: Attempted to set empty folder path",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
        }
        Logger::logMessage(std::format("Vulkan_Network::setPipelineFolder: Changing compute shader folder to {}", _pipeline_folder),
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        pipeline_folder = _pipeline_folder;
    }
};
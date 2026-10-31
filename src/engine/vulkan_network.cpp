#include "engine/vulkan_network.h"
#include "engine/vulkan_context.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <ranges>
#include <stdexcept>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"

std::vector<char> Vulkan_Network::readSpirvFile(const std::string &_file_path) const
{
    Logger::logMessage(Input_Format{"Vulkan_Network::readSpirvFile: Reading SPIR-V file: {}", _file_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::ifstream file_stream(_file_path, std::ios::ate | std::ios::binary);
    if (!file_stream.is_open())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::readSpirvFile: Failed to open SPIR-V file: {}", _file_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to open SPIR-V file: " + _file_path);
    }

    size_t file_size = static_cast<size_t>(file_stream.tellg());
    std::vector<char> buffer_data(file_size);
    file_stream.seekg(0);
    file_stream.read(buffer_data.data(), static_cast<std::streamsize>(file_size));
    file_stream.close();

    return buffer_data;
}

VkShaderModule Vulkan_Network::createShaderModule(const std::vector<char> &_code_buffer) const
{
    Logger::logMessage(Input_Format{"Vulkan_Network::createShaderModule: Creating shader module of size {} bytes", _code_buffer.size()},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    VkShaderModuleCreateInfo create_information{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = _code_buffer.size(),
        .pCode = reinterpret_cast<const uint32_t *>(_code_buffer.data())};

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

VkPipeline Vulkan_Network::createComputePipeline(const std::string &_shader_path, bool is_coop) const
{
    Logger::logMessage(Input_Format{"Vulkan_Network::createComputePipeline: Creating compute pipeline for shader: {}", _shader_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::vector<char> shader_code = readSpirvFile(_shader_path);
    VkShaderModule shader_module = createShaderModule(shader_code);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_subgroup_size_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = 32};

    bool require_subgroup = is_coop && (context != nullptr && context->isSubgroupSizeControlEnabled());

    VkPipelineShaderStageCreateInfo shader_stage_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = require_subgroup ? &required_subgroup_size_create_information : nullptr,
        .flags = require_subgroup ? static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : VkPipelineShaderStageCreateFlags{0},
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
        Logger::logMessage(Input_Format{"Vulkan_Network::createComputePipeline: Failed to create compute pipeline for shader: {}", _shader_path},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to create compute pipeline");
    }

    vkDestroyShaderModule(device, shader_module, nullptr);
    return pipeline;
}

VkPipeline Vulkan_Network::createPipelineForEnum(Compute_Pipeline _pipeline) const
{
    bool is_fp16_pipeline = (_pipeline >= Compute_Pipeline::CAST_FP32_TO_FP16 &&
                             _pipeline < Compute_Pipeline::COMPUTE_PIPELINE_END &&
                             _pipeline != Compute_Pipeline::REDUCE_ABSMEAN_PASS1 &&
                             _pipeline != Compute_Pipeline::REDUCE_ABSMEAN_PASS2 &&
                             _pipeline != Compute_Pipeline::QUANTIZE_PACK_TERNARY);
    if (is_fp16_pipeline && context != nullptr && !context->isFloat16Enabled())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::createPipelineForEnum: Skipping float16 pipeline {} because device does not have float16 enabled", magic_enum::enum_name(_pipeline)},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return VK_NULL_HANDLE;
    }

    bool is_coop_pipeline = (_pipeline == Compute_Pipeline::MATMUL_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::MATMUL_ADD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::CONV2D_WEIGHT_GRADIENT_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::LINEAR_BACKWARD_INPUT_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_BIAS_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::LINEAR_BACKWARD_WEIGHT_ADAM_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::FLASH_ATTENTION_FP16 ||
                             _pipeline == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_FP16 ||
                             _pipeline == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DQ_FP16 ||
                             _pipeline == Compute_Pipeline::FLASH_ATTENTION_BACKWARD_DKV_FP16 ||
                             _pipeline == Compute_Pipeline::UPSAMPLE2D_FORWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::UPSAMPLE2D_BACKWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::INSTANCE_NORM2D_FORWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::INSTANCE_NORM2D_BACKWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::DICE_BCE_LOSS_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::DICE_BCE_GRADIENT_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::LEAKY_RELU_FORWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::LEAKY_RELU_BACKWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::SIGMOID_FORWARD_COOPMAT_FP16 ||
                             _pipeline == Compute_Pipeline::SIGMOID_BACKWARD_COOPMAT_FP16);
    if (is_coop_pipeline && context != nullptr && !context->isCooperativeMatrixEnabled())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::createPipelineForEnum: Skipping cooperative matrix pipeline {} because device does not have cooperative matrix enabled", magic_enum::enum_name(_pipeline)},
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return VK_NULL_HANDLE;
    }

    std::string shader_name = std::string(magic_enum::enum_name(_pipeline));
    std::string shader_path = pipeline_folder + "/" + shader_name + ".spv";
    std::ranges::transform(shader_path, shader_path.begin(), [](unsigned char character)
                           { return static_cast<char>(std::tolower(character)); });

    try
    {
        return createComputePipeline(shader_path, is_coop_pipeline);
    }
    catch (const std::exception &e)
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::createPipelineForEnum: Failed to compile pipeline {}: {}", shader_name, e.what()},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return VK_NULL_HANDLE;
    }
}

void Vulkan_Network::createAllPipelines(const std::string &_folder_path)
{
    if (!_folder_path.empty())
    {
        pipeline_folder = _folder_path;
    }

    Logger::logMessage(Input_Format{"Vulkan_Network::createAllPipelines: Pre-compiling compute pipelines from folder: {}", pipeline_folder},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        auto pipeline_enum_value = magic_enum::enum_cast<Compute_Pipeline>(i);
        if (pipeline_enum_value.has_value())
        {
            getPipeline(pipeline_enum_value.value());
        }
    }
}

void Vulkan_Network::cleanUp() noexcept
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

Vulkan_Network::Vulkan_Network(const Vulkan_Context &_context, const std::string &_pipeline_folder)
    : context(&_context), pipeline_folder(_pipeline_folder), device(_context.getDevice())
{
    Logger::logMessage(Input_Format{"Vulkan_Network::Vulkan_Network: Initializing Vulkan Network with shader folder: {}", _pipeline_folder},
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
    for (uint32_t i = 0; i < DESCRIPTOR_BINDINGS_COUNT; ++i)
    {
        layout_bindings[i] = VkDescriptorSetLayoutBinding{
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr};
    }

    VkDescriptorSetLayoutCreateFlags layout_flags = 0;
    if (context != nullptr && context->isPushDescriptorEnabled())
    {
        layout_flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        is_push_descriptor_active = true;
        Logger::logMessage("Vulkan_Network::Vulkan_Network: Configured descriptor set layout with VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR",
                           Log_Level::LOG_INFO,
                           true,
                           0,
                           Log_Feature::DEVICE_MANAGEMENT | Log_Feature::DISPATCH_EXECUTION);
    }

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_information{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = layout_flags,
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

    pipelines.fill(VK_NULL_HANDLE);

    guard.is_active = false;
}

Vulkan_Network::~Vulkan_Network()
{
    cleanUp();
}

VkPipeline Vulkan_Network::getPipeline(Compute_Pipeline _pipeline) const
{
    size_t pipeline_index = static_cast<size_t>(_pipeline);
    if (pipeline_index >= pipelines.size())
    {
        Logger::logMessage(Input_Format{"Vulkan_Network::getPipeline: Pipeline index out of bounds ({})", pipeline_index},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::DISPATCH_EXECUTION);
        return VK_NULL_HANDLE;
    }

    // Fast path: already compiled
    VkPipeline cached_pipeline = pipelines[pipeline_index];
    if (cached_pipeline != VK_NULL_HANDLE)
    {
        return cached_pipeline;
    }

    // Slow path: thread-safe lazy compilation with double-checked locking
    if (!pipeline_mutex)
    {
        pipeline_mutex = std::make_unique<std::mutex>();
    }
    std::lock_guard<std::mutex> lock(*pipeline_mutex);

    if (pipelines[pipeline_index] != VK_NULL_HANDLE)
    {
        return pipelines[pipeline_index];
    }

    VkPipeline new_pipeline = createPipelineForEnum(_pipeline);
    pipelines[pipeline_index] = new_pipeline;
    return new_pipeline;
}

void Vulkan_Network::setPipelineFolder(const std::string &_pipeline_folder)
{
    if (_pipeline_folder.empty())
    {
        Logger::logMessage("Vulkan_Network::setPipelineFolder: Attempted to set empty folder path",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
    Logger::logMessage(Input_Format{"Vulkan_Network::setPipelineFolder: Changing compute shader folder to {}", _pipeline_folder},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);
    pipeline_folder = _pipeline_folder;
}

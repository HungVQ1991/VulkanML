#include "engine/pipeline_cache_manager.h"

#include <format>
#include <fstream>
#include <stdexcept>

#include "engine/vulkan_context.h"
#include "helper/logger.h"

Pipeline_Cache_Manager::Pipeline_Cache_Manager(const Vulkan_Context &_context, VkPipelineLayout _pipeline_layout)
    : context(_context), pipeline_layout(_pipeline_layout)
{
    initializePipelineCache();
}

Pipeline_Cache_Manager::~Pipeline_Cache_Manager()
{
    savePipelineCache();

    VkDevice device = context.getDevice();
    if (pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    }

    for (auto &pipeline_pair : cached_pipelines)
    {
        if (pipeline_pair.second != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline_pair.second, nullptr);
        }
    }
}

void Pipeline_Cache_Manager::initializePipelineCache(const std::string &_cache_file_path)
{
    cache_file_path = _cache_file_path;
    Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::initializePipelineCache: Initializing cache from file '{}'", cache_file_path},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::filesystem::path path(cache_file_path);
    if (path.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    std::vector<char> cache_data;
    if (std::ifstream cache_file_stream(cache_file_path, std::ios::binary | std::ios::ate); cache_file_stream.is_open())
    {
        std::streamsize file_size = cache_file_stream.tellg();
        cache_file_stream.seekg(0, std::ios::beg);
        cache_data.resize(static_cast<size_t>(file_size));
        if (cache_file_stream.read(cache_data.data(), file_size))
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::initializePipelineCache: Loaded {} bytes from cache file", file_size},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
        }
        else
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::initializePipelineCache: Failed to read data from file '{}'", cache_file_path},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            cache_data.clear();
        }
    }
    else
    {
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::initializePipelineCache: Cache file '{}' not found, creating empty pipeline cache", cache_file_path},
                           Log_Level::LOG_WARNING,
                           false,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }

    VkPipelineCacheCreateInfo pipeline_cache_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .initialDataSize = cache_data.size(),
        .pInitialData = cache_data.empty() ? nullptr : cache_data.data()};

    if (vkCreatePipelineCache(context.getDevice(), &pipeline_cache_create_information, nullptr, &pipeline_cache) != VK_SUCCESS)
    {
        pipeline_cache = VK_NULL_HANDLE;
        Logger::logMessage("Pipeline_Cache_Manager::initializePipelineCache: Failed to create VkPipelineCache",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
    else
    {
        Logger::logMessage("Pipeline_Cache_Manager::initializePipelineCache: Successfully created VkPipelineCache",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
}

void Pipeline_Cache_Manager::savePipelineCache()
{
    std::lock_guard<std::mutex> lock(cache_mutex);

    if (pipeline_cache == VK_NULL_HANDLE || cache_file_path.empty())
    {
        Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Cannot save pipeline cache due to invalid handle or empty file path",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return;
    }

    VkDevice device = context.getDevice();
    size_t cache_data_size = 0;
    if (vkGetPipelineCacheData(device, pipeline_cache, &cache_data_size, nullptr) != VK_SUCCESS || cache_data_size == 0)
    {
        Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Failed to retrieve pipeline cache size or size is zero",
                           Log_Level::LOG_WARNING,
                           false,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return;
    }

    std::vector<char> cache_data(cache_data_size);
    if (vkGetPipelineCacheData(device, pipeline_cache, &cache_data_size, cache_data.data()) != VK_SUCCESS)
    {
        Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Failed to retrieve pipeline cache binary data",
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        return;
    }

    std::filesystem::path path(cache_file_path);
    if (path.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    if (std::ofstream cache_file_stream(cache_file_path, std::ios::binary); cache_file_stream.is_open())
    {
        cache_file_stream.write(cache_data.data(), static_cast<std::streamsize>(cache_data.size()));
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::savePipelineCache: Successfully saved {} bytes to '{}'", cache_data_size, cache_file_path},
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
    else
    {
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::savePipelineCache: Failed to open file '{}' for writing", cache_file_path},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
}

VkPipeline Pipeline_Cache_Manager::getOrCreatePipeline(const std::string &_glsl_code)
{
    size_t code_hash = std::hash<std::string>{}(_glsl_code);

    if (is_frozen.load(std::memory_order_acquire))
    {
        if (auto pipeline_iterator = cached_pipelines.find(code_hash); pipeline_iterator != cached_pipelines.end())
        {
            return pipeline_iterator->second;
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (auto pipeline_iterator = cached_pipelines.find(code_hash); pipeline_iterator != cached_pipelines.end())
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Cache hit for code hash {:x}", code_hash},
                               Log_Level::LOG_DEBUG,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            return pipeline_iterator->second;
        }
    }

    Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Cache miss for code hash {:x}, compiling GLSL to SPIR-V", code_hash},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    std::vector<uint32_t> spirv_code = shader_compiler.compileGlslToSpirv(_glsl_code, "fused_compute_shader");

    VkShaderModuleCreateInfo shader_module_create_information{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spirv_code.size() * sizeof(uint32_t),
        .pCode = spirv_code.data()};

    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.getDevice(), &shader_module_create_information, nullptr, &shader_module) != VK_SUCCESS)
    {
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Failed to create VkShaderModule for hash {:x}", code_hash},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to create shader module for fused pipeline");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_subgroup_size_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = 32};

    VkPipelineShaderStageCreateInfo shader_stage_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = is_coop ? &required_subgroup_size_create_information : nullptr,
        .flags = is_coop ? static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : VkPipelineShaderStageCreateFlags{0},
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = nullptr};

    VkComputePipelineCreateInfo compute_pipeline_create_information{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = shader_stage_create_information,
        .layout = pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1};

    VkPipeline new_pipeline = VK_NULL_HANDLE;
    VkResult creation_result = vkCreateComputePipelines(context.getDevice(), pipeline_cache, 1, &compute_pipeline_create_information, nullptr, &new_pipeline);

    vkDestroyShaderModule(context.getDevice(), shader_module, nullptr);

    if (creation_result != VK_SUCCESS || new_pipeline == VK_NULL_HANDLE)
    {
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Failed to create compute pipeline for hash {:x}", code_hash},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Failed to create compute pipeline for fused shader");
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto [pipeline_iterator, is_inserted] = cached_pipelines.try_emplace(code_hash, new_pipeline);
        if (!is_inserted)
        {
            vkDestroyPipeline(context.getDevice(), new_pipeline, nullptr);
            return pipeline_iterator->second;
        }
    }

    Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Successfully created and cached compute pipeline for hash {:x}", code_hash},
                       Log_Level::LOG_DEBUG,
                       true,
                       0,
                       Log_Feature::SHADER_GENERATION);

    return new_pipeline;
}

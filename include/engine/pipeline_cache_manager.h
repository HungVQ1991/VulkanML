#pragma once

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <string>
#include <functional>
#include <mutex>
#include <vector>
#include <fstream>
#include <format>

#include "vulkan_context.h"
#include "shader_compiler.h"
#include "helper/logger.h"

#ifndef ENABLE_PIPELINE_CACHE_DEBUG_LOGS
#define ENABLE_PIPELINE_CACHE_DEBUG_LOGS 0
#endif

#if ENABLE_PIPELINE_CACHE_DEBUG_LOGS
#define PIPELINE_CACHE_LOG_DEBUG(msg) Logger::logMessage(msg, LOG_DEBUG)
#else
#define PIPELINE_CACHE_LOG_DEBUG(msg) ((void)0)
#endif

class Pipeline_Cache_Manager
{
private:
    const Vulkan_Context &context;
    VkPipelineLayout pipeline_layout;

    Shader_Compiler compiler;

    std::unordered_map<std::size_t, VkPipeline> pipeline_map;
    std::mutex cache_mutex;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    std::string cache_file_path = "temp/pipeline_cache.bin";

public:
    Pipeline_Cache_Manager(const Vulkan_Context &ctx, VkPipelineLayout layout)
        : context(ctx), pipeline_layout(layout) {}

    ~Pipeline_Cache_Manager()
    {
        savePipelineCache();

        VkDevice device = context.getDevice();
        if (pipeline_cache != VK_NULL_HANDLE)
        {
            vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        }

        for (auto &pair : pipeline_map)
        {
            if (pair.second != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, pair.second, nullptr);
            }
        }
    }

    Pipeline_Cache_Manager(const Pipeline_Cache_Manager &) = delete;
    Pipeline_Cache_Manager &operator=(const Pipeline_Cache_Manager &) = delete;

    void initPipelineCache(const std::string &file_path = "temp/pipeline_cache.bin")
    {
        cache_file_path = file_path;
        PIPELINE_CACHE_LOG_DEBUG(std::format("Pipeline_Cache_Manager::initPipelineCache: Initializing cache from file '{}'", cache_file_path));

        std::vector<char> cache_data;
        if (std::ifstream file(cache_file_path, std::ios::binary | std::ios::ate); file.is_open())
        {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            cache_data.resize(static_cast<std::size_t>(size));
            if (file.read(cache_data.data(), size))
            {
                PIPELINE_CACHE_LOG_DEBUG(std::format("Pipeline_Cache_Manager::initPipelineCache: Loaded {} bytes from cache file", size));
            }
            else
            {
                Logger::logMessage(std::format("Pipeline_Cache_Manager::initPipelineCache: Failed to read data from file '{}'", cache_file_path), LOG_WARNING);
                cache_data.clear();
            }
        }
        else
        {
            Logger::logMessage(std::format("Pipeline_Cache_Manager::initPipelineCache: Cache file '{}' not found, creating empty pipeline cache", cache_file_path), LOG_WARNING);
        }

        VkPipelineCacheCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        create_info.initialDataSize = cache_data.size();
        create_info.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

        if (vkCreatePipelineCache(context.getDevice(), &create_info, nullptr, &pipeline_cache) != VK_SUCCESS)
        {
            pipeline_cache = VK_NULL_HANDLE;
            Logger::logMessage("Pipeline_Cache_Manager::initPipelineCache: Failed to create VkPipelineCache", LOG_WARNING);
        }
        else
        {
            PIPELINE_CACHE_LOG_DEBUG("Pipeline_Cache_Manager::initPipelineCache: Successfully created VkPipelineCache");
        }
    }

    void savePipelineCache()
    {
        std::lock_guard<std::mutex> lock(cache_mutex);

        if (pipeline_cache == VK_NULL_HANDLE || cache_file_path.empty())
        {
            Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Cannot save pipeline cache due to invalid handle or empty file path", LOG_WARNING);
            return;
        }

        VkDevice device = context.getDevice();
        std::size_t data_size = 0;
        if (vkGetPipelineCacheData(device, pipeline_cache, &data_size, nullptr) != VK_SUCCESS || data_size == 0)
        {
            Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Failed to retrieve pipeline cache size or size is zero", LOG_WARNING);
            return;
        }

        std::vector<char> cache_data(data_size);
        if (vkGetPipelineCacheData(device, pipeline_cache, &data_size, cache_data.data()) != VK_SUCCESS)
        {
            Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Failed to retrieve pipeline cache binary data", LOG_WARNING);
            return;
        }

        if (std::ofstream file(cache_file_path, std::ios::binary); file.is_open())
        {
            file.write(cache_data.data(), static_cast<std::streamsize>(cache_data.size()));
            PIPELINE_CACHE_LOG_DEBUG(std::format("Pipeline_Cache_Manager::savePipelineCache: Successfully saved {} bytes to '{}'", data_size, cache_file_path));
        }
        else
        {
            Logger::logMessage(std::format("Pipeline_Cache_Manager::savePipelineCache: Failed to open file '{}' for writing", cache_file_path), LOG_WARNING);
        }
    }

    VkPipeline getOrCreatePipeline(const std::string &glsl_code)
    {
        std::size_t code_hash = std::hash<std::string>{}(glsl_code);

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (auto it = pipeline_map.find(code_hash); it != pipeline_map.end())
            {
                PIPELINE_CACHE_LOG_DEBUG(std::format("Pipeline_Cache_Manager::getOrCreatePipeline: Cache hit for code hash {:x}", code_hash));
                return it->second;
            }
        }

        PIPELINE_CACHE_LOG_DEBUG(std::format("Pipeline_Cache_Manager::getOrCreatePipeline: Cache miss for code hash {:x}, compiling GLSL to SPIR-V", code_hash));

        std::vector<std::uint32_t> spv_code = compiler.compileGlslToSpv(glsl_code, "fused_compute_shader");

        VkShaderModuleCreateInfo shader_module_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_module_info.codeSize = spv_code.size() * sizeof(std::uint32_t);
        shader_module_info.pCode = spv_code.data();

        VkShaderModule shader_module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(context.getDevice(), &shader_module_info, nullptr, &shader_module) != VK_SUCCESS)
        {
            Logger::logMessage(std::format("Pipeline_Cache_Manager::getOrCreatePipeline: Failed to create VkShaderModule for hash {:x}", code_hash), LOG_ERROR, true);
            throw std::runtime_error("Failed to create shader module for fused pipeline");
        }

        VkComputePipelineCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader_module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pipeline_layout;

        VkPipeline new_pipeline = VK_NULL_HANDLE;
        VkResult result = vkCreateComputePipelines(context.getDevice(), pipeline_cache, 1, &pipeline_info, nullptr, &new_pipeline);

        vkDestroyShaderModule(context.getDevice(), shader_module, nullptr);

        if (result != VK_SUCCESS || new_pipeline == VK_NULL_HANDLE)
        {
            Logger::logMessage(std::format("Pipeline_Cache_Manager::getOrCreatePipeline: Failed to create compute pipeline for hash {:x}", code_hash), LOG_ERROR, true);
            throw std::runtime_error("Failed to create compute pipeline for fused shader");
        }

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto [it, inserted] = pipeline_map.try_emplace(code_hash, new_pipeline);
            if (!inserted)
            {
                vkDestroyPipeline(context.getDevice(), new_pipeline, nullptr);
                return it->second;
            }
        }

        PIPELINE_CACHE_LOG_DEBUG(std::format("Pipeline_Cache_Manager::getOrCreatePipeline: Successfully created and cached compute pipeline for hash {:x}", code_hash));

        return new_pipeline;
    }
};
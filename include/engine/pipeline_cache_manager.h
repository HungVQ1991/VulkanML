#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

#include "shader_compiler.h"

class Vulkan_Context;

class Pipeline_Cache_Manager
{
private:
    const Vulkan_Context &context;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    Shader_Compiler shader_compiler;

    std::unordered_map<size_t, VkPipeline> cached_pipelines;
    mutable std::mutex cache_mutex;
    std::atomic<bool> is_frozen{false};
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    std::string cache_file_path = "temp/pipeline_cache.bin";

public:
    Pipeline_Cache_Manager(const Vulkan_Context &_context, VkPipelineLayout _pipeline_layout);
    ~Pipeline_Cache_Manager();

    Pipeline_Cache_Manager(const Pipeline_Cache_Manager &) = delete;
    Pipeline_Cache_Manager &operator=(const Pipeline_Cache_Manager &) = delete;

    Pipeline_Cache_Manager(Pipeline_Cache_Manager &&_other) noexcept = default;
    Pipeline_Cache_Manager &operator=(Pipeline_Cache_Manager &&_other) noexcept = default;

    void initializePipelineCache(const std::string &_cache_file_path = "temp/pipeline_cache.bin");
    void savePipelineCache();

    VkPipeline getOrCreatePipeline(const std::string &_glsl_code);

    const std::unordered_map<size_t, VkPipeline> &getCachedPipelines() const noexcept { return cached_pipelines; }
    const std::string &getCacheFilePath() const noexcept { return cache_file_path; }
    const Shader_Compiler &getShaderCompiler() const noexcept { return shader_compiler; }
    Shader_Compiler &getShaderCompiler() noexcept { return shader_compiler; }
    const Vulkan_Context &getContext() const noexcept { return context; }
    size_t getCachedPipelineCount() const noexcept
    {
        if (is_frozen.load(std::memory_order_relaxed))
        {
            return cached_pipelines.size();
        }
        std::lock_guard<std::mutex> lock(cache_mutex);
        return cached_pipelines.size();
    }
    VkPipelineLayout getPipelineLayout() const noexcept { return pipeline_layout; }
    VkPipelineCache getPipelineCache() const noexcept { return pipeline_cache; }
    bool hasPipeline(size_t _code_hash) const noexcept
    {
        if (is_frozen.load(std::memory_order_relaxed))
        {
            return cached_pipelines.contains(_code_hash);
        }
        std::lock_guard<std::mutex> lock(cache_mutex);
        return cached_pipelines.contains(_code_hash);
    }
    bool isFrozen() const noexcept { return is_frozen.load(std::memory_order_relaxed); }

    void setCachedPipelines(const std::unordered_map<size_t, VkPipeline> &_pipelines)
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cached_pipelines = _pipelines;
    }
    void setCacheFilePath(const std::string &_cache_file_path) { cache_file_path = _cache_file_path; }
    void setPipelineLayout(VkPipelineLayout _pipeline_layout) noexcept { pipeline_layout = _pipeline_layout; }
    void setPipelineCache(VkPipelineCache _pipeline_cache) noexcept { pipeline_cache = _pipeline_cache; }
    void setFrozen(bool _freeze) noexcept { is_frozen.store(_freeze, std::memory_order_release); }
    void freezeCache(bool _freeze = true) noexcept { is_frozen.store(_freeze, std::memory_order_release); }
};
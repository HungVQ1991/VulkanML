#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "shader_compiler.h"

class Vulkan_Context;

class Pipeline_Cache_Manager
{
private:
    const Vulkan_Context &context;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    Shader_Compiler shader_compiler;

    std::unordered_map<std::string, VkPipeline> cached_pipelines;
    mutable std::shared_mutex cache_mutex;
    std::atomic<bool> is_frozen{false};
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    std::string cache_file_path;
    std::string spirv_cache_dir;
    bool is_spirv_cache_dir_explicit = false;
    std::atomic<uint64_t> spirv_disk_hits{0};
    std::atomic<uint64_t> shaderc_compiles{0};

public:
    Pipeline_Cache_Manager(const Vulkan_Context &_context, VkPipelineLayout _pipeline_layout);
    ~Pipeline_Cache_Manager();

    Pipeline_Cache_Manager(const Pipeline_Cache_Manager &) = delete;
    Pipeline_Cache_Manager &operator=(const Pipeline_Cache_Manager &) = delete;

    Pipeline_Cache_Manager(Pipeline_Cache_Manager &&_other) = delete;
    Pipeline_Cache_Manager &operator=(Pipeline_Cache_Manager &&_other) = delete;

    static std::string buildDeviceKeyedCachePath(const Vulkan_Context &_context, const std::string &_base_path_or_dir = "");
    void initializePipelineCache(const std::string &_cache_file_path = "");
    void savePipelineCache();

    VkResult createComputePipelineWithCache(const VkComputePipelineCreateInfo &_create_info, VkPipeline *_out_pipeline) const;
    VkPipeline getOrCreatePipeline(const std::string &_glsl_code);
    VkPipeline getOrCreatePipeline(const std::string &_cache_key,
                                   const std::string &_glsl_code,
                                   const std::vector<uint32_t> &_spec_constants);

    void setSpirvCacheDir(const std::string &_dir);
    const std::string &getSpirvCacheDir() const noexcept { return spirv_cache_dir; }
    uint64_t getSpirvDiskHits() const noexcept { return spirv_disk_hits.load(std::memory_order_acquire); }
    uint64_t getShadercCompileCount() const noexcept { return shaderc_compiles.load(std::memory_order_acquire); }
    void resetCacheStats() noexcept
    {
        spirv_disk_hits.store(0, std::memory_order_release);
        shaderc_compiles.store(0, std::memory_order_release);
    }
    void clearMemoryPipelines();

    std::unordered_map<std::string, VkPipeline> getCachedPipelines() const
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return cached_pipelines;
    }
    const std::string &getCacheFilePath() const noexcept { return cache_file_path; }
    const Shader_Compiler &getShaderCompiler() const noexcept { return shader_compiler; }
    Shader_Compiler &getShaderCompiler() noexcept { return shader_compiler; }
    const Vulkan_Context &getContext() const noexcept { return context; }
    size_t getCachedPipelineCount() const
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return cached_pipelines.size();
    }
    VkPipelineLayout getPipelineLayout() const
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return pipeline_layout;
    }
    VkPipelineCache getPipelineCache() const
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return pipeline_cache;
    }
    bool hasPipeline(const std::string &_glsl_code) const
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return cached_pipelines.contains(_glsl_code);
    }
    bool isFrozen() const noexcept { return is_frozen.load(std::memory_order_acquire); }

    void setCachedPipelines(const std::unordered_map<std::string, VkPipeline> &_pipelines);
    void setCacheFilePath(const std::string &_cache_file_path) { cache_file_path = _cache_file_path; }
    void setPipelineLayout(VkPipelineLayout _pipeline_layout);
    void setPipelineCache(VkPipelineCache _pipeline_cache);
    void setFrozen(bool _freeze) noexcept { is_frozen.store(_freeze, std::memory_order_release); }
    void freezeCache(bool _freeze = true) noexcept { is_frozen.store(_freeze, std::memory_order_release); }
};

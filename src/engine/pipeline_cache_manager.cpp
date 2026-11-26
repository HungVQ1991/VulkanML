#include "engine/pipeline_cache_manager.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

#include "engine/vulkan_context.h"
#include "helper/logger.h"
#include "helper/resource_path.h"

namespace
{

bool endsWithBinIgnoreCase(const std::string &path_str)
{
    if (path_str.size() < 4)
    {
        return false;
    }
    const std::string suffix = path_str.substr(path_str.size() - 4);
    std::string lower;
    lower.reserve(4);
    for (unsigned char c : suffix)
    {
        lower.push_back(static_cast<char>(std::tolower(c)));
    }
    return lower == ".bin";
}

}

static uint32_t queryPhysicalDeviceApiVersion(const Vulkan_Context &_ctx) noexcept
{
    if (_ctx.getPhysicalDevice() == VK_NULL_HANDLE)
    {
        return 0;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(_ctx.getPhysicalDevice(), &props);
    return props.apiVersion;
}

std::string Pipeline_Cache_Manager::buildDeviceKeyedCachePath(const Vulkan_Context &_context, const std::string &_base_path_or_dir)
{
    namespace fs = std::filesystem;

    std::string trimmed = _base_path_or_dir;
    const bool explicit_auto = trimmed.empty() || trimmed == "auto";
    if (!explicit_auto && trimmed != "temp" && endsWithBinIgnoreCase(trimmed))
    {
        std::error_code ec;
        if (!fs::is_directory(fs::path(trimmed), ec))
        {
            return fs::path(trimmed).lexically_normal().generic_string();
        }
    }

    const std::string dir_hint = explicit_auto ? "temp" : trimmed;
    const std::string resolved_dir = Resource_Path::resolveWritableDirectory(
        dir_hint, explicit_auto ? "VULKANML_PIPELINE_CACHE_DIR" : nullptr, "temp");

    std::string filename = "pipeline_cache_cpu_fallback.bin";
    if (_context.getPhysicalDevice() != VK_NULL_HANDLE)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(_context.getPhysicalDevice(), &props);
        uint32_t uuid_prefix = 0;
        std::memcpy(&uuid_prefix, props.pipelineCacheUUID, sizeof(uuid_prefix));
        filename = std::format("pipeline_cache_{:04x}_{:04x}_d{:08x}_{:08x}.bin",
                               props.vendorID,
                               props.deviceID,
                               props.driverVersion,
                               uuid_prefix);
    }

    return (fs::path(resolved_dir) / filename).lexically_normal().generic_string();
}

Pipeline_Cache_Manager::Pipeline_Cache_Manager(const Vulkan_Context &_context, VkPipelineLayout _pipeline_layout)
    : context(_context), pipeline_layout(_pipeline_layout), shader_compiler(queryPhysicalDeviceApiVersion(_context))
{
    initializePipelineCache();
}

Pipeline_Cache_Manager::~Pipeline_Cache_Manager()
{
    if (pipeline_cache != VK_NULL_HANDLE && !cache_file_path.empty())
    {
        try
        {
            savePipelineCache();
        }
        catch (...)
        {
        }
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    VkDevice device = context.getDevice();
    if (pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        pipeline_cache = VK_NULL_HANDLE;
    }

    for (auto &pipeline_pair : cached_pipelines)
    {
        if (pipeline_pair.second != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline_pair.second, nullptr);
            pipeline_pair.second = VK_NULL_HANDLE;
        }
    }
    cached_pipelines.clear();
}

void Pipeline_Cache_Manager::initializePipelineCache(const std::string &_cache_file_path)
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);

    if (pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(context.getDevice(), pipeline_cache, nullptr);
        pipeline_cache = VK_NULL_HANDLE;
    }

    cache_file_path = buildDeviceKeyedCachePath(context, _cache_file_path);
    if (!is_spirv_cache_dir_explicit)
    {
        std::filesystem::path p(cache_file_path);
        std::filesystem::path parent = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
        spirv_cache_dir = (parent / "spirv_cache").lexically_normal().generic_string();
    }
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
    std::error_code status_ec;
    if (std::filesystem::is_regular_file(path, status_ec))
    {
        if (std::ifstream cache_file_stream(cache_file_path, std::ios::binary | std::ios::ate); cache_file_stream.is_open())
        {
            std::streampos end_pos = cache_file_stream.tellg();
            if (end_pos > 0)
            {
                std::streamsize file_size = static_cast<std::streamsize>(end_pos);
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

    if (!cache_data.empty())
    {
        bool header_valid = false;
        if (cache_data.size() >= sizeof(VkPipelineCacheHeaderVersionOne))
        {
            VkPipelineCacheHeaderVersionOne header{};
            std::memcpy(&header, cache_data.data(), sizeof(header));
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(context.getPhysicalDevice(), &props);
            header_valid = (header.headerSize >= sizeof(VkPipelineCacheHeaderVersionOne)) &&
                           (header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE) &&
                           (header.vendorID == props.vendorID) &&
                           (header.deviceID == props.deviceID) &&
                           (std::memcmp(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE) == 0);
        }
        if (!header_valid)
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::initializePipelineCache: Invalid or mismatched pipeline cache header in '{}', resetting cache", cache_file_path},
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            cache_data.clear();
            std::error_code rm_ec;
            std::filesystem::remove(path, rm_ec);
        }
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
        if (!cache_data.empty())
        {
            Logger::logMessage("Pipeline_Cache_Manager::initializePipelineCache: Failed to create VkPipelineCache from initial data, retrying with empty cache",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            std::error_code rm_ec;
            std::filesystem::remove(path, rm_ec);

            VkPipelineCacheCreateInfo empty_cache_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .initialDataSize = 0,
                .pInitialData = nullptr};
            if (vkCreatePipelineCache(context.getDevice(), &empty_cache_info, nullptr, &pipeline_cache) != VK_SUCCESS)
            {
                pipeline_cache = VK_NULL_HANDLE;
                Logger::logMessage("Pipeline_Cache_Manager::initializePipelineCache: Failed to create empty VkPipelineCache",
                                   Log_Level::LOG_WARNING,
                                   true,
                                   0,
                                   Log_Feature::SHADER_GENERATION);
                return;
            }
        }
        else
        {
            Logger::logMessage("Pipeline_Cache_Manager::initializePipelineCache: Failed to create VkPipelineCache",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            return;
        }
    }

    if (!cache_data.empty())
    {
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::initializePipelineCache: Created VkPipelineCache ({} bytes loaded from '{}')",
                                        cache_data.size(), cache_file_path},
                           Log_Level::LOG_INFO,
                           false,
                           0,
                           Log_Feature::PIPELINE_CACHE | Log_Feature::SHADER_GENERATION);
    }
    else
    {
        Logger::logMessage("Pipeline_Cache_Manager::initializePipelineCache: Created empty VkPipelineCache",
                           Log_Level::LOG_DEBUG,
                           true,
                           0,
                           Log_Feature::PIPELINE_CACHE | Log_Feature::SHADER_GENERATION);
    }
}

void Pipeline_Cache_Manager::savePipelineCache()
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);

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
    std::vector<char> cache_data;
    VkResult get_res = VK_INCOMPLETE;
    for (int attempt = 0; attempt < 4; ++attempt)
    {
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

        cache_data.resize(cache_data_size);
        get_res = vkGetPipelineCacheData(device, pipeline_cache, &cache_data_size, cache_data.data());
        if (get_res == VK_SUCCESS)
        {
            cache_data.resize(cache_data_size);
            break;
        }
        if (get_res != VK_INCOMPLETE)
        {
            Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Failed to retrieve pipeline cache binary data",
                               Log_Level::LOG_WARNING,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            return;
        }
    }

    if (get_res != VK_SUCCESS || cache_data.empty())
    {
        Logger::logMessage("Pipeline_Cache_Manager::savePipelineCache: Incomplete pipeline cache retrieval",
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

    std::filesystem::path temp_path = path.string() + ".tmp";
    bool write_ok = false;
    {
        std::ofstream cache_file_stream(temp_path, std::ios::binary | std::ios::trunc);
        if (cache_file_stream.is_open())
        {
            cache_file_stream.write(cache_data.data(), static_cast<std::streamsize>(cache_data.size()));
            cache_file_stream.flush();
            write_ok = cache_file_stream.good();
        }
    }

    if (write_ok)
    {
        std::error_code ec;
        std::filesystem::rename(temp_path, path, ec);
        if (ec)
        {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp_path, path, ec);
        }
        if (!ec)
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::savePipelineCache: Saved {} bytes ({} pipelines) to '{}'",
                                            cache_data.size(), cached_pipelines.size(), cache_file_path},
                               Log_Level::LOG_INFO,
                               false,
                               0,
                               Log_Feature::PIPELINE_CACHE | Log_Feature::SHADER_GENERATION);
            return;
        }
        std::filesystem::remove(temp_path, ec);
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::savePipelineCache: Failed to finalize cache file '{}'", cache_file_path},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
    else
    {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::savePipelineCache: Failed to write file '{}'", cache_file_path},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
    }
}

VkResult Pipeline_Cache_Manager::createComputePipelineWithCache(const VkComputePipelineCreateInfo &_create_info, VkPipeline *_out_pipeline) const
{
    if (_out_pipeline == nullptr)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    return vkCreateComputePipelines(context.getDevice(), pipeline_cache, 1, &_create_info, nullptr, _out_pipeline);
}

VkPipeline Pipeline_Cache_Manager::getOrCreatePipeline(const std::string &_glsl_code)
{
    return getOrCreatePipeline(_glsl_code, _glsl_code, {});
}

void Pipeline_Cache_Manager::setSpirvCacheDir(const std::string &_dir)
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    if (_dir.empty())
    {
        is_spirv_cache_dir_explicit = false;
        std::filesystem::path p(cache_file_path);
        std::filesystem::path parent = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
        spirv_cache_dir = (parent / "spirv_cache").lexically_normal().generic_string();
        return;
    }
    is_spirv_cache_dir_explicit = true;
    spirv_cache_dir = std::filesystem::path(_dir).lexically_normal().generic_string();
    std::error_code ec;
    std::filesystem::create_directories(spirv_cache_dir, ec);
}

void Pipeline_Cache_Manager::clearMemoryPipelines()
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    VkDevice device = context.getDevice();
    if (device != VK_NULL_HANDLE && !cached_pipelines.empty())
    {
        vkDeviceWaitIdle(device);
        for (auto &kv : cached_pipelines)
        {
            if (kv.second != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, kv.second, nullptr);
            }
        }
    }
    cached_pipelines.clear();
}

VkPipeline Pipeline_Cache_Manager::getOrCreatePipeline(const std::string &_cache_key,
                                                       const std::string &_glsl_code,
                                                       const std::vector<uint32_t> &_spec_constants)
{
    auto computeHash128 = [](const std::string &src) -> std::string
    {
        uint64_t h1 = 14695981039346656037ULL;
        uint64_t h2 = 0x9e3779b97f4a7c15ULL;
        for (unsigned char c : src)
        {
            h1 ^= c;
            h1 *= 1099511628211ULL;
            h2 ^= static_cast<uint64_t>(c) + 0x9e3779b97f4a7c15ULL + (h2 << 6) + (h2 >> 2);
        }
        return std::format("{:016x}{:016x}", h1, h2);
    };

    const std::string glsl_hash128 = computeHash128(_glsl_code);
    std::string ram_key = _cache_key + "|" + glsl_hash128;
    if (!_spec_constants.empty())
    {
        ram_key += "|spec:";
        for (uint32_t v : _spec_constants)
        {
            ram_key += std::format("{:x}_", v);
        }
    }
    size_t code_hash = std::hash<std::string>{}(ram_key);

    VkPipelineLayout current_layout = VK_NULL_HANDLE;
    std::string active_spirv_dir;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        if (auto pipeline_iterator = cached_pipelines.find(ram_key); pipeline_iterator != cached_pipelines.end())
        {
            return pipeline_iterator->second;
        }
        if (is_frozen.load(std::memory_order_acquire))
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Cache is frozen, rejecting hash {:x}", code_hash},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("Pipeline_Cache_Manager::getOrCreatePipeline: Pipeline cache is frozen");
        }
        current_layout = pipeline_layout;
        active_spirv_dir = spirv_cache_dir;
    }

    if (current_layout == VK_NULL_HANDLE)
    {
        Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: pipeline_layout is VK_NULL_HANDLE for hash {:x}", code_hash},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::SHADER_GENERATION);
        throw std::runtime_error("Pipeline_Cache_Manager::getOrCreatePipeline: pipeline_layout is VK_NULL_HANDLE");
    }

    std::vector<uint32_t> spirv_code;
    bool loaded_from_disk = false;
    std::filesystem::path spirv_disk_path;
    if (!active_spirv_dir.empty())
    {
        spirv_disk_path = std::filesystem::path(active_spirv_dir) / (glsl_hash128 + "_v1.spv");
        std::error_code ec;
        if (std::filesystem::is_regular_file(spirv_disk_path, ec))
        {
            std::ifstream in(spirv_disk_path, std::ios::binary | std::ios::ate);
            if (in.is_open())
            {
                std::streampos sz = in.tellg();
                if (sz >= 20 && (static_cast<size_t>(sz) % sizeof(uint32_t)) == 0)
                {
                    std::vector<uint32_t> candidate(static_cast<size_t>(sz) / sizeof(uint32_t));
                    in.seekg(0, std::ios::beg);
                    if (in.read(reinterpret_cast<char *>(candidate.data()), sz) &&
                        !candidate.empty() && candidate[0] == 0x07230203u)
                    {
                        spirv_code = std::move(candidate);
                        loaded_from_disk = true;
                        spirv_disk_hits.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    if (!loaded_from_disk)
    {
        spirv_code = shader_compiler.compileGlslToSpirv(_glsl_code, std::format("fused_{:x}", code_hash));
        shaderc_compiles.fetch_add(1, std::memory_order_relaxed);
        if (spirv_code.empty())
        {
            Logger::logMessage(Input_Format{"Pipeline_Cache_Manager::getOrCreatePipeline: Empty SPIR-V generated for hash {:x}", code_hash},
                               Log_Level::LOG_ERROR,
                               true,
                               0,
                               Log_Feature::SHADER_GENERATION);
            throw std::runtime_error("Pipeline_Cache_Manager::getOrCreatePipeline: Empty SPIR-V output");
        }
        if (!active_spirv_dir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(active_spirv_dir, ec);
            static std::atomic<uint64_t> tmp_seq{0};
            std::filesystem::path tmp_path = spirv_disk_path;
            tmp_path += std::format(".tmp.{}", tmp_seq.fetch_add(1, std::memory_order_relaxed));
            bool wrote_ok = false;
            {
                std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                if (out.is_open())
                {
                    out.write(reinterpret_cast<const char *>(spirv_code.data()),
                              static_cast<std::streamsize>(spirv_code.size() * sizeof(uint32_t)));
                    wrote_ok = out.good();
                }
            }
            if (wrote_ok)
            {
                std::filesystem::rename(tmp_path, spirv_disk_path, ec);
                if (ec)
                {
                    std::filesystem::remove(tmp_path, ec);
                }
            }
            else
            {
                std::filesystem::remove(tmp_path, ec);
            }
        }
    }

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

    const bool uses_coop = (_glsl_code.find("GL_KHR_cooperative_matrix") != std::string::npos);
    const bool require_subgroup = uses_coop && context.canRequireSubgroupSize(32);
    const bool require_full_subgroups = require_subgroup && context.isComputeFullSubgroupsSupported();

    std::vector<VkSpecializationMapEntry> spec_entries;
    VkSpecializationInfo spec_info{};
    if (!_spec_constants.empty())
    {
        spec_entries.reserve(_spec_constants.size());
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(_spec_constants.size()); ++idx)
        {
            spec_entries.push_back(VkSpecializationMapEntry{
                .constantID = idx,
                .offset = static_cast<uint32_t>(idx * sizeof(uint32_t)),
                .size = sizeof(uint32_t)});
        }
        spec_info = VkSpecializationInfo{
            .mapEntryCount = static_cast<uint32_t>(spec_entries.size()),
            .pMapEntries = spec_entries.data(),
            .dataSize = _spec_constants.size() * sizeof(uint32_t),
            .pData = _spec_constants.data()};
    }

    VkPipelineShaderStageCreateInfo shader_stage_create_information{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = require_subgroup ? &required_subgroup_size_create_information : nullptr,
        .flags = require_full_subgroups ? static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : VkPipelineShaderStageCreateFlags{0},
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = _spec_constants.empty() ? nullptr : &spec_info};

    VkComputePipelineCreateInfo compute_pipeline_create_information{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = shader_stage_create_information,
        .layout = current_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1};

    VkPipeline new_pipeline = VK_NULL_HANDLE;
    VkResult creation_result = createComputePipelineWithCache(compute_pipeline_create_information, &new_pipeline);

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
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        if (is_frozen.load(std::memory_order_acquire))
        {
            vkDestroyPipeline(context.getDevice(), new_pipeline, nullptr);
            throw std::runtime_error("Pipeline_Cache_Manager::getOrCreatePipeline: Pipeline cache is frozen");
        }
        auto [pipeline_iterator, is_inserted] = cached_pipelines.try_emplace(ram_key, new_pipeline);
        if (!is_inserted)
        {
            vkDestroyPipeline(context.getDevice(), new_pipeline, nullptr);
            return pipeline_iterator->second;
        }
    }

    return new_pipeline;
}

void Pipeline_Cache_Manager::setCachedPipelines(const std::unordered_map<std::string, VkPipeline> &_pipelines)
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    VkDevice device = context.getDevice();
    if (device != VK_NULL_HANDLE)
    {
        bool has_obsolete = false;
        for (const auto &kv : cached_pipelines)
        {
            auto it = _pipelines.find(kv.first);
            if (kv.second != VK_NULL_HANDLE && (it == _pipelines.end() || it->second != kv.second))
            {
                has_obsolete = true;
                break;
            }
        }
        if (has_obsolete)
        {
            vkDeviceWaitIdle(device);
            for (const auto &kv : cached_pipelines)
            {
                auto it = _pipelines.find(kv.first);
                if (kv.second != VK_NULL_HANDLE && (it == _pipelines.end() || it->second != kv.second))
                {
                    vkDestroyPipeline(device, kv.second, nullptr);
                }
            }
        }
    }
    cached_pipelines = _pipelines;
}

void Pipeline_Cache_Manager::setPipelineLayout(VkPipelineLayout _pipeline_layout)
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    if (pipeline_layout == _pipeline_layout)
    {
        return;
    }
    VkDevice device = context.getDevice();
    if (device != VK_NULL_HANDLE && !cached_pipelines.empty())
    {
        vkDeviceWaitIdle(device);
        for (auto &kv : cached_pipelines)
        {
            if (kv.second != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, kv.second, nullptr);
            }
        }
        cached_pipelines.clear();
    }
    pipeline_layout = _pipeline_layout;
}

void Pipeline_Cache_Manager::setPipelineCache(VkPipelineCache _pipeline_cache)
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    if (pipeline_cache == _pipeline_cache)
    {
        return;
    }
    VkDevice device = context.getDevice();
    if (device != VK_NULL_HANDLE && pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    }
    pipeline_cache = _pipeline_cache;
}

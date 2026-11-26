#include "helper/runtime.h"

#include <utility>
#include <vulkan/vulkan.h>

#include "engine/execution_engine.h"
#include "engine/vulkan_context.h"
#include "helper/logger.h"
#include "helper/user_preferences.h"

Runtime::Runtime(const Runtime_Config &cfg)
    : config(cfg), owns_session(true)
{
    if (config.load_preferences)
    {
        auto &prefs = User_Preferences::getInstance();
        prefs.load(config.preferences_path);
        Logger::applyPreferences(prefs.getLoggingPreferences());
    }
    if (config.apply_env_logging)
    {
        Logger::configureFromEnvironment();
    }

    Execution_Engine &engine = Execution_Engine::getInstance();
    Vulkan_Context &ctx = engine.getContext();

    if (config.enable_pipeline_cache)
    {
        engine.getPipelineCacheManager().initializePipelineCache(config.pipeline_cache_path);
    }

    if (config.validate_hardware)
    {
        if (config.enable_fp16 && !ctx.isFloat16Supported())
        {
            Logger::logMessage("Runtime: FP16 requested but not supported on active Vulkan device",
                               Log_Level::LOG_WARNING, true, 0, Log_Feature::DEVICE_MANAGEMENT);
        }
        if (config.enable_cooperative_matrix && !engine.isCooperativeMatrixSupported())
        {
            Logger::logMessage("Runtime: Cooperative Matrix requested but not supported on active Vulkan device",
                               Log_Level::LOG_WARNING, true, 0, Log_Feature::DEVICE_MANAGEMENT);
        }
    }

    ctx.setFloat16Enabled(config.enable_fp16);
    engine.setCooperativeMatrixEnabled(config.enable_cooperative_matrix);
    engine.enableStaticGraph(config.enable_static_graph);
    engine.setSingleSubmitEnabled(config.enable_single_submit);
    engine.enableFusedGemmAdam(config.enable_fused_gemm_adam);
    engine.setGpuProfilingEnabled(config.enable_gpu_profiling);

    if (config.validate_hardware)
    {
        const Runtime_Info current_info = queryInfo();
        Logger::logMessage(
            Input_Format{"Runtime: Device='{}' | FP16={}/{} | Coop={}/{} | StaticGraph={} | SingleSubmit={} | FusedGemmAdam={}",
                         current_info.device_name,
                         current_info.fp16_enabled ? "on" : "off",
                         current_info.fp16_supported ? "yes" : "no",
                         current_info.cooperative_matrix_enabled ? "on" : "off",
                         current_info.cooperative_matrix_supported ? "yes" : "no",
                         current_info.static_graph_enabled ? "on" : "off",
                         current_info.single_submit_enabled ? "on" : "off",
                         current_info.fused_gemm_adam_enabled ? "on" : "off"},
            Log_Level::LOG_INFO, true, 0, Log_Feature::DEVICE_MANAGEMENT);
    }
}

Runtime::Runtime(Runtime &&other) noexcept
    : config(std::move(other.config)),
      owns_session(std::exchange(other.owns_session, false))
{
}

Runtime &Runtime::operator=(Runtime &&other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    if (owns_session && config.save_pipeline_cache_on_exit && config.enable_pipeline_cache)
    {
        try
        {
            waitIdle();
            savePipelineCache();
        }
        catch (...)
        {
        }
    }
    config = std::move(other.config);
    owns_session = std::exchange(other.owns_session, false);
    return *this;
}

Runtime::~Runtime()
{
    if (!owns_session)
    {
        return;
    }
    if (config.save_pipeline_cache_on_exit && config.enable_pipeline_cache)
    {
        try
        {
            waitIdle();
            savePipelineCache();
        }
        catch (...)
        {
        }
    }
}

Runtime_Info Runtime::info() const
{
    return queryInfo();
}

Runtime_Info Runtime::queryInfo()
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    Vulkan_Context &ctx = engine.getContext();

    Runtime_Info out;
    if (ctx.getPhysicalDevice() != VK_NULL_HANDLE)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.getPhysicalDevice(), &props);
        out.device_name = props.deviceName;
    }
    out.fp16_supported = ctx.isFloat16Supported();
    out.fp16_enabled = ctx.isFloat16Enabled();
    out.cooperative_matrix_supported = engine.isCooperativeMatrixSupported();
    out.cooperative_matrix_enabled = engine.isCooperativeMatrixEnabled();
    out.timestamp_query_supported = engine.isTimestampQuerySupported();
    out.static_graph_enabled = engine.isStaticGraphEnabled();
    out.single_submit_enabled = engine.isSingleSubmitEnabled();
    out.fused_gemm_adam_enabled = engine.isFusedGemmAdamEnabled();
    out.gpu_profiling_enabled = engine.isGpuProfilingEnabled();
    const Allocator_Stats vram_stats = ctx.getAllocator().getStats();
    out.allocated_vram_bytes = vram_stats.total_allocated_bytes;
    out.used_vram_bytes = vram_stats.total_used_bytes;
    out.free_vram_bytes = vram_stats.total_free_bytes;
    out.vram_chunk_count = vram_stats.active_chunk_count;
    out.active_vram_chunk_count = vram_stats.active_chunk_count;
    out.empty_vram_chunk_count = vram_stats.empty_chunk_count;
    out.cached_graph_template_count = engine.getCachedGraphTemplates().size();
    return out;
}

bool Runtime::isGpuAvailable() noexcept
{
    try
    {
        return !queryInfo().device_name.empty();
    }
    catch (...)
    {
        return false;
    }
}

void Runtime::initializePipelineCache(const std::string &path)
{
    Execution_Engine::getInstance().getPipelineCacheManager().initializePipelineCache(path);
}

void Runtime::savePipelineCache()
{
    Execution_Engine::getInstance().getPipelineCacheManager().savePipelineCache();
}

bool Runtime::isFloat16Supported()
{
    return Execution_Engine::getInstance().getContext().isFloat16Supported();
}

bool Runtime::isFloat16Enabled()
{
    return Execution_Engine::getInstance().getContext().isFloat16Enabled();
}

void Runtime::setFloat16Enabled(bool enable)
{
    Execution_Engine::getInstance().getContext().setFloat16Enabled(enable);
}

bool Runtime::isCooperativeMatrixSupported()
{
    return Execution_Engine::getInstance().isCooperativeMatrixSupported();
}

bool Runtime::isCooperativeMatrixEnabled()
{
    return Execution_Engine::getInstance().isCooperativeMatrixEnabled();
}

void Runtime::setCooperativeMatrixEnabled(bool enable)
{
    Execution_Engine::getInstance().setCooperativeMatrixEnabled(enable);
}

bool Runtime::isStaticGraphEnabled()
{
    return Execution_Engine::getInstance().isStaticGraphEnabled();
}

void Runtime::setStaticGraphEnabled(bool enable)
{
    Execution_Engine::getInstance().enableStaticGraph(enable);
}

bool Runtime::isSingleSubmitEnabled()
{
    return Execution_Engine::getInstance().isSingleSubmitEnabled();
}

void Runtime::setSingleSubmitEnabled(bool enable)
{
    Execution_Engine::getInstance().setSingleSubmitEnabled(enable);
}

bool Runtime::isFusedGemmAdamEnabled()
{
    return Execution_Engine::getInstance().isFusedGemmAdamEnabled();
}

void Runtime::setFusedGemmAdamEnabled(bool enable)
{
    Execution_Engine::getInstance().enableFusedGemmAdam(enable);
}

bool Runtime::isTimestampQuerySupported()
{
    return Execution_Engine::getInstance().isTimestampQuerySupported();
}

bool Runtime::isGpuProfilingEnabled()
{
    return Execution_Engine::getInstance().isGpuProfilingEnabled();
}

void Runtime::setGpuProfilingEnabled(bool enable)
{
    Execution_Engine::getInstance().setGpuProfilingEnabled(enable);
}

void Runtime::flushGpuProfileResults()
{
    Execution_Engine::getInstance().flushGpuProfileResults();
}

Gpu_Profile_Summary Runtime::getGpuProfileSummary()
{
    return Execution_Engine::getInstance().getGpuProfileSummary();
}

void Runtime::resetGpuProfileStats()
{
    Execution_Engine::getInstance().resetGpuProfileStats();
}

std::string Runtime::formatGpuProfileTable(size_t top_k, size_t batch_divisor)
{
    return Execution_Engine::getInstance().formatGpuProfileTable(top_k, batch_divisor);
}

void Runtime::logGpuProfileTable(size_t top_k, size_t batch_divisor, Log_Level level, bool print_to_console)
{
    Execution_Engine::getInstance().logGpuProfileTable(top_k, batch_divisor, level, print_to_console);
}

void Runtime::printGpuProfileTable(size_t top_k, size_t batch_divisor)
{
    Execution_Engine::getInstance().printGpuProfileTable(top_k, batch_divisor);
}

void Runtime::printGpuProfileTable(std::ostream &os, size_t top_k, size_t batch_divisor)
{
    Execution_Engine::getInstance().printGpuProfileTable(os, top_k, batch_divisor);
}

void Runtime::executeGraph()
{
    Execution_Engine::getInstance().executeGraph();
}

void Runtime::executePendingTransfers()
{
    Execution_Engine::getInstance().getContext().executePendingTransfers();
}

void Runtime::waitIdle()
{
    Execution_Engine::getInstance().waitIdle();
}

void Runtime::invalidateStaticGraph()
{
    Execution_Engine::getInstance().invalidateStaticGraph();
}

void Runtime::invalidateGraphCache()
{
    Execution_Engine::getInstance().invalidateGraphCache();
}

bool Runtime::isAsyncLossEnabled()
{
    return Execution_Engine::getInstance().isAsyncLossEnabled();
}

void Runtime::setAsyncLossEnabled(bool enable)
{
    Execution_Engine::getInstance().setAsyncLossEnabled(enable);
}

float Runtime::readPendingLoss(uint32_t frame_index)
{
    return Execution_Engine::getInstance().readPendingLoss(frame_index);
}

void Runtime::prepareCurrentFrame()
{
    Execution_Engine::getInstance().prepareCurrentFrame();
}

void Runtime::collectGarbage(bool reclaim_empty_vram)
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.waitIdle();
    Vulkan_Context &ctx = engine.getContext();
    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame)
    {
        ctx.cleanGarbage(frame);
    }
    if (reclaim_empty_vram)
    {
        ctx.getAllocator().reclaimEmptyChunks(false);
    }
}

uint64_t Runtime::reclaimEmptyVram(bool keep_one_warm_chunk_per_type)
{
    Execution_Engine &engine = Execution_Engine::getInstance();
    engine.waitIdle();
    return engine.getContext().getAllocator().reclaimEmptyChunks(keep_one_warm_chunk_per_type);
}

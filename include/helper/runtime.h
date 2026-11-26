#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

#include "helper/logger.h"
#include "helper/training_profiler.h"

struct Runtime_Config
{
    bool load_preferences = true;
    std::string preferences_path = "user_preferences.json";
    bool apply_env_logging = true;
    std::string pipeline_cache_path = "";
    bool enable_pipeline_cache = true;
    bool enable_fp16 = false;
    bool enable_cooperative_matrix = false;
    bool enable_static_graph = false;
    bool enable_single_submit = true;
    bool enable_fused_gemm_adam = false;
    bool enable_gpu_profiling = false;
    bool save_pipeline_cache_on_exit = true;
    bool validate_hardware = true;
};

struct Runtime_Info
{
    std::string device_name;
    bool fp16_supported = false;
    bool fp16_enabled = false;
    bool cooperative_matrix_supported = false;
    bool cooperative_matrix_enabled = false;
    bool timestamp_query_supported = false;
    bool static_graph_enabled = false;
    bool single_submit_enabled = true;
    bool fused_gemm_adam_enabled = false;
    bool gpu_profiling_enabled = false;
    uint64_t allocated_vram_bytes = 0;
    uint64_t used_vram_bytes = 0;
    uint64_t free_vram_bytes = 0;
    size_t vram_chunk_count = 0;
    size_t active_vram_chunk_count = 0;
    size_t empty_vram_chunk_count = 0;
    size_t cached_graph_template_count = 0;
};

class Runtime
{
private:
    Runtime_Config config;
    bool owns_session = true;

public:
    explicit Runtime(const Runtime_Config &config = {});
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    Runtime(Runtime &&other) noexcept;
    Runtime &operator=(Runtime &&other) noexcept;

    const Runtime_Config &getConfig() const noexcept { return config; }
    Runtime_Info info() const;

    static Runtime_Info queryInfo();
    static bool isGpuAvailable() noexcept;
    static void initializePipelineCache(const std::string &path = "");
    static void savePipelineCache();

    static bool isFloat16Supported();
    static bool isFloat16Enabled();
    static void setFloat16Enabled(bool enable);

    static bool isCooperativeMatrixSupported();
    static bool isCooperativeMatrixEnabled();
    static void setCooperativeMatrixEnabled(bool enable);

    static bool isStaticGraphEnabled();
    static void setStaticGraphEnabled(bool enable);

    static bool isSingleSubmitEnabled();
    static void setSingleSubmitEnabled(bool enable);

    static bool isFusedGemmAdamEnabled();
    static void setFusedGemmAdamEnabled(bool enable);

    static bool isTimestampQuerySupported();
    static bool isGpuProfilingEnabled();
    static void setGpuProfilingEnabled(bool enable);
    static void flushGpuProfileResults();
    static Gpu_Profile_Summary getGpuProfileSummary();
    static void resetGpuProfileStats();
    static std::string formatGpuProfileTable(size_t top_k = 20, size_t batch_divisor = 1);
    static void logGpuProfileTable(size_t top_k = 20, size_t batch_divisor = 1, Log_Level level = Log_Level::LOG_INFO, bool print_to_console = true);
    static void printGpuProfileTable(size_t top_k = 20, size_t batch_divisor = 1);
    static void printGpuProfileTable(std::ostream &os, size_t top_k = 20, size_t batch_divisor = 1);

    static void executeGraph();
    static void executePendingTransfers();
    static void waitIdle();
    static void invalidateStaticGraph();
    static void invalidateGraphCache();
    static bool isAsyncLossEnabled();
    static void setAsyncLossEnabled(bool enable);
    static float readPendingLoss(uint32_t frame_index = UINT32_MAX);
    static void prepareCurrentFrame();
    static void collectGarbage(bool reclaim_empty_vram = true);
    static uint64_t reclaimEmptyVram(bool keep_one_warm_chunk_per_type = false);
};

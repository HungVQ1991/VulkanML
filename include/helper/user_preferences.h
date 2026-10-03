#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct Device_Preferences
{
    std::string preferred_type = "INTEGRATED_GPU";
    uint32_t preferred_device_index = 0;
};

struct Execution_Preferences
{
    std::string default_execution_target = "VULKAN_GPU";
    bool enable_shader_fusion = true;
    bool enable_cooperative_matrix = true;
    uint32_t max_frames_in_flight = 2;
    std::string default_precision = "FLOAT32";
};

struct Memory_Preferences
{
    uint32_t staging_pool_initial_size_mb = 64;
    bool enable_timeline_semaphore = true;
};

struct Logging_Preferences
{
    bool enable_logging = true;
    std::string level = "LOG_INFO";
    std::vector<std::string> features = {
        "DEVICE_MANAGEMENT",
        "FP16_METRICS",
        "SYNCHRONIZATION"
    };
};

class User_Preferences
{
private:
    mutable std::mutex preferences_mutex;
    std::string config_file_path = "user_preferences.json";
    bool is_loaded = false;

    Device_Preferences device_prefs;
    Execution_Preferences execution_prefs;
    Memory_Preferences memory_prefs;
    Logging_Preferences logging_prefs;

    User_Preferences();

public:
    static User_Preferences &getInstance();

    User_Preferences(const User_Preferences &) = delete;
    User_Preferences &operator=(const User_Preferences &) = delete;

    bool load(const std::string &path = "user_preferences.json");
    bool save(const std::string &path = "user_preferences.json") const;

    const Device_Preferences &getDevicePreferences() const noexcept { return device_prefs; }
    const Execution_Preferences &getExecutionPreferences() const noexcept { return execution_prefs; }
    const Memory_Preferences &getMemoryPreferences() const noexcept { return memory_prefs; }
    const Logging_Preferences &getLoggingPreferences() const noexcept { return logging_prefs; }

    uint32_t getStagingPoolSizeMb() const noexcept { return memory_prefs.staging_pool_initial_size_mb; }
    bool isTimelineSemaphoreEnabled() const noexcept { return memory_prefs.enable_timeline_semaphore; }
    bool isShaderFusionEnabled() const noexcept { return execution_prefs.enable_shader_fusion; }
    bool isCooperativeMatrixEnabled() const noexcept { return execution_prefs.enable_cooperative_matrix; }
};

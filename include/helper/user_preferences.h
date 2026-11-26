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
    bool enable_push_descriptor = true;
};

struct Logging_Preferences
{
    bool enable_logging = true;
    std::string level = "LOG_INFO";
    bool console_enabled = true;
    std::string console_level = "LOG_DEBUG";
    bool file_enabled = false;
    std::string file_level = "LOG_DEBUG";
    std::string log_directory = "logs";
    std::string color_mode = "auto";
    bool quiet_mode = false;
    bool errors_to_stderr = false;
    bool filter_info_by_feature = false;
    bool include_timestamp_ms = false;
    bool include_thread_id = false;
    bool include_source_location = false;
    std::vector<std::string> features = {
        "DEVICE_MANAGEMENT",
        "FP16_METRICS",
        "TERNARY_METRICS",
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
    bool saveUnlocked(const std::string &path) const;

public:
    static User_Preferences &getInstance();

    User_Preferences(const User_Preferences &) = delete;
    User_Preferences &operator=(const User_Preferences &) = delete;

    bool load(const std::string &path = "user_preferences.json");
    bool save() const;
    bool save(const std::string &path) const;

    bool isLoaded() const
    {
        std::lock_guard lock(preferences_mutex);
        return is_loaded;
    }
    std::string getConfigFilePath() const
    {
        std::lock_guard lock(preferences_mutex);
        return config_file_path;
    }

    void setDevicePreferences(const Device_Preferences &prefs);
    void setExecutionPreferences(const Execution_Preferences &prefs);
    void setMemoryPreferences(const Memory_Preferences &prefs);
    void setLoggingPreferences(const Logging_Preferences &prefs);

    Device_Preferences getDevicePreferences() const
    {
        std::lock_guard lock(preferences_mutex);
        return device_prefs;
    }
    Execution_Preferences getExecutionPreferences() const
    {
        std::lock_guard lock(preferences_mutex);
        return execution_prefs;
    }
    Memory_Preferences getMemoryPreferences() const
    {
        std::lock_guard lock(preferences_mutex);
        return memory_prefs;
    }
    Logging_Preferences getLoggingPreferences() const
    {
        std::lock_guard lock(preferences_mutex);
        return logging_prefs;
    }

    uint32_t getStagingPoolSizeMb() const
    {
        std::lock_guard lock(preferences_mutex);
        return memory_prefs.staging_pool_initial_size_mb;
    }
    bool isTimelineSemaphoreEnabled() const
    {
        std::lock_guard lock(preferences_mutex);
        return memory_prefs.enable_timeline_semaphore;
    }
    bool isPushDescriptorEnabled() const
    {
        std::lock_guard lock(preferences_mutex);
        return memory_prefs.enable_push_descriptor;
    }
    bool isShaderFusionEnabled() const
    {
        std::lock_guard lock(preferences_mutex);
        return execution_prefs.enable_shader_fusion;
    }
    bool isCooperativeMatrixEnabled() const
    {
        std::lock_guard lock(preferences_mutex);
        return execution_prefs.enable_cooperative_matrix;
    }
};

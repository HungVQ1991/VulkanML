#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "json.hpp"

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

    User_Preferences()
    {
        load(config_file_path);
    }

public:
    static User_Preferences &getInstance()
    {
        static User_Preferences instance;
        return instance;
    }

    User_Preferences(const User_Preferences &) = delete;
    User_Preferences &operator=(const User_Preferences &) = delete;

    bool load(const std::string &path = "user_preferences.json")
    {
        std::lock_guard lock(preferences_mutex);
        config_file_path = path;

        if (!std::filesystem::exists(path))
        {
            save(path);
            is_loaded = true;
            return true;
        }

        try
        {
            std::ifstream file(path);
            if (!file.is_open())
            {
                return false;
            }

            nlohmann::json j;
            file >> j;

            if (j.contains("device"))
            {
                const auto &dev = j["device"];
                if (dev.contains("preferred_type")) device_prefs.preferred_type = dev["preferred_type"].get<std::string>();
                if (dev.contains("preferred_device_index")) device_prefs.preferred_device_index = dev["preferred_device_index"].get<uint32_t>();
            }

            if (j.contains("execution"))
            {
                const auto &exec = j["execution"];
                if (exec.contains("default_execution_target")) execution_prefs.default_execution_target = exec["default_execution_target"].get<std::string>();
                if (exec.contains("enable_shader_fusion")) execution_prefs.enable_shader_fusion = exec["enable_shader_fusion"].get<bool>();
                if (exec.contains("enable_cooperative_matrix")) execution_prefs.enable_cooperative_matrix = exec["enable_cooperative_matrix"].get<bool>();
                if (exec.contains("max_frames_in_flight")) execution_prefs.max_frames_in_flight = exec["max_frames_in_flight"].get<uint32_t>();
                if (exec.contains("default_precision")) execution_prefs.default_precision = exec["default_precision"].get<std::string>();
            }

            if (j.contains("memory"))
            {
                const auto &mem = j["memory"];
                if (mem.contains("staging_pool_initial_size_mb")) memory_prefs.staging_pool_initial_size_mb = mem["staging_pool_initial_size_mb"].get<uint32_t>();
                if (mem.contains("enable_timeline_semaphore")) memory_prefs.enable_timeline_semaphore = mem["enable_timeline_semaphore"].get<bool>();
            }

            if (j.contains("logging"))
            {
                const auto &log = j["logging"];
                if (log.contains("enable_logging")) logging_prefs.enable_logging = log["enable_logging"].get<bool>();
                if (log.contains("level")) logging_prefs.level = log["level"].get<std::string>();
                if (log.contains("features") && log["features"].is_array())
                {
                    logging_prefs.features.clear();
                    for (const auto &feat : log["features"])
                    {
                        logging_prefs.features.push_back(feat.get<std::string>());
                    }
                }
            }

            is_loaded = true;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "User_Preferences::load: Warning - failed to parse " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    bool save(const std::string &path = "user_preferences.json") const
    {
        try
        {
            nlohmann::json j;
            j["device"]["preferred_type"] = device_prefs.preferred_type;
            j["device"]["preferred_device_index"] = device_prefs.preferred_device_index;

            j["execution"]["default_execution_target"] = execution_prefs.default_execution_target;
            j["execution"]["enable_shader_fusion"] = execution_prefs.enable_shader_fusion;
            j["execution"]["enable_cooperative_matrix"] = execution_prefs.enable_cooperative_matrix;
            j["execution"]["max_frames_in_flight"] = execution_prefs.max_frames_in_flight;
            j["execution"]["default_precision"] = execution_prefs.default_precision;

            j["memory"]["staging_pool_initial_size_mb"] = memory_prefs.staging_pool_initial_size_mb;
            j["memory"]["enable_timeline_semaphore"] = memory_prefs.enable_timeline_semaphore;

            j["logging"]["enable_logging"] = logging_prefs.enable_logging;
            j["logging"]["level"] = logging_prefs.level;
            j["logging"]["features"] = logging_prefs.features;

            std::ofstream file(path);
            if (!file.is_open())
            {
                return false;
            }
            file << j.dump(2) << std::endl;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    const Device_Preferences &getDevicePreferences() const noexcept { return device_prefs; }
    const Execution_Preferences &getExecutionPreferences() const noexcept { return execution_prefs; }
    const Memory_Preferences &getMemoryPreferences() const noexcept { return memory_prefs; }
    const Logging_Preferences &getLoggingPreferences() const noexcept { return logging_prefs; }

    uint32_t getStagingPoolSizeMb() const noexcept { return memory_prefs.staging_pool_initial_size_mb; }
    bool isTimelineSemaphoreEnabled() const noexcept { return memory_prefs.enable_timeline_semaphore; }
    bool isShaderFusionEnabled() const noexcept { return execution_prefs.enable_shader_fusion; }
    bool isCooperativeMatrixEnabled() const noexcept { return execution_prefs.enable_cooperative_matrix; }
};

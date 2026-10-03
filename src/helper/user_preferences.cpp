#include "helper/user_preferences.h"
#include "helper/json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

User_Preferences::User_Preferences()
{
    load(config_file_path);
}

User_Preferences &User_Preferences::getInstance()
{
    static User_Preferences instance;
    return instance;
}

bool User_Preferences::load(const std::string &path)
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
            if (dev.contains("preferred_type")) dev.at("preferred_type").get_to(device_prefs.preferred_type);
            if (dev.contains("preferred_device_index")) dev.at("preferred_device_index").get_to(device_prefs.preferred_device_index);
        }

        if (j.contains("execution"))
        {
            const auto &exec = j["execution"];
            if (exec.contains("default_execution_target")) exec.at("default_execution_target").get_to(execution_prefs.default_execution_target);
            if (exec.contains("enable_shader_fusion")) exec.at("enable_shader_fusion").get_to(execution_prefs.enable_shader_fusion);
            if (exec.contains("enable_cooperative_matrix")) exec.at("enable_cooperative_matrix").get_to(execution_prefs.enable_cooperative_matrix);
            if (exec.contains("max_frames_in_flight")) exec.at("max_frames_in_flight").get_to(execution_prefs.max_frames_in_flight);
            if (exec.contains("default_precision")) exec.at("default_precision").get_to(execution_prefs.default_precision);
        }

        if (j.contains("memory"))
        {
            const auto &mem = j["memory"];
            if (mem.contains("staging_pool_initial_size_mb")) mem.at("staging_pool_initial_size_mb").get_to(memory_prefs.staging_pool_initial_size_mb);
            if (mem.contains("enable_timeline_semaphore")) mem.at("enable_timeline_semaphore").get_to(memory_prefs.enable_timeline_semaphore);
        }

        if (j.contains("logging"))
        {
            const auto &log = j["logging"];
            if (log.contains("enable_logging")) log.at("enable_logging").get_to(logging_prefs.enable_logging);
            if (log.contains("level")) log.at("level").get_to(logging_prefs.level);
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

bool User_Preferences::save(const std::string &path) const
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

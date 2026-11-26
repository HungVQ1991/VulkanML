#include "helper/user_preferences.h"
#include "helper/json.hpp"
#include "helper/logger.h"
#include "helper/resource_path.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

User_Preferences::User_Preferences()
{
    const std::string resolved = Resource_Path::resolveFile(
        config_file_path, "VULKANML_PREFERENCES_PATH", "user_preferences.json");
    load(resolved);
}

User_Preferences &User_Preferences::getInstance()
{
    static User_Preferences instance;
    return instance;
}

bool User_Preferences::load(const std::string &path)
{
    std::lock_guard lock(preferences_mutex);
    if (path.empty())
    {
        return false;
    }

    std::string effective_path = path;
    if (path == "user_preferences.json")
    {
        std::error_code local_ec;
        if (!std::filesystem::exists(path, local_ec))
        {
            const std::string resolved = Resource_Path::resolveFile(
                path, "VULKANML_PREFERENCES_PATH", "user_preferences.json");
            std::error_code resolved_ec;
            if (std::filesystem::is_regular_file(resolved, resolved_ec))
            {
                effective_path = resolved;
            }
        }
    }

    try
    {
        std::error_code ec;
        if (!std::filesystem::exists(effective_path, ec) || ec)
        {
            if (ec)
            {
                return false;
            }
            bool saved = saveUnlocked(effective_path);
            if (saved)
            {
                config_file_path = effective_path;
                is_loaded = true;
            }
            return saved;
        }

        if (!std::filesystem::is_regular_file(effective_path, ec) || ec)
        {
            return false;
        }

        std::ifstream file(effective_path);
        if (!file.is_open())
        {
            return false;
        }

        nlohmann::json j;
        file >> j;
        if (!j.is_object())
        {
            return false;
        }

        Device_Preferences tmp_dev{};
        Execution_Preferences tmp_exec{};
        Memory_Preferences tmp_mem{};
        Logging_Preferences tmp_log{};

        if (j.contains("device"))
        {
            const auto &dev = j["device"];
            if (!dev.is_object()) return false;
            if (dev.contains("preferred_type")) dev.at("preferred_type").get_to(tmp_dev.preferred_type);
            if (dev.contains("preferred_device_index")) dev.at("preferred_device_index").get_to(tmp_dev.preferred_device_index);
        }

        if (j.contains("execution"))
        {
            const auto &exec = j["execution"];
            if (!exec.is_object()) return false;
            if (exec.contains("default_execution_target")) exec.at("default_execution_target").get_to(tmp_exec.default_execution_target);
            if (exec.contains("enable_shader_fusion")) exec.at("enable_shader_fusion").get_to(tmp_exec.enable_shader_fusion);
            if (exec.contains("enable_cooperative_matrix")) exec.at("enable_cooperative_matrix").get_to(tmp_exec.enable_cooperative_matrix);
            if (exec.contains("max_frames_in_flight")) exec.at("max_frames_in_flight").get_to(tmp_exec.max_frames_in_flight);
            if (exec.contains("default_precision")) exec.at("default_precision").get_to(tmp_exec.default_precision);
        }

        if (j.contains("memory"))
        {
            const auto &mem = j["memory"];
            if (!mem.is_object()) return false;
            if (mem.contains("staging_pool_initial_size_mb")) mem.at("staging_pool_initial_size_mb").get_to(tmp_mem.staging_pool_initial_size_mb);
            if (mem.contains("enable_timeline_semaphore")) mem.at("enable_timeline_semaphore").get_to(tmp_mem.enable_timeline_semaphore);
            if (mem.contains("enable_push_descriptor")) mem.at("enable_push_descriptor").get_to(tmp_mem.enable_push_descriptor);
        }

        if (j.contains("logging"))
        {
            const auto &log = j["logging"];
            if (!log.is_object()) return false;
            if (log.contains("enable_logging")) log.at("enable_logging").get_to(tmp_log.enable_logging);
            if (log.contains("level")) log.at("level").get_to(tmp_log.level);
            if (log.contains("console_enabled")) log.at("console_enabled").get_to(tmp_log.console_enabled);
            if (log.contains("console_level")) log.at("console_level").get_to(tmp_log.console_level);
            if (log.contains("file_enabled")) log.at("file_enabled").get_to(tmp_log.file_enabled);
            if (log.contains("file_level")) log.at("file_level").get_to(tmp_log.file_level);
            if (log.contains("log_directory")) log.at("log_directory").get_to(tmp_log.log_directory);
            if (log.contains("color_mode")) log.at("color_mode").get_to(tmp_log.color_mode);
            if (log.contains("quiet_mode")) log.at("quiet_mode").get_to(tmp_log.quiet_mode);
            if (log.contains("errors_to_stderr")) log.at("errors_to_stderr").get_to(tmp_log.errors_to_stderr);
            if (log.contains("filter_info_by_feature")) log.at("filter_info_by_feature").get_to(tmp_log.filter_info_by_feature);
            if (log.contains("include_timestamp_ms")) log.at("include_timestamp_ms").get_to(tmp_log.include_timestamp_ms);
            if (log.contains("include_thread_id")) log.at("include_thread_id").get_to(tmp_log.include_thread_id);
            if (log.contains("include_source_location")) log.at("include_source_location").get_to(tmp_log.include_source_location);
            if (log.contains("features"))
            {
                if (!log["features"].is_array()) return false;
                tmp_log.features.clear();
                for (const auto &feat : log["features"])
                {
                    if (!feat.is_string()) return false;
                    tmp_log.features.push_back(feat.get<std::string>());
                }
            }
        }

        if (tmp_exec.max_frames_in_flight < 1 || tmp_mem.staging_pool_initial_size_mb < 1)
        {
            return false;
        }

        device_prefs = std::move(tmp_dev);
        execution_prefs = std::move(tmp_exec);
        memory_prefs = std::move(tmp_mem);
        logging_prefs = std::move(tmp_log);
        config_file_path = effective_path;
        is_loaded = true;
        return true;
    }
    catch (const std::exception &e)
    {
        Logger::warn(Input_Format{"User_Preferences::load: Failed to parse {}: {}", effective_path, e.what()},
                     Log_Feature::DEVICE_MANAGEMENT);
        return false;
    }
}

bool User_Preferences::saveUnlocked(const std::string &path) const
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
        j["memory"]["enable_push_descriptor"] = memory_prefs.enable_push_descriptor;

        j["logging"]["enable_logging"] = logging_prefs.enable_logging;
        j["logging"]["level"] = logging_prefs.level;
        j["logging"]["console_enabled"] = logging_prefs.console_enabled;
        j["logging"]["console_level"] = logging_prefs.console_level;
        j["logging"]["file_enabled"] = logging_prefs.file_enabled;
        j["logging"]["file_level"] = logging_prefs.file_level;
        j["logging"]["log_directory"] = logging_prefs.log_directory;
        j["logging"]["color_mode"] = logging_prefs.color_mode;
        j["logging"]["quiet_mode"] = logging_prefs.quiet_mode;
        j["logging"]["errors_to_stderr"] = logging_prefs.errors_to_stderr;
        j["logging"]["filter_info_by_feature"] = logging_prefs.filter_info_by_feature;
        j["logging"]["include_timestamp_ms"] = logging_prefs.include_timestamp_ms;
        j["logging"]["include_thread_id"] = logging_prefs.include_thread_id;
        j["logging"]["include_source_location"] = logging_prefs.include_source_location;
        j["logging"]["features"] = logging_prefs.features;

        std::ofstream file(path, std::ios::trunc);
        if (!file.is_open())
        {
            return false;
        }
        file << j.dump(2) << std::endl;
        file.flush();
        return static_cast<bool>(file);
    }
    catch (...)
    {
        return false;
    }
}

bool User_Preferences::save() const
{
    std::lock_guard lock(preferences_mutex);
    return saveUnlocked(config_file_path);
}

bool User_Preferences::save(const std::string &path) const
{
    std::lock_guard lock(preferences_mutex);
    return saveUnlocked(path.empty() ? config_file_path : path);
}

void User_Preferences::setDevicePreferences(const Device_Preferences &prefs)
{
    std::lock_guard lock(preferences_mutex);
    device_prefs = prefs;
}

void User_Preferences::setExecutionPreferences(const Execution_Preferences &prefs)
{
    if (prefs.max_frames_in_flight < 1)
    {
        throw std::invalid_argument("User_Preferences::setExecutionPreferences: max_frames_in_flight must be >= 1");
    }
    std::lock_guard lock(preferences_mutex);
    execution_prefs = prefs;
}

void User_Preferences::setMemoryPreferences(const Memory_Preferences &prefs)
{
    if (prefs.staging_pool_initial_size_mb < 1)
    {
        throw std::invalid_argument("User_Preferences::setMemoryPreferences: staging_pool_initial_size_mb must be >= 1");
    }
    std::lock_guard lock(preferences_mutex);
    memory_prefs = prefs;
}

void User_Preferences::setLoggingPreferences(const Logging_Preferences &prefs)
{
    std::lock_guard lock(preferences_mutex);
    logging_prefs = prefs;
}

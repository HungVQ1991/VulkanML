#include "helper/logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>

bool is_coop = false;

Logger::Logger() = default;

Logger::~Logger()
{
    if (log_file_stream.is_open())
    {
        try
        {
            log_file_stream << std::format("[{}] [{:<5}] [{:<18}] END OF LOG INSTANCE\n", timestamp(), "INFO", "General");
            log_file_stream.flush();
        }
        catch (...)
        {
        }
    }
}

Logger &Logger::getInstance()
{
    static Logger instance;
    return instance;
}

std::string Logger::timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm, &t_c);
#else
    localtime_r(&t_c, &tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buffer);
}

std::string Logger::featureToString(Log_Feature _feature)
{
    std::string result;
    auto append_tag = [&result](std::string_view _tag)
    {
        if (!result.empty())
        {
            result += "|";
        }
        result += _tag;
    };

    if ((_feature & Log_Feature::DEVICE_MANAGEMENT) != Log_Feature::NONE)
        append_tag("Device");
    if ((_feature & Log_Feature::MEMORY_ALLOCATION) != Log_Feature::NONE)
        append_tag("MemAlloc");
    if ((_feature & Log_Feature::MEMORY_TRANSFER) != Log_Feature::NONE)
        append_tag("MemTransfer");
    if ((_feature & Log_Feature::SYNCHRONIZATION) != Log_Feature::NONE)
        append_tag("Sync");

    if ((_feature & Log_Feature::GRAPH_RECORDING) != Log_Feature::NONE)
        append_tag("GraphRecord");
    if ((_feature & Log_Feature::OPERATOR_FUSION) != Log_Feature::NONE)
        append_tag("Fusion");
    if ((_feature & Log_Feature::SHADER_GENERATION) != Log_Feature::NONE)
        append_tag("ShaderGen");
    if ((_feature & Log_Feature::DISPATCH_EXECUTION) != Log_Feature::NONE)
        append_tag("Dispatch");

    if ((_feature & Log_Feature::FORWARD_EVALUATION) != Log_Feature::NONE)
        append_tag("Forward");
    if ((_feature & Log_Feature::BACKWARD_PROPAGATION) != Log_Feature::NONE)
        append_tag("Backward");

    if ((_feature & Log_Feature::DENSE_COMPUTE) != Log_Feature::NONE)
        append_tag("Dense");
    if ((_feature & Log_Feature::CONV2D_COMPUTE) != Log_Feature::NONE)
        append_tag("Conv2D");
    if ((_feature & Log_Feature::POOLING_COMPUTE) != Log_Feature::NONE)
        append_tag("Pooling");
    if ((_feature & Log_Feature::NORMALIZATION_COMPUTE) != Log_Feature::NONE)
        append_tag("Norm");
    if ((_feature & Log_Feature::ACTIVATION_COMPUTE) != Log_Feature::NONE)
        append_tag("Activation");
    if ((_feature & Log_Feature::LOSS_COMPUTE) != Log_Feature::NONE)
        append_tag("Loss");

    if ((_feature & Log_Feature::OPTIMIZER_STEP) != Log_Feature::NONE)
        append_tag("Optimizer");
    if ((_feature & Log_Feature::LR_SCHEDULER) != Log_Feature::NONE)
        append_tag("LRScheduler");
    if ((_feature & Log_Feature::DATA_PIPELINE) != Log_Feature::NONE)
        append_tag("DataPipeline");
    if ((_feature & Log_Feature::MODEL_SERIALIZATION) != Log_Feature::NONE)
        append_tag("Serialize");
    if ((_feature & Log_Feature::TENSOR_INSPECTION) != Log_Feature::NONE)
        append_tag("TensorDump");
    if ((_feature & Log_Feature::LAYER_INSPECTION) != Log_Feature::NONE)
        append_tag("LayerInspect");
    if ((_feature & Log_Feature::FP16_METRICS) != Log_Feature::NONE)
        append_tag("FP16");

    if (result.empty())
    {
        return "General";
    }
    return result;
}

void Logger::openNewLogFile()
{
    std::error_code error_code;
    std::filesystem::create_directories(log_directory, error_code);

    std::vector<std::filesystem::path> log_files;
    if (std::filesystem::exists(log_directory))
    {
        for (const auto &entry : std::filesystem::directory_iterator(log_directory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".log" &&
                entry.path().filename().string().starts_with("log_"))
            {
                log_files.push_back(entry.path());
            }
        }
    }

    std::ranges::sort(log_files, [](const auto &_a, const auto &_b)
                      { return std::filesystem::last_write_time(_a) < std::filesystem::last_write_time(_b); });

    while (log_files.size() >= MAX_LOG_FILES)
    {
        std::error_code remove_error;
        std::filesystem::remove(log_files.front(), remove_error);
        log_files.erase(log_files.begin());
    }

    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm, &t_c);
#else
    localtime_r(&t_c, &tm);
#endif
    char date_buffer[32];
    std::strftime(date_buffer, sizeof(date_buffer), "log_%Y%m%d_%H%M%S", &tm);
    const std::string base_name = date_buffer;

    current_log_file = log_directory / (base_name + ".log");
    size_t suffix_index = 1;

    while (std::filesystem::exists(current_log_file))
    {
        current_log_file = log_directory / std::format("{}_{}.log", base_name, suffix_index++);
    }

    log_file_stream.open(current_log_file, std::ios::out | std::ios::app);
    if (!log_file_stream.is_open())
    {
        throw std::runtime_error("Failed to open log file: " + current_log_file.string());
    }

    log_file_stream << std::format("[{}] [{:<5}] [{:<18}] START OF LOG INSTANCE\n", timestamp(), "INFO", "General");
    log_file_stream.flush();
}

void Logger::initialize(const std::string &_directory)
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    if (instance.log_file_stream.is_open())
    {
        instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}] REDIRECTING LOG INSTANCE\n", timestamp(), "INFO", "General");
        instance.log_file_stream.close();
    }
    instance.log_directory = _directory;
    if (instance.is_file_logging_enabled.load(std::memory_order_relaxed))
    {
        instance.openNewLogFile();
    }
}

void Logger::init(const std::string &_directory)
{
    initialize(_directory);
}

bool Logger::logMessage(
    const std::string &_message,
    Log_Level _level,
    bool _print_to_console,
    size_t _repetition_count,
    Log_Feature _feature,
    const std::source_location _location)
{
#if !ENABLE_LOGGING
    return false;
#endif

    Logger &instance = getInstance();

    if (_level == Log_Level::LOG_DEBUG)
    {
        uint64_t current_features = instance.active_features.load(std::memory_order_relaxed);
        if (_feature != Log_Feature::NONE && ((current_features & static_cast<uint64_t>(_feature)) == 0))
        {
            return false;
        }
    }

    std::lock_guard<std::mutex> lock(instance.logger_mutex);

    if (_repetition_count > 0)
    {
        auto &line_map = instance.call_site_counters[_location.file_name()];
        size_t &current_count = line_map[_location.line()];
        if (current_count >= _repetition_count)
        {
            return false;
        }
        current_count++;
    }

    Log_Record record{
        .timestamp = timestamp(),
        .level = _level,
        .feature = _feature,
        .message = _message,
        .file = _location.file_name(),
        .line = _location.line()};

    if (instance.ring_buffer.size() >= MAX_RING_BUFFER_ENTRIES)
    {
        instance.ring_buffer.pop_front();
    }
    instance.ring_buffer.push_back(record);

    std::string feat_str = featureToString(_feature);

    if (instance.is_file_logging_enabled.load(std::memory_order_relaxed) && instance.log_file_stream.is_open())
    {
        instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}] {}\n",
                                                record.timestamp,
                                                levelToString(_level),
                                                feat_str,
                                                _message);
        instance.log_file_stream.flush();
    }

    bool force_console = instance.is_force_all_console_enabled.load(std::memory_order_relaxed);
    bool console_enabled = instance.is_console_enabled.load(std::memory_order_relaxed);
    bool should_print_to_console = force_console || _print_to_console || _level == Log_Level::LOG_ERROR;

    if (should_print_to_console && (console_enabled || _level == Log_Level::LOG_ERROR || force_console))
    {
        std::cout << std::format("{}[{}] [{:<5}] [{:<18}] {}\033[0m\n",
                                 levelToAnsiColor(_level),
                                 record.timestamp,
                                 levelToString(_level),
                                 feat_str,
                                 _message);
    }
    return true;
}

void Logger::logFp16TensorStats(
    std::string_view tensor_name,
    const std::vector<float> &data,
    size_t element_count,
    std::string_view data_type_name,
    Log_Level level,
    const std::source_location location)
{
    if (data.empty() || element_count == 0)
    {
        return;
    }
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    float sum_val = 0.0f;
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
    size_t subnormal_count = 0;

    for (float val : data)
    {
        if (std::isnan(val))
        {
            nan_count++;
            continue;
        }
        if (std::isinf(val))
        {
            inf_count++;
            continue;
        }
        if (val == 0.0f)
        {
            zero_count++;
        }
        else if (std::abs(val) < 6.1035e-5f)
        {
            subnormal_count++;
        }
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }

    float valid_elements = static_cast<float>(data.size() - nan_count - inf_count);
    float mean_val = (valid_elements > 0.0f) ? (sum_val / valid_elements) : 0.0f;

    logMessage(
        Input_Format{"[FP16 Stats] {}: type={}, count={}, min={:.5e}, max={:.5e}, mean={:.5e}, zeros={}/{} ({:.1f}%), subnormals={}/{} ({:.1f}%), NaNs={}, Infs={}",
                     tensor_name, data_type_name, element_count, min_val, max_val, mean_val,
                     zero_count, element_count, (100.0f * static_cast<float>(zero_count) / static_cast<float>(element_count)),
                     subnormal_count, element_count, (100.0f * static_cast<float>(subnormal_count) / static_cast<float>(element_count)),
                     nan_count, inf_count},
        level,
        false,
        0,
        Log_Feature::FP16_METRICS,
        location);
}

void Logger::dumpRecentLogs(std::ostream &_output_stream)
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);

    _output_stream << "\n=== IN-MEMORY LOG DUMP (" << instance.ring_buffer.size() << " ENTRIES) ===\n";
    for (const auto &rec : instance.ring_buffer)
    {
        _output_stream << std::format("[{}] [{:<5}] [{:<18}] ({}:{}) {}\n",
                                      rec.timestamp,
                                      levelToString(rec.level),
                                      featureToString(rec.feature),
                                      rec.file,
                                      rec.line,
                                      rec.message);
    }
    _output_stream << "=== END OF DUMP ===\n\n";
}

void Logger::resetLogCounters()
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    instance.call_site_counters.clear();
}

void Logger::resetSpecificLog()
{
}

std::string Logger::getCurrentLogFilepath()
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    return instance.current_log_file.string();
}

std::string Logger::getLogDirectory()
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    return instance.log_directory.string();
}

size_t Logger::getMaxRingBufferEntries() noexcept { return MAX_RING_BUFFER_ENTRIES; }
size_t Logger::getMaxLogFiles() noexcept { return MAX_LOG_FILES; }
Log_Feature Logger::getActiveFeatures() noexcept { return static_cast<Log_Feature>(getInstance().active_features.load(std::memory_order_relaxed)); }
bool Logger::isForceAllConsoleOutputEnabled() noexcept { return getInstance().is_force_all_console_enabled.load(std::memory_order_relaxed); }
bool Logger::isFileLoggingEnabled() noexcept { return getInstance().is_file_logging_enabled.load(std::memory_order_relaxed); }
bool Logger::isConsoleEnabled() noexcept { return getInstance().is_console_enabled.load(std::memory_order_relaxed); }

void Logger::setFileLogging(bool _enable) noexcept
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    instance.is_file_logging_enabled.store(_enable, std::memory_order_relaxed);
    if (_enable && !instance.log_file_stream.is_open())
    {
        instance.openNewLogFile();
    }
    else if (!_enable && instance.log_file_stream.is_open())
    {
        instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}] PAUSING LOG INSTANCE\n", timestamp(), "INFO", "General");
        instance.log_file_stream.close();
    }
}

void Logger::enableFeature(Log_Feature _feature, bool _enable) noexcept
{
    Logger &instance = getInstance();
    uint64_t feature_mask = static_cast<uint64_t>(_feature);
    if (_enable)
    {
        instance.active_features.fetch_or(feature_mask, std::memory_order_relaxed);
    }
    else
    {
        instance.active_features.fetch_and(~feature_mask, std::memory_order_relaxed);
    }
}

void Logger::setLogDirectory(const std::string &_directory) { initialize(_directory); }
void Logger::setOnlyActiveFeatures(Log_Feature _feature_mask) noexcept { getInstance().active_features.store(static_cast<uint64_t>(_feature_mask), std::memory_order_relaxed); }
void Logger::setForceAllConsoleOutput(bool _enable) noexcept { getInstance().is_force_all_console_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::forceAllConsoleOutput(bool _enable) noexcept { setForceAllConsoleOutput(_enable); }
void Logger::enableFileLogging(bool _enable) noexcept { setFileLogging(_enable); }
void Logger::setConsoleOutput(bool _enable) noexcept { getInstance().is_console_enabled.store(_enable, std::memory_order_relaxed); }

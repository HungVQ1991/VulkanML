#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef ENABLE_LOGGING
#define ENABLE_LOGGING 0
#endif

bool is_coop = false;

enum class Log_Level : std::uint8_t
{
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_LEVEL_END
};

enum class Log_Feature : std::uint64_t
{
    NONE = 0,

    DEVICE_MANAGEMENT = 1ULL << 0,
    MEMORY_ALLOCATION = 1ULL << 1,
    MEMORY_TRANSFER = 1ULL << 2,
    SYNCHRONIZATION = 1ULL << 3,

    GRAPH_RECORDING = 1ULL << 4,
    OPERATOR_FUSION = 1ULL << 5,
    SHADER_GENERATION = 1ULL << 6,
    DISPATCH_EXECUTION = 1ULL << 7,

    FORWARD_EVALUATION = 1ULL << 8,
    BACKWARD_PROPAGATION = 1ULL << 9,

    DENSE_COMPUTE = 1ULL << 10,
    CONV2D_COMPUTE = 1ULL << 11,
    POOLING_COMPUTE = 1ULL << 12,
    NORMALIZATION_COMPUTE = 1ULL << 13,
    ACTIVATION_COMPUTE = 1ULL << 14,
    LOSS_COMPUTE = 1ULL << 15,

    OPTIMIZER_STEP = 1ULL << 16,
    LR_SCHEDULER = 1ULL << 17,
    DATA_PIPELINE = 1ULL << 18,
    MODEL_SERIALIZATION = 1ULL << 19,
    TENSOR_INSPECTION = 1ULL << 20,
    LAYER_INSPECTION = 1ULL << 21,

    HARDWARE = DEVICE_MANAGEMENT | MEMORY_ALLOCATION | MEMORY_TRANSFER | SYNCHRONIZATION,
    GRAPH = GRAPH_RECORDING | OPERATOR_FUSION | SHADER_GENERATION | DISPATCH_EXECUTION,
    OPERATIONS = DENSE_COMPUTE | CONV2D_COMPUTE | POOLING_COMPUTE | NORMALIZATION_COMPUTE | ACTIVATION_COMPUTE | LOSS_COMPUTE,
    TRAINING = FORWARD_EVALUATION | BACKWARD_PROPAGATION | OPTIMIZER_STEP | LR_SCHEDULER | LOSS_COMPUTE | DATA_PIPELINE | LAYER_INSPECTION,

    ALL = 0xFFFFFFFFFFFFFFFFULL
};

constexpr Log_Feature operator|(Log_Feature lhs, Log_Feature rhs) noexcept
{
    return static_cast<Log_Feature>(static_cast<std::uint64_t>(lhs) | static_cast<std::uint64_t>(rhs));
}

constexpr Log_Feature operator&(Log_Feature lhs, Log_Feature rhs) noexcept
{
    return static_cast<Log_Feature>(static_cast<std::uint64_t>(lhs) & static_cast<std::uint64_t>(rhs));
}

constexpr Log_Feature operator~(Log_Feature feature) noexcept
{
    return static_cast<Log_Feature>(~static_cast<std::uint64_t>(feature));
}

struct Log_Record
{
    std::string timestamp;
    Log_Level level;
    Log_Feature feature;
    std::string message;
    std::string file;
    std::uint_least32_t line;
};

class Logger
{
private:
    static constexpr std::size_t MAX_RING_BUFFER_ENTRIES = 2048;
    static constexpr std::size_t MAX_LOG_FILES = 15;

    std::atomic<std::uint64_t> active_features{static_cast<std::uint64_t>(Log_Feature::ALL)};
    std::atomic<bool> is_console_enabled{true};
    std::atomic<bool> is_file_logging_enabled{false};
    std::atomic<bool> is_force_all_console_enabled{false};

    std::filesystem::path log_directory{"logs"};
    std::filesystem::path current_log_file;
    std::ofstream log_file_stream;

    std::deque<Log_Record> ring_buffer;
    std::unordered_map<std::string, std::unordered_map<std::uint_least32_t, std::size_t>> call_site_counters;
    std::mutex logger_mutex;

    Logger() = default;

    ~Logger()
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

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    static Logger &getInstance()
    {
        static Logger instance;
        return instance;
    }

    static std::string timestamp()
    {
        const auto now = std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now());
        const std::chrono::zoned_time zt{std::chrono::current_zone(), now};
        return std::format("{:%Y-%m-%d %H:%M:%S}", zt);
    }

    static constexpr std::string_view levelToString(Log_Level _level) noexcept
    {
        switch (_level)
        {
        case Log_Level::LOG_DEBUG:
            return "DEBUG";
        case Log_Level::LOG_INFO:
            return "INFO";
        case Log_Level::LOG_WARNING:
            return "WARN";
        case Log_Level::LOG_ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
        }
    }

    static constexpr std::string_view levelToAnsiColor(Log_Level _level) noexcept
    {
        switch (_level)
        {
        case Log_Level::LOG_DEBUG:
            return "\033[36m";
        case Log_Level::LOG_INFO:
            return "\033[32m";
        case Log_Level::LOG_WARNING:
            return "\033[33m";
        case Log_Level::LOG_ERROR:
            return "\033[31m";
        default:
            return "\033[0m";
        }
    }

    static std::string featureToString(Log_Feature _feature)
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

        if (result.empty())
        {
            return "General";
        }
        return result;
    }

    void openNewLogFile()
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

        const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        const std::chrono::zoned_time zt{std::chrono::current_zone(), now};
        const std::string base_name = std::format("log_{:%Y%m%d_%H%M%S}", zt);

        current_log_file = log_directory / (base_name + ".log");
        std::size_t suffix_index = 1;

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

public:
    static void initialize(const std::string &_directory)
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

    static void init(const std::string &_directory)
    {
        initialize(_directory);
    }

    static void setFileLogging(bool _enable) noexcept
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

    static void enableFileLogging(bool _enable = true) noexcept
    {
        setFileLogging(_enable);
    }

    [[nodiscard]] static bool isFileLoggingEnabled() noexcept
    {
        return getInstance().is_file_logging_enabled.load(std::memory_order_relaxed);
    }

    static void setForceAllConsoleOutput(bool _enable = true) noexcept
    {
        getInstance().is_force_all_console_enabled.store(_enable, std::memory_order_relaxed);
    }

    static void forceAllConsoleOutput(bool _enable = true) noexcept
    {
        setForceAllConsoleOutput(_enable);
    }

    [[nodiscard]] static bool isForceAllConsoleOutputEnabled() noexcept
    {
        return getInstance().is_force_all_console_enabled.load(std::memory_order_relaxed);
    }

    static std::string getCurrentLogFilepath()
    {
        Logger &instance = getInstance();
        std::lock_guard<std::mutex> lock(instance.logger_mutex);
        return instance.current_log_file.string();
    }

    static void enableFeature(Log_Feature _feature, bool _enable = true) noexcept
    {
        Logger &instance = getInstance();
        std::uint64_t feature_mask = static_cast<std::uint64_t>(_feature);
        if (_enable)
        {
            instance.active_features.fetch_or(feature_mask, std::memory_order_relaxed);
        }
        else
        {
            instance.active_features.fetch_and(~feature_mask, std::memory_order_relaxed);
        }
    }

    static void setOnlyActiveFeatures(Log_Feature _feature_mask) noexcept
    {
        getInstance().active_features.store(static_cast<std::uint64_t>(_feature_mask), std::memory_order_relaxed);
    }

    static void setConsoleOutput(bool _enable) noexcept
    {
        getInstance().is_console_enabled.store(_enable, std::memory_order_relaxed);
    }

    static bool logMessage(
        const std::string &_message,
        Log_Level _level = Log_Level::LOG_INFO,
        bool _print_to_console = false,
        std::size_t _repetition_count = 0,
        Log_Feature _feature = Log_Feature::NONE,
        const std::source_location _location = std::source_location::current())
    {
#if !ENABLE_LOGGING
        return false;
#endif

        Logger &instance = getInstance();

        if (_level == Log_Level::LOG_DEBUG)
        {
            std::uint64_t current_features = instance.active_features.load(std::memory_order_relaxed);
            if (_feature != Log_Feature::NONE && ((current_features & static_cast<std::uint64_t>(_feature)) == 0))
            {
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(instance.logger_mutex);

        if (_repetition_count > 0)
        {
            auto &line_map = instance.call_site_counters[_location.file_name()];
            std::size_t &current_count = line_map[_location.line()];
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

    static void dumpRecentLogs(std::ostream &_output_stream = std::cerr)
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

    static void clearCallSiteCounters()
    {
        Logger &instance = getInstance();
        std::lock_guard<std::mutex> lock(instance.logger_mutex);
        instance.call_site_counters.clear();
    }
};
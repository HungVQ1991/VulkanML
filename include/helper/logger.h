#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef ENABLE_LOGGING
#define ENABLE_LOGGING 0
#endif

extern bool is_coop;

enum class Log_Level : std::uint8_t
{
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_LEVEL_END
};

enum class Log_Feature : uint64_t
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
    FP16_METRICS = 1ULL << 22,

    FP16 = FP16_METRICS,
    FP16_COMPUTE = FP16_METRICS,
    MIXED_PRECISION = FP16_METRICS,

    HARDWARE = DEVICE_MANAGEMENT | MEMORY_ALLOCATION | MEMORY_TRANSFER | SYNCHRONIZATION,
    GRAPH = GRAPH_RECORDING | OPERATOR_FUSION | SHADER_GENERATION | DISPATCH_EXECUTION,
    OPERATIONS = DENSE_COMPUTE | CONV2D_COMPUTE | POOLING_COMPUTE | NORMALIZATION_COMPUTE | ACTIVATION_COMPUTE | LOSS_COMPUTE | FP16_METRICS,
    TRAINING = FORWARD_EVALUATION | BACKWARD_PROPAGATION | OPTIMIZER_STEP | LR_SCHEDULER | LOSS_COMPUTE | DATA_PIPELINE | LAYER_INSPECTION | FP16_METRICS,

    ALL = 0xFFFFFFFFFFFFFFFFULL
};

constexpr Log_Feature operator|(Log_Feature lhs, Log_Feature rhs) noexcept
{
    return static_cast<Log_Feature>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr Log_Feature operator&(Log_Feature lhs, Log_Feature rhs) noexcept
{
    return static_cast<Log_Feature>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

constexpr Log_Feature operator~(Log_Feature feature) noexcept
{
    return static_cast<Log_Feature>(~static_cast<uint64_t>(feature));
}

template <typename... Args>
struct Input_Format
{
    std::string_view format_string;
    std::tuple<Args...> arguments;

    constexpr Input_Format(std::format_string<Args...> _fmt, Args... _args)
        : format_string(_fmt.get()), arguments(std::move(_args)...)
    {
    }

    std::string toString() const
    {
        if constexpr (sizeof...(Args) == 0)
        {
            return std::string(format_string);
        }
        else
        {
            return std::apply([this](const auto &...unpacked_args)
                              { return std::vformat(format_string, std::make_format_args(unpacked_args...)); }, arguments);
        }
    }

    operator std::string() const
    {
        return toString();
    }
};

template <typename... Args>
Input_Format(std::format_string<Args...>, Args...) -> Input_Format<Args...>;

struct Log_Record
{
    std::string timestamp;
    Log_Level level;
    Log_Feature feature;
    std::string message;
    std::string file;
    std::uint_least32_t line;

    const std::string &getMessage() const noexcept { return message; }
    const std::string &getTimestamp() const noexcept { return timestamp; }
    const std::string &getFile() const noexcept { return file; }
    Log_Feature getFeature() const noexcept { return feature; }
    std::uint_least32_t getLine() const noexcept { return line; }
    Log_Level getLevel() const noexcept { return level; }

    void setMessage(const std::string &_message) { message = _message; }
    void setTimestamp(const std::string &_timestamp) { timestamp = _timestamp; }
    void setFile(const std::string &_file) { file = _file; }
    void setFeature(Log_Feature _feature) noexcept { feature = _feature; }
    void setLine(std::uint_least32_t _line) noexcept { line = _line; }
    void setLevel(Log_Level _level) noexcept { level = _level; }
};

class Logger
{
private:
    static constexpr size_t MAX_RING_BUFFER_ENTRIES = 2048;
    static constexpr size_t MAX_LOG_FILES = 15;

    std::atomic<uint64_t> active_features{static_cast<uint64_t>(Log_Feature::ALL)};
    std::atomic<bool> is_console_enabled{true};
    std::atomic<bool> is_file_logging_enabled{false};
    std::atomic<bool> is_force_all_console_enabled{false};

    std::filesystem::path log_directory{"logs"};
    std::filesystem::path current_log_file;
    std::ofstream log_file_stream;

    std::deque<Log_Record> ring_buffer;
    std::unordered_map<std::string, std::unordered_map<std::uint_least32_t, size_t>> call_site_counters;
    std::mutex logger_mutex;

    Logger();
    ~Logger();

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    static Logger &getInstance();
    static std::string timestamp();
    static std::string featureToString(Log_Feature _feature);
    void openNewLogFile();

public:
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

    static void initialize(const std::string &_directory);
    static void init(const std::string &_directory);

    static bool logMessage(
        const std::string &_message,
        Log_Level _level = Log_Level::LOG_INFO,
        bool _print_to_console = false,
        size_t _repetition_count = 0,
        Log_Feature _feature = Log_Feature::NONE,
        const std::source_location _location = std::source_location::current());

    template <typename... Args>
    static bool logMessage(
        const Input_Format<Args...> &_format,
        Log_Level _level = Log_Level::LOG_INFO,
        bool _print_to_console = false,
        size_t _repetition_count = 0,
        Log_Feature _feature = Log_Feature::NONE,
        const std::source_location _location = std::source_location::current())
    {
#if !ENABLE_LOGGING
        return false;
#else
        return logMessage(_format.toString(), _level, _print_to_console, _repetition_count, _feature, _location);
#endif
    }

    static void logFp16TensorStats(
        std::string_view tensor_name,
        const std::vector<float> &data,
        size_t element_count,
        std::string_view data_type_name = "FLOAT16",
        Log_Level level = Log_Level::LOG_DEBUG,
        const std::source_location location = std::source_location::current());

    static void dumpRecentLogs(std::ostream &_output_stream = std::cerr);
    static void resetLogCounters();
    static void resetSpecificLog();
    static std::string getCurrentLogFilepath();
    static std::string getLogDirectory();
    static size_t getMaxRingBufferEntries() noexcept;
    static size_t getMaxLogFiles() noexcept;
    static Log_Feature getActiveFeatures() noexcept;
    static bool isForceAllConsoleOutputEnabled() noexcept;
    static bool isFileLoggingEnabled() noexcept;
    static bool isConsoleEnabled() noexcept;

    static void setFileLogging(bool _enable) noexcept;
    static void enableFeature(Log_Feature _feature, bool _enable = true) noexcept;
    static void setLogDirectory(const std::string &_directory);
    static void setOnlyActiveFeatures(Log_Feature _feature_mask) noexcept;
    static void setForceAllConsoleOutput(bool _enable = true) noexcept;
    static void forceAllConsoleOutput(bool _enable = true) noexcept;
    static void enableFileLogging(bool _enable = true) noexcept;
    static void setConsoleOutput(bool _enable) noexcept;
};
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef ENABLE_LOGGING
#define ENABLE_LOGGING 1
#endif

#if defined(ENABLE_LOGGING) && ((ENABLE_LOGGING + 0) != 0)
#define VULKANML_LOGGING_ENABLED 1
#else
#define VULKANML_LOGGING_ENABLED 0
#endif

struct Logging_Preferences;

enum class Log_Level : std::uint8_t
{
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_LEVEL_END
};

enum class Log_Color_Mode : std::uint8_t
{
    AUTO = 0,
    ALWAYS,
    NEVER
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
    TERNARY_METRICS = 1ULL << 23,

    STATIC_GRAPH = 1ULL << 24,
    PIPELINE_CACHE = 1ULL << 25,
    LLM_INFERENCE = 1ULL << 26,
    TOKENIZER = 1ULL << 27,
    TRANSFORMER = 1ULL << 28,
    PROFILER = 1ULL << 29,
    RL_AGENT = 1ULL << 30,
    CHECKPOINT_IO = 1ULL << 31,
    GRADIENT_HEALTH = 1ULL << 32,
    DATASET_IO = 1ULL << 33,

    HARDWARE = DEVICE_MANAGEMENT | MEMORY_ALLOCATION | MEMORY_TRANSFER | SYNCHRONIZATION,
    GRAPH = GRAPH_RECORDING | OPERATOR_FUSION | SHADER_GENERATION | DISPATCH_EXECUTION | STATIC_GRAPH | PIPELINE_CACHE,
    OPERATIONS = DENSE_COMPUTE | CONV2D_COMPUTE | POOLING_COMPUTE | NORMALIZATION_COMPUTE | ACTIVATION_COMPUTE | LOSS_COMPUTE | FP16_METRICS | TERNARY_METRICS,
    TRAINING = FORWARD_EVALUATION | BACKWARD_PROPAGATION | OPTIMIZER_STEP | LR_SCHEDULER | LOSS_COMPUTE | DATA_PIPELINE | LAYER_INSPECTION | FP16_METRICS | TERNARY_METRICS | GRADIENT_HEALTH | PROFILER,
    LLM = TRANSFORMER | LLM_INFERENCE | TOKENIZER | STATIC_GRAPH,
    IO = MODEL_SERIALIZATION | CHECKPOINT_IO | DATASET_IO,

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

constexpr Log_Feature &operator|=(Log_Feature &lhs, Log_Feature rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

constexpr Log_Feature &operator&=(Log_Feature &lhs, Log_Feature rhs) noexcept
{
    lhs = lhs & rhs;
    return lhs;
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
            return std::vformat(format_string, std::make_format_args());
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

struct Logger_Config
{
    bool enable_logging = true;
    Log_Level min_level = Log_Level::LOG_DEBUG;
    Log_Level console_min_level = Log_Level::LOG_DEBUG;
    Log_Level file_min_level = Log_Level::LOG_DEBUG;
    Log_Feature active_features = Log_Feature::ALL;
    bool filter_info_by_feature = false;
    bool console_enabled = true;
    bool file_enabled = false;
    bool force_all_console = false;
    bool quiet_mode = false;
    bool errors_to_stderr = false;
    bool include_timestamp_ms = false;
    bool include_thread_id = false;
    bool include_source_location = false;
    Log_Color_Mode color_mode = Log_Color_Mode::AUTO;
    std::string log_directory = "logs";
};

class Logger
{
private:
    static constexpr size_t MAX_RING_BUFFER_ENTRIES = 2048;
    static constexpr size_t MAX_LOG_FILES = 15;

    struct Transparent_String_Hash
    {
        using is_transparent = void;
        size_t operator()(std::string_view sv) const noexcept
        {
            return std::hash<std::string_view>{}(sv);
        }
        size_t operator()(const std::string &s) const noexcept
        {
            return std::hash<std::string_view>{}(s);
        }
        size_t operator()(const char *s) const noexcept
        {
            return std::hash<std::string_view>{}(s ? std::string_view(s) : std::string_view{});
        }
    };

    struct Transparent_String_Equal
    {
        using is_transparent = void;
        bool operator()(std::string_view lhs, std::string_view rhs) const noexcept
        {
            return lhs == rhs;
        }
    };

    std::atomic<uint64_t> active_features{static_cast<uint64_t>(Log_Feature::ALL)};
    std::atomic<Log_Level> min_log_level{Log_Level::LOG_DEBUG};
    std::atomic<Log_Level> min_console_level{Log_Level::LOG_DEBUG};
    std::atomic<Log_Level> min_file_level{Log_Level::LOG_DEBUG};
    std::atomic<bool> is_filter_info_by_feature_enabled{false};
    std::atomic<bool> is_console_enabled{true};
    std::atomic<bool> is_file_logging_enabled{false};
    std::atomic<bool> is_force_all_console_enabled{false};
    std::atomic<bool> is_quiet_mode_enabled{false};
    std::atomic<bool> is_errors_to_stderr_enabled{false};
    std::atomic<bool> is_timestamp_ms_enabled{false};
    std::atomic<bool> is_thread_id_enabled{false};
    std::atomic<bool> is_source_location_enabled{false};
    std::atomic<Log_Color_Mode> color_mode{Log_Color_Mode::AUTO};

    std::filesystem::path log_directory{"logs"};
    std::filesystem::path current_log_file;
    std::ofstream log_file_stream;
    size_t lines_since_flush{0};

    std::deque<Log_Record> ring_buffer;
    std::unordered_map<std::string, std::unordered_map<std::uint_least32_t, size_t>, Transparent_String_Hash, Transparent_String_Equal> call_site_counters;
    std::mutex logger_mutex;

    Logger();
    ~Logger();

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    static Logger &getInstance();
    static void shutdownFlushAtExit() noexcept;
    static std::string timestamp();
    static std::string timestampFormatted(bool _include_ms);
    bool openNewLogFile() noexcept;
    static bool logMessageImpl(
        const std::string &_message,
        Log_Level _level,
        bool _print_to_console,
        size_t _repetition_count,
        Log_Feature _feature,
        const std::source_location _location);

public:
    static std::string featureToString(Log_Feature _feature);

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

    static inline bool logMessage(
        const std::string &_message,
        Log_Level _level = Log_Level::LOG_INFO,
        bool _print_to_console = false,
        size_t _repetition_count = 0,
        Log_Feature _feature = Log_Feature::NONE,
        const std::source_location _location = std::source_location::current())
    {
#if !VULKANML_LOGGING_ENABLED
        (void)_message;
        (void)_level;
        (void)_print_to_console;
        (void)_repetition_count;
        (void)_feature;
        (void)_location;
        return false;
#else
        return logMessageImpl(_message, _level, _print_to_console, _repetition_count, _feature, _location);
#endif
    }

    template <typename... Args>
    static bool logMessage(
        const Input_Format<Args...> &_format,
        Log_Level _level = Log_Level::LOG_INFO,
        bool _print_to_console = false,
        size_t _repetition_count = 0,
        Log_Feature _feature = Log_Feature::NONE,
        const std::source_location _location = std::source_location::current())
    {
#if !VULKANML_LOGGING_ENABLED
        (void)_format;
        (void)_level;
        (void)_print_to_console;
        (void)_repetition_count;
        (void)_feature;
        (void)_location;
        return false;
#else
        if (!shouldLog(_level, _feature))
        {
            return false;
        }
        return logMessageImpl(_format.toString(), _level, _print_to_console, _repetition_count, _feature, _location);
#endif
    }

    static inline bool debug(
        const std::string &_message,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_message, Log_Level::LOG_DEBUG, _print_to_console, _repetition_count, _feature, _location);
    }

    template <typename... Args>
    static bool debug(
        const Input_Format<Args...> &_format,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_format, Log_Level::LOG_DEBUG, _print_to_console, _repetition_count, _feature, _location);
    }

    static inline bool info(
        const std::string &_message,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_message, Log_Level::LOG_INFO, _print_to_console, _repetition_count, _feature, _location);
    }

    template <typename... Args>
    static bool info(
        const Input_Format<Args...> &_format,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_format, Log_Level::LOG_INFO, _print_to_console, _repetition_count, _feature, _location);
    }

    static inline bool warn(
        const std::string &_message,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_message, Log_Level::LOG_WARNING, _print_to_console, _repetition_count, _feature, _location);
    }

    template <typename... Args>
    static bool warn(
        const Input_Format<Args...> &_format,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_format, Log_Level::LOG_WARNING, _print_to_console, _repetition_count, _feature, _location);
    }

    static inline bool error(
        const std::string &_message,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_message, Log_Level::LOG_ERROR, _print_to_console, _repetition_count, _feature, _location);
    }

    template <typename... Args>
    static bool error(
        const Input_Format<Args...> &_format,
        Log_Feature _feature = Log_Feature::NONE,
        bool _print_to_console = true,
        size_t _repetition_count = 0,
        const std::source_location _location = std::source_location::current())
    {
        return logMessage(_format, Log_Level::LOG_ERROR, _print_to_console, _repetition_count, _feature, _location);
    }

    static bool shouldLog(Log_Level _level, Log_Feature _feature) noexcept;

    static void logFp16TensorStats(
        std::string_view tensor_name,
        const std::vector<float> &data,
        size_t element_count,
        std::string_view data_type_name = "FLOAT16",
        Log_Level level = Log_Level::LOG_DEBUG,
        const std::source_location location = std::source_location::current());

    static void logTernaryTensorStats(
        std::string_view tensor_name,
        const std::vector<float> &data,
        size_t element_count,
        float gamma = 1.0f,
        Log_Level level = Log_Level::LOG_DEBUG,
        const std::source_location location = std::source_location::current());

    static void dumpRecentLogs(std::ostream &_output_stream = std::cerr);
    static void resetLogCounters();
    static void resetSpecificLog(const std::string &_file, std::uint_least32_t _line);
    static std::string getCurrentLogFilepath();
    static std::string getLogDirectory();
    static size_t getMaxRingBufferEntries() noexcept;
    static size_t getMaxLogFiles() noexcept;
    static Log_Feature getActiveFeatures() noexcept;
    static Log_Level getMinLogLevel() noexcept;
    static Log_Level getConsoleMinLevel() noexcept;
    static Log_Level getFileMinLevel() noexcept;
    static bool isFilterInfoByFeatureEnabled() noexcept;
    static bool isForceAllConsoleOutputEnabled() noexcept;
    static bool isFileLoggingEnabled() noexcept;
    static bool isConsoleEnabled() noexcept;
    static bool isQuietModeEnabled() noexcept;
    static bool isErrorsToStderrEnabled() noexcept;
    static bool isIncludeTimestampMsEnabled() noexcept;
    static bool isIncludeThreadIdEnabled() noexcept;
    static bool isIncludeSourceLocationEnabled() noexcept;
    static Log_Color_Mode getColorMode() noexcept;

    static void enableFileLogging(bool _enable = true) noexcept;
    static void flush() noexcept;
    static void enableFeature(Log_Feature _feature, bool _enable = true) noexcept;
    static void setMinLogLevel(Log_Level _level) noexcept;
    static void setConsoleMinLevel(Log_Level _level) noexcept;
    static void setFileMinLevel(Log_Level _level) noexcept;
    static void setFilterInfoByFeature(bool _enable) noexcept;
    static void setLogDirectory(const std::string &_directory);
    static void setOnlyActiveFeatures(Log_Feature _feature_mask) noexcept;
    static void setForceAllConsoleOutput(bool _enable = true) noexcept;
    static void setConsoleOutput(bool _enable) noexcept;
    static void setQuietMode(bool _enable) noexcept;
    static void setErrorsToStderr(bool _enable) noexcept;
    static void setColorMode(Log_Color_Mode _mode) noexcept;
    static void setIncludeTimestampMs(bool _enable) noexcept;
    static void setIncludeThreadId(bool _enable) noexcept;
    static void setIncludeSourceLocation(bool _enable) noexcept;

    static void configure(const Logger_Config &_config);
    static Logger_Config getConfig();
    static void applyPreferences(const Logging_Preferences &_prefs);
    static void configureFromEnvironment();
    static void applyCliArgs(int argc, char *argv[]);
    static int loggerCliFlagArity(std::string_view _arg) noexcept;
    static bool isLoggerCliFlag(std::string_view _arg) noexcept;
    static std::string loggerCliHelpText();
    static bool parseLevel(std::string_view _text, Log_Level &_out_level) noexcept;
    static bool parseColorMode(std::string_view _text, Log_Color_Mode &_out_mode) noexcept;
    static Log_Feature parseFeatureMask(std::string_view _csv, Log_Feature _base = Log_Feature::NONE) noexcept;
};

#if VULKANML_LOGGING_ENABLED
#define VML_LOG_DEBUG(feat, fmt, ...) \
    do { \
        if (Logger::shouldLog(Log_Level::LOG_DEBUG, (feat))) { \
            Logger::logMessage(Input_Format{(fmt) __VA_OPT__(,) __VA_ARGS__}, Log_Level::LOG_DEBUG, true, 0, (feat)); \
        } \
    } while (0)

#define VML_LOG_INFO(feat, fmt, ...) \
    do { \
        if (Logger::shouldLog(Log_Level::LOG_INFO, (feat))) { \
            Logger::logMessage(Input_Format{(fmt) __VA_OPT__(,) __VA_ARGS__}, Log_Level::LOG_INFO, true, 0, (feat)); \
        } \
    } while (0)

#define VML_LOG_WARN(feat, fmt, ...) \
    do { \
        if (Logger::shouldLog(Log_Level::LOG_WARNING, (feat))) { \
            Logger::logMessage(Input_Format{(fmt) __VA_OPT__(,) __VA_ARGS__}, Log_Level::LOG_WARNING, true, 0, (feat)); \
        } \
    } while (0)

#define VML_LOG_ERROR(feat, fmt, ...) \
    do { \
        if (Logger::shouldLog(Log_Level::LOG_ERROR, (feat))) { \
            Logger::logMessage(Input_Format{(fmt) __VA_OPT__(,) __VA_ARGS__}, Log_Level::LOG_ERROR, true, 0, (feat)); \
        } \
    } while (0)

#define VML_LOG_ONCE(lvl, feat, fmt, ...) \
    do { \
        if (Logger::shouldLog((lvl), (feat))) { \
            Logger::logMessage(Input_Format{(fmt) __VA_OPT__(,) __VA_ARGS__}, (lvl), true, 1, (feat)); \
        } \
    } while (0)
#else
#define VML_LOG_DEBUG(feat, fmt, ...) ((void)0)
#define VML_LOG_INFO(feat, fmt, ...) ((void)0)
#define VML_LOG_WARN(feat, fmt, ...) ((void)0)
#define VML_LOG_ERROR(feat, fmt, ...) ((void)0)
#define VML_LOG_ONCE(lvl, feat, fmt, ...) ((void)0)
#endif

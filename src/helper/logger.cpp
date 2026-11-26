#include "helper/logger.h"
#include "helper/user_preferences.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace
{
std::string toUpperAscii(std::string_view sv)
{
    std::string out;
    out.reserve(sv.size());
    for (unsigned char c : sv)
    {
        out.push_back(static_cast<char>(std::toupper(c)));
    }
    return out;
}

std::string_view trimAscii(std::string_view sv) noexcept
{
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r' || sv.front() == '\n'))
    {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r' || sv.back() == '\n'))
    {
        sv.remove_suffix(1);
    }
    return sv;
}

std::string getEnvString(const char *name)
{
#if defined(_MSC_VER)
    char *buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr)
    {
        std::string val(buf);
        std::free(buf);
        return val;
    }
    return "";
#else
    const char *val = std::getenv(name);
    return val ? std::string(val) : std::string{};
#endif
}

bool parseBoolText(std::string_view text, bool &out_val) noexcept
{
    const std::string up = toUpperAscii(trimAscii(text));
    if (up == "1" || up == "TRUE" || up == "ON" || up == "YES" || up == "ENABLE" || up == "ENABLED")
    {
        out_val = true;
        return true;
    }
    if (up == "0" || up == "FALSE" || up == "OFF" || up == "NO" || up == "DISABLE" || up == "DISABLED")
    {
        out_val = false;
        return true;
    }
    return false;
}

Log_Feature lookupSingleFeatureToken(std::string_view up) noexcept
{
    static constexpr auto table = std::to_array<std::pair<std::string_view, Log_Feature>>({
        {"NONE", Log_Feature::NONE},
        {"DEVICE_MANAGEMENT", Log_Feature::DEVICE_MANAGEMENT},
        {"DEVICE", Log_Feature::DEVICE_MANAGEMENT},
        {"MEMORY_ALLOCATION", Log_Feature::MEMORY_ALLOCATION},
        {"MEMALLOC", Log_Feature::MEMORY_ALLOCATION},
        {"MEMORY_TRANSFER", Log_Feature::MEMORY_TRANSFER},
        {"MEMTRANSFER", Log_Feature::MEMORY_TRANSFER},
        {"SYNCHRONIZATION", Log_Feature::SYNCHRONIZATION},
        {"SYNC", Log_Feature::SYNCHRONIZATION},
        {"GRAPH_RECORDING", Log_Feature::GRAPH_RECORDING},
        {"GRAPHRECORD", Log_Feature::GRAPH_RECORDING},
        {"OPERATOR_FUSION", Log_Feature::OPERATOR_FUSION},
        {"FUSION", Log_Feature::OPERATOR_FUSION},
        {"SHADER_GENERATION", Log_Feature::SHADER_GENERATION},
        {"SHADERGEN", Log_Feature::SHADER_GENERATION},
        {"DISPATCH_EXECUTION", Log_Feature::DISPATCH_EXECUTION},
        {"DISPATCH", Log_Feature::DISPATCH_EXECUTION},
        {"FORWARD_EVALUATION", Log_Feature::FORWARD_EVALUATION},
        {"FORWARD", Log_Feature::FORWARD_EVALUATION},
        {"BACKWARD_PROPAGATION", Log_Feature::BACKWARD_PROPAGATION},
        {"BACKWARD", Log_Feature::BACKWARD_PROPAGATION},
        {"DENSE_COMPUTE", Log_Feature::DENSE_COMPUTE},
        {"DENSE", Log_Feature::DENSE_COMPUTE},
        {"CONV2D_COMPUTE", Log_Feature::CONV2D_COMPUTE},
        {"CONV2D", Log_Feature::CONV2D_COMPUTE},
        {"POOLING_COMPUTE", Log_Feature::POOLING_COMPUTE},
        {"POOLING", Log_Feature::POOLING_COMPUTE},
        {"NORMALIZATION_COMPUTE", Log_Feature::NORMALIZATION_COMPUTE},
        {"NORM", Log_Feature::NORMALIZATION_COMPUTE},
        {"ACTIVATION_COMPUTE", Log_Feature::ACTIVATION_COMPUTE},
        {"ACTIVATION", Log_Feature::ACTIVATION_COMPUTE},
        {"LOSS_COMPUTE", Log_Feature::LOSS_COMPUTE},
        {"LOSS", Log_Feature::LOSS_COMPUTE},
        {"OPTIMIZER_STEP", Log_Feature::OPTIMIZER_STEP},
        {"OPTIMIZER", Log_Feature::OPTIMIZER_STEP},
        {"LR_SCHEDULER", Log_Feature::LR_SCHEDULER},
        {"LRSCHEDULER", Log_Feature::LR_SCHEDULER},
        {"DATA_PIPELINE", Log_Feature::DATA_PIPELINE},
        {"DATAPIPELINE", Log_Feature::DATA_PIPELINE},
        {"MODEL_SERIALIZATION", Log_Feature::MODEL_SERIALIZATION},
        {"SERIALIZE", Log_Feature::MODEL_SERIALIZATION},
        {"TENSOR_INSPECTION", Log_Feature::TENSOR_INSPECTION},
        {"TENSORDUMP", Log_Feature::TENSOR_INSPECTION},
        {"LAYER_INSPECTION", Log_Feature::LAYER_INSPECTION},
        {"LAYERINSPECT", Log_Feature::LAYER_INSPECTION},
        {"FP16_METRICS", Log_Feature::FP16_METRICS},
        {"FP16", Log_Feature::FP16_METRICS},
        {"FP16_COMPUTE", Log_Feature::FP16_METRICS},
        {"MIXED_PRECISION", Log_Feature::FP16_METRICS},
        {"TERNARY_METRICS", Log_Feature::TERNARY_METRICS},
        {"TERNARY", Log_Feature::TERNARY_METRICS},
        {"TERNARY_COMPUTE", Log_Feature::TERNARY_METRICS},
        {"BITNET", Log_Feature::TERNARY_METRICS},
        {"STATIC_GRAPH", Log_Feature::STATIC_GRAPH},
        {"STATICGRAPH", Log_Feature::STATIC_GRAPH},
        {"PIPELINE_CACHE", Log_Feature::PIPELINE_CACHE},
        {"PIPELINECACHE", Log_Feature::PIPELINE_CACHE},
        {"SHADER_CACHE", Log_Feature::PIPELINE_CACHE},
        {"LLM_INFERENCE", Log_Feature::LLM_INFERENCE},
        {"LLMINFER", Log_Feature::LLM_INFERENCE},
        {"KV_CACHE", Log_Feature::LLM_INFERENCE},
        {"TOKENIZER", Log_Feature::TOKENIZER},
        {"TRANSFORMER", Log_Feature::TRANSFORMER},
        {"PROFILER", Log_Feature::PROFILER},
        {"PERF", Log_Feature::PROFILER},
        {"RL_AGENT", Log_Feature::RL_AGENT},
        {"RL", Log_Feature::RL_AGENT},
        {"CHECKPOINT_IO", Log_Feature::CHECKPOINT_IO},
        {"CHECKPOINT", Log_Feature::CHECKPOINT_IO},
        {"GRADIENT_HEALTH", Log_Feature::GRADIENT_HEALTH},
        {"GRADHEALTH", Log_Feature::GRADIENT_HEALTH},
        {"DATASET_IO", Log_Feature::DATASET_IO},
        {"DATASET", Log_Feature::DATASET_IO},
        {"HARDWARE", Log_Feature::HARDWARE},
        {"GRAPH", Log_Feature::GRAPH},
        {"OPERATIONS", Log_Feature::OPERATIONS},
        {"TRAINING", Log_Feature::TRAINING},
        {"LLM", Log_Feature::LLM},
        {"IO", Log_Feature::IO},
        {"ALL", Log_Feature::ALL}
    });

    for (const auto &kv : table)
    {
        if (kv.first == up)
        {
            return kv.second;
        }
    }
    return Log_Feature::NONE;
}

void enableWindowsVtOnce() noexcept
{
#if defined(_WIN32) || defined(_WIN64)
    static std::once_flag vt_flag;
    std::call_once(vt_flag, []() noexcept {
        for (DWORD std_id : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE})
        {
            HANDLE h = GetStdHandle(std_id);
            if (h != nullptr && h != INVALID_HANDLE_VALUE)
            {
                DWORD mode = 0;
                if (GetConsoleMode(h, &mode))
                {
                    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }
        }
    });
#endif
}

bool isStreamTty(bool _use_stderr) noexcept
{
    FILE *f = _use_stderr ? stderr : stdout;
#if defined(_WIN32) || defined(_WIN64)
    return _isatty(_fileno(f)) != 0;
#else
    return isatty(fileno(f)) != 0;
#endif
}

std::string_view basenameOf(const char *path) noexcept
{
    if (!path)
    {
        return "";
    }
    std::string_view sv(path);
    size_t pos = sv.find_last_of("/\\");
    return (pos == std::string_view::npos) ? sv : sv.substr(pos + 1);
}

std::string threadIdShort()
{
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}
}

Logger::Logger() = default;

Logger::~Logger()
{
    std::lock_guard<std::mutex> lock(logger_mutex);
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

void Logger::shutdownFlushAtExit() noexcept
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    if (instance.log_file_stream.is_open())
    {
        try
        {
            instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}] END OF LOG INSTANCE\n", timestamp(), "INFO", "General");
            instance.log_file_stream.flush();
            instance.log_file_stream.close();
            instance.lines_since_flush = 0;
        }
        catch (...)
        {
        }
    }
}

Logger &Logger::getInstance()
{
    static Logger *instance = []() {
        Logger *ptr = new Logger();
        std::atexit(&Logger::shutdownFlushAtExit);
        return ptr;
    }();
    return *instance;
}

std::string Logger::timestampFormatted(bool _include_ms)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    bool converted = false;
#if defined(_WIN32) || defined(_WIN64)
    converted = (localtime_s(&tm, &t_c) == 0);
#else
    converted = (localtime_r(&t_c, &tm) != nullptr);
#endif
    char buffer[32];
    if (!converted || std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm) == 0)
    {
        return _include_ms ? "0000-00-00 00:00:00.000" : "0000-00-00 00:00:00";
    }
    if (!_include_ms)
    {
        return std::string(buffer);
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    if (ms < 0)
    {
        ms += 1000;
    }
    return std::format("{}.{:03d}", buffer, static_cast<int>(ms));
}

std::string Logger::timestamp()
{
    return timestampFormatted(false);
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
    if ((_feature & Log_Feature::TERNARY_METRICS) != Log_Feature::NONE)
        append_tag("Ternary");

    if ((_feature & Log_Feature::STATIC_GRAPH) != Log_Feature::NONE)
        append_tag("StaticGraph");
    if ((_feature & Log_Feature::PIPELINE_CACHE) != Log_Feature::NONE)
        append_tag("PipelineCache");
    if ((_feature & Log_Feature::LLM_INFERENCE) != Log_Feature::NONE)
        append_tag("LLMInfer");
    if ((_feature & Log_Feature::TOKENIZER) != Log_Feature::NONE)
        append_tag("Tokenizer");
    if ((_feature & Log_Feature::TRANSFORMER) != Log_Feature::NONE)
        append_tag("Transformer");
    if ((_feature & Log_Feature::PROFILER) != Log_Feature::NONE)
        append_tag("Profiler");
    if ((_feature & Log_Feature::RL_AGENT) != Log_Feature::NONE)
        append_tag("RL");
    if ((_feature & Log_Feature::CHECKPOINT_IO) != Log_Feature::NONE)
        append_tag("Checkpoint");
    if ((_feature & Log_Feature::GRADIENT_HEALTH) != Log_Feature::NONE)
        append_tag("GradHealth");
    if ((_feature & Log_Feature::DATASET_IO) != Log_Feature::NONE)
        append_tag("Dataset");

    if (result.empty())
    {
        return "General";
    }
    return result;
}

bool Logger::openNewLogFile() noexcept
{
    try
    {
        std::error_code error_code;
        std::filesystem::create_directories(log_directory, error_code);
        if (error_code)
        {
            return false;
        }

        std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> log_files;
        if (std::filesystem::exists(log_directory, error_code) && !error_code)
        {
            std::filesystem::directory_iterator dir_it(log_directory, error_code);
            if (!error_code)
            {
                for (const auto &entry : dir_it)
                {
                    std::error_code entry_ec;
                    if (entry.is_regular_file(entry_ec) && !entry_ec &&
                        entry.path().extension() == ".log" &&
                        entry.path().filename().string().starts_with("log_"))
                    {
                        std::error_code time_ec;
                        auto mtime = std::filesystem::last_write_time(entry.path(), time_ec);
                        if (!time_ec)
                        {
                            log_files.emplace_back(entry.path(), mtime);
                        }
                    }
                }
            }
        }

        std::ranges::sort(log_files, [](const auto &_a, const auto &_b)
                          { return _a.second < _b.second; });

        while (log_files.size() >= MAX_LOG_FILES)
        {
            std::error_code remove_error;
            std::filesystem::remove(log_files.front().first, remove_error);
            log_files.erase(log_files.begin());
        }

        const auto now = std::chrono::system_clock::now();
        const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32) || defined(_WIN64)
        if (localtime_s(&tm, &t_c) != 0)
        {
            return false;
        }
#else
        if (localtime_r(&t_c, &tm) == nullptr)
        {
            return false;
        }
#endif
        char date_buffer[32];
        if (std::strftime(date_buffer, sizeof(date_buffer), "log_%Y%m%d_%H%M%S", &tm) == 0)
        {
            return false;
        }
        const std::string base_name = date_buffer;

        current_log_file = log_directory / (base_name + ".log");
        size_t suffix_index = 1;

        while (std::filesystem::exists(current_log_file, error_code) && !error_code)
        {
            current_log_file = log_directory / std::format("{}_{}.log", base_name, suffix_index++);
        }
        if (error_code)
        {
            return false;
        }

        log_file_stream.open(current_log_file, std::ios::out | std::ios::app);
        if (!log_file_stream.is_open())
        {
            return false;
        }

        log_file_stream << std::format("[{}] [{:<5}] [{:<18}] START OF LOG INSTANCE\n", timestamp(), "INFO", "General");
        log_file_stream.flush();
        if (!log_file_stream.good())
        {
            log_file_stream.close();
            return false;
        }
        return true;
    }
    catch (...)
    {
        try
        {
            if (log_file_stream.is_open())
            {
                log_file_stream.close();
            }
        }
        catch (...)
        {
        }
        return false;
    }
}

void Logger::initialize(const std::string &_directory)
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    if (instance.log_file_stream.is_open())
    {
        try
        {
            instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}] REDIRECTING LOG INSTANCE\n", timestamp(), "INFO", "General");
        }
        catch (...)
        {
        }
        instance.log_file_stream.close();
    }
    instance.log_directory = _directory;
    if (instance.is_file_logging_enabled.load(std::memory_order_relaxed))
    {
        if (!instance.openNewLogFile())
        {
            instance.is_file_logging_enabled.store(false, std::memory_order_relaxed);
        }
    }
}

bool Logger::shouldLog(Log_Level _level, Log_Feature _feature) noexcept
{
#if !VULKANML_LOGGING_ENABLED
    (void)_level;
    (void)_feature;
    return false;
#else
    Logger &instance = getInstance();
    const auto min_lvl = static_cast<uint8_t>(instance.min_log_level.load(std::memory_order_relaxed));
    if (static_cast<uint8_t>(_level) < min_lvl)
    {
        return false;
    }
    const bool filter_info = (_level == Log_Level::LOG_INFO) &&
                             instance.is_filter_info_by_feature_enabled.load(std::memory_order_relaxed);
    if (_level == Log_Level::LOG_DEBUG || filter_info)
    {
        const uint64_t current_features = instance.active_features.load(std::memory_order_relaxed);
        if (current_features == 0)
        {
            return false;
        }
        if (_feature != Log_Feature::NONE && ((current_features & static_cast<uint64_t>(_feature)) == 0))
        {
            return false;
        }
    }
    return true;
#endif
}

bool Logger::logMessageImpl(
    const std::string &_message,
    Log_Level _level,
    bool _print_to_console,
    size_t _repetition_count,
    Log_Feature _feature,
    const std::source_location _location)
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
    if (!shouldLog(_level, _feature))
    {
        return false;
    }

    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);

    if (_repetition_count > 0)
    {
        const std::string_view file_sv = _location.file_name() ? std::string_view(_location.file_name()) : std::string_view{};
        auto file_it = instance.call_site_counters.find(file_sv);
        if (file_it == instance.call_site_counters.end())
        {
            file_it = instance.call_site_counters.emplace(std::string(file_sv), std::unordered_map<std::uint_least32_t, size_t>{}).first;
        }
        size_t &current_count = file_it->second[_location.line()];
        if (current_count >= _repetition_count)
        {
            return false;
        }
        current_count++;
    }

    const bool inc_ms = instance.is_timestamp_ms_enabled.load(std::memory_order_relaxed);
    const bool inc_tid = instance.is_thread_id_enabled.load(std::memory_order_relaxed);
    const bool inc_loc = instance.is_source_location_enabled.load(std::memory_order_relaxed);

    Log_Record record{
        .timestamp = timestampFormatted(inc_ms),
        .level = _level,
        .feature = _feature,
        .message = _message,
        .file = _location.file_name() ? _location.file_name() : "",
        .line = _location.line()};

    if (instance.ring_buffer.size() >= MAX_RING_BUFFER_ENTRIES)
    {
        instance.ring_buffer.pop_front();
    }
    instance.ring_buffer.push_back(record);

    const std::string feat_str = featureToString(_feature);
    std::string extras;
    if (inc_tid)
    {
        extras += std::format(" [T:{}]", threadIdShort());
    }
    if (inc_loc)
    {
        extras += std::format(" ({}:{})", basenameOf(_location.file_name()), _location.line());
    }

    const auto file_min = static_cast<uint8_t>(instance.min_file_level.load(std::memory_order_relaxed));
    if (instance.is_file_logging_enabled.load(std::memory_order_relaxed) &&
        instance.log_file_stream.is_open() &&
        static_cast<uint8_t>(_level) >= file_min)
    {
        instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}]{} {}\n",
                                                record.timestamp,
                                                levelToString(_level),
                                                feat_str,
                                                extras,
                                                _message);
        if (_level >= Log_Level::LOG_WARNING || ++instance.lines_since_flush >= 64)
        {
            instance.log_file_stream.flush();
            instance.lines_since_flush = 0;
        }
        if (!instance.log_file_stream.good())
        {
            try
            {
                instance.log_file_stream.close();
            }
            catch (...)
            {
            }
            instance.lines_since_flush = 0;
            instance.is_file_logging_enabled.store(false, std::memory_order_relaxed);
        }
    }

    const bool force_console = instance.is_force_all_console_enabled.load(std::memory_order_relaxed);
    const bool quiet_mode = instance.is_quiet_mode_enabled.load(std::memory_order_relaxed);
    const bool console_enabled = instance.is_console_enabled.load(std::memory_order_relaxed);
    const auto console_min = static_cast<uint8_t>(instance.min_console_level.load(std::memory_order_relaxed));

    const bool console_allowed = (!quiet_mode || force_console) &&
                                 (force_console ||
                                  _level == Log_Level::LOG_ERROR ||
                                  (console_enabled && _print_to_console && static_cast<uint8_t>(_level) >= console_min));

    if (console_allowed)
    {
        const bool use_stderr = (_level == Log_Level::LOG_ERROR) &&
                                instance.is_errors_to_stderr_enabled.load(std::memory_order_relaxed);
        const Log_Color_Mode cm = instance.color_mode.load(std::memory_order_relaxed);
        bool use_color = false;
        if (cm == Log_Color_Mode::ALWAYS)
        {
            enableWindowsVtOnce();
            use_color = true;
        }
        else if (cm == Log_Color_Mode::AUTO && isStreamTty(use_stderr))
        {
            enableWindowsVtOnce();
            use_color = true;
        }

        std::ostream &out_stream = use_stderr ? std::cerr : std::cout;
        if (use_color)
        {
            out_stream << std::format("{}[{}] [{:<5}] [{:<18}]{} {}\033[0m\n",
                                      levelToAnsiColor(_level),
                                      record.timestamp,
                                      levelToString(_level),
                                      feat_str,
                                      extras,
                                      _message);
        }
        else
        {
            out_stream << std::format("[{}] [{:<5}] [{:<18}]{} {}\n",
                                      record.timestamp,
                                      levelToString(_level),
                                      feat_str,
                                      extras,
                                      _message);
        }
    }
    return true;
#endif
}

void Logger::logFp16TensorStats(
    std::string_view tensor_name,
    const std::vector<float> &data,
    size_t element_count,
    std::string_view data_type_name,
    Log_Level level,
    const std::source_location location)
{
#if !VULKANML_LOGGING_ENABLED
    (void)tensor_name;
    (void)data;
    (void)element_count;
    (void)data_type_name;
    (void)level;
    (void)location;
    return;
#else
    if (!shouldLog(level, Log_Feature::FP16_METRICS))
    {
        return;
    }
    const size_t n = std::min(element_count, data.size());
    if (n == 0)
    {
        return;
    }
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    double sum_val = 0.0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
    size_t subnormal_count = 0;

    for (size_t i = 0; i < n; ++i)
    {
        float val = data[i];
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
        sum_val += static_cast<double>(val);
    }

    const size_t valid_count = n - nan_count - inf_count;
    float mean_val = 0.0f;
    if (valid_count == 0)
    {
        min_val = 0.0f;
        max_val = 0.0f;
    }
    else
    {
        mean_val = static_cast<float>(sum_val / static_cast<double>(valid_count));
    }

    logMessage(
        Input_Format{"[FP16 Stats] {}: type={}, count={}, min={:.5e}, max={:.5e}, mean={:.5e}, zeros={}/{} ({:.1f}%), subnormals={}/{} ({:.1f}%), NaNs={}, Infs={}",
                     tensor_name, data_type_name, n, min_val, max_val, mean_val,
                     zero_count, n, (100.0f * static_cast<float>(zero_count) / static_cast<float>(n)),
                     subnormal_count, n, (100.0f * static_cast<float>(subnormal_count) / static_cast<float>(n)),
                     nan_count, inf_count},
        level,
        false,
        0,
        Log_Feature::FP16_METRICS,
        location);
#endif
}

void Logger::logTernaryTensorStats(
    std::string_view tensor_name,
    const std::vector<float> &data,
    size_t element_count,
    float gamma,
    Log_Level level,
    const std::source_location location)
{
#if !VULKANML_LOGGING_ENABLED
    (void)tensor_name;
    (void)data;
    (void)element_count;
    (void)gamma;
    (void)level;
    (void)location;
    return;
#else
    if (!shouldLog(level, Log_Feature::TERNARY_METRICS))
    {
        return;
    }
    const size_t n = std::min(element_count, data.size());
    if (n == 0)
    {
        return;
    }

    const float abs_gamma = (std::isfinite(gamma) && gamma > 0.0f) ? gamma : 1.0f;
    const float tol = std::max(1e-7f, 1e-3f * abs_gamma);

    size_t neg_count = 0;
    size_t zero_count = 0;
    size_t pos_count = 0;
    size_t other_count = 0;

    for (size_t i = 0; i < n; ++i)
    {
        float val = data[i];
        if (!std::isfinite(val))
        {
            other_count++;
        }
        else if (val == 0.0f)
        {
            zero_count++;
        }
        else if (val > 0.0f && (std::abs(val - 1.0f) < 1e-4f || std::abs(val - abs_gamma) <= tol))
        {
            pos_count++;
        }
        else if (val < 0.0f && (std::abs(val + 1.0f) < 1e-4f || std::abs(val + abs_gamma) <= tol))
        {
            neg_count++;
        }
        else
        {
            other_count++;
        }
    }

    float total_f = static_cast<float>(n);
    float pct_neg = 100.0f * static_cast<float>(neg_count) / total_f;
    float pct_zero = 100.0f * static_cast<float>(zero_count) / total_f;
    float pct_pos = 100.0f * static_cast<float>(pos_count) / total_f;

    logMessage(
        Input_Format{"[Ternary Stats] {}: count={}, gamma={:.5e}, {{-1}}: {} ({:.1f}%), {{0}}: {} ({:.1f}%), {{+1}}: {} ({:.1f}%), other={}",
                     tensor_name, n, gamma,
                     neg_count, pct_neg,
                     zero_count, pct_zero,
                     pos_count, pct_pos,
                     other_count},
        level,
        false,
        0,
        Log_Feature::TERNARY_METRICS,
        location);
#endif
}

void Logger::dumpRecentLogs(std::ostream &_output_stream)
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    if (instance.log_file_stream.is_open())
    {
        instance.log_file_stream.flush();
        instance.lines_since_flush = 0;
    }

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

void Logger::resetSpecificLog(const std::string &_file, std::uint_least32_t _line)
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    auto file_it = instance.call_site_counters.find(std::string_view(_file));
    if (file_it != instance.call_site_counters.end())
    {
        file_it->second.erase(_line);
        if (file_it->second.empty())
        {
            instance.call_site_counters.erase(file_it);
        }
    }
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
Log_Level Logger::getMinLogLevel() noexcept { return getInstance().min_log_level.load(std::memory_order_relaxed); }
Log_Level Logger::getConsoleMinLevel() noexcept { return getInstance().min_console_level.load(std::memory_order_relaxed); }
Log_Level Logger::getFileMinLevel() noexcept { return getInstance().min_file_level.load(std::memory_order_relaxed); }
bool Logger::isFilterInfoByFeatureEnabled() noexcept { return getInstance().is_filter_info_by_feature_enabled.load(std::memory_order_relaxed); }
bool Logger::isForceAllConsoleOutputEnabled() noexcept { return getInstance().is_force_all_console_enabled.load(std::memory_order_relaxed); }
bool Logger::isFileLoggingEnabled() noexcept { return getInstance().is_file_logging_enabled.load(std::memory_order_relaxed); }
bool Logger::isConsoleEnabled() noexcept { return getInstance().is_console_enabled.load(std::memory_order_relaxed); }
bool Logger::isQuietModeEnabled() noexcept { return getInstance().is_quiet_mode_enabled.load(std::memory_order_relaxed); }
bool Logger::isErrorsToStderrEnabled() noexcept { return getInstance().is_errors_to_stderr_enabled.load(std::memory_order_relaxed); }
bool Logger::isIncludeTimestampMsEnabled() noexcept { return getInstance().is_timestamp_ms_enabled.load(std::memory_order_relaxed); }
bool Logger::isIncludeThreadIdEnabled() noexcept { return getInstance().is_thread_id_enabled.load(std::memory_order_relaxed); }
bool Logger::isIncludeSourceLocationEnabled() noexcept { return getInstance().is_source_location_enabled.load(std::memory_order_relaxed); }
Log_Color_Mode Logger::getColorMode() noexcept { return getInstance().color_mode.load(std::memory_order_relaxed); }

void Logger::flush() noexcept
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    if (instance.log_file_stream.is_open())
    {
        try
        {
            instance.log_file_stream.flush();
            instance.lines_since_flush = 0;
        }
        catch (...)
        {
        }
    }
}

void Logger::enableFileLogging(bool _enable) noexcept
{
    Logger &instance = getInstance();
    std::lock_guard<std::mutex> lock(instance.logger_mutex);
    if (_enable)
    {
        bool ready = instance.log_file_stream.is_open();
        if (!ready)
        {
            ready = instance.openNewLogFile();
        }
        instance.lines_since_flush = 0;
        instance.is_file_logging_enabled.store(ready, std::memory_order_relaxed);
    }
    else
    {
        instance.is_file_logging_enabled.store(false, std::memory_order_relaxed);
        if (instance.log_file_stream.is_open())
        {
            try
            {
                instance.log_file_stream << std::format("[{}] [{:<5}] [{:<18}] PAUSING LOG INSTANCE\n", timestamp(), "INFO", "General");
                instance.log_file_stream.flush();
                instance.log_file_stream.close();
                instance.lines_since_flush = 0;
            }
            catch (...)
            {
            }
        }
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

void Logger::setMinLogLevel(Log_Level _level) noexcept
{
    if (static_cast<uint8_t>(_level) <= static_cast<uint8_t>(Log_Level::LOG_LEVEL_END))
    {
        getInstance().min_log_level.store(_level, std::memory_order_relaxed);
    }
}

void Logger::setConsoleMinLevel(Log_Level _level) noexcept
{
    if (static_cast<uint8_t>(_level) <= static_cast<uint8_t>(Log_Level::LOG_LEVEL_END))
    {
        getInstance().min_console_level.store(_level, std::memory_order_relaxed);
    }
}

void Logger::setFileMinLevel(Log_Level _level) noexcept
{
    if (static_cast<uint8_t>(_level) <= static_cast<uint8_t>(Log_Level::LOG_LEVEL_END))
    {
        getInstance().min_file_level.store(_level, std::memory_order_relaxed);
    }
}

void Logger::setFilterInfoByFeature(bool _enable) noexcept { getInstance().is_filter_info_by_feature_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setLogDirectory(const std::string &_directory) { initialize(_directory); }
void Logger::setOnlyActiveFeatures(Log_Feature _feature_mask) noexcept { getInstance().active_features.store(static_cast<uint64_t>(_feature_mask), std::memory_order_relaxed); }
void Logger::setForceAllConsoleOutput(bool _enable) noexcept { getInstance().is_force_all_console_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setConsoleOutput(bool _enable) noexcept { getInstance().is_console_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setQuietMode(bool _enable) noexcept { getInstance().is_quiet_mode_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setErrorsToStderr(bool _enable) noexcept { getInstance().is_errors_to_stderr_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setColorMode(Log_Color_Mode _mode) noexcept { getInstance().color_mode.store(_mode, std::memory_order_relaxed); }
void Logger::setIncludeTimestampMs(bool _enable) noexcept { getInstance().is_timestamp_ms_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setIncludeThreadId(bool _enable) noexcept { getInstance().is_thread_id_enabled.store(_enable, std::memory_order_relaxed); }
void Logger::setIncludeSourceLocation(bool _enable) noexcept { getInstance().is_source_location_enabled.store(_enable, std::memory_order_relaxed); }

bool Logger::parseLevel(std::string_view _text, Log_Level &_out_level) noexcept
{
    const std::string up = toUpperAscii(trimAscii(_text));
    if (up == "DEBUG" || up == "LOG_DEBUG" || up == "0")
    {
        _out_level = Log_Level::LOG_DEBUG;
        return true;
    }
    if (up == "INFO" || up == "LOG_INFO" || up == "1")
    {
        _out_level = Log_Level::LOG_INFO;
        return true;
    }
    if (up == "WARN" || up == "WARNING" || up == "LOG_WARNING" || up == "2")
    {
        _out_level = Log_Level::LOG_WARNING;
        return true;
    }
    if (up == "ERR" || up == "ERROR" || up == "LOG_ERROR" || up == "3")
    {
        _out_level = Log_Level::LOG_ERROR;
        return true;
    }
    if (up == "OFF" || up == "NONE" || up == "END" || up == "LOG_LEVEL_END" || up == "4")
    {
        _out_level = Log_Level::LOG_LEVEL_END;
        return true;
    }
    return false;
}

bool Logger::parseColorMode(std::string_view _text, Log_Color_Mode &_out_mode) noexcept
{
    const std::string up = toUpperAscii(trimAscii(_text));
    if (up == "AUTO" || up == "DEFAULT")
    {
        _out_mode = Log_Color_Mode::AUTO;
        return true;
    }
    if (up == "ALWAYS" || up == "ON" || up == "1" || up == "TRUE" || up == "YES")
    {
        _out_mode = Log_Color_Mode::ALWAYS;
        return true;
    }
    if (up == "NEVER" || up == "OFF" || up == "NONE" || up == "0" || up == "FALSE" || up == "NO")
    {
        _out_mode = Log_Color_Mode::NEVER;
        return true;
    }
    return false;
}

Log_Feature Logger::parseFeatureMask(std::string_view _csv, Log_Feature _base) noexcept
{
    uint64_t mask = static_cast<uint64_t>(_base);
    size_t pos = 0;
    while (pos < _csv.size())
    {
        size_t next = _csv.find_first_of(",;| \t\r\n", pos);
        if (next == std::string_view::npos)
        {
            next = _csv.size();
        }
        std::string_view token = trimAscii(_csv.substr(pos, next - pos));
        pos = next + 1;
        if (token.empty())
        {
            continue;
        }
        char op = 0;
        if (token.front() == '+' || token.front() == '-')
        {
            op = token.front();
            token.remove_prefix(1);
            token = trimAscii(token);
        }
        const std::string up = toUpperAscii(token);
        if (up.empty())
        {
            continue;
        }
        if (up == "NONE")
        {
            if (op == 0)
            {
                mask = 0;
            }
            continue;
        }
        const Log_Feature f = lookupSingleFeatureToken(up);
        const uint64_t bits = static_cast<uint64_t>(f);
        if (op == '-')
        {
            mask &= ~bits;
        }
        else
        {
            mask |= bits;
        }
    }
    return static_cast<Log_Feature>(mask);
}

void Logger::configure(const Logger_Config &_config)
{
    if (!_config.log_directory.empty() && _config.log_directory != getLogDirectory())
    {
        initialize(_config.log_directory);
    }

    if (!_config.enable_logging)
    {
        setMinLogLevel(Log_Level::LOG_LEVEL_END);
        setConsoleMinLevel(_config.console_min_level);
        setFileMinLevel(_config.file_min_level);
        setOnlyActiveFeatures(Log_Feature::NONE);
        setFilterInfoByFeature(_config.filter_info_by_feature);
        setForceAllConsoleOutput(false);
        setQuietMode(true);
        setErrorsToStderr(_config.errors_to_stderr);
        setColorMode(_config.color_mode);
        setIncludeTimestampMs(_config.include_timestamp_ms);
        setIncludeThreadId(_config.include_thread_id);
        setIncludeSourceLocation(_config.include_source_location);
        setConsoleOutput(false);
        enableFileLogging(false);
        return;
    }

    setMinLogLevel(_config.min_level);
    setConsoleMinLevel(_config.console_min_level);
    setFileMinLevel(_config.file_min_level);
    setOnlyActiveFeatures(_config.active_features);
    setFilterInfoByFeature(_config.filter_info_by_feature);
    setForceAllConsoleOutput(_config.force_all_console);
    setQuietMode(_config.quiet_mode);
    setErrorsToStderr(_config.errors_to_stderr);
    setColorMode(_config.color_mode);
    setIncludeTimestampMs(_config.include_timestamp_ms);
    setIncludeThreadId(_config.include_thread_id);
    setIncludeSourceLocation(_config.include_source_location);
    setConsoleOutput(_config.console_enabled);
    enableFileLogging(_config.file_enabled);
}

Logger_Config Logger::getConfig()
{
    Logger_Config cfg;
    cfg.min_level = getMinLogLevel();
    cfg.enable_logging = (cfg.min_level != Log_Level::LOG_LEVEL_END);
    cfg.console_min_level = getConsoleMinLevel();
    cfg.file_min_level = getFileMinLevel();
    cfg.active_features = getActiveFeatures();
    cfg.filter_info_by_feature = isFilterInfoByFeatureEnabled();
    cfg.console_enabled = isConsoleEnabled();
    cfg.file_enabled = isFileLoggingEnabled();
    cfg.force_all_console = isForceAllConsoleOutputEnabled();
    cfg.quiet_mode = isQuietModeEnabled();
    cfg.errors_to_stderr = isErrorsToStderrEnabled();
    cfg.include_timestamp_ms = isIncludeTimestampMsEnabled();
    cfg.include_thread_id = isIncludeThreadIdEnabled();
    cfg.include_source_location = isIncludeSourceLocationEnabled();
    cfg.color_mode = getColorMode();
    cfg.log_directory = getLogDirectory();
    return cfg;
}

void Logger::applyPreferences(const Logging_Preferences &_prefs)
{
    Logger_Config cfg = getConfig();
    cfg.enable_logging = _prefs.enable_logging;

    Log_Level parsed_lvl = cfg.min_level;
    if (parseLevel(_prefs.level, parsed_lvl))
    {
        cfg.min_level = parsed_lvl;
    }
    if (parseLevel(_prefs.console_level, parsed_lvl))
    {
        cfg.console_min_level = parsed_lvl;
    }
    if (parseLevel(_prefs.file_level, parsed_lvl))
    {
        cfg.file_min_level = parsed_lvl;
    }

    Log_Color_Mode parsed_color = cfg.color_mode;
    if (parseColorMode(_prefs.color_mode, parsed_color))
    {
        cfg.color_mode = parsed_color;
    }

    cfg.console_enabled = _prefs.console_enabled;
    cfg.file_enabled = _prefs.file_enabled;
    if (!_prefs.log_directory.empty())
    {
        cfg.log_directory = _prefs.log_directory;
    }
    cfg.quiet_mode = _prefs.quiet_mode;
    cfg.errors_to_stderr = _prefs.errors_to_stderr;
    cfg.filter_info_by_feature = _prefs.filter_info_by_feature;
    cfg.include_timestamp_ms = _prefs.include_timestamp_ms;
    cfg.include_thread_id = _prefs.include_thread_id;
    cfg.include_source_location = _prefs.include_source_location;

    if (_prefs.features.empty())
    {
        cfg.active_features = Log_Feature::NONE;
    }
    else
    {
        Log_Feature mask = Log_Feature::NONE;
        for (const auto &feat_name : _prefs.features)
        {
            mask = parseFeatureMask(feat_name, mask);
        }
        cfg.active_features = mask;
    }

    configure(cfg);
}

void Logger::configureFromEnvironment()
{
    if (const std::string val = getEnvString("VULKANML_LOG_DIR"); !val.empty())
    {
        initialize(val);
    }
    if (const std::string val = getEnvString("VULKANML_LOG_LEVEL"); !val.empty())
    {
        Log_Level lvl{};
        if (parseLevel(val, lvl))
        {
            setMinLogLevel(lvl);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_CONSOLE_LEVEL"); !val.empty())
    {
        Log_Level lvl{};
        if (parseLevel(val, lvl))
        {
            setConsoleMinLevel(lvl);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_FILE_LEVEL"); !val.empty())
    {
        Log_Level lvl{};
        if (parseLevel(val, lvl))
        {
            setFileMinLevel(lvl);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_FEATURES"); !val.empty())
    {
        std::string_view trimmed = trimAscii(val);
        Log_Feature base = (!trimmed.empty() && (trimmed.front() == '+' || trimmed.front() == '-'))
                               ? getActiveFeatures()
                               : Log_Feature::NONE;
        setOnlyActiveFeatures(parseFeatureMask(trimmed, base));
    }
    if (const std::string val = getEnvString("VULKANML_LOG_FILTER_INFO"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            setFilterInfoByFeature(b);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_FILE"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            enableFileLogging(b);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_COLOR"); !val.empty())
    {
        Log_Color_Mode mode{};
        if (parseColorMode(val, mode))
        {
            setColorMode(mode);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_QUIET"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            setQuietMode(b);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_STDERR"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            setErrorsToStderr(b);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_MS"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            setIncludeTimestampMs(b);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_TID"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            setIncludeThreadId(b);
        }
    }
    if (const std::string val = getEnvString("VULKANML_LOG_LOC"); !val.empty())
    {
        bool b = false;
        if (parseBoolText(val, b))
        {
            setIncludeSourceLocation(b);
        }
    }
}

int Logger::loggerCliFlagArity(std::string_view _arg) noexcept
{
    if (_arg == "--log-filter-info" ||
        _arg == "--no-log-filter-info" ||
        _arg == "--log-file" ||
        _arg == "--no-log-file" ||
        _arg == "--no-color" ||
        _arg == "--log-quiet" ||
        _arg == "--no-log-quiet" ||
        _arg == "--log-verbose" ||
        _arg == "--log-stderr" ||
        _arg == "--log-ms" ||
        _arg == "--log-thread-id" ||
        _arg == "--log-location")
    {
        return 1;
    }

    if (_arg == "--log-level" ||
        _arg == "--log-console-level" ||
        _arg == "--log-file-level" ||
        _arg == "--log-features" ||
        _arg == "--log-dir" ||
        _arg == "--log-color")
    {
        return 2;
    }

    if (_arg.starts_with("--log-level=") ||
        _arg.starts_with("--log-console-level=") ||
        _arg.starts_with("--log-file-level=") ||
        _arg.starts_with("--log-features=") ||
        _arg.starts_with("--log-dir=") ||
        _arg.starts_with("--log-color="))
    {
        return 1;
    }

    return 0;
}

bool Logger::isLoggerCliFlag(std::string_view _arg) noexcept
{
    return loggerCliFlagArity(_arg) > 0;
}

void Logger::applyCliArgs(int argc, char *argv[])
{
    if (argc <= 1 || argv == nullptr)
    {
        return;
    }

    auto readValue = [&](int &i, std::string_view flag, std::string_view prefix) -> std::string_view
    {
        if (flag.starts_with(prefix) && flag.size() > prefix.size() && flag[prefix.size()] == '=')
        {
            return flag.substr(prefix.size() + 1);
        }
        if (flag == prefix && (i + 1) < argc && argv[i + 1] != nullptr)
        {
            return std::string_view(argv[++i]);
        }
        return "";
    };

    for (int i = 1; i < argc; ++i)
    {
        if (argv[i] == nullptr)
        {
            continue;
        }
        const std::string_view arg(argv[i]);

        if (arg == "--log-level" || arg.starts_with("--log-level="))
        {
            const std::string_view val = readValue(i, arg, "--log-level");
            Log_Level lvl{};
            if (!val.empty() && parseLevel(val, lvl))
            {
                setMinLogLevel(lvl);
            }
        }
        else if (arg == "--log-console-level" || arg.starts_with("--log-console-level="))
        {
            const std::string_view val = readValue(i, arg, "--log-console-level");
            Log_Level lvl{};
            if (!val.empty() && parseLevel(val, lvl))
            {
                setConsoleMinLevel(lvl);
            }
        }
        else if (arg == "--log-file-level" || arg.starts_with("--log-file-level="))
        {
            const std::string_view val = readValue(i, arg, "--log-file-level");
            Log_Level lvl{};
            if (!val.empty() && parseLevel(val, lvl))
            {
                setFileMinLevel(lvl);
            }
        }
        else if (arg == "--log-features" || arg.starts_with("--log-features="))
        {
            const std::string_view val = trimAscii(readValue(i, arg, "--log-features"));
            if (!val.empty())
            {
                Log_Feature base = (val.front() == '+' || val.front() == '-')
                                       ? getActiveFeatures()
                                       : Log_Feature::NONE;
                setOnlyActiveFeatures(parseFeatureMask(val, base));
            }
        }
        else if (arg == "--log-dir" || arg.starts_with("--log-dir="))
        {
            const std::string_view val = readValue(i, arg, "--log-dir");
            if (!val.empty())
            {
                initialize(std::string(val));
            }
        }
        else if (arg == "--log-color" || arg.starts_with("--log-color="))
        {
            const std::string_view val = readValue(i, arg, "--log-color");
            Log_Color_Mode mode{};
            if (!val.empty() && parseColorMode(val, mode))
            {
                setColorMode(mode);
            }
        }
        else if (arg == "--no-color")
        {
            setColorMode(Log_Color_Mode::NEVER);
        }
        else if (arg == "--log-filter-info")
        {
            setFilterInfoByFeature(true);
        }
        else if (arg == "--no-log-filter-info")
        {
            setFilterInfoByFeature(false);
        }
        else if (arg == "--log-file")
        {
            enableFileLogging(true);
        }
        else if (arg == "--no-log-file")
        {
            enableFileLogging(false);
        }
        else if (arg == "--log-quiet")
        {
            setQuietMode(true);
        }
        else if (arg == "--no-log-quiet")
        {
            setQuietMode(false);
        }
        else if (arg == "--log-verbose")
        {
            setQuietMode(false);
            setConsoleOutput(true);
            setMinLogLevel(Log_Level::LOG_DEBUG);
            setConsoleMinLevel(Log_Level::LOG_DEBUG);
            setForceAllConsoleOutput(true);
            if (getActiveFeatures() == Log_Feature::NONE)
            {
                setOnlyActiveFeatures(Log_Feature::ALL);
            }
        }
        else if (arg == "--log-stderr")
        {
            setErrorsToStderr(true);
        }
        else if (arg == "--log-ms")
        {
            setIncludeTimestampMs(true);
        }
        else if (arg == "--log-thread-id")
        {
            setIncludeThreadId(true);
        }
        else if (arg == "--log-location")
        {
            setIncludeSourceLocation(true);
        }
    }
}

std::string Logger::loggerCliHelpText()
{
    return "Logger Options:\n"
           "  --log-level <lvl>         Set minimum log level (debug|info|warn|error|off)\n"
           "  --log-console-level <lvl> Set console minimum level\n"
           "  --log-file-level <lvl>    Set log file minimum level\n"
           "  --log-features <csv>      Comma-separated feature flags (+FEAT,-FEAT,ALL,NONE)\n"
           "  --log-filter-info         Also apply feature filter to LOG_INFO messages\n"
           "  --no-log-filter-info      Disable feature filtering on LOG_INFO\n"
           "  --log-file / --no-log-file Enable/disable file logging\n"
           "  --log-dir <dir>           Set log output directory (default: logs)\n"
           "  --log-color <mode>        ANSI colors (auto|always|never)\n"
           "  --no-color                Disable ANSI colors\n"
           "  --log-quiet / --no-log-quiet Suppress/restore console log output\n"
           "  --log-verbose             Enable debug console output for all features\n"
           "  --log-stderr              Route LOG_ERROR output to stderr\n"
           "  --log-ms                  Include milliseconds in timestamps\n"
           "  --log-thread-id           Include thread ID in log entries\n"
           "  --log-location            Include file:line source location\n";
}

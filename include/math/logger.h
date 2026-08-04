#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

enum Log_Level
{
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_DEBUG,
    LOG_LEVEL_END
};

class Logger
{
private:
    static constexpr std::size_t MAX_LOG_FILES = 15;

    std::filesystem::path log_directory;
    std::filesystem::path current_log_file;
    std::ofstream file;


    Logger() : log_directory("logs") { openNewLogFile(); }

    ~Logger()
    {
        if (file.is_open())
        {
            try
            {
                file << std::format("[{}] [INFO] END OF LOG INSTANCE\n", timestamp());
                file.flush();
            }
            catch (...) {}
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
        auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        return std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::zoned_time{std::chrono::current_zone(), now});
    }

    void openNewLogFile()
    {
        std::filesystem::create_directories(log_directory);

        std::vector<std::filesystem::path> logs;
        for (const auto &entry : std::filesystem::directory_iterator(log_directory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".log" &&
                entry.path().filename().string().starts_with("log_"))
            {
                logs.push_back(entry.path());
            }
        }

        std::ranges::sort(logs, [](const auto &a, const auto &b)
                          { return std::filesystem::last_write_time(a) < std::filesystem::last_write_time(b); });

        while (logs.size() >= MAX_LOG_FILES)
        {
            std::error_code ec;
            std::filesystem::remove(logs.front(), ec);
            logs.erase(logs.begin());
        }

        const std::string base_name = std::format("log_{:%Y%m%d_%H%M%S}",
                                           std::chrono::zoned_time{std::chrono::current_zone(), std::chrono::system_clock::now()});
        current_log_file = log_directory / (base_name + ".log");
        std::size_t suffix = 1;

        while (std::filesystem::exists(current_log_file))
        {
            current_log_file = log_directory / std::format("{}_{}.log", base_name, suffix++);
        }

        file.open(current_log_file, std::ios::out | std::ios::app);
        if (!file.is_open())
            throw std::runtime_error("Failed to open log file: " + current_log_file.string());

        file << std::format("[{}] [INFO] START OF LOG INSTANCE\n", timestamp());
        file.flush();
    }

public:
    static void init(const std::string &directory)
    {
        Logger &instance = getInstance();
        if (instance.file.is_open())
        {
            instance.file << std::format("[{}] [INFO] REDIRECTING LOG INSTANCE\n", timestamp());
            instance.file.close();
        }
        instance.log_directory = directory;
        instance.openNewLogFile();
    }

    static void logMessage(const std::string &message, Log_Level level = LOG_INFO, bool print_terminal = false)
    {
        Logger &instance = getInstance();

        if (!instance.file.is_open())
            throw std::runtime_error("Logger is not initialized");

        constexpr std::string_view level_strings[] = {"INFO", "WARNING", "ERROR", "DEBUG"};

        const std::string line = std::format("[{}] [{}] {}\n",
                                             timestamp(),
                                             level_strings[level],
                                             message);

        instance.file << line;
        instance.file.flush();

        if (print_terminal || level == LOG_DEBUG)
            std::cout << line;
    }

    static std::string getCurrentLogFilepath() { return getInstance().current_log_file.string(); }
};
#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "helper/logger.h"

namespace lr_internal
{

inline void throwInvalid(const char *msg, const char *log_ctx = nullptr)
{
    if (log_ctx != nullptr)
    {
        Logger::logMessage(std::string(log_ctx) + ": " + msg,
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::LR_SCHEDULER);
    }
    throw std::invalid_argument(msg);
}

inline void requirePositiveFinite(float val, const char *msg, const char *log_ctx = nullptr)
{
    if (!std::isfinite(val) || val <= 0.0f)
    {
        throwInvalid(msg, log_ctx);
    }
}

inline void requireNonNegativeFinite(float val, const char *msg, const char *log_ctx = nullptr)
{
    if (!std::isfinite(val) || val < 0.0f)
    {
        throwInvalid(msg, log_ctx);
    }
}

inline void requireFinite(float val, const char *msg)
{
    if (!std::isfinite(val))
    {
        throw std::invalid_argument(msg);
    }
}

inline void requireMinLrInBounds(float min_lr, float max_lr, const char *msg, const char *log_ctx = nullptr)
{
    if (!std::isfinite(min_lr) || min_lr < 0.0f || min_lr > max_lr)
    {
        throwInvalid(msg, log_ctx);
    }
}

inline void requireUnitOpenInterval(float val, const char *msg, const char *log_ctx = nullptr)
{
    if (!std::isfinite(val) || val <= 0.0f || val >= 1.0f)
    {
        throwInvalid(msg, log_ctx);
    }
}

inline void requireLearningRateSetterBounds(float lr, float min_lr, const char *msg)
{
    if (!std::isfinite(lr) || lr <= 0.0f || min_lr > lr)
    {
        throw std::invalid_argument(msg);
    }
}

inline void requirePositiveInt(int val, const char *msg, const char *log_ctx = nullptr)
{
    if (val <= 0)
    {
        throwInvalid(msg, log_ctx);
    }
}

inline void requireNonNegativeInt(int val, const char *msg, const char *log_ctx = nullptr)
{
    if (val < 0)
    {
        throwInvalid(msg, log_ctx);
    }
}

inline void requireOpenOut(const std::ofstream &os, const char *ctx)
{
    if (!os.is_open())
    {
        throw std::runtime_error(std::string(ctx) + ": Output stream is not open");
    }
}

inline void requireOpenIn(const std::ifstream &is, const char *ctx)
{
    if (!is.is_open())
    {
        throw std::runtime_error(std::string(ctx) + ": Input stream is not open");
    }
}

inline void requireGoodStream(const std::ios &stream, const char *err_msg)
{
    if (!stream.good())
    {
        throw std::runtime_error(err_msg);
    }
}

template <typename... Ts>
inline void writePods(std::ofstream &os, const char *ctx, const Ts &...vals)
{
    requireOpenOut(os, ctx);
    (os.write(reinterpret_cast<const char *>(&vals), sizeof(vals)), ...);
    if (!os.good())
    {
        throw std::runtime_error(std::string(ctx) + ": Failed to write checkpoint");
    }
}

template <typename... Ts>
inline void readPodsOrThrow(std::ifstream &is, const char *err_msg, Ts &...vals)
{
    (is.read(reinterpret_cast<char *>(&vals), sizeof(vals)), ...);
    if (!is.good())
    {
        throw std::runtime_error(err_msg);
    }
}

inline bool isValidBaseParams(float lr, float min_lr, float cur_lr, int cur_ep) noexcept
{
    return std::isfinite(lr) && lr > 0.0f &&
           std::isfinite(min_lr) && min_lr >= 0.0f && min_lr <= lr &&
           std::isfinite(cur_lr) && cur_lr >= 0.0f &&
           cur_ep >= 0;
}

inline void logClampedBelowMin(const char *ctx)
{
    Logger::logMessage(std::string(ctx) + ": Learning rate decayed below minimum_learning_rate, clamped.",
                       Log_Level::LOG_DEBUG, false, 0, Log_Feature::LR_SCHEDULER);
}

}

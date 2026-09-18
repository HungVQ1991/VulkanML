#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <string_view>
#include <vector>

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#endif

enum class Data_Type
{
    FLOAT32,
    FLOAT16
};

#if defined(__STDCPP_FLOAT16_T__)
using float16_t = std::float16_t;
#else
struct float16_t
{
    std::uint16_t value = 0;
};
#endif

constexpr std::size_t getDataTypeSize(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return 2;
    case Data_Type::FLOAT32:
    default:
        return 4;
    }
}

constexpr std::string_view getDataTypeGlslName(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return "float16_t";
    case Data_Type::FLOAT32:
    default:
        return "float";
    }
}

constexpr std::string_view getDataTypeName(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return "FLOAT16";
    case Data_Type::FLOAT32:
    default:
        return "FLOAT32";
    }
}

inline void convertFp32ToFp16(const float *_source, float16_t *_destination, std::size_t _count) noexcept
{
#if defined(__STDCPP_FLOAT16_T__)
    for (std::size_t i = 0; i < _count; ++i)
    {
        _destination[i] = static_cast<std::float16_t>(_source[i]);
    }
#else
    for (std::size_t i = 0; i < _count; ++i)
    {
        std::uint32_t x = 0;
        std::memcpy(&x, &_source[i], sizeof(float));
        std::uint32_t sign = (x >> 31) & 0x1;
        std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
        std::uint32_t mantissa = x & 0x7FFFFF;
        std::uint16_t h = 0;
        if (exp <= 0)
        {
            h = static_cast<std::uint16_t>(sign << 15);
        }
        else if (exp >= 31)
        {
            h = static_cast<std::uint16_t>((sign << 15) | 0x7C00);
        }
        else
        {
            h = static_cast<std::uint16_t>((sign << 15) | (exp << 10) | (mantissa >> 13));
        }
        _destination[i].value = h;
    }
#endif
}

inline void convertFp16ToFp32(const float16_t *_source, float *_destination, std::size_t _count) noexcept
{
#if defined(__STDCPP_FLOAT16_T__)
    for (std::size_t i = 0; i < _count; ++i)
    {
        _destination[i] = static_cast<float>(_source[i]);
    }
#else
    for (std::size_t i = 0; i < _count; ++i)
    {
        std::uint16_t h = _source[i].value;
        std::uint32_t sign = (h >> 15) & 0x1;
        std::uint32_t exp = (h >> 10) & 0x1F;
        std::uint32_t mantissa = h & 0x3FF;
        std::uint32_t f = 0;
        if (exp == 0)
        {
            f = sign << 31;
        }
        else if (exp == 31)
        {
            f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        }
        else
        {
            f = (sign << 31) | ((exp - 15 + 127) << 23) | (mantissa << 13);
        }
        std::memcpy(&_destination[i], &f, sizeof(float));
    }
#endif
}

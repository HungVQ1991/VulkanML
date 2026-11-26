#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

#include "math/shape.h"

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#endif

enum class Data_Type
{
    FLOAT32,
    FLOAT16,
    TERNARY_INT2
};

inline uint16_t floatToHalfBits(float _value) noexcept
{
    uint32_t x = 0;
    std::memcpy(&x, &_value, sizeof(float));
    uint32_t sign = (x >> 31) & 1u;
    uint32_t exp_fp32 = (x >> 23) & 0xFFu;
    uint32_t mantissa = x & 0x7FFFFFu;

    if (exp_fp32 == 0xFFu)
    {
        if (mantissa != 0u)
        {
            return static_cast<uint16_t>((sign << 15) | 0x7E00u | (mantissa >> 13));
        }
        return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    }

    int32_t exp_unbiased = static_cast<int32_t>(exp_fp32) - 127;
    if (exp_unbiased > 15)
    {
        return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    }

    if (exp_unbiased >= -14)
    {
        uint32_t exp16 = static_cast<uint32_t>(exp_unbiased + 15);
        uint32_t m16 = mantissa >> 13;
        uint32_t rem = mantissa & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1u)))
        {
            ++m16;
            if (m16 == 0x400u)
            {
                m16 = 0u;
                ++exp16;
                if (exp16 >= 31u)
                {
                    return static_cast<uint16_t>((sign << 15) | 0x7C00u);
                }
            }
        }
        return static_cast<uint16_t>((sign << 15) | (exp16 << 10) | m16);
    }

    if (exp_unbiased >= -25)
    {
        uint32_t sig = 0x00800000u | mantissa;
        uint32_t shift = static_cast<uint32_t>(126 - static_cast<int32_t>(exp_fp32));
        uint32_t m16 = sig >> shift;
        uint32_t rem = sig & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (m16 & 1u)))
        {
            ++m16;
        }
        return static_cast<uint16_t>((sign << 15) | m16);
    }

    return static_cast<uint16_t>(sign << 15);
}

inline float halfBitsToFloat(uint16_t _h) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(_h) >> 15) & 1u;
    uint32_t exp = (static_cast<uint32_t>(_h) >> 10) & 0x1Fu;
    uint32_t mantissa = static_cast<uint32_t>(_h) & 0x3FFu;
    uint32_t f = 0;

    if (exp == 0u)
    {
        if (mantissa == 0u)
        {
            f = sign << 31;
        }
        else
        {
            uint32_t e = 113u;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0u)
            {
                m <<= 1u;
                --e;
            }
            m &= 0x3FFu;
            f = (sign << 31) | (e << 23) | (m << 13);
        }
    }
    else if (exp == 31u)
    {
        f = (sign << 31) | 0x7F800000u | (mantissa << 13);
    }
    else
    {
        f = (sign << 31) | ((exp + 112u) << 23) | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

#if defined(__STDCPP_FLOAT16_T__)
using float16_t = std::float16_t;
#else
struct float16_t
{
    uint16_t value = 0;
    constexpr float16_t() noexcept = default;
    explicit float16_t(float _v) noexcept : value(floatToHalfBits(_v)) {}
    explicit operator float() const noexcept { return halfBitsToFloat(value); }
};
#endif

constexpr bool isSubBytePacked(Data_Type _type) noexcept
{
    return _type == Data_Type::TERNARY_INT2;
}

constexpr size_t getDataTypeBitWidth(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return 16;
    case Data_Type::TERNARY_INT2:
        return 2;
    case Data_Type::FLOAT32:
        return 32;
    }
    std::unreachable();
}

constexpr size_t getDataTypeStorageWordBytes(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return 2;
    case Data_Type::TERNARY_INT2:
        return 4;
    case Data_Type::FLOAT32:
        return 4;
    }
    std::unreachable();
}

constexpr size_t getElementsPerStorageWord(Data_Type _type) noexcept
{
    return (_type == Data_Type::TERNARY_INT2) ? 16 : 1;
}

constexpr size_t getElementCountFromStorageBytes(Data_Type _type, size_t _byte_size) noexcept
{
    return (_byte_size / getDataTypeStorageWordBytes(_type)) * getElementsPerStorageWord(_type);
}

constexpr size_t getDataTypeSize(Data_Type _type) noexcept
{
    return getDataTypeStorageWordBytes(_type);
}

constexpr std::string_view getDataTypeGlslName(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return "float16_t";
    case Data_Type::TERNARY_INT2:
        return "uint";
    case Data_Type::FLOAT32:
        return "float";
    }
    std::unreachable();
}

constexpr std::string_view getDataTypeName(Data_Type _type) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return "FLOAT16";
    case Data_Type::TERNARY_INT2:
        return "TERNARY_INT2";
    case Data_Type::FLOAT32:
        return "FLOAT32";
    }
    std::unreachable();
}

inline size_t getTensorStorageBytes(Data_Type _type, const Shape &_shape) noexcept
{
    switch (_type)
    {
    case Data_Type::FLOAT16:
        return _shape.getTotalElements() * 2;
    case Data_Type::FLOAT32:
        return _shape.getTotalElements() * 4;
    case Data_Type::TERNARY_INT2:
    {
        if (_shape.getTotalElements() == 0)
        {
            return 0;
        }
        size_t rank = _shape.getRank();
        size_t last_dim = (rank == 0) ? 1 : _shape[rank - 1];
        size_t num_rows = (rank <= 1) ? 1 : (_shape.getTotalElements() / last_dim);
        size_t words_per_row = (last_dim + 15) / 16;
        return num_rows * words_per_row * sizeof(uint32_t);
    }
    }
    std::unreachable();
}

inline void convertFp32ToFp16(const float *_source, float16_t *_destination, size_t _count) noexcept
{
#if defined(__STDCPP_FLOAT16_T__)
    for (size_t i = 0; i < _count; ++i)
    {
        _destination[i] = static_cast<std::float16_t>(_source[i]);
    }
#else
    for (size_t i = 0; i < _count; ++i)
    {
        _destination[i].value = floatToHalfBits(_source[i]);
    }
#endif
}

inline void convertFp16ToFp32(const float16_t *_source, float *_destination, size_t _count) noexcept
{
#if defined(__STDCPP_FLOAT16_T__)
    for (size_t i = 0; i < _count; ++i)
    {
        _destination[i] = static_cast<float>(_source[i]);
    }
#else
    for (size_t i = 0; i < _count; ++i)
    {
        _destination[i] = halfBitsToFloat(_source[i].value);
    }
#endif
}

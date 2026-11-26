#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm/weight_loader.h"

namespace weights::internal
{

inline uint16_t readU16Le(const uint8_t* p) noexcept
{
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t readU32Le(const uint8_t* p) noexcept
{
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t readU64Le(const uint8_t* p) noexcept
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
    {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

inline size_t denseDtypeByteSize(Weight_Dtype dt) noexcept
{
    switch (dt)
    {
    case Weight_Dtype::F32: return 4;
    case Weight_Dtype::F16:
    case Weight_Dtype::BF16: return 2;
    case Weight_Dtype::F64: return 8;
    case Weight_Dtype::I8:
    case Weight_Dtype::U8: return 1;
    case Weight_Dtype::I16: return 2;
    case Weight_Dtype::I32: return 4;
    case Weight_Dtype::I64: return 8;
    default: return 0;
    }
}

inline float convertScalarToF32(const uint8_t* ptr, Weight_Dtype dt)
{
    switch (dt)
    {
    case Weight_Dtype::F32:
    {
        float f = 0.0f;
        std::memcpy(&f, ptr, 4);
        return f;
    }
    case Weight_Dtype::F16: return f16ToF32(readU16Le(ptr));
    case Weight_Dtype::BF16: return bf16ToF32(readU16Le(ptr));
    case Weight_Dtype::F64:
    {
        double d = 0.0;
        std::memcpy(&d, ptr, 8);
        return static_cast<float>(d);
    }
    case Weight_Dtype::I8: return static_cast<float>(*reinterpret_cast<const int8_t*>(ptr));
    case Weight_Dtype::U8: return static_cast<float>(*ptr);
    case Weight_Dtype::I16:
    {
        int16_t v = 0;
        std::memcpy(&v, ptr, 2);
        return static_cast<float>(v);
    }
    case Weight_Dtype::I32:
    {
        int32_t v = 0;
        std::memcpy(&v, ptr, 4);
        return static_cast<float>(v);
    }
    case Weight_Dtype::I64:
    {
        int64_t v = 0;
        std::memcpy(&v, ptr, 8);
        return static_cast<float>(v);
    }
    default:
        throw std::runtime_error("weight_loader: unsupported dense tensor dtype");
    }
}

inline std::vector<float> convertDenseRawToF32(const uint8_t* raw,
                                                size_t total_elems,
                                                Weight_Dtype dt,
                                                const char* ctx)
{
    const size_t elem_bytes = denseDtypeByteSize(dt);
    if (elem_bytes == 0)
    {
        throw std::runtime_error(std::string(ctx) + ": unsupported tensor dtype");
    }
    std::vector<float> out(total_elems, 0.0f);
    if (total_elems == 0)
    {
        return out;
    }
    if (dt == Weight_Dtype::F32)
    {
        std::memcpy(out.data(), raw, total_elems * 4ULL);
        return out;
    }
    for (size_t i = 0; i < total_elems; ++i)
    {
        out[i] = convertScalarToF32(raw + i * elem_bytes, dt);
    }
    return out;
}

}

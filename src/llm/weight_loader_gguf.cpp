#include "llm/weight_loader.h"
#include "weight_loader_internal.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace weights
{

namespace
{

constexpr uint32_t GGUF_MAGIC = 0x46554747u;

enum class Gguf_Kv_Type : uint32_t
{
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

enum class Ggml_Type : uint32_t
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,
    BF16 = 30
};

Weight_Dtype ggmlToWeightDtype(uint32_t t) noexcept
{
    switch (static_cast<Ggml_Type>(t))
    {
    case Ggml_Type::F32: return Weight_Dtype::F32;
    case Ggml_Type::F16: return Weight_Dtype::F16;
    case Ggml_Type::Q4_0: return Weight_Dtype::Q4_0;
    case Ggml_Type::Q4_1: return Weight_Dtype::Q4_1;
    case Ggml_Type::Q5_0: return Weight_Dtype::Q5_0;
    case Ggml_Type::Q5_1: return Weight_Dtype::Q5_1;
    case Ggml_Type::Q8_0: return Weight_Dtype::Q8_0;
    case Ggml_Type::Q4_K: return Weight_Dtype::Q4_K;
    case Ggml_Type::Q6_K: return Weight_Dtype::Q6_K;
    case Ggml_Type::BF16: return Weight_Dtype::BF16;
    default: return Weight_Dtype::UNKNOWN;
    }
}

uint64_t ggmlRowBytes(Weight_Dtype dtype, size_t total_elems)
{
    switch (dtype)
    {
    case Weight_Dtype::F32: return total_elems * 4ULL;
    case Weight_Dtype::F16:
    case Weight_Dtype::BF16: return total_elems * 2ULL;
    case Weight_Dtype::Q4_0: return (total_elems / 32ULL) * 18ULL;
    case Weight_Dtype::Q4_1: return (total_elems / 32ULL) * 20ULL;
    case Weight_Dtype::Q5_0: return (total_elems / 32ULL) * 22ULL;
    case Weight_Dtype::Q5_1: return (total_elems / 32ULL) * 24ULL;
    case Weight_Dtype::Q8_0: return (total_elems / 32ULL) * 34ULL;
    case Weight_Dtype::Q4_K: return (total_elems / 256ULL) * 144ULL;
    case Weight_Dtype::Q6_K: return (total_elems / 256ULL) * 210ULL;
    default: return 0ULL;
    }
}

using internal::readU16Le;
using internal::readU32Le;

template <typename T>
T readPod(std::ifstream& in)
{
    T val{};
    if (!in.read(reinterpret_cast<char*>(&val), sizeof(T)))
    {
        throw std::runtime_error("GGUF: unexpected EOF");
    }
    return val;
}

std::string readGgufString(std::ifstream& in)
{
    const uint64_t len = readPod<uint64_t>(in);
    if (len > (64ULL * 1024ULL * 1024ULL))
    {
        throw std::runtime_error("GGUF: string exceeds safety bound");
    }
    std::string s(static_cast<size_t>(len), '\0');
    if (len > 0 && !in.read(s.data(), static_cast<std::streamsize>(len)))
    {
        throw std::runtime_error("GGUF: truncated string");
    }
    return s;
}

void skipScalarKv(std::ifstream& in, Gguf_Kv_Type t)
{
    switch (t)
    {
    case Gguf_Kv_Type::UINT8:
    case Gguf_Kv_Type::INT8:
    case Gguf_Kv_Type::BOOL: in.seekg(1, std::ios::cur); break;
    case Gguf_Kv_Type::UINT16:
    case Gguf_Kv_Type::INT16: in.seekg(2, std::ios::cur); break;
    case Gguf_Kv_Type::UINT32:
    case Gguf_Kv_Type::INT32:
    case Gguf_Kv_Type::FLOAT32: in.seekg(4, std::ios::cur); break;
    case Gguf_Kv_Type::UINT64:
    case Gguf_Kv_Type::INT64:
    case Gguf_Kv_Type::FLOAT64: in.seekg(8, std::ios::cur); break;
    case Gguf_Kv_Type::STRING: (void)readGgufString(in); break;
    case Gguf_Kv_Type::ARRAY:
    {
        const auto elem_t = static_cast<Gguf_Kv_Type>(readPod<uint32_t>(in));
        const uint64_t n = readPod<uint64_t>(in);
        for (uint64_t i = 0; i < n; ++i)
        {
            skipScalarKv(in, elem_t);
        }
        break;
    }
    default:
        throw std::runtime_error("GGUF: unknown KV type");
    }
}

bool readScalarAsInt(std::ifstream& in, Gguf_Kv_Type t, int64_t& out)
{
    switch (t)
    {
    case Gguf_Kv_Type::UINT8: out = static_cast<int64_t>(readPod<uint8_t>(in)); return true;
    case Gguf_Kv_Type::INT8: out = static_cast<int64_t>(readPod<int8_t>(in)); return true;
    case Gguf_Kv_Type::UINT16: out = static_cast<int64_t>(readPod<uint16_t>(in)); return true;
    case Gguf_Kv_Type::INT16: out = static_cast<int64_t>(readPod<int16_t>(in)); return true;
    case Gguf_Kv_Type::UINT32: out = static_cast<int64_t>(readPod<uint32_t>(in)); return true;
    case Gguf_Kv_Type::INT32: out = static_cast<int64_t>(readPod<int32_t>(in)); return true;
    case Gguf_Kv_Type::UINT64: out = static_cast<int64_t>(readPod<uint64_t>(in)); return true;
    case Gguf_Kv_Type::INT64: out = readPod<int64_t>(in); return true;
    case Gguf_Kv_Type::BOOL: out = readPod<uint8_t>(in) ? 1 : 0; return true;
    default: return false;
    }
}

void readKvValue(std::ifstream& in, Gguf_Kv_Type t, const std::string& key, Weight_Metadata& meta)
{
    switch (t)
    {
    case Gguf_Kv_Type::UINT8: meta.kv[key] = static_cast<uint64_t>(readPod<uint8_t>(in)); break;
    case Gguf_Kv_Type::INT8: meta.kv[key] = static_cast<int64_t>(readPod<int8_t>(in)); break;
    case Gguf_Kv_Type::UINT16: meta.kv[key] = static_cast<uint64_t>(readPod<uint16_t>(in)); break;
    case Gguf_Kv_Type::INT16: meta.kv[key] = static_cast<int64_t>(readPod<int16_t>(in)); break;
    case Gguf_Kv_Type::UINT32: meta.kv[key] = static_cast<uint64_t>(readPod<uint32_t>(in)); break;
    case Gguf_Kv_Type::INT32: meta.kv[key] = static_cast<int64_t>(readPod<int32_t>(in)); break;
    case Gguf_Kv_Type::UINT64: meta.kv[key] = readPod<uint64_t>(in); break;
    case Gguf_Kv_Type::INT64: meta.kv[key] = readPod<int64_t>(in); break;
    case Gguf_Kv_Type::FLOAT32: meta.kv[key] = static_cast<double>(readPod<float>(in)); break;
    case Gguf_Kv_Type::FLOAT64: meta.kv[key] = readPod<double>(in); break;
    case Gguf_Kv_Type::BOOL: meta.kv[key] = (readPod<uint8_t>(in) != 0); break;
    case Gguf_Kv_Type::STRING: meta.kv[key] = readGgufString(in); break;
    case Gguf_Kv_Type::ARRAY:
    {
        const auto elem_t = static_cast<Gguf_Kv_Type>(readPod<uint32_t>(in));
        const uint64_t n = readPod<uint64_t>(in);
        if (elem_t == Gguf_Kv_Type::STRING)
        {
            std::vector<std::string> arr;
            arr.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; ++i)
            {
                arr.push_back(readGgufString(in));
            }
            meta.kv[key] = std::move(arr);
        }
        else if (elem_t == Gguf_Kv_Type::FLOAT32 || elem_t == Gguf_Kv_Type::FLOAT64)
        {
            std::vector<double> arr;
            arr.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; ++i)
            {
                arr.push_back((elem_t == Gguf_Kv_Type::FLOAT32)
                    ? static_cast<double>(readPod<float>(in))
                    : readPod<double>(in));
            }
            meta.kv[key] = std::move(arr);
        }
        else if (elem_t == Gguf_Kv_Type::ARRAY)
        {
            for (uint64_t i = 0; i < n; ++i) skipScalarKv(in, elem_t);
        }
        else
        {
            std::vector<int64_t> arr;
            arr.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; ++i)
            {
                int64_t val = 0;
                if (!readScalarAsInt(in, elem_t, val))
                {
                    skipScalarKv(in, elem_t);
                }
                else
                {
                    arr.push_back(val);
                }
            }
            meta.kv[key] = std::move(arr);
        }
        break;
    }
    default:
        throw std::runtime_error("GGUF: unknown KV type for key '" + key + "'");
    }
}

void getScaleMinK4(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) noexcept
{
    if (j < 4)
    {
        d_out = static_cast<uint8_t>(q[j] & 63u);
        m_out = static_cast<uint8_t>(q[j + 4] & 63u);
    }
    else
    {
        d_out = static_cast<uint8_t>((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        m_out = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void dequantizeQ4_0(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 18;
        const float d = f16ToF32(readU16Le(blk));
        const uint8_t* qs = blk + 2;
        float* y = dst + b * 32;
        for (size_t j = 0; j < 16; ++j)
        {
            const int x0 = static_cast<int>(qs[j] & 0x0Fu) - 8;
            const int x1 = static_cast<int>(qs[j] >> 4) - 8;
            y[j] = d * static_cast<float>(x0);
            y[j + 16] = d * static_cast<float>(x1);
        }
    }
}

void dequantizeQ4_1(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 20;
        const float d = f16ToF32(readU16Le(blk));
        const float m = f16ToF32(readU16Le(blk + 2));
        const uint8_t* qs = blk + 4;
        float* y = dst + b * 32;
        for (size_t j = 0; j < 16; ++j)
        {
            const int x0 = static_cast<int>(qs[j] & 0x0Fu);
            const int x1 = static_cast<int>(qs[j] >> 4);
            y[j] = d * static_cast<float>(x0) + m;
            y[j + 16] = d * static_cast<float>(x1) + m;
        }
    }
}

void dequantizeQ5_0(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 22;
        const float d = f16ToF32(readU16Le(blk));
        const uint32_t qh = readU32Le(blk + 2);
        const uint8_t* qs = blk + 6;
        float* y = dst + b * 32;
        for (size_t j = 0; j < 16; ++j)
        {
            const uint8_t xh_0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
            const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10u);
            const int x0 = static_cast<int>((qs[j] & 0x0Fu) | xh_0) - 16;
            const int x1 = static_cast<int>((qs[j] >> 4) | xh_1) - 16;
            y[j] = d * static_cast<float>(x0);
            y[j + 16] = d * static_cast<float>(x1);
        }
    }
}

void dequantizeQ5_1(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 24;
        const float d = f16ToF32(readU16Le(blk));
        const float m = f16ToF32(readU16Le(blk + 2));
        const uint32_t qh = readU32Le(blk + 4);
        const uint8_t* qs = blk + 8;
        float* y = dst + b * 32;
        for (size_t j = 0; j < 16; ++j)
        {
            const uint8_t xh_0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10u);
            const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10u);
            const int x0 = static_cast<int>((qs[j] & 0x0Fu) | xh_0);
            const int x1 = static_cast<int>((qs[j] >> 4) | xh_1);
            y[j] = d * static_cast<float>(x0) + m;
            y[j + 16] = d * static_cast<float>(x1) + m;
        }
    }
}

void dequantizeQ8_0(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 34;
        const float d = f16ToF32(readU16Le(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        float* y = dst + b * 32;
        for (size_t j = 0; j < 32; ++j)
        {
            y[j] = d * static_cast<float>(qs[j]);
        }
    }
}

void dequantizeQ4_K(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 256;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 144;
        const float d = f16ToF32(readU16Le(blk));
        const float dmin = f16ToF32(readU16Le(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        float* y = dst + b * 256;

        int is = 0;
        for (int j = 0; j < 256; j += 64)
        {
            uint8_t sc = 0;
            uint8_t m = 0;
            getScaleMinK4(is, scales, sc, m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = dmin * static_cast<float>(m);
            getScaleMinK4(is + 1, scales, sc, m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = dmin * static_cast<float>(m);
            const uint8_t* q = qs + (j / 2);
            for (int l = 0; l < 32; ++l)
            {
                y[j + l] = d1 * static_cast<float>(q[l] & 0x0Fu) - m1;
            }
            for (int l = 0; l < 32; ++l)
            {
                y[j + 32 + l] = d2 * static_cast<float>(q[l] >> 4) - m2;
            }
            is += 2;
        }
    }
}

void dequantizeQ6_K(const uint8_t* src, float* dst, size_t n)
{
    const size_t nb = n / 256;
    for (size_t b = 0; b < nb; ++b)
    {
        const uint8_t* blk = src + b * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
        const float d = f16ToF32(readU16Le(blk + 208));
        float* y = dst + b * 256;

        for (int n_off = 0; n_off < 256; n_off += 128)
        {
            const uint8_t* ql_p = ql + (n_off / 2);
            const uint8_t* qh_p = qh + (n_off / 4);
            const int8_t* sc_p = scales + (n_off / 16);
            for (int l = 0; l < 32; ++l)
            {
                const int is = l / 16;
                const int q1 = static_cast<int>((ql_p[l] & 0x0Fu) | (((qh_p[l] >> 0) & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((ql_p[l + 32] & 0x0Fu) | (((qh_p[l] >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((ql_p[l] >> 4) | (((qh_p[l] >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((ql_p[l + 32] >> 4) | (((qh_p[l] >> 6) & 3u) << 4)) - 32;
                y[n_off + l + 0] = d * static_cast<float>(sc_p[is + 0]) * static_cast<float>(q1);
                y[n_off + l + 32] = d * static_cast<float>(sc_p[is + 2]) * static_cast<float>(q2);
                y[n_off + l + 64] = d * static_cast<float>(sc_p[is + 4]) * static_cast<float>(q3);
                y[n_off + l + 96] = d * static_cast<float>(sc_p[is + 6]) * static_cast<float>(q4);
            }
        }
    }
}

class Gguf_Weight_Source final : public IWeight_Source
{
private:
    struct Entry
    {
        Weight_Tensor_Info info;
        uint64_t rel_offset = 0;
    };

    std::filesystem::path file_path;
    uint64_t data_base_offset = 0;
    std::unordered_map<std::string, Entry> entries;
    std::vector<std::string> ordered_names;
    Weight_Metadata meta;

public:
    explicit Gguf_Weight_Source(const std::filesystem::path& path)
        : file_path(path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("GGUF: failed to open '" + path.string() + "'");
        }
        const uint32_t magic = readPod<uint32_t>(in);
        if (magic != GGUF_MAGIC)
        {
            throw std::runtime_error("GGUF: invalid magic header in '" + path.string() + "'");
        }
        const uint32_t version = readPod<uint32_t>(in);
        if (version < 2 || version > 3)
        {
            throw std::runtime_error("GGUF: unsupported version " + std::to_string(version));
        }
        const uint64_t tensor_count = readPod<uint64_t>(in);
        const uint64_t kv_count = readPod<uint64_t>(in);

        for (uint64_t i = 0; i < kv_count; ++i)
        {
            const std::string key = readGgufString(in);
            const auto kv_type = static_cast<Gguf_Kv_Type>(readPod<uint32_t>(in));
            readKvValue(in, kv_type, key, meta);
        }

        const uint64_t alignment = std::max<uint64_t>(1, meta.getUint("general.alignment", 32));
        std::vector<Entry> raw_entries;
        raw_entries.reserve(static_cast<size_t>(tensor_count));
        for (uint64_t i = 0; i < tensor_count; ++i)
        {
            Entry e;
            e.info.name = readGgufString(in);
            const uint32_t n_dims = readPod<uint32_t>(in);
            if (n_dims > 8)
            {
                throw std::runtime_error("GGUF: invalid tensor rank for '" + e.info.name + "'");
            }
            std::vector<size_t> ne(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d)
            {
                ne[d] = static_cast<size_t>(readPod<uint64_t>(in));
            }
            e.info.shape.assign(ne.rbegin(), ne.rend());
            const uint32_t ggml_t = readPod<uint32_t>(in);
            e.info.dtype = ggmlToWeightDtype(ggml_t);
            e.rel_offset = readPod<uint64_t>(in);
            e.info.byte_size = ggmlRowBytes(e.info.dtype, e.info.elementCount());
            raw_entries.push_back(std::move(e));
        }

        const uint64_t header_end = static_cast<uint64_t>(in.tellg());
        const uint64_t rem = header_end % alignment;
        data_base_offset = (rem == 0) ? header_end : (header_end + (alignment - rem));

        for (auto& e : raw_entries)
        {
            ordered_names.push_back(e.info.name);
            entries.emplace(e.info.name, std::move(e));
        }
    }

    Weight_Format format() const noexcept override { return Weight_Format::GGUF; }

    bool hasTensor(const std::string& name) const noexcept override
    {
        return entries.find(name) != entries.end();
    }

    std::vector<std::string> tensorNames() const override { return ordered_names; }

    Weight_Tensor_Info getTensorInfo(const std::string& name) const override
    {
        auto it = entries.find(name);
        if (it == entries.end())
        {
            throw std::runtime_error("GGUF: missing tensor '" + name + "'");
        }
        return it->second.info;
    }

    const Weight_Metadata& metadata() const noexcept override { return meta; }

    std::vector<float> readTensorF32(const std::string& name) const override
    {
        auto it = entries.find(name);
        if (it == entries.end())
        {
            throw std::runtime_error("GGUF: missing tensor '" + name + "'");
        }
        const Entry& e = it->second;
        const size_t total = e.info.elementCount();
        if (e.info.dtype == Weight_Dtype::UNKNOWN || (e.info.byte_size == 0 && total > 0))
        {
            throw std::runtime_error("GGUF: unsupported tensor quantization type for '" + name + "'");
        }

        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("GGUF: cannot reopen '" + file_path.string() + "'");
        }
        in.seekg(static_cast<std::streamoff>(data_base_offset + e.rel_offset), std::ios::beg);

        std::vector<float> out(total, 0.0f);
        if (e.info.dtype == Weight_Dtype::F32)
        {
            if (total > 0 && !in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(total * 4ULL)))
            {
                throw std::runtime_error("GGUF: read error for '" + name + "'");
            }
            return out;
        }
        if (e.info.dtype == Weight_Dtype::F16 || e.info.dtype == Weight_Dtype::BF16)
        {
            std::vector<uint16_t> raw(total);
            if (total > 0 && !in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total * 2ULL)))
            {
                throw std::runtime_error("GGUF: read error for '" + name + "'");
            }
            if (e.info.dtype == Weight_Dtype::F16)
            {
                for (size_t i = 0; i < total; ++i) out[i] = f16ToF32(raw[i]);
            }
            else
            {
                for (size_t i = 0; i < total; ++i) out[i] = bf16ToF32(raw[i]);
            }
            return out;
        }

        std::vector<uint8_t> raw(static_cast<size_t>(e.info.byte_size));
        if (!raw.empty() && !in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
        {
            throw std::runtime_error("GGUF: quantized read error for '" + name + "'");
        }
        switch (e.info.dtype)
        {
        case Weight_Dtype::Q4_0: dequantizeQ4_0(raw.data(), out.data(), total); break;
        case Weight_Dtype::Q4_1: dequantizeQ4_1(raw.data(), out.data(), total); break;
        case Weight_Dtype::Q5_0: dequantizeQ5_0(raw.data(), out.data(), total); break;
        case Weight_Dtype::Q5_1: dequantizeQ5_1(raw.data(), out.data(), total); break;
        case Weight_Dtype::Q8_0: dequantizeQ8_0(raw.data(), out.data(), total); break;
        case Weight_Dtype::Q4_K: dequantizeQ4_K(raw.data(), out.data(), total); break;
        case Weight_Dtype::Q6_K: dequantizeQ6_K(raw.data(), out.data(), total); break;
        default:
            throw std::runtime_error("GGUF: unsupported dtype for '" + name + "'");
        }
        return out;
    }
};

}

std::unique_ptr<IWeight_Source> openGguf(const std::filesystem::path& path)
{
    return std::make_unique<Gguf_Weight_Source>(path);
}

}

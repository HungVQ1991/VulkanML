#include "llm/weight_loader.h"
#include "weight_loader_internal.h"

#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace weights
{

namespace
{

enum class Wire_Type : uint32_t
{
    VARINT = 0,
    FIXED64 = 1,
    LEN = 2,
    START_GROUP = 3,
    END_GROUP = 4,
    FIXED32 = 5
};

Weight_Dtype onnxToWeightDtype(int32_t dt) noexcept
{
    switch (dt)
    {
    case 1: return Weight_Dtype::F32;
    case 2: return Weight_Dtype::U8;
    case 3: return Weight_Dtype::I8;
    case 5: return Weight_Dtype::I16;
    case 6: return Weight_Dtype::I32;
    case 7: return Weight_Dtype::I64;
    case 10: return Weight_Dtype::F16;
    case 11: return Weight_Dtype::F64;
    case 16: return Weight_Dtype::BF16;
    default: return Weight_Dtype::UNKNOWN;
    }
}

uint64_t readVarint(std::ifstream& in, uint64_t end_pos)
{
    uint64_t result = 0;
    uint32_t shift = 0;
    while (static_cast<uint64_t>(in.tellg()) < end_pos)
    {
        char c = 0;
        if (!in.get(c))
        {
            throw std::runtime_error("ONNX: unexpected EOF while reading varint");
        }
        const uint8_t b = static_cast<uint8_t>(c);
        result |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u)
        {
            return result;
        }
        shift += 7;
        if (shift >= 64)
        {
            throw std::runtime_error("ONNX: varint overflow");
        }
    }
    throw std::runtime_error("ONNX: varint out of bounds");
}

uint32_t readFixed32(std::ifstream& in)
{
    uint32_t v = 0;
    if (!in.read(reinterpret_cast<char*>(&v), 4))
    {
        throw std::runtime_error("ONNX: unexpected EOF reading fixed32");
    }
    return v;
}

uint64_t readFixed64(std::ifstream& in)
{
    uint64_t v = 0;
    if (!in.read(reinterpret_cast<char*>(&v), 8))
    {
        throw std::runtime_error("ONNX: unexpected EOF reading fixed64");
    }
    return v;
}

void skipField(std::ifstream& in, Wire_Type wt, uint64_t end_pos)
{
    switch (wt)
    {
    case Wire_Type::VARINT: (void)readVarint(in, end_pos); break;
    case Wire_Type::FIXED64: in.seekg(8, std::ios::cur); break;
    case Wire_Type::FIXED32: in.seekg(4, std::ios::cur); break;
    case Wire_Type::LEN:
    {
        const uint64_t len = readVarint(in, end_pos);
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);
        break;
    }
    default:
        throw std::runtime_error("ONNX: unsupported protobuf wire type");
    }
}

std::string readStringSpan(std::ifstream& in, uint64_t len)
{
    std::string s(static_cast<size_t>(len), '\0');
    if (len > 0 && !in.read(s.data(), static_cast<std::streamsize>(len)))
    {
        throw std::runtime_error("ONNX: truncated string field");
    }
    return s;
}

struct Onnx_Entry
{
    Weight_Tensor_Info info;
    bool has_raw = false;
    uint64_t raw_offset = 0;
    uint64_t raw_len = 0;
    bool is_external = false;
    std::filesystem::path external_path;
    uint64_t ext_offset = 0;
    uint64_t ext_len = 0;
    std::vector<float> inline_floats;
    std::vector<int64_t> inline_ints;
};

void validateSafeRelativePath(const std::filesystem::path& base_dir, const std::string& rel_str, std::filesystem::path& out_full)
{
    std::filesystem::path rel(rel_str);
    if (rel.is_absolute())
    {
        throw std::runtime_error("ONNX: external_data absolute path disallowed: '" + rel_str + "'");
    }
    const auto norm_rel = rel.lexically_normal();
    for (const auto& part : norm_rel)
    {
        if (part == "..")
        {
            throw std::runtime_error("ONNX: external_data path traversal disallowed: '" + rel_str + "'");
        }
    }
    out_full = base_dir / norm_rel;
}

void parseStringStringEntry(std::ifstream& in, uint64_t end_pos, std::string& key_out, std::string& val_out)
{
    while (static_cast<uint64_t>(in.tellg()) < end_pos)
    {
        const uint64_t tag = readVarint(in, end_pos);
        const uint32_t fn = static_cast<uint32_t>(tag >> 3);
        const auto wt = static_cast<Wire_Type>(tag & 0x7u);
        if (wt == Wire_Type::LEN && (fn == 1 || fn == 2))
        {
            const uint64_t len = readVarint(in, end_pos);
            std::string s = readStringSpan(in, len);
            if (fn == 1) key_out = std::move(s);
            else val_out = std::move(s);
        }
        else
        {
            skipField(in, wt, end_pos);
        }
    }
}

Onnx_Entry parseTensorProto(std::ifstream& in, uint64_t end_pos, const std::filesystem::path& base_dir)
{
    Onnx_Entry e;
    int32_t raw_dt = 0;
    int32_t data_loc = 0;
    std::string ext_loc;
    uint64_t ext_off = 0;
    uint64_t ext_len = 0;

    while (static_cast<uint64_t>(in.tellg()) < end_pos)
    {
        const uint64_t tag = readVarint(in, end_pos);
        const uint32_t fn = static_cast<uint32_t>(tag >> 3);
        const auto wt = static_cast<Wire_Type>(tag & 0x7u);

        if (fn == 1 && wt == Wire_Type::VARINT)
        {
            e.info.shape.push_back(static_cast<size_t>(readVarint(in, end_pos)));
        }
        else if (fn == 1 && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            const uint64_t sub_end = static_cast<uint64_t>(in.tellg()) + len;
            while (static_cast<uint64_t>(in.tellg()) < sub_end)
            {
                e.info.shape.push_back(static_cast<size_t>(readVarint(in, sub_end)));
            }
        }
        else if (fn == 2 && wt == Wire_Type::VARINT)
        {
            raw_dt = static_cast<int32_t>(readVarint(in, end_pos));
        }
        else if (fn == 4 && wt == Wire_Type::FIXED32)
        {
            uint32_t bits = readFixed32(in);
            float f = 0.0f;
            std::memcpy(&f, &bits, 4);
            e.inline_floats.push_back(f);
        }
        else if (fn == 4 && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            const size_t n = static_cast<size_t>(len / 4);
            const size_t old = e.inline_floats.size();
            e.inline_floats.resize(old + n);
            if (n > 0 && !in.read(reinterpret_cast<char*>(e.inline_floats.data() + old), static_cast<std::streamsize>(n * 4)))
            {
                throw std::runtime_error("ONNX: failed reading packed float_data");
            }
        }
        else if ((fn == 5 || fn == 7) && wt == Wire_Type::VARINT)
        {
            e.inline_ints.push_back(static_cast<int64_t>(readVarint(in, end_pos)));
        }
        else if ((fn == 5 || fn == 7) && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            const uint64_t sub_end = static_cast<uint64_t>(in.tellg()) + len;
            while (static_cast<uint64_t>(in.tellg()) < sub_end)
            {
                e.inline_ints.push_back(static_cast<int64_t>(readVarint(in, sub_end)));
            }
        }
        else if (fn == 8 && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            e.info.name = readStringSpan(in, len);
        }
        else if (fn == 9 && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            e.has_raw = true;
            e.raw_offset = static_cast<uint64_t>(in.tellg());
            e.raw_len = len;
            in.seekg(static_cast<std::streamoff>(len), std::ios::cur);
        }
        else if (fn == 10 && wt == Wire_Type::FIXED64)
        {
            uint64_t bits = readFixed64(in);
            double d = 0.0;
            std::memcpy(&d, &bits, 8);
            e.inline_floats.push_back(static_cast<float>(d));
        }
        else if (fn == 10 && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            const size_t n = static_cast<size_t>(len / 8);
            std::vector<double> tmp(n);
            if (n > 0 && !in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n * 8)))
            {
                throw std::runtime_error("ONNX: failed reading packed double_data");
            }
            for (double d : tmp) e.inline_floats.push_back(static_cast<float>(d));
        }
        else if (fn == 13 && wt == Wire_Type::LEN)
        {
            const uint64_t len = readVarint(in, end_pos);
            const uint64_t sub_end = static_cast<uint64_t>(in.tellg()) + len;
            std::string k;
            std::string v;
            parseStringStringEntry(in, sub_end, k, v);
            if (k == "location") ext_loc = v;
            else if (k == "offset") ext_off = std::stoull(v);
            else if (k == "length") ext_len = std::stoull(v);
        }
        else if (fn == 14 && wt == Wire_Type::VARINT)
        {
            data_loc = static_cast<int32_t>(readVarint(in, end_pos));
        }
        else
        {
            skipField(in, wt, end_pos);
        }
    }

    e.info.dtype = onnxToWeightDtype(raw_dt);
    if (data_loc == 1 && !ext_loc.empty())
    {
        e.is_external = true;
        validateSafeRelativePath(base_dir, ext_loc, e.external_path);
        e.ext_offset = ext_off;
        e.ext_len = ext_len;
        e.info.byte_size = ext_len;
    }
    else if (e.has_raw)
    {
        e.info.byte_size = e.raw_len;
    }
    else if (!e.inline_floats.empty())
    {
        e.info.byte_size = e.inline_floats.size() * 4ULL;
    }
    else if (!e.inline_ints.empty())
    {
        e.info.byte_size = e.inline_ints.size() * 4ULL;
    }
    return e;
}

void parseConstantNode(std::ifstream& in,
                       uint64_t end_pos,
                       const std::filesystem::path& base_dir,
                       std::vector<Onnx_Entry>& out)
{
    std::string op_type;
    std::string first_output;
    std::vector<Onnx_Entry> attr_tensors;

    auto parseAttr = [&](uint64_t attr_end) {
        std::string attr_name;
        std::optional<Onnx_Entry> tval;
        while (static_cast<uint64_t>(in.tellg()) < attr_end)
        {
            const uint64_t tag = readVarint(in, attr_end);
            const uint32_t fn = static_cast<uint32_t>(tag >> 3);
            const auto wt = static_cast<Wire_Type>(tag & 0x7u);
            if (fn == 1 && wt == Wire_Type::LEN)
            {
                attr_name = readStringSpan(in, readVarint(in, attr_end));
            }
            else if (fn == 5 && wt == Wire_Type::LEN)
            {
                const uint64_t sub_len = readVarint(in, attr_end);
                tval = parseTensorProto(in, static_cast<uint64_t>(in.tellg()) + sub_len, base_dir);
            }
            else
            {
                skipField(in, wt, attr_end);
            }
        }
        if (attr_name == "value" && tval.has_value())
        {
            attr_tensors.push_back(std::move(*tval));
        }
    };

    while (static_cast<uint64_t>(in.tellg()) < end_pos)
    {
        const uint64_t tag = readVarint(in, end_pos);
        const uint32_t fn = static_cast<uint32_t>(tag >> 3);
        const auto wt = static_cast<Wire_Type>(tag & 0x7u);
        if (fn == 2 && wt == Wire_Type::LEN && first_output.empty())
        {
            first_output = readStringSpan(in, readVarint(in, end_pos));
        }
        else if (fn == 4 && wt == Wire_Type::LEN)
        {
            op_type = readStringSpan(in, readVarint(in, end_pos));
        }
        else if (fn == 5 && wt == Wire_Type::LEN)
        {
            const uint64_t sub_len = readVarint(in, end_pos);
            parseAttr(static_cast<uint64_t>(in.tellg()) + sub_len);
        }
        else
        {
            skipField(in, wt, end_pos);
        }
    }

    if (op_type == "Constant")
    {
        for (auto& t : attr_tensors)
        {
            if (t.info.name.empty() && !first_output.empty())
            {
                t.info.name = first_output;
            }
            if (!t.info.name.empty())
            {
                out.push_back(std::move(t));
            }
        }
    }
}

class Onnx_Weight_Source final : public IWeight_Source
{
private:
    std::filesystem::path file_path;
    std::unordered_map<std::string, Onnx_Entry> entries;
    std::vector<std::string> ordered_names;
    Weight_Metadata meta;

    void parseGraphProto(std::ifstream& in, uint64_t end_pos, const std::filesystem::path& base_dir)
    {
        while (static_cast<uint64_t>(in.tellg()) < end_pos)
        {
            const uint64_t tag = readVarint(in, end_pos);
            const uint32_t fn = static_cast<uint32_t>(tag >> 3);
            const auto wt = static_cast<Wire_Type>(tag & 0x7u);
            if (fn == 1 && wt == Wire_Type::LEN)
            {
                const uint64_t sub_len = readVarint(in, end_pos);
                std::vector<Onnx_Entry> consts;
                parseConstantNode(in, static_cast<uint64_t>(in.tellg()) + sub_len, base_dir, consts);
                for (auto& c : consts)
                {
                    if (entries.find(c.info.name) == entries.end())
                    {
                        ordered_names.push_back(c.info.name);
                    }
                    entries.insert_or_assign(c.info.name, std::move(c));
                }
            }
            else if (fn == 5 && wt == Wire_Type::LEN)
            {
                const uint64_t sub_len = readVarint(in, end_pos);
                Onnx_Entry t = parseTensorProto(in, static_cast<uint64_t>(in.tellg()) + sub_len, base_dir);
                if (!t.info.name.empty())
                {
                    if (entries.find(t.info.name) == entries.end())
                    {
                        ordered_names.push_back(t.info.name);
                    }
                    entries.insert_or_assign(t.info.name, std::move(t));
                }
            }
            else
            {
                skipField(in, wt, end_pos);
            }
        }
    }

public:
    explicit Onnx_Weight_Source(const std::filesystem::path& path)
        : file_path(path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("ONNX: failed to open '" + path.string() + "'");
        }
        in.seekg(0, std::ios::end);
        const uint64_t file_size = static_cast<uint64_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        const std::filesystem::path base_dir = path.parent_path();

        while (static_cast<uint64_t>(in.tellg()) < file_size)
        {
            const uint64_t tag = readVarint(in, file_size);
            const uint32_t fn = static_cast<uint32_t>(tag >> 3);
            const auto wt = static_cast<Wire_Type>(tag & 0x7u);
            if (fn == 7 && wt == Wire_Type::LEN)
            {
                const uint64_t sub_len = readVarint(in, file_size);
                parseGraphProto(in, static_cast<uint64_t>(in.tellg()) + sub_len, base_dir);
            }
            else if (fn == 4 && wt == Wire_Type::LEN)
            {
                meta.kv["onnx.producer"] = readStringSpan(in, readVarint(in, file_size));
            }
            else
            {
                skipField(in, wt, file_size);
            }
        }
    }

    Weight_Format format() const noexcept override { return Weight_Format::ONNX; }

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
            throw std::runtime_error("ONNX: missing tensor '" + name + "'");
        }
        return it->second.info;
    }

    const Weight_Metadata& metadata() const noexcept override { return meta; }

    std::vector<float> readTensorF32(const std::string& name) const override
    {
        auto it = entries.find(name);
        if (it == entries.end())
        {
            throw std::runtime_error("ONNX: missing tensor '" + name + "'");
        }
        const Onnx_Entry& e = it->second;
        const size_t total = e.info.elementCount();
        if (!e.inline_floats.empty())
        {
            return e.inline_floats;
        }
        if (!e.inline_ints.empty())
        {
            std::vector<float> out(e.inline_ints.size());
            for (size_t i = 0; i < e.inline_ints.size(); ++i) out[i] = static_cast<float>(e.inline_ints[i]);
            return out;
        }

        const std::filesystem::path data_file = e.is_external ? e.external_path : file_path;
        const uint64_t data_off = e.is_external ? e.ext_offset : e.raw_offset;
        const uint64_t data_len = e.is_external ? e.ext_len : e.raw_len;
        if (!e.is_external && !e.has_raw)
        {
            throw std::runtime_error("ONNX: tensor '" + name + "' contains no data");
        }

        std::ifstream in(data_file, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("ONNX: failed opening data file '" + data_file.string() + "'");
        }
        in.seekg(static_cast<std::streamoff>(data_off), std::ios::beg);

        const size_t elem_bytes = internal::denseDtypeByteSize(e.info.dtype);
        if (elem_bytes == 0)
        {
            throw std::runtime_error("ONNX: unsupported tensor dtype for '" + name + "'");
        }
        const size_t bytes = total * elem_bytes;
        if (data_len != 0 && bytes > data_len)
        {
            throw std::runtime_error("ONNX: raw byte length mismatch for '" + name + "'");
        }
        std::vector<uint8_t> raw(bytes);
        if (bytes > 0 && !in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes)))
        {
            throw std::runtime_error("ONNX: read error for '" + name + "'");
        }
        return internal::convertDenseRawToF32(raw.data(), total, e.info.dtype, "ONNX");
    }
};

}

std::unique_ptr<IWeight_Source> openOnnx(const std::filesystem::path& path)
{
    return std::make_unique<Onnx_Weight_Source>(path);
}

}

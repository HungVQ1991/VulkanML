#include "llm/weight_loader.h"
#include "weight_loader_internal.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "helper/json.hpp"

namespace weights
{

namespace
{

Weight_Dtype parseSafetensorsDtype(const std::string& s)
{
    if (s == "F32") return Weight_Dtype::F32;
    if (s == "F16") return Weight_Dtype::F16;
    if (s == "BF16") return Weight_Dtype::BF16;
    if (s == "F64") return Weight_Dtype::F64;
    if (s == "I8") return Weight_Dtype::I8;
    if (s == "U8") return Weight_Dtype::U8;
    if (s == "I16") return Weight_Dtype::I16;
    if (s == "I32") return Weight_Dtype::I32;
    if (s == "I64") return Weight_Dtype::I64;
    return Weight_Dtype::UNKNOWN;
}

class Safetensors_Weight_Source final : public IWeight_Source
{
private:
    struct Shard
    {
        std::filesystem::path file_path;
        uint64_t data_base_offset = 0;
    };

    struct Entry
    {
        Weight_Tensor_Info info;
        uint64_t data_begin = 0;
        uint64_t data_end = 0;
        size_t shard_index = 0;
    };

    std::vector<Shard> shards;
    std::unordered_map<std::string, Entry> entries;
    std::vector<std::string> ordered_names;
    Weight_Metadata meta;

    void parseShardFile(const std::filesystem::path& path, size_t shard_idx)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Safetensors: failed to open '" + path.string() + "'");
        }
        uint64_t header_len = 0;
        if (!in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len)))
        {
            throw std::runtime_error("Safetensors: failed reading header length in '" + path.string() + "'");
        }
        if (header_len == 0 || header_len > (256ULL * 1024ULL * 1024ULL))
        {
            throw std::runtime_error("Safetensors: invalid header length in '" + path.string() + "'");
        }
        std::string header_str(static_cast<size_t>(header_len), '\0');
        if (!in.read(header_str.data(), static_cast<std::streamsize>(header_len)))
        {
            throw std::runtime_error("Safetensors: truncated JSON header in '" + path.string() + "'");
        }
        shards[shard_idx].data_base_offset = 8ULL + header_len;

        nlohmann::json header = nlohmann::json::parse(header_str);
        for (auto it = header.begin(); it != header.end(); ++it)
        {
            if (it.key() == "__metadata__")
            {
                if (it.value().is_object())
                {
                    for (auto mit = it.value().begin(); mit != it.value().end(); ++mit)
                    {
                        if (mit.value().is_string())
                        {
                            meta.kv["safetensors." + mit.key()] = mit.value().get<std::string>();
                        }
                    }
                }
                continue;
            }
            const auto& obj = it.value();
            if (!obj.is_object() || !obj.contains("dtype") || !obj.contains("shape") || !obj.contains("data_offsets"))
            {
                continue;
            }
            Entry e;
            e.info.name = it.key();
            e.info.dtype = parseSafetensorsDtype(obj["dtype"].get<std::string>());
            for (const auto& d : obj["shape"])
            {
                e.info.shape.push_back(d.get<size_t>());
            }
            e.data_begin = obj["data_offsets"].at(0).get<uint64_t>();
            e.data_end = obj["data_offsets"].at(1).get<uint64_t>();
            if (e.data_end < e.data_begin)
            {
                throw std::runtime_error("Safetensors: invalid offsets for '" + it.key() + "'");
            }
            e.info.byte_size = e.data_end - e.data_begin;
            e.shard_index = shard_idx;
            if (entries.find(it.key()) == entries.end())
            {
                ordered_names.push_back(it.key());
            }
            entries.insert_or_assign(it.key(), std::move(e));
        }
    }

public:
    explicit Safetensors_Weight_Source(const std::filesystem::path& path)
    {
        std::error_code ec;
        if (std::filesystem::is_regular_file(path, ec))
        {
            shards.push_back(Shard{ path, 0 });
            parseShardFile(path, 0);
            return;
        }

        const std::filesystem::path index_path = path / "model.safetensors.index.json";
        const std::filesystem::path single_path = path / "model.safetensors";
        if (std::filesystem::exists(index_path, ec))
        {
            std::ifstream idx_in(index_path);
            if (!idx_in.is_open())
            {
                throw std::runtime_error("Safetensors: failed to open index '" + index_path.string() + "'");
            }
            nlohmann::json idx_json = nlohmann::json::parse(idx_in);
            if (!idx_json.contains("weight_map") || !idx_json["weight_map"].is_object())
            {
                throw std::runtime_error("Safetensors: missing weight_map in index JSON");
            }
            std::unordered_map<std::string, size_t> shard_map;
            for (auto it = idx_json["weight_map"].begin(); it != idx_json["weight_map"].end(); ++it)
            {
                const std::string rel = it.value().get<std::string>();
                if (shard_map.find(rel) == shard_map.end())
                {
                    const size_t sid = shards.size();
                    shards.push_back(Shard{ path / rel, 0 });
                    shard_map.emplace(rel, sid);
                }
            }
            for (size_t i = 0; i < shards.size(); ++i)
            {
                parseShardFile(shards[i].file_path, i);
            }
        }
        else if (std::filesystem::exists(single_path, ec))
        {
            shards.push_back(Shard{ single_path, 0 });
            parseShardFile(single_path, 0);
        }
        else
        {
            throw std::runtime_error("Safetensors: no safetensors file or index found at '" + path.string() + "'");
        }
    }

    Weight_Format format() const noexcept override { return Weight_Format::SAFETENSORS; }

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
            throw std::runtime_error("Safetensors: missing tensor '" + name + "'");
        }
        return it->second.info;
    }

    const Weight_Metadata& metadata() const noexcept override { return meta; }

    std::vector<float> readTensorF32(const std::string& name) const override
    {
        auto it = entries.find(name);
        if (it == entries.end())
        {
            throw std::runtime_error("Safetensors: missing tensor '" + name + "'");
        }
        const Entry& e = it->second;
        const size_t total = e.info.elementCount();
        const Shard& shard = shards[e.shard_index];
        std::ifstream in(shard.file_path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Safetensors: cannot reopen shard '" + shard.file_path.string() + "'");
        }
        in.seekg(static_cast<std::streamoff>(shard.data_base_offset + e.data_begin), std::ios::beg);

        const size_t elem_bytes = internal::denseDtypeByteSize(e.info.dtype);
        if (elem_bytes == 0 || static_cast<uint64_t>(total) * elem_bytes != e.info.byte_size)
        {
            throw std::runtime_error("Safetensors: byte size or dtype mismatch for '" + name + "'");
        }
        std::vector<uint8_t> raw(static_cast<size_t>(e.info.byte_size));
        if (!raw.empty() && !in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
        {
            throw std::runtime_error("Safetensors: read error for '" + name + "'");
        }
        return internal::convertDenseRawToF32(raw.data(), total, e.info.dtype, "Safetensors");
    }
};

}

std::unique_ptr<IWeight_Source> openSafetensors(const std::filesystem::path& path)
{
    return std::make_unique<Safetensors_Weight_Source>(path);
}

}

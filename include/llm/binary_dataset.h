#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Binary_Token_Dataset
{
public:
    static constexpr uint32_t MAGIC_HEADER = 0x44424C53;
    static constexpr uint32_t CURRENT_VERSION = 1;

    struct Header
    {
        uint32_t magic = MAGIC_HEADER;
        uint32_t version = CURRENT_VERSION;
        uint64_t total_tokens = 0;
    };

    static bool save(const std::string &file_path, const std::vector<int32_t> &tokens);
    static bool load(const std::string &file_path, std::vector<int32_t> &tokens);
    static bool existsAndValid(const std::string &file_path);
};

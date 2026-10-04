#include "llm/binary_dataset.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

bool Binary_Token_Dataset::save(const std::string &file_path, const std::vector<int32_t> &tokens)
{
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    Header header;
    header.total_tokens = tokens.size();

    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!tokens.empty())
    {
        out.write(reinterpret_cast<const char *>(tokens.data()),
                  static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
    }

    return out.good();
}

bool Binary_Token_Dataset::load(const std::string &file_path, std::vector<int32_t> &tokens)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }

    Header header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in.good() || header.magic != MAGIC_HEADER || header.version != CURRENT_VERSION)
    {
        return false;
    }

    tokens.resize(header.total_tokens);
    if (header.total_tokens > 0)
    {
        in.read(reinterpret_cast<char *>(tokens.data()),
                static_cast<std::streamsize>(header.total_tokens * sizeof(int32_t)));
    }

    return in.good();
}

bool Binary_Token_Dataset::existsAndValid(const std::string &file_path)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }

    Header header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    return in.good() && header.magic == MAGIC_HEADER && header.version == CURRENT_VERSION;
}

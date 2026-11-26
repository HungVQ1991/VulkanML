#include "llm/binary_dataset.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "helper/logger.h"

namespace
{
    constexpr size_t CHUNK_BYTES = 512ULL * 1024ULL * 1024ULL;
    constexpr size_t CHUNK_TOKENS = CHUNK_BYTES / sizeof(int32_t);

    template <typename T>
    constexpr T swapIfBigEndian(T value) noexcept
    {
        if constexpr (std::endian::native == std::endian::big)
        {
            return std::byteswap(value);
        }
        return value;
    }

    std::filesystem::path toFsPath(const std::string &file_path)
    {
        return std::filesystem::path(std::u8string_view(
            reinterpret_cast<const char8_t *>(file_path.data()), file_path.size()));
    }
}

bool Binary_Token_Dataset::save(const std::string &file_path, const std::vector<int32_t> &tokens)
{
    if (file_path.empty())
    {
        return false;
    }

    const uint64_t token_count = static_cast<uint64_t>(tokens.size());
    if (token_count > (std::numeric_limits<uint64_t>::max() - sizeof(Header)) / sizeof(int32_t))
    {
        return false;
    }

    std::ofstream out(toFsPath(file_path), std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::save: Failed to open '{}' for writing", file_path);
        return false;
    }

    Header header;
    header.magic = swapIfBigEndian(header.magic);
    header.version = swapIfBigEndian(header.version);
    header.total_tokens = swapIfBigEndian(token_count);

    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!out.good())
    {
        return false;
    }

    size_t offset = 0;
    while (offset < tokens.size())
    {
        const size_t batch = std::min(CHUNK_TOKENS, tokens.size() - offset);
        if constexpr (std::endian::native == std::endian::little)
        {
            out.write(reinterpret_cast<const char *>(tokens.data() + offset),
                      static_cast<std::streamsize>(batch * sizeof(int32_t)));
        }
        else
        {
            std::vector<int32_t> le_batch(batch);
            for (size_t i = 0; i < batch; ++i)
            {
                le_batch[i] = swapIfBigEndian(tokens[offset + i]);
            }
            out.write(reinterpret_cast<const char *>(le_batch.data()),
                      static_cast<std::streamsize>(batch * sizeof(int32_t)));
        }
        if (!out.good())
        {
            return false;
        }
        offset += batch;
    }

    out.flush();
    out.close();
    const bool ok = !out.fail();
    if (ok)
    {
        Logger::logMessage(Input_Format{"Binary_Token_Dataset::save: Saved {} tokens to '{}'", tokens.size(), file_path},
                           Log_Level::LOG_INFO,
                           false,
                           0,
                           Log_Feature::DATASET_IO);
    }
    else
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::save: Write error while saving '{}'", file_path);
    }
    return ok;
}

bool Binary_Token_Dataset::load(const std::string &file_path, std::vector<int32_t> &tokens)
{
    if (file_path.empty())
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: Empty file path");
        return false;
    }

    const auto fs_path = toFsPath(file_path);
    std::error_code ec;
    const uint64_t file_bytes = std::filesystem::file_size(fs_path, ec);
    if (ec || file_bytes < sizeof(Header))
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: File '{}' missing or smaller than header ({} bytes)", file_path, file_bytes);
        return false;
    }

    std::ifstream in(fs_path, std::ios::binary);
    if (!in.is_open())
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: Failed to open '{}'", file_path);
        return false;
    }

    Header header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    header.magic = swapIfBigEndian(header.magic);
    header.version = swapIfBigEndian(header.version);
    header.total_tokens = swapIfBigEndian(header.total_tokens);
    if (!in.good() || header.magic != MAGIC_HEADER || header.version != CURRENT_VERSION)
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: Invalid magic/version in '{}'", file_path);
        return false;
    }

    if (header.total_tokens > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(int32_t) ||
        header.total_tokens > (std::numeric_limits<uint64_t>::max() - sizeof(Header)) / sizeof(int32_t))
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: Token count overflow in '{}'", file_path);
        return false;
    }

    const uint64_t expected_payload_bytes = header.total_tokens * sizeof(int32_t);
    if (file_bytes - sizeof(Header) < expected_payload_bytes ||
        file_bytes != sizeof(Header) + expected_payload_bytes)
    {
        VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: Payload size mismatch in '{}' (file={} bytes, expected={} bytes)",
                      file_path, file_bytes, sizeof(Header) + expected_payload_bytes);
        return false;
    }

    std::vector<int32_t> loaded_tokens(static_cast<size_t>(header.total_tokens));
    size_t offset = 0;
    while (offset < loaded_tokens.size())
    {
        const size_t batch = std::min(CHUNK_TOKENS, loaded_tokens.size() - offset);
        in.read(reinterpret_cast<char *>(loaded_tokens.data() + offset),
                static_cast<std::streamsize>(batch * sizeof(int32_t)));
        if (!in.good())
        {
            VML_LOG_ERROR(Log_Feature::DATASET_IO, "Binary_Token_Dataset::load: Truncated payload while reading '{}'", file_path);
            return false;
        }
        offset += batch;
    }

    if constexpr (std::endian::native == std::endian::big)
    {
        for (int32_t &tok : loaded_tokens)
        {
            tok = swapIfBigEndian(tok);
        }
    }

    tokens = std::move(loaded_tokens);
    Logger::logMessage(Input_Format{"Binary_Token_Dataset::load: Loaded {} tokens ({:.2f} MB) from '{}'",
                                    tokens.size(), static_cast<double>(file_bytes) / (1024.0 * 1024.0), file_path},
                       Log_Level::LOG_INFO,
                       false,
                       0,
                       Log_Feature::DATASET_IO);
    return true;
}

bool Binary_Token_Dataset::existsAndValid(const std::string &file_path)
{
    if (file_path.empty())
    {
        return false;
    }

    const auto fs_path = toFsPath(file_path);
    std::error_code ec;
    const uint64_t file_bytes = std::filesystem::file_size(fs_path, ec);
    if (ec || file_bytes < sizeof(Header))
    {
        return false;
    }

    std::ifstream in(fs_path, std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }

    Header header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    header.magic = swapIfBigEndian(header.magic);
    header.version = swapIfBigEndian(header.version);
    header.total_tokens = swapIfBigEndian(header.total_tokens);
    if (!in.good() || header.magic != MAGIC_HEADER || header.version != CURRENT_VERSION)
    {
        return false;
    }

    if (header.total_tokens > (std::numeric_limits<uint64_t>::max() - sizeof(Header)) / sizeof(int32_t))
    {
        return false;
    }

    return file_bytes == sizeof(Header) + header.total_tokens * sizeof(int32_t);
}

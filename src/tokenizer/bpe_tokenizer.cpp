#include "tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "helper/json.hpp"
#include "helper/logger.h"

bool Bpe_Tokenizer::parseMergeString(const std::string& merge_str, std::string& piece_a, std::string& piece_b)
{
    const size_t space_pos = (!merge_str.empty() && merge_str[0] == ' ')
        ? merge_str.find(' ', 1)
        : merge_str.find(' ');
    if (space_pos == std::string::npos)
    {
        return false;
    }
    piece_a = merge_str.substr(0, space_pos);
    piece_b = merge_str.substr(space_pos + 1);
    return !piece_a.empty() && !piece_b.empty();
}

void Bpe_Tokenizer::registerMergePair(const std::string& piece_a,
                                      const std::string& piece_b,
                                      int32_t rank,
                                      const std::unordered_map<std::string, int32_t>& vocab,
                                      std::unordered_map<uint64_t, Merge_Rule>& merges)
{
    const std::string merged = piece_a + piece_b;
    auto it_a = vocab.find(piece_a);
    auto it_b = vocab.find(piece_b);
    auto it_m = vocab.find(merged);
    if (it_a != vocab.end() && it_b != vocab.end() && it_m != vocab.end())
    {
        uint64_t pair_key = (static_cast<uint64_t>(it_a->second) << 32) | static_cast<uint32_t>(it_b->second);
        merges[pair_key] = Merge_Rule{ rank, it_m->second };
    }
}

void Bpe_Tokenizer::sortAddedTokens(
    std::unordered_map<unsigned char, std::vector<std::pair<std::string, int32_t>>>& added_tokens_by_first_byte)
{
    for (auto& [first_byte, candidates] : added_tokens_by_first_byte)
    {
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
                return a.first.size() > b.first.size();
            });
    }
}

Bpe_Tokenizer::Bpe_Tokenizer(const std::string& tokenizer_json_path)
{
    load(tokenizer_json_path);
}

bool Bpe_Tokenizer::load(const std::string& tokenizer_json_path)
{
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open())
    {
        VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: Failed to open '{}'", tokenizer_json_path);
        return false;
    }

    nlohmann::json j;
    try
    {
        file >> j;
    }
    catch (const std::exception& e)
    {
        VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: JSON parse error in '{}': {}", tokenizer_json_path, e.what());
        return false;
    }

    constexpr int64_t MAX_ALLOWED_TOKEN_ID = 16'777'216LL;

    int32_t tmp_bos_id = 1;
    int32_t tmp_eos_id = 2;
    int32_t tmp_unk_id = 0;
    bool tmp_prepend_space = true;
    bool tmp_split_metaspace = false;
    bool tmp_byte_level = false;
    bool tmp_has_bos = true;
    bool found_explicit_bos = false;
    bool found_im_end = false;
    std::string tmp_metaspace_replacement = "\xE2\x96\x81";

    std::unordered_map<std::string, int32_t> tmp_vocab;
    std::vector<std::string> tmp_id_to_token;
    std::unordered_map<uint64_t, Merge_Rule> tmp_merges;
    std::unordered_set<int32_t> tmp_special_tokens;
    std::unordered_map<int32_t, uint8_t> tmp_byte_fallback_tokens;
    std::unordered_map<unsigned char, std::vector<std::pair<std::string, int32_t>>> tmp_added_tokens_by_first_byte;

    try
    {
        if (!j.is_object() ||
            !j.contains("model") || !j["model"].is_object() ||
            !j["model"].contains("vocab") || !j["model"]["vocab"].is_object())
        {
            VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: Missing 'model.vocab' object in '{}'", tokenizer_json_path);
            return false;
        }

        const auto& vocab_json = j["model"]["vocab"];
        int32_t max_id = 0;
        for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it)
        {
            if (!it.value().is_number_integer())
            {
                VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: Non-integer vocab ID in '{}'", tokenizer_json_path);
                return false;
            }
            int64_t raw_id = it.value().get<int64_t>();
            if (raw_id < 0 || raw_id > MAX_ALLOWED_TOKEN_ID)
            {
                VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: Vocab ID {} out of bounds in '{}'", raw_id, tokenizer_json_path);
                return false;
            }
            int32_t id = static_cast<int32_t>(raw_id);
            tmp_vocab[it.key()] = id;
            if (id > max_id)
            {
                max_id = id;
            }
            uint8_t byte_val = 0;
            if (isByteFallbackToken(it.key(), byte_val))
            {
                tmp_byte_fallback_tokens[id] = byte_val;
            }
        }

        struct PendingAddedToken
        {
            int32_t id;
            std::string content;
            bool is_special;
        };
        std::vector<PendingAddedToken> added_list;
        if (j.contains("added_tokens") && j["added_tokens"].is_array())
        {
            for (const auto& item : j["added_tokens"])
            {
                if (item.is_object() && item.contains("id") && item.contains("content") &&
                    item["id"].is_number_integer() && item["content"].is_string())
                {
                    int64_t raw_id = item["id"].get<int64_t>();
                    if (raw_id < 0 || raw_id > MAX_ALLOWED_TOKEN_ID)
                    {
                        VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: Added token ID {} out of bounds in '{}'", raw_id, tokenizer_json_path);
                        return false;
                    }
                    int32_t id = static_cast<int32_t>(raw_id);
                    std::string content = item["content"].get<std::string>();
                    bool is_special = item.value("special", false);
                    max_id = std::max(max_id, id);
                    added_list.push_back({ id, std::move(content), is_special });
                }
            }
        }

        tmp_id_to_token.resize(static_cast<size_t>(max_id) + 1);
        for (const auto& [token_str, id] : tmp_vocab)
        {
            if (id >= 0 && id <= max_id)
            {
                tmp_id_to_token[static_cast<size_t>(id)] = token_str;
            }
        }

        for (const auto& added : added_list)
        {
            tmp_vocab.try_emplace(added.content, added.id);
            tmp_id_to_token[static_cast<size_t>(added.id)] = added.content;

            if (added.is_special)
            {
                tmp_special_tokens.insert(added.id);
            }
            if (added.content == "<s>" || added.content == "<|begin_of_text|>" || added.content == "<|startoftext|>")
            {
                tmp_bos_id = added.id;
                found_explicit_bos = true;
            }
            if (added.content == "</s>" || (added.content == "<|endoftext|>" && !found_im_end))
            {
                tmp_eos_id = added.id;
            }
            if (added.content == "<|im_end|>")
            {
                tmp_eos_id = added.id;
                found_im_end = true;
            }
            if (added.content == "<unk>" || added.content == "<|unk|>")
            {
                tmp_unk_id = added.id;
            }

            uint8_t byte_val = 0;
            if (isByteFallbackToken(added.content, byte_val))
            {
                tmp_byte_fallback_tokens[added.id] = byte_val;
            }
            else if (!added.content.empty())
            {
                tmp_added_tokens_by_first_byte[static_cast<unsigned char>(added.content[0])]
                    .emplace_back(added.content, added.id);
            }
        }

        sortAddedTokens(tmp_added_tokens_by_first_byte);

        if (j["model"].contains("merges") && j["model"]["merges"].is_array())
        {
            const auto& merges_json = j["model"]["merges"];
            int32_t rank = 0;
            for (const auto& item : merges_json)
            {
                std::string piece_a;
                std::string piece_b;
                bool valid_merge = false;

                if (item.is_array() && item.size() == 2 && item[0].is_string() && item[1].is_string())
                {
                    piece_a = item[0].get<std::string>();
                    piece_b = item[1].get<std::string>();
                    valid_merge = (!piece_a.empty() && !piece_b.empty());
                }
                else if (item.is_string())
                {
                    valid_merge = parseMergeString(item.get<std::string>(), piece_a, piece_b);
                }

                if (valid_merge)
                {
                    registerMergePair(piece_a, piece_b, rank, tmp_vocab, tmp_merges);
                }
                rank++;
            }
        }

        if (j.contains("decoder") && j["decoder"].is_object())
        {
            if (j["decoder"].value("type", "") == "ByteLevel")
            {
                tmp_byte_level = true;
            }
        }

        if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object())
        {
            const auto& pre_tok = j["pre_tokenizer"];
            const std::string pre_type = pre_tok.value("type", "");
            if (pre_type == "Metaspace")
            {
                tmp_metaspace_replacement = pre_tok.value("replacement", "\xE2\x96\x81");
                tmp_prepend_space = (pre_tok.value("prepend_scheme", "first") == "first");
                tmp_split_metaspace = pre_tok.value("split", true);
            }
            else if (pre_type == "ByteLevel")
            {
                tmp_byte_level = true;
            }
            else if (pre_type == "Sequence" && pre_tok.contains("pretokenizers") && pre_tok["pretokenizers"].is_array())
            {
                for (const auto& sub_pt : pre_tok["pretokenizers"])
                {
                    if (sub_pt.is_object() && sub_pt.value("type", "") == "ByteLevel")
                    {
                        tmp_byte_level = true;
                    }
                }
            }
        }

        if (tmp_byte_level)
        {
            tmp_prepend_space = false;
            tmp_split_metaspace = false;
            tmp_metaspace_replacement.clear();
            tmp_has_bos = found_explicit_bos;
            if (!found_im_end)
            {
                auto it_im_end = tmp_vocab.find("<|im_end|>");
                if (it_im_end != tmp_vocab.end())
                {
                    tmp_eos_id = it_im_end->second;
                }
                else
                {
                    auto it_eot = tmp_vocab.find("<|endoftext|>");
                    if (it_eot != tmp_vocab.end())
                    {
                        tmp_eos_id = it_eot->second;
                    }
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::load: Exception parsing '{}': {}", tokenizer_json_path, e.what());
        return false;
    }

    bos_id_ = tmp_bos_id;
    eos_id_ = tmp_eos_id;
    unk_id_ = tmp_unk_id;
    prepend_space_ = tmp_prepend_space;
    split_metaspace_ = tmp_split_metaspace;
    byte_level_ = tmp_byte_level;
    has_bos_ = tmp_has_bos;
    metaspace_replacement_ = std::move(tmp_metaspace_replacement);
    vocab_ = std::move(tmp_vocab);
    id_to_token_ = std::move(tmp_id_to_token);
    merges_ = std::move(tmp_merges);
    special_tokens_ = std::move(tmp_special_tokens);
    byte_fallback_tokens_ = std::move(tmp_byte_fallback_tokens);
    added_tokens_by_first_byte_ = std::move(tmp_added_tokens_by_first_byte);
    if (byte_level_)
    {
        initByteLevelTables();
    }
    else
    {
        unicode_to_byte_.clear();
    }
    is_loaded_ = true;
    Logger::logMessage(Input_Format{"Bpe_Tokenizer::load: Loaded '{}' (vocab={}, merges={}, special={}, byte_level={})",
                                    tokenizer_json_path, id_to_token_.size(), merges_.size(), special_tokens_.size(), byte_level_},
                       Log_Level::LOG_INFO,
                       false,
                       0,
                       Log_Feature::TOKENIZER);
    return true;
}

bool Bpe_Tokenizer::loadFromVocabAndMerges(const std::vector<std::string>& tokens,
                                           const std::vector<std::string>& merges,
                                           const std::vector<int32_t>& token_types,
                                           bool is_byte_level,
                                           int32_t bos_id,
                                           int32_t eos_id,
                                           int32_t unk_id,
                                           bool add_bos)
{
    if (tokens.empty())
    {
        VML_LOG_ERROR(Log_Feature::TOKENIZER, "Bpe_Tokenizer::loadFromVocabAndMerges: empty tokens list");
        return false;
    }

    int32_t tmp_bos_id = (bos_id >= 0) ? bos_id : 1;
    int32_t tmp_eos_id = (eos_id >= 0) ? eos_id : 2;
    int32_t tmp_unk_id = (unk_id >= 0) ? unk_id : 0;
    bool found_im_end = false;

    std::unordered_map<std::string, int32_t> tmp_vocab;
    tmp_vocab.reserve(tokens.size());
    std::vector<std::string> tmp_id_to_token = tokens;
    std::unordered_map<uint64_t, Merge_Rule> tmp_merges;
    tmp_merges.reserve(merges.size());
    std::unordered_set<int32_t> tmp_special_tokens;
    std::unordered_map<int32_t, uint8_t> tmp_byte_fallback_tokens;
    std::unordered_map<unsigned char, std::vector<std::pair<std::string, int32_t>>> tmp_added_tokens_by_first_byte;

    for (size_t i = 0; i < tokens.size(); ++i)
    {
        const int32_t id = static_cast<int32_t>(i);
        const std::string& tok = tokens[i];
        tmp_vocab[tok] = id;

        const int32_t ttype = (i < token_types.size()) ? token_types[i] : 1;
        const bool is_special = (ttype == 3 || ttype == 4);
        if (is_special)
        {
            tmp_special_tokens.insert(id);
            if (!tok.empty())
            {
                tmp_added_tokens_by_first_byte[static_cast<unsigned char>(tok[0])].emplace_back(tok, id);
            }
        }
        if (bos_id < 0 && (tok == "<s>" || tok == "<|begin_of_text|>" || tok == "<|startoftext|>"))
        {
            tmp_bos_id = id;
        }
        if (tok == "<|im_end|>")
        {
            if (eos_id < 0)
            {
                tmp_eos_id = id;
            }
            found_im_end = true;
            tmp_special_tokens.insert(id);
            tmp_added_tokens_by_first_byte[static_cast<unsigned char>(tok[0])].emplace_back(tok, id);
        }
        else if (tok == "<|im_start|>" || tok == "<|endoftext|>" || tok == "<s>" || tok == "</s>")
        {
            if (eos_id < 0 && !found_im_end && (tok == "</s>" || tok == "<|endoftext|>"))
            {
                tmp_eos_id = id;
            }
            tmp_special_tokens.insert(id);
            tmp_added_tokens_by_first_byte[static_cast<unsigned char>(tok[0])].emplace_back(tok, id);
        }

        uint8_t byte_val = 0;
        if (isByteFallbackToken(tok, byte_val))
        {
            tmp_byte_fallback_tokens[id] = byte_val;
        }
    }

    sortAddedTokens(tmp_added_tokens_by_first_byte);

    int32_t rank = 0;
    for (const std::string& merge_str : merges)
    {
        std::string piece_a;
        std::string piece_b;
        if (parseMergeString(merge_str, piece_a, piece_b))
        {
            registerMergePair(piece_a, piece_b, rank, tmp_vocab, tmp_merges);
        }
        rank++;
    }

    bos_id_ = tmp_bos_id;
    eos_id_ = tmp_eos_id;
    unk_id_ = tmp_unk_id;
    byte_level_ = is_byte_level;
    has_bos_ = add_bos;
    prepend_space_ = !is_byte_level;
    split_metaspace_ = !is_byte_level;
    metaspace_replacement_ = is_byte_level ? std::string() : std::string("\xE2\x96\x81");
    vocab_ = std::move(tmp_vocab);
    id_to_token_ = std::move(tmp_id_to_token);
    merges_ = std::move(tmp_merges);
    special_tokens_ = std::move(tmp_special_tokens);
    byte_fallback_tokens_ = std::move(tmp_byte_fallback_tokens);
    added_tokens_by_first_byte_ = std::move(tmp_added_tokens_by_first_byte);
    if (byte_level_)
    {
        initByteLevelTables();
    }
    else
    {
        unicode_to_byte_.clear();
    }
    is_loaded_ = true;
    Logger::logMessage(Input_Format{"Bpe_Tokenizer::loadFromVocabAndMerges: Loaded (vocab={}, merges={}, special={}, byte_level={})",
                                    id_to_token_.size(), merges_.size(), special_tokens_.size(), byte_level_},
                       Log_Level::LOG_INFO,
                       false,
                       0,
                       Log_Feature::TOKENIZER);
    return true;
}

void Bpe_Tokenizer::initByteLevelTables()
{
    unicode_to_byte_.clear();
    std::array<bool, 256> direct{};
    for (int b = 33; b <= 126; ++b) direct[static_cast<size_t>(b)] = true;
    for (int b = 161; b <= 172; ++b) direct[static_cast<size_t>(b)] = true;
    for (int b = 174; b <= 255; ++b) direct[static_cast<size_t>(b)] = true;

    int extra_cp = 256;
    for (int b = 0; b < 256; ++b)
    {
        int cp = direct[static_cast<size_t>(b)] ? b : extra_cp++;
        std::string utf8;
        if (cp < 0x80)
        {
            utf8.push_back(static_cast<char>(cp));
        }
        else
        {
            utf8.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        byte_to_unicode_[static_cast<size_t>(b)] = utf8;
        unicode_to_byte_[utf8] = static_cast<uint8_t>(b);
    }
}

std::vector<std::string> Bpe_Tokenizer::splitByteLevelChunks(const std::string& text) const
{
    std::vector<std::string> chunks;
    const size_t n = text.size();
    size_t i = 0;

    auto is_ascii_alpha = [](unsigned char c) noexcept {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };
    auto is_ascii_digit = [](unsigned char c) noexcept {
        return c >= '0' && c <= '9';
    };
    auto is_ascii_space = [](unsigned char c) noexcept {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
    };
    auto is_word_start = [&](size_t pos) noexcept {
        if (pos >= n) return false;
        unsigned char c = static_cast<unsigned char>(text[pos]);
        return is_ascii_alpha(c) || c >= 0x80u;
    };
    auto is_ascii_punct = [&](size_t pos) noexcept {
        if (pos >= n) return false;
        unsigned char c = static_cast<unsigned char>(text[pos]);
        return c < 0x80u && !is_ascii_alpha(c) && !is_ascii_digit(c) && !is_ascii_space(c);
    };
    auto match_contraction = [&](size_t pos) -> size_t {
        if (pos + 1 >= n || text[pos] != '\'') return 0;
        char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[pos + 1])));
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
        if (pos + 2 < n)
        {
            char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[pos + 2])));
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
            {
                return 3;
            }
        }
        return 0;
    };

    while (i < n)
    {
        if (const size_t clen = match_contraction(i); clen > 0)
        {
            chunks.push_back(text.substr(i, clen));
            i += clen;
            continue;
        }

        if (is_word_start(i))
        {
            const size_t start = i;
            while (i < n && is_word_start(i))
            {
                const size_t step = (static_cast<unsigned char>(text[i]) >= 0x80u)
                    ? getValidatedUtf8CharLength(text, i)
                    : 1u;
                i += step;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        unsigned char c0 = static_cast<unsigned char>(text[i]);
        if (c0 != '\r' && c0 != '\n' && !is_ascii_digit(c0) && (i + 1 < n) && is_word_start(i + 1))
        {
            const size_t start = i;
            ++i;
            while (i < n && is_word_start(i))
            {
                const size_t step = (static_cast<unsigned char>(text[i]) >= 0x80u)
                    ? getValidatedUtf8CharLength(text, i)
                    : 1u;
                i += step;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        if (is_ascii_digit(c0))
        {
            chunks.push_back(text.substr(i, 1));
            ++i;
            continue;
        }

        if (is_ascii_punct(i) || (c0 == ' ' && (i + 1 < n) && is_ascii_punct(i + 1)))
        {
            const size_t start = i;
            if (c0 == ' ') ++i;
            while (i < n && is_ascii_punct(i)) ++i;
            while (i < n && (text[i] == '\r' || text[i] == '\n')) ++i;
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        if (is_ascii_space(c0))
        {
            const size_t start = i;
            size_t last_nl = std::string::npos;
            while (i < n && is_ascii_space(static_cast<unsigned char>(text[i])))
            {
                if (text[i] == '\r' || text[i] == '\n')
                {
                    last_nl = i + 1;
                }
                ++i;
            }
            if (last_nl != std::string::npos)
            {
                chunks.push_back(text.substr(start, last_nl - start));
                i = last_nl;
            }
            else
            {
                if (i < n && (i - start) > 1)
                {
                    --i;
                }
                chunks.push_back(text.substr(start, i - start));
            }
            continue;
        }

        chunks.push_back(text.substr(i, 1));
        ++i;
    }
    return chunks;
}

std::vector<int32_t> Bpe_Tokenizer::encodeSegmentBpe(const std::string& prepended_chunk) const
{
    if (prepended_chunk.empty())
    {
        return {};
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(prepended_chunk.size());

    size_t i = 0;
    const size_t text_len = prepended_chunk.size();
    while (i < text_len)
    {
        const size_t char_len = getValidatedUtf8CharLength(prepended_chunk, i);
        std::string cp = prepended_chunk.substr(i, char_len);
        auto it = vocab_.find(cp);
        if (it != vocab_.end())
        {
            token_ids.push_back(it->second);
        }
        else
        {
            for (size_t b_idx = 0; b_idx < char_len; ++b_idx)
            {
                unsigned char b = static_cast<unsigned char>(prepended_chunk[i + b_idx]);
                char hex_buf[16];
                std::snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", b);
                auto fb_it = vocab_.find(hex_buf);
                if (fb_it != vocab_.end())
                {
                    token_ids.push_back(fb_it->second);
                }
                else
                {
                    token_ids.push_back(unk_id_);
                }
            }
        }
        i += char_len;
    }

    if (token_ids.size() < 2)
    {
        return token_ids;
    }

    const int32_t n = static_cast<int32_t>(token_ids.size());
    std::vector<int32_t> prev(n);
    std::vector<int32_t> next(n);
    std::vector<uint8_t> alive(n, 1);
    std::vector<uint32_t> version(n, 0);

    for (int32_t idx = 0; idx < n; ++idx)
    {
        prev[idx] = idx - 1;
        next[idx] = (idx + 1 < n) ? (idx + 1) : -1;
    }

    struct HeapEntry
    {
        int32_t rank;
        int32_t left;
        int32_t right;
        uint32_t ver_l;
        uint32_t ver_r;
        int32_t merged_id;
    };

    struct HeapCompare
    {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const noexcept
        {
            if (a.rank != b.rank)
            {
                return a.rank > b.rank;
            }
            return a.left > b.left;
        }
    };

    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCompare> pq;

    auto pushPair = [&](int32_t l, int32_t r) {
        if (l == -1 || r == -1)
        {
            return;
        }
        uint64_t pair_key = (static_cast<uint64_t>(token_ids[l]) << 32) | static_cast<uint32_t>(token_ids[r]);
        auto it = merges_.find(pair_key);
        if (it != merges_.end())
        {
            pq.push(HeapEntry{ it->second.rank, l, r, version[l], version[r], it->second.merged_id });
        }
    };

    for (int32_t idx = 0; idx + 1 < n; ++idx)
    {
        pushPair(idx, idx + 1);
    }

    while (!pq.empty())
    {
        HeapEntry top = pq.top();
        pq.pop();

        if (!alive[top.left] || !alive[top.right] ||
            next[top.left] != top.right ||
            version[top.left] != top.ver_l || version[top.right] != top.ver_r)
        {
            continue;
        }

        token_ids[top.left] = top.merged_id;
        version[top.left]++;
        alive[top.right] = 0;

        int32_t r_next = next[top.right];
        next[top.left] = r_next;
        if (r_next != -1)
        {
            prev[r_next] = top.left;
        }

        pushPair(prev[top.left], top.left);
        pushPair(top.left, next[top.left]);
    }

    std::vector<int32_t> merged;
    merged.reserve(n);
    for (int32_t cur = 0; cur != -1; cur = next[cur])
    {
        if (alive[cur])
        {
            merged.push_back(token_ids[cur]);
        }
    }
    return merged;
}

std::vector<int32_t> Bpe_Tokenizer::encode(const std::string& text, bool add_bos) const
{
    if (!is_loaded_ || text.empty())
    {
        return (add_bos && has_bos_ && is_loaded_) ? std::vector<int32_t>{ bos_id_ } : std::vector<int32_t>{};
    }

    std::vector<int32_t> tokens;
    if (add_bos && has_bos_)
    {
        tokens.push_back(bos_id_);
    }

    auto processPlainSlice = [&](const std::string& slice, bool at_text_start) {
        if (slice.empty())
        {
            return;
        }

        if (byte_level_)
        {
            for (const std::string& chunk : splitByteLevelChunks(slice))
            {
                std::string mapped;
                mapped.reserve(chunk.size() * 2);
                for (unsigned char b : chunk)
                {
                    mapped += byte_to_unicode_[b];
                }
                auto seg_ids = encodeSegmentBpe(mapped);
                tokens.insert(tokens.end(), seg_ids.begin(), seg_ids.end());
            }
            return;
        }

        std::string prepended;
        prepended.reserve(slice.size() + metaspace_replacement_.size() + 8);
        if (at_text_start && prepend_space_ && slice[0] != ' ')
        {
            prepended += metaspace_replacement_;
        }

        for (char c : slice)
        {
            if (c == ' ')
            {
                prepended += metaspace_replacement_;
            }
            else
            {
                prepended += c;
            }
        }

        if (prepended.empty())
        {
            return;
        }

        if (split_metaspace_ && !metaspace_replacement_.empty())
        {
            size_t chunk_start = 0;
            size_t search_from = 0;
            if (prepended.compare(0, metaspace_replacement_.size(), metaspace_replacement_) == 0)
            {
                search_from = metaspace_replacement_.size();
            }
            while (search_from < prepended.size())
            {
                size_t match_pos = prepended.find(metaspace_replacement_, search_from);
                if (match_pos == std::string::npos)
                {
                    break;
                }
                if (match_pos > chunk_start)
                {
                    auto seg_ids = encodeSegmentBpe(prepended.substr(chunk_start, match_pos - chunk_start));
                    tokens.insert(tokens.end(), seg_ids.begin(), seg_ids.end());
                }
                chunk_start = match_pos;
                search_from = match_pos + metaspace_replacement_.size();
            }
            if (chunk_start < prepended.size())
            {
                auto seg_ids = encodeSegmentBpe(prepended.substr(chunk_start));
                tokens.insert(tokens.end(), seg_ids.begin(), seg_ids.end());
            }
        }
        else
        {
            auto seg_ids = encodeSegmentBpe(prepended);
            tokens.insert(tokens.end(), seg_ids.begin(), seg_ids.end());
        }
    };

    if (added_tokens_by_first_byte_.empty())
    {
        processPlainSlice(text, true);
        return tokens;
    }

    size_t pos = 0;
    size_t plain_start = 0;
    const size_t n = text.size();
    while (pos < n)
    {
        auto map_it = added_tokens_by_first_byte_.find(static_cast<unsigned char>(text[pos]));
        int32_t matched_id = -1;
        size_t matched_len = 0;
        if (map_it != added_tokens_by_first_byte_.end())
        {
            for (const auto& [added_str, added_id] : map_it->second)
            {
                if (pos + added_str.size() <= n &&
                    text.compare(pos, added_str.size(), added_str) == 0)
                {
                    matched_id = added_id;
                    matched_len = added_str.size();
                    break;
                }
            }
        }

        if (matched_len > 0)
        {
            if (pos > plain_start)
            {
                processPlainSlice(text.substr(plain_start, pos - plain_start), plain_start == 0);
            }
            tokens.push_back(matched_id);
            pos += matched_len;
            plain_start = pos;
        }
        else
        {
            ++pos;
        }
    }

    if (plain_start < n)
    {
        processPlainSlice(text.substr(plain_start), plain_start == 0);
    }

    return tokens;
}

std::string Bpe_Tokenizer::decode(const std::vector<int32_t>& token_ids,
                                  bool skip_special,
                                  bool strip_leading_space) const
{
    if (!is_loaded_ || token_ids.empty())
    {
        return "";
    }

    std::vector<uint8_t> raw_bytes;
    raw_bytes.reserve(token_ids.size() * 4);

    for (int32_t id : token_ids)
    {
        if (skip_special && isSpecialToken(id))
        {
            continue;
        }

        if (byte_level_)
        {
            if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size())
            {
                continue;
            }
            const std::string& piece = id_to_token_[static_cast<size_t>(id)];
            if (isSpecialToken(id))
            {
                raw_bytes.insert(raw_bytes.end(), piece.begin(), piece.end());
                continue;
            }
            size_t p = 0;
            while (p < piece.size())
            {
                const size_t step = getValidatedUtf8CharLength(piece, p);
                const std::string ch = piece.substr(p, step);
                auto uit = unicode_to_byte_.find(ch);
                if (uit != unicode_to_byte_.end())
                {
                    raw_bytes.push_back(uit->second);
                }
                else
                {
                    raw_bytes.insert(raw_bytes.end(), ch.begin(), ch.end());
                }
                p += step;
            }
            continue;
        }

        auto bf_it = byte_fallback_tokens_.find(id);
        if (bf_it != byte_fallback_tokens_.end())
        {
            raw_bytes.push_back(bf_it->second);
            continue;
        }

        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size())
        {
            const std::string& piece = id_to_token_[static_cast<size_t>(id)];
            raw_bytes.insert(raw_bytes.end(), piece.begin(), piece.end());
        }
    }

    std::string decoded(raw_bytes.begin(), raw_bytes.end());
    if (byte_level_)
    {
        return decoded;
    }

    if (!metaspace_replacement_.empty() && metaspace_replacement_ != " ")
    {
        size_t meta_pos = 0;
        while ((meta_pos = decoded.find(metaspace_replacement_, meta_pos)) != std::string::npos)
        {
            decoded.replace(meta_pos, metaspace_replacement_.length(), " ");
            meta_pos += 1;
        }
    }

    if (strip_leading_space && prepend_space_ && !decoded.empty() && decoded[0] == ' ')
    {
        decoded.erase(0, 1);
    }

    return decoded;
}

bool Bpe_Tokenizer::isLoaded() const
{
    return is_loaded_;
}

bool Bpe_Tokenizer::isSpecialToken(int32_t id) const
{
    return special_tokens_.contains(id);
}

int32_t Bpe_Tokenizer::tokenToId(const std::string& token) const
{
    auto it = vocab_.find(token);
    return it != vocab_.end() ? it->second : unk_id_;
}

std::string Bpe_Tokenizer::idToToken(int32_t id) const
{
    if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size() && !id_to_token_[static_cast<size_t>(id)].empty())
    {
        return id_to_token_[static_cast<size_t>(id)];
    }
    return "<unk>";
}

size_t Bpe_Tokenizer::getValidatedUtf8CharLength(const std::string& text, size_t pos)
{
    const size_t n = text.size();
    if (pos >= n)
    {
        return 1;
    }

    const unsigned char b0 = static_cast<unsigned char>(text[pos]);
    if (b0 < 0x80)
    {
        return 1;
    }

    if ((b0 & 0xE0) == 0xC0)
    {
        if (b0 >= 0xC2 && pos + 1 < n)
        {
            const unsigned char b1 = static_cast<unsigned char>(text[pos + 1]);
            if ((b1 & 0xC0) == 0x80)
            {
                return 2;
            }
        }
        return 1;
    }

    if ((b0 & 0xF0) == 0xE0)
    {
        if (pos + 2 < n)
        {
            const unsigned char b1 = static_cast<unsigned char>(text[pos + 1]);
            const unsigned char b2 = static_cast<unsigned char>(text[pos + 2]);
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80)
            {
                if (!(b0 == 0xE0 && b1 < 0xA0) && !(b0 == 0xED && b1 >= 0xA0))
                {
                    return 3;
                }
            }
        }
        return 1;
    }

    if ((b0 & 0xF8) == 0xF0)
    {
        if (b0 <= 0xF4 && pos + 3 < n)
        {
            const unsigned char b1 = static_cast<unsigned char>(text[pos + 1]);
            const unsigned char b2 = static_cast<unsigned char>(text[pos + 2]);
            const unsigned char b3 = static_cast<unsigned char>(text[pos + 3]);
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80)
            {
                if (!(b0 == 0xF0 && b1 < 0x90) && !(b0 == 0xF4 && b1 > 0x8F))
                {
                    return 4;
                }
            }
        }
        return 1;
    }

    return 1;
}

int Bpe_Tokenizer::hexCharToInt(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

bool Bpe_Tokenizer::isByteFallbackToken(const std::string& token, uint8_t& out_byte)
{
    if (token.size() == 6 &&
        token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>')
    {
        int hi = hexCharToInt(token[3]);
        int lo = hexCharToInt(token[4]);
        if (hi != -1 && lo != -1)
        {
            out_byte = static_cast<uint8_t>((hi << 4) | lo);
            return true;
        }
    }
    return false;
}

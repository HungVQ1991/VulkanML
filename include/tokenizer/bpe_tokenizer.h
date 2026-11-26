#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class Bpe_Tokenizer
{
public:
    struct Merge_Rule
    {
        int32_t rank;
        int32_t merged_id;
    };

    Bpe_Tokenizer() = default;
    explicit Bpe_Tokenizer(const std::string& tokenizer_json_path);
    bool load(const std::string& tokenizer_json_path);
    bool loadFromVocabAndMerges(const std::vector<std::string>& tokens,
                                const std::vector<std::string>& merges,
                                const std::vector<int32_t>& token_types = {},
                                bool is_byte_level = true,
                                int32_t bos_id = -1,
                                int32_t eos_id = -1,
                                int32_t unk_id = 0,
                                bool add_bos = false);
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int32_t>& token_ids,
                       bool skip_special = true,
                       bool strip_leading_space = true) const;

    size_t getVocabSize() const
    {
        return vocab_.size();
    }

    int32_t getBosId() const { return bos_id_; }
    int32_t getEosId() const { return eos_id_; }
    int32_t getUnkId() const { return unk_id_; }
    bool isLoaded() const;
    bool isByteLevel() const noexcept { return byte_level_; }
    bool hasBosToken() const noexcept { return has_bos_; }
    bool isSpecialToken(int32_t id) const;
    int32_t tokenToId(const std::string& token) const;
    std::string idToToken(int32_t id) const;

private:
    static size_t getValidatedUtf8CharLength(const std::string& text, size_t pos);
    static int hexCharToInt(char c);
    static bool isByteFallbackToken(const std::string& token, uint8_t& out_byte);
    static bool parseMergeString(const std::string& merge_str, std::string& piece_a, std::string& piece_b);
    static void registerMergePair(const std::string& piece_a,
                                  const std::string& piece_b,
                                  int32_t rank,
                                  const std::unordered_map<std::string, int32_t>& vocab,
                                  std::unordered_map<uint64_t, Merge_Rule>& merges);
    static void sortAddedTokens(
        std::unordered_map<unsigned char, std::vector<std::pair<std::string, int32_t>>>& added_tokens_by_first_byte);
    void initByteLevelTables();
    std::vector<std::string> splitByteLevelChunks(const std::string& text) const;
    std::vector<int32_t> encodeSegmentBpe(const std::string& prepended_chunk) const;

    bool is_loaded_ = false;
    bool byte_level_ = false;
    bool has_bos_ = true;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t unk_id_ = 0;
    bool prepend_space_ = true;
    bool split_metaspace_ = false;
    std::string metaspace_replacement_ = "\xE2\x96\x81";

    std::array<std::string, 256> byte_to_unicode_{};
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<uint64_t, Merge_Rule> merges_;
    std::unordered_set<int32_t> special_tokens_;
    std::unordered_map<int32_t, uint8_t> byte_fallback_tokens_;
    std::unordered_map<unsigned char, std::vector<std::pair<std::string, int32_t>>> added_tokens_by_first_byte_;
};

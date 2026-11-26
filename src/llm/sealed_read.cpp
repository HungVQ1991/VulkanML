#include "llm/sealed_read.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "engine/execution_engine.h"
#include "helper/json.hpp"
#include "optimizer/adam_optimizer.h"

namespace sealedread
{

namespace
{

constexpr int64_t NEVER_OPEN = (1LL << 60);

std::string trimCopy(const std::string &s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
    {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    {
        --e;
    }
    return s.substr(b, e - b);
}

std::string trimRightCopy(const std::string &s)
{
    size_t e = s.size();
    while (e > 0 && std::isspace(static_cast<unsigned char>(s[e - 1])))
    {
        --e;
    }
    return s.substr(0, e);
}

bool startsWithAt(const std::string &text, size_t pos, const std::string &prefix)
{
    if (pos + prefix.size() > text.size())
    {
        return false;
    }
    return text.compare(pos, prefix.size(), prefix) == 0;
}

std::string replaceCaseInsensitive(const std::string &haystack, const std::string &needle, const std::string &rep)
{
    if (needle.empty())
    {
        return haystack;
    }
    std::string low_h = haystack;
    std::string low_n = needle;
    for (char &c : low_h)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (char &c : low_n)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    std::string out;
    size_t pos = 0;
    while (pos < haystack.size())
    {
        size_t hit = low_h.find(low_n, pos);
        if (hit == std::string::npos)
        {
            out.append(haystack, pos, haystack.size() - pos);
            break;
        }
        out.append(haystack, pos, hit - pos);
        out.append(rep);
        pos = hit + needle.size();
    }
    return out;
}

std::vector<std::string> splitNormalizedTokens(const std::string &s)
{
    std::string norm = normalizeAnswer(s);
    std::vector<std::string> toks;
    std::istringstream iss(norm);
    std::string tok;
    while (iss >> tok)
    {
        toks.push_back(tok);
    }
    return toks;
}

std::vector<int32_t> extractRowIndicesFromTarget(const Seq &seq, const std::vector<Step> &steps, const ISealed_Read_Tokenizer &tok, size_t step_idx)
{
    std::vector<int32_t> rows;
    std::vector<size_t> positions;
    for (size_t i = 0; i < seq.size(); ++i)
    {
        if (seq.stp[i] == static_cast<int32_t>(step_idx))
        {
            positions.push_back(i);
        }
    }
    const size_t n_open = tok.encode("<" + steps[step_idx].tag + ">").size();
    const size_t n_text = tok.encode(steps[step_idx].text).size();
    for (size_t j = n_open; j < n_open + n_text && j < positions.size(); ++j)
    {
        if (positions[j] > 0)
        {
            rows.push_back(static_cast<int32_t>(positions[j] - 1));
        }
    }
    return rows;
}

std::vector<float> computeLogSoftmaxRows(Causal_LM &model, const Seq &seq, const std::vector<int32_t> &rows, size_t chunk = 128)
{
    if (rows.empty())
    {
        return {};
    }
    const Execution_Target dev = model.getConfig().execution_target;
    const size_t T = seq.size();
    const size_t V = model.getConfig().vocab_size;

    std::vector<float> id_f(T);
    std::vector<float> pos_f(T);
    for (size_t i = 0; i < T; ++i)
    {
        id_f[i] = static_cast<float>(seq.ids[i]);
        pos_f[i] = static_cast<float>(seq.pos[i]);
    }
    Tensor ids_t(Shape{ 1, T }, std::move(id_f), dev);
    Tensor pos_t(Shape{ 1, T }, std::move(pos_f), dev);

    std::vector<int32_t> all_k(T);
    std::iota(all_k.begin(), all_k.end(), 0);
    Tensor mask_t = seq.buildDenseMask(0, T, all_k, dev);

    ops::llm::Attention_Config attn_cfg{};
    attn_cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
    attn_cfg.attn_mask = &mask_t;

    const bool prev_train = model.getTrainingMode();
    if (prev_train)
    {
        model.setTrainingMode(false);
    }
    Tensor hidden = model.forwardHidden(ids_t, false, &pos_t, &attn_cfg);
    if (dev == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::vector<float> out(rows.size() * V, 0.0f);
    for (size_t off = 0; off < rows.size(); off += chunk)
    {
        const size_t cur = std::min(chunk, rows.size() - off);
        std::vector<float> idx_f(cur);
        for (size_t j = 0; j < cur; ++j)
        {
            idx_f[j] = static_cast<float>(rows[off + j]);
        }
        Tensor idx_t(Shape{ cur }, std::move(idx_f), dev);
        Tensor logits = model.forwardTargetLogits(hidden, idx_t);
        if (dev == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().executeGraph();
        }
        std::vector<float> host = logits.getData();
        for (size_t r = 0; r < cur; ++r)
        {
            const float *src_row = host.data() + r * V;
            float *dst_row = out.data() + (off + r) * V;
            float max_v = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < V; ++v)
            {
                max_v = std::max(max_v, src_row[v]);
            }
            double sum_exp = 0.0;
            for (size_t v = 0; v < V; ++v)
            {
                sum_exp += std::exp(static_cast<double>(src_row[v] - max_v));
            }
            const float log_z = max_v + static_cast<float>(std::log(sum_exp));
            for (size_t v = 0; v < V; ++v)
            {
                dst_row[v] = src_row[v] - log_z;
            }
        }
    }
    if (prev_train)
    {
        model.setTrainingMode(true);
    }
    return out;
}

const std::vector<std::string> &specialTokensByte()
{
    static const std::vector<std::string> special = { "<|im_start|>", "<|im_end|>", "<|endoftext|>" };
    return special;
}

std::string fixFormatDriver(Episode &ep, const std::string &tag, std::string text, bool force, const Driver_Config &cfg)
{
    if (tag == "reason")
    {
        if (force && parseAlpha(text).value_or("") != "answer")
        {
            text = forceAlphaAnswer(text);
        }
        if (!parseAlpha(text).has_value())
        {
            if (!cfg.lenient)
            {
                ep.error = "reason without alpha";
                ep.done = true;
                return {};
            }
            ep.stats.format_fixes++;
            text = trimRightCopy(text);
            if (!text.empty())
            {
                text += " ";
            }
            text += "alpha=query";
        }
    }
    else if (tag == "recheck" && !parseBeta(text).has_value())
    {
        if (!cfg.lenient)
        {
            ep.error = "recheck without beta";
            ep.done = true;
            return {};
        }
        ep.stats.format_fixes++;
        text = trimRightCopy(text);
        if (!text.empty())
        {
            text += " ";
        }
        text += "beta=valid_to_think";
    }
    return text;
}

}

std::vector<int32_t> ISealed_Read_Tokenizer::encodePieces(const std::vector<std::string> &pieces) const
{
    std::vector<int32_t> out;
    for (const auto &p : pieces)
    {
        auto part = encode(p);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<int32_t> Byte_Tokenizer::encode(const std::string &text) const
{
    const auto &sp = specialTokensByte();
    std::vector<int32_t> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size())
    {
        bool matched_sp = false;
        for (size_t j = 0; j < sp.size(); ++j)
        {
            if (startsWithAt(text, i, sp[j]))
            {
                out.push_back(static_cast<int32_t>(256 + j));
                i += sp[j].size();
                matched_sp = true;
                break;
            }
        }
        if (matched_sp)
        {
            continue;
        }
        size_t nxt = text.size();
        for (const auto &s : sp)
        {
            size_t found = text.find(s, i);
            if (found != std::string::npos && found < nxt)
            {
                nxt = found;
            }
        }
        for (size_t k = i; k < nxt; ++k)
        {
            out.push_back(static_cast<int32_t>(static_cast<unsigned char>(text[k])));
        }
        i = nxt;
    }
    return out;
}

std::string Byte_Tokenizer::decode(const std::vector<int32_t> &ids) const
{
    const auto &sp = specialTokensByte();
    std::string out;
    out.reserve(ids.size());
    for (int32_t t : ids)
    {
        if (t >= 0 && t < 256)
        {
            out.push_back(static_cast<char>(t));
        }
        else if (t >= 256 && t < static_cast<int32_t>(256 + sp.size()))
        {
            out += sp[static_cast<size_t>(t - 256)];
        }
    }
    return out;
}

std::unordered_set<int32_t> Byte_Tokenizer::eosIds() const
{
    return { IM_END_ID, ENDOFTEXT_ID };
}

Bpe_Sealed_Read_Tokenizer::Bpe_Sealed_Read_Tokenizer(const Bpe_Tokenizer &_tok)
    : tok(&_tok)
{
}

size_t Bpe_Sealed_Read_Tokenizer::vocabSize() const noexcept
{
    return tok ? tok->getVocabSize() : 0;
}

std::vector<int32_t> Bpe_Sealed_Read_Tokenizer::encode(const std::string &text) const
{
    if (!tok || text.empty())
    {
        return {};
    }
    return tok->encode(text, false);
}

std::string Bpe_Sealed_Read_Tokenizer::decode(const std::vector<int32_t> &ids) const
{
    if (!tok || ids.empty())
    {
        return {};
    }
    return tok->decode(ids, false, false);
}

std::unordered_set<int32_t> Bpe_Sealed_Read_Tokenizer::eosIds() const
{
    std::unordered_set<int32_t> out;
    if (!tok)
    {
        return out;
    }
    if (tok->getEosId() >= 0)
    {
        out.insert(tok->getEosId());
    }
    const auto im_end = tok->encode("<|im_end|>", false);
    if (im_end.size() == 1)
    {
        out.insert(im_end[0]);
    }
    const auto eot = tok->encode("<|endoftext|>", false);
    if (eot.size() == 1)
    {
        out.insert(eot[0]);
    }
    return out;
}

std::vector<int32_t> Sample::goldIds() const
{
    std::vector<int32_t> out;
    for (const auto &s : segments)
    {
        if (s.gold)
        {
            out.push_back(s.sid);
        }
    }
    return out;
}

std::unordered_map<int32_t, size_t> Sample::order() const
{
    std::unordered_map<int32_t, size_t> out;
    out.reserve(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        out[segments[i].sid] = i;
    }
    return out;
}

std::unordered_set<int32_t> Sample::ids() const
{
    std::unordered_set<int32_t> out;
    out.reserve(segments.size());
    for (const auto &s : segments)
    {
        out.insert(s.sid);
    }
    return out;
}

const Segment *Sample::findBySid(int32_t sid) const noexcept
{
    for (const auto &s : segments)
    {
        if (s.sid == sid)
        {
            return &s;
        }
    }
    return nullptr;
}

const Segment &Sample::bySid(int32_t sid) const
{
    const Segment *p = findBySid(sid);
    if (!p)
    {
        throw std::out_of_range("Sample::bySid: segment ID not found: " + std::to_string(sid));
    }
    return *p;
}

std::vector<Sample> generateSyntheticDataset(const Synthetic_Dataset_Config &cfg)
{
    static const std::vector<std::string> nations = {
        "French", "German", "Italian", "Spanish", "Polish", "Dutch", "Greek", "Swedish"
    };
    static const std::vector<std::string> cities = {
        "Riverton", "Oakridge", "Fairhaven", "Kingsford", "Ashford", "Westbrook", "Greenvale", "Rockport"
    };

    std::mt19937 rng(cfg.seed);
    std::vector<Sample> out;
    out.reserve(cfg.n_samples);

    auto allocUniqueSids = [&](size_t count) {
        std::unordered_set<int32_t> used;
        std::vector<int32_t> ids;
        ids.reserve(count);
        std::uniform_int_distribution<int32_t> dist(100, 9899);
        while (ids.size() < count)
        {
            int32_t cand = dist(rng);
            if (used.insert(cand).second)
            {
                ids.push_back(cand);
            }
        }
        return ids;
    };

    const size_t n_nocontext = static_cast<size_t>(std::round(static_cast<double>(cfg.n_samples) * cfg.nocontext_ratio));
    for (size_t idx = 0; idx < cfg.n_samples; ++idx)
    {
        const int32_t item_num = static_cast<int32_t>(idx) + cfg.id_offset;
        Sample s;
        s.answer_kind = "open";

        if (idx < n_nocontext)
        {
            const int a = 12 + static_cast<int>((item_num * 7) % 40);
            const int b = 5 + static_cast<int>((item_num * 11) % 30);
            s.qid = "nc_" + std::to_string(item_num);
            s.dataset = "synthetic_arith";
            s.qtype = "nocontext";
            s.cluster = s.qid;
            s.question = "What is " + std::to_string(a) + " + " + std::to_string(b) + "?";
            s.answers = { std::to_string(a + b) };
            auto sids = allocUniqueSids(cfg.n_distractors);
            for (size_t d = 0; d < cfg.n_distractors; ++d)
            {
                Segment seg;
                seg.sid = sids[d];
                seg.title = "Village " + std::to_string(item_num) + "-" + std::to_string(d);
                seg.section = "Introduction";
                seg.text = seg.title + " has " + std::to_string(100 + static_cast<int>(d) * 13) + " inhabitants and a quiet market.";
                seg.doc = seg.title;
                seg.gold = false;
                s.segments.push_back(std::move(seg));
            }
        }
        else if (idx % 2 == 0)
        {
            const std::string &nation = nations[static_cast<size_t>(item_num) % nations.size()];
            const std::string film = "Film Alpha " + std::to_string(item_num);
            const std::string person = "Director Person " + std::to_string(item_num);
            s.qid = "w_" + std::to_string(item_num);
            s.dataset = "2wikimultihopqa_syn";
            s.qtype = "multihop";
            s.cluster = s.qid;
            s.question = "What is the nationality of the director of " + film + "?";
            s.answers = { nation };

            auto sids = allocUniqueSids(2 + cfg.n_distractors);
            std::vector<Segment> raw;
            raw.push_back(Segment{
                sids[0],
                film,
                "Introduction",
                film + " is a " + std::to_string(1950 + (item_num % 45)) + " drama film directed by " + person + ". It was shot on location.",
                film,
                true
            });
            raw.push_back(Segment{
                sids[1],
                person,
                "Biography",
                person + " is a " + nation + " film director. He was active for " + std::to_string(10 + (item_num % 30)) + " years.",
                person,
                true
            });
            for (size_t d = 0; d < cfg.n_distractors; ++d)
            {
                std::string title = "Town " + std::to_string(item_num) + "-" + std::to_string(d);
                raw.push_back(Segment{
                    sids[2 + d],
                    title,
                    "Overview",
                    title + " is a small settlement with " + std::to_string(80 + static_cast<int>(d) * 9) + " houses.",
                    title,
                    false
                });
            }
            s.hops = {
                Hop{ { sids[0] }, person },
                Hop{ { sids[1] }, nation }
            };
            std::shuffle(raw.begin(), raw.end(), rng);
            s.segments = std::move(raw);
        }
        else
        {
            const std::string book = "Harbor Story " + std::to_string(item_num);
            const std::string author = "Lena Marsh " + std::to_string(item_num);
            const std::string city = cities[static_cast<size_t>(item_num) % cities.size()] + " " + std::to_string(item_num);
            s.qid = "h_" + std::to_string(item_num);
            s.dataset = "hotpotqa_syn";
            s.qtype = "multihop";
            s.cluster = s.qid;
            s.question = "Where was the author of " + book + " born?";
            s.answers = { city };

            auto sids = allocUniqueSids(2 + cfg.n_distractors);
            std::vector<Segment> raw;
            raw.push_back(Segment{
                sids[0],
                book,
                "Introduction",
                book + " is a novel written by " + author + ".",
                book,
                true
            });
            raw.push_back(Segment{
                sids[1],
                author,
                "Early Life",
                author + " was born in " + city + ".",
                author,
                true
            });
            for (size_t d = 0; d < cfg.n_distractors; ++d)
            {
                std::string title = "Lake " + std::to_string(item_num) + "-" + std::to_string(d);
                raw.push_back(Segment{
                    sids[2 + d],
                    title,
                    "Geography",
                    title + " is a lake covering " + std::to_string(2 + static_cast<int>(d)) + " square kilometres.",
                    title,
                    false
                });
            }
            s.hops = {
                Hop{ { sids[0] }, author },
                Hop{ { sids[1] }, city }
            };
            std::shuffle(raw.begin(), raw.end(), rng);
            s.segments = std::move(raw);
        }
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<Sample> loadSamplesJsonl(const std::string &path, size_t limit)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("loadSamplesJsonl: unable to open file: " + path);
    }
    auto json_str = [](const nlohmann::json &obj, const char *key, const std::string &def = "") -> std::string {
        auto it = obj.find(key);
        return (it != obj.end() && it->is_string()) ? it->get<std::string>() : def;
    };
    auto json_int = [](const nlohmann::json &obj, const char *key, int64_t def = 0) -> int64_t {
        auto it = obj.find(key);
        if (it == obj.end() || !it->is_number())
        {
            return def;
        }
        return it->is_number_integer() ? it->get<int64_t>() : static_cast<int64_t>(it->get<double>());
    };
    auto json_bool = [](const nlohmann::json &obj, const char *key, bool def = false) -> bool {
        auto it = obj.find(key);
        return (it != obj.end() && it->is_boolean()) ? it->get<bool>() : def;
    };
    std::vector<Sample> out;
    std::string line;
    while (std::getline(ifs, line))
    {
        std::string t = trimCopy(line);
        if (t.empty())
        {
            continue;
        }
        nlohmann::json root = nlohmann::json::parse(t);
        Sample s;
        if (root.is_object())
        {
            s.qid = json_str(root, "qid");
            s.dataset = json_str(root, "dataset", "synthetic");
            s.qtype = json_str(root, "qtype", "multihop");
            s.question = json_str(root, "question");
            s.answer_kind = json_str(root, "answer_kind", "open");
            s.cluster = json_str(root, "cluster", s.qid);
            s.length = static_cast<size_t>(std::max<int64_t>(0, json_int(root, "length", 0)));
            if (auto it = root.find("answers"); it != root.end() && it->is_array())
            {
                for (const auto &item : *it)
                {
                    s.answers.push_back(item.is_string() ? item.get<std::string>() : std::string());
                }
            }
            if (auto it = root.find("segments"); it != root.end() && it->is_array())
            {
                for (const auto &seg_v : *it)
                {
                    if (!seg_v.is_object())
                    {
                        continue;
                    }
                    Segment seg;
                    seg.sid = static_cast<int32_t>(json_int(seg_v, "sid"));
                    seg.title = json_str(seg_v, "title");
                    seg.section = json_str(seg_v, "section", "Introduction");
                    seg.text = json_str(seg_v, "text");
                    seg.doc = json_str(seg_v, "doc");
                    seg.gold = json_bool(seg_v, "gold", false);
                    s.segments.push_back(std::move(seg));
                }
            }
            if (auto it = root.find("hops"); it != root.end() && it->is_array())
            {
                for (const auto &hv : *it)
                {
                    if (!hv.is_object())
                    {
                        continue;
                    }
                    Hop h;
                    h.bridge = json_str(hv, "bridge");
                    if (auto sit = hv.find("sids"); sit != hv.end() && sit->is_array())
                    {
                        for (const auto &sid_v : *sit)
                        {
                            const int64_t val = sid_v.is_number()
                                ? (sid_v.is_number_integer() ? sid_v.get<int64_t>() : static_cast<int64_t>(sid_v.get<double>()))
                                : 0;
                            h.sids.push_back(static_cast<int32_t>(val));
                        }
                    }
                    s.hops.push_back(std::move(h));
                }
            }
        }
        out.push_back(std::move(s));
        if (limit > 0 && out.size() >= limit)
        {
            break;
        }
    }
    return out;
}

size_t saveSamplesJsonl(const std::string &path, const std::vector<Sample> &samples)
{
    std::filesystem::path p(path);
    if (p.has_parent_path())
    {
        std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(path);
    if (!ofs.is_open())
    {
        throw std::runtime_error("saveSamplesJsonl: unable to open file: " + path);
    }
    for (const auto &s : samples)
    {
        nlohmann::json segs = nlohmann::json::array();
        for (const auto &seg : s.segments)
        {
            segs.push_back({
                { "sid", seg.sid },
                { "title", seg.title },
                { "section", seg.section },
                { "text", seg.text },
                { "doc", seg.doc },
                { "gold", seg.gold }
            });
        }
        nlohmann::json hops = nlohmann::json::array();
        for (const auto &h : s.hops)
        {
            hops.push_back({
                { "sids", h.sids },
                { "bridge", h.bridge }
            });
        }
        nlohmann::json row = {
            { "qid", s.qid },
            { "dataset", s.dataset },
            { "qtype", s.qtype },
            { "question", s.question },
            { "answer_kind", s.answer_kind },
            { "answers", s.answers },
            { "segments", std::move(segs) },
            { "hops", std::move(hops) }
        };
        ofs << row.dump() << '\n';
    }
    return samples.size();
}

std::string formatLabel(int32_t sid)
{
    return "[[S_" + std::to_string(sid) + "]]";
}

std::string formatCloseLabel(int32_t sid)
{
    return "[[/S_" + std::to_string(sid) + "]]";
}

std::vector<std::string> locatePieces(const std::vector<int32_t> &ids)
{
    std::vector<std::string> out;
    out.reserve(ids.empty() ? 0 : ids.size() * 2 - 1);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
        {
            out.push_back(" ");
        }
        out.push_back(formatLabel(ids[i]));
    }
    return out;
}

bool isMaskedTag(const std::string &tag)
{
    return tag == "reason" || tag == "query" || tag == "recheck" || tag == "answer";
}

std::optional<std::string> parseAlpha(const std::string &text)
{
    const std::string t = trimRightCopy(text);
    const size_t pos = t.rfind("alpha");
    if (pos == std::string::npos)
    {
        return std::nullopt;
    }
    size_t i = pos + 5;
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    if (i >= t.size() || t[i] != '=')
    {
        return std::nullopt;
    }
    ++i;
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    const std::string rest = t.substr(i);
    if (rest == "query" || rest == "answer")
    {
        return rest;
    }
    return std::nullopt;
}

std::optional<std::string> parseBeta(const std::string &text)
{
    const std::string t = trimRightCopy(text);
    const size_t pos = t.rfind("beta");
    if (pos == std::string::npos)
    {
        return std::nullopt;
    }
    size_t i = pos + 4;
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    if (i >= t.size() || t[i] != '=')
    {
        return std::nullopt;
    }
    ++i;
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    const std::string rest = t.substr(i);
    if (rest == "valid_to_think" || rest == "explore_query" || rest == "reject_and_requery")
    {
        return rest;
    }
    return std::nullopt;
}

std::vector<int32_t> parseIds(const std::string &text)
{
    std::vector<int32_t> out;
    size_t pos = 0;
    while ((pos = text.find("[[S_", pos)) != std::string::npos)
    {
        size_t num_start = pos + 4;
        size_t num_end = num_start;
        while (num_end < text.size() && std::isdigit(static_cast<unsigned char>(text[num_end])) && (num_end - num_start) < 4)
        {
            ++num_end;
        }
        if (num_end > num_start && startsWithAt(text, num_end, "]]"))
        {
            out.push_back(std::stoi(text.substr(num_start, num_end - num_start)));
            pos = num_end + 2;
        }
        else
        {
            pos = num_start;
        }
    }
    return out;
}

std::string forceAlphaAnswer(const std::string &text)
{
    std::string t = trimRightCopy(text);
    const size_t pos = t.rfind("alpha");
    if (pos != std::string::npos)
    {
        size_t i = pos + 5;
        while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
        if (i < t.size() && t[i] == '=')
        {
            ++i;
            while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
            const std::string rest = t.substr(i);
            if (rest == "query" || rest == "answer")
            {
                return t.substr(0, pos) + "alpha=answer";
            }
        }
    }
    if (!t.empty())
    {
        t += " ";
    }
    return trimCopy(t + "alpha=answer");
}

size_t locateRounds(const std::vector<Step> &steps)
{
    size_t count = 0;
    for (const auto &s : steps)
    {
        if (s.tag == "locate")
        {
            ++count;
        }
    }
    return count;
}

std::vector<int32_t> pastedIds(const std::vector<Step> &steps)
{
    std::vector<int32_t> out;
    for (const auto &s : steps)
    {
        if (s.tag == "knowledge")
        {
            out.insert(out.end(), s.ids.begin(), s.ids.end());
        }
    }
    return out;
}

std::vector<int32_t> locatedIds(const std::vector<Step> &steps)
{
    std::vector<int32_t> out;
    for (const auto &s : steps)
    {
        if (s.tag == "locate")
        {
            out.insert(out.end(), s.ids.begin(), s.ids.end());
        }
    }
    return out;
}

bool mustAnswer(const std::vector<Step> &steps, const Proto_Config &cfg)
{
    return locateRounds(steps) >= cfg.r_max;
}

std::optional<std::string> nextTag(const std::vector<Step> &steps, const Proto_Config &cfg, std::string *err)
{
    if (err)
    {
        err->clear();
    }
    if (steps.empty())
    {
        return std::string("reason");
    }
    const Step &last = steps.back();
    if (last.tag == "reason")
    {
        const auto a = parseAlpha(last.text);
        if (!a.has_value())
        {
            if (err) *err = "reason without alpha";
            return std::nullopt;
        }
        if (*a == "answer")
        {
            return std::string("answer");
        }
        if (locateRounds(steps) >= cfg.r_max)
        {
            if (err) *err = "alpha=query after r_max";
            return std::nullopt;
        }
        return std::string("query");
    }
    if (last.tag == "query")
    {
        return std::string("locate");
    }
    if (last.tag == "locate")
    {
        return std::string("knowledge");
    }
    if (last.tag == "knowledge")
    {
        return std::string("recheck");
    }
    if (last.tag == "recheck")
    {
        const auto b = parseBeta(last.text);
        if (!b.has_value())
        {
            if (err) *err = "recheck without beta";
            return std::nullopt;
        }
        if (*b == "reject_and_requery" && locateRounds(steps) < cfg.r_max)
        {
            return std::string("query");
        }
        return std::string("reason");
    }
    if (last.tag == "answer")
    {
        return std::nullopt;
    }
    if (err) *err = "unknown tag " + last.tag;
    return std::nullopt;
}

std::vector<int32_t> orderInContext(const std::unordered_map<int32_t, size_t> &order, const std::vector<int32_t> &ids)
{
    std::vector<int32_t> sorted = ids;
    std::sort(sorted.begin(), sorted.end(), [&](int32_t a, int32_t b) {
        return order.at(a) < order.at(b);
    });
    return sorted;
}

std::string checkLocate(const std::vector<int32_t> &ids,
                        const std::unordered_set<int32_t> &existing,
                        const std::unordered_set<int32_t> &already_pasted,
                        size_t k_max)
{
    if (ids.empty() || ids.size() > k_max)
    {
        return "locate with " + std::to_string(ids.size()) + " IDs";
    }
    std::unordered_set<int32_t> uniq;
    for (int32_t s : ids)
    {
        if (!uniq.insert(s).second)
        {
            return "locate repeats an ID";
        }
        if (existing.find(s) == existing.end())
        {
            return "locate ID " + std::to_string(s) + " does not exist";
        }
        if (already_pasted.find(s) != already_pasted.end())
        {
            return "locate ID " + std::to_string(s) + " was already pasted";
        }
    }
    return {};
}

std::vector<std::string> validateSteps(const std::vector<Step> &steps,
                                       const Sample &sample,
                                       const Proto_Config &cfg,
                                       const ISealed_Read_Tokenizer *tok)
{
    const auto existing = sample.ids();
    const auto order = sample.order();
    std::vector<std::string> errs;
    std::vector<Step> hist;
    std::unordered_set<int32_t> already;
    bool broken = false;

    for (size_t i = 0; i < steps.size(); ++i)
    {
        const Step &st = steps[i];
        std::string next_err;
        auto want = nextTag(hist, cfg, &next_err);
        if (!next_err.empty())
        {
            errs.push_back("step " + std::to_string(i) + ": " + next_err);
            broken = true;
            break;
        }
        if (!want.has_value())
        {
            errs.push_back("step " + std::to_string(i) + ": content after <answer>");
            broken = true;
            break;
        }
        if (st.tag != *want)
        {
            errs.push_back("step " + std::to_string(i) + ": <" + st.tag + "> where <" + *want + "> was due");
            broken = true;
            break;
        }
        if (st.tag == "reason" && mustAnswer(hist, cfg) && parseAlpha(st.text).value_or("") != "answer")
        {
            errs.push_back("step " + std::to_string(i) + ": reason after round r_max must answer");
        }
        if (st.tag == "locate")
        {
            std::string e = checkLocate(st.ids, existing, already, cfg.k);
            if (!e.empty())
            {
                errs.push_back("step " + std::to_string(i) + ": " + e);
            }
            if (tok != nullptr && st.by == "model")
            {
                size_t n = tok->encodePieces(locatePieces(st.ids)).size();
                if (n > cfg.locate_max_tokens)
                {
                    errs.push_back("step " + std::to_string(i) + ": locate over cap");
                }
            }
        }
        if (st.tag == "knowledge")
        {
            std::vector<int32_t> new_ids;
            if (!hist.empty())
            {
                std::unordered_set<int32_t> seen_prev;
                for (int32_t sid : hist.back().ids)
                {
                    if (seen_prev.insert(sid).second &&
                        order.find(sid) != order.end() &&
                        already.find(sid) == already.end())
                    {
                        new_ids.push_back(sid);
                    }
                }
            }
            const auto want_ids = orderInContext(order, new_ids);
            if (st.by != "engine" || st.ids != want_ids)
            {
                errs.push_back("step " + std::to_string(i) + ": knowledge not written by the engine as pasted");
            }
            already.insert(st.ids.begin(), st.ids.end());
        }
        if (isMaskedTag(st.tag) && tok != nullptr)
        {
            auto it = cfg.caps.find(st.tag);
            if (it != cfg.caps.end() && it->second > 0 && tok->encode(st.text).size() > it->second)
            {
                errs.push_back("step " + std::to_string(i) + ": <" + st.tag + "> over its cap of " + std::to_string(it->second) + " tokens");
            }
        }
        hist.push_back(st);
    }

    if (!broken)
    {
        std::string next_err;
        auto want = nextTag(hist, cfg, &next_err);
        if (!next_err.empty())
        {
            errs.push_back(next_err);
        }
        else if (want.has_value())
        {
            errs.push_back("solution does not end with <answer>");
        }
    }
    if (locateRounds(steps) > cfg.r_max)
    {
        errs.push_back(std::to_string(locateRounds(steps)) + " rounds > r_max");
    }
    size_t answer_count = 0;
    for (const auto &s : steps)
    {
        if (s.tag == "answer")
        {
            ++answer_count;
        }
    }
    if (answer_count != 1)
    {
        errs.push_back("not exactly one <answer>");
    }
    return errs;
}

std::string answerOf(const std::vector<Step> &steps)
{
    for (auto it = steps.rbegin(); it != steps.rend(); ++it)
    {
        if (it->tag == "answer")
        {
            return trimCopy(it->text);
        }
    }
    return {};
}

std::string Prompts::head() const
{
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
           "<|im_start|>user\n" + instruction + "\n\n";
}

std::string Prompts::question(const Sample &s) const
{
    return "Question: " + s.question + "\n\n";
}

std::string Prompts::segment(const Segment &seg, std::optional<int32_t> shown) const
{
    const int32_t k = shown.value_or(seg.sid);
    if (show_titles)
    {
        return formatLabel(k) + " (" + seg.title + " / " + seg.section + ") " + seg.text + " " + formatCloseLabel(k);
    }
    return formatLabel(k) + " " + seg.text + " " + formatCloseLabel(k);
}

std::vector<std::pair<std::string, int32_t>> Prompts::contextPieces(const Sample &s) const
{
    std::vector<std::pair<std::string, int32_t>> out;
    out.emplace_back("Documents:\n\n", -1);
    for (size_t i = 0; i < s.segments.size(); ++i)
    {
        if (i > 0)
        {
            out.emplace_back("\n\n", -1);
        }
        out.emplace_back(segment(s.segments[i]), s.segments[i].sid);
    }
    out.emplace_back("\n\n", -1);
    return out;
}

std::string Prompts::questionRepeat(const Sample &s, bool repeat_q) const
{
    return (repeat_q ? ("Question (repeated): " + s.question) : std::string()) + "<|im_end|>\n";
}

std::string Prompts::assistantStart() const
{
    return "<|im_start|>assistant\n" + assistant_prefix;
}

std::string Prompts::knowledgeBody(const Sample &s,
                                   const std::vector<int32_t> &ids,
                                   const std::unordered_map<int32_t, int32_t> &alias) const
{
    std::string out = "\n";
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
        {
            out += "\n";
        }
        std::optional<int32_t> shown;
        auto it = alias.find(ids[i]);
        if (it != alias.end())
        {
            shown = it->second;
        }
        out += segment(s.bySid(ids[i]), shown);
    }
    out += "\n";
    return out;
}

bool visible(size_t q_idx, size_t k_idx, const Seq &seq, const Mask_Opts &opts) noexcept
{
    const Token_Kind qk = seq.kind[q_idx];
    const Token_Kind kk = seq.kind[k_idx];
    if (qk == Token_Kind::PAD)
    {
        return k_idx == q_idx;
    }
    if (k_idx > q_idx || kk == Token_Kind::PAD)
    {
        return false;
    }
    if (qk <= Token_Kind::C)
    {
        return true;
    }

    const bool q_loc = (qk == Token_Kind::LOC);
    if (kk == Token_Kind::LOC)
    {
        return opts.hide_locate ? (q_loc && seq.rnd[k_idx] == seq.rnd[q_idx]) : true;
    }
    if (kk == Token_Kind::C)
    {
        if (opts.branch == Branch::NOMASK)
        {
            return true;
        }
        if (opts.branch == Branch::DAFT)
        {
            return q_loc || (seq.open_at[k_idx] <= static_cast<int64_t>(q_idx));
        }
        return q_loc;
    }
    if (opts.locate_history == Locate_History::QUERY_ONLY)
    {
        const bool is_hist = (kk == Token_Kind::GEN || kk == Token_Kind::CPY || kk == Token_Kind::KNOW);
        if (q_loc && is_hist)
        {
            return seq.stp[k_idx] == seq.qref[q_idx];
        }
    }
    return true;
}

std::vector<int32_t> Seq::targetIndices() const
{
    std::vector<int32_t> out;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (target[i] != 0)
        {
            out.push_back(static_cast<int32_t>(i));
        }
    }
    return out;
}

std::vector<int32_t> Seq::maskedTargetRows() const
{
    std::vector<int32_t> rows;
    for (size_t i = 1; i < ids.size(); ++i)
    {
        if (target[i] != 0 && kind[i] == Token_Kind::GEN)
        {
            rows.push_back(static_cast<int32_t>(i - 1));
        }
    }
    return rows;
}

bool Seq::isVisible(size_t q_idx, size_t k_idx) const noexcept
{
    return visible(q_idx, k_idx, *this, opts);
}

std::vector<int32_t> Seq::collectActiveKvIndices(size_t q_start, size_t q_end) const
{
    const size_t k_hi = std::min(q_end, ids.size());
    std::vector<int32_t> keep;
    keep.reserve(k_hi);
    for (size_t k = 0; k < k_hi; ++k)
    {
        bool any = false;
        for (size_t q = q_start; q < q_end; ++q)
        {
            if (visible(q, k, *this, opts))
            {
                any = true;
                break;
            }
        }
        if (any)
        {
            keep.push_back(static_cast<int32_t>(k));
        }
    }
    return keep;
}

Tensor Seq::buildDenseMask(size_t q_start,
                           size_t q_end,
                           const std::vector<int32_t> &k_indices,
                           Execution_Target target_dev) const
{
    const size_t q_len = (q_end > q_start) ? (q_end - q_start) : 0;
    const size_t k_len = k_indices.size();
    std::vector<float> mask_vals(q_len * k_len, 0.0f);
    for (size_t qi = 0; qi < q_len; ++qi)
    {
        const size_t q_idx = q_start + qi;
        for (size_t ki = 0; ki < k_len; ++ki)
        {
            const size_t k_idx = static_cast<size_t>(k_indices[ki]);
            if (visible(q_idx, k_idx, *this, opts))
            {
                mask_vals[qi * k_len + ki] = 1.0f;
            }
        }
    }
    return Tensor(Shape{ 1, q_len, k_len }, std::move(mask_vals), target_dev);
}

Seq_Builder::Seq_Builder(const ISealed_Read_Tokenizer &_tok, const Mask_Opts &_opts, View_Mode _view)
    : tok(&_tok),
      opts(_opts),
      view(_view)
{
    data.opts = _opts;
    data.view = _view;
}

std::pair<size_t, size_t> Seq_Builder::addIds(const std::vector<int32_t> &toks,
                                              Token_Kind kind,
                                              bool target,
                                              int32_t seg,
                                              const std::vector<int32_t> *explicit_pos)
{
    const size_t start = data.ids.size();
    if (explicit_pos != nullptr && explicit_pos->size() != toks.size())
    {
        throw std::invalid_argument("Seq_Builder::addIds: token count != position count");
    }
    for (size_t j = 0; j < toks.size(); ++j)
    {
        data.ids.push_back(toks[j]);
        if (explicit_pos == nullptr)
        {
            data.pos.push_back(next_pos);
            next_pos += 1;
        }
        else
        {
            data.pos.push_back((*explicit_pos)[j]);
        }
        data.kind.push_back(kind);
        data.rnd.push_back(round);
        data.seg.push_back(seg);
        data.open_at.push_back(NEVER_OPEN);
        data.stp.push_back(step);
        data.qref.push_back((kind == Token_Kind::LOC) ? cur_qref : -1);
        data.target.push_back(target ? 1 : 0);
    }
    if (kind == Token_Kind::C)
    {
        c_tokens += toks.size();
        if (seg >= 0)
        {
            auto &vec = c_spans[seg];
            for (size_t idx = start; idx < data.ids.size(); ++idx)
            {
                vec.push_back(idx);
            }
        }
    }
    return { start, data.ids.size() };
}

std::pair<size_t, size_t> Seq_Builder::addText(const std::string &text, Token_Kind kind, bool target, int32_t seg)
{
    if (!tok)
    {
        throw std::logic_error("Seq_Builder::addText: null tokenizer");
    }
    return addIds(tok->encode(text), kind, target, seg);
}

std::pair<size_t, size_t> Seq_Builder::addPieces(const std::vector<std::string> &pieces,
                                                 Token_Kind kind,
                                                 bool target,
                                                 const std::vector<int32_t> *explicit_pos)
{
    if (!tok)
    {
        throw std::logic_error("Seq_Builder::addPieces: null tokenizer");
    }
    return addIds(tok->encodePieces(pieces), kind, target, -1, explicit_pos);
}

void Seq_Builder::skip(size_t n)
{
    next_pos += static_cast<int32_t>(n);
    c_tokens += n;
}

void Seq_Builder::open(const std::vector<int32_t> &sids, int64_t at)
{
    for (int32_t sid : sids)
    {
        auto it = c_spans.find(sid);
        if (it == c_spans.end())
        {
            continue;
        }
        for (size_t idx : it->second)
        {
            if (data.open_at[idx] == NEVER_OPEN)
            {
                data.open_at[idx] = at;
            }
        }
    }
}

void Seq_Builder::truncate(size_t n)
{
    if (n >= data.ids.size())
    {
        return;
    }
    data.ids.resize(n);
    data.pos.resize(n);
    data.kind.resize(n);
    data.rnd.resize(n);
    data.seg.resize(n);
    data.open_at.resize(n);
    data.stp.resize(n);
    data.qref.resize(n);
    data.target.resize(n);

    int32_t max_p = -1;
    for (int32_t p : data.pos)
    {
        max_p = std::max(max_p, p);
    }
    next_pos = data.pos.empty() ? 0 : (max_p + 1);
}

std::vector<std::string> tagPieces(const std::string &tag, const std::string &text, const Prompts &prompts)
{
    std::vector<std::string> pieces = { "<" + tag + ">", text, "</" + tag + ">", "\n" };
    if (tag == "answer")
    {
        pieces.push_back(prompts.eos);
    }
    return pieces;
}

std::vector<std::string> copyPieces(const Step &st)
{
    std::vector<int32_t> shown;
    shown.reserve(st.ids.size());
    for (int32_t sid : st.ids)
    {
        auto it = st.alias.find(sid);
        shown.push_back((it != st.alias.end()) ? it->second : sid);
    }
    std::sort(shown.begin(), shown.end());
    std::vector<std::string> pieces = { "<locate>" };
    auto loc = locatePieces(shown);
    pieces.insert(pieces.end(), loc.begin(), loc.end());
    pieces.push_back("</locate>");
    return pieces;
}

void Seq_Builder::addPrompt(const Sample &s,
                            const Prompts &prompts,
                            bool with_c,
                            Frame_Mode frame,
                            bool repeat_q,
                            std::optional<size_t> c_count)
{
    addText(prompts.head(), Token_Kind::A);
    addText(prompts.question(s), Token_Kind::Q);
    const auto pieces = prompts.contextPieces(s);
    if (with_c)
    {
        for (const auto &[txt, sid] : pieces)
        {
            addText(txt, Token_Kind::C, false, sid);
        }
    }
    else if (frame == Frame_Mode::FULL)
    {
        if (c_count.has_value())
        {
            skip(*c_count);
        }
        else
        {
            size_t total_c = 0;
            for (const auto &[txt, _] : pieces)
            {
                total_c += tok->encode(txt).size();
            }
            skip(total_c);
        }
    }
    addText(prompts.questionRepeat(s, repeat_q), Token_Kind::QP);
    addText(prompts.assistantStart(), Token_Kind::TP);
}

void Seq_Builder::addLocateCopy(const Step &st, size_t loc_start, size_t loc_end)
{
    const auto cpy_pieces = copyPieces(st);
    const auto cpy_toks = tok->encodePieces(cpy_pieces);
    std::vector<int32_t> cpy_pos(cpy_toks.size());
    const int32_t base_pos = data.pos[loc_start];
    for (size_t j = 0; j < cpy_toks.size(); ++j)
    {
        cpy_pos[j] = (loc_start + j < loc_end) ? data.pos[loc_start + j] : (base_pos + static_cast<int32_t>(j));
    }
    auto [cs, ce] = addIds(cpy_toks, Token_Kind::CPY, false, -1, &cpy_pos);
    (void)ce;
    if (!cpy_pos.empty())
    {
        next_pos = std::max(next_pos, cpy_pos.back() + 1);
    }
    if (opts.branch == Branch::DAFT && view == View_Mode::CHAIN)
    {
        open(st.ids, static_cast<int64_t>(cs));
    }
}

void Seq_Builder::addStep(const Sample &s, const Step &st, int32_t idx, const Prompts &prompts)
{
    step = idx;
    if (isMaskedTag(st.tag))
    {
        if (st.tag == "query")
        {
            cur_qref = idx;
        }
        const bool tgt = (view == View_Mode::CHAIN || view == View_Mode::MASKED);
        addPieces(tagPieces(st.tag, st.text, prompts), Token_Kind::GEN, tgt);
    }
    else if (st.tag == "locate")
    {
        round += 1;
        if ((view == View_Mode::CHAIN || view == View_Mode::LOCATE) && st.by == "model")
        {
            std::vector<std::string> loc_pieces = { "<locate>" };
            auto body = locatePieces(st.ids);
            loc_pieces.insert(loc_pieces.end(), body.begin(), body.end());
            loc_pieces.push_back("</locate>");
            const bool tgt = (view == View_Mode::CHAIN || view == View_Mode::LOCATE);
            auto [start_i, end_i] = addPieces(loc_pieces, Token_Kind::LOC, tgt);
            addLocateCopy(st, start_i, end_i);
        }
        else
        {
            auto [cs, ce] = addPieces(copyPieces(st), Token_Kind::CPY, false);
            (void)ce;
            if (opts.branch == Branch::DAFT && view == View_Mode::CHAIN)
            {
                open(st.ids, static_cast<int64_t>(cs));
            }
        }
    }
    else if (st.tag == "knowledge")
    {
        if (opts.branch == Branch::DAFT && view == View_Mode::CHAIN)
        {
            addPieces({ "\n", "<knowledge>", "</knowledge>", "\n" }, Token_Kind::KNOW, false);
        }
        else
        {
            addPieces({ "\n", "<knowledge>", prompts.knowledgeBody(s, st.ids, st.alias), "</knowledge>", "\n" },
                      Token_Kind::KNOW, false);
        }
    }
    else
    {
        throw std::invalid_argument("Seq_Builder::addStep: unknown tag " + st.tag);
    }
}

Seq buildSeq(const Sample &sample,
             const std::vector<Step> &steps,
             const ISealed_Read_Tokenizer &tok,
             const Prompts &prompts,
             View_Mode view,
             Branch branch,
             Frame_Mode frame,
             bool repeat_q,
             const Mask_Opts *opts_override)
{
    Mask_Opts opts{};
    if (opts_override != nullptr)
    {
        opts = *opts_override;
    }
    else
    {
        opts.branch = (view == View_Mode::LOCATE) ? Branch::NOMASK : branch;
    }
    Seq_Builder builder(tok, opts, view);
    builder.addPrompt(sample, prompts, view != View_Mode::MASKED, frame, repeat_q);
    for (size_t i = 0; i < steps.size(); ++i)
    {
        builder.addStep(sample, steps[i], static_cast<int32_t>(i), prompts);
    }
    return builder.seq();
}

Locate_Trie::Locate_Trie(const ISealed_Read_Tokenizer &tok, const std::vector<int32_t> &allowed_ids, size_t _k_max)
    : k_max(_k_max),
      root(std::make_unique<Node>()),
      cursor(root.get())
{
    if (allowed_ids.empty())
    {
        throw std::invalid_argument("Locate_Trie: allowed_ids is empty");
    }
    for (int32_t sid : allowed_ids)
    {
        const auto toks = tok.encode(formatLabel(sid));
        Node *node = root.get();
        node->sids.insert(sid);
        for (int32_t t : toks)
        {
            auto &ch = node->children[t];
            if (!ch)
            {
                ch = std::make_unique<Node>();
            }
            node = ch.get();
            node->sids.insert(sid);
            if (node->terminal.has_value())
            {
                throw std::runtime_error("Locate_Trie: label tokenizations are not prefix-free");
            }
        }
        if (!node->children.empty())
        {
            throw std::runtime_error("Locate_Trie: label tokenizations are not prefix-free");
        }
        node->terminal = sid;
    }
    sep_tokens = tok.encode(" ");
    end_tokens = tok.encode("</locate>");
    if (sep_tokens.empty() || end_tokens.empty() || sep_tokens[0] == end_tokens[0])
    {
        throw std::runtime_error("Locate_Trie: separator and end tag must start with distinct tokens");
    }
}

std::vector<int32_t> Locate_Trie::openChildren(const Node *node) const
{
    std::vector<int32_t> out;
    for (const auto &[tok_id, child] : node->children)
    {
        bool has_unchosen = false;
        for (int32_t sid : child->sids)
        {
            if (chosen_set.find(sid) == chosen_set.end())
            {
                has_unchosen = true;
                break;
            }
        }
        if (has_unchosen)
        {
            out.push_back(tok_id);
        }
    }
    return out;
}

void Locate_Trie::stepForced()
{
    const auto &seq = (phase == Phase::SEP) ? sep_tokens : end_tokens;
    ++forced_idx;
    if (forced_idx >= seq.size())
    {
        if (phase == Phase::SEP)
        {
            phase = Phase::ITEM;
            cursor = root.get();
        }
        else
        {
            phase = Phase::DONE;
        }
    }
}

std::vector<int32_t> Locate_Trie::allowedTokens() const
{
    if (phase == Phase::ITEM)
    {
        return openChildren(cursor);
    }
    if (phase == Phase::BETWEEN)
    {
        std::vector<int32_t> out = { end_tokens[0] };
        if (chosen_ids.size() < k_max && !openChildren(root.get()).empty())
        {
            out.push_back(sep_tokens[0]);
        }
        return out;
    }
    if (phase == Phase::SEP)
    {
        return { sep_tokens[forced_idx] };
    }
    if (phase == Phase::END)
    {
        return { end_tokens[forced_idx] };
    }
    return {};
}

void Locate_Trie::advance(int32_t token_id)
{
    const auto allowed = allowedTokens();
    if (std::find(allowed.begin(), allowed.end(), token_id) == allowed.end())
    {
        throw std::invalid_argument("Locate_Trie::advance: token not allowed in current phase");
    }
    if (phase == Phase::ITEM)
    {
        cursor = cursor->children.at(token_id).get();
        if (cursor->terminal.has_value())
        {
            chosen_ids.push_back(*cursor->terminal);
            chosen_set.insert(*cursor->terminal);
            phase = Phase::BETWEEN;
        }
    }
    else if (phase == Phase::BETWEEN)
    {
        if (token_id == end_tokens[0])
        {
            phase = Phase::END;
            forced_idx = 0;
        }
        else
        {
            phase = Phase::SEP;
            forced_idx = 0;
        }
        stepForced();
    }
    else if (phase == Phase::SEP || phase == Phase::END)
    {
        stepForced();
    }
}

void Locate_Trie::maskLogits(std::vector<float> &logits) const
{
    const auto allowed = allowedTokens();
    std::vector<uint8_t> keep(logits.size(), 0);
    for (int32_t tid : allowed)
    {
        if (tid >= 0 && static_cast<size_t>(tid) < logits.size())
        {
            keep[static_cast<size_t>(tid)] = 1;
        }
    }
    for (size_t i = 0; i < logits.size(); ++i)
    {
        if (keep[i] == 0)
        {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

Scripted_Policy::Scripted_Policy(float _noise, float _wrong, uint32_t seed)
    : noise(_noise),
      wrong(_wrong),
      rng(seed)
{
}

const Hop *Scripted_Policy::nextHop(const Episode &ep) const
{
    const auto p_vec = pastedIds(ep.steps);
    const std::unordered_set<int32_t> done(p_vec.begin(), p_vec.end());
    for (const auto &h : ep.sample.hops)
    {
        bool covered = true;
        for (int32_t sid : h.sids)
        {
            if (done.find(sid) == done.end())
            {
                covered = false;
                break;
            }
        }
        if (!covered)
        {
            return &h;
        }
    }
    return nullptr;
}

std::pair<std::string, bool> Scripted_Policy::generateMasked(Episode &ep, const std::string &tag, bool force_answer)
{
    const Hop *hop = nextHop(ep);
    if (tag == "reason")
    {
        const bool need = (hop != nullptr) && !force_answer && !ep.sample.goldIds().empty();
        std::string txt = need ? ("Need " + hop->bridge) : "Enough evidence";
        txt += ". alpha=";
        txt += need ? "query" : "answer";
        return { std::move(txt), false };
    }
    if (tag == "query")
    {
        std::string target = (hop != nullptr) ? hop->bridge : ep.sample.question;
        return { "step towards " + target, false };
    }
    if (tag == "recheck")
    {
        return { "The pasted text answers the query. beta=valid_to_think", false };
    }
    if (tag == "answer")
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < wrong)
        {
            return { "wrong answer", false };
        }
        std::string ans = ep.sample.answers.empty() ? std::string("") : ep.sample.answers.front();
        return { std::move(ans), false };
    }
    return { "", false };
}

std::vector<int32_t> Scripted_Policy::generateLocate(Episode &ep, const std::vector<int32_t> &allowed_ids)
{
    const Hop *hop = nextHop(ep);
    std::unordered_set<int32_t> al_set(allowed_ids.begin(), allowed_ids.end());
    std::vector<int32_t> ids;
    if (hop != nullptr)
    {
        for (int32_t sid : hop->sids)
        {
            if (al_set.find(sid) != al_set.end())
            {
                ids.push_back(sid);
                if (ids.size() == 2)
                {
                    break;
                }
            }
        }
    }
    if (ids.empty() && !allowed_ids.empty())
    {
        ids.push_back(allowed_ids.front());
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng) < noise)
    {
        const auto gold_vec = ep.sample.goldIds();
        const std::unordered_set<int32_t> gold_set(gold_vec.begin(), gold_vec.end());
        std::vector<int32_t> extra;
        for (int32_t sid : allowed_ids)
        {
            if (std::find(ids.begin(), ids.end(), sid) == ids.end() && gold_set.find(sid) == gold_set.end())
            {
                extra.push_back(sid);
            }
        }
        if (!extra.empty())
        {
            std::uniform_int_distribution<size_t> pick(0, extra.size() - 1);
            ids.push_back(extra[pick(rng)]);
        }
    }
    if (ids.size() > 3)
    {
        ids.resize(3);
    }
    return ids;
}

void Scripted_Policy::release(Episode &)
{
}

struct Pending_Last
{
    size_t step_idx = 0;
    size_t start_pos = 0;
    std::string content;
};

struct Ref_Policy::Impl
{
    Causal_LM &model;
    const ISealed_Read_Tokenizer &tok;
    Prompts prompts;
    Ref_Policy_Config cfg;
    std::unordered_set<int32_t> eos;

    Episode *active_ep = nullptr;
    Seq_Builder builder;
    size_t fed = 0;
    size_t synced = 0;
    std::optional<std::pair<size_t, size_t>> pending_loc;
    std::optional<Pending_Last> last;
    Tensor last_hidden;
    size_t last_token_offset = 0;
    std::mt19937 rng;

    Impl(Causal_LM &_m, const ISealed_Read_Tokenizer &_t, Prompts _p, Ref_Policy_Config _c)
        : model(_m),
          tok(_t),
          prompts(std::move(_p)),
          cfg(std::move(_c)),
          eos(_t.eosIds()),
          last_hidden(_m.getConfig().execution_target)
    {
    }
};

Ref_Policy::Ref_Policy(Causal_LM &model,
                       const ISealed_Read_Tokenizer &tok,
                       const Prompts &prompts,
                       const Ref_Policy_Config &cfg)
    : impl(std::make_unique<Impl>(model, tok, prompts, cfg))
{
    if (cfg.view != View_Mode::CHAIN && cfg.view != View_Mode::MASKED)
    {
        throw std::invalid_argument("Ref_Policy: view must be CHAIN or MASKED");
    }
}

Ref_Policy::~Ref_Policy() = default;

void Ref_Policy::ensureEpisodeState(Episode &ep)
{
    if (impl->active_ep == &ep)
    {
        return;
    }
    impl->active_ep = &ep;
    impl->model.setTrainingMode(false);
    impl->model.resetKVCaches();
    impl->builder = Seq_Builder(impl->tok, impl->cfg.opts, impl->cfg.view);
    impl->builder.addPrompt(ep.sample, impl->prompts, impl->cfg.view == View_Mode::CHAIN, impl->cfg.frame, impl->cfg.repeat_q);
    impl->fed = 0;
    impl->synced = 0;
    impl->pending_loc.reset();
    impl->last.reset();
    impl->rng.seed(ep.seed);

    const auto &kinds = impl->builder.seq().kind;
    size_t n_pre = impl->builder.size();
    for (size_t i = 0; i < kinds.size(); ++i)
    {
        if (kinds[i] >= Token_Kind::QP)
        {
            n_pre = i;
            break;
        }
    }
    feed(n_pre, true);
    feedAll();
}

void Ref_Policy::feed(size_t upto, bool causal)
{
    const size_t lo = impl->fed;
    const size_t hi = std::min(upto, impl->builder.size());
    if (hi <= lo)
    {
        return;
    }
    if (hi > impl->model.getConfig().max_seq_len)
    {
        throw std::runtime_error("Ref_Policy::feed: sequence length exceeds model max_seq_len");
    }
    const Execution_Target dev = impl->model.getConfig().execution_target;
    const Seq &seq = impl->builder.seq();
    const size_t q_len = hi - lo;

    std::vector<float> id_f(q_len);
    std::vector<float> pos_f(q_len);
    for (size_t i = 0; i < q_len; ++i)
    {
        id_f[i] = static_cast<float>(seq.ids[lo + i]);
        pos_f[i] = static_cast<float>(seq.pos[lo + i]);
    }
    Tensor ids_t(Shape{ 1, q_len }, std::move(id_f), dev);
    Tensor pos_t(Shape{ 1, q_len }, std::move(pos_f), dev);

    if (causal && lo == 0)
    {
        impl->last_hidden = impl->model.forwardHidden(ids_t, true, &pos_t, nullptr);
    }
    else
    {
        const auto keep = seq.collectActiveKvIndices(lo, hi);
        Tensor kv_idx_t(Shape{ keep.size() }, std::vector<float>(keep.begin(), keep.end()), dev);
        Tensor mask_t = seq.buildDenseMask(lo, hi, keep, dev);
        ops::llm::Attention_Config attn_cfg{};
        attn_cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
        attn_cfg.attn_mask = &mask_t;
        attn_cfg.kv_index = &kv_idx_t;
        impl->last_hidden = impl->model.forwardHidden(ids_t, true, &pos_t, &attn_cfg);
    }

    if (dev == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    impl->last_token_offset = q_len - 1;
    impl->fed = hi;
}

void Ref_Policy::feedAll()
{
    feed(impl->builder.size(), false);
}

void Ref_Policy::truncate(size_t n)
{
    impl->builder.truncate(n);
    impl->model.truncateKVCaches(n);
    if (impl->fed > n)
    {
        impl->fed = n;
    }
}

void Ref_Policy::sync(Episode &ep)
{
    ensureEpisodeState(ep);
    if (impl->last.has_value())
    {
        const Pending_Last prev = *impl->last;
        impl->last.reset();
        if (prev.step_idx < ep.steps.size() && ep.steps[prev.step_idx].text != prev.content)
        {
            truncate(prev.start_pos);
            impl->synced = prev.step_idx;
        }
        else if (prev.step_idx >= ep.steps.size())
        {
            truncate(prev.start_pos);
            impl->synced = ep.steps.size();
        }
    }

    for (size_t i = impl->synced; i < ep.steps.size(); ++i)
    {
        const Step &st = ep.steps[i];
        if (st.tag == "locate" && st.by == "model" && impl->pending_loc.has_value())
        {
            const auto [lo, hi] = *impl->pending_loc;
            impl->builder.step = static_cast<int32_t>(i);
            impl->builder.addLocateCopy(st, lo, hi);
            impl->pending_loc.reset();
        }
        else
        {
            impl->builder.addStep(ep.sample, st, static_cast<int32_t>(i), impl->prompts);
        }
    }
    impl->synced = ep.steps.size();
    feedAll();
}

int32_t Ref_Policy::sampleNext(const std::vector<int32_t> *allowed)
{
    const Execution_Target dev = impl->model.getConfig().execution_target;
    Tensor idx_t(Shape{ 1 }, { static_cast<float>(impl->last_token_offset) }, dev);
    Tensor logits_t = impl->model.forwardTargetLogits(impl->last_hidden, idx_t);
    if (dev == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    std::vector<float> logits = logits_t.getData();
    const size_t V = std::min(logits.size(), impl->tok.vocabSize());
    logits.resize(V);

    if (allowed != nullptr)
    {
        std::vector<uint8_t> keep(V, 0);
        for (int32_t id : *allowed)
        {
            if (id >= 0 && static_cast<size_t>(id) < V)
            {
                keep[static_cast<size_t>(id)] = 1;
            }
        }
        for (size_t i = 0; i < V; ++i)
        {
            if (keep[i] == 0)
            {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    if (impl->cfg.temperature <= 0.0f)
    {
        size_t best_i = 0;
        float best_v = logits[0];
        for (size_t i = 1; i < V; ++i)
        {
            if (logits[i] > best_v)
            {
                best_v = logits[i];
                best_i = i;
            }
        }
        return static_cast<int32_t>(best_i);
    }

    float max_l = -std::numeric_limits<float>::infinity();
    for (float v : logits)
    {
        if (std::isfinite(v))
        {
            max_l = std::max(max_l, v);
        }
    }
    std::vector<std::pair<float, size_t>> probs;
    probs.reserve(V);
    double sum = 0.0;
    for (size_t i = 0; i < V; ++i)
    {
        if (!std::isfinite(logits[i]))
        {
            continue;
        }
        const float p = std::exp((logits[i] - max_l) / impl->cfg.temperature);
        probs.emplace_back(p, i);
        sum += p;
    }
    if (probs.empty() || sum <= 0.0)
    {
        return (allowed && !allowed->empty()) ? allowed->front() : 0;
    }
    for (auto &[p, _] : probs)
    {
        p = static_cast<float>(p / sum);
    }
    if (impl->cfg.top_p < 1.0f && probs.size() > 1)
    {
        std::sort(probs.begin(), probs.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        double cum = 0.0;
        size_t keep_n = probs.size();
        for (size_t i = 0; i < probs.size(); ++i)
        {
            if (cum > impl->cfg.top_p && i > 0)
            {
                keep_n = i;
                break;
            }
            cum += probs[i].first;
        }
        probs.resize(keep_n);
    }
    double renorm = 0.0;
    for (const auto &[p, _] : probs)
    {
        renorm += p;
    }
    std::uniform_real_distribution<double> u01(0.0, renorm);
    double r = u01(impl->rng);
    for (const auto &[p, idx] : probs)
    {
        r -= p;
        if (r <= 0.0)
        {
            return static_cast<int32_t>(idx);
        }
    }
    return static_cast<int32_t>(probs.back().second);
}

void Ref_Policy::appendSingleToken(int32_t token_id, Token_Kind kind)
{
    impl->builder.addIds({ token_id }, kind, true);
    feedAll();
}

std::pair<std::string, bool> Ref_Policy::generateMasked(Episode &ep, const std::string &tag, bool force_answer)
{
    sync(ep);
    impl->builder.step = static_cast<int32_t>(ep.steps.size());
    if (tag == "query")
    {
        impl->builder.cur_qref = static_cast<int32_t>(ep.steps.size());
    }
    const size_t open_at = impl->builder.size();
    impl->builder.addText("<" + tag + ">", Token_Kind::GEN, true);
    feedAll();
    const size_t start = impl->builder.size();

    const std::string close_tag = "</" + tag + ">";
    std::vector<int32_t> gen;
    bool hit = true;
    size_t cap = 64;
    if (auto it = impl->cfg.proto.caps.find(tag); it != impl->cfg.proto.caps.end())
    {
        cap = it->second + 16;
    }
    const size_t max_allowed = impl->model.getConfig().max_seq_len;
    const size_t tail_reserve = close_tag.size() + impl->prompts.eos.size() + 4;
    for (size_t step_i = 0; step_i < cap && impl->builder.size() + tail_reserve < max_allowed; ++step_i)
    {
        const int32_t t = sampleNext(nullptr);
        gen.push_back(t);
        appendSingleToken(t, Token_Kind::GEN);
        const size_t tail_from = (gen.size() > 24) ? (gen.size() - 24) : 0;
        std::vector<int32_t> tail(gen.begin() + static_cast<std::ptrdiff_t>(tail_from), gen.end());
        if (impl->tok.decode(tail).find(close_tag) != std::string::npos)
        {
            hit = false;
            break;
        }
        if (impl->eos.find(t) != impl->eos.end())
        {
            break;
        }
    }

    std::string text = impl->tok.decode(gen);
    const size_t close_pos = text.find(close_tag);
    std::string content = (close_pos != std::string::npos) ? text.substr(0, close_pos) : text;
    for (const char *sp : { "<|im_end|>", "<|endoftext|>" })
    {
        content = replaceCaseInsensitive(content, sp, "");
    }
    content = trimCopy(content);
    if (force_answer && tag == "reason")
    {
        content = forceAlphaAnswer(content);
    }

    truncate(start);
    const auto tp = tagPieces(tag, content, impl->prompts);
    std::vector<std::string> tail_pieces(tp.begin() + 1, tp.end());
    impl->builder.addPieces(tail_pieces, Token_Kind::GEN, true);
    feedAll();
    impl->synced = ep.steps.size() + 1;
    impl->last = Pending_Last{ ep.steps.size(), open_at, content };
    return { std::move(content), hit };
}

std::vector<int32_t> Ref_Policy::generateLocate(Episode &ep, const std::vector<int32_t> &allowed_ids)
{
    if (impl->cfg.view != View_Mode::CHAIN)
    {
        throw std::logic_error("Ref_Policy::generateLocate requires View_Mode::CHAIN");
    }
    sync(ep);
    impl->builder.step = static_cast<int32_t>(ep.steps.size());
    impl->builder.round += 1;
    auto [start, end_open] = impl->builder.addText("<locate>", Token_Kind::LOC, true);
    (void)end_open;
    feedAll();

    Locate_Trie trie(impl->tok, allowed_ids, impl->cfg.proto.k);
    while (!trie.isDone())
    {
        const auto allowed = trie.allowedTokens();
        const int32_t t = sampleNext(&allowed);
        trie.advance(t);
        appendSingleToken(t, Token_Kind::LOC);
    }
    impl->pending_loc = std::make_pair(start, impl->builder.size());
    impl->synced = ep.steps.size();
    return trie.chosenIds();
}

void Ref_Policy::release(Episode &ep)
{
    if (impl->active_ep == &ep)
    {
        impl->active_ep = nullptr;
        impl->last.reset();
        impl->pending_loc.reset();
    }
}

Seq Ref_Policy::forceTrace(Episode &ep, const std::vector<Step> &steps)
{
    ep.steps = steps;
    sync(ep);
    return impl->builder.seq();
}

Proto_Config Driver_Config::effectiveProto() const
{
    if (locator == "oracle")
    {
        Proto_Config p = proto;
        p.r_max = 1;
        return p;
    }
    return proto;
}

std::vector<Episode> runEpisodes(IPolicy &policy,
                                 const std::vector<Sample> &samples,
                                 size_t n_per_sample,
                                 const Driver_Config &cfg,
                                 uint32_t seed)
{
    const Proto_Config proto = cfg.effectiveProto();
    std::vector<Episode> eps;
    eps.reserve(samples.size() * n_per_sample);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        for (size_t j = 0; j < n_per_sample; ++j)
        {
            Episode ep;
            ep.sample = samples[i];
            ep.seed = static_cast<uint32_t>(static_cast<uint64_t>(seed) * 1000003ULL + i * 101ULL + j);
            ep.sample_idx = static_cast<int32_t>(j);
            eps.push_back(std::move(ep));
        }
    }

    for (Episode &ep : eps)
    {
        while (!ep.done)
        {
            if (ep.steps.size() >= cfg.max_steps)
            {
                ep.error = "too many steps";
                ep.done = true;
                break;
            }
            std::string err;
            auto tag_opt = nextTag(ep.steps, proto, &err);
            if (!err.empty())
            {
                ep.error = err;
                ep.done = true;
                break;
            }
            if (!tag_opt.has_value())
            {
                ep.done = true;
                break;
            }
            const std::string tag = *tag_opt;
            if (tag == "locate")
            {
                const auto p_vec = pastedIds(ep.steps);
                const std::unordered_set<int32_t> done(p_vec.begin(), p_vec.end());
                std::vector<int32_t> allowed;
                for (const auto &seg : ep.sample.segments)
                {
                    if (done.find(seg.sid) == done.end())
                    {
                        allowed.push_back(seg.sid);
                    }
                }
                if (allowed.empty())
                {
                    ep.error = "nothing left to locate";
                    ep.done = true;
                    break;
                }
                std::vector<int32_t> ids;
                std::string by = "model";
                if (cfg.locator == "oracle")
                {
                    by = "engine";
                    for (int32_t gsid : ep.sample.goldIds())
                    {
                        if (done.find(gsid) == done.end())
                        {
                            ids.push_back(gsid);
                        }
                    }
                    if (ids.empty())
                    {
                        ep.error = "oracle locate without gold";
                        ep.done = true;
                        break;
                    }
                }
                else
                {
                    ids = policy.generateLocate(ep, allowed);
                    std::vector<int32_t> uniq;
                    std::unordered_set<int32_t> seen;
                    for (int32_t sid : ids)
                    {
                        if (seen.insert(sid).second)
                        {
                            uniq.push_back(sid);
                        }
                    }
                    ep.stats.dropped_ids += ids.size() - uniq.size();
                    ids = std::move(uniq);
                    std::string chk = checkLocate(ids, ep.sample.ids(), done, proto.k);
                    if (!chk.empty())
                    {
                        ep.error = chk;
                        ep.done = true;
                        break;
                    }
                }
                if (ids.empty())
                {
                    ep.error = "empty locate";
                    ep.done = true;
                    break;
                }
                const auto order = orderInContext(ep.sample.order(), ids);
                ep.steps.push_back(Step{ "locate", "", std::move(ids), by, {} });
                ep.steps.push_back(Step{ "knowledge", "", std::move(order), "engine", {} });
            }
            else if (tag == "knowledge")
            {
                ep.error = "knowledge written unexpectedly";
                ep.done = true;
                break;
            }
            else
            {
                const bool force = (tag == "reason" && mustAnswer(ep.steps, proto));
                auto [content, hit] = policy.generateMasked(ep, tag, force);
                if (hit)
                {
                    ep.stats.cap_hits++;
                    if (!cfg.lenient)
                    {
                        ep.error = "<" + tag + "> hit its cap";
                        ep.done = true;
                        break;
                    }
                }
                std::string fixed = fixFormatDriver(ep, tag, std::move(content), force, cfg);
                if (ep.done)
                {
                    break;
                }
                ep.steps.push_back(Step{ tag, std::move(fixed), {}, "model", {} });
            }
        }
        policy.release(ep);
    }
    return eps;
}

std::string normalizeAnswer(const std::string &s)
{
    std::string no_punct;
    no_punct.reserve(s.size());
    for (unsigned char ch : s)
    {
        if (!std::ispunct(ch))
        {
            no_punct.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    std::istringstream iss(no_punct);
    std::string word;
    std::string out;
    while (iss >> word)
    {
        if (word == "a" || word == "an" || word == "the")
        {
            continue;
        }
        if (!out.empty())
        {
            out.push_back(' ');
        }
        out += word;
    }
    return out;
}

float exactMatch(const std::string &pred, const std::vector<std::string> &answers)
{
    const std::string norm_pred = normalizeAnswer(pred);
    for (const auto &g : answers)
    {
        if (norm_pred == normalizeAnswer(g))
        {
            return 1.0f;
        }
    }
    return 0.0f;
}

float bestF1(const std::string &pred, const std::vector<std::string> &answers)
{
    const auto p_toks = splitNormalizedTokens(pred);
    float best = 0.0f;
    for (const auto &g : answers)
    {
        const auto g_toks = splitNormalizedTokens(g);
        if (p_toks.empty() || g_toks.empty())
        {
            best = std::max(best, (p_toks == g_toks) ? 1.0f : 0.0f);
            continue;
        }
        std::unordered_map<std::string, int> pc;
        std::unordered_map<std::string, int> gc;
        for (const auto &t : p_toks) pc[t]++;
        for (const auto &t : g_toks) gc[t]++;
        int same = 0;
        for (const auto &[k, cnt] : pc)
        {
            auto it = gc.find(k);
            if (it != gc.end())
            {
                same += std::min(cnt, it->second);
            }
        }
        if (same == 0)
        {
            continue;
        }
        const float prec = static_cast<float>(same) / static_cast<float>(p_toks.size());
        const float rec = static_cast<float>(same) / static_cast<float>(g_toks.size());
        best = std::max(best, 2.0f * prec * rec / (prec + rec));
    }
    return best;
}

std::pair<float, float> precisionRecall(const std::vector<int32_t> &gold, const std::vector<int32_t> &got)
{
    if (gold.empty() || got.empty())
    {
        return { 0.0f, 0.0f };
    }
    std::unordered_set<int32_t> g(gold.begin(), gold.end());
    std::unordered_set<int32_t> s(got.begin(), got.end());
    size_t tp = 0;
    for (int32_t id : s)
    {
        if (g.find(id) != g.end())
        {
            ++tp;
        }
    }
    return {
        static_cast<float>(tp) / static_cast<float>(s.size()),
        static_cast<float>(tp) / static_cast<float>(g.size())
    };
}

float f2Score(const std::vector<int32_t> &gold, const std::vector<int32_t> &got)
{
    const auto [p, r] = precisionRecall(gold, got);
    if (p == 0.0f && r == 0.0f)
    {
        return 0.0f;
    }
    return 5.0f * p * r / (4.0f * p + r);
}

Score scoreEpisode(const std::vector<Step> &steps,
                   const Sample &sample,
                   const Reward_Config &cfg,
                   bool valid,
                   float lam_override,
                   bool gate)
{
    const float lam = (lam_override >= 0.0f) ? lam_override : cfg.lam;
    const auto ids = locatedIds(steps);
    const auto gold = sample.goldIds();
    const size_t n_loc = locateRounds(steps);
    const float ra = valid ? exactMatch(answerOf(steps), sample.answers) : 0.0f;
    const auto [p, r] = precisionRecall(gold, ids);
    const float f2 = f2Score(gold, ids);
    const float acc = static_cast<float>(ids.size()) / static_cast<float>(std::max<size_t>(1, cfg.k * cfg.r_max));
    std::optional<float> r_mu;
    if (valid)
    {
        if (gold.empty())
        {
            r_mu = ra - cfg.mu0 * static_cast<float>(std::min<size_t>(n_loc, 3));
        }
        else if (!gate)
        {
            r_mu = ra - cfg.mu * acc;
        }
        else
        {
            r_mu = ra * (lam + (1.0f - lam) * f2) - cfg.mu * acc;
        }
    }
    return Score{ valid, ra, f2, p, r, ids.size(), n_loc, acc, r_mu };
}

Rest_Em_Result selectRestEm(const std::vector<Episode> &episodes,
                            const Proto_Config &proto,
                            const Reward_Config &rcfg,
                            Rest_Em_Variant variant,
                            size_t keep,
                            size_t cap,
                            float recall_min,
                            uint32_t seed,
                            const ISealed_Read_Tokenizer *tok)
{
    std::unordered_map<std::string, std::vector<const Episode *>> by_q;
    std::vector<std::string> q_order;
    for (const auto &ep : episodes)
    {
        if (by_q.find(ep.sample.qid) == by_q.end())
        {
            q_order.push_back(ep.sample.qid);
        }
        by_q[ep.sample.qid].push_back(&ep);
    }

    auto passes = [&](const Episode &ep) {
        if (!ep.error.empty())
        {
            return false;
        }
        if (!validateSteps(ep.steps, ep.sample, proto, tok).empty())
        {
            return false;
        }
        const Score sc = scoreEpisode(ep.steps, ep.sample, rcfg, true);
        if (sc.r_ans < 1.0f)
        {
            return false;
        }
        if (variant != Rest_Em_Variant::NOGATE && !ep.sample.goldIds().empty() && sc.recall < recall_min)
        {
            return false;
        }
        return true;
    };

    auto rankScore = [&](const Episode &ep) {
        if (variant == Rest_Em_Variant::NOGATE)
        {
            return scoreEpisode(ep.steps, ep.sample, rcfg, true, -1.0f, false);
        }
        const float lam_v = (variant == Rest_Em_Variant::HARD) ? 0.0f : rcfg.lam;
        return scoreEpisode(ep.steps, ep.sample, rcfg, true, lam_v, true);
    };

    std::vector<std::vector<const Episode *>> learnable;
    std::vector<std::vector<const Episode *>> easy;

    for (const auto &qid : q_order)
    {
        const auto &group = by_q[qid];
        std::vector<const Episode *> ok;
        for (const Episode *e : group)
        {
            if (passes(*e))
            {
                ok.push_back(e);
            }
        }
        if (ok.empty())
        {
            continue;
        }
        std::sort(ok.begin(), ok.end(), [&](const Episode *a, const Episode *b) {
            const Score sa = rankScore(*a);
            const Score sb = rankScore(*b);
            const float ra = sa.r_mu.value_or(0.0f);
            const float rb = sb.r_mu.value_or(0.0f);
            if (ra != rb)
            {
                return ra > rb;
            }
            return sa.n_id < sb.n_id;
        });
        if (ok.size() < group.size())
        {
            if (ok.size() > keep)
            {
                ok.resize(keep);
            }
            learnable.push_back(std::move(ok));
        }
        else
        {
            easy.push_back({ ok.front() });
        }
    }

    std::mt19937 rng(seed);
    std::shuffle(learnable.begin(), learnable.end(), rng);
    std::shuffle(easy.begin(), easy.end(), rng);

    Rest_Em_Result res;
    res.questions = q_order.size();
    res.learnable = learnable.size();
    res.all_pass = easy.size();

    auto appendGroups = [&](const std::vector<std::vector<const Episode *>> &groups) {
        for (const auto &g : groups)
        {
            if (res.selected.size() >= cap)
            {
                break;
            }
            for (const Episode *ep : g)
            {
                if (res.selected.size() >= cap)
                {
                    break;
                }
                res.selected.push_back(*ep);
            }
        }
    };
    appendGroups(learnable);
    appendGroups(easy);
    res.kept = res.selected.size();
    return res;
}

Invariance_Report probeInvariance(Causal_LM &model,
                                  const Sample &sample,
                                  const std::vector<Step> &steps,
                                  const ISealed_Read_Tokenizer &tok,
                                  const Prompts &prompts,
                                  const Mask_Opts &opts,
                                  uint32_t seed)
{
    const Seq seq = buildSeq(sample, steps, tok, prompts, View_Mode::CHAIN, opts.branch, Frame_Mode::FULL, true, &opts);
    const auto rows = seq.maskedTargetRows();

    const auto pasted_vec = pastedIds(steps);
    const std::unordered_set<int32_t> keep_sids(pasted_vec.begin(), pasted_vec.end());
    std::vector<int32_t> pool;
    for (size_t i = 0; i < seq.size(); ++i)
    {
        if (seq.kind[i] == Token_Kind::C)
        {
            pool.push_back(seq.ids[i]);
        }
    }
    Seq alt = seq;
    if (!pool.empty())
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        for (size_t i = 0; i < alt.size(); ++i)
        {
            if (alt.kind[i] == Token_Kind::C && keep_sids.find(alt.seg[i]) == keep_sids.end())
            {
                int32_t replacement = pool[pick(rng)];
                if (pool.size() > 1 && replacement == alt.ids[i])
                {
                    replacement = pool[(pick(rng) + 1) % pool.size()];
                }
                alt.ids[i] = replacement;
            }
        }
    }

    Invariance_Report rep;
    rep.qid = sample.qid;
    rep.branch = opts.branch;
    rep.positions = rows.size();
    if (rows.empty())
    {
        return rep;
    }

    const size_t V = model.getConfig().vocab_size;
    const auto lp_orig = computeLogSoftmaxRows(model, seq, rows);
    const auto lp_alt = computeLogSoftmaxRows(model, alt, rows);

    double kl_sum = 0.0;
    double kl_max = 0.0;
    double max_abs_diff = 0.0;
    for (size_t r = 0; r < rows.size(); ++r)
    {
        double kl_r = 0.0;
        for (size_t v = 0; v < V; ++v)
        {
            const double p_log = static_cast<double>(lp_orig[r * V + v]);
            const double q_log = static_cast<double>(lp_alt[r * V + v]);
            max_abs_diff = std::max(max_abs_diff, std::abs(p_log - q_log));
            kl_r += std::exp(p_log) * (p_log - q_log);
        }
        kl_r = std::max(0.0, kl_r);
        kl_sum += kl_r;
        kl_max = std::max(kl_max, kl_r);
    }

    rep.kl_mean = static_cast<float>(kl_sum / static_cast<double>(rows.size()));
    rep.kl_max = static_cast<float>(kl_max);
    rep.max_abs_logp = static_cast<float>(max_abs_diff);
    return rep;
}

std::optional<Dependence_Report> probeDependence(Causal_LM &model,
                                                 const Sample &sample,
                                                 const std::vector<Step> &steps,
                                                 const ISealed_Read_Tokenizer &tok,
                                                 const Prompts &prompts,
                                                 const std::string &new_value,
                                                 const Mask_Opts &opts)
{
    if (sample.answers.empty())
    {
        return std::nullopt;
    }
    const std::string &ans = sample.answers.front();
    const auto pasted_vec = pastedIds(steps);
    const std::unordered_set<int32_t> pasted_set(pasted_vec.begin(), pasted_vec.end());
    const std::string norm_ans = normalizeAnswer(ans);
    if (norm_ans.empty())
    {
        return std::nullopt;
    }

    Sample changed = sample;
    bool hit_any = false;
    for (auto &seg : changed.segments)
    {
        if (pasted_set.find(seg.sid) != pasted_set.end() &&
            normalizeAnswer(seg.text).find(norm_ans) != std::string::npos)
        {
            seg.text = replaceCaseInsensitive(seg.text, ans, new_value);
            hit_any = true;
        }
    }
    if (!hit_any)
    {
        return std::nullopt;
    }

    std::optional<size_t> ans_step_idx;
    for (size_t i = 0; i < steps.size(); ++i)
    {
        if (steps[i].tag == "answer")
        {
            ans_step_idx = i;
        }
    }
    if (!ans_step_idx.has_value())
    {
        return std::nullopt;
    }

    const Seq seq1 = buildSeq(sample, steps, tok, prompts, View_Mode::CHAIN, opts.branch, Frame_Mode::FULL, true, &opts);
    const Seq seq2 = buildSeq(changed, steps, tok, prompts, View_Mode::CHAIN, opts.branch, Frame_Mode::FULL, true, &opts);
    const auto rows1 = extractRowIndicesFromTarget(seq1, steps, tok, *ans_step_idx);
    const auto rows2 = extractRowIndicesFromTarget(seq2, steps, tok, *ans_step_idx);
    if (rows1.empty() || rows1.size() != rows2.size())
    {
        return std::nullopt;
    }

    const size_t V = model.getConfig().vocab_size;
    const auto lp1 = computeLogSoftmaxRows(model, seq1, rows1);
    const auto lp2 = computeLogSoftmaxRows(model, seq2, rows2);

    double sum1 = 0.0;
    double sum2 = 0.0;
    double max_abs = 0.0;
    for (size_t r = 0; r < rows1.size(); ++r)
    {
        const size_t tok_id = static_cast<size_t>(seq1.ids[static_cast<size_t>(rows1[r] + 1)]);
        const double l1 = static_cast<double>(lp1[r * V + tok_id]);
        const double l2 = static_cast<double>(lp2[r * V + tok_id]);
        sum1 += l1;
        sum2 += l2;
        max_abs = std::max(max_abs, std::abs(l1 - l2));
    }

    Dependence_Report rep;
    rep.qid = sample.qid;
    rep.branch = opts.branch;
    rep.answer_tokens = rows1.size();
    rep.logp_orig = static_cast<float>(sum1);
    rep.logp_changed = static_cast<float>(sum2);
    rep.drop = static_cast<float>(sum1 - sum2);
    rep.max_abs_delta_logp = static_cast<float>(max_abs);
    return rep;
}

Sft_Metrics trainSft(Causal_LM &model,
                     const std::vector<Episode> &traces,
                     const ISealed_Read_Tokenizer &tok,
                     const Prompts &prompts,
                     const Sft_Config &cfg)
{
    if (cfg.enable_lora_if_inactive && !model.hasActiveLoRA())
    {
        model.enableLoRA(cfg.lora_rank,
                         cfg.lora_alpha,
                         0.0f,
                         { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" },
                         true);
    }

    const size_t max_seq = model.getConfig().max_seq_len;
    std::vector<Seq> seqs;
    seqs.reserve(traces.size());
    for (const auto &ep : traces)
    {
        Seq seq = buildSeq(ep.sample, ep.steps, tok, prompts, View_Mode::CHAIN, cfg.mask_opts.branch, Frame_Mode::FULL, true, &cfg.mask_opts);
        if (seq.size() >= 2 && seq.size() <= max_seq && !seq.targetIndices().empty())
        {
            seqs.push_back(std::move(seq));
        }
    }
    if (seqs.empty())
    {
        throw std::invalid_argument("trainSft: no eligible traces within max_seq_len");
    }

    std::vector<size_t> order(seqs.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(cfg.seed);
    std::shuffle(order.begin(), order.end(), rng);

    Adam_Optimizer optimizer(cfg.learning_rate, 0.9f, 0.999f, 1e-8f, cfg.max_grad_norm, cfg.weight_decay);
    const Execution_Target dev = model.getConfig().execution_target;

    Sft_Metrics metrics{};
    metrics.eligible_traces = seqs.size();
    metrics.min_loss = std::numeric_limits<float>::infinity();

    const size_t batch = std::max<size_t>(1, cfg.batch_traces);
    const size_t steps = std::max<size_t>(1, cfg.max_steps);
    size_t cursor = 0;

    model.setTrainingMode(true);
    {
        struct Async_Loss_Guard
        {
            bool active = false;
            bool prev_async = false;
            bool prev_defer = false;
            explicit Async_Loss_Guard(bool enable)
                : active(enable)
            {
                if (active)
                {
                    auto &eng = Execution_Engine::getInstance();
                    prev_async = eng.isAsyncLossEnabled();
                    prev_defer = eng.isDeferForwardLossActive();
                    eng.setAsyncLossEnabled(false);
                    eng.setDeferForwardLossActive(false);
                }
            }
            ~Async_Loss_Guard()
            {
                if (active)
                {
                    auto &eng = Execution_Engine::getInstance();
                    eng.setAsyncLossEnabled(prev_async);
                    eng.setDeferForwardLossActive(prev_defer);
                }
            }
        } async_guard(dev == Execution_Target::VULKAN_GPU);

        for (size_t step = 0; step < steps; ++step)
        {
            const double progress = (steps > 1) ? static_cast<double>(step) / static_cast<double>(steps - 1) : 1.0;
            const double cos_mul = 0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
            const float lr = cfg.min_learning_rate + static_cast<float>(cos_mul * (cfg.learning_rate - cfg.min_learning_rate));
            optimizer.setLearningRate(lr);

            model.resetGradients();
            model.setAccumulated(true);

            double step_loss_sum = 0.0;
            const float inv_batch = 1.0f / static_cast<float>(batch);
            for (size_t b = 0; b < batch; ++b)
            {
                if (cursor >= order.size())
                {
                    std::shuffle(order.begin(), order.end(), rng);
                    cursor = 0;
                }
                const Seq &seq = seqs[order[cursor++]];
                const size_t T = seq.size();
                const auto tgts = seq.targetIndices();

                std::vector<float> id_f(T);
                std::vector<float> pos_f(T);
                for (size_t i = 0; i < T; ++i)
                {
                    id_f[i] = static_cast<float>(seq.ids[i]);
                    pos_f[i] = static_cast<float>(seq.pos[i]);
                }
                std::vector<float> tgt_idx_f;
                std::vector<float> tgt_lbl_f;
                tgt_idx_f.reserve(tgts.size());
                tgt_lbl_f.reserve(tgts.size());
                for (int32_t t : tgts)
                {
                    if (t > 0)
                    {
                        tgt_idx_f.push_back(static_cast<float>(t - 1));
                        tgt_lbl_f.push_back(static_cast<float>(seq.ids[static_cast<size_t>(t)]));
                    }
                }

                Tensor in_t(Shape{ 1, T }, std::move(id_f), dev);
                Tensor pos_t(Shape{ 1, T }, std::move(pos_f), dev);
                Tensor idx_t(Shape{ tgt_idx_f.size() }, std::move(tgt_idx_f), dev);
                Tensor lbl_t(Shape{ tgt_lbl_f.size() }, std::move(tgt_lbl_f), dev);

                std::vector<int32_t> all_k(T);
                std::iota(all_k.begin(), all_k.end(), 0);
                Tensor mask_t = seq.buildDenseMask(0, T, all_k, dev);
                ops::llm::Attention_Config attn_cfg{};
                attn_cfg.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
                attn_cfg.attn_mask = &mask_t;

                const float trace_loss = model.computeTargetLossAndBackward(in_t, idx_t, lbl_t, &pos_t, &attn_cfg, inv_batch);
                step_loss_sum += static_cast<double>(trace_loss);
                metrics.traces_processed++;
            }

            model.stepOptimizer(optimizer, cfg.max_grad_norm);
            const float mean_loss = static_cast<float>(step_loss_sum / static_cast<double>(batch));
            if (step == 0)
            {
                metrics.initial_loss = mean_loss;
            }
            metrics.final_loss = mean_loss;
            metrics.min_loss = std::min(metrics.min_loss, mean_loss);
            metrics.steps_done++;
        }
    }
    model.setAccumulated(false);
    model.setTrainingMode(false);
    return metrics;
}

Aggregate_Metrics aggregateEpisodes(const std::vector<Episode> &episodes,
                                    const Proto_Config &proto,
                                    const Reward_Config &rcfg,
                                    const ISealed_Read_Tokenizer *tok)
{
    Aggregate_Metrics agg{};
    agg.total = episodes.size();
    if (episodes.empty())
    {
        return agg;
    }
    double sum_em = 0.0;
    double sum_f1 = 0.0;
    double sum_p = 0.0;
    double sum_r = 0.0;
    double sum_f2 = 0.0;
    double sum_acc = 0.0;
    double sum_rmu = 0.0;
    size_t gold_count = 0;

    for (const auto &ep : episodes)
    {
        const bool ok = ep.error.empty() && validateSteps(ep.steps, ep.sample, proto, tok).empty();
        if (ok)
        {
            agg.valid++;
        }
        const Score sc = scoreEpisode(ep.steps, ep.sample, rcfg, ok);
        const std::string pred = ok ? answerOf(ep.steps) : std::string();
        sum_em += ok ? static_cast<double>(sc.r_ans) : 0.0;
        sum_f1 += ok ? static_cast<double>(bestF1(pred, ep.sample.answers)) : 0.0;
        sum_acc += static_cast<double>(sc.acc);
        sum_rmu += static_cast<double>(sc.r_mu.value_or(-1.0f));
        if (!ep.sample.goldIds().empty())
        {
            sum_p += static_cast<double>(sc.precision);
            sum_r += static_cast<double>(sc.recall);
            sum_f2 += static_cast<double>(sc.f2);
            ++gold_count;
        }
    }

    const double n = static_cast<double>(agg.total);
    agg.em = static_cast<float>(sum_em / n);
    agg.f1 = static_cast<float>(sum_f1 / n);
    agg.acc = static_cast<float>(sum_acc / n);
    agg.r_mu = static_cast<float>(sum_rmu / n);
    if (gold_count > 0)
    {
        const double ng = static_cast<double>(gold_count);
        agg.precision = static_cast<float>(sum_p / ng);
        agg.recall = static_cast<float>(sum_r / ng);
        agg.f2 = static_cast<float>(sum_f2 / ng);
    }
    return agg;
}

}

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llm/causal_lm.h"
#include "llm/llm_configs.h"
#include "tokenizer/bpe_tokenizer.h"

namespace sealedread
{

class ISealed_Read_Tokenizer
{
public:
    virtual ~ISealed_Read_Tokenizer() = default;
    virtual std::string name() const = 0;
    virtual size_t vocabSize() const noexcept = 0;
    virtual std::string eosText() const = 0;
    virtual std::vector<int32_t> encode(const std::string &text) const = 0;
    virtual std::string decode(const std::vector<int32_t> &ids) const = 0;
    virtual std::unordered_set<int32_t> eosIds() const = 0;

    std::vector<int32_t> encodePieces(const std::vector<std::string> &pieces) const;
};

class Byte_Tokenizer final : public ISealed_Read_Tokenizer
{
public:
    static constexpr int32_t IM_START_ID = 256;
    static constexpr int32_t IM_END_ID = 257;
    static constexpr int32_t ENDOFTEXT_ID = 258;
    static constexpr size_t VOCAB_COUNT = 259;

    std::string name() const override { return "bytes"; }
    size_t vocabSize() const noexcept override { return VOCAB_COUNT; }
    std::string eosText() const override { return "<|im_end|>"; }
    std::vector<int32_t> encode(const std::string &text) const override;
    std::string decode(const std::vector<int32_t> &ids) const override;
    std::unordered_set<int32_t> eosIds() const override;
};

class Bpe_Sealed_Read_Tokenizer final : public ISealed_Read_Tokenizer
{
private:
    const Bpe_Tokenizer *tok = nullptr;

public:
    explicit Bpe_Sealed_Read_Tokenizer(const Bpe_Tokenizer &_tok);

    std::string name() const override { return "bpe"; }
    size_t vocabSize() const noexcept override;
    std::string eosText() const override { return "<|im_end|>"; }
    std::vector<int32_t> encode(const std::string &text) const override;
    std::string decode(const std::vector<int32_t> &ids) const override;
    std::unordered_set<int32_t> eosIds() const override;
};

struct Segment
{
    int32_t sid = 0;
    std::string title;
    std::string section = "Introduction";
    std::string text;
    std::string doc;
    bool gold = false;
};

struct Hop
{
    std::vector<int32_t> sids;
    std::string bridge;
};

struct Sample
{
    std::string qid;
    std::string dataset = "synthetic";
    std::string qtype = "multihop";
    std::string question;
    std::vector<std::string> answers;
    std::string answer_kind = "open";
    std::vector<Segment> segments;
    std::vector<Hop> hops;
    std::string cluster;
    size_t length = 0;

    std::vector<int32_t> goldIds() const;
    std::unordered_map<int32_t, size_t> order() const;
    std::unordered_set<int32_t> ids() const;
    const Segment *findBySid(int32_t sid) const noexcept;
    const Segment &bySid(int32_t sid) const;
};

struct Step
{
    std::string tag;
    std::string text;
    std::vector<int32_t> ids;
    std::string by = "model";
    std::unordered_map<int32_t, int32_t> alias;
};

struct Episode_Stats
{
    size_t format_fixes = 0;
    size_t cap_hits = 0;
    size_t dropped_ids = 0;
};

struct Episode
{
    Sample sample;
    std::vector<Step> steps;
    bool done = false;
    std::string error;
    uint32_t seed = 0;
    int32_t sample_idx = 0;
    Episode_Stats stats{};
};

struct Synthetic_Dataset_Config
{
    size_t n_samples = 40;
    size_t n_distractors = 6;
    float nocontext_ratio = 0.2f;
    uint32_t seed = 0;
    int32_t id_offset = 0;
};

std::vector<Sample> generateSyntheticDataset(const Synthetic_Dataset_Config &cfg);
std::vector<Sample> loadSamplesJsonl(const std::string &path, size_t limit = 0);
size_t saveSamplesJsonl(const std::string &path, const std::vector<Sample> &samples);

struct Proto_Config
{
    size_t k = 3;
    size_t r_max = 8;
    std::unordered_map<std::string, size_t> caps = {
        { "reason", 320 },
        { "query", 48 },
        { "recheck", 128 },
        { "answer", 48 }
    };
    size_t locate_max_tokens = 48;
};

std::string formatLabel(int32_t sid);
std::string formatCloseLabel(int32_t sid);
std::vector<std::string> locatePieces(const std::vector<int32_t> &ids);
bool isMaskedTag(const std::string &tag);
std::optional<std::string> parseAlpha(const std::string &text);
std::optional<std::string> parseBeta(const std::string &text);
std::vector<int32_t> parseIds(const std::string &text);
std::string forceAlphaAnswer(const std::string &text);
size_t locateRounds(const std::vector<Step> &steps);
std::vector<int32_t> pastedIds(const std::vector<Step> &steps);
std::vector<int32_t> locatedIds(const std::vector<Step> &steps);
bool mustAnswer(const std::vector<Step> &steps, const Proto_Config &cfg);
std::optional<std::string> nextTag(const std::vector<Step> &steps, const Proto_Config &cfg, std::string *err = nullptr);
std::vector<int32_t> orderInContext(const std::unordered_map<int32_t, size_t> &order, const std::vector<int32_t> &ids);
std::string checkLocate(const std::vector<int32_t> &ids,
                        const std::unordered_set<int32_t> &existing,
                        const std::unordered_set<int32_t> &already_pasted,
                        size_t k_max);
std::vector<std::string> validateSteps(const std::vector<Step> &steps,
                                       const Sample &sample,
                                       const Proto_Config &cfg,
                                       const ISealed_Read_Tokenizer *tok = nullptr);
std::string answerOf(const std::vector<Step> &steps);

enum class Branch : uint32_t
{
    SEALEDREAD = 0,
    NOMASK = 1,
    DAFT = 2
};

enum class Locate_History : uint32_t
{
    ALL = 0,
    QUERY_ONLY = 1
};

enum class View_Mode : uint32_t
{
    CHAIN = 0,
    MASKED = 1,
    LOCATE = 2
};

enum class Frame_Mode : uint32_t
{
    FULL = 0,
    COMPACT = 1
};

enum class Token_Kind : uint8_t
{
    A = 0,
    Q = 1,
    C = 2,
    QP = 3,
    TP = 4,
    GEN = 5,
    LOC = 6,
    CPY = 7,
    KNOW = 8,
    PAD = 9
};

struct Mask_Opts
{
    Branch branch = Branch::SEALEDREAD;
    bool hide_locate = true;
    Locate_History locate_history = Locate_History::ALL;
};

struct Prompts
{
    std::string system_prompt = "You are a careful assistant that answers questions about long documents.";
    std::string instruction =
        "You answer a question about a long set of documents. The documents are split into segments labelled [[S_k]] ... "
        "[[/S_k]], where k is an ID. While you think you cannot see the documents: you can only read the segments you ask for.\n\n"
        "Write your solution as a sequence of tags:\n"
        "<reason> what you know so far and what is still missing. End with alpha=query (you need to look something up) or "
        "alpha=answer (you have enough evidence). </reason>\n"
        "<query> the one thing to look up now </query>\n"
        "<locate> 1 to 3 segment IDs, e.g. [[S_12]] [[S_407]] </locate>\n"
        "The chosen segments are then pasted for you in <knowledge> ... </knowledge>.\n"
        "<recheck> check the pasted text against your query. End with beta=valid_to_think (it answers the query), "
        "beta=explore_query (it helps partly) or beta=reject_and_requery (it does not help: ask again). </recheck>\n"
        "<answer> the final answer only, as short as possible </answer>\n\n"
        "Rules: never ask again for a segment that is already pasted; before answering, check every step of your reasoning "
        "against the pasted text in a final <reason>; if the question needs no document (e.g. arithmetic), answer directly.";
    std::string assistant_prefix;
    std::string eos = "<|im_end|>";
    bool show_titles = true;

    std::string head() const;
    std::string question(const Sample &s) const;
    std::string segment(const Segment &seg, std::optional<int32_t> shown = std::nullopt) const;
    std::vector<std::pair<std::string, int32_t>> contextPieces(const Sample &s) const;
    std::string questionRepeat(const Sample &s, bool repeat_q = true) const;
    std::string assistantStart() const;
    std::string knowledgeBody(const Sample &s,
                              const std::vector<int32_t> &ids,
                              const std::unordered_map<int32_t, int32_t> &alias = {}) const;
};

struct Seq
{
    std::vector<int32_t> ids;
    std::vector<int32_t> pos;
    std::vector<Token_Kind> kind;
    std::vector<int32_t> rnd;
    std::vector<int32_t> seg;
    std::vector<int64_t> open_at;
    std::vector<int32_t> stp;
    std::vector<int32_t> qref;
    std::vector<uint8_t> target;
    Mask_Opts opts{};
    View_Mode view = View_Mode::CHAIN;

    size_t size() const noexcept { return ids.size(); }
    bool empty() const noexcept { return ids.empty(); }
    std::vector<int32_t> targetIndices() const;
    std::vector<int32_t> maskedTargetRows() const;
    bool isVisible(size_t q_idx, size_t k_idx) const noexcept;
    std::vector<int32_t> collectActiveKvIndices(size_t q_start, size_t q_end) const;
    Tensor buildDenseMask(size_t q_start,
                          size_t q_end,
                          const std::vector<int32_t> &k_indices,
                          Execution_Target target_dev) const;
};

class Seq_Builder
{
private:
    const ISealed_Read_Tokenizer *tok = nullptr;
    Mask_Opts opts{};
    View_Mode view = View_Mode::CHAIN;
    Seq data{};
    int32_t next_pos = 0;
    std::unordered_map<int32_t, std::vector<size_t>> c_spans;
    size_t c_tokens = 0;

public:
    int32_t round = 0;
    int32_t step = -1;
    int32_t cur_qref = -1;

    Seq_Builder() = default;
    Seq_Builder(const ISealed_Read_Tokenizer &_tok, const Mask_Opts &_opts = {}, View_Mode _view = View_Mode::CHAIN);

    size_t size() const noexcept { return data.size(); }
    bool empty() const noexcept { return data.empty(); }
    int32_t nextPos() const noexcept { return next_pos; }
    const Seq &seq() const noexcept { return data; }
    Seq &mutableSeq() noexcept { return data; }

    std::pair<size_t, size_t> addIds(const std::vector<int32_t> &toks,
                                     Token_Kind kind,
                                     bool target = false,
                                     int32_t seg = -1,
                                     const std::vector<int32_t> *explicit_pos = nullptr);
    std::pair<size_t, size_t> addText(const std::string &text,
                                      Token_Kind kind,
                                      bool target = false,
                                      int32_t seg = -1);
    std::pair<size_t, size_t> addPieces(const std::vector<std::string> &pieces,
                                        Token_Kind kind,
                                        bool target = false,
                                        const std::vector<int32_t> *explicit_pos = nullptr);
    void skip(size_t n);
    void open(const std::vector<int32_t> &sids, int64_t at);
    void truncate(size_t n);

    void addPrompt(const Sample &s,
                   const Prompts &prompts,
                   bool with_c = true,
                   Frame_Mode frame = Frame_Mode::FULL,
                   bool repeat_q = true,
                   std::optional<size_t> c_count = std::nullopt);
    void addLocateCopy(const Step &st, size_t loc_start, size_t loc_end);
    void addStep(const Sample &s, const Step &st, int32_t idx, const Prompts &prompts);
};

std::vector<std::string> tagPieces(const std::string &tag, const std::string &text, const Prompts &prompts);
std::vector<std::string> copyPieces(const Step &st);
Seq buildSeq(const Sample &sample,
             const std::vector<Step> &steps,
             const ISealed_Read_Tokenizer &tok,
             const Prompts &prompts,
             View_Mode view = View_Mode::CHAIN,
             Branch branch = Branch::SEALEDREAD,
             Frame_Mode frame = Frame_Mode::FULL,
             bool repeat_q = true,
             const Mask_Opts *opts_override = nullptr);

bool visible(size_t q_idx, size_t k_idx, const Seq &seq, const Mask_Opts &opts) noexcept;

class Locate_Trie
{
public:
    enum class Phase : uint8_t
    {
        ITEM = 0,
        BETWEEN = 1,
        SEP = 2,
        END = 3,
        DONE = 4
    };

private:
    struct Node
    {
        std::unordered_map<int32_t, std::unique_ptr<Node>> children;
        std::unordered_set<int32_t> sids;
        std::optional<int32_t> terminal;
    };

    size_t k_max = 3;
    std::unique_ptr<Node> root;
    const Node *cursor = nullptr;
    std::vector<int32_t> sep_tokens;
    std::vector<int32_t> end_tokens;
    std::vector<int32_t> chosen_ids;
    std::unordered_set<int32_t> chosen_set;
    Phase phase = Phase::ITEM;
    size_t forced_idx = 0;

    std::vector<int32_t> openChildren(const Node *node) const;
    void stepForced();

public:
    Locate_Trie(const ISealed_Read_Tokenizer &tok, const std::vector<int32_t> &allowed_ids, size_t _k_max = 3);

    std::vector<int32_t> allowedTokens() const;
    void advance(int32_t token_id);
    bool isDone() const noexcept { return phase == Phase::DONE; }
    Phase getPhase() const noexcept { return phase; }
    const std::vector<int32_t> &chosenIds() const noexcept { return chosen_ids; }
    void maskLogits(std::vector<float> &logits) const;
};

struct Driver_Config
{
    Proto_Config proto{};
    std::string locator = "model";
    bool lenient = false;
    bool permute_ids = false;
    size_t max_steps = 64;

    Proto_Config effectiveProto() const;
};

class IPolicy
{
public:
    virtual ~IPolicy() = default;
    virtual std::pair<std::string, bool> generateMasked(Episode &ep, const std::string &tag, bool force_answer) = 0;
    virtual std::vector<int32_t> generateLocate(Episode &ep, const std::vector<int32_t> &allowed_ids) = 0;
    virtual void release(Episode &ep) = 0;
};

class Scripted_Policy final : public IPolicy
{
private:
    float noise = 0.0f;
    float wrong = 0.0f;
    std::mt19937 rng{};

    const Hop *nextHop(const Episode &ep) const;

public:
    explicit Scripted_Policy(float _noise = 0.0f, float _wrong = 0.0f, uint32_t seed = 0);

    std::pair<std::string, bool> generateMasked(Episode &ep, const std::string &tag, bool force_answer) override;
    std::vector<int32_t> generateLocate(Episode &ep, const std::vector<int32_t> &allowed_ids) override;
    void release(Episode &ep) override;
};

struct Ref_Policy_Config
{
    Mask_Opts opts{};
    View_Mode view = View_Mode::CHAIN;
    Frame_Mode frame = Frame_Mode::FULL;
    bool repeat_q = true;
    Proto_Config proto{};
    float temperature = 0.0f;
    float top_p = 1.0f;
};

class Ref_Policy final : public IPolicy
{
private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    void ensureEpisodeState(Episode &ep);
    void feed(size_t upto, bool causal);
    void feedAll();
    void truncate(size_t n);
    void sync(Episode &ep);
    int32_t sampleNext(const std::vector<int32_t> *allowed);
    void appendSingleToken(int32_t token_id, Token_Kind kind);

public:
    Ref_Policy(Causal_LM &model,
               const ISealed_Read_Tokenizer &tok,
               const Prompts &prompts = Prompts{},
               const Ref_Policy_Config &cfg = Ref_Policy_Config{});
    ~Ref_Policy() override;

    std::pair<std::string, bool> generateMasked(Episode &ep, const std::string &tag, bool force_answer) override;
    std::vector<int32_t> generateLocate(Episode &ep, const std::vector<int32_t> &allowed_ids) override;
    void release(Episode &ep) override;

    Seq forceTrace(Episode &ep, const std::vector<Step> &steps);
};

std::vector<Episode> runEpisodes(IPolicy &policy,
                                 const std::vector<Sample> &samples,
                                 size_t n_per_sample,
                                 const Driver_Config &cfg,
                                 uint32_t seed = 0);

struct Reward_Config
{
    float lam = 0.2f;
    float mu = 0.25f;
    float mu0 = 0.25f;
    size_t k = 3;
    size_t r_max = 8;
};

struct Score
{
    bool valid = true;
    float r_ans = 0.0f;
    float f2 = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    size_t n_id = 0;
    size_t n_loc = 0;
    float acc = 0.0f;
    std::optional<float> r_mu;
};

std::string normalizeAnswer(const std::string &s);
float exactMatch(const std::string &pred, const std::vector<std::string> &answers);
float bestF1(const std::string &pred, const std::vector<std::string> &answers);
std::pair<float, float> precisionRecall(const std::vector<int32_t> &gold, const std::vector<int32_t> &got);
float f2Score(const std::vector<int32_t> &gold, const std::vector<int32_t> &got);
Score scoreEpisode(const std::vector<Step> &steps,
                   const Sample &sample,
                   const Reward_Config &cfg,
                   bool valid = true,
                   float lam_override = -1.0f,
                   bool gate = true);

enum class Rest_Em_Variant : uint32_t
{
    SOFT = 0,
    HARD = 1,
    NOGATE = 2
};

struct Rest_Em_Result
{
    std::vector<Episode> selected;
    size_t questions = 0;
    size_t learnable = 0;
    size_t all_pass = 0;
    size_t kept = 0;
};

Rest_Em_Result selectRestEm(const std::vector<Episode> &episodes,
                            const Proto_Config &proto,
                            const Reward_Config &rcfg,
                            Rest_Em_Variant variant = Rest_Em_Variant::SOFT,
                            size_t keep = 2,
                            size_t cap = 4000,
                            float recall_min = 0.5f,
                            uint32_t seed = 0,
                            const ISealed_Read_Tokenizer *tok = nullptr);

struct Invariance_Report
{
    std::string qid;
    Branch branch = Branch::SEALEDREAD;
    size_t positions = 0;
    float kl_mean = 0.0f;
    float kl_max = 0.0f;
    float max_abs_logp = 0.0f;
};

struct Dependence_Report
{
    std::string qid;
    Branch branch = Branch::SEALEDREAD;
    size_t answer_tokens = 0;
    float logp_orig = 0.0f;
    float logp_changed = 0.0f;
    float drop = 0.0f;
    float max_abs_delta_logp = 0.0f;
};

Invariance_Report probeInvariance(Causal_LM &model,
                                  const Sample &sample,
                                  const std::vector<Step> &steps,
                                  const ISealed_Read_Tokenizer &tok,
                                  const Prompts &prompts,
                                  const Mask_Opts &opts,
                                  uint32_t seed = 0);

std::optional<Dependence_Report> probeDependence(Causal_LM &model,
                                                 const Sample &sample,
                                                 const std::vector<Step> &steps,
                                                 const ISealed_Read_Tokenizer &tok,
                                                 const Prompts &prompts,
                                                 const std::string &new_value,
                                                 const Mask_Opts &opts);

struct Sft_Config
{
    Mask_Opts mask_opts{};
    size_t lora_rank = 8;
    float lora_alpha = 16.0f;
    float learning_rate = 2e-3f;
    float min_learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    size_t batch_traces = 4;
    size_t max_steps = 12;
    uint32_t seed = 0;
    bool enable_lora_if_inactive = true;
};

struct Sft_Metrics
{
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    float min_loss = 0.0f;
    size_t steps_done = 0;
    size_t traces_processed = 0;
    size_t eligible_traces = 0;
};

Sft_Metrics trainSft(Causal_LM &model,
                     const std::vector<Episode> &traces,
                     const ISealed_Read_Tokenizer &tok,
                     const Prompts &prompts,
                     const Sft_Config &cfg);

struct Aggregate_Metrics
{
    size_t total = 0;
    size_t valid = 0;
    float em = 0.0f;
    float f1 = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f2 = 0.0f;
    float acc = 0.0f;
    float r_mu = 0.0f;
};

Aggregate_Metrics aggregateEpisodes(const std::vector<Episode> &episodes,
                                    const Proto_Config &proto,
                                    const Reward_Config &rcfg,
                                    const ISealed_Read_Tokenizer *tok = nullptr);

}

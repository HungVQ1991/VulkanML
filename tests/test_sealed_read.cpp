#include "test_common.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "llm/sealed_read.h"

namespace
{

sealedread::Sample makeHandcraftedSample()
{
    sealedread::Sample s;
    s.qid = "q_unit_0";
    s.dataset = "unit";
    s.qtype = "multihop";
    s.question = "What city was the director of Dawn Harbour born in?";
    s.answers = { "Bergen" };
    s.segments = {
        sealedread::Segment{ 412, "Unrelated", "Intro", "Mount Peak is 1200 metres high.", "Mount Peak", false },
        sealedread::Segment{ 73, "Dawn Harbour", "Intro", "Dawn Harbour is a 1984 film directed by Erik Solberg.", "Dawn Harbour", true },
        sealedread::Segment{ 9105, "Valley", "Intro", "Green Valley exports apples each autumn.", "Valley", false },
        sealedread::Segment{ 508, "Erik Solberg", "Bio", "Erik Solberg was born in Bergen in 1951.", "Erik Solberg", true }
    };
    s.hops = {
        sealedread::Hop{ { 73 }, "Erik Solberg" },
        sealedread::Hop{ { 508 }, "Bergen" }
    };
    return s;
}

bool testProtocolAndValidator()
{
    const char *tag = "SR-Proto";
    const sealedread::Sample s = makeHandcraftedSample();
    const sealedread::Proto_Config cfg;
    const sealedread::Byte_Tokenizer tok;

    if (sealedread::parseAlpha("thinking step. alpha=query").value_or("") != "query" ||
        sealedread::parseAlpha("enough facts alpha = answer ").value_or("") != "answer" ||
        sealedread::parseAlpha("alpha=query extra").has_value())
    {
        return failTestStep(tag, "parseAlpha failed");
    }
    if (sealedread::parseBeta("helpful. beta=valid_to_think").value_or("") != "valid_to_think" ||
        sealedread::parseBeta("retry beta=reject_and_requery").value_or("") != "reject_and_requery")
    {
        return failTestStep(tag, "parseBeta failed");
    }
    if (sealedread::forceAlphaAnswer("still unsure alpha=query") != "still unsure alpha=answer" ||
        sealedread::forceAlphaAnswer("summary text") != "summary text alpha=answer")
    {
        return failTestStep(tag, "forceAlphaAnswer failed");
    }

    const auto ids = sealedread::parseIds("check [[S_508]] then [[S_73]] end");
    if (ids != std::vector<int32_t>{ 508, 73 })
    {
        return failTestStep(tag, "parseIds failed");
    }
    const auto ordered = sealedread::orderInContext(s.order(), { 508, 73 });
    if (ordered != std::vector<int32_t>{ 73, 508 })
    {
        return failTestStep(tag, "orderInContext failed");
    }

    if (!sealedread::checkLocate({ 73, 508 }, s.ids(), {}, 3).empty())
    {
        return failTestStep(tag, "valid checkLocate failed");
    }
    if (sealedread::checkLocate({ 73, 73 }, s.ids(), {}, 3).empty() ||
        sealedread::checkLocate({ 999 }, s.ids(), {}, 3).empty() ||
        sealedread::checkLocate({ 73 }, s.ids(), { 73 }, 3).empty() ||
        sealedread::checkLocate({}, s.ids(), {}, 3).empty())
    {
        return failTestStep(tag, "invalid checkLocate cases did not fail");
    }

    sealedread::Scripted_Policy oracle(0.0f, 0.0f, 1);
    const auto eps = sealedread::runEpisodes(oracle, { s }, 1, sealedread::Driver_Config{ cfg, "model", false, false, 64 }, 1);
    if (eps.size() != 1 || !eps[0].error.empty())
    {
        return failTestStep(tag, "Scripted_Policy oracle run failed");
    }
    const auto errs = sealedread::validateSteps(eps[0].steps, s, cfg, &tok);
    if (!errs.empty())
    {
        return failTestStep(tag, "validateSteps rejected oracle trace: " + errs.front());
    }

    std::vector<sealedread::Step> rmax_steps;
    for (int r = 0; r < 8; ++r)
    {
        rmax_steps.push_back(sealedread::Step{ "locate", "", { 73 }, "model", {} });
    }
    if (!sealedread::mustAnswer(rmax_steps, cfg))
    {
        return failTestStep(tag, "mustAnswer did not trigger at r_max");
    }
    return true;
}

bool testLocateTrieGrammar()
{
    const char *tag = "SR-Trie";
    const sealedread::Byte_Tokenizer tok;
    const std::vector<int32_t> allowed = { 73, 508 };
    sealedread::Locate_Trie trie(tok, allowed, 2);

    const std::string item1 = "[[S_508]]";
    for (int32_t t : tok.encode(item1))
    {
        const auto al = trie.allowedTokens();
        if (std::find(al.begin(), al.end(), t) == al.end())
        {
            return failTestStep(tag, "Locate_Trie rejected valid first item token");
        }
        trie.advance(t);
    }
    if (trie.chosenIds() != std::vector<int32_t>{ 508 } || trie.isDone())
    {
        return failTestStep(tag, "Locate_Trie state mismatch after first item");
    }
    for (int32_t t : tok.encode(" "))
    {
        trie.advance(t);
    }
    const std::vector<int32_t> dup_508 = tok.encode("[[S_508]]");
    const std::vector<int32_t> prefix_open = tok.encode("[[S_");
    for (int32_t t : prefix_open)
    {
        trie.advance(t);
    }
    const auto rem = trie.allowedTokens();
    const int32_t five_tok = static_cast<int32_t>('5');
    if (std::find(rem.begin(), rem.end(), five_tok) != rem.end())
    {
        return failTestStep(tag, "Locate_Trie failed to prune already-chosen sid 508");
    }
    (void)dup_508;
    for (int32_t t : tok.encode("73]]"))
    {
        trie.advance(t);
    }
    const auto between_al = trie.allowedTokens();
    if (between_al.size() != 1)
    {
        return failTestStep(tag, "Locate_Trie allowed separator past k_max");
    }
    for (int32_t t : tok.encode("</locate>"))
    {
        trie.advance(t);
    }
    if (!trie.isDone() || trie.chosenIds() != std::vector<int32_t>{ 508, 73 })
    {
        return failTestStep(tag, "Locate_Trie did not finish cleanly with chosen IDs");
    }
    return true;
}

bool testLayoutAndMaskRules()
{
    const char *tag = "SR-Layout";
    const sealedread::Sample s = makeHandcraftedSample();
    const sealedread::Byte_Tokenizer tok;
    const sealedread::Prompts prompts;
    const sealedread::Proto_Config proto;
    sealedread::Scripted_Policy oracle(0.0f, 0.0f, 3);
    const auto eps = sealedread::runEpisodes(oracle, { s }, 1, sealedread::Driver_Config{ proto, "model", false, false, 64 }, 3);
    const auto &steps = eps[0].steps;

    const auto seq_sr = sealedread::buildSeq(s, steps, tok, prompts, sealedread::View_Mode::CHAIN, sealedread::Branch::SEALEDREAD);
    const auto seq_nm = sealedread::buildSeq(s, steps, tok, prompts, sealedread::View_Mode::CHAIN, sealedread::Branch::NOMASK);
    const auto seq_df = sealedread::buildSeq(s, steps, tok, prompts, sealedread::View_Mode::CHAIN, sealedread::Branch::DAFT);

    std::vector<size_t> c_indices;
    std::vector<size_t> loc_r1;
    std::vector<size_t> cpy_r1;
    std::vector<size_t> final_gen;
    for (size_t i = 0; i < seq_sr.size(); ++i)
    {
        if (seq_sr.kind[i] == sealedread::Token_Kind::C) c_indices.push_back(i);
        if (seq_sr.kind[i] == sealedread::Token_Kind::LOC && seq_sr.rnd[i] == 1) loc_r1.push_back(i);
        if (seq_sr.kind[i] == sealedread::Token_Kind::CPY && seq_sr.rnd[i] == 1) cpy_r1.push_back(i);
        if (seq_sr.kind[i] == sealedread::Token_Kind::GEN) final_gen.push_back(i);
    }
    if (c_indices.empty() || loc_r1.empty() || cpy_r1.size() != loc_r1.size() || final_gen.empty())
    {
        return failTestStep(tag, "Missing expected token kinds in buildSeq");
    }
    for (size_t i = 0; i < loc_r1.size(); ++i)
    {
        if (seq_sr.pos[loc_r1[i]] != seq_sr.pos[cpy_r1[i]])
        {
            return failTestStep(tag, "D2 failed: CPY position ID did not match LOC position ID");
        }
    }

    auto lastGenIndex = [](const sealedread::Seq &seq) -> std::optional<size_t> {
        for (size_t i = seq.size(); i > 0; --i)
        {
            if (seq.kind[i - 1] == sealedread::Token_Kind::GEN)
            {
                return i - 1;
            }
        }
        return std::nullopt;
    };
    const auto ans_sr = lastGenIndex(seq_sr);
    const auto ans_nm = lastGenIndex(seq_nm);
    const auto ans_df = lastGenIndex(seq_df);
    if (!ans_sr.has_value() || !ans_nm.has_value() || !ans_df.has_value())
    {
        return failTestStep(tag, "Missing GEN answer token in sequence");
    }
    const size_t first_loc_tok = loc_r1.front();
    for (size_t c_i : c_indices)
    {
        if (seq_sr.isVisible(*ans_sr, c_i))
        {
            return failTestStep(tag, "SEALEDREAD allowed GEN token to attend to C");
        }
        if (!seq_nm.isVisible(*ans_nm, c_i))
        {
            return failTestStep(tag, "NOMASK hid C from GEN token");
        }
        if (seq_sr.seg[c_i] == 73 || seq_sr.seg[c_i] == 508)
        {
            if (!seq_df.isVisible(*ans_df, c_i))
            {
                return failTestStep(tag, "DAFT failed to open pasted segment in C");
            }
        }
        else if (seq_df.isVisible(*ans_df, c_i))
        {
            return failTestStep(tag, "DAFT leaked unpasted segment in C");
        }
    }

    for (size_t li : loc_r1)
    {
        if (seq_sr.isVisible(*ans_sr, li))
        {
            return failTestStep(tag, "hide_locate failed: post-LOC GEN token sees original LOC KV");
        }
    }

    sealedread::Mask_Opts qonly{ sealedread::Branch::SEALEDREAD, true, sealedread::Locate_History::QUERY_ONLY };
    const auto seq_qo = sealedread::buildSeq(s, steps, tok, prompts, sealedread::View_Mode::CHAIN, sealedread::Branch::SEALEDREAD, sealedread::Frame_Mode::FULL, true, &qonly);
    size_t loc_q = first_loc_tok;
    for (size_t k = 0; k < loc_q; ++k)
    {
        if (seq_qo.kind[k] == sealedread::Token_Kind::GEN)
        {
            const bool same_q = (seq_qo.stp[k] == seq_qo.qref[loc_q]);
            if (seq_qo.isVisible(loc_q, k) != same_q)
            {
                return failTestStep(tag, "Locate_History::QUERY_ONLY visibility mismatch");
            }
        }
    }
    return true;
}

bool testRewardAndRestEm()
{
    const char *tag = "SR-Reward";
    const sealedread::Sample s = makeHandcraftedSample();
    const sealedread::Proto_Config proto;
    const sealedread::Reward_Config rcfg;

    std::vector<sealedread::Sample> qs = { s };
    sealedread::Scripted_Policy noisy(0.4f, 0.3f, 7);
    const auto rollouts = sealedread::runEpisodes(noisy, qs, 4, sealedread::Driver_Config{ proto, "model", false, false, 64 }, 7);

    int passed = 0;
    for (const auto &ep : rollouts)
    {
        const auto sc = sealedread::scoreEpisode(ep.steps, ep.sample, rcfg, ep.error.empty());
        if (sc.r_ans >= 1.0f && sc.recall >= 0.5f)
        {
            passed++;
        }
    }
    if (passed == 0)
    {
        return failTestStep(tag, "Expected at least one passing noisy rollout");
    }

    const auto soft_res = sealedread::selectRestEm(rollouts, proto, rcfg, sealedread::Rest_Em_Variant::SOFT, 2, 100, 0.5f, 0);
    const auto hard_res = sealedread::selectRestEm(rollouts, proto, rcfg, sealedread::Rest_Em_Variant::HARD, 2, 100, 0.5f, 0);
    const auto ng_res = sealedread::selectRestEm(rollouts, proto, rcfg, sealedread::Rest_Em_Variant::NOGATE, 2, 100, 0.5f, 0);
    if (soft_res.selected.empty() || hard_res.selected.empty() || ng_res.selected.empty())
    {
        return failTestStep(tag, "selectRestEm returned zero traces");
    }
    if (soft_res.selected.size() > 2)
    {
        return failTestStep(tag, "selectRestEm keep cap exceeded");
    }
    return true;
}

bool runProperty1AndKvParityOnTarget(Execution_Target target)
{
    Scoped_Coop_Mat_Guard coop_guard;
    if (target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().setCooperativeMatrixEnabled(false);
    }
    const char *tag = (target == Execution_Target::VULKAN_GPU) ? "SR-ProbeKV-GPU" : "SR-ProbeKV-CPU";

    const sealedread::Sample s = makeHandcraftedSample();
    const sealedread::Byte_Tokenizer tok;
    const sealedread::Prompts prompts;
    const sealedread::Proto_Config proto;
    sealedread::Scripted_Policy oracle(0.0f, 0.0f, 11);
    const auto eps = sealedread::runEpisodes(oracle, { s }, 1, sealedread::Driver_Config{ proto, "model", false, false, 64 }, 11);
    const auto &steps = eps[0].steps;

    Causal_LM_Config cfg;
    cfg.vocab_size = Causal_LM::alignVocabSize(tok.vocabSize(), 16);
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 2;
    cfg.intermediate_dim = 128;
    cfg.num_layers = 2;
    cfg.max_seq_len = 4096;
    cfg.rms_norm_eps = 1e-6f;
    cfg.execution_target = target;
    cfg.data_type = Data_Type::FLOAT32;
    cfg.use_qk_norm = true;
    cfg.attention_bias = false;
    cfg.mlp_bias = false;
    cfg.tie_word_embeddings = true;
    cfg.use_fused_projections = false;
    cfg.use_loss_scaler = false;
    cfg.tokenizer_path = "";

    Causal_LM lm(cfg);

    const auto inv_sr = sealedread::probeInvariance(lm, s, steps, tok, prompts,
        sealedread::Mask_Opts{ sealedread::Branch::SEALEDREAD, true, sealedread::Locate_History::ALL }, 99);
    const auto inv_nm = sealedread::probeInvariance(lm, s, steps, tok, prompts,
        sealedread::Mask_Opts{ sealedread::Branch::NOMASK, true, sealedread::Locate_History::ALL }, 99);
    const auto inv_df = sealedread::probeInvariance(lm, s, steps, tok, prompts,
        sealedread::Mask_Opts{ sealedread::Branch::DAFT, true, sealedread::Locate_History::ALL }, 99);

    const float sr_tol = (target == Execution_Target::CPU) ? 1e-6f : 1e-4f;
    if (inv_sr.positions == 0 || inv_sr.kl_max > sr_tol || inv_sr.max_abs_logp > 1e-3f)
    {
        return failTestStep(tag, std::format("SEALEDREAD invariance failed (kl_max={:.6e}, max_logp={:.6e})", inv_sr.kl_max, inv_sr.max_abs_logp));
    }
    if (inv_nm.kl_max <= 1e-7f || inv_df.kl_max <= 1e-8f || inv_nm.max_abs_logp <= 1e-4f)
    {
        return failTestStep(tag, std::format("Expected leak on NOMASK/DAFT (nm={:.6e}, df={:.6e})", inv_nm.kl_max, inv_df.kl_max));
    }

    const auto dep = sealedread::probeDependence(lm, s, steps, tok, prompts, "Oslo",
        sealedread::Mask_Opts{ sealedread::Branch::SEALEDREAD, true, sealedread::Locate_History::ALL });
    if (!dep.has_value() || dep->max_abs_delta_logp <= 1e-5f)
    {
        return failTestStep(tag, "Evidence dependence probe failed");
    }

    const auto full_seq = sealedread::buildSeq(s, steps, tok, prompts, sealedread::View_Mode::CHAIN, sealedread::Branch::SEALEDREAD);
    const size_t T = full_seq.size();
    std::vector<float> id_f(T), pos_f(T);
    for (size_t i = 0; i < T; ++i)
    {
        id_f[i] = static_cast<float>(full_seq.ids[i]);
        pos_f[i] = static_cast<float>(full_seq.pos[i]);
    }
    Tensor full_in(Shape{ 1, T }, std::move(id_f), target);
    Tensor full_pos(Shape{ 1, T }, std::move(pos_f), target);
    std::vector<int32_t> all_k(T);
    std::iota(all_k.begin(), all_k.end(), 0);
    Tensor full_mask = full_seq.buildDenseMask(0, T, all_k, target);
    ops::llm::Attention_Config full_attn{};
    full_attn.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
    full_attn.attn_mask = &full_mask;
    Tensor full_h = lm.forwardHidden(full_in, false, &full_pos, &full_attn);
    Tensor last_idx(Shape{ 1 }, { static_cast<float>(T - 1) }, target);
    Tensor full_logits_t = lm.forwardTargetLogits(full_h, last_idx);
    flushIfGpu(target, true);
    const auto full_logits = full_logits_t.getData();

    sealedread::Episode inc_ep{};
    inc_ep.sample = s;
    sealedread::Ref_Policy_Config rcfg;
    rcfg.opts = sealedread::Mask_Opts{ sealedread::Branch::SEALEDREAD, true, sealedread::Locate_History::ALL };
    sealedread::Ref_Policy ref_pol(lm, tok, prompts, rcfg);
    const auto kv_seq = ref_pol.forceTrace(inc_ep, steps);
    ref_pol.release(inc_ep);

    if (kv_seq.ids != full_seq.ids || kv_seq.pos != full_seq.pos || kv_seq.kind != full_seq.kind)
    {
        return failTestStep(tag, "Ref_Policy forced layout differs from buildSeq");
    }

    const size_t T_half = T / 2;
    std::vector<float> p1_id(full_seq.ids.begin(), full_seq.ids.begin() + static_cast<std::ptrdiff_t>(T_half));
    std::vector<float> p1_pos(full_seq.pos.begin(), full_seq.pos.begin() + static_cast<std::ptrdiff_t>(T_half));
    std::vector<float> p2_id(full_seq.ids.begin() + static_cast<std::ptrdiff_t>(T_half), full_seq.ids.end());
    std::vector<float> p2_pos(full_seq.pos.begin() + static_cast<std::ptrdiff_t>(T_half), full_seq.pos.end());

    lm.resetKVCaches();
    Tensor in1(Shape{ 1, T_half }, std::move(p1_id), target);
    Tensor pos1(Shape{ 1, T_half }, std::move(p1_pos), target);
    std::vector<int32_t> k1(T_half);
    std::iota(k1.begin(), k1.end(), 0);
    Tensor m1 = full_seq.buildDenseMask(0, T_half, k1, target);
    ops::llm::Attention_Config a1{};
    a1.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
    a1.attn_mask = &m1;
    (void)lm.forwardHidden(in1, true, &pos1, &a1);
    flushIfGpu(target, true);

    const size_t T2 = T - T_half;
    const auto keep2 = full_seq.collectActiveKvIndices(T_half, T);
    Tensor in2(Shape{ 1, T2 }, std::move(p2_id), target);
    Tensor pos2(Shape{ 1, T2 }, std::move(p2_pos), target);
    Tensor kv_idx2(Shape{ keep2.size() }, std::vector<float>(keep2.begin(), keep2.end()), target);
    Tensor m2 = full_seq.buildDenseMask(T_half, T, keep2, target);
    ops::llm::Attention_Config a2{};
    a2.mask_mode = ops::llm::Attention_Mask_Mode::DENSE_BOOL;
    a2.attn_mask = &m2;
    a2.kv_index = &kv_idx2;
    Tensor h2 = lm.forwardHidden(in2, true, &pos2, &a2);
    Tensor idx2(Shape{ 1 }, { static_cast<float>(T2 - 1) }, target);
    Tensor kv_logits_t = lm.forwardTargetLogits(h2, idx2);
    flushIfGpu(target, true);
    const auto kv_logits = kv_logits_t.getData();

    double max_diff = 0.0;
    for (size_t i = 0; i < full_logits.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(full_logits[i]) - static_cast<double>(kv_logits[i])));
    }
    const double kv_tol = (target == Execution_Target::CPU) ? 1e-4 : 2e-3;
    if (max_diff > kv_tol)
    {
        return failTestStep(tag, std::format("Incremental KV_Cache + DENSE_BOOL + kv_index parity failed (max_diff={:.6f})", max_diff));
    }
    return true;
}

bool testProperty1AndKvParity()
{
    if (!runProperty1AndKvParityOnTarget(Execution_Target::CPU))
    {
        return false;
    }
    if (hasVulkanDevice() && !runProperty1AndKvParityOnTarget(Execution_Target::VULKAN_GPU))
    {
        return false;
    }
    return true;
}

bool testSftAndRefPolicyEndToEnd()
{
    const char *tag = "SR-SftRef";
    const sealedread::Byte_Tokenizer tok;
    const sealedread::Prompts prompts;
    const auto samples = sealedread::generateSyntheticDataset(sealedread::Synthetic_Dataset_Config{ 8, 4, 0.25f, 5, 0 });
    const sealedread::Proto_Config proto;
    sealedread::Scripted_Policy teacher(0.2f, 0.0f, 5);
    const auto traces = sealedread::runEpisodes(teacher, samples, 1, sealedread::Driver_Config{ proto, "model", false, false, 64 }, 5);

    Causal_LM_Config cfg;
    cfg.vocab_size = Causal_LM::alignVocabSize(tok.vocabSize(), 16);
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 2;
    cfg.intermediate_dim = 128;
    cfg.num_layers = 2;
    cfg.max_seq_len = 4096;
    cfg.rms_norm_eps = 1e-6f;
    cfg.execution_target = Execution_Target::CPU;
    cfg.data_type = Data_Type::FLOAT32;
    cfg.use_qk_norm = true;
    cfg.attention_bias = false;
    cfg.mlp_bias = false;
    cfg.tie_word_embeddings = true;
    cfg.use_fused_projections = false;
    cfg.use_loss_scaler = false;
    cfg.tokenizer_path = "";

    Causal_LM lm(cfg);
    sealedread::Sft_Config sft_cfg;
    sft_cfg.lora_rank = 4;
    sft_cfg.lora_alpha = 8.0f;
    sft_cfg.learning_rate = 5e-3f;
    sft_cfg.min_learning_rate = 1e-3f;
    sft_cfg.batch_traces = 2;
    sft_cfg.max_steps = 6;
    sft_cfg.seed = 5;

    const auto m = sealedread::trainSft(lm, traces, tok, prompts, sft_cfg);
    if (!std::isfinite(m.initial_loss) || !std::isfinite(m.final_loss) || !(m.final_loss < m.initial_loss))
    {
        return failTestStep(tag, std::format("SFT did not converge (init={:.4f}, final={:.4f})", m.initial_loss, m.final_loss));
    }

    sealedread::Ref_Policy_Config ref_cfg;
    ref_cfg.proto = sealedread::Proto_Config{ 3, 1, { { "reason", 12 }, { "query", 8 }, { "recheck", 10 }, { "answer", 6 } }, 24 };
    ref_cfg.temperature = 0.0f;
    sealedread::Ref_Policy ref(lm, tok, prompts, ref_cfg);
    std::vector<sealedread::Sample> two(samples.begin(), samples.begin() + 2);
    const auto eps = sealedread::runEpisodes(ref, two, 1, sealedread::Driver_Config{ ref_cfg.proto, "model", true, false, 20 }, 5);
    for (const auto &ep : eps)
    {
        if (ep.steps.empty() || ep.steps.back().tag != "answer")
        {
            return failTestStep(tag, "Ref_Policy episode failed to terminate at <answer>");
        }
    }
    return true;
}

}

void runSealedReadSuite(Test_Result_Counter &counter)
{
    counter.record("  SealedRead Protocol & Validator:  ", testProtocolAndValidator());
    counter.record("  SealedRead Locate_Trie Grammar:   ", testLocateTrieGrammar());
    counter.record("  SealedRead Layout & Mask Rules:   ", testLayoutAndMaskRules());
    counter.record("  SealedRead Reward & ReST-EM:      ", testRewardAndRestEm());
    counter.record("  SealedRead Prop1 Probe & KV Parity:", testProperty1AndKvParity());
    counter.record("  SealedRead SFT + RefPolicy E2E:   ", testSftAndRefPolicyEndToEnd());
}

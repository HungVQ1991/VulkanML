#include "llm/causal_lm.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cost_function/fused_cce_cost.h"
#include "ops/loss/loss_ops.h"
#include "engine/execution_engine.h"
#include "engine/loss_scaler.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"
#include "layer/embedding_layer.h"
#include "layer/linear_layer.h"
#include "layer/rmsnorm_layer.h"
#include "layer/transformer_block.h"
#include "llm/kv_cache_manager.h"
#include "math/gpu_tensor_impl.h"
#include "math/tensor.h"
#include "optimizer/ioptimizer.h"
#include "tokenizer/bpe_tokenizer.h"

namespace
{
    void ensureParentDirectoryExists(const std::string& file_path)
    {
        std::filesystem::path parent = std::filesystem::path(file_path).parent_path();
        if (!parent.empty())
        {
            std::filesystem::create_directories(parent);
        }
    }

    const char* targetName(Execution_Target target) noexcept
    {
        return (target == Execution_Target::VULKAN_GPU) ? "VULKAN_GPU" : "CPU";
    }

    const char* dataTypeName(Data_Type dtype) noexcept
    {
        switch (dtype)
        {
        case Data_Type::FLOAT16:
            return "FLOAT16";
        case Data_Type::TERNARY_INT2:
            return "TERNARY_INT2";
        default:
            return "FLOAT32";
        }
    }

    struct Static_Graph_Suspend_Guard
    {
        bool suspended = false;

        explicit Static_Graph_Suspend_Guard(Execution_Target target)
        {
            if (target == Execution_Target::VULKAN_GPU)
            {
                auto& engine = Execution_Engine::getInstance();
                if (engine.isStaticGraphEnabled())
                {
                    engine.enableStaticGraph(false);
                    suspended = true;
                }
            }
        }

        ~Static_Graph_Suspend_Guard()
        {
            if (suspended)
            {
                try
                {
                    Execution_Engine::getInstance().enableStaticGraph(true);
                }
                catch (...)
                {
                }
            }
        }
    };
}

struct Causal_LM::Runtime_State
{
    struct Static_Bake_Key
    {
        uint64_t bake_generation = 0;
        VkBuffer in_handle = VK_NULL_HANDLE;
        size_t in_vector_id = 0;
        Shape in_shape{};
        Data_Type in_dtype = Data_Type::FLOAT32;
        VkBuffer tgt_handle = VK_NULL_HANDLE;
        size_t tgt_vector_id = 0;
        Shape tgt_shape{};
        Data_Type tgt_dtype = Data_Type::FLOAT32;
        float loss_scale = 1.0f;
        float max_grad_norm = 0.0f;
        bool is_mixed_precision = false;
        bool is_fused_gemm_adam = false;
        bool profile_active = false;
        const IOptimizer* optimizer_ptr = nullptr;
        uint64_t optimizer_config_generation = 0;
        const ICost_Function* cost_function_ptr = nullptr;

        bool operator==(const Static_Bake_Key&) const = default;
    };

    struct Async_Loss_Override_Guard
    {
        bool active = false;
        bool prev_state = false;

        Async_Loss_Override_Guard() = default;
        Async_Loss_Override_Guard(const Async_Loss_Override_Guard&) = delete;
        Async_Loss_Override_Guard& operator=(const Async_Loss_Override_Guard&) = delete;
        Async_Loss_Override_Guard(Async_Loss_Override_Guard&& other) noexcept
            : active(std::exchange(other.active, false)),
              prev_state(other.prev_state)
        {
        }
        Async_Loss_Override_Guard& operator=(Async_Loss_Override_Guard&& other) noexcept
        {
            if (this != &other)
            {
                release();
                active = std::exchange(other.active, false);
                prev_state = other.prev_state;
            }
            return *this;
        }
        ~Async_Loss_Override_Guard() noexcept
        {
            release();
        }
        void acquire()
        {
            if (!active)
            {
                prev_state = Execution_Engine::getInstance().isAsyncLossEnabled();
                Execution_Engine::getInstance().setAsyncLossEnabled(true);
                active = true;
            }
        }
        void release() noexcept
        {
            if (active)
            {
                try
                {
                    if (!prev_state)
                    {
                        Execution_Engine::getInstance().setLatestLoss(0.0f);
                    }
                    Execution_Engine::getInstance().setAsyncLossEnabled(prev_state);
                }
                catch (...)
                {
                }
                active = false;
            }
        }
    };

    struct Static_Node_Counts
    {
        size_t fwd_loss = 0;
        size_t bwd = 0;
        size_t opt = 0;
    };

    std::array<Static_Bake_Key, MAX_FRAMES_IN_FLIGHT> static_bake_keys{};
    std::array<Static_Node_Counts, MAX_FRAMES_IN_FLIGHT> static_node_counts{};
    Async_Loss_Override_Guard async_loss_guard{};
};

int32_t Causal_LM::sampleToken(std::vector<float>& logits, float temperature, float top_p, size_t top_k) const
{
    if (logits.empty())
    {
        return 0;
    }

    for (float& val : logits)
    {
        if (std::isnan(val) || std::isinf(val))
        {
            val = -1e9f;
        }
    }

    if (temperature <= 1e-4f)
    {
        return static_cast<int32_t>(std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    for (float& val : logits)
    {
        val /= temperature;
    }

    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float& val : logits)
    {
        val = std::exp(val - max_logit);
        sum_exp += val;
    }
    if (sum_exp <= 0.0f || !std::isfinite(sum_exp))
    {
        return 0;
    }
    float inv_sum = 1.0f / sum_exp;
    for (float& val : logits)
    {
        val *= inv_sum;
    }

    static thread_local std::mt19937 default_rng(std::random_device{}());
    std::mt19937& active_rng = has_sampling_seed ? const_cast<std::mt19937&>(sampling_rng) : default_rng;

    if (top_k == 0 && (top_p <= 0.0f || top_p >= 1.0f))
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(active_rng);
        float acc = 0.0f;
        size_t best_idx = 0;
        float best_prob = logits[0];
        for (size_t i = 0; i < logits.size(); ++i)
        {
            if (logits[i] > best_prob)
            {
                best_prob = logits[i];
                best_idx = i;
            }
            acc += logits[i];
            if (r <= acc)
            {
                return static_cast<int32_t>(i);
            }
        }
        return static_cast<int32_t>(best_idx);
    }

    std::vector<std::pair<float, int32_t>> prob_idx(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
    {
        prob_idx[i] = { logits[i], static_cast<int32_t>(i) };
    }
    auto desc_cmp = [](const auto& a, const auto& b) {
        return a.first > b.first;
    };

    size_t cutoff = prob_idx.size();
    if (top_k > 0 && top_k < cutoff)
    {
        std::partial_sort(prob_idx.begin(), prob_idx.begin() + top_k, prob_idx.end(), desc_cmp);
        cutoff = top_k;
    }
    else
    {
        std::sort(prob_idx.begin(), prob_idx.end(), desc_cmp);
    }

    if (top_p > 0.0f && top_p < 1.0f)
    {
        float cumsum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i)
        {
            cumsum += prob_idx[i].first;
            if (cumsum >= top_p)
            {
                cutoff = i + 1;
                break;
            }
        }
    }

    float total_prob = 0.0f;
    for (size_t i = 0; i < cutoff; ++i)
    {
        total_prob += prob_idx[i].first;
    }

    std::uniform_real_distribution<float> dist(0.0f, std::max(total_prob, 1e-8f));
    float r = dist(active_rng);
    float acc = 0.0f;
    for (size_t i = 0; i < cutoff; ++i)
    {
        acc += prob_idx[i].first;
        if (r <= acc)
        {
            return prob_idx[i].second;
        }
    }
    return prob_idx[0].second;
}

const Causal_LM_Config& Causal_LM::validateConfig(const Causal_LM_Config& _config)
{
    if (_config.vocab_size == 0 || _config.hidden_dim == 0 || _config.num_heads == 0 ||
        _config.intermediate_dim == 0 || _config.max_seq_len == 0)
    {
        throw std::invalid_argument("Causal_LM: vocab_size, hidden_dim, num_heads, intermediate_dim, and max_seq_len must be > 0");
    }
    const size_t eff_head_dim = (_config.head_dim == 0)
        ? (_config.hidden_dim / _config.num_heads)
        : _config.head_dim;
    if ((_config.head_dim == 0 && (_config.hidden_dim % _config.num_heads != 0)) ||
        eff_head_dim == 0 || (eff_head_dim % 2) != 0)
    {
        throw std::invalid_argument("Causal_LM: head_dim must be non-zero and even (or hidden_dim divisible by num_heads with even head_dim)");
    }
    const size_t eff_kv = (_config.num_kv_heads == 0) ? _config.num_heads : _config.num_kv_heads;
    if (eff_kv == 0 || eff_kv > _config.num_heads || (_config.num_heads % eff_kv) != 0)
    {
        throw std::invalid_argument("Causal_LM: num_heads must be cleanly divisible by num_kv_heads (1 <= num_kv_heads <= num_heads)");
    }
    if (!std::isfinite(_config.rms_norm_eps) || _config.rms_norm_eps <= 0.0f)
    {
        throw std::invalid_argument("Causal_LM: rms_norm_eps must be finite and > 0");
    }
    if (!std::isfinite(_config.rope_base) || _config.rope_base <= 0.0f)
    {
        throw std::invalid_argument("Causal_LM: rope_base must be finite and > 0");
    }
    if (!std::isfinite(_config.residual_scale))
    {
        throw std::invalid_argument("Causal_LM: residual_scale must be finite");
    }
    if (_config.tie_word_embeddings && _config.ternary_lm_head)
    {
        throw std::invalid_argument("Causal_LM: tie_word_embeddings cannot be combined with ternary_lm_head");
    }
    return _config;
}

size_t Causal_LM::alignVocabSize(size_t raw_vocab_size, size_t alignment) noexcept
{
    if (alignment <= 1 || raw_vocab_size == 0)
    {
        return raw_vocab_size;
    }
    const size_t rem = raw_vocab_size % alignment;
    return rem == 0 ? raw_vocab_size : (raw_vocab_size + (alignment - rem));
}

Causal_LM_Config Causal_LM::prepareConfig(const Causal_LM_Config& _config)
{
    Causal_LM_Config prepared = _config;
    if (prepared.vocab_size == 0)
    {
        if (prepared.tokenizer_path.empty())
        {
            throw std::invalid_argument("Causal_LM: vocab_size == 0 requires a non-empty tokenizer_path");
        }
        Bpe_Tokenizer temp_tok;
        if (!temp_tok.load(prepared.tokenizer_path) || temp_tok.getVocabSize() == 0)
        {
            throw std::runtime_error("Causal_LM: Failed to infer vocab_size from tokenizer: " + prepared.tokenizer_path);
        }
        prepared.vocab_size = temp_tok.getVocabSize();
    }
    if (prepared.auto_align_vocab)
    {
        prepared.vocab_size = alignVocabSize(prepared.vocab_size, 16);
    }
    if (prepared.num_kv_heads == 0)
    {
        prepared.num_kv_heads = prepared.num_heads;
    }
    if (prepared.use_qk_norm || prepared.yarn.enabled)
    {
        prepared.use_fused_projections = false;
    }
    (void)validateConfig(prepared);
    return prepared;
}

size_t Causal_LM::resolveHeadDim() const noexcept
{
    return (config.head_dim == 0) ? (config.hidden_dim / config.num_heads) : config.head_dim;
}

void Causal_LM::syncTiedWeights() noexcept
{
    if (config.tie_word_embeddings)
    {
        lm_head.setUseBias(false);
        lm_head.tieWeights(&token_embedding.getWeight(), &token_embedding.getWeightGradient(), true);
        token_embedding.setAccumulateForTiedWeights(true);
    }
    else
    {
        lm_head.untieWeights();
        token_embedding.setAccumulateForTiedWeights(false);
    }
}

Causal_LM::Causal_LM(const Causal_LM_Config& _config)
    : config(prepareConfig(_config)),
    token_embedding(config.vocab_size, config.hidden_dim, config.execution_target,
        (config.data_type == Data_Type::TERNARY_INT2) ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32) : config.data_type),
    final_norm(config.hidden_dim, config.rms_norm_eps, config.execution_target,
        (config.data_type == Data_Type::TERNARY_INT2) ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32) : config.data_type),
    lm_head(config.hidden_dim, config.vocab_size, config.execution_target, 0.0004f * static_cast<float>(config.hidden_dim),
        (config.data_type == Data_Type::TERNARY_INT2 && !config.ternary_lm_head) ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32) : config.data_type,
        !config.tie_word_embeddings),
    cost_function(std::make_unique<Fused_Cross_Entropy>(config.ignore_index)),
    final_residual_buf(config.execution_target),
    runtime_state(std::make_unique<Runtime_State>())
{
    if ((config.data_type == Data_Type::FLOAT16 || config.data_type == Data_Type::TERNARY_INT2) && config.use_loss_scaler)
    {
        loss_scaler = std::make_unique<Loss_Scaler>(
            config.initial_loss_scale,
            config.loss_scale_growth_factor,
            config.loss_scale_backoff_factor,
            config.loss_scale_growth_interval,
            true);
    }

    syncTiedWeights();

    Data_Type kv_dtype = (config.data_type == Data_Type::TERNARY_INT2) ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32) : config.data_type;
    const size_t eff_head_dim = resolveHeadDim();

    blocks.reserve(config.num_layers);
    kv_caches.reserve(config.num_layers);
    shared_scratchpad = std::make_unique<Transformer_Scratchpad>(config.execution_target);
    for (size_t i = 0; i < config.num_layers; ++i)
    {
        Transformer_Block_Config blk_cfg;
        blk_cfg.hidden_dim = config.hidden_dim;
        blk_cfg.num_heads = config.num_heads;
        blk_cfg.head_dim = eff_head_dim;
        blk_cfg.num_kv_heads = config.num_kv_heads;
        blk_cfg.intermediate_dim = config.intermediate_dim;
        blk_cfg.rms_norm_eps = config.rms_norm_eps;
        blk_cfg.rope_base = config.rope_base;
        blk_cfg.yarn = config.yarn;
        blk_cfg.execution_target = config.execution_target;
        blk_cfg.data_type = config.data_type;
        blk_cfg.num_layers = config.num_layers;
        blk_cfg.residual_scale = config.residual_scale;
        blk_cfg.use_fused_projections = config.use_fused_projections && (config.data_type != Data_Type::TERNARY_INT2) && !config.use_qk_norm && !config.yarn.enabled;
        blk_cfg.use_qk_norm = config.use_qk_norm;
        blk_cfg.use_attention_bias = config.attention_bias;
        blk_cfg.use_mlp_bias = config.mlp_bias;
        blk_cfg.gradient_checkpointing = config.gradient_checkpointing;
        auto block = std::make_unique<Transformer_Block>(blk_cfg);
        block->setSharedScratchpad(shared_scratchpad.get(), i & 1u);
        blocks.push_back(std::move(block));

        kv_caches.push_back(std::make_unique<KV_Cache_Manager>(
            1,
            config.num_kv_heads,
            config.max_seq_len,
            eff_head_dim,
            config.execution_target,
            kv_dtype,
            KV_Cache_Mode::RING_BUFFER));
    }

    if (!config.tokenizer_path.empty() && !tokenizer.load(config.tokenizer_path))
    {
        Logger::logMessage(Input_Format{"Causal_LM: Failed to load tokenizer from '{}'", config.tokenizer_path},
                           Log_Level::LOG_WARNING,
                           true,
                           0,
                           Log_Feature::TOKENIZER);
    }

    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        warmupWeightCaches();
    }

    Logger::logMessage(
        Input_Format{"Causal_LM: Initialized (vocab={}, hidden={}, layers={}, heads={}/{}, ffn={}, max_seq={}, target={}, dtype={}, params={})",
                     config.vocab_size, config.hidden_dim, config.num_layers,
                     config.num_heads, config.num_kv_heads, config.intermediate_dim,
                     config.max_seq_len, targetName(config.execution_target), dataTypeName(config.data_type),
                     getParameterCount()},
        Log_Level::LOG_INFO,
        true,
        0,
        Log_Feature::TRANSFORMER);
}

Causal_LM::~Causal_LM() = default;

Causal_LM::Causal_LM(Causal_LM&& other) noexcept
    : config(std::move(other.config)),
      token_embedding(std::move(other.token_embedding)),
      shared_scratchpad(std::move(other.shared_scratchpad)),
      blocks(std::move(other.blocks)),
      kv_caches(std::move(other.kv_caches)),
      final_norm(std::move(other.final_norm)),
      lm_head(std::move(other.lm_head)),
      tokenizer(std::move(other.tokenizer)),
      loss_scaler(std::move(other.loss_scaler)),
      cost_function(std::move(other.cost_function)),
      is_training_mode(other.is_training_mode),
      d_logits_tensor(std::move(other.d_logits_tensor)),
      final_residual_buf(std::move(other.final_residual_buf)),
      cache_hidden_2d(std::move(other.cache_hidden_2d)),
      cache_target_hidden_shape(other.cache_target_hidden_shape),
      cache_last_input_shape(other.cache_last_input_shape),
      sampling_rng(std::move(other.sampling_rng)),
      has_sampling_seed(other.has_sampling_seed),
      runtime_state(std::move(other.runtime_state))
{
    syncTiedWeights();
    invalidateStaticBufferBindings();
    other.is_training_mode = false;
    other.has_sampling_seed = false;
}

Causal_LM& Causal_LM::operator=(Causal_LM&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    config = std::move(other.config);
    token_embedding = std::move(other.token_embedding);
    shared_scratchpad = std::move(other.shared_scratchpad);
    blocks = std::move(other.blocks);
    kv_caches = std::move(other.kv_caches);
    final_norm = std::move(other.final_norm);
    lm_head = std::move(other.lm_head);
    tokenizer = std::move(other.tokenizer);
    loss_scaler = std::move(other.loss_scaler);
    cost_function = std::move(other.cost_function);
    is_training_mode = other.is_training_mode;
    d_logits_tensor = std::move(other.d_logits_tensor);
    final_residual_buf = std::move(other.final_residual_buf);
    cache_hidden_2d = std::move(other.cache_hidden_2d);
    cache_target_hidden_shape = other.cache_target_hidden_shape;
    cache_last_input_shape = other.cache_last_input_shape;
    sampling_rng = std::move(other.sampling_rng);
    has_sampling_seed = other.has_sampling_seed;
    runtime_state = std::move(other.runtime_state);
    syncTiedWeights();
    invalidateStaticBufferBindings();
    other.is_training_mode = false;
    other.has_sampling_seed = false;
    return *this;
}

std::vector<std::pair<Tensor*, Tensor*>> Causal_LM::getParametersAndGradients()
{
    std::vector<std::pair<Tensor*, Tensor*>> params;
    auto add = [&params](ILayer& layer) {
        auto p = layer.getParametersAndGradients();
        params.insert(params.end(), p.begin(), p.end());
        };
    add(token_embedding);
    for (auto& block : blocks)
    {
        add(*block);
    }
    add(final_norm);
    add(lm_head);
    return params;
}

void Causal_LM::warmupWeightCaches()
{
    if (config.execution_target != Execution_Target::VULKAN_GPU)
    {
        return;
    }
    if (config.data_type == Data_Type::FLOAT32 &&
        !Execution_Engine::getInstance().isCooperativeMatrixEnabled())
    {
        return;
    }

    auto params = getParametersAndGradients();
    for (auto& [param, grad] : params)
    {
        if (param && !param->isEmpty())
        {
            param->prewarmFp16Cache();
        }
    }
    Execution_Engine::getInstance().executeGraph();
}

void Causal_LM::invalidateWeightCaches()
{
    token_embedding.invalidateWeightCache();
    for (auto& block : blocks)
    {
        block->invalidateWeightCache();
    }
    final_norm.invalidateWeightCache();
    lm_head.invalidateWeightCache();
}

void Causal_LM::invalidateStaticBufferBindings() noexcept
{
    if (runtime_state)
    {
        runtime_state->static_bake_keys.fill(Runtime_State::Static_Bake_Key{});
        runtime_state->static_node_counts.fill(Runtime_State::Static_Node_Counts{});
    }
}

bool Causal_LM::supportsStaticGraph() const noexcept
{
    if (config.execution_target != Execution_Target::VULKAN_GPU || isTernary())
    {
        return false;
    }
    if (config.gradient_checkpointing || config.use_qk_norm || config.yarn.enabled ||
        config.tie_word_embeddings || hasActiveLoRA())
    {
        return false;
    }
    if (!token_embedding.supportsStaticGraph() ||
        !final_norm.supportsStaticGraph() ||
        !lm_head.supportsStaticGraph())
    {
        return false;
    }
    for (const auto& block : blocks)
    {
        if (!block || !block->supportsStaticGraph())
        {
            return false;
        }
    }
    return true;
}

void Causal_LM::setExecutionTarget(Execution_Target new_target)
{
    if (new_target != Execution_Target::VULKAN_GPU)
    {
        if (runtime_state)
        {
            runtime_state->async_loss_guard.release();
        }
        Execution_Engine::getInstance().invalidateStaticGraph();
    }
    invalidateStaticBufferBindings();
    config.execution_target = new_target;
    if (shared_scratchpad)
    {
        shared_scratchpad->setExecutionTarget(new_target);
    }
    token_embedding.setExecutionTarget(new_target);
    for (auto& block : blocks)
    {
        block->setExecutionTarget(new_target);
    }
    for (auto& cache : kv_caches)
    {
        cache->setExecutionTarget(new_target);
    }
    final_norm.setExecutionTarget(new_target);
    lm_head.setExecutionTarget(new_target);
    syncTiedWeights();
    d_logits_tensor = Tensor(new_target);
    final_residual_buf = Tensor(new_target);
    cache_hidden_2d = Tensor(new_target);
    invalidateWeightCaches();
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        if (is_training_mode && runtime_state)
        {
            runtime_state->async_loss_guard.acquire();
        }
        warmupWeightCaches();
    }
}

void Causal_LM::setTrainingMode(bool is_training)
{
    const bool changed = (is_training != is_training_mode);
    is_training_mode = is_training;
    token_embedding.setTrainingMode(is_training);
    for (auto& block : blocks)
    {
        block->setTrainingMode(is_training);
    }
    final_norm.setTrainingMode(is_training);
    lm_head.setTrainingMode(is_training);
    if (changed)
    {
        invalidateStaticBufferBindings();
        if (!is_training && config.execution_target == Execution_Target::VULKAN_GPU)
        {
            invalidateWeightCaches();
            Execution_Engine::getInstance().invalidateStaticGraph();
        }
    }
    if (config.execution_target == Execution_Target::VULKAN_GPU && runtime_state)
    {
        if (is_training)
        {
            runtime_state->async_loss_guard.acquire();
        }
        else
        {
            runtime_state->async_loss_guard.release();
        }
    }
}

void Causal_LM::resetKVCaches() noexcept
{
    for (auto& cache : kv_caches)
    {
        cache->reset();
    }
}

void Causal_LM::truncateKVCaches(size_t new_seq_len)
{
    for (auto& cache : kv_caches)
    {
        cache->truncate(new_seq_len);
    }
}

void Causal_LM::resetForwardState()
{
    token_embedding.resetForwardState();
    for (auto& block : blocks)
    {
        block->resetForwardState();
    }
    final_norm.resetForwardState();
    lm_head.resetForwardState();
}

void Causal_LM::resetGradients()
{
    invalidateStaticBufferBindings();
    token_embedding.resetGradients();
    for (auto& block : blocks)
    {
        block->resetGradients();
    }
    final_norm.resetGradients();
    lm_head.resetGradients();
}

std::vector<std::vector<std::pair<Tensor*, Tensor*>>> Causal_LM::getParameterGroups()
{
    std::vector<std::vector<std::pair<Tensor*, Tensor*>>> groups;
    groups.push_back(token_embedding.getParametersAndGradients());
    for (auto& block : blocks)
    {
        groups.push_back(block->getParametersAndGradients());
    }
    std::vector<std::pair<Tensor*, Tensor*>> out_group;
    auto fn_p = final_norm.getParametersAndGradients();
    out_group.insert(out_group.end(), fn_p.begin(), fn_p.end());
    auto lm_p = lm_head.getParametersAndGradients();
    out_group.insert(out_group.end(), lm_p.begin(), lm_p.end());
    groups.push_back(out_group);
    return groups;
}

size_t Causal_LM::getParameterCount() const
{
    size_t count = token_embedding.getParameterCount();
    for (const auto& block : blocks)
    {
        if (block)
        {
            count += block->getParameterCount();
        }
    }
    count += final_norm.getWeights().getTotalElements();
    if (!config.tie_word_embeddings)
    {
        count += lm_head.getInputDimension() * lm_head.getOutputDimension();
    }
    if (lm_head.hasBias())
    {
        count += lm_head.getBiases().getTotalElements();
    }
    return count;
}

void Causal_LM::setAccumulated(bool is_accumulated)
{
    token_embedding.setAccumulated(is_accumulated);
    for (auto& block : blocks)
    {
        block->setAccumulated(is_accumulated);
    }
    final_norm.setAccumulated(is_accumulated);
    lm_head.setAccumulated(is_accumulated);
}

void Causal_LM::enableLoRA(size_t rank,
                           float alpha,
                           float dropout_rate,
                           const std::vector<std::string>& target_modules,
                           bool freeze_base)
{
    invalidateStaticBufferBindings();
    for (auto& block : blocks)
    {
        if (block)
        {
            block->enableLoRA(rank, alpha, dropout_rate, target_modules, freeze_base);
        }
    }
    if (freeze_base)
    {
        token_embedding.setTrainable(false);
        final_norm.setTrainable(false);
        lm_head.setTrainable(false);
    }
}

void Causal_LM::disableLoRA()
{
    invalidateStaticBufferBindings();
    for (auto& block : blocks)
    {
        if (block)
        {
            block->disableLoRA();
        }
    }
    token_embedding.setTrainable(true);
    final_norm.setTrainable(true);
    lm_head.setTrainable(true);
}

void Causal_LM::mergeLoRA()
{
    for (auto& block : blocks)
    {
        if (block)
        {
            block->mergeLoRA();
        }
    }
}

void Causal_LM::unmergeLoRA()
{
    for (auto& block : blocks)
    {
        if (block)
        {
            block->unmergeLoRA();
        }
    }
}

bool Causal_LM::hasActiveLoRA() const noexcept
{
    for (const auto& block : blocks)
    {
        if (block && block->hasActiveLoRA())
        {
            return true;
        }
    }
    return false;
}

void Causal_LM::setBaseTrainable(bool trainable) noexcept
{
    invalidateStaticBufferBindings();
    token_embedding.setTrainable(trainable);
    final_norm.setTrainable(trainable);
    lm_head.setTrainable(trainable);
    for (auto& block : blocks)
    {
        if (block)
        {
            block->setBaseTrainable(trainable);
        }
    }
}

Tensor Causal_LM::forwardHidden(const Tensor& input_tensor,
                                bool use_cache,
                                const Tensor* position_ids,
                                const ops::llm::Attention_Config* attn_override)
{
    cache_last_input_shape = input_tensor.getShape();
    if (input_tensor.isEmpty())
    {
        Data_Type h_dtype = (config.data_type == Data_Type::TERNARY_INT2)
            ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
            : config.data_type;
        cache_hidden_2d = Tensor(Shape{ 0, config.hidden_dim }, h_dtype, config.execution_target);
        cache_target_hidden_shape = Shape{ 0, config.hidden_dim };
        return Tensor(Shape{ 1, 0, config.hidden_dim }, h_dtype, config.execution_target);
    }

    size_t B = 1;
    size_t S = 1;
    if (input_tensor.getShape().getRank() == 2)
    {
        B = input_tensor.getShape()[0];
        S = input_tensor.getShape()[1];
    }
    else
    {
        B = 1;
        S = input_tensor.getTotalElements();
    }

    const size_t eff_head_dim = resolveHeadDim();
    if (use_cache && !kv_caches.empty() && kv_caches[0]->getBatchSize() != B)
    {
        Data_Type kv_dtype = kv_caches[0]->getDataType();
        KV_Cache_Mode kv_mode = kv_caches[0]->getMode();
        const size_t cache_max_seq = kv_caches[0]->getMaxSeqLen();
        const bool cache_auto_grow = kv_caches[0]->isAutoGrowEnabled();
        for (size_t l = 0; l < kv_caches.size(); ++l)
        {
            kv_caches[l] = std::make_unique<KV_Cache_Manager>(
                B,
                config.num_kv_heads,
                cache_max_seq,
                eff_head_dim,
                config.execution_target,
                kv_dtype,
                kv_mode,
                cache_auto_grow);
        }
    }

    Tensor x = token_embedding.forward(input_tensor);
    Tensor x_2d;
    if (!blocks.empty())
    {
        std::pair<Tensor, Tensor> res_pair{ x, Tensor(config.execution_target) };
        for (size_t l = 0; l < blocks.size(); ++l)
        {
            KV_Cache_Manager* cache_ptr = use_cache ? kv_caches[l].get() : nullptr;
            const float in_scale = (l > 0) ? blocks[l - 1]->getResidualScale() : 1.0f;
            res_pair = blocks[l]->forwardResidualPair(res_pair.first, res_pair.second, in_scale, cache_ptr, l > 0, B, S, position_ids, attn_override);
        }
        x_2d = final_norm.forwardResidualAdd(
            res_pair.first, res_pair.second, blocks.back()->getResidualScale(), final_residual_buf);
    }
    else
    {
        Tensor x_flat = (x.getShape().getRank() != 2 || x.getColumns() != config.hidden_dim)
            ? x.view(Shape{ B * S, config.hidden_dim })
            : x;
        x_2d = final_norm.forward(x_flat);
    }

    cache_hidden_2d = x_2d;
    cache_target_hidden_shape = Shape{ B * S, config.hidden_dim };
    if (input_tensor.getShape().getRank() == 2)
    {
        return x_2d.view(Shape{ B, S, config.hidden_dim });
    }
    return x_2d;
}

Tensor Causal_LM::forwardTargetLogits(const Tensor& hidden_states, const Tensor& target_indices)
{
    Tensor hidden_2d = (hidden_states.getShape().getRank() != 2 || hidden_states.getColumns() != config.hidden_dim)
        ? hidden_states.view(Shape{ hidden_states.getTotalElements() / config.hidden_dim, config.hidden_dim })
        : hidden_states;
    cache_target_hidden_shape = hidden_2d.getShape();
    return lm_head.forwardIndexed(hidden_2d, target_indices);
}

Tensor Causal_LM::forward(const Tensor& input_tensor,
                          bool use_cache,
                          const Tensor* position_ids,
                          const ops::llm::Attention_Config* attn_override)
{
    if (input_tensor.isEmpty())
    {
        Data_Type logits_dtype = (config.data_type == Data_Type::TERNARY_INT2) ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32) : config.data_type;
        return Tensor(Shape{ 1, 0, config.vocab_size }, logits_dtype, config.execution_target);
    }

    size_t B = 1;
    size_t S = 1;
    if (input_tensor.getShape().getRank() == 2)
    {
        B = input_tensor.getShape()[0];
        S = input_tensor.getShape()[1];
    }
    else
    {
        B = 1;
        S = input_tensor.getTotalElements();
    }

    (void)forwardHidden(input_tensor, use_cache, position_ids, attn_override);
    Tensor logits = lm_head.forward(cache_hidden_2d);
    if (input_tensor.getShape().getRank() == 2)
    {
        logits.reshape(Shape{ B, S, config.vocab_size });
    }

    if (config.execution_target == Execution_Target::VULKAN_GPU && !is_training_mode)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    return logits;
}

Tensor Causal_LM::forward(const Tensor& input_tensor, bool use_cache)
{
    return forward(input_tensor, use_cache, nullptr, nullptr);
}

Tensor Causal_LM::forward(const std::vector<int32_t>& token_ids, bool use_cache)
{
    if (token_ids.empty())
    {
        Data_Type logits_dtype = (config.data_type == Data_Type::TERNARY_INT2) ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32) : config.data_type;
        return Tensor(Shape{ 1, 0, config.vocab_size }, logits_dtype, config.execution_target);
    }

    Tensor in_tensor(Shape{ 1, token_ids.size() }, config.execution_target);
    std::vector<float> f_data(token_ids.begin(), token_ids.end());
    in_tensor.uploadData(f_data);

    Tensor logits = forward(in_tensor, use_cache);
    logits.reshape(Shape{ token_ids.size(), config.vocab_size });
    return logits;
}

Tensor Causal_LM::forward(const std::vector<std::vector<int32_t>>& batch_token_ids, bool use_cache)
{
    Data_Type logits_dtype = (config.data_type == Data_Type::TERNARY_INT2)
        ? (config.execution_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32)
        : config.data_type;
    if (batch_token_ids.empty())
    {
        return Tensor(Shape{ 0, 0, config.vocab_size }, logits_dtype, config.execution_target);
    }

    size_t B = batch_token_ids.size();
    size_t S = batch_token_ids[0].size();
    for (size_t b = 1; b < B; ++b)
    {
        if (batch_token_ids[b].size() != S)
        {
            throw std::invalid_argument("Causal_LM::forward: All sequences in batch_token_ids must have identical length");
        }
    }
    if (S == 0)
    {
        return Tensor(Shape{ B, 0, config.vocab_size }, logits_dtype, config.execution_target);
    }

    Tensor in_tensor(Shape{ B, S }, config.execution_target);
    std::vector<float> f_data(B * S, 0.0f);
    for (size_t b = 0; b < B; ++b)
    {
        for (size_t s = 0; s < S; ++s)
        {
            f_data[b * S + s] = static_cast<float>(batch_token_ids[b][s]);
        }
    }
    in_tensor.uploadData(f_data);
    return forward(in_tensor, use_cache);
}

Tensor Causal_LM::backwardFromHidden(Tensor d_x)
{
    d_x = final_norm.backward(d_x);

    Tensor prev_dx_scaled(config.execution_target);
    const Tensor* prescaled_branch_grad = nullptr;
    for (size_t i = blocks.size(); i > 0; --i)
    {
        const bool produce_scaled_dx = (i > 1);
        auto bwd_pair = blocks[i - 1]->backwardResidualChained(d_x, prescaled_branch_grad, produce_scaled_dx);
        d_x = bwd_pair.first;
        if (produce_scaled_dx)
        {
            prev_dx_scaled = bwd_pair.second;
            prescaled_branch_grad = &prev_dx_scaled;
        }
        else
        {
            prescaled_branch_grad = nullptr;
        }
    }

    return token_embedding.backward(d_x);
}

Tensor Causal_LM::backward(const Tensor& d_logits, bool defer_execution)
{
    if (d_logits.isEmpty() || d_logits.getTotalElements() == 0)
    {
        return d_logits;
    }
    const size_t nodes_before_bwd = (config.execution_target == Execution_Target::VULKAN_GPU)
        ? Execution_Engine::getInstance().getCurrentGraph().getNodes().size()
        : 0;
    auto t_bwd_rec_start = std::chrono::high_resolution_clock::now();
    size_t total_tokens = d_logits.getTotalElements() / config.vocab_size;
    Tensor d_logits_2d = d_logits;
    if (d_logits_2d.getShape().getRank() != 2 || d_logits_2d.getColumns() != config.vocab_size)
    {
        d_logits_2d.reshape(Shape{ total_tokens, config.vocab_size });
    }

    Tensor d_x = lm_head.backward(d_logits_2d);
    d_x = backwardFromHidden(std::move(d_x));
    auto t_bwd_rec_end = std::chrono::high_resolution_clock::now();
    Step_Timings::getInstance()[Timing_Stage::BWD_RECORD] += std::chrono::duration<double, std::milli>(t_bwd_rec_end - t_bwd_rec_start).count();

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            if (defer_execution)
            {
                const size_t nodes_after_bwd = Execution_Engine::getInstance().getCurrentGraph().getNodes().size();
                Step_Timings::getInstance()[Node_Stage::BWD_GPU_NODES] = (nodes_after_bwd >= nodes_before_bwd)
                    ? (nodes_after_bwd - nodes_before_bwd)
                    : nodes_after_bwd;
            }
            else
            {
                Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::BACKWARD);
                Step_Timings::getInstance()[Node_Stage::BWD_GPU_NODES] = Execution_Engine::getInstance().getLastExecutedNodeCount();
            }
        }

    return d_x;
}

float Causal_LM::computeLossAndGradient(const Tensor& logits, const std::vector<int32_t>& target_token_ids, Tensor& d_logits) const
{
    if (cost_function)
    {
        return cost_function->computeLossAndGradient(logits, target_token_ids, d_logits);
    }

    uint32_t valid = 0;
    for (int32_t id : target_token_ids)
    {
        if (id >= 0 && id != config.ignore_index && static_cast<size_t>(id) < config.vocab_size)
        {
            valid++;
        }
    }
    return ops::loss::fusedCrossEntropy(logits, target_token_ids, d_logits, valid, config.ignore_index);
}

float Causal_LM::computeLossAndGradient(const Tensor& logits, const Tensor& target_tensor, Tensor& d_logits, uint32_t valid_tokens) const
{
    if (cost_function)
    {
        if (valid_tokens > 0)
        {
            if (auto* fused = dynamic_cast<const Fused_Cross_Entropy*>(cost_function.get()))
            {
                return fused->computeLossAndGradient(logits, target_tensor, d_logits, valid_tokens);
            }
        }
        return cost_function->computeLossAndGradient(logits, target_tensor, d_logits);
    }
    return ops::loss::fusedCrossEntropy(logits, target_tensor, d_logits, valid_tokens, config.ignore_index);
}

float Causal_LM::computeTargetLossAndBackward(const Tensor& input_tensor,
                                             const Tensor& target_indices,
                                             const Tensor& target_labels,
                                             const Tensor* position_ids,
                                             const ops::llm::Attention_Config* attn_override,
                                             float loss_scale)
{
    Tensor hidden = forwardHidden(input_tensor, false, position_ids, attn_override);
    Tensor target_logits = forwardTargetLogits(hidden, target_indices);
    Tensor d_logits(config.execution_target);
    const float loss = computeLossAndGradient(target_logits, target_labels, d_logits);

    Tensor d_logits_scaled(config.execution_target);
    const Tensor* grad_ptr = &d_logits;
    if (loss_scale != 1.0f)
    {
        d_logits.mulScalar(loss_scale, d_logits_scaled);
        grad_ptr = &d_logits_scaled;
    }

    Tensor d_x = lm_head.backwardIndexed(*grad_ptr, target_indices, cache_target_hidden_shape);
    backwardFromHidden(std::move(d_x));
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::BACKWARD);
    }
    return loss;
}

float Causal_LM::clipGradients(float max_norm)
{
    if (max_norm <= 0.0f)
    {
        return 0.0f;
    }

    if (config.execution_target == Execution_Target::VULKAN_GPU &&
        !Execution_Engine::getInstance().getCurrentGraph().getNodes().empty())
    {
        Execution_Engine::getInstance().executeGraph();
    }

    auto param_groups = getParameterGroups();
    double sum_squares = 0.0;
    bool has_nan_inf = false;
    size_t p_idx = 0;
    std::vector<Tensor*> non_empty_grads;

    for (size_t g = 0; g < param_groups.size(); ++g)
    {
        for (auto& [param, grad] : param_groups[g])
        {
            if (grad && !grad->isEmpty())
            {
                std::vector<float> data = grad->getData();
                size_t nan_count = 0;
                double p_sum_sq = 0.0;
                float p_max_abs = 0.0f;
                for (float val : data)
                {
                    if (std::isnan(val) || std::isinf(val))
                    {
                        has_nan_inf = true;
                        nan_count++;
                    }
                    else
                    {
                        p_sum_sq += static_cast<double>(val) * val;
                        p_max_abs = std::max(p_max_abs, std::abs(val));
                    }
                }
                sum_squares += p_sum_sq;
                non_empty_grads.push_back(grad);

                double p_norm = std::sqrt(p_sum_sq);
                if (p_norm > 20.0)
                {
                    Logger::logMessage(
                        Input_Format{"Causal_LM::clipGradients: Large gradient on param #{} shape={} norm={:.4f} max_abs={:.4f}",
                                     p_idx, grad->getShape().toString(), p_norm, p_max_abs},
                        Log_Level::LOG_DEBUG,
                        false,
                        0,
                        Log_Feature::BACKWARD_PROPAGATION);
                }
                if (nan_count > 0)
                {
                    float first_val = data.empty() ? 0.0f : data[0];
                    Logger::logMessage(
                        Input_Format{"Causal_LM::clipGradients: Overflow pinpoint on param #{} shape={} count={} first_val={}",
                                     p_idx, grad->getShape().toString(), nan_count, first_val},
                        Log_Level::LOG_WARNING,
                        true,
                        0,
                        Log_Feature::BACKWARD_PROPAGATION);
                }
            }
            p_idx++;
        }
    }

    if (has_nan_inf)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    float total_norm = static_cast<float>(std::sqrt(sum_squares));
    bool any_clipped = false;
    if (total_norm > max_norm && total_norm > 1e-6f)
    {
        float scale = max_norm / total_norm;
        for (Tensor* grad : non_empty_grads)
        {
            grad->mulScalar(scale, *grad);
        }
        any_clipped = true;
    }

    if (any_clipped && config.execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    return total_norm;
}

bool Causal_LM::stepOptimizer(IOptimizer& optimizer, float max_grad_norm, float loss_scale, bool overflow_hint, bool is_chained_with_backward)
{
    float actual_loss_scale = (loss_scale > 0.0f)
        ? loss_scale
        : ((loss_scaler && loss_scaler->isEnabled()) ? loss_scaler->getScale() : 1.0f);

    bool has_overflow = overflow_hint;
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        if (max_grad_norm > 0.0f)
        {
            if (is_chained_with_backward)
            {
                optimizer.setMaxGradient(max_grad_norm);
                Logger::logMessage(
                    Input_Format{"Causal_LM::stepOptimizer: Chained GPU step uses per-element gradient clamp ({:.4f}) instead of sync L2 norm clip", max_grad_norm},
                    Log_Level::LOG_DEBUG,
                    true,
                    1,
                    Log_Feature::OPTIMIZER_STEP);
            }
            else
            {
                float norm = clipGradients(max_grad_norm * actual_loss_scale);
                if (std::isnan(norm) || std::isinf(norm))
                {
                    has_overflow = true;
                }
            }
        }
        else if (!is_chained_with_backward && !has_overflow && loss_scaler && loss_scaler->isEnabled())
        {
            auto params_and_grads = getParametersAndGradients();
            if (loss_scaler->hasOverflow(params_and_grads))
            {
                has_overflow = true;
            }
        }
    }
    else
    {
        if (max_grad_norm > 0.0f)
        {
            float norm = clipGradients(max_grad_norm * actual_loss_scale);
            if (std::isnan(norm) || std::isinf(norm))
            {
                has_overflow = true;
            }
        }
        else if (loss_scaler && loss_scaler->isEnabled())
        {
            auto params_and_grads = getParametersAndGradients();
            if (loss_scaler->hasOverflow(params_and_grads))
            {
                has_overflow = true;
            }
        }
    }

    if (has_overflow)
    {
        Logger::logMessage(
            Input_Format{"Causal_LM::stepOptimizer: Non-finite gradient/loss overflow detected (scale={:.1f}), skipping optimizer step", actual_loss_scale},
            Log_Level::LOG_WARNING,
            true,
            4,
            Log_Feature::GRADIENT_HEALTH);
        (void)optimizer.consumeGpuOverflowDetected();
        if (loss_scaler && loss_scaler->isEnabled())
        {
            loss_scaler->step(true);
        }
        if (is_chained_with_backward && config.execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getCurrentGraph().clear();
            Execution_Engine::getInstance().getContext().clearTransferTasks();
            auto& slot = Execution_Engine::getInstance().getLossSlot(
                Execution_Engine::getInstance().getContext().getCurrentFrame());
            slot.has_pending_read = false;
            slot.is_submitted = false;
        }
        return false;
    }

    const size_t raw_nodes_before_opt = (config.execution_target == Execution_Target::VULKAN_GPU)
        ? Execution_Engine::getInstance().getCurrentGraph().getNodes().size()
        : 0;

    if (is_chained_with_backward && config.execution_target == Execution_Target::VULKAN_GPU)
    {
        auto& nodes = Execution_Engine::getInstance().getCurrentGraph().getNodes();
        if (!nodes.empty())
        {
            nodes.back().is_barrier_required_after = true;
        }
    }

    auto& timings = Step_Timings::getInstance();
    double fence_wait_before = timings[Timing_Stage::BWD_FENCE_WAIT]
                             + timings[Timing_Stage::OPT_FENCE_WAIT]
                             + timings[Timing_Stage::FWD_LOSS_FENCE_WAIT];
    auto t_opt_rec_start = std::chrono::high_resolution_clock::now();
    auto params_and_grads = getParametersAndGradients();
    optimizer.step(params_and_grads, actual_loss_scale);
    auto t_opt_rec_end = std::chrono::high_resolution_clock::now();
    double fence_wait_after = timings[Timing_Stage::BWD_FENCE_WAIT]
                            + timings[Timing_Stage::OPT_FENCE_WAIT]
                            + timings[Timing_Stage::FWD_LOSS_FENCE_WAIT];
    double opt_rec_ms = std::chrono::duration<double, std::milli>(t_opt_rec_end - t_opt_rec_start).count()
                      - std::max(0.0, fence_wait_after - fence_wait_before);
    timings[Timing_Stage::OPT_RECORD] += std::max(0.0, opt_rec_ms);

    const bool async_gpu_overflow = (config.execution_target == Execution_Target::VULKAN_GPU)
        ? optimizer.consumeGpuOverflowDetected()
        : false;
    if (loss_scaler && loss_scaler->isEnabled())
    {
        loss_scaler->step(async_gpu_overflow);
    }

    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Stage stage = is_chained_with_backward ? Execution_Stage::BACKWARD_OPTIMIZER : Execution_Stage::OPTIMIZER;
        Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, stage);
        size_t total_nodes = Execution_Engine::getInstance().getLastExecutedNodeCount();
        if (is_chained_with_backward)
        {
            size_t fwd_nodes = Step_Timings::getInstance()[Node_Stage::FWD_LOSS_GPU_NODES];
            size_t bwd_nodes = Step_Timings::getInstance()[Node_Stage::BWD_GPU_NODES];
            size_t base_nodes = (raw_nodes_before_opt > bwd_nodes) ? (fwd_nodes + bwd_nodes) : bwd_nodes;
            Step_Timings::getInstance()[Node_Stage::OPT_GPU_NODES] = (total_nodes >= base_nodes)
                ? (total_nodes - base_nodes)
                : 0;
            if (raw_nodes_before_opt <= bwd_nodes)
            {
                Step_Timings::getInstance()[Node_Stage::BWD_GPU_NODES] = std::min(bwd_nodes, total_nodes);
            }
        }
        else
        {
            Step_Timings::getInstance()[Node_Stage::OPT_GPU_NODES] = total_nodes;
        }
    }
    return true;
}

namespace
{
struct Defer_Forward_Loss_Guard
{
    Execution_Engine* engine = nullptr;
    bool prev = false;

    Defer_Forward_Loss_Guard(Execution_Engine* eng, bool enable) noexcept
        : engine((enable && eng != nullptr) ? eng : nullptr)
    {
        if (engine != nullptr)
        {
            prev = engine->isDeferForwardLossActive();
            engine->setDeferForwardLossActive(true);
        }
    }

    ~Defer_Forward_Loss_Guard() noexcept
    {
        if (engine != nullptr)
        {
            engine->setDeferForwardLossActive(prev);
        }
    }

    Defer_Forward_Loss_Guard(const Defer_Forward_Loss_Guard&) = delete;
    Defer_Forward_Loss_Guard& operator=(const Defer_Forward_Loss_Guard&) = delete;
};

template <typename InputT, typename TargetT>
float runForwardLossAndBackwardImpl(
    Causal_LM& lm,
    const Causal_LM_Config& config,
    Tensor& d_logits_tensor,
    const InputT& input_arg,
    const TargetT& target_arg,
    float loss_scale,
    bool defer_backward_execution)
{
    const bool is_gpu = (config.execution_target == Execution_Target::VULKAN_GPU);
    Execution_Engine* engine = is_gpu ? &Execution_Engine::getInstance() : nullptr;
    const bool use_single_submit = is_gpu && defer_backward_execution && engine->isAsyncLossEnabled() && engine->isSingleSubmitEnabled();

    if (use_single_submit)
    {
        uint32_t cur_frame = engine->getContext().getCurrentFrame();
        if (!engine->getContext().isFrameReady(cur_frame))
        {
            engine->prepareCurrentFrame(Execution_Stage::FORWARD);
        }
    }

    lm.setTrainingMode(true);

    auto t_fwd_rec_start = std::chrono::high_resolution_clock::now();
    Tensor logits = lm.forward(input_arg, false);
    auto t_fwd_rec_end = std::chrono::high_resolution_clock::now();
    Step_Timings::getInstance()[Timing_Stage::FWD_RECORD] += std::chrono::duration<double, std::milli>(t_fwd_rec_end - t_fwd_rec_start).count();

    auto t_loss_start = std::chrono::high_resolution_clock::now();
    uint32_t loss_frame = is_gpu ? engine->getContext().getCurrentFrame() : 0;
    float loss = 0.0f;
    {
        Defer_Forward_Loss_Guard defer_guard(engine, use_single_submit);
        if (is_gpu)
        {
            engine->setExecutionStage(Execution_Stage::FORWARD);
        }
        loss = lm.computeLossAndGradient(logits, target_arg, d_logits_tensor);
        if (is_gpu)
        {
            engine->setExecutionStage(Execution_Stage::NONE);
        }
    }
    auto t_loss_end = std::chrono::high_resolution_clock::now();

    if (is_gpu)
    {
        if (use_single_submit)
        {
            Step_Timings::getInstance()[Node_Stage::FWD_LOSS_GPU_NODES] = engine->getCurrentGraph().getNodes().size();
        }
        else
        {
            double fwd_gpu = engine->getLastExecutionTimeMs();
            Step_Timings::getInstance()[Timing_Stage::FWD_LOSS_EXEC] += fwd_gpu;
            Step_Timings::getInstance()[Node_Stage::FWD_LOSS_GPU_NODES] = engine->getLastExecutedNodeCount();
            if (!engine->isAsyncLossEnabled())
            {
                double total_loss_call = std::chrono::duration<double, std::milli>(t_loss_end - t_loss_start).count();
                double loss_read = std::max(0.0, total_loss_call - fwd_gpu);
                Step_Timings::getInstance()[Timing_Stage::LOSS_READBACK] += loss_read;
            }
        }
    }

    if (loss_scale != 1.0f && loss_scale > 0.0f)
    {
        auto t_mul_start = std::chrono::high_resolution_clock::now();
        d_logits_tensor.mulScalar(loss_scale, d_logits_tensor);
        auto t_mul_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance()[Timing_Stage::MUL_SCALAR] += std::chrono::duration<double, std::milli>(t_mul_end - t_mul_start).count();
    }

    lm.backward(d_logits_tensor, defer_backward_execution);
    if (!is_gpu || !engine->isAsyncLossEnabled())
    {
        return loss;
    }
    if (use_single_submit)
    {
        return engine->getLatestLoss();
    }
    float current_loss = engine->getLatestLoss();
    if (current_loss <= 0.0f)
    {
        current_loss = engine->readPendingLoss(loss_frame);
    }
    return current_loss;
}

template <typename InputT, typename TargetT>
float runTrainStepImpl(
    Causal_LM& lm,
    const Causal_LM_Config& config,
    Loss_Scaler* loss_scaler,
    const InputT& input_arg,
    const TargetT& target_arg,
    IOptimizer& optimizer,
    float max_grad_norm,
    bool* out_step_applied = nullptr)
{
    auto t_step_start = std::chrono::high_resolution_clock::now();
    lm.setAccumulated(false);

    auto t_reset_start = std::chrono::high_resolution_clock::now();
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        lm.resetForwardState();
    }
    else
    {
        lm.resetGradients();
    }
    auto t_reset_end = std::chrono::high_resolution_clock::now();
    Step_Timings::getInstance()[Timing_Stage::RESET_GRAD] += std::chrono::duration<double, std::milli>(t_reset_end - t_reset_start).count();

    float current_scale = (loss_scaler && loss_scaler->isEnabled()) ? loss_scaler->getScale() : 1.0f;
    const bool is_gpu = (config.execution_target == Execution_Target::VULKAN_GPU);
    const uint32_t step_frame = is_gpu ? Execution_Engine::getInstance().getContext().getCurrentFrame() : 0;
    float loss = lm.forwardLossAndBackward(input_arg, target_arg, current_scale, is_gpu);
    bool overflow = std::isnan(loss) || std::isinf(loss);

    const bool ok = lm.stepOptimizer(optimizer, max_grad_norm, current_scale, overflow, is_gpu);
    if (out_step_applied)
    {
        *out_step_applied = ok;
    }
    if (ok && is_gpu && Execution_Engine::getInstance().isAsyncLossEnabled() && loss <= 0.0f)
    {
        loss = Execution_Engine::getInstance().readPendingLoss(step_frame);
    }

    auto t_step_end = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
    Step_Timings::getInstance()[Timing_Stage::TOTAL_BATCH] += step_ms;
    return loss;
}
}

float Causal_LM::forwardLossAndBackward(const std::vector<int32_t>& input_tokens, const std::vector<int32_t>& target_tokens, float loss_scale, bool defer_backward_execution)
{
    if (input_tokens.size() != target_tokens.size())
    {
        throw std::invalid_argument("Causal_LM::forwardLossAndBackward: input_tokens and target_tokens must have identical length");
    }
    if (input_tokens.empty())
    {
        return 0.0f;
    }

    return runForwardLossAndBackwardImpl(*this, config, d_logits_tensor, input_tokens, target_tokens, loss_scale, defer_backward_execution);
}

float Causal_LM::forwardLossAndBackward(const Tensor& input_tensor, const Tensor& target_tensor, float loss_scale, bool defer_backward_execution)
{
    if (input_tensor.isEmpty() || input_tensor.getTotalElements() == 0 ||
        target_tensor.isEmpty() || target_tensor.getTotalElements() == 0)
    {
        return 0.0f;
    }
    if (input_tensor.getTotalElements() != target_tensor.getTotalElements())
    {
        throw std::invalid_argument("Causal_LM::forwardLossAndBackward: input_tensor and target_tensor element counts must match");
    }

    return runForwardLossAndBackwardImpl(*this, config, d_logits_tensor, input_tensor, target_tensor, loss_scale, defer_backward_execution);
}

float Causal_LM::trainStep(const std::vector<int32_t>& input_tokens, const std::vector<int32_t>& target_tokens, IOptimizer& optimizer, float max_grad_norm)
{
    if (input_tokens.size() != target_tokens.size())
    {
        throw std::invalid_argument("Causal_LM::trainStep: input_tokens and target_tokens must have identical length");
    }
    if (input_tokens.empty())
    {
        return 0.0f;
    }

    return runTrainStepImpl(*this, config, loss_scaler.get(), input_tokens, target_tokens, optimizer, max_grad_norm);
}

float Causal_LM::trainStep(const Tensor& input_tensor, const Tensor& target_tensor, IOptimizer& optimizer, float max_grad_norm, bool* out_step_applied)
{
    if (input_tensor.isEmpty() || input_tensor.getTotalElements() == 0 ||
        target_tensor.isEmpty() || target_tensor.getTotalElements() == 0)
    {
        if (out_step_applied)
        {
            *out_step_applied = false;
        }
        return 0.0f;
    }
    if (input_tensor.getTotalElements() != target_tensor.getTotalElements())
    {
        throw std::invalid_argument("Causal_LM::trainStep: input_tensor and target_tensor element counts must match");
    }

    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        if (!is_training_mode)
        {
            setTrainingMode(true);
        }

        auto& engine = Execution_Engine::getInstance();
        uint32_t cur_frame = engine.getContext().getCurrentFrame();
        if (!engine.getContext().isFrameReady(cur_frame))
        {
            engine.prepareCurrentFrame(Execution_Stage::FORWARD);
            cur_frame = engine.getContext().getCurrentFrame();
        }

        auto extractGpuHandle = [](const Tensor& t) noexcept -> VkBuffer {
            auto impl = t.getImplementation();
            if (!impl || t.getExecutionTarget() != Execution_Target::VULKAN_GPU)
            {
                return VK_NULL_HANDLE;
            }
            const auto& gpu = Gpu_Tensor_Impl::castToGpu(*impl);
            return gpu.getStorageBuffer() ? gpu.getStorageBuffer()->getBuffer() : VK_NULL_HANDLE;
        };
        auto extractGpuVectorId = [](const Tensor& t) noexcept -> size_t {
            auto impl = t.getImplementation();
            if (!impl || t.getExecutionTarget() != Execution_Target::VULKAN_GPU)
            {
                return 0;
            }
            const auto& gpu = Gpu_Tensor_Impl::castToGpu(*impl);
            return gpu.getStorageBuffer() ? gpu.getStorageBuffer()->getId() : 0;
        };
        auto isContiguousZeroOffset = [](const Tensor& t) noexcept -> bool {
            auto impl = t.getImplementation();
            return impl && impl->isContiguous() && impl->getByteOffset() == 0;
        };

        if (!runtime_state)
        {
            runtime_state = std::make_unique<Runtime_State>();
        }

        const float current_scale = (loss_scaler && loss_scaler->isEnabled()) ? loss_scaler->getScale() : 1.0f;
        const bool static_ok = supportsStaticGraph();

        Runtime_State::Static_Bake_Key current_key{};
        current_key.bake_generation = engine.getGraphExecutor().getStaticBakeGeneration(cur_frame);
        current_key.in_handle = extractGpuHandle(input_tensor);
        current_key.in_vector_id = extractGpuVectorId(input_tensor);
        current_key.in_shape = input_tensor.getShape();
        current_key.in_dtype = input_tensor.getDataType();
        current_key.tgt_handle = extractGpuHandle(target_tensor);
        current_key.tgt_vector_id = extractGpuVectorId(target_tensor);
        current_key.tgt_shape = target_tensor.getShape();
        current_key.tgt_dtype = target_tensor.getDataType();
        current_key.loss_scale = current_scale;
        current_key.max_grad_norm = max_grad_norm;
        current_key.is_mixed_precision = (config.data_type == Data_Type::FLOAT16 || engine.isCooperativeMatrixEnabled());
        current_key.is_fused_gemm_adam = Graph_Optimizer::isFusedGemmAdamEnabled();
        current_key.profile_active = engine.getGraphExecutor().isGpuProfilingEnabled();
        current_key.optimizer_ptr = &optimizer;
        current_key.optimizer_config_generation = optimizer.getConfigGeneration();
        current_key.cost_function_ptr = cost_function.get();

        const bool can_fast_replay =
            static_ok &&
            engine.isStaticGraphEnabled() &&
            engine.isAsyncLossEnabled() &&
            engine.isSingleSubmitEnabled() &&
            engine.getGraphExecutor().isStaticBaked(cur_frame) &&
            engine.getGraphExecutor().isStaticBakedWithProfiling(cur_frame) == current_key.profile_active &&
            engine.getGraphExecutor().areStaticBakedBuffersValid(cur_frame) &&
            current_key.bake_generation != 0 &&
            current_key.in_handle != VK_NULL_HANDLE &&
            current_key.tgt_handle != VK_NULL_HANDLE &&
            isContiguousZeroOffset(input_tensor) &&
            isContiguousZeroOffset(target_tensor) &&
            runtime_state->static_bake_keys[cur_frame] == current_key;

        if (can_fast_replay)
        {
            auto t_step_start = std::chrono::high_resolution_clock::now();

            optimizer.stepDynamicParams(current_scale);
            const bool gpu_overflow = optimizer.consumeGpuOverflowDetected();
            if (loss_scaler && loss_scaler->isEnabled())
            {
                loss_scaler->step(gpu_overflow);
            }

            for (auto& [p, g] : getParametersAndGradients())
            {
                if (p && p->getImplementation() && p->getExecutionTarget() == Execution_Target::VULKAN_GPU)
                {
                    Gpu_Tensor_Impl::castToGpu(*p->getImplementation()).clearHostCache();
                }
            }

            auto& slot = engine.getLossSlot(cur_frame);
            slot.valid_tokens = 0;
            slot.total_tokens = input_tensor.getTotalElements();
            slot.has_pending_read = true;
            slot.is_submitted = false;

            engine.executeStaticReplay(VK_NULL_HANDLE, Execution_Stage::BACKWARD_OPTIMIZER);

            const auto& cached_nodes = runtime_state->static_node_counts[cur_frame];
            auto& timings = Step_Timings::getInstance();
            timings.setNodeCount(Node_Stage::FWD_LOSS_GPU_NODES, cached_nodes.fwd_loss);
            timings.setNodeCount(Node_Stage::BWD_GPU_NODES, cached_nodes.bwd);
            timings.setNodeCount(Node_Stage::OPT_GPU_NODES, cached_nodes.opt);

            float loss = engine.getLatestLoss();
            if (loss <= 0.0f)
            {
                loss = engine.readPendingLoss(cur_frame);
            }

            auto t_step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
            Step_Timings::getInstance()[Timing_Stage::TOTAL_BATCH] += step_ms;

            if (out_step_applied)
            {
                *out_step_applied = true;
            }
            return loss;
        }

        if (engine.isStaticGraphEnabled() &&
            Logger::shouldLog(Log_Level::LOG_DEBUG, Log_Feature::STATIC_GRAPH))
        {
            const auto& prev_key = runtime_state->static_bake_keys[cur_frame];
            const char* reason = "config_key_mismatch";
            if (!static_ok) reason = "model_unsupported";
            else if (!engine.isAsyncLossEnabled()) reason = "async_loss_disabled";
            else if (!engine.isSingleSubmitEnabled()) reason = "single_submit_disabled";
            else if (!engine.getGraphExecutor().isStaticBaked(cur_frame)) reason = "not_baked_yet";
            else if (engine.getGraphExecutor().isStaticBakedWithProfiling(cur_frame) != current_key.profile_active) reason = "profiling_mode_changed";
            else if (!engine.getGraphExecutor().areStaticBakedBuffersValid(cur_frame)) reason = "baked_buffers_invalid";
            else if (current_key.bake_generation == 0) reason = "bake_generation_zero";
            else if (current_key.in_handle == VK_NULL_HANDLE || current_key.tgt_handle == VK_NULL_HANDLE) reason = "null_io_buffer";
            else if (!isContiguousZeroOffset(input_tensor) || !isContiguousZeroOffset(target_tensor)) reason = "non_contiguous_or_offset";
            else if (prev_key.bake_generation != current_key.bake_generation) reason = "bake_generation_mismatch";
            else if (prev_key.in_handle != current_key.in_handle || prev_key.in_vector_id != current_key.in_vector_id) reason = "input_buffer_changed";
            else if (prev_key.tgt_handle != current_key.tgt_handle || prev_key.tgt_vector_id != current_key.tgt_vector_id) reason = "target_buffer_changed";
            else if (prev_key.in_shape != current_key.in_shape || prev_key.tgt_shape != current_key.tgt_shape) reason = "shape_changed";
            else if (prev_key.loss_scale != current_key.loss_scale) reason = "loss_scale_changed";
            else if (prev_key.optimizer_ptr != current_key.optimizer_ptr || prev_key.optimizer_config_generation != current_key.optimizer_config_generation) reason = "optimizer_changed";
            Logger::logMessage(Input_Format{"Causal_LM::trainStep: Fast replay miss (frame={}): {}", cur_frame, reason},
                               Log_Level::LOG_DEBUG, true, 4, Log_Feature::STATIC_GRAPH);
        }

        if (engine.isStaticGraphEnabled() && static_ok && current_key.is_mixed_precision)
        {
            bool needs_warmup = false;
            for (const auto& [p, g] : getParametersAndGradients())
            {
                if (!p || p->isEmpty() || p->getDataType() != Data_Type::FLOAT32)
                {
                    continue;
                }
                const auto& p_gpu = Gpu_Tensor_Impl::castToGpu(*p->getImplementation());
                if (p_gpu.isFp16CacheDirty() || !p_gpu.getCachedFp16Storage())
                {
                    needs_warmup = true;
                    break;
                }
            }
            if (needs_warmup)
            {
                warmupWeightCaches();
            }
        }

        const uint32_t submitted_frame = engine.getContext().getCurrentFrame();
        bool step_ok = false;
        float loss = runTrainStepImpl(*this, config, loss_scaler.get(), input_tensor, target_tensor, optimizer, max_grad_norm, &step_ok);
        if (out_step_applied)
        {
            *out_step_applied = step_ok;
        }

        if (step_ok &&
            static_ok &&
            engine.isStaticGraphEnabled() &&
            engine.getGraphExecutor().isStaticBaked(submitted_frame))
        {
            current_key.bake_generation = engine.getGraphExecutor().getStaticBakeGeneration(submitted_frame);
            current_key.optimizer_config_generation = optimizer.getConfigGeneration();
            current_key.loss_scale = current_scale;
            runtime_state->static_bake_keys[submitted_frame] = current_key;
            auto& timings = Step_Timings::getInstance();
            runtime_state->static_node_counts[submitted_frame] = Runtime_State::Static_Node_Counts{
                timings[Node_Stage::FWD_LOSS_GPU_NODES],
                timings[Node_Stage::BWD_GPU_NODES],
                timings[Node_Stage::OPT_GPU_NODES]
            };
        }
        else
        {
            runtime_state->static_bake_keys[submitted_frame] = Runtime_State::Static_Bake_Key{};
            runtime_state->static_node_counts[submitted_frame] = Runtime_State::Static_Node_Counts{};
        }
        return loss;
    }

    return runTrainStepImpl(*this, config, loss_scaler.get(), input_tensor, target_tensor, optimizer, max_grad_norm, out_step_applied);
}

float Causal_LM::trainStep(const std::vector<std::vector<int32_t>>& batch_inputs, const std::vector<std::vector<int32_t>>& batch_targets, IOptimizer& optimizer, float max_grad_norm)
{
    if (batch_inputs.size() != batch_targets.size())
    {
        throw std::invalid_argument("Causal_LM::trainStep: batch_inputs and batch_targets batch sizes must match");
    }
    if (batch_inputs.empty())
    {
        return 0.0f;
    }

    size_t B = batch_inputs.size();
    size_t S = batch_inputs[0].size();
    for (size_t b = 0; b < B; ++b)
    {
        if (batch_inputs[b].size() != S || batch_targets[b].size() != S)
        {
            throw std::invalid_argument("Causal_LM::trainStep: All sequences in batch_inputs and batch_targets must have identical length");
        }
    }
    if (S == 0)
    {
        return 0.0f;
    }

    auto t_prep_start = std::chrono::high_resolution_clock::now();

    Tensor in_tensor(Shape{ B, S }, config.execution_target);
    Tensor tgt_tensor(Shape{ B, S }, config.execution_target);

    std::vector<float> in_data(B * S, 0.0f);
    std::vector<float> tgt_data(B * S, -1.0f);

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t s = 0; s < S; ++s)
        {
            in_data[b * S + s] = static_cast<float>(batch_inputs[b][s]);
            tgt_data[b * S + s] = static_cast<float>(batch_targets[b][s]);
        }
    }

    in_tensor.uploadData(in_data);
    tgt_tensor.uploadData(tgt_data);
    auto t_prep_end = std::chrono::high_resolution_clock::now();
    double prep_ms = std::chrono::duration<double, std::milli>(t_prep_end - t_prep_start).count();
    Step_Timings::getInstance()[Timing_Stage::DATA_PREP] += prep_ms;

        float loss = trainStep(in_tensor, tgt_tensor, optimizer, max_grad_norm);

    Step_Timings::getInstance()[Timing_Stage::TOTAL_BATCH] += prep_ms;
        return loss;
}

float Causal_LM::trainStep(const std::string& text, IOptimizer& optimizer)
{
    if (!tokenizer.isLoaded())
    {
        throw std::runtime_error("Causal_LM::trainStep: Tokenizer is not loaded");
    }

    auto t_prep_start = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> tokens = tokenizer.encode(text, true);
    if (tokens.size() < 2)
    {
        return 0.0f;
    }

    if (tokens.size() > config.max_seq_len + 1)
    {
        tokens.resize(config.max_seq_len + 1);
    }

    std::vector<int32_t> input_tokens(tokens.begin(), tokens.end() - 1);
    std::vector<int32_t> target_tokens(tokens.begin() + 1, tokens.end());
    auto t_prep_end = std::chrono::high_resolution_clock::now();
    double prep_ms = std::chrono::duration<double, std::milli>(t_prep_end - t_prep_start).count();
    Step_Timings::getInstance()[Timing_Stage::DATA_PREP] += prep_ms;

        float loss = trainStep(input_tokens, target_tokens, optimizer);
    Step_Timings::getInstance()[Timing_Stage::TOTAL_BATCH] += prep_ms;
        return loss;
}

namespace
{
    size_t utf8CompletePrefixLength(const std::string& text)
    {
        const size_t n = text.size();
        if (n == 0)
        {
            return 0;
        }
        const unsigned char last = static_cast<unsigned char>(text[n - 1]);
        if ((last & 0x80) == 0)
        {
            return n;
        }

        const size_t max_scan = std::min<size_t>(4, n);
        for (size_t k = 1; k <= max_scan; ++k)
        {
            const unsigned char b = static_cast<unsigned char>(text[n - k]);
            if ((b & 0xC0) == 0x80)
            {
                continue;
            }
            if (b < 0x80)
            {
                return n;
            }

            size_t expected_len = 1;
            if ((b & 0xE0) == 0xC0 && b >= 0xC2)
            {
                expected_len = 2;
            }
            else if ((b & 0xF0) == 0xE0)
            {
                expected_len = 3;
            }
            else if ((b & 0xF8) == 0xF0 && b <= 0xF4)
            {
                expected_len = 4;
            }

            if (expected_len > 1 && k < expected_len)
            {
                return n - k;
            }
            return n;
        }
        return n;
    }

    size_t stepBackToUtf8Boundary(const std::string& text, size_t pos, size_t min_pos)
    {
        while (pos > min_pos && pos < text.size() &&
               (static_cast<unsigned char>(text[pos]) & 0xC0) == 0x80)
        {
            --pos;
        }
        return pos;
    }
}

void Causal_LM::prepareInference()
{
    invalidateStaticBufferBindings();
    for (auto &b : blocks)
    {
        b->prepareInference();
    }
    if (lm_head.isTernary())
    {
        lm_head.prepareInference();
    }
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }
    Logger::logMessage(Input_Format{"Causal_LM::prepareInference: Packed inference weights across {} blocks (ternary_lm_head={})",
                                    blocks.size(), lm_head.isTernary() ? "true" : "false"},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::TERNARY_METRICS);
}

void Causal_LM::quantizeToTernary()
{
    invalidateStaticBufferBindings();
    config.data_type = Data_Type::TERNARY_INT2;
    for (auto &b : blocks)
    {
        b->quantizeToTernary();
    }
    if (config.ternary_lm_head)
    {
        lm_head.quantizeToTernary();
    }
    Logger::logMessage(Input_Format{"Causal_LM::quantizeToTernary: Quantized {} blocks (ternary_lm_head={})",
                                    blocks.size(), config.ternary_lm_head ? "true" : "false"},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::TERNARY_METRICS);
}

void Causal_LM::saveCheckpoint(const std::string& file_path) const
{
    ensureParentDirectoryExists(file_path);
    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        Logger::logMessage(Input_Format{"Causal_LM::saveCheckpoint: Failed to open file '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::saveCheckpoint: Failed to open file: " + file_path);
    }

    uint32_t magic = 0x434D4C43u;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    constexpr uint64_t CAUSAL_LM_GQA_FLAG = (1ULL << 63);
    constexpr uint64_t CAUSAL_LM_EXT_FLAG = (1ULL << 62);
    const size_t eff_kv = (config.num_kv_heads == 0) ? config.num_heads : config.num_kv_heads;
    const size_t eff_head_dim = resolveHeadDim();
    const bool write_gqa = (eff_kv != config.num_heads);
    const bool write_ext = (eff_head_dim != config.hidden_dim / config.num_heads) ||
                           ((config.hidden_dim % config.num_heads) != 0) ||
                           config.use_qk_norm || !config.attention_bias || !config.mlp_bias ||
                           config.tie_word_embeddings || config.residual_scale >= 0.0f ||
                           config.gradient_checkpointing;
    uint64_t v_size = config.vocab_size;
    uint64_t h_dim = config.hidden_dim;
    uint64_t n_heads = static_cast<uint64_t>(config.num_heads)
                     | (write_gqa ? CAUSAL_LM_GQA_FLAG : 0ULL)
                     | (write_ext ? CAUSAL_LM_EXT_FLAG : 0ULL);
    uint64_t i_dim = config.intermediate_dim;
    uint64_t n_layers = config.num_layers;
    uint64_t m_seq = config.max_seq_len;
    out.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
    out.write(reinterpret_cast<const char*>(&h_dim), sizeof(h_dim));
    out.write(reinterpret_cast<const char*>(&n_heads), sizeof(n_heads));
    out.write(reinterpret_cast<const char*>(&i_dim), sizeof(i_dim));
    out.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));
    out.write(reinterpret_cast<const char*>(&m_seq), sizeof(m_seq));
    out.write(reinterpret_cast<const char*>(&config.rms_norm_eps), sizeof(config.rms_norm_eps));
    out.write(reinterpret_cast<const char*>(&config.rope_base), sizeof(config.rope_base));
    if (write_gqa)
    {
        uint64_t n_kv = static_cast<uint64_t>(eff_kv);
        out.write(reinterpret_cast<const char*>(&n_kv), sizeof(n_kv));
    }
    if (write_ext)
    {
        uint64_t h_d = static_cast<uint64_t>(eff_head_dim);
        uint8_t ext_flags = 0u;
        if (config.use_qk_norm) ext_flags |= 0x01u;
        if (config.attention_bias) ext_flags |= 0x02u;
        if (config.mlp_bias) ext_flags |= 0x04u;
        if (config.tie_word_embeddings) ext_flags |= 0x08u;
        if (config.gradient_checkpointing) ext_flags |= 0x10u;
        out.write(reinterpret_cast<const char*>(&h_d), sizeof(h_d));
        out.write(reinterpret_cast<const char*>(&config.residual_scale), sizeof(config.residual_scale));
        out.write(reinterpret_cast<const char*>(&ext_flags), sizeof(ext_flags));
    }

    token_embedding.saveCheckpoint(out);
    for (const auto& block : blocks)
    {
        block->saveCheckpoint(out);
    }
    final_norm.saveCheckpoint(out);
    if (!config.tie_word_embeddings)
    {
        lm_head.saveCheckpoint(out);
    }

    const uint32_t trailer_magic = 0x32434C43u;
    const uint32_t saved_dtype = static_cast<uint32_t>(config.data_type);
    const uint8_t saved_ternary_lm_head = config.ternary_lm_head ? 1 : 0;
    const uint8_t has_scaler = (loss_scaler != nullptr) ? 1 : 0;
    const float scaler_scale = loss_scaler ? loss_scaler->getScaleFactor() : 1.0f;
    const uint32_t scaler_good_steps = loss_scaler ? loss_scaler->getGoodSteps() : 0u;
    const uint8_t scaler_enabled = (loss_scaler && loss_scaler->isEnabled()) ? 1 : 0;

    out.write(reinterpret_cast<const char*>(&trailer_magic), sizeof(trailer_magic));
    out.write(reinterpret_cast<const char*>(&saved_dtype), sizeof(saved_dtype));
    out.write(reinterpret_cast<const char*>(&saved_ternary_lm_head), sizeof(saved_ternary_lm_head));
    out.write(reinterpret_cast<const char*>(&has_scaler), sizeof(has_scaler));
    out.write(reinterpret_cast<const char*>(&scaler_scale), sizeof(scaler_scale));
    out.write(reinterpret_cast<const char*>(&scaler_good_steps), sizeof(scaler_good_steps));
    out.write(reinterpret_cast<const char*>(&scaler_enabled), sizeof(scaler_enabled));
    out.flush();

    if (!out.good())
    {
        Logger::logMessage(Input_Format{"Causal_LM::saveCheckpoint: Failed writing checkpoint '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::saveCheckpoint: Failed writing checkpoint to: " + file_path);
    }
    Logger::logMessage(Input_Format{"Causal_LM::saveCheckpoint: Saved checkpoint to '{}' (dtype={})",
                                    file_path, dataTypeName(config.data_type)},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::CHECKPOINT_IO);
}

void Causal_LM::loadCheckpoint(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        Logger::logMessage(Input_Format{"Causal_LM::loadCheckpoint: Failed to open file '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::loadCheckpoint: Failed to open file: " + file_path);
    }

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!in.good() || magic != 0x434D4C43u)
    {
        Logger::logMessage(Input_Format{"Causal_LM::loadCheckpoint: Invalid magic header in '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::loadCheckpoint: Invalid magic header in: " + file_path);
    }

    constexpr uint64_t CAUSAL_LM_GQA_FLAG = (1ULL << 63);
    constexpr uint64_t CAUSAL_LM_EXT_FLAG = (1ULL << 62);
    uint64_t v_size = 0, h_dim = 0, n_heads_raw = 0, i_dim = 0, n_layers = 0, m_seq = 0;
    float loaded_eps = 0.0f;
    float loaded_rope = 0.0f;
    in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
    in.read(reinterpret_cast<char*>(&h_dim), sizeof(h_dim));
    in.read(reinterpret_cast<char*>(&n_heads_raw), sizeof(n_heads_raw));
    in.read(reinterpret_cast<char*>(&i_dim), sizeof(i_dim));
    in.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
    in.read(reinterpret_cast<char*>(&m_seq), sizeof(m_seq));
    in.read(reinterpret_cast<char*>(&loaded_eps), sizeof(loaded_eps));
    in.read(reinterpret_cast<char*>(&loaded_rope), sizeof(loaded_rope));
    const bool has_gqa = (n_heads_raw & CAUSAL_LM_GQA_FLAG) != 0;
    const bool has_ext = (n_heads_raw & CAUSAL_LM_EXT_FLAG) != 0;
    const uint64_t n_heads = n_heads_raw & ~(CAUSAL_LM_GQA_FLAG | CAUSAL_LM_EXT_FLAG);
    uint64_t n_kv = n_heads;
    if (has_gqa)
    {
        in.read(reinterpret_cast<char*>(&n_kv), sizeof(n_kv));
    }
    const size_t eff_head_dim = resolveHeadDim();
    uint64_t loaded_head_dim = (config.hidden_dim % config.num_heads == 0) ? (config.hidden_dim / config.num_heads) : 0;
    float loaded_res_scale = -1.0f;
    uint8_t loaded_ext_flags = 0x06u;
    if (has_ext)
    {
        in.read(reinterpret_cast<char*>(&loaded_head_dim), sizeof(loaded_head_dim));
        in.read(reinterpret_cast<char*>(&loaded_res_scale), sizeof(loaded_res_scale));
        in.read(reinterpret_cast<char*>(&loaded_ext_flags), sizeof(loaded_ext_flags));
    }
    if (!in.good())
    {
        throw std::runtime_error("Causal_LM::loadCheckpoint: Truncated checkpoint header in: " + file_path);
    }

    const size_t eff_kv = (config.num_kv_heads == 0) ? config.num_heads : config.num_kv_heads;
    if (v_size != config.vocab_size || h_dim != config.hidden_dim ||
        n_heads != config.num_heads || n_kv != eff_kv ||
        i_dim != config.intermediate_dim ||
        n_layers != config.num_layers || m_seq != config.max_seq_len ||
        (has_ext && (loaded_head_dim != eff_head_dim ||
                     ((loaded_ext_flags & 0x01u) != 0) != config.use_qk_norm ||
                     ((loaded_ext_flags & 0x02u) != 0) != config.attention_bias ||
                     ((loaded_ext_flags & 0x04u) != 0) != config.mlp_bias ||
                     ((loaded_ext_flags & 0x08u) != 0) != config.tie_word_embeddings)))
    {
        throw std::runtime_error("Causal_LM::loadCheckpoint: Architecture dimensions mismatch");
    }

    if (!std::isfinite(loaded_eps) || loaded_eps <= 0.0f ||
        !std::isfinite(loaded_rope) || loaded_rope <= 0.0f)
    {
        throw std::runtime_error("Causal_LM::loadCheckpoint: Invalid rms_norm_eps or rope_base in: " + file_path);
    }

    config.rms_norm_eps = loaded_eps;
    config.rope_base = loaded_rope;
    final_norm.setEpsilon(config.rms_norm_eps);
    for (auto& block : blocks)
    {
        block->setRmsNormEps(config.rms_norm_eps);
        block->setRopeBase(config.rope_base);
    }

    token_embedding.loadCheckpoint(in);
    for (auto& block : blocks)
    {
        block->loadCheckpoint(in);
    }
    final_norm.loadCheckpoint(in);
    if (!config.tie_word_embeddings)
    {
        lm_head.loadCheckpoint(in);
    }
    else
    {
        syncTiedWeights();
    }

    if (in.fail() || in.bad())
    {
        throw std::runtime_error("Causal_LM::loadCheckpoint: Failed reading weights from: " + file_path);
    }

    uint32_t trailer_magic = 0;
    if (in.read(reinterpret_cast<char*>(&trailer_magic), sizeof(trailer_magic)))
    {
        if (trailer_magic != 0x32434C43u)
        {
            throw std::runtime_error("Causal_LM::loadCheckpoint: Invalid trailer magic in: " + file_path);
        }
        uint32_t loaded_dtype_raw = 0;
        uint8_t loaded_tern_head = 0;
        uint8_t loaded_has_scaler = 0;
        float loaded_scale = 1.0f;
        uint32_t loaded_good_steps = 0;
        uint8_t loaded_scaler_enabled = 0;
        in.read(reinterpret_cast<char*>(&loaded_dtype_raw), sizeof(loaded_dtype_raw));
        in.read(reinterpret_cast<char*>(&loaded_tern_head), sizeof(loaded_tern_head));
        in.read(reinterpret_cast<char*>(&loaded_has_scaler), sizeof(loaded_has_scaler));
        in.read(reinterpret_cast<char*>(&loaded_scale), sizeof(loaded_scale));
        in.read(reinterpret_cast<char*>(&loaded_good_steps), sizeof(loaded_good_steps));
        in.read(reinterpret_cast<char*>(&loaded_scaler_enabled), sizeof(loaded_scaler_enabled));

        const Data_Type loaded_dtype = static_cast<Data_Type>(loaded_dtype_raw);
        const bool valid_dtype = (loaded_dtype == Data_Type::FLOAT32 ||
                                  loaded_dtype == Data_Type::FLOAT16 ||
                                  loaded_dtype == Data_Type::TERNARY_INT2);
        if (!in.good() || !valid_dtype || !std::isfinite(loaded_scale) || loaded_scale <= 0.0f)
        {
            throw std::runtime_error("Causal_LM::loadCheckpoint: Corrupt trailer metadata in: " + file_path);
        }

        config.data_type = loaded_dtype;
        config.ternary_lm_head = (loaded_tern_head != 0);
        if (loaded_has_scaler != 0)
        {
            if (!loss_scaler)
            {
                loss_scaler = std::make_unique<Loss_Scaler>(
                    loaded_scale,
                    config.loss_scale_growth_factor,
                    config.loss_scale_backoff_factor,
                    config.loss_scale_growth_interval,
                    loaded_scaler_enabled != 0);
            }
            loss_scaler->setScaleFactor(loaded_scale);
            loss_scaler->setGoodSteps(loaded_good_steps);
            loss_scaler->setEnabled(loaded_scaler_enabled != 0);
        }
        else
        {
            loss_scaler.reset();
        }
    }
    else if (!in.eof() || in.bad())
    {
        throw std::runtime_error("Causal_LM::loadCheckpoint: Stream error reading checkpoint trailer in: " + file_path);
    }
    else
    {
        in.clear();
        config.data_type = !blocks.empty()
            ? blocks[0]->getDataType()
            : (lm_head.isTernary() ? Data_Type::TERNARY_INT2 : lm_head.getDataType());
        config.ternary_lm_head = lm_head.isTernary();
    }

    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    invalidateStaticBufferBindings();
    invalidateWeightCaches();
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        warmupWeightCaches();
    }
    Logger::logMessage(Input_Format{"Causal_LM::loadCheckpoint: Loaded checkpoint from '{}' (dtype={})",
                                    file_path, dataTypeName(config.data_type)},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::CHECKPOINT_IO);
}

void Causal_LM::saveInference(const std::string& file_path) const
{
    ensureParentDirectoryExists(file_path);
    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        Logger::logMessage(Input_Format{"Causal_LM::saveInference: Failed to open file '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::saveInference: Failed to open file: " + file_path);
    }

    uint32_t magic = 0x494D4C43u;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    constexpr uint64_t CAUSAL_LM_GQA_FLAG = (1ULL << 63);
    constexpr uint64_t CAUSAL_LM_EXT_FLAG = (1ULL << 62);
    const size_t eff_kv = (config.num_kv_heads == 0) ? config.num_heads : config.num_kv_heads;
    const size_t eff_head_dim = resolveHeadDim();
    const bool write_gqa = (eff_kv != config.num_heads);
    const bool write_ext = (eff_head_dim != config.hidden_dim / config.num_heads) ||
                           ((config.hidden_dim % config.num_heads) != 0) ||
                           config.use_qk_norm || !config.attention_bias || !config.mlp_bias ||
                           config.tie_word_embeddings || config.residual_scale >= 0.0f ||
                           config.gradient_checkpointing;
    uint64_t v_size = config.vocab_size;
    uint64_t h_dim = config.hidden_dim;
    uint64_t n_heads = static_cast<uint64_t>(config.num_heads)
                     | (write_gqa ? CAUSAL_LM_GQA_FLAG : 0ULL)
                     | (write_ext ? CAUSAL_LM_EXT_FLAG : 0ULL);
    uint64_t i_dim = config.intermediate_dim;
    uint64_t n_layers = config.num_layers;
    uint64_t m_seq = config.max_seq_len;
    out.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
    out.write(reinterpret_cast<const char*>(&h_dim), sizeof(h_dim));
    out.write(reinterpret_cast<const char*>(&n_heads), sizeof(n_heads));
    out.write(reinterpret_cast<const char*>(&i_dim), sizeof(i_dim));
    out.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));
    out.write(reinterpret_cast<const char*>(&m_seq), sizeof(m_seq));
    out.write(reinterpret_cast<const char*>(&config.rms_norm_eps), sizeof(config.rms_norm_eps));
    out.write(reinterpret_cast<const char*>(&config.rope_base), sizeof(config.rope_base));
    if (write_gqa)
    {
        uint64_t n_kv = static_cast<uint64_t>(eff_kv);
        out.write(reinterpret_cast<const char*>(&n_kv), sizeof(n_kv));
    }
    if (write_ext)
    {
        uint64_t h_d = static_cast<uint64_t>(eff_head_dim);
        uint8_t ext_flags = 0u;
        if (config.use_qk_norm) ext_flags |= 0x01u;
        if (config.attention_bias) ext_flags |= 0x02u;
        if (config.mlp_bias) ext_flags |= 0x04u;
        if (config.tie_word_embeddings) ext_flags |= 0x08u;
        if (config.gradient_checkpointing) ext_flags |= 0x10u;
        out.write(reinterpret_cast<const char*>(&h_d), sizeof(h_d));
        out.write(reinterpret_cast<const char*>(&config.residual_scale), sizeof(config.residual_scale));
        out.write(reinterpret_cast<const char*>(&ext_flags), sizeof(ext_flags));
    }

    token_embedding.saveInference(out);
    for (const auto& block : blocks)
    {
        block->saveInference(out);
    }
    final_norm.saveInference(out);
    if (!config.tie_word_embeddings)
    {
        lm_head.saveInference(out);
    }
    out.flush();

    if (!out.good())
    {
        Logger::logMessage(Input_Format{"Causal_LM::saveInference: Failed writing inference file '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::saveInference: Failed writing inference file: " + file_path);
    }
    Logger::logMessage(Input_Format{"Causal_LM::saveInference: Saved inference model to '{}' (dtype={})",
                                    file_path, dataTypeName(config.data_type)},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::CHECKPOINT_IO);
}

void Causal_LM::loadInference(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        Logger::logMessage(Input_Format{"Causal_LM::loadInference: Failed to open file '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::loadInference: Failed to open file: " + file_path);
    }

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!in.good() || magic != 0x494D4C43u)
    {
        Logger::logMessage(Input_Format{"Causal_LM::loadInference: Invalid magic header in '{}'", file_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
        throw std::runtime_error("Causal_LM::loadInference: Invalid magic header in: " + file_path);
    }

    constexpr uint64_t CAUSAL_LM_GQA_FLAG = (1ULL << 63);
    constexpr uint64_t CAUSAL_LM_EXT_FLAG = (1ULL << 62);
    uint64_t v_size = 0, h_dim = 0, n_heads_raw = 0, i_dim = 0, n_layers = 0, m_seq = 0;
    float loaded_eps = 0.0f;
    float loaded_rope = 0.0f;
    in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
    in.read(reinterpret_cast<char*>(&h_dim), sizeof(h_dim));
    in.read(reinterpret_cast<char*>(&n_heads_raw), sizeof(n_heads_raw));
    in.read(reinterpret_cast<char*>(&i_dim), sizeof(i_dim));
    in.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
    in.read(reinterpret_cast<char*>(&m_seq), sizeof(m_seq));
    in.read(reinterpret_cast<char*>(&loaded_eps), sizeof(loaded_eps));
    in.read(reinterpret_cast<char*>(&loaded_rope), sizeof(loaded_rope));
    const bool has_gqa = (n_heads_raw & CAUSAL_LM_GQA_FLAG) != 0;
    const bool has_ext = (n_heads_raw & CAUSAL_LM_EXT_FLAG) != 0;
    const uint64_t n_heads = n_heads_raw & ~(CAUSAL_LM_GQA_FLAG | CAUSAL_LM_EXT_FLAG);
    uint64_t n_kv = n_heads;
    if (has_gqa)
    {
        in.read(reinterpret_cast<char*>(&n_kv), sizeof(n_kv));
    }
    const size_t eff_head_dim = resolveHeadDim();
    uint64_t loaded_head_dim = (config.hidden_dim % config.num_heads == 0) ? (config.hidden_dim / config.num_heads) : 0;
    float loaded_res_scale = -1.0f;
    uint8_t loaded_ext_flags = 0x06u;
    if (has_ext)
    {
        in.read(reinterpret_cast<char*>(&loaded_head_dim), sizeof(loaded_head_dim));
        in.read(reinterpret_cast<char*>(&loaded_res_scale), sizeof(loaded_res_scale));
        in.read(reinterpret_cast<char*>(&loaded_ext_flags), sizeof(loaded_ext_flags));
    }
    if (!in.good())
    {
        throw std::runtime_error("Causal_LM::loadInference: Truncated header in: " + file_path);
    }

    const size_t eff_kv = (config.num_kv_heads == 0) ? config.num_heads : config.num_kv_heads;
    if (v_size != config.vocab_size || h_dim != config.hidden_dim ||
        n_heads != config.num_heads || n_kv != eff_kv ||
        i_dim != config.intermediate_dim ||
        n_layers != config.num_layers || m_seq != config.max_seq_len ||
        (has_ext && (loaded_head_dim != eff_head_dim ||
                     ((loaded_ext_flags & 0x01u) != 0) != config.use_qk_norm ||
                     ((loaded_ext_flags & 0x02u) != 0) != config.attention_bias ||
                     ((loaded_ext_flags & 0x04u) != 0) != config.mlp_bias ||
                     ((loaded_ext_flags & 0x08u) != 0) != config.tie_word_embeddings)))
    {
        throw std::runtime_error("Causal_LM::loadInference: Architecture dimensions mismatch");
    }

    if (!std::isfinite(loaded_eps) || loaded_eps <= 0.0f ||
        !std::isfinite(loaded_rope) || loaded_rope <= 0.0f)
    {
        throw std::runtime_error("Causal_LM::loadInference: Invalid rms_norm_eps or rope_base in: " + file_path);
    }

    config.rms_norm_eps = loaded_eps;
    config.rope_base = loaded_rope;
    final_norm.setEpsilon(config.rms_norm_eps);
    for (auto& block : blocks)
    {
        block->setRmsNormEps(config.rms_norm_eps);
        block->setRopeBase(config.rope_base);
    }

    token_embedding.loadInference(in);
    for (auto& block : blocks)
    {
        block->loadInference(in);
    }
    final_norm.loadInference(in);
    if (!config.tie_word_embeddings)
    {
        lm_head.loadInference(in);
    }
    else
    {
        syncTiedWeights();
    }

    if (in.fail() || in.bad())
    {
        throw std::runtime_error("Causal_LM::loadInference: Failed reading weights from: " + file_path);
    }

    config.data_type = !blocks.empty()
        ? blocks[0]->getDataType()
        : (lm_head.isTernary() ? Data_Type::TERNARY_INT2 : lm_head.getDataType());
    config.ternary_lm_head = lm_head.isTernary();

    setTrainingMode(false);
    invalidateStaticBufferBindings();
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    invalidateWeightCaches();
    if (config.execution_target == Execution_Target::VULKAN_GPU)
    {
        warmupWeightCaches();
    }
    Logger::logMessage(Input_Format{"Causal_LM::loadInference: Loaded inference model from '{}' (dtype={})",
                                    file_path, dataTypeName(config.data_type)},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::CHECKPOINT_IO);
}

bool Causal_LM::loadAuto(const std::string& file_path)
{
    uint32_t magic = 0;
    {
        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open())
        {
            Logger::logMessage(Input_Format{"Causal_LM::loadAuto: Failed to open '{}'", file_path},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
            throw std::runtime_error("Causal_LM::loadAuto: Failed to open file: " + file_path);
        }
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (!in.good())
        {
            Logger::logMessage(Input_Format{"Causal_LM::loadAuto: Failed reading header in '{}'", file_path},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
            throw std::runtime_error("Causal_LM::loadAuto: Failed reading magic header from: " + file_path);
        }
    }

    if (magic == 0x434D4C43u)
    {
        Logger::logMessage(Input_Format{"Causal_LM::loadAuto: Detected checkpoint format in '{}'", file_path},
                           Log_Level::LOG_INFO, true, 0, Log_Feature::CHECKPOINT_IO);
        loadCheckpoint(file_path);
        return true;
    }
    if (magic == 0x494D4C43u)
    {
        Logger::logMessage(Input_Format{"Causal_LM::loadAuto: Detected inference format in '{}'", file_path},
                           Log_Level::LOG_INFO, true, 0, Log_Feature::CHECKPOINT_IO);
        loadInference(file_path);
        return false;
    }

    Logger::logMessage(Input_Format{"Causal_LM::loadAuto: Unknown magic header 0x{:08X} in '{}'", magic, file_path},
                       Log_Level::LOG_ERROR, true, 0, Log_Feature::CHECKPOINT_IO);
    throw std::runtime_error("Causal_LM::loadAuto: Unsupported file magic header in: " + file_path);
}

std::string Causal_LM::generate(const std::string& prompt, const Generate_Config& gen_config)
{
    return generateImpl(
        prompt,
        gen_config.max_new_tokens,
        gen_config.temperature,
        gen_config.top_p,
        gen_config.eos_token_id,
        gen_config.skip_special,
        gen_config.token_callback,
        gen_config.repetition_penalty,
        gen_config.top_k,
        gen_config.stop_sequences,
        gen_config.logits_processor);
}

std::string Causal_LM::generate(const std::string& prompt, size_t max_new_tokens, float temperature, float top_p, int32_t eos_token_id, bool skip_special, const std::function<void(const std::string&)>& token_callback, float repetition_penalty, size_t top_k, const std::vector<std::string>& stop_sequences)
{
    return generateImpl(
        prompt,
        max_new_tokens,
        temperature,
        top_p,
        eos_token_id,
        skip_special,
        token_callback,
        repetition_penalty,
        top_k,
        stop_sequences,
        nullptr);
}

std::string Causal_LM::generateImpl(const std::string& prompt,
                                    size_t max_new_tokens,
                                    float temperature,
                                    float top_p,
                                    int32_t eos_token_id,
                                    bool skip_special,
                                    const std::function<void(const std::string&)>& token_callback,
                                    float repetition_penalty,
                                    size_t top_k,
                                    const std::vector<std::string>& stop_sequences,
                                    const std::function<void(std::vector<float>&, const std::vector<int32_t>&)>& logits_processor)
{
    if (!tokenizer.isLoaded())
    {
        throw std::runtime_error("Causal_LM::generate: Tokenizer is not loaded");
    }

    Static_Graph_Suspend_Guard static_guard(config.execution_target);

    std::vector<int32_t> prompt_tokens = tokenizer.encode(prompt, true);
    if (prompt_tokens.empty())
    {
        return "";
    }

    if (config.max_seq_len > 0 && prompt_tokens.size() > config.max_seq_len)
    {
        Logger::logMessage(Input_Format{"Causal_LM::generate: Prompt truncated from {} to {} tokens (max_seq_len)",
                                        prompt_tokens.size(), config.max_seq_len},
                           Log_Level::LOG_WARNING, true, 0, Log_Feature::LLM_INFERENCE);
        prompt_tokens.erase(prompt_tokens.begin(),
            prompt_tokens.begin() + (prompt_tokens.size() - config.max_seq_len));
    }

    setTrainingMode(false);
    resetKVCaches();

    bool has_any_non_special_in_vocab = false;
    for (size_t id = 0; id < config.vocab_size; ++id)
    {
        if (!tokenizer.isSpecialToken(static_cast<int32_t>(id)))
        {
            has_any_non_special_in_vocab = true;
            break;
        }
    }
    const bool effective_skip_special = skip_special && has_any_non_special_in_vocab;

    std::unordered_set<int32_t> seen_tokens(prompt_tokens.begin(), prompt_tokens.end());
    size_t max_stop_len = 0;
    for (const auto& stop_str : stop_sequences)
    {
        max_stop_len = std::max(max_stop_len, stop_str.size());
    }
    const size_t holdback_len = (max_stop_len > 1) ? (max_stop_len - 1) : 0;

    std::vector<int32_t> generated_ids;
    generated_ids.reserve(max_new_tokens);
    std::vector<int32_t> single_token(1, 0);
    const size_t V = config.vocab_size;

    size_t emitted_len = 0;
    bool hit_stop = false;
    std::string final_decoded;

    for (size_t step = 0; step < max_new_tokens; ++step)
    {
        const std::vector<int32_t>& step_input = (step == 0) ? prompt_tokens : single_token;
        Tensor logits = forward(step_input, true);
        const size_t S_curr = step_input.size();

        Tensor last_row = (S_curr > 1) ? logits.slice(0, S_curr - 1, 1) : logits;
        std::vector<float> last_logits = last_row.getData();
        if (last_logits.size() < V)
        {
            break;
        }
        if (last_logits.size() > V)
        {
            last_logits.resize(V);
        }

        if (repetition_penalty > 1.0f)
        {
            for (int32_t prev_id : seen_tokens)
            {
                if (prev_id >= 0 && static_cast<size_t>(prev_id) < last_logits.size())
                {
                    if (last_logits[prev_id] > 0.0f)
                    {
                        last_logits[prev_id] /= repetition_penalty;
                    }
                    else
                    {
                        last_logits[prev_id] *= repetition_penalty;
                    }
                }
            }
        }

        if (logits_processor)
        {
            logits_processor(last_logits, generated_ids);
        }

        int32_t next_token = sampleToken(last_logits, temperature, top_p, top_k);
        if (next_token == eos_token_id)
        {
            break;
        }

        generated_ids.push_back(next_token);
        seen_tokens.insert(next_token);
        single_token[0] = next_token;

        const bool need_full_decode = token_callback || !stop_sequences.empty();
        if (need_full_decode)
        {
            std::string decoded_so_far = tokenizer.decode(generated_ids, effective_skip_special, true);

            if (!stop_sequences.empty())
            {
                size_t earliest_stop = std::string::npos;
                for (const auto& stop_str : stop_sequences)
                {
                    if (!stop_str.empty())
                    {
                        size_t found_pos = decoded_so_far.find(stop_str);
                        if (found_pos != std::string::npos && found_pos < earliest_stop)
                        {
                            earliest_stop = found_pos;
                        }
                    }
                }
                if (earliest_stop != std::string::npos)
                {
                    decoded_so_far.resize(earliest_stop);
                    final_decoded = std::move(decoded_so_far);
                    hit_stop = true;
                    break;
                }
            }

            if (token_callback)
            {
                const size_t utf8_safe_len = utf8CompletePrefixLength(decoded_so_far);
                size_t emit_end = (utf8_safe_len > holdback_len) ? (utf8_safe_len - holdback_len) : 0;
                emit_end = stepBackToUtf8Boundary(decoded_so_far, emit_end, emitted_len);
                if (emit_end > emitted_len)
                {
                    token_callback(decoded_so_far.substr(emitted_len, emit_end - emitted_len));
                    emitted_len = emit_end;
                }
            }
        }
    }

    std::string res = hit_stop
        ? std::move(final_decoded)
        : tokenizer.decode(generated_ids, effective_skip_special, true);

    if (token_callback && res.size() > emitted_len)
    {
        token_callback(res.substr(emitted_len));
    }
    Logger::logMessage(Input_Format{"Causal_LM::generate: prompt_tokens={}, generated_tokens={}, hit_stop={}",
                                    prompt_tokens.size(), generated_ids.size(), hit_stop ? "true" : "false"},
                       Log_Level::LOG_INFO, true, 0, Log_Feature::LLM_INFERENCE);
    return res;
}

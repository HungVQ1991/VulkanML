#include "layer/transformer_block.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/layer.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "math/tensor.h"
#include "ops/llm/attention.h"
#include "ops/llm/rope.h"
#include "ops/llm/swiglu.h"

namespace
{

constexpr uint32_t TRANSFORMER_BLOCK_LORA_MAGIC = 0x524C4254u;

size_t resolveTransformerHeadDim(size_t _hidden_dim,
                                 size_t _num_heads,
                                 size_t _head_dim,
                                 size_t _intermediate_dim,
                                 float _rms_norm_eps,
                                 float _rope_base,
                                 float _residual_scale)
{
    if (_hidden_dim == 0 || _num_heads == 0 || _intermediate_dim == 0)
    {
        throw std::invalid_argument("Transformer_Block: hidden_dim, num_heads, and intermediate_dim must be > 0");
    }
    size_t resolved_head_dim = _head_dim;
    if (resolved_head_dim == 0)
    {
        if (_hidden_dim % _num_heads != 0)
        {
            throw std::invalid_argument("Transformer_Block: hidden_dim must be divisible by num_heads when head_dim is 0");
        }
        resolved_head_dim = _hidden_dim / _num_heads;
    }
    if (resolved_head_dim == 0 || (resolved_head_dim % 2) != 0)
    {
        throw std::invalid_argument("Transformer_Block: head_dim must be non-zero and even for RoPE");
    }
    if (!std::isfinite(_rms_norm_eps) || _rms_norm_eps <= 0.0f)
    {
        throw std::invalid_argument("Transformer_Block: rms_norm_eps must be finite and > 0");
    }
    if (!std::isfinite(_rope_base) || _rope_base <= 0.0f)
    {
        throw std::invalid_argument("Transformer_Block: rope_base must be finite and > 0");
    }
    if (!std::isfinite(_residual_scale))
    {
        throw std::invalid_argument("Transformer_Block: residual_scale must be finite");
    }
    return resolved_head_dim;
}

size_t resolveTransformerKvHeads(size_t _num_heads, size_t _num_kv_heads)
{
    size_t eff_kv = (_num_kv_heads == 0) ? _num_heads : _num_kv_heads;
    if (_num_heads == 0 || eff_kv == 0 || eff_kv > _num_heads || (_num_heads % eff_kv) != 0)
    {
        throw std::invalid_argument("Transformer_Block: num_heads must be cleanly divisible by num_kv_heads (1 <= num_kv_heads <= num_heads)");
    }
    return eff_kv;
}

Transformer_Block_Config makeLegacyBlockConfig(size_t _hidden_dim,
                                               size_t _num_heads,
                                               size_t _intermediate_dim,
                                               float _rms_norm_eps,
                                               float _rope_base,
                                               Execution_Target _execution_target,
                                               Data_Type _data_type,
                                               size_t _num_layers,
                                               size_t _num_kv_heads,
                                               bool _use_fused_projections)
{
    Transformer_Block_Config cfg;
    cfg.hidden_dim = _hidden_dim;
    cfg.num_heads = _num_heads;
    cfg.head_dim = 0;
    cfg.num_kv_heads = _num_kv_heads;
    cfg.intermediate_dim = _intermediate_dim;
    cfg.rms_norm_eps = _rms_norm_eps;
    cfg.rope_base = _rope_base;
    cfg.execution_target = _execution_target;
    cfg.data_type = _data_type;
    cfg.num_layers = _num_layers;
    cfg.residual_scale = -1.0f;
    cfg.use_fused_projections = _use_fused_projections;
    cfg.use_qk_norm = false;
    cfg.use_attention_bias = true;
    cfg.use_mlp_bias = true;
    cfg.gradient_checkpointing = false;
    return cfg;
}

Tensor makeTensorPreservingDtype(const Shape &shape, std::vector<float> values, Execution_Target target, Data_Type dtype)
{
    Tensor t(shape, std::move(values), target);
    if (dtype == Data_Type::FLOAT16)
    {
        Tensor fp16_t;
        t.to(Data_Type::FLOAT16, fp16_t);
        return fp16_t;
    }
    return t;
}

void mergeLinearColumns(const std::vector<const Linear_Layer *> &srcs,
                        Linear_Layer &dst,
                        size_t in_dim,
                        Execution_Target target,
                        float init_gain,
                        bool mixed_precision,
                        bool copy_grads,
                        bool use_bias)
{
    if (srcs.empty())
    {
        throw std::invalid_argument("mergeLinearColumns: srcs must not be empty");
    }
    size_t total_out = 0;
    std::vector<size_t> col_offsets;
    std::vector<size_t> out_dims;
    std::vector<std::vector<float>> w_vecs;
    std::vector<std::vector<float>> b_vecs;
    std::vector<std::vector<float>> wg_vecs;
    std::vector<std::vector<float>> bg_vecs;
    col_offsets.reserve(srcs.size());
    out_dims.reserve(srcs.size());
    w_vecs.reserve(srcs.size());
    b_vecs.reserve(srcs.size());

    for (const Linear_Layer *src : srcs)
    {
        if (!src)
        {
            throw std::invalid_argument("mergeLinearColumns: null source layer");
        }
        if (src->getInputDimension() != in_dim)
        {
            throw std::runtime_error("mergeLinearColumns: source layer input dimension mismatch");
        }
        const size_t c = src->getOutputDimension();
        col_offsets.push_back(total_out);
        out_dims.push_back(c);
        total_out += c;
        w_vecs.push_back(src->getWeights().getData());
        b_vecs.push_back(src->getBiases().getData());
        if (w_vecs.back().size() != in_dim * c || b_vecs.back().size() != c)
        {
            throw std::runtime_error("mergeLinearColumns: source weight or bias buffer size mismatch");
        }
        if (copy_grads)
        {
            wg_vecs.push_back(src->getWeightsGradient().getData());
            bg_vecs.push_back(src->getBiasesGradient().getData());
            if (wg_vecs.back().size() != in_dim * c || bg_vecs.back().size() != c)
            {
                throw std::runtime_error("mergeLinearColumns: source gradient buffer size mismatch");
            }
        }
    }

    const Data_Type w_dtype = srcs.front()->getWeights().getDataType();
    const Data_Type b_dtype = srcs.front()->getBiases().getDataType();

    std::vector<float> merged_w(in_dim * total_out, 0.0f);
    std::vector<float> merged_b(total_out, 0.0f);
    for (size_t r = 0; r < in_dim; ++r)
    {
        for (size_t s = 0; s < srcs.size(); ++s)
        {
            std::copy_n(w_vecs[s].data() + r * out_dims[s], out_dims[s], merged_w.data() + r * total_out + col_offsets[s]);
        }
    }
    for (size_t s = 0; s < srcs.size(); ++s)
    {
        std::copy_n(b_vecs[s].data(), out_dims[s], merged_b.data() + col_offsets[s]);
    }

    const Data_Type ctor_dtype = (w_dtype == Data_Type::FLOAT16 || mixed_precision) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    dst = Linear_Layer(in_dim, total_out, target, init_gain, ctor_dtype, use_bias);
    dst.setMixedPrecision(mixed_precision || w_dtype == Data_Type::FLOAT16);
    dst.setWeights(makeTensorPreservingDtype(Shape{ in_dim, total_out }, std::move(merged_w), target, w_dtype));
    if (use_bias)
    {
        dst.setBiases(makeTensorPreservingDtype(Shape{ 1, total_out }, std::move(merged_b), target, b_dtype));
    }

    if (copy_grads)
    {
        const Data_Type wg_dtype = srcs.front()->getWeightsGradient().getDataType();
        const Data_Type bg_dtype = srcs.front()->getBiasesGradient().getDataType();
        std::vector<float> merged_wg(in_dim * total_out, 0.0f);
        std::vector<float> merged_bg(total_out, 0.0f);
        for (size_t r = 0; r < in_dim; ++r)
        {
            for (size_t s = 0; s < srcs.size(); ++s)
            {
                std::copy_n(wg_vecs[s].data() + r * out_dims[s], out_dims[s], merged_wg.data() + r * total_out + col_offsets[s]);
            }
        }
        for (size_t s = 0; s < srcs.size(); ++s)
        {
            std::copy_n(bg_vecs[s].data(), out_dims[s], merged_bg.data() + col_offsets[s]);
        }
        dst.setWeightsGradient(makeTensorPreservingDtype(Shape{ in_dim, total_out }, std::move(merged_wg), target, wg_dtype));
        if (use_bias)
        {
            dst.setBiasesGradient(makeTensorPreservingDtype(Shape{ 1, total_out }, std::move(merged_bg), target, bg_dtype));
        }
    }
    dst.invalidateWeightCache();
}

void splitLinearColumns(const Linear_Layer &src,
                        const std::vector<Linear_Layer *> &dsts,
                        const std::vector<size_t> &col_counts,
                        size_t in_dim,
                        Execution_Target target,
                        float init_gain,
                        bool mixed_precision,
                        bool copy_grads,
                        bool use_bias)
{
    const size_t total_out = src.getOutputDimension();
    const Data_Type w_dtype = src.getWeights().getDataType();
    const Data_Type b_dtype = src.getBiases().getDataType();
    const Data_Type wg_dtype = src.getWeightsGradient().getDataType();
    const Data_Type bg_dtype = src.getBiasesGradient().getDataType();

    const std::vector<float> w_src = src.getWeights().getData();
    const std::vector<float> b_src = src.getBiases().getData();
    std::vector<float> wg_src;
    std::vector<float> bg_src;
    if (copy_grads)
    {
        wg_src = src.getWeightsGradient().getData();
        bg_src = src.getBiasesGradient().getData();
    }

    size_t col_off = 0;
    for (size_t s = 0; s < dsts.size(); ++s)
    {
        const size_t c = col_counts[s];
        std::vector<float> sub_w(in_dim * c, 0.0f);
        std::vector<float> sub_b(c, 0.0f);
        for (size_t r = 0; r < in_dim; ++r)
        {
            std::copy_n(w_src.data() + r * total_out + col_off, c, sub_w.data() + r * c);
        }
        std::copy_n(b_src.data() + col_off, c, sub_b.data());

        const Data_Type ctor_dtype = (w_dtype == Data_Type::FLOAT16 || mixed_precision) ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        Linear_Layer &dst = *dsts[s];
        dst = Linear_Layer(in_dim, c, target, init_gain, ctor_dtype, use_bias);
        dst.setMixedPrecision(mixed_precision || w_dtype == Data_Type::FLOAT16);
        dst.setWeights(makeTensorPreservingDtype(Shape{ in_dim, c }, std::move(sub_w), target, w_dtype));
        if (use_bias)
        {
            dst.setBiases(makeTensorPreservingDtype(Shape{ 1, c }, std::move(sub_b), target, b_dtype));
        }

        if (copy_grads)
        {
            std::vector<float> sub_wg(in_dim * c, 0.0f);
            std::vector<float> sub_bg(c, 0.0f);
            for (size_t r = 0; r < in_dim; ++r)
            {
                std::copy_n(wg_src.data() + r * total_out + col_off, c, sub_wg.data() + r * c);
            }
            std::copy_n(bg_src.data() + col_off, c, sub_bg.data());
            dst.setWeightsGradient(makeTensorPreservingDtype(Shape{ in_dim, c }, std::move(sub_wg), target, wg_dtype));
            if (use_bias)
            {
                dst.setBiasesGradient(makeTensorPreservingDtype(Shape{ 1, c }, std::move(sub_bg), target, bg_dtype));
            }
        }
        dst.invalidateWeightCache();
        col_off += c;
    }
}

struct Checkpoint_Recompute_Guard
{
    bool &flag;
    explicit Checkpoint_Recompute_Guard(bool &f) noexcept : flag(f) { flag = true; }
    ~Checkpoint_Recompute_Guard() noexcept { flag = false; }
};

}

Transformer_Block::Transformer_Block(const Transformer_Block_Config &_config)
    : hidden_dim(_config.hidden_dim),
      num_heads(_config.num_heads),
      head_dim(resolveTransformerHeadDim(_config.hidden_dim, _config.num_heads, _config.head_dim, _config.intermediate_dim, _config.rms_norm_eps, _config.rope_base, _config.residual_scale)),
      num_kv_heads(resolveTransformerKvHeads(_config.num_heads, _config.num_kv_heads)),
      intermediate_dim(_config.intermediate_dim),
      rms_norm_eps(_config.rms_norm_eps),
      rope_base(_config.rope_base),
      yarn_config(_config.yarn),
      execution_target(_config.execution_target),
      data_type(_config.data_type),
      num_layers(std::max<size_t>(1, _config.num_layers)),
      residual_scale((_config.residual_scale < 0.0f) ? (1.0f / std::sqrt(2.0f * static_cast<float>(std::max<size_t>(1, _config.num_layers)))) : _config.residual_scale),
      use_fused_projections(_config.use_fused_projections && (_config.data_type != Data_Type::TERNARY_INT2) && !_config.use_qk_norm && !_config.yarn.enabled),
      use_qk_norm(_config.use_qk_norm),
      use_attention_bias(_config.use_attention_bias),
      use_mlp_bias(_config.use_mlp_bias),
      gradient_checkpointing(_config.gradient_checkpointing),
      input_layernorm(_config.hidden_dim, _config.rms_norm_eps, _config.execution_target, _config.data_type),
      q_proj(use_fused_projections ? LoRA_Linear_Layer() : LoRA_Linear_Layer(_config.hidden_dim, _config.num_heads * head_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_attention_bias)),
      k_proj(use_fused_projections ? LoRA_Linear_Layer() : LoRA_Linear_Layer(_config.hidden_dim, num_kv_heads * head_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_attention_bias)),
      v_proj(use_fused_projections ? LoRA_Linear_Layer() : LoRA_Linear_Layer(_config.hidden_dim, num_kv_heads * head_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_attention_bias)),
      qkv_proj(use_fused_projections ? Linear_Layer(_config.hidden_dim, _config.num_heads * head_dim + 2 * num_kv_heads * head_dim, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_attention_bias) : Linear_Layer()),
      o_proj(_config.num_heads * head_dim, _config.hidden_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.num_heads * head_dim), _config.data_type, _config.use_attention_bias),
      q_norm(use_qk_norm ? std::make_unique<RMSNorm_Layer>(head_dim, _config.rms_norm_eps, _config.execution_target, _config.data_type) : nullptr),
      k_norm(use_qk_norm ? std::make_unique<RMSNorm_Layer>(head_dim, _config.rms_norm_eps, _config.execution_target, _config.data_type) : nullptr),
      post_attention_layernorm(_config.hidden_dim, _config.rms_norm_eps, _config.execution_target, _config.data_type),
      gate_proj(use_fused_projections ? LoRA_Linear_Layer() : LoRA_Linear_Layer(_config.hidden_dim, _config.intermediate_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_mlp_bias)),
      up_proj(use_fused_projections ? LoRA_Linear_Layer() : LoRA_Linear_Layer(_config.hidden_dim, _config.intermediate_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_mlp_bias)),
      gate_up_proj(use_fused_projections ? Linear_Layer(_config.hidden_dim, 2 * _config.intermediate_dim, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type, _config.use_mlp_bias) : Linear_Layer()),
      down_proj(_config.intermediate_dim, _config.hidden_dim, 0, 1.0f, 0.0f, _config.execution_target, 0.0004f * static_cast<float>(_config.intermediate_dim), _config.data_type, _config.use_mlp_bias),
      cache_x(_config.execution_target),
      cache_h1(_config.execution_target),
      cache_act(_config.execution_target),
      cache_q_rope(_config.execution_target),
      cache_k_rope(_config.execution_target),
      cache_v(_config.execution_target),
      cache_k_all(_config.execution_target),
      cache_v_all(_config.execution_target),
      cache_attn_out(_config.execution_target),
      cache_l_stats(_config.execution_target),
      cache_ckpt_input(_config.execution_target),
      cache_ckpt_branch(_config.execution_target),
      cache_position_ids(_config.execution_target),
      cache_attn_mask(_config.execution_target),
      cache_attn_kv_index(_config.execution_target),
      cache_scaled_attn(_config.execution_target),
      cache_attn_permuted(_config.execution_target),
      cache_scaled_ffn(_config.execution_target),
      cache_ffn_out(_config.execution_target),
      cache_h2(_config.execution_target),
      cache_dy_branch(_config.execution_target),
      cache_dg(_config.execution_target),
      cache_du(_config.execution_target),
      cache_d_norm_h1(_config.execution_target),
      cache_dh1(_config.execution_target),
      cache_dh1_branch(_config.execution_target),
      cache_d_attn_permuted(_config.execution_target),
      cache_dq(_config.execution_target),
      cache_dk(_config.execution_target),
      cache_dv(_config.execution_target),
      cache_dq_unrope(_config.execution_target),
      cache_dk_unrope(_config.execution_target),
      cache_dv_permuted(_config.execution_target),
      cache_d_norm_qk(_config.execution_target),
      cache_d_norm_attn(_config.execution_target),
      cache_d_gate_up(_config.execution_target),
      cache_dqkv(_config.execution_target),
      cache_dx(_config.execution_target),
      cache_dx_scaled(_config.execution_target)
{
    if (_config.data_type == Data_Type::FLOAT16)
    {
        is_mixed_precision_enabled = true;
    }
    Logger::logMessage(Input_Format{"Transformer_Block: hidden={}, heads={}, head_dim={}, kv_heads={}, inter={}, qk_norm={}, fused_proj={}",
                                    hidden_dim, num_heads, head_dim, num_kv_heads, intermediate_dim, use_qk_norm, use_fused_projections},
                       Log_Level::LOG_DEBUG,
                       false,
                       1,
                       Log_Feature::TRANSFORMER);
}

Transformer_Block::Transformer_Block(size_t _hidden_dim,
                                     size_t _num_heads,
                                     size_t _intermediate_dim,
                                     float _rms_norm_eps,
                                     float _rope_base,
                                     Execution_Target _execution_target,
                                     Data_Type _data_type,
                                     size_t _num_layers,
                                     size_t _num_kv_heads,
                                     bool _use_fused_projections)
    : Transformer_Block(makeLegacyBlockConfig(_hidden_dim, _num_heads, _intermediate_dim, _rms_norm_eps, _rope_base,
                                              _execution_target, _data_type, _num_layers, _num_kv_heads, _use_fused_projections))
{
}

void Transformer_Block::clearIntermediateForwardCaches()
{
    cache_q_rope = Tensor(execution_target);
    cache_k_rope = Tensor(execution_target);
    cache_v = Tensor(execution_target);
    cache_k_all = Tensor(execution_target);
    cache_v_all = Tensor(execution_target);
    cache_attn_out = Tensor(execution_target);
    cache_l_stats = Tensor(execution_target);
    cache_attn_permuted = Tensor(execution_target);
    cache_act = Tensor(execution_target);
}

void Transformer_Block::enableFusedProjections(bool _enable)
{
    if (_enable == use_fused_projections)
    {
        return;
    }
    if (_enable)
    {
        if (isTernary())
        {
            throw std::logic_error("Transformer_Block::enableFusedProjections: Fused projections are not supported in TERNARY_INT2 mode");
        }
        if (use_qk_norm || hasActiveLoRA() || yarn_config.enabled)
        {
            throw std::logic_error("Transformer_Block::enableFusedProjections: Fused projections are not compatible with QK-norm, LoRA, or YaRN");
        }
    }

    const float attn_gain = 0.0004f * static_cast<float>(hidden_dim);
    const size_t q_dim = num_heads * head_dim;
    const size_t kv_hidden = num_kv_heads * head_dim;
    if (_enable)
    {
        mergeLinearColumns({ &q_proj.getBaseLayer(), &k_proj.getBaseLayer(), &v_proj.getBaseLayer() },
                           qkv_proj, hidden_dim, execution_target, attn_gain, is_mixed_precision_enabled, true, use_attention_bias);
        mergeLinearColumns({ &gate_proj.getBaseLayer(), &up_proj.getBaseLayer() },
                           gate_up_proj, hidden_dim, execution_target, attn_gain, is_mixed_precision_enabled, true, use_mlp_bias);
        qkv_proj.setAccumulated(is_accumulated);
        gate_up_proj.setAccumulated(is_accumulated);
        qkv_proj.setTrainingMode(is_training_mode);
        gate_up_proj.setTrainingMode(is_training_mode);
        qkv_proj.setTrainable(is_trainable);
        gate_up_proj.setTrainable(is_trainable);
        q_proj = LoRA_Linear_Layer();
        k_proj = LoRA_Linear_Layer();
        v_proj = LoRA_Linear_Layer();
        gate_proj = LoRA_Linear_Layer();
        up_proj = LoRA_Linear_Layer();
    }
    else
    {
        Linear_Layer tq, tk, tv, tgate, tup;
        splitLinearColumns(qkv_proj, { &tq, &tk, &tv }, { q_dim, kv_hidden, kv_hidden },
                           hidden_dim, execution_target, attn_gain, is_mixed_precision_enabled, true, use_attention_bias);
        splitLinearColumns(gate_up_proj, { &tgate, &tup }, { intermediate_dim, intermediate_dim },
                           hidden_dim, execution_target, attn_gain, is_mixed_precision_enabled, true, use_mlp_bias);
        q_proj = LoRA_Linear_Layer(std::move(tq));
        k_proj = LoRA_Linear_Layer(std::move(tk));
        v_proj = LoRA_Linear_Layer(std::move(tv));
        gate_proj = LoRA_Linear_Layer(std::move(tgate));
        up_proj = LoRA_Linear_Layer(std::move(tup));
        for (LoRA_Linear_Layer *l : { &q_proj, &k_proj, &v_proj, &gate_proj, &up_proj })
        {
            l->setAccumulated(is_accumulated);
            l->setTrainingMode(is_training_mode);
            l->setTrainable(is_trainable);
        }
        qkv_proj = Linear_Layer();
        gate_up_proj = Linear_Layer();
    }
    use_fused_projections = _enable;
    is_forward_completed = false;
    last_forward_used_kv_cache = false;
    cache_is_checkpointed = false;
}

void Transformer_Block::setResidualScale(float scale) noexcept
{
    residual_scale = scale;
}

float Transformer_Block::getResidualScale() const noexcept
{
    return residual_scale;
}

void Transformer_Block::setRmsNormEps(float _eps)
{
    if (!std::isfinite(_eps) || _eps <= 0.0f)
    {
        throw std::invalid_argument("Transformer_Block: rms_norm_eps must be finite and > 0");
    }
    rms_norm_eps = _eps;
    forEachNorm([_eps](RMSNorm_Layer &norm) { norm.setEpsilon(_eps); });
}

void Transformer_Block::setRopeBase(float _rope_base)
{
    if (!std::isfinite(_rope_base) || _rope_base <= 0.0f)
    {
        throw std::invalid_argument("Transformer_Block: rope_base must be finite and > 0");
    }
    rope_base = _rope_base;
}

void Transformer_Block::enableLoRA(size_t _rank,
                                   float _alpha,
                                   float _dropout_rate,
                                   const std::vector<std::string> &_target_modules,
                                   bool _freeze_base)
{
    if (_rank == 0)
    {
        throw std::invalid_argument("Transformer_Block::enableLoRA: rank must be > 0");
    }
    if (isTernary())
    {
        throw std::logic_error("Transformer_Block::enableLoRA: LoRA cannot be enabled on TERNARY_INT2 block");
    }
    for (const std::string &name : _target_modules)
    {
        if (name != "q_proj" && name != "k_proj" && name != "v_proj" && name != "o_proj" &&
            name != "gate_proj" && name != "up_proj" && name != "down_proj")
        {
            throw std::invalid_argument("Transformer_Block::enableLoRA: Unknown target module '" + name + "'");
        }
    }
    if (use_fused_projections)
    {
        enableFusedProjections(false);
    }
    auto apply_module = [&](const std::string &mod_name, LoRA_Linear_Layer &layer) {
        const bool targeted = std::find(_target_modules.begin(), _target_modules.end(), mod_name) != _target_modules.end();
        if (targeted)
        {
            layer.enableLoRA(_rank, _alpha, _dropout_rate, _freeze_base);
        }
        else
        {
            layer.disableLoRA();
            if (_freeze_base)
            {
                layer.setBaseTrainable(false);
            }
        }
    };
    apply_module("q_proj", q_proj);
    apply_module("k_proj", k_proj);
    apply_module("v_proj", v_proj);
    apply_module("o_proj", o_proj);
    apply_module("gate_proj", gate_proj);
    apply_module("up_proj", up_proj);
    apply_module("down_proj", down_proj);

    if (_freeze_base)
    {
        forEachNorm([](RMSNorm_Layer &norm) { norm.setTrainable(false); });
    }
}

void Transformer_Block::disableLoRA()
{
    forEachLoRAProjection([this](LoRA_Linear_Layer &l) {
        l.disableLoRA();
        l.setBaseTrainable(is_trainable);
    });
    if (use_fused_projections)
    {
        qkv_proj.setTrainable(is_trainable);
        gate_up_proj.setTrainable(is_trainable);
    }
    forEachNorm([this](RMSNorm_Layer &norm) { norm.setTrainable(is_trainable); });
}

void Transformer_Block::mergeLoRA()
{
    forEachLoRAProjection([](LoRA_Linear_Layer &l) { l.mergeWeights(); });
}

void Transformer_Block::unmergeLoRA()
{
    forEachLoRAProjection([](LoRA_Linear_Layer &l) { l.unmergeWeights(); });
}

bool Transformer_Block::hasActiveLoRA() const noexcept
{
    if (o_proj.isLoRAEnabled() || down_proj.isLoRAEnabled())
    {
        return true;
    }
    if (!use_fused_projections)
    {
        return q_proj.isLoRAEnabled() || k_proj.isLoRAEnabled() || v_proj.isLoRAEnabled() ||
               gate_proj.isLoRAEnabled() || up_proj.isLoRAEnabled();
    }
    return false;
}

void Transformer_Block::setBaseTrainable(bool _trainable) noexcept
{
    forEachNorm([_trainable](RMSNorm_Layer &norm) { norm.setTrainable(_trainable); });
    forEachLoRAProjection([_trainable](LoRA_Linear_Layer &l) { l.setBaseTrainable(_trainable); });
    if (use_fused_projections)
    {
        qkv_proj.setTrainable(_trainable);
        gate_up_proj.setTrainable(_trainable);
    }
}

void Transformer_Block::setKVCache(KV_Cache_Manager *cache) noexcept
{
    active_kv_cache = cache;
}

KV_Cache_Manager *Transformer_Block::getKVCache() const noexcept
{
    return active_kv_cache;
}

void Transformer_Block::setAccumulated(bool _is_accumulated) noexcept
{
    is_accumulated = _is_accumulated;
    forEachNorm([_is_accumulated](RMSNorm_Layer &norm) { norm.setAccumulated(_is_accumulated); });
    forEachActiveProjection([_is_accumulated](ILayer &proj) { proj.setAccumulated(_is_accumulated); });
}

void Transformer_Block::setMixedPrecision(bool _enable) noexcept
{
    ILayer::setMixedPrecision(_enable);
    if (!isTernary())
    {
        data_type = _enable ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    }
    forEachNorm([_enable](RMSNorm_Layer &norm) { norm.setMixedPrecision(_enable); });
    forEachActiveProjection([_enable](ILayer &proj) { proj.setMixedPrecision(_enable); });
}

void Transformer_Block::setTrainable(bool _trainable) noexcept
{
    ILayer::setTrainable(_trainable);
    if (!hasActiveLoRA())
    {
        forEachNorm([_trainable](RMSNorm_Layer &norm) { norm.setTrainable(_trainable); });
    }
    forEachActiveProjection([_trainable](ILayer &proj) { proj.setTrainable(_trainable); });
}

void Transformer_Block::invalidateWeightCache() noexcept
{
    forEachNorm([](RMSNorm_Layer &norm) { norm.invalidateWeightCache(); });
    forEachActiveProjection([](ILayer &proj) { proj.invalidateWeightCache(); });
}

Tensor Transformer_Block::forward(const Tensor &_input_tensor)
{
    return forward(_input_tensor, active_kv_cache, nullptr, nullptr);
}

Tensor Transformer_Block::forward(const Tensor &x, KV_Cache_Manager *kv_cache)
{
    return executeForwardCore(x, nullptr, residual_scale, kv_cache, true, 0, 0, nullptr, nullptr).first;
}

Tensor Transformer_Block::forward(const Tensor &x,
                                  KV_Cache_Manager *kv_cache,
                                  const Tensor *position_ids,
                                  const ops::llm::Attention_Config *attn_override)
{
    return executeForwardCore(x, nullptr, residual_scale, kv_cache, true, 0, 0, position_ids, attn_override).first;
}

std::pair<Tensor, Tensor> Transformer_Block::forwardResidualPair(const Tensor &_residual,
                                                                 const Tensor &_branch,
                                                                 float _branch_scale,
                                                                 KV_Cache_Manager *_kv_cache,
                                                                 bool _fuse_incoming,
                                                                 size_t _batch_size,
                                                                 size_t _seq_len,
                                                                 const Tensor *_position_ids,
                                                                 const ops::llm::Attention_Config *_attn_override)
{
    return executeForwardCore(_residual, _fuse_incoming ? &_branch : nullptr, _branch_scale,
                              _kv_cache, false, _batch_size, _seq_len, _position_ids, _attn_override);
}

std::pair<Tensor, Tensor> Transformer_Block::executeForwardCore(const Tensor &x,
                                                                  const Tensor *branch_in,
                                                                  float branch_scale,
                                                                  KV_Cache_Manager *kv_cache,
                                                                  bool fuse_tail,
                                                                  size_t override_batch,
                                                                  size_t override_seq,
                                                                  const Tensor *position_ids,
                                                                  const ops::llm::Attention_Config *attn_override)
{
    if (!x.getImplementation())
    {
        throw std::invalid_argument("Transformer_Block::forward: Input tensor has null implementation");
    }
    if (x.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Transformer_Block::forward: TERNARY_INT2 input is not supported");
    }
    if (x.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Transformer_Block::forward: Execution_Target mismatch");
    }
    const Shape &in_shape = x.getShape();
    const size_t rank = in_shape.getRank();
    if (rank < 1 || x.getTotalElements() == 0 || in_shape[rank - 1] != hidden_dim)
    {
        throw std::invalid_argument("Transformer_Block::forward: Input must have rank >= 1, non-zero elements, and last dimension equal to hidden_dim");
    }

    const bool has_custom_attn = (attn_override != nullptr) &&
        (attn_override->mask_mode != ops::llm::Attention_Mask_Mode::NONE ||
         attn_override->attn_mask != nullptr ||
         attn_override->kv_index != nullptr ||
         !attn_override->is_causal ||
         attn_override->causal_offset != 0 ||
         attn_override->scale != 0.0f ||
         attn_override->kv_seq_len != 0);
    if (use_fused_projections &&
        (use_qk_norm || hasActiveLoRA() || position_ids != nullptr || yarn_config.enabled || has_custom_attn))
    {
        enableFusedProjections(false);
    }

    size_t B = 1;
    size_t S = 1;
    bool is_input_3d = (rank == 3);
    if (override_batch > 0 && override_seq > 0 && override_batch * override_seq * hidden_dim == x.getTotalElements())
    {
        B = override_batch;
        S = override_seq;
    }
    else if (is_input_3d)
    {
        B = in_shape[0];
        S = in_shape[1];
    }
    else if (rank == 2)
    {
        B = 1;
        S = in_shape[0];
    }
    else
    {
        size_t total_elements = x.getTotalElements();
        S = total_elements / hidden_dim;
        B = 1;
    }
    cache_batch_size = B;
    cache_seq_len = S;
    cache_is_3d = is_input_3d;

    if (cache_l_stats.getExecutionTarget() != execution_target)
    {
        cache_l_stats = Tensor(execution_target);
    }

    const bool should_checkpoint = gradient_checkpointing && is_training_mode && (kv_cache == nullptr) && !checkpoint_recompute_active;
    if (should_checkpoint)
    {
        cache_ckpt_input = x;
        cache_ckpt_has_branch = (branch_in != nullptr);
        cache_ckpt_branch = (branch_in != nullptr) ? *branch_in : Tensor(execution_target);
        cache_ckpt_branch_scale = branch_scale;
        cache_ckpt_fuse_tail = fuse_tail;
    }

    cache_has_position_ids = (position_ids != nullptr);
    if (position_ids != nullptr)
    {
        cache_position_ids = *position_ids;
    }
    cache_has_attn_override = (attn_override != nullptr);
    if (attn_override != nullptr)
    {
        cache_attn_mask_mode = attn_override->mask_mode;
        cache_attn_is_causal = attn_override->is_causal;
        cache_attn_causal_offset = attn_override->causal_offset;
        cache_attn_scale = attn_override->scale;
        cache_attn_has_mask = (attn_override->attn_mask != nullptr);
        if (cache_attn_has_mask)
        {
            cache_attn_mask = *attn_override->attn_mask;
        }
        cache_attn_has_kv_index = (attn_override->kv_index != nullptr);
        if (cache_attn_has_kv_index)
        {
            cache_attn_kv_index = *attn_override->kv_index;
        }
    }

    Tensor x_2d = (x.getShape().getRank() != 2 || x.getColumns() != hidden_dim)
        ? x.view(Shape{ B * S, hidden_dim })
        : x;

    Tensor norm_x;
    if (branch_in != nullptr)
    {
        Tensor br_2d = (branch_in->getShape().getRank() != 2 || branch_in->getColumns() != hidden_dim)
            ? branch_in->view(Shape{ B * S, hidden_dim })
            : *branch_in;
        norm_x = input_layernorm.forwardResidualAdd(x_2d, br_2d, branch_scale, cache_x);
        x_2d = cache_x;
    }
    else
    {
        cache_x = x;
        norm_x = input_layernorm.forward(x_2d);
    }
    if (kv_cache != nullptr && !kv_cache->isAutoGrowEnabled() && S > kv_cache->getMaxSeqLen())
    {
        throw std::invalid_argument("Transformer_Block::forward: Sequence length exceeds KV_Cache_Manager max_seq_len");
    }
    uint32_t past_len = (kv_cache != nullptr) ? static_cast<uint32_t>(kv_cache->getTotalTokensSeen()) : 0u;
    const size_t q_dim = num_heads * head_dim;
    Tensor attn_proj;

    if (use_fused_projections)
    {
        Tensor qkv = qkv_proj.forward(norm_x);
        ops::llm::qkvSplitRopeForward(qkv, cache_q_rope, cache_k_rope, cache_v,
                                      static_cast<uint32_t>(B), static_cast<uint32_t>(S),
                                      static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_kv_heads),
                                      static_cast<uint32_t>(head_dim), rope_base,
                                      (kv_cache != nullptr), past_len);
        if (kv_cache != nullptr)
        {
            if (kv_cache->getNumHeads() != num_kv_heads || kv_cache->getHeadDim() != head_dim)
            {
                throw std::invalid_argument("Transformer_Block::forward: KV_Cache_Manager head dimensions do not match num_kv_heads / head_dim");
            }
            kv_cache->append(cache_k_rope, cache_v);
            kv_cache->getK().contiguous(cache_k_all);
            kv_cache->getV().contiguous(cache_v_all);
            size_t T = kv_cache->getCurrentSeqLen();

            if (S > 1)
            {
                uint32_t eff_past_len = (T >= S) ? static_cast<uint32_t>(T - S) : 0u;
                cache_q_rope.flashAttentionForward(cache_k_all, cache_v_all, cache_attn_out, num_heads,
                                                   static_cast<uint32_t>(S), static_cast<uint32_t>(head_dim),
                                                   true, 0.0f, &cache_l_stats,
                                                   static_cast<uint32_t>(T), eff_past_len,
                                                   static_cast<uint32_t>(num_kv_heads));
            }
            else
            {
                cache_q_rope.singleTokenAttentionForward(cache_k_all, cache_v_all, cache_attn_out, num_heads, head_dim, T, num_kv_heads);
            }

            cache_attn_out.permute({ 0, 2, 1, 3 }).contiguous(cache_attn_permuted);
            cache_attn_permuted.reshape(Shape{ B * S, q_dim });
            attn_proj = o_proj.forward(cache_attn_permuted);
        }
        else
        {
            ops::llm::Attention_Config attn_cfg{};
            attn_cfg.num_heads = static_cast<uint32_t>(num_heads);
            attn_cfg.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            attn_cfg.seq_len = static_cast<uint32_t>(S);
            attn_cfg.head_dim = static_cast<uint32_t>(head_dim);
            attn_cfg.is_causal = true;
            attn_cfg.is_bshd = true;
            ops::llm::flashAttentionForward(cache_q_rope, cache_k_rope, cache_v, cache_attn_out, attn_cfg, &cache_l_stats);

            cache_attn_permuted = cache_attn_out.view(Shape{ B * S, q_dim });
            attn_proj = o_proj.forward(cache_attn_permuted);
        }
    }
    else
    {
        Tensor q = q_proj.forward(norm_x);
        Tensor k = k_proj.forward(norm_x);
        Tensor v = v_proj.forward(norm_x);

        q.reshape(Shape{ B, S, num_heads, head_dim });
        k.reshape(Shape{ B, S, num_kv_heads, head_dim });
        v.reshape(Shape{ B, S, num_kv_heads, head_dim });

        if (use_qk_norm)
        {
            q = q_norm->forward(q);
            k = k_norm->forward(k);
        }

        ops::llm::RoPE_Config rope_q_cfg{};
        rope_q_cfg.seq_len = static_cast<uint32_t>(S);
        rope_q_cfg.head_dim = static_cast<uint32_t>(head_dim);
        rope_q_cfg.direction = ops::llm::RoPE_Direction::FORWARD;
        rope_q_cfg.base = rope_base;
        rope_q_cfg.num_heads = static_cast<uint32_t>(num_heads);
        rope_q_cfg.mode = (kv_cache != nullptr) ? ops::llm::RoPE_Mode::FUSED_PERMUTE : ops::llm::RoPE_Mode::BSHD_INPLACE;
        rope_q_cfg.position_offset = past_len;
        rope_q_cfg.position_ids = position_ids;
        rope_q_cfg.yarn = yarn_config;

        ops::llm::RoPE_Config rope_k_cfg = rope_q_cfg;
        rope_k_cfg.num_heads = static_cast<uint32_t>(num_kv_heads);

        ops::llm::applyRoPE(q, cache_q_rope, rope_q_cfg);
        ops::llm::applyRoPE(k, cache_k_rope, rope_k_cfg);

        if (kv_cache != nullptr)
        {
            v.permute({ 0, 2, 1, 3 }).contiguous(cache_v);
            if (kv_cache->getNumHeads() != num_kv_heads || kv_cache->getHeadDim() != head_dim)
            {
                throw std::invalid_argument("Transformer_Block::forward: KV_Cache_Manager head dimensions do not match num_kv_heads / head_dim");
            }
            kv_cache->append(cache_k_rope, cache_v);
            kv_cache->getK().contiguous(cache_k_all);
            kv_cache->getV().contiguous(cache_v_all);
            size_t T = kv_cache->getCurrentSeqLen();

            if (attn_override != nullptr)
            {
                ops::llm::Attention_Config attn_cfg = *attn_override;
                attn_cfg.num_heads = static_cast<uint32_t>(num_heads);
                attn_cfg.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
                attn_cfg.seq_len = static_cast<uint32_t>(S);
                attn_cfg.kv_seq_len = static_cast<uint32_t>(T);
                attn_cfg.head_dim = static_cast<uint32_t>(head_dim);
                attn_cfg.is_bshd = false;
                if (attn_cfg.mask_mode == ops::llm::Attention_Mask_Mode::NONE && attn_cfg.is_causal && attn_cfg.causal_offset == 0 && T >= S)
                {
                    attn_cfg.causal_offset = static_cast<uint32_t>(T - S);
                }
                ops::llm::flashAttentionForward(cache_q_rope, cache_k_all, cache_v_all, cache_attn_out, attn_cfg, &cache_l_stats);
            }
            else if (S > 1)
            {
                uint32_t eff_past_len = (T >= S) ? static_cast<uint32_t>(T - S) : 0u;
                cache_q_rope.flashAttentionForward(cache_k_all, cache_v_all, cache_attn_out, num_heads,
                                                   static_cast<uint32_t>(S), static_cast<uint32_t>(head_dim),
                                                   true, 0.0f, &cache_l_stats,
                                                   static_cast<uint32_t>(T), eff_past_len,
                                                   static_cast<uint32_t>(num_kv_heads));
            }
            else
            {
                cache_q_rope.singleTokenAttentionForward(cache_k_all, cache_v_all, cache_attn_out, num_heads, head_dim, T, num_kv_heads);
            }

            cache_attn_out.permute({ 0, 2, 1, 3 }).contiguous(cache_attn_permuted);
            cache_attn_permuted.reshape(Shape{ B * S, q_dim });
            attn_proj = o_proj.forward(cache_attn_permuted);
        }
        else
        {
            cache_v = v.view(Shape{ B, S, num_kv_heads, head_dim });
            ops::llm::Attention_Config attn_cfg = (attn_override != nullptr) ? *attn_override : ops::llm::Attention_Config{};
            attn_cfg.num_heads = static_cast<uint32_t>(num_heads);
            attn_cfg.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            attn_cfg.seq_len = static_cast<uint32_t>(S);
            attn_cfg.head_dim = static_cast<uint32_t>(head_dim);
            attn_cfg.is_bshd = true;
            if (attn_override == nullptr)
            {
                attn_cfg.is_causal = true;
            }
            ops::llm::flashAttentionForward(cache_q_rope, cache_k_rope, cache_v, cache_attn_out, attn_cfg, &cache_l_stats);

            cache_attn_permuted = cache_attn_out.view(Shape{ B * S, q_dim });
            attn_proj = o_proj.forward(cache_attn_permuted);
        }
    }

    Tensor norm_h1 = post_attention_layernorm.forwardResidualAdd(x_2d, attn_proj, residual_scale, cache_h1);
    if (use_fused_projections)
    {
        Tensor gate_up = gate_up_proj.forward(norm_h1);
        ops::llm::fusedSwiGLUForward(gate_up, cache_act);
    }
    else
    {
        Tensor g = gate_proj.forward(norm_h1);
        Tensor u = up_proj.forward(norm_h1);
        g.swigluForward(u, cache_act);
    }
    Tensor ffn_out = down_proj.forward(cache_act);
    is_forward_completed = true;
    last_forward_used_kv_cache = (kv_cache != nullptr);

    if (!fuse_tail)
    {
        cache_ffn_out = ffn_out;
        if (should_checkpoint)
        {
            cache_is_checkpointed = true;
            clearIntermediateForwardCaches();
        }
        return { cache_h1, cache_ffn_out };
    }

    Tensor &scaled_ffn_buf = shared_scratchpad ? shared_scratchpad->fwd_scaled : cache_scaled_ffn;
    ffn_out.mulScalar(residual_scale, scaled_ffn_buf);
    cache_h1.add(scaled_ffn_buf, cache_h2);
    if (is_input_3d)
    {
        cache_h2.reshape(Shape{ B, S, hidden_dim });
    }
    if (should_checkpoint)
    {
        cache_is_checkpointed = true;
        clearIntermediateForwardCaches();
    }
    return { cache_h2, Tensor() };
}

Tensor Transformer_Block::backward(const Tensor &_output_gradient)
{
    return executeBackwardCore(_output_gradient, nullptr, false).first;
}

std::pair<Tensor, Tensor> Transformer_Block::backwardResidualChained(const Tensor &_output_gradient,
                                                                     const Tensor *_prescaled_branch_grad,
                                                                     bool _produce_scaled_dx)
{
    return executeBackwardCore(_output_gradient, _prescaled_branch_grad, _produce_scaled_dx);
}

std::pair<Tensor, Tensor> Transformer_Block::executeBackwardCore(const Tensor &_output_gradient,
                                                                   const Tensor *_prescaled_branch_grad,
                                                                   bool _produce_scaled_dx)
{
    if (!is_forward_completed)
    {
        throw std::logic_error("Transformer_Block::backward: Backward called before forward");
    }
    if (last_forward_used_kv_cache)
    {
        throw std::logic_error("Transformer_Block::backward cannot be called after KV-cached forward");
    }
    if (!_output_gradient.getImplementation())
    {
        throw std::invalid_argument("Transformer_Block::backward: Output gradient has null implementation");
    }
    if (_output_gradient.getDataType() == Data_Type::TERNARY_INT2)
    {
        throw std::invalid_argument("Transformer_Block::backward: TERNARY_INT2 gradient is not supported");
    }
    if (_output_gradient.getExecutionTarget() != execution_target)
    {
        throw std::invalid_argument("Transformer_Block::backward: Execution_Target mismatch");
    }

    size_t B = cache_batch_size;
    size_t S = cache_seq_len;
    const Shape &grad_shape = _output_gradient.getShape();
    const bool valid_grad_shape =
        (grad_shape == cache_h2.getShape()) ||
        (grad_shape.getRank() == 2 && grad_shape[0] == B * S && grad_shape[1] == hidden_dim) ||
        (grad_shape.getRank() == 3 && grad_shape[0] == B && grad_shape[1] == S && grad_shape[2] == hidden_dim);
    if (!valid_grad_shape)
    {
        throw std::invalid_argument("Transformer_Block::backward: Output gradient shape mismatch");
    }

    if (cache_is_checkpointed)
    {
        Checkpoint_Recompute_Guard guard(checkpoint_recompute_active);
        ops::llm::Attention_Config rebuilt_override{};
        const ops::llm::Attention_Config *override_ptr = nullptr;
        if (cache_has_attn_override)
        {
            rebuilt_override.mask_mode = cache_attn_mask_mode;
            rebuilt_override.is_causal = cache_attn_is_causal;
            rebuilt_override.causal_offset = cache_attn_causal_offset;
            rebuilt_override.scale = cache_attn_scale;
            rebuilt_override.attn_mask = cache_attn_has_mask ? &cache_attn_mask : nullptr;
            rebuilt_override.kv_index = cache_attn_has_kv_index ? &cache_attn_kv_index : nullptr;
            override_ptr = &rebuilt_override;
        }
        executeForwardCore(cache_ckpt_input,
                           cache_ckpt_has_branch ? &cache_ckpt_branch : nullptr,
                           cache_ckpt_branch_scale,
                           nullptr,
                           cache_ckpt_fuse_tail,
                           B,
                           S,
                           cache_has_position_ids ? &cache_position_ids : nullptr,
                           override_ptr);
        cache_is_checkpointed = false;
    }

    Tensor dy = (_output_gradient.getShape().getRank() != 2 || _output_gradient.getColumns() != hidden_dim)
        ? _output_gradient.view(Shape{ B * S, hidden_dim })
        : _output_gradient;

    Tensor &dy_branch_buf = shared_scratchpad ? shared_scratchpad->dy_branch : cache_dy_branch;
    Tensor &dg_buf = shared_scratchpad ? shared_scratchpad->dg : cache_dg;
    Tensor &du_buf = shared_scratchpad ? shared_scratchpad->du : cache_du;
    Tensor &d_norm_h1_buf = shared_scratchpad ? shared_scratchpad->d_norm_h1 : cache_d_norm_h1;
    Tensor &dh1_buf = shared_scratchpad ? shared_scratchpad->dh1 : cache_dh1;
    Tensor &dh1_branch_buf = shared_scratchpad ? shared_scratchpad->dh1_branch : cache_dh1_branch;
    Tensor &dq_buf = shared_scratchpad ? shared_scratchpad->dq : cache_dq;
    Tensor &dk_buf = shared_scratchpad ? shared_scratchpad->dk : cache_dk;
    Tensor &dv_buf = shared_scratchpad ? shared_scratchpad->dv : cache_dv;
    Tensor &dq_unrope_buf = shared_scratchpad ? shared_scratchpad->dq_unrope : cache_dq_unrope;
    Tensor &dk_unrope_buf = shared_scratchpad ? shared_scratchpad->dk_unrope : cache_dk_unrope;
    Tensor &dv_permuted_buf = shared_scratchpad ? shared_scratchpad->dv_permuted : cache_dv_permuted;
    Tensor &d_norm_qk_buf = shared_scratchpad ? shared_scratchpad->d_norm_qk : cache_d_norm_qk;
    Tensor &d_norm_attn_buf = shared_scratchpad ? shared_scratchpad->d_norm_attn : cache_d_norm_attn;
    Tensor &dx_out_buf = shared_scratchpad ? shared_scratchpad->dx_slots[scratch_dx_slot] : cache_dx;
    Tensor &dx_scaled_out_buf = shared_scratchpad ? shared_scratchpad->dx_scaled_slots[scratch_dx_slot] : cache_dx_scaled;

    const Tensor *dy_branch_ptr = &dy_branch_buf;
    if (_prescaled_branch_grad != nullptr)
    {
        dy_branch_ptr = _prescaled_branch_grad;
    }
    else
    {
        dy.mulScalar(residual_scale, dy_branch_buf);
    }
    Tensor d_act = down_proj.backward(*dy_branch_ptr);

    Tensor d_norm_h1_local;
    const Tensor *d_norm_h1_ptr = &d_norm_h1_buf;
    if (use_fused_projections)
    {
        Tensor &d_gate_up_buf = shared_scratchpad ? shared_scratchpad->d_gate_up : cache_d_gate_up;
        ops::llm::fusedSwiGLUBackward(gate_up_proj.getOutput(), d_act, d_gate_up_buf);
        d_norm_h1_local = gate_up_proj.backward(d_gate_up_buf);
        d_norm_h1_ptr = &d_norm_h1_local;
    }
    else
    {
        Tensor g = gate_proj.getOutput();
        Tensor u = up_proj.getOutput();
        g.swigluBackward(d_act, u, dg_buf, du_buf);

        Tensor d_norm_gate = gate_proj.backward(dg_buf);
        Tensor d_norm_up = up_proj.backward(du_buf);
        d_norm_gate.add(d_norm_up, d_norm_h1_buf);
    }

    post_attention_layernorm.backwardResidualAdd(*d_norm_h1_ptr, dy, dh1_buf, &dh1_branch_buf, residual_scale);

    Tensor d_attn_proj = o_proj.backward(dh1_branch_buf);
    d_attn_proj.reshape(Shape{ B, S, num_heads, head_dim });
    const size_t q_dim = num_heads * head_dim;
    const size_t kv_hidden_dim = num_kv_heads * head_dim;

    ops::llm::Attention_Config attn_bwd_cfg{};
    if (cache_has_attn_override)
    {
        attn_bwd_cfg.mask_mode = cache_attn_mask_mode;
        attn_bwd_cfg.is_causal = cache_attn_is_causal;
        attn_bwd_cfg.causal_offset = cache_attn_causal_offset;
        attn_bwd_cfg.scale = cache_attn_scale;
        attn_bwd_cfg.attn_mask = cache_attn_has_mask ? &cache_attn_mask : nullptr;
        attn_bwd_cfg.kv_index = cache_attn_has_kv_index ? &cache_attn_kv_index : nullptr;
    }
    else
    {
        attn_bwd_cfg.is_causal = true;
    }
    attn_bwd_cfg.num_heads = static_cast<uint32_t>(num_heads);
    attn_bwd_cfg.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    attn_bwd_cfg.seq_len = static_cast<uint32_t>(S);
    attn_bwd_cfg.head_dim = static_cast<uint32_t>(head_dim);
    attn_bwd_cfg.is_bshd = true;
    ops::llm::flashAttentionBackward(cache_q_rope, cache_k_rope, cache_v, cache_attn_out, d_attn_proj,
                                     dq_buf, dk_buf, dv_buf, attn_bwd_cfg, &cache_l_stats);

    Tensor d_norm_attn_local;
    const Tensor *d_norm_attn_ptr = &d_norm_attn_buf;
    if (use_fused_projections)
    {
        Tensor &dqkv_buf = shared_scratchpad ? shared_scratchpad->dqkv : cache_dqkv;
        ops::llm::qkvPackUnropeBackward(dq_buf, dk_buf, dv_buf, dqkv_buf,
                                        static_cast<uint32_t>(B), static_cast<uint32_t>(S),
                                        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_kv_heads),
                                        static_cast<uint32_t>(head_dim), rope_base, false, 0u);
        d_norm_attn_local = qkv_proj.backward(dqkv_buf);
        d_norm_attn_ptr = &d_norm_attn_local;
    }
    else
    {
        ops::llm::RoPE_Config rope_bwd_q{};
        rope_bwd_q.seq_len = static_cast<uint32_t>(S);
        rope_bwd_q.head_dim = static_cast<uint32_t>(head_dim);
        rope_bwd_q.direction = ops::llm::RoPE_Direction::BACKWARD;
        rope_bwd_q.base = rope_base;
        rope_bwd_q.num_heads = static_cast<uint32_t>(num_heads);
        rope_bwd_q.mode = ops::llm::RoPE_Mode::BSHD_INPLACE;
        rope_bwd_q.position_ids = cache_has_position_ids ? &cache_position_ids : nullptr;
        rope_bwd_q.yarn = yarn_config;

        ops::llm::RoPE_Config rope_bwd_k = rope_bwd_q;
        rope_bwd_k.num_heads = static_cast<uint32_t>(num_kv_heads);

        ops::llm::applyRoPE(dq_buf, dq_unrope_buf, rope_bwd_q);
        ops::llm::applyRoPE(dk_buf, dk_unrope_buf, rope_bwd_k);

        if (use_qk_norm)
        {
            dq_unrope_buf = q_norm->backward(dq_unrope_buf);
            dk_unrope_buf = k_norm->backward(dk_unrope_buf);
        }

        dq_unrope_buf.reshape(Shape{ B * S, q_dim });
        dk_unrope_buf.reshape(Shape{ B * S, kv_hidden_dim });
        dv_permuted_buf = dv_buf.view(Shape{ B * S, kv_hidden_dim });

        Tensor d_norm_q = q_proj.backward(dq_unrope_buf);
        Tensor d_norm_k = k_proj.backward(dk_unrope_buf);
        Tensor d_norm_v = v_proj.backward(dv_permuted_buf);
        d_norm_q.add(d_norm_k, d_norm_qk_buf);
        d_norm_qk_buf.add(d_norm_v, d_norm_attn_buf);
    }

    input_layernorm.backwardResidualAdd(*d_norm_attn_ptr, dh1_buf, dx_out_buf,
                                        _produce_scaled_dx ? &dx_scaled_out_buf : nullptr,
                                        residual_scale);
    if (cache_is_3d)
    {
        dx_out_buf.reshape(Shape{ B, S, hidden_dim });
    }
    return { dx_out_buf, _produce_scaled_dx ? dx_scaled_out_buf : Tensor() };
}

void Transformer_Block::resetGradient()
{
    is_forward_completed = false;
    last_forward_used_kv_cache = false;
    cache_is_checkpointed = false;
    forEachNorm([](RMSNorm_Layer &norm) { norm.resetGradient(); });
    forEachActiveProjection([](ILayer &proj) { proj.resetGradient(); });
}

void Transformer_Block::resetGradients()
{
    is_forward_completed = false;
    last_forward_used_kv_cache = false;
    cache_is_checkpointed = false;
    forEachNorm([](RMSNorm_Layer &norm) { norm.resetGradients(); });
    forEachActiveProjection([](ILayer &proj) { proj.resetGradients(); });
}

void Transformer_Block::setTrainingMode(bool _is_training)
{
    is_training_mode = _is_training;
    forEachNorm([_is_training](RMSNorm_Layer &norm) { norm.setTrainingMode(_is_training); });
    forEachActiveProjection([_is_training](ILayer &proj) { proj.setTrainingMode(_is_training); });
}

std::vector<std::pair<Tensor *, Tensor *>> Transformer_Block::getParametersAndGradients()
{
    std::vector<std::pair<Tensor *, Tensor *>> params;
    auto add = [&params](ILayer &layer) {
        auto p = layer.getParametersAndGradients();
        params.insert(params.end(), p.begin(), p.end());
    };
    add(input_layernorm);
    if (use_fused_projections)
    {
        add(qkv_proj);
    }
    else
    {
        add(q_proj);
        add(k_proj);
        add(v_proj);
    }
    if (q_norm)
    {
        add(*q_norm);
    }
    if (k_norm)
    {
        add(*k_norm);
    }
    add(o_proj);
    add(post_attention_layernorm);
    if (use_fused_projections)
    {
        add(gate_up_proj);
    }
    else
    {
        add(gate_proj);
        add(up_proj);
    }
    add(down_proj);
    return params;
}

std::vector<const Tensor *> Transformer_Block::getNoDecayParameters() const
{
    std::vector<const Tensor *> params;
    auto add = [&params](const ILayer &layer) {
        auto p = layer.getNoDecayParameters();
        params.insert(params.end(), p.begin(), p.end());
    };
    add(input_layernorm);
    if (use_fused_projections)
    {
        add(qkv_proj);
    }
    else
    {
        add(q_proj);
        add(k_proj);
        add(v_proj);
    }
    if (q_norm)
    {
        add(*q_norm);
    }
    if (k_norm)
    {
        add(*k_norm);
    }
    add(o_proj);
    add(post_attention_layernorm);
    if (use_fused_projections)
    {
        add(gate_up_proj);
    }
    else
    {
        add(gate_proj);
        add(up_proj);
    }
    add(down_proj);
    return params;
}

void Transformer_Block::setExecutionTarget(Execution_Target _execution_target)
{
    if (execution_target == _execution_target)
    {
        return;
    }
    logChangeExecutionTarget(_execution_target);
    execution_target = _execution_target;
    is_forward_completed = false;
    last_forward_used_kv_cache = false;
    cache_is_checkpointed = false;
    input_layernorm.setExecutionTarget(_execution_target);
    if (q_norm)
    {
        q_norm->setExecutionTarget(_execution_target);
    }
    if (k_norm)
    {
        k_norm->setExecutionTarget(_execution_target);
    }
    o_proj.setExecutionTarget(_execution_target);
    post_attention_layernorm.setExecutionTarget(_execution_target);
    down_proj.setExecutionTarget(_execution_target);
    if (use_fused_projections)
    {
        qkv_proj.setExecutionTarget(_execution_target);
        gate_up_proj.setExecutionTarget(_execution_target);
    }
    else
    {
        q_proj.setExecutionTarget(_execution_target);
        k_proj.setExecutionTarget(_execution_target);
        v_proj.setExecutionTarget(_execution_target);
        gate_proj.setExecutionTarget(_execution_target);
        up_proj.setExecutionTarget(_execution_target);
    }
    cache_x = Tensor(_execution_target);
    cache_h1 = Tensor(_execution_target);
    cache_act = Tensor(_execution_target);
    cache_q_rope = Tensor(_execution_target);
    cache_k_rope = Tensor(_execution_target);
    cache_v = Tensor(_execution_target);
    cache_k_all = Tensor(_execution_target);
    cache_v_all = Tensor(_execution_target);
    cache_attn_out = Tensor(_execution_target);
    cache_l_stats = Tensor(_execution_target);
    cache_ckpt_input = Tensor(_execution_target);
    cache_ckpt_branch = Tensor(_execution_target);
    cache_position_ids = Tensor(_execution_target);
    cache_attn_mask = Tensor(_execution_target);
    cache_attn_kv_index = Tensor(_execution_target);
    cache_scaled_attn = Tensor(_execution_target);
    cache_attn_permuted = Tensor(_execution_target);
    cache_scaled_ffn = Tensor(_execution_target);
    cache_ffn_out = Tensor(_execution_target);
    cache_h2 = Tensor(_execution_target);
    cache_dy_branch = Tensor(_execution_target);
    cache_dg = Tensor(_execution_target);
    cache_du = Tensor(_execution_target);
    cache_d_norm_h1 = Tensor(_execution_target);
    cache_dh1 = Tensor(_execution_target);
    cache_dh1_branch = Tensor(_execution_target);
    cache_d_attn_permuted = Tensor(_execution_target);
    cache_dq = Tensor(_execution_target);
    cache_dk = Tensor(_execution_target);
    cache_dv = Tensor(_execution_target);
    cache_dq_unrope = Tensor(_execution_target);
    cache_dk_unrope = Tensor(_execution_target);
    cache_dv_permuted = Tensor(_execution_target);
    cache_d_norm_qk = Tensor(_execution_target);
    cache_d_norm_attn = Tensor(_execution_target);
    cache_d_gate_up = Tensor(_execution_target);
    cache_dqkv = Tensor(_execution_target);
    cache_dx = Tensor(_execution_target);
    cache_dx_scaled = Tensor(_execution_target);
}

std::unique_ptr<ILayer> Transformer_Block::clone() const
{
    Transformer_Block_Config cfg;
    cfg.hidden_dim = hidden_dim;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.num_kv_heads = num_kv_heads;
    cfg.intermediate_dim = intermediate_dim;
    cfg.rms_norm_eps = rms_norm_eps;
    cfg.rope_base = rope_base;
    cfg.yarn = yarn_config;
    cfg.execution_target = execution_target;
    cfg.data_type = data_type;
    cfg.num_layers = num_layers;
    cfg.residual_scale = residual_scale;
    cfg.use_fused_projections = use_fused_projections;
    cfg.use_qk_norm = use_qk_norm;
    cfg.use_attention_bias = use_attention_bias;
    cfg.use_mlp_bias = use_mlp_bias;
    cfg.gradient_checkpointing = gradient_checkpointing;

    auto copy = std::make_unique<Transformer_Block>(cfg);
    input_layernorm.copyStateTo(copy->input_layernorm);
    if (use_fused_projections)
    {
        qkv_proj.copyStateTo(copy->qkv_proj);
        gate_up_proj.copyStateTo(copy->gate_up_proj);
    }
    else
    {
        copy->q_proj = q_proj;
        copy->k_proj = k_proj;
        copy->v_proj = v_proj;
        copy->gate_proj = gate_proj;
        copy->up_proj = up_proj;
    }
    if (q_norm && copy->q_norm)
    {
        q_norm->copyStateTo(*copy->q_norm);
    }
    if (k_norm && copy->k_norm)
    {
        k_norm->copyStateTo(*copy->k_norm);
    }
    copy->o_proj = o_proj;
    post_attention_layernorm.copyStateTo(copy->post_attention_layernorm);
    copy->down_proj = down_proj;
    copy->setAccumulated(is_accumulated);
    copy->setMixedPrecision(is_mixed_precision_enabled);
    copy->setTrainingMode(is_training_mode);
    copy->ILayer::setTrainable(is_trainable);
    return copy;
}

void Transformer_Block::prepareInference()
{
    o_proj.prepareInference();
    down_proj.prepareInference();
    if (use_fused_projections)
    {
        qkv_proj.prepareInference();
        gate_up_proj.prepareInference();
    }
    else
    {
        q_proj.prepareInference();
        k_proj.prepareInference();
        v_proj.prepareInference();
        gate_proj.prepareInference();
        up_proj.prepareInference();
    }
}

void Transformer_Block::quantizeToTernary()
{
    if (hasActiveLoRA())
    {
        throw std::logic_error("Transformer_Block::quantizeToTernary: Disable LoRA before quantizing to TERNARY_INT2");
    }
    if (use_fused_projections)
    {
        Logger::logMessage("Transformer_Block::quantizeToTernary: Disabling fused QKV/GateUp projections for ternary weights",
                           Log_Level::LOG_INFO,
                           false,
                           1,
                           Log_Feature::TRANSFORMER | Log_Feature::TERNARY_METRICS);
        enableFusedProjections(false);
    }
    data_type = Data_Type::TERNARY_INT2;
    q_proj.quantizeToTernary();
    k_proj.quantizeToTernary();
    v_proj.quantizeToTernary();
    o_proj.quantizeToTernary();
    gate_proj.quantizeToTernary();
    up_proj.quantizeToTernary();
    down_proj.quantizeToTernary();
}

void Transformer_Block::dequantizeFromTernary()
{
    data_type = is_mixed_precision_enabled ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
    o_proj.dequantizeFromTernary();
    down_proj.dequantizeFromTernary();
    if (use_fused_projections)
    {
        qkv_proj.dequantizeFromTernary();
        gate_up_proj.dequantizeFromTernary();
    }
    else
    {
        q_proj.dequantizeFromTernary();
        k_proj.dequantizeFromTernary();
        v_proj.dequantizeFromTernary();
        gate_proj.dequantizeFromTernary();
        up_proj.dequantizeFromTernary();
    }
}

void Transformer_Block::saveConfiguration(std::ofstream &_output_file_stream) const
{
    constexpr uint64_t TRANSFORMER_BLOCK_GQA_FLAG = (1ULL << 63);
    constexpr uint64_t TRANSFORMER_BLOCK_EXT_FLAG = (1ULL << 62);
    const bool write_gqa = (num_kv_heads != num_heads);
    const bool write_ext = (head_dim != hidden_dim / num_heads) || ((hidden_dim % num_heads) != 0) ||
                           use_qk_norm || !use_attention_bias || !use_mlp_bias || gradient_checkpointing;
    uint64_t h_dim = hidden_dim;
    uint64_t n_heads = static_cast<uint64_t>(num_heads)
                     | (write_gqa ? TRANSFORMER_BLOCK_GQA_FLAG : 0ULL)
                     | (write_ext ? TRANSFORMER_BLOCK_EXT_FLAG : 0ULL);
    uint64_t i_dim = intermediate_dim;
    uint64_t n_layers = num_layers;
    _output_file_stream.write(reinterpret_cast<const char *>(&h_dim), sizeof(h_dim));
    _output_file_stream.write(reinterpret_cast<const char *>(&n_heads), sizeof(n_heads));
    _output_file_stream.write(reinterpret_cast<const char *>(&i_dim), sizeof(i_dim));
    _output_file_stream.write(reinterpret_cast<const char *>(&rms_norm_eps), sizeof(rms_norm_eps));
    _output_file_stream.write(reinterpret_cast<const char *>(&rope_base), sizeof(rope_base));
    _output_file_stream.write(reinterpret_cast<const char *>(&n_layers), sizeof(n_layers));
    _output_file_stream.write(reinterpret_cast<const char *>(&residual_scale), sizeof(residual_scale));
    if (write_gqa)
    {
        uint64_t n_kv_heads = static_cast<uint64_t>(num_kv_heads);
        _output_file_stream.write(reinterpret_cast<const char *>(&n_kv_heads), sizeof(n_kv_heads));
    }
    if (write_ext)
    {
        uint64_t h_d = static_cast<uint64_t>(head_dim);
        uint8_t ext_flags = 0u;
        if (use_qk_norm) ext_flags |= 0x01u;
        if (use_attention_bias) ext_flags |= 0x02u;
        if (use_mlp_bias) ext_flags |= 0x04u;
        if (gradient_checkpointing) ext_flags |= 0x08u;
        _output_file_stream.write(reinterpret_cast<const char *>(&h_d), sizeof(h_d));
        _output_file_stream.write(reinterpret_cast<const char *>(&ext_flags), sizeof(ext_flags));
    }
}

void Transformer_Block::saveInference(std::ofstream &_output_file_stream) const
{
    input_layernorm.saveInference(_output_file_stream);
    if (use_fused_projections)
    {
        const float gain = 0.0004f * static_cast<float>(hidden_dim);
        const size_t q_dim = num_heads * head_dim;
        const size_t kv_hidden = num_kv_heads * head_dim;
        Linear_Layer tq, tk, tv, tgate, tup;
        std::vector<Linear_Layer *> qkv_dsts{ &tq, &tk, &tv };
        splitLinearColumns(qkv_proj, qkv_dsts, { q_dim, kv_hidden, kv_hidden }, hidden_dim, Execution_Target::CPU, gain, false, false, use_attention_bias);
        tq.saveInference(_output_file_stream);
        tk.saveInference(_output_file_stream);
        tv.saveInference(_output_file_stream);
        o_proj.getBaseLayer().saveInference(_output_file_stream);
        post_attention_layernorm.saveInference(_output_file_stream);
        std::vector<Linear_Layer *> gu_dsts{ &tgate, &tup };
        splitLinearColumns(gate_up_proj, gu_dsts, { intermediate_dim, intermediate_dim }, hidden_dim, Execution_Target::CPU, gain, false, false, use_mlp_bias);
        tgate.saveInference(_output_file_stream);
        tup.saveInference(_output_file_stream);
        down_proj.getBaseLayer().saveInference(_output_file_stream);
    }
    else
    {
        q_proj.getBaseLayer().saveInference(_output_file_stream);
        k_proj.getBaseLayer().saveInference(_output_file_stream);
        v_proj.getBaseLayer().saveInference(_output_file_stream);
        o_proj.getBaseLayer().saveInference(_output_file_stream);
        post_attention_layernorm.saveInference(_output_file_stream);
        gate_proj.getBaseLayer().saveInference(_output_file_stream);
        up_proj.getBaseLayer().saveInference(_output_file_stream);
        down_proj.getBaseLayer().saveInference(_output_file_stream);
    }
    if (use_qk_norm)
    {
        q_norm->saveInference(_output_file_stream);
        k_norm->saveInference(_output_file_stream);
    }
    if (hasActiveLoRA())
    {
        const std::array<const LoRA_Linear_Layer *, 7> lora_mods{ &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj };
        uint8_t mask = 0u;
        for (size_t i = 0; i < lora_mods.size(); ++i)
        {
            if (lora_mods[i]->isLoRAEnabled())
            {
                mask |= static_cast<uint8_t>(1u << i);
            }
        }
        _output_file_stream.write(reinterpret_cast<const char *>(&TRANSFORMER_BLOCK_LORA_MAGIC), sizeof(TRANSFORMER_BLOCK_LORA_MAGIC));
        _output_file_stream.write(reinterpret_cast<const char *>(&mask), sizeof(mask));
        for (size_t i = 0; i < lora_mods.size(); ++i)
        {
            if ((mask & static_cast<uint8_t>(1u << i)) == 0u)
            {
                continue;
            }
            const uint32_t r = static_cast<uint32_t>(lora_mods[i]->getRank());
            const float a = lora_mods[i]->getAlpha();
            const float d = lora_mods[i]->getDropoutRate();
            const uint8_t merged = lora_mods[i]->isMerged() ? 1u : 0u;
            const uint8_t base_tr = lora_mods[i]->isBaseTrainable() ? 1u : 0u;
            _output_file_stream.write(reinterpret_cast<const char *>(&r), sizeof(r));
            _output_file_stream.write(reinterpret_cast<const char *>(&a), sizeof(a));
            _output_file_stream.write(reinterpret_cast<const char *>(&d), sizeof(d));
            _output_file_stream.write(reinterpret_cast<const char *>(&merged), sizeof(merged));
            _output_file_stream.write(reinterpret_cast<const char *>(&base_tr), sizeof(base_tr));
            lora_mods[i]->getLoRAA().saveInference(_output_file_stream);
            lora_mods[i]->getLoRAB().saveInference(_output_file_stream);
        }
    }
}

void Transformer_Block::loadInference(std::ifstream &_input_file_stream)
{
    disableLoRA();
    input_layernorm.loadInference(_input_file_stream);
    if (use_fused_projections)
    {
        const float gain = 0.0004f * static_cast<float>(hidden_dim);
        const size_t q_dim = num_heads * head_dim;
        const size_t kv_hidden = num_kv_heads * head_dim;
        Linear_Layer tq, tk, tv, tgate, tup;
        tq.configureDimensionsForLoad(hidden_dim, q_dim);
        tk.configureDimensionsForLoad(hidden_dim, kv_hidden);
        tv.configureDimensionsForLoad(hidden_dim, kv_hidden);
        tgate.configureDimensionsForLoad(hidden_dim, intermediate_dim);
        tup.configureDimensionsForLoad(hidden_dim, intermediate_dim);
        tq.loadInference(_input_file_stream);
        tk.loadInference(_input_file_stream);
        tv.loadInference(_input_file_stream);
        o_proj.getBaseLayer().loadInference(_input_file_stream);
        post_attention_layernorm.loadInference(_input_file_stream);
        tgate.loadInference(_input_file_stream);
        tup.loadInference(_input_file_stream);
        down_proj.getBaseLayer().loadInference(_input_file_stream);
        o_proj.setAccumulated(is_accumulated);
        o_proj.setTrainingMode(is_training_mode);
        o_proj.setMixedPrecision(is_mixed_precision_enabled || o_proj.getBaseLayer().isMixedPrecisionEnabled());
        down_proj.setAccumulated(is_accumulated);
        down_proj.setTrainingMode(is_training_mode);
        down_proj.setMixedPrecision(is_mixed_precision_enabled || down_proj.getBaseLayer().isMixedPrecisionEnabled());
        if (tq.isTernary() || tk.isTernary() || tv.isTernary() || tgate.isTernary() || tup.isTernary())
        {
            use_fused_projections = false;
            tq.setExecutionTarget(execution_target);
            tk.setExecutionTarget(execution_target);
            tv.setExecutionTarget(execution_target);
            tgate.setExecutionTarget(execution_target);
            tup.setExecutionTarget(execution_target);
            q_proj = LoRA_Linear_Layer(std::move(tq));
            k_proj = LoRA_Linear_Layer(std::move(tk));
            v_proj = LoRA_Linear_Layer(std::move(tv));
            gate_proj = LoRA_Linear_Layer(std::move(tgate));
            up_proj = LoRA_Linear_Layer(std::move(tup));
            for (LoRA_Linear_Layer *p : { &q_proj, &k_proj, &v_proj, &gate_proj, &up_proj })
            {
                p->setAccumulated(is_accumulated);
                p->setTrainingMode(is_training_mode);
                p->setMixedPrecision(is_mixed_precision_enabled || p->getBaseLayer().isMixedPrecisionEnabled());
            }
            qkv_proj = Linear_Layer();
            gate_up_proj = Linear_Layer();
            data_type = q_proj.getBaseLayer().getDataType();
        }
        else
        {
            mergeLinearColumns({ &tq, &tk, &tv }, qkv_proj, hidden_dim, execution_target, gain, is_mixed_precision_enabled || tq.isMixedPrecisionEnabled(), false, use_attention_bias);
            mergeLinearColumns({ &tgate, &tup }, gate_up_proj, hidden_dim, execution_target, gain, is_mixed_precision_enabled || tgate.isMixedPrecisionEnabled(), false, use_mlp_bias);
            qkv_proj.setAccumulated(is_accumulated);
            qkv_proj.setTrainingMode(is_training_mode);
            gate_up_proj.setAccumulated(is_accumulated);
            gate_up_proj.setTrainingMode(is_training_mode);
            data_type = qkv_proj.getDataType();
        }
    }
    else
    {
        q_proj.getBaseLayer().loadInference(_input_file_stream);
        k_proj.getBaseLayer().loadInference(_input_file_stream);
        v_proj.getBaseLayer().loadInference(_input_file_stream);
        o_proj.getBaseLayer().loadInference(_input_file_stream);
        post_attention_layernorm.loadInference(_input_file_stream);
        gate_proj.getBaseLayer().loadInference(_input_file_stream);
        up_proj.getBaseLayer().loadInference(_input_file_stream);
        down_proj.getBaseLayer().loadInference(_input_file_stream);
        for (LoRA_Linear_Layer *p : { &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj })
        {
            p->setAccumulated(is_accumulated);
            p->setTrainingMode(is_training_mode);
            p->setMixedPrecision(is_mixed_precision_enabled || p->getBaseLayer().isMixedPrecisionEnabled());
        }
        data_type = q_proj.getBaseLayer().getDataType();
    }
    if (use_qk_norm)
    {
        q_norm->loadInference(_input_file_stream);
        k_norm->loadInference(_input_file_stream);
    }
    if (_input_file_stream.good() && _input_file_stream.peek() != std::char_traits<char>::eof())
    {
        const std::streampos saved_pos = _input_file_stream.tellg();
        uint32_t magic = 0u;
        if (_input_file_stream.read(reinterpret_cast<char *>(&magic), sizeof(magic)) && magic == TRANSFORMER_BLOCK_LORA_MAGIC)
        {
            if (use_fused_projections)
            {
                enableFusedProjections(false);
            }
            uint8_t mask = 0u;
            if (!_input_file_stream.read(reinterpret_cast<char *>(&mask), sizeof(mask)))
            {
                throw std::runtime_error("Transformer_Block::loadInference: Failed reading LoRA mask");
            }
            const std::array<LoRA_Linear_Layer *, 7> lora_mods{ &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj };
            for (size_t i = 0; i < lora_mods.size(); ++i)
            {
                if ((mask & static_cast<uint8_t>(1u << i)) == 0u)
                {
                    continue;
                }
                uint32_t r = 0u;
                float a = 1.0f;
                float d = 0.0f;
                uint8_t merged = 0u;
                uint8_t base_tr = 0u;
                if (!_input_file_stream.read(reinterpret_cast<char *>(&r), sizeof(r)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&a), sizeof(a)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&d), sizeof(d)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&merged), sizeof(merged)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&base_tr), sizeof(base_tr)))
                {
                    throw std::runtime_error("Transformer_Block::loadInference: Truncated LoRA metadata");
                }
                lora_mods[i]->enableLoRA(r, a, d, base_tr == 0u);
                lora_mods[i]->getLoRAA().loadInference(_input_file_stream);
                lora_mods[i]->getLoRAB().loadInference(_input_file_stream);
                if (merged != 0u)
                {
                    lora_mods[i]->mergeWeights();
                }
            }
        }
        else
        {
            _input_file_stream.clear();
            _input_file_stream.seekg(saved_pos);
        }
    }
}

void Transformer_Block::saveCheckpoint(std::ofstream &_output_file_stream) const
{
    input_layernorm.saveCheckpoint(_output_file_stream);
    if (use_fused_projections)
    {
        const float gain = 0.0004f * static_cast<float>(hidden_dim);
        const size_t q_dim = num_heads * head_dim;
        const size_t kv_hidden = num_kv_heads * head_dim;
        Linear_Layer tq, tk, tv, tgate, tup;
        std::vector<Linear_Layer *> qkv_dsts{ &tq, &tk, &tv };
        splitLinearColumns(qkv_proj, qkv_dsts, { q_dim, kv_hidden, kv_hidden }, hidden_dim, Execution_Target::CPU, gain, false, true, use_attention_bias);
        tq.saveCheckpoint(_output_file_stream);
        tk.saveCheckpoint(_output_file_stream);
        tv.saveCheckpoint(_output_file_stream);
        o_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        post_attention_layernorm.saveCheckpoint(_output_file_stream);
        std::vector<Linear_Layer *> gu_dsts{ &tgate, &tup };
        splitLinearColumns(gate_up_proj, gu_dsts, { intermediate_dim, intermediate_dim }, hidden_dim, Execution_Target::CPU, gain, false, true, use_mlp_bias);
        tgate.saveCheckpoint(_output_file_stream);
        tup.saveCheckpoint(_output_file_stream);
        down_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
    }
    else
    {
        q_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        k_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        v_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        o_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        post_attention_layernorm.saveCheckpoint(_output_file_stream);
        gate_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        up_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
        down_proj.getBaseLayer().saveCheckpoint(_output_file_stream);
    }
    if (use_qk_norm)
    {
        q_norm->saveCheckpoint(_output_file_stream);
        k_norm->saveCheckpoint(_output_file_stream);
    }
    if (hasActiveLoRA())
    {
        const std::array<const LoRA_Linear_Layer *, 7> lora_mods{ &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj };
        uint8_t mask = 0u;
        for (size_t i = 0; i < lora_mods.size(); ++i)
        {
            if (lora_mods[i]->isLoRAEnabled())
            {
                mask |= static_cast<uint8_t>(1u << i);
            }
        }
        _output_file_stream.write(reinterpret_cast<const char *>(&TRANSFORMER_BLOCK_LORA_MAGIC), sizeof(TRANSFORMER_BLOCK_LORA_MAGIC));
        _output_file_stream.write(reinterpret_cast<const char *>(&mask), sizeof(mask));
        for (size_t i = 0; i < lora_mods.size(); ++i)
        {
            if ((mask & static_cast<uint8_t>(1u << i)) == 0u)
            {
                continue;
            }
            const uint32_t r = static_cast<uint32_t>(lora_mods[i]->getRank());
            const float a = lora_mods[i]->getAlpha();
            const float d = lora_mods[i]->getDropoutRate();
            const uint8_t merged = lora_mods[i]->isMerged() ? 1u : 0u;
            const uint8_t base_tr = lora_mods[i]->isBaseTrainable() ? 1u : 0u;
            _output_file_stream.write(reinterpret_cast<const char *>(&r), sizeof(r));
            _output_file_stream.write(reinterpret_cast<const char *>(&a), sizeof(a));
            _output_file_stream.write(reinterpret_cast<const char *>(&d), sizeof(d));
            _output_file_stream.write(reinterpret_cast<const char *>(&merged), sizeof(merged));
            _output_file_stream.write(reinterpret_cast<const char *>(&base_tr), sizeof(base_tr));
            lora_mods[i]->getLoRAA().saveCheckpoint(_output_file_stream);
            lora_mods[i]->getLoRAB().saveCheckpoint(_output_file_stream);
        }
    }
}

void Transformer_Block::loadCheckpoint(std::ifstream &_input_file_stream)
{
    disableLoRA();
    input_layernorm.loadCheckpoint(_input_file_stream);
    if (use_fused_projections)
    {
        const float gain = 0.0004f * static_cast<float>(hidden_dim);
        const size_t q_dim = num_heads * head_dim;
        const size_t kv_hidden = num_kv_heads * head_dim;
        Linear_Layer tq, tk, tv, tgate, tup;
        tq.configureDimensionsForLoad(hidden_dim, q_dim);
        tk.configureDimensionsForLoad(hidden_dim, kv_hidden);
        tv.configureDimensionsForLoad(hidden_dim, kv_hidden);
        tgate.configureDimensionsForLoad(hidden_dim, intermediate_dim);
        tup.configureDimensionsForLoad(hidden_dim, intermediate_dim);
        tq.loadCheckpoint(_input_file_stream);
        tk.loadCheckpoint(_input_file_stream);
        tv.loadCheckpoint(_input_file_stream);
        o_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        post_attention_layernorm.loadCheckpoint(_input_file_stream);
        tgate.loadCheckpoint(_input_file_stream);
        tup.loadCheckpoint(_input_file_stream);
        down_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        o_proj.setAccumulated(is_accumulated);
        o_proj.setTrainingMode(is_training_mode);
        o_proj.setMixedPrecision(is_mixed_precision_enabled || o_proj.getBaseLayer().isMixedPrecisionEnabled());
        down_proj.setAccumulated(is_accumulated);
        down_proj.setTrainingMode(is_training_mode);
        down_proj.setMixedPrecision(is_mixed_precision_enabled || down_proj.getBaseLayer().isMixedPrecisionEnabled());
        if (tq.isTernary() || tk.isTernary() || tv.isTernary() || tgate.isTernary() || tup.isTernary())
        {
            use_fused_projections = false;
            tq.setExecutionTarget(execution_target);
            tk.setExecutionTarget(execution_target);
            tv.setExecutionTarget(execution_target);
            tgate.setExecutionTarget(execution_target);
            tup.setExecutionTarget(execution_target);
            q_proj = LoRA_Linear_Layer(std::move(tq));
            k_proj = LoRA_Linear_Layer(std::move(tk));
            v_proj = LoRA_Linear_Layer(std::move(tv));
            gate_proj = LoRA_Linear_Layer(std::move(tgate));
            up_proj = LoRA_Linear_Layer(std::move(tup));
            for (LoRA_Linear_Layer *p : { &q_proj, &k_proj, &v_proj, &gate_proj, &up_proj })
            {
                p->setAccumulated(is_accumulated);
                p->setTrainingMode(is_training_mode);
                p->setMixedPrecision(is_mixed_precision_enabled || p->getBaseLayer().isMixedPrecisionEnabled());
            }
            qkv_proj = Linear_Layer();
            gate_up_proj = Linear_Layer();
            data_type = q_proj.getBaseLayer().getDataType();
        }
        else
        {
            mergeLinearColumns({ &tq, &tk, &tv }, qkv_proj, hidden_dim, execution_target, gain, is_mixed_precision_enabled || tq.isMixedPrecisionEnabled(), true, use_attention_bias);
            mergeLinearColumns({ &tgate, &tup }, gate_up_proj, hidden_dim, execution_target, gain, is_mixed_precision_enabled || tgate.isMixedPrecisionEnabled(), true, use_mlp_bias);
            qkv_proj.setAccumulated(is_accumulated);
            qkv_proj.setTrainingMode(is_training_mode);
            gate_up_proj.setAccumulated(is_accumulated);
            gate_up_proj.setTrainingMode(is_training_mode);
            data_type = qkv_proj.getDataType();
        }
    }
    else
    {
        q_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        k_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        v_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        o_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        post_attention_layernorm.loadCheckpoint(_input_file_stream);
        gate_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        up_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        down_proj.getBaseLayer().loadCheckpoint(_input_file_stream);
        for (LoRA_Linear_Layer *p : { &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj })
        {
            p->setAccumulated(is_accumulated);
            p->setTrainingMode(is_training_mode);
            p->setMixedPrecision(is_mixed_precision_enabled || p->getBaseLayer().isMixedPrecisionEnabled());
        }
        data_type = q_proj.getBaseLayer().getDataType();
    }
    if (use_qk_norm)
    {
        q_norm->loadCheckpoint(_input_file_stream);
        k_norm->loadCheckpoint(_input_file_stream);
    }
    if (_input_file_stream.good() && _input_file_stream.peek() != std::char_traits<char>::eof())
    {
        const std::streampos saved_pos = _input_file_stream.tellg();
        uint32_t magic = 0u;
        if (_input_file_stream.read(reinterpret_cast<char *>(&magic), sizeof(magic)) && magic == TRANSFORMER_BLOCK_LORA_MAGIC)
        {
            if (use_fused_projections)
            {
                enableFusedProjections(false);
            }
            uint8_t mask = 0u;
            if (!_input_file_stream.read(reinterpret_cast<char *>(&mask), sizeof(mask)))
            {
                throw std::runtime_error("Transformer_Block::loadCheckpoint: Failed reading LoRA mask");
            }
            const std::array<LoRA_Linear_Layer *, 7> lora_mods{ &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj };
            for (size_t i = 0; i < lora_mods.size(); ++i)
            {
                if ((mask & static_cast<uint8_t>(1u << i)) == 0u)
                {
                    continue;
                }
                uint32_t r = 0u;
                float a = 1.0f;
                float d = 0.0f;
                uint8_t merged = 0u;
                uint8_t base_tr = 0u;
                if (!_input_file_stream.read(reinterpret_cast<char *>(&r), sizeof(r)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&a), sizeof(a)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&d), sizeof(d)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&merged), sizeof(merged)) ||
                    !_input_file_stream.read(reinterpret_cast<char *>(&base_tr), sizeof(base_tr)))
                {
                    throw std::runtime_error("Transformer_Block::loadCheckpoint: Truncated LoRA metadata");
                }
                lora_mods[i]->enableLoRA(r, a, d, base_tr == 0u);
                lora_mods[i]->getLoRAA().loadCheckpoint(_input_file_stream);
                lora_mods[i]->getLoRAB().loadCheckpoint(_input_file_stream);
                if (merged != 0u)
                {
                    lora_mods[i]->mergeWeights();
                }
            }
        }
        else
        {
            _input_file_stream.clear();
            _input_file_stream.seekg(saved_pos);
        }
    }
}

size_t Transformer_Block::getParameterCount() const noexcept
{
    auto count_linear = [](const Linear_Layer &l) -> size_t {
        return l.getWeights().getTotalElements() + (l.hasBias() ? l.getBiases().getTotalElements() : 0u);
    };
    auto count_lora_layer = [&](const LoRA_Linear_Layer &l) -> size_t {
        size_t c = count_linear(l.getBaseLayer());
        if (l.isLoRAEnabled())
        {
            c += l.getLoRAA().getWeights().getTotalElements() + l.getLoRAB().getWeights().getTotalElements();
        }
        return c;
    };

    size_t count = input_layernorm.getWeights().getTotalElements()
                 + count_lora_layer(o_proj)
                 + post_attention_layernorm.getWeights().getTotalElements()
                 + count_lora_layer(down_proj);
    if (q_norm)
    {
        count += q_norm->getWeights().getTotalElements();
    }
    if (k_norm)
    {
        count += k_norm->getWeights().getTotalElements();
    }
    if (use_fused_projections)
    {
        count += count_linear(qkv_proj) + count_linear(gate_up_proj);
    }
    else
    {
        count += count_lora_layer(q_proj)
               + count_lora_layer(k_proj)
               + count_lora_layer(v_proj)
               + count_lora_layer(gate_proj)
               + count_lora_layer(up_proj);
    }
    return count;
}

const RMSNorm_Layer &Transformer_Block::getInputLayernorm() const noexcept
{
    return input_layernorm;
}

RMSNorm_Layer &Transformer_Block::getInputLayernorm() noexcept
{
    return input_layernorm;
}

const Linear_Layer &Transformer_Block::getQProj() const noexcept
{
    return q_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getQProj() noexcept
{
    return q_proj.getBaseLayer();
}

const Linear_Layer &Transformer_Block::getKProj() const noexcept
{
    return k_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getKProj() noexcept
{
    return k_proj.getBaseLayer();
}

const Linear_Layer &Transformer_Block::getVProj() const noexcept
{
    return v_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getVProj() noexcept
{
    return v_proj.getBaseLayer();
}

const Linear_Layer &Transformer_Block::getOProj() const noexcept
{
    return o_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getOProj() noexcept
{
    return o_proj.getBaseLayer();
}

const RMSNorm_Layer &Transformer_Block::getPostAttentionLayernorm() const noexcept
{
    return post_attention_layernorm;
}

RMSNorm_Layer &Transformer_Block::getPostAttentionLayernorm() noexcept
{
    return post_attention_layernorm;
}

const Linear_Layer &Transformer_Block::getGateProj() const noexcept
{
    return gate_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getGateProj() noexcept
{
    return gate_proj.getBaseLayer();
}

const Linear_Layer &Transformer_Block::getUpProj() const noexcept
{
    return up_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getUpProj() noexcept
{
    return up_proj.getBaseLayer();
}

const Linear_Layer &Transformer_Block::getDownProj() const noexcept
{
    return down_proj.getBaseLayer();
}

Linear_Layer &Transformer_Block::getDownProj() noexcept
{
    return down_proj.getBaseLayer();
}

const Linear_Layer &Transformer_Block::getQkvProj() const noexcept
{
    return qkv_proj;
}

Linear_Layer &Transformer_Block::getQkvProj() noexcept
{
    return qkv_proj;
}

const Linear_Layer &Transformer_Block::getGateUpProj() const noexcept
{
    return gate_up_proj;
}

Linear_Layer &Transformer_Block::getGateUpProj() noexcept
{
    return gate_up_proj;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <fstream>

#include "layer/ilayer.h"
#include "math/tensor.h"
#include "layer/linear_layer.h"
#include "layer/rmsnorm_layer.h"
#include "layer/swiglu_layer.h"
#include "llm/kv_cache_manager.h"
#include "engine/execution_engine.h"

class Transformer_Block : public ILayer
{
private:
    size_t hidden_dim;
    size_t num_heads;
    size_t head_dim;
    size_t intermediate_dim;
    float rms_norm_eps;
    float rope_base;
    Execution_Target execution_target;
    Data_Type data_type;
    size_t num_layers = 1;
    float residual_scale = 1.0f;

    RMSNorm_Layer input_layernorm;
    Linear_Layer q_proj;
    Linear_Layer k_proj;
    Linear_Layer v_proj;
    Linear_Layer o_proj;

    RMSNorm_Layer post_attention_layernorm;
    Linear_Layer gate_proj;
    Linear_Layer up_proj;
    Linear_Layer down_proj;
    SwiGLU_Layer swiglu;

    mutable KV_Cache_Manager *active_kv_cache = nullptr;
    mutable Tensor cache_x;
    mutable Tensor cache_h1;
    mutable Tensor cache_act;
    mutable Tensor cache_q_rope;
    mutable Tensor cache_k_rope;
    mutable Tensor cache_v;
    mutable Tensor cache_attn_out;
    mutable Tensor cache_l_stats;
    mutable size_t cache_batch_size = 1;
    mutable size_t cache_seq_len = 1;
    mutable bool cache_is_3d = false;

    // Persistent activation & gradient tensors for Zero-Allocation Replay
    mutable Tensor cache_scaled_attn;
    mutable Tensor cache_attn_permuted;
    mutable Tensor cache_scaled_ffn;
    mutable Tensor cache_h2;

    mutable Tensor cache_dy_branch;
    mutable Tensor cache_dg;
    mutable Tensor cache_du;
    mutable Tensor cache_d_norm_h1;
    mutable Tensor cache_dh1;
    mutable Tensor cache_dh1_branch;
    mutable Tensor cache_d_attn_permuted;
    mutable Tensor cache_dq;
    mutable Tensor cache_dk;
    mutable Tensor cache_dv;
    mutable Tensor cache_dq_unrope;
    mutable Tensor cache_dk_unrope;
    mutable Tensor cache_dv_permuted;
    mutable Tensor cache_d_norm_qk;
    mutable Tensor cache_d_norm_attn;
    mutable Tensor cache_dx;

public:
    Transformer_Block(size_t _hidden_dim,
                      size_t _num_heads,
                      size_t _intermediate_dim,
                      float _rms_norm_eps = 1e-5f,
                      float _rope_base = 10000.0f,
                      Execution_Target _execution_target = Execution_Target::CPU,
                      Data_Type _data_type = Data_Type::FLOAT32,
                      size_t _num_layers = 1)
        : hidden_dim(_hidden_dim),
          num_heads(_num_heads),
          head_dim(_hidden_dim / _num_heads),
          intermediate_dim(_intermediate_dim),
          rms_norm_eps(_rms_norm_eps),
          rope_base(_rope_base),
          execution_target(_execution_target),
          data_type(_data_type),
          num_layers(_num_layers),
          residual_scale(1.0f / std::sqrt(2.0f * static_cast<float>(std::max<size_t>(1, _num_layers)))),
          input_layernorm(_hidden_dim, _rms_norm_eps, _execution_target, _data_type),
          q_proj(_hidden_dim, _hidden_dim, _execution_target, 0.0004f * static_cast<float>(_hidden_dim), _data_type),
          k_proj(_hidden_dim, _hidden_dim, _execution_target, 0.0004f * static_cast<float>(_hidden_dim), _data_type),
          v_proj(_hidden_dim, _hidden_dim, _execution_target, 0.0004f * static_cast<float>(_hidden_dim), _data_type),
          o_proj(_hidden_dim, _hidden_dim, _execution_target, 0.0004f * static_cast<float>(_hidden_dim), _data_type),
          post_attention_layernorm(_hidden_dim, _rms_norm_eps, _execution_target, _data_type),
          gate_proj(_hidden_dim, _intermediate_dim, _execution_target, 0.0004f * static_cast<float>(_hidden_dim), _data_type),
          up_proj(_hidden_dim, _intermediate_dim, _execution_target, 0.0004f * static_cast<float>(_hidden_dim), _data_type),
          down_proj(_intermediate_dim, _hidden_dim, _execution_target, 0.0004f * static_cast<float>(_intermediate_dim), _data_type),
          swiglu(_execution_target),
          cache_x(_execution_target),
          cache_h1(_execution_target),
          cache_act(_execution_target),
          cache_q_rope(_execution_target),
          cache_k_rope(_execution_target),
          cache_v(_execution_target),
          cache_attn_out(_execution_target),
          cache_l_stats(_execution_target),
          cache_scaled_attn(_execution_target),
          cache_attn_permuted(_execution_target),
          cache_scaled_ffn(_execution_target),
          cache_h2(_execution_target),
          cache_dy_branch(_execution_target),
          cache_dg(_execution_target),
          cache_du(_execution_target),
          cache_d_norm_h1(_execution_target),
          cache_dh1(_execution_target),
          cache_dh1_branch(_execution_target),
          cache_d_attn_permuted(_execution_target),
          cache_dq(_execution_target),
          cache_dk(_execution_target),
          cache_dv(_execution_target),
          cache_dq_unrope(_execution_target),
          cache_dk_unrope(_execution_target),
          cache_dv_permuted(_execution_target),
          cache_d_norm_qk(_execution_target),
          cache_d_norm_attn(_execution_target),
          cache_dx(_execution_target)
    {
        if (_data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
    }

    void setResidualScale(float scale) noexcept { residual_scale = scale; }
    float getResidualScale() const noexcept { return residual_scale; }

    void setKVCache(KV_Cache_Manager *cache) noexcept { active_kv_cache = cache; }
    KV_Cache_Manager *getKVCache() const noexcept { return active_kv_cache; }

    void setAccumulated(bool _is_accumulated) noexcept override
    {
        is_accumulated = _is_accumulated;
        input_layernorm.setAccumulated(_is_accumulated);
        q_proj.setAccumulated(_is_accumulated);
        k_proj.setAccumulated(_is_accumulated);
        v_proj.setAccumulated(_is_accumulated);
        o_proj.setAccumulated(_is_accumulated);
        post_attention_layernorm.setAccumulated(_is_accumulated);
        gate_proj.setAccumulated(_is_accumulated);
        up_proj.setAccumulated(_is_accumulated);
        down_proj.setAccumulated(_is_accumulated);
    }

    void setMixedPrecision(bool _enable) noexcept override
    {
        is_mixed_precision_enabled = _enable;
        data_type = _enable ? Data_Type::FLOAT16 : Data_Type::FLOAT32;
        input_layernorm.setMixedPrecision(_enable);
        q_proj.setMixedPrecision(_enable);
        k_proj.setMixedPrecision(_enable);
        v_proj.setMixedPrecision(_enable);
        o_proj.setMixedPrecision(_enable);
        post_attention_layernorm.setMixedPrecision(_enable);
        gate_proj.setMixedPrecision(_enable);
        up_proj.setMixedPrecision(_enable);
        down_proj.setMixedPrecision(_enable);
    }

    void invalidateWeightCache() noexcept override
    {
        input_layernorm.invalidateWeightCache();
        q_proj.invalidateWeightCache();
        k_proj.invalidateWeightCache();
        v_proj.invalidateWeightCache();
        o_proj.invalidateWeightCache();
        post_attention_layernorm.invalidateWeightCache();
        gate_proj.invalidateWeightCache();
        up_proj.invalidateWeightCache();
        down_proj.invalidateWeightCache();
    }

    Tensor forward(const Tensor &_input_tensor) override
    {
        return forward(_input_tensor, active_kv_cache);
    }

    Tensor forward(const Tensor &x, KV_Cache_Manager *kv_cache)
    {
        size_t B = 1;
        size_t S = 1;
        bool is_input_3d = (x.getShape().getRank() == 3);
        if (is_input_3d)
        {
            B = x.getShape()[0];
            S = x.getShape()[1];
        }
        else if (x.getShape().getRank() == 2)
        {
            B = 1;
            S = x.getShape()[0];
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

        Tensor x_2d = x;
        if (x_2d.getShape().getRank() != 2 || x_2d.getColumns() != hidden_dim)
        {
            x_2d.reshape(Shape{ B * S, hidden_dim });
        }
        cache_x = x;

        Tensor norm_x = input_layernorm.forward(x_2d);

        // 2. Q, K, V Projections
        Tensor q = q_proj.forward(norm_x);
        Tensor k = k_proj.forward(norm_x);
        Tensor v = v_proj.forward(norm_x);

        q.reshape(Shape{ B, S, num_heads, head_dim });
        k.reshape(Shape{ B, S, num_heads, head_dim });
        v.reshape(Shape{ B, S, num_heads, head_dim });
        v.permute({ 0, 2, 1, 3 }).contiguous(cache_v);

        q.applyRoPE(cache_q_rope, S, head_dim, 1, rope_base, num_heads, 1 /* FUSED_PERMUTE */);
        k.applyRoPE(cache_k_rope, S, head_dim, 1, rope_base, num_heads, 1 /* FUSED_PERMUTE */);

        if (kv_cache != nullptr)
        {
            kv_cache->append(cache_k_rope, cache_v);
            Tensor k_all = kv_cache->getK().contiguous();
            Tensor v_all = kv_cache->getV().contiguous();
            size_t T = kv_cache->getCurrentSeqLen();

            if (S > 1)
            {
                cache_q_rope.flashAttentionForward(k_all, v_all, cache_attn_out, num_heads, S, head_dim, true, 0.0f, &cache_l_stats);
            }
            else
            {
                if (T == 1)
                {
                    cache_q_rope.flashAttentionForward(k_all, v_all, cache_attn_out, num_heads, 1, head_dim, false, 0.0f, &cache_l_stats);
                }
                else
                {
                    cache_q_rope.singleTokenAttentionForward(k_all, v_all, cache_attn_out, num_heads, head_dim, T);
                }
            }
        }
        else
        {
            cache_q_rope.flashAttentionForward(cache_k_rope, cache_v, cache_attn_out, num_heads, S, head_dim, true, 0.0f, &cache_l_stats);
        }

        // Permute back to [B, S, H, D] -> [B*S, D] 
        cache_attn_out.permute({ 0, 2, 1, 3 }).contiguous(cache_attn_permuted);
        cache_attn_permuted.reshape(Shape{ B * S, hidden_dim });
        Tensor attn_proj = o_proj.forward(cache_attn_permuted);

        // Residual 1 with DeepNorm Scaling
        attn_proj.mulScalar(residual_scale, cache_scaled_attn);
        x_2d.add(cache_scaled_attn, cache_h1);

        // 5. FFN Post-RMSNorm
        Tensor norm_h1 = post_attention_layernorm.forward(cache_h1);
        Tensor g = gate_proj.forward(norm_h1);
        Tensor u = up_proj.forward(norm_h1);
        g.swigluForward(u, cache_act);
        Tensor ffn_out = down_proj.forward(cache_act);

        // Residual 2 with DeepNorm Scaling
        ffn_out.mulScalar(residual_scale, cache_scaled_ffn);
        cache_h1.add(cache_scaled_ffn, cache_h2);
        if (is_input_3d)
        {
            cache_h2.reshape(Shape{ B, S, hidden_dim });
        }
        return cache_h2;
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        size_t B = cache_batch_size;
        size_t S = cache_seq_len;

        Tensor dy = _output_gradient;
        if (dy.getShape().getRank() != 2 || dy.getColumns() != hidden_dim)
        {
            dy.reshape(Shape{ B * S, hidden_dim });
        }

        // 1. Backward through FFN & Residual 2
        // h2 = h1 + residual_scale * ffn_out => d_ffn_out = dy * residual_scale
        dy.mulScalar(residual_scale, cache_dy_branch);
        Tensor d_act = down_proj.backward(cache_dy_branch);
        Tensor norm_h1 = post_attention_layernorm.getOutput();
        Tensor g = gate_proj.getOutput();
        Tensor u = up_proj.getOutput();

        g.swigluBackward(d_act, u, cache_dg, cache_du);

        Tensor d_norm_gate = gate_proj.backward(cache_dg);
        Tensor d_norm_up = up_proj.backward(cache_du);
        d_norm_gate.add(d_norm_up, cache_d_norm_h1);

        Tensor d_h1_from_norm = post_attention_layernorm.backward(cache_d_norm_h1);
        dy.add(d_h1_from_norm, cache_dh1);

        // 2. Backward through Attention & Residual 1
        // h1 = x_2d + residual_scale * attn_proj => d_attn_proj = dh1 * residual_scale
        cache_dh1.mulScalar(residual_scale, cache_dh1_branch);
        Tensor d_attn_proj = o_proj.backward(cache_dh1_branch);

        d_attn_proj.reshape(Shape{ B, S, num_heads, head_dim });
        d_attn_proj.permute({ 0, 2, 1, 3 }).contiguous(cache_d_attn_permuted);

        cache_q_rope.flashAttentionBackward(cache_k_rope, cache_v, cache_attn_out, cache_d_attn_permuted,
                                            cache_dq, cache_dk, cache_dv, num_heads, S, head_dim, true, 0.0f, &cache_l_stats);

        // Backward through RoPE with fused unpermute (inverse rotation: direction = -1, mode = 2)
        cache_dq.applyRoPE(cache_dq_unrope, S, head_dim, -1, rope_base, num_heads, 2 /* FUSED_UNPERMUTE */);
        cache_dk.applyRoPE(cache_dk_unrope, S, head_dim, -1, rope_base, num_heads, 2 /* FUSED_UNPERMUTE */);

        // dq_unrope and dk_unrope are already in [B, S, H, D] -> reshape to [B*S, D_total]
        cache_dq_unrope.reshape(Shape{ B * S, hidden_dim });
        cache_dk_unrope.reshape(Shape{ B * S, hidden_dim });

        cache_dv.permute({ 0, 2, 1, 3 }).contiguous(cache_dv_permuted);
        cache_dv_permuted.reshape(Shape{ B * S, hidden_dim });

        // Backward through Q, K, V projections
        Tensor d_norm_q = q_proj.backward(cache_dq_unrope);
        Tensor d_norm_k = k_proj.backward(cache_dk_unrope);
        Tensor d_norm_v = v_proj.backward(cache_dv_permuted);
        d_norm_q.add(d_norm_k, cache_d_norm_qk);
        cache_d_norm_qk.add(d_norm_v, cache_d_norm_attn);

        // Backward through input RMSNorm
        Tensor d_x_attn = input_layernorm.backward(cache_d_norm_attn);

        cache_dh1.add(d_x_attn, cache_dx);
        if (cache_is_3d)
        {
            cache_dx.reshape(Shape{ B, S, hidden_dim });
        }
        return cache_dx;
    }

    void resetGradient() override
    {
        input_layernorm.resetGradient();
        q_proj.resetGradient();
        k_proj.resetGradient();
        v_proj.resetGradient();
        o_proj.resetGradient();
        post_attention_layernorm.resetGradient();
        gate_proj.resetGradient();
        up_proj.resetGradient();
        down_proj.resetGradient();
    }

    void resetGradients() override
    {
        resetGradient();
        input_layernorm.resetGradients();
        q_proj.resetGradients();
        k_proj.resetGradients();
        v_proj.resetGradients();
        o_proj.resetGradients();
        post_attention_layernorm.resetGradients();
        gate_proj.resetGradients();
        up_proj.resetGradients();
        down_proj.resetGradients();
    }

    void setTrainingMode(bool _is_training) override
    {
        input_layernorm.setTrainingMode(_is_training);
        q_proj.setTrainingMode(_is_training);
        k_proj.setTrainingMode(_is_training);
        v_proj.setTrainingMode(_is_training);
        o_proj.setTrainingMode(_is_training);
        post_attention_layernorm.setTrainingMode(_is_training);
        gate_proj.setTrainingMode(_is_training);
        up_proj.setTrainingMode(_is_training);
        down_proj.setTrainingMode(_is_training);
        swiglu.setTrainingMode(_is_training);
    }

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override
    {
        std::vector<std::pair<Tensor *, Tensor *>> params;
        auto add = [&params](ILayer &layer) {
            auto p = layer.getParametersAndGradients();
            params.insert(params.end(), p.begin(), p.end());
        };
        add(input_layernorm);
        add(q_proj);
        add(k_proj);
        add(v_proj);
        add(o_proj);
        add(post_attention_layernorm);
        add(gate_proj);
        add(up_proj);
        add(down_proj);
        return params;
    }

    bool hasParameters() const override { return true; }
    Layer_Type getLayerType() const override { return Layer_Type::TRANSFORMER_BLOCK; }
    Execution_Target getExecutionTarget() const override { return execution_target; }

    void setExecutionTarget(Execution_Target _execution_target) override
    {
        logChangeExecutionTarget(_execution_target);
        execution_target = _execution_target;
        input_layernorm.setExecutionTarget(_execution_target);
        q_proj.setExecutionTarget(_execution_target);
        k_proj.setExecutionTarget(_execution_target);
        v_proj.setExecutionTarget(_execution_target);
        o_proj.setExecutionTarget(_execution_target);
        post_attention_layernorm.setExecutionTarget(_execution_target);
        gate_proj.setExecutionTarget(_execution_target);
        up_proj.setExecutionTarget(_execution_target);
        down_proj.setExecutionTarget(_execution_target);
        swiglu.setExecutionTarget(_execution_target);
        cache_x = Tensor(_execution_target);
        cache_h1 = Tensor(_execution_target);
        cache_act = Tensor(_execution_target);
        cache_q_rope = Tensor(_execution_target);
        cache_k_rope = Tensor(_execution_target);
        cache_v = Tensor(_execution_target);
        cache_attn_out = Tensor(_execution_target);
        cache_l_stats = Tensor(_execution_target);
        cache_scaled_attn = Tensor(_execution_target);
        cache_attn_permuted = Tensor(_execution_target);
        cache_scaled_ffn = Tensor(_execution_target);
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
        cache_dx = Tensor(_execution_target);
    }

    std::unique_ptr<ILayer> clone() const override
    {
        auto copy = std::make_unique<Transformer_Block>(hidden_dim, num_heads, intermediate_dim,
                                                        rms_norm_eps, rope_base, execution_target, data_type, num_layers);
        return copy;
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        uint64_t h_dim = hidden_dim;
        uint64_t n_heads = num_heads;
        uint64_t i_dim = intermediate_dim;
        _output_file_stream.write(reinterpret_cast<const char *>(&h_dim), sizeof(h_dim));
        _output_file_stream.write(reinterpret_cast<const char *>(&n_heads), sizeof(n_heads));
        _output_file_stream.write(reinterpret_cast<const char *>(&i_dim), sizeof(i_dim));
        _output_file_stream.write(reinterpret_cast<const char *>(&rms_norm_eps), sizeof(rms_norm_eps));
        _output_file_stream.write(reinterpret_cast<const char *>(&rope_base), sizeof(rope_base));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        input_layernorm.saveInference(_output_file_stream);
        q_proj.saveInference(_output_file_stream);
        k_proj.saveInference(_output_file_stream);
        v_proj.saveInference(_output_file_stream);
        o_proj.saveInference(_output_file_stream);
        post_attention_layernorm.saveInference(_output_file_stream);
        gate_proj.saveInference(_output_file_stream);
        up_proj.saveInference(_output_file_stream);
        down_proj.saveInference(_output_file_stream);
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        input_layernorm.loadInference(_input_file_stream);
        q_proj.loadInference(_input_file_stream);
        k_proj.loadInference(_input_file_stream);
        v_proj.loadInference(_input_file_stream);
        o_proj.loadInference(_input_file_stream);
        post_attention_layernorm.loadInference(_input_file_stream);
        gate_proj.loadInference(_input_file_stream);
        up_proj.loadInference(_input_file_stream);
        down_proj.loadInference(_input_file_stream);
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        input_layernorm.saveCheckpoint(_output_file_stream);
        q_proj.saveCheckpoint(_output_file_stream);
        k_proj.saveCheckpoint(_output_file_stream);
        v_proj.saveCheckpoint(_output_file_stream);
        o_proj.saveCheckpoint(_output_file_stream);
        post_attention_layernorm.saveCheckpoint(_output_file_stream);
        gate_proj.saveCheckpoint(_output_file_stream);
        up_proj.saveCheckpoint(_output_file_stream);
        down_proj.saveCheckpoint(_output_file_stream);
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        input_layernorm.loadCheckpoint(_input_file_stream);
        q_proj.loadCheckpoint(_input_file_stream);
        k_proj.loadCheckpoint(_input_file_stream);
        v_proj.loadCheckpoint(_input_file_stream);
        o_proj.loadCheckpoint(_input_file_stream);
        post_attention_layernorm.loadCheckpoint(_input_file_stream);
        gate_proj.loadCheckpoint(_input_file_stream);
        up_proj.loadCheckpoint(_input_file_stream);
        down_proj.loadCheckpoint(_input_file_stream);
    }

    size_t getHiddenDim() const noexcept { return hidden_dim; }
    size_t getNumHeads() const noexcept { return num_heads; }
    size_t getHeadDim() const noexcept { return head_dim; }
    size_t getIntermediateDim() const noexcept { return intermediate_dim; }

    size_t getParameterCount() const noexcept
    {
        return input_layernorm.getWeights().getTotalElements()
             + q_proj.getWeights().getTotalElements() + q_proj.getBiases().getTotalElements()
             + k_proj.getWeights().getTotalElements() + k_proj.getBiases().getTotalElements()
             + v_proj.getWeights().getTotalElements() + v_proj.getBiases().getTotalElements()
             + o_proj.getWeights().getTotalElements() + o_proj.getBiases().getTotalElements()
             + post_attention_layernorm.getWeights().getTotalElements()
             + gate_proj.getWeights().getTotalElements() + gate_proj.getBiases().getTotalElements()
             + up_proj.getWeights().getTotalElements() + up_proj.getBiases().getTotalElements()
             + down_proj.getWeights().getTotalElements() + down_proj.getBiases().getTotalElements();
    }

    const RMSNorm_Layer &getInputLayernorm() const noexcept { return input_layernorm; }
    RMSNorm_Layer &getInputLayernorm() noexcept { return input_layernorm; }
    const Linear_Layer &getQProj() const noexcept { return q_proj; }
    Linear_Layer &getQProj() noexcept { return q_proj; }
    const Linear_Layer &getKProj() const noexcept { return k_proj; }
    Linear_Layer &getKProj() noexcept { return k_proj; }
    const Linear_Layer &getVProj() const noexcept { return v_proj; }
    Linear_Layer &getVProj() noexcept { return v_proj; }
    const Linear_Layer &getOProj() const noexcept { return o_proj; }
    Linear_Layer &getOProj() noexcept { return o_proj; }
    const RMSNorm_Layer &getPostAttentionLayernorm() const noexcept { return post_attention_layernorm; }
    RMSNorm_Layer &getPostAttentionLayernorm() noexcept { return post_attention_layernorm; }
    const Linear_Layer &getGateProj() const noexcept { return gate_proj; }
    Linear_Layer &getGateProj() noexcept { return gate_proj; }
    const Linear_Layer &getUpProj() const noexcept { return up_proj; }
    Linear_Layer &getUpProj() noexcept { return up_proj; }
    const Linear_Layer &getDownProj() const noexcept { return down_proj; }
    Linear_Layer &getDownProj() noexcept { return down_proj; }
};

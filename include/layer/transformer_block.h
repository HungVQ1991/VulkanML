#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "layer/ilayer.h"
#include "layer/linear_layer.h"
#include "layer/lora_linear_layer.h"
#include "layer/rmsnorm_layer.h"
#include "llm/kv_cache_manager.h"
#include "llm/llm_configs.h"
#include "math/tensor.h"

struct Transformer_Block_Config
{
    size_t hidden_dim = 0;
    size_t num_heads = 0;
    size_t head_dim = 0;
    size_t num_kv_heads = 0;
    size_t intermediate_dim = 0;
    float rms_norm_eps = 1e-5f;
    float rope_base = 10000.0f;
    ops::llm::YaRN_Config yarn{};
    Execution_Target execution_target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT32;
    size_t num_layers = 1;
    float residual_scale = -1.0f;
    bool use_fused_projections = false;
    bool use_qk_norm = false;
    bool use_attention_bias = true;
    bool use_mlp_bias = true;
    bool gradient_checkpointing = false;
};

struct Transformer_Scratchpad
{
    Tensor fwd_scaled;
    Tensor dy_branch;
    Tensor dg;
    Tensor du;
    Tensor d_norm_h1;
    Tensor dh1;
    Tensor dh1_branch;
    Tensor d_attn_permuted;
    Tensor dq;
    Tensor dk;
    Tensor dv;
    Tensor dq_unrope;
    Tensor dk_unrope;
    Tensor dv_permuted;
    Tensor d_norm_qk;
    Tensor d_norm_attn;
    Tensor d_gate_up;
    Tensor dqkv;
    std::array<Tensor, 2> dx_slots;
    std::array<Tensor, 2> dx_scaled_slots;

    explicit Transformer_Scratchpad(Execution_Target target = Execution_Target::CPU)
        : fwd_scaled(target),
          dy_branch(target),
          dg(target),
          du(target),
          d_norm_h1(target),
          dh1(target),
          dh1_branch(target),
          d_attn_permuted(target),
          dq(target),
          dk(target),
          dv(target),
          dq_unrope(target),
          dk_unrope(target),
          dv_permuted(target),
          d_norm_qk(target),
          d_norm_attn(target),
          d_gate_up(target),
          dqkv(target),
          dx_slots{ Tensor(target), Tensor(target) },
          dx_scaled_slots{ Tensor(target), Tensor(target) }
    {
    }

    void setExecutionTarget(Execution_Target target)
    {
        fwd_scaled = Tensor(target);
        dy_branch = Tensor(target);
        dg = Tensor(target);
        du = Tensor(target);
        d_norm_h1 = Tensor(target);
        dh1 = Tensor(target);
        dh1_branch = Tensor(target);
        d_attn_permuted = Tensor(target);
        dq = Tensor(target);
        dk = Tensor(target);
        dv = Tensor(target);
        dq_unrope = Tensor(target);
        dk_unrope = Tensor(target);
        dv_permuted = Tensor(target);
        d_norm_qk = Tensor(target);
        d_norm_attn = Tensor(target);
        d_gate_up = Tensor(target);
        dqkv = Tensor(target);
        dx_slots[0] = Tensor(target);
        dx_slots[1] = Tensor(target);
        dx_scaled_slots[0] = Tensor(target);
        dx_scaled_slots[1] = Tensor(target);
    }
};

class Transformer_Block : public ILayer
{
private:
    size_t hidden_dim;
    size_t num_heads;
    size_t head_dim;
    size_t num_kv_heads;
    size_t intermediate_dim;
    float rms_norm_eps;
    float rope_base;
    ops::llm::YaRN_Config yarn_config{};
    Execution_Target execution_target;
    Data_Type data_type;
    size_t num_layers = 1;
    float residual_scale = 1.0f;
    bool use_fused_projections = false;
    bool use_qk_norm = false;
    bool use_attention_bias = true;
    bool use_mlp_bias = true;
    bool gradient_checkpointing = false;
    bool checkpoint_recompute_active = false;
    bool is_training_mode = false;

    RMSNorm_Layer input_layernorm;
    LoRA_Linear_Layer q_proj;
    LoRA_Linear_Layer k_proj;
    LoRA_Linear_Layer v_proj;
    Linear_Layer qkv_proj;
    LoRA_Linear_Layer o_proj;
    std::unique_ptr<RMSNorm_Layer> q_norm;
    std::unique_ptr<RMSNorm_Layer> k_norm;

    RMSNorm_Layer post_attention_layernorm;
    LoRA_Linear_Layer gate_proj;
    LoRA_Linear_Layer up_proj;
    Linear_Layer gate_up_proj;
    LoRA_Linear_Layer down_proj;

    mutable KV_Cache_Manager *active_kv_cache = nullptr;
    mutable Tensor cache_x;
    mutable Tensor cache_h1;
    mutable Tensor cache_act;
    mutable Tensor cache_q_rope;
    mutable Tensor cache_k_rope;
    mutable Tensor cache_v;
    mutable Tensor cache_k_all;
    mutable Tensor cache_v_all;
    mutable Tensor cache_attn_out;
    mutable Tensor cache_l_stats;
    mutable size_t cache_batch_size = 1;
    mutable size_t cache_seq_len = 1;
    mutable bool cache_is_3d = false;
    mutable bool is_forward_completed = false;
    mutable bool last_forward_used_kv_cache = false;

    mutable bool cache_is_checkpointed = false;
    mutable bool cache_ckpt_has_branch = false;
    mutable bool cache_ckpt_fuse_tail = true;
    mutable float cache_ckpt_branch_scale = 1.0f;
    mutable Tensor cache_ckpt_input;
    mutable Tensor cache_ckpt_branch;

    mutable bool cache_has_position_ids = false;
    mutable Tensor cache_position_ids;
    mutable bool cache_has_attn_override = false;
    mutable ops::llm::Attention_Mask_Mode cache_attn_mask_mode = ops::llm::Attention_Mask_Mode::NONE;
    mutable bool cache_attn_has_mask = false;
    mutable Tensor cache_attn_mask;
    mutable bool cache_attn_has_kv_index = false;
    mutable Tensor cache_attn_kv_index;
    mutable bool cache_attn_is_causal = true;
    mutable uint32_t cache_attn_causal_offset = 0;
    mutable float cache_attn_scale = 0.0f;

    mutable Tensor cache_scaled_attn;
    mutable Tensor cache_attn_permuted;
    mutable Tensor cache_scaled_ffn;
    mutable Tensor cache_ffn_out;
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
    mutable Tensor cache_d_gate_up;
    mutable Tensor cache_dqkv;
    mutable Tensor cache_dx;
    mutable Tensor cache_dx_scaled;
    mutable Transformer_Scratchpad *shared_scratchpad = nullptr;
    size_t scratch_dx_slot = 0;

    void clearIntermediateForwardCaches();

    template <typename Fn>
    void forEachNorm(Fn &&fn)
    {
        fn(input_layernorm);
        fn(post_attention_layernorm);
        if (q_norm)
        {
            fn(*q_norm);
        }
        if (k_norm)
        {
            fn(*k_norm);
        }
    }

    template <typename Fn>
    void forEachLoRAProjection(Fn &&fn)
    {
        fn(o_proj);
        fn(down_proj);
        if (!use_fused_projections)
        {
            fn(q_proj);
            fn(k_proj);
            fn(v_proj);
            fn(gate_proj);
            fn(up_proj);
        }
    }

    template <typename Fn>
    void forEachActiveProjection(Fn &&fn)
    {
        fn(static_cast<ILayer &>(o_proj));
        fn(static_cast<ILayer &>(down_proj));
        if (use_fused_projections)
        {
            fn(static_cast<ILayer &>(qkv_proj));
            fn(static_cast<ILayer &>(gate_up_proj));
        }
        else
        {
            fn(static_cast<ILayer &>(q_proj));
            fn(static_cast<ILayer &>(k_proj));
            fn(static_cast<ILayer &>(v_proj));
            fn(static_cast<ILayer &>(gate_proj));
            fn(static_cast<ILayer &>(up_proj));
        }
    }

    std::pair<Tensor, Tensor> executeForwardCore(const Tensor &x_or_res,
                                                 const Tensor *branch_in,
                                                 float branch_scale,
                                                 KV_Cache_Manager *kv_cache,
                                                 bool fuse_tail,
                                                 size_t override_batch = 0,
                                                 size_t override_seq = 0,
                                                 const Tensor *position_ids = nullptr,
                                                 const ops::llm::Attention_Config *attn_override = nullptr);

    std::pair<Tensor, Tensor> executeBackwardCore(const Tensor &_output_gradient,
                                                  const Tensor *_prescaled_branch_grad,
                                                  bool _produce_scaled_dx);

public:
    explicit Transformer_Block(const Transformer_Block_Config &_config);
    Transformer_Block(size_t _hidden_dim,
                      size_t _num_heads,
                      size_t _intermediate_dim,
                      float _rms_norm_eps = 1e-5f,
                      float _rope_base = 10000.0f,
                      Execution_Target _execution_target = Execution_Target::CPU,
                      Data_Type _data_type = Data_Type::FLOAT32,
                      size_t _num_layers = 1,
                      size_t _num_kv_heads = 0,
                      bool _use_fused_projections = false);
    bool usesFusedProjections() const noexcept { return use_fused_projections; }
    void enableFusedProjections(bool _enable);
    void setResidualScale(float scale) noexcept;
    float getResidualScale() const noexcept;
    void setRmsNormEps(float _eps);
    float getRmsNormEps() const noexcept { return rms_norm_eps; }
    void setRopeBase(float _rope_base);
    float getRopeBase() const noexcept { return rope_base; }
    void setYaRNConfig(const ops::llm::YaRN_Config &_yarn) noexcept { yarn_config = _yarn; }
    const ops::llm::YaRN_Config &getYaRNConfig() const noexcept { return yarn_config; }
    void setGradientCheckpointing(bool _enable) noexcept { gradient_checkpointing = _enable; }
    bool isGradientCheckpointingEnabled() const noexcept { return gradient_checkpointing; }
    bool usesQKNorm() const noexcept { return use_qk_norm; }
    bool usesAttentionBias() const noexcept { return use_attention_bias; }
    bool usesMlpBias() const noexcept { return use_mlp_bias; }

    void enableLoRA(size_t _rank,
                    float _alpha,
                    float _dropout_rate = 0.0f,
                    const std::vector<std::string> &_target_modules = { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" },
                    bool _freeze_base = true);
    void disableLoRA();
    void mergeLoRA();
    void unmergeLoRA();
    bool hasActiveLoRA() const noexcept;
    void setBaseTrainable(bool _trainable) noexcept;

    void setSharedScratchpad(Transformer_Scratchpad *_pad, size_t _dx_slot_index = 0) noexcept { shared_scratchpad = _pad; scratch_dx_slot = _dx_slot_index & 1u; }
    void setKVCache(KV_Cache_Manager *cache) noexcept;
    KV_Cache_Manager *getKVCache() const noexcept;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setMixedPrecision(bool _enable) noexcept override;
    void setTrainable(bool _trainable) noexcept override;
    void invalidateWeightCache() noexcept override;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &x, KV_Cache_Manager *kv_cache);
    Tensor forward(const Tensor &x,
                   KV_Cache_Manager *kv_cache,
                   const Tensor *position_ids,
                   const ops::llm::Attention_Config *attn_override = nullptr);
    std::pair<Tensor, Tensor> forwardResidualPair(const Tensor &_residual,
                                                  const Tensor &_branch,
                                                  float _branch_scale,
                                                  KV_Cache_Manager *_kv_cache = nullptr,
                                                  bool _fuse_incoming = true,
                                                  size_t _batch_size = 0,
                                                  size_t _seq_len = 0,
                                                  const Tensor *_position_ids = nullptr,
                                                  const ops::llm::Attention_Config *_attn_override = nullptr);
    Tensor backward(const Tensor &_output_gradient) override;
    std::pair<Tensor, Tensor> backwardResidualChained(const Tensor &_output_gradient,
                                                      const Tensor *_prescaled_branch_grad = nullptr,
                                                      bool _produce_scaled_dx = false);
    void resetGradient() override;
    void resetGradients() override;
    void setTrainingMode(bool _is_training) override;
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<const Tensor *> getNoDecayParameters() const override;
    const Tensor &getInput() const override { return cache_x; }
    const Tensor &getOutput() const override { return cache_h2; }

    bool hasParameters() const override { return true; }
    bool supportsStaticGraph() const noexcept override
    {
        if (gradient_checkpointing || hasActiveLoRA() || use_qk_norm || yarn_config.enabled)
        {
            return false;
        }
        bool ok = input_layernorm.supportsStaticGraph() &&
                  o_proj.supportsStaticGraph() &&
                  post_attention_layernorm.supportsStaticGraph() &&
                  down_proj.supportsStaticGraph();
        if (use_fused_projections)
        {
            return ok && qkv_proj.supportsStaticGraph() && gate_up_proj.supportsStaticGraph();
        }
        return ok &&
               q_proj.supportsStaticGraph() && k_proj.supportsStaticGraph() &&
               v_proj.supportsStaticGraph() && gate_proj.supportsStaticGraph() &&
               up_proj.supportsStaticGraph();
    }
    bool supportsPopulationBatch() const noexcept override { return false; }
    Layer_Type getLayerType() const override { return Layer_Type::TRANSFORMER_BLOCK; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    void setExecutionTarget(Execution_Target _execution_target) override;
    std::unique_ptr<ILayer> clone() const override;
    void prepareInference() override;
    void quantizeToTernary() override;
    void dequantizeFromTernary() override;
    bool isTernary() const noexcept override { return data_type == Data_Type::TERNARY_INT2; }
    Data_Type getDataType() const noexcept { return data_type; }
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    size_t getHiddenDim() const noexcept { return hidden_dim; }
    size_t getNumHeads() const noexcept { return num_heads; }
    size_t getNumKvHeads() const noexcept { return num_kv_heads; }
    size_t getHeadDim() const noexcept { return head_dim; }
    size_t getIntermediateDim() const noexcept { return intermediate_dim; }
    size_t getParameterCount() const noexcept;
    const RMSNorm_Layer &getInputLayernorm() const noexcept;
    RMSNorm_Layer &getInputLayernorm() noexcept;
    const Linear_Layer &getQProj() const noexcept;
    Linear_Layer &getQProj() noexcept;
    const Linear_Layer &getKProj() const noexcept;
    Linear_Layer &getKProj() noexcept;
    const Linear_Layer &getVProj() const noexcept;
    Linear_Layer &getVProj() noexcept;
    const Linear_Layer &getOProj() const noexcept;
    Linear_Layer &getOProj() noexcept;
    const RMSNorm_Layer &getPostAttentionLayernorm() const noexcept;
    RMSNorm_Layer &getPostAttentionLayernorm() noexcept;
    const Linear_Layer &getGateProj() const noexcept;
    Linear_Layer &getGateProj() noexcept;
    const Linear_Layer &getUpProj() const noexcept;
    Linear_Layer &getUpProj() noexcept;
    const Linear_Layer &getDownProj() const noexcept;
    Linear_Layer &getDownProj() noexcept;
    const Linear_Layer &getQkvProj() const noexcept;
    Linear_Layer &getQkvProj() noexcept;
    const Linear_Layer &getGateUpProj() const noexcept;
    Linear_Layer &getGateUpProj() noexcept;

    LoRA_Linear_Layer &getQLoRA() noexcept { return q_proj; }
    const LoRA_Linear_Layer &getQLoRA() const noexcept { return q_proj; }
    LoRA_Linear_Layer &getKLoRA() noexcept { return k_proj; }
    const LoRA_Linear_Layer &getKLoRA() const noexcept { return k_proj; }
    LoRA_Linear_Layer &getVLoRA() noexcept { return v_proj; }
    const LoRA_Linear_Layer &getVLoRA() const noexcept { return v_proj; }
    LoRA_Linear_Layer &getOLoRA() noexcept { return o_proj; }
    const LoRA_Linear_Layer &getOLoRA() const noexcept { return o_proj; }
    LoRA_Linear_Layer &getGateLoRA() noexcept { return gate_proj; }
    const LoRA_Linear_Layer &getGateLoRA() const noexcept { return gate_proj; }
    LoRA_Linear_Layer &getUpLoRA() noexcept { return up_proj; }
    const LoRA_Linear_Layer &getUpLoRA() const noexcept { return up_proj; }
    LoRA_Linear_Layer &getDownLoRA() noexcept { return down_proj; }
    const LoRA_Linear_Layer &getDownLoRA() const noexcept { return down_proj; }

    RMSNorm_Layer *getQNorm() noexcept { return q_norm.get(); }
    const RMSNorm_Layer *getQNorm() const noexcept { return q_norm.get(); }
    RMSNorm_Layer *getKNorm() noexcept { return k_norm.get(); }
    const RMSNorm_Layer *getKNorm() const noexcept { return k_norm.get(); }
};

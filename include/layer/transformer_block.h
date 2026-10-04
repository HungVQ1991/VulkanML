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

public:    Transformer_Block(size_t _hidden_dim,
                      size_t _num_heads,
                      size_t _intermediate_dim,
                      float _rms_norm_eps = 1e-5f,
                      float _rope_base = 10000.0f,
                      Execution_Target _execution_target = Execution_Target::CPU,
                      Data_Type _data_type = Data_Type::FLOAT32,
                      size_t _num_layers = 1);
    void setResidualScale(float scale) noexcept;
    float getResidualScale() const noexcept;
    void setKVCache(KV_Cache_Manager *cache) noexcept;
    KV_Cache_Manager *getKVCache() const noexcept;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &x, KV_Cache_Manager *kv_cache);
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    void setTrainingMode(bool _is_training) override;
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;


    bool hasParameters() const override { return true; }
    Layer_Type getLayerType() const override { return Layer_Type::TRANSFORMER_BLOCK; }
    Execution_Target getExecutionTarget() const override { return execution_target; }    void setExecutionTarget(Execution_Target _execution_target) override;
    std::unique_ptr<ILayer> clone() const override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;


    size_t getHiddenDim() const noexcept { return hidden_dim; }
    size_t getNumHeads() const noexcept { return num_heads; }
    size_t getHeadDim() const noexcept { return head_dim; }
    size_t getIntermediateDim() const noexcept { return intermediate_dim; }    size_t getParameterCount() const noexcept;
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

};;

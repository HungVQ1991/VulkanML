#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <functional>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>

#include "math/tensor.h"
#include "layer/embedding_layer.h"
#include "layer/transformer_block.h"
#include "layer/rmsnorm_layer.h"
#include "layer/linear_layer.h"
#include "llm/kv_cache_manager.h"
#include "tokenizer/bpe_tokenizer.h"
#include "engine/execution_engine.h"
#include "engine/loss_scaler.h"
#include "optimizer/ioptimizer.h"
#include "helper/training_profiler.h"
#include "cost_function/fused_cce_cost.h"

struct Causal_LM_Config
{
    size_t vocab_size = 32768;
    size_t hidden_dim = 64;
    size_t num_heads = 4;
    size_t intermediate_dim = 128;
    size_t num_layers = 2;
    size_t max_seq_len = 256;
    float rms_norm_eps = 1e-5f;
    float rope_base = 10000.0f;
    int32_t ignore_index = -1;
    Execution_Target execution_target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT32;
    bool use_loss_scaler = true;
    float initial_loss_scale = 1024.0f;
    float loss_scale_growth_factor = 2.0f;
    float loss_scale_backoff_factor = 0.5f;
    uint32_t loss_scale_growth_interval = 2000;
    std::string tokenizer_path = "tokenizer/tokenizer.json";
};

class Causal_LM
{
private:
    Causal_LM_Config config;

    Embedding_Layer token_embedding;
    std::vector<std::unique_ptr<Transformer_Block>> blocks;
    std::vector<std::unique_ptr<KV_Cache_Manager>> kv_caches;
    RMSNorm_Layer final_norm;
    Linear_Layer lm_head;
    Bpe_Tokenizer tokenizer;
    std::unique_ptr<Loss_Scaler> loss_scaler;
    std::unique_ptr<ICost_Function> cost_function;
    bool is_training_mode = false;
    mutable Tensor d_logits_tensor;    int32_t sampleToken(std::vector<float>& logits, float temperature, float top_p, size_t top_k = 0) const;


public:    
    explicit Causal_LM(const Causal_LM_Config& _config);


      Loss_Scaler* getLossScaler() const noexcept { return loss_scaler.get(); }
      void setLossScaler(std::unique_ptr<Loss_Scaler> scaler) noexcept { loss_scaler = std::move(scaler); }    std::vector<std::pair<Tensor*, Tensor*>> getParametersAndGradients();
      void warmupWeightCaches();
      void invalidateWeightCaches();
      void setExecutionTarget(Execution_Target new_target);
      void setTrainingMode(bool is_training);


      bool getTrainingMode() const noexcept { return is_training_mode; }
      const Step_Timings& getLastStepTimings() const noexcept { return Step_Timings::getInstance(); }
      void setStepDataPrepMs(double ms) noexcept { Step_Timings::getInstance()[Timing_Stage::DATA_PREP] += ms; }    void resetKVCaches() noexcept;
      void resetGradients();


      void resetGradient()
      {
          resetGradients();
      }    std::vector<std::vector<std::pair<Tensor*, Tensor*>>> getParameterGroups();
      size_t getParameterCount() const;
      void setAccumulated(bool is_accumulated);
      Tensor forward(const Tensor& input_tensor, bool use_cache = false);
      Tensor forward(const std::vector<int32_t>& token_ids, bool use_cache = false);
      Tensor forward(const std::vector<std::vector<int32_t>>& batch_token_ids, bool use_cache = false);
      Tensor backward(const Tensor& d_logits, bool defer_execution = false);
      float computeLossAndGradient(const Tensor& logits,
          const std::vector<int32_t>& target_token_ids,
          Tensor& d_logits) const;
      float computeLossAndGradient(const Tensor& logits,
          const Tensor& target_tensor,
          Tensor& d_logits,
          uint32_t valid_tokens = 0) const;
      float clipGradients(float max_norm);
      bool stepOptimizer(IOptimizer& optimizer, float max_grad_norm = 1.0f, float loss_scale = -1.0f, bool overflow_hint = false, bool is_chained_with_backward = false);
      float forwardLossAndBackward(const std::vector<int32_t>& input_tokens,
          const std::vector<int32_t>& target_tokens,
          float loss_scale = 1.0f,
          bool defer_backward_execution = false);
      float forwardLossAndBackward(const Tensor& input_tensor,
          const Tensor& target_tensor,
          float loss_scale = 1.0f,
          bool defer_backward_execution = false);
      float trainStep(const std::vector<int32_t>& input_tokens,
          const std::vector<int32_t>& target_tokens,
          IOptimizer& optimizer,
          float max_grad_norm = 1.0f);
      float trainStep(const Tensor& input_tensor,
          const Tensor& target_tensor,
          IOptimizer& optimizer,
          float max_grad_norm = 1.0f);
      float trainStep(const std::vector<std::vector<int32_t>>& batch_inputs,
          const std::vector<std::vector<int32_t>>& batch_targets,
          IOptimizer& optimizer,
          float max_grad_norm = 1.0f);
      float trainStep(const std::string& text, IOptimizer& optimizer);
      void saveCheckpoint(const std::string& file_path) const;
      void loadCheckpoint(const std::string& file_path);
      void saveInference(const std::string& file_path) const;
      void loadInference(const std::string& file_path);
      std::string generate(const std::string& prompt,
          size_t max_new_tokens = 16,
          float temperature = 0.7f,
          float top_p = 0.9f,
          int32_t eos_token_id = 2,
          bool skip_special = true,
          const std::function<void(const std::string&)>& token_callback = nullptr,
          float repetition_penalty = 1.2f,
          size_t top_k = 0,
          const std::vector<std::string>& stop_sequences = {});


      const Causal_LM_Config& getConfig() const noexcept { return config; }
      Bpe_Tokenizer& getTokenizer() noexcept { return tokenizer; }
      const Bpe_Tokenizer& getTokenizer() const noexcept { return tokenizer; }

      const Embedding_Layer& getTokenEmbedding() const noexcept { return token_embedding; }
      Embedding_Layer& getTokenEmbedding() noexcept { return token_embedding; }

      const std::vector<std::unique_ptr<Transformer_Block>>& getBlocks() const noexcept { return blocks; }
      std::vector<std::unique_ptr<Transformer_Block>>& getBlocks() noexcept { return blocks; }

      const RMSNorm_Layer& getFinalNorm() const noexcept { return final_norm; }
      RMSNorm_Layer& getFinalNorm() noexcept { return final_norm; }

      const Linear_Layer& getLMHead() const noexcept { return lm_head; }
      Linear_Layer& getLMHead() noexcept { return lm_head; }

      const ICost_Function& getCostFunction() const noexcept { return *cost_function; }
      ICost_Function& getCostFunction() noexcept { return *cost_function; }
      void setCostFunction(std::unique_ptr<ICost_Function> _cost_function) { cost_function = std::move(_cost_function); }

      template <std::derived_from<ICost_Function> Cost_Type_T, typename... Args>
      Cost_Type_T& setCostFunction(Args&&... args)
      {
          auto cost = std::make_unique<Cost_Type_T>(std::forward<Args>(args)...);
          Cost_Type_T& cost_reference = *cost;
          cost_function = std::move(cost);
          return cost_reference;
      }
};
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine/async_data_pipeline.h"
#include "learning_rate/ilearning_rate.h"
#include "llm/binary_dataset.h"
#include "llm/causal_lm.h"
#include "optimizer/ioptimizer.h"

class Lambda_Data_Pipeline final : public Async_Data_Pipeline
{
public:
    using Producer_Fn = std::function<void(size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets)>;

private:
    size_t batch_count = 0;
    Producer_Fn producer_fn;

protected:
    void prepareBatchHost(size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets) override;

public:
    Lambda_Data_Pipeline(size_t _batch_count,
                         Producer_Fn _producer_fn,
                         Execution_Target _execution_target = Execution_Target::VULKAN_GPU);
    ~Lambda_Data_Pipeline() override { stop(); }

    size_t getBatchSize() const override { return batch_count; }
};

struct LM_Sample
{
    std::vector<int32_t> inputs;
    std::vector<int32_t> targets;
};

struct LM_Train_Config
{
    size_t epochs = 1;
    size_t batch_size = 8;
    size_t seq_len = 128;
    size_t stride = 64;
    size_t max_batches_per_epoch = 0;
    size_t grad_accum = 1;
    float grad_clip = 1.0f;
    bool use_static_graph = false;
    bool gpu_profiling = false;
    size_t log_every_batches = 0;
    std::string checkpoint_best_path;
    std::string checkpoint_latest_path;
    std::string inference_model_path;
};

struct LM_Batch_Metrics
{
    size_t epoch = 0;
    size_t batch_index = 0;
    size_t total_batches = 0;
    size_t samples_processed = 0;
    float step_loss = 0.0f;
    float avg_loss = 0.0f;
    double tokens_per_sec = 0.0;
    bool step_applied = true;
};

struct LM_Epoch_Metrics
{
    size_t epoch = 0;
    size_t total_epochs = 0;
    size_t batches_run = 0;
    size_t samples_processed = 0;
    float avg_loss = 0.0f;
    float perplexity = 0.0f;
    double tokens_per_sec = 0.0;
    double elapsed_sec = 0.0;
    bool is_best = false;
};

struct LM_Train_Callbacks
{
    std::function<void(const LM_Batch_Metrics &)> on_batch;
    std::function<void(const LM_Epoch_Metrics &)> on_epoch_end;
    std::function<bool(const LM_Batch_Metrics &)> should_stop;
};

class LM_Trainer
{
private:
    Causal_LM &model;

public:
    explicit LM_Trainer(Causal_LM &_model) noexcept : model(_model) {}

    static std::vector<LM_Sample> buildSamples(const std::vector<int32_t> &tokens, size_t seq_len, size_t stride);
    float evaluateLoss(const std::vector<LM_Sample> &samples, size_t max_samples = 0);

    void fit(const std::vector<LM_Sample> &samples,
             IOptimizer &optimizer,
             ILearning_Rate *scheduler = nullptr,
             const LM_Train_Config &config = {},
             const LM_Train_Callbacks &callbacks = {});

    void fit(const Binary_Token_Dataset &dataset,
             const std::string &dataset_path,
             IOptimizer &optimizer,
             ILearning_Rate *scheduler = nullptr,
             const LM_Train_Config &config = {},
             const LM_Train_Callbacks &callbacks = {});

    void fitTokens(const std::vector<int32_t> &tokens,
                   IOptimizer &optimizer,
                   ILearning_Rate *scheduler = nullptr,
                   const LM_Train_Config &config = {},
                   const LM_Train_Callbacks &callbacks = {});
};

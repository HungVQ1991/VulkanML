#include "helper/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"

namespace
{
    void saveCheckpointSafely(Causal_LM &model, const std::string &path)
    {
        if (path.empty())
        {
            return;
        }
        if (model.getConfig().execution_target == Execution_Target::VULKAN_GPU)
        {
            auto &eng = Execution_Engine::getInstance();
            eng.waitIdle();
            eng.enableStaticGraph(false);
        }
        model.saveCheckpoint(path);
    }

    void saveInferenceSafely(Causal_LM &model, const std::string &path)
    {
        if (path.empty())
        {
            return;
        }
        if (model.getConfig().execution_target == Execution_Target::VULKAN_GPU)
        {
            auto &eng = Execution_Engine::getInstance();
            eng.waitIdle();
            eng.enableStaticGraph(false);
        }
        model.saveInference(path);
    }
}

Lambda_Data_Pipeline::Lambda_Data_Pipeline(size_t _batch_count,
                                           Producer_Fn _producer_fn,
                                           Execution_Target _execution_target)
    : Async_Data_Pipeline(_execution_target),
      batch_count(_batch_count),
      producer_fn(std::move(_producer_fn))
{
    if (!producer_fn)
    {
        throw std::invalid_argument("Lambda_Data_Pipeline: producer function must not be null");
    }
}

void Lambda_Data_Pipeline::prepareBatchHost(size_t batch_step, std::vector<float> &output_inputs, std::vector<float> &output_targets)
{
    producer_fn(batch_step, output_inputs, output_targets);
}

std::vector<LM_Sample> LM_Trainer::buildSamples(const std::vector<int32_t> &tokens, size_t seq_len, size_t stride)
{
    std::vector<LM_Sample> dataset;
    if (seq_len == 0 || tokens.size() < 2)
    {
        return dataset;
    }
    const size_t effective_stride = std::max<size_t>(1, stride);
    if (tokens.size() <= seq_len)
    {
        dataset.push_back({
            std::vector<int32_t>(tokens.begin(), tokens.end() - 1),
            std::vector<int32_t>(tokens.begin() + 1, tokens.end())
        });
        return dataset;
    }

    for (size_t start = 0; start + seq_len < tokens.size(); start += effective_stride)
    {
        dataset.push_back({
            std::vector<int32_t>(tokens.begin() + static_cast<ptrdiff_t>(start),
                                 tokens.begin() + static_cast<ptrdiff_t>(start + seq_len)),
            std::vector<int32_t>(tokens.begin() + static_cast<ptrdiff_t>(start + 1),
                                 tokens.begin() + static_cast<ptrdiff_t>(start + seq_len + 1))
        });
    }
    return dataset;
}

float LM_Trainer::evaluateLoss(const std::vector<LM_Sample> &samples, size_t max_samples)
{
    if (samples.empty())
    {
        return 0.0f;
    }
    const auto &cfg = model.getConfig();
    const size_t seq_len = cfg.max_seq_len;
    const size_t count = (max_samples > 0) ? std::min(max_samples, samples.size()) : std::min<size_t>(8, samples.size());
    Tensor b_in(Shape{count, seq_len}, cfg.execution_target);
    Tensor b_tgt(Shape{count, seq_len}, cfg.execution_target);
    std::vector<float> in_d(count * seq_len, 0.0f);
    std::vector<float> tgt_d(count * seq_len, static_cast<float>(cfg.ignore_index));
    for (size_t b = 0; b < count; ++b)
    {
        for (size_t s = 0; s < seq_len && s < samples[b].inputs.size(); ++s)
        {
            in_d[b * seq_len + s] = static_cast<float>(samples[b].inputs[s]);
        }
        for (size_t s = 0; s < seq_len && s < samples[b].targets.size(); ++s)
        {
            tgt_d[b * seq_len + s] = static_cast<float>(samples[b].targets[s]);
        }
    }
    b_in.uploadData(in_d);
    b_tgt.uploadData(tgt_d);
    const bool prev_training = model.getTrainingMode();
    struct Eval_Mode_Guard
    {
        Causal_LM &m;
        bool saved_training;
        ~Eval_Mode_Guard()
        {
            try
            {
                m.setTrainingMode(saved_training);
            }
            catch (...)
            {
            }
        }
    } mode_guard{model, prev_training};
    if (prev_training)
    {
        model.setTrainingMode(false);
    }
    Tensor logits = model.forward(b_in, false);
    Tensor dummy_grad;
    float loss = model.computeLossAndGradient(logits, b_tgt, dummy_grad);
    return loss;
}

void LM_Trainer::fit(const std::vector<LM_Sample> &samples,
                     IOptimizer &optimizer,
                     ILearning_Rate *scheduler,
                     const LM_Train_Config &config,
                     const LM_Train_Callbacks &callbacks)
{
    if (config.batch_size == 0 || config.seq_len == 0 || config.grad_accum == 0)
    {
        throw std::invalid_argument("LM_Trainer::fit: batch_size, seq_len, and grad_accum must be > 0");
    }
    if (samples.empty())
    {
        Logger::logMessage("LM_Trainer::fit: Sample list is empty; skipping training",
                           Log_Level::LOG_WARNING, true, 0, Log_Feature::TRAINING);
        return;
    }

    const Execution_Target target = model.getConfig().execution_target;
    const bool is_gpu_target = (target == Execution_Target::VULKAN_GPU);
    const float pad_target = static_cast<float>(model.getConfig().ignore_index);

    const size_t actual_batch_size = std::min(config.batch_size, samples.size());
    size_t num_batches = (samples.size() + actual_batch_size - 1) / actual_batch_size;
    if (num_batches == 0)
    {
        num_batches = 1;
    }
    const size_t effective_batches = (config.max_batches_per_epoch > 0)
        ? std::min(config.max_batches_per_epoch, num_batches)
        : num_batches;

    Tensor batch_inputs(Shape{actual_batch_size, config.seq_len}, target);
    Tensor batch_targets(Shape{actual_batch_size, config.seq_len}, target);
    std::vector<float> in_data(actual_batch_size * config.seq_len, 0.0f);
    std::vector<float> tgt_data(actual_batch_size * config.seq_len, pad_target);

    if (is_gpu_target)
    {
        model.warmupWeightCaches();
    }

    const bool saved_training_mode = model.getTrainingMode();
    struct Static_Graph_Restore_Guard
    {
        Causal_LM &m;
        bool saved_training = false;
        bool is_gpu = false;
        bool saved_static = false;
        Static_Graph_Restore_Guard(Causal_LM &mdl, bool tr, bool gpu)
            : m(mdl), saved_training(tr), is_gpu(gpu)
        {
            if (is_gpu)
            {
                saved_static = Execution_Engine::getInstance().isStaticGraphEnabled();
            }
        }
        ~Static_Graph_Restore_Guard()
        {
            try
            {
                m.setAccumulated(false);
                m.setTrainingMode(saved_training);
            }
            catch (...)
            {
            }
            if (is_gpu)
            {
                try
                {
                    Execution_Engine::getInstance().enableStaticGraph(saved_static);
                }
                catch (...)
                {
                }
            }
        }
    } static_guard(model, saved_training_mode, is_gpu_target);

    float best_loss = std::numeric_limits<float>::infinity();
    bool stop_requested = false;

    for (size_t epoch = 1; epoch <= config.epochs && !stop_requested; ++epoch)
    {
        if (is_gpu_target)
        {
            Execution_Engine::getInstance().enableStaticGraph(config.use_static_graph);
        }
        model.setTrainingMode(true);

        const auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss_sum = 0.0f;
        size_t total_samples_processed = 0;
        size_t valid_loss_samples = 0;
        size_t batches_run = 0;
        Step_Timings::getInstance().reset();
        if (config.gpu_profiling && is_gpu_target)
        {
            auto &eng = Execution_Engine::getInstance();
            eng.flushGpuProfileResults();
            eng.resetGpuProfileStats();
        }

        for (size_t batch_idx = 0; batch_idx < effective_batches; ++batch_idx)
        {
            const auto batch_step_start = std::chrono::high_resolution_clock::now();
            const size_t start_idx = batch_idx * actual_batch_size;
            const size_t batch_samples = std::min(actual_batch_size, samples.size() - start_idx);
            total_samples_processed += batch_samples;

            const auto data_prep_start = std::chrono::high_resolution_clock::now();
            std::fill(in_data.begin(), in_data.end(), 0.0f);
            std::fill(tgt_data.begin(), tgt_data.end(), pad_target);
            for (size_t b = 0; b < batch_samples; ++b)
            {
                const auto &sample = samples[start_idx + b];
                for (size_t s = 0; s < config.seq_len && s < sample.inputs.size(); ++s)
                {
                    in_data[b * config.seq_len + s] = static_cast<float>(sample.inputs[s]);
                }
                for (size_t s = 0; s < config.seq_len && s < sample.targets.size(); ++s)
                {
                    tgt_data[b * config.seq_len + s] = static_cast<float>(sample.targets[s]);
                }
            }

            auto &timings = Step_Timings::getInstance();
            const double fence_wait_before_prep = timings.getDuration(Timing_Stage::BWD_FENCE_WAIT)
                                                + timings.getDuration(Timing_Stage::OPT_FENCE_WAIT)
                                                + timings.getDuration(Timing_Stage::FWD_LOSS_FENCE_WAIT);
            uint32_t step_frame = 0;
            if (is_gpu_target)
            {
                auto &eng = Execution_Engine::getInstance();
                step_frame = eng.getContext().getCurrentFrame();
                if (!eng.getContext().isFrameReady(step_frame))
                {
                    eng.prepareCurrentFrame();
                    step_frame = eng.getContext().getCurrentFrame();
                }
            }

            batch_inputs.uploadData(in_data);
            batch_targets.uploadData(tgt_data);
            const auto data_prep_end = std::chrono::high_resolution_clock::now();
            const double fence_wait_after_prep = timings.getDuration(Timing_Stage::BWD_FENCE_WAIT)
                                               + timings.getDuration(Timing_Stage::OPT_FENCE_WAIT)
                                               + timings.getDuration(Timing_Stage::FWD_LOSS_FENCE_WAIT);
            const double data_prep_ms = std::chrono::duration<double, std::milli>(data_prep_end - data_prep_start).count()
                                      - std::max(0.0, fence_wait_after_prep - fence_wait_before_prep);
            timings.recordDuration(Timing_Stage::DATA_PREP, std::max(0.0, data_prep_ms));

            float step_loss = 0.0f;
            bool step_applied = true;
            bool is_end_of_accum = true;
            if (config.grad_accum == 1 && is_gpu_target)
            {
                step_applied = false;
                const double total_batch_before = timings.getDuration(Timing_Stage::TOTAL_BATCH);
                step_loss = model.trainStep(batch_inputs, batch_targets, optimizer, config.grad_clip, &step_applied);
                timings.setDuration(Timing_Stage::TOTAL_BATCH, total_batch_before);
                if (step_applied)
                {
                    if (scheduler)
                    {
                        const auto sched_start = std::chrono::high_resolution_clock::now();
                        scheduler->step();
                        const auto sched_end = std::chrono::high_resolution_clock::now();
                        timings.recordDuration(Timing_Stage::SCHED_STEP,
                            std::chrono::duration<double, std::milli>(sched_end - sched_start).count());
                    }
                }
                else
                {
                    const auto reset_start = std::chrono::high_resolution_clock::now();
                    model.resetGradients();
                    const auto reset_end = std::chrono::high_resolution_clock::now();
                    timings.recordDuration(Timing_Stage::RESET_GRAD,
                        std::chrono::duration<double, std::milli>(reset_end - reset_start).count());
                }
            }
            else
            {
                const size_t window_start = (batch_idx / config.grad_accum) * config.grad_accum;
                const size_t window_len = std::min(config.grad_accum, effective_batches - window_start);
                const bool is_start_of_accum = (batch_idx % config.grad_accum == 0);
                is_end_of_accum = ((batch_idx + 1) % config.grad_accum == 0) ||
                                  (batch_idx + 1 == effective_batches);

                if (is_start_of_accum)
                {
                    model.setAccumulated(false);
                    const auto reset_start = std::chrono::high_resolution_clock::now();
                    if (is_gpu_target)
                    {
                        model.resetForwardState();
                    }
                    else
                    {
                        model.resetGradients();
                    }
                    const auto reset_end = std::chrono::high_resolution_clock::now();
                    timings.recordDuration(Timing_Stage::RESET_GRAD,
                        std::chrono::duration<double, std::milli>(reset_end - reset_start).count());
                }
                else
                {
                    model.setAccumulated(true);
                }

                const float current_scale = (model.getLossScaler() && model.getLossScaler()->isEnabled())
                    ? model.getLossScaler()->getScale()
                    : 1.0f;
                const bool defer_bwd = is_end_of_accum && is_gpu_target;
                uint32_t fwd_frame = is_gpu_target ? Execution_Engine::getInstance().getContext().getCurrentFrame() : step_frame;
                step_loss = model.forwardLossAndBackward(
                    batch_inputs, batch_targets, current_scale / static_cast<float>(window_len), defer_bwd);

                if (!defer_bwd && step_loss <= 0.0f && is_gpu_target && Execution_Engine::getInstance().isAsyncLossEnabled())
                {
                    step_loss = Execution_Engine::getInstance().readPendingLoss(fwd_frame);
                }

                if (is_end_of_accum)
                {
                    const auto opt_start = std::chrono::high_resolution_clock::now();
                    const bool overflow_hint = std::isnan(step_loss) || std::isinf(step_loss);
                    step_applied = model.stepOptimizer(optimizer, config.grad_clip, current_scale, overflow_hint, defer_bwd);
                    const auto opt_end = std::chrono::high_resolution_clock::now();
                    timings.recordDuration(Timing_Stage::OPT_EXEC,
                        std::chrono::duration<double, std::milli>(opt_end - opt_start).count());

                    if (step_applied && step_loss <= 0.0f && is_gpu_target && Execution_Engine::getInstance().isAsyncLossEnabled())
                    {
                        step_loss = Execution_Engine::getInstance().readPendingLoss(fwd_frame);
                    }

                    if (step_applied)
                    {
                        if (scheduler)
                        {
                            const auto sched_start = std::chrono::high_resolution_clock::now();
                            scheduler->step();
                            const auto sched_end = std::chrono::high_resolution_clock::now();
                            timings.recordDuration(Timing_Stage::SCHED_STEP,
                                std::chrono::duration<double, std::milli>(sched_end - sched_start).count());
                        }
                    }
                    else
                    {
                        const auto reset_start = std::chrono::high_resolution_clock::now();
                        model.resetGradients();
                        const auto reset_end = std::chrono::high_resolution_clock::now();
                        timings.recordDuration(Timing_Stage::RESET_GRAD,
                            std::chrono::duration<double, std::milli>(reset_end - reset_start).count());
                    }
                }
            }

            if (std::isfinite(step_loss) && (step_applied || !is_end_of_accum))
            {
                epoch_loss_sum += step_loss * static_cast<float>(batch_samples);
                valid_loss_samples += batch_samples;
            }
            ++batches_run;

            const auto batch_step_end = std::chrono::high_resolution_clock::now();
            timings.recordDuration(Timing_Stage::TOTAL_BATCH,
                std::chrono::duration<double, std::milli>(batch_step_end - batch_step_start).count());

            const double cur_sec = std::chrono::duration<double>(batch_step_end - epoch_start).count();
            const double cur_tok_s = (cur_sec > 0.0)
                ? (static_cast<double>(total_samples_processed * config.seq_len) / cur_sec)
                : 0.0;
            const float cur_avg_loss = (valid_loss_samples > 0)
                ? (epoch_loss_sum / static_cast<float>(valid_loss_samples))
                : 0.0f;

            const LM_Batch_Metrics batch_metrics{
                epoch,
                batch_idx,
                effective_batches,
                total_samples_processed,
                step_loss,
                cur_avg_loss,
                cur_tok_s,
                step_applied
            };

            if (config.log_every_batches > 0 &&
                (((batch_idx + 1) % config.log_every_batches == 0) || (batch_idx + 1 == effective_batches)))
            {
                Logger::logMessage(
                    Input_Format{"LM_Trainer: Epoch {}/{} Batch {}/{} | step_loss={:.4f} avg_loss={:.4f} speed={:.0f} tok/s",
                                 epoch, config.epochs, batch_idx + 1, effective_batches,
                                 step_loss, cur_avg_loss, cur_tok_s},
                    Log_Level::LOG_INFO, true, 0, Log_Feature::TRAINING);
            }

            if (callbacks.on_batch)
            {
                callbacks.on_batch(batch_metrics);
            }
            if (callbacks.should_stop && callbacks.should_stop(batch_metrics))
            {
                stop_requested = true;
                break;
            }
        }

        if (is_gpu_target)
        {
            Execution_Engine::getInstance().readPendingLoss(UINT32_MAX);
        }

        const auto epoch_end = std::chrono::high_resolution_clock::now();
        const double elapsed_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        const double tokens_per_sec = (elapsed_sec > 0.0)
            ? (static_cast<double>(total_samples_processed * config.seq_len) / elapsed_sec)
            : 0.0;
        const float avg_loss = (valid_loss_samples > 0)
            ? (epoch_loss_sum / static_cast<float>(valid_loss_samples))
            : 0.0f;
        const float ppl = std::exp(std::min(avg_loss, 20.0f));

        bool is_best = false;
        if (!stop_requested && valid_loss_samples > 0 && std::isfinite(avg_loss) && avg_loss < best_loss)
        {
            best_loss = avg_loss;
            is_best = true;
            saveCheckpointSafely(model, config.checkpoint_best_path);
        }

        if (is_gpu_target)
        {
            Execution_Engine::getInstance().enableStaticGraph(false);
        }

        const LM_Epoch_Metrics epoch_metrics{
            epoch,
            config.epochs,
            batches_run,
            total_samples_processed,
            avg_loss,
            ppl,
            tokens_per_sec,
            elapsed_sec,
            is_best
        };

        if (callbacks.on_epoch_end)
        {
            callbacks.on_epoch_end(epoch_metrics);
        }
        Logger::resetLogCounters();
    }

    if (is_gpu_target)
    {
        Execution_Engine::getInstance().enableStaticGraph(false);
    }
    saveCheckpointSafely(model, config.checkpoint_latest_path);
    saveInferenceSafely(model, config.inference_model_path);
}

void LM_Trainer::fit(const Binary_Token_Dataset &,
                     const std::string &dataset_path,
                     IOptimizer &optimizer,
                     ILearning_Rate *scheduler,
                     const LM_Train_Config &config,
                     const LM_Train_Callbacks &callbacks)
{
    std::vector<int32_t> tokens;
    if (!Binary_Token_Dataset::load(dataset_path, tokens) || tokens.size() < 2)
    {
        Logger::logMessage(Input_Format{"LM_Trainer::fit: Failed to load valid dataset from '{}'", dataset_path},
                           Log_Level::LOG_ERROR, true, 0, Log_Feature::DATASET_IO);
        throw std::runtime_error("LM_Trainer::fit: Failed to load valid dataset: " + dataset_path);
    }
    fitTokens(tokens, optimizer, scheduler, config, callbacks);
}

void LM_Trainer::fitTokens(const std::vector<int32_t> &tokens,
                           IOptimizer &optimizer,
                           ILearning_Rate *scheduler,
                           const LM_Train_Config &config,
                           const LM_Train_Callbacks &callbacks)
{
    auto samples = buildSamples(tokens, config.seq_len, config.stride);
    fit(samples, optimizer, scheduler, config, callbacks);
}

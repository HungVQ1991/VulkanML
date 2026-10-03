#include "llm/causal_lm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cost_function/fused_cce_cost.h"
#include "engine/execution_engine.h"
#include "engine/loss_scaler.h"
#include "helper/logger.h"
#include "helper/training_profiler.h"
#include "layer/embedding_layer.h"
#include "layer/linear_layer.h"
#include "layer/rmsnorm_layer.h"
#include "layer/transformer_block.h"
#include "llm/kv_cache_manager.h"
#include "math/tensor.h"
#include "optimizer/ioptimizer.h"
#include "tokenizer/bpe_tokenizer.h"

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
        if (sum_exp <= 0.0f || std::isnan(sum_exp))
        {
            return 0;
        }
        float inv_sum = 1.0f / sum_exp;
        for (float& val : logits)
        {
            val *= inv_sum;
        }

        std::vector<std::pair<float, int32_t>> prob_idx(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            prob_idx[i] = { logits[i], static_cast<int32_t>(i) };
        }
        std::sort(prob_idx.begin(), prob_idx.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
            });

        size_t cutoff = prob_idx.size();
        if (top_k > 0 && top_k < cutoff)
        {
            cutoff = top_k;
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

        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, std::max(total_prob, 1e-8f));
        float r = dist(rng);
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

Causal_LM::Causal_LM(const Causal_LM_Config& _config)
    : config(_config),
        token_embedding(_config.vocab_size, _config.hidden_dim, _config.execution_target, _config.data_type),
        final_norm(_config.hidden_dim, _config.rms_norm_eps, _config.execution_target, _config.data_type),
        lm_head(_config.hidden_dim, _config.vocab_size, _config.execution_target, 0.0004f * static_cast<float>(_config.hidden_dim), _config.data_type),
        cost_function(std::make_unique<Fused_Cross_Entropy>(_config.ignore_index))
{
        if (config.data_type == Data_Type::FLOAT16 && config.use_loss_scaler)
        {
            loss_scaler = std::make_unique<Loss_Scaler>(
                config.initial_loss_scale,
                config.loss_scale_growth_factor,
                config.loss_scale_backoff_factor,
                config.loss_scale_growth_interval,
                true);
        }

        blocks.reserve(config.num_layers);
        kv_caches.reserve(config.num_layers);
        for (size_t i = 0; i < config.num_layers; ++i)
        {
            blocks.push_back(std::make_unique<Transformer_Block>(
                config.hidden_dim,
                config.num_heads,
                config.intermediate_dim,
                config.rms_norm_eps,
                config.rope_base,
                config.execution_target,
                config.data_type,
                config.num_layers));

            kv_caches.push_back(std::make_unique<KV_Cache_Manager>(
                1,
                config.num_heads,
                config.max_seq_len,
                config.hidden_dim / config.num_heads,
                config.execution_target,
                config.data_type,
                KV_Cache_Mode::RING_BUFFER));
        }

        if (!config.tokenizer_path.empty())
        {
            tokenizer.load(config.tokenizer_path);
        }

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            warmupWeightCaches();
        }
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

void Causal_LM::setExecutionTarget(Execution_Target new_target)
{
        config.execution_target = new_target;
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
        invalidateWeightCaches();
        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            warmupWeightCaches();
        }
    }

void Causal_LM::setTrainingMode(bool is_training)
{
        is_training_mode = is_training;
        token_embedding.setTrainingMode(is_training);
        for (auto& block : blocks)
        {
            block->setTrainingMode(is_training);
        }
        final_norm.setTrainingMode(is_training);
        lm_head.setTrainingMode(is_training);
        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().setAsyncLossEnabled(is_training);
        }
    }

void Causal_LM::resetKVCaches() noexcept
{
        for (auto& cache : kv_caches)
        {
            cache->reset();
        }
    }

void Causal_LM::resetGradients()
{
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
        size_t count = 0;
        auto params = const_cast<Causal_LM*>(this)->getParametersAndGradients();
        for (const auto& p : params)
        {
            if (p.first)
            {
                count += p.first->getTotalElements();
            }
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

Tensor Causal_LM::forward(const Tensor& input_tensor, bool use_cache)
{
        if (input_tensor.isEmpty())
        {
            return Tensor(Shape{ 1, 0, config.vocab_size }, config.data_type, config.execution_target);
        }

        Tensor x = token_embedding.forward(input_tensor);
        for (size_t l = 0; l < blocks.size(); ++l)
        {
            KV_Cache_Manager* cache_ptr = use_cache ? kv_caches[l].get() : nullptr;
            x = blocks[l]->forward(x, cache_ptr);
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

        Tensor x_2d = x;
        if (x_2d.getShape().getRank() != 2 || x_2d.getColumns() != config.hidden_dim)
        {
            x_2d.reshape(Shape{ B * S, config.hidden_dim });
        }

        x_2d = final_norm.forward(x_2d);
        Tensor logits = lm_head.forward(x_2d);

        if (input_tensor.getShape().getRank() == 2)
        {
            logits.reshape(Shape{ B, S, config.vocab_size });
        }

        if (config.execution_target == Execution_Target::VULKAN_GPU && !is_training_mode)
        {
            Execution_Engine::getInstance().getCurrentGraph().print();
            Execution_Engine::getInstance().executeGraph();
        }

        return logits;
    }

Tensor Causal_LM::forward(const std::vector<int32_t>& token_ids, bool use_cache)
{
        if (token_ids.empty())
        {
            return Tensor(Shape{ 1, 0, config.vocab_size }, config.data_type, config.execution_target);
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
        if (batch_token_ids.empty())
        {
            return Tensor(Shape{ 0, 0, config.vocab_size }, config.data_type, config.execution_target);
        }

        size_t B = batch_token_ids.size();
        size_t S = batch_token_ids[0].size();
        Tensor in_tensor(Shape{ B, S }, config.execution_target);
        std::vector<float> f_data(B * S, 0.0f);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t s = 0; s < S && s < batch_token_ids[b].size(); ++s)
            {
                f_data[b * S + s] = static_cast<float>(batch_token_ids[b][s]);
            }
        }
        in_tensor.uploadData(f_data);
        return forward(in_tensor, use_cache);
    }

Tensor Causal_LM::backward(const Tensor& d_logits, bool defer_execution)
{
        auto t_bwd_rec_start = std::chrono::high_resolution_clock::now();
        size_t total_tokens = d_logits.getTotalElements() / config.vocab_size;
        Tensor d_logits_2d = d_logits;
        if (d_logits_2d.getShape().getRank() != 2 || d_logits_2d.getColumns() != config.vocab_size)
        {
            d_logits_2d.reshape(Shape{ total_tokens, config.vocab_size });
        }

        Tensor d_x = lm_head.backward(d_logits_2d);

        d_x = final_norm.backward(d_x);

        for (size_t i = blocks.size(); i > 0; --i)
        {
            d_x = blocks[i - 1]->backward(d_x);
        }

        token_embedding.backward(d_x);
        auto t_bwd_rec_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().bwd_record_ms += std::chrono::duration<double, std::milli>(t_bwd_rec_end - t_bwd_rec_start).count();

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            if (defer_execution)
            {
                Step_Timings::getInstance().bwd_gpu_nodes = Execution_Engine::getInstance().getCurrentGraph().getNodes().size();
            }
            else
            {
                Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, Execution_Stage::BACKWARD);
                Step_Timings::getInstance().bwd_gpu_nodes = Execution_Engine::getInstance().getLastExecutedNodeCount();
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
        return logits.fusedCrossEntropyLoss(target_token_ids, d_logits, valid);
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
        return logits.fusedCrossEntropyLoss(target_tensor, d_logits, valid_tokens);
    }

float Causal_LM::clipGradients(float max_norm)
{
        if (max_norm <= 0.0f)
        {
            return 0.0f;
        }

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            return 1.0f;
        }

        auto param_groups = getParameterGroups();
        double sum_squares = 0.0;
        bool has_nan_inf = false;
        size_t p_idx = 0;
        std::vector<double> group_sum_squares(param_groups.size(), 0.0);

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
                    group_sum_squares[g] += p_sum_sq;
                    sum_squares += p_sum_sq;

                    double p_norm = std::sqrt(p_sum_sq);
                    if (p_norm > 20.0)
                    {
                        std::cout << "[Large Param] #" << p_idx << " shape=" << grad->getShape().toString()
                            << " norm=" << p_norm << " max_abs=" << p_max_abs << "\n";
                    }
                    if (nan_count > 0)
                    {
                        std::cout << "[Overflow Pinpoint] Param #" << p_idx << " shape=" << grad->getShape().toString()
                            << " has " << nan_count << " NaN/Inf values! First val=" << data[0] << "\n";
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
        for (size_t g = 0; g < param_groups.size(); ++g)
        {
            float g_norm = static_cast<float>(std::sqrt(group_sum_squares[g]));
            if (g_norm > max_norm && g_norm > 1e-6f)
            {
                float scale = max_norm / g_norm;
                for (auto& [param, grad] : param_groups[g])
                {
                    if (grad && !grad->isEmpty())
                    {
                        grad->mulScalar(scale, *grad);
                    }
                }
                any_clipped = true;
            }
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
                optimizer.setMaxGradient(max_grad_norm);
            }
        }
        else
        {
            if (max_grad_norm > 0.0f)
            {
                float norm = clipGradients(max_grad_norm);
                if (std::isnan(norm) || std::isinf(norm))
                {
                    has_overflow = true;
                }
            }
            else if (loss_scaler && loss_scaler->isEnabled())
            {
                auto params_and_grads = getParametersAndGradients();
                for (auto& [param, grad] : params_and_grads)
                {
                    if (grad && !grad->isEmpty())
                    {
                        std::vector<float> data = grad->getData();
                        for (float val : data)
                        {
                            if (std::isnan(val) || std::isinf(val))
                            {
                                has_overflow = true;
                                break;
                            }
                        }
                        if (has_overflow)
                        {
                            break;
                        }
                    }
                }
            }
        }

        if (loss_scaler && loss_scaler->isEnabled())
        {
            loss_scaler->step(has_overflow);
        }

        if (has_overflow)
        {
            if (is_chained_with_backward && config.execution_target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().getCurrentGraph().clear();
            }
            return false;
        }

        if (is_chained_with_backward && config.execution_target == Execution_Target::VULKAN_GPU)
        {
            auto& nodes = Execution_Engine::getInstance().getCurrentGraph().getNodes();
            if (!nodes.empty())
            {
                nodes.back().is_barrier_required_after = true;
            }
        }

        auto t_opt_rec_start = std::chrono::high_resolution_clock::now();
        auto params_and_grads = getParametersAndGradients();
        optimizer.step(params_and_grads, actual_loss_scale);
        auto t_opt_rec_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().opt_record_ms += std::chrono::duration<double, std::milli>(t_opt_rec_end - t_opt_rec_start).count();

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Stage stage = is_chained_with_backward ? Execution_Stage::BACKWARD_OPTIMIZER : Execution_Stage::OPTIMIZER;
            Execution_Engine::getInstance().executeGraph(VK_NULL_HANDLE, stage);
            size_t total_nodes = Execution_Engine::getInstance().getLastExecutedNodeCount();
            if (is_chained_with_backward)
            {
                Step_Timings::getInstance().opt_gpu_nodes = (total_nodes >= Step_Timings::getInstance().bwd_gpu_nodes)
                    ? (total_nodes - Step_Timings::getInstance().bwd_gpu_nodes)
                    : total_nodes;
                Step_Timings::getInstance().bwd_gpu_nodes = total_nodes;
            }
            else
            {
                Step_Timings::getInstance().opt_gpu_nodes = total_nodes;
            }
        }
        return true;
    }

float Causal_LM::forwardLossAndBackward(const std::vector<int32_t>& input_tokens, const std::vector<int32_t>& target_tokens, float loss_scale, bool defer_backward_execution)
{
        auto t_flb_start = std::chrono::high_resolution_clock::now();
        setTrainingMode(true);

        auto t_fwd_rec_start = std::chrono::high_resolution_clock::now();
        Tensor logits = forward(input_tokens, false);
        auto t_fwd_rec_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().fwd_record_ms += std::chrono::duration<double, std::milli>(t_fwd_rec_end - t_fwd_rec_start).count();

        auto t_loss_start = std::chrono::high_resolution_clock::now();
        Execution_Engine::getInstance().setExecutionStage(Execution_Stage::FORWARD);
        float loss = computeLossAndGradient(logits, target_tokens, d_logits_tensor);
        Execution_Engine::getInstance().setExecutionStage(Execution_Stage::NONE);
        auto t_loss_end = std::chrono::high_resolution_clock::now();

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            double fwd_gpu = Execution_Engine::getInstance().getLastExecutionTimeMs();
            Step_Timings::getInstance().fwd_gpu_ms += fwd_gpu;
            Step_Timings::getInstance().fwd_loss_exec_ms += fwd_gpu;
            Step_Timings::getInstance().fwd_gpu_nodes = Execution_Engine::getInstance().getLastExecutedNodeCount();
            Step_Timings::getInstance().fwd_loss_gpu_nodes = Step_Timings::getInstance().fwd_gpu_nodes;
            if (!Execution_Engine::getInstance().isAsyncLossEnabled())
            {
                double total_loss_call = std::chrono::duration<double, std::milli>(t_loss_end - t_loss_start).count();
                double loss_read = std::max(0.0, total_loss_call - fwd_gpu);
                Step_Timings::getInstance().loss_read_ms += loss_read;
                Step_Timings::getInstance().loss_readback_ms += loss_read;
            }
        }

        if (loss_scale != 1.0f && loss_scale > 0.0f)
        {
            auto t_mul_start = std::chrono::high_resolution_clock::now();
            d_logits_tensor.mulScalar(loss_scale, d_logits_tensor);
            auto t_mul_end = std::chrono::high_resolution_clock::now();
            Step_Timings::getInstance().mul_scalar_ms += std::chrono::duration<double, std::milli>(t_mul_end - t_mul_start).count();
        }

        backward(d_logits_tensor, defer_backward_execution);
        auto t_flb_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().fwd_loss_bwd_ms += std::chrono::duration<double, std::milli>(t_flb_end - t_flb_start).count();
        float current_loss = Execution_Engine::getInstance().isAsyncLossEnabled()
            ? Execution_Engine::getInstance().getLatestLoss()
            : loss;
        if (Execution_Engine::getInstance().isAsyncLossEnabled() && current_loss <= 0.0f)
        {
            current_loss = Execution_Engine::getInstance().readPendingLoss(0);
        }
        return current_loss;
    }

float Causal_LM::forwardLossAndBackward(const Tensor& input_tensor, const Tensor& target_tensor, float loss_scale, bool defer_backward_execution)
{
        auto t_flb_start = std::chrono::high_resolution_clock::now();
        setTrainingMode(true);

        auto t_fwd_rec_start = std::chrono::high_resolution_clock::now();
        Tensor logits = forward(input_tensor, false);
        auto t_fwd_rec_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().fwd_record_ms += std::chrono::duration<double, std::milli>(t_fwd_rec_end - t_fwd_rec_start).count();

        auto t_loss_start = std::chrono::high_resolution_clock::now();
        Execution_Engine::getInstance().setExecutionStage(Execution_Stage::FORWARD);
        float loss = computeLossAndGradient(logits, target_tensor, d_logits_tensor);
        Execution_Engine::getInstance().setExecutionStage(Execution_Stage::NONE);
        auto t_loss_end = std::chrono::high_resolution_clock::now();

        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            double fwd_gpu = Execution_Engine::getInstance().getLastExecutionTimeMs();
            Step_Timings::getInstance().fwd_gpu_ms += fwd_gpu;
            Step_Timings::getInstance().fwd_loss_exec_ms += fwd_gpu;
            Step_Timings::getInstance().fwd_gpu_nodes = Execution_Engine::getInstance().getLastExecutedNodeCount();
            Step_Timings::getInstance().fwd_loss_gpu_nodes = Step_Timings::getInstance().fwd_gpu_nodes;
            if (!Execution_Engine::getInstance().isAsyncLossEnabled())
            {
                double total_loss_call = std::chrono::duration<double, std::milli>(t_loss_end - t_loss_start).count();
                double loss_read = std::max(0.0, total_loss_call - fwd_gpu);
                Step_Timings::getInstance().loss_read_ms += loss_read;
                Step_Timings::getInstance().loss_readback_ms += loss_read;
            }
        }

        if (loss_scale != 1.0f && loss_scale > 0.0f)
        {
            auto t_mul_start = std::chrono::high_resolution_clock::now();
            d_logits_tensor.mulScalar(loss_scale, d_logits_tensor);
            auto t_mul_end = std::chrono::high_resolution_clock::now();
            Step_Timings::getInstance().mul_scalar_ms += std::chrono::duration<double, std::milli>(t_mul_end - t_mul_start).count();
        }

        backward(d_logits_tensor, defer_backward_execution);
        auto t_flb_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().fwd_loss_bwd_ms += std::chrono::duration<double, std::milli>(t_flb_end - t_flb_start).count();
        float current_loss = Execution_Engine::getInstance().isAsyncLossEnabled()
            ? Execution_Engine::getInstance().getLatestLoss()
            : loss;
        if (Execution_Engine::getInstance().isAsyncLossEnabled() && current_loss <= 0.0f)
        {
            current_loss = Execution_Engine::getInstance().readPendingLoss(0);
        }
        return current_loss;
    }

float Causal_LM::trainStep(const std::vector<int32_t>& input_tokens, const std::vector<int32_t>& target_tokens, IOptimizer& optimizer, float max_grad_norm)
{
        auto t_step_start = std::chrono::high_resolution_clock::now();
        setAccumulated(false);

        auto t_reset_start = std::chrono::high_resolution_clock::now();
        resetGradients();
        auto t_reset_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().reset_grad_ms += std::chrono::duration<double, std::milli>(t_reset_end - t_reset_start).count();

        float current_scale = (loss_scaler && loss_scaler->isEnabled()) ? loss_scaler->getScale() : 1.0f;
        bool defer_bwd = (config.execution_target == Execution_Target::VULKAN_GPU);
        float loss = forwardLossAndBackward(input_tokens, target_tokens, current_scale, defer_bwd);
        bool overflow = std::isnan(loss) || std::isinf(loss);

        auto t_opt_start = std::chrono::high_resolution_clock::now();
        stepOptimizer(optimizer, max_grad_norm, current_scale, overflow, defer_bwd);
        auto t_opt_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().opt_step_ms += std::chrono::duration<double, std::milli>(t_opt_end - t_opt_start).count();

        auto t_step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        Step_Timings::getInstance().total_batch_ms += step_ms;
        Step_Timings::getInstance().total_ms = Step_Timings::getInstance().total_batch_ms;
        Step_Timings::getInstance().syncAliases();
        return loss;
    }

float Causal_LM::trainStep(const Tensor& input_tensor, const Tensor& target_tensor, IOptimizer& optimizer, float max_grad_norm)
{
        auto t_step_start = std::chrono::high_resolution_clock::now();
        setAccumulated(false);

        auto t_reset_start = std::chrono::high_resolution_clock::now();
        resetGradients();
        auto t_reset_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().reset_grad_ms += std::chrono::duration<double, std::milli>(t_reset_end - t_reset_start).count();

        float current_scale = (loss_scaler && loss_scaler->isEnabled()) ? loss_scaler->getScale() : 1.0f;
        bool defer_bwd = (config.execution_target == Execution_Target::VULKAN_GPU);
        float loss = forwardLossAndBackward(input_tensor, target_tensor, current_scale, defer_bwd);
        bool overflow = std::isnan(loss) || std::isinf(loss);

        auto t_opt_start = std::chrono::high_resolution_clock::now();
        stepOptimizer(optimizer, max_grad_norm, current_scale, overflow, defer_bwd);
        auto t_opt_end = std::chrono::high_resolution_clock::now();
        Step_Timings::getInstance().opt_step_ms += std::chrono::duration<double, std::milli>(t_opt_end - t_opt_start).count();

        auto t_step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        Step_Timings::getInstance().total_batch_ms += step_ms;
        Step_Timings::getInstance().total_ms = Step_Timings::getInstance().total_batch_ms;
        Step_Timings::getInstance().syncAliases();
        return loss;
    }

float Causal_LM::trainStep(const std::vector<std::vector<int32_t>>& batch_inputs, const std::vector<std::vector<int32_t>>& batch_targets, IOptimizer& optimizer, float max_grad_norm)
{
        if (batch_inputs.empty() || batch_targets.empty())
        {
            return 0.0f;
        }

        auto t_prep_start = std::chrono::high_resolution_clock::now();
        size_t B = batch_inputs.size();
        size_t S = batch_inputs[0].size();

        Tensor in_tensor(Shape{ B, S }, config.execution_target);
        Tensor tgt_tensor(Shape{ B, S }, config.execution_target);

        std::vector<float> in_data(B * S, 0.0f);
        std::vector<float> tgt_data(B * S, -1.0f);

        for (size_t b = 0; b < B; ++b)
        {
            for (size_t s = 0; s < S && s < batch_inputs[b].size(); ++s)
            {
                in_data[b * S + s] = static_cast<float>(batch_inputs[b][s]);
            }
            for (size_t s = 0; s < S && s < batch_targets[b].size(); ++s)
            {
                tgt_data[b * S + s] = static_cast<float>(batch_targets[b][s]);
            }
        }

        in_tensor.uploadData(in_data);
        tgt_tensor.uploadData(tgt_data);
        auto t_prep_end = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double, std::milli>(t_prep_end - t_prep_start).count();
        Step_Timings::getInstance().data_prep_ms += prep_ms;

        float loss = trainStep(in_tensor, tgt_tensor, optimizer, max_grad_norm);

        Step_Timings::getInstance().total_batch_ms += prep_ms;
        Step_Timings::getInstance().total_ms = Step_Timings::getInstance().total_batch_ms;
        Step_Timings::getInstance().syncAliases();
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
        Step_Timings::getInstance().data_prep_ms += prep_ms;

        float loss = trainStep(input_tokens, target_tokens, optimizer);
        Step_Timings::getInstance().total_batch_ms += prep_ms;
        Step_Timings::getInstance().total_ms = Step_Timings::getInstance().total_batch_ms;
        return loss;
    }

void Causal_LM::saveCheckpoint(const std::string& file_path) const
{
        std::ofstream out(file_path, std::ios::binary);
        if (!out.is_open())
        {
            throw std::runtime_error("Causal_LM::saveCheckpoint: Failed to open file: " + file_path);
        }

        uint32_t magic = 0x434D4C43;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        uint64_t v_size = config.vocab_size;
        uint64_t h_dim = config.hidden_dim;
        uint64_t n_heads = config.num_heads;
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

        token_embedding.saveCheckpoint(out);
        for (const auto& block : blocks)
        {
            block->saveCheckpoint(out);
        }
        final_norm.saveCheckpoint(out);
        lm_head.saveCheckpoint(out);
    }

void Causal_LM::loadCheckpoint(const std::string& file_path)
{
        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Causal_LM::loadCheckpoint: Failed to open file: " + file_path);
        }

        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x434D4C43)
        {
            throw std::runtime_error("Causal_LM::loadCheckpoint: Invalid magic header in: " + file_path);
        }

        uint64_t v_size, h_dim, n_heads, i_dim, n_layers, m_seq;
        in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
        in.read(reinterpret_cast<char*>(&h_dim), sizeof(h_dim));
        in.read(reinterpret_cast<char*>(&n_heads), sizeof(n_heads));
        in.read(reinterpret_cast<char*>(&i_dim), sizeof(i_dim));
        in.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
        in.read(reinterpret_cast<char*>(&m_seq), sizeof(m_seq));
        in.read(reinterpret_cast<char*>(&config.rms_norm_eps), sizeof(config.rms_norm_eps));
        in.read(reinterpret_cast<char*>(&config.rope_base), sizeof(config.rope_base));

        if (v_size != config.vocab_size || h_dim != config.hidden_dim || n_layers != config.num_layers)
        {
            throw std::runtime_error("Causal_LM::loadCheckpoint: Architecture dimensions mismatch");
        }

        token_embedding.loadCheckpoint(in);
        for (auto& block : blocks)
        {
            block->loadCheckpoint(in);
        }
        final_norm.loadCheckpoint(in);
        lm_head.loadCheckpoint(in);
        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
        invalidateWeightCaches();
        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            warmupWeightCaches();
        }
    }

void Causal_LM::saveInference(const std::string& file_path) const
{
        std::ofstream out(file_path, std::ios::binary);
        if (!out.is_open())
        {
            throw std::runtime_error("Causal_LM::saveInference: Failed to open file: " + file_path);
        }

        uint32_t magic = 0x494D4C43;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        uint64_t v_size = config.vocab_size;
        uint64_t h_dim = config.hidden_dim;
        uint64_t n_heads = config.num_heads;
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

        token_embedding.saveInference(out);
        for (const auto& block : blocks)
        {
            block->saveInference(out);
        }
        final_norm.saveInference(out);
        lm_head.saveInference(out);
    }

void Causal_LM::loadInference(const std::string& file_path)
{
        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Causal_LM::loadInference: Failed to open file: " + file_path);
        }

        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x494D4C43)
        {
            throw std::runtime_error("Causal_LM::loadInference: Invalid magic header in: " + file_path);
        }

        uint64_t v_size, h_dim, n_heads, i_dim, n_layers, m_seq;
        in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
        in.read(reinterpret_cast<char*>(&h_dim), sizeof(h_dim));
        in.read(reinterpret_cast<char*>(&n_heads), sizeof(n_heads));
        in.read(reinterpret_cast<char*>(&i_dim), sizeof(i_dim));
        in.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
        in.read(reinterpret_cast<char*>(&m_seq), sizeof(m_seq));
        in.read(reinterpret_cast<char*>(&config.rms_norm_eps), sizeof(config.rms_norm_eps));
        in.read(reinterpret_cast<char*>(&config.rope_base), sizeof(config.rope_base));

        if (v_size != config.vocab_size || h_dim != config.hidden_dim || n_layers != config.num_layers)
        {
            throw std::runtime_error("Causal_LM::loadInference: Architecture dimensions mismatch");
        }

        token_embedding.loadInference(in);
        for (auto& block : blocks)
        {
            block->loadInference(in);
        }
        final_norm.loadInference(in);
        lm_head.loadInference(in);
        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            Execution_Engine::getInstance().getContext().executePendingTransfers();
        }
        invalidateWeightCaches();
        if (config.execution_target == Execution_Target::VULKAN_GPU)
        {
            warmupWeightCaches();
        }
    }

std::string Causal_LM::generate(const std::string& prompt, size_t max_new_tokens, float temperature, float top_p, int32_t eos_token_id, bool skip_special, const std::function<void(const std::string&)>& token_callback, float repetition_penalty, size_t top_k, const std::vector<std::string>& stop_sequences)
{
        if (!tokenizer.isLoaded())
        {
            throw std::runtime_error("Causal_LM::generate: Tokenizer is not loaded");
        }

        std::vector<int32_t> current_tokens = tokenizer.encode(prompt, true);
        if (current_tokens.empty())
        {
            return "";
        }

        setTrainingMode(false);
        std::vector<int32_t> generated_ids;
        generated_ids.reserve(max_new_tokens);
        size_t V = config.vocab_size;

        for (size_t step = 0; step < max_new_tokens; ++step)
        {
            if (current_tokens.size() > config.max_seq_len)
            {
                current_tokens.erase(current_tokens.begin(),
                    current_tokens.begin() + (current_tokens.size() - config.max_seq_len));
            }

            Tensor logits = forward(current_tokens, false);
            size_t S_curr = current_tokens.size();

            std::vector<float> all_logits = logits.getData();
            if (all_logits.size() < S_curr * V)
            {
                break;
            }

            std::vector<float> last_logits(all_logits.begin() + (S_curr - 1) * V,
                all_logits.begin() + S_curr * V);

            if (repetition_penalty > 1.0f)
            {
                for (int32_t prev_id : current_tokens)
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

            int32_t next_token = sampleToken(last_logits, temperature, top_p, top_k);
            generated_ids.push_back(next_token);
            current_tokens.push_back(next_token);

            if (token_callback)
            {
                token_callback(tokenizer.decode({ next_token }, skip_special, false));
            }

            if (next_token == eos_token_id)
            {
                break;
            }

            if (!stop_sequences.empty())
            {
                std::string full_so_far = tokenizer.decode(generated_ids, skip_special);
                bool should_stop = false;
                for (const auto& stop_str : stop_sequences)
                {
                    if (!stop_str.empty() && full_so_far.size() >= stop_str.size() &&
                        full_so_far.compare(full_so_far.size() - stop_str.size(), stop_str.size(), stop_str) == 0)
                    {
                        should_stop = true;
                        break;
                    }
                }
                if (should_stop)
                {
                    break;
                }
            }
        }

        std::string res = tokenizer.decode(generated_ids, skip_special);
        if (res.empty() && !generated_ids.empty())
        {
            res = tokenizer.decode(generated_ids, false);
        }
        return res;
    }


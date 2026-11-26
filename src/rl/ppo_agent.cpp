#include "rl/ppo_agent.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "layer/gelu.h"
#include "layer/linear_layer.h"
#include "layer/ppo_actor_critic_layer.h"
#include "math/tensor.h"
#include "optimizer/adam_optimizer.h"

Ppo_Agent::Ppo_Agent(std::size_t _state_dimension,
                     std::size_t _action_space_size,
                     float _learning_rate,
                     float _gamma,
                     float _gae_lambda,
                     float _clip_epsilon,
                     float _value_loss_coeff,
                     float _entropy_coeff,
                     std::size_t _epochs_per_update,
                     std::size_t _mini_batch_size,
                     Execution_Target _execution_target,
                     std::uint32_t _seed)
    : state_dimension(_state_dimension),
      action_space_size(_action_space_size),
      gamma(_gamma),
      gae_lambda(_gae_lambda),
      clip_epsilon(_clip_epsilon),
      value_loss_coeff(_value_loss_coeff),
      entropy_coeff(_entropy_coeff),
      epochs_per_update(_epochs_per_update),
      mini_batch_size(_mini_batch_size),
      execution_target(_execution_target),
      random_engine(_seed)
{
    if (state_dimension == 0 || action_space_size == 0 || mini_batch_size == 0)
    {
        throw std::invalid_argument("Ppo_Agent: state_dimension, action_space_size, and mini_batch_size must be > 0");
    }

    model = std::make_unique<PPO_Actor_Critic_Layer>(action_space_size, execution_target);

    model->addActorLayer<Linear_Layer>(state_dimension, 128, execution_target);
    model->addActorLayer<Gelu_Layer>(execution_target);
    model->addActorLayer<Linear_Layer>(128, 128, execution_target);
    model->addActorLayer<Gelu_Layer>(execution_target);
    model->addActorLayer<Linear_Layer>(128, action_space_size, execution_target);

    model->addCriticLayer<Linear_Layer>(state_dimension, 128, execution_target);
    model->addCriticLayer<Gelu_Layer>(execution_target);
    model->addCriticLayer<Linear_Layer>(128, 128, execution_target);
    model->addCriticLayer<Gelu_Layer>(execution_target);
    model->addCriticLayer<Linear_Layer>(128, 1, execution_target);

    optimizer = std::make_unique<Adam_Optimizer>(_learning_rate, 0.9f, 0.999f, 1e-8f, 1.0f);
}

Ppo_Agent::~Ppo_Agent() noexcept
{
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        try
        {
            Execution_Engine::getInstance().waitIdle();
        }
        catch (...)
        {
        }
    }
}
Ppo_Agent::Ppo_Agent(Ppo_Agent &&) noexcept = default;
Ppo_Agent &Ppo_Agent::operator=(Ppo_Agent &&) noexcept = default;

Ppo_Action_Result Ppo_Agent::selectAction(const std::vector<float> &state, bool explore) const
{
    if (state.size() != state_dimension)
    {
        throw std::invalid_argument("Ppo_Agent::selectAction: State dimension mismatch");
    }

    Tensor input_matrix(1, state_dimension, state, execution_target);
    Tensor output_matrix = const_cast<PPO_Actor_Critic_Layer *>(model.get())->forward(input_matrix);
    const auto &out_data = output_matrix.getData();
    if (out_data.size() < action_space_size + 1)
    {
        throw std::runtime_error("Ppo_Agent::selectAction: Model output size smaller than expected");
    }

    float value = out_data[action_space_size];
    if (!std::isfinite(value))
    {
        throw std::runtime_error("Ppo_Agent::selectAction: Non-finite state value encountered");
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t a = 0; a < action_space_size; ++a)
    {
        if (!std::isfinite(out_data[a]))
        {
            throw std::runtime_error("Ppo_Agent::selectAction: Non-finite action logit encountered");
        }
        max_logit = std::max(max_logit, out_data[a]);
    }

    std::vector<float> probs(action_space_size);
    float sum_exp = 0.0f;
    for (std::size_t a = 0; a < action_space_size; ++a)
    {
        probs[a] = std::exp(out_data[a] - max_logit);
        sum_exp += probs[a];
    }
    if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp))
    {
        throw std::runtime_error("Ppo_Agent::selectAction: Non-finite or zero softmax normalizer");
    }
    const float log_sum_exp = std::log(sum_exp);
    for (std::size_t a = 0; a < action_space_size; ++a)
    {
        probs[a] /= sum_exp;
    }

    std::size_t selected_action = 0;
    if (!explore)
    {
        float max_prob = -1.0f;
        for (std::size_t a = 0; a < action_space_size; ++a)
        {
            if (probs[a] > max_prob)
            {
                max_prob = probs[a];
                selected_action = a;
            }
        }
    }
    else
    {
        std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
        selected_action = dist(random_engine);
    }

    float log_prob = (out_data[selected_action] - max_logit) - log_sum_exp;
    return {selected_action, log_prob, value};
}

float Ppo_Agent::estimateValue(const std::vector<float> &state) const
{
    if (state.size() != state_dimension)
    {
        throw std::invalid_argument("Ppo_Agent::estimateValue: State dimension mismatch");
    }

    Tensor input_matrix(1, state_dimension, state, execution_target);
    Tensor output_matrix = const_cast<PPO_Actor_Critic_Layer *>(model.get())->forward(input_matrix);
    const auto &out_data = output_matrix.getData();
    if (out_data.size() < action_space_size + 1)
    {
        throw std::runtime_error("Ppo_Agent::estimateValue: Model output size smaller than expected");
    }
    float value = out_data[action_space_size];
    if (!std::isfinite(value))
    {
        throw std::runtime_error("Ppo_Agent::estimateValue: Non-finite value estimated");
    }
    return value;
}

void Ppo_Agent::storeTransition(const Ppo_Transition &transition)
{
    if (transition.state.size() != state_dimension || transition.action >= action_space_size)
    {
        throw std::invalid_argument("Ppo_Agent::storeTransition: state dimension or action index out of range");
    }
    trajectory_buffer.push_back(transition);
}

void Ppo_Agent::storeTransition(Ppo_Transition &&transition)
{
    if (transition.state.size() != state_dimension || transition.action >= action_space_size)
    {
        throw std::invalid_argument("Ppo_Agent::storeTransition: state dimension or action index out of range");
    }
    trajectory_buffer.push_back(std::move(transition));
}

void Ppo_Agent::trainStep()
{
    trainStep(0.0f);
}

void Ppo_Agent::trainStep(float last_value)
{
    if (!std::isfinite(last_value))
    {
        throw std::invalid_argument("Ppo_Agent::trainStep: last_value must be finite");
    }
    if (trajectory_buffer.empty() || mini_batch_size == 0)
    {
        return;
    }

    const std::size_t total_steps = trajectory_buffer.size();
    if (total_steps < mini_batch_size)
    {
        return;
    }

    std::vector<float> advantages(total_steps, 0.0f);
    std::vector<float> returns(total_steps, 0.0f);

    float gae = 0.0f;
    for (std::size_t t = total_steps; t-- > 0;)
    {
        float next_value = (t + 1 == total_steps) ? last_value : trajectory_buffer[t + 1].value;
        bool non_terminal = !trajectory_buffer[t].done;
        float delta = trajectory_buffer[t].reward + gamma * next_value * (non_terminal ? 1.0f : 0.0f) - trajectory_buffer[t].value;
        gae = delta + gamma * gae_lambda * (non_terminal ? 1.0f : 0.0f) * gae;
        advantages[t] = gae;
        returns[t] = advantages[t] + trajectory_buffer[t].value;
    }

    float adv_mean = 0.0f;
    for (float a : advantages) adv_mean += a;
    adv_mean /= static_cast<float>(total_steps);

    float adv_var = 0.0f;
    for (float a : advantages) adv_var += (a - adv_mean) * (a - adv_mean);
    float adv_std = std::sqrt(adv_var / static_cast<float>(total_steps)) + 1e-8f;

    for (float &a : advantages)
    {
        a = (a - adv_mean) / adv_std;
    }

    std::vector<std::size_t> indices(total_steps);
    std::iota(indices.begin(), indices.end(), 0);

    const std::size_t output_dimension = action_space_size + 1;

    for (std::size_t epoch = 0; epoch < epochs_per_update; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), random_engine);

        for (std::size_t start = 0; start < total_steps; start += mini_batch_size)
        {
            const std::size_t cur_batch_size = std::min(mini_batch_size, total_steps - start);
            std::vector<float> batch_states;
            batch_states.reserve(cur_batch_size * state_dimension);

            for (std::size_t b = 0; b < cur_batch_size; ++b)
            {
                std::size_t idx = indices[start + b];
                batch_states.insert(batch_states.end(),
                                    trajectory_buffer[idx].state.begin(),
                                    trajectory_buffer[idx].state.end());
            }

            Tensor batch_matrix(cur_batch_size, state_dimension, std::move(batch_states), execution_target);
            Tensor output_matrix = model->forward(batch_matrix);
            const auto &out_data = output_matrix.getData();
            if (out_data.size() < cur_batch_size * output_dimension)
            {
                throw std::runtime_error("Ppo_Agent::trainStep: Model output size smaller than expected");
            }

            std::vector<float> grad_data(cur_batch_size * output_dimension, 0.0f);
            bool batch_non_finite = false;

            for (std::size_t b = 0; b < cur_batch_size; ++b)
            {
                std::size_t idx = indices[start + b];
                std::size_t row_offset = b * output_dimension;

                float max_logit = -std::numeric_limits<float>::infinity();
                for (std::size_t a = 0; a < action_space_size; ++a)
                {
                    if (!std::isfinite(out_data[row_offset + a]))
                    {
                        batch_non_finite = true;
                        break;
                    }
                    max_logit = std::max(max_logit, out_data[row_offset + a]);
                }
                if (batch_non_finite)
                {
                    break;
                }

                std::vector<float> current_probs(action_space_size);
                float current_sum_exp = 0.0f;
                for (std::size_t a = 0; a < action_space_size; ++a)
                {
                    current_probs[a] = std::exp(out_data[row_offset + a] - max_logit);
                    current_sum_exp += current_probs[a];
                }
                if (!(current_sum_exp > 0.0f) || !std::isfinite(current_sum_exp))
                {
                    batch_non_finite = true;
                    break;
                }
                const float log_sum_exp = std::log(current_sum_exp);
                std::vector<float> current_log_probs(action_space_size);
                float entropy = 0.0f;
                for (std::size_t a = 0; a < action_space_size; ++a)
                {
                    current_probs[a] /= current_sum_exp;
                    current_log_probs[a] = (out_data[row_offset + a] - max_logit) - log_sum_exp;
                    if (current_probs[a] > 0.0f)
                    {
                        entropy -= current_probs[a] * current_log_probs[a];
                    }
                }

                std::size_t act = trajectory_buffer[idx].action;
                if (act >= action_space_size)
                {
                    throw std::out_of_range("Ppo_Agent::trainStep: Action index out of range");
                }
                float current_log_prob = current_log_probs[act];
                float old_log_prob = trajectory_buffer[idx].log_prob;
                float adv = advantages[idx];
                float current_value = out_data[row_offset + action_space_size];
                float target_return = returns[idx];
                if (!std::isfinite(old_log_prob) || !std::isfinite(adv) ||
                    !std::isfinite(current_value) || !std::isfinite(target_return) ||
                    !std::isfinite(entropy))
                {
                    batch_non_finite = true;
                    break;
                }

                float clamped_log_diff = std::clamp(current_log_prob - old_log_prob, -20.0f, 20.0f);
                float ratio = std::exp(clamped_log_diff);

                float d_loss_d_ratio = 0.0f;
                if ((adv > 0.0f && ratio < 1.0f + clip_epsilon) ||
                    (adv < 0.0f && ratio > 1.0f - clip_epsilon))
                {
                    d_loss_d_ratio = -adv;
                }

                for (std::size_t k = 0; k < action_space_size; ++k)
                {
                    float d_ratio_d_zk = ratio * ((k == act ? 1.0f : 0.0f) - current_probs[k]);
                    float d_policy_d_zk = (d_loss_d_ratio != 0.0f) ? (d_loss_d_ratio * d_ratio_d_zk) : 0.0f;

                    float d_entropy_d_zk = entropy_coeff * current_probs[k] * (current_log_probs[k] + entropy);

                    grad_data[row_offset + k] = (d_policy_d_zk + d_entropy_d_zk) / static_cast<float>(cur_batch_size);
                }

                float val_grad = value_loss_coeff * (current_value - target_return);
                grad_data[row_offset + action_space_size] = val_grad / static_cast<float>(cur_batch_size);
            }

            if (batch_non_finite)
            {
                Logger::logMessage("Ppo_Agent::trainStep: Non-finite logits, log_prob, value, or advantage encountered; skipping mini-batch.",
                                   Log_Level::LOG_WARNING, true, 0, Log_Feature::RL_AGENT);
                continue;
            }

            Tensor output_gradient(cur_batch_size, output_dimension, std::move(grad_data), execution_target);
            model->backward(output_gradient);
            optimizer->step(model->getParametersAndGradients());
            model->resetGradient();
            if (execution_target == Execution_Target::VULKAN_GPU)
            {
                Execution_Engine::getInstance().executeGraph();
            }
        }
    }

    trajectory_buffer.clear();
}

namespace
{
    constexpr std::uint32_t PPO_INFER_MAGIC_V2 = 0x494F5050u;
    constexpr std::uint32_t PPO_CKPT_MAGIC_V2 = 0x434F5050u;
    constexpr std::uint32_t PPO_FORMAT_VERSION_V2 = 2u;
}

void Ppo_Agent::saveInference(const std::string &path) const
{
    std::ofstream out_stream(path, std::ios::binary | std::ios::trunc);
    if (!out_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::saveInference: Could not open file " + path);
    }
    const std::uint64_t sd = static_cast<std::uint64_t>(state_dimension);
    const std::uint64_t ad = static_cast<std::uint64_t>(action_space_size);
    out_stream.write(reinterpret_cast<const char *>(&PPO_INFER_MAGIC_V2), sizeof(PPO_INFER_MAGIC_V2));
    out_stream.write(reinterpret_cast<const char *>(&PPO_FORMAT_VERSION_V2), sizeof(PPO_FORMAT_VERSION_V2));
    out_stream.write(reinterpret_cast<const char *>(&sd), sizeof(sd));
    out_stream.write(reinterpret_cast<const char *>(&ad), sizeof(ad));
    model->saveInference(out_stream);
    out_stream.flush();
    if (!out_stream.good())
    {
        throw std::runtime_error("Ppo_Agent::saveInference: Failed writing inference file " + path);
    }
    Logger::logMessage(Input_Format{"Ppo_Agent::saveInference: Saved to '{}'", path},
                       Log_Level::LOG_INFO, false, 0, Log_Feature::RL_AGENT | Log_Feature::CHECKPOINT_IO);
}

void Ppo_Agent::loadInference(const std::string &path)
{
    std::ifstream in_stream(path, std::ios::binary);
    if (!in_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::loadInference: Could not open file " + path);
    }
    std::uint32_t magic = 0;
    in_stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (in_stream.fail())
    {
        throw std::runtime_error("Ppo_Agent::loadInference: Failed reading header from " + path);
    }
    if (magic == PPO_INFER_MAGIC_V2)
    {
        std::uint32_t version = 0;
        std::uint64_t sd = 0;
        std::uint64_t ad = 0;
        in_stream.read(reinterpret_cast<char *>(&version), sizeof(version));
        in_stream.read(reinterpret_cast<char *>(&sd), sizeof(sd));
        in_stream.read(reinterpret_cast<char *>(&ad), sizeof(ad));
        if (in_stream.fail() || version != PPO_FORMAT_VERSION_V2 ||
            sd != state_dimension || ad != action_space_size)
        {
            throw std::runtime_error("Ppo_Agent::loadInference: Invalid V2 header or dimension mismatch in " + path);
        }
    }
    else
    {
        in_stream.clear();
        in_stream.seekg(0, std::ios::beg);
    }
    model->loadInference(in_stream);
    if (in_stream.fail())
    {
        throw std::runtime_error("Ppo_Agent::loadInference: Corrupt or truncated inference file " + path);
    }
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        model->invalidateWeightCache();
    }
    Logger::logMessage(Input_Format{"Ppo_Agent::loadInference: Loaded from '{}'", path},
                       Log_Level::LOG_INFO, false, 0, Log_Feature::RL_AGENT | Log_Feature::CHECKPOINT_IO);
}

void Ppo_Agent::saveCheckpoint(const std::string &path, std::size_t epoch) const
{
    std::ofstream out_stream(path, std::ios::binary | std::ios::trunc);
    if (!out_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::saveCheckpoint: Could not open file " + path);
    }
    const std::uint64_t sd = static_cast<std::uint64_t>(state_dimension);
    const std::uint64_t ad = static_cast<std::uint64_t>(action_space_size);
    const std::uint64_t ep = static_cast<std::uint64_t>(epoch);
    out_stream.write(reinterpret_cast<const char *>(&PPO_CKPT_MAGIC_V2), sizeof(PPO_CKPT_MAGIC_V2));
    out_stream.write(reinterpret_cast<const char *>(&PPO_FORMAT_VERSION_V2), sizeof(PPO_FORMAT_VERSION_V2));
    out_stream.write(reinterpret_cast<const char *>(&sd), sizeof(sd));
    out_stream.write(reinterpret_cast<const char *>(&ad), sizeof(ad));
    out_stream.write(reinterpret_cast<const char *>(&ep), sizeof(ep));
    model->saveCheckpoint(out_stream);
    optimizer->saveCheckpoint(out_stream);
    out_stream.flush();
    if (!out_stream.good())
    {
        throw std::runtime_error("Ppo_Agent::saveCheckpoint: Failed writing checkpoint " + path);
    }
    Logger::logMessage(Input_Format{"Ppo_Agent::saveCheckpoint: Saved epoch {} to '{}'", epoch, path},
                       Log_Level::LOG_INFO, false, 0, Log_Feature::RL_AGENT | Log_Feature::CHECKPOINT_IO);
}

void Ppo_Agent::loadCheckpoint(const std::string &path)
{
    std::ifstream in_stream(path, std::ios::binary);
    if (!in_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::loadCheckpoint: Could not open file " + path);
    }
    std::uint32_t magic = 0;
    in_stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (in_stream.fail())
    {
        throw std::runtime_error("Ppo_Agent::loadCheckpoint: Failed reading header from " + path);
    }
    if (magic == PPO_CKPT_MAGIC_V2)
    {
        std::uint32_t version = 0;
        std::uint64_t sd = 0;
        std::uint64_t ad = 0;
        std::uint64_t ep = 0;
        in_stream.read(reinterpret_cast<char *>(&version), sizeof(version));
        in_stream.read(reinterpret_cast<char *>(&sd), sizeof(sd));
        in_stream.read(reinterpret_cast<char *>(&ad), sizeof(ad));
        in_stream.read(reinterpret_cast<char *>(&ep), sizeof(ep));
        if (in_stream.fail() || version != PPO_FORMAT_VERSION_V2 ||
            sd != state_dimension || ad != action_space_size)
        {
            throw std::runtime_error("Ppo_Agent::loadCheckpoint: Invalid V2 header or dimension mismatch in " + path);
        }
        last_loaded_epoch = static_cast<std::size_t>(ep);
        model->loadCheckpoint(in_stream);
        optimizer->loadCheckpoint(in_stream, execution_target);
    }
    else
    {
        in_stream.clear();
        in_stream.seekg(0, std::ios::beg);
        std::size_t epoch = 0;
        in_stream.read(reinterpret_cast<char *>(&epoch), sizeof(epoch));
        if (in_stream.fail())
        {
            throw std::runtime_error("Ppo_Agent::loadCheckpoint: Failed reading V1 header from " + path);
        }
        last_loaded_epoch = epoch;
        model->loadCheckpoint(in_stream);
    }
    if (in_stream.fail())
    {
        throw std::runtime_error("Ppo_Agent::loadCheckpoint: Corrupt or truncated checkpoint " + path);
    }
    Logger::logMessage(Input_Format{"Ppo_Agent::loadCheckpoint: Loaded epoch {} from '{}'", last_loaded_epoch, path},
                       Log_Level::LOG_INFO, false, 0, Log_Feature::RL_AGENT | Log_Feature::CHECKPOINT_IO);
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
        model->invalidateWeightCache();
    }
    Logger::logMessage(Input_Format{"Ppo_Agent::loadInference: Loaded from '{}'", path},
                       Log_Level::LOG_INFO, false, 0, Log_Feature::RL_AGENT | Log_Feature::CHECKPOINT_IO);
}

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
    model = std::make_unique<PPO_Actor_Critic_Layer>(action_space_size, execution_target);

    // Actor Network Branch: state_dim -> 128 -> GELU -> 128 -> GELU -> action_dim
    model->addActorLayer<Linear_Layer>(state_dimension, 128, execution_target);
    model->addActorLayer<Gelu_Layer>(execution_target);
    model->addActorLayer<Linear_Layer>(128, 128, execution_target);
    model->addActorLayer<Gelu_Layer>(execution_target);
    model->addActorLayer<Linear_Layer>(128, action_space_size, execution_target);

    // Critic Network Branch: state_dim -> 128 -> GELU -> 128 -> GELU -> 1
    model->addCriticLayer<Linear_Layer>(state_dimension, 128, execution_target);
    model->addCriticLayer<Gelu_Layer>(execution_target);
    model->addCriticLayer<Linear_Layer>(128, 128, execution_target);
    model->addCriticLayer<Gelu_Layer>(execution_target);
    model->addCriticLayer<Linear_Layer>(128, 1, execution_target);

    optimizer = std::make_unique<Adam_Optimizer>(_learning_rate, 0.9f, 0.999f, 1e-8f, 1.0f);
}

Ppo_Agent::~Ppo_Agent() noexcept = default;
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

    float value = out_data[action_space_size];

    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t a = 0; a < action_space_size; ++a)
    {
        max_logit = std::max(max_logit, out_data[a]);
    }

    std::vector<float> probs(action_space_size);
    float sum_exp = 0.0f;
    for (std::size_t a = 0; a < action_space_size; ++a)
    {
        probs[a] = std::exp(out_data[a] - max_logit);
        sum_exp += probs[a];
    }
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

    float log_prob = std::log(std::max(probs[selected_action], 1e-8f));
    return {selected_action, log_prob, value};
}

void Ppo_Agent::storeTransition(const Ppo_Transition &transition)
{
    trajectory_buffer.push_back(transition);
}

void Ppo_Agent::storeTransition(Ppo_Transition &&transition)
{
    trajectory_buffer.push_back(std::move(transition));
}

void Ppo_Agent::trainStep()
{
    if (trajectory_buffer.empty())
    {
        return;
    }

    const std::size_t total_steps = trajectory_buffer.size();
    if (total_steps < mini_batch_size)
    {
        return;
    }

    // 1. Compute Generalized Advantage Estimation (GAE) and Returns
    std::vector<float> advantages(total_steps, 0.0f);
    std::vector<float> returns(total_steps, 0.0f);

    float gae = 0.0f;
    for (int t = static_cast<int>(total_steps) - 1; t >= 0; --t)
    {
        float next_value = (t == static_cast<int>(total_steps) - 1) ? 0.0f : trajectory_buffer[t + 1].value;
        bool non_terminal = !trajectory_buffer[t].done;
        float delta = trajectory_buffer[t].reward + gamma * next_value * (non_terminal ? 1.0f : 0.0f) - trajectory_buffer[t].value;
        gae = delta + gamma * gae_lambda * (non_terminal ? 1.0f : 0.0f) * gae;
        advantages[t] = gae;
        returns[t] = advantages[t] + trajectory_buffer[t].value;
    }

    // 2. Normalize advantages for training stability
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

    // 3. Mini-batch PPO Optimization Epochs
    std::vector<std::size_t> indices(total_steps);
    std::iota(indices.begin(), indices.end(), 0);

    const std::size_t output_dimension = action_space_size + 1;

    for (std::size_t epoch = 0; epoch < epochs_per_update; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), random_engine);

        for (std::size_t start = 0; start + mini_batch_size <= total_steps; start += mini_batch_size)
        {
            std::vector<float> batch_states;
            batch_states.reserve(mini_batch_size * state_dimension);

            for (std::size_t b = 0; b < mini_batch_size; ++b)
            {
                std::size_t idx = indices[start + b];
                batch_states.insert(batch_states.end(),
                                    trajectory_buffer[idx].state.begin(),
                                    trajectory_buffer[idx].state.end());
            }

            Tensor batch_matrix(mini_batch_size, state_dimension, std::move(batch_states), execution_target);
            Tensor output_matrix = model->forward(batch_matrix);
            const auto &out_data = output_matrix.getData();

            std::vector<float> grad_data(mini_batch_size * output_dimension, 0.0f);

            for (std::size_t b = 0; b < mini_batch_size; ++b)
            {
                std::size_t idx = indices[start + b];
                std::size_t row_offset = b * output_dimension;

                // Logits for current sample
                float max_logit = -std::numeric_limits<float>::infinity();
                for (std::size_t a = 0; a < action_space_size; ++a)
                {
                    max_logit = std::max(max_logit, out_data[row_offset + a]);
                }

                std::vector<float> current_probs(action_space_size);
                float current_sum_exp = 0.0f;
                for (std::size_t a = 0; a < action_space_size; ++a)
                {
                    current_probs[a] = std::exp(out_data[row_offset + a] - max_logit);
                    current_sum_exp += current_probs[a];
                }
                float entropy = 0.0f;
                for (std::size_t a = 0; a < action_space_size; ++a)
                {
                    current_probs[a] /= current_sum_exp;
                    if (current_probs[a] > 1e-8f)
                    {
                        entropy -= current_probs[a] * std::log(current_probs[a]);
                    }
                }

                std::size_t act = trajectory_buffer[idx].action;
                float current_log_prob = std::log(std::max(current_probs[act], 1e-8f));
                float old_log_prob = trajectory_buffer[idx].log_prob;
                float ratio = std::exp(current_log_prob - old_log_prob);

                float adv = advantages[idx];

                // Clipped surrogate loss derivative w.r.t ratio
                float d_loss_d_ratio = 0.0f;
                if ((adv > 0.0f && ratio < 1.0f + clip_epsilon) ||
                    (adv < 0.0f && ratio > 1.0f - clip_epsilon))
                {
                    d_loss_d_ratio = -adv;
                }

                // Compute gradient for each action logit
                for (std::size_t k = 0; k < action_space_size; ++k)
                {
                    float d_ratio_d_zk = ratio * ((k == act ? 1.0f : 0.0f) - current_probs[k]);
                    float d_policy_d_zk = d_loss_d_ratio * d_ratio_d_zk;

                    float d_entropy_d_zk = entropy_coeff * current_probs[k] * (std::log(std::max(current_probs[k], 1e-8f)) + entropy);

                    grad_data[row_offset + k] = (d_policy_d_zk + d_entropy_d_zk) / static_cast<float>(mini_batch_size);
                }

                // Value Loss: 0.5 * (V - R)^2 -> derivative = (V - R)
                float current_value = out_data[row_offset + action_space_size];
                float target_return = returns[idx];
                float val_grad = value_loss_coeff * (current_value - target_return) * 2.0f;
                grad_data[row_offset + action_space_size] = val_grad / static_cast<float>(mini_batch_size);
            }

            Tensor output_gradient(mini_batch_size, output_dimension, std::move(grad_data), execution_target);
            model->backward(output_gradient);
            optimizer->step(model->getParametersAndGradients());
            model->resetGradient();
        }
    }

    trajectory_buffer.clear();
}

void Ppo_Agent::saveInference(const std::string &path) const
{
    std::ofstream out_stream(path, std::ios::binary);
    if (!out_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::saveInference: Could not open file " + path);
    }
    model->saveInference(out_stream);
}

void Ppo_Agent::loadInference(const std::string &path)
{
    std::ifstream in_stream(path, std::ios::binary);
    if (!in_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::loadInference: Could not open file " + path);
    }
    model->loadInference(in_stream);
}

void Ppo_Agent::saveCheckpoint(const std::string &path, std::size_t epoch) const
{
    std::ofstream out_stream(path, std::ios::binary);
    if (!out_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::saveCheckpoint: Could not open file " + path);
    }
    out_stream.write(reinterpret_cast<const char *>(&epoch), sizeof(epoch));
    model->saveCheckpoint(out_stream);
}

void Ppo_Agent::loadCheckpoint(const std::string &path)
{
    std::ifstream in_stream(path, std::ios::binary);
    if (!in_stream.is_open())
    {
        throw std::runtime_error("Ppo_Agent::loadCheckpoint: Could not open file " + path);
    }
    std::size_t epoch = 0;
    in_stream.read(reinterpret_cast<char *>(&epoch), sizeof(epoch));
    model->loadCheckpoint(in_stream);
}

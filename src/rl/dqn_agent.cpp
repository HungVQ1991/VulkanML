#include "rl/dqn_agent.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "math/tensor.h"
#include "neural_network.h"

Dqn_Agent::Dqn_Agent(Neural_Network &&_network_prototype,
                     std::size_t _state_dimension,
                     std::size_t _action_space_size,
                     std::size_t _replay_capacity,
                     float _gamma,
                     float _epsilon,
                     float _epsilon_min,
                     float _epsilon_decay,
                     Execution_Target _execution_target,
                     std::uint32_t _seed)
    : q_network(std::move(_network_prototype)),
      target_network(_execution_target),
      replay_buffer(_replay_capacity, _seed),
      state_dimension(_state_dimension),
      action_space_size(_action_space_size),
      gamma(_gamma),
      epsilon(_epsilon),
      epsilon_min(_epsilon_min),
      epsilon_decay(_epsilon_decay),
      execution_target(_execution_target),
      random_engine(_seed)
{
    q_network.setExecutionTarget(_execution_target);
    synchronizeTargetNetworkHard();
}

std::size_t Dqn_Agent::selectAction(const std::vector<float> &_state, bool _explore) const
{
    if (_state.size() != state_dimension)
    {
        throw std::invalid_argument("State dimension mismatch");
    }

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    if (_explore && distribution(random_engine) < epsilon)
    {
        std::uniform_int_distribution<std::size_t> action_dist(0, action_space_size - 1);
        return action_dist(random_engine);
    }

    Tensor state_matrix(1, state_dimension, _state, execution_target);
    Tensor q_values_matrix = const_cast<Neural_Network &>(q_network).forward(state_matrix);
    std::vector<float> q_values = q_values_matrix.getData();

    std::size_t best_action = 0;
    float max_q_value = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < action_space_size; ++i)
    {
        if (q_values[i] > max_q_value)
        {
            max_q_value = q_values[i];
            best_action = i;
        }
    }

    return best_action;
}

void Dqn_Agent::synchronizeTargetNetworkHard()
{
    auto q_params = q_network.getParametersAndGradients();
    auto target_params = target_network.getParametersAndGradients();

    if (target_params.empty() && !q_params.empty())
    {
        return;
    }

    for (std::size_t i = 0; i < q_params.size(); ++i)
    {
        target_params[i].first->uploadData(q_params[i].first->getData());
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
}

void Dqn_Agent::synchronizeTargetNetworkSoft(float _tau)
{
    auto q_params = q_network.getParametersAndGradients();
    auto target_params = target_network.getParametersAndGradients();

    for (std::size_t i = 0; i < q_params.size(); ++i)
    {
        std::vector<float> q_data = q_params[i].first->getData();
        std::vector<float> target_data = target_params[i].first->getData();

        for (std::size_t j = 0; j < target_data.size(); ++j)
        {
            target_data[j] = _tau * q_data[j] + (1.0f - _tau) * target_data[j];
        }

        target_params[i].first->uploadData(target_data);
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
}

void Dqn_Agent::trainStep(std::size_t _batch_size)
{
    if (!replay_buffer.isReady(_batch_size))
    {
        return;
    }

    Transition_Batch batch = replay_buffer.sample(_batch_size);

    Tensor next_states_matrix(_batch_size, state_dimension, std::move(batch.next_states), execution_target);
    Tensor target_q_matrix = target_network.forward(next_states_matrix);
    std::vector<float> target_q_data = target_q_matrix.getData();

    Tensor states_matrix(_batch_size, state_dimension, batch.states, execution_target);
    Tensor current_q_matrix = q_network.forward(states_matrix);
    std::vector<float> desired_q_data = current_q_matrix.getData();

    for (std::size_t i = 0; i < _batch_size; ++i)
    {
        float max_next_q = -std::numeric_limits<float>::infinity();
        if (!batch.terminals[i])
        {
            for (std::size_t a = 0; a < action_space_size; ++a)
            {
                max_next_q = std::max(max_next_q, target_q_data[i * action_space_size + a]);
            }
        }
        else
        {
            max_next_q = 0.0f;
        }

        float target_value = batch.rewards[i] + (batch.terminals[i] ? 0.0f : gamma * max_next_q);
        desired_q_data[i * action_space_size + batch.actions[i]] = target_value;
    }

    Tensor desired_target_matrix(_batch_size, action_space_size, std::move(desired_q_data), execution_target);

    q_network.trainStep(states_matrix, desired_target_matrix);

    train_step_counter++;

    if (is_soft_update)
    {
        synchronizeTargetNetworkSoft(tau);
    }
    else if (train_step_counter % target_update_interval == 0)
    {
        synchronizeTargetNetworkHard();
    }
}

void Dqn_Agent::initializeTargetNetworkFromPrototype(Neural_Network &&_target_prototype)
{
    target_network = std::move(_target_prototype);
    target_network.setExecutionTarget(execution_target);
    synchronizeTargetNetworkHard();
}

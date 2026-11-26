#include "rl/dqn_agent.h"

#include <algorithm>
#include <cmath>
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
    if (state_dimension == 0 || action_space_size == 0)
    {
        throw std::invalid_argument("Dqn_Agent: state_dimension and action_space_size must be > 0");
    }
    if (!std::isfinite(gamma) || gamma < 0.0f || gamma > 1.0f ||
        !std::isfinite(epsilon) || epsilon < 0.0f || epsilon > 1.0f ||
        !std::isfinite(epsilon_min) || epsilon_min < 0.0f || epsilon_min > 1.0f ||
        !std::isfinite(epsilon_decay) || epsilon_decay < 0.0f || epsilon_decay > 1.0f)
    {
        throw std::invalid_argument("Dqn_Agent: gamma, epsilon, epsilon_min, and epsilon_decay must be finite in [0, 1]");
    }
    if (q_network.getLayers().empty())
    {
        throw std::invalid_argument("Dqn_Agent: q_network must contain at least one layer");
    }
    if (!q_network.getTrainingContext().hasOptimizer() || !q_network.getTrainingContext().hasCostFunction())
    {
        throw std::invalid_argument("Dqn_Agent: q_network must have optimizer and cost function configured");
    }

    q_network.setExecutionTarget(_execution_target);
    for (const auto &layer : q_network.getLayers())
    {
        if (!layer)
        {
            throw std::invalid_argument("Dqn_Agent: q_network contains null layer");
        }
        auto cloned = layer->clone();
        if (!cloned)
        {
            throw std::runtime_error("Dqn_Agent: failed to clone q_network layer for target_network");
        }
        target_network.addLayer(std::move(cloned));
    }
    target_network.setExecutionTarget(_execution_target);
    synchronizeTargetNetworkHard();
}

void Dqn_Agent::setTargetUpdateParameters(std::size_t _interval, bool _use_soft_update, float _tau)
{
    if (_interval == 0)
    {
        throw std::invalid_argument("Dqn_Agent::setTargetUpdateParameters: interval must be > 0");
    }
    if (!std::isfinite(_tau) || _tau < 0.0f || _tau > 1.0f)
    {
        throw std::invalid_argument("Dqn_Agent::setTargetUpdateParameters: tau must be finite in [0, 1]");
    }
    target_update_interval = _interval;
    is_soft_update = _use_soft_update;
    tau = _tau;
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
    if (q_values.size() < action_space_size)
    {
        throw std::runtime_error("Dqn_Agent::selectAction: Q-network output size smaller than action_space_size");
    }

    std::size_t best_action = 0;
    float max_q_value = -std::numeric_limits<float>::infinity();
    bool found_finite = false;

    for (std::size_t i = 0; i < action_space_size; ++i)
    {
        if (std::isfinite(q_values[i]) && (!found_finite || q_values[i] > max_q_value))
        {
            max_q_value = q_values[i];
            best_action = i;
            found_finite = true;
        }
    }

    if (!found_finite)
    {
        Logger::logMessage("Dqn_Agent::selectAction: All Q-values are non-finite; falling back to action 0.",
                           Log_Level::LOG_WARNING, true, 0, Log_Feature::RL_AGENT);
        return 0;
    }

    return best_action;
}

void Dqn_Agent::storeTransition(const Transition &_transition)
{
    if (_transition.state.size() != state_dimension ||
        _transition.next_state.size() != state_dimension ||
        _transition.action >= action_space_size)
    {
        throw std::invalid_argument("Dqn_Agent::storeTransition: transition dimensions or action out of range");
    }
    replay_buffer.push(_transition);
}

void Dqn_Agent::storeTransition(Transition &&_transition)
{
    if (_transition.state.size() != state_dimension ||
        _transition.next_state.size() != state_dimension ||
        _transition.action >= action_space_size)
    {
        throw std::invalid_argument("Dqn_Agent::storeTransition: transition dimensions or action out of range");
    }
    replay_buffer.push(std::move(_transition));
}

void Dqn_Agent::synchronizeTargetNetworkHard()
{
    auto q_params = q_network.getParametersAndGradients();
    auto target_params = target_network.getParametersAndGradients();

    if (q_params.size() != target_params.size())
    {
        throw std::invalid_argument("Dqn_Agent::synchronizeTargetNetworkHard: parameter count mismatch between q_network and target_network");
    }

    for (std::size_t i = 0; i < q_params.size(); ++i)
    {
        if (!q_params[i].first || !target_params[i].first ||
            q_params[i].first->getTotalElements() != target_params[i].first->getTotalElements())
        {
            throw std::invalid_argument("Dqn_Agent::synchronizeTargetNetworkHard: parameter tensor shape mismatch");
        }
        target_params[i].first->uploadData(q_params[i].first->getData());
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().getContext().executePendingTransfers();
    }
    target_network.invalidateLayerWeightCaches();
    Logger::logMessage("Dqn_Agent::synchronizeTargetNetworkHard: Hard target sync completed",
                       Log_Level::LOG_DEBUG, false, 0, Log_Feature::RL_AGENT);
}

void Dqn_Agent::synchronizeTargetNetworkSoft(float _tau)
{
    if (!std::isfinite(_tau) || _tau < 0.0f || _tau > 1.0f)
    {
        throw std::invalid_argument("Dqn_Agent::synchronizeTargetNetworkSoft: tau must be finite in [0, 1]");
    }

    auto q_params = q_network.getParametersAndGradients();
    auto target_params = target_network.getParametersAndGradients();

    if (q_params.size() != target_params.size())
    {
        throw std::invalid_argument("Dqn_Agent::synchronizeTargetNetworkSoft: parameter count mismatch between q_network and target_network");
    }

    for (std::size_t i = 0; i < q_params.size(); ++i)
    {
        if (!q_params[i].first || !target_params[i].first ||
            q_params[i].first->getTotalElements() != target_params[i].first->getTotalElements())
        {
            throw std::invalid_argument("Dqn_Agent::synchronizeTargetNetworkSoft: parameter tensor shape mismatch");
        }

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
    target_network.invalidateLayerWeightCaches();
}

void Dqn_Agent::trainStep(std::size_t _batch_size)
{
    if (_batch_size == 0 || !replay_buffer.isReady(_batch_size))
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

    if (target_q_data.size() != _batch_size * action_space_size ||
        desired_q_data.size() != _batch_size * action_space_size)
    {
        throw std::runtime_error("Dqn_Agent::trainStep: Q-network output size mismatch with action_space_size");
    }

    auto all_finite = [](const std::vector<float> &vals) {
        return std::all_of(vals.begin(), vals.end(), [](float v) { return std::isfinite(v); });
    };
    bool rewards_finite = true;
    for (std::size_t i = 0; i < _batch_size; ++i)
    {
        if (!std::isfinite(batch.rewards[i]))
        {
            rewards_finite = false;
            break;
        }
    }
    if (!all_finite(target_q_data) || !all_finite(desired_q_data) || !rewards_finite)
    {
        Logger::logMessage("Dqn_Agent::trainStep: Non-finite Q-value or reward encountered; skipping step.",
                           Log_Level::LOG_WARNING, true, 0, Log_Feature::RL_AGENT);
        return;
    }

    for (std::size_t i = 0; i < _batch_size; ++i)
    {
        if (batch.actions[i] >= action_space_size)
        {
            throw std::out_of_range("Dqn_Agent::trainStep: sampled action index out of range");
        }

        const bool non_bootstrapping_terminal =
            batch.terminals[i] && !(i < batch.truncated.size() && batch.truncated[i]);

        float max_next_q = 0.0f;
        if (!non_bootstrapping_terminal)
        {
            max_next_q = target_q_data[i * action_space_size];
            for (std::size_t a = 1; a < action_space_size; ++a)
            {
                max_next_q = std::max(max_next_q, target_q_data[i * action_space_size + a]);
            }
        }

        float target_value = batch.rewards[i] + (non_bootstrapping_terminal ? 0.0f : gamma * max_next_q);
        if (!std::isfinite(target_value))
        {
            Logger::logMessage("Dqn_Agent::trainStep: Non-finite Bellman target computed; skipping step.",
                               Log_Level::LOG_WARNING, true, 0, Log_Feature::RL_AGENT);
            return;
        }
        desired_q_data[i * action_space_size + batch.actions[i]] = target_value;
    }

    Tensor desired_target_matrix(_batch_size, action_space_size, std::move(desired_q_data), execution_target);

    q_network.backwardAndStep(desired_target_matrix);

    train_step_counter++;

    if (is_soft_update)
    {
        synchronizeTargetNetworkSoft(tau);
    }
    else if (target_update_interval > 0 && (train_step_counter % target_update_interval == 0))
    {
        synchronizeTargetNetworkHard();
    }
}

void Dqn_Agent::initializeTargetNetworkFromPrototype(Neural_Network &&_target_prototype)
{
    Neural_Network candidate(std::move(_target_prototype));
    candidate.setExecutionTarget(execution_target);

    auto q_params = q_network.getParametersAndGradients();
    auto cand_params = candidate.getParametersAndGradients();
    if (q_params.size() != cand_params.size())
    {
        throw std::invalid_argument("Dqn_Agent::initializeTargetNetworkFromPrototype: parameter count mismatch with q_network");
    }
    for (std::size_t i = 0; i < q_params.size(); ++i)
    {
        if (!q_params[i].first || !cand_params[i].first ||
            q_params[i].first->getTotalElements() != cand_params[i].first->getTotalElements())
        {
            throw std::invalid_argument("Dqn_Agent::initializeTargetNetworkFromPrototype: tensor shape mismatch with q_network");
        }
    }

    target_network = std::move(candidate);
    synchronizeTargetNetworkHard();
}

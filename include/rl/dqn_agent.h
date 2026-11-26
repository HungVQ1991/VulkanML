#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "neural_network.h"
#include "replay_buffer.h"
#include "transition.h"

class Dqn_Agent
{
private:
    Neural_Network q_network;
    Neural_Network target_network;
    Replay_Buffer replay_buffer;

    std::size_t state_dimension = 0;
    std::size_t action_space_size = 0;

    float gamma = 0.99f;
    float epsilon = 1.0f;
    float epsilon_min = 0.01f;
    float epsilon_decay = 0.995f;
    float tau = 0.005f;

    std::size_t target_update_interval = 100;
    std::size_t train_step_counter = 0;
    bool is_soft_update = false;

    Execution_Target execution_target = Execution_Target::CPU;
    mutable std::mt19937 random_engine;

public:
    Dqn_Agent(Neural_Network &&_network_prototype,
              std::size_t _state_dimension,
              std::size_t _action_space_size,
              std::size_t _replay_capacity = 50000,
              float _gamma = 0.99f,
              float _epsilon = 1.0f,
              float _epsilon_min = 0.01f,
              float _epsilon_decay = 0.995f,
              Execution_Target _execution_target = Execution_Target::CPU,
              std::uint32_t _seed = std::random_device{}());

    void setTargetUpdateParameters(std::size_t _interval, bool _use_soft_update = false, float _tau = 0.005f);

    void setEpsilon(float _epsilon) noexcept { epsilon = _epsilon; }
    float getEpsilon() const noexcept { return epsilon; }
    void decayEpsilon() noexcept
    {
        if (epsilon > epsilon_min)
        {
            epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
        }
    }

    std::size_t selectAction(const std::vector<float> &_state, bool _explore = true) const;

    void storeTransition(const Transition &_transition);
    void storeTransition(Transition &&_transition);

    void synchronizeTargetNetworkHard();
    void synchronizeTargetNetworkSoft(float _tau);

    void trainStep(std::size_t _batch_size);
    void initializeTargetNetworkFromPrototype(Neural_Network &&_target_prototype);

    Neural_Network &getQNetwork() noexcept { return q_network; }
    const Neural_Network &getQNetwork() const noexcept { return q_network; }

    Neural_Network &getTargetNetwork() noexcept { return target_network; }
    const Neural_Network &getTargetNetwork() const noexcept { return target_network; }

    Replay_Buffer &getReplayBuffer() noexcept { return replay_buffer; }
    const Replay_Buffer &getReplayBuffer() const noexcept { return replay_buffer; }
};

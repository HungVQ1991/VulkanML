#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "layer/ppo_actor_critic_layer.h"
#include "optimizer/adam_optimizer.h"

struct Ppo_Action_Result
{
    std::size_t action = 0;
    float log_prob = 0.0f;
    float value = 0.0f;
};

struct Ppo_Transition
{
    std::vector<float> state;
    std::size_t action = 0;
    float reward = 0.0f;
    bool done = false;
    float log_prob = 0.0f;
    float value = 0.0f;
};

class Ppo_Agent
{
private:
    std::unique_ptr<PPO_Actor_Critic_Layer> model;
    std::unique_ptr<Adam_Optimizer> optimizer;

    std::size_t state_dimension = 0;
    std::size_t action_space_size = 0;

    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_epsilon = 0.2f;
    float value_loss_coeff = 0.5f;
    float entropy_coeff = 0.01f;
    std::size_t epochs_per_update = 4;
    std::size_t mini_batch_size = 32;

    Execution_Target execution_target = Execution_Target::CPU;
    mutable std::mt19937 random_engine;

    std::vector<Ppo_Transition> trajectory_buffer;
    std::size_t last_loaded_epoch = 0;

public:
    Ppo_Agent(std::size_t _state_dimension,
              std::size_t _action_space_size,
              float _learning_rate = 0.0003f,
              float _gamma = 0.99f,
              float _gae_lambda = 0.95f,
              float _clip_epsilon = 0.2f,
              float _value_loss_coeff = 0.5f,
              float _entropy_coeff = 0.01f,
              std::size_t _epochs_per_update = 4,
              std::size_t _mini_batch_size = 32,
              Execution_Target _execution_target = Execution_Target::CPU,
              std::uint32_t _seed = std::random_device{}());

    ~Ppo_Agent() noexcept;

    Ppo_Agent(const Ppo_Agent &) = delete;
    Ppo_Agent &operator=(const Ppo_Agent &) = delete;
    Ppo_Agent(Ppo_Agent &&) noexcept;
    Ppo_Agent &operator=(Ppo_Agent &&) noexcept;

    Ppo_Action_Result selectAction(const std::vector<float> &state, bool explore = true) const;

    void storeTransition(const Ppo_Transition &transition);
    void storeTransition(Ppo_Transition &&transition);

    std::size_t getTrajectorySize() const noexcept { return trajectory_buffer.size(); }
    void clearTrajectories() noexcept { trajectory_buffer.clear(); }

    void trainStep();
    void trainStep(float last_value);
    float estimateValue(const std::vector<float> &state) const;

    void saveInference(const std::string &path) const;
    void loadInference(const std::string &path);

    void saveCheckpoint(const std::string &path, std::size_t epoch) const;
    void loadCheckpoint(const std::string &path);
    std::size_t getLastLoadedEpoch() const noexcept { return last_loaded_epoch; }

    PPO_Actor_Critic_Layer &getModel() noexcept { return *model; }
    const PPO_Actor_Critic_Layer &getModel() const noexcept { return *model; }
};

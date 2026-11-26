#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

struct Cartpole_Step_Result
{
    std::vector<float> next_state;
    float reward = 0.0f;
    bool is_terminal = false;
    bool is_terminated = false;
    bool is_truncated = false;
};

class Cartpole_Env
{
private:
    float gravity = 9.8f;
    float mass_cart = 1.0f;
    float mass_pole = 0.1f;
    float total_mass = 1.1f;
    float length = 0.5f;
    float pole_mass_length = 0.05f;
    float force_mag = 10.0f;
    float time_step = 0.02f;

    float theta_threshold_radians = 12.0f * 2.0f * 3.1415926535f / 360.0f;
    float x_threshold = 2.4f;

    float x = 0.0f;
    float x_dot = 0.0f;
    float theta = 0.0f;
    float theta_dot = 0.0f;

    std::size_t step_count = 0;
    std::size_t max_steps = 200;
    bool is_done = false;

    mutable std::mt19937 random_engine;

public:
    explicit Cartpole_Env(std::size_t _max_steps = 200, std::uint32_t _seed = std::random_device{}());

    std::vector<float> reset();
    Cartpole_Step_Result step(std::size_t _action);

    bool isDone() const noexcept { return is_done; }
    std::size_t getStateDimension() const noexcept { return 4; }
    std::size_t getActionSpaceSize() const noexcept { return 2; }
};

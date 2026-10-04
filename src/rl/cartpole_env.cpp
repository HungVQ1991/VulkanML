#include "rl/cartpole_env.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

Cartpole_Env::Cartpole_Env(std::size_t _max_steps, std::uint32_t _seed)
    : max_steps(_max_steps),
      random_engine(_seed)
{
}

std::vector<float> Cartpole_Env::reset()
{
    std::uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    x = distribution(random_engine);
    x_dot = distribution(random_engine);
    theta = distribution(random_engine);
    theta_dot = distribution(random_engine);
    step_count = 0;
    return {x, x_dot, theta, theta_dot};
}

Cartpole_Step_Result Cartpole_Env::step(std::size_t _action)
{
    float force = (_action == 1) ? force_mag : -force_mag;
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);

    float temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) / total_mass;
    float theta_acc = (gravity * sin_theta - cos_theta * temp) /
                      (length * (4.0f / 3.0f - mass_pole * cos_theta * cos_theta / total_mass));
    float x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

    x += time_step * x_dot;
    x_dot += time_step * x_acc;
    theta += time_step * theta_dot;
    theta_dot += time_step * theta_acc;

    step_count++;

    bool is_failed = (x < -x_threshold || x > x_threshold ||
                      theta < -theta_threshold_radians || theta > theta_threshold_radians);
    bool is_timeout = (step_count >= max_steps);
    bool is_terminal = is_failed || is_timeout;

    float reward = is_failed ? 0.0f : 1.0f;

    return Cartpole_Step_Result{
        .next_state = {x, x_dot, theta, theta_dot},
        .reward = reward,
        .is_terminal = is_terminal};
}

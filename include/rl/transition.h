#pragma once

#include <cstddef>
#include <vector>

struct Transition
{
    std::vector<float> state;
    std::size_t action = 0;
    float reward = 0.0f;
    std::vector<float> next_state;
    bool is_terminal = false;
    bool is_truncated = false;
};

struct Transition_Batch
{
    std::vector<float> states;
    std::vector<std::size_t> actions;
    std::vector<float> rewards;
    std::vector<float> next_states;
    std::vector<bool> terminals;
    std::vector<bool> truncated;
    std::size_t batch_size = 0;
    std::size_t state_dimension = 0;
};

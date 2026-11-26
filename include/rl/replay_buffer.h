#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "transition.h"

class Replay_Buffer
{
private:
    std::size_t capacity = 10000;
    std::size_t head_index = 0;
    std::size_t current_size = 0;
    std::size_t expected_state_dim = 0;
    std::vector<Transition> buffer;
    mutable std::mt19937 random_engine;

public:
    explicit Replay_Buffer(std::size_t _capacity = 10000, std::uint32_t _seed = std::random_device{}());

    void push(const Transition &_transition);
    void push(Transition &&_transition);

    Transition_Batch sample(std::size_t _batch_size) const;

    bool isReady(std::size_t _batch_size) const noexcept { return _batch_size > 0 && current_size >= _batch_size; }
    std::size_t getSize() const noexcept { return current_size; }
    std::size_t getCapacity() const noexcept { return capacity; }

    void clear() noexcept;
};

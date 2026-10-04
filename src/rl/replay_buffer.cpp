#include "rl/replay_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helper/logger.h"

Replay_Buffer::Replay_Buffer(std::size_t _capacity, std::uint32_t _seed)
    : capacity(_capacity),
      head_index(0),
      current_size(0),
      buffer(_capacity),
      random_engine(_seed)
{
    if (capacity == 0)
    {
        Logger::logMessage("Capacity must be greater than zero", Log_Level::LOG_ERROR, true);
        throw std::invalid_argument("Capacity must be greater than zero");
    }
}

void Replay_Buffer::push(const Transition &_transition)
{
    buffer[head_index] = _transition;
    head_index = (head_index + 1) % capacity;
    current_size = std::min(current_size + 1, capacity);
}

void Replay_Buffer::push(Transition &&_transition)
{
    buffer[head_index] = std::move(_transition);
    head_index = (head_index + 1) % capacity;
    current_size = std::min(current_size + 1, capacity);
}

Transition_Batch Replay_Buffer::sample(std::size_t _batch_size) const
{
    if (_batch_size > current_size)
    {
        Logger::logMessage("Batch size exceeds current buffer size", Log_Level::LOG_ERROR, true);
        throw std::runtime_error("Batch size exceeds current buffer size");
    }

    std::size_t state_dimension = buffer[0].state.size();

    Transition_Batch batch;
    batch.batch_size = _batch_size;
    batch.state_dimension = state_dimension;
    batch.states.resize(_batch_size * state_dimension);
    batch.next_states.resize(_batch_size * state_dimension);
    batch.actions.reserve(_batch_size);
    batch.rewards.reserve(_batch_size);
    batch.terminals.reserve(_batch_size);

    std::uniform_int_distribution<std::size_t> distribution(0, current_size - 1);

    for (std::size_t i = 0; i < _batch_size; ++i)
    {
        std::size_t sample_index = distribution(random_engine);
        const auto &transition = buffer[sample_index];

        std::copy(transition.state.begin(), transition.state.end(), batch.states.begin() + i * state_dimension);
        std::copy(transition.next_state.begin(), transition.next_state.end(), batch.next_states.begin() + i * state_dimension);

        batch.actions.push_back(transition.action);
        batch.rewards.push_back(transition.reward);
        batch.terminals.push_back(transition.is_terminal);
    }

    return batch;
}

void Replay_Buffer::clear() noexcept
{
    head_index = 0;
    current_size = 0;
}

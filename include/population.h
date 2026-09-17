#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/shape.h"
#include "math/tensor.h"
#include "neural_network.h"

inline std::size_t calculateNumel(const Shape &shape)
{
    std::size_t total = 1;
    for (std::size_t dimension : shape.getDimensions())
    {
        total *= dimension;
    }
    return total;
}

inline Shape prependPopulationDim(std::size_t population_size, const Shape &shape)
{
    std::vector<std::size_t> dimensions{population_size};
    dimensions.insert(dimensions.end(), shape.getDimensions().begin(), shape.getDimensions().end());
    return Shape(dimensions);
}

struct Population_Layer_Adapter
{
    const ILayer *template_layer = nullptr;
    std::vector<Shape> param_shapes;
    std::vector<bool> param_evolvable;
    std::vector<std::size_t> param_numel;
    std::vector<Tensor> batched_params;
    std::vector<std::vector<float>> host_params;
};

class Population
{
private:
    std::size_t population_size = 0;
    std::size_t state_dimension = 0;
    std::size_t action_space_size = 0;
    Execution_Target execution_target = Execution_Target::CPU;
    mutable std::mt19937 random_engine;

    std::vector<Population_Layer_Adapter> layer_adapters;

    mutable Tensor batched_input_tensor;
    mutable std::vector<float> batched_input_host;

    void initializeWeights()
    {
        for (auto &adapter : layer_adapters)
        {
            adapter.host_params.resize(adapter.param_shapes.size());
            for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                std::size_t total = population_size * adapter.param_numel[p];
                adapter.host_params[p].resize(total);

                auto init_fn = adapter.template_layer->getPopulationParameterInitializer(p);
                for (float &value : adapter.host_params[p])
                {
                    value = init_fn(random_engine);
                }
            }
        }
        syncHostToDevice();
    }

    void syncHostToDevice()
    {
        for (auto &adapter : layer_adapters)
        {
            for (std::size_t p = 0; p < adapter.batched_params.size(); ++p)
            {
                adapter.batched_params[p].uploadData(adapter.host_params[p]);
            }
        }
    }

    void syncDeviceToHost()
    {
        for (auto &adapter : layer_adapters)
        {
            for (std::size_t p = 0; p < adapter.batched_params.size(); ++p)
            {
                adapter.host_params[p] = adapter.batched_params[p].getData();
            }
        }
    }

public:
    Population(std::size_t _population_size,
               const Neural_Network &_template_net,
               std::size_t _state_dimension,
               std::size_t _action_dimension,
               Execution_Target _execution_target = Execution_Target::CPU,
               std::uint32_t _seed = std::random_device{}())
        : population_size(_population_size),
          state_dimension(_state_dimension),
          action_space_size(_action_dimension),
          execution_target(_execution_target),
          random_engine(_seed)
    {
        for (const auto &layer_ptr : _template_net.getLayers())
        {
            if (!layer_ptr->supportsPopulationBatch())
            {
                throw std::invalid_argument(std::format("Layer {} not supported in population mode",
                                                        magic_enum::enum_name(layer_ptr->getLayerType())));
            }

            Population_Layer_Adapter adapter;
            adapter.template_layer = layer_ptr.get();
            adapter.param_shapes = layer_ptr->getPopulationParameterDims();
            adapter.param_evolvable = layer_ptr->getPopulationParameterIsEvolvable();

            if (adapter.param_evolvable.size() != adapter.param_shapes.size())
            {
                throw std::logic_error(std::format("{}: getPopulationParameterIsEvolvable() size mismatch",
                                                   magic_enum::enum_name(layer_ptr->getLayerType())));
            }

            for (const Shape &shape : adapter.param_shapes)
            {
                adapter.param_numel.push_back(calculateNumel(shape));
                adapter.batched_params.emplace_back(prependPopulationDim(_population_size, shape), _execution_target);
            }

            layer_adapters.push_back(std::move(adapter));
        }

        if (layer_adapters.empty())
        {
            throw std::invalid_argument("Template neural network has no population-capable layers");
        }

        batched_input_tensor = Tensor(Shape{_population_size, 1, _state_dimension}, _execution_target);
        batched_input_host.assign(_population_size * _state_dimension, 0.0f);

        initializeWeights();
    }

    std::size_t selectAction(std::size_t individual_index, const std::vector<float> &state_data) const
    {
        return selectAction(individual_index, state_data.data());
    }

    std::size_t selectAction(std::size_t individual_index, const float *state_data) const
    {
        if (individual_index >= population_size)
        {
            throw std::out_of_range("Individual index out of range in selectAction");
        }

        Neural_Network individual_net = getIndividual(individual_index);
        std::vector<float> input_vector(state_data, state_data + state_dimension);
        Matrix input_matrix(1, state_dimension, std::move(input_vector), execution_target);
        Matrix output_matrix = individual_net.forward(input_matrix);

        const auto &output_data = output_matrix.getData();
        if (output_data.empty())
        {
            return 0;
        }

        return static_cast<std::size_t>(std::distance(
            output_data.begin(),
            std::max_element(output_data.begin(), output_data.end())));
    }

    void selectBatchActions(const float *states_flat,
                            const std::size_t *active_indices,
                            std::size_t active_count,
                            std::size_t *actions_out) const
    {
        if (active_count == 0)
        {
            return;
        }

        if (batched_input_host.size() != population_size * state_dimension)
        {
            batched_input_host.assign(population_size * state_dimension, 0.0f);
        }

        if (!active_indices && active_count == population_size)
        {
            std::copy(states_flat, states_flat + (population_size * state_dimension), batched_input_host.begin());
        }
        else
        {
            for (std::size_t i = 0; i < active_count; ++i)
            {
                std::size_t agent_idx = active_indices ? active_indices[i] : i;
                if (agent_idx < population_size)
                {
                    const float *src = states_flat + (i * state_dimension);
                    float *dst = batched_input_host.data() + (agent_idx * state_dimension);
                    std::copy(src, src + state_dimension, dst);
                }
            }
        }

        batched_input_tensor.uploadData(batched_input_host);

        Tensor current_tensor = batched_input_tensor;
        for (const auto &adapter : layer_adapters)
        {
            current_tensor = adapter.template_layer->forward(current_tensor, adapter.batched_params);
        }

        std::vector<float> action_values = current_tensor.getData();
        for (std::size_t i = 0; i < active_count; ++i)
        {
            std::size_t agent_idx = active_indices ? active_indices[i] : i;
            std::size_t offset = agent_idx * action_space_size;
            std::size_t best_action = 0;
            float max_value = action_values[offset];
            for (std::size_t a = 1; a < action_space_size; ++a)
            {
                if (action_values[offset + a] > max_value)
                {
                    max_value = action_values[offset + a];
                    best_action = a;
                }
            }
            actions_out[i] = best_action;
        }
    }

    void evolve(const float *fitness_scores,
                float elitism_ratio = 0.15f,
                float mutation_rate = 0.15f,
                float mutation_strength = 0.15f,
                float crossover_rate = 0.5f,
                std::size_t tournament_size = 3,
                std::size_t explicit_elite_count = 0)
    {
        std::vector<std::size_t> indices(population_size);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [fitness_scores](std::size_t a, std::size_t b)
                  { return fitness_scores[a] > fitness_scores[b]; });

        std::size_t elite_count = (explicit_elite_count > 0)
            ? std::min(explicit_elite_count, population_size)
            : std::max<std::size_t>(1, static_cast<std::size_t>(population_size * elitism_ratio));

        std::uniform_int_distribution<std::size_t> pop_dist(0, population_size - 1);
        std::normal_distribution<float> norm_dist(0.0f, mutation_strength);
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

        std::vector<std::vector<std::vector<float>>> prev_params;
        prev_params.reserve(layer_adapters.size());
        for (const auto &adapter : layer_adapters)
        {
            prev_params.push_back(adapter.host_params);
        }

        auto selectParentTournament = [&]() -> std::size_t {
            if (tournament_size <= 1 || population_size <= 1)
            {
                return pop_dist(random_engine);
            }
            std::size_t best_cand = pop_dist(random_engine);
            float best_fit = fitness_scores[best_cand];
            for (std::size_t t = 1; t < tournament_size; ++t)
            {
                std::size_t cand = pop_dist(random_engine);
                if (fitness_scores[cand] > best_fit)
                {
                    best_fit = fitness_scores[cand];
                    best_cand = cand;
                }
            }
            return best_cand;
        };

        for (std::size_t i = 0; i < elite_count; ++i)
        {
            std::size_t elite_src = indices[i];
            for (std::size_t l = 0; l < layer_adapters.size(); ++l)
            {
                auto &adapter = layer_adapters[l];
                for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
                {
                    std::size_t len = adapter.param_numel[p];
                    std::copy_n(prev_params[l][p].data() + elite_src * len, len,
                                adapter.host_params[p].data() + i * len);
                }
            }
        }

        for (std::size_t i = elite_count; i < population_size; ++i)
        {
            std::size_t target_idx = i;
            std::size_t parent_a = selectParentTournament();
            std::size_t parent_b = selectParentTournament();

            for (std::size_t l = 0; l < layer_adapters.size(); ++l)
            {
                auto &adapter = layer_adapters[l];
                for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
                {
                    std::size_t len = adapter.param_numel[p];
                    auto &host = adapter.host_params[p];
                    const auto &prev = prev_params[l][p];

                    if (!adapter.param_evolvable[p])
                    {
                        std::copy_n(prev.data() + parent_a * len, len, host.data() + target_idx * len);
                        continue;
                    }

                    std::size_t dst = target_idx * len;
                    std::size_t pa = parent_a * len;
                    std::size_t pb = parent_b * len;

                    for (std::size_t k = 0; k < len; ++k)
                    {
                        float val = (prob_dist(random_engine) < crossover_rate) ? prev[pb + k] : prev[pa + k];
                        if (prob_dist(random_engine) < mutation_rate)
                        {
                            val += norm_dist(random_engine);
                        }
                        host[dst + k] = val;
                    }
                }
            }
        }

        syncHostToDevice();
    }

    Neural_Network getIndividual(std::size_t index) const
    {
        if (index >= population_size)
        {
            throw std::out_of_range("Individual index out of range in getIndividual");
        }

        Neural_Network individual_net(Execution_Target::CPU);
        for (const auto &adapter : layer_adapters)
        {
            auto layer_clone = adapter.template_layer->clone();
            for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                std::size_t len = adapter.param_numel[p];
                std::vector<float> slice(adapter.host_params[p].begin() + index * len,
                                         adapter.host_params[p].begin() + (index + 1) * len);
                layer_clone->setPopulationParameter(p, std::move(slice));
            }
            individual_net.addLayer(std::move(layer_clone));
        }
        individual_net.setExecutionTarget(execution_target);
        return individual_net;
    }

    Neural_Network getBestIndividual(const float *fitness_scores) const
    {
        std::size_t best_idx = 0;
        float max_fitness = fitness_scores[0];
        for (std::size_t i = 1; i < population_size; ++i)
        {
            if (fitness_scores[i] > max_fitness)
            {
                max_fitness = fitness_scores[i];
                best_idx = i;
            }
        }
        return getIndividual(best_idx);
    }

    bool saveIndividual(std::size_t individual_index, const std::string &file_path) const
    {
        if (individual_index >= population_size)
        {
            return false;
        }
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            return false;
        }

        const char magic_header[4] = {'N', 'N', 'I', 'K'};
        out_file.write(magic_header, 4);

        std::uint32_t file_version = 1;
        out_file.write(reinterpret_cast<const char *>(&file_version), sizeof(file_version));

        std::uint64_t state_dim_val = static_cast<std::uint64_t>(state_dimension);
        std::uint64_t action_dim_val = static_cast<std::uint64_t>(action_space_size);
        std::uint64_t layer_count_val = static_cast<std::uint64_t>(layer_adapters.size());

        out_file.write(reinterpret_cast<const char *>(&state_dim_val), sizeof(state_dim_val));
        out_file.write(reinterpret_cast<const char *>(&action_dim_val), sizeof(action_dim_val));
        out_file.write(reinterpret_cast<const char *>(&layer_count_val), sizeof(layer_count_val));

        for (const auto &adapter : layer_adapters)
        {
            Layer_Type layer_type = adapter.template_layer->getLayerType();
            out_file.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));

            std::uint64_t param_count = static_cast<std::uint64_t>(adapter.param_shapes.size());
            out_file.write(reinterpret_cast<const char *>(&param_count), sizeof(param_count));

            for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                std::uint64_t len = static_cast<std::uint64_t>(adapter.param_numel[p]);
                out_file.write(reinterpret_cast<const char *>(&len), sizeof(len));
                const float *ptr = adapter.host_params[p].data() + individual_index * len;
                out_file.write(reinterpret_cast<const char *>(ptr), len * sizeof(float));
            }
        }

        return true;
    }

    bool saveBestIndividual(const std::string &file_path, const float *fitness_scores) const
    {
        std::size_t best_idx = 0;
        float max_fitness = fitness_scores[0];
        for (std::size_t i = 1; i < population_size; ++i)
        {
            if (fitness_scores[i] > max_fitness)
            {
                max_fitness = fitness_scores[i];
                best_idx = i;
            }
        }
        return saveIndividual(best_idx, file_path);
    }

    bool loadIndividual(std::size_t individual_index, const std::string &file_path)
    {
        if (individual_index >= population_size)
        {
            return false;
        }
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            return false;
        }

        char magic_header[4];
        in_file.read(magic_header, 4);
        if (magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'I' || magic_header[3] != 'K')
        {
            return false;
        }

        std::uint32_t file_version = 0;
        in_file.read(reinterpret_cast<char *>(&file_version), sizeof(file_version));
        if (file_version != 1)
        {
            return false;
        }

        std::uint64_t state_dim_val = 0;
        std::uint64_t action_dim_val = 0;
        std::uint64_t layer_count_val = 0;

        in_file.read(reinterpret_cast<char *>(&state_dim_val), sizeof(state_dim_val));
        in_file.read(reinterpret_cast<char *>(&action_dim_val), sizeof(action_dim_val));
        in_file.read(reinterpret_cast<char *>(&layer_count_val), sizeof(layer_count_val));

        if (state_dim_val != state_dimension || action_dim_val != action_space_size || layer_count_val != layer_adapters.size())
        {
            return false;
        }

        for (auto &adapter : layer_adapters)
        {
            Layer_Type layer_type;
            in_file.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type));
            if (layer_type != adapter.template_layer->getLayerType())
            {
                return false;
            }

            std::uint64_t param_count = 0;
            in_file.read(reinterpret_cast<char *>(&param_count), sizeof(param_count));
            if (param_count != adapter.param_shapes.size())
            {
                return false;
            }

            for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                std::uint64_t len = 0;
                in_file.read(reinterpret_cast<char *>(&len), sizeof(len));
                if (len != adapter.param_numel[p])
                {
                    return false;
                }

                float *ptr = adapter.host_params[p].data() + individual_index * len;
                in_file.read(reinterpret_cast<char *>(ptr), len * sizeof(float));
            }
        }

        syncHostToDevice();
        return true;
    }

    bool loadBestIndividual(const std::string &file_path)
    {
        return loadIndividual(0, file_path);
    }

    bool saveCheckpoint(const std::string &file_path, std::uint64_t generation = 0, const float *fitness_scores = nullptr) const
    {
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
        {
            return false;
        }

        const char magic_header[4] = {'N', 'N', 'P', 'K'};
        out_file.write(magic_header, 4);

        std::uint32_t file_version = 1;
        out_file.write(reinterpret_cast<const char *>(&file_version), sizeof(file_version));

        std::uint64_t generation_val = generation;
        std::uint64_t pop_size_val = static_cast<std::uint64_t>(population_size);
        std::uint64_t state_dim_val = static_cast<std::uint64_t>(state_dimension);
        std::uint64_t action_dim_val = static_cast<std::uint64_t>(action_space_size);
        std::uint64_t layer_count_val = static_cast<std::uint64_t>(layer_adapters.size());

        out_file.write(reinterpret_cast<const char *>(&generation_val), sizeof(generation_val));
        out_file.write(reinterpret_cast<const char *>(&pop_size_val), sizeof(pop_size_val));
        out_file.write(reinterpret_cast<const char *>(&state_dim_val), sizeof(state_dim_val));
        out_file.write(reinterpret_cast<const char *>(&action_dim_val), sizeof(action_dim_val));
        out_file.write(reinterpret_cast<const char *>(&layer_count_val), sizeof(layer_count_val));

        for (const auto &adapter : layer_adapters)
        {
            Layer_Type layer_type = adapter.template_layer->getLayerType();
            out_file.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));

            std::uint64_t param_count = static_cast<std::uint64_t>(adapter.param_shapes.size());
            out_file.write(reinterpret_cast<const char *>(&param_count), sizeof(param_count));

            for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                std::uint64_t len = static_cast<std::uint64_t>(adapter.param_numel[p]);
                out_file.write(reinterpret_cast<const char *>(&len), sizeof(len));
                out_file.write(reinterpret_cast<const char *>(adapter.host_params[p].data()),
                               adapter.host_params[p].size() * sizeof(float));
            }
        }

        std::uint8_t has_fitness = (fitness_scores != nullptr) ? 1 : 0;
        out_file.write(reinterpret_cast<const char *>(&has_fitness), sizeof(has_fitness));
        if (has_fitness != 0)
        {
            out_file.write(reinterpret_cast<const char *>(fitness_scores), population_size * sizeof(float));
        }

        return true;
    }

    bool loadCheckpoint(const std::string &file_path, std::uint64_t &generation, float *fitness_scores_out = nullptr)
    {
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open())
        {
            return false;
        }

        char magic_header[4];
        in_file.read(magic_header, 4);
        if (magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'P' || magic_header[3] != 'K')
        {
            return false;
        }

        std::uint32_t file_version = 0;
        in_file.read(reinterpret_cast<char *>(&file_version), sizeof(file_version));
        if (file_version != 1)
        {
            return false;
        }

        std::uint64_t generation_val = 0;
        std::uint64_t pop_size_val = 0;
        std::uint64_t state_dim_val = 0;
        std::uint64_t action_dim_val = 0;
        std::uint64_t layer_count_val = 0;

        in_file.read(reinterpret_cast<char *>(&generation_val), sizeof(generation_val));
        in_file.read(reinterpret_cast<char *>(&pop_size_val), sizeof(pop_size_val));
        in_file.read(reinterpret_cast<char *>(&state_dim_val), sizeof(state_dim_val));
        in_file.read(reinterpret_cast<char *>(&action_dim_val), sizeof(action_dim_val));
        in_file.read(reinterpret_cast<char *>(&layer_count_val), sizeof(layer_count_val));

        if (pop_size_val != population_size || state_dim_val != state_dimension || action_dim_val != action_space_size || layer_count_val != layer_adapters.size())
        {
            return false;
        }

        generation = generation_val;

        for (auto &adapter : layer_adapters)
        {
            Layer_Type layer_type;
            in_file.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type));
            if (layer_type != adapter.template_layer->getLayerType())
            {
                return false;
            }

            std::uint64_t param_count = 0;
            in_file.read(reinterpret_cast<char *>(&param_count), sizeof(param_count));
            if (param_count != adapter.param_shapes.size())
            {
                return false;
            }

            for (std::size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                std::uint64_t len = 0;
                in_file.read(reinterpret_cast<char *>(&len), sizeof(len));
                if (len != adapter.param_numel[p])
                {
                    return false;
                }

                in_file.read(reinterpret_cast<char *>(adapter.host_params[p].data()),
                             adapter.host_params[p].size() * sizeof(float));
            }
        }

        std::uint8_t has_fitness = 0;
        in_file.read(reinterpret_cast<char *>(&has_fitness), sizeof(has_fitness));
        if (has_fitness != 0)
        {
            if (fitness_scores_out != nullptr)
            {
                in_file.read(reinterpret_cast<char *>(fitness_scores_out), population_size * sizeof(float));
            }
            else
            {
                in_file.seekg(population_size * sizeof(float), std::ios::cur);
            }
        }

        syncHostToDevice();
        return true;
    }

    std::size_t getPopulationSize() const noexcept
    {
        return population_size;
    }

    std::size_t getStateDimension() const noexcept
    {
        return state_dimension;
    }

    std::size_t getActionSpaceSize() const noexcept
    {
        return action_space_size;
    }

    Execution_Target getExecutionTarget() const noexcept
    {
        return execution_target;
    }
};
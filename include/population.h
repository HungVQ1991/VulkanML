#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer/gelu.h"
#include "layer/linear_layer.h"
#include "layer/relu.h"
#include "math/tensor.h"
#include "neural_network.h"

enum class Population_Activation
{
    NONE,
    RELU,
    GELU
};

struct Population_Layer_Block
{
    std::size_t in_dim = 0;
    std::size_t out_dim = 0;
    Population_Activation activation = Population_Activation::NONE;
    Tensor weights;
    Tensor biases;

    Population_Layer_Block(std::size_t pop_size, std::size_t in_d, std::size_t out_d, Population_Activation act, Execution_Target target)
        : in_dim(in_d), out_dim(out_d), activation(act),
          weights(Shape{pop_size, in_d, out_d}, target),
          biases(Shape{pop_size, 1, out_d}, target)
    {
    }
};

class Population
{
private:
    std::size_t population_size = 0;
    std::size_t state_dimension = 0;
    std::size_t action_space_size = 0;
    std::size_t hidden_dimension = 128;
    Execution_Target execution_target = Execution_Target::CPU;
    mutable std::mt19937 random_engine;

    std::vector<Population_Layer_Block> dense_layers;

    std::vector<std::vector<float>> host_weights;
    std::vector<std::vector<float>> host_biases;

    mutable Tensor batched_input_tensor;
    mutable std::vector<float> batched_input_host;

    void initializeWeights()
    {
        host_weights.resize(dense_layers.size());
        host_biases.resize(dense_layers.size());

        for (std::size_t l = 0; l < dense_layers.size(); ++l)
        {
            std::size_t in_d = dense_layers[l].in_dim;
            std::size_t out_d = dense_layers[l].out_dim;
            std::size_t w_total = population_size * in_d * out_d;
            std::size_t b_total = population_size * out_d;

            host_weights[l].resize(w_total);
            host_biases[l].assign(b_total, 0.0f);

            float std_dev = std::sqrt(2.0f / static_cast<float>(in_d));
            std::normal_distribution<float> dist(0.0f, std_dev);

            for (std::size_t i = 0; i < w_total; ++i)
            {
                host_weights[l][i] = dist(random_engine);
            }
        }

        batched_input_tensor = Tensor(Shape{population_size, 1, state_dimension}, execution_target);
        batched_input_host.assign(population_size * state_dimension, 0.0f);

        syncHostToDevice();
    }

    void syncHostToDevice()
    {
        for (std::size_t l = 0; l < dense_layers.size(); ++l)
        {
            dense_layers[l].weights.uploadData(host_weights[l]);
            dense_layers[l].biases.uploadData(host_biases[l]);
        }
    }

    void syncDeviceToHost()
    {
        for (size_t l = 0; l < dense_layers.size(); ++l)
        {
            host_weights[l] = dense_layers[l].weights.getData();
            host_biases[l] = dense_layers[l].weights.getData();
        }
    }

public:
    Population(std::size_t pop_size,
               const Neural_Network &template_net,
               Execution_Target target = Execution_Target::CPU,
               std::uint32_t seed = std::random_device{}())
        : population_size(pop_size), execution_target(target), random_engine(seed)
    {
        const auto &layers = template_net.getLayers();
        for (std::size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i]->getLayerType() == Layer_Type::LINEAR)
            {
                auto *lin = dynamic_cast<const Linear_Layer *>(layers[i].get());
                if (lin)
                {
                    std::size_t in_d = lin->getWeights().getRows();
                    std::size_t out_d = lin->getWeights().getColumns();
                    Population_Activation act = Population_Activation::NONE;
                    if (i + 1 < layers.size())
                    {
                        Layer_Type next_t = layers[i + 1]->getLayerType();
                        if (next_t == Layer_Type::GELU)
                        {
                            act = Population_Activation::GELU;
                        }
                        else if (next_t == Layer_Type::RELU)
                        {
                            act = Population_Activation::RELU;
                        }
                    }
                    dense_layers.emplace_back(pop_size, in_d, out_d, act, target);
                }
            }
        }

        if (dense_layers.empty())
        {
            throw std::invalid_argument("Template neural network contains no Linear layers");
        }

        state_dimension = dense_layers.front().in_dim;
        action_space_size = dense_layers.back().out_dim;
        hidden_dimension = dense_layers.front().out_dim;

        initializeWeights();
    }

    Population(std::size_t pop_size,
               std::size_t state_dim,
               std::size_t action_dim,
               std::size_t hidden_dim = 128,
               Execution_Target target = Execution_Target::CPU,
               std::uint32_t seed = std::random_device{}())
        : population_size(pop_size),
          state_dimension(state_dim),
          action_space_size(action_dim),
          hidden_dimension(hidden_dim),
          execution_target(target),
          random_engine(seed)
    {
        dense_layers.emplace_back(pop_size, state_dim, hidden_dim, Population_Activation::GELU, target);
        dense_layers.emplace_back(pop_size, hidden_dim, hidden_dim, Population_Activation::GELU, target);
        dense_layers.emplace_back(pop_size, hidden_dim, action_dim, Population_Activation::NONE, target);

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

        std::vector<float> current_act(state_data, state_data + state_dimension);
        std::vector<float> next_act;

        for (std::size_t l = 0; l < dense_layers.size(); ++l)
        {
            const auto &layer = dense_layers[l];
            std::size_t in_d = layer.in_dim;
            std::size_t out_d = layer.out_dim;
            const float *w_ptr = host_weights[l].data() + individual_index * (in_d * out_d);
            const float *b_ptr = host_biases[l].data() + individual_index * out_d;

            next_act.assign(out_d, 0.0f);

            for (std::size_t j = 0; j < out_d; ++j)
            {
                next_act[j] = b_ptr[j];
            }

            for (std::size_t k = 0; k < in_d; ++k)
            {
                float in_val = current_act[k];
                const float *w_row = w_ptr + k * out_d;
                for (std::size_t j = 0; j < out_d; ++j)
                {
                    next_act[j] += in_val * w_row[j];
                }
            }

            if (layer.activation == Population_Activation::GELU)
            {
                constexpr float ALPHA = 0.7978845608F;
                constexpr float BETA = 0.044715F;
                for (std::size_t j = 0; j < out_d; ++j)
                {
                    float x = next_act[j];
                    float tanh_in = std::tanh(ALPHA * (x + BETA * x * x * x));
                    next_act[j] = 0.5F * x * (1.0F + tanh_in);
                }
            }
            else if (layer.activation == Population_Activation::RELU)
            {
                for (std::size_t j = 0; j < out_d; ++j)
                {
                    next_act[j] = std::max(0.0F, next_act[j]);
                }
            }

            current_act = std::move(next_act);
        }

        std::size_t best_action = 0;
        float max_val = current_act[0];
        for (std::size_t i = 1; i < current_act.size(); ++i)
        {
            if (current_act[i] > max_val)
            {
                max_val = current_act[i];
                best_action = i;
            }
        }
        return best_action;
    }

    void selectBatchActions(const float *states_flat, const std::size_t *active_indices, std::size_t active_count, std::size_t *actions_out) const
    {
        if (active_count == 0)
        {
            return;
        }

        if (execution_target == Execution_Target::VULKAN_GPU)
        {
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
            for (std::size_t l = 0; l < dense_layers.size(); ++l)
            {
                current_tensor = current_tensor.matmulAdd(dense_layers[l].weights, dense_layers[l].biases);
                if (dense_layers[l].activation == Population_Activation::GELU)
                {
                    current_tensor = current_tensor.gelu();
                }
                else if (dense_layers[l].activation == Population_Activation::RELU)
                {
                    current_tensor = current_tensor.relu();
                }
            }

            std::vector<float> q_values = current_tensor.getData();
            for (std::size_t i = 0; i < active_count; ++i)
            {
                std::size_t agent_idx = active_indices ? active_indices[i] : i;
                std::size_t offset = agent_idx * action_space_size;
                std::size_t best_act = 0;
                float max_val = q_values[offset];
                for (std::size_t a = 1; a < action_space_size; ++a)
                {
                    if (q_values[offset + a] > max_val)
                    {
                        max_val = q_values[offset + a];
                        best_act = a;
                    }
                }
                actions_out[i] = best_act;
            }
            return;
        }

        for (std::size_t i = 0; i < active_count; ++i)
        {
            std::size_t agent_idx = active_indices ? active_indices[i] : i;
            const float *agent_state = states_flat + (i * state_dimension);
            actions_out[i] = selectAction(agent_idx, agent_state);
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

        auto prev_weights = host_weights;
        auto prev_biases = host_biases;

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

        for (std::size_t i = elite_count; i < population_size; ++i)
        {
            std::size_t target_idx = indices[i];
            std::size_t parent_a_idx = selectParentTournament();
            std::size_t parent_b_idx = selectParentTournament();

            for (std::size_t l = 0; l < dense_layers.size(); ++l)
            {
                std::size_t w_len = dense_layers[l].in_dim * dense_layers[l].out_dim;
                std::size_t b_len = dense_layers[l].out_dim;

                std::size_t w_dst_offset = target_idx * w_len;
                std::size_t w_p1_offset = parent_a_idx * w_len;
                std::size_t w_p2_offset = parent_b_idx * w_len;

                for (std::size_t w = 0; w < w_len; ++w)
                {
                    float val = (prob_dist(random_engine) < crossover_rate)
                                    ? prev_weights[l][w_p2_offset + w]
                                    : prev_weights[l][w_p1_offset + w];
                    if (prob_dist(random_engine) < mutation_rate)
                    {
                        val += norm_dist(random_engine);
                    }
                    host_weights[l][w_dst_offset + w] = val;
                }

                std::size_t b_dst_offset = target_idx * b_len;
                std::size_t b_p1_offset = parent_a_idx * b_len;
                std::size_t b_p2_offset = parent_b_idx * b_len;

                for (std::size_t b = 0; b < b_len; ++b)
                {
                    float val = (prob_dist(random_engine) < crossover_rate)
                                    ? prev_biases[l][b_p2_offset + b]
                                    : prev_biases[l][b_p1_offset + b];
                    if (prob_dist(random_engine) < mutation_rate)
                    {
                        val += norm_dist(random_engine);
                    }
                    host_biases[l][b_dst_offset + b] = val;
                }
            }
        }

        syncHostToDevice();
    }

    std::size_t getPopulationSize() const noexcept
    {
        return population_size;
    }

    Neural_Network getIndividual(std::size_t index) const
    {
        if (index >= population_size)
        {
            throw std::out_of_range("Individual index out of range in getIndividual");
        }

        Neural_Network net(Execution_Target::CPU);
        for (std::size_t l = 0; l < dense_layers.size(); ++l)
        {
            std::size_t in_d = dense_layers[l].in_dim;
            std::size_t out_d = dense_layers[l].out_dim;

            auto &lin = net.addLayer<Linear_Layer>(in_d, out_d, Execution_Target::CPU);

            std::vector<float> w(host_weights[l].begin() + index * (in_d * out_d),
                                 host_weights[l].begin() + (index + 1) * (in_d * out_d));
            std::vector<float> b(host_biases[l].begin() + index * out_d,
                                 host_biases[l].begin() + (index + 1) * out_d);

            lin.setWeights(Matrix(in_d, out_d, std::move(w), Execution_Target::CPU));
            lin.setBiases(Matrix(1, out_d, std::move(b), Execution_Target::CPU));

            if (dense_layers[l].activation == Population_Activation::GELU)
            {
                net.addLayer<Gelu_Layer>();
            }
            else if (dense_layers[l].activation == Population_Activation::RELU)
            {
                net.addLayer<Relu_Layer>();
            }
        }

        net.setExecutionTarget(execution_target);
        return net;
    }

    bool saveIndividual(std::size_t individual_index, const std::string &file_path) const
    {
        if (individual_index >= population_size) return false;
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open()) return false;

        std::size_t layer_count = dense_layers.size() * 2;
        out_file.write(reinterpret_cast<const char *>(&layer_count), sizeof(layer_count));

        for (std::size_t l = 0; l < dense_layers.size(); ++l)
        {
            std::size_t w_len = dense_layers[l].in_dim * dense_layers[l].out_dim;
            std::size_t b_len = dense_layers[l].out_dim;

            out_file.write(reinterpret_cast<const char *>(&w_len), sizeof(w_len));
            const float *w_ptr = host_weights[l].data() + individual_index * w_len;
            out_file.write(reinterpret_cast<const char *>(w_ptr), w_len * sizeof(float));

            out_file.write(reinterpret_cast<const char *>(&b_len), sizeof(b_len));
            const float *b_ptr = host_biases[l].data() + individual_index * b_len;
            out_file.write(reinterpret_cast<const char *>(b_ptr), b_len * sizeof(float));
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
        if (individual_index >= population_size) return false;
        std::ifstream in_file(file_path, std::ios::binary);
        if (!in_file.is_open()) return false;

        std::size_t layer_count = 0;
        in_file.read(reinterpret_cast<char *>(&layer_count), sizeof(layer_count));
        if (layer_count != dense_layers.size() * 2) return false;

        for (std::size_t l = 0; l < dense_layers.size(); ++l)
        {
            std::size_t w_len = 0;
            in_file.read(reinterpret_cast<char *>(&w_len), sizeof(w_len));
            if (w_len != dense_layers[l].in_dim * dense_layers[l].out_dim) return false;

            float *w_ptr = host_weights[l].data() + individual_index * w_len;
            in_file.read(reinterpret_cast<char *>(w_ptr), w_len * sizeof(float));

            std::size_t b_len = 0;
            in_file.read(reinterpret_cast<char *>(&b_len), sizeof(b_len));
            if (b_len != dense_layers[l].out_dim) return false;

            float *b_ptr = host_biases[l].data() + individual_index * b_len;
            in_file.read(reinterpret_cast<char *>(b_ptr), b_len * sizeof(float));
        }

        syncHostToDevice();
        return true;
    }

    bool loadBestIndividual(const std::string &file_path)
    {
        return loadIndividual(0, file_path);
    }
};